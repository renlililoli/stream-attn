"""Explicit H3 shapes, callback contracts and per-operator calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal

from ..specs import MemoryPool, RateProfile, finite_number, positive_int


@dataclass(frozen=True)
class H3BlockShape:
    segments: tuple[int, ...]
    hidden_features: int
    ffn_features: int
    heads: int
    head_dim: int
    element_bytes: int = 2
    activation_dtype: str = "bfloat16"
    rope_dim: int = 96
    modulation_rows: int = 3
    timestep_features: int = 2688
    modulation_modalities: int = 3
    modulation_element_bytes: int = 4

    def __post_init__(self):
        sizes = {"bfloat16": 2, "float16": 2, "float32": 4}
        if self.activation_dtype not in sizes or sizes[self.activation_dtype] != self.element_bytes:
            raise ValueError(
                "activation dtype and byte size must describe BF16, FP16 or FP32 storage"
            )
        if not isinstance(self.segments, tuple) or not self.segments:
            raise ValueError("segments must be a non-empty tuple")
        for tokens in self.segments:
            positive_int("segment length", tokens)
        for name in (
            "hidden_features",
            "ffn_features",
            "heads",
            "head_dim",
            "element_bytes",
            "modulation_rows",
            "timestep_features",
            "modulation_modalities",
            "modulation_element_bytes",
        ):
            positive_int(name, getattr(self, name))
        if self.modulation_rows % self.modulation_modalities:
            raise ValueError("modulation_rows must be a multiple of modulation_modalities")
        positive_int("rope_dim", self.rope_dim, allow_zero=True)
        if self.rope_dim > self.head_dim or self.rope_dim % 2:
            raise ValueError("rope_dim must be even and at most head_dim")

    @property
    def tokens(self):
        return sum(self.segments)

    @property
    def attention_features(self):
        return self.heads * self.head_dim


@dataclass(frozen=True)
class H3ExecutionConfig:
    q_chunk_tokens: int
    kv_tile_tokens: int
    projection_tile_tokens: int
    ffn_tile_tokens: int
    execution_mode: Literal["materialized", "recompute"] = "materialized"
    num_kv_buffers: int = 2
    num_projection_buffers: int = 2
    num_output_buffers: int = 2
    qkv_capacity_tokens: int | None = None
    kv_capacity_tokens: int | None = None
    workspace_margin_bytes: int = 32 * 2**20
    attention_mode: str = "dense"

    @classmethod
    def from_attention_plan(
        cls,
        plan,
        *,
        projection_tile_tokens,
        ffn_tile_tokens,
        execution_mode="materialized",
        num_projection_buffers=2,
        num_output_buffers=2,
    ):
        if getattr(plan, "backend_workspace_bytes", 0) or getattr(plan, "backend", None) == "sage3":
            raise ValueError(
                "the offline H3 estimator models online-softmax execution, not Sage3 query preparation/LSE merging"
            )
        if plan.output_mode != "device_consumer":
            raise ValueError("H3 requires a device-consumer attention plan")
        if plan.q_heads != plan.kv_heads:
            raise ValueError("H3 callbacks require equal Q/KV head counts")
        return cls(
            plan.q_chunk_tokens,
            plan.kv_chunk_tokens,
            projection_tile_tokens,
            ffn_tile_tokens,
            execution_mode=execution_mode,
            num_kv_buffers=plan.num_kv_buffers,
            num_projection_buffers=num_projection_buffers,
            num_output_buffers=num_output_buffers,
            qkv_capacity_tokens=plan.max_q_tokens,
            kv_capacity_tokens=plan.max_kv_tokens,
        )

    def __post_init__(self):
        if self.execution_mode not in {"materialized", "recompute"}:
            raise ValueError("execution_mode must be materialized or recompute")
        if self.attention_mode != "dense":
            raise ValueError("H3 estimation currently requires explicit dense attention")
        for name in (
            "q_chunk_tokens",
            "kv_tile_tokens",
            "projection_tile_tokens",
            "ffn_tile_tokens",
        ):
            positive_int(name, getattr(self, name))
        for name in ("num_kv_buffers", "num_projection_buffers"):
            if getattr(self, name) not in {1, 2, 3}:
                raise ValueError(f"{name} must be 1, 2 or 3")
        if self.num_output_buffers not in {1, 2}:
            raise ValueError("num_output_buffers must be 1 or 2")
        for name in ("qkv_capacity_tokens", "kv_capacity_tokens"):
            if getattr(self, name) is not None:
                positive_int(name, getattr(self, name))
        positive_int("workspace_margin_bytes", self.workspace_margin_bytes, allow_zero=True)


@dataclass(frozen=True)
class H3CallbackConfig:
    """Concrete no-LoRA callbacks: benchmark linear-only or norm/modulation/RoPE.

    Operator workspace profiles account for implementation-private temporaries
    (e.g. INT8 ConvRot). Named tensor results and their aliases are modeled here.
    """

    variant: Literal["block25", "modulated"] = "modulated"
    qkv_result_layout: Literal["strided", "contiguous"] = "strided"
    # Actual recompute callbacks return fresh Q/KV then copy into runtime slots.
    recompute_direct_write: bool = False
    # With fused materialized QK normalization/RoPE, the QKV allocation is mutated.
    materialized_qk_inplace: bool = True
    compute_modulation: bool = True
    # FC2 may fuse input activation; its workspace must include any hidden
    # activation intermediates when this is True.
    fused_swiglu_fc2: bool | None = None
    name: str = "explicit H3 SwiGLU callbacks"
    linear_memory: Literal["dense", "int8_eager", "profile"] = "profile"
    convrot_group: int = 256
    per_channel_weight_scale: bool = True

    def __post_init__(self):
        if self.fused_swiglu_fc2 is None:
            object.__setattr__(self, "fused_swiglu_fc2", self.variant == "modulated")
        if (
            self.variant == "modulated"
            and self.linear_memory != "dense"
            and not self.fused_swiglu_fc2
        ):
            raise ValueError(
                "production INT8 H3 uses linear_input_act; separate SwiGLU/FC2 is a different callback"
            )
        if self.variant == "modulated" and self.qkv_result_layout != "strided":
            raise ValueError(
                "production H3 returns strided QKV views; contiguous outputs require a different callback"
            )
        if self.linear_memory not in {"dense", "int8_eager", "profile"}:
            raise ValueError("unknown linear memory implementation")
        positive_int("convrot_group", self.convrot_group)
        if self.convrot_group & (self.convrot_group - 1):
            raise ValueError("convrot_group must be a power of two")
        if self.variant not in {"block25", "modulated"}:
            raise ValueError("unknown callback variant")
        if self.qkv_result_layout not in {"strided", "contiguous"}:
            raise ValueError("qkv_result_layout must be strided or contiguous")
        if self.variant == "block25" and self.compute_modulation:
            object.__setattr__(self, "compute_modulation", False)
        if self.recompute_direct_write and self.linear_memory == "int8_eager":
            raise ValueError(
                "eager INT8 returns fresh tensors; direct-write needs its own operator profile"
            )
        if self.recompute_direct_write and self.variant == "modulated":
            raise ValueError(
                "modulated callbacks use fresh norm/RoPE results before direct-write copies"
            )


@dataclass(frozen=True)
class H3OperatorSample:
    tokens: int
    seconds: float
    extra_workspace_bytes: int
    kv_tokens: int = 0
    work: float | None = None

    def __post_init__(self):
        positive_int("tokens", self.tokens)
        positive_int("kv_tokens", self.kv_tokens, allow_zero=True)
        finite_number("seconds", self.seconds)
        if self.work is not None:
            finite_number("sample work", self.work)
        positive_int("extra_workspace_bytes", self.extra_workspace_bytes, allow_zero=True)


@dataclass(frozen=True)
class H3ScratchBuffer:
    """Additional implementation workspace in an explicitly scoped physical pool."""

    pool: str
    fixed_bytes: int = 0
    bytes_per_token: int = 0
    component: str = "local operator workspace"

    def __post_init__(self):
        if not self.pool or not self.component:
            raise ValueError("scratch pool and component must be named")
        positive_int("fixed_bytes", self.fixed_bytes, allow_zero=True)
        positive_int("bytes_per_token", self.bytes_per_token, allow_zero=True)


@dataclass(frozen=True)
class H3OperatorProfile:
    """One implementation, with exact-shape samples preferred over its rate model.

    Rate work units are assigned by the builder: GEMM/attention FLOPs, transfer
    bytes, vector elements. Extra workspace excludes declared input/output tensors.
    Samples are not silently interpolated; missing shapes require an explicit rate.
    """

    rate: RateProfile | None = None
    samples: tuple[H3OperatorSample, ...] = ()
    extra_workspace_bytes: int = 0
    extra_bytes_per_token: int = 0
    provenance: str = "analytical operator model"
    additional_workspaces: tuple[H3ScratchBuffer, ...] = ()

    def __post_init__(self):
        if self.rate is None and not self.samples:
            raise ValueError("an operator requires measured samples or an explicit rate")
        positive_int("extra_workspace_bytes", self.extra_workspace_bytes, allow_zero=True)
        positive_int("extra_bytes_per_token", self.extra_bytes_per_token, allow_zero=True)
        keys = [(s.tokens, s.kv_tokens, s.work) for s in self.samples]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate operator sample shape")

    def resolve(self, tokens, work, kv_tokens=0):
        matching = [s for s in self.samples if (s.tokens, s.kv_tokens) == (tokens, kv_tokens)]
        sample = next((s for s in matching if s.work == work), None)
        if sample is None:
            sample = next((s for s in matching if s.work is None), None)
        if sample is not None:
            return sample.seconds, sample.extra_workspace_bytes, self.provenance, "sample"
        if self.rate is None:
            raise ValueError(
                f"no calibrated operator sample for tokens={tokens}, kv_tokens={kv_tokens}"
            )
        return (
            self.rate.seconds(work),
            self.extra_workspace_bytes + tokens * self.extra_bytes_per_token,
            self.provenance + "; " + self.rate.provenance,
            "rate",
        )


@dataclass(frozen=True)
class H3DeviceProfile:
    name: str
    device_pool: MemoryPool
    host_pool: MemoryPool
    operators: dict[str, H3OperatorProfile]
    # Explicit stream-to-resource mapping. Same-stream order is always enforced.
    compute_resources: tuple[str, ...] = ("compute",)
    h2d_resources: tuple[str, ...] = ("H2D",)
    d2h_resources: tuple[str, ...] = ("D2H",)
    q_alignment: int = 1
    kv_alignment: int = 1
    local_pools: tuple[MemoryPool, ...] = ()
    shape_signature: dict[str, object] | None = None
    # None preserves legacy D2D resource contention. An observed concurrent pack
    # path can use a distinct resource; effective rates must include contention.
    projection_pack_resources: tuple[str, ...] | None = None

    def __post_init__(self):
        pool_names = [
            self.device_pool.name,
            self.host_pool.name,
            *(p.name for p in self.local_pools),
        ]
        if len(set(pool_names)) != len(pool_names):
            raise ValueError("H3 physical pool names must be distinct; aliases must share a pool")
        for key, op in self.operators.items():
            if key in {"h2d", "d2h", "d2d"} and any(s.work is None for s in op.samples):
                raise ValueError(
                    "transfer samples must specify work=payload_bytes, not only token count"
                )
            if any(w.pool not in pool_names for w in op.additional_workspaces):
                raise ValueError("operator workspace refers to an unknown physical pool")
        for value in (self.q_alignment, self.kv_alignment):
            positive_int("alignment", value)
        resource_groups = (self.compute_resources, self.h2d_resources, self.d2h_resources)
        if self.projection_pack_resources is not None:
            resource_groups += (self.projection_pack_resources,)
        for resources in resource_groups:
            if (
                not isinstance(resources, (tuple, list))
                or not resources
                or any(not isinstance(r, str) or not r.strip() for r in resources)
                or len(set(resources)) != len(resources)
            ):
                raise ValueError("resource groups must be sequences of non-empty unique names")

    def for_shape(self, shape):
        signature = asdict(shape)
        signature.pop("segments")
        return replace(self, shape_signature=signature)

    def validate_shape(self, shape):
        signature = asdict(shape)
        signature.pop("segments")
        if self.shape_signature is not None and signature != self.shape_signature:
            raise ValueError(
                "operator calibration shape/dtype signature does not match the H3 block"
            )
        if self.shape_signature is None and any(op.samples for op in self.operators.values()):
            raise ValueError("measured operator samples require profile.for_shape(shape)")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        values = dict(data)
        values["device_pool"] = MemoryPool(**values["device_pool"])
        values["host_pool"] = MemoryPool(**values["host_pool"])
        values["local_pools"] = tuple(MemoryPool(**p) for p in values.get("local_pools", ()))
        for name in (
            "compute_resources",
            "h2d_resources",
            "d2h_resources",
            "projection_pack_resources",
        ):
            if name not in values or (name == "projection_pack_resources" and values[name] is None):
                continue
            if not isinstance(values[name], (list, tuple)):
                raise TypeError(f"{name} must be a sequence of resource names")
            values[name] = tuple(values[name])
        operators = {}
        for key, entry in values["operators"].items():
            entry = dict(entry)
            if entry.get("rate") is not None:
                rate = dict(entry["rate"])
                rate["resources"] = tuple(rate["resources"])
                entry["rate"] = RateProfile(**rate)
            entry["samples"] = tuple(H3OperatorSample(**s) for s in entry.get("samples", ()))
            entry["additional_workspaces"] = tuple(
                H3ScratchBuffer(**w) for w in entry.get("additional_workspaces", ())
            )
            operators[key] = H3OperatorProfile(**entry)
        values["operators"] = operators
        return cls(**values)

    @classmethod
    def from_rates(cls, name, *, device_pool, host_pool, attention, gemm, vector, h2d, d2h, d2d):
        """Create a fully explicit analytical profile; vector rate is elements/s.

        This convenience initializer supplies zero implementation-private scratch.
        Override individual operators with measured samples/extra workspaces for
        quantized, fused or compiler-specific kernels. No hardware peaks are used.
        """
        rates = {
            key: gemm for key in ("qkv", "q", "kv", "out", "fc1", "fc2", "swiglu_fc2", "adaln")
        }
        rates.update(
            {
                key: vector
                for key in (
                    "norm1",
                    "modulate1",
                    "qk_rope",
                    "q_rope",
                    "k_rope",
                    "finalize",
                    "attention_residual",
                    "norm2",
                    "modulate2",
                    "swiglu",
                    "ffn_residual",
                )
            }
        )
        rates.update(
            {
                key: vector
                for key in ("rope_angles", "rope_table", "qk_norm", "rope_rotate", "rope_concat")
            }
        )
        rates.update(attention=attention, h2d=h2d, d2h=d2h, d2d=d2d)
        return cls(
            name,
            device_pool,
            host_pool,
            {key: H3OperatorProfile(rate=rate) for key, rate in rates.items()},
            compute_resources=attention.resources,
            h2d_resources=h2d.resources,
            d2h_resources=d2h.resources,
        )


@dataclass(frozen=True)
class H3WeightPolicy:
    """Declared weight storage, separate from activation/workspace accounting.

    Resident weights have already been prepared. Stage-leased weights include
    H2D and follow the runner's projection/consumer context lifetimes.
    Zero sizes explicitly exclude weights from the estimate.
    """

    mode: Literal["resident", "staged"] = "resident"
    projection_bytes: int = 0
    consumer_bytes: int = 0
    auxiliary_bytes: int = 0

    def __post_init__(self):
        if self.mode not in {"resident", "staged"}:
            raise ValueError("weight mode must be resident or staged")
        for name in ("projection_bytes", "consumer_bytes", "auxiliary_bytes"):
            positive_int(name, getattr(self, name), allow_zero=True)
