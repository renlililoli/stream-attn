"""Web parameter schema and the bridge to the canonical H3 estimator."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, replace

from ..h3 import (
    H3BlockShape,
    H3CallbackConfig,
    H3DeviceProfile,
    H3ExecutionConfig,
    H3OperatorProfile,
    H3WeightPolicy,
    build_h3_block_execution,
)
from ..report import timeline_report_data
from ..search import estimate_activation_memory
from ..specs import MemoryPool, RateProfile

MAX_CANDIDATES = 16
MAX_ESTIMATED_EVENTS = 120_000


@dataclass(frozen=True)
class Field:
    name: str
    label: str
    group: str
    kind: str
    default: object
    minimum: float | None = None
    maximum: float | None = None
    step: object = "any"
    choices: tuple[tuple[str, str], ...] = ()
    hint: str = ""


FIELDS = (
    Field("device_label", "设备 / Profile 名称", "模型", "text", "自定义设备"),
    Field("tokens", "序列 tokens", "模型", "integer", 81159, 1, 16777216, 1),
    Field("hidden_features", "Hidden width", "模型", "integer", 5376, 1, 131072, 1),
    Field("ffn_features", "FFN width", "模型", "integer", 14336, 1, 262144, 1),
    Field("heads", "Attention heads", "模型", "integer", 56, 1, 4096, 1),
    Field("head_dim", "Head dimension", "模型", "integer", 128, 8, 1024, 8),
    Field(
        "dtype",
        "激活 dtype",
        "模型",
        "choice",
        "bfloat16",
        choices=(("bfloat16", "BF16"), ("float16", "FP16"), ("float32", "FP32")),
    ),
    Field(
        "execution_mode",
        "执行方式",
        "切块",
        "choice",
        "materialized",
        choices=(
            ("materialized", "Materialized"),
            ("recompute", "Recompute"),
            ("compare", "对比两种方式"),
        ),
    ),
    Field("q_chunk_tokens", "驻留 Q tokens", "切块", "integer", 3840, 1, 16777216, 1),
    Field("kv_tile_tokens", "KV tile tokens", "切块", "integer", 4096, 1, 16777216, 1),
    Field(
        "projection_tile_tokens", "Projection tile tokens", "切块", "integer", 4096, 1, 16777216, 1
    ),
    Field("ffn_tile_tokens", "FFN tile tokens", "切块", "integer", 4096, 1, 16777216, 1),
    Field(
        "callback_variant",
        "回调路径",
        "切块",
        "choice",
        "modulated",
        choices=(("modulated", "完整 H3（含调制 / RoPE）"), ("block25", "Block 25 benchmark 路径")),
    ),
    Field("fa_tflops", "Attention · TFLOP/s", "吞吐与带宽", "number", 200, 0.000001, 10000000),
    Field("gemm_tflops", "GEMM · TFLOP/s", "吞吐与带宽", "number", 120, 0.000001, 10000000),
    Field(
        "vector_gelements", "Vector · G elements/s", "吞吐与带宽", "number", 100, 0.000001, 10000000
    ),
    Field("h2d_gbps", "Host → Device · GB/s", "吞吐与带宽", "number", 50, 0.000001, 10000000),
    Field("d2h_gbps", "Device → Host · GB/s", "吞吐与带宽", "number", 40, 0.000001, 10000000),
    Field("d2d_gbps", "Device → Device · GB/s", "吞吐与带宽", "number", 500, 0.000001, 10000000),
    Field("launch_us", "每次调用固定延迟 · μs", "吞吐与带宽", "number", 5, 0, 1000000),
    Field(
        "linear_memory",
        "Linear 工作区模型",
        "内存",
        "choice",
        "int8_eager",
        choices=(
            ("int8_eager", "INT8 eager ConvRot"),
            ("dense", "Dense"),
            ("profile", "仅使用导入 Profile 的私有工作区"),
        ),
    ),
    Field(
        "weight_scale",
        "INT8 weight scale",
        "内存",
        "choice",
        "channel",
        choices=(("channel", "逐通道"), ("tensor", "逐张量")),
    ),
    Field("convrot_group", "ConvRot group", "内存", "integer", 256, 1, 65536, 1),
    Field(
        "device_capacity_gib",
        "设备存储预算 · GiB",
        "内存",
        "optional_number",
        None,
        0.000001,
        1048576,
        hint="留空表示不限制；只检查图中声明的分配",
    ),
    Field(
        "host_capacity_gib",
        "Host 存储预算 · GiB",
        "内存",
        "optional_number",
        None,
        0.000001,
        1048576,
    ),
    Field("num_kv_buffers", "K/V slots", "高级参数", "integer", 2, 1, 3, 1),
    Field("num_projection_buffers", "Projection slots", "高级参数", "integer", 2, 1, 3, 1),
    Field("num_output_buffers", "Output slots", "高级参数", "integer", 2, 1, 2, 1),
    Field(
        "segments",
        "Packed segment 长度",
        "高级参数",
        "text",
        "",
        hint="逗号分隔；留空使用上方 tokens",
    ),
    Field(
        "ffn_candidates",
        "对比 FFN tiles",
        "高级参数",
        "text",
        "",
        hint="如 2048,4096,8192,16384；留空只模拟当前 tile",
    ),
    Field("target_percent", "候选吞吐目标 · %", "高级参数", "number", 95, 0.01, 100),
    Field(
        "rope_dim",
        "RoPE dimension",
        "高级参数",
        "optional_integer",
        None,
        0,
        1024,
        2,
        hint="留空为 min(96, head dimension)",
    ),
    Field("modulation_rows", "Modulation rows", "高级参数", "integer", 3, 1, 4096, 1),
    Field("modulation_modalities", "Modulation modalities", "高级参数", "integer", 3, 1, 128, 1),
    Field("timestep_features", "Timestep width", "高级参数", "integer", 2688, 1, 131072, 1),
    Field(
        "qkv_layout",
        "QKV 返回布局",
        "高级参数",
        "choice",
        "strided",
        choices=(("strided", "Strided：包含打包拷贝"), ("contiguous", "Contiguous")),
    ),
    Field(
        "workspace_margin_mib",
        "Workspace 预留 · MiB",
        "高级参数",
        "number",
        32,
        0,
        1048576,
        hint="单独记入预算，不绘制为实际分配",
    ),
    Field(
        "qkv_tflops",
        "QKV · TFLOP/s",
        "分算子速率",
        "optional_number",
        None,
        0.000001,
        10000000,
        hint="留空沿用 GEMM",
    ),
    Field(
        "out_tflops",
        "Output projection · TFLOP/s",
        "分算子速率",
        "optional_number",
        None,
        0.000001,
        10000000,
    ),
    Field("fc1_tflops", "FC1 · TFLOP/s", "分算子速率", "optional_number", None, 0.000001, 10000000),
    Field("fc2_tflops", "FC2 · TFLOP/s", "分算子速率", "optional_number", None, 0.000001, 10000000),
    Field(
        "weight_mode",
        "权重存储方式",
        "权重与来源",
        "choice",
        "resident",
        choices=(("resident", "已驻留"), ("staged", "分阶段搬运 / 释放")),
    ),
    Field("projection_weight_mib", "Projection 权重 · MiB", "权重与来源", "number", 0, 0, 1048576),
    Field("consumer_weight_mib", "Consumer 权重 · MiB", "权重与来源", "number", 0, 0, 1048576),
    Field("auxiliary_weight_mib", "辅助权重 · MiB", "权重与来源", "number", 0, 0, 1048576),
    Field("provenance", "数值来源", "权重与来源", "text", "手动参数；初始速率为示例，非实测"),
)


def schema():
    return {
        "version": 1,
        "fields": [asdict(f) for f in FIELDS],
        "max_candidates": MAX_CANDIDATES,
        "max_estimated_events": MAX_ESTIMATED_EVENTS,
    }


def normalize(parameters):
    if not isinstance(parameters, dict):
        raise TypeError("parameters 必须是对象")
    known = {f.name for f in FIELDS}
    if set(parameters) - known:
        raise ValueError("未知参数：" + ", ".join(sorted(set(parameters) - known)))
    result = {}
    for f in FIELDS:
        value = parameters.get(f.name, f.default)
        if f.kind.startswith("optional") and value is None:
            result[f.name] = None
            continue
        if f.kind in {"text", "choice"}:
            if not isinstance(value, str) or len(value) > 4096:
                raise ValueError(f"{f.label} 必须是长度不超过 4096 的文本")
            value = value.strip()
            if f.kind == "choice" and value not in dict(f.choices):
                raise ValueError(f"{f.label} 选项无效")
        else:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{f.label} 必须是有限数字")
            if "integer" in f.kind and (not isinstance(value, int) or isinstance(value, bool)):
                raise ValueError(f"{f.label} 必须是整数")
            if (
                f.minimum is not None
                and value < f.minimum
                or f.maximum is not None
                and value > f.maximum
            ):
                raise ValueError(f"{f.label} 必须在 {f.minimum}–{f.maximum} 之间")
        result[f.name] = value
    return result


def _integers(text, label):
    values = re.split(r"[,，;；\s]+", text.strip())
    if len(values) > 256 or any(not re.fullmatch(r"[0-9]+", v) for v in values):
        raise ValueError(f"{label} 需要用逗号分隔的正整数，最多 256 项")
    numbers = tuple(int(v) for v in values)
    if any(not 0 < v <= 16777216 for v in numbers):
        raise ValueError(f"{label} 数值超出范围")
    return numbers


def prepare_request(parameters, imported_profile=None):
    p = normalize(parameters)
    segments = _integers(p["segments"], "分段长度") if p["segments"] else (p["tokens"],)
    if sum(segments) > 16777216:
        raise ValueError("分段 token 总数超出范围")
    p["tokens"] = sum(segments)
    ffn_tiles = (
        tuple(dict.fromkeys(_integers(p["ffn_candidates"], "FFN tiles")))
        if p["ffn_candidates"]
        else (p["ffn_tile_tokens"],)
    )
    modes = (
        ("materialized", "recompute")
        if p["execution_mode"] == "compare"
        else (p["execution_mode"],)
    )
    if len(ffn_tiles) * len(modes) > MAX_CANDIDATES:
        raise ValueError(f"交互模式最多比较 {MAX_CANDIDATES} 个候选")
    q_passes = sum((s + p["q_chunk_tokens"] - 1) // p["q_chunk_tokens"] for s in segments)
    updates = sum(
        ((s + p["q_chunk_tokens"] - 1) // p["q_chunk_tokens"])
        * ((s + p["kv_tile_tokens"] - 1) // p["kv_tile_tokens"])
        for s in segments
    )
    projection = (sum(segments) + p["projection_tile_tokens"] - 1) // p["projection_tile_tokens"]
    bound = 0
    for mode in modes:
        for ffn in ffn_tiles:
            ffn_calls = (sum(segments) + ffn - 1) // ffn
            bound += (
                (
                    30 * projection + 20 * q_passes + 6 * updates
                    if mode == "materialized"
                    else 40 * q_passes + 30 * updates
                )
                + 12 * ffn_calls
                + 100
            )
    # The eager linear memory model also iterates through scaling chunks.
    # Account for that work before building a trace, even with very large tiles.
    if p["linear_memory"] == "int8_eager":
        a = p["heads"] * p["head_dim"]
        rows_for = lambda width: max(1, (256 * 2**20) // (width * 4))
        repeated_kv_tokens = sum(
            ((s + p["q_chunk_tokens"] - 1) // p["q_chunk_tokens"]) * s for s in segments
        )
        for mode in modes:
            projection_work = (
                p["tokens"] // rows_for(3 * a)
                if mode == "materialized"
                else (p["tokens"] // rows_for(a) + repeated_kv_tokens // rows_for(2 * a))
            )
            bound += len(ffn_tiles) * (
                projection_work
                + p["tokens"] // rows_for(2 * p["ffn_features"])
                + 2 * p["tokens"] // rows_for(p["hidden_features"])
            )
    if bound > MAX_ESTIMATED_EVENTS:
        raise ValueError(
            f"此组合预计最多生成 {bound:,} 个事件，超过交互上限 {MAX_ESTIMATED_EVENTS:,}。请增大 tile、减少候选，或使用 Python API。"
        )
    dtype = p["dtype"]
    shape = H3BlockShape(
        segments,
        p["hidden_features"],
        p["ffn_features"],
        p["heads"],
        p["head_dim"],
        element_bytes=4 if dtype == "float32" else 2,
        activation_dtype=dtype,
        rope_dim=p["rope_dim"] if p["rope_dim"] is not None else min(96, p["head_dim"]),
        modulation_rows=p["modulation_rows"],
        modulation_modalities=p["modulation_modalities"],
        timestep_features=p["timestep_features"],
    )

    def rate(name, value, resource, kind="compute"):
        return RateProfile(
            name,
            value,
            (resource,),
            kind=kind,
            latency_seconds=p["launch_us"] / 1e6,
            provenance=p["provenance"],
        )

    profile = H3DeviceProfile.from_rates(
        p["device_label"],
        device_pool=MemoryPool("device.memory", allocation_alignment_bytes=512),
        host_pool=MemoryPool("host.DRAM"),
        attention=rate("attention FLOP/s", p["fa_tflops"] * 1e12, "compute"),
        gemm=rate("GEMM FLOP/s", p["gemm_tflops"] * 1e12, "compute"),
        vector=rate("vector elements/s", p["vector_gelements"] * 1e9, "compute"),
        h2d=rate("H2D byte/s", p["h2d_gbps"] * 1e9, "H2D", "io"),
        d2h=rate("D2H byte/s", p["d2h_gbps"] * 1e9, "D2H", "io"),
        d2d=rate("D2D byte/s", p["d2d_gbps"] * 1e9, "compute"),
    )
    operators = dict(profile.operators)
    for field, key in (
        ("qkv_tflops", "qkv"),
        ("out_tflops", "out"),
        ("fc1_tflops", "fc1"),
        ("fc2_tflops", "fc2"),
    ):
        if p[field] is not None:
            operators[key] = H3OperatorProfile(
                rate=rate(key + " FLOP/s", p[field] * 1e12, "compute")
            )
    profile = replace(profile, operators=operators)
    if imported_profile is not None:
        if not isinstance(imported_profile, dict):
            raise ValueError("导入的 profile 必须是对象")
        operators = imported_profile.get("operators")
        if not isinstance(operators, dict) or len(operators) > 64:
            raise ValueError("交互 Profile 需要 operators 对象，最多 64 个算子")
        if len(imported_profile.get("local_pools", ())) > 8:
            raise ValueError("交互 Profile 最多支持 8 个额外存储池；更多请使用 Python API")
        if any(
            not isinstance(op, dict)
            or len(op.get("additional_workspaces", ())) > 4
            or len(op.get("samples", ())) > 2048
            for op in operators.values()
        ):
            raise ValueError("交互 Profile 每个算子最多 4 个额外工作区和 2048 个测量样本")
        profile = H3DeviceProfile.from_dict(imported_profile)
    # Explicit UI capacities override only when filled, preserving imported pool names and limits otherwise.
    changes = {}
    for field, member in (
        ("device_capacity_gib", "device_pool"),
        ("host_capacity_gib", "host_pool"),
    ):
        if p[field] is not None:
            changes[member] = replace(
                getattr(profile, member), capacity_bytes=max(1, round(p[field] * 2**30))
            )
    profile = replace(profile, **changes)
    callbacks = H3CallbackConfig(
        variant=p["callback_variant"],
        linear_memory=p["linear_memory"],
        per_channel_weight_scale=p["weight_scale"] == "channel",
        convrot_group=p["convrot_group"],
        qkv_result_layout=p["qkv_layout"],
    )
    weights = H3WeightPolicy(
        mode=p["weight_mode"],
        projection_bytes=round(p["projection_weight_mib"] * 2**20),
        consumer_bytes=round(p["consumer_weight_mib"] * 2**20),
        auxiliary_bytes=round(p["auxiliary_weight_mib"] * 2**20),
    )
    configs = tuple(
        H3ExecutionConfig(
            p["q_chunk_tokens"],
            p["kv_tile_tokens"],
            p["projection_tile_tokens"],
            ffn,
            execution_mode=mode,
            num_kv_buffers=p["num_kv_buffers"],
            num_projection_buffers=p["num_projection_buffers"],
            num_output_buffers=p["num_output_buffers"],
            workspace_margin_bytes=round(p["workspace_margin_mib"] * 2**20),
        )
        for mode in modes
        for ffn in ffn_tiles
    )
    return p, shape, configs, profile, callbacks, weights, bound


def simulate(parameters, imported_profile=None):
    p, shape, configs, profile, callbacks, weights, bound = prepare_request(
        parameters, imported_profile
    )
    specs = [
        build_h3_block_execution(shape, c, profile, callbacks=callbacks, weights=weights)
        for c in configs
    ]
    result = estimate_activation_memory(
        specs,
        objective_pool=profile.device_pool.name,
        target_throughput_fraction=p["target_percent"] / 100,
        objective_owners=frozenset({"operator", "callback", "caller"}),
    )
    return {
        "parameters": p,
        "report": timeline_report_data(result),
        "estimated_event_bound": bound,
        "actual_events": sum(len(s.operations) for s in specs),
        "profile_imported": imported_profile is not None,
    }
