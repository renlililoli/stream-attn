from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from typing import Any

DEFAULT_CHECKPOINT = Path(
    "/opt/ComfyUI/models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
)
EXPECTED_COMFYUI_VERSION = "0.30.0"
EXPECTED_COMFYUI_COMMIT = "9a9fdb10ed144ce760d9682cb247526ea23cc525"
EXPECTED_TORCH_VERSION = "2.10.0+cu128"
EXPECTED_TORCH_CUDA_VERSION = "12.8"
EXPECTED_COMFY_AIMDO_VERSION = "0.4.11"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare materialized and large-tile QKV recompute on MiniMax-H3 block 25"
    )
    parser.add_argument(
        "--mode", choices=("materialized", "recompute", "compare"), default="compare"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--comfyui-root", type=Path, default=Path("/opt/ComfyUI"))
    parser.add_argument("--block-index", type=int, default=25)
    parser.add_argument("--tokens", type=int, default=262_720)
    parser.add_argument("--q-chunk-tokens", type=int, default=16_384)
    parser.add_argument("--kv-chunk-tokens", type=int, default=4_096)
    parser.add_argument("--qkv-tile-tokens", type=int, default=2_048)
    parser.add_argument("--mlp-tile-tokens", type=int, default=2_048)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--validation-tokens", type=int, default=16)
    parser.add_argument("--sample-interval-ms", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")
    os.replace(temporary, path)


def _require_version(actual: str, expected: str, name: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{name} mismatch: expected {expected}, found {actual}")


def _mode_command(args: argparse.Namespace, mode: str, output: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        mode,
        "--checkpoint",
        str(args.checkpoint),
        "--comfyui-root",
        str(args.comfyui_root),
        "--block-index",
        str(args.block_index),
        "--tokens",
        str(args.tokens),
        "--q-chunk-tokens",
        str(args.q_chunk_tokens),
        "--kv-chunk-tokens",
        str(args.kv_chunk_tokens),
        "--qkv-tile-tokens",
        str(args.qkv_tile_tokens),
        "--mlp-tile-tokens",
        str(args.mlp_tile_tokens),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--validation-tokens",
        str(args.validation_tokens),
        "--sample-interval-ms",
        str(args.sample_interval_ms),
        "--seed",
        str(args.seed),
        "--output",
        str(output),
    ]


def _run_comparison(args: argparse.Namespace) -> None:
    children: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="seqattn-convrot-qkv-") as temporary:
        temporary_path = Path(temporary)
        for mode in ("materialized", "recompute"):
            output = temporary_path / f"{mode}.json"
            completed = subprocess.run(
                _mode_command(args, mode, output),
                check=False,
                capture_output=True,
                text=True,
            )
            if output.exists():
                payload = json.loads(output.read_text(encoding="ascii"))
            else:
                payload = {
                    "status": "missing_output",
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            payload["child_returncode"] = completed.returncode
            if completed.returncode != 0:
                payload["child_stdout"] = completed.stdout
                payload["child_stderr"] = completed.stderr
            children[mode] = payload

    materialized = children["materialized"]
    recompute = children["recompute"]
    both_succeeded = all(payload.get("status") == "success" for payload in children.values())
    comparison: dict[str, Any] = {}
    if both_succeeded:
        materialized_summary = materialized["summary"]
        recompute_summary = recompute["summary"]
        comparison = {
            "recompute_over_materialized_wall_median": (
                recompute_summary["wall_median_seconds"]
                / materialized_summary["wall_median_seconds"]
            ),
            "recompute_over_materialized_wall_mean": (
                recompute_summary["wall_mean_seconds"] / materialized_summary["wall_mean_seconds"]
            ),
            "logical_host_activation_reduction_bytes": (
                materialized_summary["logical_host_activation_bytes"]
                - recompute_summary["logical_host_activation_bytes"]
            ),
            "logical_host_activation_reduction_fraction": 1.0
            - (
                recompute_summary["logical_host_activation_bytes"]
                / materialized_summary["logical_host_activation_bytes"]
            ),
        }

    result = {
        "status": "success" if both_succeeded else "child_failure",
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "output"
        },
        "process_isolation": "one independent child process per mode",
        "validity": {
            "current_design": "large attention-tile Q-only/KV-only direct-write callbacks",
            "excluded_prior_results": [
                "q_chunk_tokens=196608 experiments",
                "the removed recompute path that reused materialized projection subtiles",
            ],
        },
        "materialized": materialized,
        "recompute": recompute,
        "comparison": comparison,
    }
    _atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "comparison": comparison}, indent=2))


def _run_mode(args: argparse.Namespace) -> None:
    result: dict[str, Any] = {
        "status": "runtime_error",
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    sampler = None
    try:
        if args.mode not in {"materialized", "recompute"}:
            raise ValueError("child execution requires materialized or recompute mode")
        if args.tokens <= 0 or args.repeats <= 0 or args.warmup < 0:
            raise ValueError("tokens/repeats must be positive and warmup must be non-negative")
        if (
            min(
                args.q_chunk_tokens,
                args.kv_chunk_tokens,
                args.qkv_tile_tokens,
                args.mlp_tile_tokens,
                args.validation_tokens,
            )
            <= 0
        ):
            raise ValueError("all chunk and validation sizes must be positive")
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
        if not args.comfyui_root.is_dir():
            raise FileNotFoundError(f"ComfyUI root not found: {args.comfyui_root}")
        import comfy_aimdo.control

        if not comfy_aimdo.control.init(nvml_pressure=True):
            raise RuntimeError("failed to initialize comfy-aimdo before torch import")

        sys.argv = [sys.argv[0]]
        sys.path.insert(0, str(args.comfyui_root))
        os.chdir(args.comfyui_root)

        from importlib.metadata import version

        import comfy.sd
        import comfy_kitchen
        import comfyui_version
        import torch
        from comfy import memory_management, model_management, model_patcher
        from comfy import ops as comfy_ops

        from seqattn_core import (
            H3BlockOps,
            H3DiTStats,
            H3MaterializedProjection,
            H3MaterializedRunner,
            H3RecomputeProjection,
            H3RecomputeRunner,
            H3SequenceMeta,
            ProjectedAttentionRunner,
            ProjectionPipelineConfig,
            RecomputedAttentionRunner,
            StreamingAttentionConfig,
            build_plan,
        )
        from seqattn_core.benchmarking.common import ProcessMemorySampler

        if not torch.cuda.is_available():
            raise RuntimeError("a visible CUDA GPU is required")
        comfyui_commit = subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={args.comfyui_root}",
                "-C",
                str(args.comfyui_root),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip()
        comfy_aimdo_version = version("comfy-aimdo")
        _require_version(
            comfyui_version.__version__,
            EXPECTED_COMFYUI_VERSION,
            "ComfyUI version",
        )
        _require_version(comfyui_commit, EXPECTED_COMFYUI_COMMIT, "ComfyUI commit")
        _require_version(torch.__version__, EXPECTED_TORCH_VERSION, "PyTorch version")
        _require_version(
            torch.version.cuda or "",
            EXPECTED_TORCH_CUDA_VERSION,
            "PyTorch CUDA version",
        )
        _require_version(
            comfy_aimdo_version,
            EXPECTED_COMFY_AIMDO_VERSION,
            "comfy-aimdo version",
        )
        torch.set_num_threads(1)
        devices = model_management.get_all_torch_devices()
        if not comfy_aimdo.control.init_devices((device.index, 0) for device in devices):
            raise RuntimeError("failed to initialize comfy-aimdo devices")
        model_patcher.CoreModelPatcher = model_patcher.ModelPatcherDynamic
        memory_management.aimdo_enabled = True

        sampler = ProcessMemorySampler(args.sample_interval_ms / 1000.0)
        sampler.__enter__()
        model = comfy.sd.load_diffusion_model(str(args.checkpoint))
        model_management.load_models_gpu([model])
        diffusion_model = model.model.diffusion_model
        blocks = list(diffusion_model.blocks)
        if not 0 <= args.block_index < len(blocks):
            raise ValueError(f"block index must be within [0, {len(blocks)})")
        block = blocks[args.block_index]
        device = torch.device(model.load_device)
        dtype = torch.bfloat16
        hidden_features = int(diffusion_model.hidden_size)
        heads = int(block.attn.heads)
        head_dim = int(block.attn.head_dim)
        attention_features = heads * head_dim

        modules = []
        seen = set()
        for module in block.modules():
            if not hasattr(module, "_v") or id(module) in seen:
                continue
            seen.add(id(module))
            modules.append(module)
        registerable_bytes = sum(
            memory_management.vram_aligned_size([module.weight, module.bias]) for module in modules
        )
        prepare_stream = comfy_ops.cast_modules_with_vbar(
            modules,
            None,
            device,
            None,
            True,
        )
        if not model_management.args.fast_disk:
            model_management.ensure_pin_registerable(registerable_bytes)
        if prepare_stream is not None:
            prepare_stream.synchronize()

        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        hidden_a = torch.empty(
            (args.tokens, hidden_features),
            dtype=dtype,
        ).pin_memory()
        hidden_a.normal_(generator=generator)
        hidden_b = torch.empty_like(hidden_a, pin_memory=True) if args.mode == "recompute" else None
        sequence_meta = H3SequenceMeta(cu_seqlens=torch.tensor([0, args.tokens], dtype=torch.int32))
        attention_config = StreamingAttentionConfig(
            backend="triton",
            q_chunk_tokens=args.q_chunk_tokens,
            kv_chunk_tokens=args.kv_chunk_tokens,
            output_mode="device_consumer",
        )
        attention_plan = build_plan(
            q_heads=heads,
            kv_heads=heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
            max_q_tokens=args.tokens,
            max_kv_tokens=args.tokens,
            config=attention_config,
        )

        qkv_module = block.attn.qkv_proj
        if qkv_module.quant_format != "int8_tensorwise":
            raise RuntimeError(f"expected int8_tensorwise QKV, got {qkv_module.quant_format}")
        active_qkv: dict[str, Any] = {}

        @contextmanager
        def qkv_lease():
            weight, bias, offload = comfy_ops.cast_bias_weight(
                qkv_module,
                input=None,
                dtype=qkv_module.weight.dtype,
                device=device,
                bias_dtype=dtype,
                offloadable=True,
                compute_dtype=dtype,
                want_requant=True,
            )
            if not hasattr(weight, "_qdata") or not hasattr(weight, "_params"):
                raise RuntimeError("QKV lease did not produce a comfy-kitchen quantized weight")
            active_qkv.update(weight=weight, bias=bias)
            try:
                yield
            finally:
                active_qkv.clear()
                comfy_ops.uncast_bias_weight(qkv_module, weight, bias, offload)

        def project_rows(tile, row_start, row_stop):
            weight = active_qkv.get("weight")
            if weight is None:
                raise RuntimeError("QKV projector called outside its weight lease")
            params = weight._params
            qdata = weight._qdata
            if qdata.ndim != 2 or qdata.shape[0] != 3 * attention_features:
                raise RuntimeError(f"unexpected QKV weight shape: {tuple(qdata.shape)}")
            scale = params.scale
            if scale.numel() != 1:
                if scale.shape[0] != qdata.shape[0]:
                    raise RuntimeError(f"unexpected QKV scale shape: {tuple(scale.shape)}")
                scale = scale[row_start:row_stop].contiguous()
            bias = active_qkv["bias"]
            if bias is not None:
                bias = bias[row_start:row_stop].contiguous()
            return comfy_kitchen.int8_linear(
                tile.contiguous(),
                qdata[row_start:row_stop].contiguous(),
                scale,
                bias,
                out_dtype=tile.dtype,
                convrot=bool(params.convrot),
                convrot_groupsize=int(params.convrot_groupsize),
            )

        def project_qkv(tile, start, stop):
            del start, stop
            projected = project_rows(tile, 0, 3 * attention_features)
            q, k, v = projected.split(attention_features, dim=-1)
            tokens = tile.shape[0]
            return (
                q.view(tokens, heads, head_dim),
                k.view(tokens, heads, head_dim),
                v.view(tokens, heads, head_dim),
            )

        def project_q(tile, destination_q, start, stop):
            del start, stop
            q = project_rows(tile, 0, attention_features)
            destination_q.copy_(q.view(tile.shape[0], heads, head_dim))

        def project_kv(tile, destination_k, destination_v, start, stop):
            del start, stop
            kv = project_rows(tile, attention_features, 3 * attention_features)
            k, v = kv.split(attention_features, dim=-1)
            destination_k.copy_(k.view(tile.shape[0], heads, head_dim))
            destination_v.copy_(v.view(tile.shape[0], heads, head_dim))

        validation_tokens = min(args.validation_tokens, args.tokens)
        with qkv_lease():
            validation_hidden = hidden_a[:validation_tokens].to(device, non_blocking=True)
            full = qkv_module(validation_hidden)
            q_only = project_rows(validation_hidden, 0, attention_features)
            kv_only = project_rows(
                validation_hidden,
                attention_features,
                3 * attention_features,
            )
            sliced = torch.cat((q_only, kv_only), dim=-1)
            torch.testing.assert_close(sliced, full, atol=0.0, rtol=0.0)
            validation_max_abs = float((sliced.float() - full.float()).abs().max().item())
        del validation_hidden, full, q_only, kv_only, sliced
        torch.cuda.synchronize(device)

        def _lease(module):
            lease = getattr(module, "computation_lease", None)
            return nullcontext(module) if lease is None else lease(allow_preparing=True)

        @contextmanager
        def consumer_lease():
            with ExitStack() as stack:
                for module in (block.attn.out_proj, block.mlp.fc1, block.mlp.fc2):
                    stack.enter_context(_lease(module))
                yield

        def attention_epilogue(attention, residual_host, start, stop):
            update = block.attn.out_proj(attention)
            residual = residual_host[start:stop].to(device, non_blocking=True)
            return residual.add_(update)

        def mlp(post_attention, start, stop):
            del start, stop
            return post_attention.add_(block.mlp(block.norm2(post_attention)))

        ops = H3BlockOps(
            attention_epilogue=attention_epilogue,
            mlp=mlp,
            consumer_lease=consumer_lease,
        )
        if args.mode == "materialized":
            projected_attention = ProjectedAttentionRunner(
                attention_plan,
                attention_config,
                ProjectionPipelineConfig(
                    projection_chunk_tokens=args.qkv_tile_tokens,
                    num_projection_buffers=2,
                ),
            )
            runner = H3MaterializedRunner(
                projected_attention,
                hidden_features=hidden_features,
                mlp_chunk_tokens=args.mlp_tile_tokens,
            )
            projection = H3MaterializedProjection(project_qkv, weight_lease=qkv_lease)

            def run_once(stats):
                return runner.run_block_(
                    hidden_a,
                    sequence_meta,
                    projection,
                    ops,
                    softmax_scale=head_dim**-0.5,
                    stats=stats,
                )

            def current_hidden():
                return hidden_a

            logical_host_activation_bytes = (
                args.tokens * (hidden_features + 3 * attention_features) * hidden_a.element_size()
            )
        else:
            assert hidden_b is not None
            recomputed_attention = RecomputedAttentionRunner(
                attention_plan,
                hidden_features=hidden_features,
                attention_config=attention_config,
            )
            runner = H3RecomputeRunner(
                recomputed_attention,
                mlp_chunk_tokens=args.mlp_tile_tokens,
            )
            projection = H3RecomputeProjection(
                project_q,
                project_kv,
                weight_lease=qkv_lease,
            )
            buffers = [hidden_a, hidden_b]

            def run_once(stats):
                result_hidden = runner.run_block(
                    buffers[0],
                    buffers[1],
                    sequence_meta,
                    projection,
                    ops,
                    softmax_scale=head_dim**-0.5,
                    stats=stats,
                )
                buffers.reverse()
                return result_hidden

            def current_hidden():
                return buffers[0]

            logical_host_activation_bytes = (
                args.tokens * 2 * hidden_features * hidden_a.element_size()
            )

        def reset_current_hidden():
            generator.manual_seed(args.seed)
            current_hidden().normal_(generator=generator)

        for _ in range(args.warmup):
            reset_current_hidden()
            run_once(H3DiTStats())
            torch.cuda.synchronize(device)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        records = []
        for repeat in range(args.repeats):
            reset_current_hidden()
            stats = H3DiTStats()
            started = time.perf_counter()
            run_once(stats)
            torch.cuda.synchronize(device)
            records.append(
                {
                    "repeat": repeat,
                    "wall_seconds": time.perf_counter() - started,
                    "stats": stats.as_dict(),
                }
            )

        output_hidden = current_hidden()
        signature_indices = sorted(
            {0, args.tokens // 4, args.tokens // 2, 3 * args.tokens // 4, args.tokens - 1}
        )
        output_signature = {
            str(index): output_hidden[index, :8].float().tolist() for index in signature_indices
        }
        sampled_output_is_finite = all(
            math.isfinite(value) for values in output_signature.values() for value in values
        )
        if not sampled_output_is_finite:
            raise RuntimeError("sampled block output contains non-finite values")
        walls = [record["wall_seconds"] for record in records]
        sampler.__exit__()
        process_peak_rss_bytes = sampler.peak_rss_bytes
        nvml_process_peak_bytes = sampler.peak_vram_bytes
        sampler = None
        properties = torch.cuda.get_device_properties(device)
        result.update(
            status="success",
            environment={
                "hostname": platform.node(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "comfyui_version": comfyui_version.__version__,
                "comfyui_commit": comfyui_commit,
                "comfy_aimdo_version": comfy_aimdo_version,
                "comfyui_root": str(args.comfyui_root),
                "gpu": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "pid": os.getpid(),
            },
            model={
                "checkpoint": str(args.checkpoint),
                "block_index": args.block_index,
                "hidden_features": hidden_features,
                "heads": heads,
                "head_dim": head_dim,
                "attention_features": attention_features,
                "qkv_quant_format": qkv_module.quant_format,
                "qkv_weight_shape": list(qkv_module.weight.shape),
                "prepared_vbar_module_count": len(modules),
                "prepared_vbar_registerable_bytes": registerable_bytes,
            },
            projection_validation={
                "tokens": validation_tokens,
                "atol": 0.0,
                "rtol": 0.0,
                "max_abs_difference": validation_max_abs,
            },
            plan={key: value for key, value in vars(runner.plan).items()},
            records=records,
            output_signature=output_signature,
            output_validation={
                "sampled_values": sum(len(values) for values in output_signature.values()),
                "sampled_all_finite": sampled_output_is_finite,
            },
            summary={
                "wall_median_seconds": statistics.median(walls),
                "wall_mean_seconds": statistics.mean(walls),
                "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "process_peak_rss_bytes": process_peak_rss_bytes,
                "nvml_process_peak_bytes": nvml_process_peak_bytes,
                "logical_host_activation_bytes": logical_host_activation_bytes,
            },
        )
    except Exception as error:  # noqa: BLE001 - benchmark records failures as JSON
        try:
            import torch

            out_of_memory = isinstance(error, torch.OutOfMemoryError)
        except ImportError:
            out_of_memory = False
        result["status"] = "oom" if out_of_memory else "runtime_error"
        result["failure_message"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    finally:
        if sampler is not None:
            sampler.__exit__()
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "success":
        raise SystemExit(1)


def main() -> None:
    args = _parse_args()
    if args.mode == "compare":
        _run_comparison(args)
    else:
        _run_mode(args)


if __name__ == "__main__":
    main()
