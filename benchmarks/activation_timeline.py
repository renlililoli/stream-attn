"""Generate a full H3 block timeline from explicit, initially synthetic calibration rates."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from seqattn_core.estimation import (
    H3BlockShape,
    H3CallbackConfig,
    H3DeviceProfile,
    H3ExecutionConfig,
    MemoryPool,
    RateProfile,
    build_h3_block_execution,
    estimate_activation_memory,
    write_timeline_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("h3-activation-timeline.html"))
    parser.add_argument(
        "--profile-json", type=Path, help="Load an explicit H3 device/operator profile"
    )
    parser.add_argument("--tokens", type=int, default=81159)
    parser.add_argument("--fa-tflops", type=float, default=200.0)
    parser.add_argument("--gemm-tflops", type=float, default=120.0)
    parser.add_argument("--vector-gelements", type=float, default=100.0)
    parser.add_argument("--h2d-gbps", type=float, default=50.0)
    parser.add_argument("--d2h-gbps", type=float, default=40.0)
    parser.add_argument("--d2d-gbps", type=float, default=500.0)
    parser.add_argument("--callback-variant", choices=("block25", "modulated"), default="modulated")
    parser.add_argument(
        "--linear-memory", choices=("dense", "int8_eager", "profile"), default="int8_eager"
    )
    parser.add_argument("--weight-scale", choices=("channel", "tensor"), default="channel")
    parser.add_argument("--device-label", default="Synthetic accelerator (not a hardware claim)")
    parser.add_argument("--provenance", default="Synthetic example rates; not measured")
    parser.add_argument("--target-fraction", type=float, default=0.95)
    args = parser.parse_args()
    shape = H3BlockShape(
        (args.tokens,), hidden_features=5376, ffn_features=14336, heads=56, head_dim=128
    )

    def rate(name, value, resources, kind="compute"):
        return RateProfile(
            name, value, resources, kind=kind, latency_seconds=5e-6, provenance=args.provenance
        )

    profile = H3DeviceProfile.from_rates(
        args.device_label,
        device_pool=MemoryPool("accelerator.memory", allocation_alignment_bytes=512),
        host_pool=MemoryPool("host.DRAM"),
        attention=rate("attention FLOP/s", args.fa_tflops * 1e12, ("compute",)),
        gemm=rate("GEMM FLOP/s", args.gemm_tflops * 1e12, ("compute",)),
        vector=rate("vector elements/s", args.vector_gelements * 1e9, ("compute",)),
        h2d=rate("H2D byte/s", args.h2d_gbps * 1e9, ("H2D",), "io"),
        d2h=rate("D2H byte/s", args.d2h_gbps * 1e9, ("D2H",), "io"),
        d2d=rate("D2D byte/s", args.d2d_gbps * 1e9, ("compute",)),
    )
    if args.profile_json is not None:
        profile = H3DeviceProfile.from_dict(json.loads(args.profile_json.read_text()))
    callbacks = H3CallbackConfig(
        variant=args.callback_variant,
        linear_memory=args.linear_memory,
        per_channel_weight_scale=args.weight_scale == "channel",
    )
    baseline = H3ExecutionConfig(3840, 4096, 4096, 4096)
    candidates = [
        build_h3_block_execution(
            shape,
            replace(baseline, ffn_tile_tokens=ffn, execution_mode=mode),
            profile,
            callbacks=callbacks,
        )
        for mode in ("materialized", "recompute")
        for ffn in (2048, 4096, 8192, 16384)
    ]
    result = estimate_activation_memory(
        candidates,
        objective_pool=profile.device_pool.name,
        target_throughput_fraction=args.target_fraction,
        objective_owners=frozenset({"operator", "callback", "caller"}),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_timeline_report(result, args.output, json_path=args.output.with_suffix(".json"))
    print(f"Wrote {args.output} and {args.output.with_suffix('.json')}")
    for item in result.candidates:
        selected = " *" if item is result.selected else ""
        print(
            f"{item.trace.name}: {item.trace.duration_seconds:.6f} s, {item.peak_bytes / 2**20:.3f} MiB, "
            f"FFN calls={len(item.trace.metadata['ffn_ranges'])}{selected}"
        )


if __name__ == "__main__":
    main()
