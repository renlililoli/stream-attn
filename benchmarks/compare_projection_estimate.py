"""Compare only the materialization phase against an unprofiled measurement.

Reuses the H3 builder's phase implementation. Attention/FFN are deliberately not
built, so their unmeasured tail shapes do not require invented calibration samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from seqattn_core.estimation import (
    H3BlockShape,
    H3CallbackConfig,
    H3DeviceProfile,
    H3ExecutionConfig,
    H3WeightPolicy,
    schedule_execution,
)
from seqattn_core.estimation.h3.callbacks import H3Callbacks
from seqattn_core.estimation.h3.graph import H3Graph
from seqattn_core.estimation.h3.pipeline import _allocate, _materialize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrent-pack", action="store_true")
    args = parser.parse_args()
    configuration = json.loads(args.configuration.read_text())
    measurement = json.loads(args.measurement.read_text())
    shape = H3BlockShape(**{**configuration["shape"], "segments": (measurement["tokens"],)})
    config = H3ExecutionConfig(**configuration["execution"])
    if config.projection_tile_tokens != measurement["projection_tile_tokens"]:
        raise ValueError("measured and predicted projection tiles differ")
    if config.num_projection_buffers != measurement["slots"]:
        raise ValueError("measured and predicted slot counts differ")
    # The real projection benchmark prepares modulation/weights before timing.
    callbacks = H3CallbackConfig(**{**configuration["callbacks"], "compute_modulation": False})
    data = json.loads(args.profile.read_text())
    if args.concurrent_pack:
        data["projection_pack_resources"] = ("projection.pack",)
    profile = H3DeviceProfile.from_dict(data)
    profile.validate_shape(shape)
    graph = H3Graph(shape, config, profile, callbacks, H3WeightPolicy())
    _allocate(graph)
    cb = H3Callbacks(graph)
    cb.prepare()
    _materialize(graph, cb, None)
    trace = schedule_execution(graph.finish("H3 projection phase"))
    result = {
        "tokens": shape.tokens,
        "predicted_seconds": trace.duration_seconds,
        "actual_median_seconds": measurement["median_seconds"],
        "relative_error_percent": 100
        * (trace.duration_seconds / measurement["median_seconds"] - 1),
        "concurrent_pack": args.concurrent_pack,
        "rates_refitted": False,
        "scope": "projection phase only; no attention/FFN or full-block accuracy claim",
        "source_profile": str(args.profile),
        "source_configuration": str(args.configuration),
        "profile_resolutions": trace.metadata["profile_resolutions"],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
