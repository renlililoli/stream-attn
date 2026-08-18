#!/usr/bin/env bash
set -euo pipefail

output_prefix=${1:-seqattn-profile}
shift || true

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  --output="${output_prefix}" \
  python -m seqattn.benchmark --nvtx "$@"
