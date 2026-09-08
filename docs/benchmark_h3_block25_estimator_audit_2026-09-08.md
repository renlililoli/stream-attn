# H3 block-25 evidence and estimator mapping audit

Audit date: 2026-09-08. Dataset date: 2026-08-25.

Status: historical findings about the initial generic template. The replacement
[full H3 model](activation_memory_estimation.md) now implements the carry schedule
and explicit callback tensors. See the
[full-model validation](benchmark_h3_full_estimator_2026-09-08.md) for current results.

The initial estimator validation missed a parent-project dataset because its
folder is named `seqattn_single_block25_81159_20260825`, without `h3` in the name.
There are **11 JSON files, two Nsight SQLite exports and two .nsys-rep captures**
under:

```text
../../workspace/benchmarks/results/seqattn_single_block25_81159_20260825/
```

The parent does contain complete block timings, native CUDA-event stage
timings, projection/MLP tile sweeps, and Torch allocated/reserved peaks for
seven streamed variants. This corrects the earlier incomplete evidence search.
The [pure-attention validation](benchmark_activation_estimator_5090_2026-09-08.md)
remains valid within its stated scope; its error percentages do not characterize
this complete H3 block.

The machine-readable [audit JSON](experiments/h3_block25_81159_estimator_audit_20260908/audit.json)
retains every raw repeat, file hashes, available memory fields, stage timings,
profile metadata and the concrete FFN schedule mismatch below.

## Measurements found

All cases describe block 25, 81,159 tokens, hidden width 5,376, 56 heads,
head dimension 128 and BF16 activations. Streamed attention uses Q=3840 and
KV=4096. SQLite metadata identifies an RTX 5090 at PCI bus `0000:e1:00.0`.

| Unprofiled case | QKV tile | MLP tile | Wall median | Torch allocated peak |
|---|---:|---:|---:|---:|
| Native FA4 | n/a | n/a | 1.291733 s | Not recorded |
| Streaming baseline | 2048 | 2048 | 1.535516 s | Not recorded |
| Streaming | 2048 | 4096 | 1.504778 s | 2048.736 MiB |
| Streaming | 2048 | 8192 | 1.490434 s | 2984.346 MiB |
| Streaming | 4096 | 8192 | 1.471253 s | 3026.346 MiB |
| Streaming | 8192 | 8192 | 1.484030 s | 3110.346 MiB |
| Streaming | 4096 | 12288 | 1.462356 s | 4025.611 MiB |
| Streaming | 4096 | 16384 | 1.462620 s | 5147.664 MiB |
| Streaming configuration-file case | 4096 | 4096 | 2.253628 s | 2090.736 MiB |

The configuration-file case ranges from 1.563002 to 2.394845 seconds over its
five repeats. Its variability is retained and must be treated separately from
the stable-looking runs; these records alone do not establish the cause.
The two additional `_profile.json` records are retained for attribution and
excluded from the primary timing table.

`block25_native_fa4.json` also provides independent CUDA-event stage timings.
Its five-repeat stage means are:

| Native stage | Mean GPU time |
|---|---:|
| QKV projection | 124.643 ms |
| Attention | 864.110 ms |
| Output projection | 47.222 ms |
| MLP | 265.035 ms |

These are useful performance inputs. They measure the native full-sequence
shapes, not a saturation curve for 2K/4K/8K GEMM tiles. The native JSON's `plan`
section includes streaming-related fields; those fields must not be taken as
the native GPU GEMM tile sizes.

The captured profiles record CUDA `13.1.1.006`, PyTorch
`2.10.0a0+a36e1d3`, and NGC PyTorch `26.01`. The unprofiled JSONs do not include a
separate environment section. Do not silently combine this experiment with
the previous CUDA 12.8/PyTorch 2.10.0+cu128 attention calibration.

## A concrete mismatch in the initial generic block template

The H3 consumer accumulates post-attention tokens across resident-Q chunks.
The generic `build_block_execution()` template currently calls its FFN inside
each Q chunk. The observed and modeled call counts consequently differ:

| MLP tile | Recorded H3 MLP calls | Generic template calls |
|---|---:|---:|
| 2048 | 40 | 43 |
| 4096 | 20 | 22 |
| 8192 | 10 | 22 |
| 12288 | 7 | 22 |
| 16384 | 5 | 22 |

For MLP tiles larger than Q=3840, the initial template's actual FFN operation
size is capped by Q. Increasing its requested FFN capacity grows output slots,
but does not reproduce H3's larger carried FFN invocation and temporaries.
This is a template-to-runtime mapping limitation, not an error in the physical
allocation event sweep.

The two QKV=4096 cases with MLP=8192 and MLP=16384 provide a useful incremental
memory check with fixed attention/projection shape:

| Growth when MLP tile doubles | Bytes | MiB |
|---|---:|---:|
| Recorded Torch allocated peak growth | 2,224,363,008 | 2121.318 |
| Recorded H3 core workspace growth | 264,241,152 | 252 |
| Current template output-slot growth | 176,160,768 | 168 |
| H3 carry-buffer growth absent from the template | 88,080,384 | 84 |

The initial per-Q template only accounts for 168 MiB of capacity growth here.
H3's core carry accounts for another 84 MiB; the much larger measured Torch
increment requires modeling the consumer's actual large-tile temporaries and
operator allocation behavior. The 92.08% shortfall obtained by comparing
168 MiB with 2121.318 MiB is an **incremental growth mismatch**, not an absolute
whole-block activation error. Fixed weights, scratch and ownership still need
explicit accounting for an absolute peak comparison.

Therefore this dataset supports a real full-block validation effort, and also
shows why passing pure-attention memory checks cannot establish H3 accuracy.
The H3 mapping must represent carry batching and callback lifetimes before
using this template to select H3 FFN tiles.

## What the SQLite exports can and cannot establish

The streaming capture contains 18,060 kernel records and 6,720 memcpy records;
the native capture contains 2,025 and 175 respectively. They can establish
GPU identity, stream overlap and operation attribution. Neither export contains
the allocation-event tables checked by this audit. They are not complete
per-tensor allocation/free traces.

Use the unprofiled JSON wall/CUDA-event measurements for latency comparisons.
Use SQLite for scheduling attribution, and keep supplied allocation declarations
separate from measured kernel/copy timestamps. No Nsight durations were promoted
to primary benchmark latency in this audit.

## Reproduce

```bash
python benchmarks/audit_h3_block_evidence.py \
  --directory ../../workspace/benchmarks/results/seqattn_single_block25_81159_20260825 \
  --output /tmp/h3-block25-estimator-audit.json
```

The audit uses only Python's standard library and opens SQLite read-only. It
does not load checkpoints, import framework adapters or run accelerator kernels.
