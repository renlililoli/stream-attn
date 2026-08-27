# MiniMax-H3 QKV recompute 2K/4K profile

Date: 2026-08-27

## Scope

This calibration compares `C_proj=2048, C_mlp=2048` with
`C_proj=4096, C_mlp=4096` for the standalone block-25 INT8 ConvRot benchmark.
The attention plan is fixed at `Q=16384` and `KV=4096`, so the experiment does
not mix Q/KV attention planning with projection or MLP tile calibration.

The run used physical GPU 2, an NVIDIA GeForce RTX 5090, without explicit NUMA
binding. The authoritative Community environment was ComfyUI `0.30.0` commit
`9a9fdb10ed144ce760d9682cb247526ea23cc525`, PyTorch `2.10.0+cu128`, CUDA
`12.8`, and comfy-aimdo `0.4.11`.

Each formal mode ran in an independent process with one warmup and three
measured repeats. Every repeat restored the same synthetic pinned BF16 input
outside the timed region. Q-only and K/V-only projection matched the complete
QKV operator exactly, and the materialized and recompute sampled outputs were
finite and identical.

## Formal wall time

| Mode | 2K median | 4K median | Change |
|---|---:|---:|---:|
| Materialized | 11.1793 s | 11.0222 s | -1.41% |
| Recompute | 15.8284 s | 15.7279 s | -0.64% |
| Recompute / materialized | 1.4159x | 1.4269x | +0.0110x |

The 4K tiles improve both modes, but materialized benefits from both the larger
QKV projection tile and the larger MLP tile. Recompute benefits only from the
MLP tile because its Q and K/V projection ranges are fixed by the attention
plan, not `C_proj`.

## Memory effect

| Mode | Metric | 2K | 4K | Change |
|---|---|---:|---:|---:|
| Materialized | estimated workspace | 1,090,519,040 B | 1,200,619,520 B | +10.10% |
| Materialized | Torch allocated peak | 2,716,321,280 B | 2,873,559,040 B | +5.79% |
| Recompute | estimated workspace | 1,222,639,616 B | 1,288,699,904 B | +5.40% |
| Recompute | Torch allocated peak | 2,875,701,248 B | 2,960,558,080 B | +2.95% |

The additional bounded workspace is small relative to the RTX 5090 capacity
and does not change logical host activation. For this dedicated H3 deployment,
4K is the better default. This result does not change the generic core default
for other models or devices.

## Nsight Systems breakdown

Nsight Systems captured one stable measured block after warmup using the CUDA
profiler API capture range. NVTX GPU projected times may overlap across CUDA
streams, so transfer and compute rows must not be added as if they were
strictly serial. Same-stream projection, attention, epilogue, and MLP rows are
directly useful for attribution.

### 2K to 4K operator change

| GPU projected range | 2K | 4K | Change |
|---|---:|---:|---:|
| Materialized complete QKV projection | 0.5276 s | 0.4560 s | -13.57% |
| Materialized MLP | 1.0612 s | 0.9448 s | -10.97% |
| Recompute MLP | 1.0594 s | 0.9434 s | -10.94% |

The complete QKV callback count falls from 129 to 65, and the MLP callback
count falls from 129 to 65. The larger tiles save about 72 ms of QKV projection
GPU time and 116 ms of MLP GPU time in the profiled block. Pipeline overlap and
minor run-to-run variation reduce the formal wall-time improvement to 157 ms
for materialized and 101 ms for recompute.

### 4K materialized versus recompute

| GPU projected range | Materialized | Recompute |
|---|---:|---:|
| Fused online-softmax update | 9.0429 s | 8.9816 s |
| Complete materialized QKV projection | 0.4560 s | n/a |
| Recompute Q-only projection | n/a | 0.1630 s |
| Recompute K/V-only projection | n/a | 5.5688 s |
| H3 attention epilogue | 0.2371 s | 0.2375 s |
| H3 MLP | 0.9448 s | 0.9434 s |
| Attention finalize | 0.0069 s | 0.0071 s |

CUDA memory-operation totals were:

| Transfer | Materialized | Recompute |
|---|---:|---:|
| Host to device | 3.8144 s | 1.4472 s |
| Device to host | 0.5052 s | 0.0889 s |

The attention kernel, MLP, epilogue, and finalize times are effectively equal.
Recompute also transfers less data than materialized. The slowdown is the
5.57 seconds spent regenerating K/V on the compute stream, not attention or
MLP.

At `tokens=262720`, `Q=16384`, and `KV=4096`, there are 17 Q chunks and 65 K/V
tiles per complete sequence. Recompute therefore invokes the K/V projector
`17 * 65 = 1105` times. Materialized projects the complete sequence once, then
streams the host K/V backing. The repeated K/V projection accounts for more
than the net wall-time gap because recompute recovers part of that cost through
lower H2D/D2H traffic and by avoiding the materialized preprojection stage.

## Conclusion

- Use `C_proj=4096` and `C_mlp=4096` for this MiniMax-H3 RTX 5090 deployment.
- Do not expect `C_proj` to improve recompute; recompute has no projection
  subtile and its callback ranges remain `Q=16384` and `KV=4096`.
- The next meaningful recompute tuning variable is resident Q size. A larger Q
  range reduces the number of complete K/V rescans and therefore both K/V
  projection calls and hidden H2D traffic. It must be swept against the larger
  attention workspace rather than inferred from the 4K projection result.
- Reworking MLP or the attention update kernel is not the priority for the
  current gap; their measured times already match between storage policies.

Generated JSON, SQLite exports, and `.nsys-rep` files are experiment artifacts
under `/tmp` and are not committed.
