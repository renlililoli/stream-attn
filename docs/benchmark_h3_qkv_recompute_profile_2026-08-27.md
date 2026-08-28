# MiniMax-H3 QKV recompute 2K/4K profile

Date: 2026-08-27

Status: current calibration evidence for the H3 defaults documented in
[`design_dit_mlp_chunk_model.md`](design_dit_mlp_chunk_model.md).

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

## Formal measurements

| Mode | 2K mean | 2K median | 4K mean | 4K median | Median change |
|---|---:|---:|---:|---:|---:|
| Materialized | 11.179503 s | 11.179313 s | 11.018168 s | 11.022211 s | -1.41% |
| Recompute | 15.831779 s | 15.828434 s | 15.729613 s | 15.727886 s | -0.64% |

The recompute/materialized median ratio is `1.4159x` at 2K and `1.4269x` at
4K.

The unrounded measured wall times were:

| Configuration | Repeat 0 | Repeat 1 | Repeat 2 |
|---|---:|---:|---:|
| 2K materialized | 11.182602211 s | 11.179312767 s | 11.176594866 s |
| 2K recompute | 15.818961724 s | 15.828433870 s | 15.847941785 s |
| 4K materialized | 11.003184426 s | 11.029109673 s | 11.022211228 s |
| 4K recompute | 15.727885503 s | 15.726514061 s | 15.734438875 s |

The 4K tiles improve both modes, but materialized benefits from both the larger
QKV projection tile and the larger MLP tile. Recompute benefits only from the
MLP tile because its Q and K/V projection ranges are fixed by the attention
plan, not `C_proj`.

## Memory measurements

| Metric | 2K materialized | 2K recompute | 4K materialized | 4K recompute |
|---|---:|---:|---:|---:|
| Estimated workspace | 1,090,519,040 B | 1,222,639,616 B | 1,200,619,520 B | 1,288,699,904 B |
| Torch allocated peak | 2,716,321,280 B | 2,875,701,248 B | 2,873,559,040 B | 2,960,558,080 B |
| Torch reserved peak | 3,797,942,272 B | 3,300,917,248 B | 3,506,438,144 B | 3,363,831,808 B |
| PID NVML peak | 5,140,119,552 B | 4,643,094,528 B | 4,848,615,424 B | 4,706,009,088 B |
| Process RSS peak | 19,448,967,168 B | 10,797,760,512 B | 19,446,194,176 B | 10,797,703,168 B |
| Logical host activation | 14,123,827,200 B | 5,649,530,880 B | 14,123,827,200 B | 5,649,530,880 B |

Moving from 2K to 4K changes estimated workspace by `+10.10%` and Torch
allocated peak by `+5.79%` for materialized. For recompute, the corresponding
changes are `+5.40%` and `+2.95%`.

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

### Complete NVTX GPU projected data

| GPU projected range | 2K materialized | 2K recompute | 4K materialized | 4K recompute |
|---|---:|---:|---:|---:|
| Complete recomputed attention | n/a | 16.110651 s | n/a | 15.980300 s |
| Fused online-softmax update | 9.047553 s | 8.985275 s | 9.042882 s | 8.981593 s |
| Complete materialized QKV projection | 0.527557 s | n/a | 0.455966 s | n/a |
| Recompute Q-only projection | n/a | 0.162772 s | n/a | 0.163012 s |
| Recompute K/V-only projection | n/a | 5.570879 s | n/a | 5.568778 s |
| H3 attention epilogue | 0.236714 s | 0.241496 s | 0.237078 s | 0.237508 s |
| H3 MLP | 1.061202 s | 1.059353 s | 0.944820 s | 0.943448 s |
| Device output consumer | 1.316805 s | 1.309567 s | 1.222730 s | 1.200605 s |
| Attention finalize | 0.006940 s | 0.007101 s | 0.006930 s | 0.007114 s |
| Materialized projection hidden H2D | 0.143980 s | n/a | 0.150694 s | n/a |
| Materialized projection QKV D2H | 0.623935 s | n/a | 0.575651 s | n/a |
| Materialized attention Q H2D | 0.112544 s | n/a | 0.124803 s | n/a |
| Materialized attention K/V H2D | 3.437015 s | n/a | 3.463104 s | n/a |
| Recompute hidden H2D | n/a | 1.454790 s | n/a | 1.370812 s |

CUDA memory-operation totals were:

| Transfer | 2K materialized | 2K recompute | 4K materialized | 4K recompute |
|---|---:|---:|---:|---:|
| Host to device | 3.768962 s | 1.535125 s | 3.814383 s | 1.447236 s |
| Device to host | 0.480234 s | 0.091123 s | 0.505188 s | 0.088928 s |
| Device to device | 0.037189 s | 0.194613 s | 0.003182 s | 0.172414 s |

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

## Validation data

Both formal configurations completed without OOM or runtime failure. For each
mode and tile size:

- the 16-token Q-only plus K/V-only ConvRot projection had
  `max_abs_difference=0.0` against complete QKV, with `atol=0` and `rtol=0`;
- all 40 sampled output values were finite;
- materialized and recompute sampled output signatures were exactly equal;
- recompute issued 17 Q projector calls and 1,105 K/V projector calls per
  measured block;
- recompute allocated no host Q/K/V backing.

## Artifacts

The experiment produced these uncommitted artifacts:

```text
/tmp/minimax_h3_convrot_qkv_recompute_formal_gpu2_reset.json
/tmp/minimax_h3_convrot_qkv_recompute_formal_gpu2_4k.json
/tmp/seqattn_h3_2k_materialized_profile.json
/tmp/seqattn_h3_2k_recompute_profile.json
/tmp/seqattn_h3_4k_materialized_profile.json
/tmp/seqattn_h3_4k_recompute_profile.json
/tmp/seqattn_h3_2k_materialized.nsys-rep
/tmp/seqattn_h3_2k_recompute.nsys-rep
/tmp/seqattn_h3_4k_materialized.nsys-rep
/tmp/seqattn_h3_4k_recompute.nsys-rep
/tmp/seqattn_h3_2k_materialized.sqlite
/tmp/seqattn_h3_2k_recompute.sqlite
/tmp/seqattn_h3_4k_materialized.sqlite
/tmp/seqattn_h3_4k_recompute.sqlite
```

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
