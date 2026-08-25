# MiniMax-H3 DiT MLP sequence-streaming chunk model

Date: 2026-08-25

## Scope

This note derives a chunk-size model for the MiniMax-H3 DiT SwiGLU MLP when:

- model weights remain resident in GPU memory;
- the full sequence hidden state is backed by pinned CPU memory;
- each GPU chunk executes `norm -> AdaLN -> FC1 -> SwiGLU -> FC2 -> gate/residual`;
- the completed hidden chunk is copied back to CPU memory;
- H2D, compute, and D2H use independent CUDA streams and persistent buffers;
- inference uses no backward graph or dropout.

The main result is that bandwidth and GEMM throughput alone determine whether
PCIe traffic is asymptotically hideable, but they do not determine a unique
minimum chunk size. A concrete minimum additionally requires copy latency,
GPU fixed overhead, HBM traffic, and the chunk-dependent GEMM efficiency
curve.

## Notation

| Symbol | Meaning |
|---|---|
| `C` | MLP chunk size in tokens |
| `H` | model hidden size |
| `F` | SwiGLU intermediate size |
| `s` | activation element size in bytes |
| `s_w` | weight element size in bytes |
| `P(C)` | effective combined FC1+FC2 throughput at chunk `C`, FLOP/s |
| `P` | plateau effective combined FC1+FC2 throughput, FLOP/s |
| `B_in` | effective pinned-CPU-to-GPU bandwidth, byte/s |
| `B_out` | effective GPU-to-pinned-CPU bandwidth, byte/s |
| `B_hbm` | effective GPU HBM bandwidth available to the MLP, byte/s |
| `alpha_in` | fixed H2D time per chunk |
| `alpha_out` | fixed D2H time per chunk |
| `alpha_gpu` | fixed GPU busy time per MLP chunk |
| `W` | FC1+FC2 weight bytes read per chunk |
| `m` | activation and temporary HBM bytes per token |
| `A` | required token alignment for the selected GEMM implementation |

`B_in` should be an end-to-end measured bandwidth. It includes the limiting
effect of CPU DRAM, NUMA placement, PCIe, the CUDA copy engine, and pinned-memory
behavior. A PCIe link-rate specification is not sufficient.

## MLP operation count

For MiniMax-H3, the tile-local MLP is:

```text
[C, H]
  -> FC1 [H, 2F]
  -> split gate/up
  -> SiLU(gate) * up [C, F]
  -> FC2 [F, H]
  -> gate/residual [C, H]
```

Counting one fused multiply-add as two FLOPs:

```text
FC1 FLOPs = 2 * C * H * 2F = 4CHF
FC2 FLOPs = 2 * C * F * H  = 2CHF
```

The two GEMMs therefore execute:

```text
F_mlp(C) = 6CHF
```

Define the GEMM work per token as:

```text
f = 6HF
```

The SiLU, multiply, normalization, AdaLN, and residual kernels add work, but
the two GEMMs dominate the first-order model. Their measured time can be added
to `alpha_gpu` or modeled separately when higher accuracy is required.

## Pipeline timing model

With pinned CPU buffers and asynchronous copies:

```text
T_h2d(C) = alpha_in  + C * sH / B_in
T_d2h(C) = alpha_out + C * sH / B_out
```

Once the GEMMs have reached their plateau throughput:

```text
T_compute(C) = alpha_gpu + 6CHF / P
```

A first-order HBM roofline term is:

```text
T_hbm(C) = (W + mC) / B_hbm
```

The modeled GPU stage is:

```text
T_gpu(C) = max(T_compute(C), T_hbm(C))
```

For an H2D/compute/D2H ring with independent copy directions, steady-state
time per chunk is:

```text
T_steady(C) = max(T_h2d(C), T_gpu(C), T_d2h(C))
```

Copy traffic is fully hidden in steady state when:

```text
T_gpu(C) >= max(T_h2d(C), T_d2h(C))
```

Pipeline fill and drain remain visible for the first and last chunks.

If the hardware or runtime cannot overlap H2D and D2H, replace the condition
with:

```text
T_gpu(C) >= T_h2d(C) + T_d2h(C)
```

## Why bandwidth and compute alone do not produce a chunk size

Ignoring fixed latency and assuming constant effective throughput, hiding H2D
requires:

```text
6CHF / P >= CsH / B_in
```

After cancelling `CH`:

```text
6F / P >= s / B_in
```

Equivalently:

```text
P <= 6F * B_in / s
```

The chunk size cancels. The same happens for D2H. Therefore bandwidth and
plateau GEMM throughput answer only one question:

> Once GEMM is saturated, is there enough computation per token to cover the
> per-token transfer time?

They do not say at which chunk the GEMM reaches that plateau. The practical
critical chunk is usually determined by GEMM utilization and weight-traffic
amortization rather than by PCIe bandwidth.

## Closed-form PCIe latency threshold

Assume the GPU stage is compute-bound and already achieves constant `P`.
Hiding one copy direction `d` requires:

```text
alpha_gpu + C * 6HF / P >= alpha_d + C * sH / B_d
```

Define the per-token timing margin:

```text
delta_d = 6HF / P - sH / B_d
```

If `delta_d > 0`, the minimum chunk that covers fixed copy latency is:

```text
C_pcie,d = ceil(max(0, alpha_d - alpha_gpu) / delta_d)
```

For independent H2D and D2H engines:

```text
C_pcie = max(C_pcie,in, C_pcie,out)
```

If `delta_d <= 0`, no sufficiently large chunk can hide that direction's
steady-state transfer. The implementation must increase effective compute per
token, reduce transferred bytes, or increase effective bandwidth.

For serialized bidirectional copies with equal bandwidth `B`:

```text
delta_serial = 6HF / P - 2sH / B

C_pcie,serial =
    ceil(max(0, alpha_in + alpha_out - alpha_gpu) / delta_serial)
```

This latency-derived threshold is commonly very small for the H3 MLP. It is a
lower bound, not the final production chunk.

## HBM weight-amortization threshold

The FC1 and FC2 weights contain:

```text
FC1 elements = 2HF
FC2 elements = HF
total         = 3HF
```

Therefore:

```text
W = 3HF * s_w
```

Ignoring activation HBM traffic, the compute-bound condition is:

```text
6CHF / P >= W / B_hbm
```

This gives:

```text
C_hbm = ceil(W * P / (6HF * B_hbm))
```

Substituting `W = 3HF * s_w`:

```text
C_hbm = ceil(s_w * P / (2 * B_hbm))
```

For BF16 weights, `s_w = 2`, so the lower bound simplifies to:

```text
C_hbm = ceil(P / B_hbm)
```

When activation and temporary traffic is included:

```text
6CHF / P >= (W + mC) / B_hbm
```

If the denominator is positive:

```text
C_hbm = ceil(
    (W / B_hbm) /
    (6HF / P - m / B_hbm)
)
```

This remains a roofline lower bound. Cache reuse, quantized-weight unpacking,
GEMM layouts, and kernel fusion change the effective values of `W`, `m`, `P`,
and `B_hbm`.

## GEMM saturation threshold

The plateau throughput `P` is generally invalid for small `C`. Define a target
efficiency `eta`, for example `eta = 0.80`, and measure:

```text
P(C) = 6CHF / measured_gpu_seconds(C)
```

Then define:

```text
C_sat = min C such that P(C) >= eta * P_plateau
```

This term captures effects that a bandwidth/peak-FLOP formula does not model:

- insufficient GEMM `M` dimension;
- Tensor Core tile and wave quantization;
- kernel launch overhead;
- split-K or workspace policy changes;
- quantized-weight dequantization and synchronization;
- insufficient thread-block parallelism;
- cache and HBM behavior.

For this reason, `C_sat` must come from a small device-specific sweep or a
calibrated GEMM performance model. Nominal GPU peak FLOPs should not be used as
`P_plateau` unless measured efficiency is applied.

## Activation-memory upper bound

The current MiniMax-H3 fused-MLP activation estimate is:

```text
A_single = (6H + 3F) * s bytes/token
```

This represents one compute tile's FC1 output, gated intermediate, residual,
normalized/modulated hidden, FC2 output, and epilogue temporaries.

A practical three-stage ring can keep the large FC1 and gated scratch single
buffered while double-buffering only input and output hidden tiles. The
approximate per-token storage is then:

```text
A_ring = A_single + 2Hs
       = (8H + 3F) * s bytes/token
```

Given `M_available` bytes reserved for the MLP pipeline:

```text
C_memory = floor(M_available / A_ring)
```

The selected chunk must satisfy:

```text
C <= C_memory
```

Framework allocation behavior may require an additional fixed margin. A
custom preallocated implementation can usually follow the formula more closely
than eager `nn.Linear` calls that create new output tensors per chunk.

## Final selection rule

Choose the smallest aligned chunk that satisfies the PCIe, HBM, and GEMM
saturation constraints:

```text
C_required = max(C_pcie, C_hbm, C_sat)

C_target = align_up(C_required, A)
```

Then verify memory feasibility:

```text
C_target <= C_memory
```

If it does not fit, there is no chunk that meets all current assumptions. The
options are to reduce ring storage, lower the target GEMM efficiency, fuse
temporaries, increase the activation budget, or accept partially exposed copy
time.

In executable pseudocode:

```python
f = 6 * H * F
b_in = s * H
b_out = s * H

compute_slope = f / P
h2d_slope = b_in / B_in
d2h_slope = b_out / B_out

if compute_slope <= h2d_slope:
    raise ValueError("H2D cannot be fully hidden at the supplied throughput")
if compute_slope <= d2h_slope:
    raise ValueError("D2H cannot be fully hidden at the supplied throughput")

C_pcie_in = ceil(
    max(0.0, alpha_in - alpha_gpu) /
    (compute_slope - h2d_slope)
)
C_pcie_out = ceil(
    max(0.0, alpha_out - alpha_gpu) /
    (compute_slope - d2h_slope)
)
C_pcie = max(C_pcie_in, C_pcie_out)

W = 3 * H * F * s_w
hbm_denominator = compute_slope - m / B_hbm
if hbm_denominator <= 0:
    raise ValueError("HBM traffic cannot be amortized at the supplied throughput")
C_hbm = ceil((W / B_hbm) / hbm_denominator)

C_required = max(C_pcie, C_hbm, C_sat)
C_target = align_up(C_required, alignment)
C_memory = M_available // ((8 * H + 3 * F) * s)

if C_target > C_memory:
    raise ValueError("the required streaming chunk exceeds the memory budget")
```

## MiniMax-H3 BF16 example

Use the H3 dimensions:

```text
H = 5,376
F = 14,336
s = 2 bytes
s_w = 2 bytes
```

Per-token GEMM work:

```text
f = 6HF
  = 462,422,016 FLOP/token
```

Per-token transfer:

```text
H2D = sH = 10,752 bytes/token
D2H = sH = 10,752 bytes/token
total      21,504 bytes/token
```

BF16 FC1+FC2 weights:

```text
W = 3HF * 2
  = 462,422,016 bytes
  = approximately 441 MiB
```

Assume the following measured or calibrated values:

```text
P       = 200 TFLOP/s effective
B_in    = 25 GB/s
B_out   = 25 GB/s
B_hbm   = 1.5 TB/s effective
alpha_in = alpha_out = 20 us
alpha_gpu = 0 us for the conservative latency calculation
```

Per-token plateau compute time:

```text
462,422,016 / 200e12 = 2.312 us/token
```

Per-token one-direction copy time:

```text
10,752 / 25e9 = 0.430 us/token
```

The maximum effective GEMM throughput for which one copy direction remains
asymptotically hideable is:

```text
P_max,one-way = 6F * B / s
              = 1.0752 PFLOP/s
```

If H2D and D2H must be serialized, the corresponding limit is:

```text
P_max,serial = 3F * B / s
             = 537.6 TFLOP/s
```

The assumed 200 TFLOP/s effective MLP throughput is below both limits, so the
steady-state transfer slopes are theoretically hideable.

The single-direction PCIe latency lower bound is therefore:

```text
C_pcie,d = ceil(20 / (2.312 - 0.430))
         = 11 tokens
```

The BF16 weight-amortization lower bound is:

```text
C_hbm = ceil(P / B_hbm)
       = ceil(200 / 1.5)
       = 134 tokens
```

With 128-token alignment:

```text
align_up(134, 128) = 256 tokens
```

This 256-token result is still only a roofline lower bound. If a measured sweep
shows that combined FC1+FC2 throughput reaches 80% of its plateau at 512 or
1,024 tokens, then `C_sat` dominates and the production chunk should be 512 or
1,024 rather than 256.

For the approximate double-I/O ring:

```text
A_ring = (8H + 3F) * 2
       = 172,032 bytes/token
```

With a 512 MiB MLP activation budget:

```text
C_memory = floor(512 MiB / 172,032)
         = 3,120 tokens
```

A 2,048-token chunk therefore fits this estimate, while a 4,096-token chunk
does not. If the measured `C_sat` is at most 1,024, using 2,048 spends extra HBM
without improving PCIe hiding and may reduce the resident-query capacity
available to attention.

## Recommended calibration protocol

1. Measure pinned H2D and D2H time for the exact hidden-row size and fit:

   ```text
   T_copy(C) = alpha + bytes(C) / B
   ```

2. Benchmark the real FC1, SwiGLU, FC2, and epilogue path at aligned chunks:

   ```text
   128, 256, 512, 1024, 2048, 4096
   ```

3. Record GPU event time, not host launch time, and calculate `P(C)`.

4. Set `C_sat` to the smallest chunk reaching the chosen fraction of plateau
   throughput, normally 80% to 90%.

5. Compute `C_pcie`, `C_hbm`, and `C_memory` from measured values.

6. Select the smallest aligned feasible chunk and validate it in the actual
   three-stream ring. Separate-copy and standalone-GEMM measurements do not
   capture HBM contention or stream synchronization mistakes.

7. Confirm with a timeline that steady-state chunks execute as:

   ```text
   H2D[i + 1] || MLP[i] || D2H[i - 1]
   ```

The expected result for H3 is that PCIe latency produces a very small lower
bound, weight amortization produces a low-hundreds lower bound, and measured
GEMM saturation determines the practical chunk. Chunks larger than the
saturation point should be justified by measured end-to-end improvement,
because their main effect is to consume activation memory that could otherwise
increase the attention resident-Q set.
