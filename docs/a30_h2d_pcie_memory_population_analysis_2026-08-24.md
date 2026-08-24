# A30 H2D, PCIe, and Host-Memory Population Analysis

Date: 2026-08-24

## Conclusion

The measured 12.3 GB/s pinned H2D bandwidth is normal for this host. The A30
device supports PCIe Gen4, but the Intel Xeon Gold 6226R host is limited to
PCIe Gen3. All four A30 GPUs negotiate Gen3 x16.

The result is not caused by missing host-memory channels. EDAC reports all six
DDR4 channels populated on each socket. The system has eight 64 GiB DIMMs per
socket: channel 0 of each memory controller has two DIMMs, while channels 1
and 2 each have one DIMM.

## PCIe Limit

For GPU0, `nvidia-smi -q` reports:

```text
Device Max:  PCIe Gen4
Host Max:    PCIe Gen3
Current:     PCIe Gen3 x16
Replays:     0
```

The PCIe Gen3 x16 data-rate ceiling after 128b/130b encoding is:

```text
8 GT/s * (128 / 130) * 16 lanes / 8 = 15.753846 GB/s
```

The primary measured rate, 12.357768 GB/s, is 78.443% of that pre-protocol
ceiling. The remaining difference includes PCIe packet and flow-control
overhead plus the CUDA copy path. NVIDIA's CUDA Best Practices Guide describes
roughly 12 GB/s as an attainable pinned-memory rate on PCIe x16 Gen3, so this
node is operating at the expected practical ceiling.

## Measurements

All measurements use BF16, 56 KV heads, head dimension 128, 10 warmups, and
50 measured CUDA-event samples. The transfer benchmark issues two distinct
asynchronous copies, matching SeqAttn's K/V transfer pattern.

| GPU | Memory placement | Payload | Median GB/s |
|---|---|---:|---:|
| GPU0 | local NUMA0, `seqattn-a30` | 112 MiB | 12.357768 |
| GPU1 | local NUMA0 | 112 MiB | 12.314395 |
| GPU1 | remote NUMA1 | 112 MiB | 12.316027 |
| GPU1 | interleave NUMA0+1 | 112 MiB | 12.315614 |
| GPU2 | local NUMA1 | 112 MiB | 12.330178 |
| GPU3 | local NUMA1 | 112 MiB | 12.338427 |

The four local-GPU measurements have a mean of 12.335192 GB/s and a maximum
spread of 0.352%. This rules out a GPU0-specific or root-port-specific fault.

On GPU1, local, remote, and explicitly interleaved memory differ by at most
0.013%. Even the cross-socket UPI path can supply more data than the PCIe copy
path consumes. Unlike the RTX 5090 host experiment, interleaving memory across
two NUMA nodes provides no H2D gain on this PCIe Gen3 system.

The local GPU1 payload-size sweep is also flat:

| Payload | Median GB/s |
|---:|---:|
| 56 MiB | 12.298610 |
| 112 MiB | 12.314395 |
| 224 MiB | 12.320751 |

Increasing the payload by four times improves bandwidth by only 0.18%, so the
112 MiB K/V payload is already large enough to saturate the path.

FA2 partial-forward plus split-state merge preserves the same concurrent H2D
bandwidth on GPU0:

| FA2 Q tokens | Concurrent H2D median GB/s |
|---:|---:|
| 8192 | 12.357497 |
| 16384 | 12.357830 |

The FA2 compute workload therefore does not reduce the available copy-engine
bandwidth for this transfer pattern.

## Memory Population

Each Xeon Gold 6226R socket exposes two memory controllers with three channels
each. Local EDAC enumeration reports the following layout on both sockets:

```text
IMC0 channel0: slot0 + slot1
IMC0 channel1: slot0
IMC0 channel2: slot0
IMC1 channel0: slot0 + slot1
IMC1 channel1: slot0
IMC1 channel2: slot0
```

Every memory channel has at least one DIMM. Not every physical DIMM slot is
occupied, but filling the remaining second slots would add capacity rather
than memory channels. It can also lower the supported memory clock under a
two-DIMM-per-channel configuration.

The CPU supports six DDR4-2933 channels, with a nominal aggregate data rate of
140.784 GB/s per socket before efficiency losses. Privileged DMI access was
not available to confirm the configured DIMM clock. This does not affect the
bottleneck diagnosis because local, remote, and interleaved host memory all
saturate the same 12.3 GB/s PCIe path.

## SeqAttn Implication

Use approximately 12.35 GB/s as this node's measured concurrent pinned-H2D
roof. Do not model the node using the A30 device's Gen4 capability. Adding
DIMMs is not expected to raise SeqAttn H2D throughput; a PCIe Gen4-capable host
platform would be required for a material link-bandwidth increase.

Raw artifacts are kept outside the repository under:

```text
workspace/benchmarks/results/a30_host_memory_roofline_experiment0_20260824/
```
