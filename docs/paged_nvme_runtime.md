# Paged CPU/NVMe runtime

## Scope

The paged API is Linux-only, inference-only, single-GPU dense attention. It
supports packed variable-length sequences, bottom-right causal masking, MHA,
GQA, and MQA. BF16/FP16 stores are exact with respect to the same online-softmax
algorithm. INT8 K/V is a separate approximate mode.

It does not implement backward, sparse masks, model-weight paging, cross-request
HBM residency, io_uring, or GPUDirect Storage.

## Public types

- `PageSource`: Q and/or paired K/V page metadata plus a reader session that
  fills caller-owned buffers.
- `PageSink`: creates a writer session that consumes completed output pages.
- `MemoryPageSource` and `MemoryPageSink`: adapters for existing CPU tensors.
- `NvmeQKVStore`: validated manifest and aligned Q/KV data files.
- `NvmeOutputSink`: aligned, atomic output store publication.
- `CallbackOutputSink`: synchronous immediate page consumption.
- `NvmeQKVWriter`: tensor convenience construction and page-iterator streaming
  construction.
- `HostMemoryPlan`: category and total operator-memory enforcement.
- `PagedAttentionRunner`: bounded cache, I/O, H2D/compute/D2H, and sink pipeline.

## Failure behavior

The runtime rejects incompatible dtype/head/page layouts, segment/page overlap,
corrupt or incomplete manifests, file-size mismatch, unaligned records, short
reads/writes, direct-I/O rejection, and memory-budget overruns. Data-file and
manifest publication is ordered so a manifest never advertises an unfinished
temporary file.

Disk-full and cancellation errors propagate to the caller after temporary
output cleanup. Persistent stores are never deleted automatically.

## Measurement policy

Exact BF16/FP16 and INT8 results must be separate. Numerical reports should
include relative L2, maximum absolute error, and cosine similarity. Preparation
time includes initial INT8 quantization; reused-store runs report it separately.

NVMe acceptance requires a separately verified local device with at least
7GB/s sequential throughput and a working set larger than the configured DRAM
cache. The target is paged exact end-to-end latency no greater than 2x the
DRAM-only path. A result that misses this target remains a valid measured result
and must identify I/O, queue, copy, or kernel stalls rather than reporting only
kernel time.

This repository's current node is not an NVMe performance reference. Publish
only correctness and memory-cap results from it.
