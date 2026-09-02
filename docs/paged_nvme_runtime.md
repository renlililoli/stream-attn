# Paged CPU and NVMe runtime

Status: current for `seqattn-core 0.4.0a1` on 2026-09-02.

The paged runtime executes exact dense attention without requiring complete
CPU Q/K/V tensors. It is Linux-only, inference-only, and single-GPU. It
supports packed variable-length sequences, bottom-right causal masking, MHA,
GQA, and MQA.

## Public contracts

| Type | Responsibility |
|---|---|
| `PageSource` | Q and/or paired K/V layouts, page descriptors, and reader creation |
| `PageReader` | Fill caller-provided Q or K/V staging buffers |
| `PageSink` | Create a writer for output pages |
| `PageWriter` | Consume complete output pages and publish or return the result |
| `PagedAttentionRunner` | Validate layouts, enforce budgets, schedule I/O, cache, copy, and attention |

Included adapters are:

- `MemoryPageSource` and `MemoryPageSink` for existing CPU tensors;
- `CallbackOutputSink` for synchronous page consumption;
- `NvmeQKVWriter`, `NvmeQKVStore`, and `NvmeOutputSink` for persistent stores;
- `SimulatedNvmeDevice`, `SimulatedPageSource`, and `SimulatedPageSink` for
  scheduler experiments.

Page descriptors and packed boundaries are validated before execution. A page
may not cross a packed segment.

## Host-memory budget

`PagedAttentionConfig.host_memory_budget_bytes` covers allocations owned by the
paged operator:

```text
total host budget
  = pinned staging limit
  + direct-I/O bounce limit
  + metadata margin
  + remaining bounded K/V cache
```

Default policy:

| Category | Default |
|---|---:|
| Total operator host budget | 8 GiB |
| Pinned staging limit | 1 GiB |
| Direct-I/O bounce limit | 512 MiB |
| Metadata margin | 128 MiB |
| Target logical page size | 16 MiB |

Caller-owned tensors behind memory adapters are excluded. Process RSS can
therefore exceed the operator budget without violating the contract.

The CUDA attention workspace is separately controlled by
`config.attention.workspace_budget_bytes`; it is not part of the host budget.

## Execution model

```text
validate source and sink contracts
  -> allocate bounded staging
  -> open readers and writer
  -> allocate remaining K/V cache capacity
  -> schedule page reads through a fixed worker pool
  -> H2D resident Q and streamed K/V
  -> exact FP32 online-softmax recurrence
  -> D2H output page
  -> PageWriter
```

Memory-backed K/V bypasses the operator cache because the caller already owns
the complete backing. NVMe-backed K/V uses a bounded two-region cache under the
remaining host budget.

Runners are single-flight. Create separate runners for concurrent requests.

## Storage modes

| `kv_storage_dtype` | Semantics |
|---|---|
| `bf16` | Exact storage mode |
| `fp16` | Exact storage mode |
| `fp32` | Exact reference-compatible mode; not supported by paged Triton execution |
| `int8` | Approximate symmetric K/V storage with FP16 scales per 64-token group and KV head |

Exact and approximate results must be named and measured separately. INT8
reports should include relative L2, maximum absolute error, and cosine
similarity, and should report initial quantization time separately from reused
store execution.

## Persistent store publication

An NVMe Q/K/V store contains a manifest and aligned Q and K/V data files. The
loader validates:

- format and version;
- alignment;
- tensor and storage layouts;
- packed boundaries and page order;
- file names and exact file sizes.

Writers publish data before publishing the manifest, so a completed manifest
never points to unfinished temporary files. Output writers clean up temporary
files after failure; persistent input stores are never deleted automatically.

## Direct I/O

`direct_io=True` is strict. SeqAttn does not silently fall back to buffered
I/O. Unsupported filesystems, invalid offsets or lengths, rejected `O_DIRECT`
operations, and short reads or writes raise explicit errors.

Use `direct_io=False` only when buffered I/O is intentionally part of the
experiment or deployment. Do not compare buffered and direct-I/O results as if
they were the same storage mode.

## Simulation boundary

The in-memory simulator applies fixed command latency and shared aggregate
bandwidth timelines to page operations. It is useful for queue-depth, cache,
read-ahead, and overlap experiments.

It does not emulate filesystem behavior, alignment cost, controller firmware,
PCIe contention, writeback, thermal throttling, or device failure. Simulated
results are not physical NVMe performance evidence.

## Failure behavior

The runtime rejects incompatible dtype, head, page, segment, and direct-I/O
contracts before or during execution. Disk-full, cancellation, callback, and
I/O errors propagate to the caller after worker shutdown and temporary output
cleanup.

No current path implements backward, sparse masks, model-weight paging,
cross-request HBM residency, `io_uring`, or GPUDirect Storage.

## Measurement policy

Physical storage claims require a separately verified local device and a
working set larger than the configured DRAM cache. Record physical and logical
bytes, cache hit ratio, read and write service time, queue depth, host-memory
peak, CUDA workspace, and full wall latency. Treat emitted JSON as the source
of truth and retain failures and timeouts.
