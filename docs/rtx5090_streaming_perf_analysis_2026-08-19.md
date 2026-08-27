# RTX 5090 SeqAttn Streaming 性能分析与优化路线

> 日期：2026-08-19
> 范围：独立 `seqattn` runtime，重点分析完整 Q/K/V 已驻留 CPU DRAM 时的 exact streaming attention。
> 结论口径：表中的延迟使用未插桩 benchmark；Nsight Systems 和 Nsight Compute 只用于解释瓶颈，不与未插桩延迟直接混算。

## 1. 执行摘要

对 61,312-token、56-head、head dimension 128、BF16、non-causal MHA、2 GiB operator HBM workspace 的分析得到以下结论：

1. 原始 streaming 路径主要受 Triton update kernel 限制，不是 PCIe H2D 限制。Nsight Systems 中 update kernel 占 GPU kernel 时间的 99.8%，H2D 大部分可与计算重叠。
2. Blackwell `sm_120` 上，把 kernel launch profile 从 `64x64 / 4 warps / 2 stages` 调整为 `128x64 / 8 warps / 3 stages`，将 Tensor pipeline utilization 从 65.1% 提升到 93.2%，achieved occupancy 从 8.33% 提升到 16.67%。
3. 结合 planner 当前选择的 4K KV chunk，代表性 workload 从 0.7510 秒降至 0.5412 秒，提升 27.94%，有效吞吐从 143.5 TFLOPS 提升到 199.2 TFLOPS。
4. 优化在 causal、GQA、FP16 和 132K-token workload 上均有约 25%-29% 收益；短序列收益较小，1K tokens 只有约 3%。
5. 当前 D=128 Blackwell kernel 已接近 Tensor pipeline 饱和。继续只调整 `BLOCK_M/BLOCK_N/warps/stages` 的预期收益有限，后续更值得投入 planner 泛化、其他 head dimension/架构的 profile、模型 projection pipeline 和自动执行路径选择。
6. 当完整 Q/K/V 已在 DRAM 时，应使用 contiguous streaming，不应使用 paged DRAM runtime。Paged 的价值是固定 host-memory budget、NVMe/iterator backing 和分页 output，而不是提高 DRAM 路径速度。

## 2. 测试环境与代表性形状

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 32 GiB |
| GPU architecture | Blackwell, `sm_120` |
| Driver | 595.84 |
| PyTorch | 2.10.0+cu128 |
| Triton | 3.6.0 |
| Nsight Systems | 2025.6.1 |
| Nsight Compute | 2025.4.1 |
| Tokens | 61,312 |
| Heads | Q/K/V = 56/56/56 |
| Head dimension | 128 |
| Dtype | BF16，另测 FP16 |
| Causal | false，另测 true |
| HBM workspace | 2 GiB |
| Host input | caller-owned contiguous pinned CPU tensors |

GPU3 在正式测量时为空闲并由容器独占。CPU 进程绑定到 GPU3 对应的 CPU affinity；未使用不被容器允许的 NUMA `membind`。

## 3. 当前主要执行方式

### 3.1 GPU-resident FlashAttention

```text
完整 Q/K/V 驻留 HBM
→ FlashAttention
→ output 留在 HBM
```

适用条件：完整 Q/K/V/output 和模型其他活跃 tensor 能同时放入 HBM。

这是最低延迟路径，也是容量允许时应优先选择的路径。它不承担 CPU DRAM 到 GPU 的持续搬运成本，不能与包含 H2D/D2H 的 streaming 结果视为完全相同的端到端口径。

### 3.2 Contiguous DRAM streaming

```text
完整 pinned CPU Q/K/V
→ resident Q chunk
→ 双缓冲 streamed K/V
→ online-softmax Triton update
→ pinned CPU output 或 GPU consumer
```

适用条件：完整 Q/K/V 能放进 CPU DRAM，但不能或不希望完整驻留 HBM。

Streaming 直接从 caller pinned tensor slice 发起 H2D，不需要 PageSource、DRAM page cache、CPU I/O thread pool 或 page staging。其逻辑传输量近似为：

```text
H2D = |Q| + Q passes × (|K| + |V|)
D2H = |output|
```

因此 workspace 的关键作用是决定 resident Q 大小和完整 K/V 的重复扫描次数。

### 3.3 Projection pipeline

```text
CPU hidden
→ chunked GPU QKV projection
→ pinned CPU Q/K/V backing
→ streamed attention
→ GPU output projection / gate / residual consumer
```

适用条件：attention 位于完整模型 block 中，调用方能够提供 GPU-side output consumer。

这条路径可以复用最终 Q buffer 交付 attention output，避免 raw attention output 的额外 HBM buffer和 CPU 往返。在 MiniMax-H3 integration 中，它同时降低了显存峰值和延迟。它仍要求完整 projected Q/K/V 驻留 CPU DRAM，不是固定 host-memory 路径。

### 3.4 Paged / NVMe runtime

```text
PageSource / NVMe backing
→ bounded DRAM cache
→ pinned staging
→ HBM workspace
→ PageSink / callback / NVMe output
```

适用条件：完整 Q/K/V 无法放入 CPU DRAM，或者数据来自 NVMe、iterator、远程/自定义 source，或者 output 不能完整物化在 DRAM。

Paged 解决的是容量和数据来源问题，不是 DRAM 性能问题。若输入已经是完整 pinned `MemoryPageSource`，合理的产品行为应是自动 dispatch 到 contiguous streaming fast path；Memory paged 主要保留用于正确性测试、存储模拟和 API 兼容。

### 3.5 路径选择表

| 场景 | 推荐路径 | 原因 |
|---|---|---|
| 完整工作集能放入 HBM | GPU-resident FlashAttention | 最低延迟，无持续 PCIe streaming |
| Q/K/V 能放入 DRAM，但不能放入 HBM | Contiguous streaming | 直接 pinned H2D，运行时开销最低 |
| 位于模型 block，后面紧接 GPU projection | Projection pipeline | 避免 raw attention output 往返和独立 output HBM buffer |
| Q/K/V 超过 DRAM 或来自外部 backing | Paged runtime | 固定 host budget，按页读取和输出 |
| 只想验证 NVMe 调度模型 | Simulated NVMe | 可复现实验，不代表物理 NVMe 性能 |
| 可接受近似以降低存储/H2D | Explicit INT8 K/V paged | 降低 payload，但必须单独报告误差 |

## 4. 已完成并验证的优化

### 4.1 Blackwell D=128 kernel specialization

旧 portable profile：

```text
BLOCK_M = 64
BLOCK_N = 64
num_warps = 4
num_stages = 2
```

RTX 5090 上验证后的 profile：

```text
BLOCK_M = 128
BLOCK_N = 64
num_warps = 8
num_stages = 3
```

Planner 仅在以下条件全部满足时自动选择新 profile：

- CUDA device capability 为 `sm_120` 或更高；
- dtype 为 FP16 或 BF16；
- `head_dim == 128`；
- 用户没有显式指定任何 launch parameter。

若用户显式设置任一 launch parameter，其余参数从 portable profile 补齐。这使 benchmark 可以稳定复现旧配置，也避免把 Blackwell/D=128 的经验值错误应用到其他架构和 shape。

`num_stages=4` 在该 shape 上需要 122,880 bytes shared memory，超过 RTX 5090 单 block 101,376 bytes 上限，因此不可用。当前 `num_stages=3` 已使用约 98.3 KiB dynamic shared memory，也说明该 profile 不能直接推广到 D=256。

### 4.2 Joint Q/KV planning

2 GiB workspace 下，当前 planner 的实际选择是：

| KV chunk | Resident Q chunk | 61K 实测趋势 |
|---:|---:|---|
| 4,096 | 32,512 | 最快，约 0.541 秒 |
| 8,192 | 28,416 | 约 0.560 秒 |
| 16,384 | 20,224 | 约 0.612 秒 |

更大的 KV chunk 会减少 launch 数，但也压缩 resident Q，可能增加 Q pass 和 FP32 state traffic。当前 shape 上 4K 是更好的总体折中，因此不应固定追求 8K 或 16K KV tile。

### 4.3 Device-consumer output path

当 attention output 立即被 GPU consumer 使用时，最终 KV tile 完成后可以复用 Q HBM buffer 存放 normalized output，而不是保留独立 raw-output HBM buffer。Q buffer 的 free event 延后到 consumer 完成读取，consumer tensor 使用 `record_stream()` 管理生命周期。

该模式适用于 output projection、gate/residual 等模型内融合，不适用于必须立即得到完整 CPU output 的普通 API。

### 4.4 Benchmark 与诊断能力

已增加或扩展：

- `benchmarks/kernel_sweep.py`：扫描 block、warp、stage 组合，记录失败与 sampled output parity；
- streaming benchmark launch 参数：`--block-m`、`--block-n`、`--num-warps`、`--num-stages`；
- buffer 数参数：`--num-kv-buffers`、`--num-output-buffers`；
- plan JSON 中记录最终 resolved launch profile；
- planner 单元测试覆盖 CPU portable profile、Blackwell profile 和显式 override。

## 5. 性能结果

### 5.1 代表性 61K MHA

| 配置 | 延迟 | 有效 TFLOPS | 相对旧配置 |
|---|---:|---:|---:|
| 旧 explicit kernel | 0.751015 s | 143.52 | 1.000x |
| 新 automatic profile + 当前 planner | **0.541162 s** | **199.17** | **1.388x** |

延迟改善：

```text
(0.751015 - 0.541162) / 0.751015 = 27.94%
```

单独固定 8K KV chunk 时，旧配置约 0.763 秒，新 profile 约 0.560 秒，收益约 26%。这说明主要收益来自 kernel specialization，4K planner choice 进一步提供了小幅改善。

### 5.2 Shape 泛化

| Shape | 旧配置 | 新配置 | 改善 |
|---|---:|---:|---:|
| Causal BF16 MHA | 0.763948 s | 0.574637 s | 24.8% |
| BF16 GQA 56/8 | 0.723207 s | 0.518754 s | 28.3% |
| FP16 MHA | 0.753159 s | 0.560453 s | 25.6% |
| 132,288-token BF16 MHA | 3.356678 s | 2.386892 s | 28.9% |

测试范围内未观察到回退，但这不等价于所有 head dimension、架构和短序列都适用同一 profile。

### 5.3 短序列

| Tokens | 旧配置 | 新配置 | 改善 |
|---:|---:|---:|---:|
| 1,024 | 1.957 ms | 1.898 ms | 3.0% |
| 4,096 | 9.901 ms | 8.956 ms | 9.5% |
| 8,192 | 25.755 ms | 21.844 ms | 15.2% |
| 16,384 | 70.594 ms | 54.565 ms | 22.7% |

短序列中固定 launch、Python/CUDA API 和 finalize 开销占比更高，kernel specialization 的收益随序列增长逐渐显现。

### 5.4 MiniMax-H3 262K 完整 DiT 验证

Standalone kernel 结果随后在真实 MiniMax-H3 NF4 集成中验证。测试沿用
README 的 720x1280 workload，只把时间长度扩展到 957 frames；模型对齐后
为 736x1280、262,720 packed tokens。单层 BF16 Q/K/V 为 10.523GiB，连同
attention output 为 14.031GiB，明显超过 8GiB。

两次运行均使用 RTX 5090 GPU3、完整 50 blocks、一个 denoise step、2GiB
SeqAttn workspace、4,096-token KV chunk 和 8,192MiB whole-process target：

| 指标 | 旧 kernel + split MLP | 自动 Blackwell + fused MLP | 改善 |
|---|---:|---:|---:|
| Denoise step | 806.465 s | **570.980 s** | **29.20%** |
| Pipeline | 818.109 s | **583.017 s** | **28.74%** |
| CPU RSS peak | 66,048 MiB | **57,769 MiB** | **8,279 MiB / 12.54%** |
| PID NVML peak | **7,564 MiB** | 7,866 MiB | 两者均低于 8GiB |
| Logical H2D | 4,912.582 GiB | **4,430.275 GiB** | **482.307 GiB** |
| Logical D2H | 1,139.999 GiB | **789.230 GiB** | **350.769 GiB** |

旧配置显式固定 `64x64/4/2` 并使用 split MLP；新配置使用自动
`128x64/8/3` 和 fused MLP。两者 resident Q 分别为 26,048 和 25,984
tokens，均需要 11 个 Q passes，attention H2D 也完全相同。因此：

- kernel specialization 的收益来自更快的 update execution，不是减少 KV scan；
- fused MLP 删除完整 FC1 intermediate 的 D2H/H2D，以及 duplicate residual H2D；
- 每个 denoise step 总共减少 833.076GiB logical PCIe traffic；
- fused 路径的 host-memory 收益随序列增长，在 262K 时降低约 8.1GiB RSS。

同一 old-to-current 组合随序列长度的结果为：

| Packed tokens | 旧配置 | 新配置 | 改善 |
|---:|---:|---:|---:|
| 14,912 | 26.314 s | **20.431 s** | **22.36%** |
| 30,976 | 28.982 s | **22.813 s** | **21.29%** |
| 61,056 | 65.990 s | **50.230 s** | **23.88%** |
| 262,720 | 806.465 s | **570.980 s** | **29.20%** |

前三行使用 warmed step 2；262K 是一次完整 capacity run。该结果证明
standalone kernel 优化和 MLP fusion 能转化为真实 50-block 模型收益，但
单次观测不提供误差条或统计显著性。主延迟结论只引用未插桩 JSON。

## 6. Profiler 结论

### 6.1 Nsight Systems

三次代表性 streaming run 的汇总显示：

| 项目 | 结果 |
|---|---:|
| Update kernels | 72 次，2.072 秒 |
| Update kernel 占 GPU kernel 时间 | 99.8% |
| Finalize kernels | 4.99 ms |
| H2D | 595.8 ms，大部分与 compute 重叠 |
| D2H | 158.5 ms |

因此该 shape 的主要优化对象是 update kernel 和影响 update 次数的 Q/KV planning，而不是简单增加 H2D buffer 或 CPU I/O 线程。

### 6.2 Nsight Compute

| 指标 | 旧 `64x64/4/2` | 新 `128x64/8/3` |
|---|---:|---:|
| Replay duration | 59.77 ms | 41.60 ms |
| Compute throughput | 65.06% | 93.19% |
| Tensor pipeline utilization | 65.1% | 93.2% |
| Achieved occupancy | 8.33% | 16.67% |
| Active warps / SM | 4 | 8 |
| Registers / thread | 246 | 213 |
| Dynamic shared memory / block | 57.34 KiB | 98.30 KiB |

新配置通过增加 block 的 query work 和 active warps，显著提高 Tensor Core 利用率。由于 Tensor pipeline 已达到约 93%，在不改变算法或代码生成的情况下，当前 shape 的剩余 kernel 优化空间更可能是个位数百分比，而不是再次获得 20%-30%。

## 7. 后续可优化方向

### 7.1 高优先级：按架构和 shape 建立 profile registry

当前自动 specialization 只覆盖 Blackwell D=128。下一步应分别 profile：

- D=64、D=96、D=256；
- MHA、GQA、MQA；
- causal 与 non-causal；
- Ampere、Ada、Hopper、Blackwell；
- FP16、BF16；
- 不同 resident Q 和 KV chunk 范围。

每个 profile 必须有 shared-memory legality 检查和性能回退阈值。没有实测支持的 shape 应继续使用 portable profile，而不是扩大当前 heuristic。

### 7.2 高优先级：让 planner 使用设备实测模型

当前 cost model 使用经验带宽、state traffic 和 launch penalty。可以在安装或首次 benchmark 时测量：

- pinned H2D/D2H bandwidth；
- update kernel 在不同 Q/KV chunk 下的吞吐；
- launch latency；
- FP32 accumulator/state spill 成本；
- output consumer 是否消除 D2H。

Planner 随后根据设备和 shape 选择 Q chunk、KV chunk、buffer 数和 launch profile，而不是依赖一组跨设备常数。

### 7.3 中优先级：降低 FP32 online-softmax state 成本

每个 resident query/head 需要 running max、running sum 和 FP32 accumulator。更大的 Q chunk 会扩大这些状态并降低 cache locality。可能方向包括：

- 调整 accumulator layout 和写回粒度；
- 减少不必要的 state reload/store；
- 对特定 shape 使用更持久的 CTA scheduling；
- 研究不降低 exactness 的 mixed accumulator 策略。

这是算法和 kernel codegen 级工作，需要用 Nsight Compute 验证寄存器、shared memory、L2 和 Tensor pipeline 的综合变化。

### 7.4 中优先级：扩大 device-consumer 融合

在模型场景中，attention 后通常紧接 output projection、gate、residual 或 normalization。继续扩大 GPU consumer API 可以：

- 消除 raw output D2H；
- 消除下一算子的 raw output H2D；
- 复用 Q/output HBM storage；
- 降低 pinned output 容量和 event 数量。

这通常比继续微调已接近饱和的 standalone kernel 更有端到端价值。

### 7.5 中优先级：自动执行路径选择

Public API 可以根据 residency 自动选择：

```text
完整 CUDA Q/K/V          → FlashAttention / native GPU backend
完整 pinned CPU Q/K/V    → contiguous streaming
projection callbacks     → projection pipeline
非物化 PageSource        → paged runtime
```

这能避免用户把完整 DRAM tensor错误送入 paged runtime，也能让 benchmark 明确比较容量路径和性能路径。

### 7.6 较低优先级：更多 buffering 和 CUDA Graph

当前代表性长序列已经 compute-bound，盲目增加 KV buffer 通常不会显著改善 wall time，反而会占用 HBM并缩小 resident Q。CUDA Graph 更可能帮助短序列和重复固定 shape，但对长序列 update kernel 主导的 workload 收益有限。

应先用 trace 证明存在明显 launch gap，再引入这些复杂度。

### 7.7 较低优先级：NUMA 与 pinned allocator 稳定性

大规模 pinned tensor 的首次分配、page fault 和 NUMA placement 会影响 preparation time。推荐继续保持：

- pageable memory 并行填充后再 pin；
- CPU affinity 与 GPU PCIe/NUMA locality 对齐；
- 区分 preparation time 和 steady-state execution；
- 重复运行时复用 pinned input/output 和 CUDA workspace。

这些优化主要改善准备时间和波动，对当前 compute-bound kernel 的 steady-state latency 影响较小。

## 8. 建议的优化优先级

| 优先级 | 工作项 | 预期价值 | 风险 |
|---|---|---|---|
| P0 | 保持 Blackwell D=128 automatic profile | 已验证约 28% | 低，已有严格 guard |
| P0 | MiniMax-H3 默认保持 fused MLP | 262K 下快 29.2%，RSS 低 8.1GiB | 低，保留 split fallback |
| P0 | DRAM input 自动选择 contiguous streaming | 避免错误进入 paged path | 低 |
| P1 | 其他架构/head dimension profile registry | 扩大收益覆盖面 | 中，需要大量 benchmark |
| P1 | Device-consumer 与模型 projection 融合 | 最大端到端传输收益 | 中，涉及调用方 contract |
| P1 | Device-aware planner calibration | 改善不同 GPU/shape 的 chunk 选择 | 中 |
| P2 | FP32 state traffic/codegen 优化 | 当前 kernel 的剩余核心空间 | 高，易增加寄存器/shared memory |
| P2 | 短序列 CUDA Graph/launch 优化 | 改善 1K-8K 场景 | 中，长序列收益有限 |
| P3 | 增加 buffer 数或通用预取 | 仅在 trace 显示 copy gap 时有效 | 低到中，可能压缩 Q chunk |

## 9. 正确性与验证状态

已完成：

- 67 项 test suite 全部通过；
- changed files Ruff check 通过；
- Python compileall 通过；
- sampled output 在 kernel sweep 中与 baseline 一致；
- causal、GQA、FP16、长序列和短序列均完成未插桩 A/B；
- 262,720-token、完整 50-block H3 old/new 对比在 8GiB target 下完成；
- Nsight Systems 用于确认 kernel/H2D/D2H 时间关系；
- Nsight Compute 用于确认 occupancy、register、shared memory 和 Tensor pipeline。

数值测试仍应区分：

- BF16/FP16 exact streaming 与 FP32 reference 的误差；
- 不同 launch profile 的 parity；
- 可选 INT8 K/V 的 approximate error。INT8 结果不得描述为 exact。

## 10. 最终建议

当前版本应将 Blackwell D=128 specialization 作为默认自动 profile，并保留 portable override。对于完整 CPU Q/K/V，contiguous streaming 是正式的 DRAM 性能路径；paged runtime 只承担 out-of-core 和非物化 source/sink 场景。

短期内最值得投入的不是继续对同一 RTX 5090 D=128 kernel 做大范围参数搜索，而是：

1. 建立多架构、多 head dimension 的实测 profile registry；
2. 将 planner 校准为 device-aware；
3. 扩展 GPU output consumer 和模型 projection 融合；
4. 自动选择 GPU-resident、DRAM streaming 和 paged 路径；
5. 只有 profiler 显示明确新瓶颈时，再进行 state layout、CUDA Graph 或额外 buffering 优化。
