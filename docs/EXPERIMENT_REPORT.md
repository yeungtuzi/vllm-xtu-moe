# vllm-xtu-moe 长 prefill GPU 加速 —— 实验报告

> 项目名:**vllm-xtu-moe**(XTU = X Transformers Unity,读音「小兔」)。
> 内部标识符(`vllm_xiaotu_moe` / `xiaotu_moe` 包名、`XIAOTU_*` 环境变量、目录名)不变。

> 作为可随 PR 提交的完整实验报告:①硬件配置(工具核验)②本次开发工作摘要
> ③对主线 vLLM 的修改与扩充清单 ④期望的合并方式 ⑤本机实测(1 张 / 2 张 A100)。
> 数据源:`/home/user/lvllm/results.txt`(逐条留档)、`report/*.jsonl`(机器可读)、
> `report/fig/*.png`(图表)。除注明外,所有数字均为**本机实测**;测量命令见附录 A。

---

## 0. 摘要

**做了什么。** 给 `vllm-xtu-moe`(vLLM 主线 OOT 插件:DeepSeek-V4-Flash 的 43 层
routed MoE 放 CPU、注意力放 GPU)增加一条**逐层流式 GPU prefill** 通路:当某层 prefill
的 token 数 ≥ 阈值(`VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS`,默认 2048)时,把该层
**原始 MXFP4 权重**(3.19 GiB/层;43 层合计 137 GiB)**按 K-major 从 pinned 主机缓存
异步 DMA 到 GPU**,用**分组 Triton GEMM(内核内反量化)**算完即释放,并用**双 slot
预取流水**让下一层的 H2D 与当前层的 tensor core 计算重叠。短 prefill 仍走 CPU 引擎。

**关键结论。**

1. **可行性成立**:A100(40 GB)装不下 137 GiB 的 fp4 专家权重,但"逐层流式 + 算完即
   释放"只需 **6.4 GiB VRAM**(2 个 ping-pong slot),且数值正确(golden 全维度通过,
   RMS 相对误差 5.6e-3 ≈ bf16 输出精度,见 §5.2)。
2. **端到端加速显著且随上下文放大**(单卡,同一进程内实测):

   | prompt | CPU prefill | GPU prefill | 加速 |
   |---|---|---|---|
   | 2091 token | 112.93 s(18.5 tok/s) | **5.64 s(371 tok/s)** | **20.0×** |
   | 4137 token | 439.38 s(9.4 tok/s) | **5.76 s(718 tok/s)** | **76.2×** |
   | 8229 token | — | **8.64 s(953 tok/s)** | — |
   | 16039 token | — | **18.24 s(880 tok/s)** | — |

3. **H2D 地板决定短上下文**:137 GiB / 25 GB/s = **5.9 s/次 prefill**(实测 5.5 s),
   与 batch 内 token 数无关 ⇒ 2K 与 4K 的 GPU TTFT 几乎相同(5.64 vs 5.76 s)。
   但 CPU 路径每 token 每层要 0.44–2.5 ms(GPU 只要 11.2 µs),所以 **CPU/GPU 交叉点
   只有 100–300 token** —— 默认阈值 2048 明显保守,可以下调(§5.6d、§7.6)。
4. **预取重叠是最大的单项收益**:双 slot 预取把 H2D 藏进计算,单层耗时在 T=4K/16K/32K
   从 217.6/356.1/539.0 ms 降到 **127.4/198.7/382.2 ms**(1.71×/1.79×/1.41×);
   纯 MoE 吞吐 T=16K → **1918 tok/s**、T=32K → **1994 tok/s**,跨过 1500 tok/s 的
   规模从 41K 提前到 **8K**(单卡,§5.6a)。
5. **双卡(TP=2 + 专家并行)有效**:每 rank 只流 `E/2` 个专家(1.59 GiB/层)、只算一半
   token-expert pair,H2D 地板与算力各减半 → 纯 MoE T=16K **3803 tok/s**、T=32K
   **3979 tok/s**(等效微基准,§5.6b);端到端 16K **18.24 s → 13.96 s(1149 tok/s)**,
   2K 并发场景 **1.93×**(§5.4)。
6. **端到端仍未达 1500 tok/s,瓶颈是"分散开销"而不是注意力**:进程内 profiler
   (§5.10)显示 16K 的 GPU 时间里 MoE 内核占 31%、逐元素小内核占 29%、
   H2D 占 19%、dense GEMM 占 18%,而**注意力/indexer 只有 2.9%**。
   第 1 轮"注意力占 9.65 s"是减法伪影,已更正(§5.7/§7.1)。

---

## 1. 硬件配置(全部由工具核验,命令见附录 A)

### 1.1 平台 / 操作系统

| 项 | 值 | 核验方式 |
|---|---|---|
| CPU | 2 × AMD EPYC 9654(Genoa,Zen4),96 核/路 | `lscpu` |
| 逻辑 CPU | 384 个(0-383),**在线 192**(SMT 关闭,192-383 离线) | `lscpu` |
| 频率 | 基频 1.5 GHz,max 3.709 GHz | `lscpu` |
| 指令集 | AVX-512F/BW/VL/VNNI/BF16、GFNI、VAES、BMI2(**无 AMX**) | `lscpu` flags |
| L1d / L1i | 6 MiB / 6 MiB(192 × 32 KiB) | `lscpu` |
| L2 | 192 MiB(192 × 1 MiB) | `lscpu` |
| L3 | 768 MiB(24 × 32 MiB) | `lscpu` |
| 内存 | 1.5 TiB,**8 NUMA 节点**(NPS4),约 192 GiB/节点 | `numactl -H` |
| 内存规格 | **DDR5-4800**,24 条 × 64 GiB(12 通道/路)⇒ 理论 **921.6 GB/s** | 用户确认(`dmidecode` 需 root) |
| OS / kernel | Ubuntu 22.04.5 LTS,kernel 5.15.0-179-generic | `uname` |
| THP | `madvise` | `/sys/.../transparent_hugepage/enabled` |

### 1.2 NUMA 与 GPU 亲和性

`nvidia-smi topo -m` + `lspci -s <bdf> -vv`:

| GPU | PCI BDF | 归属 NUMA 节点 | 本地 CPU | 与其它 GPU 互连 |
|---|---|---|---|---|
| GPU0 | `0000:21:00.0` | node 2 | 48-71 | `SYS`(PCIe + 跨 socket SMP) |
| GPU1 | `0000:41:00.0` | node 0 | 0-23 | `SYS` |
| GPU2 | `0000:c1:00.0` | node 4 | 96-119 | `SYS` |

**无 NVLink**:三张卡均为 A100-PCIE,任意两卡之间都是 `SYS`(经 PCIe + SMP),
因此 TP 的 all-reduce 只能走 PCIe。

### 1.3 GPU

| 项 | 值 | 核验方式 |
|---|---|---|
| 型号 | NVIDIA A100-PCIE-40GB(SM 8.0,GA100,108 SM) | `nvidia-smi -q` |
| 数量 | 3(serial …404617 / …405012 / …405004) | `nvidia-smi -q` |
| 显存 | 40960 MiB HBM2 ×3 | `nvidia-smi` |
| 驱动 / CUDA | 580.159.03 / CUDA 13.0 | `nvidia-smi` |
| VBIOS | 92.00.25.00.08 | `nvidia-smi -q` |
| 功耗上限 | 250 W | `nvidia-smi` |
| 最大 SM / 显存时钟 | 1410 MHz / 1215 MHz | `nvidia-smi` |
| PCIe | **Gen4 ×16**(current == max),无 P2P | `nvidia-smi` / `lspci` |

### 1.4 实测带宽(不是纸面参数)

**(a) 主机内存带宽** —— 这一项**第 2 轮重新调查过,结论与第 1 版不同**。
机器是 24 条 **DDR5-4800**(12 通道/路),理论 **921.6 GB/s**。

第一版用 `-O3 -march=native` 的朴素 STREAM + `numactl --interleave=all` 只测到
422 GB/s,看起来远低于理论值。逐项排查后确认那是**三个测量伪影叠加**,不是硬件问题:

| 因素 | 为什么有害 | 修掉后的收益 |
|---|---|---|
| `numactl --interleave=all` | 强制每个线程 7/8 的访问跨节点(走 xGMI/IO die) | +12% |
| 普通写(读改写,RFO) | Triad 记 3N 字节,实际 DRAM 流量是 4N | +29% |
| 标量归约 / 128-bit 向量 | GCC 11.4 不认 znver4,`-march=native`=znver3 且默认 `-mprefer-vector-width=128` | 单流读 +37% |
| 用满 192 线程 | 比 96–144 线程更差且方差极大(与 `numa_balancing=1` 的页迁移有关) | — |

修正后的实测(`report/stream_nt.c` + `rd1.c`,first-touch 本地分配、
`-mprefer-vector-width=512`、AVX-512 非临时存储)。**归因矩阵**(2×2×2,固定数组
3.2 GB/条,取 5 次最好值):

| 线程 | 内存放置 | 普通写(RFO) | NT 写 | 只读 |
|---:|---|---:|---:|---:|
| 96 | first-touch 本地 | 574.8 | **743.6** | 754.5 |
| 96 | `--interleave=all` | 386.9 | 397.7 | 440.8 |
| 192 | first-touch 本地 | 507.8 | 647.0 | 714.0 |
| 192 | `--interleave=all` | 398.6 | 461.7 | 456.3 |

单流只读(1 数组,96 线程)**860.5 GB/s = 93% 理论**。

> **结论:这台机器的内存子系统是健康的,~740 GB/s 是它的实际可达值(80%)**;
> 第 1 版的 422 GB/s 只反映了测量方式的问题。三个因素按贡献排序:
> ①**页放置**:`--interleave=all` 让每个线程 7/8 的访问跨节点,NT 口径下损失
> 87%(743.6 → 397.7);**改成 first-touch 本地**是最关键的一步。
> ②**NT 存储**:省掉写的 read-for-ownership,+29%(574.8 → 743.6)。
> ③**线程数 96 而不是 192**:+15%(647.0 → 743.6);满核反而更差且方差极大
> (与 `numa_balancing=1` 的页迁移有关,曾观察到单次 217 GB/s)。
> ④**向量宽度 / `-ffast-math` / `-mtune` 无影响**:`-mprefer-vector-width=128/256/512`、
> `-mtune=generic/znver3`、`-ffast-math` 在 96 线程下互差 <1%(见 §1.4 末"编译器调查")。
>
> **对 CPU MoE 引擎的含义(第 12 版更正,见 §7.2b)**:引擎不是带宽受限 ——
> 真实路由下每层只读到约 **10 GB/s**(3.2 GB / 0.32 s),而机器可做 740 GB/s;
> 它跑在 **1.7–2.0 TFLOP/s ≈ AVX-512 BF16 峰值的 15–18%**(不是早期微基准里
> 那个 0.64 TFLOP/s —— 那是**固定路由**下的失真值,见 §7.2b)。
> 引擎自带相位计时显示 **99.7% 的时间在 gate_up(57%)与 down(28%)两个 GEMM 相位**。
> 第 1 版"CPU MoE 是带宽敏感型"、第 4 版"反量化 ALU 受限"两种说法**都不成立**。

**编译器调查(GCC 11.4,Zen4)。** `-march=native` 在本机解析成 **znver3**
(GCC 11 不认识 `znver4`;`-march=znver4` 直接报错),默认 `-mprefer-vector-width=128`。
为此对 **STREAM** 和 **xiaotu CPU 引擎**各做了 5 组编译参数 A/B:

| 变体 | 参数 | STREAM Triad_NT(96 线程) | 引擎 B=2048 |
|---|---|---:|---:|
| 原始 | `-O3 -march=native -ffast-math` | 743.3 GB/s | 962 ms/层 |
| 512 位 | `+ -mprefer-vector-width=512` | 743.1 | 964 |
| 256 位 | `+ -mprefer-vector-width=256` | — | 968 |
| generic tune | `-mtune=generic -mprefer-vector-width=512` | 743.1 | 966 |
| znver3 tune + unroll | `-mtune=znver3 -funroll-loops -fno-math-errno` | 741.7 | 961 |

⇒ **两边都没有可测差异(<1%)**。原因:STREAM 在 96 线程下已经是内存受限;
引擎的热循环是手写 AVX-512 BF16 内联函数(`VDPBF16PS`),编译器向量化器不参与。
**结论:GCC 11.4 下没有"调参数就能提速"的空间**;若要试,只能换 GCC 12+/LLVM 用
`-march=znver4`,而本机实测显示它不会动引擎的瓶颈(ALU 反量化)。

**(b) PCIe H2D/D2H 与 HBM**(`report/pcie_bw.py`,256 MiB 缓冲 ×10 次):

| 方向 | 实测 |
|---|---|
| pinned H2D | **22.7 GB/s** |
| pinned D2H | 22.4 GB/s |
| pageable H2D(连续大缓冲) | 25.0 GB/s |
| HBM device-to-device copy | **1372 GB/s**(读+写合计,≈88% 峰值) |

> 逐层流式实际达成 3.19 GiB / 127 ms ≈ **26.9 GB/s**,已接近 Gen4 ×16 实用上限。
> **PCIe 是本方案唯一无法绕过的硬地板。**
> 另:`cudaHostRegister` 就地 pin 实测 0.37 s / 2 GiB(比 `pin_memory` 快 3.7×),
> H2D 速率相同 —— 见 §7.4。

### 1.5 被加速的模型

`deepseek-ai/DeepSeek-V4-Flash-0731`(本地 156 GB,48 个 safetensors 分片):

| 项 | 值 |
|---|---|
| 层数 | 43(+1 MTP nextn 层) |
| hidden / moe_intermediate | 4096 / 2048 |
| routed experts / topk / shared | 256 / 6 / 1 |
| 专家权重 | `expert_dtype=fp4`(**MXFP4**:e2m1 半字节 + e8m0 块标量,block=32) |
| 注意力 | MLA,`head_dim=512`,64 heads,1 KV head,q_lora/o_lora=1024 |
| indexer | 128 维 / 64 heads / `index_topk=512` |
| compress_ratios | `[0,0,4,128,4,128,…,4,0,0,0]` |
| sliding_window / max_pos | 128 / 1,048,576 |

**每层 routed expert 原始字节(= 逐层流式的搬运量):**

| 张量 | 形状 | 大小 |
|---|---|---|
| `w13_weight` | [256, 4096, 2048] u8 | 2.000 GiB |
| `w13_weight_scale` | [256, 4096, 128] u8 | 0.125 GiB |
| `w2_weight` | [256, 4096, 1024] u8 | 1.000 GiB |
| `w2_weight_scale` | [256, 4096, 64] u8 | 0.0625 GiB |
| **合计** | | **3.1875 GiB/层 = 3.423 GB/层** |

→ 43 层 **137.1 GiB / 147.2 GB**;按 25 GB/s 计 **H2D 地板 5.89 s/次 prefill**
(实测 127 ms/层 × 43 = 5.48 s)。

**KV cache 开销(直接决定可用上下文):**

| max_model_len | 可用 KV | 每 token |
|---|---|---|
| 1024 | 21.52 GiB / 55,677 tokens | **405 KiB** |
| 16384 | 17.02 GiB / 44,434 tokens | **400 KiB** |

> 单张 40 GB A100 扣掉权重/激活后只剩 ~17 GiB 给 KV ⇒ **上下文上限约 44K token**。
> 这是"hundreds of K 上下文"在单卡上的硬墙。400 KiB/token 来自 SWA + compressed KV
> + indexer KV + compressor 状态四类缓存叠加,是 DS-V4 的架构特性。

---

## 2. 本次开发工作摘要

### 2.1 问题与约束

- 目标:让 DS-V4-Flash 在 **A100-40GB** 上做长 prefill(2K ~ hundreds of K token)
  时用上 GPU,初步吞吐目标 **1500 token/s**;短 prefill(<~1K)继续走 CPU。
- 约束:专家权重 137 GiB ≫ 40 GB 显存;vLLM 的 `Marlin` 只支持 NVFP4(group-16),
  **不支持 MXFP4(block-32)**;主线没有可复用的 fp4 MoE 后端。
- 设计选择(与 ktransformers `KT_GPU_PREFILL_TOKEN_THRESHOLD`、fork
  `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE` 同构):**逐层流式** —— 每层把原始 MXFP4 权重
  H2D,算完立刻释放,VRAM 只保留 2 个 slot。

### 2.2 实现要点

1. **主机侧 K-major 缓存**:`_pinned_kmajor()` 把 `[E,2I,H/2]` 一次性转成
   `[E,H/2,2I]` 并 pin 住(按 storage 指针 + offset + shape 缓存,并持有源 storage
   防止地址复用命中脏数据)。H2D 落进 GPU 即为内核所需布局 → 去掉每层一次的设备端
   transpose。
2. **内核内反量化**:e2m1 半字节 → 值(`{0,0.5,1,1.5,2,3,4,6}` 符号对称),再乘
   `2^(byte-127)` 的 e8m0 块标量(block=32),全程 fp32 累加。VRAM 里只放**原始字节**
   (3.19 GiB/层);反量化成 bf16 的 12.9 GiB/层会直接 OOM。
3. **单 launch 分组 GEMM**:`program = (expert, N-tile)`,内部 m-loop 处理该专家被
   选中的所有行;一层只有 **2 次 kernel launch**(gate_up + down),而不是 256×2 次。
   分组用 `seg_ptr`(expert → 行区间)表达,输入按 expert 排序一次。
4. **双 slot 预取流水**:`prefetch_layer()` 在**独立 CUDA stream** 上发起 H2D 并记录
   event,计算流只 `wait_event`;slot 被消费后记录 `busy` event,下次写入同一 slot 前
   `wait_event(busy)`。层 L 的 forward 里**先发起 L+1 的预取**,再做 L 的内核。
   显存不足以放 ping-pong 缓冲时自动降级为同步路径并打印提示。
5. **专家并行(TP>1)**:每 rank 只流 `E/TP` 个专家,路由 id 减 rank 偏移、越界置 -1、
   权重置 0,算完对 bf16 结果做 `tensor_model_parallel_all_reduce`。
6. **环境变量**:`VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS`(0=关闭)、
   `XIAOTU_GPU_PREFETCH_AHEAD`(默认 1)、`XIAOTU_GPU_PREFILL_{BM,BN,BK,BH,STAGES}`。

### 2.3 优化轮次与每轮实测收益(单层,T=4K)

| 轮次 | 改动 | 每层耗时 | 说明 |
|---|---|---|---|
| 0 | 每专家 2 次 launch + 页式 H2D + torch 反量化 | ~830 ms | 22016 次 launch/层集,occupancy 极低 |
| 1 | 分组单 launch + pinned H2D + 内核内反量化 | ~205–220 ms | H2D 3.4 → 25 GB/s;VRAM 12.9 → 3.2 GiB/层 |
| 2 | 主机侧 K-major(去掉内核内 `tl.trans`) | ~218 ms | 计算率 20 → 22 TFLOP/s,为流水化铺路 |
| 3 | **双 slot 预取流水** | **127.4 ms** | H2D 被计算完全掩盖,**1.71×** |
| 4 | tile 扫描(BM/BN/BK/BH/NS 共 48 组,T=16K) | 199.3 ms | 默认 (64,64,64,64,NS=2) 即最优点 |

### 2.4 踩过的坑(便于复现)

| 现象 | 根因 | 修法 |
|---|---|---|
| Triton 报 `Expected dtype ['fp32','fp64'] but got uint8` | 半字节算术用了 u8 | 先 `.to(tl.int32)` |
| 数值整体偏 2 的幂 | 忘了 e8m0 的 `2^(byte-127)` | `tl.exp2(scale - 127)` |
| 打包字节错位 | 偏移写成 `kk + arange(BK/2)` | `(kk>>1) + arange(BK/2)` |
| `tl.dot` 报 b 维度错 | `join+reshape` 沿列交错 | `reshape(trans(join(...)))` |
| `bincount` 报错 | profile pass 产生 -1 id | 过滤 `(ids>=0)&(ids<E)` + `A==0` 早退 |
| `cannot pin 'torch.cuda.ByteTensor'` | golden 传了 GPU 权重 | 设备无关化 |
| OOM | bf16 反量化权重 12.9 GiB/层 | 内核内反量化,只留原始字节 |
| H2D 只有 3.4 GB/s | 页式内存 | pinned 缓存 |
| pinned 缓存命中脏数据 | 按 `data_ptr` 缓存,张量释放后地址复用 | 按 (storage, offset, shape, dtype) 缓存并**持有源 storage** |
| **GPU prefill 每层慢 2.6 s、主机内存暴涨到 OOM** | `_pinned_kmajor` 的缓存键建在**临时**的转置张量上(每次调用都是新分配)→ 永远 miss → 每层重新 transpose+pin 3.19 GiB(2.6 s),并把 pinned 条目越积越多 | 缓存键改到**源张量自己的 storage**(长生命周期的模型参数)上;修后每层 0.2 s→命中 |
| 客户端 `os.environ[...] = ...` 改了阈值却不生效 | 模型跑在 EngineCore/worker 子进程里,看不到客户端的 `os.environ` 修改 | 加 `XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE`,阈值改为每层读文件 |
| TP=2 启动 `Engine core initialization failed` | **①`VLLM_ENGINE_READY_TIMEOUT_S=600` 小于加载耗时;②未设 `XIAOTU_MOE_SINGLECOPY=1`,每个 worker 持 ~2.7 份权重(430–505 GB RSS),node 0 被打满 → 内核 OOM killer 杀掉 worker** | 设 `VLLM_ENGINE_READY_TIMEOUT_S=3600` + `XIAOTU_MOE_SINGLECOPY=1` |
| 主机 OOM(1.2 TB RSS) | 引擎 scratch + 权重副本 + 重复 pin 叠加 | 同上两条 + 长 prompt 的 CPU 测量加长度上限 |

---

## 3. 对主线 vLLM 的修改与扩充清单

基线:`vllm-project/vllm` @ `6c73b08dec2af5052288169663549687ba61f330`(0.1.dev1,仅 V1 engine)。
补丁包:**`patches/mainline_sm80_mixed_mode.patch`,27 个文件,+6977 / −133**。

### 3.1 A. 通用 GPU/CPU 混合模式(与模型无关,4 个文件)

| 文件 | 改动 | 为什么必须改 |
|---|---|---|
| `vllm/envs.py` | 新增 `VLLM_EXPERTS_LOAD_DEVICE` | 让专家参数在 **CPU** 构造,否则构造期就在 GPU 上 materialize 137 GiB → OOM |
| `model_executor/layers/fused_moe/routed_experts.py` | 按开关 `with torch.device("cpu")` | 上游没有"专家权重放 CPU"的入口 |
| `model_executor/layers/fused_moe/oracle/mxfp4.py` | 混合模式下**强制** CPU 后端 + 跳过 AMX prepack | 否则 A100 会选 Marlin → `b_q_weight is not on GPU` |
| `vllm/envs.py` | 新增 5 个 `VLLM_TRITON_MLA_SPARSE*` | fork 的 Triton sparse-MLA 路径开关 |

### 3.2 B. SM80(Ampere)可移植 Triton 路径(17 个文件,主体工作量)

主线 DS-V4 只走 SM90+/SM100 的 DeepGEMM / CUTLASS / cutedsl 路径,A100(SM 8.0)
在 sparse-MLA、fp8 einsum、MHC 等处全部不可用。移植策略(纪律 #2):
**fork `Lvllmds4-x` 已有的实现逐字节采用,只做主线新版 API 适配。**

新增(与 fork 逐字节一致):

| 文件 | 行数 |
|---|---|
| `v1/attention/backends/mla/sparse_mla_env.py` | 119 |
| `v1/attention/backends/mla/sparse_mla_kernels.py` | 3517 |
| `models/deepseek_v4/nvidia/ops/sm12x_deep_gemm_fallbacks.py` | 711 |
| `models/deepseek_v4/nvidia/ops/sm12x_mqa.py` | 756 |
| `models/deepseek_v4/nvidia/ops/fp8_einsum.py` | 320 |

适配合并:

| 文件 | 要点 |
|---|---|
| `models/deepseek_v4/nvidia/flashmla.py` | 换成 fork 版(1112 行),仅适配 `combine_topk_swa_indices` 主线新签名 `out=(idx,lens)` |
| `models/deepseek_v4/attention.py` | `_fused_qnorm_rope_kv_insert` 等按主线新结构保留 |
| `v1/attention/backends/mla/{indexer,sparse_swa}.py` | SM80 跳过 DeepGEMM scheduler metadata;补 `prefill_gather_lens_cpu` |
| `models/deepseek_v4/common/ops/{cache_utils,fused_indexer_q,fused_inv_rope_fp8_quant,fused_compress_quant_cache}.py` | `.to(tl.float8e4nv)` → `_f32_to_e4m3_uint8()`(Ampere 无 fp8e4nv);fp8 缓冲以 uint8 视图传入 |
| `quantization/utils/fp8_utils.py` | 新增 `@triton.jit _e4m3_uint8_to_f32 / _f32_to_e4m3_uint8`(fork 原实现) |
| `models/deepseek_v4/compressor.py` | head=512 的 cutedsl 只在 SM89+ 用,Ampere 走 Triton |
| `utils/deep_gemm.py` | `_use_sm12x_mqa_fallback()` 扩到 SM8.x;`fp8_fp4_mqa_logits`/`paged`/`tf32_hc_prenorm_gemm` 加 SM80 分支 |
| `utils/import_utils.py` | `has_cutedsl()` 在 SM8.x 返回 False(cutedsl 只出 SM90+ PTX) |
| `models/deepseek_v4/sparse_mla.py` | `supports_compute_capability` 接受 major==8 |
| `model_executor/kernels/mhc/tilelang.py` | 首层 broadcast prenorm 走 `_torch_hc_prenorm_gemm` |
| `warmup/flashinfer_sparse_mla_warmup.py` | SM80 不做 FlashInfer sparse-MLA warmup |

### 3.3 C. 数值正确性修复(2 个文件,可独立上游)

| 文件 | 改动 | 根因 |
|---|---|---|
| `model_executor/layers/quantization/fp8.py` | `is_bmm` 特判:SM80 上不让 Marlin 重打包 `wo_a.weight`,改为就地反量化成 bf16 + `use_marlin=False` | Marlin 的 int32 打包布局会**破坏 o_proj 的 fp8 einsum**,输出数值错误(不是精度差,是错) |
| `models/deepseek_v4/nvidia/ops/o_proj.py` | 换成 fork 版:SM80 走 `deepseek_v4_fp8_einsum`(`DECODE_E4M3` + `B_BF16`),保留主线 SM90/100/110 分支 | 同上 |

### 3.4 D. 调试脚手架(上游前必须清理)

`XIAOTU_DEBUG_L1..L4` / `XIAOTU_TIMING` 埋点分布在
`models/deepseek_v4/nvidia/model.py`、`models/deepseek_v4/attention.py`、
`models/deepseek_v4/nvidia/flashmla.py`。全部 env-gated、默认关闭。

### 3.5 插件侧(不在主线,`vllm-xtu-moe` 仓库)

| 文件 | 作用 |
|---|---|
| `vllm_xiaotu_moe/hybrid_model.py` | OOT 覆盖 `DeepseekV4ForCausalLM`;`CpuXiaotuMoE` 替换 `DeepseekV4MoE`(专家参数在 CPU,gate/共享专家在 GPU) |
| `vllm_xiaotu_moe/gpu_prefill.py` | 长 prefill 逐层流式 GPU MoE(本报告主角) |
| `vllm_xiaotu_moe/mixed_experts.py` | 通用 CPU experts 后端注册表:MXFP4/FP8/INT4 三个格式,把 `_supports_current_device` 放宽到"x86 无 AMX"(见 §3.6) |
| `xiaotu_moe/csrc/` | CPU MoE 引擎(MXFP4/BF16/FP8/NVFP4/WNA16),header-only,`binding.cpp` 导出 |

### 3.6 通用 CPU experts 后端(2026-09-09 夜新增,进行中)

把"只覆盖 DS-V4"的 OOT 模型覆盖,升级为**格式无关的 CPU experts 后端**:任何使用主线
`FusedMoEFactory` 的 MoE 模型,只要权重量化格式在支持列表里,就会在混合模式下自动选中
xiaotu 引擎(不再需要模型级覆盖)。

| 机制 | 实现 |
|---|---|
| 后端注册 | `mixed_experts.register_mixed_cpu_backend()` 把 `cpu_moe.{CPUExpertsMxfp4,CPUExpertsFp8,CPUExpertsInt4}` 替换为 `XiaotuCPUExperts*`(oracle 对这些类是惰性 import,替换模块属性即可) |
| 设备选择 | `_supports_current_device()` = 混合模式(`VLLM_EXPERTS_LOAD_DEVICE=cpu`)+ x86;主线原版要求 `is_cpu()` + AMX |
| **路由** | 复用主线 router(`create_fused_moe_router`),覆盖 softmax / sigmoid+noaux_tc / **sqrtsoftplus** / grouped-topk / custom_routing_function。主线 `cpu_moe.select_experts` 硬编码 `scoring_func="softmax"`,对 GLM(sigmoid)、DS-V4(sqrtsoftplus)会**选错专家**(已列为上游 bug) |
| 激活 | `moe_config.swiglu_limit/alpha/beta` 透传到引擎 `activation_type=1`,语义对齐主线 `silu_and_mul_with_clamp`(GLM/DS-V4=10.0、MiniMax-M3=7.0) |
| 拒绝而非静默错算 | 非 SILU 激活、EP(`expert_map`)、`apply_router_weight_on_input` 目前显式 `raise NotImplementedError` |

**已验证**:

- oracle 选择(`scripts/probe_oracle.py`,不加载权重):
  GLM-5.3 fp8 块 128x128 + sigmoid/noaux_tc → `CPU / XiaotuCPUExpertsFp8` ✅(修 `_supports_routing_method` 前是 `MARLIN`);
  DS-V4 fp8 + sqrtsoftplus → 同上 ✅;INT4(WNA16)→ 支持 ✅;INT8 → 不支持(引擎无 int8,如实报告)。
- 引擎激活语义(`scripts/test_swiglu_clamp.py`,BF16 路径,5 组 limit/alpha/beta):rel_rms ≤ 1.7e-3;
  真实 DS-V4 layer-1 MXFP4 权重(`scripts/test_swiglu_clamp_mxfp4.py`):max_rel 6.8e-4;
  **真实 GLM-5.3-Flash fp8 专家权重**(`scripts/test_glm53_fp8_layer.py`,layer 3 / 8 专家):
  引擎 vs numpy 参考 **rms_rel = 9.3e-5**(引擎在 act 后转 bf16,这就是下限)。
- 顺带修掉引擎一个真实 bug:`fp8_dequant.hpp` 的 e4m3 **subnormal 解码**用了 `2^-7`(应为 `2^-6`),
  导致每个 subnormal 权重小 2 倍。修后与 `torch.float8_e4m3fn` 逐字节一致;
  GLM 权重里非零 subnormal 占 0.010%(聚合影响小,但语义必须对)。

**阻塞(与插件无关)**:GLM-5.3-Flash 在 A100/SM80 上**完全无法启动** —— 它的 MLA 维度
(`qk_nope_head_dim=256, qk_rope_head_dim=0, v_head_dim=256`)没有任何可用 attention 后端:
sparse 打开时所有 MLA 后端以 compute-capability/sparse 拒绝;用 `hf_overrides {"index_topk": null}`
关掉 sparse 后,MLA prefill 选择器只剩 FLASH_ATTN,而它只支持 (128,64,128)/(192,64,256)/(64,64,128)。
FLASHINFER 仅 Blackwell 且维度是 (128,64,128)。→ GLM-5.3 需要 SM90+;本报告用**层内真实权重**
验证代替端到端(见上)。日志:`logs/glm53_smoke.log`、`logs/glm53_smoke2.log`;关键报错摘录已入库:
`docs/evidence/glm53_a100_blocker.txt`。

---

## 4. 期望的合并方式(upstream 策略)

主线目前**没有**"CPU 专家 + 逐层流式 GPU prefill"通路,也没有 SM80 的 DS-V4 路径。
建议**拆成 4 个独立 PR + 1 个 RFC**,按"可独立评审、可独立回滚"推进:

| # | 范围 | 规模 | 评审要点 | 依赖 |
|---|---|---|---|---|
| **PR1** | `VLLM_EXPERTS_LOAD_DEVICE` + mxfp4 oracle 混合模式(A) | ~120 行 | 纯开关,默认行为不变;需 e2e 测试证明 GPU 路径无回归 | 无 |
| **PR2** | fp8 `is_bmm` / o_proj 数值修复(C) | ~60 行 | **bug fix**:SM80 上 Marlin 重打包破坏 fp8 einsum,附最小复现 | 无 |
| **PR3** | SM80 Triton sparse-MLA / fp8 einsum / MHC 回退(B) | ~7000 行 | 体量大,建议先发 **RFC issue**,再按子模块拆(attention / ops / utils) | 依赖 PR2 |
| **PR4** | 通用"CPU-offload MoE 的逐层 GPU prefill"钩子 | ~400 行 | 把 §2.2 的机制抽象成与引擎无关的接口:`VLLM_GPU_PREFILL_MIN_TOKENS` + `stream_layer_weights()/release()` 契约;MXFP4 Triton 内核可作为**参考后端**(主线 Marlin 不支持 MXFP4 block-32,这是真实空白) | 需要 RFC |
| **RFC** | "长上下文下 CPU-offload MoE 的分层 GPU prefill" | — | 说明收益边界(§5)、H2D 地板、与 `PrefetchOffloader` 的关系 | — |

**明确不打算上游的部分:** `xiaotu_moe/csrc/`(自研 CPU 引擎,属本项目);
`XIAOTU_DEBUG_*` 埋点(§3.4);逐层流式的调度策略建议先以插件形式存在,
内核可单独以上述"MXFP4 grouped MoE backend"形式上游。

**与已有上游机制的关系:** 主线已有 `model_executor/offloader/prefetch.py`
(`PrefetchOffloader` + `StaticBufferPool`),语义是"GPU 常驻 slot_capacity 层权重 +
异步预取",**要求权重放得进 VRAM**;DS-V4 的 137 GiB 放不进 40 GB,所以本方案改成
"每层算完即释放"。PR4 应复用 `PrefetchOffloader` 的 stream/event 骨架而不是另造一套。

### 4.1 决策点(D1–D8,已由项目所有者拍板 2026-09-09)

| # | 决策点 | 决定 | 执行状态 |
|---|---|---|---|
| D1 | 先发仓库还是直接提 PR | **先发仓库攒反馈** | ✅ 仓库已转 public:<https://github.com/yeungtuzi/vllm-xtu-moe>(内网地址已脱敏) |
| D2 | PR3 的代码来源与署名 | **逐字节采用 fork `Lvllmds4-x`,保留署名** | ✅ 分支已推;文件保留 SPDX 头,提交信息里写明来源 |
| D3 | PR2 单独提 | **单独小 PR** | ✅ `xtu/pr2-fp8-sm80-o-proj`(4 文件 +424/−12) |
| D4 | PR4 形态 | **先发 RFC issue** | ✅ 草稿 `patches/rfc_layerwise_gpu_prefill.md`(待你发) |
| D5 | MXFP4 内核是否上游 | **暂留插件** | ✅ 未纳入任何 PR |
| D6 | 谁提交 PR | **改成:我生成全部 PR(draft),你检查后点 "Ready for review"** | ✅ 三个 draft PR 已开:#56118 / #56119 / #56120 |
| D7 | 清理范围 | **删掉全部 `XIAOTU_DEBUG_*`/`XIAOTU_TIMING` 埋点;内部命名前缀不改** | ✅ PR3 里用 AST 精确删除(共 38 处/122 行),`grep XIAOTU` = 0;插件侧包名/env 前缀保持不变 |
| D8 | 许可证与署名 | **我方 Apache-2.0,署名"大河马(BigHippo) dahema@me.com";第三方按其 license 要求署名** | ✅ 新增 `NOTICE`、`THIRD_PARTY_NOTICES.md`;`pyproject` 作者更新 |

### 4.2 上游分支现状(D9=A:已冻结,等你点提交的动作暂缓)

> **🧊 2026-09-09 冻结**:项目目标改为「通用 CPU experts 后端(支持 DeepSeek/GLM/Qwen)」后,
> 用户拍板 **三个 draft PR 全部冻结**(不点 Ready、不改描述、不主动 rebase,只记录漂移)。
> 解冻条件与重排方案见 `docs/BACKLOG.md` §3「D9 已决」。下表为冻结时的状态快照。

| PR | 分支(`yeungtuzi/vllm`) | 规模(vs 最新 main) | Draft PR | 依赖 |
|---|---|---|---|---|
| PR1 | `xtu/pr1-experts-load-device` | 3 文件 +59/−2 | [#56118](https://github.com/vllm-project/vllm/pull/56118) | 无 |
| PR2 | `xtu/pr2-fp8-sm80-o-proj` | 4 文件 +422/−12 | [#56119](https://github.com/vllm-project/vllm/pull/56119) | 无 |
| PR3 | `xtu/pr3-sm80-port` | 21 文件 +6350/−119 | [#56120](https://github.com/vllm-project/vllm/pull/56120) | 建议在 PR2 之后 |
| RFC | — | issue 草稿 | 未发 | 建议等 PR1 有回应 |

**三个分支都已 rebase 到 `main@1454b71`(behind=0)**。PR3 需要额外说明:上游重构了
`common/ops/{cache_utils,fused_indexer_q,fused_inv_rope_fp8_quant}.py`(新的
`VllmTritonJitKernel`/`LaunchSpec` 结构),我们的改动**按新结构重新落了一遍**
(fp8 编解码改用 `_f32_to_e4m3_uint8`/`_e4m3_uint8_to_f32`、`has_cutedsl()` 加
`not is_ampere_or_ada()` 门、移植的 SM80 内核追加在文件末尾);已验证全部文件可解析、
无冲突标记,**A100 运行时验证仍待补**(PR 描述里已如实标注)。

- 标题、完整描述(含测试说明)与一键 compare 链接:`patches/UPSTREAM_PRS.md`
- PR1 与 PR3 都会改 `vllm/envs.py`(互不重叠的两组变量),合入时可能有一次
  trivial 冲突,已在 PR 说明里注明。
- PR3 的代码来自 fork `Lvllmds4-x`(Apache-2.0),只做了主线 API 适配
  (如 `combine_topk_swa_indices` 改为返回 `(idx, lens)`)。

### 4.3 上游漂移与维护成本(实测,2026-09-09)

上游 vLLM 的更新节奏(实测):**近 90 天 3682 个提交(≈41/天)**;
我们基线 `6c73b08` 之后 **36 小时内上游走了 85–86 个提交、涉及 300 个文件**,
而且在我做这几次检查的十几分钟里 `main` 又前进了一次。

**我们的三个 PR 对当天上游 main 的可应用性**(脚本 `scripts/check_upstream_drift.sh`):

| PR | 结果 | 需要人工 rebase 的文件 |
|---|---|---|
| PR1 | ⚠️ 1 文件 / **1 处冲突标记** | `fused_moe/routed_experts.py`(上游在同一位置加了 `intermediate_size_full` 的 hunk) |
| PR2 | ✅ **干净应用** | — |
| PR3 | ⚠️ 3 文件 / **6 处冲突标记** | `common/ops/{cache_utils,fused_indexer_q,fused_inv_rope_fp8_quant}.py` |

**插件侧(我们自己的代码)受上游影响很小**:

- 插件只从上游导入 **15 个模块**(见脚本 §2 的清单),`grep` 全部仍存在;
- 上游近 90 天对这些 API 文件有 9–16 次提交,但**近 1 天内没有任何一次改动
  `def`/`class` 签名**(实测 `cpu_moe.py` / `modular_kernel.py` / `config.py` /
  `gate_linear.py` / `fused_topk_bias_router.py` 的签名改动数 = 0)。
- 风险形态是"改名/删函数"这类破坏性变更 —— 一旦发生,插件在 **import 时立刻报错**,
  不会静默算错。

**会被上游高频繁改动的"高危文件"**(近 90 天提交数):

| 提交数 | 文件 |
|---:|---|
| 79 | `vllm/envs.py` |
| 28 | `models/deepseek_v4/attention.py` |
| 26 | `models/deepseek_v4/nvidia/model.py` |
| 26 | `fused_moe/oracle/mxfp4.py` |
| 25 | `fused_moe/routed_experts.py` |
| 22 | `quantization/fp8.py` |
| 17 | `v1/attention/backends/mla/indexer.py` |

**结论与维护建议:**

1. **不是"基本不受影响",而是"要按天跟"** —— 但冲突量很小(PR1 一行、PR3 六处),
   因为我们的改动 ~90% 是**新增**(新文件、新分支、新函数),不是重写既有逻辑。
2. 提 PR 前**必须**先跑 `scripts/check_upstream_drift.sh`,必要时 rebase;
   建议每周至少跑一次,或在上游合并高峰后跑。
3. **能上游的就尽快上游**:补丁被主线吸收后,本地维护成本直接归零
   (插件只剩 ~1200 行 Python)。
4. 插件侧建议加**版本守卫**:导入失败时给出"当前支持的 vLLM 版本区间"而不是裸 import 错误。
5. 若上游重构 `FusedMoE` 的模块化接口(`modular_kernel`/`config`)或重写 DS-V4 的
   attention 路径,那就是"大改",届时插件与 PR3 都要跟着改 —— 这是唯一的系统性风险点。

---

## 5. 本机实测报告

> 测量纪律:同一次进程内、同一 prompt、连续测量;原始数据见
> `report/curve_tp1.jsonl`、`report/curve_tp2.jsonl`、`report/moe_micro*.json`、
> `report/golden.txt`。

### 5.1 方法

| 指标 | 方法 |
|---|---|
| TTFT | `LLM.generate(prompt, max_tokens=1)` 墙钟(含 tokenize + 调度 + prefill + 采样) |
| 纯 MoE 吞吐 | 单层微基准 × 43 层折算(`bench_gpu_moe_prefetch.py`),权重形状/字节数与真实权重一致 |
| 正确性 | `gpu_prefill_golden.py` / `gpu_prefill_prefetch_test.py` vs 纯 torch 参考(同一 MXFP4 布局) |
| 并发 | 同一 prompt 复制 c 份一次 `generate`,报告聚合 tok/s |
| 分解 | 端到端 TTFT − 43 × 单层微基准 = 非 MoE 部分(注意力/indexer/路由/采样) |
| 预热 | 首个 GPU prefill 会构建 pinned 缓存(~2.6 s/层)并 JIT 内核,测量前先跑一次预热并单独记录 |

### 5.2 正确性

`report/golden.txt`(同步路径)与 `gpu_prefill_prefetch_test.py`(slot / 2-slot ring
复用)对纯 torch 参考实现全部通过:

| 用例 | 维度 | max_abs | **RMS 相对误差** |
|---|---|---|---|
| 小维度 | H=256 I=128 E=8 T=32 K=3 | 16 | 4.1e-3 |
| 小维度(ragged M) | H=256 I=128 E=8 T=7 K=2 | 8 | 3.6e-3 |
| **全 DS-V4 维度** | H=4096 I=2048 E=256 T=4 K=6 | 1024 | **5.6e-3** |
| slot 路径 + ring×4 | 同上 | 16 / 8 | 同量级 |

> RMS 相对误差 3.6e-3 ~ 5.6e-3 ≈ bf16 的 `2^-8 = 3.9e-3`,即误差就是**输出 bf16
> 量化精度**本身(全维度用例输出幅值 ~1.6e4,1 ulp = 128,实测 max_abs 1024 = 8 ulp)。
> 参考实现本身把 gate/up 中间结果舍入到 bf16,精度还低于我们的 fp32 累加路径。

### 5.3 单卡:TTFT vs 上下文长度

同一进程内连续测量(CPU / GPU 两种模式、同一 prompt),`XIAOTU_MOE_SINGLECOPY=1`,
`max_model_len = max_num_batched_tokens = 16384`,阈值 2048:

| prompt tokens | CPU prefill TTFT | CPU tok/s | **GPU prefill TTFT** | **GPU tok/s** | 加速 |
|---|---:|---:|---:|---:|---:|
| 2091 | 112.93 s | 18.5 | **5.638 s** | 370.9 | **20.0×** |
| 4137 | 439.38 s | 9.4 | **5.763 s** | 717.9 | **76.2×** |
| 8229 | (未测,主机内存不足) | — | **8.639 s** | 952.5 | — |
| 16039 | (未测) | — | **18.237 s** | 879.5 | — |

**读数要点:**

- **GPU 路径在 T ≤ 4K 完全由 H2D 地板支配**:2091 与 4137 token 的 TTFT 几乎相同
  (5.64 vs 5.76 s),与 §1.5 算出的 5.5 s 地板一致。
- **CPU 路径随 T 超线性劣化**:2091→4137 token(2×)耗时 112.9→439.4 s(3.9×)。
  原因是它逐层读取全部 256 个专家的权重(3.4 GB/层)且引擎 scratch 随 batch 膨胀。
  因此 GPU 的相对收益随上下文迅速放大:**20× → 76×**。
- 8K 起计算 + 注意力开始显现(8.64 s);16K 端到端 18.24 s。
- **分解(第 2 轮用进程内 torch profiler 实测,不再是减法估算)**:见 §5.10。
  16K 的一次 prefill 里,**MoE 内核 ~9.0 s(约一半墙钟)**,其余是 dense 层
  (o_proj fp8 einsum 1.6 s、Marlin GEMM 0.9 s、sgemm/bmm 0.7 s)、
  大量逐元素小内核(~1.0 s / 约 2700 次)、以及未被重叠掉的 H2D。
  **注意力/indexer 内核只有 ~0.85 s —— 它不是瓶颈**(第 1 轮用"TTFT − 微基准"推出的
  "9.7 s 注意力"是减法伪影,已在 §5.10 更正)。
- 首 token 文本:2K 时 CPU 与 GPU 都输出 `'2'`;4K 时 CPU `'1'` / GPU `'2'`
  —— 两者是**不同的量化内核**(CPU 引擎 MXFP4 vs GPU 内核 MXFP4,累加顺序与中间
  精度不同),按纪律 #5,不追求字节一致,追求数值在 bf16 精度内(§5.2)。

### 5.4 双卡:A100 ×2(TP=2 + 专家并行)

`CUDA_VISIBLE_DEVICES=0,1 TP=2`,每 rank 只流 `E/2=128` 个专家(1.59 GiB/层)、只算自己
那半 token-expert pair,再对 bf16 部分和做 all-reduce;`XIAOTU_MOE_SINGLECOPY=1`;
`max_model_len = max_num_batched_tokens = 16384`;阈值 2048。

| prompt tokens | TP=1 GPU | **TP=2 GPU** | 加速 |
|---|---:|---:|---:|
| 4137 | 5.763 s(717.9 tok/s) | **3.993 s(1036.2 tok/s)** | 1.44× |
| 16039 | 18.237 s(879.5 tok/s) | **13.956 s(1149.3 tok/s)** | **1.31×** |

并发(2091-token prompt × c 份):

| c | TP=1 聚合 | **TP=2 聚合** | 加速 |
|---|---:|---:|---:|
| 1 | 370.9 tok/s | **715.9 tok/s** | **1.93×** |
| 2 | 371.8 | **717.6** | 1.93× |
| 4 | 605.7 | **978.6** | 1.62× |
| 8 | 677.2 | **1015.8** | 1.50× |

**读数要点:**

- 2K–4K 区间(完全 H2D 受限)几乎线性加速(**1.93×**):每 rank 的地板从 5.48 s 降到
  2.74 s,实测 63.7 ms/层 —— 与 §5.6b 的等效微基准一致。
- 16K 端到端只加速 1.31×:TP=2 把 MoE 砍半(按 §5.6b 的每 rank 5.6 µs/token/层,
  16K 时 MoE ≈ 4.31 s),但 **dense 层 / 逐元素小内核 / H2D 这几块开销不随 TP 变**,
  于是它们的占比上升(§5.10 的分解:注意力只占 2.9%,不是这里的原因)。
- **TP=2 端到端 16K = 1149 tok/s**,距 1500 还差 1.3×,差的是那几块分散开销。
- decode 反而变慢(TP=2 1.71 tok/s vs TP=1 4.49 tok/s):CPU decode 在两个 rank 上
  做**冗余的全量计算**(每 rank 都算全部 256 专家),这是 §7.6 的待办。
- 加载耗时 942 s(2 个 worker 各自加载 156 GB + 建 43 个引擎),首次启动需要
  `VLLM_ENGINE_READY_TIMEOUT_S=3600` 与 `XIAOTU_MOE_SINGLECOPY=1`(§2.4)。

### 5.5 并发与聚合吞吐

单卡,同一 2091-token prompt × c 份,一次 `generate`:

| 并发度 c | 总 token | 墙钟 | 聚合吞吐 |
|---|---:|---:|---:|
| 1 | 2091 | 5.637 s | 370.9 tok/s |
| 2 | 4182 | 11.247 s | 371.8 tok/s |
| 4 | 8364 | 13.810 s | 605.7 tok/s |
| 8 | 16728 | 24.702 s | 677.2 tok/s |

**已定位(第 2 轮,含真实 HTTP 服务端复测)。** 给插件加 `XIAOTU_DEBUG_QLEN=1`
打印每个 MoE forward 的 qlen 后,真相是**请求到达时序**而非调度策略:

| c | 墙钟 | 聚合 tok/s | 每个 step 的 token 数 |
|---:|---:|---:|---|
| 1 | 6.08 s | 337 | `[2111]` |
| 2 | 11.25 s | 364 | `[2111]` + `[2111]` |
| 4 | 13.83 s | 592 | `[2111]` + `[6333]` |
| 8 | 24.87 s | 659 | `[2111]` + `[14777]` |

引擎一有第一个请求就开 step,其余请求晚几毫秒到达 → **永远"1 个独占一步 + 其余合批一步"**,
H2D 被流 2 次。但合批本身有效:c=8 时 7 个请求共享一次权重流,比顺序执行快 2.4×。
**上限**是"一次大 prefill"的吞吐(16K ≈ 900 tok/s),并发只能摊掉地板、不能突破它。
要再进一步需要上游调度器加一个微批窗口(估算 1.2–1.5×)。

> 测量陷阱(已记录):用**完全相同**的 prompt 测并发时,vLLM 默认开启的 prefix cache
> 会让第 2 个起全部命中缓存,c=8 假报 2003 tok/s。必须让 prompt 首 token 就不同。

**decode 吞吐(对照)**:64 个 token 用了 14.26 s = **4.49 tok/s** —— decode 仍走
CPU 引擎(逐 token 读 6 个专家/层),是另一条独立的优化线(`docs/THREAD_GEOMETRY.md`)。

### 5.6 消融:重叠 / tile / 交叉点

**(a) 预取重叠(`report/moe_micro.json`,E=256,真实层形状,43 层折算)**

| T | 无重叠 ms/层 | 重叠后 ms/层 | 加速 | 无重叠 tok/s | **重叠后 tok/s** |
|---|---|---|---|---|---|
| 2048 | 189.2 | 127.5 | 1.48× | 252 | 373 |
| 4096 | 217.6 | 127.4 | 1.71× | 438 | 748 |
| 8192 | 265.6 | 127.4 | 2.08× | 717 | 1495 |
| 16384 | 356.1 | 198.7 | 1.79× | 1070 | **1918** |
| 32768 | 539.0 | 382.2 | 1.41× | 1414 | **1994** |
| 49152 | 722.1 | 567.9 | 1.27× | 1583 | **2013** |

- 拟合(重叠后):`t(T) = max(127.4 ms, 15.5 ms + 11.2 µs × T)`;H2D 地板 = 127.4 ms。
- **1500 tok/s 交叉点从 41K 提前到 8K**(纯 MoE 层面)。

**(b) TP=2 专家并行等效微基准**(`report/moe_micro_tp2.json`;E=128 + topk=3,
即每 rank 只算自己那半的 pair)

| T | 无重叠 ms/层 | 重叠后 ms/层 | **重叠后 tok/s** |
|---|---|---|---|
| 2048 | 112.7 | 63.8 | 746 |
| 4096 | 108.9 | 63.7 | 1495 |
| 8192 | 131.7 | 63.7 | 2989 |
| 16384 | ~200 | 100.2 | **3803** |
| 32768 | ~400 | 197 | **3979** |
| 49152 | — | 292 | **4038** |

- 每 rank H2D 地板降到 63.7 ms(1.59 GiB);1500 tok/s 交叉点 ≈ **11.4K**。

> **第 2 轮更新**:`_down_kernel` 的输出累加改成 `sem="relaxed"` 原子加后,
> T=16K 的每层耗时 198.7 → **191.4 ms**(吞吐 1918 → **1991 tok/s**),
> T=32K 1994 → **2066 tok/s**;golden 全通过。上表仍是"默认原子加"那一版的数据,
> 保留作为对比。测到的下一档:换成"per-slot 缓冲 + 归约"可到 177.4 ms(+6.3%)。

**(c) tile 扫描**(`scripts/sweep_gpu_moe_tiles.py`,T=16384,48 组):

| 配置 | ms/层 |
|---|---|
| **(BM,BN,BK,BH,NS)=(64,64,64,64,2)** | **199.3(最优)** |
| (64,128,64,64,2) | 207.1 |
| (64,64,128,64,2) | 258.0 |
| (64,64,128,128,2) | 1741.8(寄存器溢出) |
| (128,256,128,128,2) | 2812.5(寄存器溢出) |

→ 默认配置已是局部最优,不需要改。

**(d) CPU / GPU 交叉点。** CPU 路径的实测每 token 每层成本(由 §5.3 的端到端反推):

| CPU 配置 | 实测 | 每 token/层 |
|---|---|---|
| `XIAOTU_MOE_SINGLECOPY=1` @2091 token | 112.93 s | 1.26 ms |
| `XIAOTU_MOE_SINGLECOPY=1` @4137 token | 439.38 s | 2.47 ms |
| 2 份 socket 副本 @2091 token | 55.86 s | 0.62 ms |
| 2 份 socket 副本 @4096 token | 77.72 s(早期测量,机器高负载) | 0.44 ms |

GPU 路径重叠后是 `127.4 ms/层 + 11.2 µs/token/层`。令二者相等:

```
43 × c_cpu × T = 43 × (0.1274 s + 11.2µs × T)
c_cpu = 0.44 ms  →  T ≈ 300 token
c_cpu = 1.26 ms  →  T ≈ 100 token
```

⇒ **交叉点只有 100–300 token**,远低于 `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS` 的默认
2048。也就是说:GPU 路径的固定成本(5.5 s)虽然可观,但 CPU 路径实在太慢
(每 token 每层 0.44–2.5 ms,而 GPU 只需 11.2 µs),**阈值可以大幅下调**(§7.6)。

### 5.7 与目标的差距

**目标:端到端 1500 tok/s。** 实测(16K prompt):

| 配置 | 16K 端到端 | MoE 内核 | 非 MoE(GPU 内核) | 达标还差 |
|---|---:|---:|---:|---:|
| TP=1 | 18.24 s(880 tok/s) | 9.05 s(§5.10 实测) | ~5.3 s 内核 + H2D/调度开销 | 1.67× |
| **TP=2(EP)** | **13.96 s(1149 tok/s)** | **~4.3 s**(每 rank 一半专家) | 同上 | **1.28×** |

- **TP=2 把 MoE 砍半(8.54 → 4.31 s),但端到端只快 1.31×** —— 因为非 MoE 部分没变,
  它现在占了 69% 的 TTFT。
- 按 §5.10 的实测分解,16K 的非 MoE 开销是:逐元素小内核(大量)+ dense GEMM
  (o_proj fp8 einsum 最大)+ H2D 未重叠部分 + 启动调度开销;**注意力/indexer 只占 2.9%**。
  ⇒ **要达标需要把这堆分散开销从 ~9 s 压到 ≤6.4 s**,而不是去优化注意力。
- **换个角度,聚合吞吐目标已接近达成**:TP=2 并发 c=8 时聚合 1015.8 tok/s;
  按 §5.8 的模型,若并发请求能真正落在同一个 prefill step(§5.5 的待办),
  16K 级 batch 的聚合吞吐应为 `总 token / (H2D + 计算)` ≈ 1500–2000 tok/s。

### 5.9 GPU prefill 阈值推荐表(实测)

`VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS` 决定"多长的 prefill 才值得走 GPU"。
在**本机配置**(1×A100-PCIE-40GB,EPYC 9654,`XIAOTU_MOE_SINGLECOPY=1`,
`XIAOTU_MOE_THREADS=96`,`max_model_len = max_num_batched_tokens = 16384`)上逐长度实测:

| prompt tokens | CPU prefill TTFT | GPU prefill TTFT | GPU 相对收益 |
|---:|---:|---:|---:|
| 139 | 5.69 s(含首次调用预热) | **5.57 s** | 1.02× |
| 275 | 6.53 s | **5.57 s** | 1.17× |
| 411 | 9.85 s | **5.58 s** | 1.77× |
| 548 | 12.25 s | **5.66 s** | 2.16× |
| 793 | 18.05 s | **5.67 s** | 3.18× |
| 1057 | 23.52 s | **5.65 s** | 4.16× |
| 2091 | 92.60 s | **5.71 s** | 16.2× |
| 4137 | 439.38 s | **5.76 s** | 76.3× |
| 8229 | — | **8.61 s** | — |
| 16039 | — | **17.74 s** | — |

读数:

- **GPU 侧在 T ≤ 4K 是常数 ~5.6 s**(H2D 地板,§1.5),与 token 数无关;8K 起
  计算/注意力开始显现(8.61 s),16K 为 17.74 s。
- **CPU 侧约 22.4 ms/token**(548–1057 的稳定区间),2091 起跳到 44 ms/token
  (引擎 scratch 随 batch 增长)。
- **交叉点**:`5.57 s ÷ 22.4 ms/token ≈ 249 token`。若 CPU 用 2 份 socket 权重副本
  (不用 `SINGLECOPY=1`,多 ~150 GB 内存、CPU 快约 2.2×),交叉点右移到 **≈ 550 token**。

**推荐起始值(仅限本硬件配置,其他机器请自行实测):**

| CPU 配置 | 推荐阈值 | 理由 |
|---|---|---|
| `XIAOTU_MOE_SINGLECOPY=1`(serve 脚本默认) | **384** | 越过 ~249 的交叉点并留 1.5× 余量,且避开 <200 token 由"首次调用预热"主导的区间 |
| 2 份 socket 副本(内存换 CPU 速度) | **512–768** | 交叉点 ~550 |

> ⚠️ 这只是**本机(2×A100 + EPYC 9654 + DDR5-4800)的实测结论**。换 GPU/CPU/模型,
> 或改 `XIAOTU_MOE_THREADS` / `SINGLECOPY`,交叉点都会移动 —— 建议用户用
> `scripts/dsv4_prefill_curve.py`(扫几个长度 + `MODES=cpu,gpu`)在自己机器上重跑再定。
> 原始数据:`report/curve_thr.jsonl`、`report/curve_small.jsonl`。


### 5.10 内核级分解(进程内 torch profiler,16K prefill)

**为什么要重做**:第 1 轮用"端到端 TTFT − 单层微基准 × 43"来估算非 MoE 部分,得到
"注意力/indexer 9.7 s"。第 2 轮用进程内 profiler 直接测(模型跑在 vLLM 的 worker 进程里,
外部 nsys 看不到它的 kernel;改成插件内 `torch.profiler` 钩子才拿到数据):

| 分组 | 2 次 pass 的 Self CUDA | 每次 pass | 占比 |
|---|---:|---:|---:|
| **MoE 内核(本项目)** `_gate_up_kernel` + `_down_kernel` | 18.09 s | **9.05 s** | 31% |
| 逐元素/其它(`add_`/`mul_`/`clamp_min_` 等,~2700 个小内核/pass) | 16.70 s | 8.35 s | 29% |
| H2D memcpy(pinned → device) | 10.94 s | 5.47 s | 19% |
| dense GEMM(o_proj fp8 einsum 1.57 s、Marlin 0.92 s、sgemm/bmm 0.73 s、hc_prenorm 0.31 s) | 10.48 s | 5.24 s | 18% |
| **注意力 / indexer**(`_indexed_d512_split_*`、rope、sink、accumulate) | 1.70 s | **0.85 s** | **2.9%** |

单内核 Top(2 次 pass 合计):

| 内核 | 时间 | 调用数 |
|---|---:|---:|
| `Memcpy HtoD (Pinned -> Device)` | 10.94 s | 536 |
| `_down_kernel`(本项目) | 10.12 s | 83 |
| `_gate_up_kernel`(本项目) | 7.97 s | 83 |
| `_deepseek_v4_sm12x_fp8_einsum_kernel`(o_proj) | 3.13 s | 257 |
| `_C::marlin_gemm` / `marlin::Marlin<…>` | 1.87 / 1.83 s | 1327 / 2712 |
| `aten::bmm` / `ampere_sgemm_32x128_nn` | 1.46 / 1.15 s | 5499 / 4032 |
| `aten::add_` / `elementwise_kernel` | 0.92 s | 5376 |
| `hc_prenorm_gemm_block_m_tilelang_kernel` | 0.63 s | 169 |
| `_indexed_d512_split_score_kernel` | 0.51 s | 2583 |
| `_indexed_d512_split_value_kernel` | 0.32 s | 2583 |

**结论(更正)**:

- **瓶颈不是注意力/indexer**(仅 2.9%)。真正的大头依次是:
  ①MoE 内核 9.05 s/pass;②海量逐元素小内核(~1.0 s/pass 的 kernel 时间,但 2700 次
  启动/pass 的调度开销可能更大);③H2D 5.47 s/pass(与计算有重叠,但实测并未完全藏住);
  ④dense 层 GEMM 5.24 s/pass,其中 **o_proj 的 fp8 einsum 1.57 s 是最大的单个非 MoE 内核**。
- 因此 §7.1 的优先级要改:先做"减少逐元素小内核 / 融合",再做 o_proj einsum 与 Marlin,
  注意力/indexer 放到最后。**"1500 tok/s 的最后一公里"不是注意力,而是这些分散的开销。**
- 数据:`report/prof16k_kernels.txt`(profiler 表)、`scripts/dsv4_profile_prefill.py` +
  插件内 `XIAOTU_TORCH_PROFILE` 钩子。

### 5.8 性能模型小结

```
单卡, 43 层, 重叠后:
  T <= 11.4K :  t_layer = 127.4 ms                (H2D 受限)
  T >  11.4K :  t_layer = 15.5 + 11.2µs × T       (计算受限)
  TTFT_MoE = 43 × t_layer
  端到端   = TTFT_MoE + attn(T) + 路由/采样开销

双卡 EP(每 rank 一半专家、一半 pair):
  T <= 11.4K :  t_layer = 63.7 ms
  T >  11.4K :  t_layer = 7.8 + 5.6µs × T
  端到端   = 43 × t_layer + attn(T) + all_reduce

非 MoE 部分(由 §5.3/§5.4 端到端减去 MoE 得到,两卡基本相同):
  attn(T) ≈ 0.3 s @4K, 3.2 s @8K, 9.7 s @16K   (≈ O(T^1.6),见 §7.1)
```

---

## 6. 图表

`report/fig/`(`report/make_figs.py` 从 JSONL 直接绘制):

| 图 | 内容 |
|---|---|
| `fig1_hw_bandwidth.png` | 主机内存 STREAM(3 档)+ PCIe H2D/D2H + HBM D2D |
| `fig2_ttft_vs_len.png` | TTFT vs prompt 长度(CPU / GPU / GPU×2),含交叉点 |
| `fig3_moe_throughput.png` | 纯 MoE tok/s vs T(无重叠 / 重叠 / 2 卡),1500 目标线 |
| `fig4_scaling.png` | 1 卡 vs 2 卡:TTFT 与端到端吞吐 |
| `fig5_kv_capacity.png` | 每 token KV 与可用上下文 |
| `fig6_concurrency.png` | 聚合吞吐 vs 并发度 |

---

## 7. 还能提升/优化的空间(按预期收益排序)

### 7.1 (已更正)真正剩下的是"分散开销",不是注意力

> **第 2 轮实测更正**:§5.10 的 profiler 分解显示注意力/indexer 内核只占 **2.9%**
> (0.85 s/pass),第 1 轮"9.7 s 注意力"是减法伪影。下面保留原文以便追溯,但优先级要按
> 新的顺序:**①逐元素小内核(约 2700 次/pass)②o_proj fp8 einsum(1.57 s)
> ③Marlin/sgemm(1.65 s)④H2D 重叠缺口 ⑤注意力/indexer(最后)**。

### 7.1b (原分析,已更正)注意力 / indexer

- **实测(§5.3/§5.4 端到端减去 MoE)**:非 MoE 部分 ≈ **0.3 s @4K、3.2 s @8K、
  9.7 s @16K**(≈ O(T^1.6));TP=2 下 16K 的 TTFT 里它已占 **69%**(9.65 / 13.96 s),
  而且**双卡没有让它变快**(TP=1 9.70 s vs TP=2 9.65 s)。
- 构成:稀疏 MLA(compress_ratio 4 与 128 两套 KV)+ DSA indexer(打分
  O(T²/compress),compress=4 的层贡献最大)+ 路由/采样。
- 可做:①indexer 打分融合/分块(已有 `fused_indexer_q`,可再合
  `fused_compress_quant_cache`);②indexer KV 用 fp4 缓存(`indexer.py` 已有
  `dsa_indexer_uses_fp4`,但当前实现**只允许 SM100+**,A100 需要新增 SM80 的 mxfp4
  indexer 路径);③`index_topk=512` 对长上下文自适应调小;④让注意力也随 TP 分片
  (indexer 目前两个 rank 做同样的活);⑤SM80 Triton 路径的 tile 调优。
  **目标是把 9.7 s 压到 ≤6.4 s(1.5×)—— 这就是 1500 tok/s 的最后一公里。**

### 7.2 MoE 计算率(7% MFU → 目标 20%)

- 现状 22 TFLOP/s(T=16K 单卡),A100 bf16 峰值 312 TFLOP/s。瓶颈是**内核内反量化**
  的 ALU 开销 + `BN=64` 的窄 tile(扫描证明加宽反而溢出,说明当前内核结构已到边界)。
- 可做:①反量化改成 shared-memory LUT(16 项 e2m1 表)替代算术展开;②两半
  (even/odd k)分开做 `tl.dot`,消掉 `join+trans+reshape` 的寄存器搬移;
  ③直接写 CUTLASS/CUDA 的 weight-stationary grouped MXFP4 内核(ktransformers/lk 路线),
  预期 **2–3×**。按 2× 估算:纯 MoE T=32K 从 382 → ~190 ms/层,吞吐 1994 → **~4000 tok/s**。
- 优先级已下降:**TP=2 已经用另一条路(每 rank 一半专家)把 MoE 砍半了**
  (16K:8.54 → 4.31 s),继续压 MoE 的边际收益不如攻 §7.1。

### 7.2b CPU 引擎的真实瓶颈(第 12 版更正)

**先认错**:第 4 版说"CPU 引擎瓶颈是反量化 ALU",这是错的;而当时用来支撑它的
"BF16 预反量化只快 1.2×"实验也**不成立**。两个问题叠在一起:

1. **微基准的路由是错的**。`bench_cpu_engine.py` 原本用
   `np.tile(ids, (B,1))` —— 所有 token 命中**同样的 6 个专家**(每专家 2048 行),
   而真实模型是 **256 个专家 × 每专家约 48 行**。固定路由把权重工作集从 3.2 GB(DRAM)
   压到几十 MB(L3),数字完全不同。
2. **BF16 对照不是同一个 kernel**。MXFP4 路径是**批处理**的(M = 该专家的行数,
   4 行分块),BF16 路径在 `BF16WeightTraits::gate_up_impl` 里是
   `matmul_bf16(..., 1, inter, hidden)` —— **M=1 的矩阵-向量乘**。两者结构不同,
   不能用来隔离"反量化"的成本。

**修正后的实测**(96 线程,真实逐 token 随机路由,引擎自带 `XIAOTU_MOE_PROFILE=1` 计时):

| 批次 B | 每层耗时 | 每 token/层 | 算力 | gate_up(A) | down(B) | 组合(C) |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 89.1 ms | 0.174 ms | 1.73 TFLOP/s | – | – | – |
| 2048 | 315.7 ms | 0.154 ms | 1.96 TFLOP/s | 182 ms | 88 ms | 8.6 ms |
| 8192 | 1235.0 ms | 0.151 ms | 2.00 TFLOP/s | – | – | – |

- **每 token 成本随 batch 改善**(0.174 → 0.151 ms),不是恶化;
- **99.7% 的时间在两个 GEMM 相位**(A 57% / B 28% / C 3%);
- 算力 **1.7–2.0 TFLOP/s ≈ AVX-512 BF16 峰值(11.4 TFLOP/s)的 15–18%**;
- 权重流量约 **10 GB/s**,远低于 740 GB/s ⇒ **不是带宽受限**;
- **反量化占比未被任何实验确定** —— 要测它必须给**同一个批处理 kernel**加一个
  "跳过解码"的编译开关(结果错、指令数对齐),而不是换 kernel。

**BF16 预反量化(真实路由)**:MXFP4 320 ms/层 vs BF16 **4628 ms/层** ⇒ **慢 14×**,
因为那条路径是 M=1 matvec,在"256 专家 × 48 行"的分布下完全没有批处理收益。

**顺带发现的第二个缺口**:真实模型里 CPU prefill 是 **2.15 s/层**
(2091 token 共 92.6 s),而隔离引擎只有 **0.32 s/层** ⇒ **6.7× 的差距来自混合路径的
每层开销**(D2H/H2D、逐层引擎调用、43 个引擎各自的 scratch),不是 GEMM 本身。
这是 CPU 侧下一个真正值得优化的点。

### 7.3 摊销 / 消除 H2D 地板

- 地板 5.48 s/次 prefill(137 GiB @27 GB/s)。三种摊法:
  ①**并发/连续批处理**:一个 batch 里多个请求共享同一次权重流(权重与 batch 内 token
  数无关)。**实测没有拿到预期收益**(§5.5:c=1→8 只从 371 升到 677 tok/s,c=2 的墙钟
  几乎正好是 c=1 的两倍)⇒ 先要定位"为什么这些请求没落进同一个 prefill step"
  (vLLM V1 调度、`max_num_batched_tokens`、请求提交方式三者之一),这是**最容易
  拿到 2–4× 聚合吞吐的一条路**;
  ②**TP=2 + EP**:每 rank 只流 73.5 GiB → 地板 2.74 s(实测 63.7 ms/层);
  ③**层内流水**:把 gate_up 的 K 分块与 H2D 交错,小 T 时也能掩盖(当前只做了层间流水)。
- 另:只流"本 batch 真正命中的专家"——T ≥ 1K 时 256 个专家几乎全命中,收益≈0;
  但 T < 1K 时可把地板降到 ~0,从而把阈值再往下推。

### 7.4 一次性启动成本(pinned 缓存)

- 首个 GPU prefill 要构建 147 GB 的 pinned K-major 缓存:实测 `pin_memory` 只有
  **1.6 GB/s**、`transpose().contiguous()` 7.4 GB/s → **~2.6 s/层 ≈ 114 s**。
- 可做:①用 `cudaHostRegister` 就地 pin(实测 0.37 s / 2 GiB,比 `pin_memory` 快
  **3.7×** 且省一份内存);②多线程并行(实测 8 线程 3.5 GB/s,约 2×);
  ③启动时预建而非首个请求时惰性建。合计可把 114 s 压到 **~20 s**。

### 7.5 KV 容量(hundreds-of-K 上下文的硬约束)

- 400 KiB/token ⇒ 单卡 44K token 上限;要跑 128K+ 必须:①KV 量化(fp8/fp4 KV cache,
  2–4×);②更多卡(2 卡 ≈ 88K);③indexer KV 换 fp4(见 7.1②)。

### 7.6 其它

- **阈值可以大幅下调**:实测交叉点只有 100–300 token(§5.6d),而默认阈值是 2048。
  在 1K–2K 的 prefill 上 GPU 路径已经快 5–20×,建议把默认值改到 512–1024,
  或改成基于 `T × per_token_cost > H2D_floor` 的在线估算(见下)。
- **decode 路径**:当前 `enforce_eager=True`,decode 未用 CUDA graph;decode 走 CPU
  MoE(逐 token),优化见 `docs/THREAD_GEOMETRY.md`。
- **阈值自适应**:交叉点随机器负载/GPU 型号变化,建议把固定阈值换成基于
  `T × per_token_cost > H2D_floor` 的在线估算。
- **CPU 路径的专家并行**:TP=2 时 CPU 路径目前两个 rank 各算全量(冗余 2×),
  若把 CPU 引擎也做 EP 分片,CPU prefill 也能加速近 2×。

---

## 附录 A. 复现命令

```bash
# 环境
source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate vllm-xiaotu-moe
MODEL=/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/master

# 硬件核验
lscpu; numactl -H; nvidia-smi topo -m; nvidia-smi -q
report/stream                              # OpenMP STREAM(report/stream.c)
python report/pcie_bw.py

# 正确性
TEST_FULL=1 python scripts/gpu_prefill_golden.py          # 同步路径(全维度)
python scripts/gpu_prefill_prefetch_test.py               # slot / ring 路径

# 纯 MoE 微基准(无重叠 vs 重叠;E=256 与 E=128/topk=3 等效 EP)
TS=2048,4096,8192,16384,32768,49152 E_LIST=256,128 \
  OUT=report/moe_micro.json python scripts/bench_gpu_moe_prefetch.py
TS=2048,4096,8192,16384,32768,49152 E_LIST=128 TOPKS=3 \
  OUT=report/moe_micro_tp2.json python scripts/bench_gpu_moe_prefetch.py
TS=16384 python scripts/sweep_gpu_moe_tiles.py            # tile 扫描

# 端到端(单卡;CPU + GPU 同进程对比。阈值必须在 shell 里导出 —— 模型跑在
# EngineCore/worker 子进程里,客户端 os.environ 改不动它;运行期切换用 thr.txt)
TEST_MODEL=$MODEL TP=1 GLM_MAXLEN=16384 MAX_NBT=16384 MODES=cpu,gpu GPU_UTIL=0.85 \
  SKIP_CPU_ABOVE=4096 XIAOTU_MOE_SINGLECOPY=1 VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=2048 \
  OUT=report/curve_tp1.jsonl python -u scripts/dsv4_prefill_curve.py

# 端到端(双卡 TP=2 + EP;加载 ~15 min,必须放大 ready 超时 + SINGLECOPY)
CUDA_VISIBLE_DEVICES=0,1 TEST_MODEL=$MODEL TP=2 GLM_MAXLEN=16384 MAX_NBT=16384 \
  MODES=gpu GPU_UTIL=0.85 XIAOTU_MOE_SINGLECOPY=1 VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  OUT=report/curve_tp2.jsonl python -u scripts/dsv4_prefill_curve.py

# 图表
python report/make_figs.py
```

---

## 修订记录

- **2026-09-09(第 14 版)** — §4.2 记录 **D9=A:三个 draft PR 全部冻结**(用户拍板),
  解冻条件见 `docs/BACKLOG.md` §3。依据:用户 2026-09-09 选择。
- **2026-09-09(第 13 版)** — 新增 §3.6「通用 CPU experts 后端(进行中)」:①
  `mixed_experts.py` 从"只覆盖 DS-V4 的 OOT 覆盖"升级为格式无关后端(MXFP4/FP8/INT4),
  路由改用主线 router(修掉主线 `cpu_moe.select_experts` 硬编码 softmax 的 bug 影响);
  ②引擎实现 `swiglu_limit/alpha/beta`(语义对齐 `silu_and_mul_with_clamp`)并给出三路数值验证
  (BF16 rel_rms ≤1.7e-3 / 真实 DS-V4 MXFP4 max_rel 6.8e-4 / **真实 GLM-5.3 fp8 专家 rms_rel 9.3e-5**);
  ③修掉引擎 e4m3 subnormal 解码 bug(`2^-7→2^-6`);④记录 **GLM-5.3-Flash 在 A100 无法启动**
  的确切原因(MLA 维度 256/0/256 无后端)与替代验证方式。依据:`scripts/probe_oracle.py`、
  `scripts/test_swiglu_clamp*.py`、`scripts/test_glm53_fp8_layer.py`、`logs/glm53_smoke*.log`。
- **2026-09-09(第 12 版)** — **更正两个错误结论**(用户指出):①"CPU 引擎瓶颈是
  fp4→bf16 反量化 ALU"不成立;②早期 CPU 微基准用了**固定路由**(所有 token 命中同样
  6 个专家),导致 CPU 数字偏悲观 3×、并得出"每 token 成本随 batch 恶化"的假象。
  修正后:真实路由下 1.7–2.0 TFLOP/s(峰值 15–18%)、~0.15 ms/token/层、随 batch 改善;
  BF16 预反量化在真实路由下**慢 14×**(不是快 1.2×)——那个对照本身也不成立
  (两条路径的 M 结构不同)。详见新增的 §7.2b。依据:`XIAOTU_MOE_PROFILE=1` 相位计时 +
  修正后的 `scripts/bench_cpu_engine*.py`。
- **2026-09-09(第 11 版)** — D6 改为"我生成全部 PR":三个 draft PR 已开
  (#56118/#56119/#56120),三个分支全部 rebase 到 `main@1454b71`;PR3 因上游重构
  `common/ops/*` 做了手工移植(细节见 §4.2),运行时验证待补。
- **2026-09-09(第 10 版)** — 新增 §4.3 上游漂移与维护成本(实测:上游 41 提交/天、
  36h 走 85 个提交;PR2 干净、PR1 1 处冲突、PR3 6 处冲突;插件 15 个导入模块全在、
  近 1 天无签名变更),并给出维护建议。新增 `scripts/check_upstream_drift.sh`。
- **2026-09-09(第 9 版)** — §4.1 换成 D1–D8 的**决策与执行状态表**;新增 §4.2
  上游分支现状(PR1/PR2/PR3 已推送到 `yeungtuzi/vllm`);新增 `NOTICE`、
  `THIRD_PARTY_NOTICES.md`;`pyproject` 作者改为"大河马 (BigHippo)"。
  依据:用户 2026-09-09 对 D1–D8 的回复。
- **2026-09-09(第 8 版)** — 优化项 ⑤ 落地一项:`_down_kernel` 输出累加改
  `sem="relaxed"`(T=16K 198.7→191.4 ms,32K 1994→2066 tok/s),并记录 5 项被证伪的
  变体与下一档(per-slot 缓冲 + 归约,可再 +6.3%)。
- **2026-09-09(第 7 版)** — ①新增 §5.10 内核级分解(进程内 torch profiler),
  **更正**第 1–6 版"注意力/indexer 是第一瓶颈"的说法(实测仅占 2.9%,真正的开销是
  逐元素小内核 + dense GEMM + H2D 缺口);②§5.5 补 HTTP 服务端复测与 prefix-cache
  测量陷阱;③§5.7/§7.1 按新分解重排优先级。依据:`report/prof16k_kernels.txt`、
  `report/server_conc2.jsonl`。
- **2026-09-09(第 6 版)** — 补 §5.9 GPU prefill 阈值推荐表(逐长度实测 + 交叉点推导 +
  推荐值 384/512–768,注明仅限本硬件)与 §4.1 upstream 需要用户拍板的 8 个决策点。
  依据:`report/curve_thr.jsonl`、`report/curve_small.jsonl`。
- **2026-09-09(第 5 版)** — 项目对外名称改为 **vllm-xtu-moe**(XTU = X Transformers Unity,
  读音「小兔」);同步改了 `pyproject.toml` 的 `name`、插件注册横幅、README 标题、
  文档标题。**内部标识符刻意不改**(`vllm_xiaotu_moe` / `xiaotu_moe` 包名、`XIAOTU_*`
  环境变量、conda env 名、目录名),以免破坏既有脚本与配置。依据:用户 2026-09-09 指令。
- **2026-09-09(第 4 版)** — ①§1.1 补内存规格 **DDR5-4800**(24×64 GiB,理论 921.6 GB/s);
  ②§1.4(a) 重写:给出 2×2×2 归因矩阵与"编译器参数无影响"的 A/B 结论,并**更正**
  第 3 版"CPU MoE 是带宽敏感型"的说法(见 §7.2b 的最终更正)。依据:`report/stream_nt.c`、`report/rd1.c`、`/tmp/s_*`、
  `scripts/bench_cpu_engine.py`、`/tmp/engine_*.so`。
- **2026-09-09(第 3 版)** — 补齐全部实测(§5.3–5.7、§0);①§5.3 单卡 TTFT 曲线(CPU 112.9/439.4 s vs
  GPU 5.64/5.76/8.64/18.24 s);②§5.4 双卡 TP=2 EP 实测(16K 13.96 s = 1149 tok/s,
  2K 并发 1.93×);③§5.5 并发与 decode;④§5.7 差距表(TP=2 下非 MoE 占 69%);
  ⑤§0 摘要换成实测数字;⑥§1.4 STREAM 重测(1 socket 274.5 GB/s、全机 422.3 GB/s,
  旧值 138/316 是高负载下测的);⑦§5.6d 交叉点从 2.5K 更正为 **100–300 token**
  (旧值基于一个把 GPU 路径误标成 CPU 的测量);⑧§2.4 新增两个致命 bug
  (`_pinned_kmajor` 缓存键、客户端 os.environ 不生效)。依据:`report/curve_tp1*.jsonl`、
  `report/curve_tp2.jsonl`、`report/moe_micro*.json`、`report/hw_bandwidth.json`。
- **2026-09-09(第 2 版)** — ①§5.2 补正确性表(RMS 相对误差,全维度 5.6e-3);
  ②§5.6 补三项消融(重叠前后、TP=2 等效 EP、tile 扫描)与 CPU/GPU 交叉点推导;
  ③§2.4 补 TP=2 启动失败的两个根因(`VLLM_ENGINE_READY_TIMEOUT_S=600` 与
  `XIAOTU_MOE_SINGLECOPY=1` 缺失导致的 node-0 OOM);④§7.4 新增"一次性启动成本"
  实测(pin_memory 1.6 GB/s、cudaHostRegister 快 3.7×);⑤§1.4 标注 STREAM 测量时刻
  的负载;⑥更正第 1 版里"7.8×/14× 提速"的表述 —— 那是与修复前、高负载下的 CPU 基线
  对比,不公平;本版一律用同进程实测曲线。依据:`results.txt` 2026-09-09「长 prefill
  第 3 轮」、`report/moe_micro*.json`、`report/golden.txt`、`journalctl -k` OOM 记录。
- **2026-09-09(第 1 版)** — 首版:§1 硬件(工具核验)、§2 开发摘要、§3 主线改动清单、
  §4 upstream 策略、§5 实测框架、§7 优化空间。
