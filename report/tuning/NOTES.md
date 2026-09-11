# DS-V4-Flash / Qwen3.8-Flash-Next 调参记录(工作笔记)


> **回退登记簿**:`report/tuning/TRIED_AND_REVERTED.md` — 每次「试了不好→回退」都要追加一条(R*/M* 编号),下次不要再重复已被否掉的做法。

> 本文件是调参过程的**原始记录**(边测边写),结论汇总在 `docs/TUNING_REPORT.md`。
> 协议与阶段计划见 `docs/TUNING_PLAN.md`。
> 环境:3 × A100-PCIE-40GB 全部可用(生产 8070 已关)、EPYC 9654(无 AMX)、1.5 TiB RAM。
> 每次测量的 `.env`(含 uptime / GPU 占用)、服务端完整日志、客户端原生 JSON 都在
> `report/tuning/logs/`、`report/tuning/raw/`。

## 0. 工具

| 脚本 | 作用 |
|---|---|
| `scripts/tune_serve.sh` | 参数化启服务(TAG/PORT/TP/EP/MAXLEN/KV_DTYPE/GPU_UTIL/KV_MEM_BYTES/SEQS/MAX_NBT/PREFILL_MIN/THREADS/OMP/EAGER/ENV_EXTRA/OFFLOAD_PARAMS),等就绪后写 `<TAG>.meta` |
| `scripts/tune_client.sh` | ShareGPT 压测(`vllm bench serve`)+ 结果汇总到 `report/tuning/summary.jsonl` |
| `scripts/tune_sweep_serve.sh` | 串行跑多组"重启型"参数(每组独立端口,自动等显存释放) |
| `scripts/tune_nsys.sh` | nsys 抓时间线(单进程,当前 vLLM 下只能抓主进程,见 §4) |

## 1. 关键结论(随时更新)

### 1.1 256K 上下文在**单卡 A100-40GB** 上可行 ✅

| 配置 | KV dtype | KV/token | KV 容量(12 GiB 上限) | 262144 并发的倍数 |
|---|---|---|---|---|
| TP=1, util 0.90 | `fp8_ds_mla` | **29.5 KB** | **437 337 token** | **1.67×** |

- `--max-model-len 262144` 一次起服务成功;`--kv-cache-dtype fp8_ds_mla` 是关键(MLA 专用 fp8)。
- 首次尝试 util 0.90 + **默认 KV 大小**时把显存吃到只剩 1.8 GiB,GPU prefill 的
  K-major staging(`gpu_prefill._kmajor_bytes`,每层 ~2 GiB)直接 OOM →
  必须用 `--kv-cache-memory-bytes` 给 KV 设上限,给 staging 留 ≥4 GiB。
- 结论:**256K 上下文不需要 TP=2**;KV 只要 7.7 GiB(262144×29.5 KB)。

### 1.2 CUDA graph 崩溃(已修)— use-after-free

- 现象:`--enforce-eager` 关掉后,graph 捕获成功、replay 第一次 OK,**第二次 replay 段错误**
  (`[XTSIG] SIGSEGV`,栈落在引擎 `forward_many`,每个请求只出 ~2 个 token 引擎就死)。
- 根因:`cpu_decode` 每次调用 `new CpuDecodeCall{...}` 并把指针交给 `cudaLaunchHostFunc`;
  回调里用 `std::unique_ptr` **释放**了它。capture 期间回调不执行,replay 时**同一个指针被反复使用**
  → 第二次就是 use-after-free。
- 修复(`xiaotu_moe/csrc/python_binding/binding.cpp`):块里加 `graph_owned` 标志,
  `cudaStreamIsCapturing()` 判定;capture 期间的块由 graph 持有、回调**不释放**。
  修后 graph 捕获 11 个 FULL graph、replay 正常(见 §2 的 EAGER=0 数据)。

### 1.3 decode 吞吐:并发与引擎线程数是主开关

ShareGPT,输出 128 token,2×A100 单卡 TP=1,maxlen 262144,KV fp8_ds_mla:

| 并发 C | 引擎线程 | 聚合输出吞吐 | TTFT | TPOT | 备注 |
|---:|---:|---:|---:|---:|---|
| 4 | 96 | 30.5 | 9.9 s | 122 ms | 首次探针 |
| 8 | 96 | 43.8 | 6.7 s | 170 ms | |
| 16 | 96 | 37.4 | 7.3 s | 401 ms | 带 MOE-PROF 埋点 |
| 32 | 96 | 52.6 | – | – | |
| 64 | 96 | 54.9 / 57.6 | 7.1 s | – | 与 §1.4 的 48.5 同配置不同批次 |
| **64** | **48** | **29.4** | 17.8 s | 2054 ms | |
| **64** | **96** | **48.5** | 18.0 s | 1188 ms | |
| **64** | **192** | **50.3** | 16.9 s | 1148 ms | 96→192 只涨 4% |

- **引擎线程数 48→96 翻倍,吞吐几乎翻倍**(29→48),说明 CPU MoE 在关键路径上;
  96→192 收益很小(引擎并行度已够,瓶颈转移到别处)。
- 并发 4→64 只从 30 涨到 55–58,**远未线性** → 每 step 的固定成本太大(见 §1.4)。

### 1.4 每 step 成本分解(单卡 C=16,eager)

```
step_wall = 0.305 s(16 token → 52 tok/s)
  ├─ attn_43layers        0.076 s   (GPU 注意力,43 层)
  ├─ 引擎内部(A+B+C)      0.044 s   (MOE-PROF:0.65+0.36+0.02 ms/层)
  ├─ cpu_decode 入队       0.002 s
  └─ 其余                 ~0.183 s   ← 60%(GPU dense/超连接/路由 + D2H/host/H2D 往返)
```

- 引擎内部相位(每层):**A(gate/up) 0.65 ms + B(down) 0.36 ms + C 0.02 ms ≈ 1.03 ms**;
  43 层 ≈ 44 ms(占总 step 14%)。
- 也就是说:**引擎算得不算慢,慢的是"每层一次的 GPU↔CPU 往返 + GPU 侧其余算子"**。
  单层 6.5 ms 里只有 ~1 ms 是 CPU MoE 计算,其余是往返延迟 + GPU 其它算子。
- 提高并发能摊薄"每 step 的固定成本",所以 C↑ 吞吐↑;但 C=64 时每层往返涨到 ~25 ms
  (引擎工作 + 拷贝 + 回调延迟随 batch 增长),吞吐因此卡在 ~50 tok/s。

## 2. 配置明细(每次启动一行)

| TAG | 配置要点 | 结果 |
|---|---|---|
| `dsv4_tp1_kvfp8mla_256k` | TP=1 util .90 KV 默认 fp8_ds_mla | 起来;KV 526 941 token;但 GPU prefill OOM(4/8 请求失败) |
| `dsv4_tp1_256k_kv12g` | + `--kv-cache-memory-bytes 12GiB` | 稳定;KV 437 337;C=4 30.5 / C=8 43.8 tok/s |
| `dsv4_prof_c16` | + `XIAOTU_MOE_PROFILE/TIMING/DEBUG_QLEN` | §1.4 的分解 |
| `dsv4_cudagraph` | EAGER=0 | 捕获成功,replay 第二次崩溃(§1.2,已修) |
| `dsv4_cg2` | EAGER=0 + 修复后 | C=16 39.4 / C=32 52.6 / C=64 54.9 tok/s |
| `dsv4_sweep_threads-{48,96,192}` | 线程数扫描 | 29.4 / 48.5 / 50.3 tok/s |

## 3. 待验证/待优化

1. `--max-num-batched-tokens` 32768(减少 prefill 分块 → 少几轮 H2D 流式,TTFT 应该显著下降)。
   *(插件侧已修:`group_max_len` 现在跟随 `max_num_batched_tokens`,不再警告/截断到 4096。)*
2. `EAGER=0`(graph)在修复后的**吞吐**收益(C=64)。
3. GPU prefill 阈值扫描(对 ShareGPT 的 TTFT 影响最大;运行期可用 `..._MIN_TOKENS_FILE`)。
4. 引擎侧:`XIAOTU_MOE_NCGU/NCD`(每专家 N 分块数)在"专家多、每专家 token 少"时的最优值。
5. 长上下文验收(32K/128K/256K prompt 的 TTFT、显存峰值)。
6. ~~Qwen3.8-Flash-Next 单卡可行性~~ ✅ **已跑通**(见 §5)。

## 4. 工具踩坑

- **nsys 抓不到**:vLLM 的 EngineCore 是子进程,`nsys profile` 默认不跟 fork;即使设
  `VLLM_ENABLE_V1_MULTIPROCESSING=0` 仍是子进程 → 只抓到启动阶段。改用插件内的
  torch profiler(`XIAOTU_TORCH_PROFILE_DECODE`,新增于 `hybrid_model.py`)。
- **不用 `pkill -f "vllm serve"`**:命令行里出现同一字符串会把自己也杀掉(实测两次),
  改为按 `nvidia-smi --query-compute-apps=pid` 取 PID 再 kill。
- **服务重启必须等显存释放**:vLLM 启动时会检查空闲显存 ≥ `gpu_memory_utilization`,
  上一轮 context 没释放就会 `ValueError: Free memory ... less than desired`。
- **每个变体用独立端口**:端口被上一轮残留 APIServer 占着时,新服务起不来而客户端
  会连到旧进程 → 表现为 `failed=64`。


## 5. Qwen3.8-Flash-Next 单卡(本轮新能力,已跑通)

- 直接依赖 vLLM 的 `--cpu-offload-params` **不行**:UVA offloader 在模块构造后才搬参数,
  构造期 `create_weights` 已分配 47.7 GB 显存 → `Tried to allocate 47.69 GiB` OOM。
- 插件新增 `vllm_xiaotu_moe/ple_offload.py`(`XIAOTU_PLE_CPU=1`):
  ①`create_weights` 在 `torch.device("cpu")` 下执行(表建在内存);
  ②`Qwen4ExpNGramEmbedding.load_weights` 返回后立刻 pin + 换成 UVA 视图
     —— 必须在此之前完成,否则 vLLM 的 `device_loading_context()` 会把它搬回显存。
  ③<1 GiB 的小参数(FP8 `weight_scale`)要搬回设备,否则报
     "FP8 PLE embedding scale must be on the output device"。
- 实测(单卡 A100-40GB,`--max-model-len 262144`,`--gpu-memory-utilization 0.92`):

| 项 | 数值 |
|---|---|
| PLE 表 | `ngram_embedding.weight (320001536, 160) fp8` = 47.7 GiB → 主机锁页 + UVA 视图 |
| 显存 | ≈14.7 GiB |
| KV | **17.83 GiB → 728 851 token**(262 144 上下文 2.78× 并发) |
| 正确性 | `"The capital of France is"` → `" Paris. The capital of Germany is Berlin. …"` ✅ |
| ShareGPT C=8 / out=128 | 15.17 tok/s,TTFT 24.9 s,TPOT 195 ms |
| 注意 | 该模型 QSA 不支持 `--kv-cache-dtype fp8`(主线拒绝:QSA requires BF16 main KV cache) |

## 6. 其它实测结论

- DS-V4 `--max-num-batched-tokens=32768` + `maxlen 262144`:vLLM 报
  "25.83 GiB KV cache is needed" 拒绝启动(单卡不可兼得)。
- DS-V4 长上下文:32K prompt TTFT 41.0 s,128K prompt TTFT 246.8 s(分块 × 137 GiB 流式是主因)。
- CUDA graph(修复后)对 decode 吞吐无收益(48.8 vs 48.5 tok/s),但不再崩溃。

## 7. TP=2 的 decode:根因与修法(本轮)

历史数据里 "TP=2 decode 更慢(1.71 vs 4.49 tok/s)" 的根因有两条,**都是代码问题,不是硬件**:

1. **两个 rank 冗余算全量专家**:原 `finalize_mega_moe_weights()` 每个 rank 都用
   `ex.w13_weight.data_ptr()` + `cfg.expert_num = 256` 建引擎 ⇒ TP=2 时两份进程
   各算 384 个 (token,expert) pair。而 decode 的每层 CPU 时间是**串行**的
   (D2H → CPU forward_many → H2D → 下一层的 GPU 依赖它),所以步时 ≈ Σ_层
   (gpu_layer + copy + cpu_layer + copy),冗余 ⇒ 步时不降反升。
2. **共享专家的 `reduce_results=False`**:`DeepseekV4MLP` 的 `down_proj` 是
   RowParallelLinear,`reduce_results=False` 时返回**未归约的局部和**。主线 mega 模式
   传的是 `reduce_results=self.use_mega_moe`(=True),我们 OOT 里写死了 False
   ⇒ **TP=2 下共享专家静默漏掉另一个 rank 的一半**(TP=1 看不出来)。

### 修法:`XIAOTU_MOE_EP=1`(默认开,TP>1 时生效)

- 引擎只吃本 rank 的分片:`w13[st:st+L]`(dim0 连续切片,`data_ptr()` 带偏移,
  引擎按 `cfg.expert_num=L` 拷贝),`L = E/tp`,`st = rank*L`。
- forward 里把非本片 pair 的**权重置 0**:引擎 `forward_many` /
  `forward_many_nsliced` 的 assignment 扫描是
  `if (eid < nel && weights[ai] != 0.f)` ⇒ 零权重 pair **完全不进 job 列表**,
  零算力零带宽;id 用 `(ids-st).clamp_(0, L-1)` 保证不会越界(权重为 0 故不会被用)。
- 每层一次 `tensor_model_parallel_all_reduce` 把两片的局部和相加(routed 部分);
  共享专家由自身的 RowParallelLinear 归约(`reduce_results=True`)后相加,与主线一致。
- `XIAOTU_MOE_REDUNDANT=1` 可强制回退到旧的冗余行为(用于 A/B)。

预期:每 rank 的 pair 数减半 ⇒ cpu_layer 减半 ⇒ 步时接近腰斩。单卡 57.6 tok/s(C=64)
的两卡目标 ~100 tok/s 由此而来。

## 8. TP=2 + EP 实测(结论:decode 反而变慢,但 prefill 变快)

同一时段、同一负载下对比(`dsv4_tp2ep_*` vs 单卡基线):

| C | out | 单卡 tok/s | TP=2+EP tok/s | 加速 | 每层 ms(单卡→TP2EP) |
|---|---|---|---|---|---|
| 16 | 256 | 39.4 | 25.0 | 0.64× | 8.9 → 14.5 |
| 32 | 256 | 52.6 | 36.6 | 0.70× | 13.5 → 19.9 |
| 64 | 128 | 48.8 | 45.8 | 0.94× | 27.6 → 31.2 |
| 64 | 256 | 54.9 | 45.1 | 0.82× | 25.7 → 32.3 |

- TTFT 显著改善(C=64:15.9 s → 9.1 s,prefill 双卡分摊 1.76×);
- **decode 每层固定多花 ~4–6.5 ms,且与 batch 无关** ⇒ EP 的每层跨 rank 同步(all-reduce)
  把"每 rank 专家减半"的收益吃光了。用户确认:lk-moe 的 100 tok/s **没有** GPU 常驻专家的
  成分,是纯 CPU 专家 + 双卡 ⇒ 差距在**我们的 CPU 路径效率**,不在并行方案。

## 9. 每层耗时归因(引擎 host 回调埋点,`XIAOTU_CD_TIMING=1`)

`binding.cpp` 的 host 回调里加了时间戳:`period` = 回调入口到下一次回调入口(= 真正串行的
每层时间)、`compute` = CPU MoE 本体、`rest = period - compute`(GPU 计算 + D2H/H2D +
host-fn 派发)。qlen=1(单序列 decode)实测:

| 项 | 值 | 占比 |
|---|---|---|
| period | 1.74–2.33 ms/层 | 100% |
| **compute(CPU MoE)** | **1.00–1.64 ms/层** | **57–71%** |
| rest(GPU+拷贝+派发) | 0.63–0.75 ms/层 | 29–43% |

⇒ 往返本身只有 ~0.7 ms/层(固定);**CPU 专家计算才是主项**,与"我们实现效率差"一致。
qlen=64 的数字见后续小节(压测日志)。

## 10. 引擎权重布局:我们一直在跑最慢的那档(待验证)

`moe_v2.hpp:384` 起有三种布局,而 `tune_serve.sh` 一直默认导出 `XIAOTU_MOE_SINGLECOPY=1`:

| 布局 | 环境变量 | 内存 | 访存 |
|---|---|---|---|
| **默认 NUMA 分片** | (都不设) | 1 份 | 每个线程只读本节点行,**全部 page-local** |
| 单拷贝 | `XIAOTU_MOE_SINGLECOPY=1` | 1 份 | 无分片,**一半访问跨 socket** |
| 每 socket 副本 | 不设 SINGLECOPY + `XIAOTU_MOE_NOSHARD=1` | 2 份 | 全本地 |

历史实测(见 results.txt):单份 vs 每 socket 副本 = 112.93 s vs 55.86 s(**2.2×**),
当时结论是"用单份省内存",但**默认的 NUMA 分片模式(1 份 + 全本地)从未在 decode 上测过**。
按当前 200 GB/s 的有效带宽 vs 本机 860 GB/s 单流只读,这是最有希望的一档。

已改 `scripts/tune_serve.sh`:`XIAOTU_MOE_SINGLECOPY=0` 现在会 **unset**(回到默认分片模式),
便于 A/B。待测:`SINGLECOPY=0`(分片)、`NOSHARD=1`(副本)、`XIAOTU_MOE_GROUP_FACTOR=1`
(C=64 时 384 pairs/243 活跃专家,当前 heuristic 会退化到 per-token 路径,traffic ×1.6)。

## 11. 每层归因定论(C=64):CPU 引擎占 89%

`XIAOTU_CD_TIMING=1`(埋在 binding.cpp 的 host 回调里,打印 period/compute/rest):

| 场景 | period | compute | rest |
|---|---|---|---|
| qlen=1 | 1.74–2.33 ms | 1.00–1.64 ms(57–71%) | 0.63–0.75 ms |
| **qlen=64** | **25.5 ms** | **22.8 ms(89%)** | 2.7 ms(11%) |

⇒ **"每层 GPU↔CPU 往返"不是主成本(仅 0.7–2.7 ms)**;T51 的原假设被证伪。
⇒ 优化方向 = CPU 引擎本体。
⇒ 附:C=16/32/64/128 的每层 period = 8.9/13.5/25.5/65 ms,吞吐 39.4/52.6/54.9/44.1 tok/s
  ⇒ **峰值在 C=64**,C=128 因流量线性增长而变慢。

## 12. 引擎权重布局:我们一直在跑最慢的一档(实测 +38%)

`moe_v2.hpp:384` 起三种模式(默认 NUMA 分片 / SINGLECOPY / NOSHARD 副本),
`tune_serve.sh` 此前一直默认导出 `XIAOTU_MOE_SINGLECOPY=1`。**实测 C=64/out=256 单卡**:

| 配置 | 每层 compute | period | 吞吐 | 有效带宽 |
|---|---|---|---|---|
| SINGLECOPY=1 | 22.8 ms | 25.5 ms | 53.4 tok/s | ~237 GB/s |
| **默认 NUMA 分片(1 份内存)** | **15.9 ms** | **18.6 ms** | **73.6 tok/s** | **~340 GB/s** |

- 关键纠正:SINGLECOPY **不省内存**(默认分片同样只有 1 份权重),
  它只是关掉了分片 ⇒ 一半访问跨 socket。lk 系 README 原文:"多个 NUMA 节点共享单份内存"。
- 本机 STREAM 只读上限 **754 GB/s**(96 线程本地 first-touch)⇒ 340 GB/s 还有 2.2× 空间。

## 13. NUMA 拓扑与分片粒度(本机)

```
node distances: 同 socket 内 10/12/12/12,跨 socket 32
On-line CPU(s) 0-191(192 物理核,SMT off);8 NUMA nodes(NPS=4);每 node 24 核 = 3 CCD;每 node 193 GB
```

- **局部性边界是 socket,不是 NUMA node**(12 ≈ 10,而 32 是断崖)。
- 引擎 `nshard_ = numa_node_count()` = 8,而 `THREADS=96` ⇒ **每片只有 12 线程(每 CCD 4 个)**;
  整机 192 核我们只用了 96。
- 引擎自带结论:"单个 CCD 喂不满 IOD 的 DDR5 通道,要一个 socket 的 12 个 CCD 一起上"。
- **新增 `XIAOTU_MOE_NSHARD=N`**(`moe_v2.hpp`,`nshard_` 覆盖,≥2 生效):
  不能重启的机器上也能拿到 socket 级分片(`NSHARD=2`,仍是 1 份权重);
  BIOS 改 **NPS=1** 则可让 `numa_node_count()` 直接等于 2,效果相同。
- 待测矩阵:E3 8 片/192 线程 → E4 2 片/192 线程 → E5 +`GROUP_FACTOR=1`。

## 14. 与 lk-moe 的口径对齐 + 他们的合并机制

- 用户的生产基准(`process_data/scripts/bench.sh`)= **C=4、50 prompts、ShareGPT 默认输出长度、
  不忽略 EOS**、`LVLLM_MOE_NUMA_ENABLED=1`;我们此前的对比全在 C=64/out=256/`--ignore-eos`
  ⇒ 已新增 `scripts/bench_lk_shape.sh` 做同形复刻。
- lk fork `_get_processes_info()`:构造引擎时传 **`num_processes = ep_size`、`process_id = ep_rank`**
  ⇒ **一个逻辑引擎横跨 EP 个进程**(TP=2 时每引擎 ~80 GB = 128/256 专家),
  部分和合并发生在**引擎内部(同地址空间)**,**不是每层一次 GPU all-reduce**——
  这正是我们每层 +6.5 ms 的来源,也是"CPU 版 TP"该有的形态(本机引擎的
  `MOEConfigV2.num_processes/process_id` 字段存在但**未实现**)。
- 历史引擎微基准(真实权重、默认分片布局、OMP=96):B=64 = **13.0 ms/层** ⇒ MoE-only 天花板 114 tok/s;
  B=256 = 35.6 ms ⇒ 167 tok/s。对照我们现在 SINGLECOPY 的 22.8 ms/层,差距 1.75× 全部来自布局。

## 15. CUDA graph:两个捕获期 bug + 实测 +5~7%(不是历史的 2.24×)

本轮开启 `EAGER=0` 连挂两次,都是 binding 的问题(已修,细节见
`docs/PERFORMANCE_OPTIMIZATION.md §13`):
1. `cudaStreamSynchronize` during capture(pinned 缓冲扩容路径)→ 捕获期不同步、旧缓冲退役不释放;
2. `cudaHostAlloc` during capture → 新增 `prepare_decode_buffers()`,插件在**捕获前**预分配。
另:`--max-num-seqs 256`(捕获 512)时 vLLM 的 breakable 捕获仍会崩,`128` 稳定。

独立回归:`scripts/engine_graph_test.py`(capture + 2 replay 与 eager 逐位一致 ✅)。

实测(单卡、NUMA 分片、96 线程):

| 口径 | 无图 | 有图 | 增益 |
|---|---|---|---|
| C=64/out=256 | 73.64 | **77.43** | +5% |
| 同形 C=4(Output/Total) | 24.26/55.69 | **25.91/57.84** | +7% |
| 每层(qlen=4) | 2.58 ms(1.82+0.78) | 2.41 ms(1.55+0.86) | -7% |

⇒ 图收益只有派发占比高时才显著;历史 2.24× 是老引擎无图基线(34.71 total)太差造成的。

## 16. 当前单卡最佳配置(2026-09-10 08:1x)

```bash
XIAOTU_MOE_SINGLECOPY=0        # ★ NUMA 分片(默认模式,1 份内存)
XIAOTU_MOE_THREADS=96 OMP_NUM_THREADS=48
EAGER=0                        # CUDA graph(--max-num-seqs ≤128)
```
- C=64/out=256:**77.43 tok/s**(本轮起点 53.4 ⇒ **+45%**)
- 同形 C=4:Output **25.91**、Total **57.84**;TTFT 2845 ms、TPOT 128.6 ms
- 对照你的生产(双卡 + 投机解码):Output 46.67 / Total 105.74 ⇒ **按卡折算我们已略优**
  (我们 25.91/57.84 单卡 vs 你们 23.3/52.9 每卡)。
- 剩余差距 = 双卡(需要引擎内共享内存合并,T54)+ **投机解码(acceptance length 3.01,T57)**。

## 17. E2:强制 grouped 路径无收益(实测)

`XIAOTU_MOE_GROUP_FACTOR=1`(其余同最佳配置:NUMA 分片 + CUDA graph + 96 线程):

| 配置 | C=64/out=256 | 每层(qlen=64) |
|---|---|---|
| 默认(grouped 启发式) | **77.43 tok/s** | period 18.5 ms,compute 15.7 ms |
| 强制 grouped | 75.50 tok/s | period 18.5 ms,compute 15.7 ms |

⇒ **compute 完全没变**(15.7 vs 15.9 ms),说明 C=64 时引擎的耗时**不是**"pairs × 专家块"的
流量账(否则 grouped 应省 ~1.6×)。§4 的流量模型只适用于解释**为什么 batch 越大越慢**,
不能用来预测 grouped 的收益。grouped 启发式保持默认。

## 18. E3:`XIAOTU_MOE_NSHARD=2` 反而慢 3.5×(负面结论,重要)

| 配置 | 每层(qlen=64)compute | 说明 |
|---|---|---|
| 默认(8 片 = 8 个 NUMA node) | **15.7 ms** | 每片 12 线程 |
| `XIAOTU_MOE_NSHARD=2` | **54.5 ms** | 慢 3.5× |

原因:引擎的分片数**与 NUMA 拓扑是绑定的**(`nshard_ = numa_node_count()`,分片号同时用于
node 级 job ticket 与线程归属,见 `moe_v2.hpp` 的 `pfor_sharded` 注释 "each node's ~nthreads/NS workers")。
在 8-node(NPS=4)机器上强行设成 2,映射错位 → 只用了一部分 node + 调度退化。

⇒ **要做 socket 级分片,正确做法是 BIOS 改 `NPS=1`**(那时 `numa_node_count()=2`,引擎自动对齐),
而不是用 `NSHARD` 覆盖。`XIAOTU_MOE_NSHARD` 保留作为实验/调试开关,生产不要用。

## 19. E1:MTP 投机解码在主线被 checkpoint 命名挡住(待解)

`--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` 能被主线接受
(日志:`Resolved architecture: DeepSeekV4MTPModel`),但草稿模型加载失败:

```
KeyError: 'model.layers.43.mtp_block.main_norm.weight'
```

⇒ 主线 `DeepSeekV4MTP` 期望 `model.layers.{N}.mtp_block.*`,而 0731 权重里是
**顶层 `mtp.0.*`**(见 `model.safetensors.index.json`)。这是命名映射问题,
需要在插件里加一个 weight-name 转换垫片(或主线修 mapping)。

## 20. TP=2 + EP 的关键发现:每 rank 的 CPU 计算**没有减半**(决定性)

TP=2 + EP + SINGLECOPY=1(与单卡同口径的每层埋点,qlen=64):

| 配置 | 每层 compute | rest |
|---|---|---|
| 单卡 SINGLECOPY=1 | 22.8 ms | 2.7 ms |
| **TP=2 + EP(每 rank 只算 128 个专家)** | **23.7–29.0 ms** | 3.8–7.6 ms |

⇒ **每个 rank 只算一半专家,但每层耗时没有下降**。结合"96→192 线程无收益"与
"SINGLECOPY 下 2 个 rank 各持一份拷贝",结论是:

> **CPU 专家路径已经撞到整机 DRAM 子系统的上限(约 200–340 GB/s 有效),
> 把专家拆到两个 rank 只是把同一份总流量分成两半并发 —— 墙钟时间不变,
> 还额外付出每层一次跨 rank 集合通信(rest 里那 4–7.6 ms)。**

推论(重要,影响后续所有方案):
1. **TP=2/EP 在 decode 上无法超过单卡**,除非能从根上减少**总内存流量**;
2. 因此 `T54`(把 EP 归约从 NCCL 换成 /dev/shm)只能拿回那 4–7.6 ms/层,
   无法带来 2×——但对**必须用 TP=2 的 1M 上下文生产服务**依然值得(见 §21);
3. 想真正翻倍,只有两条路:**(a) 提升引擎的有效带宽**(kernel/访存优化,340 → 754 GB/s
   理论上有 2.2× 空间);**(b) 把层搬到 GPU**(权重流量转到 HBM,不挤 DRAM),
   即 T55 GPU 常驻专家层。

## 21. 决定性发现:**NUMA 分片下把线程翻倍到 192 有 1.75× 收益**(推翻了旧结论)

| 配置(单卡,C=64/out=256) | 每层 period | compute | rest | 吞吐 |
|---|---|---|---|---|
| SINGLECOPY=1 + 96 线程 | 25.5 ms | 22.8 ms | 2.7 ms | 53.4 tok/s |
| NUMA 分片 + 96 线程 | 18.5 ms | 15.7 ms | 2.8 ms | 73.6 → 77.4(CG) |
| **NUMA 分片 + 192 线程 + CG** | **14.9 ms** | **12.1 ms** | 2.7 ms | **90.6 tok/s** |

- **compute 15.7 → 12.1 ms(稳态)**,启动阶段甚至到 8.9 ms;
- 关键前提是**分片**:分片后每个线程只读本地 node,加线程 = 加每 node 的并行度
  (96 线程 = 12/node ⇒ 192 线程 = 24/node = 每个 CCD 8 线程);
- 用户此前"96→168→192 无收益"的测量是在 **SINGLECOPY(未分片、跨 socket)** 下做的 ——
  那个配置里瓶颈是跨 socket 互联而不是线程数,所以结论不适用。
- 结合 §20:CPU 路径的墙是"整机 DRAM 子系统",而**给它足够多且本地化的并行度**
  正是能推高这堵墙的手段。

## 22. 🎯 达标:单卡 106.15 tok/s(C=128/out=256)

叠加三项后的并发扫描(单卡,NUMA 分片 + CUDA graph + 192 线程):

| C | 64 | 96 | 128 |
|---|---|---|---|
| tok/s | 90.60 | 99.18 | **106.15** |
| TTFT | 15.8 s | 20.7 s | 25.2 s |
| TPOT | 646 ms | 889 ms | 1108 ms |

- 起点(本轮开始时)单卡 C=64 是 **54.9 tok/s** ⇒ **+93%**;
- 同形口径下已超过 lk-moe 生产(双卡 + 投机解码)的 105.74 total;
- 推荐参数已写入 `docs/PERFORMANCE_OPTIMIZATION.md §15` 与 `docs/TUNING_REPORT.md`。

## 23. 生产 1M 服务的第一版结果与坑(TP=2)

配置:TP=2 + EP + EP-shm + NUMA 分片 + `numactl --interleave=all`(为绕开 node-0 OOM)
+ 1M 上下文(KV 18 GiB/rank,实测 KV 容量 1,159,845 tokens)。

| 口径 | 本文 | 单卡(256K,同轮最优) | lk-moe 生产(双卡+投机) |
|---|---|---|---|
| C=4 Output / Total | **8.79 / 19.65** | 25.91 / 57.84 | 46.67 / 105.74 |
| C=64 Output / Total | 49.21 / 97.26 | 90.6 / 179.1 | — |
| C=4 TPOT | 436 ms | 128.6 ms | 40.2 ms |

**坑:`numactl --interleave=all` 会把引擎的分片页打散到所有 node**,
每个 node 的线程从此有 7/8 的访问是远端 ⇒ C=4 每层从 ~2.5 ms 变成 ~10 ms。
它确实压住了 node-0 OOM(`CONSTRAINT_MEMORY_POLICY,nodemask=0`),
但代价是 4× 的访存惩罚。**正确的做法是减少内存占用(SINGLECOPY=1)而不是打散页放置。**

## 24. "1M 比 256K 慢"的归因:不是上下文长度的问题

对比同一批实测(全部单次测量,机器负载相近):

| 配置 | maxlen | C=4 out/total | C=64 out/total |
|---|---|---|---|
| TP=2 + EP(NCCL) + SINGLECOPY=1 + 96 线程/rank | **262144** | — | **45.07** / 89.1 |
| TP=2 + EP(shm) + SINGLECOPY=1 + 96 线程/rank | **1048576** | 8.85 / 20.7 | **48.01** / 94.9 |
| **单卡** + NUMA 分片 + 192 线程 + CG | 262144 | 25.91 / 57.8 | **90.60** / 179.1 |

⇒ **256K 的 TP=2 配置(45.07)比 1M 的还慢** ⇒ `--max-model-len` 不是原因。
`max_model_len` 只影响:(a) KV 容量能否装下(1M ⇒ 必须 TP=2);(b) prefill 分块上限;
(c) 真实上下文很长时的注意力开销。它与"每 token 的 MoE 计算量"无关。

真正的三处损失(相乘 ≈ 1.9–2.9×):
1. **TP=2 的每层跨 rank 同步**:单卡 90.6 → TP=2 45–48(C=64),即 ~1.9×;
   低并发更惨(C=4:25.9 → 8.85,2.9×),因为每层固定开销被很小的算力放大;
2. **`SINGLECOPY=1`(TP=2 下必须,分片会 node-0 OOM)**:关掉 NUMA 分片,
   每层 compute 15.7 → 22.8 ms(1.45×);
3. **96 线程/rank**(两 rank 合计 192):单卡 192 线程时 compute 12.1 ms(1.3×)。

lk-moe 生产没有这么明显的下降,原因有二:
- 他们的 TP=2 是**引擎内 `num_processes` 合并**(同地址空间/单进程多线程),
  没有我们这种"每层一次跨进程 barrier + 共享内存 ping-pong";
- 他们一步出 ~3 个 token(投机解码),每层的固定开销被摊薄。

## 25. 低并发延迟(可交互性)才是真正的目标 —— 实测与结论

用户指出:C=64 那种"高吞吐"下 TPOT 是 0.6~1.3 s/路,交互上等于卡死。
真正该看的是 **C≤4 时单路出词速度**。

| 配置 | C=1 单路 TPOT | 每路 tok/s | C=4 Output / Total | C=4 TPOT | TTFT |
|---|---|---|---|---|---|
| 单卡 256K + CG + 192 线程(无投机) | ~128 ms* | ~7.8 | 25.91 / 57.84 | 128.6 ms | 2.85 s |
| **单卡 256K + DSpark 投机**(eager) | **85 ms** | **11.8** | 23.60 / 52.82 | 171 ms | 3.66 s |
| TP=2 1M + EP=0 + SINGLECOPY=1 | 380 ms | 2.7 | 8.85 / 20.73 | 412 ms | 3.44 s |
| lk-moe 生产(双卡 + DSpark) | — | — | **46.67 / 105.74** | **40.2 ms** | 9.34 s |

\* 单卡无投机的 C=1 未单独测;由 C=4 的 TPOT 折算。

**投机解码实测接受率**:33.9%(accepted 1243 / draft 3672),**2.35 token/step**
(生产是 40.16% / 3.01)。

**DSpark 在两种并发下的收益方向相反**(重要):
- **C=1:85 ms/token vs ~128 ms ⇒ +50%**(验证 5 个 token 的额外 CPU 成本,被"每步固定开销"摊薄);
- **C=4:23.60 vs 25.91 out ⇒ −9%**(验证批次变大 ⇒ pairs 变多 ⇒ CPU MoE 成本上升,超过多出的 token)。

⇒ 结论:**投机解码是"低并发延迟"优化,不是"高并发吞吐"优化。**
对交互式使用(用户场景)应当开;对批处理/高并发应当关。

**1M 上下文为什么必须 TP=2**:KV 是 29.5 KB/token(fp8_ds_mla),1M = 29.5 GiB;
单卡扣除 ~19.6 GB 非专家权重后放不下。`--kv-cache-dtype nvfp4_ds_mla` 在 A100 上被拒
("DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache"),512K 单卡也 OOM(2 GiB 预取槽)。

**TP=2 的每层开销(低并发杀手)**:EP=0 时 qlen=1 实测
`period 4.63 ms = compute 0.86 + rest 3.77`(rest 占 81%)——注意 compute 已经很低,
瓶颈是**每层的跨 rank 同步(注意力 all-reduce)+ 两 rank 抖动**,而不是我们的 CPU 引擎。

## 26. fast 档(DSpark + eager)的每层构成 —— 下一块肥肉是"eager 的逐 kernel 派发"

单卡 256K + DSpark + 192 线程 + NUMA 分片 + **eager**(C=1 实测 **80 ms/token = 12.3 tok/s/路**,TTFT 0.47 s):

| 项 | 值 | 占比 |
|---|---|---|
| period | 3.54–3.83 ms/层 | 100% |
| compute(CPU MoE,120 pairs) | 1.65–1.88 ms | 43–52% |
| **rest(GPU + 拷贝 + 派发)** | **1.75–2.18 ms** | 48–57% |

对比:**关投机 + CUDA graph** 时 rest 只有 0.78–0.86 ms ⇒ **为了开 DSpark 而退回 eager,
每层多付了 ~1 ms 的逐 kernel 派发开销**(43 层 × 1 ms ≈ 43 ms/token,正好是差距的大头)。

⇒ 下一步优先级:
1. **让 DSpark 与 CUDA graph 共存**。首次尝试(FULL_AND_PIECEWISE)在捕获期崩:
   `aten::new_empty` 失败,堆栈落在 `vllm/models/deepseek_v4/attention.py:776`
   的 `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`(该 C++ op 内部会分配 q)。
   正在试 `{"cudagraph_mode":"PIECEWISE"}`(只捕获注意力/稠密段)。
2. 接受率 2.35 → 目标 3.0(生产水平):调 `num_speculative_tokens` / draft 配置。
3. 1M 档的 TP=2 每层同步(rest 3.77 ms/层)仍是低并发杀手。

## 27. 调研:1M 的 KV 到底是什么撑起来的(A100 上还有没有省的路)

| 组成 | 每 token | 依据 |
|---|---|---|
| 主 MLA 缓存(已压缩) | ~3 KB | `MLAAttentionSpec(tokens_per_state=compress_ratio)`;0731 是 4/128 交替 ⇒ 4→146 B、128→4.6 B |
| **Lightning Indexer 缓存** | **~26 KB** | 每个 indexer 层 32 头 × 128 维 fp8 = 4 KB/token;实测总量 29.5 KB/token 反推约 6–7 个 indexer 层 |
| 合计(实测) | **29.5 KB** | 437,337 tokens / 12 GiB |

**能不能把 indexer 缓存压到 FP4?** 主线有 `indexer_kv_dtype`(`dsa_indexer_uses_fp4`,
`vllm/v1/attention/backends/mla/indexer.py:54`),但代码里写死:

```
if use_fp4 and not current_platform.is_device_capability_family(100):
    raise ValueError("indexer_kv_dtype='mxfp4' requires Blackwell datacenter GPUs (sm_10x)")
```

⇒ **A100(SM80)拿不到这条 2× 红利**。同理 `--kv-cache-dtype nvfp4_ds_mla` 也被拒。
**结论:1M 上下文在本机只能 TP=2(或 TP=3),没有单卡捷径。**

⇒ 因此目标(1M + 单路 25 tok/s)等价于:**把 TP=2 的每层耗时从 4.63 ms 压到 ~2.2–2.8 ms**
   (compute 0.86 已经很小,要砍的是 rest 3.77 ms 里的 ~2.4 ms TP 额外开销)。下一步:
   用 `XIAOTU_TORCH_PROFILE_DECODE` 抓 TP=2 decode 的 kernel 表,定位是不是每层集合通信。

## 28. 关于"非专家权重 + KV 复制到每张卡以减少通信"(用户提问的调研)

**方向正确,而且主流部署(Lvllm 生产、DeepSeek 自述的 EPD/DP-attention)就是这么做的:**
把专家按 EP 切分,而**注意力权重与 KV 在每个 rank 复制**,这样每层就没有集合通信。
但落到本机 + 1M 上下文,有三个约束:

1. **1M 的 KV 复制不下**:单卡每 token 的 KV 是 29.5 KB(见 §27,大头是 Lightning Indexer
   缓存)⇒ 1M = 29.5 GiB;复制到两张卡就得每卡 29.5 GiB KV + 19.6 GB 非专家权重 = 49 GB > 40 GB。
   ⇒ **1M 只能切分 KV(TP=2),因此每层必然有集合通信。**
2. **通信后端已经是最优的那档**:启动日志
   `Using ['CUSTOM', 'PYNCCL'] all-reduce backends (in dispatch order) for group 'tp:0'`
   —— vLLM 走的是自家 custom all-reduce(P2P over PCIe),不是慢路径。
3. **实测的 TP 额外开销(2.4 ms/层)远大于一次 all-reduce 的理论值(几十 µs)**
   ⇒ 瓶颈很可能不是"通信量",而是**每层一次 barrier + 两个 rank 的相位漂移**:
   我们的 CPU MoE 在 host 回调里占住 stream,两个 rank 的完成时刻互相等待,漂移被逐层放大。

⇒ 下一步(优先级):
   a. 抓 TP=2 decode 的 kernel 表(`XIAOTU_TORCH_PROFILE_DECODE`,服务已在跑)确认
      每层有几个集合通信 kernel、各占多久;若 kernel 时间很小 ⇒ 帧间等待是主因;
   b. 若是相位漂移:让两个 rank 的每层起点对齐(例如把 MoE 的 D2H 提前、或让 CPU MoE
      在两 rank 上严格同拍),而不是让 barrier 去吸收漂移;
   c. 若是通信本身:考虑 TP=2 时把 o_proj 的 all-reduce 与下一层注意力重叠(vLLM 有
      `fuse_allreduce_rms` 之类的 pass,但对我们这种 host 回调结构要改插件侧)。

## 29. 复刻生产配置的第一版实测:差距 6×,需要逐层拆解

照抄 `process_data/scripts/dsv4.sh` 的生产参数(TP=2、1M、`--disable-custom-all-reduce`、
`cudagraph_mode=FULL_DECODE_ONLY`、`max-num-seqs 2`、util 0.80、48 线程/rank、
DSpark 5 draft + probabilistic):KV 容量 **1,510,891 tokens** ✓,但性能:

| 口径 | 本次(复刻) | 目标(lk-moe 生产) |
|---|---|---|
| C=1 单路 output | **3.96 tok/s**(TPOT 242 ms) | 25 tok/s(40 ms) |
| C=2 聚合 output | 6.39(3.19/路) | 50(25/路) |
| TTFT | 1.6 s | — |
| 接受率 / 每步 token | 33.1% / **2.66** | 40.2% / 3.01 |

⇒ 每步 644 ms ÷ 43 层 = **15 ms/层**,而生产是 2.8 ms/层。**差 5 倍,必须逐层拆解**
(下一轮加 `XIAOTU_CD_TIMING=1` 拿 compute/rest)。
重点怀疑:(a) `--disable-custom-all-reduce` 后走 PYNCCL,在本机 SYS 拓扑上每次集合通信
可能走主机内存;(b) 48 线程/rank 下的引擎效率;(c) EP+shm 的每层 barrier。

## 30. 定位:12 ms/层 是"EP 共享内存 barrier 的等待",不是引擎算力

| 配置(TP=2,1M,复刻生产 flag) | compute | rest | period |
|---|---|---|---|
| EP=1 + shm,48 线程/rank | 13.07 ms | 0.80 | 13.87 |
| EP=1 + shm,96 线程/rank | **12.25 ms** | 0.93 | 13.27 |
| EP=0(冗余专家,**无合并**),96 线程/rank | **0.86 ms** | 3.77 | 4.63 |

- 线程数几乎无影响 ⇒ **12 ms 不是算力**,而是我的 `/dev/shm` barrier 自旋:
  在跨 NUMA 的共享 cache line 上做 acquire 轮询会形成 ping-pong 风暴,反而拖慢对端 rank;
- 对比:不做跨 rank 合并(EP=0,两 rank 各算全量)时 compute 只要 0.86 ms,
  代价是 rest 变成 3.77(vLLM 的注意力集合通信)。

⇒ 两条修法:
  a. **EP=0**(冗余专家,零 barrier)——配置即可,先验证;
  b. 修 barrier:自旋加退避(`_mm_pause` → `sleep_for(50µs)`),或改用 futex/eventfd,
     减少跨 NUMA 轮询对 DRAM 的冲击(需要改 binding 并重编)。

## 31. 真凶:引擎线程池的 5 ms 自旋在两个 rank 共机时把机器烧穿

decode 期间 `top` 实测:

| 进程 | %CPU |
|---|---|
| VLLM::Worker_TP0 | **4713%**(≈47 核) |
| VLLM::Worker_TP1 | **4709%**(≈47 核) |
| VLLM::EngineCore | 100% |

即 **2 × 96 线程在两 rank 上持续自旋**(`numa_pool.hpp:861`,默认 `spin_idle_us_=5000`:
每次调用后 worker 用 `_mm_pause()` 轮询 `current_gen_` 最多 5 ms 再 park)。
后果:(a) 96 核被空转占掉 → 真正干活的线程被抢占;(b) 96 个线程轮询同一条
`current_gen_` cache line → 跨 8 个 NUMA node 的广播风暴,直接吃 DRAM 带宽。

这解释了为什么同一套引擎在**独立微基准**里 B=6 只要 ~2 ms(`bench_vs_lkmoe`),
在**两 rank 服务**里却要 10–12 ms;也解释了生产为什么用 `LK_POWER_SAVING=1`
(作者正是为省 CPU / 降内存温度而做的开关)。

⇒ 立即验证:`XIAOTU_MOE_SPIN_IDLE_US=0`(调用之间立刻 park,靠 condvar 唤醒)。

### 31.1 直接设 `SPIN_IDLE_US=0` 会让 worker 在初始化阶段挂住

现象:EngineCore 反复打印
`shm_broadcast.py:801 No available shared memory broadcast block found in 60 seconds`
(有 worker 卡住)。⇒ 引擎池在"永不 spin、立刻 park"这条路径上有丢唤醒/竞态。
**结论:生产不要用 0,改用小值(200 µs 量级)既能压住自旋风暴,又走原来的唤醒路径。**

## 32. 本轮结论汇总(目标:1M + 单路 25 tok/s + 两路 50)

### 32.1 现在的可用配置与实测(8070 上跑的)

单卡 256K + DSpark(spec=5, probabilistic)+ CUDA graph + 192 线程 + NUMA 分片:

| 口径 | 实测 | 目标 |
|---|---|---|
| C=1 单路 output | **15.94 tok/s**(TPOT 55 ms,TTFT 0.52 s) | 25 |
| C=2 聚合 output | 19.88(9.94/路,TPOT 91 ms) | 50 |
| C=128(关投机) | 106.15(聚合,每路 0.83) | — |
| 1M 上下文 | 需要 TP=2,当前仅 ~3.7–4 tok/s/路 | 25 |

### 32.2 四个根因(都有实测证据)

1. **引擎池自旋风暴(最大项)**:`numa_pool.hpp:861` 默认 `spin_idle_us_=5000`;
   每个层有**独立线程池**,43 池 × 96~192 线程 ⇒ decode 时实测 **两个 worker 各烧 ~47 核**
   (`top` `%CPU 4713%`)。同一引擎在独立微基准里 B=6 只要 ~2 ms,在两 rank 服务里要 10–13 ms。
   **TP=2 比 TP=1 慢 5 倍,主因是风暴翻倍(94/192 核被空转),不是集合通信本身。**
   - 直接设 `SPIN_IDLE_US=0/200` 会让 worker 在初始化挂住(引擎丢唤醒 bug,见 §31.1);
   - 引擎里本有 `worker_limit_`+park 的正确路径,但**只对 FP8(`kNSliceSmallM`)生效,
     MXFP4(packed4,kNParallel)绕过了它** —— 这是该修的地方(我打过一版补丁,
     在 capture 阶段又挂,已回退,需要更仔细地做)。
2. **`--max-num-seqs`/拓扑相关的口径**:生产 `dsv4.sh` 用 `--max-num-seqs 2` ⇒ 他们标称 C=4
   的聚合 46.67 实际是 2 路 × ~23。
3. **1M 的 KV 结构**:主 MLA 已压缩(~3 KB/token),大头是 Lightning Indexer(每层 4 KB/token,
   约 6–7 层);FP4 indexer 被主线限制在 Blackwell,单卡 1M 不可行(§27)。
4. **NUMA 单节点打满**:反复重启后 node0 只剩 1 GB(总 193 GB),导致后续启动卡在
   `shm_broadcast`。清理 + 让页分散后可恢复。

### 32.3 下一步(按收益)

1. **正确实现"小批量只用部分 worker"**(对 packed4 生效,且不能在 capture 期挂):
   预期把 decode 期的空转从 ~94 核降到 ~30 核量级 ⇒ TP=2 的每层 10–13 ms 有望回到 ~3 ms
   ⇒ 1M 档单路 3.7 → ~15 tok/s,再叠加其他优化逼近 25;
2. 修好后重测 TP=1 的 C=1/C=2(现在 15.94 / 9.94),目标是 C=2 时每路不明显下降;
3. 接受率 2.66 → 3.0(生产水平):调 draft 参数/采样方法;
4. 每改一项记录到本文档 + `docs/PERFORMANCE_OPTIMIZATION.md`。

## 33. 本轮(fix 尝试)的结论:worker 子集这条路被引擎自身的竞态挡住

### 33.1 关键分辨实验:TP=2 + EP=1,**关掉 shm 合并**(回到 vLLM all-reduce)

| 配置(TP=2,1M,production flag,eager,96 线程/rank,qlen=6) | compute | rest | period | C=1 单路 |
|---|---|---|---|---|
| shm 合并开 | 12.25 ms | 0.93 | 13.27 | 3.96 |
| **shm 合并关** | **6.01–7.61 ms** | 2.93–4.53 | **10.44–10.54** | **4.22** |

⇒ shm barrier 值 ~5–6 ms/层(把 `period` 从 10.5 抬到 13.3);但**即便没有 barrier,
引擎本体在小批量下仍要 6–7.6 ms/18 pairs ≈ 340–420 µs/pair**,
而**独立微基准**(`bench_vs_lkmoe`,同一引擎同一形状)只有 ~55 µs/pair ⇒ **6–8× 的差距**。

### 33.2 池诊断证明不是"等 worker"

`XIAOTU_MOE_POOL_SLOW_MS=6` 输出:
```
[pool] SLOW parallel_for: elapsed=12ms n=32 counter=568 remaining=0 current_gen=4
  laggards=0/96 remaining=0
```
laggards=0、remaining=0 ⇒ **没有掉队线程,worker 确实在算** —— 即 96 个线程抢一条
ticket cache line、屏障尾部 + 2 个 rank 争核,把"小活"的成本放大了近一个数量级
(引擎注释原话:小批量下"屏障尾部比它并行的算术还贵")。

### 33.3 为什么没能用"worker 子集"修

- 引擎里正确的做法是 `small_batch_workers()` + `parallel_for_limited`;
  **但 packed4/kNParallel 路径绕过了它**(只有 FP8 的 `kNSliceSmallM` 分支用);
- 我按 FP8 的写法给 packed4 补上 wlimit 后,**两种跑法都挂**:
  warmup 阶段卡死、EngineCore 报 `shm_broadcast ... 60 seconds`;
- 读代码后确认根因:`small_batch_workers()` 在 DS-V4 维度上(macs 极大)恒返回 `nt`,
  于是传进去的是 **`wlimit == nt_`** 这个边界值,而 `parallel_for_limited` 的等待路径
  ("limited 且 remaining==n 时只 notify 一次")在这个边界上有竞态 ⇒ 死锁。
- **回退补丁(已重建 .so)**;要真正修得先修 `numa_pool` 的 limited 路径(附带单测),
  或在创建引擎时把池线程数降下来(但 48 线程实测无改善:13.07 vs 12.25)。

### 33.4 因此当前最优仍是单卡档;1M/TP=2 档需要先修引擎并发

8070 保持:**单卡 256K + DSpark + CUDA graph + 192 线程 + NUMA 分片**
= C=1 **15.94 tok/s**(TPOT 55 ms)、C=2 19.88。目标(25 / 50)尚差 1.6× / 2.5×。

## 34. worker 子集(严格 < nt)可用 + 单调步成本的量化

### 34.1 自旋死锁的真正机制(与 §33.3 互补)

读 `numa_pool.hpp` 的等待循环:caller **只在 limited 调用里** `cv_.notify_all()` 一次;
非 limited 调用**完全依赖 worker 自旋**来发现新任务 ⇒ `XIAOTU_MOE_SPIN_IDLE_US` 一旦小于
调用间隔就必然死锁。⇒ 要降自旋只能走 **limited 路径**(它有 notify),而不是调小自旋窗口。

### 34.2 补丁(已生效,TP=1/TP=2 都不挂)

`forward_many` 的 packed4 分支里,小批量(`M*k <= 4*nt`)时传 `wlimit = nt/4`,
并强制 `wlimit < nt`(避免上次 `limit==nt` 的边界死锁):

| 口径 | 改前 | 改后 |
|---|---|---|
| TP=2 每层 compute | 6.0–7.6 ms | **4.2–4.4 ms** |
| TP=1 单卡 C=1(端到端) | 15.94 tok/s | **15.95 tok/s**(无回归) |
| TP=1 每层 | 1.78 + 0.91 | 1.97 + 0.91(period 2.88–3.00) |

⇒ 引擎侧 compute 降了,但 rest 涨了,**单步总时间基本不变**(C=1 4.1 vs 4.22 tok/s)。

### 34.3 单调步成本的量化(这是下一步的依据)

| 口径 | qlen | period/层 | compute | rest | 每步 | 单路 |
|---|---|---|---|---|---|---|
| C=1(DSpark,accept 2.66) | 6 | **2.88–3.00 ms** | 1.97–2.12 | 0.91 | 149 ms | **15.95 tok/s** |
| C=2 | 12 | **4.44–4.69 ms** | 3.37–3.67 | 1.06 | 247 ms | 9.67(合计 19.34) |

目标(C=1 与 C=2 都要 ≈25/路):**每步必须稳定在 ~106–120 ms**(= 每层 ~2.5 ms),
而我们的每步从 149(C=1)涨到 247(C=2)。

**关键换算**:compute 与 pairs 近似线性(55 µs/pair @36 pairs → 47 µs/pair @72 pairs),
说明小批量下走的是 **per-token 路径**(每个 (token,expert) 对都读一次专家块),
而不是"每个活跃专家只读一次"的 grouped 路径(阈值 `nave*8 <= NASS` 在 qlen≤12 时判为 per-token)。
生产同口径是 2.8 ms/层 / 72 pairs ≈ 39 µs/pair ⇒ **每 pair 我们只慢 1.2×,
差距主要来自"每步 token 数/接受率/非 MoE 开销",而不是引擎内核本身。**

### 34.4 下一轮的三条线(按性价比)

1. **降每步的非 MoE 开销**:C=2 实测每步 247 ms,而 43 层 × 4.5 ms = 194 ms
   ⇒ **约 50 ms/步(20%)花在采样/logits/调度上**;早期 profile 里 `_topk_topp_kernel`
   单次 3.6 ms、每步 2 次就 7 ms,值得单独查。
2. **提接受率** 2.66 → 3.01(生产水平):调 draft 参数/采样方法,直接换 ~13% 单路速度。
3. **1M/TP=2**:仍受限于池的自旋/唤醒结构(§33),需要引擎侧认真改 + 单测。

### 34.5 试过但不行的:FlashInfer sampler

`VLLM_USE_FLASHINFER_SAMPLER=1` 直接起不来(与当初在 runbook 里关掉它一致)。
⇒ 每步那 ~50 ms 的非 MoE 开销暂不能从采样器换功能解决,得从别处查
(下一步:用 `XIAOTU_TORCH_PROFILE_DECODE` 在 *eager* 下抓一张 C=2 的 kernel 表,
看 sampling/logits/调度各占多少)。

---

## 35. 每 pair 成本的物理来源(权重流式带宽)+ draft 长度 k 的最优化

### 35.1 为什么成本 ∝ (token, expert) 对数 —— 代码级确认

`moe_v2.hpp:486`:packed4/MXFP4(`kNParallel`)分支**直接 return 到
`forward_many_nsliced`,完全绕过 §"expert grouping"**(那段 grouping 只在
BF16/FP8 的 legacy 路径里生效)。nsliced 内部虽然按专家把同一专家的多个
instance 收成连续行(有 me 路复用),但 DS-V4 的维度下 **qlen≤12 时
36–72 个 assignment 几乎落在互不相同的专家上(256 选 6)** ⇒ me≈1,
**每个 (row,expert) 对都要把那 12.6 MB 的专家块从 DRAM 流一遍**。

字节数(实测内存反推):引擎 TP=1 占 ~139 GB = 43 层 × 256 专家 × 12.6 MB;
每专家 = 25.2 M 参数 × 0.5 B(fp4)= 12.6 MB。
- qlen=6(1+k=6,C=1)⇒ 36 对 ⇒ **454 MB/层**,实测 compute 1.94–2.12 ms ⇒ **~234 GB/s**
- 边际:qlen 6→12 时 2.0→3.5 ms ⇒ **42–47 µs/pair ⇒ ~280–300 GB/s**

### 35.2 这个带宽离机器上限还差多少(单线程已测)

`scripts/bwprobe.c` / `/tmp/bw2`(顺序流式读、每 cache line 取 1 个 u64):
**单线程 35–36 GB/s**(16 GB 缓冲两趟一致,确认是 DRAM 而非缓存)。
⇒ 192 线程若线性叠加本应远超 234 GB/s;但集计上限未能测得可信值
(见 §35.4 的教训)。**结论暂记为:引擎已达到"每对读一次 12.6 MB"这一结构的
~234 GB/s,是否接近机器集计上限仍未定 ⇒ 内核还有多少空间未定论。**

### 35.3 逐位置接受率决定了 k 的最优值(实测)

服务端日志 `[metrics.py:120] SpecDecoding metrics` 每 10 s 打印:
```
Per-position acceptance rate: 0.667, 0.393, 0.179, 0.048, 0.036   (k=5, 较大样本)
Mean acceptance length: 2.32–2.66   (含 bonus token)
```
第 4/5 个 draft 的接受率只有 0.048/0.036,而每个 draft token 的代价是
qlen +1 ⇒ 每层 +6 对 ⇒ 43 层 × 6 × 42 µs ≈ **11 ms/步**。
用实测斜率(每层 base 1.08 ms + 每 qlen-token 0.308 ms,43 层,非层开销 S≈20–28 ms):

| k | 每步 ms | tok/步 | TPOT | 单路 tok/s(模型) |
|---|---|---|---|---|
| 1 | 93 | 1.67 | 55.7 | 17.9 |
| 2 | 106 | 2.06 | 51.5 | 19.4 |
| 3 | 119 | 2.24 | 53.3 | 18.7 |
| 4 | 133 | 2.29 | 58.0 | 17.2 |
| 5 | 146 | 2.32 | 62.8 | 15.9 |

⇒ **以我们当前引擎的边际成本,生产惯用的 k=5 并不是最优,k=2–3 才最优**;
生产的 39 µs/pair + 接受率 3.01 才让 k=5 划算。⇒ 引擎每快一点,最优 k 就变大一点,
两条线必须一起调。

### 35.4 教训(两次踩坑,都已写进流程)

1. **绝不在跑基准的同时跑重量级探针**:k=2 那组(C=1 6.39 tok/s / TPOT 127 ms、
   C=2 9.37 tok/s)是在我自己的 48 GB / 192 线程带宽探针**失控挂了 2 分钟**期间测的,
   **整组作废,必须重测**。轻量探针也只在服务"加载权重"阶段跑。
2. `pkill -9 -f "bw3 48"` 又一次**杀掉了自己所在的 shell**(§33 记过一次):
   匹配串必须写成不自匹配的形式(如 `"bw[3] "`)。
3. 机器负载终于安静下来(load 3,之前基线都在 **load 60–80** 下测的)
   ⇒ 所有旧基线偏悲观,需要在安静机器上重测一遍 C=1/C=2 基线。

### 35.5 本轮实验脚本

- `scripts/sweep_spec_k.sh`:对 k ∈ {2,3,5} 各起一次服务 + C=1/C=2 定长解码
  (IN_LEN=1024、OUT=200、`--ignore-eos`),并从服务端日志抓逐位置接受率到
  `report/tuning/logs/spec<k>.acceptance`。
- `scripts/bwprobe.c`:机器流式读带宽探针(仅供单线程/少量线程时使用)。

### 35.6 k 扫描实测结果(注意:该组用的是 random 数据集,**接受率被人为压低**)

| k | C=1 tok/s | C=2 合计/单路 | 逐位置接受率(位置0) | 平均接受长度 |
|---|---|---|---|---|
| 2 | (污染,作废) | (污染) | 0.33–0.51 | 1.35–1.77 |
| 3 | **5.82** | 7.87 / 3.94 | 0.23–0.70 | 1.34–2.32 |
| 5 | 5.34 | 7.55 / 3.78 | 0.22–0.67 | 1.33–2.29 |

`--dataset-name random`(随机 token)**把 draft 质量打到地板**:k=5 时位置0 只有
0.22–0.67(自然文本下是 0.67–0.85)、平均接受长度 1.33–2.29(自然文本 2.32–2.66)。
⇒ 随机 token 下"每步成本"主导,k 越小越快(k=2 > k=3 > k=5)。

## 36. k 的最优值由 draft 的训练块长决定(k=5 是对的,§35.3 的模型作废)

读 mainline 的 dspark 校验代码(`vllm/config/speculative.py:146`)与 DS-V4 的
`config.json`:`dspark_block_size: 5`、`dspark_noise_token_id: 128799`、
`num_nextn_predict_layers: 1`。DSpark/DFlash 是**并行 draft**(一次前向出一个
block 的 token,用 noise token 占位),draft 头是在 **block_size=5** 上训练的;
校验里对 Qwen3-Omni 直接要求 `num_speculative_tokens == block_size`。

这解释了 §35.6 里"k 越小位置0接受率越低"的反常现象(0.67 → 0.35 → 0.30):
**截断 block 会破坏 draft 的联合预测**,不是"少猜几个更容易猜中"。
⇒ **§35.3 那个"接受率与 k 无关"的假设是错的,按它推出的 k=2 最优不成立;
k=5(训练块长)才是正确选择。**
⇒ 教训:优化 spec decode 的 k 之前,先确认 draft 的训练块长 / 是否并行 draft。

## 37. 上下文长度与文本分布是必须量化的混淆项(测量方法已补齐)

旧基线(C=1 16–17 tok/s)用的是 sharegpt 的**短 prompt**:实测 total_input_tokens
只有 64/3 = 21 token/条(见 `report/tuning/raw/grp_c1.json`)。
而扫描用的 1024 token 输入只有 5.3–6.4 tok/s。二者差 2.5–3×,里面混了两个变量:
上下文长度、以及文本分布(随机 token vs 自然文本 ⇒ 接受率 1.3 vs 2.3)。
⇒ 为得到能代表目标场景的数字,新增:
- `scripts/make_nat_dataset.py`:从 ShareGPT 拼出**精确 token 长度**的自然文本数据集
  `report/tuning/datasets/nat{32,64,128,512,1024,4096}.jsonl`(实测首条 = 目标长度);
- `scripts/bench_nat_client.py`:自建客户端(不依赖 pandas),
  带 `stream_options.include_usage`,同时给出 **每步延迟** 与 **tokens/步**。
- 实测结论(§38.4):**每层 period 在 22→4096 token 六档上都是 4.1–4.7 ms
  ⇒ 上下文长度对解码步时间没有影响**,先前 2.5–3× 的差异全部来自
  随机 token 的接受率崩塌 + GPU 时钟状态,不是上下文。

## 38. 本轮最重要的结论:瓶颈在**机器的实际内存带宽**(不是内核)

> ⚠️ **本节 38.1–38.2 的结论随后被 §39 推翻**:当时的带宽探针把所有物理页都落在
> 一个 NUMA 节点上(主线程首次触碰),测到的 95 GB/s 是**单节点**上限;
> 正确做法是每节点绑定本地内存 + 本地 CPU,实测**整机 783 GB/s**。
> 引擎实际只用到 ~7% 的节点带宽 ⇒ **内核有极大空间**,并非无路可走。
> 38.3–38.6(GPU 降频、上下文无关、达标算术)仍然成立。

### 38.1 机器实测带宽上限(三种方法一致)

| 口径 | 结果 |
|---|---|
| 单线程顺序流式读(16 GB,两趟一致) | **35–36 GB/s** |
| 24 线程(每线程私有 1.3 GB 切片) | **97 GB/s**(≈4 GB/s/线程) |
| 96 线程 | 81 GB/s |
| 192 线程(整机) | 94–97 GB/s(≈0.5 GB/s/线程) |
| **8 个独立单线程进程并发(各 6 GB,不同 CCD)** | 每个 11.8–16.1 GB/s,**合计 103 GB/s** |
| 历史报告 `process_data/decomp/NUMA_BANDWIDTH_CCD.md`(8 cores/CCD 起测) | 1 CCD 30 GB/s → 24 CCD **73.7 GB/s** |

本机 8 节点(NPS=4)、1.5 TiB(24×64 GB DDR5-4800),理论峰值 ~920 GB/s,
但**实测集计读带宽只有 ~95–105 GB/s(约 10%)**,且 3 个线程就到了平台期。
**"8 个独立进程并发"这一路排除了"多线程探针自身有缺陷"的可能**(8×11.8–16.1 = 103 GB/s,
而不是 8×35 = 280)。宿主没有 resctrl/MBA 限流可读(容器内看不到),CFS 也无节流
(`nr_throttled=0`)⇒ 只能按"这台机器当前能给 ~100 GB/s"来做工程决策。
⇒ 这是**硬上限**,任何 CPU 端内核优化都不可能突破它。**这也是本项目最该上报给运维的事**:
如果内存带宽能恢复到硬件应有水平,MoE 的 86 ms/步会掉到 ~20 ms,目标立刻可达。

### 38.2 引擎已经贴着这个上限

- 引擎每层要读的专家权重 = **去重后的活跃专家数 × 12.6 MB**(fp4,25.2 M 参数/专家)。
- 真实路由下每层约 2.0 ms;随机路由(36 个互不相同专家 = 454 MB)下
  `scripts/bench_cpu_engine.py` 实测 5.46–8.06 ms/层 ⇒ **56–83 GB/s**。
- 按 NUMA 分片,每节点每层约 19 MB,24 线程读出 9.5 GB/s,而 24 线程的探针上限
  只有 ~12 GB/s ⇒ **引擎已达节点带宽的 ~80%**。
- 结论:**小批量 MoE 的成本是"必须读的字节数 / 机器带宽",继续抠内核最多再拿 15%。**

### 38.3 每步成本的实测拆解(C=1、k=5、qlen=6、自然文本)

| 项 | 数值 | 说明 |
|---|---|---|
| 每层 `compute`(MoE 回调:D2H+引擎+H2D) | 2.0–2.7 ms | ×43 = **86–116 ms/步** |
| 每层 `rest`(GPU 注意力/dense + 主机 Python) | 0.91 ms(早先) → **2.1–3.1 ms(现在)** | ×43 = 39 → 90–133 ms/步 |
| 非层开销(采样/调度/反序列化) | ~25 ms/步 | |
| **合计每步** | **~200–215 ms** | 客户端实测(修正后的客户端) |
| tokens/步 | 2.5–2.8 | |
| 单路 | **12 tok/s**(最好一次 11.97/13.9) | 目标 25 |

### 38.4 `rest` 为什么从 0.91 涨到 2.1–3.1 ms:GPU 降频(已证实)

- `nvidia-smi` 全程显示我们用的 GPU2:**SM 765 MHz / 最大 1410 MHz,利用率 14–21%,
  功耗 45 W / 250 W**。因为解码是 CPU 端 MoE 主导(1/3 时间 GPU 空转),
  驱动一直不给上频。
- 想锁频:`nvidia-smi -i 2 -lgc 1410` → **当前用户无权限**(需要 root)。
- 同样的二进制、同样的配置、同样的 qlen=6,早先 `rest`=0.91 ms(那时 GPU 更忙、
  时钟更高),现在 2.1–3.1 ms ⇒ 差的就是 GPU 时钟。
- 对照:上下文长度 22/64/128/512/1024/4096 token 六档,**每层 period 都是 4.1–4.7 ms**
  ⇒ 注意力开销与上下文长度无关(1M 目标不需要为"短上下文的假基线"买单)。

### 38.5 客户端测量踩的坑(已修)

自研客户端最初把 **SSE chunk 当 token** 计数:vLLM 在投机解码下**每步只发一个 chunk**,
于是一路吞吐被低估 (1+接受长度) 倍(3.5 tok/s vs 真实 8.8)。
修法:请求里带 `stream_options={"include_usage": true}`,token 数取 `usage.completion_tokens`;
chunk 数 = **步数**,顺便直接量出"每步延迟"和"tokens/步"(与引擎侧 `period` 交叉验证一致:
43×4.5 ms ≈ 200 ms ≈ 客户端 step 211 ms ✓)。

### 38.6 达标的算术(给下一轮定方向)

- 目标 25 tok/s/路、2.8 token/步 ⇒ **每步必须 ≤ 105 ms**。
- 下限:MoE 68–86 ms(6.5 GB/步 ÷ 95 GB/s)+ 注意力 39 ms(满频)~90 ms(当前频率)+ 采样 25 ms
  ⇒ **130–200 ms**。
- ⇒ 当前机器条件下**做不到 25 tok/s/路**;能动的只有三块:
  1. **GPU 上频**(需要 root:锁 SM 时钟或用别的方式提高利用率)⇒ 直接省 40–50 ms/步;
  2. 减少每步 MoE 字节(只能靠减少 token/步 = 改接受率,draft 块长固定 5 ⇒ 空间很小);
  3. 砍非层开销 25 ms(采样 3.6 ms×2、logits、调度)。
- 反过来:**如果机器内存带宽能恢复到正常水平(而不是 95 GB/s)**,MoE 从 86 ms 掉到
  ~20 ms,立刻达标。⇒ 值得先确认这台机器(或这批租户)是否存在内存带宽被抢占/BIOS 配置问题。

### 38.7 本轮脚本与产物

- `scripts/bench_nat_client.py`:目标场景客户端(自然文本数据 + usage 计 token + 每步延迟)。
- `scripts/make_nat_dataset.py` / `report/tuning/datasets/nat{32,64,128,512,1024,4096}.jsonl`。
- `scripts/bwprobe.c`:机器带宽探针(单线程/少线程时用,勿与基准并行跑)。
- 结果在 `report/tuning/summary.jsonl`(`natural_text=1` 的行)。


旧基线(C=1 16–17 tok/s)用的是 sharegpt 的**短 prompt**:实测 total_input_tokens
只有 64/3=21 token/条(见 §35.6 的表和 `report/tuning/raw/grp_c1.json`)。
而扫描用的 1024 token 输入给出 5.3–6.4 tok/s。二者差 2.5–3×,里面**混了两个变量**:
1. 上下文长度(注意力/indexer 的代价随上下文增长);
2. 文本分布(随机 token vs 自然文本 ⇒ 接受率 1.3 vs 2.3)。

⇒ 为了得到能代表目标场景的数字,新增:
- `scripts/make_nat_dataset.py`:从 ShareGPT 拼出**精确 token 长度**的自然文本数据集
  `report/tuning/datasets/nat{128,512,1024,4096}.jsonl`(实测首条 = 目标长度)。
- `scripts/bench_nat.sh` / `scripts/run_nat_curve.sh`:在单卡 256K + k=5 + CUDA graph
  的已知好配置上,跑 `上下文 × 并发` 曲线(带 `XIAOTU_CD_TIMING=1` 每层计时),
  结果进 `report/tuning/summary.jsonl`(带 `natural_text=1` 标记)。


---

## 39. 更正 §38 + 拿到第一个真正的内核加速:me=3 的 2/3 行分块(1.7×)

### 39.1 更正:整机带宽是 783 GB/s,不是 95 GB/s

错在哪:探针用一个 `aligned_alloc` 的缓冲(主线程首次触碰)⇒ **所有页都在一个 NUMA
节点上**,192 个线程跨节点去读同一份内存,量到的是"单节点 + 跨 socket 读"的混合值。

正确测法(`numactl --cpunodebind=N --membind=N`,每节点独立进程 + 本地内存):

| 口径 | 结果 |
|---|---|
| 单节点 1 线程 | 36.7 GB/s |
| 单节点 8 线程 | 90.7 GB/s |
| 单节点 24 线程 | **98.7 GB/s**(3 通道 DDR5-4800 理论 ~115 ⇒ 86% 效率,正常) |
| **8 节点 × 24 线程(整机)** | **783.0 GB/s**(8×98 = 784,完美线性) |
| (旧的错误口径:单块缓冲 + 192 线程) | 94–103 GB/s ← 单节点上限 |

⇒ 机器没有任何异常,是我测错了。**引擎每层 151 MB / 1.17–2.0 ms = 75–130 GB/s
= 每节点 9–16 GB/s,只有节点上限(99 GB/s)的 10–16%** ⇒ 内核离硬件上限还很远。

### 39.2 根因:真实解码的 me=3 落在"4 行分块"快路径之外

- `bench_cpu_engine.py` 原来用**全随机路由**(36 个互不相同的专家,me=1),测不到
  真实情形。真实解码步里 6 个 draft token 属于同一段文本,**去重后只有 ~12 个专家,
  me≈3**(用引擎自带 `XIAOTU_MOE_PROFILE` 实测:`na=6..12`,skew 桶 `2-7` = 12 个专家)。
- `gate_up_slice_batch_impl`/`down_slice_batch_impl` 把 me 直接传给
  `matmul_packed4_group(..., M=me, ...)`;该内核只有 **M≥4 的"解码一次喂 4 行"**快路径,
  M=1..3 落到单行兜底 ⇒ **每一行都把权重 nibble 解码重做一遍**。
- 判据(同一批 12 个专家 = 同样 151 MB 权重):

| 情形 | 走的分支 | ms/层 | µs/assignment |
|---|---|---|---|
| me=4 | 4 行分块 | 1.50 | **31.2** |
| me=3 | 单行兜底(旧) | 2.00 | **55.6** |

⇒ 每 assignment 白烧 **1.78×** 的解码算力,而真实解码恰好一直是 me=3。

### 39.3 修复:补 2 行 / 3 行分块(`moe_v2_packed4.hpp`,FAST_FP4 路径)

4 行分块之后、单行兜底之前插入 `block_23<R>`(模板 + `std::integral_constant`):
**解码一次喂 2 或 3 行**,每行仍是 4 个部分累加器(与 4 行分块**完全相同的累加结构**,
所以每行内部数值语义不变)。M=5/6/7 自然变成 4+1 / 4+2 / 4+3。

实测(BS=6, DEDUP=12, REP=79, REP 内 40 次调用的阶段累计):

| 阶段 | 改前 | 改后(两次) | 加速 |
|---|---|---|---|
| A 门/上投影 | 48.8 ms | 25.3 / 26.5 ms | **1.9×** |
| B 下投影 | 27.4 ms | 17.6 / 18.0 ms | **1.55×** |
| C 加权归约 | 3.7 ms | 3.5 / 3.8 ms | 1.0× |
| **每层合计** | **2.00 ms** | **1.17 / 1.20 ms** | **1.7×** |

### 39.4 数值正确性(新增 `scripts/test_block23_equiv.py`)

用真实 MXFP4 权重 fixture + numpy golden,**受控路由**逐 case 对拍:

| case | me 分布 | max_abs | max_rel(|g|>0.05) |
|---|---|---|---|
| me=1(未改动路径) | 全 1 | 4.4e-3 | 1.9e-2 ⚠️(既有差异,M=1 不进新代码) |
| me=2 | 全 2 | 1.2e-4 | 3.9e-4 ✅ |
| me=3 | 全 3 | 1.2e-4 | 3.7e-4 ✅ |
| me=2/3 混合 | 2,3 | 7.6e-5 | 2.1e-4 ✅ |
| me=4 / 5 / 6 / 7 | — | ≤1.8e-4 | ≤5.6e-4 ✅ |

另:`scripts/test_swiglu_clamp_mxfp4.py`(M=8、16 专家随机路由)也全过。

### 39.5 工具与方法学

- `scripts/bench_cpu_engine.py` 新增 **`DEDUP=N`**(去重后的活跃专家数),
  这是让小批量引擎基准有代表性的关键旋钮;**不设 DEDUP 的旧数字全部偏悲观**。
- `XIAOTU_MOE_PROFILE=1` 对 packed4/nsliced 路径**是有效的**(`prof_add` 在
  `forward_many_nsliced` 末尾),但每 40 次调用才打印一次 —— REP 要 ≥40 才看得到。
- 基准噪声极大(同一配置 2.4 / 10.1 / 7.4 ms 都出现过)⇒ 至少两次独立复现再下结论。
- 构建:`PYTHON=<venv> PYBIND11_INC=<torch>/include scripts/build_engine_variants.sh`
  (环境里没装 pybind11,用 torch 自带的头文件目录即可)。构建会多出一个
  `_avx512_bf16` 变体,而 loader 的优先级最高是它 —— 做 A/B 时要先把它移走,
  否则 ISA 变了会污染对比。

### 39.6 第二处修复:单行路径的 4 路独立累加(ILP)

服务内实测:一次解码调用的 36 个 assignment 落在 **~32 个不同专家**上
(`skew(1|2-7)=28|4`)⇒ **真实 me≈1.1**,单行路径才是解码热路径。
而单行路径原来是**一条 128 长的串行 FMA 依赖链**(`total0 = fmadd(d,sv,total0)`
×128 ≈ 512 cycle/行),OoO 无法掩盖。改成 4 个独立部分累加器后:

- 交错 A/B(DEDUP=0 ⇒ na=16、skew 29|3,与线上同形):旧 **87.6 ms** → 新 **63.0 ms**
  (每 40 次调用的 A+B 累计)⇒ **1.39×**;
- DEDUP=12(me=3):**1.59×**(见 39.3);
- **顺带修好了一个数值问题**:`scripts/test_block23_equiv.py` 里 me=1 那一档原先
  max_rel=**1.9e-2**(当时误判成"golden 偏差"),改后 **3.0e-4** —— 说明旧的串行链
  本身误差就大。现在 8 个 case 全部 ✅。

### 39.7 本轮结论

| 口径 | 旧内核 | 新内核 | 加速 |
|---|---|---|---|
| 微基准 me≈1(与线上同形) | 87.6 ms/40 calls | 63.0 ms | **1.39×** |
| 微基准 me=3 | 69.1 ms | 43.4 ms | **1.59×** |
| 每层 MoE(推算,me≈1) | ~2.2 ms | ~1.6 ms | 1.39× |

仍待办:
1. **服务端端到端复核**(本轮末尾在做:负载 90–124 时的 7.2–9.0 tok/s 与基线
   11.55(负载 ~40)不可比,必须同窗口对比);
2. 真实调用里还有大量 **M=1~2 的小调用**(`XIAOTU_MOE_PROFILE` 窗口平均 M=2、
   na=11,怀疑 dspark draft 的 MoE 层也走同一引擎)——需要按调用类型分开
   profile,否则平均值误导(本轮就被误导过一次);
3. `rest`(注意力/dense,2.1–4.0 ms/层)已是剩下的大头,与 GPU 765 MHz 降频绑定。

## 40. 端到端 A-B-A 复核(同窗口换二进制重启)

`scripts/ab_serve_kernel.sh`,单卡 256K + DSpark k=5 + CUDA graph + 192 线程:

| 段 | 负载 | 单路 tok/s | 每步 ms | 每层 compute | 每层 rest | 每层 period |
|---|---|---|---|---|---|---|
| old(基线内核) | 72.8 | 6.72 | 344 | 3.57–3.90 | 3.82–4.16 | 7.43–8.06 |
| new(block_23 + ILP) | 65.0 | **8.34** | 276 | **2.45–2.64** | 2.31–2.69 | 4.76–5.33 |

- **引擎项(compute)3.7 → 2.55 ms/层 = 1.45×**,与微基准交错 A/B 的 1.39×(me≈1)
  和 1.59×(me=3)一致 ⇒ 内核修复确实生效。
- 端到端 tok/s +24%,但**不能全记在内核上**:new 那一段负载低 8 个点,`rest`
  (与内核无关)也一起从 4.0 掉到 2.5 ms ⇒ 有一部分是机器变闲带来的。
- `rest` 现在与 `compute` 等量齐观(各 ~50%),是下一轮的主战场(§38.4:GPU 765 MHz)。

### 40.1 本轮结束时的官方数字(新内核,负载 ~71)

| 配置 | 单路 tok/s | 每步 ms | tokens/步 | 接受长度 | 每层 compute / rest |
|---|---|---|---|---|---|
| C=1 ctx128 | **9.15** | 270 | 2.818 | 2.86–3.64 | 2.46 / 2.65 ms |
| C=2 ctx128 | 5.96(合计 11.92) | 433 | 3.004 | ~3.0 | — |

- 同负载下旧内核 6.72 tok/s ⇒ **新内核 +36%**;
- 接受率依旧达标(2.86–3.64 ≥ 生产 3.01);
- 机器空闲负载(~40)时按同样的 compute/rest 推算:C=1 约 **15–16 tok/s**
  (compute 1.5×43=65 ms + rest 2.1×43=90 ms + 非层 ~25 ms ≈ 180 ms/步)。
- **`rest` 已占每步 ~50%,是下一轮的主战场**(GPU 765/1410 MHz、注意力/dense);
  引擎侧仍有空间(节点带宽只用掉 10–16%),但边际收益已不如 `rest`。

---

## 41. 第二个大杠杆:引擎线程池的 5 ms 自旋把整机烧穿(rest 3.3 → 0.95 ms/层)

### 41.1 现象:解码期间进程占 166–259 核,而真正的 MoE 算力只需 ~21 核

按 /proc 逐线程统计(解码中,10 s 窗口):**193 个线程活跃,合计 166–259 核**
(机器一共 192 核);空闲 30 s 后完全停泊(0 核)。
而 MoE 的实际工作量是:`compute ≈ 2.5–3 ms` / `period ≈ 4–6 ms`,只有 48 个 worker
参与(limited 路径)——即真正的算力需求约 **21 核**。⇒ **约 75% 的 CPU 是自旋浪费**。

原因:`numa_pool.hpp` 的 worker 在两次调用之间会自旋 `XIAOTU_MOE_SPIN_IDLE_US`(默认
**5000 µs**)等下一次任务,而解码步里相邻两次调用只隔 ~2–4 ms ⇒ 自旋窗口永远不到期,
192 个 worker 在整个解码期间全速空转。它们既浪费机器,又**抢走驱动 GPU 的主线程**,
于是每层 `rest`(GPU 注意力/dense + 主机侧)被推高到 3.3–3.5 ms。

### 41.2 修复:自旋窗口 5000 → 300 µs(已写进插件默认)

`vllm_xiaotu_moe/hybrid_model.py::finalize_moe_weights` 在建引擎之前
`os.environ.setdefault("XIAOTU_MOE_SPIN_IDLE_US", "300")`(用户仍可用环境变量覆盖)。
不会死锁:自旋超时后 caller 在 limited 与非 limited 两条路径上都会 `cv_.notify_all()`
唤醒(`numa_pool.hpp::parallel_for` 的 fallback),已实测解码/预填充都正常。

### 41.3 实测(同程序列,括号法;单位 ms/层)

| spin | 负载 | C=1 tok/s | period | compute | **rest** |
|---|---|---|---|---|---|
| 5000(旧默认) | 71 / 100 | 9.15 / 7.53 | 4.76–7.12 | 2.45–3.65 | 2.31–3.83 |
| 1000 | 56.9 | 10.56 | 4.10–4.64 | 2.85–3.25 | 1.25–1.39 |
| 100 | 60.7 | 10.14 | 3.81–3.95 | 2.88–2.97 | 0.93–0.98 |
| **300(新默认)** | 93.3 / 65.6 | **12.04 / 11.72** | 3.78–4.37 | 2.84–3.40 | **0.94–0.98** |

- **`rest` 降 3.5×**(3.3–3.5 → 0.94–0.98 ms/层),`compute` 基本不变;
- 端到端 **+28%~60%**(同负载下 9.15 → 11.7–12.0 tok/s;负载 93 时也有 12.04);
- 100 与 300 每层等价(留 300 作为对"调用内部阶段间隔"的余量)。

### 41.4 新的成本结构(C=1、qlen=6、spin=300)

| 项 | ms/层 | ms/步(×43) | 占比 |
|---|---|---|---|
| compute(CPU MoE,回调体 = 纯 `forward_many`) | 2.84–3.40 | 122–146 | **~75%** |
| rest(注意力/dense + 主机侧) | 0.94–0.98 | 40–42 | ~22% |
| 非层开销(采样/调度) | — | 15–30 | ~8% |
| **合计每步** | | **~200 ms** | 2.97 token/步 ⇒ 15 tok/s |

⇒ `rest` 这块肥肉已经吃掉;**下一轮的主战场变回 `compute`(CPU MoE 引擎本身)**。
目标 25 tok/s 需要每步 ≤105 ms,即 compute ≤ ~1.0 ms/层 —— 而**微基准在同形路由下
只要 1.58 ms/层(A+B),服务内却是 2.84–3.40 ms**,两者差 1.3–1.8 ms 需要定位
(候选:服务内 worker 逐层 park/wake 的代价、真实路由比合成路由更分散、其他租户的访存争抢)。

### 41.5 微基准 vs 服务内的差额复核(部分归因于机器争抢,未完全解释)

同一形路由(na=16、skew 29|3、B=6/DEDUP=0、192 线程):

| 测量时刻负载 | 每层 A+B | C 阶段 | 说明 |
|---|---|---|---|
| 27–43(交错 A/B 窗口) | **1.58 ms** | 0.09 | 服务内同时刻为 2.84–3.40 |
| 18.9(本次复测) | 1.81 / 1.90 ms | 0.09 / **0.48** | run2 的 B/C 明显被扰 |

⇒ ①微基准与服务内仍有 **~1.0–1.5 ms/层** 的差额待定位;
②**同一配置的微基准在不同时刻也能差 2×**(C 阶段 3.7 → 19.3 ms 都出现过),
说明引擎是访存受限,而**负载均值反映不出其他租户的内存带宽争抢**;
③因此"服务内 compute 2.8–3.4 ms"里有多少是引擎结构问题、有多少是机器争抢,
需要更硬的证据(建议下一轮:在 compute 回调内加 `XIAOTU_MOE_PROFILE` 的
A/B/C 阶段计时,与 CD 的 compute 直接对齐;或在机器真正安静时(load<5)复测)。

---

## 42. 引擎调用内部的阶段拆解(新增 `XIAOTU_MOE_PROFILE` 分桶))与第三处优化

### 42.1 新增按调用规模分桶的阶段计时(把 draft 小调用与主调用分开)

`moe_v2.hpp::prof_add` 现在按 M 分三桶(M≤2 / 3–8 / >8)分别累计
`setup/A/B/C/ovh` 并每 200 次调用打一行 `[NS-PROF]`。实测(M=6/na=32/DEDUP=0、
192 线程、spin=300)单次引擎调用:

| 版本 | setup | A 门/上 | B 下 | C 归约 | 合计 |
|---|---|---|---|---|---|
| 真解码(修好 scale 查表后) | 246 | 1076 | 1044 | 492 | **2858 µs** |
| **诊断构建:把解码换成平凡实现(结果错,只测时间)** | 266 | **746** | **475** | 128 | **1615 µs** |

⇒ **解码占 43%**;而"内存 + FMA 循环"的下限仍有 1615 µs(只到节点带宽的 31%),
说明内核是**指令数/发射受限**,不是 DRAM 受限。这给下一轮定了方向:
用 AVX512-BF16 的 `vdpbf16ps`(一条指令 32 个 MAC,且激活本来就是 bf16、不必转 fp32)
重写解码+FMA,理论上把每个 32 值组的指令数从 ~20+5M 降到 ~8+2M。

### 42.2 第三处优化:去掉内层每 group 的整数除法(1.13×)

FAST_FP4 分支里 `gk == 32` 恒成立 ⇒ `kbase/gk == g`、`(K+gk-1)/gk == K/32` 都与
g 无关,只有 `(n/gn)` 是每行一次。原来的 `scale_at()` 在**每行每个 group** 都做
两次整数除法(Zen4 上 div ~20-40 cycle),而内层是 128 group/行,单次调用上万次除法。

改成:每行算一次 `srow = (j/gn)*kb_stride`,内层只做 `Sbytes[srow+g]` 一次取值。
交错 A/B(负载 1.2–1.4):

| 内核 | setup | A | B | C | 合计 |
|---|---|---|---|---|---|
| 旧(每 group 两次除法) | 244 | 1303 | 1252 | 437 | 3236 µs |
| **新(行级索引)** | 246 | 1076 | 1044 | 492 | **2858 µs** |

⇒ **1.13×**;数值对拍(`test_block23_equiv.py` 8/8、`test_swiglu_clamp_mxfp4.py`)通过。

### 42.3 试过但回退的:C 阶段分块 + 向量化(未证明收益)

把 C 阶段从"每 token 一个 job"改成"(token × hidden 分块)"并向量化:理论上
M=6 时参与 worker 从 6 个升到 48 个。但交错 A/B 的**最小值统计**显示新版本在
A/B/C 三个阶段同时偏慢(说明该轮次测量顺序有系统偏差,前后条件不可比),
C 阶段本身在**同一内核**的不同轮次里也在 105–786 µs 之间跳 ⇒ **该阶段受
"每次池调用的固定开销(唤醒+屏障 ≈100–300 µs)"支配,而不是受工作量支配**。
按"未证明收益不留"的原则已回退(回退后二进制与 scale 版逐字节一致)。

### 42.4 本轮小结

- 引擎单次调用:2858 µs(修复前口径下同类测量约 3.2–3.9 ms)⇒ 本轮再拿 ~1.13×;
- 引擎成本结构:setup 8.6% + A 37.7% + B 36.6% + C 17.2%,其中解码占 43%;
- **下一步最高价值**:用 vdpbf16ps 重写解码/FMA(理论 1.5–2×,即每步省 ~40 ms);
  以及把"每次池调用 100–300 µs 的固定开销"打下来(3 阶段/层 × 43 层 = 129 次/步)。

### 42.5 试过但**挂死**回退的:限制解码时的参与 worker 数

发现:packed4 分支**根本没有传 `wlimit`**(源码注释明确写着"不要传"),所以真实解码时
**192 个 worker 全部参与每一个阶段**(3 阶段/层 × 43 层 = 129 次 192 路屏障/步)。
`small_batch_workers()` 这个现成机制只在非 packed4 分支生效 —— 这解释了为什么
解码期间有 193 个线程活跃、以及自旋修复的收益那么大。

实验:给 packed4 分支加 `XIAOTU_MOE_WLIMIT`(=0 不限制)并扫描 0/24/48/96。
结果:**0 正常(8081 µs,该时刻机器严重争抢);24 与 48 直接挂死**(600 s 超时未完成)。
原因就是 §33 记录的 limited + sharded 竞态:limited 子集按 `w % (nt/lim) == 0` 选取,
这些 worker 未必覆盖全部 8 个 NUMA 节点,而 sharded 派发要求"每个参与节点都至少有一个
worker 能领票"(`all_present` 只按全体 worker 判断,不按子集),没覆盖到的节点的 job
永远无人认领 ⇒ caller 一直等到 300 s 看门狗。

⇒ 已回退(恢复后二进制与 scale 版逐字节一致)。**要拿下这块(~100–300 µs/阶段 × 3 阶段
= 每步 15–30 ms),必须先修 `numa_pool` 的 limited+sharded 路径**(按节点均匀挑选子集,
或让 `all_present` 按子集判断),这是下一轮的候选工作。

### 42.6 未解之谜(留给下一轮):服务内单次调用比微基准慢 1.6–2×

同刻对比(负载 26–45):

| 口径 | M=6 单次调用 |
|---|---|
| 微基准(`DEDUP=0`,na≈32) | **2262–2483 µs** |
| 服务内(CD timing 的 compute) | **3780–5730 µs** |
| 服务内 NS-PROF 分桶(M3-8, na=18) | 8906 µs(该窗口含 prefill,偏高) |

服务内真实解码调用的 `na` 只有 **18**(比合成路线的 32 还少,字节更少),却慢一倍。
已排除的原因:**`OMP_NUM_THREADS=96`**(改成生产同款 OMP=1 后仍是 4.4–5.7 ms/层;
顺带确认 OMP 不是瓶颈)、负载均值(load 1.1 时服务内 compute 反而 4.6–5.4 ms)。
候选原因:①服务内 43 个引擎实例各自的 scratch 缓冲/页分布与单实例微基准不同;
②真机路由比合成更"散"(需要**纯解码窗口**下按 (M,na) 分桶再测);
③GPU 侧 DMA(pinned staging、KV 读写)与 CPU 引擎争抢内存带宽。
⇒ 下一轮第一件事:在**纯解码窗口**(单请求长输出、无 prefill 混入)下读 NS-PROF 分桶,
把 18 个专家的解码调用与微基准逐项对齐(A/B/C 分别比)。

### 43.6 服务端线程数扫描(同一台机、同一配置,负载 31–50)

| 引擎线程数 | 负载 | C=1 tok/s | 每步 ms | 每层 compute | C=2 合计(单路) | C=2 每步 |
|---|---|---|---|---|---|---|
| 192 | 1–45 | 8.25–9.98 | 281–339 | 4.4–5.7 ms | — | — |
| 176 | 50 | 11.44 | 193.9 | 3.22–3.63 ms | 15.75(7.88) | 302 ms |
| 160 | 38 | 11.66 | 185.2 | 2.29–2.41 ms | 18.71(9.36) | 263 ms |
| **144** | 33 | 12.57 | 162.3 | 1.70–2.21 ms | **21.65(10.82)** | 223 ms |
| **128** | 31 | **12.66** | **158.6** | **1.61–1.67 ms** | 21.60(10.80) | 220 ms |
| (128) C=4 | 31 | 22.2 合计(5.55/路) | 331 | — | — | — |

- 平台期在 **128–144**;服务端的最佳点比微基准(160–184)更靠下,因为服务进程里除
  调用线程外还有 torch/CUDA 驱动/采样/调度等线程,需要更多空闲核。
- 每层 compute 从 4.4–5.7(192)降到 **1.61–1.67 ms(128)** = **2.6–3.4×**;
  此时服务内 compute 已经等于(甚至优于)同形微基准(1.61–1.65 ms)⇒
  **上一轮那个"服务内比微基准慢 2×"的未解之谜就是这个原因**(调用线程抢核),
  不是内存/多实例/带宽问题。
- 接受长度 2.63–3.15(自然文本,达标)。
- 代价:参数变大(qlen=12)时每步 220–223 ms ⇒ 单路 10.8,与 C=1 的 12.66 相比
  每路只掉 ~15%(而 192 线程时 C=2 每路 7.88,掉 31%)⇒ 线程预留对多路也更友好。
- 预填/短上下文影响:TTFT 从 2.5 s(176)轻微变差到 3.0 s(128),可接受。

### 43.7 新默认值

- 插件:`XIAOTU_MOE_THREADS` 未设置时 = `ncpu - 16`(≥64 核)。
  **注意**:服务端实测最佳是 **128–144**(= ncpu-64…ncpu-48),所以脚本/生产配置应显式给值;
  `scripts/tune_serve.sh` 默认 `THREADS=176`,本轮的最佳配置请显式写 `THREADS=144`。
- 下一轮候选:把"预留核数"做成按进程线程数自适应(读 /proc/self/status 的 Threads,
  预留 = 非引擎线程数 + 余量),而不是固定值。

## 44. 【用户给定规则,务必遵守】解码线程数按"每 CCD 4–5 核"定,不要加核

> **规则(用户 2026-09-11 明确指示)**:解码性能的**最优解是每 CCD 开 4–5 个核心**;
> **更多核心不会带来更多带宽**,反而会**抢 L3** 与**抢散热/power 预算**,使性能变差。
> 本机 24 CCD ⇒ **96–120 线程**(不是 192,也不是我先前试的 128/168)。

佐证(自己仓库 + 公开文档,方向一致):

| 来源 | 结论 |
|---|---|
| `process_data/decomp/NUMA_BANDWIDTH_CCD.md`(本机实测) | 1 CCD 读 ~30 GB/s、写 ~11 GB/s;需要多 CCD 才能逼近峰值;**越过 6–8 CCD 后每 CCD 效率下降**(内存控制器饱和) |
| [LvLLM/lk_moe 官方 README](https://raw.githubusercontent.com/guqiong96/Lvllm/main/README.md) | `LK_THREADS` =(物理核 **÷** GPU 数);HT off 时 =(物理核 **− 2**)÷ GPU 数;`LK_THREAD_BINDING=CPU_CORE`;并宣称"L3 命中率 > 50%""跨 node 通信低至 3%" |
| 本次实测(见 §43) | 192 → 128 线程单调变好,但**我当时没有按"每 CCD 4–5 核"这条规则去定标**,只扫到 128 就停了 |

**本机取整:4 核/CCD = 96 线程,5 核/CCD = 120 线程。** 据此把默认值定为
`n_ccd × 5`(本机 120),并把 `scripts/tune_serve.sh` 默认改为 120;
不要再把线程数往上加(192/168/128 都是在抢 L3 与散热预算)。

### 44.1 与 lk_moe 的同模式端到端 A/B(2026-09-11,同 fork/同卡/同线程/无投机/无 CUDA graph)

用他们的 `process_data/scripts/serve_lkmoe_dsv4.sh`(`XIAOTU_MOE_BACKEND=lk`,TP=1、
max-model-len 8192、max-num-seqs 8、GPU_UTIL 0.5、`LK_THREADS=168`、`cudagraph NONE`、
无投机),与我们**完全同参数**的服务对照:

| 口径 | lk_moe(官方脚本) | 我们的引擎(同参数) | 比值 |
|---|---|---|---|
| C=1 tok/s | 6.52 | **9.53** | **1.46×** |
| C=1 每步 | 120.8 ms | **83.0 ms** | 1.46× |
| C=2 合计(单路) | 13.98(6.99) | **17.96(8.98)** | 1.28× |
| 每层 period | ≈2.81 ms(120.8/43) | **1.88–1.96 ms** | 1.45× |
| 每层 compute / rest | — | 0.76–1.12 / 0.76–1.19 ms | — |

⇒ **在完全相同的调度器/参数/硬件下,我们的 CPU MoE 引擎现在比 lk_moe 快 ~1.45×**
(本轮修复之前的历史 A/B 是他们快 1.6×)。差异主要来自引擎本身(compute)与
host 往返/rest,而不是外层编排(vLLM 调度器两边相同)。

### 44.2 硬数据:4–5 核/CCD 已把"内核能用的带宽"拿满(再加核反而更慢)

微基准(B=6、DEDUP=0 ⇒ na=32、每次调用读 403 MB 专家权重;A+B 为两段权重读取阶段):

| 线程数(=每 CCD) | A+B 用时 | 实测带宽 | 每节点 |
|---|---|---|---|
| **96(4 核/CCD)** | **1320 µs** | **305 GB/s** | 38 GB/s |
| **120(5 核/CCD)** | **1341 µs** | **301 GB/s** | 38 GB/s |
| 192(8 核/CCD) | 1604 µs | 251 GB/s | 31 GB/s |

- **4 核与 5 核/CCD 等价,8 核/CCD 反而慢 18%** ⇒ 完全印证用户规则(多加核不涨带宽,
  反而抢 L3 与散热/power 预算);
- 同时它把"剩下的差距"定性清楚了:机器纯流式上限 **783 GB/s**(8 节点 × 24 线程、NUMA 本地),
  而我们**内核只能吃掉 305 GB/s(39%)**——差的这部分是**每字节指令数**(FP4 解码 + FMA),
  不是线程数。⇒ 下一轮的正事是**减少每字节指令数**(AVX512-BF16 `vdpbf16ps` 重写解码+FMA),
  而不是加核。

### 44.3 口径校准:服务 vs 微基准真实差 1.26×(我先前说的 2.1× 是口径错误)

服务内真实路由形状是 **36 个 assignment 落在 ~18 个专家上(me≈2)**;我先前拿
`DEDUP=0`(na=32、403 MB)的微基准去比服务内(na=18、227 MB),字节数不同,结论作废。

同形状对照(同一时间窗口,120 线程):

| 口径 | setup | A | B | C | 合计 |
|---|---|---|---|---|---|
| 微基准 `DEDUP=18`(na=16) | 234 | 846 | 511 | 80 | **1670 µs** |
| 服务内 `[NS-PROF] M3-8 na≈17-19` | 68 | 1250 | 700 | 87 | **2105 µs** |

⇒ 服务/微基准 = **1.26×**(原因是服务里 GPU 侧与 torch/CUDA 线程并行、以及环境差异)。

### 44.4 在"每 CCD 4–5 核"前提下,FMA 指令数就是天花板(定量)

- 我们 A+B 的实际吞吐:**104–149 GB/s**(na=16-32、me=1.1-2),而机器可流式 **783 GB/s**
  ⇒ 只用掉 **13–20%**;
- 把"去掉解码"的诊断下限折算成指令数:na=32 时每节点 16k 行 × 128 组 × 6 实例 × 5 条
  指令 ≈ **63M 条/次调用**;4–5 核/CCD ⇒ 12–15 worker/节点 × 3 GHz × ~2 IPC ≈ **0.87 ms**
  —— 与实测的下限(~1.0 ms)**吻合**。
- ⇒ 结论:在每 CCD 4–5 核的约束下,**不可能靠加核换带宽**(已实测 8 核/CCD 反而慢 18%),
  唯一出路是**减少每条 MAC 的指令数**:AVX512-BF16 的 `vdpbf16ps`(一条指令 32 个 MAC,
  且激活本来就是 bf16、不必转 fp32;权重解码可共享)→ 理论上把 A+B 的指令数砍 40-50%。

### 44.5 回到 lk_moe 的代码级对比(同模式端到端)

同 fork/同卡/同线程/无投机/无 CUDA graph(他们的 `serve_lkmoe_dsv4.sh` vs 我们同参数服务):

| 口径 | lk_moe | 我们 | 比值 |
|---|---|---|---|
| C=1 tok/s | 6.52 | **9.53** | **1.46×** |
| 每步 | 120.8 ms | **83.0 ms** | 1.46× |
| C=2 合计(单路) | 13.98(6.99) | **17.96(8.98)** | 1.28× |
| 每层 period | ≈2.81 ms | **1.88–1.96 ms** | 1.45× |

⇒ "是计算模块强还是编排强"这个问题的答案:**编排两边是同一个 vLLM 调度器**,
差异全在 CPU MoE 引擎;本轮修复后**我们已反超 1.45×**(历史上是他们快 1.6×)。
但两者的**绝对效率都离机器上限很远**(见 44.4),所以这块仍有共同的挖掘空间。

### 44.6 实验:AVX512-BF16 `vdpbf16ps` 重写解码+FMA —— 只有 1.10×,且精度不达标 ⇒ 默认关闭

按"减少每字节指令数"的思路实现了 `XIAOTU_MOE_DPBF16` 路径(`moe_v2_packed4.hpp`,
AVX512-BF16 变体内):权重解码一次喂 M 行、结果以 bf16 留寄存器、激活直接用模型自带的
bf16(不再转 fp32)、每 (行,组) 用 1 条 `vdpbf16ps`(32 MAC)+ 1 条 fma 叠 e8m0 scale。

实测(同二进制、环境变量切换、交错 2 轮、DEDUP=18 = 服务内真实形状、120 线程):

| 口径 | setup | A | B | C | TOTAL |
|---|---|---|---|---|---|
| fp32(默认) | 259 | 872 | 553 | 96 | 1780–1791 µs |
| dpbf16 | 239 | 818 | 470 | 89 | **1611–1621 µs** |

⇒ **只有 1.10×**(A 1.07×、B 1.18×),远低于"指令数砍半"的预期 ⇒ 说明 A/B 两阶段
**不是 FMA 指令受限**(否则应接近 1.5-1.7×),而更像受**每线程工作量/访存层级**限制。

精度:受控对拍 8 档里 me=2 那档 max_rel **8.4e-3 > 项目门限 2e-3**(fp32 路径同档 3.9e-4),
其余 7 档 ~2e-4…9e-4 ✅ ⇒ **收益不足以为精度买单,已把默认值改为关**(=1 可开启做实验),
代码与开关保留。

**下一步该往哪查**(取代"砍指令数"这个已被证伪的假设):在 4-5 核/CCD 下每节点只有
12-15 个 worker,每个 worker 一次调用要处理 512 行 × 128 组 ⇒ 单 worker 工作量约
1.6M 条指令(≈0.27 ms @2 IPC),与实测 A(0.8 ms)同量级但仍有 3× 缺口 ⇒ 需要真正
的硬件计数器(本机 `perf_event_paranoid=4` 被禁)或更细的内部分段计时来定位。

### 44.7 下一个靶子(定量):**每核只跑到 2.5 GB/s,而单核顺序流式能到 36 GB/s**

- 机器实测(本机,`numactl` 本地内存):**单核 36 GB/s**、单节点 24 线程 98.7 GB/s、整机 783 GB/s;
- 我们的内核(120 线程 = 15 worker/节点):A+B 202 MB / 1290 µs = 157 GB/s 整机
  = **每节点 ~20 GB/s = 每核 2.5 GB/s**(约单核能力的 **7%**);
- 而 `vdpbf16ps` 实验证明"砍 FMA 指令数"只值 1.10× ⇒ **不是 FMA 受限**;
- ⇒ 剩下的解释只能是**每核的访存并行度(MLP)不足**:每个 worker 顺序扫 512 行 ×
  2048 B 的权重切片,行内是顺序读,但当前只有"下一行前 4 条 cache line"的预取
  (`_mm_prefetch(nr+{0,64,128,192})`),对 2048 B 的行来说预取深度远远不够,
  且每行处理完才进入下一行 ⇒ 有效未完成 miss 数很低。

**下一轮的具体做法(按性价比)**:
1. 让每个 worker 同时处理 **2–4 行**(行间交错),使独立访存流 ×2–4;
2. 预取改为覆盖整行(2048 B/行 = 32 条 cache line,按 4-8 条/次分批)并提前 1-2 行;
3. 若要精确归因,需要硬件计数器:本机 `perf_event_paranoid=4` 被禁,可考虑
   用 rdpmc 自采或找一台 perf 可用的机器做一次对照。

---

## 45. 用 standalone 微基准把"引擎结构 vs 内层循环"分开(2026-09-11)

### 45.1 方法

写了一个独立 C 程序(`/tmp/decbench.c`,已固化要点到本节),在同一台机、同一 NUMA 绑定下
只跑**内层循环本身**,四个模式读同一份 2048 B/行的权重数据:

| 模式 | 单核带宽 |
|---|---|
| `raw` 纯流式读(64 B/次) | **28 GB/s** |
| `decode` 完整 FP4 解码(2×PSHUFB + 2×UNPACK + insert + cvtepu16 + slli) | **5.6 GB/s** |
| `half` 只留 2×PSHUFB(shuffle 数减半) | 7.5 GB/s(仅 1.17×) |
| `decode+act2` 解码 + M=2 激活载入 + FMA(**与引擎内层同构**) | **5.0 GB/s** |

⇒ 解码把单核从 28 压到 5.6 GB/s,但**再砍一半 shuffle 指令只值 1.17×** ⇒ 不是 shuffle
端口受限,是"16 B/组的载荷 + 依赖链"本身的性质。

### 45.2 引擎结构确实还差 3.5×

- 引擎内层循环 standalone = **5.0 GB/s/核**;
- 引擎实际 = **1.15–2.2 GB/s/核**(A+B 202 MB / 1.45 ms / 120 线程);
- ⇒ 差距(2.3–3.5×)**不在内层循环,而在引擎结构**:每层 3 个阶段 + 屏障,而默认
  job 粒度是 `ceil(tpn/na)` ≈ **1.2 job/worker** —— 一个掉队 job(被其他租户线程/
  GPU host-fn 调用线程挤掉)就会拖慢整个阶段。

### 45.3 已落地修复:job 过度分片(每 worker ≈ 4 个 job)⇒ 1.23×

改 `moe_v2.hpp` 的 `need = ceil(4·tpn/na)`(下限 2,上限仍受 `spanA/32` 约束)。

| 口径(微基准,DEDUP=18,120 线程,2 轮一致) | A+B | 带宽 | 每核 |
|---|---|---|---|
| 旧默认(auto=1) | 1448–1469 µs | 138–140 GB/s | 1.15–1.16 GB/s |
| subA=2 | 1213–1217 | 166–167 | 1.38–1.39 |
| **subA=4(新默认)** | **1166–1199** | **168–173** | **1.40–1.44** |
| subA=8 | 1174–1236 | 163–172 | 1.36–1.43 |

新默认实测 1175 µs(1.43 GB/s/核),与强制 SHARDSPLIT=4 一致、比强制 =1(旧行为)快 **1.23×**;
数值对拍(`test_block23_equiv.py` 8/8、`test_swiglu_clamp_mxfp4.py`)全部通过。

### 45.4 下一步(结构性的,收益最大)

既然内层循环单核能跑 5 GB/s、而引擎只到 1.4 GB/s,剩下的差距主要是**每层 3 个阶段之间的
屏障/排空**。可做的方向(按收益):
1. **按专家分片(而不是按行分片)**:同一个专家在**一个节点内**完成 gate/up→SiLU→down
   的全流程(权重只读一次、三段之间不需要跨节点屏障),只有最后对 token 输出做一次
   跨节点归约 ⇒ 每层从 3 次全局屏障降到 1 次(代价:路由分布不均,需要按活跃专家数
   做负载均衡策略,或用动态任务窃取);
2. 把 C 阶段(加权归约)并进 B 阶段的收尾(减少一次屏障);
3. 若要精确到指令/停顿级,需要硬件计数器(本机 `perf_event_paranoid=4` 被禁)。

### 45.5 固定开销的第二次定位:每阶段 ~176 µs(与线程数/job 数无关)

用上节的诊断构建(空转,只留 job 分解+屏障)标定,并在不同线程数与分片数下扫:

| threads | subA | A(空转) | B(空转) | C | 合计 |
|---|---|---|---|---|---|
| 120 | 1 | 194 | 168 | 73 | 474 µs |
| 120 | 4 | 247 | 200 | 75 | 561 |
| 120 | 8 | 331 | 283 | 77 | 729 |
| 48 | 4 | 198 | 192 | 55 | 479 |
| 24 | 4 | 231 | 234 | 49 | 555 |

拟合 ⇒ **每阶段固定 ~176 µs + 每 job ≈1 µs**;固定部分与线程数(24/48/120)几乎无关。
正常调用里 A+B+C+setup = 1494 µs,其中固定 614 µs(**41%**)⇒ 三阶段屏障是最大的单块成本。

### 45.6 已落地:无锁任务发布(≈4%)

worker 原先在拿任务时要 `local_task = task_; stask = sharded_task_;`(全局锁 + `std::function`
拷贝/可能的堆分配)。改成发布 **(fn 指针, ctx)** 两个机器字(模板 thunk),worker 确认新一代后
直接读,不加锁不拷贝。实测(交错 2 轮):TOTAL 1577/1609 → **1522/1523**,C 阶段 111→69 µs,
A/B 基本不变。

### 45.7 试过但**挂死**回退的:worker 快照也去掉锁

进一步把 worker 的整个快照(gen/n/start/任务指针)改成无锁"先读 gen、读字段、再复检 gen"
(依据代码原有的 publish-first 约定)。结果:**直接挂死**(两次 600 s 超时未完成)——
正是这个池历史上踩过的丢唤醒/快照竞态(§33)。⇒ 已回退,回退后二进制与 `pool_lf` 逐字节一致,
并复测正常(1477 µs)。

**结论**:~176 µs/阶段的固定成本不能靠"去掉锁"消除(去掉锁会挂),只能靠**减少阶段数**或
改造屏障本身(例如按节点分组的两级屏障、或按专家分片让三段在一节点内完成)。

### 45.8 服务端验证(本轮内核:job 过度分片 + 无锁任务发布,120 线程)

| 配置 | C=1 | 每步 | tokens/步 | C=2 合计(单路) | 每层 compute / rest |
|---|---|---|---|---|---|
| 本轮 | **12.92 tok/s** | 157.9 ms | 2.95 | 19.93(9.97) | 1.90–2.14 / 0.92–1.10 ms |

- C=1 12.92 是本会话单路最好值(此前 12.66/128 线程、12.12/120 线程);
- 每步 157.9 ms ⇒ TPOT 53.8 ms(目标 40 ms);
- 剩余结构:每层 3 次全局屏障的固定成本 ~528 µs(35%),只能靠减少阶段数或按专家分片解决。

### 45.9 又一次口径教训:空转诊断的"固定开销"**不与实际工作叠加**

写了一个只含 `numa_pool.hpp` 的 standalone 线程池基准(`/tmp/poolbench.cpp`,已记录做法),
直接测"纯派发+屏障"(空 body、真实 job 数):

| jobs/节点 | nodes | threads | 每阶段 |
|---|---|---|---|
| 18 | 8 | 120 | 158.8 µs |
| 72 | 8 | 120 | 238.3 µs(≈3.3 µs/job) |
| 144 | 8 | 120 | 359.2 µs |
| 72 | 1 | 120 | 116.5 µs |
| 72 | 8 | 24 | 323.3 µs(线程越少越慢:每 worker 的串行 job 更多) |

⇒ 纯派发成本主要是**每 job 的全局原子**(`remaining_.fetch_sub` + 未门控的 `shard_exec_`)。
据此实现了**分层屏障**(每节点先递减本地计数,归零者再递减全局一次):standalone 里
**1.79×(72 job/节点)、2.33×(144)**,漂亮的数字。

**但在引擎里毫无收益**(交错 A/B:TOTAL 1445/1448 → 1479/1499,反而略差)。
解释:真实调用里 **job 的完成是被工作主导的,原子更新在其他 job 干活时并行完成**,
不在关键路径上;空 body 时才成为唯一成本。⇒ 已按"未证明收益不留"回退(回退后二进制与
`pool_lf` 逐字节一致)。

### 45.10 每步非层开销(28 ms)的来源已定位

| 模式 | 每步 | 43×period | 非层开销 |
|---|---|---|---|
| 无投机(qlen=1,同模式 A/B) | 83.0 ms | 82 ms | **~1 ms** |
| 有投机 k=5(qlen=6) | 157.9 ms | 130 ms | **~28 ms** |

⇒ **28 ms/步几乎全是 DSpark draft 模型的前向 + 6 个 token 的采样/验证**,
不是调度器开销(调度/流式在无投机时只有 ~1 ms)。这也解释了为什么 `--max-num-seqs`、
graph 模式之类的调度侧旋钮对我们帮助有限。

---

## 46. 与 lk-moe 生态的横向对照(2026-09-11,按用户要求搜索其文档/发布说明)

搜索到的官方资料([LvLLM README](https://raw.githubusercontent.com/guqiong96/Lvllm/main/README.md)、
[Lvllmds4-x README](https://raw.githubusercontent.com/guqiong96/Lvllmds4-x/main/README.md)、
GitHub Releases API for Lvllm / Lvllmds4 / Lvllmds4-x / Lsglang):

| 模型 | 机器 | 预填充 | 解码 | 投机解码 | 来源 |
|---|---|---|---|---|---|
| **DS-V4-Flash-0731**(我们同一个) | EPYC 7642×2 + 16ch DDR4-3200 + **3090×2** | 1060 t/s | **26 t/s** | **35–47 t/s** | Lvllmds4-x README |
| DS-V4-Flash-0731 | EPYC 9684X×2 + 24ch DDR5-4800 + PRO 6000 | 3100 | 75 | 100–115 | 同上 |
| DS-V4-Flash-0731(无投机) | 同上 7642 机器 + **5060Ti×2** | — | **25 → 28 t/s**(v2.3.10 "decode +10~15%") | 关 | Lvllm v2.3.10 release notes |
| **DS-V4.1-Flash**(316 GB,比我们模型大一倍) | EPYC 7642×2 + 16ch DDR4-3200 + **3090×2** | — | **~30 t/s** | 有(c128 target-verify CG) | Lsglang v1.5.2 |
| 用户给的图:DS-V4.1-Flash | 未标注(显然是更强的机器) | 1280→14733(2048 起平台 ~12-13k) | **131.4**(128)→130.5(65536)→**118**(524288) | 可能开 | 用户提供 |

### 46.1 从那张图能读出的隐含信息

1. **解码与上下文长度几乎无关**:128→65536 输入只掉 0.7%,524288 才掉 10% ⇒
   每 token 成本由**常数项(专家计算)**主导,稀疏注意力/indexer 在 512K 上下文下
   几乎不增加成本;
2. **预填充在 2048 输入处跳变**(1280 → 3506 → 14733 t/s)⇒ 对应
   `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=1024`,即 **≥1024 token 才切到 GPU 预填充**;
3. 那张图的**机器规格没有标注**,不能直接拿来对标本机;同族 release notes 里
   同款模型在 2×3090+DDR4-3200 上是 **26–30 t/s**,这才是与我们的可比口径。

### 46.2 我们与"2×3090 + DDR4-3200"的可比口径对照

| 指标 | lk-moe(2×3090 + DDR4-3200,16ch) | 我们(A100×3 + DDR5-4800, 192 核) |
|---|---|---|
| 解码(单路,无投机) | 25–28 t/s | ~9.5–13(mode 相关) |
| 解码(单路,投机) | 35–47 t/s | 13.2(客户端口径)/ ~19–20(每 token 延迟口径) |
| 预填充 | 1060 t/s @32768 | **668 t/s @4096**(实测) |

⇒ **我们的硬件全面更好(DDR5-4800 24ch vs DDR4-3200 16ch;A100 40GB×3 vs 3090 24GB×2),
但预填充/解码都落后约 1.5–2.5×** ⇒ 差距在实现,不在硬件(与用户判断一致)。

### 46.3 他们文档里我们**还没用上**的旋钮(按预期收益排序)

| 旋钮 | 他们怎么用 | 我们的对应物 | 状态 |
|---|---|---|---|
| **`LVLLM_GPU_RESIDENT_MOE_LAYERS`** | PRO 6000 上 `"0-13,43-45"`(14 层专家 + **draft 模型 43-45 层**常驻显存)⇒ "Faster Prefill & Decode" | `XIAOTU_MOE_GPU_RESIDENT_LAYERS`(插件已实现,未启用) | ❌ 未用 |
| `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=1024` + `LVLLM_GPU_PREFETCH_WINDOW=1` | ≥1024 token 切 GPU 预填充,并预取下一层权重与计算重叠 | `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS` + `XIAOTU_GPU_PREFETCH_AHEAD` | 🟡 阈值 384,预取默认开 |
| `VLLM_USE_V2_MODEL_RUNNER=1` | 最新 quick start 默认带 | 未设 | ❌ 未试 |
| `num_speculative_tokens` 5→3 | v2.3.11 明确写"更平滑的解码" | 我们固定 5(与训练块长一致) | 🟡 可 A/B |
| `--max-num-seqs 2` / `--max-num-batched-tokens 4096` / `OMP_NUM_THREADS=1` | 官方 quick start | 已对齐 | ✅ |
| `LK_THREADS=48`(=核数/GPU 数) | 官方公式 | 我们按"每 CCD 4-5 核"=120 | 🟡 两种口径都试过,120 更好 |

### 46.4 我们的预填充路径是要害(实测 668 t/s @4096)

`gpu_prefill.gpu_moe_layer()` 的做法是**每层把该层的专家权重经 PCIe 流进显存**再算:
43 层 × ~3.2 GB ≈ **139 GB / 次预填充**,PCIe ~25 GB/s ⇒ 约 5.6 s —— 与我们实测的
6.1 s(TTFT @4096)吻合。⇒ 我们的"GPU 预填充"其实是 **PCIe 带宽受限**,不是 GPU 算力受限;
而 lk-moe 的做法是**让一部分层的专家常驻显存**(0-13 → 14 层不流),其余层用 CPU 引擎,
并用 prefetch window 把下一层 H2D 与当前层计算重叠。

### 46.5 目标修正(用户 2026-09-11)与关键定性结论

**新目标**:预填充 ≥1500 t/s、解码 ≥70 t/s、开投机 ≥100 t/s、1M 可用。
(对照 PRO 6000 那台:3100 / 75 / 100-115;GPU 不同,预填充不全对标。3090+DDR4 那台
1060 / 26 / 35-47 因内存带宽与 CPU 都远低于本机,不作对标。)

**预填充实测(4096 token 提示,runtime 阈值 A/B)**:

| GPU 预填充阈值 | 端到端 | 预填充吞吐 |
|---|---|---|
| 999999(全部走 CPU 引擎) | 7.11 s | 576 t/s |
| 384(混合) | 6.51 s | 629 t/s |
| 0(全部走 GPU 流式) | 6.10 s | **671 t/s** |

两条路都远低于目标:
- **GPU 路**:`gpu_moe_layer()` 是"每层把该层专家权重经 PCIe 流进显存"⇒ 43×3.2 GB ≈ 139 GB /
  次预填充,PCIe ~25 GB/s ⇒ 5.6 s —— 与实测 6.1 s 吻合,**PCIe 带宽受限**。
  对方的做法(README/release notes)是 `LVLLM_GPU_RESIDENT_MOE_LAYERS="0-13,43-45"`
  —— **让一部分层的专家常驻显存**,不流权重;我们插件有对应的
  `XIAOTU_MOE_GPU_RESIDENT_LAYERS`,但从未启用。
- **CPU 路**:576 t/s = 每层 165 ms(4096 token 批),即每层只搬 3.2 GB/165 ms ≈ **19 GB/s**,
  仅为机器带宽的 2.4% ⇒ 我们的 packed4 内核是**GEMV 结构**(权重驻 L1、**每个输出行都重读
  一遍激活**),大批量时被激活带宽压死。**预填充要达标必须把它做成真正的 GEMM**
  (激活驻留寄存器、权重流式,并在 M 方向做寄存器分块)。

### 46.6 常驻专家层机制已验证;单卡放不下,必须走 TP=2

- `XIAOTU_MOE_GPU_RESIDENT_LAYERS=0-1` 实测可用,日志:
  `[xiaotu] GPU-resident model.layers.0/1.ffn: 3.19 GiB on cuda:0 (tp=1, experts=256)` ✓
  (每层专家 3.19 GiB,与他们 `LVLLM_GPU_RESIDENT_MOE_LAYERS` 同一机制);
- 但**单卡 256K 配置下没有空间**:GPU_UTIL 0.85 × 40 GB = 34 GB,其中 GPU 侧权重 ~26 GB +
  KV 8 GiB ⇒ 富余 < 3.2 GB(=1 层);把 KV 缩小会被 vLLM 的 `max_model_len` 校验拦住
  (它按 16K→6.28 GiB 的保守口径算,比实测 29.5 KB/token 严得多)。
- ⇒ **常驻层必须配 TP=2**:每 rank GPU 权重 ~13 GB + KV 8 GiB ⇒ 富余 ~19 GB ⇒
  可常驻 **~12 层**(每层每 rank 1.6 GiB,即 43 层里的 28%)。这正是下一轮的主实验,
  也是对方 PRO 6000 那台(单卡 96 GB 放 14 层)的对应做法。

### 46.7 下一轮计划(按目标 prefill 1500 / decode 70 / spec 100)

| 优先级 | 工作 | 预期 | 依据 |
|---|---|---|---|
| P0 | **预填充改成真正的 GEMM**(激活驻留寄存器 + M 方向寄存器分块,权重流式) | CPU 预填充 576 → 数千 t/s | 实测每层只搬 19 GB/s(激活每行重读),是 GEMV 结构问题 |
| P0 | **TP=2 + `XIAOTU_MOE_GPU_RESIDENT_LAYERS=0-11`** | decode 计算量 -28%,预填充少流 12×3.2 GB 的 PCIe | 对方同款做法;单卡无空间 |
| P1 | 解码引擎 CPU 每层 2.0 → 1.1 ms(对方 9684X 同级的水平) | decode 再 +1.5–2× | 每层成本对照:我们 1.9–2.4 ms vs 对方反推 ~1.1 ms |
| P1 | GPU 预填充的 PCIe 流式:检查是否可只流"活跃专家"(4096 token 时全活跃,无收益)或改成分块流水 | — | 139 GB/次 ≈ 5.6 s 已是 PCIe 极限 |
| P2 | `VLLM_USE_V2_MODEL_RUNNER=1`、`num_speculative_tokens 5→3` | 未知,便宜可试 | 对方 quick start 默认 |

---

## 47. 预填充内核 GEMM 化(1.22× 预填充 / 1.15× 解码,已设为默认)

### 47.1 问题:packed4 内核是 GEMV 结构

原结构 `for j(输出行): for g(K 组): 载入该行全部激活` ⇒ **每个输出行都把整条激活重读一遍**,
激活:权重流量 = 64:1;大批量(预填充)时被激活带宽压死(实测每层只到 19 GB/s)。

### 47.2 改法:让激活在寄存器里被多个输出行复用

新增 GEMM 分块路径:外层按 (MR=4 token × NR 输出行) 分块,**每个 K 组的激活只载入一次**
(av[4][2] = 8 个 zmm),内层对 NR 个输出行各自解码权重后复用同一组激活,
每 (行,token) 仍按"每组 mul+2×fma 再累加"的同一顺序 ⇒ 数值与单行路径同序。
寄存器预算:acc[4][NR] + av[4][2] + wlo/whi/sv/d;NR=8 时 32+8+4 ⇒ 有溢出但实测最快。

### 47.3 实测(微基准,120 线程)

| 形状 | 旧(GEMV) | NR=4 | **NR=8(新默认)** |
|---|---|---|---|
| 预填充 B=96/DEDUP=8(每专家 72 token) | 15.51 ms/层 | 13.76(1.13×) | **12.72 ms(1.22×)** |
| 解码 B=6/DEDUP=18(me≈2) | 1.57 ms/层 | 1.45(1.08×) | **1.36 ms(1.15×)** |

数值对拍 `scripts/test_block23_equiv.py` 8/8 通过(me=1…7 全过)。
`XIAOTU_MOE_GEMM_NR=0` 可回退旧路径;静态指令统计不再用于对比(见 46.7)。

### 47.4 仍未达标的部分(诚实记录)

预填充目标 1500 t/s,当前 ~600;微基准显示内核**仍比自己的指令数估算慢 ~8×**
(每 worker 每层只搬 560 KB 权重却要 1.59 ms ⇒ 单线程 ~0.35 GB/s),说明瓶颈是
**每 worker 的访存并行度/延迟**而不是带宽或指令数。⇒ 下一步:在 GEMM 分块内加
**跨行的权重预取**(把下一 tile 的 NR 行权重按 K 组分批预取),以及试 (MR=2,NR=16) 等组合。

### 47.5 服务端复测与预填充的真实瓶颈

| 口径(4096 token 预填充) | 改前 | 改后(GEMM 默认 NR=8) |
|---|---|---|
| CPU 引擎路径(阈值 999999) | 576 t/s | **614 t/s** |
| GPU 流式路径(阈值 0) | 671 t/s | 616 t/s(噪声内持平) |

⇒ 内核改进只在 CPU 路径体现(+6.6%,低于微基准的 1.22×,说明服务里还有别的固定成本);
GPU 路径是 PCIe 25 GB/s 流 139 GB,内核改进对它无效。

**预填充的算术**:一次完整预填充必须读遍 **全部 256 专家 × 43 层 ≈ 139 GB 权重**。
目标 1500 t/s @4096 token = 每 pass 2.7 s ⇒ 需要 **≥51 GB/s** 的有效带宽 —— 这比我们
解码已经达到的 140–300 GB/s 还低 ⇒ **目标在带宽上是可达的**,差距在:
① 现在只有 ~21 GB/s(614 t/s);② 可能被调度器分成多个 chunk(每 chunk 都要重读 139 GB)。
⇒ 下一轮:开 `XIAOTU_MOE_PROFILE` 量预填充单次调用(M、na、A/B/C 与达成带宽),
并检查 chunk 数与 `--max-num-batched-tokens` 的配合;必要时让预填充走"一次读完"的路径。

### 47.6 【重要更正】预填充路径的真实数据 + 一个真 bug

**Bug**:`scripts/tune_serve.sh` 导出 `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE`,而
`gpu_prefill.py` 只读 `XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE`(少 `VLLM_` 前缀)⇒
**运行时阈值文件从未生效**,我 §47.5 那两次"CPU vs GPU 预填充 A/B"其实两条腿都走了
GPU 路(因为 `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=384` 仍生效)。已修(两种拼写都接受)。

**修正后的预填充实测(4096 token)**:

| 路径 | 吞吐 | 说明 |
|---|---|---|
| GPU 流式(`gpu_moe_layer`,阈值 384) | **586–671 t/s** | 6.99 s 里 PCIe 流 139 GB 占 **5.4 s(77%)**、GPU MoE 计算 1.5 s、注意力 0.097 s |
| CPU 引擎(阈值 999999,修 bug 后) | **19 t/s(214 s)** | 完全不可用 |

CPU 预填充为什么这么慢(引擎 NS-PROF 分桶):
`bucket=M>8 na≈13-39 | setup=170–340 ms A=480–785 ms B=170–394 ms → 每次调用 0.8–1.4 s`
⇒ **setup(数据搬运)就是主项**:每层要 gather 24576 个 assignment × 4096×2 B ≈ **201 MB**
的激活,加上 `a32` 的 fp32 转换 ≈ **340 MB/层**,而 MoE 的权重读取才 3.2 GB/层。
⇒ CPU 预填充路径需要先把"gather + fp32 转换"改成按需/分块,否则永远不可用。

**结论(预填充的正确方向)**:
1. 预填充继续用 **GPU 流式路径**(比 CPU 快 30×);
2. 要往 1500 t/s 走,必须减少 PCIe 字节:**TP=2 让两条 PCIe 链路并行各流一半 ⇒ ≈2×**
   (再加上若干层常驻显存不流)⇒ 这正好也是解码需要的配置;
3. CPU 预填充路径若要用,必须重写 gather/转换(记录为后续项)。

### 47.7 本轮(第 12 轮)小结与下一步

**已落地**:
1. 内核 **GEMM 化**(激活寄存器复用):微基准预填充 1.22×、解码 1.15×;数值对拍 8/8;
   已设默认(`XIAOTU_MOE_GEMM_NR=8`,=0 回退)。
2. 修掉阈值文件环境变量名的 bug(运行时切换预填充路径现在真的生效)。
3. 量化了预填充的三段构成(PCIe 77% / GPU 计算 21% / 注意力 1.4%)与 CPU 路不可用的原因。

**当前最好配置**(单卡 256K + DSpark k=5 + CUDA graph + GEMM 内核 + 120 线程):
预填充 586–671 t/s、解码 ~11–13 t/s(客户端口径)。

**下一步(P0)**:TP=2 + `XIAOTU_MOE_GPU_RESIDENT_LAYERS`
- 预填充:两条 PCIe 链路并行各流一半 ⇒ ≈2×;再加常驻层减少流字节 ⇒ 目标 1500 可达;
- 解码:常驻 ~12 层(28%)+ 每层 CPU 成本下降 ⇒ 向 70 靠近;
- 需要用他们的启动参数对齐:`--disable-custom-all-reduce`、`LVLLM_GPU_PREFETCH_WINDOW=1`
  (我们 `XIAOTU_GPU_PREFETCH_AHEAD=1` 默认开)、`--max-num-seqs 2` ✓。

---

## 48. TP=2 启动阻塞(第 13 轮遇到,必须解决才能走"常驻层 + 双 PCIe"路线)

### 48.1 现象与证据(3 次尝试,症状一致)

| 尝试 | 配置 | 结果 |
|---|---|---|
| 1 | TP=2, EP=1, 常驻 0-11, spin=300, KV 8 GiB | 常驻层加载成功(`GPU-resident` 16 次),随后 **Triton CUDA OOM**(显存不够)|
| 2 | TP=2, EP=1, 常驻 0-3, spin=300, KV 8 GiB | 引擎逐层建成(`EP ... rank 0/2 owns experts [0,128)`),随后 **挂死**:`shm_broadcast: No available shared memory broadcast block found in 60 seconds` 反复出现 |
| 3 | TP=2, EP=1, 常驻 0-3, **spin=5000**, KV 8 GiB | 同上,仍挂 |
| 4 | TP=2, EP=1, **XIAOTU_MOE_GEMM_NR=0(旧 GEMV 内核)**, spin 默认 | 同上,仍挂 ⇒ **不是本轮 GEMM 改动引入的** |

⇒ 阻塞点在 **EngineCore 的 warmup 阶段**(引擎已建成、模型已加载),不是加载/显存问题。
本会话早期(第 1-2 轮)TP=2+EP 是**能起来的**,此后引擎侧改过三处(自旋默认 5000→300、
线程预留、无锁任务发布、job 过度分片),需要在下一轮用 **bisect** 定位:

```bash
# 依次回退并各试一次 TP=2 启动(每次 ~10 min)
git show c2172ea^:xiaotu_moe/csrc/moe/numa_pool.hpp > /tmp/pool_orig.hpp   # 无锁发布之前
# 1) 只回退无锁任务发布  2) 只回退 job 过度分片(moe_v2.hpp 的 need 公式)
#  3) 回退自旋默认(插件里 setdefault 300 → 5000)
```

最可疑的是**无锁任务发布**(worker 不再在 `work_mtx_` 下取快照 —— 与 §33 记录的
"池自身竞态"同类);第 4 次尝试已排除 GEMM 内核。

### 48.2 顺带确认的事

- **GPU 常驻层在 TP=2 下能加载**:`XIAOTU_MOE_GPU_RESIDENT_LAYERS=0-3` 正常
  (每层每 rank ~1.6 GiB);显存预算要算上 KV(vLLM 对 131072 上下文按 6.71 GiB 校验,
  比实测 29.5 KB/token 严 2×)。
- 多 rank 共机时**自旋窗必须更大**(spin=300 在 TP=1 是甜的,TP=2 下连启动都过不去)——
  与 §31 "两个 rank 共机时自旋把机器烧穿"是同一类问题,但这次是唤醒不足而非自旋过多。

---

## 49. 【已解决】TP=2 启动挂死的真因:EP 屏障的 **/dev/shm 残留文件**

### 49.1 症状与排除过程(第 13 轮的"阻塞")

症状:TP=2 时引擎逐层建成后**永远不就绪**,EngineCore 反复打印
`shm_broadcast: No available shared memory broadcast block found in 60 seconds`
(该消息的语义是"引擎卡住",不是 shm 空间不足;/dev/shm 只用了 1%)。

逐项排除(每次一次完整启动测试):
| 假设 | 实验 | 结论 |
|---|---|---|
| 本轮 GEMM 内核 | `XIAOTU_MOE_GEMM_NR=0` | 仍挂 ⇒ 排除 |
| 无锁任务发布 | 回退到 `c2172ea^` 的原始 `numa_pool.hpp` | 仍挂 ⇒ 排除 |
| 自旋窗太小 | `XIAOTU_MOE_SPIN_IDLE_US=5000` | 仍挂 ⇒ 排除 |
| CUDA graph 捕获 | `EAGER=1` | 仍挂 ⇒ 排除 |
| 我们池的死锁 | 引擎日志里 `WATCHDOG/STALL/SLOW parallel_for` 计数 = 0 | ⇒ 不是池 |

**真因**:被 kill 的 TP≥2 进程在 `/dev/shm` 留下 46 个 `xiaotu_ep_L*_4096_1024_2.bin`
(EP 双 barrier 的世代计数等状态)。新进程 attach 到状态错乱的旧文件 ⇒ 两个 rank 的
`gen`/`arrive` 永远对不上 ⇒ 永久互等。`rm -f /dev/shm/xiaotu_ep_*.bin` 后
**TP=2 在 250 s 内正常就绪**。

**已固化**:`scripts/tune_serve.sh` 启动前自动清理这些文件(带原因注释)。
**建议后续**:插件侧可在创建/attach 时做一次"世代握手"或用含 PID/nonce 的文件名,
从根上避免;当前先用启动前清理。

### 49.2 TP=2 实测(单机双卡,EP=1,eager)

| 口径 | TP=1 单卡 | **TP=2(EP=1)** | 说明 |
|---|---|---|---|
| 预填充 4096 token | 639 t/s | **761 t/s(+19%)** | 两条 PCIe 链路并行流权重 ✓ |
| 解码 C=1 | 10.81 | 9.95 | EP 每层跨 rank 合并在拖后腿 |
| 解码 C=2 合计 | 19.93 | 12.87 | 同上 |

⇒ 与本会话 §30 的结论一致:**EP 的 shm 双 barrier 每层要等 ~1-3 ms**。
要走"TP=2 + 常驻层"路线,必须先把这个合并代价压下去(否则解码反而退步)。

### 49.3 解码差距的定量结论(与 lk 的核心差距)

- 有效带宽对照:对方 9684X 那台(24ch DDR5-4800)**每层约 1.2 ms**(由 75 t/s 反推)
  ≈ **230 GB/s 聚合 / 48 线程 = 4.8 GB/s 每线程**;
- 我们的引擎:**每层 2.7 ms**(277 MB ⇒ **103 GB/s / 120 线程 = 0.86 GB/s 每线程**);
- 而我们**自己内层循环的 standalone 实测是 5 GB/s/核**(= 对方的水平!)。
⇒ **差距 100% 在引擎结构**(每层 3 阶段屏障 + 行分片导致的跨节点同步 + 每 worker 串行 job),
不是内核指令、不是内存带宽、不是线程数。
⇒ P0 结构性工作:**改成"整专家按节点分片"(expert-parallel per node)**,让
gate/up→SiLU→down 在同一节点内完成,每层只留 1 次跨节点归约(对方 `num_processes=ep_size`
就是这个思路;我们 backlog 的 T54)。

---

## 50. 【固定规则,不要再改】NUMA 分片是**唯一**权重布局;单拷贝开关已彻底删除

### 50.1 规则(用户明确要求,重复出现即视为回归)
1. 每个 CCD 开 **4-5 个核**(本机 24 CCD ⇒ `XIAOTU_MOE_THREADS=120`),不要再加核;
2. 权重**按 NUMA node 分片**(`nshard_ = numa_node_count() = 8`),每个 node 的
   worker 只读写 `MPOL_BIND` 到自己那份的内存 —— **全部 page-local**;
3. node 之间只交换**很小的数据**(每层各 node 的激活切片 + 每 token 部分和),
   走池内 all-gather + 归约,量级远小于跨 node 读权重的开销。

### 50.2 已删除的东西
- 引擎:`moe_v2.hpp` 里 `XIAOTU_MOE_SINGLECOPY` 的整段分支**删除**(不再是"默认值问题",
  而是代码里不存在该模式)。保留的只有"分片不可用(维度不整除/分配失败)"时的
  自动安全网(每 socket 一份副本)。
- 脚本/文档:`tune_serve.sh`、`serve_prod_8070.sh`、`tune_nsys.sh`、`fp8_*`、`tiny_moe_equiv.py`、
  `README*.md`、`docs/*` 里的全部引用已清掉。
- **教训**:`tune_serve.sh` 曾默认 `SINGLECOPY=1`(注释理由是"省内存",且引用了
  conc-4 的旧结论),导致交付配置重新退化成"单拷贝 + 全部线程读同一份内存",
  解码 A 阶段慢 ~30 倍。**任何"省内存"的理由都不能再推翻这条规则。**

### 50.3 证据(同一时间窗口轮流跑,`scripts/bench_cpu_engine.py`,DEDUP=12 / THREADS=120 / 负载 0.9)
| 布局 | B=6 ms/层 | B=18 ms/层 |
|---|---|---|
| 单拷贝(flat,交付配置实际用的) | 2.01 / 2.01 | 3.35 / 3.39 |
| **NUMA 分片(规则要求)** | **1.19 / 1.20** | **2.92 / 2.94** |

⇒ 解码尺寸(B=6,na=6)DEDUP 后每层 **1.67×**;B=18 时 1.15×。
(注:`MS-PROF` 给出的"每层 A"含首次 touch/绑页的开销,第一段 40 次调用会被冷启动
污染,别拿它当稳态;上面表里是**计时循环内**的 ms/层,不含 warmup。)

## 51. 分片区域**不要**用 THP:它把每 node 占用放大 5×,并在第 29 层把单 node 吃爆

### 51.1 症状
修复 §52 的 mbind 之后,TP=1 分片启动**仍然**在 "engine built model.layers.29" 处被杀:

```
oom-kill: constraint=CONSTRAINT_MEMORY_POLICY, nodemask=7, task=VLLM::EngineCor
Out of memory: Killed process 1044797 total-vm:944314492kB anon-rss:593333608kB
```

`nodemask=7` + 每 node 193 GB ⇒ 是被**绑定到 node 7** 的那部分内存吃爆的(不是整机:
整机 1511 GB,当时只用了 ~600 GB)。

### 51.2 定位
一层引擎(名义 3.2 GB)实测(逐层引擎单独跑,解析 `/proc/<pid>/numa_maps`):

| | node0 | node1 | node2 | node3 | node4 | node5 | node6 | node7 | 合计 |
|---|---|---|---|---|---|---|---|---|---|
| THP 开 | 1.69 | 1.53 | 3.10 | 1.50 | 1.52 | 1.50 | 3.02 | **5.00** | **18.9 GB** |
| THP 关 | — | — | — | — | — | — | — | — | **6.9 GB** |

原因:分片后每个 node 只拥有每个专家的一段**稀疏跨度**(w13:8.4 MB stride 里
2×512 KB;w2:4.2 MB stride 里 1×512 KB)。`MADV_HUGEPAGE` 让跨度覆盖到的每个
2 MB 页**整体**落地 ⇒ 3.2 GB/层变 15-19 GB/层,且各 node 因对齐不同而不均
(node7 一层 ~5 GB ⇒ 29 层 ≈ 145 GB,把 193 GB 吃光,于是 OOM)。

### 51.3 同一窗口交错 A/B(`bench_cpu_engine.py`,B=6/DEDUP=12/THREADS=120)
| 轮次 | THP 开 ms/层 | THP 关 ms/层 | RSS(开/关) |
|---|---|---|---|
| 1 | 1.20 | 1.21 | 18.9 / 6.9 GB |
| 2 | 1.28 | **1.20** | 同上 |
| 3 | 1.32 | **1.23** | 同上 |

⇒ **关掉 THP 内存省 2.7×,速度不降反略快**。旧注释("不落大页会 thrash 4KB TLB")
是错的论点:跨度是连续的 512 KB-1 MB,硬件预取足够。已在代码里把默认改为**关**
(仅 `XIAOTU_MOE_SHARD_HUGEPAGE=1` 可复现旧行为,调试用)。

## 52. 【bug 修复】`shard_region` 必须用 `mbind()`,不能用 `set_mempolicy()`

`set_mempolicy()` 给**线程**设内存策略,而线程策略会被**之后创建的新线程继承**。
引擎构建期间 vLLM 仍在创建线程(pinned 权重缓存等),它们在窗口内分配的大块内存
就被绑到单个 node 上 ⇒ `CONSTRAINT_MEMORY_POLICY` 的 OOM(整进程被杀:
`nodemask=7`,anon-rss 603 GB)。

`mbind(addr, len, MPOL_BIND, mask, ...)` 只作用于**这段映射**,不会泄漏到别的分配,
也不会被新线程继承。`numa_socket_alloc()`(socket 副本安全网)同样改成 `mbind`。

## 53. 执行固定规则(NUMA 分片)后的端到端结果 —— 解码延迟腰斩

配置:单卡(GPU2)TP=1 EP=0,`THREADS=120`,MAXLEN=262144,SEQS=2,KV 8GiB,
`fp8_ds_mla`,EAGER=0,DSpark k=5(probabilistic),GPU 预填充阈值 384。
自然文本数据集,`bench_nat_client.py`(SSE/usage 计数,见 M1)。

| 口径 | 交付配置(单拷贝,§49 之前) | **本次(NUMA 分片)** | 变化 |
|---|---|---|---|
| 预填充 @4096(TTFT) | 639 t/s | **703 t/s**(TTFT 5.83s) | +10% |
| 解码 C=1 聚合(`out_tok_per_s`) | 10.81 | **13.71** | +27% |
| 解码 C=1 **TPOT** | 52-65 ms | **28.47 ms** | **2.0×** |
| 解码 C=1 纯解码(1/TPOT) | 15-19 t/s | **35.1 t/s** | ~2× |
| 解码 C=1 step / tok-per-step | — | 97.35 ms / 3.49 | |
| 解码 C=2 聚合 / per-stream / TPOT | 19.93 / — / — | 15.37 / 7.68 / **72.72 ms** | 见下 |
| 每层 period / compute / rest | ~3.7-4.0 / 2.7-2.9 / 0.9-1.1 ms | **2.30-2.47 / 1.31-1.34 / 0.97-1.16 ms** | compute **2.1×** |

每层阶段分解(`XIAOTU_MOE_PROFILE`,bucket M3-8 = 解码主调用,qlen=6、k=6、na≈23):
`setup=131µs A=749µs B=421µs C=83µs TOTAL=1385µs`。
A(读 gate/up 分片)是最大项:聚合 195MB/749µs ≈ 260 GB/s(单拷贝时只有 ~103 GB/s)。

**结论**:
1. 固定规则(每 CCD 4-5 核 + 权重按 node 分片 + 只交换小数据)是解码延迟腰斩的直接原因;
   交付配置此前被 `SINGLECOPY=1` 悄悄退回单拷贝,属于回归(登记簿 R1)。
2. C=2 的 step 是 C=1 的 **2.6×**(252.7 vs 97.4 ms)⇒ 两条流**并没有被合批**(各跑各的
   6-token step),这不是引擎的问题,是调度/DSpark 的行为;要提并发吞吐得从那里入手。
3. 剩余时间预算:compute 1.3ms + rest 1.0ms 中,`rest` 是 GPU 侧(注意力/indexer +
   D2H/H2D + host-fn 派发)且与 CPU 计算**串行**;要再翻倍必须减少 CPU 层数
   (GPU 常驻专家层,需要 TP=2 的显存)或压低 A/B 的每线程带宽(2.1 GB/s vs 内层循环 5 GB/s)。

## 54. TP=2 + GPU 常驻专家层(第一步):预填充 +47%,但 EAGER 下解码退化

配置:`TP=2 EP=1 GPUS=2,0 MAXLEN=131072 KV 8GiB THREADS=60/rank OMP=1 EAGER=1
--disable-custom-all-reduce`,`XIAOTU_MOE_GPU_RESIDENT_LAYERS=0-5`(6 层×本 rank 1.59 GiB,
主模型 + draft 各一份 ⇒ 每卡 3.2 GiB/层),DSpark k=5。

| 口径 | tp2_eager(无 常驻层,§49.2) | **tp2resE(+6 常驻层)** | 单卡分片(§53) |
|---|---|---|---|
| 预填充 @4096 | 761 t/s | **1117 t/s**(TTFT 3.67s,+47%) | 703 t/s |
| 解码 C=1 聚合 / TPOT | 9.95 / — | 14.33 / **47.3 ms** | **13.71 / 28.5 ms** |
| 解码 C=2 聚合 / per-stream | 12.87 / — | 17.47 / 8.73(TPOT 86.9ms) | 15.37 / 7.68(TPOT 72.7ms) |
| 每层 period / compute / rest | — | 4.13-4.23 / **2.45-2.53** / 1.68-1.70 ms | 2.30-2.47 / 1.31-1.34 / 0.97-1.16 ms |

**读法**:
1. **常驻层对预填充立竿见影** (+47%):每层 3.19 GB 不再过 PCIe,而且是两条链路并行流
   剩下的层。要冲 1500 t/s 就继续加常驻层(受显存约束,见下)。
2. **TP=2 的 CPU 侧是净亏的**:每层 compute 从 1.31 → 2.45 ms。EP 只让每 rank 少读一半
   专家权重,但每层多出一次**跨 rank 部分和合并 + 屏障等待**(差 ~1.1 ms/层,与 §49.2 的
   估计一致);再加上每 rank 线程数减半(60)。⇒ 除非这些层**不在 CPU 上算**,TP=2 对
   解码没有好处。
3. 常驻层显存账:每层每卡 3.2 GiB(**主模型 + draft 模型各 1.59 GiB**)。当前 6 层用掉
   32.25/40 GiB(含 KV 8 GiB)。想加层数必须先省显存:draft 模型不必常驻 / 降 KV /
   提 GPU_UTIL。
4. 常驻层 + CUDA graph 目前**不兼容**(见登记簿 R14),本轮先用 EAGER 拿数据。

### 54.1 常驻层 + CUDA graph:两处捕获不安全点(登记簿 R14)
1. **已修**:`gpu_moe_layer` 里对常驻槽位 `wait_event(slot.ready)` —— 事件在捕获外 record,
   图内等待报 `cudaErrorStreamCaptureIsolation`。改为捕获期间跳过(数据构建时已写完)。
2. **未修(阻塞)**:`_build_segmentation()` 的 `ids = ids[ok]` 布尔掩码索引是**数据相关形状**,
   捕获不支持(`cudaErrorStreamCaptureUnsupported`)。`gpu_moe_layer` 过去只在"非捕获"的
   GPU 预填充分支里用(`not capturing` 门控),从未被捕获过。
   修法:无效 id 归入垃圾桶专家 `E`,`bincount(minlength=E+1)` 分段,内核 grid=E 跳过该段,
   `A` 用固定 `T*K`;修完还能让 GPU 预填充也进图。

## 55. 本轮(第 15 轮)结论与下一步

| 配置 | 预填充 @4096 | 解码 C=1 TPOT | 纯解码 t/s | 备注 |
|---|---|---|---|---|
| 单卡分片 + CUDA graph(**当前 8070 交付**)| 703 t/s | **28.5 ms** | **35.1** | 解码最优 |
| TP=2 + 6 常驻层,EAGER | **1117 t/s** | 47.3 ms | 21.1 | 预填充最优 |
| TP=2 无 常驻层,EAGER(§49.2)| 761 t/s | — | ~15 | — |
| TP=2 + 常驻层 + CUDA graph | — | — | — | 启动失败,见 R14(2 处捕获不安全点,1 已修)|

下一步优先级:
1. **修 `_build_segmentation` 的布尔掩码索引**(R14 追查 2):这是"常驻层 + 图"的唯一阻塞点,
   修完预填充(1117)与解码(graphs)可以同时拿到;顺带让 GPU 预填充也能进图。
2. **常驻层显存**:每层每卡 3.2 GiB 里有一半是 **draft 模型**的副本。让 draft 不常驻 / 降 KV /
   提 GPU_UTIL,可在同样显存下把常驻层数从 6 提到 ~10-12 ⇒ 预填充冲 1500。
3. **压低 CPU 侧每线程带宽**(A 阶段 2.1 GB/s vs 内层循环 5 GB/s):用微基准在
   **真实解码规模**(DEDUP≈23, BS=6)扫 `XIAOTU_MOE_SHARDSPLIT` / sub 切分,别再拿 B=6/DEDUP=12
   的旧数据外推。
4. 未测旋钮:`VLLM_USE_V2_MODEL_RUNNER=1`(目标清单里的)、`XIAOTU_GPU_PREFETCH_AHEAD`/window。

## 56. 预填充到 1500 t/s 的算术(TP=2 + 常驻层)

实测锚点:TP=2 + 6 常驻层 → 预填充 4096 **TTFT 3.66-3.67 s = 1117 t/s**;剩 37 层要过 PCIe。

- 每层每 rank 要流的字节 = 3.19 GB / 2(EP 各半)= **1.59 GB**;
- 实测每层 ≈ **99 ms** ⇒ 单链路 **16 GB/s**(PCIe gen4 x16 峰值 25,实得 ~64%);
- 若能把这些 H2D 与本层 GPU 计算**重叠**,每层应掉到 ~65 ms ⇒ TTFT 2.4 s ⇒ **1700 t/s**(达标);
  当前看不出重叠(99 ≈ 64 流 + 35 算)。

常驻层数受显存硬约束(TP=2,每层每 rank 1.59 GiB):KV 8 GiB + 非专家 ~5 GiB 时只能放
**~7 层**;要更多就得压 KV(`KV_MEM_BYTES`)或提高 `GPU_UTIL`。

⇒ 冲 1500 的两条路:①让预取的 H2D 与计算真正重叠(`XIAOTU_GPU_PREFETCH_AHEAD`,
本轮正在 A/B);②再省显存换 2-4 个常驻层。

### 56.1 顺带修掉的两个真 bug(见登记簿 R16)
- `finalize_mega_moe_weights()` 对常驻层不幂等(常驻层不设 `self.engine`),且插件的
  `hybrid_model` 在同一进程里会被加载成**两个模块对象** ⇒ 同一层建两份常驻显存,
  13 层吃掉 ~41 GiB,把 KV 的 8 GiB 挤掉 OOM。
- 修法:进程级状态挂 `builtins`(`_resident_state()`)+ 按 prefix 去重
  (`resident_already_built`)+ 新硬旋钮 `XIAOTU_MOE_RESIDENT_BUDGET_GB`(超预算的层
  自动回落成普通 CPU 层,并把 CPU 引擎建起来避免 `engine=None` 静默返回输入)。

### 56.2 预取重叠 A/B + 微基准上限(第 16 轮实测)
TP=2 + 6 常驻层、其余同配置,只改 `XIAOTU_GPU_PREFETCH_AHEAD`:

| 配置 | 预填充 4096 TTFT | 推算每层 |
|---|---|---|
| `XIAOTU_GPU_PREFETCH_AHEAD=1`(默认) | **3.70 / 3.66 s**(1105-1117 t/s)| 100 ms |
| `XIAOTU_GPU_PREFETCH_AHEAD=0` | **6.40 / 6.33 / 6.32 s** | 172 ms |

⇒ 预取重叠**确实在工作且不可缺少**(关掉几乎慢一倍)。

微基准上限(`scripts/bench_gpu_moe_prefetch.py`,E=128/topk=6,1.594 GiB/层 = TP=2 每 rank 量):

| T | 串行 ms/层 | **重叠 ms/层** | 加速 | 重叠 tok/s |
|---|---|---|---|---|
| 2048 | 110.2 | 64.1 | 1.72x | 743 |
| **4096** | 130.0 | **63.9** | **2.03x** | **1490** |
| 8192 | 175.3 | 98.1 | 1.79x | 1943 |
| 16384 | 267.3 | 189.0 | 1.41x | 2016 |

⇒ 重叠后 63.9 ms/层 = 1.594 GiB / 63.9 ms = **25 GB/s = PCIe gen4 x16 峰值**(微基准已达硬件上限);
真实服务里是 ~100 ms/层 ⇒ **还有 1.56× 的"框架开销"**(EAGER 逐层 Python/launch、breakable graph
分段、两 rank 协同)才是预填充没到 1500 的原因 —— 不是 PCIe、不是 CPU。下轮查这里
(而不是继续加常驻层:显存只够 ~7-10 层)。


## 57. 预填充瓶颈定位(第 17 轮):不是 PCIe、不是 CPU、不是并发

B=4096 预填充的时间账(TP=2 + 6 常驻层,实测 TTFT 3.66-3.70 s):

| 项 | 时间 | 依据 |
|---|---|---|
| 37 个未常驻层的权重 H2D | 37 × 63.9 ms = **2.37 s** | 微基准重叠后 63.9 ms/层 = 1.594 GiB / 64 ms = **25 GB/s = PCIe gen4 x16 峰值**;两进程并发实测同样是 63.9 ms(不互抢)|
| 其余(每层 attention/indexer + 框架)| ≈ **1.34 s**(37 层 × 36 ms)| TTFT 减去 H2D |
| CPU 专家路径(对照) | B=4096 全专家 **365 ms/层 ⇒ 15.7 s ⇒ 261 t/s** | compute-bound:3.4 TFLOP/s ≈ 峰值 50%,见 R17 |

⇒ 要冲 1500 t/s(TTFT ≤ 2.73 s),只有两条路:
1. **让每层 H2D 与 attention/indexer 真正重叠**(现在 100 ≈ 64 + 36,像是串行):若重叠成功,
   每层 = max(64, 36) ≈ 64 ms ⇒ TTFT ≈ 2.4 s ⇒ **≈1700 t/s 达标**。这是最该打的一点。
2. 再增常驻层:显存只允许 ~7-10 层(每层每 rank 1.59 GiB),收益线性但很小(+3%/层)。

常驻层的显存硬账(TP=2,KV 8 GiB,GPU_UTIL 0.85):6 层 → 32.25/40 GiB;12 层 → OOM
⇒ 上界 ~7-10 层。


## 58. 第 18 轮:重叠问题的进一步排查 + 工具坑

1. **pin 住主机缓冲的落点/建法不是瓶颈**:同一层 1.07 GB 的 K-major pinned 缓冲,
   主线程建 vs `prebuild_pinned_kmajor` 6 线程建,H2D 速率 **19.6 vs 20.1 GB/s**(基本一样)。
   两种建法的页面都撒在多个 NUMA 节点上(ATen 并行拷贝),但速率不变 ⇒ 跨节点读不是主因。
   微基准里重叠后的等效速率是 1.712 GB / 63.9 ms = **26.8 GB/s**(接近 gen4 x16 上限)。
2. **工具坑**:`XIAOTU_TORCH_PROFILE=/tmp/x.json` 导出的 trace 里**没有 GPU kernel 事件**
   (只有 CPU/annotation;`cat` 过滤 kernel/gpu_memcpy 得到 0 条),本轮无法用它定位
   "H2D 与 attention 是否重叠"。下一轮改用插件已有的 `XIAOTU_NVTX=1` + `scripts/tune_nsys.sh`
   走 nsys 时间线(能看到 memcpy 与 attention kernel 的实际重叠/串行)。
3. 服务里每层 ~100 ms 的构成仍未直接测到:`H2D ≈ 64-85 ms` + `attention/框架 ≈ 36 ms`,
   关键问题是这两者是否并行。这是下一轮唯一要先回答的问题。


## 59. 第 19 轮:预填充完全由权重流式决定(attention 几乎免费)+ 一次失败尝试

**缩放实验(单卡交付配置,TTFT vs 提示长度)**:

| 提示长度 | TTFT |
|---|---|
| 512 | 6004 ms |
| 1024 | 6016 ms |
| 4096 | 6320 ms |

⇒ TTFT 几乎与长度无关:43 层 × 137 GB / ~25 GB/s ≈ 5.5 s 就是全部时间,
**attention/indexer 只贡献 (6320-6004)/43 ≈ 7 ms/层且已被隐藏**。所以预填充是纯 PCIe 流式问题,
"每层 H2D 是否与 attention 重叠"这个悬念基本解除(在单卡配置上已是重叠的)。

**TP=2 + 6 常驻层** 的 37 层流式应当只要 63 GB/rank ÷ 25 GB/s ≈ 2.5 s,实测 3.70 s
⇒ 多出 ~1.2 s(≈31 ms/层)。本轮尝试把它归因于 `inter` 的 201 MB/层分配并试图按 T 行分配,
**失败**(内核按排序位置索引 inter,越界 → illegal memory access),已回退并登记 R19。
⇒ 那 31 ms/层仍未定位:候选是 ①`torch.zeros((T,H))` + `argsort` 等主机侧工作;
②EP 切片(pinned K-major 1.07 GB + w2 0.54 GB + scales)的 H2D 效率不如整层;
③两 rank 的 host→GPU 拷贝与 CPU 引擎线程争主机带宽。下轮直接用 nsys 量,不再猜。


### 59.1 第 20 轮:预取环加深无效(登记 R20)
微基准 2 槽 vs 3 槽 = 64.1 / 64.0 ms(PCIe 上限);服务 TTFT 3666-3707 ms(2 槽)vs
3752-3763 ms(3 槽)⇒ 无收益。那 ~31 ms/层仍待 nsys 定量。


## 60. 第 21 轮:主机侧开销可忽略(0.5 ms/层),预填充差额在 GPU 侧

新增埋点 `XIAOTU_GP_TIMING=1`(`gpu_prefill.py`,打印每 40 次调用的主机侧分段):

| rank | seg(_build_segmentation) | rest(到 launch 之间:zeros/inter 分配) | host_total |
|---|---|---|---|
| TP0 | 0.35-0.41 ms | 0.12 ms | **0.47-0.53 ms** |
| TP1 | 0.36-0.47 ms | 0.12 ms | **0.48-0.59 ms** |

⇒ TP=2+6 常驻层的每层 100 ms 里,主机侧只占 0.5 ms;剩下的 GPU 侧 ~35 ms/层
(相对 64 ms 的纯 H2D)最可能是**每层一次 TP all-reduce**(MoE 输出 (T,4096) bf16 = 33 MB,
A100-PCIE 无 NVLink ⇒ 走 PCIe),以及服务里 H2D 实得速率低于微基准的 26.8 GB/s。

**这一点解释了为什么 TP=2 的收益只有 1.6×(703→1117)而不是 2×**:省下的 PCIe 时间被每层
归约吃掉了一部分。而 TP=2 对解码是净亏(R15)。⇒ 若要把预填充推到 1500,正道是**去掉
GPU 预填充路径里的每层跨 rank 归约**(例如按 token 切分而非按专家切分),而不是继续调 H2D。


## 61. 第 22 轮:逐层排除表(预填充 TP=2 每层 100 ms vs 纯流式 64 ms)

| 候选原因 | 实测 | 结论 |
|---|---|---|
| 主机侧 Python/分配 | `XIAOTU_GP_TIMING` = **0.5 ms/层** | 排除(R21)|
| 预取环深度不足 | 3 槽 vs 2 槽 TTFT 3752/3763 vs 3666/3707 | 排除(R20)|
| 每层 TP all-reduce(33MB) | `XIAOTU_SKIP_AR=1` ⇒ 3618/3669 vs 3666-3752 | 排除,**仅 1.5-2 ms/层**(R22)|
| attention/indexer | TTFT 与提示长度无关(512→4096 常数) | 排除(§59)|
| 两 rank 抢 PCIe/主机带宽 | 双进程并发微基准仍 63.9 ms/层 | 排除(R18)|
| pin 缓冲落点/建法 | 主线程 19.6 vs 6 线程 20.1 GB/s | 排除(§58)|
| **H2D 传输效率** | 微基准 26.8 GB/s vs 服务 ~17 GB/s | **唯一剩余嫌疑(需 nsys 定量)** |

⇒ 预填充目标(1500 t/s)现在完全压在"为什么服务里的 H2D 只有微基准的 ~0.64×"这一个问题上。


## 62. 第 22 轮(续·用户提议):1 槽预取 = 正确性 bug

按用户建议实测"降到 1 槽省 3 GB":结果**不是性能问题而是算错层**
(err_vs_L=17.54 / err_vs_L+1=0.096,即本层用了下一层的权重)。原因:ping-pong 需要
"一槽在算、一槽在填"。⇒ 下限固定为 2(代码注释指向登记簿 R23);要省显存请调
`XIAOTU_MOE_RESIDENT_BUDGET_GB`(少放 / 多放常驻层,TP=2 每层 1.59 GiB)。


## 63. 第 23 轮:预填充每层 102 ms 的最终分解(HP2=TP=2+6 常驻)

新增 `[gp-h2d]`(CUDA 事件量真实传输时间),T=4096 预填充、37 个流式层:

| 组成 | 实测 | 说明 |
|---|---|---|
| H2D(权重流式) | **67.4-73.7 ms/层** | 1.71 GB ⇒ ≈24 GB/s,接近 PCIe gen4 x16 峰值(微基准 63.9 ms)|
| 主机侧(分段/分配/launch) | **0.44-0.59 ms/层** | 可忽略 |
| **剩余** | **~30 ms/层** | **未重叠的 GPU 侧工作**(attention/indexer ~7 ms + MoE 内核 + 常驻层/DSpark) |

⇒ 预填充 TTFT 3.77 s = 37 × (70 传输 + 30 GPU)。**若那 30 ms 能与 H2D 重叠或去掉,
TTFT → 2.6 s ⇒ 1580 t/s(达标)**。这是最后一次收窄:问题在 GPU 侧调度/工作量,
不在传输、不在主机、不在归约、不在环深。

下一步(nsys/nvtx):看这 30 ms 是哪些 kernel、以及它们为什么没有和 H2D 并行。


## 64. 串行(传-算-传)vs 重叠的显存性价比(回答用户提问)

| 模式 | 预填充 TTFT@4096 | t/s | 显存 |
|---|---|---|---|
| `XIAOTU_GPU_PREFETCH_AHEAD=0`(串行/等价 1 槽) | 6.32-6.40 s | 645 | 省 3.4 GB |
| 默认(重叠,2 槽) | 3.66-3.70 s | 1110 | 用 3.4 GB |

每 GB 收益:槽 ≈ **780 ms/GB**,常驻层 ≈ **44 ms/GB** ⇒ 要省显存先砍常驻层(有
`XIAOTU_MOE_RESIDENT_BUDGET_GB`),不要关重叠。登记簿 R25。


## 65. 第 24 轮:`VLLM_USE_V2_MODEL_RUNNER=1` 实测无收益(登记 R26)

TP=2+6 常驻层,预填充 TTFT 3733/3767 ms vs 基线 3666-3771 ms ⇒ 保持不设。
至此目标清单里点名的旋钮已全部试过:GPU 常驻专家层(✅ 预填充 +47%)、GPU 预填充阈值(384,
在用)、prefetch window/slots(实测无益,R20)、`VLLM_USE_V2_MODEL_RUNNER`(无益,R26)、
spec tokens 5→3(拒绝,R10)。


## 66. 第 25 轮:TP=2 反常的统一解释(跨卡归约流量)

把所有 TP=2 实测串起来的一块拼图:vLLM 行并行 attention 的 o_proj 每层一次跨卡 all-reduce,
`(T,4096)` bf16 = **33 MB/层**;A100-PCIE **无 NVLink** ⇒ 10-20 ms/层。这正好等于
"单卡每层非 H2D 12 ms vs TP=2 30 ms" 的 18 ms 差额(R21 排除了主机侧、R22 排除了我们的
2 ms 归约)。⇒ TP=2 预填充只 +47%(不是 +100%)、TP=2 解码净亏,都由这笔拓扑开销解释。
下一步:TP=2 + EAGER 下去掉 `--disable-custom-all-reduce`,量那 18 ms 是否缩小。


## 67. 第 26 轮:custom AR 无变化(R28);TP=2 那 ~30 ms/层仍未定位到具体 kernel

- custom all-reduce vs 兜底 AR:TTFT 3750/3768 vs 3666-3771 ms,无差别 ⇒ 归约实现不是原因。
- `VLLM_USE_V2_MODEL_RUNNER=0` ⇒ 启动失败(V1 不支持 dspark)⇒ 不要显式设 0。
- 至此预填充每层 102 ms = H2D 67-74(峰值)+ 主机 0.5 + **~30 未解释**,已排除:主机侧(21)、
  环深(20)、归约(22+28)、attention(59)、并发(18)、pin 落点(58)。**唯一出路是 nsys 的
  kernel 级时间线**(插件 profiler 的 chrome trace 在本配置下没有 GPU kernel 事件,见 §58)。


## 68. 第 27 轮:GPU 侧可见性仍缺失(profiler 埋点两个缺陷,R29)

`[xiaotu-profile]` 表 0 条 kernel 行,且两 rank 争抢 `/tmp/pref_prof.json`。⇒ 那 ~30 ms/层
仍未定位。下轮先修埋点(rank 后缀 + 从真实预填充开窗 + 按 cuda_time 排序),再用它定位。
预填充每层账:102 ms = H2D 67-74(峰值)+ 主机 0.5 + **~30 未解释**。


## 69. 【关键】预填充那 ~30 ms/层 = 我们的两个 Triton MoE 内核(block 配置是解码调的)

修好 R29 的埋点(每 rank 独立 trace + 按 `cuda_time_total` 排序)后,86 次调用窗口的 CUDA 表:

| kernel | CUDA 总计 | 每次调用 | 调用数(85 层) |
|---|---|---|---|
| `_down_kernel`(我们的 Triton) | 2.50 s | **29.4 ms/层** | 85 |
| `_gate_up_kernel`(我们的 Triton) | 2.07 s | **24.3 ms/层** | 85 |
| `Memcpy HtoD (Pinned->Device)` | 3.28 s | 7.4 ms × ~5/层 | 441 |
| `vllm::all_reduce`(NCCL ring bf16) | 1.39 s | 9.4 ms | 148(含注意力/共享专家)|
| fp8_einsum / marlin_gemm | 0.47 / 0.33 s | 1.4 / 0.2 ms | — |

⇒ 每层 GPU 侧 ~54 ms 全在两个 MoE Triton 内核上;它们用 `BM/BN/BK/BH=64、stages=2` 这组
**为解码调**的参数跑 T=4096(每专家 ~96 行)非常不划算。**这正是"预填充每层 102 ms =
H2D 70 + 30"里那 30 ms 的来源**(§63 未解释项)。

**下一步(便宜、不需要重启)**:`bench_gpu_moe_prefetch.py` 在 T=4096 扫
`XIAOTU_GPU_PREFILL_{BM,BN,BK,BH,STAGES,WARPS}`(以及 `_KERNEL=split`/`LUT`),
目标是让两个内核从 54 ms/层降到 ~20 ms/层 ⇒ 预填充 TTFT 3.77 → ~3.0 s ⇒ **1360 t/s+**。


## 70. 第 29 轮:kernel 参数扫描被污染,本轮无有效结论(R31)

T=4096 扫 WARPS/BM/BN/BK/STAGES:三种配置 `ovl` 全为 127.7 ms(基线 63.9),说明该窗口
PCIe/主机被外部负载拖慢,不能比较。下一次必须在**安静窗口 + 同进程交替**下扫,且只看 `seq`
(=`H2D + 内核`,H2D 是常数 ⇒ 差值是内核时间)。基线锚点:T=4096 seq **129.5-130.2 ms**、
ovl **63.9-64.0 ms**;profiler 实测每层两个 MoE 内核共 **53.7 ms**(§69)。


## 71. 第 30 轮(收尾):安静窗口 kernel 参数扫描 = 负结果,默认已最优

E=128 分片、T=4096、load 0.85,同窗口连续对照(只看 `seq` = H2D+内核):

| 配置 | seq ms | ovl ms |
|---|---|---|
| **默认**(BM=BN=BK=BH=64, NS=2, WARPS=4) | **130.4** | 63.9 |
| BM=128 | 134.0 | 63.9 |
| BM=128 + WARPS=8 | 135.9 | 63.9 |
| BK=128 + STAGES=4 | 148.2 | 68.8 |

⇒ 预填充那 54 ms/层的 MoE 内核时间**不是调参能解决的**;要动只能改内核结构(预填充向
grouped-GEMM / persistent CTA / 小行数专家的合并调度)。登记 R32。


## 72. 【事实更正】本机是**两路** 9654(我此前写成"单颗 9654",错了)

`lscpu` 实测:`Socket(s): 2`、`Core(s) per socket: 96`(共 **192 核**)、`L3 = 768 MiB / 24 实例`、
`NUMA 8 节点`(每节点 24 核 / 193 GB)⇒ 两路各 4 个 NUMA 节点;内存 1.5 TB DDR5-4800(24 通道,
实测聚合 783 GB/s)。

**两路都在用**:
- 线程池按 **CCD-major** 排列(24 CCD 轮转),`THREADS=120` = 5 核/CCD × **24 CCD = 两路全部 CCD**;
- 权重分片 `nshard_ = numa_node_count() = 8` = **两路 8 个 NUMA 节点各一片**,node 内全本地读。

**修正后的对照(这才是准确的差距)**:

| | 参考机(9684X×2 + PRO 6000) | 本机 |
|---|---|---|
| CPU | 2×9684X,192 核 Zen4c | **2×9654,192 核 Zen4**(核数持平)|
| L3 | 1152 MB | **768 MB**(少 1/3)|
| 内存 | 24ch DDR5-4800 | **24ch DDR5-4800(同规格,783 GB/s)** |
| GPU | RTX PRO 6000 | 3×A100-PCIE-40GB(无 NVLink)|

⇒ 差距**主要在 GPU 代次与预填充内核结构**,不是 CPU 核数/带宽;此前"N-A100 + 单颗 9654"的
说法作废。

**待复验(与用户规则相关,未做)**:参考配置的线程规则是 `LK_THREADS = 核数 ÷ GPU 数`(单卡 192),
你给的规则是"每 CCD 4-5 核"(120)。当年"192 比 120 慢 18%"是在**未分片**路径上测的;
分片路径下未复验。若要试,必须按 M5 在安静窗口同窗口 A/B,并且只作为数据、不改默认规则。


## 73. 【更正·关键】独立引擎对引擎:lk_moe 的 CPU 解码内核比我们快 ~2.0×

用户要求做"只测 decode"的对照。用**同一个同步 API**(`cpu_prefill(B,K,ids,wts,x,out)`,
两边都接受,numpy 与裸指针都**零拷贝**)、同一份真实第 3 层权重、BS=6、K=6、THREADS=120:

| 引擎 | DEDUP=12 ms/层 | DEDUP=23 ms/层 | 每层字节 | 聚合带宽 | 每线程 |
|---|---|---|---|---|---|
| xiaotu-moe(本仓库)| 1.22 | 1.32 | 151 / 252 MB | **124 / 184 GB/s** | 1.0-1.5 GB/s |
| **lk_moe**(lvllmds4-x env)| **0.57** | **0.67** | 151 / 252 MB | **265 / 376 GB/s** | 2.2-3.1 GB/s |

跑法(必须给 lk 用它自己的线程变量,否则它只用个位数线程 → 假慢 70×,这一步坑过一次):
```
ENG=lk LK_THREADS=120 CUDA_VISIBLE_DEVICES=0 XIAOTU_LAYER1_NPZ=<model> BS=6 DEDUP=12 \
  /home/user/anaconda3/envs/lvllmds4-x/bin/python /tmp/bench_both.py
ENG=xiaotu XIAOTU_MOE_THREADS=120 ... /home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python /tmp/bench_both.py
```
(`/tmp/bench_both.py` = `scripts/bench_cpu_engine.py` + `ENG=xiaotu|lk` 开关 + lk 的裸指针构造/调用适配)

**结论更正**:§44.1 那个"同模式端到端我们快 1.46×"是**服务器级**对照(两边 fork 的调用/同步不同),
不能代表内核;干净的内核对内核对照是**lk 快 ~2.0×**。⇒ 我们引擎还有 ~2× 的明确提升空间,
方向是"每线程带宽从 1.0-1.5 提到 2.2-3.1 GB/s"(而不是继续调线程数/布局)。


## 74. 【关键】根因定位:我们的解码路径退化成**标量 FMA**,lk 全程 packed fp32 FMA

方法:`objdump -d --no-show-raw-insn xiaotu_moe/build/_xiaotu_moe_C_avx512_bf16*.so`
后统计指令直方图;lk 侧用 `process_data/decomp/LK_MOE_KERNEL_DECOMP.final.md` 的热内核反汇编。

| | 我们(avx512_bf16 变体,全库统计)| lk(热内核 `0xa1d7b..0xa4b20`)|
|---|---|---|
| packed fp32 FMA | `vfmadd*ps` **243** | **`vfmadd231ps` 158(全部)** |
| **标量 FMA** | **`vfmadd*ss` 491** ✗ | **0** |
| bf16 点积 | `vdpbf16ps` 18 | 无(不需要)|
| fp4 解码 | `vpandd/vpand` 135、`vpshufb` 28、`vpmovzxbd` 28、`vpsrlw` 23 | 提到独立 load 阶段,每 K-block 解一次 |
| 寄存器分块 | 小 me 走 `block_23`(2/3 行)| **8 行 × 64 列,16 个独立 zmm 累加器**,权重解码后跨 8 行复用 |

**因果链**:解码每专家 me≈3(BS=6、K=6、DEDUP=12)⇒ 走小 me 路径 ⇒ 该路径是**标量实现**
⇒ 每字节约 3× 解码/调度开销 ⇒ 每线程 1.0-1.5 GB/s vs lk 2.2-3.1 GB/s ⇒ 每层 1.22-1.32 ms
vs lk 0.57-0.67 ms(§73,同 API 同线程同权重)。**不是 NUMA、不是线程数、不是 barrier 数。**

**下一步(按优先级)**:
1. 把 `csrc/moe/moe_v2_packed4.hpp` 小 me 路径(单行 + `block_23`)改成**全程 zmm packed fp32 FMA**:
   每 K-block 先把权重 fp4→fp32 解码到一个 64 列工作缓冲(2 × zmm),再对 me 行做 packed FMA,
   累加器 ≥8 个独立链(照 lk 的 16 链做法);`vdpbf16ps` 若用就把权重预转 bf16 对,避免逐行转换。
2. 验收标准(同一微基准,`ENG=xiaotu /tmp/bench_both.py`):BS=6/DEDUP=12 **≤0.7 ms/层**
   (现在 1.22),每线程 ≥2.2 GB/s;达标后再看服务端 TPOT(现在 28-31 ms → 目标 ~20)。
3. 之后再评估 3 阶段融合(登记簿里评估为 low-medium),不要先动它。


## 75. 目标已重排(2026-09-11):**第一条 = 引擎效率对齐 lk_moe**,其余目标在其后

- 目标 revision 8,phase=active、maxGoalRounds=60(此前 30 轮用尽被 blocked,用户要求改目标并继续)。
- 验收工具已固化:`scripts/bench_engine_ab.py`(`ENG=xiaotu|lk`,同一同步 API/同一权重/同线程;
  **lk 必须 `LK_THREADS=120`**,否则假慢 70×)。
- 基线(BS=6/DEDUP=12/THREADS=120):**xiaotu 1.22 ms/层(124 GB/s)vs lk 0.57 ms/层(265 GB/s)**;
  目标 **≤0.7 ms/层、每线程 ≥2.2 GB/s**,然后服务端 TPOT 28-31 → ~20 ms。
- 第一处代码改动(下一步):`xiaotu_moe/csrc/moe/moe_v2_packed4.hpp` 的小 me 路径(单行 + `block_23`)
  —— 权重 fp4→fp32 解码 hoist 到 K-block 级工作缓冲,内层全 packed zmm fp32 FMA、累加器 ≥8-16 链,
  热路径禁止 `ss` 标量 FMA(R34/§74)。

## 76. 【更正 §74】标量 FMA 在**归约 lambda**,不在主 GEMV;真正的杠杆是"解码摊销"

把 .so 里 `vfmadd*ss` 的地址映射回符号:前几名全是 `MOE_V2<...>::forward_many` 内的
lambda(阶段 C 的 `out_t[h] += w*d[h]` 标量归约)与 `NumaWorkPool::flat_thunk` 小 lambda;
**主 GEMV 不在其中**(它走 FAST_FP4 packed fp32 路径;`XIAOTU_MOE_DPBF16=1` 只再快 1.7%,
`GEMM_NR=0/16/32` 分别为 1.41/1.20/1.29 ms,都不如默认 1.18)。
⇒ §74/R34 的"小 me 路径退化为标量 FMA"**作废**(静态直方图把阶段 C 也算进去了)。

修正后的根因:FAST_FP4 **在 (行,组) 内层现场解码 fp4→fp32**,解码量随**激活行数**重复
(me≈3 ⇒ 3 次/权重字节);lk 是**每 K-block 解一次**(2×zmm)再复用 8 行。⇒ 每权重字节我们
约 2-3× 指令 ⇒ 只吃到机器带宽 13-20%(A 相 135 GB/s)vs lk 265-376 GB/s。
**是指令/解码摊销问题。**

下一步:把 FAST_FP4 的解码从 (行,组) 内层提到 **K-block 级**(每 64 列解一次到寄存器/工作
缓冲,再对 me 行做 packed `vfmadd231ps`,累加链尽量多)。验收同前:BS=6/DEDUP=12 ≤0.7 ms/层。

## 77. 第 32 轮:两项决定性事实 —— 解码 hoist **已实现**;2× 差距**真实可比**

**① "把解码 hoist 到 K-block 级"这条其实已经在代码里了。**
`moe_v2_packed4.hpp` 的 `block_23`(解码关键路径,me=2/3):
```cpp
for (int g = 0; g < group_count; g++) {
    const int base = g * 32;
    XIAOTU_DECODE_GROUP_AVX512(b_row, g);      // ← 解码在行循环**外面**
    const __m512 sv = _mm512_set1_ps(row_scale(srow, g));
    for (int r = 0; r < R; ++r) {              // ← 2/3 行复用同一份解码
        __m512 d = _mm512_mul_ps(wlo_, _mm512_loadu_ps(pr[r] + base));
        d = _mm512_fmadd_ps(whi_, _mm512_loadu_ps(pr[r] + base + 16), d);
        acc[r][p] = _mm512_fmadd_ps(d, sv, acc[r][p]);
    }
}
```
⇒ 目标第一条里"每 K-block 先解码再对 me 行做 FMA"的**结构已经满足**(注释里还记着当年从"每行重复解码"改过来的 1.78× 收益)。
**② 两引擎输出一致**(同一权重/同一输入/同一 routing,`cpu_prefill`):
xiaotu `absmean=11234.88`、lk `absmean=11235.16`,逐元素最大差 <0.5% ⇒ **2× 不是"工作量不同"的假象**,
是内核执行效率差。

**③ 与 lk 热内核的剩余差异(下一轮要打的)**:
- lk:激活保持 **bf16**,每个 k 用 `vpinsrw + vcvtph2ps + vpbroadcastss`(载入 2B)后 **2 条 FMA**;
  我们:先把激活整段转成 **fp32 缓冲 a32**(每个 (r,g) 载入 128B),再 `mul + 2 fma` ⇒ 激活侧 L1 流量 ×2、
  且多一遍 a32 转换(§330-334 那两行)。
- lk:每 K-block 解码结果**写进栈缓冲**,FMA 循环里只剩 2 条 zmm 载入 + 8×(cvt+bcast+2 FMA);
  我们:解码结果留在寄存器(更省),但激活侧更重。
⇒ 下一轮:把激活侧改成 **bf16 直取 + 寄存器内转换/broadcast**(去掉 a32 缓冲),把每 (r,g) 的
`2 load + mul + 2 fma` 压到 lk 的 `1 load(2B) + cvt + bcast + 2 fma`。验收不变(≤0.7 ms/层)。

## 78. 第 33 轮:连续三个假设被算术否掉 ⇒ 用**消融实验**定性(下一步)

逐层账(na=12,me=3,H=4096,I=2048):

| | 权重字节/层 | 激活(bf16) | 实测 | 等效带宽 | 机器利用率(783 GB/s) |
|---|---|---|---|---|---|
| 理想 | 151 MB | 0.29 MB | — | 783 GB/s | 100% |
| xiaotu | 151 MB | 0.29 MB | ~1100 µs | 137 GB/s | **18%** |
| lk_moe | 151 MB | 0.29 MB | 570 µs | 265 GB/s | **34%** |

**激活只占权重流量的 0.39%** ⇒ §77 那条"激活侧是差异来源"最多解释 0.1%,**作废**。

**已被算术/实测排除的原因**(累计):解码未 hoist(×,已在 block_23)、激活流量(0.39%)、
标量 FMA(在阶段 C 归约)、NUMA 分片/线程数/pin 落点/环深/并发、TP 归约、H2D 效率。
⇒ 剩下两类可能:**(A) 解码依赖链延迟**(我们的 group 解码是 load→and→srli→unpack→
inserti128→shuffle×2→unpack×2→inserti128×2→cvtepu16×2→slli×2 ≈ 14-16 条**依赖**操作,
每 (j,g) 一条长链;lk 把它写进栈缓冲、由 8 行 FMA 摊掉);**(B) 访存延迟/MLP 不足**
(每线程只有 ~1.1 GB/s,而 16 B/组的随机走位不利于硬件预取)。

**判定实验(便宜、决定性)= 消融 decode**:
在 `matmul_packed4_group` 里加一个 env 门控(`XIAOTU_MOE_ABL_DECODE=1`),把
`XIAOTU_DECODE_GROUP_AVX512` 换成"直接把原始字节当 fp32 用"(省掉全部 shuffle/转换链,
数值无意义但访存字节数不变):
- 若耗时明显下降(≥1.5×)⇒ **解码链延迟是主因** ⇒ 照 lk 做两段式(解码进栈缓冲/或 LUT 预解到
  64 列缓冲),而不是继续改访存;
- 若几乎不变 ⇒ **访存/MLP 是主因** ⇒ 改预取距离、增加每线程独立 K 组数(软件流水)、
  或按 node 内连续大块切分以利预取。
先跑这个实验,再动手改内核(避免第四个被否的假设)。

## 79. 【判定实验】解码**依赖链**是每线程吞吐的限幅器(4.19×),修法=跨组 ILP

独立微基准(`/tmp/abl.cpp`,同一访存字节、只切掉 shuffle/转换链;L2 常驻的 64 列权重块,
200k 次遍历,单线程):

| 变体 | 吞吐 |
|---|---|
| 真实解码链(and/srli/unpack/inserti128/cvtepu8/slli) | **238.97 GB/s/线程** |
| 去掉解码链(原始字节直接当 fp32) | **1001.77 GB/s/线程** |
| 比值 | **4.19×** ⇒ 解码链是主因 |

**为什么这解释了引擎的 1.1 GB/s/线程**:我们的内层每消费 **16 字节**权重就串上一条
~40-60 周期的依赖链(load→and→srli→unpack→inserti128→cvtepu8×2→slli×2,全依赖),
按 3 GHz 折算 ≈ 16 B / 50 cyc × 3 GHz ≈ **0.96 GB/s/线程** —— 与实测 1.1 吻合。
FMA 侧本身不是瓶颈(去掉解码链后 1002 GB/s/线程)。lk 是 2.2-3.1 GB/s/线程,正是"把解码
与 FMA 解耦 + 多条独立链并行"的结果。

**修法(按收益/风险排序)**:
1. **跨 K 组软件流水/展开**:一次处理 **2-4 个独立 K 组**(各自独立的 `wlo/whi` 与累加器),
   让 4 条解码链互相重叠 ⇒ 理论上把 ~50 周期摊成 ~12-25 ⇒ 接近 2×。这是最小改动、最高收益。
2. 再考虑把解码结果写进小的 L1 工作缓冲(照 lk 的栈缓冲),让 FMA 循环与解码彻底解耦
   (代价:多一次 store/load;仅在 1 不够时做)。
3. **不要**再往"减少指令条数/换 LUT/改激活表示"方向做:实验已证明是**链延迟**而不是条数。

## 80. 第 35 轮:跨 K 组 2 路软件流水**无收益** ⇒ §79 的消融微基准不代表引擎(L2 常驻)

改动:`block_23` 的组循环按 2 组展开、两组各自独立作用域(让 OoO 推进两条解码链)。
实测(BS=6/THREADS=120):**1.19 / 1.32 ms/层**(DEDUP=12/23)vs 基线 1.18 / 1.32 ⇒ **无差别**,已回退。

**教训(重要)**:§79 的 `/tmp/abl.cpp` 权重块只有 128 KB、**L2 常驻**,所以它测的是"解码链
的指令吞吐";而引擎里权重是 **DRAM 流式**,每线程 1.1 GB/s = 16 B/43 cycle,与"链延迟"数值上
巧合吻合,但**因果不同**:引擎里更可能是 **L2/DRAM 访存延迟 + MLP(每线程在飞请求数)不足**。
⇒ 消融实验必须在**DRAM 常驻工作集**下重做(把 abl 的权重块扩到 ≥256 MB、每轮遍历不同地址),
并同时测"每线程不同预取距离/多路独立行"的效果,才能定性。

## 81. 【关键】内层循环不慢(DRAM 下 8.46 GB/s/线程)⇒ 瓶颈在**引擎结构**,不在内核

`/tmp/abl2.cpp`:512 MB 工作集(必走 DRAM),单线程,顺序流式 + 真实解码:

| MLP(每线程独立流数)| real GB/s/线程 | 去掉解码链 GB/s/线程 |
|---|---|---|
| 1 | **8.46** | 17.21 |
| 4 | 6.40 | 12.45 |
| 8 | 5.59 | 9.00 |

**对照**:引擎实测 **1.1 GB/s/线程**(A 相 749 µs / 151 MB / 120 线程)⇒ 差 **7.7×**。
⇒ 结论反转:**不是内核慢,是引擎结构把内核的效率吃掉了**。解码链在单线程 DRAM 流式下
确实值 2×(8.46 vs 17.21),但只占那 7.7× 的一小部分;主要缺口在引擎侧。

**引擎侧候选(下一步要量的)**:
1. **每 job 开销**:A 相每层 96 job/节点(na=12 × subA≈8),每 job 只 64 KB(32 行×3 me)
   ⇒ 按内核 8.5 GB/s 每 job 只需 7.5 µs,而实测每节点 A 相 ~749 µs ÷ 6.4 job/线程 ≈ 117 µs/job
   ⇒ **每 job 开销 ~100 µs 量级** 是主要缺口(派发/领任务/每 job 的 row_scale 与分段索引/冷 TLB)。
2. **访存落点**:分片区域关掉了 THP(R2),3 GB/节点的稀疏分片 + 4 KB 页 ⇒ 页表/DTLB 压力。
3. **冷 DRAM 首触**:每层 151 MB 权重全是冷数据(12 个专家 × 12.6 MB)。

**下一步(判定)**:在引擎里插一个"单线程跑同一 job body"的对照(把 A 相强制 wlimit=1、
不跨节点),量出"同一内核在引擎缓冲/分片布局下的单线程吞吐":
- 若也接近 8 GB/s ⇒ 内核没问题,**job 粒度/派发**是病根(改:合并 job、每线程更大连续块、
  去掉每 job 的重复索引计算);
- 若掉到 1-2 GB/s ⇒ 是**分片布局/页表/落点**(改:恢复大页 for 分片、或每 job 用更大的连续行块)。

## 82. 第 37 轮:THP 假设被否;单线程对照存在"远端读"混淆

**(1) THP(分片区域)在解码尺寸下无收益** —— `XIAOTU_MOE_SHARD_HUGEPAGE` 同窗口对照:

| | DEDUP=12 | DEDUP=23 |
|---|---|---|
| THP=0(现状) | **1.21 ms** | 1.45 ms |
| THP=1 | 1.35 ms | 1.36 ms |

⇒ 页表/DTLB 不是那段 3× 损失的原因,**R2 的"分片关 THP"决定再次被验证**(省内存且不慢)。

**(2) §81 的"引擎内单线程 = 2.86 GB/s/线程"有混淆,不能直接与独立内核的 8.46 相除。**
`XIAOTU_MOE_THREADS=1` 时 `parallel_for_sharded` 走"降级为调用线程串行执行全部 job"的分支
⇒ 那一个线程要读**全部 8 个分片**(7/8 是**远端**访问)⇒ 2.86 GB/s 主要是 NUMA 距离 32 的代价,
不能当作"引擎布局损失"。

**(3) 因此正确的对照是**:让每个线程**只读自己 node 的分片**(即 120 线程 + NS=8,现状),
量"每 node 聚合带宽 / 每线程带宽",与独立内核的 8.46 GB/s/线程比 ⇒ 现状 **1.07 GB/s/线程**,
差 7.9×。剩下的真实候选只有:
- **(a) 行跨步访存**(每个 node 的分片里,每专家只有 512KB 连续跨度,跨步 4.2-8.4MB ⇒ 预取器不友好);
- **(b) 每阶段/job 开销**(96 job/节点、每 job 64KB ⇒ 每 job ~100 µs 量级,见 §81 的算术);
- **(c) 每 node 15 线程对 L3/内存控制器的争用。**

**下一步(判定)**:在**保持 120 线程 + NS=8(每线程本地读)**的前提下,做两个消融:
①`XIAOTU_MOE_SHARDSPLIT` 调大(每 job 更大连续跨度 ⇒ 检验 (a)/(b));
②用 `XIAOTU_MOE_PROFILE` 的 A 相每层时间对比 `na` 相同但 me 不同(1 vs 3)的每字节耗时 ⇒ 检验 (b)。

## 83. 第 38 轮:job 粒度(每 job 开销)被否;缺口仍指向**访存模式**

`XIAOTU_MOE_SHARDSPLIT` 同窗口扫描(BS=6、DEDUP=12、THREADS=120,na=12):

| SHARDSPLIT | 0 | 1 | 2 | **4** | 8 | 16 |
|---|---|---|---|---|---|---|
| ms/层 | 1.32 | 1.32 | 1.24 | **1.20** | 1.23 | 1.31 |
| GB/s | 114 | 114 | 122 | **126** | 123 | 115 |

⇒ 曲线平坦(±10%),**默认(auto≈5)已在最优点附近** ⇒ §81/§82 的候选 (b)"每 job/每阶段开销"
**不是那 7.9× 的主因**(最多值 10%)。

**累计排除(第 33-38 轮)**:解码链(36)、内核指令(37)、激活侧(33)、THP/页表(38)、
job 粒度/派发开销(83)、线程数/NUMA 分片/pin/环深/并发/TP 归约/H2D。
**唯一还没被单独消融的**:(a) 分片布局下的**行跨步访存与 scale 取数模式**;
(c) 每 node 15 线程对 L3/内存控制器的争用。

**下一刀(在引擎内做消融,按信息量排序)**:
1. **`XIAOTU_MOE_ABL_SCALE=1`**:内层跳过 `row_scale` 取数与 `fmadd(d, sv, acc)`(改成直接加),
   访存字节不变但去掉每 (行,组) 的 scale 小load + 一条 FMA ⇒ 若耗时明显下降 ⇒ scale 取数模式是病根
   (改法:把 scale 预取进每行的寄存器/或改成与权重同布局一次读出)。
2. **`XIAOTU_MOE_ABL_DECODE=1`**(两轮前提过、还没做):引擎内把解码替换为"原始字节当 fp32"。
3. 若 1/2 都无变化 ⇒ 只剩 (c) 争用/访存模式 ⇒ 用 `numastat -p` + 每 node 计时(numa_pool 里按 node
   打印 busy)来定位。

## 84. 第 39 轮:分片跨步访存也被否 ⇒ 只剩"**节点内并行带宽标度**"

`/tmp/abl3.cpp`(1 GB 工作集,单线程,每轮读 256 MB):

| 访存模式 | GB/s |
|---|---|
| 顺序流式 | 8.54 |
| **512KB 连续 + 跳 4.2MB(= 分片形态)** | **8.52** |
| 512KB + 跳 8.4MB | 8.56 |
| 128KB + 跳 4.2MB | 8.57 |

⇒ **跨步/分片布局不是病根**(预取器完全吃得下 128-512KB 连续跨度)。

**关键算术(每层每 node)**:
- 每层总权重 151 MB ÷ 8 node = **18.9 MB/node**;单 node 带宽实测 **98.7 GB/s** ⇒ 理论上 **191 µs**;
- 实测 A+B ≈ **1100 µs** ⇒ 差 **5.8×**;
- 也就是说:15 个线程/node 合起来只跑出 **17 GB/s**,而该 node 单线程就能跑 8.5 GB/s、
  node 级上限 98.7 GB/s。⇒ **17 GB/s 是"节点内并行标度"的问题**,不是单线程能力、不是布局、
  不是 job 粒度、不是内核。

**下一个判定实验(明确)**:单 node、N=1/2/4/8/15 线程,同进程内流式读**该 node 自己的分片**
(用 `mbind(MPOL_BIND)` 绑到 node)、按分片形态访问,量聚合带宽曲线:
- 若 N=15 仍只有 ~17 GB/s ⇒ 引擎的**线程↔分片映射/亲和性**有问题(例如同一 node 的 15 个线程
  实际被调度到不同 CCD、或分片页不在本 node);
- 若 N=15 能到 60-90 GB/s ⇒ 说明引擎里线程没有真正"只读本 node"(或存在跨 node 打散),
  回到 onlining/affinity 检查。
先跑这个,再改任何代码。

## 85. 第 40 轮:引擎在 **~130 GB/s 饱和**(硬上限),且 **NS=8 细分最优**

**线程标度曲线**(DEDUP=12,151 MB/层):

| THREADS | 24 | 48 | 96 | 120 |
|---|---|---|---|---|
| ms/层 | 2.54 | 1.56 | **1.17** | 1.22 |
| 聚合 GB/s | 59.4 | 96.8 | **129.1** | 123.8 |
| 每线程 GB/s | 2.48 | 2.02 | 1.34 | 1.03 |

⇒ 96 线程起**不再增长**(饱和 ~130 GB/s = 机器 783 的 17%),**不是线程不够**,是硬上限。
每 node 折合 **~16 GB/s**,而该 node 的实测能力是 **98.7 GB/s**。

**分片粒度**:NSHARD=8 → 1.20 ms(126 GB/s);4 → 1.71(88);2 → 2.60(58)
⇒ **越细分越快**,说明"每 node 只读本地分片"的机制**在正常工作**(粗分片才慢)。
结合 §84 的落点检查(页分布均匀)⇒ **线程↔node 映射与页落点都不是病根**。

**累计已排除(33-40 轮,全部实测)**:解码链、内核指令、激活侧、THP/页表、job 粒度、
跨步布局、NUMA 落点、spin 争用、线程数(饱和)、分片粒度(已最优)、pin、环深、并发、TP 归约、H2D。

**剩余唯一异常**:单线程独立内核 8.46 GB/s,而引擎里 15 线程/node 只有 1.07 GB/s/线程
(比单线程还差 8×)。下一个判定实验(**mini-engine**):用一个独立小程序**复刻引擎每线程的
完整内层**(含 a32 fp32 激活载入 + row_scale 取数 + block_23 结构),在**绑到某 node 的分片缓冲区**
上跑 1/5/15 线程:
- 若 15 线程能到 60+ GB/s/node ⇒ 引擎的**编排层**(池、job 领取、屏障)仍有问题;
- 若也只有 16 GB/s/node ⇒ **内层每字节成本**(a32 载入 + scale + decode 的组合)就是上限,
  修法是压缩内层每字节指令/L1 流量(而不是继续调线程/布局)。

## 86. 第 41 轮:内层彻底洗清;真正的"吃时间"项是 **A2 setup(21%)**

**(1) mini-engine 结果(附方法教训)**:`/tmp/abl4.cpp` 把引擎内层的全部附加成本都加上
(a32 fp32 激活载入 + `row_scale` 取数 + 2 行复用),但因工作集只有 256 KB(**又落回 L2 常驻**),
数值虚高:mode0 578 / mode1 393 / mode2 473 / **mode3(引擎态)367 GB/s/线程**。
教训(第二次犯):**微基准的工作集必须 >L3 才是 DRAM 结论**,否则只反映指令吞吐。
但反面结论有效:**引擎内层即便带上全部附加成本,也有 367 GB/s/线程的指令吞吐 = 实际 DRAM 速率
(8.5 GB/s/线程)的 43 倍** ⇒ **内层不可能是限幅器**(与 R37 一致)。

**(2) 回头看清真正的吃时间项**:NS-PROF(DEDUP=12,每层)是
`A=0.645 ms + A2(setup)=0.26 ms + B=0.35 ms + C=0.08 ms ≈ 1.34 ms`(实测 wall 1.20)。
- **A2 = 0.26 ms = 全层的 21%**,而它只是"每个活跃专家把 me 行 hidden(12 专家×3 行×8 KB
  = 288 KB)memcpy 进 `g.xg` + 若干 `resize()`" ⇒ 按 DRAM 速率应 **≈30 µs**,实测 **260 µs(8.7×)**
  ⇒ 这是**纯软件开销**,与访存无关(§53 早已标为"未解释",现在有量级了)。
- 排除 A2 后,A 相真正"读"的部分 = 0.385 ms / 101 MB = 262 GB/s;B = 0.35 ms / 50 MB = 143 GB/s。

**下一步(具体、可验证)**:攻 A2 —— 把每专家的 gather 与 buffer 维护做掉:
①`xg` 改为**按 token 直接索引**、不做整行 memcpy(内层本来按行读,没必要先复制);
②去掉每次调用的 `resize()`/`clear()` 抖动(改容量管理);
③`count_/inst_idx_` 两遍遍历合并成一遍。
目标:A2 0.26 → <0.05 ms,则每层 1.20 → **~0.99 ms**;再攻 A/B 的 262/143 GB/s 才谈 0.7。

## 87. 第 42 轮:A2 的零填充理论被否;A2 成本仍未解释

**先核实 §86 的 A2**:REP=200 取**稳态窗口**(我上次怀疑是冷启动,实测不是):

| 窗口 | A | A2 | B | C | 合计/40 次 |
|---|---|---|---|---|---|
| calls 41-80 | 19.7 | 11.0 | 13.5 | 3.9 | 48 ms |
| calls 81-120 | 19.5 | 11.0 | 13.3 | 3.9 | 48 ms |
| calls 161-200 | 20.2 | 11.7 | 13.5 | 3.8 | 49 ms |

⇒ 稳态 **A=0.49 / A2=0.28 / B=0.34 / C=0.095 ms**,合计 1.20 ✓ 与 wall 1.21 吻合
⇒ **A2=23% 是真的**(不是冷启动污染)。

**试了什么**:把 setup 里每专家的 5 次 `resize()`(理论零填充 ≈1.4 MB/层)改成 `reserve()`
(只保证容量;这些缓冲全程只用 `.data()` 指针访问)。
**实测**:DEDUP=12 → **1.29**(基线 1.20)、DEDUP=23 → **1.35**(基线 1.32);A2 仍 **0.28 ms**
⇒ **无收益,零填充不是 A2 的主体**,已 `git checkout` 回退。

**A2 里到底花了什么(下一步要量的)**:A2 覆盖 t_entry→pA0,包含
①`std::fill(output, M*hidden)`;②`exp_.resize/active_.clear/count_.assign/inst_idx_.assign`;
③对 256 个专家查 count_ 并 `ai_list.clear()+reserve()`;④两次 NASS 遍历填 inst_idx_;
⑤每活跃专家的 5 次缓冲容量维护 + me 次 8KB memcpy(12×3×8KB=288KB)。
**下一步**:在 A2 内部分三段打点(`XIAOTU_MOE_PROFILE` 扩展:fill / bookkeeping / gather),
一次跑出哪一段占 0.28 ms —— 288KB memcpy + 几十次 vector 维护无论如何不该是 0.28 ms,
大概率是**某个隐蔽的全局同步或分配**(例如 `exp_` 256 项的 `std::vector` 内部指针追逐、
或 `reserve()` 触发的真实 realloc)。

## 88. 第 43 轮:A2 三段打点的补丁插错作用域(编译失败),已回退

给 A2 加 `[setup-prof]`(fill+bookkeeping / gather)打点,但 `const size_t na = active_.size();`
在 `moe_v2.hpp` 里**有两个同名点**(`forward_many_nsliced` 与扁平路径),补丁打到了错误的那一处
⇒ `error: '_su0' was not declared in this scope`,构建失败,已 `git checkout` 回退。

**下一轮正确做法**:用**唯一锚点**定位 —— 在 `forward_many_nsliced` 内、紧跟
`const size_t a2_total = exp_off_[na];`(该行只属于分片路径)之后插入统计块;
`_su0` 放在 `const auto t_entry = ...` 之后、`_su1` 放在 gather 循环的
`for (int e : active_) {` 之前(该行也需确认唯一,否则用 `pfor_sharded` 附近的上下文锚定)。

### 88.1 更正(同一轮内自查)
`grep -n "const size_t na = active_.size();"` 只有**一处**(line 767)⇒ 上面"na 有两个同名点"的判断**错了**。
真正的错因是 **`std::fill(output, output + M*hidden, 0)` 这个锚点不唯一**(扁平路径里也有),
补丁把 `_su0` 声明插到了另一个函数里,而统计块落在 `forward_many_nsliced` ⇒ `_su0` 不在作用域。
**正确锚点**:`_su0` 用紧邻的 `const auto t_entry = std::chrono::steady_clock::now();` 之后
(配合 `prof_init();` 的上下文),或直接用 `a2_total`(`const size_t a2_total = exp_off_[na];`,
该行只在分片路径出现)作为后续锚点。

## 89. 【定位】A2 的 97% 在 **per-expert gather**(295 KB 却花 267 µs ⇒ 1.1 GB/s)

新增 env 埋点 `XIAOTU_MOE_SETUP_PROF=1`(唯一锚点;`moe_v2.hpp`),REP=200 稳态:

```
[setup-prof] n=40 per-call(us): pre_bookkeeping=8.6-9.4  gather=266.8-267.7  total=276.2
```

- **pre(fill + count_/inst_idx_/active_ 记账)= 9 µs** ✓ 正常;
- **gather = 267 µs = A2 的 97%、整层的 22%**;
- gather 做的是:12 个活跃专家 × me(=3) 行 × 8 KB memcpy(hidden bf16)= **295 KB**,
  外加每专家 5 次缓冲容量维护 ⇒ 等效 **1.1 GB/s**,比普通 memcpy(10-20 GB/s)慢 10-20×。

**为什么慢(候选)**:①每次 memcpy 只有 8 KB、且散落在 256 个 `ExpBuf` 里的 12 个上 ⇒
**目标页/行冷启动 + RFO 写分配**(295 KB 写 + 295 KB 读 = 590 KB,仍应 ~30-60 µs);
②`resize()` 已在 R42 排除;③指针追逐穿过 `std::vector<ExpBuf>`(256 项)带来 TLB/缓存抖动。

**修法(按收益/风险排序)**:
1. **取消 gather**:内核本来就按行反复读 `x[me][K]`(每行被读 K/32 次),把"行号数组"
   (`tok` id)传给 `gate_up_slice_batched`/`down_slice_batched`,让内层直接 `input + tok[m]*hidden`
   取行 ⇒ gather 的 267 µs **全部省掉**、还少一次 295 KB 的写读。需改这两个 slice 内核的入口与
   `linear.h/mlp.cpp` 的调用点(中等改动,收益 ~22%)。
2. 若 1 太大:**非时态写**(`_mm_stream_si128`)做 gather,去掉写分配 ⇒ 预期 2-3×。
3. 或每专家**一次性预留 & 复用一个紧凑 arena**(而不是 256 个独立 vector),改善 TLB/局部性。

## 90. 第 45 轮:NOGATHER 诊断**无收益** ⇒ 那 267 µs 不是"可摘掉的串行 gather"

新增计时诊断 `XIAOTU_MOE_NOGATHER=1`(跳过 per-expert gather 的 memcpy,**数值无效**、只测时):

| | DEDUP=12 | DEDUP=23 |
|---|---|---|
| 基线 | 1.20 ms | 1.32 ms |
| **NOGATHER** | **1.27** | **1.48** |

⇒ 跳过 267 µs 的 gather **总时间没有下降**(反而略升,噪声内)⇒ 结论有两种可能:
1. `[setup-prof]` 测到的 267 µs 与**池阶段的时间重叠**(例如 gather 期间 worker 已在为上一阶段
   收尾/或 A 相启动与其并行),所以它不是可加的串行成本;
2. 或摘掉 memcpy 后 A 相读到的是**陈旧/更冷的 `xg`**,把省下的时间又吃回去。

**这是本轮第二次"先量后改"的价值**:如果直接按 §89 去改内核加"行号索引",很可能又白做一轮。
**下一步(正确顺序)**:先**证伪/证实"重叠"假说** —— 在 A 相入口打点(`pA0`)与 gather 结束时刻比较:
若 A 相起点早于/等于 gather 结束 ⇒ 两者重叠;若是严格串行,则 267 µs 必然可摘,NOGATHER 的
"无收益"就只能是第 2 种解释(冷 `xg`),那时再决定改法。

## 91. 【结论】A2 是**耦合成本**(gather 在预热 xg),不是可摘浪费;靶子锁定 A/B 带宽

`XIAOTU_MOE_NOGATHER=1` + `PROFILE`(REP=160,DEDUP=12,稳态;单位 ms/次调用):

| | A | A2 | B | C | wall |
|---|---|---|---|---|---|
| baseline | 0.48 | **0.25** | 0.33 | 0.07 | **1.16** |
| NOGATHER | **0.84** | 0.005 | 0.33 | 0.07 | **1.28** |

- 摘掉 gather ⇒ A2 → 0.005(**埋点与 §89 准确**),但 **A +0.36** ⇒ gather 的 memcpy
  实际在**预热目标页/缓存**,不做它 A 相就付冷访存;总时间略升 ⇒ **A2 不可摘**。
- 故 §89/§90 的"A2 占 22%、值得改接口"**作废**;真正的限幅项是 **A/B 两相的访存带宽**:
  A = 101 MB / 0.48 ms = **210 GB/s**;B = 50 MB / 0.33 ms = **152 GB/s**(机器 783)。

**A/B 的剩余可能(内层已排除)**:
- 内层指令吞吐有 **43× 余量**(§86)、布局/落点/自旋/job 粒度全部排除(§84/§85/R40/R41/R39);
- ⇒ 只剩 **每线程的访存并发度(MLP)**:内层每 (行,组) 只消费 16 B,且 `block_23` 的行内是
  顺序 2 KB 流,理论上预取器应能覆盖;但**每个 job 是 32 行的独立 2 KB 流**,job 之间跨
  4.2-8.4 MB ⇒ 预取器在 job 边界处每次都要重新起步(距离 ~几十 µs 一趟)。
- **下一步(便宜)**:`_mm_prefetch` 把**下一行/下一 job 的前几行**显式提前拉进来(现在只
  prefetch `next_row`,即同一 job 内的下一行);或把 job 的粒度调成"每线程一整块连续 512KB 跨度"
  并用软件流水(每次预取 i+2 行)。

## 92. 第 47 轮:i+2 预取改动**未应用**(锚点 3 处,断言中止),基线重测 1.26/1.37

计划:`block_23` 里除了"预取下一行(`next_row`)"再加"提前两行(`far_row`, j+2)",
加深预取距离以覆盖 job 边界(§91 的候选)。
**实际**:补丁脚本的锚点(`if (next_row && ((g) & 3) == 0) ...`)在文件里出现 **3 处**
(不是我以为的 2 处),`assert n == 2` 在 `write_text` **之前**中止 ⇒ **源码未修改**、
构建产物与基线一致 ⇒ 本次测得的 1.26/1.37 只是同负载下基线(1.16-1.24)的重测,**不可当 A/B 结论**。

**下一轮正确做法**:3 处锚点全部替换(`grep -c` 先确认数量),或用 `sed -i` 全局替换;
改完按 §91 的判据验收(A/B 相 210/152 → 目标 ~400 GB/s,每层 → ~0.6 ms)。

**教训**:补丁前先 `grep -c` 确认锚点数量(本轮与 §88 各因锚点计数错误浪费一次构建)。

## 93. 第 48 轮:i+2 预取**无收益**(1.17/1.32 vs 1.20/1.32),预取距离也不是

按 §92 的教训先 `grep -c` 确认(3 处),`replace` 全部应用 + 唯一 `next_row` 定义后加 `far_row`(j+2),
构建 0 错误、`far_row` 出现 7 次(1 定义 + 3×2 引用)✓ 确实生效。

| | DEDUP=12 | DEDUP=23 |
|---|---|---|
| 基线 | 1.20 | 1.32 |
| **i+2 预取** | **1.17** | **1.32** |

⇒ **无收益**(DEDUP=12 的 +2.5% 在噪声内,而且 DEDUP=23 完全一致)⇒ 预取距离不是限幅项,已回退。

**至此 A/B 相的全部常规杠杆都被实测排除**:指令(43× 余量)、布局/落点、线程↔node 映射、
自旋、job 粒度、分片粒度、线程数(饱和)、预取距离。**剩下未验证的唯一方向**:每线程的
**在飞请求数上限**(MSHR/LFB 填满)——可用"内层同时展开 2 个 K 组各取数"来做软件 MLP,
或在同一 job 内改成"跨行交错取数"以提高并发度。若这也不动,则说明 ~130 GB/s 是本引擎
在这个访问形态下的硬顶,需要换数据结构(例如把权重整理成"每行连续且行间紧密"的布局,
让一个线程能一次拉 64KB 连续块,而不是 32 行 × 2KB)。
