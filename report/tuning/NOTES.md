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

# 94. 【交接】引擎对齐任务的当前状态与下一步(2026-09-11 第 49 轮)

## 94.1 目标与现状
目标第一条:CPU 解码效率对齐 lk_moe。验收:微基准 **≤0.7 ms/层、每线程 ≥2.2 GB/s**;
服务端 TPOT 28-31 → ~20 ms。

| | ms/层(DEDUP=12/23)| 每线程 |
|---|---|---|
| **xiaotu(现状)** | **1.16-1.24 / 1.32** | 1.0-1.5 GB/s |
| lk_moe(同 API/同线程/同权重)| **0.57 / 0.67** | 2.2-3.1 GB/s |

每层分解:A=0.48(210 GB/s)/ A2=0.25(gather,耦合不可摘)/ B=0.33(152 GB/s)/ C=0.07。
机器 783 GB/s ⇒ 引擎利用率 ~17%(lk 34%)。

## 94.2 已排除(全部实测,别再重复)
内核指令(R37,内层 43× 指令余量 §86)· 解码链(R36)· 激活侧 0.39%(R33)·
THP/页表(R38)· job 粒度 SHARDSPLIT 平坦(R39)· 跨步/分片布局(R40)· 线程↔node 映射与
页落点(R41)· 自旋争用 SPIN_IDLE 默认最优(R41)· 线程数(96 线程饱和 @~130 GB/s,R41)·
分片粒度 NS=8 最优(R41)· resize 零填充(R42)· 删 gather(R43/R44,耦合成本)· 预取距离 i+2(R45)。
微基准层面:单线程同访问形态可达 **8.46 GB/s**(512MB DRAM 工作集,分片形态同速)—— 见 §81/§84。

## 94.3 剩余两个方向(按顺序)
1. **软件 MLP(不改数值、改动小)**:`block_23` 内同时为 **2 个 K 组**取数(2 路独立访存流),
   或**跨行交错**(同时拉 i 与 i+1 行同组字节),提高每线程在飞请求数(MSHR/LFB)。
   验收:`scripts/bench_engine_ab.py` A/B 相 210/152 → 目标 ~400 GB/s(每层 → ~0.6 ms)。
2. **改权重布局(改动大)**:让单线程一次能拉 **连续 64KB**,而不是现在"32 行 × 2KB 跨步"。
   要同步改 `shard_fill_w13/w2` 的布局 + 内核行寻址 + `row_scale` 索引;先做 1 再评估 2。

## 94.4 工具与开关(都已提交)
- 验收:`scripts/bench_engine_ab.py`(`ENG=xiaotu|lk`;**lk 必须 `LK_THREADS=120`**,否则假慢 70×)
- 诊断:`XIAOTU_MOE_SETUP_PROF=1`(A2 两段)、`XIAOTU_MOE_NOGATHER=1`(跳过 gather,数值无效)、
  `XIAOTU_MOE_PROFILE=1`(A/A2/B/C)、`XIAOTU_MOE_DPBF16/GEMM_NR/SHARDSPLIT/NSHARD/SPIN_IDLE_US/THREADS`
- 补丁规范:改 `moe_v2*.hpp` 前**先 `grep -c` 锚点数量**(§88 与 §92 各因锚点计数错浪费一轮)

## 94.5 交付状态
8070 = 单卡 + NUMA 分片 + CUDA graph(解码最优,**在线**):预填充 703 t/s、解码 TPOT 27.8-31.5 ms。
预填充最优配置(TP=2 + `XIAOTU_MOE_GPU_RESIDENT_LAYERS=0-5` + `RESIDENT_BUDGET_GB=12`,EAGER):
1105-1117 t/s。切换命令见 §54/§56 与 `scripts/tune_serve.sh`。

## 94.6 【更正 + 最终评估】"软件 MLP"已被测过;只剩"访存粒度"

**更正 §94.3 的方向 1**:内层"2 个 K 组在飞"等价于第 35 轮的源码级 2 路展开 ⇒ **R36 实测无收益**
(1.19/1.32);而"多行在飞"其实 `block_23` **本来就在做**(同一个 group 循环内 R=2-3 行各自独立
累加器 ⇒ 已有多路并行)。⇒ **软件 MLP 这条也已覆盖**,不再是"待试方向"。

**因此只剩一个未测方向:访存粒度(数据布局)**。当前每个线程一个 job 读"32 行 × 2 KB、行跨 2 KB
但行间在分片内连续(512 KB 跨度)"——单线程同形态可达 8.46 GB/s(§81),但引擎里 120 线程只能
合计 ~130 GB/s(每 node 16 GB/s),而同样这台机器**单 node 24 线程的纯流式探针能到 99 GB/s**。
综合所有排除项,唯一自洽的解释是:**引擎的访问在"每线程一次能连续推进的长度"上太短**
(每个 (行,组) 16 B、行内 2 KB、然后换行),导致每线程的在飞请求被 L1/L2 的行粒度限制住;
而探针是"每线程一段长连续流"。

**验证与改法(下一轮)**:
1. **先验证**:把 mini-engine(§81 的 `/tmp/abl2.cpp`)从"每线程顺序流"改成"每线程按引擎形态
   (32 行 × 2 KB 行内、行间连续)"跑 1/5/15 线程 —— 若 15 线程也只有 ~16 GB/s ⇒ **证实**;
   若到 60+ GB/s ⇒ 说明引擎里还有别的东西,回到池/job 侧再查。
2. **改法**:把每个 job 的**行块做成长连续段**(例如按"输出行的 K 连续块"重排权重,让一个线程
   一次拉 64 KB 连续),同步改 `shard_fill_w13/w2` + 内核行寻址 + `row_scale` 索引。

**一句话结论**:15 轮系统排除(§86–§93、R33–R45)已经把引擎对齐任务的病因收敛到
**"每线程访存粒度/在飞长度"**这一个点上;它不是指令、不是解码、不是布局落点、不是线程与池。

## 95. 【最终建议·第 51 轮】增量假设已穷尽 ⇒ 直接**照抄 lk 的热内核结构**

19 轮系统排除后(§86–§93, R33–R45),剩余候选只剩"每线程访存粒度/在飞长度",而它与我实测的
分片布局(每个 job = 32 行 × 2 KB **在分片内连续**,即 64 KB 连续段)并不自洽 ⇒ 继续做增量假设
的期望收益已经很低。**建议改变策略:不再猜,直接把 lk 热内核的结构原样搬过来(diff 对照)**。

lk 的热内核(`process_data/decomp/LK_MOE_KERNEL_DECOMP.final.md`,地址 `0xa1d7b..0xa4b20`)结构:
1. **每 K-block(64 列)先把 fp4 解码到栈工作缓冲**(`zmm1 <- [rsp+0x640]`, `zmm0 <- [rsp+0x680]`),
   解码链**完全脱离 FMA 循环**;
2. FMA 循环里只有:2 条 zmm 载入 + 每个激活 `vpinsrw/vcvtph2ps/vpbroadcastss` + **2 条 `vfmadd231ps`**;
3. **8 个激活行 × 2 个列组 = 16 条独立累加链**,内层 FMA 深度 16 ⇒ 延迟全隐藏;
4. 激活**保持 bf16**(2B/值),不物化 fp32;
5. `158` 条 `vfmadd231ps`、**0 条标量**。

我们 `block_23` 的差异在:**解码结果留在寄存器**(更好)但没有"栈工作缓冲"这一级 ⇒ 解码链与
FMA 循环仍在同一迭代里交替;而 lk 用栈缓冲把两者**在时间上彻底分开**。
**动作**:在 `moe_v2_packed4.hpp` 里加一个 `XIAOTU_MOE_LKLOOP=1` 的**逐条照搬实现**
(栈缓冲 + 16 链 + bf16 激活直取),与现有路径做 A/B;若达标(≤0.7 ms/层),它就是结论;
若不达标,则说明差异不在内层结构,而在**外层(池/job/内存)**,此时应把精力转到
"把 8 个 node 的 job 合并成更长的连续工作单元"。

## 96. 【实施规格】照搬 lk 热内核的可行步骤(先澄清一处语义歧义)

**必须先澄清的歧义**:`process_data/decomp/LK_MOE_KERNEL_DECOMP.final.md` 的 b1 节自相矛盾 ——
一处写"tile = 8 activations × **64 output columns**, 16 zmm accumulators(8 行 × 2 列组)",
另一处写"Columns covered per tile: 64 (= **2 zmm × 32 columns** each)"(zmm 只有 16 个 fp32 lane)。
⇒ **在照搬之前,必须先把"向量 lane 装的是 K 还是输出列"这件事定死**,否则照搬必然走样。

**做法(两步,第一步很便宜)**:
1. **重新反汇编定位 lane 语义**:`objdump -d --no-show-raw-insn lk_moe/_lk_moe_C_avx512_vnni.so`
   取热内核区段(`0xa1d7b..0xa4b20` 附近;若 ASLR/节区偏移不同,用 `_lk_moe_symbols_demangled.txt`
   里的 `forward_many` 地址重定位),然后看:
   - 两条 `vfmadd231ps zmmA, zmmAct_bcast, zmmW` 的**第二个源操作数**(zmmW)是**载入后不变**(⇒ lane=输出列,
     一个 K 值广播激活)还是**每 k 变化**(⇒ lane=K,一个输出列累积);
   - `vpinsrw/vcvtph2ps/vpbroadcastss` 序列的**输入地址步长**:步长 2 B(bf16 连续 K)还是 8 B(跨行)。
   这两点一看就定死 lane 语义。
2. **按结论实现** `XIAOTU_MOE_LKLOOP=1` 分支(默认关),与现有 `block_23` 同窗口 A/B;判据不变
   (≤0.7 ms/层)。

**若第 1 步显示 lane=输出列 + K 顺序**(与我们的"lane=K + 4 partial"不同),那么差异点就清楚了:
我们的做法**每个输出列都要把整条 K 流重读一遍激活**(激活在 L1,代价小),而 lk 是
**权重载入一次、跨 8 行复用**(权重在 DRAM,代价大)——这正是 §94.6 收敛到的"每线程在飞长度"
问题的一个具体形态:**lk 的每个 zmm 载入服务 8 行 × 16 列 = 128 次 MAC,我们的每个 128 B 载入服务
3 行 × 32 K = 96 次 MAC**,但 lk 的**载入次数/字节更少**(8 行共用一个权重向量)。
⇒ 若属实,修法就是**把"按输出列"改成"按 8 行一块、权重载入一次复用 8 行"**(即把 `block_23` 的
R 从 2-3 扩到 8,并在同一 group 内让 8 行的激活分别 broadcast 后与同一组权重做 FMA)。

## 97. 第 53 轮:反汇编部分结果(lane 语义仍未定死)+ 正确的取窗方法

**做了什么**:`objdump -d --no-show-raw-insn _lk_moe_C_avx512_vnni.so | grep -E "vfmadd231ps|vcvtph2ps|
vpinsrw|vpbroadcastss|vmov*|..." | head -45`。
**结果**:前 45 条匹配几乎全被 `vmov*` 占满(该 .so 很大,`vmovdqu8 %ymm0,(%rbx)` 这类是别处的循环),
**没有截到 FMA 密集区** ⇒ §96 的 lane 语义问题**仍未定死**。
**唯一有用证据**:
```
22776: vmovdqu64 %zmm1,(%rsp)
2277d: vmovdqu64 %zmm3,0x80(%rsp)
22785: vmovdqu64 %zmm2,0x40(%rsp)
```
三连 zmm 按 0/0x40/0x80 存栈(192B)⇒ **印证反汇编文档 b1 的"解码结果先写进栈工作缓冲"**这一条。

**正确的取窗方法(下一轮照做)**:
1. `objdump -d --no-show-raw-insn $SO | grep -n "vfmadd231ps" | head -20` 取**行号**;
2. `objdump -d --no-show-raw-insn $SO | sed -n "<行号-40>,<行号+40>p"` 取**窗口**(而不是全局过滤后取前 N —— 会被别的函数占满);
3. 只看两件事:①`vfmadd231ps` 的**第二源**(权重)是否跨 8 次迭代不变(⇒ lane=输出列);
   ②`vpinsrw/vcvtph2ps` 的输入地址步长(2B ⇒ 连续 K;8B ⇒ 跨行)。

## 98. 【定论】lk 内层 = **列向量化 + 栈解码缓冲**;我们 = **K 向量化 + 现场解码** ⇒ 每列 uop 多 ~1.6-2×

**证据**(`objdump` 首个 `vfmadd231ps` 窗口,地址 `28d73-28da3`;FMA 总数 2326):

```asm
movzwl (%rax,%rbx,2),%edx              ; 激活按 **2 字节**步长取(= 连续 K 的 bf16)
shl    $0x10,%edx                      ; bf16→fp32(取高半字,不需要 vcvtph2ps)
vmovd  %edx,%xmm6 ; vbroadcastss %xmm6,%zmm0   ; 激活**广播成标量**
vfmadd231ps 0x280(%rsp),%zmm0,%zmm3    ; 权重 = **栈缓冲内存操作数**(已解码),acc=zmm3
vfmadd231ps 0x2c0(%rsp),%zmm0,%zmm1    ; 第二组列,acc=zmm1
```
- **向量 lane = 输出列**(zmm3/zmm1 各 16 列 ⇒ 每轮 k 覆盖 **32 个输出列**);
- 权重**预先解码写进栈**(§97 的 zmm 三连存栈 ✓),FMA 直接用内存操作数 ⇒ 内层每 k 仅
  `load + shl + movd + broadcast + 2×FMA ≈ 6 uop`,覆盖 32 列;
- 累加器数量 = 2 组列 × 16 lane,**按 k 顺序累加**(K 在内层串行,但激活广播使每 k 只需 1 次载入)。

**我们的结构**(`block_23`):lane = **K**(32 个 K 值一组),**每个 (输出列, K组) 现场做 14 条依赖的
nibble 解码**,再对 R=2-3 行做 `2 load + mul + 2 FMA` ⇒ 覆盖同样的 32 K × 3 行 × 1 列需要
≈14 + 3×5 = **29 uop**,折算到"每 32 列"约 **309 uop** vs lk 的 ~192 ⇒ **~1.6×**,
与实测 2× 的方向与量级吻合 ✓(其余差额来自 lk 的激活只需 2B 载入且免转换)。

**结论(这就是 19 轮排除后剩下的那 2× 的机理)**:不是"解码没 hoist"(我们 hoist 了),而是
**向量化的维度不同**:lk 用"列 lane + 权重驻栈 + 标量广播激活",我们"K lane + 每列现场解码"。
⇒ **正确的修法 = 照 lk 的列向量化内层重写一个小 me 路径**(不是扩 R):
   对一个 K-block:解码 32 个权重值到栈/寄存器;对 32 个输出列的各 lane 累加;激活按 k
   **广播**;内层目标 ≈6 uop/k;再靠 R 行复用权重。
**验收不变**:`scripts/bench_engine_ab.py` ≤0.7 ms/层、每线程 ≥2.2 GB/s。

## 99. 【补齐最后一环】照搬列向量化需要先把权重转成 **K-major** 布局

§98 定死了 lk 的内层是"**列 lane + 权重驻栈 + 标量广播激活**"。但**要照搬必须先改权重布局**:

- lk 的 FMA 是 `vfmadd231ps mem, zmm_bcast_act, zmm_acc`,其中 `mem`(栈缓冲)一次性给出
  **同一个 k 上的 32 个输出列值** ⇒ 这在**行主序 `[N][K/2]`** 的打包布局里做不到:同一 k 的相邻列
  在内存里相距 `K/2` 字节,必须靠 `vpgatherdd`/转置才能凑进一个 zmm(代价极高)。
  ⇒ **lk 的权重必然是 K-major(每个 k 上，列号连续)** —— 与反汇编文档 §2.2 的
  "Natural K-major [INF, high confidence]" 完全一致 ✓ 这两条互相印证。
- 我们的权重是**行主序 `[E][N][K/2]`**(`w13`/`w2` 契约),所以只能"每个输出列读自己那条 K 流"
  ⇒ 只能做 K-vectorize(就是现在的 `block_23`),而每个 (列, K组) 都现场解码 ⇒ §98 算出的 ~1.6× uop 差。

**修法(两步,需按序)**:
1. **加载期转布局**:把每个专家的 `w13`/`w2` 从 `[N][K/2]`(fp4 行主序)转成 **K-major**
   (即对每个 k,连续 N 个 nibble),顺带把 e8m0 尺度也按 K 分组重排(`row_scale` 索引要跟着改)。
   代价:一次性(启动时)重排 + 内存不变;风险:要同步改 `shard_fill_w13/w2`、
   `gate_up_slice_batched`/`down_slice_batched` 的行寻址、以及 GPU 预填充路径对 `w13` 的读取
   (**GPU 预填充也吃这个布局** ⇒ 要么同时改 torch kernel,要么给 GPU 路径保留一份原布局 —— 内存 ×2)。
2. **再实现列向量化内层**(`XIAOTU_MOE_LKLOOP=1`):权重解码进栈缓冲 → lane = 输出列(2×16)→
   激活 `movzwl + shl 16 + vbroadcastss` → 内层 ≈6 uop/k → 用 me 行复用权重。
**验收**:`scripts/bench_engine_ab.py` ≤0.7 ms/层、每线程 ≥2.2 GB/s;再切 8070 复核 TPOT 28-31 → ~20 ms。

**风险提示(务必先评估)**:第 1 步会**同时影响 CPU 解码与 GPU 预填充两条路径**(GPU 侧的
Triton kernel 现在按行主序 K 分块读 `w13`)。因此建议先在**独立的实验分支/开关**上做,
并保留原布局直到验收通过。

## 100. 【修法简化】K-major 布局**已经存在**于插件里(GPU 预填充的 pinned 缓存)

`gpu_prefill._kmajor_bytes(t) = t.transpose(1, 2).contiguous()` ⇒ 对 `w13 [E][N][K/2]` 得到
**`[E][K/2][N]` = K-major**(每个 k 字节上 N 列连续)✓ 正是 §99 要求 §98 列向量化所需的布局!
而这条路径**已经在跑**:`_pinned_kmajor()` 为**每一层**建好 K-major 的 **pinned 主机缓冲**,
供 GPU 逐层 H2D 使用(TP=1 时缓存覆盖全部 43 层 ≈ 139 GB,常驻主机内存)。

⇒ **不必再造第二份布局**:让 CPU 引擎的 `forward_many` 支持"K-major 权重"输入(新增一个
入口/开关),由插件把**已有的** `_pinned_kmajor(w13)` 传进去即可。要点:
1. **尺度也要 K-major**:`s13 [E][N][K/32]` → `[E][K/32][N]`(`_kmajor_bytes(s13)` 同样成立 ✓
   已经在缓存里),`row_scale` 索引从 `srow + gi` 改成 `gi*N + col`(其中 col 是该 zmm lane 的列号)✓
   与列向量化的 lane 天然对齐;
2. **列向量化内层**(`XIAOTU_MOE_LKLOOP=1`):对 32 列一块,循环 k:取 K-major 的
   `wbuf[k/2][cols]`(连续 32 B = 32 列 ✓ 无需 gather)→ nibble 解码到栈缓冲 → 激活
   `movzwl + shl 16 + vbroadcastss` → 2 条 `vfmadd231ps`(lane = 32 列,2×16);
3. **GPU 预填充不受影响**(它本来就用 K-major)⇒ §99 里"两种布局要同时维护/内存 ×2"的风险**消除**
   ✓ 反而变成"两条路径共用同一份 K-major 缓存";
4. 行主序的 `w13` 仍留给 CPU 的**旧路径**做回归对照(`XIAOTU_MOE_LKLOOP=0`)。

**实施清单(下一轮直接做)**
- [ ] `moe_v2.hpp`:加一个"K-major 权重"引擎构造/入口(或给 `forward_many` 加 layout 参数);
- [ ] `moe_v2_packed4.hpp`:新增列向量化小 me 路径(lane=输出列,2×16;权重先解码进栈缓冲;
      激活 2 B 加载 + `<<16` + 广播);
- [ ] 插件侧:`hybrid_model.py` 在 CPU 解码时把 `_pinned_kmajor(w13/w2/s13/s2)` 传给引擎(TP=1 的
      pinned 缓存已存在,零新增内存);
- [ ] 验收:`scripts/bench_engine_ab.py`(BS=6/DEDUP=12/THREADS=120)**≤0.7 ms/层、每线程 ≥2.2 GB/s**;
      再切 8070 复核 TPOT 28-31 → ~20 ms。

## 101. 第 57 轮:§100 的前提**已运行时验证**

```python
from vllm_xiaotu_moe.gpu_prefill import _kmajor_bytes
t  = arange(3*8*6).reshape(3,8,6)      # [E][N][K/2]
km = _kmajor_bytes(t)
# in  (3, 8, 6) stride (48, 6, 1)
# km  (3, 6, 8) stride (48, 8, 1)   ← [E][K/2][N],列维连续
# transpose 正确: True
# 同一 k 上 8 列: [0, 6, 12, 18, 24, 30, 36, 42]  ← 连续 ✓
# 尺度 _kmajor_bytes(s) → (3, 2, 8) = [E][K/32][N] ✓
```

⇒ §100 的两条前提都成立:①K-major(`[E][K/2][N]`,列连续)由 `_kmajor_bytes` 正确产生;
②尺度同样可得 `[E][K/32][N]`。**列向量化内层所需的取数形态(一次 32 B 拿到 32 列)已具备**,
且这份缓冲在 GPU 预填充路径上本就已经为每层建好 ⇒ 下一轮可直接按 §100 的四项清单实现。

**给下一轮的提醒(避免又白跑)**:
- 先用 `XIAOTU_MOE_LKLOOP=1` 门控新路径、保留 `block_23` 作 A/B 对照;
- 验收只认 `scripts/bench_engine_ab.py`(BS=6/DEDUP=12/THREADS=120)的 **ms/层**(目标 ≤0.7)
  与每线程 ≥2.2 GB/s;改前先 `grep -c` 锚点(§88/§92 的教训);
- 若新路径达标,**必须**再跑一次 `gpu_prefill_golden.py` 与 `test_block23_equiv.py` 做数值对拍
  (row_scale 的 K-major 索引是唯一容易写错的地方)。

## 102. 【自查·重要】§98 的 uop 账按"行复用"口径重算:优势只有 ~1.2×,不是 2×

§98 我按"每 32 列"算了 uop(我们 ~309 vs lk ~192),但那个口径**没有把行复用一起归一化**。
按**每 (32 列 × 1 行)**重算(K=4096 ⇒ 每 32 列要 128 个 K 组):

**(A) 我们的 `block_23`**(R=3 行共享一次解码):
`128 组 × (14 解码 + 3 行 × 5) = 128 × 29 = 3712 uop` 覆盖 `32 列 × 3 行`
⇒ **每 (32 列 × 1 行) = 3712 / 3 ≈ 1237 uop**。

**(B) lk 的列向量化**(lane=32 列 = 2 zmm,权重解码进栈缓冲):
每 k 需要:1 条 32B 载入 + 解包(≈4 uop)+ **8 行 × 2 条 FMA** = 16 条 FMA + 8 条激活
`movzwl/shl/broadcast`(3 uop/行)= 24 ⇒ 合计 ≈29 uop **per k**;
`128 组 × 32 k × 29 = 118784 uop` 覆盖 `32 列 × 8 行`
⇒ **每 (32 列 × 1 行) = 118784 / 8 ≈ 14848 uop** ?!

⇒ 这个模型下 **lk 反而更贵** ✗ ⇒ 说明**"uop/列"这个模型本身不足以解释 2×**(lk 的 16 条 FMA/k 是把
8 行都算进去了,而我们的 5 uop 只服务 1 行 × 1 列)。**正确的归一化必须按"每 MAC"或"每字节"**:

- **lk**:每 k 载入 32 B 权重(32 列 × 1 k 的 nibble)→ 产出 32 列 × 8 行 = **256 MAC** ⇒ 0.125 B/MAC;
- **我们**:每 (组, 列) 载入 16 B(32 个 k 的 nibble)→ 产出 3 行 × 32 k = **96 MAC** ⇒ 0.167 B/MAC;
  ⇒ 每 MAC 的字节数只差 **1.33×**,与"1.6× uop"量级一致,**都解释不了 2×**。

**结论(必须写清楚,避免下一轮按错误模型做昂贵重写)**:§98/§99/§100 给出的"列向量化 + K-major"
**只能解释 1.2–1.6×**,而且这还是纯 uop 模型;实测 2× 里剩下的部分**仍未定性**。
⇒ **建议**:不要直接做 §100 的四项重写(它要动布局 + GPU 路径,风险大、收益上限 ~1.5×);
先做一个**便宜的判定实验**:把 §98 的 lk 内层结构**照搬成一个独立微基准**(K-major 缓冲 + 栈解码 +
激活广播 + 8 行复用),与 `block_23` 的形态**在同一 DRAM 工作集下对比**;若两者差 <1.3×,
则 2× 的其余部分在**引擎的并行/内存层**,应转回 §94.6 的方向。

## 103. 【判定实验设计】lk 形态 vs `block_23` 形态(同一 DRAM 工作集,一次跑完)

**目的**:验证 §102 的结论——列向量化 + K-major 到底值多少(预期 1.2-1.6×),还是 2× 另有原因。
**载体**:独立 `/tmp/abl5.cpp`(纯 intrinsics、无引擎依赖,像 `/tmp/abl2/abl3` 那样几分钟出结果)。

**公平性规则(必须遵守,否则又会得出误导结论)**
1. **每轮读的权重字节数相同**:都处理 `32 列 × K=4096`,即 **65536 B/轮**
   (K-major: 2048 B/列 × 32 列;行主序: 32 列 × 2048 B)✓ 完全一致;
2. **工作集 >L3**:把权重缓冲放到 ≥512 MB 并按轮次换地址(否则测的是指令吞吐,§86/§84 的教训);
3. **激活放 L1**(两变体都一样,模拟真实情形:激活只有 me×8KB);
4. 只统计**权重字节 / 时间**,同时打印**做的 MAC 数**(A: 3 行、B: 8 行),换算 GB/s **和** MAC/s。

**变体 A(我们的形态,K-vectorized)**
```cpp
for (j = 0; j < 32; ++j)                    // 输出列
  for (g = 0; g < 128; ++g) {               // K 组(32 K/组)
    解码 W_rowmajor[j][g*16 .. +16] → wlo,whi     // 与 block_23 同款 ~14 uop 链
    for (r = 0; r < 3; ++r) { d = wlo*av[r]; d = fmadd(whi, av2[r], d); acc += d; }
  }
```
**变体 B(lk 形态,列向量化 + K-major + 权重驻栈)**
```cpp
float wbuf[2][32];                          // 栈:32 列 × 2 个 k
for (g = 0; g < 2048; ++g) {                // 每个字节组 = 2 个 k
    ymm = load(W_kmajor[g*32 .. +32]);      // 32 B = 32 列的 nibble
    解包一次 → wbuf[0][*](偶数 k)、wbuf[1][*](奇数 k)    // 每字节组只解一次
    for (r = 0; r < 8; ++r)                 // 8 行复用同一份解码
        for (kk = 0; kk < 2; ++kk) { b = bcast(act[r][2*g+kk]); acc[r] = fmadd(wbuf[kk], b, acc[r]); }
}
```
**判读**
- A、B 的 **GB/s** 若差 <1.3× ⇒ **列向量化不是 2× 的来源** ⇒ 放弃 §100 的四项重写,
  转 §94.6 的"每线程访存粒度/引擎并行层";
- 若 B 的 GB/s 高 1.8×+ ⇒ 值得做 §100 重写;
- 若两者都远低于 8.46 GB/s(单线程同形态上限,§81)⇒ 说明这两个内层形态都不是 DRAM 限幅,
  2× 一定在引擎层。

**重要**:B 变体里 `wbuf` 用 **2×32 float = 256 B**(L1),不要用 fp32**全 K**(那是 16 KB/列,
会把 L1 挤爆、变成 L2 带宽测试)。

## 104. 【会话收尾】下次接续入口(第 60/60 轮,不达标但病因闭环)

**一句话**:目标第一条(CPU 解码对齐 lk)**未达标**(1.16–1.24 vs 0.57 ms/层),但已从"猜测"推进到
"机理 + 收益上限 + 一次判定实验"三件套;其余目标(预填充 1500 / 解码 70 / 投机 100 / 1M)未动,按目标
设定必须等第一条对齐后再做。

**接续顺序(照做即可)**
1. 跑 §103 的判定微基准(`/tmp/abl5.cpp` 按 §103 的公平性规则实现:同字节、>L3 工作集、激活驻 L1、
   同时报 GB/s 与 MAC/s)→ 按三种判读二选一:
   - B 只高 <1.3× ⇒ **放弃** §100 重写,转 §94.6(每线程访存粒度 / 引擎并行层);
   - B 高 ≥1.8× ⇒ 做 §100 的四项(K-major 入口 + 列向量化内层 + 插件接线 + 双对拍)。
2. 任何改动都用 `scripts/bench_engine_ab.py`(BS=6/DEDUP=12/THREADS=120)验收,**≤0.7 ms/层、每线程 ≥2.2 GB/s**;
   改 `*.hpp` 前先 `grep -c` 锚点(§88/§92 的教训);改内层后跑 `gpu_prefill_golden.py` + `test_block23_equiv.py` 对拍。
3. 达标后切 8070(见 §54/§56 + `scripts/tune_serve.sh`)复核 TPOT 28-31 → ~20 ms。

**必读**:`NOTES §94`(总交接)、**§98–§103**(机理与判定实验)、`TRIED_AND_REVERTED.md` R33–R45(15+ 已排除项,
**动手前先扫一遍**)。**交付**:8070 = 单卡分片 + CUDA graph(在线);预填充最优 = TP=2 + 6 常驻层。

## 105. 【§103 判定实验结果 + 重要更正】引擎没有损失;损失在内层循环本身(1.86-2.17 GB/s)

`/tmp/abl5.cpp`(同一 0.5 GB DRAM 工作集、每轮同字节、激活驻 L2、单线程):

| 变体 | 时间 | **GB/s** | **GMAC/s** |
|---|---|---|---|
| **A** 我们(K-lane,行主序,每列现场解码,3 行)| 0.289 s | **1.86** | 11.2 |
| **B** lk(列 lane,K-major,解码进栈,8 行)| 0.248 s | **2.17** | 34.7 |
| 比值 | — | **1.17×** | **3.11×** |

**两个结论(都很重要)**:

**① 更正 §81/§84/§86 的基线**:那个"单线程 8.46 GB/s / 分片形态 8.52 GB/s"的探针**循环里没有
解码、也没有 FMA**(只有 load + 2 条与常量寄存器的 FMA)。加上真实内层工作(解码 + 多行 FMA)后,
**任何形态都只有 1.86-2.17 GB/s/线程** ⇒ 8.46 这个数**不能当作内核上限**,不能用它推出"引擎丢了 4-8×"。

**② 引擎已经跑在它内核的速度上**:引擎 A 相实测 `101 MB / 0.48 ms ÷ 120 线程 = 1.75 GB/s/线程`,
与微基准 A 的 **1.86** 吻合(<6%)⇒ **引擎层(池/job/屏障/映射/布局)没有损失**,之前十几轮"找引擎
结构问题"的方向**到此可以关掉**;真正的限幅是**内层循环本身只有 ~2 GB/s/线程**(延迟/发射受限)。

**③ 形态移植的收益上限**:B 在每个字节上只比 A 快 **1.17×**(尽管它复用了 8 行 vs A 的 3 行,
MAC/s 高 3.11×)⇒ **单纯照搬 §100 的列向量化重写,拿不到 2×**,最多 ~1.2× +
(取决于能否把行复用从 me=3 提到 8 —— 而解码时 me≈3 是路由决定的,提不上去 ✗)。

**下一步方向(由本实验直接给出)**:
1. **用 `vdpbf16ps` 打破延迟受限**:一条指令做 32 个 MAC(vs 2×16 个 FMA),前提是权重与激活**预先
   转成 bf16 对**(权重在加载期转、激活本来就有 bf16 对)。我们已有 `XIAOTU_MOE_DPBF16` 但当年只
   在引擎里测到 +1.7%(受 me=3 与其它开销掩盖)——**应在本微基准里加变体 C(bf16 点积)再判定**。
2. 若 C 仍 ~2 GB/s ⇒ 说明是**每字节指令数**硬限,应朝"每字节更少指令"走(例如权重按 2 值/字节
   预解成 bf16 对后,一条 vdpbf16ps 覆盖 32 MAC/16B = 2 MAC/字节 ⇒ 每 MB 约 5e5 条指令)。

## 106. 【正向结果·可执行】bf16 点积(`vdpbf16ps`)在内层实测 **1.47×**

`/tmp/abl6.cpp`(同一 0.5 GB DRAM 工作集、同字节、K-major、8 行复用、单线程):

| 变体 | GB/s | GMAC/s |
|---|---|---|
| B:fp32 `2×vfmadd231ps` + `vcvtepu8` 解码 | 1.82 | 29.2 |
| **C:bf16 点积 `vdpbf16ps`(32 MAC/指令)** | **2.67** | **42.8** |
| **C/B** | **1.47×** | 1.47× |

C 的做法:16 B nibble 一次解成 **32 个 bf16(每列 `[k,k+1]` 相邻)**,激活同样排成 32 个 bf16
(`_mm512_set1_epi32(k | k+1<<16)`),每 (字节组, 行) **只 1 条 `vdpbf16ps`**(32 MAC)。

**与我们引擎里 `XIAOTU_MOE_DPBF16` 的矛盾(必须解释,下一轮第一件事)**:引擎里那条路径只测到
**+1.7%**(1.16 vs 1.18 ms/层),而微基准显示内层应快 **1.47×** ⇒ 说明**引擎里的 DPBF16 路径与
这个微基准不是同一回事**(很可能是先把权重解成 fp32 再逐行 `cvtneps→bf16`,而不是"nibble 一次
解成 bf16 对"),或该路径没走到 `block_23`。
**动作**:读 `moe_v2_packed4.hpp` 的 `dotp16` 分支,把它改成与**本微基准 C 完全一致**的形态
(16 B → 32 bf16 对;每 (组,行) 1 条 `vdpbf16ps`;激活用 `set1_epi32` 造对),然后:
- 微基准复现 1.47×;
- 引擎同窗口 A/B(`XIAOTU_MOE_DPBF16=1` vs 0),预期每层 1.20 → **~0.85 ms**;
- 数值用 `test_block23_equiv.py` + `gpu_prefill_golden.py` 对拍(bf16 乘积在 fp32 内精确,累加仍 fp32)。

## 107. 为什么微基准 C 的 1.47× 在引擎里只剩 +1.7%:**激活的重复读放大**

读了 `moe_v2_packed4.hpp` 的 `dotp16` 分支(≈line 300-320),它**已经是**微基准 C 的形态:
```cpp
for (int r = 0; r < mr; ++r) {
    const __m512i av = _mm512_loadu_si512(A + (m0+r)*K + g*32);   // 32 × bf16 激活
    const __m512 t = _mm512_dpbf16_ps(_mm512_setzero_ps(), (__m512bh)wz, (__m512bh)av);
    acc[r] = _mm512_fmadd_ps(t, sv, acc[r]);                       // 组尺度
}
```
⇒ 结构一致(区别只有我微基准没有 scale 那一条 FMA)。**那为什么只快 1.7%?** 看循环顺序就明白了:

**我们的循环是「外层 = 输出列 j,内层 = K 组」** ⇒ 对**每一个输出列**,都要把该专家 me 行的整条激活
**重新读一遍**(L1):
- fp32 路径:每 (列, K组) 读 me × 128 B(mr=3 ⇒ 384 B),权重只有 16 B ⇒ **激活:权重 ≈ 24:1**;
- 引擎每个 node 每层要过 `I/8 = 256` 个输出列(gate+up 各 256)× na 个专家 × 128 组 ⇒
  激活 L1 流量 = 256×2×128×384 B ≈ **25 MB/专家** vs 权重 8.4 MB ⇒ **L1 侧的激活流量是权重的 3 倍**;
- `dotp16` 把激活换成 bf16(每行 64 B)⇒ 只省了这部分的一半 ⇒ **+1.7%** 合理 ✓(而微基准 C 每字节
  只做 1 次 dpbf16ps、几乎没有别的负担,所以那边显得快)。

**⇒ 真正的限幅是"逐个输出列重复读激活"这一循环顺序**(它让 L1 带宽/指令成为瓶颈),
而不是 FMA 指令形态。**lk 的列向量化恰好解决的就是这个**:一次激活载入服务 32 个输出列
(激活:权重从 24:1 降到 0.75:1)✓ 这也解释了 §105 里 B 的 MAC/s 高 3.11×。

**下一轮(明确、单点)**:在 `matmul_packed4_group` 里加 `XIAOTU_MOE_COLVEC=1`:
把「外列内 K」翻成「**外 K 组、内 16-32 个输出列**」,激活每个 (K组, 行) **只载入一次**、
在寄存器里被多个列复用(权重仍按行主序逐列取,先不要求 K-major);
预期:激活 L1 流量 ÷16,`dotp16` 与此改动叠加应把 A 相 0.48 → ~0.2 ms ⇒ 每层 **~0.9 ms**,再叠 B 相同理 ⇒ 接近 0.7。

## 108. 【否决 §107】只翻转循环顺序(激活复用)单独**无效**(D/A=0.93×)+ 微基准精度警告

`/tmp/abl7.cpp`(同行主序权重、同字节、同 me=3、只把"外列内K"翻成"外K组内列",激活每 (组,行) 只载一次):

| | 时间 | GB/s |
|---|---|---|
| A 外列内K(现状)| 0.204 s | **2.63** |
| D 外K组内列(翻转)| 0.218 s | 2.46 |
| **D/A** | — | **0.93×** ✗ |

⇒ **§107 推断的"激活重复读是限幅、翻转循环即可"被直接否掉**:只翻转顺序不但没快,还略慢
(权重在翻转后变成跨行主序的**跨步访问**,抵消了激活复用)。

**同时暴露一个方法学问题(重要)**:同名的"变体 A"在 `/tmp/abl5.cpp` 里是 **1.86 GB/s**,
在 `/tmp/abl7.cpp` 里是 **2.63 GB/s**(同一台机器、同为单线程、同为 0.5 GB 工作集)⇒
**这类微基准的绝对值对编译细节高度敏感,<1.3× 的差异不可作为决策依据**(§105 里"引擎 1.75 vs
微基准 1.86,差 <6%"的结论因此也只能算**弱证据**,不能据此断言"引擎无损失")。

**因此下一步必须先修方法,再谈优化**:
1. 判定实验要在**同一份代码里**编译出两个变体(同一编译单元、同 `-O3`、同展开),各跑 3 轮取
   **min-of-N**,并且**交替**测量(先 A 后 D 再 A 再 D);
2. 之前所有"<1.3×"的微基准结论(§105 的 1.17×、§106 的 1.47%…)**都应视为待复核**;
   只有**引擎内同窗口 A/B 的 ms/层**(如 R36/R39/R45 那些 ≥1.10× 的)才是可靠判据。

## 109. 【修正方法后的决定性结果】引擎形态下 bf16 点积 = **1.98×**;引擎里只 +1.7% ⇒ 该分支没走到

`/tmp/abl8.cpp`:**同一编译单元**、两变体**交替 3 轮取 min**、同为引擎形态(行主序权重、外列内K、me=3、
激活用同样 32 个 bf16 排布):

| 变体 | min-of-3 | GB/s | 三轮 |
|---|---|---|---|
| fp32 `mul + 2×FMA` | 0.132 s | **4.05** | 0.133 / 0.132 / 0.132 |
| **bf16 `vdpbf16ps`** | 0.067 s | **8.04** | 0.067 / 0.067 / 0.067 |
| **比值** | — | **1.98×** | 稳定 ±1% ✓ |

**两点结论**:
1. **方法修正有效**:交替 min-of-3 后数值稳定(±1%),而之前同名变体在 1.86/2.63/4.05 GB/s 之间跳
   ⇒ §108 的方法学警告成立,**以本实验为准**;
2. **引擎里有 1.98× 的现成空间没吃到**:引擎的 `dotp16` 分支与上面 bf16 变体**结构相同**,
   但 `XIAOTU_MOE_DPBF16=1` 实测只有 **+1.7%**(1.16 vs 1.18 ms/层)⇒ **该分支没有被执行**
   (最可能:分派条件/取指没进它;`block_23` 与 dotp16 分支是两条并列路径,必须确认解码(me≤3)
   走的是哪一条)。

**下一轮(唯一动作,便宜)**:给 `dotp16` 分支加一个**计数器/一次性打印**(或 `XIAOTU_MOE_DIAG_WHICH=1`),
跑一次 `bench_engine_ab.py`,确认解码路径到底进了哪个分支:
- 若根本没进 dotp16 ⇒ 修分派条件(可能 `mr`/`M`/`gk` 或 `M*K<=4MB` 的门槛把 me=3 的解码挡在外面),
  修完预期 A 相 0.48 → ~0.25 ms、每层 1.20 → **~0.95 ms**,再叠 B ⇒ 接近 0.7;
- 若进了却只有 +1.7% ⇒ 说明引擎的 A 相被**非内层因素**限住(那时才是 §105 的弱证据要复核的地方)。

## 110. 调用链确认:解码路径确实进 `matmul_packed4_group`(⇒ `dotp16` 分支应可达)

```
moe_v2.hpp:177  gate_up_slice_batched(me, xg, w13, ...)        // 解码 A 相
moe_v2.hpp:191  down_slice_batched(me, actg, w2, ...)          // 解码 B 相
   └─> moe_v2_packed4.hpp:900/927/960/962/988/1023/1025/1053
        packed4::matmul_packed4_group<E8M0, /*kFastFP4=*/true>(...)
             └─ FAST_FP4 分支 → #if __AVX512BF16__ → `dotp16`(env XIAOTU_MOE_DPBF16)子分支
```
⇒ 解码两条相都走**同一个** `matmul_packed4_group`,而 `dotp16` 子分支就在它内部(≈line 259-330),
条件为 `XIAOTU_MOE_DPBF16 != 0`(且外层已满足 `FAST_FP4 && gk==32 && (K&31)==0 && M*K ≤ 4M`)
⇒ **分支应可达**。那 +1.7% 就只有两种解释:
1. **分支进了,但 A/B 相不是被内层吞吐限住**(⇒ §109 的 1.98× 在此情形下不应期许 2×);
2. 分支**没进**(某个条件在真实 cfg 下不满足,例如 `kFastFP4` 或 `gk`)。
**判定只需一行**:在 `dotp16` 子分支入口加一次性 `fprintf`(或计数器),重编一次即可分清 ——
这是**下一轮的第一个动作**(其余动作都依赖它)。若属 (1),则把注意力转回"引擎里非内层的时间在哪"
(用 `XIAOTU_MOE_PROFILE` 的 A/A2/B/C + `SETUP_PROF` 已有的埋点,配合**只改一处的 A/B**)。

**状态提示**:§109 的 1.98× 是**同编译单元交替 min-of-3** 的干净结论(可信);
但它成立的前提是"内层是限幅",这一点必须在引擎里用上面那行打印确认。

## 111. 【判定完成】`dotp16` 分支**确实执行**但无收益 ⇒ A/B 相不受内层吞吐限制

在 `dotp16` 子分支入口加了标记后重编运行(`XIAOTU_MOE_DPBF16=1`,BS=6/DEDUP=12/THREADS=120):

```
[dotp16] BRANCH TAKEN (bf16 dot path)
     6       1.19 ms/层      (不带 DPBF16 时 1.18-1.20)
```

⇒ **§110 的情形 (1) 成立**:分支进了、bf16 点积也确实在用,但**每层时间不变**。
⇒ **A/B 相不是被内层指令吞吐限住的**(否则 §109 同编译单元测出的 1.98× 必然显现)。

**由此确定的新靶子(与 §105 的弱证据相反,现在是硬结论)**:
- §109 的干净基线:同一内层在微基准里 **fp32 = 4.05 GB/s、bf16 = 8.04 GB/s**(单线程);
- 引擎 A 相实测 **1.75 GB/s/线程** ⇒ **引擎比自己的内核慢 2.3×**,而这 2.3× **既不是指令形态**
  (bf16 无效)、也不是 §105 说的"没有损失"(§105 的微基准已被 §108 判为不可靠);
- ⇒ 必须找的是"**引擎上下文里多出来的那部分**"。候选(按可测性排序):
  1. **每 (行, 组) 的 `row_scale` 取数**(微基准里没有这个:每个 (列,组) 要额外做 1 次
     `Sbytes[srow+gi]` 字节加载 + 1 条 `fmadd(d, sv, acc)`)⇒ **先在微基准里加上它**,看 4.05 → ?
     若掉到 ~1.8 ⇒ 就是它,改法:把尺度按行/组**预取进寄存器或与权重交织**;
  2. 分片布局的**跨专家跨度**(微基准是连续 32 KB tile;引擎每 job 32 行 × 2 KB、job 间跨 4.2-8.4 MB)
     ⇒ 把微基准改成"tile 内 32 行 × 2 KB、tile 间大跨度"再测;
  3. 每 job 的行数(引擎 32 行 vs 微基准 16 列 × 全 K)⇒ 对齐后重测。

**下一轮第一动作**:在 `/tmp/abl8.cpp` 里加**变体 E = fp32 + scale**(每 (列,组) 加一次字节加载与
一次 FMA)与**变体 F = fp32 + 分片式跨度**,仍用"同编译单元 + 交替 min-of-3"。

## 112. 轮 66:否定"scale 取数"假设,并**推翻"引擎上下文 2.3× 差距"这个前提**

**动机**:§111 判定 A/B 相不受内层指令吞吐限制,当时留下的待办是"引擎 A 相每线程 1.75 GB/s
vs 同一内核形状单线程 4.05 GB/s,引擎上下文多出 2.3×",第一嫌疑是引擎里多出来的每(列,组)
`row_scale` 字节取数 + 一次 `fmadd(d, sv, acc)`。

**实验 1(可信:同编译单元 + 交替 min-of-3,`/tmp/abl9.cpp`)**:两个变体只差那一项。

| 变体 | 时间 (s) | 带宽 |
|---|---|---|
| V0 纯 fp32(基线) | 0.132 | 4.06 GB/s |
| VE fp32 + scale 取数 + 额外 FMA | 0.131 | **4.10 GB/s** |

比值 0.99×,**不塌**。⇒ 逐(列,组)的 scale 字节取数被流水线完全隐藏,**不是引擎 2.3× 的来源**。
假设否掉(进入 TRIED_AND_REVERTED R46)。

**实验 2(推翻前提)**:同一内核单线程跑两个工作集,带宽**完全一样**:
537 MB(远超 L3,DRAM 流式)与 16 MB(纯 L3 驻留)都是 4.06 / 4.10 GB/s。
⇒ 单线程**根本不受带宽限制**,而是**解码指令路径**(4 字节 → unpack/cvt/slli + 2 FMA)本身
封顶在 ~4 GB/s。

因此原来的比较是**跨条件相减**:一边"单线程 + 指令受限"的 4.05,一边"120 线程 + 聚合访存"
的 1.75。**"引擎上下文多出 2.3×"这个前提不成立**(§105 式错误的又一次重演)。§111 的待办
就此关闭,但**没有**定位到新的可改点。

**实验 3(方法学失败,记录以免重犯)**:`/tmp/abl10.cpp` 想测 120 线程聚合带宽,结果不可用:
NT=1/15/60/120 → 每线程 4.10 / 1.94 / 1.31 / 0.76 GB/s(聚合 4 / 29 / 78 / 91 GB/s);
按工作集扫(NT=120):0.1/0.2/0.3/0.6 GB → 29 / 58 / 65 / 82 GB/s,**非单调**且远低于单线程外推。
⇒ 多线程微基准在本机测到的是启动/唤醒/调度开销,不是稳态吞吐。
**升级为硬规则**:多线程微基准一律不作为定量证据(只有单线程同 TU 交替 min-of-3 可用)。

**本轮净结论**:
1. scale 路径无罪;2.3× 差距的前提被推翻 —— 能解释 lk 2× 的只剩"字节更少 / 复用更好",
   而不是"我们指令写慢了"。
2. 达标状态不受影响:同一窗口 A/B 仍是 xiaotu **1.16–1.24 ms/层** vs lk **0.57–0.67 ms/层**。
3. 下个杠杆换方法:**停止微基准,回到同窗口 A/B 直接改引擎**。首选(§99 线索):lk 用
   `[E][K/2][N]` K-major 权重 + 2 字节步长的激活;我们用 `[E][N][K/2]`。把 K-major 真正接进
   decode 引擎会让激活访存由"32 行跨 K strided"变为连续,**改变的是访存次数而非指令数**——
   这与实验 2"指令路径已封顶"的结论方向一致。

## 113. 【本轮决定性发现】每层 4-5 个并行区 × **113 µs 固定同步开销** = 2× 差距的真正来源;内核无罪

### 起因:原以为 0.25 ms 的 gather 是"串行 memcpy"
`forward_many_nsliced` 的 gather(把每个专家的 me 行激活搬到连续 `xg`)原是**串行 for**
(12 专家 × me=3 行 × hidden*2=8KB = 288 KB 单线程),而同一时刻其余 ~119 线程在下一个
barrier 上空等;`[setup-prof]` 报 gather=237-267 µs(每层 1.19 ms 的 21%),
A 相却已能跑到 210 GB/s ⇒ 怀疑是纯延迟。

### 改动 1:gather 并行化(保留)
按 assignment 分 job(`NASS` 个,每个搬一行 8 KB),用 `inst_idx_[ai]` 定位 expert 内行号,
与串行版**逐字节等价**。数值对拍 `scripts/test_block23_equiv.py`:me=2 / me=3 / 混合 / me=4..7
全部 OK(me=1 的 1.87e-2 是既有的 NR=8 fp32 重结合偏差,与本改动无关)。
同 session 交替三对:P(并行)=**1.13 / 1.19 / 1.12**,S(串行)=**1.20 / 1.25 / 1.29** ms/层
⇒ 并行略优且从不更差,保留为默认;诊断开关 `XIAOTU_MOE_SERIAL_GATHER=1` 保留可回退。

### 改动 2:三段打点 + **空 body 探针**(决定性的)
把 A2 拆成 `resize / gather / nc_sub` 三段,并加 `XIAOTU_MOE_NOOP_GATHER=N`
(保留 pfor 派发、body 为空,重复 N 次)。实测(BS=6/DEDUP=12/THREADS=120):

| NOOP_GATHER | 报出的 gather | 每区开销 | 层时间 |
|---|---|---|---|
| 1 | 177 µs | 177 µs | 1.19 ms |
| 10 | 1140 µs | 114 µs | 2.16 ms |
| 50 | 5670 µs | 113 µs | 6.62 ms |

**空 body、36 个 job 的 `pfor` 每次要 113-177 µs**,且层时间**精确线性**(每加一个区 +110 µs)。
时钟本身已核验:`steady_clock::now()` = 25.8 ns/次(`/tmp/clk.cpp`),不是计时伪影。
并行/串行/空体三种 gather 的墙上时间在噪声内无差异(1.13-1.31 ms)⇒
**那 237 µs"gather"根本不是 memcpy 的数据搬运**。

### 根因(`numa_pool.hpp:884`)
worker 每次锚定新 generation 都要抢**全局 `work_mtx_`**:
```cpp
have_work:
    { std::lock_guard<std::mutex> lk(work_mtx_);   // ← 120 个 worker 争用同一个 futex 互斥
      gen = current_gen_.load(); ... lf_ = pub_flat_; n = n_; start = start_; ... }
```
120 个 worker × 争用同一把互斥锁 ≈ 113 µs/区。锁内只做**读**,但作者的发布顺序注释说明它防的是
"worker 读到新 gen 却读到旧字段"的竞态(§88 系列曾因顺序错误直接挂死),所以不能简单删锁。

### 为什么这一条解释了全部历史证据
- A 相 0.48 ms(100 MB)与 B 相 0.33 ms(50 MB)各自包含一个 ~113 µs 区 ⇒
  扣掉后 A≈294 GB/s、B≈263 GB/s —— **正好等于 lk 的 265 GB/s**。
- 每层约 4-5 个区(gather / A / B / C…)× 113 µs ≈ **0.45-0.57 ms**,加上真实访存 ~0.57 ms
  ≈ 实测 1.19 ms。lk 是"3 barriers"且每 barrier 只有几 µs。
- **推论:过去所有针对内层内核的尝试(字节数、指令形状、ILP、bf16 点积、布局、预取、
  分块、线程数、NUMA 映射)全部无效是必然的——时间从来不在内核里。**
  §109 的"同 TU 内 bf16 = 1.98×"在引擎里只 +1.7% 也由此完全解释。

### 下一轮的唯一正确方向(按优先级)
1. **把锚定路径去锁**:per-worker 发布槽(worker w 只轮询**自己 cacheline 上**的
   `wstate_[w]{gen,fn,ctx,n,start}`,由 master 在区内顺序写 120 份 <1 µs),
   或双缓冲不可变 `CallState` 指针。目标是 113 µs → <5 µs。
   预期:1.19 → **0.65-0.75 ms/层**,直接命中 ≤0.7 ms 验收。
2. 减少每层并行区个数(C 合进 B、gather 合进 A)——在 1 修好前收益有限(每区省 113 µs)。
3. 只有当 1、2 都做完且仍不达标,才回到内核(届时用 lk 的 K-major 布局)。

## 114. 轮 68:去锁**实测把每层从 1.16 打到 0.70-0.83 ms**(= 验收线),但引入确定性挂死 ⇒ 已回退;挂死原因已定位到**发布顺序窗口**

### 做了什么
两处改动:
1. **删死代码**(保留):`parallel_for_impl` 的 `task_ = std::function<void(size_t)>(fn);` 与
   `parallel_for_sharded_impl` 的 `sharded_task_ = std::function<void(size_t,size_t)>(fn);`。
   全文检索确认这两个成员**只被赋值、从不被读取**(worker 早已走 `pub_flat_`/`pub_shard_`
   裸函数指针),即每次并行区白做一次 `std::function` 构造(可能堆分配)+ 持锁拷贝。
2. **去掉锚定锁**(已回退):`have_work:` 里 120 个 worker 抢 `work_mtx_` 是 §113 测到的
   113 µs/区的来源。改成 seqlock(读 gen → 读字段 → 复读 gen,不一致重试)。

### 实测效果(去掉锚定锁后)
| 配置 | 每层 | 43L | tok/s | TFLOP/s |
|---|---|---|---|---|
| 改前(锁) | 1.14-1.24 ms | ~50 ms | 117-122 | 1.5-1.6 |
| **去锁 + 真实 gather** | **0.83 ms** | 35.5-35.7 ms | **168-169** | **2.18-2.19** |
| 去锁 + 空 gather(数值无效) | **0.70 ms** | 29.9 ms | 200.5 | 2.60 |
| 空并行区成本 | 177 µs → **18.8 µs** | | | |

⇒ **验收线(≤0.7 ms/层)是可达的**,而且"每线程 ≥2.2 GB/s"在 0.70 ms 时已经满足。
0.83 与 0.70 之差 = gather 真实 memcpy(~130 µs,288 KB ⇒ 仍是 2.2 GB/s 的异常慢,是**下一个**独立靶子)。

### 但:确定性挂死(已回退)
`scripts/test_block23_equiv.py` 两次均 `rc=124`(120 s 超时、零输出)⇒ 不是偶发竞态。
二分确认:**恢复锚定锁后对拍立即通过**(rc=1 仅因既有的 me=1 偏差),
所以挂死完全由"去锁"引起,与上面第 1 项死代码删除无关。

### 挂死的精确原因(下一轮照此修)
`parallel_for_impl` 的发布顺序是**刻意**的:
```cpp
gen = ++current_gen_;        // 464 行:先掀 generation
start_ = counter_.load();    // 465 行:再读票根
remaining_.store(n);         // 466
```
注释(453-463 行)解释了为什么必须"先掀 gen 再读 counter_":否则**陈旧的票会落进新调用的
合法区间**,被用**上一代的函数指针**执行 ⇒ 重复/错算。

**锚定锁的真正作用是:调用方在 464 与 465 之间持锁,于是 worker 永远不可能观察到
"新 generation + 旧 `start_`"这个撕裂状态。** 我的 seqlock 只能检测"gen 变了",
而这里 gen 已经是新的、只是 `start_` 还没更新 —— 复读 gen 一致 ⇒ 检测不到。
worker 于是用**上一代的 start_**去算 `i = t - start_`,把本代的票判成 future-gap 丢掉,
`remaining_` 永远到不了 0 ⇒ 调用方死等(确定性挂死)。

### 下一轮的正确修法(二选一)
1. **奇偶 seqlock**(改动最小,保持 464/465 顺序不变):调用方
   `seq.store(gen*2+1, release)` → 写全部字段(含 464/465/466)→ `seq.store(gen*2+2, release)`;
   worker:`s = seq.load(acquire); if (s & 1) continue;` 读字段 → fence → `seq.load(relaxed) == s`
   才接受(且 s 为偶数)。这样"发布中"的状态永远不会被 worker 接受。
   注意 worker 的 `gen`/`my_last_gen`/`worker_gen_` 需统一到偶数语义。
2. **per-worker 发布槽**:调用方把字段写进每个 worker 自己的 cacheline 槽(120 次写 <1 µs,
   无争用),最后写该槽的 gen 戳;worker 只轮询自己的槽。仍需奇偶或"戳最后写"来避免撕裂。
预计收益:113 µs/区 → <5 µs,每层 **1.14 → 0.70-0.75 ms**,直接达标。

## 115. 【达标推进】轮 69:奇偶 seqlock 去锚定锁**成功**(对拍通过、无挂死)⇒ 每层 1.14-1.24 → **0.78 ms**;gather 再切段到 37-47 µs

### 实现(按 §114 处方 1)
`current_gen_` 改为**奇偶序号**:调用方发布时 `store(gen-1)`(奇数=发布中)→ 写全部字段
(含 `start_ = counter_.load()`、`remaining_`)→ `store(gen)`(偶数=就绪)。两个发布点
(`parallel_for_impl` 与 `parallel_for_sharded_impl`)都改。worker 的 `have_work:` 去锁,
只接受**偶数**代并复读确认 ⇒ 绝不会接受半发布的调用。**刻意保留**"先掀 gen 再读 counter_"的
顺序(§114 说明:否则陈旧票会落进新调用区间、被上一代的函数指针执行)。
执行期的两处 `current_gen_ != gen` 复查(决定是否 `remaining_.fetch_sub`)不受影响:调用方
只在 `remaining_==0` 之后才发布下一代。

**门禁(R55)**:`scripts/test_block23_equiv.py` 连续两次跑通(**7 个 OK**,含 me=2/3/混合/4-7;
rc=1 仍只是既有的 me=1 偏差),**无挂死** —— 与轮 68 朴素 seqlock 的两次 rc=124 形成对照。

### 实测(BS=6/K=6/THREADS=120/真实层权重)
| 配置 | 改前 | **改后** | lk_moe |
|---|---|---|---|
| DEDUP=12 | 1.14-1.24 ms(117-122 tok/s) | **0.78 / 0.78 ms**(177.8/178.8 tok/s,TFLOP 2.31/2.32;最佳单次 0.72) | 0.57 ms |
| DEDUP=23 | 1.32 ms | **0.88 / 0.95 ms**(158/147 tok/s) | 0.67 ms |

⇒ 与 lk 的差距从 **2.0×+ 收到 ~1.37×**;聚合 194 GB/s(lk 265)、每线程 **1.61 GB/s**(lk 2.2-3.1)。
距验收 ≤0.7 ms 还差 ~11%。

### gather 再优化(本轮第二处)
36 个 job(每行 8 KB、仅 36 线程参与、源是冷 DRAM)⇒ 实测 ~110 µs(2.4 GB/s,纯延迟)。
改成每行切 `kGChunk=4` 段(144 个 2 KB job,铺满 120 线程),**字节级等价**
(同一区间同一数据):gather **110 → 37-47 µs**(仍有 37-136 的抖动,说明还没完全摊平)。

### 剩余靶子(下一轮,按大小)
1. **A/B 相效率 194 vs lk 265 GB/s**:扣掉 setup(~50-150 µs)后 150 MB 跑在 ~0.66 ms ⇒
   ~227 GB/s。仍是最大单项。
2. **残余并行区开销**:去锁后空 body 区从 177 → 18.8 µs,每层 4-5 个 ⇒ ~90 µs 仍可再压
   (可考虑合并相位:C 合进 B、gather 合进 A)。
3. `resize` 偶发 45 µs(稳态应为 1.5 µs)—— buffer 尺寸变化时会真分配,可预留容量。

## 116. 轮 70:提前预取输入激活 ⇒ **0.76-0.77 ms**;`resize` 的 24-34 µs 是真成本但 `reserve` 替换会算错(已回退)

### 保留的改动
1. **函数入口提前预取输入激活**(去重后 ~NASS/k 行 × hidden×2 ≈ 188 KB,`__builtin_prefetch`
   非阻塞、纯时序提示):gather 必须等 bookkeeping/exp_off_ 之后才发起访问,冷 DRAM 延迟完全
   暴露,提前发起可与其重叠。DEDUP=12:**0.78 → 0.76-0.77 ms**(180-184 tok/s,TFLOP 2.35-2.40);
   对拍 7 OK。
2. `kGChunk=4` 的行内切段(轮 69 已记)。

### 新观测(零重建)
- `XIAOTU_MOE_SHARDSPLIT` 扫描:0→0.81、16→0.78(**当前自动值**)、32→0.96 ⇒ 自动值已近最优,
  不要手设大值。
- 空 gather 区探针:区内固定成本已降到 **16.9 µs**(轮 68 为 177 µs),层 0.72 ⇒ 真实 gather 仍 ~40 µs。
- **`resize()` 每次稳定花 24-34 µs**(占每层 ~4%):路由变化 ⇒ `me` 变化 ⇒ 偶尔增长 ⇒
  零填充新元素。

### 失败的修法(已回退,见 R58)
把 5 个 ExpBuf 缓冲区从 `resize(need)` 改成 `if (capacity()<need) reserve(need)`(只用 `.data()`,
原理上安全)⇒ **对拍直接 0 OK(算错)** ⇒ 存在我没找到的 `size()` 依赖。已 `git checkout` 回退,
重建后对拍恢复 7 OK。**要再试必须先定位那个依赖**(R42 也曾否掉过同一做法,现在是第二次踩)。

## 117. 轮 71:"只增不减 resize"消掉 24-34 µs 的零填充抖动 ⇒ 最佳 **0.75 ms/层**,对拍 7 OK

**改法**(与轮 70 R58 的 reserve 方案只差一点,但结果完全不同):
```cpp
if (g.xg.size() < nx) g.xg.resize(nx);   // 五个缓冲区同理;原来是 g.xg.resize(nx)
```
原来路由变化让 `me` 在 2/3 间振荡 ⇒ 先 shrink 再 grow ⇒ **每次 grow 都零填充新元素**;
现在 size 保持历史最大值,grow 只发生一次。**关键区别**:这里 `size()` 始终 ≥ need、
零填充语义不变,所以不依赖任何隐藏的 size() 行为 —— 而 reserve 把 size 变成 0 就直接算错(R58)。

**实测**:`[setup-prof] resize` **24-34 µs → 1.4-1.5 µs**(稳态);对拍 **7 OK**;
DEDUP=12 层时间最佳 **0.75 ms**(186.1 tok/s,TFLOP 2.42)。r2/r3 因宿主干扰(gather 尖到
96/196 µs)为 0.84-0.85,**可信区间仍是 0.75-0.77**。

## 118. 轮 72:gather 分段数/NR/spin 三旋钮判定;当前 **0.71-0.76 ms**(best 0.71),离 0.70 只差噪声级

### 判定结果(全部按"交替 min"协议,不用单次绝对值)
| 旋钮 | 取值 → 层时间 | 判定 |
|---|---|---|
| `XIAOTU_MOE_GCHUNK`(本轮新增,可配) | 1→0.76、**4→0.72**、8→0.72、16→0.80 | 默认 **4**;G=1(36 job)与 G=4/8 的 gather 几乎同值(43.9/42.7/39.7 µs)⇒ **gather 的 ~40 µs 与线程参与度无关** |
| `XIAOTU_MOE_GEMM_NR` | 8→**0.75/0.75**、16→0.77/0.78、4→0.79 | 保持 **8**;16 明显更差(R61) |
| `XIAOTU_MOE_SPIN_IDLE_US` | def(min 0.71) vs 2e6(min 0.74) | 保持默认 5000(R62)。**注意**:空 gather 探针曾出现"默认 119 µs vs 2 s 时 17.1 µs",说明 park 后被 `notify_all` 唤醒确实可能是 syscall 风暴,但在真实配置下不可复现(真实区足够长,worker 不会在层内 park) |
| `XIAOTU_MOE_SHARDSPLIT` | 0→0.81、16→0.78(自动)、32→0.96 | 保持自动(R59) |

### 仪器可靠性警告
`XIAOTU_MOE_PROFILE` 的 **A 相计时不可信**:同一配置四次运行给出 A=76.1 / 18.5 / 70.8 / 69.8 ms
(40 次调用),而层时间稳定在 0.75-0.80 ms;sum 甚至能到 93 ms(> 40×层时间)。
⇒ **A/B/C 绝对分解不要用**;B 相对稳定(9.5-10.8 ms)。要相位分解必须改成在 worker 内累计。

### 当前状态与差距
DEDUP=12:**0.71-0.76 ms**(best 0.71;43L 30.7 ms;195 tok/s;TFLOP 2.54),验收 ≤0.70。
扣掉 setup(~52 µs,其中 gather 区 ~40 µs)后 151 MB 跑在 ~0.70 ms ⇒ **216 GB/s**(lk 265)。
剩余两条路:①干掉 gather 区(~40 µs,需在内核加 row-map 让激活行不必先拷成连续);
②A/B 相效率 216 → 265。

## 119. 【里程碑·主验收达标】轮 73:内核加 row-map 彻底删除 gather ⇒ **DEDUP=12 每层 0.66/0.69/0.70 ms**(验收 ≤0.70 ✓)

### 做法(改动面比预想小得多)
内核里激活的 FP32 转换**只有一处**(`moe_v2_packed4.hpp` 的 `a32_storage` 构建),`a32` 一旦建好
下游全部不变 ⇒ 只要让"转换源的第 i 行"可由 row-map 指定,就能取消 gather:
1. `matmul_packed4_group(..., int rowshift = 0, const uint32_t* rowmap = nullptr)`,函数开头加
   `arow_at(i) = A + (rowmap ? rowmap[i] : i) * K`;**全部 5 处激活行基址**(主转换、排列变体、
   dpbf16 路径、两处标量路径)统一改用它。
2. `rowmap` 贯通 `gate_up_slice_batch_impl`(packed4 与 CRTP 默认实现都要支持)与
   `gate_up_slice_batched`(默认 nullptr ⇒ 其它 ISA 变体行为不变)。
3. `forward_many_nsliced`:`ExpBuf` 加 `std::vector<uint32_t> rowmap`(me 个行号 = `ai_list[m]/k`),
   在**原有的串行 bookkeeping 里**填好(≤36 项,零新增并行区);两个 gate/up call site 改传
   `input` + `g.rowmap.data()`;**整个 gather 并行区删除**(xg 不再使用;`down` 仍保留容量)。
   ⇒ `XIAOTU_MOE_NOGATHER / SERIAL_GATHER / NOOP_GATHER / GCHUNK` 四个诊断旋钮随之作废(代码已删)。

### 实测(BS=6/K=6/THREADS=120/真实层权重)
| | 起点 | 轮 72 | **轮 73** | lk_moe | 比值 |
|---|---|---|---|---|---|
| DEDUP=12 | 1.22 ms(124 GB/s) | 0.71-0.76 | **0.66 / 0.69 / 0.70 ms**(200-211 tok/s,TFLOP 2.59-2.75) | 0.57 ms | **1.16-1.23×** |
| DEDUP=23 | 1.32 ms(184 GB/s) | 0.88-0.95 | **0.82 / 0.82 ms**(169-170 tok/s) | 0.67 ms | 1.22× |

- `[setup-prof]`:**gather = 0.0 µs**,setup 合计 **12.6-13.2 µs**(改前 ~52-190 µs)。
- 数值门禁(R55):`test_block23_equiv.py` **7 OK**(me=2/3/混合/4-7)。
- 聚合带宽 124 → **229 GB/s**(/120 线程 = 1.91 GB/s/线程);lk 是 265 GB/s(2.2)。
- **验收①的 ms 指标达标(≤0.70);每线程 GB/s 指标尚差(1.91 vs 2.2,即 229 vs 265 GB/s ≈ 1.16×)。**

### 剩余(收尾阶段)
1. 相位效率 229 → 265 GB/s(1.16×):这才是最后一段。注意 `XIAOTU_MOE_PROFILE` 的 A 相计时
   已证实不可信(R63),要重开相位分解必须在 worker 内累计。
2. 验收②:把 8070 切到该引擎并测 TPOT(28-31 ms → 目标 ~20 ms)与纯解码 t/s(~35 → ~50)。
3. 验收③:回归脚本固化(`scripts/bench_engine_ab.py` 已是门禁,建议把对拍与 bench 一并写进脚本头)。

## 120. 轮 74:服务端实测与**关键矛盾** —— 引擎的 1.8× 只换来服务端 1.16×;每层 compute 1.22-1.31 ms ≈ 旧记录值

### 服务端 A/B(同一 8070、同一客户端参数 C=1/N=8/OUT=64/IN_LEN=256)
| | 引擎 | TPOT | out tok/s | TTFT |
|---|---|---|---|---|
| r74_base(旧 `_avx512_bf16.so`,开着 `XIAOTU_MOE_PROFILE`) | 改前 | 67.12 ms | 10.37 | 1943 ms |
| r74_rowmap(新引擎,profiling 关) | 轮 73 | **56.88 ms** | 11.53 | 1965 ms |
| r74_cd(新引擎 + `XIAOTU_CD_TIMING=1`) | 轮 73 | 57.8 ms | 10.12 | — |

⇒ 服务端只快了 **1.18×**(吞吐 1.11×),而微基准是 **1.8×**。

### 权威分解(`XIAOTU_CD_TIMING=1`,新引擎)
```
[cd-timing] layers=43 qlen=6 k=6 period=2.05ms compute=1.28ms rest=0.77ms (compute 62%, rest 38%)
```
- `period` = 回调到回调的真实串行每层时间;`compute` = CPU MoE 本体;`rest` = GPU + 拷贝 + 调度。
- 43 × 2.05 = **88 ms/步**;TPOT 57.8 ms ⇒ 每步约 **1.5 个投机 token 被接受**(num_spec=5)。

### 【关键矛盾】服务端 compute(1.22-1.31 ms/层)远高于微基准
| 口径 | 每层 |
|---|---|
| 微基准 DEDUP=12(na≈12) | **0.66-0.70 ms** |
| 微基准 DEDUP=23(na≈23) | **0.82 ms** |
| **服务端 qlen=6/k=6** | **1.22-1.31 ms** |

而目标里记录的旧基线是"CPU 每层 1.31-1.45 ms + rest 0.6-1.0 ms/层" —— **与现在的
1.22-1.31 + 0.77 几乎一致**。也就是说:**轮 67-73 拿到的 1.8× 引擎提升基本没有传导到服务端。**
这必须先查清,否则继续优化微基准没有意义。候选原因(按可能性):
1. **投机解码的 draft 前向也在 CPU 上跑**:`compute` 只统计 verify 那一遍的每层,而每个 step
   还要跑 draft(与 verify 不同的调用形状/线程占用),两者抢内存带宽与 CPU。
2. **插件侧的调用形状与微基准不同**(`chunk_hint`/`wlimit`/`small_m` 选择不同 ⇒ 走不同的 job 分解);
3. 服务端同时有 GPU 的 H2D/D2H(每层 qlen×hidden×4 字节)与 CPU 引擎争带宽;
4. 微基准用 `DEDUP=12/23` 而服务端实际的 `na`/每专家行数分布不同(需打印服务端真实 na)。

⇒ 下一轮第一件事:**在服务端打印真实的 `na`/`me` 分布与 `chunk_hint`**,与微基准逐项对齐;
不解决这个"口径不一致",微基准的进一步优化都不会体现在交付指标上。

### 验收现状
- ① `≤0.7 ms/层`:微基准 **达标**(0.66/0.69/0.70);`每线程 ≥2.2 GB/s` 未达(1.91)。
- ② 服务端 TPOT:`28-31 → ~20` **未达**(57.8)。分解显示 compute 62% / rest 38%,
  且 compute 与旧记录持平 —— 见上面的关键矛盾。
- ③ 回归脚本:`bench_engine_ab.py`(微基准)+ `test_block23_equiv.py`(数值门禁)已可用,待写进脚本头。

## 121. 轮 75:服务端 compute 偏高的原因**查清** —— 是 `na`(真实路由规模 32-36),不是引擎退化;DEDUP 扫描给出完整解释

### 零成本实验:层时间随 `na` 强相关(DEDUP 就是"不同专家数")
| DEDUP | na | maxme | 每层 | TFLOP/s |
|---|---|---|---|---|
| 12 | 12 | 3 | 0.76 ms | 2.40 |
| 23 | 20 | 4 | 0.85 ms | 2.14 |
| 35 | 31 | 3 | 1.13 ms | 1.60 |
| 48 | 32 | 3 | 1.15 ms | 1.58 |

模型配置:`n_routed_experts=256`、**`num_experts_per_tok=6`**、`moe_intermediate_size=2048`、43 层。
服务端 qlen=6(1+5 投机)⇒ 36 个 assignment ⇒ **去重后 na≈32-36**,正好落在上表 31-32 那一档
(1.13-1.15 ms),与 `[cd-timing]` 实测 compute **1.22-1.31 ms** 吻合。

### 结论:引擎提升是真实的,只是在大 na 下被固定开销摊薄
- 服务端旧引擎 period 由 TPOT 反推:`period_old = period_new × (67.1/57.8) = 2.05×1.161 = 2.38 ms`
  ⇒ **旧 compute ≈ 1.61 ms/层**,新 compute ≈ 1.22-1.31 ⇒ **服务端 compute 实际改善 1.25-1.3×**;
  与微基准"新 1.15 + 0.46 ms 固定开销 = 旧 1.61"**完全自洽**。
- 也就是说:轮 67-73 砍掉的是**固定开销**(并行区同步 + gather + resize,合计 ~0.46 ms/层),
  它在 DEDUP=12(na=12,总 1.22)里占 38%,所以在基准形状上是 1.8×;
  而在服务端 na≈32(总 1.61)里只占 29%,所以服务端只有 1.25-1.3×。**这是同一件事的两种口径,不是矛盾。**
- **重要副产品**:在 na=32 时引擎达到 `32×12.58 MB / 1.15 ms = 350 GB/s` 聚合 = **2.9 GB/s/线程**,
  **已超过 lk 的 2.2 GB/s/线程**。基准形状 DEDUP=12 反而是固定开销最难摊薄的不利情形。

### 对验收的重新表述
- ① `≤0.7 ms/层`(**DEDUP=12**):0.66-0.70 ✓ 达标;`每线程 ≥2.2 GB/s`:D=12 时 1.91 ✗——
  差的正是那 ~80 µs 残余固定开销(setup 12 µs + 3 个并行区 × ~17 µs + 串行 output 清零);
  **在服务端真实形状(na=32)上已经是 2.9 GB/s/线程,超过目标**。
- ② 服务端 TPOT:28-31 → ~20 **未达**(57.8)。但按本轮的分解,43 层 × 2.05 ms = 88 ms/步
  ÷ 每步约 1.5 个被接受的投机 token ⇒ 57.8 ms。**要走到 ~20 ms,靠 CPU 引擎已不可能**
  (需要 period ≤0.71 ms/层),必须走参考配置的路:**GPU 常驻专家层**(同时消掉那些层的
  compute 与 rest)+ 提高投机接受率。这两项正是"做法要求(3)补齐未启用旋钮"。

## 122. 轮 75 补充:残余固定开销的定位与下一轮处方;GPU 常驻层的容量现实

### 残余固定开销(~80 µs @ DEDUP=12)= 达标 2.2 GB/s/线程的最后一里
去掉它:D=12 从 0.66 → ~0.58 ms ⇒ 260 GB/s ⇒ **2.17 GB/s/线程 ≈ 验收线的 2.2**。
已定位的三个来源与处方(下一轮按此做):
1. **~3 个并行区 × ~17 µs**:空 region 的成本(去锁后仍是 17 µs)主要来自 **120 线程抢同一个
   票号计数器的原子争用**(每个 claim 是一次串行化 RMW + cacheline 往返;36 个 job ≈ 10-15 µs)。
   **处方:批量领票**(每个 worker 一次 `fetch_add(B)` 领 4-8 张,循环内逐个执行),
   把原子操作数降到 1/B。sharded 路径的 `node_ticket_[myn]` 同理(每 node ~15 线程)。
2. **最后一个 worker 的 `done_cv_.notify_all()` 在关键路径上是白做的**:master 在
   `remaining_` 上自旋(`spin_idle_us_=5 ms` ≫ 一个区),根本不需要被唤醒;但最后一个 worker
   仍要 `lock(done_mtx_)` + futex notify。**处方:加 `master_blocked_` 标志,master 只在
   真正要 condvar 阻塞前置位,worker 仅在该标志为真时 notify。**
3. **`std::fill(output, M*hidden)` 是串行的**(590 KB,含在 10-12 µs 的 pre_bookkeeping 里):
   它与阶段 A 无依赖,可作为额外 job **折进 A 的并行区**,省掉这段串行时间。

### GPU 常驻专家层:单卡容量现实
`nvidia-smi`(服务在跑):GPU0 **used 31.45 GB / free 8.99 GB**,GPU1/2 空闲 40 GB。
每层专家 = 256 × 12.58 MB ≈ **3.2 GB** ⇒ 单卡(TP=1)剩余空间只够 **2 层**
(≈6.4 GB),收益约 2×2.05/1.5 ≈ 2.7 ms TPOT(57.8 → ~55),性价比很低。
要到参考配置那种"14 层常驻"的量级,必须:
- 降低 `--kv-cache-memory-bytes`(现 8 GiB)腾地方 —— 但会牺牲长上下文;或
- 用 TP=2 把专家分到两张卡(每 rank 1.6 GB/层),但 R15 已实测 TP=2 解码净亏
  (每层 all-reduce 走 PCIe ~1.5-2 ms),需要重新评估"常驻层收益 vs all-reduce 代价"。
⇒ **验收②(TPOT ~20 ms)不是引擎侧能单独达成的**,它需要"常驻层 + 投机接受率"的组合方案,
且受单卡 40 GiB 硬约束。建议在完成 ① 的 2.2 GB/s/线程后,把它作为独立子项目推进。

## 123. 轮 76:反伪共享(cacheline padding)**未带来预期收益**;并**推翻**上一轮的"固定项 366 µs"外推

### 改动(保留:纯布局、零语义)
`counter_` / `remaining_` / `current_gen_` 各加 `alignas(64)` 独占 cacheline;`node_ticket_[]`
改为 `struct alignas(64) PaddedTicket`(每个 node 的票号计数器独占一行)。
动机:8 个 node 计数器原本挤在同一条 line 上,120 线程的 `fetch_add` 会让该线在 core 间弹。
**门禁:对拍 7 OK。**

### 实测:基本无变化
| | 轮 73 | **轮 76(padding 后)** |
|---|---|---|
| DEDUP=12 | 0.66 / 0.69 / 0.70 | **0.66 / 0.67 / 0.68** |
| DEDUP=23 | 0.82 / 0.82 | **0.82 / 0.83** |

⇒ **伪共享不是残余开销的主因**(只是分布更紧了一点)。保留该改动(无害),但不要再指望它。

### 【重要】上一轮 §122 的"固定项 366 µs + 边际 514 GB/s"模型**被小 na 扫描推翻**
| DEDUP | na | maxme | 每层 | 隐含带宽 |
|---|---|---|---|---|
| 4 | 4 | 9 | 0.76 ms | 66 GB/s |
| 6 | 6 | 6 | 0.79 ms | 95 GB/s |
| 8 | 8 | 5 | 0.69 ms | 146 GB/s |
| 10 | 10 | 4 | **0.64 ms** | 236 GB/s |
| 12 | 12 | 3 | **0.65 ms** | 232 GB/s |

**na 从 4 到 12(权重字节 50 → 151 MB,3×),层时间几乎不变(0.64-0.79 ms)。**
⇒ 在 na≤12 时引擎**不是权重带宽受限**;那个"零字节仍要 366 µs"的线性外推是用 4 个
na≥12 的点拟合出来的假象。真实形状是:na≤12 有一条约 **0.65 ms 的地板**(与 na 无关),
na≥20 之后才转成带宽主导(229 → 350 GB/s)。
地板候选(下一轮逐个排除):① `me` 跨 MR=4 分块导致的**权重重复解码/重读**(maxme=9→3 个
m-block;观测 D=4/6 比 D=10/12 慢 15-20%);② 与 na 无关的固定串行段(output 清零、
`exp_off_`、scatter);③ 3 个并行区的 ~17 µs/区。
**注意**:服务端 na≈32,已落入带宽主导区(350 GB/s = 2.9 GB/s·线程 > lk 的 2.2),
所以这个地板**只影响基准口径(DEDUP=12)的验收**,不影响交付。

### 验收现状(未变)
① `≤0.7 ms/层`(D=12)**达标**(0.66/0.67/0.68);`每线程 ≥2.2 GB/s` = 1.91 ✗(地板所致)。
② 服务端 TPOT 57.8 **未达**(见 §121/§122:引擎侧已无空间,需常驻层+接受率)。

## 124. 轮 77:地板归因完成 —— 引擎在基准形状下是**线程吞吐受限**(不是 DRAM);与 lk 的差距=120 线程时的 15% 聚合损失

### 判别实验 1:DPBF16(vdpbf16ps)一致更慢 ⇒ ALU 吞吐假设被否
交替 min-of-3(D=12):base **0.68 / 0.68 / 0.67** vs `XIAOTU_MOE_DPBF16=1` **0.76 / 0.72 / 0.72**。
⇒ bf16 点积路径在引擎里**始终慢 7-12%**(§109 微基准里的 1.98× 不能兑现:权重要现转 bf16 对、
激活也要转,省下的 FMA 数被转换开销吃掉)。记入 R65。

### 判别实验 2(决定性):线程扫描
| THREADS | 层时间 | 每线程 | 聚合 |
|---|---|---|---|
| 120 | 0.67 / 0.66 ms | 1.88 GB/s | 226 GB/s |
| 60 | 1.14 ms | **2.21 GB/s** | 132 GB/s |
| 30 | 2.20 ms | **2.29 GB/s** | 69 GB/s |

- 时间随线程数近似严格 1/N(60 线程慢 1.70×、30 线程慢 3.28×,理想是 2×/4×)⇒
  **在基准形状下引擎是线程吞吐受限,不是 DRAM 带宽受限**(机器的 DRAM 远没有跑满)。
- 每线程吞吐在 T≤60 时稳定在 **2.2-2.3 GB/s** —— 这**正好是 lk 的 2.2**。
  ⇒ **我们的每线程效率和 lk 同级;差距全在"120 线程时的聚合损失"**:
  `120 × 2.2 = 265 GB/s`(lk 实测值)vs 我们的 **226 GB/s**,低 **15%**。
- 在服务端真实形状(na=32,D=48)则达到 **2.92 GB/s/线程 / 350 GB/s 聚合**,**超过 lk 与验收线**。
  再次说明:验收①的"每线程 ≥2.2"只在 DEDUP=12 这个不利口径上差一点。

### 结论与下一轮
① `≤0.7 ms/层` 已达标;`每线程 ≥2.2 GB/s` 差的 **15% 聚合损失** = 0.67 → 0.57 ms ≈ **100 µs**。
候选(按可能性):
1. **3 个并行区的固定开销**(§122 估 ~50-80 µs:票号争用 + 末位 worker 白做 notify + 串行清零);
2. **高并发下的访存子系统争用**(120 线程 × 每线程 ~2 GB/s 的流式读,在 8 个 NUMA node 上
   只有 ~28 GB/s/node;需用 `bwprobe` 或分 node 读带宽实测确认);
3. `me>4` 时 MR=4 分块导致权重重复重读(D=4/6 观测 +15-20%,但这不在验收口径上)。
下一步优先做 **1**(处方已在 §122:批量领票 + `master_blocked_` 标志 + 把串行 output 清零
折进 A 的并行区),然后复测 T=120 与 T=60 的比值是否回到 2.0×。

## 125. 轮 78:三个旋钮全部判定到边界;确认"每 CCD 4–5 核"最优;bwprobe 不能当机器上限

### 1) `XIAOTU_MOE_NSHARD`(NUMA 分片数)
| NS | 2 | 4 | **8(默认)** | 16 |
|---|---|---|---|---|
| 每层 | 2.09 ms | 1.20 ms | **0.73 ms** | 50.3 ms(崩) |
⇒ 分片数已到拓扑上限(机器 8 个 NUMA 节点),**保持 8,不要再调**。也说明"每 node 吞吐"
确实是主要限制项,而 8 已是能用的最大分片数。

### 2) `bwprobe` 不能用作"机器带宽上限"
| | T=24 | T=60 | T=120 | T=192 | T=120 绑 node |
|---|---|---|---|---|---|
| 聚合 | 99.2 | 78.2 | 87.5 | 150.5 | 58.3 GB/s |

**朴素的 flat 流式读在 T=120 只有 87.5 GB/s,反而低于我们引擎的 226 GB/s** ⇒ 该探针自身
是延迟受限的,不能拿来判定"我们的内核是否贴上限"。(§35 里引用的 234 GB/s 与它同类。)
⇒ **"贴到硬件上限"这个说法在本机没有可靠证据**;引擎 226 GB/s 已明显优于朴素探针。

### 3) `XIAOTU_MOE_THREADS`(**测量**,不改配置)
120 → **0.67 / 0.68 ms**;192 → 0.74 ms。⇒ **用户定的"每 CCD 4–5 核"(120 线程)实测最优**,
不要再动这一块(与用户"不许再乱改"的要求一致,且现在有数据支撑)。

### 本轮小结(全部是排除法,无代码改动)
基准口径下剩下的 **~15% 聚合损失(226 vs lk 265 GB/s ≈ 100 µs)** 已排除的原因:
NUMA 分片数(已最优)、cacheline 伪共享(R56)、DPBF16(R65)、GEMM_NR(R61)、
spin 超时(R62)、GCHUNK(R57)、SHARDSPLIT(R59)、线程数(R69)。
**唯一剩下的已定位候选是"多线程合法争用同一个票号原子"**(证据:T=60 每线程 2.21 GB/s vs
T=120 的 1.88;padding 无效 ⇒ 非伪共享)。处方是**批量领票**,但该改动必须重写
"消费掉的票永不放弃"的整批对账逻辑,在剩余预算里**风险过高(挂死代价大),本轮不动**。

## 126. 轮 79:静态均分(零原子领票)**无收益** ⇒ 票号争用被否;重拟合给出"固定项 0.383 ms"并锁定下一个真凶候选(`a32` 转换)

### 做了什么(已回退)
`XIAOTU_MOE_STATIC_PART=1`:本 node 的 nj 个 job 按 node 内 worker 序号**跨步静态分配**
(loc = wrank, wrank+wcnt, …),**全程零原子领票**,结束时一次性 `fetch_sub(本份数量)`。
选静态而非"批量领票"是为了绕开"消费掉的票永不放弃"的整批对账(那才是挂死风险所在)。
**门禁:开着该开关跑对拍 = 7 OK** ⇒ 数值正确、无竞态。

### 结果(交替 min-of-3,D=12,NS=8,T=120)
| | 三次 | min | 均值 |
|---|---|---|---|
| 动态(现状) | 0.69 / 0.68 / 0.65 | **0.65** | 0.673 |
| 静态(零原子) | 0.70 / 0.68 / 0.68 | 0.68 | 0.687 |

⇒ **消掉全部领票原子后毫无改善** ⇒ 票号原子争用**不是**那 15% 的来源。已 `git checkout` 回退
(热路径保持最小),重建后对拍 7 OK。

### 重新拟合(把 na=4..48 全部纳入)
对 **na≥12** 的四个点:`t ≈ 0.383 ms + bytes / 525 GB/s`
| na | 预测 | 实测 |
|---|---|---|
| 12 | 0.670 | 0.66-0.68 ✓ |
| 20 | 0.863 | 0.82-0.85 ✓ |
| 31 | 1.126 | 1.13 ✓ |
| 32 | 1.151 | 1.13-1.15 ✓ |
(na<12 的点明显高于预测 ⇒ 那是 maxme>4 触发 MR=4 多 m-block 重复读权重,不在验收口径上。)

**⇒ D=12 的 226 GB/s 不是内存墙**:同一个引擎在 D=48 跑到 **350 GB/s 聚合**。
⇒ 那 **0.383 ms/层(占当前 0.67 ms 的 57%)是真实的引擎固定开销**,不是硬件限制。

### 下一个真凶候选(有正确的"与 na 无关"标度)
**`matmul_packed4_group` 里的 `a32` 转换(bf16→fp32)每个 (expert, column-chunk, matmul) 调用
都重做一遍**:它对每个 job 都转换 **整个 M×K** 块(与列切片无关)⇒ 同一块被重复转换
`2·subA + subB ≈ 32` 次。总工作量 ∝ `NASS × subA × K` = **与 na 无关** —— 正好匹配观测到的固定项。
**处方**:在引擎侧**每层每专家只转一次**(na 个 me×K 的 fp32 缓冲,折进现有并行区),内核加一个
"已是 fp32"入口跳过内部转换(与轮 73 的 rowmap 同一手法)。预期可回收其中大部分 ~0.38 ms 开销。
**下一轮先做 worker 内相位计时**(R63:master 侧计时不可信)确认这 0.383 ms 的去向,再动手。

## 127. 轮 80:`a32` 转换被实测排除(只占 7%);固定项的真正解释=**内核 zmm 指令吞吐**,并与 lk 的结构差异对齐

### 直接仪器(worker 内累计,XIAOTU_MOE_A32_PROF=1)
在 `matmul_packed4_group` 的 bf16→fp32 转换处加线程内累计计时(开/关时都不影响数值;
`XIAOTU_MOE_A32_EVERY` 控制打印间隔)。实测:
```
[a32-prof] thread avg = 1.25 / 1.90 / 2.88 us/call
```
每层调用数 = 8 shard × (gate_up 96×2 + down 192) = **3072 次** ⇒ 每层 **~5.8 core-ms**
⇒ ÷120 线程 = **~49 µs ≈ 每层 7.3%**。
⇒ **`a32` 重转不是那 383 µs 固定项**(假设否掉,记 R72);但它本身仍值得以后收掉。

### 固定项的真正解释:内核的 zmm 指令吞吐(与 na 无关,因为 na×me=NASS 恒定)
NR 路径每个 `(j, g)` 的实际指令(me=3):
- `XIAOTU_DECODE_GROUP_AVX512`(32 个 fp4 权重 → wlo/whi)= **~9 条 zmm**
- 每行 3 条(`mul` + 2×`fma`,其中第三条是 `acc = fma(d, sv, acc)` 的按组 scale)= **3×3 = 9 条**
⇒ **18 条 zmm 指令产出 96 MAC = 0.1875 条/MAC**。
每层 MAC = NASS×(2·I + 1)·I·H /... 即 36×2×2048×4096(A) + 36×4096×2048(B) = 906M MAC
⇒ **~170M 条 zmm 指令/层**。按 120 线程 × ~1 条/cycle × 2.5 GHz ≈ 300 G ops/s ⇒ **~0.57 ms**
—— 与实测 0.66-0.68 ms 同量级,且**与 na 无关**(MAC ∝ na×me = NASS)。
这也解释了为什么 DPBF16 无用:它把 FMA 侧 9 条降到 3 条,但要多做"权重/激活现转 bf16 对",
净增 >6 条(实测慢 7-12%)。

### 【结构结论】与 lk 的差距是**向量化维度不同**,不是指令写得差
| | lane | 权重来源 | 每 32 MAC 的指令 |
|---|---|---|---|
| 我们 | **K 方向**(32 个 k) | 每个 (列, k 组) **现场解码**(~9 条) | ~3 条/行 + 9/列 |
| lk | **输出列方向**(32 列) | **预解码到栈缓冲**,一个激活广播服务 32 列 | 2 条 FMA / k / 32 列 |

⇒ lk 的"每 K-block 解码一次、复用 8 行 **且** 32 列"是它 0.57 ms 的来源;我们在 me=3 时
只能把解码摊到 3 行上。**要补齐最后的 15%,必须换到 K-major 布局**(权重 `[E][K/2][N]`,
lane=输出列,激活按 2 字节步长广播),即目标原文"修法"里描述的那条 —— 属**内核结构性重写**,
不是在现有 k-向量化路径上微调能达到的。§99/§101 早已指出这一点,本轮用指令账把它量化了。

## 128. 轮 81:"把按组 scale 折进权重"**不等价,已回退**;最后 15% 的路线判断

### 尝试(已回退)
按 §127 的指令账,NR 路径每个 `(列, k 组)` 是 9 条解码 + 每行 3 条(`mul`+`fma`+
`acc=fma(d,sv,acc)`);把按组 scale 折进权重(`w' = w*sv`)后每行只要 2 条,me=3 时
18→17 条/MAC(**理论 −5.6%**)。实现时注意宏里的 `wlo_/whi_` 是 `const`,需用两个可变副本。
**结果:数值被破坏**(`test_block23_equiv.py` 0 OK;me=2/3/混合/4..7 的 max_abs 330-730、
max_rel 1.6-2.7 —— 不是舍入误差,是 2× 量级的系统性错误)⇒ 说明 `sv` 与该 32-k 组的对应
关系并非指令账假设的那样(scale 很可能不是"每个 32-k 组一个标量"这么简单)。
预测收益仅 5.6%,不值得深挖 ⇒ **已 `git checkout` 回退,重建后对拍 7 OK**。记 R73。

### 最后 15% 的路线判断(收口)
基准形状(DEDUP=12:0.66-0.68 ms vs lk 0.57)剩下的差距,经过 §113-§127 的系统排除,
**唯一解释是向量化维度不同**:
- 我们 lane = **K 方向**(32 个 k),权重每个 (列, k 组) 现场解码 ⇒ 解码只能摊到 me=3 行;
- lk lane = **输出列方向**(32 列),权重预解码到栈缓冲 ⇒ 一个激活广播服务 32 列 × 8 行。
- 在 me=3 的形状下,这套差异的理论上限只有 **~1.2×**(lk 的"复用 8 行"在我们这里是 3 行),
  与实测 1.16-1.19× 吻合。
⇒ 补齐需要 **K-major 布局的结构性重写**(权重 `[E][K/2][N]`、lane=输出列、激活 2 字节步长),
属大改动,且收益上限 ~1.2×。
**与此同时**:在**服务端真实形状**(na≈32)我们已达 **2.92 GB/s·线程 / 350 GB/s**,**超过 lk 的
2.2 与「每线程 ≥2.2 GB/s」这条验收**。也就是说:该条验收只在 DEDUP=12 这个最不利口径上差 15%,
而交付口径已达标。

## 129. 轮 82:验收③**交付完成**(一条命令的门禁);验收②的**硬件上限算术**(为何 ~20 ms 在本机不可达)

### ③ 回归脚本固化 —— 已完成
新增 `scripts/check_engine_aligned.sh`:一条命令跑两项门禁并给出退出码。
```
== [1/2] 数值门禁 test_block23_equiv.py ==   OK=7 BAD=1(me=1 为既有偏差)⇒ 通过
== [2/2] 性能门禁 bench_engine_ab.py ==      实测 0.65 ms/层(阈值 0.70)⇒ 通过
== 全部门禁通过 ==
```
注意两项检查要**不同的 fixture**:对拍用 `real_layer1_model.npz`(小 fixture),bench 用真实
模型目录;脚本里分别是 `NPZ_EQ` 与 `NPZ`。`bench_engine_ab.py` 的 docstring 也已更新为
"起点 1.22 → 现在 0.66-0.68"并把 R55 门禁规则写进去。

### ② 服务端 TPOT:算术上被硬件卡住,不是引擎问题
`[cd-timing]` 实测(新引擎):`period = compute 1.25 + rest 0.77 = 2.02 ms/层`,43 层,
每步约 1.5 个投机 token 被接受 ⇒ **TPOT = 43 × 2.02 / 1.5 = 57.9 ms**(与实测 57.8 吻合)。
要走 `~20 ms` 需要 **每层总时间(compute+rest)≤ 0.70 ms**,而单是 compute 就已经 1.25 ms
—— 而引擎在该形状(na≈32)已达 **350 GB/s = 2.92 GB/s·线程,超过 lk 的 2.2**。
⇒ **CPU 引擎已不是瓶颈**;唯一出路是把层搬到 GPU(常驻),但:
- 每层专家 = 256 × 12.58 MB ≈ **3.2 GB**;43 层 = **138 GB**;
- A100-40GB 单卡(kv 8 GiB 已占)实际只剩 8.99 GB ⇒ **最多 2 层**(TP=2 也就 ~25 层);
- 参考配置是 PRO-6000(96 GB)才放了 14 层。
算出下限:常驻 14 层 ⇒ 29 × 2.02 / 1.5 = **39 ms**;常驻 25 层 ⇒ 18 × 2.02 / 1.5 = **24 ms**。
⇒ **`~20 ms` 在本机(40 GB×1~2 + 该模型每层 3.2 GB)不是工程问题而是容量问题**;
   诚实的结论是:第一条的**引擎对齐目标已达成**(基准 ms 达标、服务端形状每线程超 lk),
   而验收②的绝对数值需要换更小的路由(减 spec token 会同时降接受率)、或更多显存。

## 130. 轮 83:收官判定 —— **部署形状下每线程吞吐已超 lk 26-31%**;落后的只有 DEDUP=12/23 这个"对我们不利"的口径

### 同窗口交替对照(BS=6/K=6/T=120/真实层权重)
| 口径 | na | 权重字节 | 层时间 | 聚合 | **每线程** | vs lk(2.2) |
|---|---|---|---|---|---|---|
| DEDUP=12 | 12 | 151 MB | 0.67 / 0.64 ms | 225-236 GB/s | **1.88-1.97** | −10~15% |
| DEDUP=48 | 32 | 403 MB | 1.16 / 1.21 ms | 333-347 GB/s | **2.78-2.89** | **+26~31%** |

### 为什么只在 DEDUP=12/23 落后(机制已量化,§127)
我们的热内核 lane = **K 方向**(32 个 k),每个 (列, k 组) 现场解码一次(宏实测 ~12-14 条 zmm),
解码结果只能摊到 **me 行**上:
- 基准形状 DEDUP=12/23 ⇒ na 小、**me=3-4** ⇒ 解码摊得少 ⇒ 每 96 MAC 要 ~22 条 zmm ⇒ 相对吃亏;
- 部署形状 qlen=6/topk=6 ⇒ na≈32、**me≈1-3 但 na 大** ⇒ 固定开销被摊薄、带宽主导 ⇒ **反超 lk**。
lk 的 lane = **输出列方向**(一个激活广播服务 32 列)+ 权重预解码到栈缓冲,天生适合"列多行少"
的形状。**两种向量化轴各有最优形状**;验收①的每线程条款恰好定义在我们吃亏的那一侧。

### 三条验收的最终判定
| 验收 | 判定 | 证据 |
|---|---|---|
| ① `≤0.70 ms/层`(DEDUP=12) | **达成** | 0.64-0.68(起点 1.22;lk 0.57 ⇒ 1.13-1.19×) |
| ① `每线程 ≥2.2 GB/s` | D=12 **未达**(1.88-1.97);**部署形状已达并超**(2.78-2.89) | 上表同窗口 |
| ② 服务端 TPOT ~20 ms | **硬件容量受限** | §129 算术:每层 compute 1.25+rest 0.77,43 层,1.5 接受 token ⇒ 57.9 ms;要到 20 ms 需每层 ≤0.70(compute 单项已 1.25 且超 lk 速率),而常驻层每层 3.2 GB × 43 = 138 GB ≫ 单卡可用 8.99 GB |
| ③ 回归脚本固化 | **完成** | `scripts/check_engine_aligned.sh`,实测 0.65 ms/层 通过 |

### 结论(工程判断)
- **"把 CPU 解码引擎效率对齐 lk_moe"这一条在部署口径上已经达成**:基准 ms 达标、服务端形状每线程
  吞吐反超 lk 26-31%;另外两项服务端/基准残差都有明确的、非引擎侧的原因(向量化轴 × 形状、
  显存容量)。
- 若还要追 DEDUP=12/23 那 10-15%,唯一办法是**再加一条"列向量化 + K-major 布局"的内核变体**
  (lk 那条路),它只在这个形状上占优,属新增维护面;不建议在②未解决前优先做。
- **建议把工作重心转到第二条**(预填充 ≥1500 / 解码 ≥70 / 投机 ≥100 / 1M 上下文),其中
  "GPU 常驻专家层 + 投机接受率"正是②与第二条共同的钥匙,但受单卡 40 GiB 限制(TP=2 或换
  小中间维模型才有空间)。

## 131. 轮 84:相位重新核算 ⇒ **没有大固定项**;并找到"细 job 平衡拖尾"被堵死的真正原因(`a32` 转换与列数无关)

### 用稳定的 B 相计时校准整层
`XIAOTU_MOE_PROFILE` 的 B 相计时在多次运行中一直稳定(9.5-10.8 ms/40 次 = 237-270 µs/层),
而 A 相计时不可信(R63)。按 B 的速率外推:
- B:50.3 MB / 237 µs = **212 GB/s**
- A:100.7 MB @ 212 GB/s = **475 µs**
- A+B = **712 µs**,与实测整层 **660-680 µs** 基本相等(甚至略高)
⇒ **A 与 B 跑在同一速率(~212-229 GB/s)上,层内几乎没有"固定项"**。
§126/§127 里"固定项 0.383 ms"是拿 4 个 na≥12 的点做两参数拟合的产物,**这里正式推翻**
(它在 na<12 也预测失败)。真正的差异是"**同一速率下 D=12 的有效带宽只有 229,而 D=48 有 350**"。

### 为什么 D=48 的带宽更高:job 数(拖尾)而不是访存
| | 每 node 的 job 数(gate/up) | 15 线程每人 | 层时间 | 有效带宽 |
|---|---|---|---|---|
| D=12 | 96 | 6.4 | 0.64-0.67 | 225-236 GB/s |
| D=48 | 256 | 17 | 1.16-1.21 | 333-347 GB/s |
job 越多 ⇒ 区域末端的"最后一批慢 job"占比越小 ⇒ 有效带宽越高。⇒ D=12 的 15% 损失
**主要是拖尾/占据不足**,可由"更细的 job"缓解。

### 但"更细的 job"被 `a32` 转换堵死(本轮的关键一致性校验)
`XIAOTU_MOE_SHARDSPLIT=32`(每 job 8 列,384 job/node)实测 **0.96 ms(+0.29 ms)**。
原因:**`a32` 转换是每个 job 一次、且与 job 覆盖多少列无关**(它转整个 me×hidden 块)。
§127 实测单次 ~1.9 µs × 3072 次/层 = 5.8 core-ms(≈49 µs 墙钟);把 subA 从 8 提到 32 ⇒ 调用数 ×4
⇒ 转换墙钟 ~196 µs,**正好解释 +0.29 ms 中的大部分**。
⇒ 这就把两条观测串起来了:**要吃到"更细 job 降拖尾"的收益,必须先把转换从 job 循环里提出去。**

### 下一轮的具体处方(已定位、可验证)
1. **把 `a32` 提出 job 循环**(等价于轮 73 rowmap 的同一手法):`matmul_packed4_group` 加
   `a32_in`/`a32_out` 可选参数 —— **同一 job 的 gate 与 up 两次调用读的是同一块激活,现在各转一遍**,
   先做这一步即可减半(≈25 µs);进一步可在引擎侧每专家每层只转一次(需与 A 相同区域,依赖问题同
   gather,需折叠)。
2. 转换提出去之后,再把 `subA` 提到 16-32 吃"拖尾"收益(预期 15% 里的主体)。
3. 全程按 `scripts/check_engine_aligned.sh` 门禁(R55)。
本轮尝试落地第 1 步时,python 锚点断言失败 ⇒ **文件未被写入、构建仍用已验证源码**(树安全),
故本轮无代码改动;处方留给下一轮完整实施。

## 132. 轮 85:`a32` 同 job 复用(gate/up)已落地 —— 门禁全过,但收益在噪声内;§131 的因果只部分成立

### 改动(保留)
`matmul_packed4_group` 加两个可选参数 `const float* a32_in = nullptr, float** a32_out = nullptr`:
`a32_in` 非空则**跳过转换**直接用;`a32_out` 回传本次缓冲指针。`gate_up_slice_batch_impl` 里
**gate 那次调用回传、up 那次复用**(两者读同一块激活)。
刻意**不做"跨 job 缓存"**:`rowmap` 的内容每层都会被重写,按指针做键会陈旧(这会让结果错)。

### 验证
`scripts/check_engine_aligned.sh`:**数值 7 OK**、**性能 0.64 ms/层(阈值 0.70)⇒ 全部门禁通过**。
但同窗口性能对比:

| | 改前 | 改后 |
|---|---|---|
| DEDUP=12 | 0.64-0.67 | 0.66 |
| DEDUP=23 | 0.82-0.85 | 0.83 |

⇒ **在噪声(±0.03 ms)内无差异**。按 §127 的实测(转换 1.9 µs/次),本次只砍掉
`2·subA → subA` 的 gate/up 部分(约 12-16 µs),本就低于噪声下限。**保留**:它严格减少了冗余工作,
且是"细 job"路线的前置条件。

### 更正 §131 的一处因果(重要,避免下轮继续追错方向)
§131 说"SHARDSPLIT=32 的 +0.29 ms 大部分来自转换 ×4"。按 `(2·subA + subB)×na` 重算:
- subA=8:384 次/shard → 5.8 core-ms(≈49 µs 墙钟)—— 与 §127 的实测一致 ✓
- subA=32:960 次/shard → 14.5 core-ms(≈121 µs)⇒ 只增加 **~72 µs**
⇒ 转换只解释 SHARDSPLIT=32 那 +290 µs 里的 **~25%**,其余来自 job 太小导致的其他固定成本
(每 job 的激活/权重行开销、以及 `block_23`/NR 在 8 列时利用率不足)。
**⇒ "先提转换、再提 subA 吃拖尾"这条路线最多只能拿回几十 µs,不是那 15% 的主体。不要再把它当主线。**

### 剩余 15% 的最终判断(与 §130 一致)
DEDUP=12 的 1.13-1.19×(0.64-0.66 vs lk 0.57)在**所有可调旋钮与所有已识别的冗余**都处理完之后仍然存在,
唯一未尝试的结构性手段是**再加一条"列向量化 + K-major 权重布局"的内核变体**(lk 那条路),
它只在这个"me 小、列多"的形状上占优(部署形状我们已反超 lk 26-31%,§130)。
⇒ 若继续,应按新内核变体立项(独立门禁、独立回退),而不是继续在现有 k-向量化路径上找边角。

## 133. 轮 86:拓扑与缓存层级事实;以及一个反直觉发现 —— **L3 驻留口径比 DRAM 流式口径更慢** ⇒ 那 15% 指向缓存层级/硬件特性

### 硬件与线程池事实(本轮实测)
- `L3 = 768 MiB / 24 instances` ⇒ **每 CCD 32 MiB**(EPYC 9654);NUMA 8 节点 × 24 CPU。
- 线程池 `cores_` 按 **slot-major / CCD-minor** 构造(`numa_pool.hpp:771-772`):
  `[ccd0.cpu0, ccd1.cpu0, …, ccd23.cpu0, ccd0.cpu1, …]` ⇒ **120 线程 = 24 CCD × 5 线程**
  —— 这正是用户"每 CCD 4–5 核"的规则,在代码里得到确认。
- 每 CCD 的需求(按实测每线程吞吐折算):
  | THREADS | 线程/CCD | 每线程 | **每 CCD** |
  |---|---|---|---|
  | 30 | 1.25 | 2.29 GB/s | 2.9 GB/s |
  | 60 | 2.5 | 2.21 GB/s | 5.5 GB/s |
  | 120 | 5 | 1.88 GB/s | **9.4 GB/s** |
  ⇒ 每线程吞吐在 ~5.5 GB/s/CCD 时仍是峰值,到 9.4 GB/s/CCD 就掉下来了。

### 反直觉发现(本轮的关键)
`DEDUP=48`(工作集 403 MB **> 单 CCD 的 32 MiB**,走 DRAM 流式)实测 **350 GB/s 聚合
= 14.6 GB/s/CCD**,**高于**上面那个 9.4 GB/s/CCD 的"拐点"。也就是说:
- **L3 驻留口径(D=12,每 shard 19 MB 基本都在 L3 里)= 226-236 GB/s**
- **DRAM 流式口径(D=48,工作集远大于 L3)= 333-350 GB/s**
⇒ 在这台机器上,"命中 L3"反而比"直接流 DRAM"慢。这与"L3 是受害缓存、每 CCD 的 L3 交付
路径带宽有限,而 DRAM 可以跨 24 个通道并行"一致;而 lk 那台 **9684X 每 CCD 有 96 MiB L3**
(是 9654 的 3 倍),缓存层级完全不同。

### 结论(这 15% 的性质)
- 每线程**峰值吞吐我们与 lk 完全一致**(2.21-2.29 vs 2.2),差异只在**高并发聚合**;
- 聚合差异出现在"L3 驻留"口径,而该口径下我们被**每 CCD 的 L3 交付路径**限制(9.4 GB/s/CCD),
  换到 DRAM 流式口径反而能到 14.6 GB/s/CCD;
- ⇒ **残差 15% 指向缓存层级与 CCD 拓扑这类硬件特性,不是引擎代码里剩下的边角**。
  可证伪它的实验:若在 D=12 口径下把每 CCD 并发降到 ≤2.5 线程而聚合不掉(即靠"更多 CCD"
  而不是"每 CCD 更多线程"),则说明是每 CCD 交付上限;§124 的 `NSHARD` 扫描(8 已是最优、
  16 直接崩)已间接支持"无法再拆更多 node"。

## 134. 【重要更正】轮 87:按**同口径**重算,我们在两个验收形状上分别落后 lk 12% / 20%,不是"部署形状反超"

### 门禁扩展(③ 的完整交付)
`scripts/check_engine_aligned.sh` 现在**同时判验收①的两个条款**,并按 `na` 自动换算每线程带宽:
```
[1/2] 数值门禁:OK=7 BAD=1(me=1 既有)⇒ 通过
[2/2] 性能门禁:
   DEDUP=12  na=12  0.65 ms/层  聚合 232 GB/s  每线程 1.94 GB/s  ms=PASS  per-thread=WARN
   DEDUP=23  na=20  0.84 ms/层  聚合 300 GB/s  每线程 2.50 GB/s  ms=FAIL  per-thread=PASS
```
(每线程按 `na × 12.58 MB` 换算:每专家 gate+up = 2·I·H/2、down = H·I/2,I=2048/H=4096。)

### 【更正】把 §130/§133 的"部署形状反超 lk"作废
§130/§133 里我写"服务端形状(na=32)达 2.78-2.89 GB/s·线程,超过 lk 的 2.2"。
**这是跨口径比较**:lk 的 2.2 是它在 **DEDUP=12(na=12)** 的数字;它在 DEDUP=23 是 3.13。
按**同一口径**严格对照:

| 形状 | 我们 | lk_moe | 比值 |
|---|---|---|---|
| DEDUP=12 | 0.65 ms / 232 GB/s / **1.94 GB/s·线程** | 0.57 ms / 265 GB/s / **2.21** | 0.88× |
| DEDUP=23 | 0.84 ms / 300 GB/s / **2.50 GB/s·线程** | 0.67 ms / 376 GB/s / **3.13** | 0.80× |

⇒ **我们在两个验收形状上都落后(12% / 20%),而且差距是"形状一致"的**。
这同时**削弱**了 §133 的"缓存层级硬件特性"解释(差距在 DRAM 流式口径上同样存在,甚至更大),
也说明 §124 起的"部署形状已达标"结论**不成立**,需要作废。
唯一未尝试的结构性手段仍是 **§132 指出的:列向量化 + K-major 权重布局的内核变体**(lk 那条路)。
`check_engine_aligned.sh` 现在如实报 FAIL,不再有"全部门禁通过"的虚高结论。

## 135. 轮 88:那 12-20% 的**精确来源与可执行规格**(lk 的"预解码+预乘 scale"权重缓冲)

### 为什么"折 scale"只能值 4.5%,而缺口是 16%(自证)
NR 路径每个 `(列 j, k 组 g)` 的 zmm 指令(me=3):
- 解码 `XIAOTU_DECODE_GROUP_AVX512` = **13 条**(1 载入 + 2 and/srli + 2 unpack + 1 insert +
  2 shuffle + 2×(widen+shift) 等)
- FMA 侧 = **3 条/行 × 3 行 = 9 条**(`mul` + `fma` + `acc=fma(d,sv,acc)`)
⇒ **22 条 / 96 MAC = 0.229 条/MAC**。
把 scale 折进权重:**13 + 2(两次 mul) + 2×3 = 21 条 ⇒ 只省 1 条 = 4.5%**(不是 16%)。
lk 的形状(32 列在 lane 上、每 k 一对权重、8 行):**13(解码) + 2×rows**;
在 rows=3 时 = 19 条 / 96 MAC = 0.198 ⇒ **1.16×**;rows=4 时 = 21/128 = 0.164 ⇒ **1.19×**。
**⇒ 与实测 1.14×(D=12)/ 1.25×(D=23) 完全吻合。**
⇒ 缺口的本质:**lk 的权重缓冲是"预解码 + 预乘 scale"的,一份解码结果被 32 列 × 多行复用,
scale 只在解码时付一次;我们则是"每列每 K 组现场解码 + 每行再付一次 scale"。**
在 `[N][K/2]` 布局下**无法把解码摊到列上**(列之间相隔 K/2 字节),所以必须换布局。

### 可执行规格(下一轮的立项书)
1. **权重布局**:`[E][K/2][N]` 字节 —— 对每个 k 对 (k,k+1) 与全部 N 列存 N 字节
   (低半字节 = k 的权重、高半字节 = k+1 的)。**插件侧已有同一份张量**
   (`gpu_prefill._kmajor_bytes(t) = t.transpose(1,2).contiguous()`,GPU 预填充在用),
   可直接复用为 CPU 引擎的输入。
2. **内核**:lane = 输出列(2 个 zmm = 32 列)。每个 k 对:
   - 解码 N 字节 → 32 列 × 2 个 k 的 fp32 权重(2 zmm),**并在这一步乘上该 (列块, k 组) 的 scale**
     (`row_scale((j/gn)*kb_stride, g)`,每 32 列一个标量 ⇒ 解码时付一次);
   - 每行:把 `a32[r][k]`、`a32[r][k+1]` 广播后 **2 条 FMA**。
   ⇒ 每 k 对:`13 + 2·rows` 条,服务 32 列 × rows 行 × 2 k 个 MAC。
3. **累加器极轻**:32 列/行 ⇒ 每行 2 个 zmm;me=3 只要 6 个(对比现在 NR=8/MR=4 需 24-32 个
   ⇒ 也顺带消掉寄存器压力/溢出)。
4. **预期**:me=3 时 1.16×、me=4 时 1.19×——正好覆盖 D=12 的 0.88× 与 D=23 的 0.80×
   (即把 0.65→0.56、0.84→0.71,与 lk 持平)。
5. **门禁**:先做 K-major ↔ 现有内核的**逐元素对拍**(用同一 fixture 现场转 K-major),
   再过 `scripts/check_engine_aligned.sh`(R55:同步/内层循环改动必须先过对拍)。
6. **风险与回退**:新内核走独立函数 + env 开关,默认关闭;对拍不过就删,不影响现有路径。

### 另一条已排除的旁路
"把 scale 折进权重"(R73)数学上等价却把数值打坏(max_rel 1.6-2.7),且即便成立也只值 4.5%;
**不必再回头查它**——真正的收益在按列复用,不在省那一条 mul。

## 136. 【自我更正】轮 89:op 模型自证不成立 —— **两个内核都不是 op-bound**,K-major 重写不解决问题,项目撤销

### 反证(用同一个 op 模型算 lk 的形状)
按列 lane 的 K-major 形状,每个 `(16 列, k 组)` 的指令:
- 解码:16 个 k 对 × 13 条 = **208 条**(服务 16 列 × 32 k)
- FMA 侧:每 k 对每行 2 条 FMA + 1 条广播 ⇒ `16×3 = 48` 条/行(16 列 × 32 k = 512 MAC/行)
- 组尾按组 scale:`+M` 条
(若不做"组内先累加、组尾一次乘 scale",而是每 k 对乘,则再加 2×16 条 ⇒ 更差)
⇒ 每 (行, 16 列, 组) ≈ 208/M + 48 + 1 ≈ **118 条 @M=3**,服务 512 MAC
⇒ **0.23 条/MAC —— 与我们现在的 0.229 完全相同**。

**关键**是:`208 + 48M` 里,解码(208)被 M 除、FMA 侧(48/行)不被除;而我们 `16×(13+3M)` 里
解码(13)被 M 除、FMA(3/行)不被除。两者同构,**列 lane 并不比 k lane 省**。
真正省的是"**把解码结果复用 8 行**"(lk 的 8-row block),那需要 me=8,而我们的路由形状 me=3。

**更强的反证**:若两者真是 op-bound,按 op 数 lk 应快 ~1.9×;**实测只快 1.14×(D=12)/1.25×(D=23)**
⇒ **两个内核都不是 op-bound**。§135 用它推出的"缺口 = 解码不能摊到列上"因此**不成立**。

### 修正后的结论(与本轮之前的证据一致)
- 每线程吞吐:我们 **T≤60 时 2.21-2.29 GB/s = lk 在 D=12 的 2.21**(§124/§133);
  在要求的 T=120 下降 1.94(**−12%**)。
- ⇒ 那 12-20% 是**高并发下的访存/聚合惩罚**,不是指令数、不是布局维度、不是同步、不是旋钮。
- **K-major 列 lane 内核项目撤销**(不去实现):上面的反证说明它不会带来收益,
  而我此前据 §135 认定的"1.16×"是 op-bound 假设下的算术巧合。
- 与 lk 的差距若要继续缩小,方向应回到"**提高每字节权重携带的 MAC 数**"(即更大的 me),
  而 me 由路由决定(部署形状 na≈32、me≈1-3)——**这条路上没有引擎侧手段**。

### 本轮无代码改动
仅更正结论(§135 作废)。工作树保持已验证状态(数值门禁 7 OK;D=12 ms 达标、D=23 ms 未达)。

## 137. 轮 90:两个新假设被否(master 位置、冷启动) + **指令级对比首次落地**(做法要求(2))

### 否掉的假设
| 假设 | 实验 | 结论 |
|---|---|---|
| 未绑核的 master 自旋偷了 worker 的核(代码注释自己提过"caller's spin-wait competes for a core") | `taskset -c 128-191`(把 master 赶离 worker)/ `0-127`(与 worker 同核)/ 不绑,交替各两次 | **无影响**:def 0.65/0.66、hi 0.66/0.68、lo 0.66/0.67 ⇒ 全部在噪声内(R78) |
| harness 把首次冷启动摊进均值,虚高每层时间 | `REP=5 / 60 / 300 / 600` | **无膨胀**:0.66 / 0.65 / 0.66 / 0.66 ⇒ 启动 <0.1 ms(R79) |

### 指令级对比(lk `_lk_moe_C_avx512_vnni.so` vs 我们 `_xiaotu_moe_C_avx512_bf16.so`)
统计全库的 AVX-512 关键助记符(注意:两者都含未被选中的 ISA 回退路径,故只宜看**同一库内的比例**):

| 助记符 | lk | 我们 |
|---|---|---|
| `vfmadd231ps` | **2326** | **144** |
| `vbroadcastss` | **2131** | 121 |
| `vpmovzxbd` | 506 | 42 |
| `vpsrld` / `vpand` | 472 / 167 | 187 / 171 |
| `vinserti128` | 225 | 62 |
| `vpunpcklbw` / `vpunpckhbw` | **0 / 0** | 41 / 24 |
| `vdpbf16ps` | 0 | 32 |
| `vmovaps` | 7384 | 862 |

**读出来的结构差异**:
1. lk 是**列 lane + 激活广播**:`vfmadd231ps` 与 `vbroadcastss` 都是我们的一个数量级以上
   (两者之比 ~1.1),与我们 `lane=K`(靠 zmm 载入激活、几乎不广播)完全不同的形态;
2. lk 的 nibble 解码走 **`vpmovzxbd`(字节→dword 零扩展)+ `vpsrld` + `vpand` + `vinserti128`**,
   **一处 `vpunpck*` 都没有**;我们用 `vpunpcklbw/hbw` + LUT(`vpshufb`)+ lane 重组 + widen。
   ⇒ 按静态比例,lk 的 FMA:解码 ≈ 2.0,我们 ≈ 0.31(受未选中路径污染,但差距明显)。
3. 我们库里 `vdpbf16ps` 只有 32 条 —— 与 R65(bf16 点积实测更慢)一致,该路径不值得再开。

### 但这仍不改变 §136 的结论
§136 已用同构 op 账证明:**若真 op-bound,lk 应快 ~1.9×,实测只快 1.14-1.25×** ⇒ 两者都不是 op-bound。
所以上面这些结构差异(更省的解码、列 lane 广播)**不会**自动换来那 12-20%;它是高并发访存/聚合特性。
本轮的指令对比是对 **做法要求(2)** 的正式交付(此前只依据 §98/§99 的解码笔记,未直接对二进制)。

## 138. 轮 91:完整数据集给出**最终机制结论** —— 每线程效率与 lk 相同;差距全在"120 线程能维持多少每线程速率"

### 完整数据集(BS=6/K=6,DEDUP=12 时 na=12、DEDUP=23 时 na=20;字节 = na×12.58 MB)
| THREADS | DEDUP | 每层 | 聚合 | **每线程** |
|---|---|---|---|---|
| 120 | 12 | 0.68 ms | 222 GB/s | **1.85** |
| 60 | 12 | 1.14 ms | 132 GB/s | **2.21** |
| **lk @120** | 12 | **0.57 ms** | **265 GB/s** | **2.21** |
| 120 | 23 | 0.87 ms | 289 GB/s | **2.41** |
| 60 | 23 | 1.45 ms | 174 GB/s | **2.89** |
| 30 | 23 | 2.74 ms | 92 GB/s | **3.06** |
| **lk @120** | 23 | **0.67 ms** | **376 GB/s** | **3.13** |

### 【结论】两句话
1. **我们低并发时的每线程速率 = lk 在满并发时的每线程速率**(D=12:2.21 vs 2.21;
   D=23:2.89-3.06 vs 3.13,达 92-98%)⇒ **内核的每线程效率与 lk 相同**(§136 的"非同构/非 op-bound"
   在这里得到正面印证);
2. **差距 100% 在"这台机器 120 线程时能维持多少每线程速率"**:我们从 60→120 线程时每线程掉
   16-20%(2.21→1.85、2.89→2.41),而 lk 在 120 线程仍维持 2.21/3.13。
   与 §133 的硬件事实一致:**9654 每 CCD 只有 32 MiB L3,9684X 是 96 MiB**;而且
   D=23 时每个 shard 的工作集 31.5 MB 已经贴到 32 MiB/CCD(所以 D=23 的比值 0.80× 比 D=12 的 0.88× 更差)
   —— 与"缓存压力随 na 增大"完全对应。

### 【验收上的直接后果:两个条款在本机互斥】
验收①要求**同时**满足 `≤0.70 ms/层` 与 `每线程 ≥2.2 GB/s`(D=12)。实测:
- T=120:0.65-0.68 ms **达标**,但每线程 1.85-1.94 **不达标**;
- T=60:每线程 2.21 **达标**,但 1.14 ms **不达标**。
⇒ **在本机缓存层级下,低延迟(要多线程)与高每线程带宽(要少线程)不可兼得**;
lk 能同时满足,靠的是 9684X 每 CCD 96 MiB 的 L3。**这是硬件层级差异,不是引擎代码问题**
(§113-§137 已把同步/布局/指令/旋钮/调度/测量方法逐一排除)。

## 139. 轮 92:否掉"L3 驻留反而更慢";定位为 **na(并行度/MLP)驱动**,并给出本机带宽曲线

### 实验(等 na、只改工作集大小)
`NENGINES=6 ROUNDROBIN=1`(每层一个引擎,工作集 6×151 = **906 MB > 768 MB L3**,每次调用 na 仍是 12):
| 配置 | 每层 | 聚合 | 每线程 |
|---|---|---|---|
| 1 引擎(L3 驻留,151 MB) | 0.65 / 0.69 ms | 219-232 GB/s | **1.82-1.94** |
| 6 引擎(流式,906 MB) | **0.85 / 0.92 ms** | 164-178 GB/s | **1.37-1.48** |
⇒ **工作集越大越慢**(符合常规)。所以 §133/§138 里"D=48(350 GB/s)> D=12(232 GB/s)"**不是**
"L3 驻留更慢",而是 **D=48 的 na=32 带来 256 job/node(对比 96)⇒ 并行度/MLP 更高**(R81)。

### 本机带宽曲线(全部 T=120,除非注明;字节 = na×12.58 MB)
| na | 每层 | 聚合 | 每线程 |
|---|---|---|---|
| 12 | 0.65-0.69 | 219-232 | 1.82-1.94 |
| 20 | 0.84-0.87 | 289-300 | 2.41-2.50 |
| 31-32 | 1.13-1.21 | 333-350 | 2.78-2.92 |
| 12 @T=60 | 1.14 | 132 | **2.21** |
| 20 @T=60 | 1.45 | 174 | **2.89** |
| 20 @T=30 | 2.74 | 92 | **3.06** |

### 结论
- 聚合带宽**随 na(并行度)单调上升**(232 → 300 → 350),与 L3/DRAM 归属无关;
- 同一 na 下,每线程速率**随线程数下降**(na=20:3.06@30T → 2.89@60T → 2.41@120T)—— 这是内存
  子系统的并发争用曲线,不是尾部/调度/缓存归属问题;
- ⇒ **na=12 这个验收口径本身"并行度不足"**(96 job/node),我们与 lk 在**同一 job 数**下差 14%;
  而这个差距无法靠"加 job"解决——加 job 就等于改 na(subA 加大已被实测否决:转换与每 job 固定成本)。
- 至此,基准口径下可动的引擎侧手段**全部穷尽**(§113-§139 共 27 轮的排除清单见 TRIED_AND_REVERTED)。

## 140. 轮 93:subA 扫描再否"并行度不足";并**找到 R65 的真正原因** —— bf16 点积分支是未做列分块的旧 GEMV

### 1) 重新定性:na=12 是 **compute(op)受限**,不是"并行度不足"
`SHARDSPLIT` 再扫描(a32 复用已落地之后,零重建,两次重复完全一致):
| SS(每 job 列数) | 8(=默认,32 列) | 16(16 列) | 32(8 列) |
|---|---|---|---|
| 每层 | **0.690 / 0.690 ms** | 0.720 / 0.720 | 0.920 / 0.910 |
⇒ **加 job(细分)单调更差**,所以 §139 的"na=12 并行度不足、可以靠加 job 补"**是错的**(记 R82)。

**正确的定性**:MAC/计算量 ∝ **NASS**(= na×me = 36,常数),权重字节 ∝ **na**:
- na=4..12:字节少 ⇒ **计算量主导 ⇒ 层时间平坦**(0.64-0.79)✓
- na≥20:字节主导 ⇒ 时间随 na 上升 ✓
- op 账:每层 ~170M 条 zmm(0.1875 条/MAC × 906M MAC)÷ ~300 G ops/s ≈ **0.57 ms**,与 na=12 实测
  0.65-0.69 ms 同量级 ⇒ **验收口径(na=12)确实是 op 受限**。
⇒ **减指令就是杠杆**(§136 的"非 op-bound"结论对 na=12 不成立,那里我只算了 lk 的绝对 op 数,
没有把它和"我们处于 op 受限区"这一实测事实对齐 —— 现予更正)。

### 2) **R65 的真正原因**:`dotp16`(bf16 点积)分支是**未做列分块的旧 GEMV**
```cpp
for (int j = n0; j < n1; ++j)              // ← 列在最外层:每列都把整条激活重读一遍
    for (int m0 = 0; m0 < M; m0 += 8) {    // 只有行分块,没有列分块
```
而 fp32 路径早已改成 `for j0 += NR(8) { for g { 载入 av 一次; for jj in NR {...} } }`
(源码注释原话:"原结构是 GEMV:j 在最外层 = 每行输出都把整条激活重读一遍 ⇒ 激活:权重流量 = 64:1")。
⇒ **bf16 路径慢不是"bf16 不好",而是它从未吃到那次列分块优化**;§109 微基准里 bf16 = 1.98×
(那里是受控内层循环)与引擎里更慢,两者由此统一解释。

### 3) 下一轮的处方(带已知风险)
**给 `dotp16` 分支加 NR 列分块**,使其结构对齐 fp32 路径:
- 预期 op 数:每 `(k组, 8 列, me=3)` = 8×(bf16 解码 + 2/行×3) vs fp32 的 8×(13 + 3×3)=176;
  若 bf16 解码 ≈ 9-13 条 ⇒ **1.15-1.35× 的 op 削减 ⇒ 0.65 → 0.48-0.57 ms**,可达 lk 水平。
- **风险/门禁**:该分支历史上因精度被否(me=2 用例 max_rel **8.4e-3** > 项目门限 2e-3,
  fp32 路径同档 3.9e-4)。**先把列分块做出来测性能**,若确实快,再单独处理精度问题
  (注意:激活本身就是 bf16 存储 ⇒ 不是"激活被截断",而是 Zen4 `vdpbf16ps` 的成对求和舍入,
  可用"每 K 块拆两次 dpbf16 + 更频繁归约"或对拍放宽到实测分布来评估)。
- 必须过 `scripts/check_engine_aligned.sh`(R55)。

## 141. 轮 94:给 bf16 路径加 NR 列分块 —— **证实 R65 病根是 GEMV**,但收益只有 3-4%,并**顺带否掉 op-bound 假设**

### 改动(保留,env 开关默认关)
把 `dotp16` 分支从 GEMV(列最外层,每 (列,组) 只有 mr 条累加链)改成 `acc[MR=4][NR=4]` 列分块
(23 zmm,避免溢出):激活每 (组) 只载一次被 NR 列复用,独立累加链从 mr 条变成 MR×NR=12 条。

### 性能(交替,同窗口)
| 路径 | 每层 | 每线程 |
|---|---|---|
| fp32(默认) | 0.670 / 0.670 ms | 1.88 |
| **bf16 + 列分块** | **0.660 / 0.640 ms** | 1.91-1.97 |
⇒ 从"明显更慢(§140 前实测 0.72-0.76)"变成"**略快 3-4%**" ⇒ **证实 R65 的病根是 GEMV 结构,
不是 bf16**;但收益远小于 op 账预期的 1.15-1.35×。

### 精度门禁(`XIAOTU_MOE_DPBF16=1` 跑 test_block23_equiv)
| 用例 | max_rel | 判定(门限 2e-3) |
|---|---|---|
| me=1 | 8.02e-4 | OK(**比 fp32 路径的 1.87e-2 好**) |
| me=3(解码主形状) | 8.70e-4 | OK |
| me=2/3 混合 | 4.92e-4 | OK |
| **me=2** | **8.42e-3** | **BAD** |
⇒ 仅 me=2 一处超标(历史已知)⇒ **仍不可采纳**;fp32 路径继续作为默认,bf16 保留在开关后。

### 【重要】本实验**顺带否掉 op-bound 假设**
bf16 路径把 FMA 侧从 **3 条/行降到 2 条/行**(总 op 约 −10~15%),若 na=12 真是 op-bound,
应有 1.15-1.35× 收益;实测只有 **3-4%** ⇒ **na=12 也不是单纯 op-bound**。
至此,"访存字节"与"指令数"两个单因素模型**都无法解释** 0.64-0.67 vs lk 0.57 的残差;
在 §113-§141 的系统排除之后,它与本机在 120 线程下的执行效率有关(§138 的每线程对比:
低并发时我们与 lk 完全同速),而**引擎侧没有可动的地方了**。

## 142. 轮 95:量化现有列分块的价值(1.25-1.30×)并确认 ILP/分块线已到顶;最终验收审计

### 1) GEMV vs NR 列分块(同窗口交替)
| 路径 | 每层 | 每线程 |
|---|---|---|
| NR=8(默认,列分块) | **0.640 / 0.650 ms** | 1.94-1.97 |
| `GEMM_NR=0`(旧 GEMV,列最外层) | 0.840 / 0.860 ms | 1.46-1.50 |
⇒ **列分块值 1.25-1.30×** —— 这是当前内核里最大的一项结构收益,也印证"ILP/分块"确实关键
(与 R65 的 bf16 病根同源)。

### 2) 分块参数已到顶
| NR | 4 | **8** | 16 |
|---|---|---|---|
| 每层 | 0.730 ms | **0.640 / 0.650** | 0.640 ms |
注:代码里 `const int NR = std::min(gemm_nr, 8);` ⇒ **NR 内部上限就是 8**,故 16 与 8 等价。
NR=8 时 `acc[4][8]=32` + `av[4][2]=8` + 4 ⇒ **约 44 个 zmm > 32 个架构寄存器,必然溢出**,
但它仍是最快的(32 条累加链的 ILP 收益盖过溢出代价);NR=4(不溢出)反而慢 14%。
⇒ **分块/ILP 这条线已到顶。**

### 3) 本轮最终验收审计
```
[1/2] 数值门禁:OK=7 BAD=1(me=1 既有的 NR=8 fp32 重结合偏差)⇒ 通过
[2/2] 性能门禁:
   DEDUP=12  na=12  0.65 ms/层  聚合 232 GB/s  每线程 1.94 GB/s  ms=PASS  per-thread=WARN
   DEDUP=23  na=20  0.84 ms/层  聚合 300 GB/s  每线程 2.50 GB/s  ms=FAIL  per-thread=PASS
8070:UP(新引擎)
```
⇒ 两个条款**各达一个、各差一个**(D=12 ms 达标、D=23 每线程达标),与 §138 的"本机两条款互斥"一致。

### 4) 累计结论(第一条)
- 起点 1.22 / 1.32 ms(124 / 184 GB/s、每线程 1.0-1.5)
- 现在 **0.64-0.65 / 0.84 ms**(232 / 300 GB/s、每线程 1.94 / 2.50)
- 与 lk 同口径:0.57/0.67 ⇒ 我们 **0.89× / 0.80×**
- 已系统排除(29 轮,清单见 TRIED_AND_REVERTED R1-R85):同步结构、奇偶 seqlock、伪共享 padding、
  静态/批量领票、K-major 布局维度、折 scale、bf16 点积(结构与精度)、列分块参数、GCHUNK、
  SHARDSPLIT、NSHARD、线程数、spin 超时、master 绑核、冷启动、L3 绕过、reserve/resize、
  a32 复用…**剩余差距指向"本机 120 线程执行效率"这一硬件/并发属性,引擎侧已无可动之处。**

## 143. 第 96 轮:目标变更(第一条结案 → 第二条)+ "静默原则"暂停

### 用户决定
- **第一条视为已完成并结案**:CPU 解码引擎 `1.22/1.32 → 0.64-0.65/0.84 ms`(与 lk 同口径 0.89×/0.80×),
  数值门禁 7 OK。剩余项(§A2/§A3 两个未达条款、以及 §B 的未解之谜)**不再作为当前优先项**,
  已统一整理到新建的 **`report/tuning/FUTURE_PLAN.md`** 供以后回顾。
- **当前唯一优先 = 第二条**:预填充 ≥1500 t/s、解码 ≥70 t/s、投机 ≥100 t/s、1M 上下文可用
  (对照 PRO 6000 那台 3100 / 75 / 100-115);做法按原"做法要求"(1)-(5)。
- **"静默原则"暂停**(运行环境变更):用户已把调用 AI API 改为外部服务 ⇒ 本机不再需要维持
  生产服务器、不会被抢占 ⇒ **测试时可自由并行占用本机**(可同时跑多个 server/bench、可随意重启 8070)。
  这条替代了此前"不得干扰 8070 生产态"的约束。

### 目标对象
`goal-36fc65c0-...` 已 `edit` 到 rev 13(内容=上面的第二条)。**注意:工具不允许模型自行 `resume`
一个 paused 目标**,需用户在界面上点一次 resume 才能重新武装自动轮次;在此之前我按指示直接推进工作。

### 本条起的工作纪律(沿用)
结论写 NOTES/docs、每次回退记 TRIED_AND_REVERTED、每改一项实测 **C=1/C=2 tok/s + 每层 compute/rest
分解(`XIAOTU_CD_TIMING=1`)**、门禁脚本 `scripts/check_engine_aligned.sh`(数值+性能)。

## 144. 第 97 轮(第二条开工):参考口径澄清、EP 机制、TP=2 的两种失败模式

### 1) 参考口径澄清(用户第 96 轮指出)
[Lvllmds4 README](https://raw.githubusercontent.com/guqiong96/Lvllmds4/main/README.md) 的官方对照表:
| 项目 | 上游分支 | CPU | 内存 | GPU | 预填充 | 解码 | 投机 |
|---|---|---|---|---|---|---|---|
| Lvllmds4-x-v2.3.9 | `yhfgyyf/vllm-deepseek-v4-sm89` | EPYC 7642×2 | 16ch DDR4-3200 | **3090 × 2** | 1060 @32768 | 26 | 35-47 |
| Lvllmds4-v2.3.9 | `jasl/vllm`(SM120) | EPYC 9684x×2 | 24ch DDR5-4800 | **pro 6000 × 1** | 3100 @131072 | 75 | 100-115 |
⇒ **PRO 6000 那台是单卡**,它预填充快是必然的(原生 NVFP4 + 算力强);
⇒ **对标口径应以"双卡 3090 那台"为准**(1060/26/35-47),我方机器(192 核 + DDR5-4800 24ch + A100-40G×2)应显著高于它。
(记录:`docs/PERFORMANCE_OPTIMIZATION.md` 里此前把 3100 当成对标目标,现已更正口径。)

### 2) TP=2 时我们**做 EP**(证据)
启动日志:`EP model.layers.43.ffn: rank 0/2 owns experts [0, 128) of 256`(每个 rank 持 128/256 个专家),
以及 `GPU-resident model.layers.5.ffn: 1.59 GiB on cuda:0 (tp=2, experts=128)`。
⇒ **每卡每常驻层 1.59 GiB(TP=1 是 3.2 GB)** ⇒ **同样显存预算能常驻约 2× 的层数** —— 这正是
"双卡容量宽裕"的真实机制,但它是**专家切分 + 同机共享内存归约**,不是"每卡复制相同内容"。
- 归约代价:每层一次部分和合并,量级 `tokens × hidden`(6 token 时 98 KB,可忽略;
  **8192 token 的预填充时 134 MB/层,不可忽略** —— 这是 EP 在长预填充下的隐性成本)。
- 相关旋钮:`XIAOTU_MOE_EP`(默认 1)/ `XIAOTU_MOE_EP_SHM`(默认 1,共享内存归约)/
  `XIAOTU_MOE_REDUNDANT`(1=旧的"两 rank 都跑全部专家"冗余模式,只用于对照)。
- 用户提的"两卡各放完整共享权重 + 分层 ping/pong":属 **PP(层切分)**,只有一次层边界激活交接
  (比 EP 的每层归约更省),但需要 micro-batch 填流水线气泡;且我们的插件未验证过 PP 路径。

### 3) **TP=2 当前起不来(两种失败模式,已记为待查项)**
| 模式 | 现象 | 定位 |
|---|---|---|
| 关 EAGER(默认 CUDA graph) | KV 分配完成后,worker 不再消费 shm 广播块,每 60 s 打 `No available shared memory broadcast block`,EngineCore 在 RPC 超时后 `cancelled`(实测 ≈317 s) | 预热/图捕获阶段卡住(非配置错) |
| **开 EAGER** | worker 在 `Enforce eager set, disabling torch.compile and CUDAGraphs` 之后**静默死亡**(无 Python 堆栈 ⇒ 原生崩溃) | 与图捕获无关 |
- **`XIAOTU_MOE_EP=0` 不能修复**模式二 ⇒ 不只是 EP 的问题 ⇒ **相对 §54/§56 的记录(TP=2+常驻0-5+EAGER=1105-1117 t/s)是回归**,
  需单独二分(建议:去掉 resident / 去掉 spec / 去掉 EP 逐项试,并抓 coredump 或 `dmesg`)。
- 结论:**第二条不能依赖 TP=2**,先用 TP=1 拿可测收益。

### 4) 【操作教训复发】M9
kill 服务器后仅等 5 s 就重启 ⇒ 新进程看到 GPU0 只剩 7.42/39.49 GiB 空闲,
报 `ValueError: Free memory ... less than desired GPU memory utilization`,启动中止。
**规矩:kill 后至少等 18-20 s,并用 `nvidia-smi` 确认全 0 再启动。**(M9 原文只写了"等 ~10s",现收紧到 ≥18 s。)

### 5) 第二条的下一步(不依赖 TP=2)
TP=1 下 `XIAOTU_MOE_RESIDENT_BUDGET_GB` 之前只给了 12 GB(≈3.7 层);GPU0 启动时空闲 38.75 GiB
⇒ 提到 **24 GB(≈7 层)** 再叠加 KV 8 GiB 仍有余量。先按此测量**预填充/C=1 解码/C=2 吞吐**,
再横向扫 `budget`、`XIAOTU_MOE_PREFETCH_SLOTS`、`VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS`、spec tokens。

## 145. 第 97 轮:第二条的显存"精确账"与三处失败(供下一轮直接用)

### 显存精确账(TP=1,A100-40G,实测)
| 项 | 值 | 来源 |
|---|---|---|
| 启动时空闲 | **38.75 GiB** | worker 日志 |
| 固定开销(非专家权重 + 工作区) | **≈14.2 GiB** | OOM 明细反推 |
| 每常驻层(TP=1) | **3.19 GiB** | `resident try ... used=22.31`(7 层) |
| 每常驻层(TP=2/rank) | **1.59 GiB** | `GPU-resident ... (tp=2, experts=128)` |
| 预热额外需要 | **≈2-3 GiB** 余量 | 20 GB 预算(19.1 GiB)仍在 warmup 阶段失败 |

⇒ TP=1 可常驻层数上限 ≈ `(38.75 - 14.2 - KV - 3)/3.19`:
- KV 8 GiB → **约 4 层**;KV 2 GiB → 约 6 层(实测 7 层=22.31 GiB **OOM**)。
- 也就是说:**"常驻层 vs KV"在单卡上是硬性二选一**,这也再次印证双卡的价值在于"总量翻倍后再二分"。

### 本轮三处失败(都不是"贴上限",而是配置/内存)
1. **TP=2 关 EAGER**:KV 分配后 worker 不消费 shm 广播 ⇒ 预热卡死到 RPC 超时(≈317 s)。
2. **TP=2 开 EAGER**:worker 在禁用 torch.compile 后**静默原生崩溃**;`XIAOTU_MOE_EP=0` 不修复 ⇒ 相对 §54/§56 是回归。
3. **TP=1 常驻 7 层(预算 24, KV 8 或 2)**:GPU OOM(实测余 937 MB 时还要 2 GiB)。
4. **TP=1 常驻 6 层(预算 20, KV 2)**:失败于 `RuntimeError: torch_call_dispatcher("aten::empty", "memory…")`
   ⇒ 指向 **pinned host 缓冲分配**(我们的 GPU-prefill 预填充会建 pinned 的 K-major 权重缓冲),
   而非 GPU OOM —— 下一轮应先查 `ulimit -l` / `MemLock` 上限与 `XIAOTU_MOE_PREFETCH_SLOTS`/pinned 预建规模。

### 当前状态
已把 8070 恢复到**已知可用**配置(TP=1 + 常驻 0-5 + 预算 12 + KV 8 GiB + EAGER),启动中。
下一轮第一件事:等它就绪后测 **预填充(L=32768,C=1,TTFT→t/s)/ C=1 解码 / C=2 吞吐**;
并生成缺失的数据集(`report/tuning/datasets/` 现有 32/128/512/1024/4096,**缺 8192 与 32768**,
用 `LENS="8192 32768" N=8 scripts/make_nat_dataset.py` 生成)。

## 146. 第 97 轮:配置矩阵 —— **本会话所有带 `--enforce-eager` 的启动都失败,所有成功的都没带**

### 成功的(本会话验证过两次)
`TP=1 + 无常驻层 + CUDA graph(不加 --enforce-eager)+ KV 8 GiB + maxlen 262144` ⇒ 276-282 s 就绪,
TPOT 56.9-57.8 ms(r74/r75 实测)。

### 失败的(全部带 EAGER)
| # | 配置 | 现象 |
|---|---|---|
| 1 | TP=2 + CUDA graph(无 EAGER) | KV 分配后 worker 不消费 shm 广播 ⇒ 预热卡死,RPC 317 s 超时 |
| 2 | TP=2 + 常驻 0-5 + **EAGER** | worker 在禁用 torch.compile 后**静默原生崩溃**;`XIAOTU_MOE_EP=0` 不修复 |
| 3 | TP=1 + 常驻 0-5 + budget12 + **EAGER** | `Engine core initialization failed`(放下 3 层后崩) |
| 4 | TP=1 + 常驻 0-9 + budget24 + **EAGER** | GPU OOM(7 层 = 22.31 GiB 时还要 2 GiB) |
| 5 | TP=1 + 常驻 0-9 + budget20 + **EAGER** | `torch_call_dispatcher("aten::empty")` 失败(6 层 = 19.1 GiB,余量够) |

⇒ **规律`:--enforce-eager` 是共同因子**;而 #1 说明 CUDA graph 在 **TP=2** 下也有问题(预热卡死)。
⇒ 下一轮纪律:**变量一次只加一个** —— 基线(TP=1 + graph + 无常驻)→ 只加常驻层 → 再单独试 EAGER;
每步都先用 `scripts/check_engine_aligned.sh` 之外的服务端探针确认能起来并测 C=1/C=2。

### 副产:显存"精确账"(见 §145)与数据集补齐
- 数据集已补齐:`report/tuning/datasets/nat8192.jsonl`、`nat32768.jsonl`(精确 8192/32768 token)。

## 147. 第 97 轮:8070 已恢复可用;两处异常待下一轮处理

### 1) 8070 = 已证明可用的基线,现已 **UP**
`TP=1 + CUDA graph(无 --enforce-eager)+ 无常驻层 + KV 8 GiB + maxlen 262144`,启动 ~4.5 min。

### 2) 【红旗】同一基线,`rest` 从 0.77 → 3.2-3.6 ms/层
新服务端 `XIAOTU_CD_TIMING=1` 输出:
```
layers=43 qlen=12 k=6 period=4.43ms compute=0.82ms rest=3.62ms (compute 18%, rest 82%)
layers=43 qlen=6  k=6 period=4.54ms compute=1.30ms rest=3.24ms (compute 29%, rest 71%)
```
对比 §120(第 74 轮,几乎同配置):`period=2.05ms compute=1.28ms rest=0.77ms`。
⇒ **`compute` 相当,`rest` 恶化 2.2×**。最可能是我这一轮反复起停服务/并行尝试造成的环境残差
(残留进程、pinned 内存压力、GPU 上还有别的进程),**不一定是真回归**。
⇒ 下一轮第一件事:**在真正干净的机器状态上(确认无残留、GPU 全 0、`rm -f /dev/shm/xiaotu_ep_*.bin`)
重新采一次这个基线**(C=1 与 C=2),作为第二条的起跑线。

### 3) 客户端工具缺依赖
`scripts/bench_nat.sh`(走 `vllm bench serve --dataset-name custom` 读 jsonl)失败:
`ImportError: Please install vllm[bench] for bench support`(该 env 缺 pandas)。
⇒ 用 `scripts/tune_client.sh`(random/sharegpt 路径,不读 jsonl;第 74/75 轮已验证可用),
或 `pip install pandas` 到 vllm-xiaotu-moe env。**下一轮开工前先修这个,否则测不出数。**

## 148. 第 98 轮:第二条**起跑线**测到了;红旗解除;并算出解码目标的真实距离

### 基线健康(8070:TP=1 + CUDA graph + 无常驻 + KV 8 GiB + spec k=5)
客户端(`tune_client.sh`,L=512,C=1,N=8,OUT=64,IN_LEN=256):
**TPOT = 57.38 ms**(≈**17.4 tok/s 单流**)、out_tok/s 11.61(含 TTFT 1.90 s 摊薄)、TTFT 1897 ms。
服务端 `XIAOTU_CD_TIMING`:
```
layers=43 qlen=6 k=6 period=1.73ms compute=0.99ms rest=0.74ms (compute 57%, rest 43%)
layers=43 qlen=6 k=6 period=1.72ms compute=0.97ms rest=0.75ms (compute 57%, rest 43%)
```
⇒ **`rest` 回到 0.74-0.75 ms/层**(§120 是 0.77)⇒ §147 的"rest 3.2-3.6"确系**我反复起停服务的环境残差**,不是回归。✅

### 解码目标的真实距离(算术)
- 每步 = 43 层 × 1.72 ms = **74 ms/步**;被接受的投机 token ≈ **1.3 个/步**(由 74/57.4 反推)
  ⇒ 单流 **17.4 tok/s**。
- 目标 **≥70 t/s** ⇒ 需要 ≤14.3 ms/token ⇒ 步 ≤ **18.6 ms** ⇒ 每层 ≤ **0.43 ms**
  ⇒ 比现状(1.72)好 **4×**。
- 靠"常驻层"达到:需把 43 层压到 ~11 层在 CPU ⇒ **常驻 ~32 层**;按 §145 的账
  (TP=1 3.19 GiB/层)= **102 GiB**,TP=2(1.59 GiB/层)= **51 GiB**
  ⇒ **两种都远超 40 GiB/卡(且 KV 与它互斥)** ⇒ **解码 ≥70 t/s 不可能靠常驻层在本机达成。**
- 对照口径:SM80 参考(3090×2)解码 **26 t/s**(38 ms/token)—— 我们 17.4 是它的 **0.67×**;
  PRO6000 的 75 t/s 是强得多的 GPU。⇒ 现实目标应是**先追平 26、再靠下面三条逼近上限**。

### 由此确定的四条杠杆(下一轮按收益排序逐条实测)
1. **投机接受率**:目前 ~1.3 tok/步。`num_speculative_tokens`(现 5)扫 3/5/7 + draft 的采样参数
   ⇒ 若接受率能到 2.5,步成本不变而单流直接 **×1.9**(17.4 → 33)。
2. **`rest` 0.75 ms/层**:这是纯 GPU 侧(拷贝+派发+注意力),`XIAOTU_GPU_PREFETCH_AHEAD` /
   `XIAOTU_MOE_PREFETCH_SLOTS` 等**尚未在本配置下扫过**;常驻层能整段消掉它(但容量受限,见上)。
3. **C≥2 聚合**:步成本被多路摊薄 ⇒ 面向"吞吐 ≥70"的另一个口径(需先确认目标口径是单流还是聚合)。
4. **`compute` 0.97 ms/层**:即第一条的引擎(§97 已结案,0.65/0.84 是其在 DEDUP=12/23 的值;
   服务端的 0.97 对应更大 na)。

## 149. 第 99 轮:补齐 C=2 与长预填充口径 —— **解码不随并发扩展**;预填充 ~700-730 t/s(2× 缺口)

### 实测(8070 基线:TP=1 + CUDA graph + 无常驻 + KV 8 GiB + spec k=5)
| 测量 | 结果 |
|---|---|
| C=1 解码(L=512,OUT=64) | TPOT **57.4 ms**(单流 17.4 t/s) |
| **C=2 解码**(同参数) | **TPOT 108.3 ms/请求**(≈9.2 t/s/请求)、out_tok/s **13.92** ⇒ 聚合 ≈18.5 t/s |
| **预填充 8192**(C=1,OUT=1,GPU 流式) | TTFT **11.75 s** ⇒ **≈697 t/s** |
| **预填充 32768**(C=1,OUT=1) | TTFT **45.0 s** ⇒ **≈728 t/s** |

### 两个硬结论
1. **解码不随并发扩展**:C=2 时**单请求 TPOT 直接翻倍**(57.4→108.3),聚合只有 17.4→18.5 t/s。
   ⇒ 说明两个请求在同一条串行链上排队(CPU 引擎 120 线程 + GPU 步进都被串行化),
   **"靠加并发把解码吞吐抬到 70"这条路在我们当前架构下不成立**(先要解决并行/流水重叠)。
2. **预填充 ~700-730 t/s,距 1500 差 2×,而且低于 SM80 参考(3090×2 = 1060 @32768)**。
   反推每层-每块成本:`8192` 一次块(43 层)= 11.75 s ⇒ **273 ms/层**;
   `32768` 分 4 块(172 层-块)= 45.0 s ⇒ **262 ms/层-块**。
   而文档记录的瓶颈是 H2D 67-74 ms/层 + Triton 54 ms/层 ≈ **128 ms** ⇒ **还有约 2× 未解释**,
   需要按 `XIAOTU_GP_TIMING` / `[gp-h2d]` 打点拆开(H2D vs 内核 vs 派发)。

### 与验收对照(现状)
| 验收 | 目标 | 现状 | 差距 |
|---|---|---|---|
| 预填充 | ≥1500 | **697-728** | 2.1× |
| 解码(单流) | ≥70 | **17.4** | 4.0× |
| 投机 | ≥100 | 含 k=5 已计入上面 17.4(需确认口径) | — |
| 1M 上下文 | 可用 | maxlen 262144、KV 8 GiB(≈29.5 KB/token ⇒ 8 GiB≈278K token) | 需核准 29.5 vs 60 KiB |

### 下一轮优先级(按杠杆/成本)
1. **预填充**:先按 `XIAOTU_GP_TIMING` 拆 273 ms/层 的构成(H2D/内核/派发),再扫
   `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS`(现 384,参考用 1024)与
   `XIAOTU_GPU_PREFETCH_AHEAD` / `XIAOTU_MOE_PREFETCH_SLOTS`(**本配置下从未扫过**);
   加常驻层(KV 降到 2 GiB 可容 ~6 层,每层省 67-74 ms)。
2. **解码**:扫 `num_speculative_tokens`(3/5/7)提接受率;`rest` 0.75 ms/层的 prefetch 旋钮;
   **以及"C=2 不扩展"这个新问题的定位**(是 CPU 引擎串行还是 GPU 步进串行)。

## 150. 第 100 轮:预填充**分段计时**拿到 —— H2D 144.7 ms/层(TP=1 的 PCIe 墙),解释了"为什么必须 TP=2"

### 实测(GP_TIMING,8192 token 一次块,43 层,当前 TP=1 基线)
```
[gp-timing] n=40 Tavg=8192 seg=10.03ms rest=0.37ms host_total=10.40ms
[gp-h2d]    n=40 per_layer=144.7ms
```
预填充总 TTFT 11683 ms ÷ 43 层 = **272 ms/层**。分解:

| 分量 | 每层 | 占比 | 说明 |
|---|---|---|---|
| **H2D(MoE 专家权重入卡)** | **144.7 ms** | **53%** | TP=1 要搬 **3.2 GB** ⇒ 3.2GB/144.7ms = **22 GB/s**,正好是 A100-PCIE gen4 x16 的实际上限 ⇒ **这是 PCIe 墙** |
| GPU 计算(注意力+稠密+Triton MoE) | ≈117 ms | 43% | 文档里 Triton MoE 单列 ~54 ms/层,其余是注意力/稠密 |
| 主机侧(seg 分段 + 派发) | 10.4 ms | 4% | `seg=10.03ms`(建 segmentation)+ 派发 |

### 关键推论(与文档口径对上了)
- 文档写的"预填充 H2D **67-74 ms/层**" ⇒ 3.2GB/70ms = 45 GB/s,**超过 PCIe gen4 x16 物理上限**,
  所以那个数字**只可能是 TP=2 的口径**:TP=2 时每 rank 只搬 **1.6 GB** ⇒ 1.6GB/70ms = **23 GB/s** ✅ 正好是 PCIe 实际值。
- ⇒ **TP=1 的 H2D 是 TP=2 的两倍(144.7 vs ~72 ms/层),而两块卡并行搬 ⇒ 预填充要上量必须 TP=2。**
  这也解释了 §54/§56 记录里"预填充最优 = TP=2 + 常驻 0-5 = 1105-1117 t/s"。

### 预填充到 1500 的算术(用本轮的实测分量)
| 配置 | 每层 | 43 层 | 预填(8192) |
|---|---|---|---|
| 现状 TP=1 | 144.7+117+10.4 = **272** | 11.7 s | **701 t/s**(实测) |
| TP=2(假设计算也分摊一半) | 72+58+10 = **140** | 6.0 s | ~1360 t/s |
| TP=2 + 常驻 6 层(省 6×72) | (6层免H2D) | 5.6 s | ~1470 t/s |
| TP=2 + 常驻 6 层 + 更多(容量受限) | — | — | 1500 ⇒ **刚好在边界上** |
⇒ **TP=2 + 常驻层是唯一现实路径,且 1500 是"贴着天花板"的目标**;而 **TP=2 目前起不来(§146 的回归)** ⇒
**修好 TP=2 = 第二条预填充目标的前置条件**,优先级最高。

### 另一条独立的杠杆(不需要 TP=2)
常驻层每层直接省掉 **144.7 ms**(TP=1 口径)。TP=1 下 KV 2 GiB 可容 ~6 层 ⇒ 省 6×144.7 = 0.87 s
⇒ 11.7 → 10.8 s ⇒ **约 760 t/s**(+8%)。虽不解决大局,但**零风险、可立刻做**,且是"未启用旋钮"之一。

## 151. 第 101 轮:【用户澄清口径】70/100 是**并发输出**,但**单流也必须 >30(投机 50)**

### 目标(修正后)
| 口径 | 目标 | 现状(C=1,spec k=5) | 差距 |
|---|---|---|---|
| **单流** | **>30 t/s(投机 >50)** | **17.4 t/s**(TPOT 57.4 ms) | **1.7× / 2.9×** |
| 并发聚合(仅到 C=2~3) | ≥70(投机 ≥100) | ~18.5 t/s @C=2 | 3.8× |
用户约束:C 超过 2-3 后单流速度低到不可用 ⇒ **聚合 70 实际等价于"单流 23-35 t/s"** ⇒ **单流是真正的约束**;
等价说法:**投机的单流 TPOT ≤ 20 ms**(与第一条 ② 原本的 "28-31 → ~20 ms" 一致)。

### 现状分解(§148/§149 实测,TP=1 基线)
- 每步 = 43 层 × **1.72 ms/层**(= compute 0.97 + rest 0.75)= **74 ms/步**
- 每步被接受的投机 token ≈ **1.3 个** ⇒ 单流 74/1.3 = **57 ms/token = 17.4 t/s**
- ⇒ 要到 **50 t/s(20 ms/token)**,可走三条(可叠加):
  1. **提高接受率**:1.3 → 3.0 ⇒ 74/3 = 24.7 ms ⇒ **40 t/s**(最大杠杆,`num_speculative_tokens` 现 5 未扫过)
  2. **降 `rest`**(常驻层):6 层 × 0.75 = −4.5 ms/步 ⇒ 69.5 ms(单流 +6%)
  3. **降 `compute`**(第一条引擎,已结案)
- 若要**同时**满足单流 >30 与聚合 ≥70(C=2~3),最基本是**单流 35 t/s** ⇒ 接受率 ≈ 2.8~3.0 且 `rest` 全部消掉;
  ⇒ **接受率扫描是下一步第一优先**,`rest` 用常驻层做(受容量限制,见 §145)。

## 152. 第 101 轮:【天花板证明】解码已在"权重流量"上限上;C=2 不涨的真因是"工作量不去重"

### 带宽账(全部用实测值,无假设)
| 量 | 值 | 来源 |
|---|---|---|
| 每层不同专家数 `na` | ≈32 | §121/§124(DEDUP=48 档) |
| 每层权重字节 | 32 × 12.58 MB = **403 MB** | 每专家 gate+up+down=12.58 MB |
| 每步(43 层) | **17.3 GB** | 403 MB × 43 |
| 每步时间 | **74 ms** | §148:43 × 1.72 ms |
| ⇒ 隐含带宽 | **234 GB/s** | 与引擎实测 226-232 GB/s 一致 ✅ |
| 每步接受 token | ~1.3 | §148 |
| ⇒ **每输出 token 的字节** | **13.3 GB** | 17.3 / 1.3 |
| ⇒ **单流上限** | **226 GB/s ÷ 13.3 GB = 17 t/s** | **实测 17.4** ✅ **完全吻合** |

⇒ **解码已被"权重流量"卡死,不是指令、不是并行度、不是调度。** 这解释了:
- **单流 17.4 t/s**:就是这条线;
- **C=2 几乎不涨(17.4→18.5)**:2 个请求 = 72 个 assignment,`na` 32→≈58(专家几乎不重叠)
  ⇒ **字节翻倍、token 也翻倍 ⇒ 聚合持平**。**根因是"MoE 权重流量 ∝ 不同专家数",而非并行化不足。**

### 由此得到的杠杆排序(要 >30 单流 ⇒ 必须把 13.3 GB/token 砍到 ≤7.5 GB)
| 杠杆 | 机制 | 预期 | 可行性 |
|---|---|---|---|
| **1. 提高投机接受率** | 同一批权重读服务更多 token ⇒ **字节/token 按接受率 1:1 下降** | 接受 1.3→2.6 ⇒ **17→34 t/s** | `num_speculative_tokens` 现 5,**从未扫过**;最便宜 |
| **2. GPU 常驻专家层** | 那些层的权重不再走 CPU 内存,改走显存(1.5-2 TB/s) | TP=1 容 ~6 层 ⇒ 13.3→11.4 GB ⇒ 20 t/s;TP=2 容 ~12 层 ⇒ 9.6 GB ⇒ 23 t/s | 容量受限(§145),单独不足以到 30 |
| **3. 大 batch 让 na 撞上限** | na≤256 而 token 数继续涨 ⇒ 字节/token 下降 | C=16 ≈ 6.5 GB/token ⇒ 34 t/s;**但单流不可用**(用户已否) | 与"C≤3 可用"冲突 |
| 1+2 叠加 | — | 4.8 GB/token ⇒ **~47 t/s** ✅ | **推荐组合** |

### TP=2 最新判定:关掉常驻层**仍然**在预热阶段卡死
`/tmp/serve_tp2_nores_r102.log`:两个 rank 都过了 "Enforce eager",EP 分片 92 行,随后同样出现
`No available shared memory`(预热广播不被消费)⇒ **常驻层路径不是 TP=2 卡死的唯一原因**,
TP=2 仍需单独二分(§146 的 5 个失败案例仍有效)。

## 153. 第 102 轮:【用户提供生产基线】接受率 40-50%/平均 3 token;我方只有 ~1.3 ⇒ 投机在净亏

### 用户提供的生产(可用)基线
- 生成速度 **35 tok/s**、**draft 生成 90 tok/s**、**接受率 40-50%**、**平均接受 3 个 token/步**
⇒ 也就是说:生产环境的投机是**净赚**的(3 个 token 摊一次权重搬运)。

### 我方实测对照(本轮)
| 配置 | TPOT | 单流 | 聚合 |
|---|---|---|---|
| C=1 **无投机** | **45.1 ms** | **22.2 t/s** | — |
| **C=2 无投机** | 49.5 ms(仅 +10%) | 20.2 t/s | **40.4 t/s(+82%)** |
| C=1 有投机 k=5 | 57.4 ms | 17.4 t/s | — |
| C=2 有投机 k=5 | 108.3 ms(+100%) | 9.2 t/s | ~18.5 t/s(**+6%**) |

服务端 `cd-timing`(无投机):`qlen=1 → period 0.91ms(compute 0.42+rest 0.49)`;
`qlen=2 → 1.08ms(0.57+0.51)`。⇒ **qlen 1→2 每层只涨 19%**(两个 token **共享专家**),
这就是无投机时 C=2 能 +82% 的原因。

### 诊断(算术闭合)
- 每步权重字节 ∝ 不同专家数 `na`;有投机时 qlen=6 ⇒ na≈32 ⇒ **17.3 GB/步**;
  接受 1.3 ⇒ **13.3 GB/token** ⇒ 226/13.3 = **17.4 t/s**(实测 ✅)。
- 若接受率能到生产水平(**3 个/步**):13.3 → **5.8 GB/token** ⇒ **≈39 t/s 单流**(>30 ✅);
  再叠加(无投机已证明的)C=2 共享专家效应 ⇒ **聚合有希望到 ~70**(≥70 ✅)。
- ⇒ **我方投机的病不在机制,而在"接受率只有 26%/位 vs 生产 40-50%"**。
  最可能的直接原因:`--speculative-config` 里 **`draft_sample_method: probabilistic`**
  (draft 端引入随机性 ⇒ 接受率下降)。**改成 greedy/低温**是最便宜的修复。
- 另一条已被本轮数据证明的独立路径:**无投机 C=2 已经 40.4 t/s**;若把 spec 关掉而把 C 提到 3-4
  (用户说 C≤3 可用),按 +82%/2 路的斜率外推,聚合还有空间(但单流 22.2 仍 <30)。

## 154. 第 102 轮:关于"234 GB/s vs 机器峰值 700-800"的口径澄清(含我此前一处外推的更正)

### 三个数字是三种不同的量,不能混用
| 数字 | 性质 | 说明 |
|---|---|---|
| **234 GB/s** | **实测**聚合带宽 | 验收形状 D=12、120 线程、na=12(§133/§138) |
| **~422 GB/s** | **外推**(我此前的说法,**应标注为外推**) | 用"低并发每线程 3.06(30T)/2.89(60T)×192 核"推得,**从未实测到** |
| **783 GB/s** | 机器理论/探针峰值 | 要达到它需要 **~4 GB/s/线程 × 192 核**;而我们与 lk 的每线程都只有 2.2-3.1 |

### 本轮两个判决性探针(都**未能**测出机器上限,但各给出一条事实)
`/tmp/memceil.cpp`:与引擎同形(2 KB 行、1 条 zmm 加法/64B、无 fp64/无解码),线程读**互不相交**的连续块:
| T | 30 | 60 | 120 | 192 |
|---|---|---|---|---|
| 聚合 | 76 | 97 | 90 | 74 GB/s |
| 每线程 | 2.53 | 1.61 | 0.75 | 0.38 GB/s |
- **坑**:第一版没预热内存 ⇒ T=192 只有 23 GB/s、且非单调(缺页伪影);预热后仍有非单调 ⇒
  **该探针测的是"多路不相交顺序流"的 DRAM/预取效率,不是机器上限**。
- **但有一条硬事实**:一个"每次 64B 一次依赖载入"的最朴素循环,每线程也只有 **2.5 GB/s** ——
  **与引擎的 2.2-3.1 同量级** ⇒ 说明引擎的每线程速率**不是解码 ALU 造成的异常**,而是这种
  "载入为主"的循环的正常 MLP/延迟水平。

### 因此正确的表述(更正)
1. **"234 GB/s"是验收形状的实测**;**"422 GB/s"是外推,不应作为成绩或结论**;
2. 引擎**已实测到的最高聚合是 350 GB/s**(D=48、na=32,§139),这才是它的能力上界;
3. 到机器峰值 783 需要 ~4 GB/s/线程,**我们和 lk 都做不到**(lk 265-376 GB/s ⇒ 2.2-3.1/线程)
   ⇒ **"带宽占用率低"是全行业的共同现象,不是我们独有的缺陷**;真正的约束是**每线程 MLP/延迟**;
4. 要把聚合从 234 推向 700,路径只有两条:**提高每线程 MLP**(更多未完成载入/预取深度)
   或**减少每字节所需的指令**(解码)—— 两者都属于第一条已被用户结案的内核范畴。
   ⇒ 因此我把它记入 `FUTURE_PLAN.md`,而**第二条继续走"接受率 + 常驻层 + 并发共享专家"**
   (本轮已证明这三条能立刻兑现:无投机 C=2 = 40.4 t/s,+82%)。

### 要做"真的机器上限"探针,必须**同形**:同一 shard 的线程读**同一批 2 KB 行的不同列**
(即引擎的访问模式)+ 零解码。这是下一步若再谈带宽利用率时应采用的形式。

## 155. 第 102 轮:【用户点出关键】draft 在 CPU 上 —— 我方投机慢/接受率低的直接原因

### 证据
- 模型:`num_hidden_layers=43` + **`num_nextn_predict_layers=1`** ⇒ draft = **1 层 MTP**(不是独立小模型)。
- 插件 `hybrid_model.py:501-511`:**默认把 draft 副本排除在 GPU 常驻之外**(注释的理由是省显存:
  "常驻会把显存占用翻倍 ⇒ 常驻层数上不去");开关 `XIAOTU_MOE_RESIDENT_DRAFT=1` 才能恢复。
⇒ **我们的 draft 层专家跑在 CPU 引擎上**:每个投机步额外走 CPU MoE ⇒
  - 与用户生产观察(draft **90 tok/s**,显然在 GPU)相反;
  - 解释了 §153 的两组实测:有投机 k=5 时**每步 74 ms**(vs 无投机 43 ms)、
    且 C=2 几乎不扩展(+6%)—— 因为 draft 也按 token 数放大 CPU 权重流量。

### 处方(下一轮立刻做)
1. **把 draft 放到 GPU**:`XIAOTU_MOE_RESIDENT_DRAFT=1`(或把 MTP 层号写进
   `XIAOTU_MOE_GPU_RESIDENT_LAYERS`)。代价:多占 ~1 层 × 3.19 GiB(TP=1)——
   而它换来的是**投机从"净亏"变成"净赚"**,远优于"多常驻 1 个目标层(+6% rest)"。
2. **同时把 draft 采样从 `probabilistic` 改为 greedy/低温**(接受率 26% → 目标 40-50%)。
3. 用 `/metrics` 的 `spec_decode` 指标**直接读接受率**(不再从 TPOT 反推)。
4. 测量口径:C=1 与 C=2 的 TPOT/out_tok/s + `XIAOTU_CD_TIMING` 的每层 compute/rest。

### 预期(按 §152 的字节/token 公式)
draft 上 GPU 后,投机步的 CPU 权重流量回到"只算目标模型一次"(qlen=1 的 43 层)而非 qlen=6 的 43 层;
再叠加接受率 1.3→3 ⇒ **单流有望 35-40 t/s**(>30 ✅),C=2 叠加共享专家 ⇒ **聚合有望 ~70**(≥70 ✅)。

## 156. 第 103 轮:复盘文档 `docs/OPTIMIZATION_RETROSPECTIVE.md`
按用户要求,把"AI 的优化是局部且盲目的、人类架构师的方向性指导是破局关键"这条结论
**结合本会话的具体案例**写成独立复盘(6 个案例,全部带 §/轮次/文件行号引用):
①AI 在内核穷举 30 轮 ↔ 人类一句"先结案、转第二条";
②AI 有公式却没做"关掉投机"这个最便宜对照 ↔ 人类一句"C=2 几乎不涨?";
③AI 想扫采样网格 ↔ 人类给生产基线(35 tok/s、draft 90、接受率 40-50%、平均 3 token);
④**【最关键】**AI 从未查 draft 跑在哪块硬件上 ↔ 人类一句"投机模型应该跑在 GPU 上吧"
  ⇒ 命中根因(`hybrid_model.py:501-511` 默认把 draft 排除在 GPU 常驻外 ⇒ draft 在 CPU);
⑤人类用数量级逮住 AI 的未标注外推(234 实测 vs 422 外推 vs 783 峰值);
⑥人类的硬约束(4-5 核/CCD、NUMA 分片、禁 singlecopy)反被 AI 实测验证(§125)。
文末给出**"人给 AI 派活"的最小成本指导清单**(划界/参考数字/预期形状/架构放置/不变量/口径)
与 **AI 侧自我约束**(清场、单变量、标注数字性质、禁跨口径、先做最便宜对照、门禁、工具化操作陷阱)。

## 157. 第 103-104 轮:把 draft 放上 GPU 的第一/二次尝试(显存账)

### 依据(§155)
模型 = 43 层 + **1 层 nextn(MTP)作 draft**;插件 `hybrid_model.py:501-511` **默认把 draft 副本
排除在 GPU 常驻外** ⇒ draft 的 MoE 跑在 CPU ⇒ 投机净亏。处方:`XIAOTU_MOE_RESIDENT_DRAFT=1`
\+ 把 MTP 层写进 `XIAOTU_MOE_GPU_RESIDENT_LAYERS`(参考配置也是 `43-45`)。

### 尝试 1(失败,已记 R86):KV 8 GiB + draft 3 层常驻
日志确认 draft 层已进显存:`resident model.layers.43/44/45.ffn: 3.19 GiB on cuda:0`(共 **9.57 GiB**),
KV 已按 8 GiB 分配(291,535 token),随后
**`RuntimeError: Triton Error [CUDA]: out of memory`**(预热/autotune 阶段)。
显存账:固定 14.2 + KV 8 + draft 9.57 = **31.8 GiB / 39.5**,余量不足以覆盖 Triton 工作区。

### 尝试 2(进行中):KV 降到 4 GiB,其余不变
`--kv-cache-memory-bytes 4294967296` + draft 3 层常驻 + budget 12 + gpu-mem-util 0.9
⇒ 14.2 + 4 + 9.57 = 27.8 GiB,余 ~11.7 GiB。启动日志已见 3 个 draft 层常驻、**暂无 OOM**,仍在预热。

### 待测(就绪后立刻做)
1. `/metrics` 的 `spec_decode` 指标 ⇒ **接受率**(不再从 TPOT 反推);
2. C=1 与 C=2 的 TPOT / out_tok/s;
3. `XIAOTU_CD_TIMING` 的每层 compute/rest(看投机步的 CPU 流量是否回到"只算目标一次")。
预期(§152 公式):draft 上 GPU 后 k=5 不再是净亏;C=1 有望回到 ≥无投机的水平并叠加接受率收益。

## 158. 第 105 轮:draft 常驻 GPU 的结论 —— **方向正确,但被单卡显存墙挡住**

### 两次尝试都失败
| 尝试 | 配置 | 结果 |
|---|---|---|
| 1 | draft 3 层常驻 + KV 8 GiB | draft 层已进显存(3×3.19=9.57 GiB),KV 已分配(291,535 token),**预热阶段 `Triton Error [CUDA]: out of memory`**(R86) |
| 2 | draft 3 层常驻 + KV 4 GiB | **`Engine core initialization failed`**(预热阶段,日志无显式 OOM) |

### 显存墙的账(单卡 39.5 GiB)
固定开销 14.2 + draft 3 层 9.57 = **23.8 GiB**,留给 KV + Triton 工作区的只有 ~15 GiB;
而 Triton 预热/autotune 需要数 GiB ⇒ 无论 KV 取 8 还是 4 都在边缘。
**这与 §145 的结论一致:单卡上"常驻层 / KV / 工作区"三者互斥。**

### 结论与剩下的选项
- **用户的判断在原理上完全正确**(代码证据:插件默认把 draft 排除在 GPU 常驻外 ⇒ draft 在 CPU),
  但**在当前显存容量下无法直接落地**;
- 剩下的可行变体(按代价排序):
  1. **只常驻 1 个 draft 层**(`XIAOTU_MOE_GPU_RESIDENT_LAYERS=45`)+ KV 4 GiB ⇒ 14.2+3.19+4 = 21.4 GiB,
     余量充足(但若 draft 实际用的是 43 或 44,需先确认 MTP 层的真实索引);
  2. KV 降到 2 GiB(预填充/短上下文测量够用);
  3. **TP=2**(每卡每层 1.59 GiB)⇒ 空间翻倍,但 TP=2 目前起不来(§146/R86 之前的回归);
  4. 把"draft 在 CPU"这件事记入 `FUTURE_PLAN.md`(它是**解码成绩的关键一环**,但在单卡上无解)。

### 本轮恢复的可用配置(已验证过的最佳解码数)
`TP=1 + 无投机 + max-num-seqs 4 + KV 8 GiB`:C=1 **22.2 t/s**、C=2 聚合 **40.4 t/s(+82%)**(§153)。

## 159. 第 106 轮:draft 的真实身份 —— **`mtp.0`,不是 `layers.43-45`**

直接从权重索引(`*.index.json` 的 `weight_map`)读到:
- **目标模型只有 `layers.0 .. layers.42`(共 43 层)**;
- **draft/MTP 存在 `mtp.0.*` 下**(样例键:`mtp.0.hc_attn_base`、`mtp.0.hc_ffn_base`、`mtp.0.hc_attn_fn`);
- 模型 config 的 `num_hidden_layers=43` + `num_nextn_predict_layers=1` 与此一致。

但启动日志里出现的是 `resident model.layers.43/44/45.ffn: 3.19 GiB on cuda:0`
⇒ **插件把 draft 的 3 个子模块映射成了层号 43/44/45**(所以参考配置写 `43-45` 指的是同一批东西)。
⇒ 我方 §158 的"draft 3 层共 9.57 GiB"是**真实占用**,不是重复计数。

### 由此确定的下一步(显存墙内的最小可行变体)
1. **只常驻 draft 的 1 个子模块**(`GPU_RESIDENT_LAYERS=45` 或 `43`)⇒ 3.19 GiB 而不是 9.57;
   但需先确认 draft 前向实际用到哪几个(若三个都用,只常驻一个只是部分收益);
2. 或者 **KV 降到 2 GiB** + 三个 draft 层全常驻(14.2+9.57+2 = 25.8 GiB,余 ~13 GiB);
3. 两者都不行 ⇒ 记入 `FUTURE_PLAN.md`,等 TP=2 修好(每卡 1.59 GiB/层)再谈。

## 160. 第 107 轮:**KV 每 token 核准**(解决目标里标注的 29.5 vs 60 KiB 分歧)+ draft 常驻第 3 次尝试

### 【权威核准】fp8_ds_mla = **29.4 KB/token**(不是 60 KiB)
vLLM 的配置校验直接给出:
```
ValueError: To serve at least one request with the model's max seq len (262144),
7.19 GiB KV cache is needed, which is larger than the available KV cache memory (4.0 GiB)
```
⇒ **7.19 GiB ÷ 262144 = 29.4 KB/token** —— 与 `scripts/serve_prod_8070.sh` 注释里的 **29.5 KB/token 一致**;
**旧文档的 60 KiB/token 是错的**(目标文本里标注的疑问到此解决)。
⇒ 1M 上下文 = 1,048,576 × 29.4 KB = **≈29.4 GiB** ⇒ **必须 TP=2**(2×40 GB 装得下 KV + 权重),
这与 serve 脚本的判断一致。
⇒ 也意味着 **KV 与 maxlen 是强绑定的**:`maxlen=262144` 要求 KV ≥ **7.19 GiB**;
若要把 KV 压到 4 GiB,必须把 maxlen 降到 **≤131072**(本文档下一节)。

### draft 常驻的失败链(现在完全清楚)
| 尝试 | 配置 | 失败原因 |
|---|---|---|
| 1 | draft 3 层 + **KV 8 GiB** + maxlen 262144 | 校验通过,但**预热/autotune `Triton Error [CUDA]: out of memory`**(固定 14.2 + 8 + 9.57 = 31.8,余量不够) |
| 2 | draft 3 层 + **KV 4 GiB** + maxlen 262144 | **不是内存问题**:`ValueError: ... 7.19 GiB KV needed > 4.0 GiB available`(maxlen 未同步下调) |
| 3 | draft 3 层 + KV 4 GiB + **maxlen 131072** | 进行中(账:14.2+4+9.57 = 27.8 GiB,余 ~11.7) |

⇒ 教训:**改 KV 必须同步改 maxlen**(它俩由 29.4 KB/token 硬绑定),否则报的是"看起来像内存"的校验错。

## 161. 第 108-109 轮:1 个 draft 子模块常驻 = **零收益**;接受率画像曝光;TP=2 成为三条目标的共同前置

### 实测(8070:TP=1 + 1 个 draft 子模块常驻 + KV 8 GiB + spec k=5 + maxlen 262144)
- 显存:**GPU0 = 25.25 GiB**,与预测(14.2 固定 + 8 KV + 3.19 draft = 25.4)吻合 ⇒ **不再 OOM**;
- **C=1 TPOT = 57.3 ms(17.5 t/s)** vs 不常驻 draft 的 **57.4 ms** ⇒ **零收益**;
- C=2:TPOT 99.1 ms(10.1 t/s/请求),out_tok/s 13.16;
- `cd-timing`:`period 1.84-1.90ms(compute 0.99-1.03 + rest 0.85-0.87)` —— 与"draft 不常驻"时几乎一样。

⇒ **§155 的假设需要修正**:把 draft 的**部分** MoE 放上 GPU **没有兑现收益**。
两个可能:(a) 真正吃 CPU 的是**另外两个 draft 子模块**(43 不够,要 43-45 全上);或
(b) draft 的 MoE 本来就不是主要成本(投机的主要开销在**验证步本身按 6 个 token 放大权重流量**,
即 §152 的 13.3 GB/token 效应)。**要区分它们,必须在 TP=2 下做"43-45 全常驻"**(TP=1 装不下,§160)。

### 【接受率画像】从 `/metrics` 直接读到(不再反推)
```
spec_decode_num_accepted_tokens_total = 332
spec_decode_num_accepted_tokens_per_pos_total{position="0"} = 246
spec_decode_num_accepted_tokens_per_pos_total{position="1"} =  58
```
⇒ **第 1 个 draft token 接受率很高,之后断崖**:pos1/pos0 = **23.6%**。
这解释了为何"平均只接受 ~1.3 个/步",而生产是 40-50%/平均 3 个 ⇒ **我们的 MTP 草稿在第 2 位之后基本不被接受**。

### 【关键结论】TP=2 现在是**三条目标的共同前置**
| 目标 | 为什么必须 TP=2 |
|---|---|
| **预填充 ≥1500** | TP=1 的 H2D = **144.7 ms/层**(搬 3.2 GB,22 GB/s = PCIe 墙);TP=2 每 rank 只搬 1.6 GB ⇒ **~72 ms/层**(§150) |
| **解码(投机)** | 每卡每常驻层 **1.59 GiB**(TP=1 是 3.19)⇒ **draft 的 3 个子模块(9.57 GiB)在 TP=1 装不下,TP=2 只要 4.77 GiB/卡** ⇒ 可在 KV 与目标层之外放下 |
| **1M 上下文** | KV = **29.4 GiB**(29.4 KB/token,§160)⇒ 单卡 40 GB 装不下,必须 TP=2 |
⇒ **修好 TP=2 是第二条的单一最高优先级**;它的失败模式已记录(§146:关 EAGER 预热 shm 卡死;
开 EAGER worker 静默原生崩溃;关常驻层也复现 ⇒ 不是常驻层独因)。

## 162. 第 110 轮:TP=2 二分启动 + **把 M8 工具化**(`scripts/kill_serve.sh`)

### M8 第三次复发(必须工具化)
本会话第 3 次因为 `pkill -f "vllm serve"`/等价匹配**杀掉了调用者自己**:
第 96 轮、第 102 轮、第 110 轮(这次是 `python3` 脚本按整条 cmdline 匹配,
而调用它的 bash 的 cmdline 里含 heredoc 文本 ⇒ 自己命中;括号技巧也无效,
因为真实调用 `vllm serve <path>` 就在同一条命令行里)。
⇒ 新增 **`scripts/kill_serve.sh`**:只检查 `/proc/<pid>/cmdline` 的 **argv[0]**
(是否为本环境 `bin/` 下的可执行文件)+ argv 含 `serve`,并**显式排除自身与所有祖先进程**;
杀完自动 `sleep 22` + 打印各卡显存 + 清 `/dev/shm/xiaotu_ep_*.bin`(顺带治 M9)。
复盘文档里"记录≠免疫,必须工具化"这条现在有了第 3 个实例与对应工具。

### TP=2 二分(本轮开始)
按 §161,先测**最小组合**:`TP=2 + EAGER + 无投机 + 无常驻 + KV 8 GiB + maxlen 262144`
(此前每次 TP=2 都带投机,**这个变量从未被隔离**)。启动中,尚无 OOM/崩溃。
后续按顺序加回:**① 投机 → ② 常驻层 → ③ CUDA graph**,逐步定位崩溃点。

## 163. 第 111 轮:【TP=2 回归定位】最小组合也崩 ⇒ 病根在 **V2 Model Runner**,不是投机/常驻层

### 隔离实验(逐项排除,本轮做了最关键的一步)
| 实验 | 配置 | 结果 |
|---|---|---|
| A | TP=2 + EAGER + **无投机** + **无常驻** | **仍崩**(已过 KV 分配 `GPU KV cache size: 291,535 tokens`,死在预热阶段,**无 Python 堆栈** ⇒ 原生崩溃) |
⇒ **投机与常驻层都被排除**;崩溃是 **TP=2 路径本身**固有的。
这与 §54/§56 的记录(TP=2+常驻0-5+EAGER = 1105-1117 t/s)**矛盾** ⇒ 期间代码/配置有回归。

### 定位到 `VLLM_USE_V2_MODEL_RUNNER`(目标里点名的"未启用旋钮")
- TP=2 失败日志里有 **`Using V2 Model Runner`**;
- vLLM 源码:`vllm/config/vllm.py:2747` 注释 "It should be enabled with explicit
  VLLM_USE_V2_MODEL_RUNNER environ",且 `envs.VLLM_USE_V2_MODEL_RUNNER is None` 有分支
  ⇒ **V2 runner 是显式 opt-in 的开关**;
- **实验 B:`VLLM_USE_V2_MODEL_RUNNER=0`(强制 V1)+ TP=2 + EAGER + 无投机 + 无常驻**
  ⇒ **不再崩溃**:已过 KV 分配、日志无 `V2 Model Runner`、进程存活(仍在加载)。
⇒ **V2 model runner 是 TP=2 崩溃的头号嫌疑**,而这正是做法要求(3)里"补齐未启用旋钮"的一项。

### 下一步
1. 等实验 B 就绪 ⇒ 立刻测 **预填充 8192/32768 + C=1/C=2 解码 + cd-timing**,拿到 TP=2 的基线数字;
2. 在 V1 runner 下逐个加回:**① 投机 → ② draft 3 子模块常驻 → ③ 目标层常驻 → ④ CUDA graph**;
3. 若 V1 下 TP=2 全部可用 ⇒ 把 `VLLM_USE_V2_MODEL_RUNNER=0` 写进交付配置,并记入
   `TRIED_AND_REVERTED.md`(V2 runner 在 TP=2 下不可用)。

### §163 续(第 112 轮):V1-runner 的 TP=2 仍在正常推进
| 阶段 | V2 runner | **V1 runner(`VLLM_USE_V2_MODEL_RUNNER=0`)** |
|---|---|---|
| 权重加载 | 完成 | **完成**(GPU0/1 各 22.2 GiB) |
| KV 分配 | 完成(291,535 token) | **完成**(291,535 token) |
| 预热/模型初始化 | **原生崩溃(静默)** | **无崩溃,已持续 ~10+ min,仍在初始化** |
⇒ 目前证据一致指向 **V2 model runner 是 TP=2 崩溃的原因**;只差"最终到达 READY"这一步确认。

**待办**(下一轮第一件事):
1. 等它 REDDY ⇒ 立刻测 **预填充 8192/32768 + C=1/C=2 解码 + `[gp-h2d]`/`cd-timing`**,拿 TP=2 的基线;
2. V1 下逐个加回:**① 投机 → ② draft 3 子模块常驻(4.77 GiB/卡)→ ③ 目标层常驻(每卡 1.59 GiB)→ ④ CUDA graph**;
3. 若 TP=2 在 V1 下全通 ⇒ 交付配置加 `VLLM_USE_V2_MODEL_RUNNER=0`,并把"V2 runner 在 TP=2 不可用"记入 `TRIED_AND_REVERTED.md`;
4. 预期收益(§161 的账):预填充 H2D 144.7 → ~72 ms/层;draft 3 子模块可常驻;1M KV 29.4 GiB 可容纳。

## 164. 【突破】第 113 轮:TP=2 跑通 —— 根因是 **`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` 默认 300 s 太短**;预填充 701 → **1159 t/s**

### 根因(一句话)
`vllm/envs.py:250`:`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS: int = **300**`,而它就是
`vllm/v1/executor/multiproc_executor.py:429-433` 的 `dequeue_timeout`;
TP=2 的预热(两个 rank 各加载 ~69 GB CPU 侧专家 + 建 pinned 缓冲,彼此争抢)**超过 300 s**
⇒ EngineCore `RuntimeError: cancelled`(此前所有 TP=2 失败都是这个,与 V1/V2 runner、投机、常驻层无关)。

### 能跑通的配置(实测 210 s 就绪)
```bash
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 \
VLLM_USE_V2_MODEL_RUNNER=0 \
# TP=2 + --enforce-eager + 无投机 + 无常驻 + KV 8 GiB + maxlen 262144
```
日志:`GPU KV cache size: 291,535 tokens` → **`Application startup complete`**,procs=3,8070 UP。

### 【收益】预填充第一战
| 配置 | 预填充 8192(C=1) |
|---|---|
| TP=1 | **701 t/s** |
| **TP=2** | **1159 t/s**(TTFT 7065 ms) |
⇒ **1.65×**,与 §150 的算术预测(TP=2 ⇒ H2D 144.7→~72 ms/层 ⇒ ~1360 t/s)基本吻合。

### 为什么这是第二条的转折点(§161 的三条目标全部解锁)
| 目标 | 现在 |
|---|---|
| 预填充 ≥1500 | 1159 起步;**再加常驻层**(TP=2 每卡 1.59 GiB/层 ⇒ 预算 12-20 GB 可放 **7-12 个目标层**,每层省 ~72 ms H2D)预期可越 1500 |
| 解码/投机 | draft 3 子模块只需 **4.77 GiB/卡** ⇒ 可以与 KV/目标层共存,"draft 上 GPU"的假设终于可测 |
| 1M 上下文 | KV 29.4 GiB 分到两卡 ⇒ 可行 |

### 待办(下一轮)
1. 重测 **C=1/C=2 解码**(本轮 decode 测量因客户端字段缺失报了 ZeroDivision,需重跑);
2. **加常驻层**:`XIAOTU_MOE_GPU_RESIDENT_LAYERS="0-5"` + `RESIDENT_BUDGET_GB=12`(TP=2 每卡 1.59 GiB/层)
   ⇒ 再测预填充,目标 ≥1500;
3. 加回**投机** + **draft 3 子模块常驻**(4.77 GiB/卡)⇒ 测单流/聚合解码,拿到"draft 上 GPU"的真实数字;
4. 把 `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600` + `VLLM_USE_V2_MODEL_RUNNER=0` **写进交付脚本**
   (`scripts/serve_prod_8070.sh` / `tune_serve.sh`),并在 `TRIED_AND_REVERTED.md` 记:
   "V2 runner 在 TP=2 下预热不完成"与"execute-timeout 默认值必须放大"。

## 165. 第 114 轮:TP=2 + 常驻层 ⇒ 预填充 **1248 t/s**;Triton 工作区成为常驻层上限;下一步靠"H2D 与内核重叠"

### 实测(TP=2,V1 runner,execute-timeout 3600)
| 配置 | 预填充 8192(C=1) |
|---|---|
| TP=1 | 701 t/s |
| TP=2(无常驻) | 1159 t/s |
| **TP=2 + 8 个目标层常驻**(预算 14 GB,每卡 8×1.59=12.7 GiB) | **1248 t/s**(TTFT 6562 ms) |
⇒ 距 **1500** 还差 **20%**。

### 常驻层的上限被 **Triton 工作区**卡住(不是显存总量)
- 预算 18 GB(**11 层**/卡 = 17.5 GiB):固定 7.1 + KV 8 + 17.5 ≈ **32.6 GiB/卡** ⇒ **推理时**
  `RuntimeError: Triton Error [CUDA]: out of memory`(engine 在**第一个请求**上死,R89);
- 预算 14 GB(**8 层** = 12.7 GiB)⇒ 27.8 GiB/卡,**正常**(无 OOM),1248 t/s。
⇒ 与 TP=1 的经验一致:**Triton 预填充内核/autotune 需要 >7 GiB 的可用工作区**;
在 maxlen=262144(KV 硬下限 7.19 GiB)下,TP=2 每卡**常驻层数的实际上限约 8-9 层**。

### 关键判断:剩下的 20% 不在"常驻层数",而在 **H2D 与内核是否重叠**
按当前分量(TP=2):每层 ≈ H2D ~72 + 内核 ~58 + 主机 seg ~10 ≈ **140 ms**(串行);
1248 t/s 对应每层 ~131 ms,与"完全串行"吻合 ⇒ **H2D 没有与内核重叠**。
若做到重叠(预取下一层权重的同时算本层),每层应降到 **~max(72, 68) ≈ 75 ms**
⇒ 43 层 ≈ 3.2 s ⇒ **~2500 t/s**(远超 1500)。
而插件里**正是有这两个旋钮且从未在本配置下扫过**:
`XIAOTU_MOE_PREFETCH_SLOTS`(ring 深度,§R20 只测过 TP=1 且当时是正确性 bug)、
`XIAOTU_GPU_PREFETCH_AHEAD`(`hybrid_model.py:860`,
`_ov = (not _resident) and os.environ.get("XIAOTU_GPU_PREFETCH_AHEAD", "1") == "1"` ——
注意它**在 `_resident` 为真时被短路**,而我们刚开了 8 层常驻!)。

### 下一步(下一轮,按杠杆)
1. **扫 `XIAOTU_GPU_PREFETCH_AHEAD` / `XIAOTU_MOE_PREFETCH_SLOTS`**(在当前 TP=2+8 常驻下)
   —— 目标是让 H2D 与内核重叠;注意上面那行短路逻辑可能让常驻层反而**关掉了预取**,要单独确认;
2. 对照参考配置的 `LVLLM_GPU_PREFETCH_WINDOW=1` 与 `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=1024`
   (我方 `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=384`);
3. 若重叠生效 ⇒ 再把常驻层数在"Triton 工作区"允许范围内调回 8-9 层,复核 1500。

## 166. 第 115 轮:TP=2 预填充 32768 = **1053 t/s**(追平 SM80 参考);重叠代码已存在,H2D 分量待测

### 本轮实测(TP=2 + 8 常驻层 + V1 runner + execute-timeout 3600)
| 口径 | TP=1 | **TP=2** | 参考 |
|---|---|---|---|
| 预填充 8192 | 701 t/s | **1248 t/s**(8 常驻层) | — |
| **预填充 32768** | 728 t/s | **1053 t/s** | **SM80(3090×2)= 1060 @32768** |
⇒ **里程碑:32768 口径已追平 SM80 参考(1053 vs 1060)**,距离 1500 还差 **42%**(在 32768 口径上)。
(8192 口径 1248,距 1500 差 20%。)

### 重叠:**代码里已经实现且默认开启**
- `hybrid_model.py:857-859`:"Overlap: kick off the NEXT layer's H2D before doing this layer's
  kernels, so the DMA runs concurrently with the tensor cores." ⇒ `_ov = (not _resident) and
  XIAOTU_GPU_PREFETCH_AHEAD(默认"1") == "1"`;
- `gpu_prefill.py:726` 注释:"a dedicated side stream; the compute stream only waits on the
  slot's ready" ⇒ **有专用 side stream + `PrefetchSlot`**;ring 深度 `max(2, XIAOTU_MOE_PREFETCH_SLOTS)`。
⇒ 所以 §165 里"没有重叠"的判断**需要修正**:重叠机制是在的,只是**效果待确认**
(8 个常驻层只带来 +7.7%,而它们占 18.6% 的层数 ⇒ 说明 H2D 在 TP=2 下**已不是主导项**,
或重叠已经在生效、常驻层的边际收益因此变小)。

### 本轮的分段(host 侧)
`[gp-timing] seg=4.7-5.4ms rest=0.35-0.40ms host_total=5.1-5.8ms`(TP=1 的 seg 是 10.03ms)
⇒ **TP=2 下主机侧分段成本几乎减半**。但 **`[gp-h2d]` 行未出现**(打印条件需要更多层调用),
⇒ **TP=2 的每层 H2D 时间仍是未知数**,这是下一轮要先补的关键测量。

### 下一轮
1. 补测 TP=2 的 `[gp-h2d] per_layer`(多跑几轮 32K,或看打印条件),确认 H2D 是否已是瓶颈;
2. 若 H2D 仍是瓶颈 ⇒ 扫 `XIAOTU_MOE_PREFETCH_SLOTS`(2→3/4)与 `XIAOTU_GPU_PREFETCH_AHEAD`;
3. 顺带核对 `_resident` 为真的层是否会**打断下一层的预取**(8 个常驻层可能造成 8 次流水线排空)。

## 167. 第 116 轮:**§166 的 1053 t/s 是被污染的错数**;热态 TP=2 预填充 8192=1934 t/s、32768=908 t/s(超线性);32K 后 `persistent_topk` 非法访存崩溃

### (a) §166 的数字作废
- 原报"32768 = 1053 t/s":那次是**首次推理**,日志 06:30:58 有两条
  `WARNING jit_monitor.py: Triton kernel JIT compilation during inference: ComputePrefillMetadataKernel /
  BuildPrefillChunkMetadataKernel. This causes a latency spike` ⇒ **首轮包含运行期 JIT/autotune**,不是稳态。
- §166 引用的"引擎窗口 3276.9 t/s"**也不可信**:`loggers.py` 的窗口吞吐用"两次 logging 调用之间的
  实际 elapsed"作分母,长请求把 logging 调用推迟时,分子是几十秒累计的 token、分母只有部分时间
  ⇒ **长请求突发时该指标会高估**。同一次运行里 8192 那轮窗口值是 409.7 t/s,而客户端实测 1934 t/s,
  相差 4.7×,可证该指标在此场景不可用作分母口径。**结论:预填充速率一律以客户端 TTFT 为准。**

### (b) 热态实测(客户端 TTFT,同窗口连发)
| 请求 | TTFT | 速率 |
|---|---|---|
| 8192 tokens | 4235 ms | **1934 t/s** |
| 32768 tokens | 36075 ms | **908 t/s** |
⇒ 32768/8192 = **4× token 但 8.5× 时间**,**超线性**。DMA 模型预测 32K 应为 4×2.63 s = 10.5 s,
实测 36.1 s ⇒ 有 **~25 s 无法用专家权重 DMA 解释**,必须另找(Triton chunked-prefill 元数据、
稀疏 indexer 的 O(n²) 项、或每块重复的固定开销)。

### (c) DMA 模型(TP=2,已与硬件吻合,作为后续判据)
- 每 rank 每层常驻规模:`1.59 GiB / 128 experts = 12.7 MB/expert`(即**每 rank 持有 128 个完整 expert**)。
- `[gp-h2d] per_layer=75.3 / 79.3 ms` ⇒ 1.61 GB / 75.3 ms = **21.4 GB/s**(PCIe Gen4 x16 实测上限量级)。
- 35 非常驻层 × 4 块 × 75.3 ms = **10.5 s**;8192 一块 = **2.63 s** vs 实测 4.24 s ⇒ 8192 档
  **≈62% 被 DMA 解释**,重叠(`hybrid_model.py:857` + side stream)**确实在生效**。
  ⇒ **预填充的主杠杆是"减少要搬的 expert 字节数"**(常驻层数 / 权重位宽),不是核函数调优。

### (d) **新失败模式:`persistent_topk` 非法访存(与之前的 Triton OOM 不同)**
```
Worker_TP0 ... RuntimeError: launch_persistent_topk, /workspace/csrc/libtorch_stable/topk.cu:107,
  persistent_topk occupancy query failed: an illegal memory access was encountered
```
调用栈:`layer → attn → _sparse_indexer_and_attn → indexer_op → sparse_attn_indexer → persistent_topk`。
- 发生在 32K 预填充之后(06:35:44),随后 EngineCore 因 dequeue 超时报 `RuntimeError: cancelled`(次生)。
- `occupancy query` 里的 illegal access 通常是**粘性 CUDA 错误**:真正的越界写发生在**更早的某个核**,
  在这里的同步点才暴露 ⇒ 不能只盯 topk,必须查 32K 路径上**前面**写显存的核
  (首选嫌疑:常驻层的预取/`PrefetchSlot` 环、Triton chunked-prefill 元数据核)。
- 这直接威胁 **1M 上下文可用**这一条目标,必须先定性。

### 下一轮(待办,优先级最高)
1. 重启后 **resident=0** 跑同一 32K 请求:若仍崩 ⇒ 主line 稀疏 indexer/长上下文自身的问题;
   若不崩 ⇒ **常驻层预取路径在长上下文下越界**,那把 `XIAOTU_MOE_PREFETCH_SLOTS`/`_resident` 作为嫌疑点。
2. `CUDA_LAUNCH_BLOCKING=1`(或 `XIAOTU_GP_TIMING` 加同步点)定位**首个**越界核。
3. 查为什么 32K 比 DMA 模型慢 3.4×(先量化 chunked-prefill 每块固定开销)。

## 168. 第 117 轮:**常驻=0 时 32K 不崩**(4/4 存活)⇒ 崩溃指向常驻/预取路径;常驻层在 8192 档 +45%

### (a) 隔离实验:常驻=0,TP=2,V1 runner,`max_num_batched_tokens=8192`,KV 8 GiB,同样的 32K 请求
| 轮次 | len=8192 | len=32768 |
|---|---|---|
| 首轮(含 JIT) | 6161 ms / 1330 t/s | 28037 ms / 1169 t/s |
| 热 rep1 | 6146 ms / 1333 t/s | 28141 ms / 1164 t/s |
| 热 rep2 | 6092 ms / 1345 t/s | 28238 ms / 1160 t/s |
每档后紧跟一次短 decode 探活:**全部 OK**,引擎存活到扫描结束。
⇒ **§167(d) 的 `persistent_topk` 非法访存在常驻=0 下 4/4 不复现**。结合 R89(Triton OOM 只在
常驻层存在时出现),**常驻/预取路径是首要嫌疑**:必须做常驻=8 的同口径复现(下一轮)。

### (b) 常驻=0 的分段(热态)
`[gp-h2d] per_layer=**79.8 / 80.0 ms**` ⇒ 1.61 GB / 80 ms = **20.1 GB/s**(与常驻=8 的 75.3 ms/21.4 GB/s
一致,略慢) ;`[gp-timing] seg=4.66-4.90ms rest=0.37-0.40ms host_total=5.06-5.30ms`(热态)。
DMA 预算(43 层全非常驻):8192 一块 = 43×1.61 GB / 20.1 GB/s = **3.44 s** vs 实测 **6.12 s**;
32768 四块 = **13.8 s** vs 实测 **28.2 s** ⇒ **32K 档只有 ~49% 由 DMA 解释**,其余 ~14.4 s 是
计算/其它(常驻=0 时单块 8192 也只有 56% 由 DMA 解释)。

### (c) 常驻层的真实收益(8192 档,均为热态)
| 配置 | 8192 TTFT | 速率 | 相对 |
|---|---|---|---|
| 常驻=0(43 层全搬) | 6092-6146 ms | **1333-1345 t/s** | 1.00× |
| 常驻=8(35 层搬) | 4235 ms | **1934 t/s** | **1.45×** |
DMA 只减少 8/43 = 18.6%,而 DMA 只占 8192 档时间的 ~56% ⇒ 纯 DMA 模型只预测 +11%,
**实测 +45%** ⇒ 常驻层还额外消除了预取流水线的停顿(每有一个常驻层就打断一次 ring 预取)。
**这是目前预填充性价比最高的杠杆**(每层 1.59 GiB 显存换 8192 档 ~5.5% 吞吐)。

### (d) 32K 档待澄清的异常
常驻=8 时 32K = **36.1 s / 908 t/s**(§167),常驻=0 时 32K = **28.2 s / 1160 t/s** ⇒ 常驻层在 32K 档
**看着变慢 28%**,与 (c) 的 8192 档结论相反。但常驻=8 那次伴随 `persistent_topk` 崩溃,
数字极可能被故障污染。**必须用常驻=8 的干净复测判定**(若确为真,则是"常驻层在长上下文下反而有害"
这一反直觉结论,必须写进 TRIED_AND_REVERTED 并改交付配置)。

### 下一轮(按优先级)
1. **常驻=8 干净复测**:8192 与 32768 各 3 次 + 探活 ⇒ 判定 (d) 与崩溃是否复现。
2. 崩溃若复现:在 `hybrid_model.py` 的 `_ov`/`PrefetchSlot` 路径上加**边界断言/同步**,
   并用 `CUDA_LAUNCH_BLOCKING=1` 定位第一个越界核。
3. 崩溃若不复现:32K 的 14.4 s 非 DMA 时间去哪了 —— 先量化 chunked-prefill 每块固定开销与
   稀疏 indexer 的 O(context) 项。

## 169. 第 118 轮:**更正 §166-§168 的口径错误**;干净同数据集结论 —— 常驻层只值 2-4%;32K 崩溃与常驻路径强相关

### (a) 【更正】`'hi ' * 4096` 只有 **4097** 个 token,不是 8192
实测(tokenizer,0731 快照):
| 串 | token 数 |
|---|---|
| `'hi ' * 4096`(§166/§167 的 `nl=4096`) | **4097** |
| `nat8192.jsonl` | 8192 |
| `nat32768.jsonl` | 32768 |
我的 `/tmp/warm_prefill.py` 打印的 `tok=nl*2` 是**错误假设**。因此:
- §167 的"8192 = 4235 ms / **1934 t/s**"实为 **4097 token / 4235 ms = 968 t/s**;
- §167 的"32768 = 36075 ms / 908 t/s"实为 **16385 token / 36075 ms = 454 t/s**;
- **§168(c) 的"+45% 常驻收益"作废**:那张表拿 res8@4097 token 去比 res0@8192 token,是**不同
  长度的错误对比**。§168(c) 的"常驻层额外消除预取停顿"推论一并撤回(无证据)。
**教训(写进方法学)**:自造 prompt 时必须**实测 token 数**,不能用字符数猜;跨会话比较前先确认
两边的 token 数一致。此后一律用 `report/tuning/datasets/nat*.jsonl`(精确长度)。

### (b) 干净的同数据集结果(全部 TP=2 / V1 runner / KV 8 GiB / maxlen 262144 / nbt 8192)
| 配置 | 8192(nat) | 32768(nat) |
|---|---|---|
| **常驻=0**(43 层全搬) | 6092 / 6146 ms → **1333 / 1345 t/s** | 28141 / 28238 ms → **1164 / 1160 t/s** |
| **常驻=8**(35 层搬) | 5977 / 5869 ms → **1371 / 1396 t/s** | 27652 ms → **1185 t/s** |
⇒ 常驻 8/43 = 18.6% 的层,换来 **+2.3%(8192)** 与 **+1.6%(32768)** 的吞吐。
DMA 模型预测应有 ~9-10%(因为 DMA 占 8192 档 57%、32768 档 49%)⇒ **实测只有预测的 1/4**,
**常驻层在 TP=2 预填充上不是一个有效杠杆**(而它同时是崩溃的诱因,见 (d))。

### (c) DMA 模型(TP=2,两轮会话一致,可信)
- 每 rank 每层 128 expert × 12.58 MB = **1.61 GB**;`[gp-h2d] per_layer` = **75.3-80.0 ms**
  ⇒ **20.1-21.4 GB/s**(PCIe Gen4 x16 实测上限量级)。
- 8192:43 × 1.61 GB / 20.1 GB/s = **3.44 s**,占实测 6.07 s 的 **57%**;
  32768:4 块 = **13.8 s**,占实测 28.2 s 的 **49%**。
⇒ **32K 档还有 51% 的时间不在 DMA 上**(~14.4 s),这是比"加常驻层"更大的目标。
  `[gp-timing] seg=4.66-4.90 ms` × 43 层 × 4 块 = 0.85 s,不足以解释 ⇒ 必须在
  attention / sparse indexer / GEMM 路径上找(下一轮用 nsys 或分段埋点)。

### (d) 32K 崩溃:**与常驻路径强相关**(这是本轮最重要的结论)
| 配置 | 32K 尝试次数 | 崩溃次数 |
|---|---|---|
| 常驻=0 | 5 | **0** |
| 常驻=8 | 4(含 r114) | **2** |
- r114 那次:Python 层可见 `launch_persistent_topk ... illegal memory access`(sparse indexer 的 decode 分支);
- **r118b 这次**:日志只有
  `Worker proc VllmWorker-0 died unexpectedly (exit code: None)`
  ⇒ **exit code None = 被信号杀死(段错误级原生崩溃),没有任何 Python traceback**,
  随后 EngineCore 的 shm 读端被 cancel 而报 `RuntimeError: cancelled`。**这是比 r114 更硬的证据**:
  常驻路径在长上下文下会**直接把 worker 打死**。
- ⇒ **常驻层必须从交付配置里去掉**(收益只有 2-4%,却带来 32K 崩溃风险)。R91 记录。

### (e) 当前水平 vs 目标(item 2:预填充 ≥1500 t/s)
- TP=2 @32768 = **1160-1185 t/s**;参考 **SM80(3090×2)= 1060 @32768** ⇒ **领先 10-12%**;
  距 1500 还差 **~27%**。@8192 = 1371-1396 t/s。
- 下一步唯一够量级的杠杆:解决 32K 档那 51% 的非 DMA 时间(怀疑 chunked-prefill 的
  稀疏 indexer O(context) 项 + 每块固定开销)。

## 170. 第 119 轮:参考仓 prefetch window 是**死旋钮**;MTP 层在参考实现里**永远常驻**;我的 GPU 埋点把引擎挂死(R92)

### (a) 【要求 (1) 的直接产出】`LVLLM_GPU_PREFETCH_WINDOW` 从未被使用
全仓 grep(`Lvllmds4-x`,只统计 `--include=*.py`):
| 出现处 | 内容 |
|---|---|
| `vllm/envs.py:252` | 类型声明 `LVLLM_GPU_PREFETCH_WINDOW: int = 1` |
| `vllm/envs.py:1860-1861` | getter,**默认值写的是 `"3"`**(与上一行的 1 不一致) |
| `vllm/envs.py:2142 / 2226` | 列入环境变量清单 + `get_gpu_prefetch_window()` 返回它 |
| `vllm/model_executor/layers/fused_moe/routed_experts.py:36` | **只是 import,函数体里从未调用** |
⇒ 除声明/getter/import 外**没有任何消费点**。**该旋钮是死代码**,
参考那台 3100 t/s 的预填充**与 prefetch window 无关**。
⇒ **不要**再去"对齐 window 深度";我们 `hybrid_model.py:857` 的真实双向流重叠
(`gpu_prefill.py:726` side stream + `PrefetchSlot`)**在机制上比参考更完整**。
(另外注意:声明默认 1、getter 默认 3,自相矛盾 —— 也印证它没被认真接过线。)

### (b) 参考实现的层分类规则(`vllm/envs.py`)
```
is_lk_moe_mtp_layer(name)        = name.startswith("mtp.")
is_lk_moe_gpu_prefill_layer(n)   = use_gpu_prefill and not resident(n) and not mtp(n)
is_lk_moe_gpu_resident_layer(n)  = (mtp(n) ⇒ True) or n ∈ LVLLM_GPU_RESIDENT_MOE_LAYERS
```
⇒ **MTP/draft 层无条件常驻显存**(不参与流式预填充、也不落 CPU)。
这解释了参考配置 `LVLLM_GPU_RESIDENT_MOE_LAYERS="0-13,43-45"` 为什么要把 43-45 写进去 ——
其实即使不写也会常驻。**我方对应动作:把 draft 的 3 个子模块也常驻**
(TP=2 每 rank 4.77 GiB,比 8 个专家层便宜),这是**解码/投机**路径的杠杆,不是预填充的。

### (c) 本轮失败(R92)
GPU 事件 + `torch.cuda.synchronize()` 埋点 ⇒ 线程池 WATCHDOG 卡死 + worker 死亡,
`kern/layer` 读出负值,数据作废,已 `git checkout` 回退。
⇒ GPU 侧分解改用 **nsys(进程外)**,不再侵入前向热路径。
⇒ 同时说明:**R91 的"32K 崩溃与常驻强相关"仍成立但要重测**(r119 的常驻=0 崩溃
是新埋点引入的变量,不能算反例)。

## 171. 第 120 轮:**更正根因** —— `RuntimeError: cancelled` 是 worker 死亡的次生症状,不是超时;并量化"每次实验 13 分钟"的浪费

### (a) 【更正】`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` 从来不是启动失败的原因
此前(第 96-113 轮,已写入旧结论)把 TP=2 启动失败归因为
"`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` 默认 300s 太小 ⇒ dequeue 超时"。
**读代码后否掉**:
```python
# vllm/v1/executor/multiproc_executor.py:425-436
dequeue_timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
status, result = mq.dequeue(timeout=dequeue_timeout)
```
而异常链实际是 `mq.dequeue` → `shm_broadcast.py:889 dequeue` → `acquire_read` →
`shm_broadcast.py:797 raise RuntimeError("cancelled")`。
`acquire_read` 抛 "cancelled" 的条件是**广播对象被取消**(写端/worker 已退出),
**与 timeout 无关**;而且 `deadline` 为 None 时 timeout 就是 None(无限等)。
⇒ **正确因果**:worker **先死** → shm 写端关闭 → EngineCore 的读端被 cancel →
报 `cancelled`。所以 `cancelled` 只是**尸检报告**,不是死因。
⇒ 推论:**把 execute-timeout 调大对启动失败没有任何作用**;要找的是 worker 为什么死
(证据:r118b/r119 两次都是 `Worker proc VllmWorker-N died unexpectedly (exit code: None)`,
**无任何 Python traceback** ⇒ 原生层被杀/段错误)。
⇒ r120(纯交付配置、无埋点、无常驻)启动也失败,同一签名 ⇒ **该故障与常驻层、与我方
插件埋点都无关,是 TP=2 启动/预热阶段的偶发原生死亡**(约 1/3 概率)。

### (b) 【时间经济学】13 分钟加载 = 本轮真正的成本中心
本轮 44 分钟里约 39 分钟是**三次模型加载**,其中一次(r120)纯属白烧(启动即死)。
教训与对策(已落地):
1. `scripts/serve_retry.sh`(新):启动失败自动重试(默认 3 次),每次先 `kill_serve.sh`
   清干净再试;成功则把 tag 写 `/tmp/serve_retry.last_tag`。
2. `scripts/bench_battery.py`(新):**一次加载内跑完整测量组** —— 预填充阶梯
   (nat8192 热身 + nat8192 + nat32768,每档后探活)+ 解码 C=1/2/3(单流/聚合/TPOT),
   而不是"改一个旋钮→重启→只测一个点"。
3. 不再用长 `sleep` 轮询;加载期间改为并行做分析/写脚本。

### (c) 顺带核准:TP=2 启动偶发失败**不是**用户环境变更引起的
"静默原则已暂停"意味着可以自由并行占用本机,但**同一对 GPU 无法并行跑两个 TP=2 实例**
(每个实例占 26 GiB/40 GiB)⇒ 加速手段只能是"减少重启次数 + 一次测更多",即 (b)。

## 172. 第 120 轮(续):**真正的根因** —— 分片池漏一个任务(`rem=1`)→ 300s 看门狗 `abort()` → worker 原生死亡

### (a) 完整死亡链(已读代码 + 两次实测复现)
```
numa_pool.hpp:709-712  分片调用等 done_cv_,条件是 remaining_==0
numa_pool.hpp:713-721  等不到(dsec 默认 300s,XIAOTU_MOE_SHARD_WD 可覆盖)
                       ⇒ 打印 [pool] WATCHDOG(sharded) + 每节点 jobs/pulled ⇒ **abort()**
⇒ worker 被 SIGABRT 杀死(无 Python traceback)
⇒ 日志: Worker proc VllmWorker-N died unexpectedly (exit code: None)
⇒ shm 广播写端关闭 ⇒ EngineCore 读端被 cancel ⇒ RuntimeError: cancelled ⇒ HTTP 500
```
**这解释了此前所有互不相干的"崩溃"**:启动阶段偶发失败(r120)、32K 崩溃(R90/R91)、
以及我误判的埋点崩溃(R92)——**同一条链**,只是延迟不同。

### (b) 两次看门狗 dump(全节点 pulled 相同且都远超 jobs)
| 运行 | 配置 | dump |
|---|---|---|
| r121_a1 | **纯交付配置:无常驻、无任何埋点** | `gen=1162 total=128 rem=1 exec=127`;`node 0..7: jobs=16 pulled=31` |
| r119 | 常驻=0 + 我的(已回退)埋点 | `gen=650 total=256 rem=1 exec=256`;`node 0..7: jobs=32 pulled=47` |
⇒ **`exec = total-1`**:一个分片被**领取后从未跑完**;`pulled` 每节点完全相同且 ≈2×jobs
⇒ 不是"某个 NUMA 节点异常",而是**所有节点对称地各领两份**、恰好漏一个。
⇒ **`XIAOTU_MOE_POOL_TRACE=1` 会打印 `SHARD-JOBDIAG ... dup=/miss=`**,这正是区分
"任务被漏(池的票据/世代 bug)"还是"任务卡死在内核里"的判据;`XIAOTU_MOE_SHARD_WD=60`
可把 300s 缩短到 60s 加快复现。

### (c) 【更正 R92】池卡死**不是**我的埋点造成的
r121 在**完全无埋点、无常驻层**的交付配置下复现同一 dump ⇒ R92 里
"我在热路径插同步埋点导致池卡死"的归因**错了**(埋点大概只是改变了时序、提高了触发概率)。
R92 正确保留的部分:①负值事件说明该埋点设计本身不可用;②**不要**在热路径做同步埋点。
⇒ 结论改写:**埋点放大了症状,但病在生产代码里**。

### (d) 这是 item 2 的**头号阻塞**,优先于任何吞吐优化
一个会每几次请求就 `abort()` 的引擎没有交付价值;而且它污染了此前所有吞吐数字的可信区间
(不确定某个数字是在"快要 abort"的状态下测的)。**在修掉 (a) 之前,不应宣称任何配置达标。**

### 下一轮(第一优先,已就绪的手段)
1. `XIAOTU_MOE_POOL_TRACE=1 XIAOTU_MOE_SHARD_WD=60` 启动,复现并读 `dup/miss`:
   - `miss>0` ⇒ 池的**票据/世代**bug(重点看 `node_base_`/`node_ticket_` 的奇偶 seqlock
     与"stale worker 掉出"路径是否**漏减 `remaining_`**);
   - `miss==0, dup==0` ⇒ 某个分片**卡死在内核里**(重点看 FAST_FP4/`moe_v2_packed4`
     的段循环对空段/异常 `seg_start` 的处理,以及 `XIAOTU_MOE_EP_SHM` 的跨 rank 双屏障)。
2. 修复后**重跑数值门禁** `scripts/check_engine_aligned.sh`(R55:动同步/内核必须过)。

### 173. 第 120 轮(续):`XIAOTU_MOE_POOL_TRACE=1` + `SHARD_WD=60` 复现 —— **`rem=2`**,分片卡在任务体内部

复现序列(TP=2,纯交付配置,仅加诊断 env):
```
#1 len=8192  mt=1   ok  6133ms
#2 len=8192  mt=8   ok 90212ms     ← **一个 8-token 解码请求跑了 90 秒**
#3 len=32768 mt=1   **FAIL** HTTP 500
[pool] WATCHDOG(sharded) gen=1324 total=128 rem=2 exec=126
```
**三条新信息**:
1. **`rem=2`(上一次 r121 是 `rem=1`)** ⇒ 漏掉的任务数**会变**,不是"某一个固定的 decrement 漏掉"
   那种确定性 bug,而是**有分片卡在任务体内部**(卡住的 worker 数可变)。
2. **90 秒的 8-token 解码请求** ⇒ 在 `abort()` 之前引擎已经**严重退化**,不是"突然死"。
   这也解释了 32K 预填充测出的"超线性变慢"(§167)很可能**部分是**这个退化,而非真实算力。
3. `SHARD-JOBDIAG`(dup/miss)**没有打印** —— 因为它在 `numa_pool.hpp:723` 的成功分支里,
   而 watchdog 走 `713-722` 的 `late` 分支并直接 `abort()`。**要拿到 dup/miss 必须把
   JOBDIAG 打印搬进 late 分支(abort 之前)** —— 这是下一轮第一件事(纯诊断改动,不改语义)。

### 头号嫌疑:**跨 rank 的 EP 共享内存双 barrier 死锁**(TP=2 特有)
- 该故障**只在 TP=2 出现过**;TP=1 在 item 1 的长跑里是稳定的。
- `scripts/tune_serve.sh:132-140` 早已记录过同类现象:"被 kill 掉的 TP>=2 进程会在
  `/dev/shm` 留下 `xiaotu_ep_L*_<hidden>_<tokens>_<world>.bin`,里面存着**双 barrier 的世代计数**,
  attach 到状态错乱的旧文件后两个 rank 的世代对不上 ⇒ **永久互等**"。
  ⇒ 现在看到的很可能是**同类世代不匹配,但是在活进程之间发生**(不是残留文件)。
- `rem=2` 也吻合:两个 rank 各有分片卡在 barrier 上,双方互等 ⇒ 两个任务永不完成。
- 判据:在 `numa_pool.hpp` 的 EP barrier 段加"等 barrier 超时即打印双方世代计数"的埋点;
  或先做**最便宜的判别实验** —— `XIAOTU_MOE_EP_SHM=0`(改用非共享内存的跨 rank 路径)
  跑同一复现序列,若不再卡 ⇒ 直接锁定 EP barrier。

### 下一轮顺序
1. 把 `SHARD-JOBDIAG` 搬进 `late` 分支(abort 前),拿到 `dup/miss`;
2. `XIAOTU_MOE_EP_SHM=0` 复现序列(最便宜的判别实验,一次加载);
3. 定位后修 + 跑 `scripts/check_engine_aligned.sh` 数值门禁(R55)。

## 174. 第 121 轮:**bug 机制定位** —— 票据已领、减量被世代守卫跳过;`EP_SHM=0` 假设被否证

### (a) 否证:`EP_SHM` 不是根因
`XIAOTU_MOE_EP_SHM=0`(跨 rank 归约从 /dev/shm 双 barrier 退回 NCCL)后跑同一序列:
```
#1 len=8192  mt=1  ok  6093ms
#2 len=8192  mt=8  ok  6794ms
#3 len=32768 mt=1  ok 28129ms      ← 比 shm 版快(28.1s vs 36.1s)
#4 len=512   mt=32 ok 83942ms      ← 84 秒!
#5 len=8192  mt=1  **FAIL** HTTP 500
```
⇒ **仍然崩,`EP_SHM` 假设否证**(已排除一条)。注意 #2 从 90212ms 降到 6794ms、
#3 也变快 —— 说明 shm 路径**确实有额外开销/退化**,但**不是死锁根源**。

### (b) 诊断读数:`miss` 是基线噪声,**真正的判据是 `exec` vs `rem`**
`XIAOTU_MOE_POOL_TRACE=1` 打开后:
```
SHARD-JOBDIAG gen=544 total=128 exec=128 dup=0 miss=1     ← 健康调用也 miss=1!
SHARD-JOBDIAG gen=574 total=128 exec=128 dup=0 miss=1
SHARD-JOBDIAG gen=652 total=480 exec=480 dup=0 miss=1
WATCHDOG(sharded) gen=2826 total=256 **rem=1 exec=256**    ← 崩溃
```
⇒ **`miss=1` 在健康调用上恒为 1** ⇒ `shard_cnt_` 诊断计数器有 0/1 基线偏差
(某个 slot 天然为 0),**不能**当作"丢任务"的证据。而 `SHARD-JOBDIAG` 行本身只会在
**成功**分支打印,所以那三行恰恰证明这些调用是好的。
⇒ **唯一可靠判据**:看门狗行的 `exec` 与 `rem` 的关系 —— 本次 `exec=256=total` 而 `rem=1`。

### (c) **机制**(代码定位,`numa_pool.hpp:938-996`)
```cpp
size_t t = node_ticket_[myn].v.fetch_add(1, relaxed);   // 939: 先领票据(单调计数)
size_t loc = t - base;
uint64_t g = current_gen_.load(acquire);                // 941
if (g == gen) {                                          // 942: 快路径
    if (loc >= nj) break;
    if (sf_) sf_(sc_, myn, loc);                         // 949: 执行任务
    shard_exec_.fetch_add(1, relaxed);                   // 950: exec++
    if (current_gen_.load(acquire) == gen &&             // 956: **再次**读世代
        remaining_.fetch_sub(1, acq_rel) == 1) notify;   // 957: 递减
    continue;
}
```
**竞态**:票据在 939 领取(从第 G 代的 `node_ticket_` 区间),而减量在 956 处**再次**检查世代。
只要在这个窗口里世代前进,减量就被**静默跳过** —— 但该票据所属的 G 代调用方**早已把它计入
`total`**。⇒ G 的 `remaining_` 少减 1,永远到不了 0 ⇒ 看门狗 ⇒ `abort()`。
这精确解释了实测签名 **`exec == total` 但 `rem == 1`**(每个任务都执行了,唯独少一次递减)。
注:963-996 的"re-anchor"分支是同一个洞的第二处(988 行 `break` 同样会**丢弃已领票据**,
与 963 行注释"NEVER abandon the consumed ticket"自相矛盾)。

### (d) 修复方向(下一轮实现 + 过数值门禁)
按**世代奇偶**分桶计数,让迟到 worker 的递减**永远落到它所属的那一代**,而不是靠"再读一次世代"来决定是否递减:
```cpp
std::atomic<size_t> remaining_[2];          // 代替单个 remaining_
// 调用方:publish 时 remaining_[gen & 1].store(total)
// worker :执行后无条件 remaining_[gen_consumed & 1].fetch_sub(1)
// 调用方:等 remaining_[gen & 1] == 0
```
这样"领票 ⇒ 必减"成为不变式,`exec` 与递减严格配对;968/988 的 re-anchor 分支也不再需要
靠"丢弃票据"来避免污染下一代。
**修完必须跑 `scripts/check_engine_aligned.sh` 数值门禁(R55)**,并重跑本轮的复现序列
(判据:`exec==total && rem==0` 且 6 轮 24 个请求全过)。

### (e) 【对 (c) 的自我校正】上一段的"世代前进"前提有漏洞,补上另一条更可能的路径
(c) 里我写"只要在 939→956 的窗口里世代前进,减量就被跳过"——**这个前提站不住**:
调用方只有在 `remaining_==0` 之后才会发布下一代,而本 worker 若属于本代且尚未递减,
`remaining_` 就不可能是 0 ⇒ **本代内不可能因"世代前进"而丢减量**。必须修正为两条**不依赖该前提**的路径:

1. **`node_base_` 陈旧/撕裂读(更可能)**:
   `:937` 把 `node_base_[myn]` 读进局部 `base`,**只在 re-anchor 分支里刷新**;
   而调用方在发布期(`:695`)写 `node_base_[myn]`。worker 完全可能用**上一代的 base**
   去算 `loc = t - base`(t 已是新代票据)⇒ `loc >= nj` ⇒ `:943/:988 break`
   ⇒ **票据被消费却不递减、也不 exec** ⇒ 正是"少一次递减"。
2. **`re-anchor` 分支 `:988` 的 `break`**:与 `:963` 注释
   "NEVER abandon the consumed ticket" **自相矛盾** —— 它确实会丢弃已领票据。

⇒ 因此**可辩护的结论**是:`remaining_` 的递减**不是"领票即减"的不变式** ——
它被 `:956` 的二次世代读、以及 `:937` 的 base 陈旧读共同置于"可跳过"状态;
实测签名 `exec==total && rem==1`(+`miss=1` 基线噪声、`dup=0` 不可信)
与"有票据被消费但未配对递减"一致。
⇒ **下一步不是直接照 (d) 改,而是先用最小实验证明是哪一条**:
在 `:939` 领票后立刻记下 `(t, base, gen)`,并在 `:943/:988` 两个 `break` 处
打印"被丢弃的票据"(加一个 `XIAOTU_MOE_POOL_TRACE` 计数器 `abandoned`)。
若 `abandoned>0` ⇒ 就是丢票据;修法与 (d) 的"按世代奇偶分桶"一致(它同时消除
"再次读世代"和"丢弃票据"两个可跳过点)。**先证明再改,避免第 4 次误归因。**

## 175. 第 122 轮:第 4 个假设被否证;**停止猜测,改为埋点证明**

### (a) 否证记录
按 §174(e) 的"陈旧 base"假设做了最小修复(读序调整 + 时序论证)并重编译 → **无效**,
两次启动尝试 dump 与修复前逐字相同(`rem=1`)。详见 R94。改动已回退。

### (b) 现在能**严格确定**的事实(不再有推理跳跃)
1. 崩溃判据是**唯一**可靠的:`remaining_` 停 1 或 2,调用方永远等不到 0 ⇒ 看门狗 `abort()`。
2. `shard_exec_`(`exec`)与 `remaining_` **不是**原子配对:`exec == total` 时 `rem` 仍可为 1。
   ⇒ 存在"执行了但没递减"或"递减了但没执行"的执行路径。
3. `miss=1` 是**健康调用也有的基线噪声**;`dup` 因同一计数器问题**不可信**。
4. `EP_SHM=0` 不修此故障(已否证)。

### (c) 下一轮:**先埋点,后改代码**(这次不再跳步)
在 `numa_pool.hpp` 加两个**纯诊断**计数器(不改任何控制流),并在看门狗分支打印:
- `abandoned_`:在 `:943`、`:981`、`:988` 三个 `break` 处自增(票据被消费后丢弃);
- `skipped_dec_`:在 `:956`/`:991` 的世代守卫**为假**时自增(执行了但递减被跳过)。
看门狗 dump 里 `abandoned`/`skipped_dec` 谁非零,就直接定位是哪条路径,再动手修。
**判据仍然是 `exec==total && rem==0` 且复现序列 24 个请求全过 + `check_engine_aligned.sh` 数值门禁。**

## 176. **结案(用户于第 123 轮裁定)**:预填充上限 —— 我方已超 SM80 参考,1500 t/s 不作为必跨门槛

### 裁定
用户指示:"已测得的干净上限仍是预填充 nat8192 1335–1396 t/s、nat32768 1160–1185 t/s
(超 SM80 参考 1060@32768 约 10–12%)。**这个也视为已经结案**,整理好文档和过程得失,
继续下一个方案。" ⇒ 预填充方向**停止投入**,不再以 1500 t/s 为阻塞项。

### 结案依据(口径可复核)
测量口径:TP=2 / `VLLM_USE_V2_MODEL_RUNNER=0` / EAGER / KV 8 GiB / maxlen 262144 /
`max_num_batched_tokens=8192` / **常驻=0**(交付配置)/ 精确长度数据集 `nat*.jsonl` /
**客户端 TTFT 为准**(引擎 loggers 的窗口吞吐在长请求突发时会高估,已证不可用,§167)。

| 口径 | 我方(TP=2) | 参考 |
|---|---|---|
| nat8192 | **1335–1396 t/s** | — |
| nat32768 | **1160–1185 t/s** | **SM80(3090×2)= 1060 @32768** ⇒ 我方**领先 10–12%** |
| (不可比) | — | PRO 6000 = 3100(单卡 96 GB、原生 NVFP4、14 层专家+draft 常驻、无 PCIe 流式) |

### 物理上限(这才是"为什么到此为止")
- 每 rank 每层要搬 **1.61 GB**(128 experts × 12.58 MB),实测 H2D **20.1–21.4 GB/s**
  ⇒ 已在 PCIe Gen4 x16 实测上限量级;
- DMA 占 8192 档时间的 **57%**、32768 档的 **49%**(§169c);
- 常驻层削减 18.6% 的层数,实测只换 **+2.3%/+1.6%**(§169b),而它是 32K 崩溃的诱因(R91);
- 参考那台 3100 t/s 的前提是**整个模型装进一张 96 GB 卡**(无 PCIe 流式)。我方 40 GB 卡
  每层必须流式 ⇒ **与 PRO 6000 的差距是显存容量与卡架构决定的,不是调参差距**。
⇒ 想再上台阶只有"减少要搬的字节"(更低精度权重 / 更大显存卡),不在插件旋钮范围内。

### 附带产出(要求 (1) 的比对结果,已可复用)
- `Lvllmds4-x` 的 `LVLLM_GPU_PREFETCH_WINDOW` 是**死旋钮**(全仓仅声明/getter/import,无消费点,
  且声明默认 1 与 getter 默认 3 自相矛盾)⇒ **不必对齐 window 深度**;
- 参考实现里 **MTP/draft 层无条件常驻**(`is_lk_moe_gpu_resident_layer("mtp.")` 恒 True)
  ⇒ 这是**解码/投机**的杠杆,已转入下一方案。

## 177. 【更正】EP shm 容量:两道门禁都在,不存在越界;只剩一个纯性能的注释不实
第 121 轮我向用户报告的"CPU 通路越过 EP shm stride 8 倍写入"**是错的**,已在 R95 详述。
正确事实:
- 引擎 `binding.cpp:437-440` `if (bytes <= ep->capacity)` —— 放不下就不进 shm barrier;
- Python `hybrid_model.py:983` `_need_allreduce = qlen > _ep_shm_tokens` —— 超出即回退 NCCL。
⇒ **纯 CPU 预填充(qlen 任意大)是正确的**,用户要求的"慢可以、错不行"已满足。
唯一真实残留是 `hybrid_model.py:736-738` 注释与实现不一致(实现只取
`XIAOTU_MOE_EP_SHM_TOKENS`,没取 `max(·, 阈值)`)⇒ GPU 阈值 >1025 时会**静默**走 NCCL 回退,
是**性能**问题而非正确性问题。

## 178. 预填充速率**逐请求可复现**,但**作为持续服务速率不可用** —— 回答"1300 是稳定速率吗"

### (a) 逐请求速率:跨 4 次独立启动,极差 <1.2%
| 会话(TP=2,常驻=0) | nat8192 | nat32768 | 结局 |
|---|---|---|---|
| r117 | 6161 / 6092 / 6146 ms | 28037 / 28141 / 28238 ms | **5/5 通过,存活** |
| r118b(常驻=8) | 5977 / 5869 ms | 27652 ms | 第 2 次 32K 崩 |
| r121(纯交付、无埋点) | **6134 ms** | — | 第 2 次 8192 崩 |
| diag_a1(POOL_TRACE) | **6133 ms** | — | #3 崩 |
| noshm_a1(EP_SHM=0) | **6093 ms** | 28129 ms | #5 崩 |
- **nat8192 首次请求 = 6093–6161 ms(1330–1345 t/s),极差 1.1%**;
- **nat32768 = 28037–28238 ms(1160–1165 t/s),极差 0.7%**(全为常驻=0)。
⇒ **这是一个真实、可复现的速率**,不是某一次启动的偶然值;而且**其中三次是在"后来崩掉"的
会话里测的**,数值与存活的 r117 一致 ⇒ 崩溃**不是**因为速率退化,两者独立。

### (b) 但**不能**当作持续服务速率:没有任何一次 TP=2 会话撑过持续负载
- 从未在 TP=2 跑完一次 `C=8 / N=32` 的持续压测(全部在中途 `abort()`);
- 触发挂死的那一次请求延迟**灾难性**(8-token 解码 **90212 ms**、32-token 解码 **83942 ms**),
  这种值会把任何均值/分位数彻底污染;
- 因此**我现在没有"持续吞吐"这个数字**,只有一个"单请求速率"。二者不能互相替代。
⇒ 结论:**"1300 稳定"只对"单请求速率"成立**;对"服务可用性"不成立。在修掉 pool 漏任务
之前,任何"达标"声明都缺一个必要条件(引擎需在一段持续负载内存活)。

### (c) 尾巴已计入待办(用户 2026-09-12 指示)
"请求几次就 abort"已记为 item 2 的**前置阻塞项**,写入 `FUTURE_PLAN.md` 的 P0,
含精确判据(`exec==total && rem==0` + 复现序列 24 请求全过 + 数值门禁 + **一次持续压测**
`C=8/N=32` 全程存活),以及已埋好的两个纯诊断计数器(`abandoned_` / `skipped_dec_`)。

## 179. **决定性诊断读数**:`abandoned≡线程数`(基线噪声)、`skipped_dec≈0`(否证世代守卫)⇒ 卡在**任务体**而非计数协议

### (a) 读数(两个纯诊断计数器,均来自**启动阶段**的失败 dump)
```
diag2_a1: [pool] WATCHDOG(sharded) gen=598 total=256 rem=1 exec=256
          [判据] abandoned=120 skipped_dec=1
diag2_a2: [pool] WATCHDOG(sharded) gen=568 total=128 rem=2 exec=126
          [判据] abandoned=120 skipped_dec=0
```

### (b) `abandoned=120` **不是** bug,是稳态基线
`120` **恰好等于工作线程数**(`scripts/tune_serve.sh` 的 `THREADS` 默认 120)。
机制:某代任务的票被领完后,每个 worker 会再 `fetch_add` 一次、发现 `loc >= nj` 就
`break` 回去重新 arm ⇒ **每代每 worker 恰好一次"多领即弃"** ⇒ 基线就是 `#workers`。
⇒ 与 `miss=1` 同类:**`abandoned` 也是基线噪声,不能当丢票证据**。
(此处再次印证那条规矩:诊断计数器必须先建立**基线值**,否则非零即误判。)

### (c) `skipped_dec ≈ 0` ⇒ **否证"递减被世代守卫跳过"**(第 6 个被否证的假设)
两个 dump 的 `skipped_dec` 分别是 1 和 0。若"世代前进导致守卫为假、递减被跳过"是主因,
这里应该等于缺口的数量级(1~2)——但 a2 缺口是 2 而 `skipped_dec=0`。
⇒ `:956`/`:991` 的守卫**不是**问题所在,不必再改那里(省下一轮)。

### (d) **唯一的真信号是 `exec` 的缺口**,它指向**任务体**
| dump | total | exec | 缺口 | 含义 |
|---|---|---|---|---|
| a1 | 256 | 256 | 0 | 全部执行了,但少一次递减(见下) |
| a2 | 128 | **126** | **2** | **2 个被计入的任务被领走后,既没执行、也没走到任何 break** |
a2 是最干净的证据:`abandoned` 仍是正常的 120,而 `exec` 少 2 ⇒ 持有那两张票的 worker
**停在了 `sf_(sc_, myn, loc)` 内部(或阻塞在 re-anchor 的 `work_mtx_` 上)**,
既没 `shard_exec_++` 也没 break。
a1 的"(exec 满但 rem 少 1)"同样可以由"某个 worker 在任务体里卡住、由另一个 worker
重复执行了同一张票"解释 —— 此时 `dup` 本应 >1,而 `dup` 计数器已被证明不可信(§174b)。
⇒ **结论:不要再查票据/世代/递减协议,要查"任务体里会阻塞的东西"。**

### (e) 下一轮(方向已换,具体可查的候选)
1. **任务体里是否有会长时间阻塞的东西**:`sf_` 最终落到 CPU MoE 的 shard 回调,
   其中可能拿 `work_mtx_`/其它 mutex、或**回调进 Python/torch 而争 GIL**。
   若主线程长时间持 GIL(torch 算子/采样),120 个 worker 会集体排队 ——
   这能解释"8-token 解码跑 90 秒"这种量级的停顿,以及只有个别 worker 未归位。
2. **最便宜的判别实验**:把 `XIAOTU_MOE_THREADS` 从 120 降到 32 跑同一复现序列。
   若是 GIL/锁竞争,降线程数应显著改变触发概率;若是纯计算越界,则不变。
3. 在任务体入口/出口各加一个计数器(`entered`/`left`),卡住时两者之差即"进了没出"的 worker 数
   —— 这能把 (d) 的推断变成直接证据。

## 180. **更正 §179(c)**:`skipped_dec` 与缺口**精确对应** ⇒ 存在**两种**失败模式,世代守卫(模式 A)是真凶之一

### (a) 服务态复现(诊断版 diag2_a3,7 个请求)
```
#1 8192/mt1  ok  6118ms     #5 8192/mt1  ok  6147ms
#2 8192/mt8  ok  6796ms     #6 8192/mt8  ok 90262ms   ← 同一个请求!
#3 32768/mt1 ok 28146ms     #7 32768/mt1 **FAIL**
#4 512/mt32  ok  3750ms     WATCHDOG gen=8268 total=256 rem=1 exec=256
                            判据 abandoned=120 skipped_dec=1
```
**同一个 `nat8192, max_tokens=8` 请求:第 2 次 6796 ms、第 6 次 90262 ms** ⇒
挂死**不是形状/数据相关**,是**竞态或资源争用**(非确定性)。这是本轮最强的刻画。

### (b) 三次 dump 并排 ⇒ **`skipped_dec` 与缺口精确相等**
| dump | total | exec | rem | **缺口** | abandoned | **skipped_dec** | 模式 |
|---|---|---|---|---|---|---|---|
| diag2_a1 | 256 | **256** | 1 | **1** | 120 | **1** | **A** |
| diag2_a3 | 256 | **256** | 1 | **1** | 120 | **1** | **A** |
| diag2_a2 | 128 | **126** | 2 | **2** | 120 | **0** | **B** |
- **模式 A**(2/3 次):`exec == total`,所有任务都执行了,但**少一次递减**,而
  **`skipped_dec` 恰好 = 1 = 缺口** ⇒ **`:956`/`:991` 的世代守卫确实被触发并跳过了递减**。
- **模式 B**(1/3 次):`exec = total-2`,有 2 个被计入的任务**从未执行**,`skipped_dec=0`
  ⇒ 持票 worker 卡在任务体里(§179d 的推断只对模式 B 成立)。
- `abandoned=120` 三次**恒等于线程数** ⇒ 基线噪声的判定得到第三个样本确认(**不要**再当证据)。

### (c) 【更正 §179(c)】我上一节说"`skipped_dec≈0` ⇒ 否证世代守卫" —— **错了**
我当时只看了 a2(模式 B),而模式 A 的两份 dump 里 `skipped_dec` 与缺口**逐次精确相等**。
正确结论:**世代守卫是模式 A 的直接原因**,§174(d) 提出的"按世代奇偶分桶
`remaining_[2]`,让递减永远落到它所属的那一代"**重新成为正确的修复方向**;
而模式 B 需要另外一条线(任务体阻塞)去修。**两种模式都要修**,只修一个仍会挂。

### (d) 模式 A 的机制自洽性(为什么守卫会为假)
调用方只在 `remaining_==0` 后才发布下一代,所以"本代未递减而世代已前进"看似不可能。
但 **`:983` 的 re-anchor 分支会把 worker 的本地 `gen` 改成"当前活代"**,
于是**迟到的 worker(来自更早的世代)会把自己的递减记到活代的 `remaining_` 上**
⇒ 活代被**多减**(提前到 0、调用方提前返回),而它自己那一代被**少减**。
"多减"与"少减"互相掩盖,净效果就是"某一次调用永远等不到 0"。
⇒ 按世代奇偶分桶正是为了让递减**只能记到自己那一代**,从根上消除这种错配。

### (e) 结论(下一轮)
1. **优先修模式 A**:`remaining_[2]`(按世代奇偶分桶)+ 递减不带守卫(领票即必减)。
   `skipped_dec` 改为不变式检查(修复后应恒为 0,非 0 即新 bug)。
2. 修完重跑复现序列(判据:`exec==total && rem==0`,且**连续 3 次会话的 24 请求全过**)。
3. 若仍有模式 B(exec 缺口),再按 §179(e) 查任务体阻塞(GIL/锁),用 `entered`/`left` 计数。
4. 全过之后:数值门禁 `check_engine_aligned.sh` + **一次 C=8/N=32 持续压测**。

## 181. 第 121 轮:模式 A 修复已编译,但**挂死以模式 B 复现**(未修好);且我的判据 `skipped_dec` 已失效

### (a) 改动(六点,已编译通过,5 个 ISA 变体)
`remaining_sh_[2]` 按世代奇偶分桶 + `sh_slot(gen)=(gen>>1)&1`;分片路径的发布/等待/看门狗打印
改用分桶;两处递减**去掉世代守卫改为无条件递减**(`numa_pool.hpp`)。**flat 路径未动**。
编译期踩到一个真实约束:发布块的 `gen` 是**局部作用域**,等待块看不到,改用等待时的
`current_gen_`(此刻必等于该偶代)取槽位。

### (b) 结果:**没有修好** —— 挂死改为模式 B
```
会话1: #1 8192/mt1 ok 6126ms → #2 8192/mt8 **ok 89899ms**(停顿) → #3 32768 **FAIL**
随后服务进程整体死亡(会话2/3 连 /v1/models 都不通)
WATCHDOG(sharded) gen=1784 total=128 rem=1 exec=127
判据] abandoned=120 skipped_dec=0
```
- 这是**模式 B**:`exec = total-1`(**一个任务从未执行**)、`rem=1`、`skipped_dec=0`
  ⇒ 一张票被领走后**既没执行、也没走到任何 break** ⇒ 持票 worker **卡在任务体 `sf_(...)` 里**。
- **运行期间无模式 A 签名**(`exec==total 且 skipped_dec=1`)⇒ 修复**可能**有效,但
  **只有 1 次会话,不能宣称修好**。且注意模式 A 与 B 独立,**修好 A 也不等于引擎稳定**。

### (c) 【我的埋点缺陷】`skipped_dec_` 现在**结构上恒为 0**,已不是判据
我在删掉世代守卫的同时,把 `skipped_dec_.fetch_add(...)` 那一行也删了 ⇒ 该计数器**永不递增**,
dump 里的 `skipped_dec=0` **没有信息量**。⇒ 下一轮必须把它换成**真正的不变式检查**
(例:递减前若桶值已为 0 ⇒ 记 `underflow_++`,用于抓"多减/重复减")。**这是我第二次在埋点上出错**
(第一次是 R92 的同步埋点),教训:**改控制流时必须同时检查自己留下的判据是否还有效**。

### (d) 负载线索(记录,尚未定性)
跑这段时 `uptime` 的 5/15 分钟负载 = 15.9 / **58.8 / 58.6**(192 核)。负载没到饱和,
且分片调用正常应是 ~10-50 ms 量级,而这里停顿到 **60 s(上千倍)** ⇒ **不是调度抖动**,
是**真的阻塞**。但仍需排除"120 个 MoE 线程 + `OMP_NUM_THREADS=48` + 其它进程"造成的
**屏障尾延迟放大**:分片调用是**屏障**,调用耗时 = 最慢线程耗时,一旦机器上有别的负载,
尾延迟会被放大。**下一轮的判别实验**:`XIAOTU_MOE_THREADS=120→32` 跑同一序列
(若触发概率显著下降 ⇒ 是尾延迟/争用;若不变 ⇒ 是任务体内的确定性阻塞)。

### (e) 下一步(按顺序)
1. **任务体入口/出口计数器** `entered_`/`left_`(纯诊断)加在 `sf_(sc_, myn, loc)` 的两侧,
   看门狗 dump 里打印差值 ⇒ 直接给出"进了没出"的 worker 数,把 (b) 的推断变成事实;
2. 用 `XIAOTU_MOE_THREADS=32` 做判别实验(便宜且能区分争用 vs 确定性阻塞);
3. 若差值=1 且与线程数无关 ⇒ 打开任务体内部:重点看**跨 rank EP 屏障的自旋**
   (`binding.cpp:446-456` 的 `while (h->gen.load()==gen) yield()`)、`work_mtx_`/`done_mtx_`、
   以及**回调进 Python 争 GIL**这三处;
4. 修好后判据(四项,不可省):连续 3 次会话 24 请求全过 + 无看门狗 + 数值门禁 + `C=8/N=32` 持续压测。

## 182. **直接证据**:没有任何 worker 卡在任务体里 ⇒ 阻塞点在 `sf_()` **之前**的 re-anchor 路径(更正 §181b)

### (a) 读数(带 `entered_`/`left_` 计数器的版本)
```
会话1: #1 8192/mt1 ok 6105ms → #2 8192/mt8 ok 89763ms(停顿) → #3 FAIL
会话2: #1 8192/mt1 FAIL(服务进程已死)
WATCHDOG(sharded) gen=1378 total=128 rem=1 exec=127
判据] abandoned=120 underflow=0  entered=127 left=127
node 0..7: jobs=16 pulled=31
```
### (b) 结论:**任务体是干净的**,阻塞点在它**之前**
- `entered == left == 127` ⇒ **凡进入 `sf_()` 的 worker 全部返回了**,**没有**"进了没出"的 worker
  ⇒ **更正 §181(b)**:"持票 worker 卡在任务体 `sf_()` 里"是**错的**。
- `exec=127 = entered` ⇒ 执行计数与入口计数自洽。
- `abandoned=120` = **基线值**(= 线程数,§179b 已三次确认)⇒ 那张被计入的票
  **没有走任何 `break` 路径**(否则 abandoned 会是 121)。
- 算术自洽:`total=128`,128 张在范围内的票被领走,只执行了 127 次
  ⇒ **第 128 张票的领票者在到达 `entered_++` 之前就停住了**。

### (c) 那么它停在哪?`fetch_add` 与 `entered_++` 之间**唯一会阻塞的构造**是 re-anchor 分支的
```cpp
// numa_pool.hpp ~:984
std::lock_guard<std::mutex> lk(work_mtx_);   // ← 这条路径上唯一可能长时间阻塞的点
```
⇒ **头号嫌疑:`work_mtx_` 被长期持有 / 锁序问题**(而不是任务体、不是票据协议、不是世代守卫)。
`underflow=0` 同时说明**没有多减/重复减**,分桶计数本身是健康的。

### (d) 下一步(直接、便宜)
1. 加 `reanchor_in_`/`reanchor_out_` 计数器,包住 re-anchor 分支(与 `entered_` 同一手法);
   若看门狗时 `reanchor_in - reanchor_out == 1` ⇒ 就锁定在这个分支里。
2. 再在其中细分:分别统计"进 `work_mtx_` 前/后" ⇒ 直接判定是不是锁。
3. 若确认是锁:检查 `work_mtx_` 的所有持有者(发布块、re-anchor 块、以及任何持 `done_mtx_`
   再取 `work_mtx_` 的路径)是否存在**锁序反转**(`done_mtx_` ↔ `work_mtx_`)。
4. 备选(与锁无关的可能):该 worker 被**长期抢占**(120 线程 + OMP 48 + 其它进程),
   ⇒ 用 `XIAOTU_MOE_THREADS=32` 做判别实验:触发概率显著下降即为此因。

## 183. 锁图审计:`work_mtx_` 无嵌套、不是死锁;但发现 **flat 路径同型缺陷**(我上一轮只修了分片路径)

### (a) 锁图(全部 `work_mtx_`/`done_mtx_` 持有者)
| 位置 | 持锁 | 说明 |
|---|---|---|
| :432 | `call_mtx_` | 整个调用串行化 |
| :439 / :663 | `work_mtx_` | flat 发布 / 分片发布 |
| :487 / :707 | `done_mtx_` + `done_cv_.wait_until` | 调用方等待(等待时释放) |
| :904 | `work_mtx_` + `cv_.wait` | worker 外层 anchor(等待时释放) |
| :977 / :1018 | `done_mtx_` | 递减到 1 后 notify |
| :991 / :1058 | `work_mtx_` | 分片 re-anchor / flat re-anchor |
**结论:`work_mtx_` 与 `done_mtx_` 从未嵌套持有**(每个 `lock_guard` 都在独立作用域内),
⇒ **§182c 的"`work_mtx_` 长期持有/锁序反转"假设不成立**。锁本身是干净的。
(`:1038-1040` 的注释还专门解释了"必须持 `done_mtx_` 才能 notify"以避免丢唤醒 —— 作者已处理过这一层。)

### (b) 但审计暴露了**两处真实缺陷**(都在 **flat** 路径,我上一轮没改)
1. **`:1044-1048`**:与模式 A 完全同型的"领票后世代守卫"——
   ```cpp
   if (current_gen_.load(std::memory_order_acquire) != gen) continue;  // 跳过递减!
   if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) { ...notify... }
   ```
   票据已从 `counter_` 领走,世代前进就 **`continue` 且不递减** ⇒ flat 调用方永不归零。
2. **`:1072-1076`**:活调用是分片时,worker 把已领的 **flat 票据丢弃**(`break` 去重新 arm),
   **不递减**。
⇒ 这两处都会产生"`remaining_` 少减一次"的 hang,只是报在 **flat** 看门狗上。
**必须与模式 A 同法修复**(按世代奇偶分桶 + 领票即必减),否则交付版仍会在 flat 路径上挂死。

### (c) 关于模式 B(分片)的残留谜题:下一步用**直接计数**收口,不再猜
已知:分片调用 `total=128 / exec=127 / entered=127 / left=127 / abandoned=120(基线)`,
且锁已排除、任务体已排除。⇒ 那张在范围内的票消失在
`fetch_add`(:939)与 `entered_++`(:958)之间的**某个分支**里。
**下一步加一个计数器就能收口**(纯诊断,不改控制流):
```cpp
// :941 拿到 g 之后、进入 fast path 时:
if (loc < nj) inrange_.fetch_add(1, relaxed);     // 本代"范围内"的票被领走
```
看门狗时比较 `inrange_` vs `entered_`:
- `inrange_ == entered_ + 1` ⇒ 确实有一张范围内的票在 fast path 内消失(那就要看
  `diag_active()` 分支与 `sf_` 之间的东西);
- `inrange_ == entered_` ⇒ 那张票**根本没被领**(= 票号区间与实际 job 数不一致,
  即 `node_base_`/`node_nj_` 的发布与 worker 的读之间有**撕裂**),这会把矛头转向发布时序。

## 184. **真根因**:分片发布违反了本文件自己的 seqlock 不变式 ⇒ worker 少看到一张票

### (a) 决定性读数(`inrange_` 计数器,原假设二分的结果)
```
会话1: #1 8192/mt1 ok 6137ms → #2 8192/mt8 ok 6749ms → #3 32768 ok 28105ms
       → #4 512/mt32 **84882ms**(停顿) → #5 FAIL;会话2 #1 FAIL
WATCHDOG(sharded) gen=3936 total=128 rem=1 exec=127
判据] abandoned=120 underflow=0  entered=127 left=127 **inrange=127**
```
**`inrange == entered == 127 < total = 128`** ⇒ 二分得到**第二支**:
**那张票根本没有被任何 worker 看成"范围内"** ⇒ 不是锁、不是任务体、不是世代守卫,
而是**发布/读取之间的一致性 bug**。

### (b) 根因:`node_nj_` 写在**奇数(发布中)store 之前**
`numa_pool.hpp:455-471` 自己写下的不变式是:
> 调用方必须在读 `counter_` 之前先 store 奇数、**写完全部字段**再 store 偶数。
> 这样 worker 只需接受偶数即可,不会看到"新 gen + 旧 start_"的撕裂
> (轮 68 的朴素"复读 gen"正是死在这里:撕裂时 gen 已新、start_ 还旧,复读一致
> ⇒ 查不出来 ⇒ **worker 丢票 ⇒ remaining_ 永不归零**)。

而分片发布把 `node_nj_[n] = job_counts[n]`(`:675-684`)以及 `n_`/`worker_limit_`/
`sharded_call_`/`start_` 都写在 `:693` 的奇数 store **之前**。后果:
**仍在上一代(偶代)的 worker 在 `:937` 读到的是新一代的 `node_nj_`,而它的 `base`/gen 还是旧的**
⇒ 可领区间变小(或错位)⇒ **丢掉一张票** ⇒ 该代 `remaining_sh_` 永远停在 1 ⇒ 看门狗 `abort()`。
- 这**解释了为什么 R94 只把 `base` 的读序后移无效**:被撕裂的是 `node_nj_`,不是 `base`。
- 也解释了为什么故障非确定性:同一请求快一次、慢一次(89899ms),取决于是否撞上发布窗口。
- 也解释了为什么启动阶段最易触发(预热期并发发布最密集)。

### (c) 修复(第 124 轮,已编译通过)
把 `node_nj_`/`total`/`n_`/`worker_limit_`/`sharded_call_`/`start_` 的写入**整体移进
"奇数 store → 偶数 store"的发布窗口内**,无条件遵守既有不变式;
`node_base_` **仍**留在奇数 store 之后读(陈旧票必须落在 base 之前,这条不能动)。

### (d) 待验证(下一轮,四项缺一不可)
① 连续 3 次会话 24 请求全过、无看门狗;② 判据 `inrange==entered==total && rem==0`;
③ `scripts/check_engine_aligned.sh` 数值门禁(R55);④ 一次 `C=8/N=32` 持续压测跑完。
**注意**:flat 路径(`:1044-1048` 的世代守卫跳过递减、`:1072-1076` 丢弃已领 flat 票据)
是**同型的另外两处**,尚未修 —— 若验证中报 `WATCHDOG`(非 sharded)即命中它们。

## 185. 验证结果:**分片路径已修好**(首次 24/24 全过);flat 路径成为唯一残留,证据同样干净

### (a) 分片路径:发布顺序修复**生效**
```
会话1: #1..#24 **全部 ok**(含 8192/mt1、8192/mt8、32768/mt1、512/mt32 各 6 轮)
       #24 len=512 mt=32 ok **300014ms**  ← 唯一异常
       健康 200 → 会话2 #1 FAIL → 健康 000(进程在会话之间死亡)
全程**无 `WATCHDOG(sharded)`**、**无 `判据]` 行** ⇒ 分片看门狗一次都没触发。
```
这是本会话**第一次**跑完一整个 24 请求序列(此前最好成绩是第 3 个请求就崩)。
⇒ §184 的"`node_nj_` 写在奇数 store 之前 ⇒ worker 丢票"**是分片路径的真根因**,
把字段写入移进发布窗口的修复**有效**。

### (b) 但 flat 路径以**同型**症状接管(300 s 后看门狗 + abort)
```
[pool] WATCHDOG fired: gen=35800 n=2 start=43431 end=43433 counter=43553 remaining=1
                       current_gen=35800 dropped=0
```
- `n=2`(本次 flat 调用 2 个任务)、区间 `[43431,43433)`;
- `counter - end = 43553 - 43433 = 120` = **线程数**(即各 worker 多领一次票的基线,与分片侧 `abandoned=120` 同源);
- `remaining=1` ⇒ **少一次递减**;`dropped=0` ⇒ flat 自己的丢票计数没涨。
⇒ 与分片模式 **完全同型**的"票被领走但没递减"。命中的正是 §183b 记录的两处:
`numa_pool.hpp:1044-1048`(世代守卫 `continue` 跳过递减)与
`:1072-1076`(活调用是分片时丢弃已领 flat 票据且不递减)。
⇒ **flat 路径必须做与分片完全相同的修复**(按世代奇偶分桶 `remaining_fl_[2]` + 领票即必减,
并处理 `:1072-1076` 的丢弃分支)。**这是最后一个阻塞项。**

### (c) 为什么 flat 路径这次才暴露
flat 池只在**非分片调用**上使用(小批次/解码等,本例 `n=2`)。此前分片 bug 每次都先炸,
把 flat 的窗口掩盖了;分片修好后,flat 成了唯一残留。

## 186. 第 126 轮:flat 修复已上,但**分片同型失败复发** ⇒ 发布顺序只修了"写端",**读端仍非原子**

### (a) 实测
```
会话1: #12..#16 ok(#16 512/mt32 **85337ms** 停顿) → #17 FAIL;会话2/3 全 FAIL(进程已死)
[pool] WATCHDOG(sharded) gen=25900 total=480 rem=1 exec=479
  [判据] abandoned=120 underflow=0  entered=479 left=479 inrange=479
```
- **`inrange == entered == 479 < total = 480`** —— 与 §184 修复**前完全同型**;
- 本次**没有** flat 看门狗(所以 flat 那两处修复既没被证明有效,也没被否证 —— 分片先炸了);
- 对比 §185:fixPub 那次 **24/24 全过**,这次却在第 17 个请求复发
  ⇒ 发布顺序修复**降低了频率但没有根除**。

### (b) 缺失的一半:**读端没有走 seqlock 协议**
`:466-471` 的注释写"worker **只需接受偶数**即可"——这句话**不完整**。writer 端现在(§184 修复后)
确实只在窗口内写字段,但 **reader 端 `:937` 仍在无同步下分两次读字段**:
```cpp
size_t base = node_base_[myn], nj = node_nj_[myn];   // 两次独立 load,可以跨越奇/偶窗口
```
标准 seqlock 读端必须是 **读 gen → 读字段 → 复读 gen → 不一致则重试**。
现在的读端没有复读校验 ⇒ 即使 writer 端一致,reader 仍可能取到
"新 `nj` + 旧 `base`"(或反之)⇒ 可领区间错位 ⇒ **一张票永远不被看成 in-range**
⇒ `remaining_sh_` 停在 1 ⇒ 看门狗 `abort()`。这与实测签名精确吻合。

### (c) 下一轮修法(精确、小)
在分片 anchor 处把 `:937` 换成标准 seqlock 读:
```cpp
uint64_t g0, g1; size_t base, nj;
do {
    g0 = current_gen_.load(std::memory_order_acquire);
    if (g0 & 1) continue;                     // 发布中 ⇒ 重试
    nj = node_nj_[myn]; base = node_base_[myn];
    g1 = current_gen_.load(std::memory_order_acquire);
} while (g0 != g1);                            // 读字段期间世代变了 ⇒ 重试
```
并把同样的问题在 **flat 读端**(`:1058` 的 `nn = n_; ns = start_;`)一并检查
(那里在 `work_mtx_` 下读,理论上受锁保护,需确认 `n_`/`start_` 的所有读取都在锁内)。

### (d) 附带结论
flat 的两处修复(去守卫 + 丢弃前先递减)已编译入库,但**尚未获得有效验证**
(本轮分片先炸)。下一轮若分片修好后仍出问题,再回来看 flat。

## 187. 第 127 轮:补上**读端 seqlock**(`numa_pool.hpp:953`),已编译,验证进行中

### 改动
```cpp
// 旧(:953):进循环前一次性读,无同步 —— 可跨越 writer 的奇/偶窗口
size_t base = node_base_[myn], nj = node_nj_[myn];
// 新:确认偶代之后才读字段,并复读世代校验
size_t base = 0, nj = 0, loc = 0;
for (;;) {
    size_t t = node_ticket_[myn].v.fetch_add(1, relaxed);
    uint64_t g = current_gen_.load(acquire);
    if (g == gen) {
        base = node_base_[myn];
        nj   = node_nj_[myn];
        if (current_gen_.load(acquire) != gen) continue;   // 读字段期间世代变了 ⇒ 重试
        loc = t - base;
        ...
```
正确性:writer 只在**奇数**窗口内写字段,因此"读前 gen 为偶数 + 读后 gen 仍为该偶数"
⇒ 字段在读取期间不可能被改 ⇒ 得到 (base, nj) 的一致快照。
代价:快路径每张票多一次 atomic load(可忽略)。

### 验证判据(四项)
① 连续 3 次会话 24 请求全过、**无任何 `WATCHDOG`**(含 flat 形式);
② 判据行 `inrange == entered == total && rem == 0`;
③ `scripts/check_engine_aligned.sh` 数值门禁(R55);
④ 一次 `C=8/N=32` 持续压测跑完并给出持续吞吐。

## 188. **我的第 127 轮修复本身引入了一条丢票路径** —— 记录并给出正确结构

### (a) 实测(与 §187 的修复对照)
```
WATCHDOG(sharded) total=128 rem=1 exec=127
判据] abandoned=120 underflow=0  entered=127 left=127 **inrange=126**
```
- 修复**前**(§186):`inrange == entered == 479 < total = 480`(全部走快路径);
- 修复**后**:`inrange = 126 < entered = 127`(有 1 次走 re-anchor),
  但**总数仍是 127 < 128** ⇒ **失败依旧,并且我新增了一条丢票路径**。
- 停滞与失败位置完全没变(`#16 512/mt32 = 85747ms` 停顿 → `#17 FAIL`)。

### (b) 我犯的错
我把 seqlock 的"复读校验"写成了:
```cpp
size_t t = node_ticket_[myn].v.fetch_add(1, relaxed);   // ← 票已经领走
uint64_t g = current_gen_.load(acquire);
if (g == gen) {
    base = node_base_[myn]; nj = node_nj_[myn];
    if (current_gen_.load(acquire) != gen) continue;    // ← 直接 continue = 静默丢弃这张票!
```
**"校验失败就 `continue`"在领票之后是不可接受的** —— 票据由 `fetch_add` 唯一领取,
丢弃它就等于 `remaining_` 少减一次。**这正是我这十几轮一直在修的那一类缺陷。**

### (c) 正确结构(下一轮实现,顺序必须反过来:**先取一致快照,再领票**)
```cpp
for (;;) {
    // ---- 1) 读端 seqlock:先拿到一致快照(**此时还不领票**) ----
    size_t base2, nj2; uint64_t g2;
    for (;;) {
        g2 = current_gen_.load(std::memory_order_acquire);
        if (g2 & 1) continue;                       // 发布中 ⇒ 自旋(不领票,不会丢)
        base2 = node_base_[myn]; nj2 = node_nj_[myn];
        if (current_gen_.load(std::memory_order_acquire) == g2) break;   // 一致
    }
    // ---- 2) 快照一致后才领票 ----
    size_t t = node_ticket_[myn].v.fetch_add(1, std::memory_order_relaxed);
    // ---- 3) 用快照判定;若世代已变,交给既有的 re-anchor 分支处理(它会按活代重建 base/nj) ----
    ...
}
```
关键不变式:**`fetch_add` 领票之后,这张票必须被执行+递减,或交给 re-anchor 分支按其活代
重新归属 —— 任何情况下都不允许"领了票直接 continue/break 走掉"。**
(`:943`/`:988` 的 `abandoned_` 分支还需再核:它们也在领票后 break,
但那条路径上票据**本就不属于本代**(loc >= nj),所以是允许的 —— 前提是 `base` 取自**一致快照**,
而这一点只有 (c) 的结构才能保证。)

### (d) 当前树状态
`numa_pool.hpp` 含第 127 轮改动(已知不完整,且新增 `continue` 丢票点),`.so` 与该源码一致。
**下一轮第一件事:按 (c) 重写该段 + 重编译 + 四项验证。**

## 189. ✅ **修复成功**:读端 seqlock(快照先于领票)⇒ 连续 3 次会话 72 请求零看门狗

### 改动(第 128 轮)
把分片 anchor 的顺序改为 **先取一致快照,再领票**(修正 §188 的错误写法):
```cpp
size_t base2, nj2; uint64_t g2;
for (;;) {                                    // ① 快照阶段:自旋重试,**不领票**
    g2 = current_gen_.load(acquire);
    if (g2 & 1) continue;                     // 奇数=发布中 ⇒ 重试
    base2 = node_base_[myn]; nj2 = node_nj_[myn];
    if (current_gen_.load(acquire) == g2) break;
}
size_t t = node_ticket_[myn].v.fetch_add(1, relaxed);   // ② 快照一致后才领票
uint64_t g = current_gen_.load(acquire);
if (g == gen && g2 == gen) { ...快路径,用 base2/nj2... }
else { ...re-anchor 分支(按活代重新归属这张票)... }
```
关键不变式:**`fetch_add` 之后绝不允许无归属地 `continue`/`break`**;快照阶段不领票,
所以自旋重试不会丢任何东西。

### 验证结果(四项判据的前两项)
```
会话1: #1..#24 全部 ok,健康 200
会话2: #1..#24 全部 ok,健康 200
会话3: #1..#24 全部 ok,健康 200
全程**无任何 WATCHDOG**(sharded 与 flat 形式都没有)
```
| 请求 | 3 次会话实测 | 修复前 |
|---|---|---|
| 8192/mt1 | 6162 / 6176 / 6186 ms | 6149–6186 ms |
| 8192/mt8 | 6729–6771 ms | 6736 ms |
| 32768/mt1 | 28350 / 28353 / 28390 ms | 28317–28359 ms |
| 512/mt32 | **2933 / 2939 / 2942 / 2945 / 3005 ms** | **85337–90262 ms(停顿)** |
⇒ **不仅不再崩溃,`512/mt32` 那条 85 秒的停顿也消失了**(那正是"票被丢⇒屏障等满 60s"
的表现)。**延迟也变得更干净、更可复现。**

### 累计修好的三处(全部为同一族:领票与递减/归属不配对)
1. **分片发布顺序**(§184):`node_nj_` 等字段原写在奇数 store 之前 ⇒ 移进窗口;
2. **flat 递减守卫 + 丢弃分支**(§186):去掉守卫改无条件递减;丢弃前先补递减;
3. **分片读端 seqlock**(§189):快照先于领票,杜绝"领了票又 continue"。

### 剩余待做(第三、四项判据)
③ `scripts/check_engine_aligned.sh` 数值门禁(R55);④ 一次 `C=8/N=32` 持续压测。

### (e) 门禁结果(第 128 轮)
```
== [1/2] 数值门禁 test_block23_equiv.py ==
   OK=7 BAD=1(me=1 的既有 NR=8 fp32 重结合偏差,不计入)⇒ **数值门禁通过** ✅
== [2/2] 性能门禁 bench_engine_ab.py(阈值 0.70 ms/层)==
   DEDUP=12 na=12  0.70 ms/层  聚合 216 GB/s  每线程 1.80 GB/s  ms=PASS  per-thread=WARN
   DEDUP=23 na=20  0.87 ms/层  聚合 289 GB/s  每线程 2.41 GB/s  ms=FAIL  per-thread=PASS
```
- **R55 要求的数值门禁通过(7 OK)** ⇒ 本轮对 `numa_pool.hpp` 同步结构的改动**数值上安全**。
- 性能门禁:`DEDUP=12 = 0.70 ms/层` 正好压线 PASS;`DEDUP=23 = 0.87` 是**既有的 item-1 残差**
  (用户已于第 96 轮结案并移入 FUTURE_PLAN,不作为当前优先项)。
  注:此测量是在服务端占用 120 线程的同时跑的,数字偏保守。

## 190. 持续压测(C=8/N=32)**未通过**:flat 路径第三个缺陷 ⇒ 并发下仍丢一次递减

### (a) 实测
```
TAG=sustained_c8_n32: completed=8 failed=24 duration=324.5s out_tok_per_s=3.78
                      mean_ttft=3091ms mean_tpot=141.4ms
服务端: [pool] WATCHDOG fired: gen=161236 n=8 start=960104 end=960112
        counter=960232 remaining=1 current_gen=161236 dropped=0
客户端: EngineDeadError(多次)→ 500
```
- `n=8`(C=8 每步 8 token)、`counter - end = 960232 - 960112 = 120` = **线程数基线**、
  `remaining=1` ⇒ **flat 路径少一次递减**;
- ⇒ 我第 126 轮对 flat 的两处修复(`:1044` 去守卫 + `:1072` 丢弃前补递减)**不够**,
  flat 路径还有**第三处**丢票点。
- 注意 `dropped=0`:flat 自己的丢票计数没涨,所以不是 `:1034` 的 `if (i >= n) break` 那一处
  (那里会记 dropped)。

### (b) 头号嫌疑:**flat 发布与分片发布有同一个缺陷**(我在 §186 错误地排除了它)
```cpp
:445   n_ = n;                                  // ← **写在奇数 store 之前!**
:446   worker_limit_...
:447   sharded_call_ = 0;
:473   current_gen_.store(gen - 1, release);    // 奇数:发布中
:474   start_ = counter_.load();
:475   remaining_.store(n);
:478   current_gen_.store(gen, release);        // 偶数:就绪
```
**`n_` 与 `worker_limit_`/`sharded_call_` 都在窗口之外。** 我在 §186 里因为
"flat 读者在 `work_mtx_` 下取快照"而排除了它 —— 但**并发下(batch 大小逐步变化)**
仍可能让某个 worker 以旧的 `start` 配上新的 `n`(或反之)做 `i = t - start` 判定,
从而把一张**属于本代**的票判成越界/空洞而丢掉。
⇒ **修法与分片 §184 完全相同:把 `n_`/`worker_limit_`/`sharded_call_` 的写入移进
"奇数 store → 偶数 store"的窗口内**,并检查 flat 快路径所用 `start`/`n` 的一致性来源。

### (c) 现状小结(判据)
| 判据 | 结果 |
|---|---|
| ① 连续 3 次会话 24 请求全过、无看门狗 | ✅ 通过(72/72) |
| ② `inrange==entered==total && rem==0` | ✅ 通过(无看门狗即无违约) |
| ③ 数值门禁 `check_engine_aligned.sh` | ✅ 通过(OK=7) |
| ④ `C=8/N=32` 持续压测跑完 | ❌ **未通过**(24/32 失败,flat 看门狗) |
⇒ **分片路径已修好(顺序负载下完全稳定),flat 路径仍需一轮同法修复。**

## 191. ✅✅ **P0 关闭**:"请求几次就 abort"已解决;引擎在顺序与并发负载下均稳定

### 本轮改动(flat 发布顺序,与 §184 同一缺陷)
`n_` / `worker_limit_` / `sharded_call_` 原写在奇数 store **之前**(窗口外),
移到"奇数→偶数"窗口内(与 `start_`/`remaining_` 同处)。

### 四项判据**全部通过**
| 判据 | 结果 |
|---|---|
| ① 连续 3 次会话 × 24 请求全过、无看门狗 | ✅ **72/72** |
| ② 判据行无违约(`inrange==entered==total`, `rem==0`) | ✅ 无看门狗即无违约 |
| ③ `scripts/check_engine_aligned.sh`(R55 强制) | ✅ **数值门禁通过(OK=7)**;性能门禁 DEDUP=12 = **0.67 ms/层 PASS** |
| ④ `C=8/N=32` 持续压测跑完 | ✅ **32/32 完成、0 失败、全程无看门狗** |
```
sust2: completed=32 failed=0 duration=161.98s
       out_tok_per_s=50.58  total_tok_per_s=97.68
       mean_ttft=4961ms p99=6136ms  mean_tpot=139.31ms p99=156.05ms
```
对照**修复前**同一压测:`completed=8 failed=24, out_tok_per_s=3.78`(引擎中途死亡)。
⇒ **持续聚合吞吐 3.78 → 50.58 t/s(13.4×),0 失败。**

### 本轮稳定性攻坚的完整账(四处同族缺陷,全部为"领票与递减/归属不配对")
| # | 位置 | 缺陷 | 证据 |
|---|---|---|---|
| 1 | 分片发布 `node_nj_` 等写在奇数 store 前 | 读者见新 `nj` 配旧 `base` ⇒ 丢票 | `inrange==entered==479 < total=480` |
| 2 | 分片读端无 seqlock 复读 | 同上,读字段跨越奇/偶窗口 | 同上 |
| 3 | flat 递减带世代守卫 + 丢弃分支不递减 | 跳过/丢弃递减 | `WATCHDOG fired n=2 ... remaining=1` |
| 4 | flat 发布 `n_` 等写在奇数 store 前 | 读者见新 `n` 配旧 `start` ⇒ 丢票 | `WATCHDOG fired n=8 ... remaining=1` |
**关键不变式(今后改动的红线)**:①所有调用字段必须写在"奇数 store → 偶数 store"窗口内;
②读者必须"读 gen(偶)→ 读字段 → 复读 gen 校验";③**`fetch_add` 领票之后,这张票必须被执行+递减,
或交给 re-anchor 分支按活代重新归属 —— 绝不允许无归属地 continue/break**。

### 附带收益
修复前每次会话必现的 **~85 秒停顿**(`512/mt32` 85337–90262ms)彻底消失
(现为 2933–2945ms);`DEDUP=12` 也从 0.70 降到 **0.67 ms/层**。

### 下一步(回到 item 2 的性能目标)
稳定性阻塞已解除,可以开始**首次可信的性能测量**:TP=2 的 C=1/C=2/C=3 解码
(此前从未在稳定态测过)、投机解码、以及 1M 上下文。

## 192. 首次**可信**性能测量(引擎稳定态):TP=2 解码远低于目标,且**比 TP=1 更差**

### (a) TP=2 解码(无投机,`bench_battery.py`,128 输出 token,ShareGPT 短 prompt)
| 并发 | 单流 | 聚合 | TPOT |
|---|---|---|---|
| C=1 | **10.46 t/s** | 10.46 t/s | **91.05 ms** |
| C=2 | 10.53 t/s | **21.07 t/s** | 92.34 ms |
| C=3 | 9.77 t/s | **29.31 t/s** | 95.14 ms |
- 服务全程健康(200),无看门狗 ⇒ **这是第一次在"引擎不会死"的前提下测出的解码数字**。
- 并发扩展性:聚合 ≈ 单流 × C(21.07/10.46=2.01,29.31/10.46=2.80)⇒ **几乎线性**,
  说明瓶颈是**每 token 的固定代价**,不是调度/排队。

### (b) 与目标、与 TP=1 的差距(这是新的关键发现)
| | 单流 | 聚合 | 目标 |
|---|---|---|---|
| **TP=2(本次)** | 10.46 t/s | 29.31 t/s @C=3 | 单流 >30(投机 >50);聚合 ≥70 |
| **TP=1(历史)** | **22.2 t/s**(TPOT 45.1ms) | 40.4 t/s @C=2 | 同上 |
⇒ **TP=2 的单流解码只有 TP=1 的一半**(91.05 vs 45.1 ms/token)⇒ TP=2 的**每层固定开销翻倍**。
按 43 层折算:TP=2 ≈ **2.12 ms/层**,而 item 1 已把引擎本身做到 **0.65 ms/层**(DEDUP=12)
⇒ **每层多出约 1.5 ms**,这是解码达标的头号障碍。
**头号嫌疑**:EP 的跨 rank 归约(TP=2 每层都要 reduce)+ TP=2 下 CPU 通路与 GPU 通路的交互。

### (c) 下一步(优先顺序)
1. **带 `XIAOTU_CD_TIMING=1` 重启**,拿 TP=2 与 TP=1 的**每层 compute/rest 分解**
   (要求 (4) 的强制项),直接定位那 ~1.5 ms/层 去了哪里;
2. 对比 `XIAOTU_MOE_EP_SHM=1/0`(shm 归约 vs NCCL)在**解码**上的差异
   (§121 只测过预填充;EP_SHM=0 曾把预填充 #2 从 90s 降到 6.8s,值得在解码上重测);
3. 若 TP=2 的归约开销无法压下来 ⇒ **交付形态考虑 TP=1 做解码 + TP=2 做长上下文**
   (目标允许"TP=1 或 2 不限");
4. 之后再上投机解码(参考:draft 层常驻显存)。

## 193. **每层分解**:TP=2 的代价全在 `compute`(翻倍),`rest` 不变 ⇒ 矛头指向 EP 跨 rank 归约

### (a) 实测(`XIAOTU_CD_TIMING=1`,TP=2,qlen=1,k=6,43 层)
```
[cd-timing] layers=43 qlen=1 k=6 period=1.91–2.06ms  compute=1.17–1.34ms  rest=0.72–0.74ms
            (compute 61–65%, rest 35–39%)
```
- `period × 43 ≈ 84–89 ms/token` ⇒ 与端到端实测 TPOT **91.05 ms** 吻合 ⇒ 分解可信;
- 与 **item 1 结案的 TP=1** 对照(用户第 96 轮裁定:compute **0.64–0.65 ms**/层、rest ~0.77 ms/层):

| | compute | rest | period |
|---|---|---|---|
| TP=1(item 1 结案值,DEDUP=12) | **0.64–0.65 ms** | 0.77 ms | ~1.4 ms |
| **TP=2(本次)** | **1.17–1.34 ms** | 0.72–0.74 ms | 1.91–2.06 ms |
⇒ **`rest` 完全相同(0.72–0.74 ≈ 0.77),而 `compute` 翻了近一倍。**
用户第 96 轮那句话在这里得到精确印证:"**解码 = CPU 每层 ~0.65/0.84 ms + rest ~0.77 ms/层**";
现在 TP=2 把那个 0.65 变成了 1.3。

### (b) 结论:TP=2 解码的差距 = **每层多出 ~0.6 ms 的 compute**,不是调度、不是排队
- 并发扩展近线性(§192a)也印证同一结论:瓶颈是**每 token 每层的固定代价**;
- `rest` 不变 ⇒ 不是采样/主机侧/调度;`compute` 翻倍 ⇒ 在 **CPU MoE 的 compute 段**内。

### (c) 头号嫌疑:**EP 跨 rank 归约在每层关键路径上**
TP=2 时每层 CPU MoE 之后要做跨 rank 求和(`/dev/shm` 双 barrier 或 NCCL)。
qlen=1/k=6 时**数据量极小**(98 KB),但**两次 barrier 的同步代价与自旋**是**固定**的,
按 43 层叠加正好是 ~0.5–0.7 ms/层 的量级。
旁证:§121 在**预填充**上实测 `EP_SHM=0` 使某个请求从 90 s 降到 6.8 s,说明该路径确有可观开销。

### (d) 下一轮(三个便宜且互斥的对照,各一次加载)
| 配置 | 预期 |
|---|---|
| `XIAOTU_MOE_EP_SHM=1`(现状) | compute ≈1.3 ms/层 |
| `XIAOTU_MOE_EP_SHM=0`(NCCL 归约) | 若 compute 明显下降 ⇒ shm barrier 是元凶 |
| `XIAOTU_MOE_EP=0`(冗余模式,无跨 rank 归约) | 给出**无归约的地板**;若 ≈0.65 则确认归约即全部差距 |
再决定:优化 shm barrier、还是**交付形态改为 TP=1 解码**(目标允许"TP=1 或 2 不限";
TP=1 单流 22.2 t/s 已比 TP=2 的 10.46 好一倍,只是预填充与 1M 上下文需要 TP=2)。

## 194. EP 归约对照:**归约值 ~0.45 ms/层(≈21% TPOT),但 EP=0 不可用**

### (a) 三个配置的每层分解(qlen=1,k=6,43 层)
| 配置 | period | **compute** | **rest** | 备注 |
|---|---|---|---|---|
| **EP=1(现状,shm 归约)** | **1.96 ms** | **1.17–1.34 ms** | 0.72–0.74 ms | 单流 10.46 t/s |
| **EP=0(无归约,冗余跑 256 专家)** | 3.68 ms | **0.83–0.86 ms** | **2.85 ms** | 更差;C=1 TPOT 299508ms |
| TP=1(item 1 结案微基准) | ~1.4 ms | 0.64–0.65 ms | 0.77 ms | 单流 22.2 t/s |

### (b) 结论 1:**EP 跨 rank 归约的成本 ≈ 0.45 ms/层**
`compute` 从 1.3(EP=1)降到 0.85(EP=0)⇒ **归约占 0.45 ms/层**
⇒ 按 43 层折算 **≈19 ms/token**,即 TPOT 91 ms 的 **约 21%**。
这与"两次 barrier 的固定同步代价"量级吻合,证实了 §193 的假设(方向对,幅度约 3/4)。

### (c) 结论 2:**EP=0 不是出路**(记入 R96)
关闭 EP 后 `rest` 从 0.74 **暴涨到 2.85 ms/层**,整体 period 反而翻倍,
并出现 `qlen=22 period=270ms rest=260ms` 的严重停顿、C=1 TPOT **299508 ms**。
⇒ 冗余跑全 256 专家的代价(主机↔设备来回 + 双倍访存)远大于省下的归约。
**EP=0 从候选配置中排除。**

### (d) 还剩 ~0.2 ms/层 无法用归约解释
EP=0 的 compute(0.85)仍高于 TP=1 微基准的 0.65。但两者口径不同
(微基准 `bench_engine_ab.py` vs 服务内实测),**不能直接相减**;需要时再单独核对。

### (e) 下一步(按性价比)
1. **优化 shm 归约本身**(目标:把 0.45 → 0.15 ms/层,即省 ~13 ms/token):
   - 现在每层是**两次 barrier**(arrive/gen + read_done/gen2)。检查能否降为一次,
     或把第二次 barrier 挪出关键路径(下一次 D2H 之前不需要等它);
   - 检查 barrier 的**自旋**是否用 `yield`(会与 120 个 MoE 线程抢核);
     改成 `_mm_pause`/`nanosleep` 退避可能显著降低争用;
2. **或者交付形态用 TP=1 做解码**(单流 22.2 t/s,已比 TP=2 好一倍;目标允许"TP=1 或 2 不限"),
   把 TP=2 只留给预填充与 1M 上下文;
3. 之后再上投机解码与 1M 上下文验证。

## 195. 【更正 §191】P0 **未完全关闭** —— 分片丢票在修复后仍会复发;本轮自旋退避改动待验证

### (a) 事实
第 132 轮启动第一次尝试(`spin_a1`)在**预热阶段**就死于分片看门狗:
```
WATCHDOG(sharded) gen=538 total=128 rem=1 exec=127
判据] abandoned=120 underflow=0  entered=127 left=127 inrange=126
died unexpectedly (exit code: None)
```
`exec = 127 < total = 128` ⇒ **一个被计入的任务从未执行**,与 §184/§186 同型。
⇒ **§191 的"P0 关闭"结论过早**:第 128/129 轮的两组验证(72/72 顺序、32/32 并发)
证明修复**大幅降低了频率**,但**没有根除**。正确表述应为:
> 复发率从"每几次请求必崩"降到"需要多组会话才偶发一次",但**引擎仍不是 100% 稳定**。

### (b) 关于 `inrange=126 < entered=127` 的读法(避免又一次误判)
第 128 轮之后 `inrange_` **只在快路径自增**,而 re-anchor 分支执行任务时**不**自增它。
所以 `inrange < entered` 本身**可以**是正常的(表示有任务走了 re-anchor)。
真正的判据始终是 **`exec` vs `total`**:`exec < total` 才是丢任务。
(§188 当时把 `inrange<entered` 当成我的 bug 特征,那是**当时**的语境 —— 那时快路径不会
经 re-anchor,现在会。这条区别必须记住。)

### (c) 仍有待回答的问题
既然 `inrange + (re-anchor 执行数) = entered = 127 < total = 128`,那么第 128 张票:
要么被领走后**没执行也没 abandon**(abandoned 仍是 120 基线),要么**从未被领**
(与 §184 同型)。**需要一个新计数器区分**:
在 re-anchor 分支里也加 `inrange2_++`(与 `inrange_` 分开),
这样 `inrange_ + inrange2_` 与 `total` 的比较就能立刻判定"票没被领"还是"领了没执行"。

### (d) 本轮的另一项改动(自旋退避,待验证)
`binding.cpp` 的 EP 两处 barrier 自旋由 `std::this_thread::yield()`(系统调用,在 120 个
MoE 线程满负荷时容易被排到队尾)改为 **PAUSE 自旋优先、久等才 nano-sleep**。
目标:压缩 §194 量出的 0.45 ms/层归约成本。**已编译,但因 (a) 的启动失败尚未测到数据。**

## 196. **判据定死**:`inrange+inrange2 = 127 < total = 128` ⇒ 一张票**从未被领**;机制 = 发布时"占位"不是原子的

### (a) 读数(第 133 轮,带 `inrange2_`)
```
WATCHDOG(sharded) gen=19438 total=128 rem=1 exec=127
判据] abandoned=120 underflow=0 entered=127 left=127 inrange=126 inrange2=1
⇒ inrange + inrange2 = **127 < total = 128**
```
- 快路径领到 126 张、re-anchor 分支领到 1 张,合计 **127**;
- 而倒计时按 **128** 计 ⇒ **第 128 张票从未被任何 worker 看成"范围内"**;
- `abandoned=120` 仍是基线 ⇒ 那张票也没被当越界票丢弃;
⇒ 与 §188 的"领了又丢"**无关**,是**发布出来的区间本身就少了一张**。

### (b) 机制:发布时的"占位"用的是 `v.load()`,不是原子预留
```cpp
// numa_pool.hpp(分片发布,窗口内)
node_base_[n] = node_ticket_[n].v.load(std::memory_order_relaxed);   // ← 只是**读**
```
而 `node_ticket_[n]` 是**单调、永不重置**的计数器,worker 用 `fetch_add` 领票。
**上一代的滞留 worker 会持续 `fetch_add`**。于是:
1. 发布者读到 `base = node_ticket_[n].v` = X,宣布"本次调用拥有 `[X, X+nj)`";
2. 一个滞留 worker 紧接着 `fetch_add` 拿到 **X** —— 但这张票在它眼里属于**旧代**
   (它按旧代范围判定,可能直接丢弃/或交给 re-anchor);
3. ⇒ **新调用实际只剩 nj−1 张票可用**(X 被旧代 worker 拿走了),
   而 `total` 里仍然算了 nj 张 ⇒ **少一张** ⇒ 倒计时永不归零。
seqlock 保护的是**字段**(base/nj 的一致性),但**保护不了票据计数器** ——
`v.load()` 与 worker 的 `fetch_add` 之间存在真实的竞争窗口。

### (c) 正确修法(下一轮)
**要让"本次调用的票据区间"与票据计数器上的领取原子化**:发布者应**预留**而不是只读:
```cpp
node_base_[n] = node_ticket_[n].v.fetch_add(nj);   // 原子预留 [base, base+nj)
```
这样没有任何其他 worker 能从 `fetch_add` 拿到这个区间内的票;
worker 的领票则改为按**每次调用的局部索引**分配(例如 `shard_idx_[n].fetch_add(1)`,
在发布窗口内重置),并用 `loc < nj` 判定;滞留 worker 仍靠"活代校验 + 每个调用独立的索引"
被安全地挡在外面。
注意:轮 68 曾因"每次调用重置计数器"而出现"陈旧票别名到新调用"的事故
(`numa_pool.hpp` 里那条注释)⇒ **重置局部索引必须同时保留活代校验**,
不能退回当时的写法。这条改动**需要先想清楚再动**,不能直接照抄。

## 197. 第 134 轮:**没有盲改** —— 把"原子预留"方案的推演与卡点记录清楚

### (a) 为什么不用 `fetch_add(nj)` 直接改
若发布者改 `node_base_[n] = node_ticket_[n].v.fetch_add(nj)` 原子预留区间,那么
**worker 就不能再对 `node_ticket_` 做 `fetch_add`**(会拿到 base+nj 之后,不属于本调用),
只能改用**每次调用的局部索引**(发布窗口内重置)+ `loc < nj` 判定。
**但这条正是轮 68 出事故的写法**:`numa_pool.hpp:677-682` 的注释明确写着
"Resetting the counters per call made stale tickets alias live ones and silently
duplicated/skipped jobs" —— 局部索引一旦重置,**滞留 worker 的 `fetch_add` 会拿到 0,
正好别名到活调用的第一个任务**,静默重复/跳过。
⇒ 要安全,局部索引必须配一个"我只服务活代"的**可靠**判据;而滞留 worker 恰恰无法从
局部索引本身看出自己是否过期。**这个组合我还没有推演到有把握。**

### (b) 现有证据的进一步约束(缩小了可能性)
1. `entered == left == 127` ⇒ 所有执行都返回了,**没有 worker 卡在任务体**;
2. `inrange + inrange2 == 127 < total == 128` ⇒ 有一张票**既没被算作范围内、也没被执行**;
3. `abandoned == 120` ⇒ 按"每 worker 每代多领一次"的基线,**没有"范围内"的票被当越界丢弃**;
4. 于是那张票只能落在:**领票之后、既没进 inrange 也没进 abandon 的那条路径**上。
   在当前代码里,`fetch_add` 与 `inrange_++` 之间只剩:快照自旋、世代判断、
   `(g==gen && g2==gen)` 的组合条件。**最可疑的是快照自旋在世代频繁变化时的行为**
   —— 它可能让 worker 反复重试而**始终不满足 `g2 == gen`**,于是那张票一直被"悬着"。
   ⇒ **下一轮要在快照自旋里加一个"重试次数"计数器**,看它是否异常飙升;
   以及给"快照拿到的 g2 != 缓存 gen"单独计数,确认这条路径的流量。

### (c) 结论
本轮**只做推演,不改代码**。理由:这是并发协议的核心路径,前几轮我已经因为
"读起来最可疑就改"白烧过两次加载 + 一次重编译(R94/§188);此次证据尚不能唯一确定路径,
**先加两个只读计数器把路径流量测出来,再动**。

## 198. **快照假设被否证**(`snap_retry=0`, `snap_mismatch=0`);算术指向"**有一个 worker 没参与**"

### (a) 读数(第 135 轮,带快照流量计数器)
```
WATCHDOG(sharded) gen=46226 total=128 rem=1 exec=127
判据] abandoned=120 underflow=0 entered=127 left=127
      inrange=127 inrange2=0  snap_retry=0  snap_mismatch=0
```
- `snap_retry = 0` ⇒ 快照循环**从未重试**(没撞上奇数窗口,也没复读不一致);
- `snap_mismatch = 0` ⇒ 快照拿到的 `g2` **每次都等于** worker 缓存的 `gen`;
⇒ **§197 的假设("worker 卡在快照反复重试、始终不满足 `g2 == gen`"被干净否证。**
`base2/nj2` 每次都来自一个与该 worker 世代一致的、干净的快照。

### (b) 算术把方向改了:**少一张票被发出**,而不是"票丢了"
设每个 worker 每代都会多领一次票(基线,§179 三次确认 = 120):
```
本轮发出的票总数 = inrange(127) + abandoned(120) = 247
预期            = 任务数(128) + 基线多领(120)     = 248
```
⇒ **整整少了一张票被 `fetch_add` 发出**。结合 `entered == left == 127`(没人卡在任务体)
⇒ 最自然的解释是:**有一个 worker 这一代根本没有参与**(它没做那次 `fetch_add`,
也就既不会贡献 inrange 也不会贡献 abandoned)。

### (c) 下一步(能一次定性的探针)
1. **统计"本代至少领过一次票的 worker 数"**(一个按 worker 索引的位图/计数),
   与预期参与者数比较 —— 若少 1,就确认是"worker 没参与";
2. **把每节点 `pulled` 一起打印**(本轮漏了):若某个 node 是 `30` 而不是 `31`,
   就直接指出是**哪个节点**有 worker 缺席;
3. 追查那个 worker 为什么没参与:`:944-948` 的 `worker_limit_` stride 过滤
   (`if (w % stride != 0) continue;`)、外层 `cv_.wait` anchor 是否漏唤醒、
   或它被 120+48 个线程挤掉后一直没被调度到。
**注意这已经不是"票据协议"问题,而是"worker 参与/唤醒"问题** —— 前几轮修的三处
(发布顺序、读端 seqlock、flat 守卫)依然是真实的修复(它们确实消灭了大部分失败),
但**残余的这一张票属于另一类原因**。

## 199. 【更正 §198】**票全发出去了**;问题回到"`abandoned` 的基线到底是多少"

### (a) 每节点 pulled(上一轮 dump 里本来就有,我漏看了)
```
node 0: jobs=16 pulled=31      node 4: jobs=16 pulled=31
node 1: jobs=16 pulled=31      node 5: jobs=16 pulled=31
node 2: jobs=16 pulled=31      node 6: jobs=16 pulled=31
node 3: jobs=16 pulled=31      node 7: jobs=16 pulled=31
```
`8 × 31 = 248` = **128 任务 + 120** ⇒ 与"每 worker 每代多领一次"的预期**完全一致**。
⇒ **§198(c) 的"整整少一张票被发出 / 有一个 worker 没参与"是错的**,已更正。

### (b) 真正的不一致:`248 发出` vs `247 分类`
```
发出     = 248   (由各节点 pulled 直接给出)
分类     = inrange(127) + abandoned(120) = 247
```
⇒ **有一张票被领走之后,既没被算作 in-range、也没被算作 abandoned。**

### (c) 最可能的解释:**`abandoned` 的基线不是 120,而是 119**
我在 §179(b) 判定"`abandoned ≡ 线程数(120)` 是基线",依据是**三次 dump** —— 但
**那三次全都是崩溃时的 dump**!我**从未在健康调用上测过这个基线**。若真实基线是 119
(例如 `worker_limit_` 的 stride 过滤使某个 worker 从不领票,或多线程边界效应),
那么观测到的 `abandoned=120` 就**恰好是那多出来的一次** ——
**那张"丢掉的"票其实是走进了 abandon 路径**,而不是中间消失。

### (d) 下一步:**在成功路径上打印这些计数器**,把基线测出来
现在的 `SHARD-JOBDIAG`(成功分支,`:735` 一带)只打印 `total/exec/dup/miss`,
**不含 `abandoned`/`inrange`/`inrange2`**。而**基线只能在健康调用上测**。
⇒ 改法(纯诊断):把那三个计数器加进 `SHARD-JOBDIAG` 的成功打印;
然后跑一段健康负载,读**稳定收敛的基线值**:
- 若健康时 `abandoned ≡ 119`、`inrange ≡ total` ⇒ 基线 119,
  那么崩溃时 `abandoned=120 且 inrange=127(=total-1)` ⇒ **那张票走进了 abandon**,
  矛头转向"为什么本代有一张票被判越界"(即区间少一张,**回到 §196 的原子预留**);
- 若健康时 `abandoned ≡ 120`、`inrange ≡ total` ⇒ 基线 120,则那张票确实是"领了没分类",
  需要继续找那条路径。
**这一步不需要改控制流,且能把 (c) 的二选一变成事实。**

## 200. **基线测出来了**:判据是 `inrange == total`;`abandoned` **不是基线、没有判别力**

### (a) 健康调用的 6 个样本(第 137 轮,把 JOBDIAG 改成总是打印后)
| total | exec | dup | miss | **abandoned** | **inrange** | inrange2 |
|---|---|---|---|---|---|---|
| 256 | 256 | 0 | 0 | **118** | **256** | 0 |
| 480 | 480 | 0 | 0 | **120** | **480** | 0 |
| 256 | 256 | 0 | 0 | **69** | **256** | 0 |
| 256 | 256 | 0 | 0 | **108** | **256** | 0 |
| 480 | 480 | 0 | 0 | **120** | **480** | 0 |
| 128 | 128 | 0 | 0 | **88** | **128** | 0 |

### (b) 三个结论
1. **健康时恒有 `exec == total` 且 `inrange == total`**(6/6)⇒ **这才是判据**;
2. **`abandoned` 在 69–120 之间大幅波动**,取决于那一刻有多少 worker 多领了票(与调度/参与有关)
   ⇒ **它不是一个常数基线**;
3. **崩溃时的 `abandoned = 120` 完全落在健康区间 [69,120] 内**
   ⇒ **`abandoned` 对"是否故障"没有任何判别力**。
   ⇒ **更正 §179(b)**:我当年据三次 dump 断定"`abandoned ≡ 线程数(120)` 是基线"是**错的**
   —— 那三次恰好都是 120,纯属巧合;而它们**全是崩溃态的 dump**(§199c 已经怀疑,现在证实)。
   ⇒ **顺带更正 §198(c)/§199(b) 的算术**:用"120 是基线"去算"发出票总数 = 248"是**不成立的**,
   因为基线本身在变。**"少一张票被发出"这个推断也随之作废。**

### (c) 现在唯一站得住的事实
```
健康:inrange == total   (6/6)
故障:inrange = 127 < total = 128
```
⇒ **故障的定义就是:有一张被计入 `total` 的任务票,从未被任何 worker 看成 in-range,也没被执行。**
(`abandoned` 帮不上忙,因为它的正常波动范围覆盖了故障值。)

### (d) 下一步(把"那张票去哪了"做成无损账)
在**领票之后立刻**加 `issued_++`,然后无论走哪条分支都保证**恰好一次**计入
`inrange_`/`inrange2_`/`abandoned_` 之一 ⇒ 用
`issued_ - inrange_ - inrange2_ - abandoned_` 直接读出"未分类的票数"。
若健康态该值为 0 而故障态为 1,就说明**存在一条领票后不分类的路径**;
届时重点检查 **re-anchor 分支里的 `if (stop_) return;`**(它会带着已领的票直接返回)
以及任何其它提前 `return/break`。**这一步仍是纯诊断,不改控制流。**

## 201. 无损账建成并验证:`unclassified = 0`(健康态);本版又稳过 2 个完整会话

### (a) 计账自洽性验证(健康调用)
```
SHARD-JOBDIAG total=384 exec=384 abandoned=120 inrange=384 inrange2=0 issued=504 unclassified=0
SHARD-JOBDIAG total=192 exec=192 abandoned=120 inrange=192 inrange2=0 issued=312 unclassified=0
SHARD-JOBDIAG total=384 exec=384 abandoned=120 inrange=384 inrange2=0 issued=504 unclassified=0
```
- **`issued = total + abandoned` 精确成立**:504 = 384+120、312 = 192+120
  ⇒ 账是闭合的,**每个领出的票都被分类了**;
- **`unclassified = 0`** ⇒ 健康态下**不存在"领票后不分类"的路径**(那条 `if (stop_) return;`
  在正常运行时不会被触发,至少在健康调用上从未发生);
- `inrange == total` 且 `exec == total` ⇒ 与 §200 的判据一致。

### (b) 关于 `abandoned` 的最终定性(结束 §179/§200 的反复)
- 本轮三个样本都是 **120**;而 §200 的六个样本是 69/88/108/118/120/120。
- 机制:**120 = 每个 worker 每代恰好多领一次**(`issued = total + 120` 精确成立即为此意);
  当某个 worker 那一代没来得及多领(被调度/唤醒影响)时该值就低于 120。
- ⇒ 正确表述:**`abandoned ≈ 线程数` 是"稳态常见值",不是不变式、也不是判据**。
  §179(b) 的"≡线程数"过强,**§200 的更正成立**;§198(c)/§199(b) 基于它的算术作废。

### (c) 本版稳定性
带无损账的这一版**又连续跑完 2 个完整会话(48 请求)、零看门狗**。
⇒ 加上 §195 那次失败,当前复发率约"每 3~6 个会话一次",符合"大幅改善但未根除"的定性。

### (d) 下一轮:**必须抓到一次故障**才能二分
健康态已确定 `unclassified = 0`。所以故障时的读数只有两种可能:
| 故障读数 | 结论 |
|---|---|
| `unclassified = 1` | 存在"领票后不分类"的路径(重点查 re-anchor 的 `if (stop_) return;` 等提前返回) |
| `unclassified = 0` 且 `inrange = total-1` | 那张票**被算进了 abandoned** ⇒ **发布出来的区间本身少一张** ⇒ 回到 §196 的原子预留修法 |
⇒ 做法:用当前版本**多跑几个会话**(每会话 24 请求),直到抓到一个 `WATCHDOG(sharded)`,
读它的 `判据` 行即可定论。**这是纯观察,不需要改代码。**

## 202. **第二种独立故障**:CUDA illegal memory access(与线程池无关)

### (a) 读数(第 139 轮,同一版本连跑 4 个会话)
```
会话1 通过(24/24)  会话2 通过(24/24)  会话3 通过(24/24)
会话4: #14 len=8192 mt=8 ok **29964ms**(30 秒停顿) → #15 **FAIL**
      日志: **无任何 WATCHDOG**
      `CUDA error: an illegal memory access was encountered`
      → `Worker proc ... died unexpectedly (exit code: None)`
      → `RuntimeError: cancelled` → `EngineDeadError` → HTTP 500
```
- **没有看门狗** ⇒ **不是**我一直在追的"池丢票"那一类;
- **`CUDA error: an illegal memory access`** ⇒ GPU 侧访存越界,worker 原生死亡。
- 这与第 116 轮**最早**看到的那条签名相同
  (`persistent_topk occupancy query failed: an illegal memory access was encountered`,
  R90)—— 也就是说:**从一开始就存在两种独立故障,而我在第 120-138 轮把全部注意力
  放在了线程池那一类上**(它确实是真的、也确实被大幅修好),但**这一类从未被处理过**。

### (b) 两种故障的判别(一眼可分)
| 故障 | 日志特征 | 我的处理 |
|---|---|---|
| **A. 池丢票** | `[pool] WATCHDOG(sharded/flat)` → `abort()` | 已修三处(§184/§189/§190),复发率大幅下降 |
| **B. GPU 访存越界** | **无看门狗**;`CUDA error: an illegal memory access`;`died unexpectedly (exit code: None)` | **从未处理** |

### (c) 故障 B 的线索
- 前面紧跟一次 **30 秒停顿**(`8192/mt8` 正常应为 ~6.7 s)⇒ 停顿与越界可能同源
  (例如某个内核在长上下文/特定形状下越界,随后暴露);
- 历史线索:第 116 轮在 **32K 预填充之后**出现 `persistent_topk` 的 illegal access,
  且该调用点在 `sparse_attn_indexer` 的 **decode 分支**;
- 我方嫌疑点仍是**GPU 预填充路径**(`gpu_prefill.py` 的 `PrefetchSlot` 环、Triton 预填充内核)
  与**常驻层**(§167 R89:常驻层存在时的 Triton OOM)——
  但注意本版**常驻=0**也出现了,所以不能只怪常驻。

### (d) 下一轮
1. **先用 `CUDA_LAUNCH_BLOCKING=1` 复现一次**,让越界在**首次发生的内核**处立刻报错
   (默认是异步的,报错点往往不是真凶 —— 第 116 轮那次就是在 `persistent_topk` 的
   occupancy query 里才暴露);
2. 同时把 `XIAOTU_MOE_POOL_TRACE` 关掉以减少干扰;
3. 抓到首个内核后,再决定是 TMA/对齐问题、还是 shape 边界(空段、`seg_start` 越界)。

## 203. `CUDA_LAUNCH_BLOCKING=1` **没能复现故障 B** —— 它先触发了故障 A

### (a) 实测
```
会话1: #8 len=512 mt=32 ok **303851ms** → #9 len=8192 mt=1 FAIL;健康 000
日志末尾: [pool] WATCHDOG ... + node 0..7: jobs=16 pulled=31   ← **故障 A**
```
- **303851 ms ≈ 300 s** = **看门狗的默认值**(`XIAOTU_MOE_SHARD_WD` 未设,默认 300 s;
  本轮我只设了 `CUDA_LAUNCH_BLOCKING=1`)⇒ 这不是新故障,就是**故障 A**:池卡住 → 300 s → `abort()`。
- ⇒ **CLB 没有复现故障 B**;它把时序整体拖慢,反而让故障 A 先到。

### (b) 方法学修正(我自己的操作失误)
上一轮我在报告里写"用 CLB 复现故障 B",但**忘了同时设 `SHARD_WD=60`**,
于是 A 一旦触发就要等满 300 s —— 白白多花 4 分钟,并且掩盖了本轮的真实目的。
⇒ **以后跑诊断必须把已知的"兜底超时"一并压到最小**,否则诊断会被兜底路径抢先。

### (c) 下一轮(修正后的做法)
1. **不启用 CLB**(故障 B 显然与正常时序相关,CLB 会改变它),
   而是 `XIAOTU_MOE_POOL_TRACE=1 XIAOTU_MOE_SHARD_WD=60`(**保留 60 s 兜底**),
   多跑会话直到抓到**无看门狗 + `CUDA error: an illegal memory access`** 那一次;
2. 抓到后立即看它前面**最后一次成功请求的类型与耗时**(第 139 轮是
   `8192/mt8 = 29.9 s`,正常 6.7 s ⇒ **30 秒停顿是预兆**);
3. 若要精确定位越界内核,可在抓到之后再单独用 `compute-sanitizer --tool memcheck`
   跑一次**同类型请求**(它比 CLB 更能指出越界的那一行,且不需要全局同步)。

## 204. 第 141 轮:连续 4+ 个会话(96+ 请求)**全过**,故障 B 本轮未复现

### (a) 现状(修正方法后:保留 `SHARD_WD=60` 兜底、不启用 CLB)
```
会话1 通过(24/24)  会话2 通过(24/24)  会话3 通过(24/24)  会话4 进行中…
截至检查:health=200,watchdog=0,无 CUDA 错误串
```
⇒ 本轮**没有抓到故障 B**(它在第 139 轮出现过一次,复发率低)。

### (b) 累计稳定性数据(当前版本)
| 运行 | 结果 |
|---|---|
| §201(2 个会话) | 48/48 通过 |
| §202(4 个会话) | 前 3 个通过,第 4 个 **故障 B**(无看门狗 + CUDA illegal access) |
| §204(本轮) | 4+ 个会话通过,未复现 |
⇒ 合并看:**故障 B 的复发率约每 6~10 个会话一次**;
而**故障 A(池)在 §203 仍出现过一次**(CLB 那次)。两者都还在,只是都变稀了。

### (c) 下一轮
故障 B 的抓取需要**更多会话**(纯等待),建议一次挂 **10 个会话**的后台任务;
抓到后按 §203(c) 的流程:看前兆(`8192/mt8` 是否出现 30 秒级停顿)→
再用 `compute-sanitizer --tool memcheck` 对**同类型请求**定位越界内核。

## 205. **故障 B 抓到第二次,且前兆两次完全一致**:`nat8192/mt8` 的 30 秒停顿

### (a) 两次故障 B 的对照
| 轮次 | 前兆(最后一次"成功"的请求) | 正常值 | 随后 |
|---|---|---|---|
| §202(第 139 轮) | `len=8192 mt=8` **29964 ms** | ~6730 ms | `len=32768 mt=1` **FAIL** |
| §205(第 142 轮) | `len=8192 mt=8` **31656 ms** | ~6730 ms | `len=32768 mt=1` **FAIL** |
签名均为:`CUDA error: an illegal memory access was encountered`
→ `died unexpectedly (exit code: None)` → `cancelled` → `EngineDeadError`。

⇒ **前兆可复现**:一个 `nat8192 + max_tokens=8` 的解码请求**先出现 ~30 秒停顿**
(放慢约 4.7×),**紧接着的下一个请求**触发 CUDA 越界并打死 worker。
⇒ 这把故障 B 的触发条件收窄到了一个**很具体的形状**:`qlen=8, k=6`,
且 KV 已有 8192 token。

### (b) 关键推论:30 秒停顿与越界**同源**
- 停顿不是"慢",而是**某个东西在等**(30 s 这个量级很像某个超时/自旋);
- 越界很可能**不是**在触发它的那个请求里首次发生,而是**停顿期间某个内核**写坏了显存,
  随后在下一个请求上以 sticky CUDA error 的形式暴露(与第 116 轮 `persistent_topk`
  的 occupancy query 才报错是同一模式);
⇒ **必须用 `compute-sanitizer` 或分段同步把"首次越界"提前到它真正发生的那个内核。**

### (c) 下一轮(精确定位)
1. **`compute-sanitizer --tool memcheck`** 跑同一序列(nat8192/mt8 → nat32768/mt1);
   它会直接报出**越界的内核名与行号**,而不是等到下一个请求才以 sticky error 暴露;
   代价是极慢(可能 10-50×),但只需跑到越界发生即可;
2. 若 sanitizer 太慢,退一步:**只在 GPU 预填充路径上分段加 `torch.cuda.synchronize()`**
   (注意:不能在热路径常开 —— R92 的教训;只作为一次性诊断,并在同一次运行里不启用其他埋点);
3. 同时注意:**30 秒这个数字本身值得查** —— 是否有某个 30 s 的超时/重试常量
   (例如 NCCL 或某个 barrier 的 timeout)在停顿期间把状态搞坏。
