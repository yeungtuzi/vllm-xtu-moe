# lvllm + lk-moe  vs  主线 vLLM + xiaotu-moe —— 等效参数同机 A/B

日期:2026-09-18 · 脚本:`scripts/ab_lvllm_matrix.sh` · 原始数据:`report/tuning/raw/abmx_{a,b}_p*_o*_c*.json`
完整分析:`report/tuning/NOTES.md` §622

## 为什么这个对比可归因

| 项 | 设置 |
|---|---|
| **运行环境** | **两个 arm 都在 `conda env lvllm`(vLLM 2.5.0 基座)里** ⇒ 基座/模型加载/GPU kernel 全同,**唯一变量 = CPU MoE 引擎** |
| arm A | `lvllm + lk-moe`(`LVLLM_MOE_NUMA_ENABLED=1`,lk_moe 2.4.2) |
| arm B | 同一基座 + **我们的** `xiaotu_moe` CPU 后端,由受 `XTU_PLUGIN=1` 门控的 `.pth` 装载 |
| 纯度断言 | arm A 日志**不得**出现 `vllm-xtu-moe`;arm B 必须出现,且日志有 shim `routed_experts.lk_moe -> xiaotu_moe(LvLLM fork)` ✅ 两边都通过 |
| 参数 | 照抄 lvllm 自己的 `commands/dsv41_serve_tp2_3090_dspark.sh`:`TP=2 / MAXLEN=65536 / MBT=8192 / SEQS=2 / GPU_UTIL=0.95 / bfloat16 / kv-cache-dtype=fp8_ds_mla / tokenizer-mode=deepseek_v4 / --compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY","mode":"VLLM_COMPILE"} / prefix-caching / chunked-prefill / enable_thinking=false` |
| 唯一刻意的对齐 | **CPU 线程数两边同为 60**(参考机 96c 用 48;本机 192c,若各用各的最佳值就无法归因) |
| GPU 预填充 | **两边都关** ⇒ 量的是纯 CPU MoE 引擎 |
| 客户端 | 官方 `vllm bench serve`;`--dataset-name random` 固定 token 长度 ⇒ **两个 arm 拿到逐字节相同的 prompt**;`--ignore-eos` 强制输出长度 |

## 结果(C=1,N=4,每格独立 seed 避免前缀缓存命中)

| prompt | output | A `output_throughput` | B `output_throughput` | **B/A** | A TTFT | B TTFT | **B/A** | A TPOT | B TPOT | **B/A** |
|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 32 | 11.19 | 7.65 | **0.68×** | 2153 ms | 3022 ms | **0.71×** | 22.80 ms | 37.44 ms | **0.61×** |
| 256 | 1024 | 40.08 | 27.62 | **0.69×** | 2163 ms | 3044 ms | **0.71×** | 22.86 ms | 33.26 ms | **0.69×** |
| 8192 | 32 | 0.49 | 0.37 | **0.76×** | 64178 ms | 86117 ms | **0.75×** | 22.56 ms | 31.51 ms | **0.70×** |
| 8192 | 1024 | 11.70 | 8.59 | **0.73×** | 64162 ms | 86355 ms | **0.74×** | 22.82 ms | 32.14 ms | **0.70×** |

### 换算成速率

| | prefill(256 token) | prefill(8192 token) | 解码 `1/TPOT`(256) | 解码 `1/TPOT`(8192) |
|---|---|---|---|---|
| **A = lvllm + lk-moe** | **119 tok/s** | **128 tok/s** | **43.9 tok/s** | **44.3 tok/s** |
| **B = 主线 + xiaotu-moe** | **85 tok/s** | **95 tok/s** | **26.7-30.1 tok/s** | **31.7 tok/s** |

* 两边 prefill 都近似**线性**(固定成本小)。斜率 = 每 token 引擎开销:
  **A ≈ 7.8 ms/token,B ≈ 10.5 ms/token**。
* **A 的 TPOT 与上下文无关**(256 vs 8192:22.80 vs 22.56 ms)。B 在这个配置下也基本无关
  (37.4 vs 31.5 ms)。⇒ NOTES §621c 里"B 的 TPOT 随上下文涨到 60 ms"是**另一套服务配置**
  (MAXLEN=1M / `COMPILE=0` / 前缀缓存关)的现象,不是这个 fork 配置下的现象。

## 引擎级同形状对比(把差距切成"引擎"和"引擎之外")

工具 `scripts/bench_engine_ab.py`:直接加载两个 `.so`、同一份**真实 V4.1 第 3 层权重**、
同线程数 60、纯 CPU、无 GPU/无调度器。用 `LK_PY_PATH` 指定**服务级 A/B 里那一个** lk_moe
二进制(机器上有两个同名同版本 2.4.2 但 md5 不同的安装)。

| 形状 | xiaotu ms/层 | lk ms/层 | **我们慢** | xiaotu tok/s | lk tok/s |
|---|---|---|---|---|---|
| **BS=1, DEDUP=6**(解码) | 0.59 | 0.44 | **1.34×** | 42.4 | 57.4 |
| BS=227, DEDUP=23 | 50.24 | 29.59 | **1.70×** | 113.0 | 191.8 |
| BS=1893, DEDUP=23 | 408.00 | 229.19 | **1.78×** | 116.0 | 206.5 |

**切分(解码侧,p=8192 那格)**:TPOT 31.51 ms = 引擎 23.6 ms + 非引擎 7.9 ms;
lk 22.56 ms = 引擎 17.4 ms + 非引擎 5.2 ms
⇒ **引擎 1.34×,非引擎 1.52×**。

⚠️ **不能用 DEDUP=23 的微基准外推真实 8192-token 预填充**(真实 na≈384,形状不同),
所以预填充侧只能说"引擎级 1.70-1.78× vs 服务级 1.34-1.40×,他们的集成开销更大"。

## 结论(诚实记录)

1. **在等效参数下,lk-moe 的 prefill 快约 1.34-1.40×,解码快约 1.4-1.6×。**
   此前"16.64 tok/s vs 参考 25-27 t/s"**不能**用"口径不同"完全解释掉 —— 同口径下我们确实更慢。
2. 差距在 prefill 与解码**两个维度同时存在**,所以不是单一瓶颈;更像**每 token 的引擎固定开销**差异。
3. 下一步(已在跑):把"慢"拆成"哪个默认值造成的":
   * `SPIN_IDLE_US`:§354 记录的 **fork 最佳是 0**,而本基线用的是 **300** ⇒ 最可疑;
   * `THREADS`:参考机是"物理核 ÷ GPU 数" = 48,本机同规则是 **96**,而基线把两边都压到 60;
   * `NSLICE_SMALL`:出货默认 0(§354 说它在 fork 上曾造成 1225 ms/token 的漂移)。

## 复现

```bash
# arm A(lk-moe):先起服务,再 SKIP_LAUNCH 复用
bash scripts/ab_lvllm_matrix.sh          # 两个 arm 顺序跑(每个 arm 只起一次服务)
GPU_PREFILL=1 bash scripts/ab_lvllm_matrix.sh   # 各 arm 带 GPU 预填充
B_SPIN=0 B_THREADS=96 bash scripts/ab_lvllm_matrix.sh   # arm B 旋钮变体
```
