# DeepSeek-V4.1-Flash 分析报告

> 面向本项目(vllm-xtu-moe:消费级/受限显存下的 MoE CPU 专家后端)的架构、资源与可行性分析。
> 数据来源:`config.json`、官方 51 页技术报告 `DeepSeek_V41_Tech_Report.pdf`、
> safetensors 索引(96,085 个张量)+ 分片头部逐张量形状(zero-copy 探测,未下载 475 GiB 权重)。
> 报告日期:2026-09-10(发布当天)。

---

## 0. 结论先行

| 问题 | 结论 |
|---|---|
| 模型多大? | **552B backbone + 196B Engram = 748B 参数,磁盘 475.2 GiB** |
| 权重怎么分布? | **路由专家 269 GiB(56.6%)+ Engram 表 183 GiB(38.5%)+ 其它 23 GiB(4.9%)** |
| 我们机器(3×A100-40G + 1.5 TiB RAM)装得下吗? | **装得下,而且比 V4-Flash-0731 更适合我们的架构** —— 只需把"专家 + Engram 表"放主机内存 |
| KV 还是瓶颈吗? | **完全不是**:全局 KV 890 B/token,256K 上下文只需 0.23 GB(0731 在我们这套 vLLM 上实测 29.5 KB/token) |
| 能直接跑起来吗? | **不能**。需要 vLLM 主线先支持 `deepseek_v41`(CED + CSA2 跨层复用 + Engram + 层级索引器 + FP4 KV);我们当前主线(2026-09-08)只有其中一部分 |
| 我们该做什么? | 做我们最擅长、且已被官方设计"背书"的两件事:**Engram 表放主机内存** + **路由专家放 CPU**;其余等主线 |

**一句话**:V4.1 是本项目成立以来遇到的最"对我们路子"的模型——它把 38.5% 的权重量做成
**纯查找表**,并且官方生产栈自己就把它放在**主机内存**里用 RDMA 预取;剩下 56.6% 是
CPU 专家后端的主场。真正的障碍不是内存容量,而是**整条模型的实现依赖主线 vLLM**。

---

## 1. 基本参数(与技术报告一致)

| 项 | 数值 |
|---|---|
| 架构 | `DeepseekV41ForCausalLM` / `model_type: deepseek_v41`,多模态 MoE Transformer |
| 层数 | **40 层 = 20 层 causal encoder + 20 层 decoder**(CED) |
| hidden / MoE inter | 5120 / 2304 |
| 专家 | **1 shared + 384 routed,top-6**,`sqrtsoftplus` + `noaux_tc`,scaling 1.5,swiglu_limit 10 |
| 注意力 | head_dim 512,q_lora 1280,o_lora 1024 / o_groups 8,qk_rope 64,SWA 窗口 128 |
| 稀疏索引器 | 32 heads × 128 dim,attn top-k **512**;层级索引器候选池 2048 块 × 8 = 16,384 位置 |
| mHC | `hc_mult=4`,`hc_sinkhorn_iters=20`(Single-Pass mHC) |
| Engram | 2 个模块(层 1、14),N-gram 阶 {2,3,4} × 8 hash heads,压缩词表 99,092 |
| DSpark | 3 个 Transformer 块(SWA 128),128 专家 top-3,Markov rank 256,块大小 5 |
| 视觉 | 32 层,hidden 1024,patch 14,downsample 3,16 heads,单图 ≤1024 token |
| 上下文 | 1,048,576(YaRN factor 16,原 65,536) |
| 量化 | dense/shared fp8(block 32×32,e4m3+ue8m0)、**专家 fp4(e2m1,groupK=32,ue8m0)**、**KV fp4** |
| 激活 | **prefill 8B/token,decode 16B/token** |

### 1.1 参数账:552B / 196B / 748B 是怎么来的(逐项可核验)

网上常见说法"排除 n-gram 只有 300 多 B"**是把存储字节当成了参数量**。用实测张量形状算:

| 组成 | 计算 | 参数量 | 存储 |
|---|---|---|---|
| 单个路由专家 | w1 2304×5120 + w3 2304×5120 + w2 5120×2304 | **35.39 M** | 17.93 MiB(含 scale) |
| 路由专家合计 | 35.39M × 384 专家 × 40 层 | **543.6 B** | 253.1 GiB(fp4)+ ~16 GiB 缩放 |
| 共享专家 | 35.39M × 1/层 × 40 层 | 1.42 B | 0.7 GiB |
| Engram 表 | 384,006,168×256 + 384,016,682×256 | **196.6 B** | **183.1 GiB(fp8,1 B/参数)** |
| 注意力 + mHC + embed/lm_head + 视觉 | — | ~9 B | ~17 GiB(fp8/bf16) |
| **合计** | | **~748 B** | **475.2 GiB** |

- **backbone = 543.6 + 1.4 + 9 ≈ 554B ≈ 论文的 "552B backbone"** ✅(误差来自我对注意力/mHC 的粗算)
- **+ Engram 196B = 748B 总参数**,与技术报告 "552B backbone + 196B Engram" 完全一致。
- 两个独立交叉验证:
  - **decode 激活** = 40 层×6 专家×35.39M(8.5B)+ 共享 1.4B + 注意力/mHC ~6.4B ≈ **16.3B** ≈ 论文 16B ✅
  - **prefill 激活** = 20 层(只走 encoder)×6 专家(4.2B)+ 共享 0.7B + 注意力/mHC ~3.2B ≈ **8.1B** ≈ 论文 8B ✅
    ——同时反证了 CED 的语义:prefill 只跑 20 层 encoder。

**"300 多 B" 的三个可能来源**

| 说法 | 实际 | 错在哪 |
|---|---|---|
| "排除 n-gram 后 300 多 B" | 排除 Engram 后是 **552B**(存储 286.4 GiB = **308 GB**) | 把 **GB 存储**读成了 **B 参数**;专家是 fp4,1 字节装 2 个参数 |
| "数了一下约 270B" | 那是 fp4 打包后的 **int8 元素数**(271.8B) | 未把每字节 2 个 fp4 还原回逻辑参数 |
| "552 − 196 = 356B" | **552B 是 backbone 而不是总量** | 专家单层就有 13.6B 参数,40 层 543.6B > 356B,自相矛盾 |

核验命令(不需要下载权重,只读分片头):

```bash
curl -r 0-65535 https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main/model-00009-of-00048.safetensors \
 | python3 -c "import sys,json,struct;n=struct.unpack('<Q',sys.stdin.buffer.read(8))[0];h=json.loads(sys.stdin.buffer.read(n));\
print({k:v['shape'] for k,v in h.items() if 'experts.0.' in k})"
# 期望: w1.weight [2304, 2560] I8(每字节 2 个 fp4 -> 2304x5120 逻辑参数)、w1.scale [2304, 160] F8_E8M0(groupK=32)
```

## 2. 与 V4-Flash-0731(我们当前生产模型)的架构 diff

| 维度 | V4-Flash-0731 | V4.1-Flash | 对我们 |
|---|---|---|---|
| 层数 / 结构 | 43 层,单栈 | **40 层 = 20 enc + 20 dec(CED)** | ⚠️ 新机制 |
| hidden / inter | 4096 / 2048 | 5120 / 2304 | — |
| 路由专家 | 256,top-6 | **384,top-6** | 专家总字节 ×2 |
| 专家字节 | 137 GiB(12.75 MiB/专家) | **269 GiB(17.93 MiB/专家)** | CPU 流量 ×1.3/token |
| 专家布局 | `experts.{i}.w1/w2/w3.weight(.scale)` | 同左(fp4,scale [N, K/32] ue8m0) | ✅ 引擎契约一致 |
| **Engram** | 无 | **2 张表,各 384M 行 × 256 维 FP8 = 各 91.5 GiB** | ✅✅ 主机内存最佳候选 |
| KV | compress_ratios 交替 4/128 | CSA2:**m=2(enc)/ m=1(dec)+ 跨层复用 + FP4 KV** | ✅ KV 变得无关紧要 |
| KV 跨层复用 | 无 | `kv_source_layer_ids [2,8,14,20]` | 主线待做 |
| 索引器复用 | 无 | `index_source_layer_ids`(8→4 层)+ 层级候选池 | 主线待做 |
| 投机解码 | dspark(块 5,目标层 40-42) | DSpark(块 5,目标层 37-39,3 块 drafter) | ★ 对我们杠杆最大 |
| 视觉 | 外挂式 | 原生(32 层 + MLP projector) | 主线待做 |
| 权重总量 | 155.4 GiB | **475.2 GiB** | — |

> 0731 与我们已有结论的对照:0731 实测 156 GB/137 GiB 专家,单卡 256K KV 需 7.7 GiB。
> V4.1 的 `n_routed_experts` 从 256 → 384,而 `compress_ratios` 从"4/128 交替"变成"2/1",
> **所以 V4.1 的 KV 收益并非来自更激进的 per-layer 压缩,而是来自跨层复用 + FP4 精度 + SWA KV 移出 HBM**。

## 3. 权重清单(逐张量实测,非估算)

用 Range 请求只读 safetensors 头部得到精确形状:

| 张量 | dtype | 形状 | 大小 |
|---|---|---|---|
| `layers.1.engram.embed.weight` | F8_E4M3 | [384,006,168, 256] | **91.55 GiB** |
| `layers.14.engram.embed.weight` | F8_E4M3 | [384,016,682, 256] | **91.55 GiB** |
| `layers.{1,14}.engram.embed.scale` | F8_E8M0 | [384M, 8] | 5.7 GiB(合计) |
| `layers.N.ffn.experts.E.w1.weight` | I8(2×fp4) | [2304, 2560] | 5.62 MiB |
| `layers.N.ffn.experts.E.w1.scale` | F8_E8M0 | [2304, 160] | 0.34 MiB |
| `layers.N.ffn.experts.E.w2.weight` | I8 | [5120, 1152] | 5.62 MiB |
| `layers.N.ffn.experts.E.w2.scale` | F8_E8M0 | [5120, 72] | 0.34 MiB |
| `layers.N.ffn.shared_experts.w1.weight` | F8_E4M3 | [2304, 5120] | 11.25 MiB |
| `layers.N.attn.wq_a.weight` | F8_E4M3 | [1280, 5120] | 6.25 MiB |
| `head.weight` | BF16 | [129280, 5120] | 1.23 GiB |
| （每个路由专家合计） | | w1+w3+w2(+scale) | **17.93 MiB** |

**汇总(475.2 GiB)**

| 类别 | 大小 | 占比 | 放哪 |
|---|---|---|---|
| 路由专家 384×40(含 ue8m0 缩放) | **268.9 GiB** | 56.6% | 主机 DRAM + xiaotu 引擎(我们的主场) |
| **Engram(2 表 183.1 GiB + 缩放 5.7 GiB)** | **188.8 GiB** | 39.7% | **主机 DRAM + UVA 视图(每 token 只需 12 KB 流量)** |
| shared experts(40×3 张 fp8) | ~1.4 GiB | 0.3% | GPU(常驻) |
| 注意力(40 层 fp8) | ~1.6 GiB | 0.3% | GPU |
| lm_head(BF16)+ embed | ~2 GiB | 0.4% | GPU |
| 视觉塔(32 层 BF16)+ aligner | ~3 GiB | 0.6% | GPU |
| DSpark 3 块(128 专家 top-3)+ MTP + mHC/norm | ~9 GiB | 1.9% | GPU(可部分 CPU) |

## 4. 为什么 Engram 是"天赐良机"

Engram 的取数模式与专家权重**完全不同**:

| | 路由专家(每 token) | Engram(每 token) |
|---|---|---|
| 计算 | 每层 6 次 GEMV(fan-out 全量行读取) | 纯 gather,无矩阵乘 |
| 主机→设备流量 | **4.2 GiB**(6 专家 × 17.93 MiB × 40 层) | **12 KB**(2 模块 × 3 阶 × 8 头 × 256 B) |
| 放大比 | — | **约 1/350,000** |

也就是说:**模型 38.5% 的参数量,只贡献 0.0003% 的主机带宽开销**。把它放主机内存几乎"免费"。
更关键的是,官方技术报告 §2.4.2 明确写了生产栈的做法:

> "During inference, deterministic addressing enables embeddings to be prefetched from
> **host memory** via background RDMA transfers, with prefetching for the first module
> overlapping computation in the first Transformer block."

这与我们为 Qwen3.8 实现的 `XIAOTU_PLE_CPU=1`(表 build 在 CPU → `load_weights` 后 pin +
`get_accelerator_view_from_cpu_tensor` UVA 视图)**是同一个设计**。官方把"主机内存 + 预取"
写进了架构设计,等于给我们的路线背书。

**唯一需要先验证的约束**:Engram 两张表合计 183 GiB,我们的 `ulimit -l` 实测约 189 GB,
单进程全量 pin 会顶到上限。TP=2 时每 rank 只持 91.5 GiB ⇒ 天然规避;或者提高 RLIMIT_MEMLOCK。

## 5. KV:从"必须精打细算"变成"不用管"

| | V4-Flash-0731(vLLM 实测) | V4.1(技术报告) |
|---|---|---|
| 每 token 全局 KV | **29.5 KB**(实测:12 GiB / 437,337 token) | **890 B**(fp4 + CSA2 跨层复用) |
| 256K 上下文 | 7.7 GiB | **0.23 GB** |
| 1M 上下文 | 29.5 GiB | **0.89 GB** |
| 持久化(SSD/主机) | 全量 | V4-Flash 的 **1/8**;SWA KV 不再持久化,改放**每机 10% DRAM 的池** |

> 口径提示:论文的 "1/4" 是相对 DeepSeek 自己的 V4-Flash 推理栈;vLLM 当前对 0731 的
> 实现口径下 KV 大 8 倍(29.5 KB vs 推算 ~3.5 KB)。**KV 红利能拿到多少,取决于服务栈
> 是否真正落地了 CSA2 跨层复用 + FP4 KV**——这也解释了为什么主线里已经出现
> `fp4_kv`(`config/vllm.py`、`kv_cache_interface.py`)的钩子。

对我们的直接后果:**`--max-num-batched-tokens` 可以从 8192 放开到 32K–128K**,
长 prefill 的分块数从 16–32 降到 1–4。我们实测 0731 的 128K TTFT 是 246.8 s,
主因就是"每个 chunk 都要把 137 GiB 专家重流一遍";V4.1 的 KV 允许我们只流 1–4 遍。

## 6. 吞吐预估(估算,非实测)

**decode(CPU 专家路径,瓶颈是主机 DRAM 带宽)**

| 模型 | 每 token 每层激活专家字节 | 层数 | 每 token 合计 | 相对 |
|---|---|---|---|---|
| 0731(top-6/256,I=2048,H=4096) | 76.5 MiB | 43 | **3.25 GiB** | 1.00× |
| V4.1(top-6/384,I=2304,H=5120) | 107.6 MiB | 40 | **4.25 GiB** | **1.31×** |

⇒ 同样的实现下,V4.1 的 CPU decode 每 token 成本约高 **31%**。考虑到我们 0731 在 C=64
时的有效带宽效率约 97 GB/s(峰值 422 GB/s,即约 23%),V4.1 的**粗略**预期是:
单卡 35–45 tok/s、TP=2+EP 60–90 tok/s 量级。**这个数字的可信度只到量级**,
必须等主线可跑后实测。

**两个对 CPU 路径特别重要的杠杆**

1. **DSpark 投机解码**:DSpark 用一个 3 块、128 专家 top-3 的小 drafter 在 GPU 上草稿,
   一次验证覆盖多个 token。对带宽受限的 CPU 专家路径收益是**近似线性**的——
   这正是 DeepSeek 自己用来压低长上下文 Agent 成本的手段,也是我们应当优先接入的能力。
2. **EP 专家并行**(本项目本轮新增 `XIAOTU_MOE_EP`):V4.1 的 384 专家在 TP=2 下每 rank
   192 个,配合零权重掩码(引擎 `weights[ai] != 0.f` 过滤,零算力零带宽),
   把每 rank 的 CPU 工作量减半。

**prefill**:CED 让 prefill 只走 20 层 encoder(激活 8B),而 decode 走 40 层(16B)
⇒ **prefill 的每 token 成本约减半**。加上 KV 不再限制 chunk 大小,长 prefill 有望从
0731 的"分钟级"降到"几十秒级"。

## 7. 本项目的可复用 / 需新做 / 依赖主线

**✅ 直接复用(我们已有)**

| 能力 | 现成资产 | 在 V4.1 上的角色 |
|---|---|---|
| CPU 专家引擎 | `xiaotu_moe`(MXFP4/FP8/BF16/INT4) | 269 GiB 路由专家的计算 |
| 查找表放主机内存 | `ple_offload.py`(Qwen PLE,51.2 GB pin + UVA) | **Engram 183 GiB 的直接模板** |
| 专家并行 | `XIAOTU_MOE_EP`(零权重掩码 + TP all-reduce) | TP=2 下每 rank 192 专家 |
| GPU 流式 prefill | `gpu_prefill.py`(ping-pong slot + 预取) | 长 prompt 的专家权重 H2D |
| 调参/测量基建 | `scripts/tune_serve.sh`、`tune_client.sh`、ShareGPT 协议 | 新模型加一个 `MODE=` 即可 |

**⚠️ 需要新做(我们可以承担的部分)**

1. `engram` 模块的 CPU/主机内存实现:复用官方参考实现(`inference/engram.py` 仅 8 KB,
   含 `build_compressed_token_map`、多阶 N-gram 哈希、压缩词表 99,092),把
   `embed.weight`([384M, 256] FP8)挂到 pin + UVA 视图。**这是本项目最应该抢的活**。
2. 384 专家 / fp4 / groupK=32 / ue8m0 布局在新引擎路径上的验证(与我们 0731 的
   mega 融合布局一致,主要工作是权重融合映射的复用与校验)。
3. 新模型的推荐参数与 ShareGPT 基线(等主线可跑后)。

**❌ 不在我们范围(必须等主线 vLLM)**

CED(encoder-decoder KV 流向)、CSA2 跨层 KV/索引复用、层级稀疏索引器、FP4 KV cache、
Single-Pass mHC、视觉塔、DSpark 推理路径。我们的插件是 **OOT 覆盖 MoE 层**,不重写模型。

**主线现状(我们安装的 vLLM,commit `6c73b08`,2026-09-08)**

| V4.1 需要 | 主线已有? |
|---|---|
| `compress_ratios` / compressor | ✅ `models/deepseek_v4/compressor.py` |
| FP4 KV | ✅ 钩子已存在(`fp4_kv`,`kv_cache_interface.py`) |
| DSpark | ✅ `nvidia/dspark.py` 等 |
| SWA 窗口 | ✅ `attention.py` |
| **Engram** | ❌ 无 |
| **跨层 KV / 索引复用** | ❌ 无 |
| **CED(enc-dec)** | ❌ 无 |
| **`deepseek_v41` 架构名** | ❌ 无 |

⇒ 主线距离 V4.1 大约缺 4 个模块;按 0731 的落地节奏,预期需要数周(社区/官方适配)。

## 8. 建议路线

**P0(现在,不等主线)**
- 把本报告结论并入 `docs/MODEL_GUIDES.md` 的"后续模型"章节,明确 V4.1 的定位与前置条件。
- 把 `ple_offload.py` 抽成通用"稀疏查找表 → 主机内存"钩子(PLE / Engram 共用),
  并用 Qwen 单卡回归验证不退化。
- 在 `docs/TUNING_REPORT.md` 标注:0731 的 8192 nbt 限制**只由 KV 造成**,换模型后必须重扫。

**P1(主线出现 `deepseek_v41` 之后,1–2 周)**
- Engram 表 offload + 路由专家 CPU + 单卡/双卡 256K 参数复扫(直接复用 `tune_serve.sh`)。

**P2(中期)**
- DSpark 投机解码接入(对 CPU 专家路径的收益最大,属于"结构性"提速)。
- 前缀缓存(Agent 场景按缓存命中计费,我们的 `--no-enable-prefix-caching` 需要重新评估)。

## 9. 风险

| 风险 | 影响 | 对策 |
|---|---|---|
| 主线长期不支持 CED/CSA2/Engram | 我们无法上线 V4.1 | P0/P1 保持"引擎侧就绪",主线一到即可跑 |
| Engram 183 GiB pin 触发 RLIMIT_MEMLOCK | 进程被杀 | TP=2 分摊(每 rank 91.5 GiB)或提高 ulimit |
| CED/CSA2 在 vLLM 的实现口径与论文差距大 | KV/prefill 收益缩水 | 上线前实测 KV bytes/token,别信论文数字 |
| V4.1 的 decode 激活 16B > 0731 的 ~7B | CPU 路径每 token 更贵 | EP 分片 + DSpark + 更大 batch |
| 官方 API 降价(命中 0.02 元/百万 token) | 自建动机下降 | 定位在数据不出域 / 无按量计费 / 离线场景 |

## 10. 附录:如何自己复核这些数字

```bash
# 1) 配置(2 KB)
curl -sL https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json

# 2) 权重总量与张量名(7.4 MB)
curl -sL -o index.json https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/model.safetensors.index.json

# 3) 逐张量形状(只读 safetensors 头,不下载权重):
#    前 8 字节 = header 长度 N,随后 N 字节是 JSON(name/dtype/shape/data_offsets)
curl -r 0-7 https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main/model-00047-of-00048.safetensors | python3 -c \
 "import sys,struct;print(struct.unpack('<Q',sys.stdin.buffer.read())[0])"

# 4) 技术报告(51 页 PDF,含 §2 架构 / §3.2 推理系统)
curl -sL -o tech.pdf https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main/DeepSeek_V41_Tech_Report.pdf

# 5) 官方参考推理实现(engram.py / model.py / kernel.py / convert.py)
curl -sL https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/inference/README.md
```

License: Apache-2.0
