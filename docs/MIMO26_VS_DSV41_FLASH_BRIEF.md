# MiMo-V2.6-Flash-RL vs DeepSeek-V4.1-Flash(精简对比)

> 本文只给结论与对比,不含引用标记。**所有数据的具体出处、本机实测记录与派生公式,详见完整版
> `dev-docs/MIMO26_VS_DSV41_FLASH.md`。**
> 成本口径:decode、batch=1、无前缀复用、专家权重按 4-bit 计。

---

## 一、架构异同

| 维度 | MiMo-V2.6-Flash-RL | DeepSeek-V4.1-Flash |
|---|---|---|
| 总参数 / 激活 | 310B / **15B** | 552B + 196B Engram / **8B prefill、16B decode** |
| 层数 / Hidden | 48 / 4096 | 40(20 编码 + 20 解码)/ 5120 |
| 注意力 | hybrid **SWA-128 + Global 5:1**,GQA + **DiffKV**(192/128) | **CSA2 压缩稀疏** + 跨层 KV/索引复用 + 分层 indexer + **MLA latent** |
| KV 精度 | bf16 | **FP4 main KV** + FP8 SWA KV |
| 专家 | 256 routed / top-8(1/32),**无共享** | 384 routed / top-6(1/64)+ **1 共享** |
| 路由打分 / 激活 | sigmoid / `silu`(无 clamp) | sqrtsoftplus / SwiGLU **clamp@10** |
| 专家精度 | **MXFP4**(e8m0 block-32) | **FP4**(block 32,ue8m0) |
| 注意力精度 | FP8 E4M3 block-128 | FP8(block 32,ue8m0) |
| 位置编码 | partial-RoPE,**无 YaRN**(原生 1M) | **YaRN 16**(64K→1M) |
| 多模态 | 文本 + 图 + **视频 + 音频** | 文本 + 图 |
| 投机解码 | 3 层 MTP + 5 层 DFlash | **DSpark** 3 层 + 置信度调度 |
| 落盘 | **161.0 GiB** | **475.2 GiB** |

**一句话**:外形(稀疏 MoE + 1M + 多模态 + FP4 专家 + FP8 注意力)高度相似,但
**MiMo 靠"权重小、结构简单"省钱,DeepSeek 靠"把 KV 压到极限、非对称计算"省钱。**

---

## 二、各自独有的技术特色

**MiMo-V2.6(集中在训练范式)**
- **You Only RL Once** —— 一个混合 RL run 同训 code / general / visual / cyber,多 harness 同 batch。
- **Groupwise Agentic Grading** —— 离线用对比 rollout 造 rubric(GRS)+ 在线给通过轨迹排质量并重分配 advantage(GAR),形成自改进闭环,引导"更短路径、更少 token"。
- **MOPD2** 多前缀多教师 on-policy 蒸馏。
- **RL 稳定性**:冻结 MoE router + 四层反 reward hacking(奖励设计 / 对抗评测 / 异常检测 / 验证器交叉校验)。
- **完全开源 RL 栈**:7k+ 任务环境、端到端框架、可组合 mini-harness、Distill-Qwen-9B。
- 原生**音视频** + learned attention sink。

**DeepSeek-V4.1-Flash(集中在推理效率)**
- **CED**:不对称 8B prefill / 16B decode,prefill 计算≈减半。
- **CSA2**:Full/Reindex/Reuse 三模式跨层复用 + 分层稀疏 indexer,把深层 indexer 成本与上下文长度解耦。
- **FP4 main KV** = 890 B/token;**SWA Bounded Replay** ⇒ 持久 KV ≈1/8。
- **Single-Pass mHC** 高效残差混合。
- **Engram** 196B 稀疏 n-gram 条件记忆。
- **DSpark** 半自回归草稿 + 置信度调度验证。
- **可控推理预算 1–100**(API 暴露 low/high/max);预算控制可从单轮迁移到长程 agent 轨迹。
- **逐模态负载均衡**(text / image 各一套 correction bias)。

---

## 三、公开评分比较(分类)

> ⚠️ 两列来自**不同 harness / 评测框架**,差值只能当量级参考,不是严格排名。

| 类别 | Benchmark | MiMo-V2.6-Flash | DS-V4.1-Flash | 高者 |
|---|---|---:|---:|---|
| 代码 | DeepSWE v1.1 | 67.9 | **74.2** | DS |
| 代码 | ProgramBench | **26.0** | 20.3 | **MiMo** |
| 代码 | Terminal-Bench 4.0 | 28.8 | **31.2** | DS |
| 代码 | Terminal-Bench 2.1 | 87.6 | **90.6** | DS |
| 代码 | NL2Repo-Bench | — | 65.4 | DS only |
| 代码 | MiMo Code Bench(内部) | 61.2 | — | MiMo only |
| 通用 Agent | AutomationBench v1.0.6 | 52.3 | **54.8** | DS |
| 通用 Agent | Agents' Last Exam | 27.6 | **31.8** | DS |
| 通用 Agent | Toolathlon-Verified | 73.6 | — | MiMo only |
| 通用 Agent | OSWorld-Verified | 80.8 | — | MiMo only |
| 通用 Agent | JobBench | 61.2 | — | MiMo only |
| 网络安全 | CyberGym | **95.1** | 88.1 | **MiMo** |
| 网络安全 | SEC-Bench Pro | 47.5 | **62.8** | DS |
| 网络安全 | ExploitGym | 6.0 | **15.3** | DS |
| 网络安全 | ExploitBench | 25.3 | — | MiMo only |
| 网络安全 | MiMo Cyber Bench(内部) | 77.2 | — | MiMo only |
| 视觉 Agent | MiMo Visual Coding(内部) | 71.5 | — | MiMo only |
| 视觉 Agent | Chartography w/ tools | — | 78.9 | DS only |
| 视觉 Agent | BabyVision w/ tools | — | 89.6 | DS only |
| 视觉 Agent | ZeroBench-main w/ tools | — | 49.0 | DS only |

**推理 / 知识**:MiMo V2.6 报告未公布此类分数,DS 有完整表(GPQA Diamond 90.9、HLE 36.8、Codeforces 3471、MathArena Apex 65.6 及五维 Base 表)⇒ **无法对比**。

**综合指数**:DS-V4.1-Flash 的 AA 智能指数 **39**(cost/task $0.27,输出 225.6 tok/s);MiMo-V2.6-**Pro** 为 **46.32**(官方称最强开源);**MiMo-V2.6-Flash 的 AA 指数官方未公布**。

**分类小结**:代码/通用 agent、漏洞利用类 DeepSeek 略胜;CyberGym 与 ProgramBench MiMo 反超;推理知识无同口径分。

---

## 四、平均每 token 成本

### 4.1 空间成本

| 指标 | MiMo-V2.6-Flash-RL | DS-V4.1-Flash | 倍数 |
|---|---:|---:|---:|
| 权重落盘 | **161.0 GiB** | **475.2 GiB** | MiMo 小 **2.95×** |
| 每 token 专家权重流量 | **≈5.03 GB** | **≈5.26 GB** | ≈持平 |
| 每 token 非专家权重流量(估算) | ≈7.6 GB | ≈6.1 GB | DS 略小 |
| **合计权重流量 / token** | **≈12.7 GB** | **≈11.4 GB** | ≈持平 |
| **global KV / token** | **23,040 B(22.5 KiB)** | **890 B** | **DS 少 25.9×** |
| SWA KV / 序列(常数) | ≈25.6 MB | ≈2.6 MB | DS 少 ~10× |
| **KV @1M / 序列** | **22.52 GiB** | **0.87 GiB** | **DS 少 25.9×** |
| 本机 1M KV 池(TP=2 实测) | 2,078,802 token(dummy)/ 1,844,560(真权重) | 同 HBM 约可容 26× token | — |
| 本机主机内存(TP=2) | ≈310 GiB/进程 | 服务树峰值 ≈638–644 GiB | MiMo 更省 DRAM |

**结论**:每 token 搬运的权重字节几乎一样,差距全在 **KV(26×)**。
**长上下文 / 多轮 agent 空间成本 DeepSeek 量级性更优;冷启动与常驻内存 MiMo 更省。**

### 4.2 TOPS / 计算成本

| 指标 | MiMo-V2.6-Flash-RL | DS-V4.1-Flash |
|---|---:|---:|
| 激活参数 | 15B | **8B prefill / 16B decode** |
| **prefill FLOPs / token** | ≈**30 GFLOP** | ≈**16 GFLOP** |
| **decode FLOPs / token** | ≈**30 GFLOP** | ≈**32 GFLOP** |
| 算力 @1000 tok/s | **29.6 TOPS** | **32.0 TOPS** |
| 每 1M-token prefill 计算 | ≈30 PFLOP | **≈16 PFLOP** |
| 每 1M-token 输入 KV 显存 | 22.5 GiB | **0.87 GiB** |

**结论**:**decode 算力几乎相同**(30 vs 32 GFLOP);**prefill DeepSeek 便宜近半**(CED)。
⇒ 输入密集型 agent 负载 DeepSeek 计算成本占优。

### 4.3 API 价格(USD / 1M token)

| | cache hit | cache miss | output |
|---|---:|---:|---:|
| **MiMo-V2.6-Flash** | **$0.0028** | **$0.14** | **$0.28** |
| **DS-V4.1-Flash(peak)** | $0.006 | $0.30 | $1.20 |
| DS-V4.1-Flash(off-peak 5 折) | $0.003 | $0.15 | $0.60 |

**结论**:cache-miss 输入 MiMo 便宜 **2.1×**(vs peak)/ 1.07×(vs off-peak);输出便宜 **4.3×**(peak)/ **2.1×**(off-peak)。

---

## 五、总评(三句)

1. **架构哲学相反,代价与收益互换**:**KV 23 KB vs 890 B/token(DS 优 26×)**;**权重 161 vs 475 GiB(MiMo 优 2.95×)**。
2. **每 token 权重流量与 decode 算力打平**(≈5 GB 专家 / 30 vs 32 GFLOP);**分水岭在 prefill 与长上下文** ——
   DeepSeek 更适合长输入、多轮、超长上下文;MiMo 更适合短上下文、常驻内存敏感、需要**音视频**与**开源 RL 复现**的场景。
3. **分数各有胜负**(DeepSWE / TB2.1 / 漏洞利用 DS 强;CyberGym / ProgramBench MiMo 强),且带 harness 差异;**Flash 档缺公开 AA 指数**。
