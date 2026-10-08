# Strata 技术分析报告 —— 特色剖析与对本项目的建议

> **分析对象**:[Niko1221/Strata](https://github.com/Niko1221/Strata)(`main`,engine v0.1.40.3)
> **分析日期**:2026-10-08 · **方法**:clone 到 `/tmp/Strata`(depth-1)后通读 README / AGENTS.md / `docs/**`(31 篇)+ 核心头文件与源码目录;未逐行审内核
> **读者**:本项目(vllm-xiaotu-moe)内部 · **性质**:技术调研 + 建议,**不是**发布面材料
> ⚠️ **局限(先声明)**:① 只读文档与头文件,未运行其引擎,所有数字均为**其自报**;② 社区 benchmark 是自报数据;③ depth-1 clone 无提交历史;④ 其模型/量化档位与我们有 2–3 倍差距,数字**不可直接横向比较**

---

## 0. 结论先行

1. **Strata 与本项目是同一条技术路线的镜像**:都认"显存装不下专家权重",都靠"CPU 算专家 + 专家字节过 PCIe",都撞在同一堵墙上(Strata 自己量化为:**663.6 MB 专家字节/token ÷ ~40 GB/s = 16.2 ms,占一个 ~53 ms token 的绝大部分,且流水线只藏住 96% 中极少的一部分**)。
2. **分歧在"谁算专家、粒度多细"**:Strata 把**按路由频率排序的热专家**放进 VRAM(**不是整层**),CPU 并行算 miss;**粒度是专家,不是层** —— 这一点正是我们当初否决"专家层常驻"时没试过的坐标。
3. **Strata 有一组独立实测,直接支撑我们推迟的 `SPEC_mbt_budget`**:**453 个专家槽(0.86 GiB)之差 ⇒ 解码 13.7 vs 102 tok/s(7 倍)**,而且**慢的那次命中的缓存命中率更高** ⇒ 病根不是 miss,是 **"0 MiB free"**。这就是我们的"激活工作区 ∝ MBT"在另一个项目里的同构现象。
4. **本项目在"纪律/台账"上强于 Strata**(R34/R35、BUG_REGISTRY、可跑判据);**Strata 在"显式未验证项 + 可离线复算的收据"上强于我们**。
5. **有 1 条我们的既有结论值得重新审**:长 prefill 饿死解码时,我们只测过"**改 chunk 大小**"(结论:净更差 ⇒ 只能 PD 分离);Strata 用的是"**保持 chunk 大小、在 chunk 之间插入解码窗口 + 长 prompt 让步**" —— 这是**另一个自变量**,我们没测过。

### 建议优先级(详见 §4)

| # | 建议 | 优先级 | 是否需重启生产 | 主要代价 |
|---|---|---|---|---|
| **R1** | 显存**留白**优先于"最后一块显存";把 `XIAOTU_GP_ACT_RESERVE_GIB` 实测化 | **P0** | 否(先离线) | 少一档 MBT |
| **R2** | profile **二次校验**(= 我们 SPEC 的改动 ①②) | **P0** | 是(≈40 min) | 已在推迟队列 |
| **R3** | 重估"专家常驻":**细粒度热专家槽位**,先离线量化 | **P1** | 否 | 纯分析 |
| **R4** | **chunk 间插入解码窗口**(协作式让步)的离线估算 | **P1** | 否 | 纯估算 |
| **R5** | 短 chunk 的 CPU/GPU **门槛扫描**与分摊 | **P2** | 是(临时实例) | 机时 |
| **R6** | "复制上下文"负载的 **prompt-lookup 草稿** | **P2** | 否(先离线接受率) | 中(取决于上游) |
| **R7** | **会话边界检查点**(治"改写最后一条消息") | **P2** | 否(先查口径) | 小–中 |
| **R8** | 台账加"**证据等级**"列 + 结果的离线收据 | **P2** | 否 | 极低 |

---

## 1. Strata 是什么

| 项 | 内容 |
|---|---|
| 定位 | **从零自研的 MoE 推理引擎**,让 **Qwen3.8-Flash-Next**(125B 级 / **24,576 专家** / top-10 / 48 层)跑在**消费级单卡 12–24 GB + 系统内存 + SSD** |
| 形态 | C++/CUDA/HIP/SYCL 引擎(~10.7 万行,`src/`+`include/`)+ Python 服务端(OpenAI/Anthropic/Responses 兼容 + MCP)+ 自带 Web 面板 + 一键安装器 |
| 基座 | ⭐ **不是 fork**:自建引擎,把 **llama.cpp / ggml 当库**(CPU i-quant 点积、GGUF 读取) |
| 平台 | NVIDIA(RTX20–50)、**AMD HIP**、**Intel Arc SYCL**、Strix Halo iGPU;Windows + Linux |
| 规模 | 31 篇专题文档 / 93 个测试文件 / 98 个 bench 结果目录 / 一篇[论文 PDF](https://github.com/Niko1221/Strata/blob/main/docs/paper/Strata-Paper.pdf) / README **7 种语言** |
| 公开实测 | RTX 5070 12 GB + 64 GB RAM:Q2_0 解码 **94 tok/s**、32K prompt 预填 **2,650 tok/s**;RTX 3090 预期 100–140 tok/s |

⭐ 它把"AI 助手"当一等公民:仓库内 [AGENTS.md](https://github.com/Niko1221/Strata/blob/main/AGENTS.md)、`docs/AI_SETUP.md`(让 AI 照着装)、**MCP server**、`START-HERE.bat --calibrate`(自测最优引擎参数)。

---

## 2. 技术特色

### 2.1 核心矛盾:CPU 专家池的带宽墙

[`include/strata/core/expert_cache.hpp`](https://github.com/Niko1221/Strata/blob/main/include/strata/core/expert_cache.hpp) 开头把立项动机写成了算式:

```text
CPU 专家池 = 663.6 MB 专家字节/token ÷ ~40 GB/s = 16.2 ms
一个 token 约 53 ms
流水线只藏住 19.035 ms 中的 1.055 ms
⇒ CPU 工作是 96% 暴露的,因为残差链严格串行
⇒ 唯一出路:别再读这些字节 —— 把一部分专家放 VRAM 让 GPU 算,CPU 只算 miss
```

⭐ 这句话几乎就是本项目的立项陈述。**同一堵墙,两种解法。**

### 2.2 三级存储层次

| 层 | 放什么 | 关键设计 |
|---|---|---|
| **VRAM** | 注意力 / DeltaNet mixer、gated-residual 权重、router、共享专家、输出头、MTP draft、**KV(>64K 只留最热段)**、⭐ **专家缓存** | 缓存**随对话自适应**;**每多 1 GB ≈ 多约 700 个专家** |
| **RAM** | **全部 24,576 个专家,pinned** | CPU **原地(in place)** 算 GPU 没有的,**与 GPU 并行**;AVX-512/AVX2 + ggml i-quant 内核 |
| **SSD** | **28.8 GB n-gram 表**;低内存模式下还有 `experts.bin` | 每 token 只读几行;`--ple-io direct` 绕过 OS cache |

### 2.3 专家缓存:**静态 profile + 运行时自适应**

- 启动按 `tools/make_profile.py` 产出的 `profile.bin`(**按路由频率降序的 (layer, expert) 表**)填充;
- 之后自适应 tier(`--adapt-every` / `--adapt-swaps` / `--adapt-decay`)按实际流量换进换出;
- **可跨重启保存学到的 profile**:`"expert_profile_save"`(每 10 分钟 + 干净退出时写,临时文件 + rename ⇒ 崩溃不留半截);
- ⭐ 头注释连统计口径都自纠:`h_expert` 十折留一 **0.6447 [0.6110, 0.6750]**,并注明"README 公布的 0.6573 是 in-sample、0.4720 是单 prompt 语料、都不是这个指标";`h_layer` 留一只有 **0.0456** ⇒ **不能按层做分组内核**。

### 2.4 预填:分块 / PCIe 流式 / 下一层路由预取 / CPU 分摊

- chunk 上限 **8,192**(`--prefill auto` 取"借来的缓存槽放得下的最大块";opt-in 可到 **32,768**);
- 本层注意力计算时,**下一层专家经 PCIe 流过来**;ring 按**字节**定尺寸(`STRATA_RING_BYTES`);
- ⭐ **`STRATA_PREFILL_CPU_SHARE=auto`**:chunk < 1,024 token 时,把"少数 token 路由到的专家"交给**空闲的 CPU 池**,并**按层实测两侧耗时、求出两者同时结束的分摊比**;实测 prompt 时间 **512 token −26%、1,000 token −22%,≥2K 无变化**(三台机器一致:−26/−22、−28/−19、−35/−25)。

### 2.5 投机解码:三件套叠加

1. **模型自带 MTP draft layer**:一次起草最多 3 token,一轮校验,**平均 2.4–3.2 token/轮**,整体 **1.6–1.8×**;
   草稿词表可分语言裁剪(`--draft-vocab en|cjk|cyrillic|fr`),CJK 由 40,525 → **106,299** id,中日韩回答快 **15–38%**;
2. ⭐ **prompt lookup / suffix draft**:回复在**复制上下文**时(改代码、引用文本)从早先的副本起草最多 5 token,**代码编辑快 6–11%,其他文本不变**;
3. **`STRATA_SPEC_COUPLED` + `STRATA_SPEC_GUMBEL`**:草稿与目标共享同一随机链 / Gumbel-max 噪声 ⇒ 采样请求接受率 **52.8% → 59.9%**(41.1 → 44.9 tok/s)。

⭐ 三者的共同点:**草稿只提议,目标模型决定每个 token ⇒ 输出不变**。

### 2.6 KV 与会话复用

- KV dtype 可选(FP16 / **INT8** / q4_0);**>64K 时只把最热段留 VRAM**,其余从 RAM 流;
- `--kv-grow`(0.1.40):KV 只按请求**实际触达的 cell** 占显存,余量留给专家缓存;
- 会话复用三件套:
  - **conversation parking**(`--conversation-cache-mib/-slots`,有界、限流、可被驱逐):追踪请求**0.53 / 0.46 s** vs 首轮 **3.9 / 5.7 s**;
  - **mid-prompt checkpoint**(`--prompt-cache-every`);
  - ⭐ **`MESSAGE_BOUNDARY_CACHE`**:专治"agent 保留同一段长历史、但**改写最后一条用户消息**"—— 首次完整编辑 **17.236 → 11.527 s(−33.1%)**,可复用前缀 **22,016 → 32,831 token**;并要求为此排除 `multi_gpu`(层切分)。

### 2.7 多卡 = **层切分(pipeline)**,不是张量并行

- 每卡只留**自己那些层**的专家缓存 ⇒ 两卡约两倍专家驻留(5080+3090 预填 **+18~20%**,decode 持平或更好);
- **一 token 每 verify window 只跨卡一次**(几百 KB,pinned RAM)⇒ **不需要 NVLink / P2P**,x4 / x1 槽位也能跑;
- `--batch-groups G`:N 个 slot 分 G 组在卡间**流水**;
- ⭐ `--pipeline-windows 2`:第一卡**预跑下一个窗口**(赌本窗口被整接受),实测均值 **72.4 → 84.0 tok/s(+16%)**,并用"状态副本 + 猜错回滚"保证 token 与串行一致;
- `--trim-stage-weights`:每卡只装自己层的 dense 权重,省下的显存进专家缓存(驻留专家 **8,819 → 10,626**)。

### 2.8 显存预算与"让渡" —— ⭐ 与本项目最相关的一节

| 机制 | 内容 |
|---|---|
| `--vram-reserve-mib` / `--vram-reserve-later-mib` | 预留**在缓存定尺寸之前扣除**;显示卡所在的那张可单独给更大预留 |
| `--expert-cache auto`(默认) | ⭐ **槽真正写下去之后再查一次,不够就缩小**(固定值 `--expert-cache N` **只查一次、不复查**) |
| `--idle-unload 600` / `--min-free-vram-mib` / `--before-load` | 闲置卸载;显存不够时答 **503** 而不是硬起 |
| ⭐ **`POST /v1/vram`** + `--vram-elastic` | 专家缓存按 **512 MiB 段**分配,**运行时 unmap 还显存给别的程序、之后再 map 回来**;还 4.3 GiB 用时 78 ms,decode 44 → 33 tok/s;map 回来 92 ms,**答案与收缩前 token 级一致** |

⭐⭐ **本节最有价值的一条实测**(RTX 4080 SUPER 32 GB,IQ3_S,1M 上下文):

| expert cache | slots | 启动时 VRAM free | decode tok/s |
|---|---:|---|---|
| `--expert-cache 8900`(11,631 slots) | 11,631 | 0 MiB(`LOW`) | **13.7 / 14.7** |
| `auto` | 11,178 | 217 MiB(`LOW`) | 102.1 / 122.3 |
| `auto` + `--vram-reserve-mib 1500` | 10,766 | 1,075 MiB | 88.9 / 96.5 / 101.4 |
| `--expert-cache 7000` | 9,148 | 4,130 MiB | 94.1 / 97.3 |

> **453 个槽(0.86 GiB)之差,把 13.7 与 102 tok/s 分开,而慢的那次命中率更高(91.4/94.7% vs 90.4/93.9%)⇒ 不是 miss,是"0 MiB free"本身要 7 倍代价。**
> ⇒ **留给卡上其余部分的余量,比多塞进去的槽更值钱。**

⚠️ 同时它记录了**绕过预算的经典失败模式**:WDDM 下分配未 touch 前不计驻留,分配前的 free 读数**可能高约 1 GiB**,驱动默认把超配放进系统内存而**不报错** ⇒ "启动看起来正常、banner 照报槽数、**只有速度崩**"。

### 2.9 低内存模式谱系(五档)

`--mmap-experts`(按需映射文件)→ `--resident-experts`(只把 GPU 没有的锁进 RAM)→ `--resident-budget-gib N`(按热度锁 N GiB)→ `STRATA_ARENA_MMAP`(Linux 只映射)→ `STRATA_FILE_RELEASE`(Windows 归还文件页)。
外加启动 **read-ahead**(`madvise WILLNEED`,128 KiB 步进):就绪 **920 s → 70 s**,填充 **39 MB/s → 3.2 GB/s**。

### 2.10 工程与文档纪律

- **每个特性默认关、opt-in**,且文档固定回答四件事:① 结论来自**读代码**还是**实测**;② **验证环境**(哪两张卡、多少核、什么 pack、是否提功耗);③ **测量数字**;④ ⭐ **还没验证什么**;
- 大量 **token 级 parity 收据**:`bench/results/exchange-rotation-ab.json` + 一段可离线跑的 SHA256 校验脚本;
- ⭐ 反例也照实写:`BATCHED_DMA` 明写"提交延迟降 **82.7%**,**总传输只降 1.3%**"、"**没有建立生成速度增益**"、"**不构成改默认值的理由**";
- 数字与结论**绑定环境**:同一句结论会标"这不是这套配置的实测""单次运行,不是普遍结论"。

---

## 3. 与本项目对照

| 维度 | **vllm-xiaotu-moe(本仓)** | **Strata** |
|---|---|---|
| **基座** | **vLLM 插件 + patch 系列**(严格不 fork 主线) | **从零自研引擎**(llama.cpp/ggml 当库) |
| **靶子** | 旗舰大 MoE:**DeepSeek-V4.1 / GLM-5.3 / MiMo**,官方 FP8+FP4 | **单一模型** Qwen3.8-Flash-Next 的多档量化 |
| **硬件** | **3× A100-40GB,TP=2**;≤64 核纪律 | **单张 12–24 GB 消费卡** + 32–64 GB RAM;Windows 优先 |
| **专家分工** | **CPU 算全部专家**(host 侧大表)+ 长 prompt 时**部分专家流到 GPU 算**(GPU 预填 staging) | **热专家常驻 VRAM 缓存**(自适应)+ CPU **并行**算 miss |
| **常驻粒度** | "专家层常驻"**已实测否决**(每层 3.36 GiB,挤压 32K 预填) | ⭐ **按路由频率的专家槽位**(单专家粒度)+ 每卡只留自己层的 |
| **量化** | 官方 **FP8 / MXFP4**,自研 **INT8 激活**(M 阈值 160) | **GGUF Q2_0 / IQ2_XS / IQ3_S / IQ1_M**,CPU 走 ggml i-quant |
| **并行** | **张量并行 TP=2** + vLLM 连续批处理 | **层切分 pipeline**;`--batch-groups`;**无 TP** |
| **长上下文** | **1M**(ds41f)/ 512K;MBT 4096–8192 | **262K**(可 rope scaling 外推);chunk ≤8192 |
| **显存 OOM 策略** | 固定优先级 + **推迟的 `SPEC_mbt_budget`**:启动算预算 → 查表定 MBT → 阶梯降档 → **不允许回升** | `--vram-reserve-mib` + `auto` **二次校验** + **`POST /v1/vram` 可运行时伸缩并可回升** |
| **前缀/会话复用** | LMCache(SSD、跨重启)⇒ **因崩溃与退化已禁用**,重启需 **~21 min 重预填** | conversation parking + tail / message-boundary checkpoint(**in-RAM,跨请求不跨进程**) |
| **多厂商** | 仅 NVIDIA(注意力 SM80 后端) | **CUDA + HIP + SYCL/Arc + Strix Halo** |
| **服务面** | vLLM 原生 OpenAI server | OpenAI + **Anthropic** + **Responses/Codex** + **MCP** + 自带 Monitor 面板 |
| **工程纪律** | **实验台账(B 编号)+ 铁律(R 编号)+ QA 错题本**(判据可跑) | 每特性一篇 doc + **验证收据** + 论文 + **显式"未验证项"** |

### 3.1 三处高度同构的独立发现(互为交叉验证)

1. **"CPU 内核分组影响舍入 ⇒ 输出依赖几个 token 共享专家"** —— ⚠️ **这是【验证工具】,不是"要解决的问题"**(见 §4 判据口径 / R40)
   - Strata:`STRATA_IQ_MT_MIN=1` 强制单/多 token 用同一内核,才让"贪心输出与草稿无关";其文档明写该开关是给 **A/B 用**的(*"keep it for A/B runs"*)⇒ 是**仪器**;
   - 本仓:`XIAOTU_MOE_INT8_VNNI_MIN_TOKENS`(**M 阈值 160**,crossover 实测 ≈150)—— 同族的**门控**,不是"必须消除的差异"。
   - ⇒ ⭐ **不要把"舍入不同"当成待解决的问题**;"bit 相等"只在**做单变量对照**时才有价值 ✗
2. **"短 prompt 该不该分给 CPU"的分界线,两边数量级相同**
   - Strata:`STRATA_PREFILL_CPU_SHARE`,<1,024 token 才划算,**≥2K 无变化**;
   - 本仓:`VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS`(默认 **2560**)、MBT 分块 —— 同类"按 chunk 大小决定谁算"。
3. **"chunk 大小 ↔ 工作区 ↔ 0 MiB free ⇒ 崩/慢"是共同的死结**
   - 本仓:激活工作区 ∝ MBT,是**连续两次长 prompt OOM 的真凶**,才有了推迟的 MBT 预算表;
   - Strata:chunk 由"借来的槽放得下"反推;且**"0 MiB free"本身值 7 倍**(§2.8 表)。

---

## 4. ⭐ 对本项目的建议

> ⭐⭐ **判据口径(用户 2026-10-08 裁决 ⇒ 台账 #115 / `IRON_RULES` R40)**:
> **本报告评估任何方案时,「输出不是 bit 相同 / 舍入改变」一律【不计入代价】** ✗ ——
> "bit 相等"是**验证工具**(把改动效果单变量隔离),**不是产品判据**;
> 真正的门禁是**端到端质量验收**(GSM8K 全量、长文四针、Vision 逐项 …)✓。
> 仅当某方案**改变了模型语义**(换量化档位 / 换精度)时,代价才写「**需质量验收**」。
> ⚠️ 下文引用的 Strata "not bit-identical" 是**它的 A/B 方法论要求**,**不是**本节任何一条的代价 ✗

> 每条固定六项:**背景 / Strata 证据 / 建议动作 / 可行性 / 代价 / 判据与风险**。
> 遵守 R23:**能停在前面的一律不走到"重启服务"**。

### R1 (P0) 显存**留白**优先于"最后一块显存";把 `XIAOTU_GP_ACT_RESERVE_GIB` 实测化

- **背景**:我们的显存优先级是 `KV 池 → GPU 预填 staging → 投机 draft → 激活工作区(∝ MBT)`,而**激活工作区是唯一不参与 `vram_policy` 预检**的一项;当前 `XIAOTU_GP_ACT_RESERVE_GIB=1.5`,而代码注释称**实测激活峰 2.9 GiB** ⇒ **预留偏小**。
- **Strata 证据**:453 槽 / 0.86 GiB 之差 = **7 倍速度**,且慢的那次**命中率更高** ⇒ 病根是"没有余量",不是"缺命中";其 `--vram-reserve-mib` **在定尺寸之前扣除**,并建议"余量比多塞更值钱"。
- **动作**:① 把 `XIAOTU_GP_ACT_RESERVE_GIB` 从 1.5 提到**实测值(≈3.0)**;② 在 `SPEC_mbt_budget` 里把"留白"提升为**第一类约束**(独立项),而不是 MBT 的副产品;③ 在启动日志里**打印"全部加载后 X MiB free"并低于阈值时告警**(Strata 的做法,直接可复制)。
- **可行性**:✅ **高**。①③ 是配置/日志级;② 是文档级。**不需要重启即可先做离线核算**。
- **代价**:预留 +1.5 GiB ⇒ 少约 1,500 MiB 的可用量,**可能让 MBT 从 8192 退回一个档**(按我们自己的 `h ∝ MBT` 数据,8192→4096 省 ~8.5 GiB,故 1.5 GiB 不足以逼退一整档;需实测)。**纯粹是"峰值余量换档位"的取舍。**
- **判据**:16K 长 prompt 在 reserve 3.0 + MBT 8192 下仍成功;启动日志的 free 值 > 0 且不再出现"跑一阵才 OOM"。**风险**:预留过多会白降吞吐 ⇒ 必须用 ①② 的预检把它算成**查表**,而不是拍一个常数。
- **关联**:这正是 `SPEC_mbt_budget` 缺口里那句"表里 vLLM 的每-token 系数 `k` 需一次真实 profile"。

### R2 (P0) profile **二次校验** —— 直接补上我们 SPEC 的改动 ①②

- **背景**:我们用**两处 hack 绕过**了 vLLM 的预算计算:① `mixed_experts.py:1490-1498` 在 profile run 期间**禁用 GPU 预填**;② `bringup_prod_8070.sh` **显式传 `KV_CACHE_BYTES`**(vLLM 见到它就跳过 profiling)。⇒ vLLM 的账本 = 权重 + 2.5 GiB KV,**staging(峰值 ~7.56 GiB)完全不在账上** ⇒ 启动查不出不够,**只能运行中崩**。
- **Strata 证据**:`--expert-cache auto` **在槽真正写下去之后再查一次**、不够就缩小;**固定值 `--expert-cache N` 只查一次、不复查** ⇒ 在 WDDM 下"**启动看起来正常、只有速度崩**"。这就是"用固定值绕开真实校验"的教科书级后果。
- **动作**:执行 `SPEC_mbt_budget` 改动 ①②③④(不再显式传 `KV_CACHE_BYTES`;去掉 profile 挡板;新 `scripts/mbt_budget.py` 出表+查表;层内按表限 qlen)。
- **可行性**:✅ 规格已定案(4 处改动),**但需要一次专门的重启窗口(≈40 min,须用户当次许可)**。⛔ 期间**不得改代码、不得自行重启**。
- **代价**:重启 ≈40 min(启动 20 + 重预填 20);去掉 profile 挡板后**profile run 会更慢、峰值更高**(它要真实走 GPU 预填)⇒ 启动时间可能进一步上升;若 profile 期间 OOM,需回退到"显式声明 staging 上限"。
- **判据**:启动日志出现**真实的** `transient_peak_headroom` / `non_kv_cache_memory`;去掉 `KV_CACHE_BYTES` 后 1M 仍能起;`scripts/mbt_budget.py --query` 给出的 MBT 与实测可行档位一致。
- **风险**:⭐ **这是唯一"改了会动生产"的建议**,其余建议都可在它之前先做完。

### R3 (P1) 重估"专家常驻":从**整层**改为**按路由频率的专家槽位**(先离线量化,不立项)

- **背景**:README 记载 2026-09-20 修订"**去掉"专家层常驻" —— 实测效果很差(每层 3.36 GiB,换来的收益抵不上它对 32K 预填充的挤压)**"。⚠️ 但那次的**粒度是"层"**,不是"专家"。
- **Strata 证据**:它**从不做整层**;填的是 `48 × 512` 个专家中**按路由频率排序**的那些,并且给了量化指标:`h_expert` 留一 **0.6447**;**`h_layer` 只有 0.0456**(⇒ 4.6% 的 (层, token) 对才能"10 个专家全在" ⇒ **按层分组根本不是优化,而是错解**)。⇒ **"粒度错了会得出错误结论"**,这条对我们是直接警示。
- **动作**(纯离线,**不重启、不改生产**):① 用真实 routing 统计算 **top-N 专家覆盖率曲线**(N 对应多少 GiB);② 用留一口径复算"可预测性",避免 in-sample 自欺;③ 只有当 **≤2 GiB 能覆盖 ≥某阈值** 时才考虑立项,否则明确记为"已量化否决",关闭这条线。
- **可行性**:⚠️ **中**。需要 routing trace;我们**目前没有现成的 routing 频率 profile 工具**(Strata 有 `tools/make_profile.py` + `tools/routing_kfold.py`)。但可以是纯计算,属 R23 阶梯里**最低成本**那档。
- **代价**:离线工具 ~1–2 天;**不建议**先动内核。真正的实现量很大(新增 slot 存储 + 驻留表 + 命中/未命中的 grouped kernel + 交换与一致性)。
- **判据**:在 ≤2 GiB 预算下 top-N 覆盖率(把"阈值"留到看完曲线再定)。**风险**:与既有 GPU 预填 staging **功能重复**;必须先说清两者的分工(一个是"把专家搬来算",一个是"把最热的专家留下来算")。
- ⭐ **本条的真正价值可能不是"做",而是"把结论建立在正确粒度上"** —— 只需一次离线分析。

### R4 (P1) 用**离线估算**检验我们"长 prefill 只能靠 PD 分离"的结论

- **背景**:B277/B279 实测注入 32768-token prompt 后,**解码请求的 ITL 从 64.8 ms 暴涨到 ~8.4 s,持续整个 57.6 s 预填窗口**。我们测过的唯一旋钮是**压小 chunk**(`--long-prefill-token-threshold`)⇒ **净效果更差**(decode 1.25 → 0.57 tok/s,因为 2048-chunk 只有 ~410 tok/s 而 8192 有 ~1000)。⇒ 结论落在"**没有可用的调度旋钮,应做 PD 分离**",单流交付用 `--max-num-seqs 1`。
- **Strata 证据**:它用**另外两个自变量**处理同一问题,**且都没改 chunk 大小**:
  - `STRATA_BATCH_DECODE_SHARE`(默认 0.5):**每个 chunk 之后让 slot 解码"该 chunk 耗时的一半"** ⇒ 长 prompt 只是拖慢别人,而不是停掉别人;
  - `BYIELD`:**更短的 prompt 在等 ⇒ 长 prefill 在 chunk 边界让步**:已读部分存进 slot、短请求先做、长请求再从 slot 继续(每请求最多两次)。
- ⭐ **关键洞察**:我们只测了"**改 chunk 大小**",**没测"chunk 间插入解码窗口"**。两者的代价结构不同:前者**同时**毁掉预填效率(权重重流变多),后者**保持** chunk 大小、只延长预填总时长。
- **动作**(零机时):用**已有台账**里的 `8.2 s/chunk`(B277 时间轴,非 B273 的 p99 8.7 s)、chunk 数与解码 ITL 数据,**离线估算**"每 chunk 后让出 f 比例时间"时,ITL 尖峰与预填总时长的 Pareto 曲线(f = 0.25 / 0.5)。若曲线在 f≈0.3 处就把 8.4 s 压到秒级以内,则值得单独立项。
- **可行性**:✅ **估算高**(纯计算);❌ **实现中**(vLLM 调度器是上游代码,要动就得加 patch,与"尽量不改主线"冲突,或需上游化)。
- **代价**:预填总时长变长(与用户"agent 长 prefill 等 20 分钟"的痛点**直接冲突**,必须显式权衡);实现中-高。
- **判据**:同一条 32K prompt + 一路解码的配对 A/B:**解码 ITL** 与**预填总墙钟**两个数一起报。**风险**:⚠️ 预填 = **DMA/装配受限**,**不是**计算受限 —— ⭐ B279 的"计算受限"结论**已被用户指正撤回**(见 **B284/B285**):`nvidia-smi` 的 `utilization.gpu` 只表示"有 kernel 在执行"的时长占比,**不衡量 FLOPs** ⇒ "SM 满 + mem 不满"同样符合"**SM 空转等 DMA/自旋**";真瓶颈是**设备侧 strided 转置**(84 GB/s strided vs 1361 GB/s 连续),`asm` 占每层 **85%(250 ms/层)**,而该层 MoE 计算只要 **5–50 ms** ⇒ DMA 必然主导。
  * ⇒ ⭐ 这**反而降低**了本条的预期风险:插入的**纯解码步**用 CPU 算专家(不走预填那条 DMA/装配链)⇒ 争用面**小于**"算力争用";但**仍是推断**,上机才算 ✓
- ✅ **已完成(2026-10-08,零机时离线估算)** ⇒ 台账 **B337**;工具 [scripts/estimate_prefill_yield_pareto.py](scripts/estimate_prefill_yield_pareto.py)。**结论:这一条的 Pareto 远优于我们当年测的"压 chunk"**:
  - 关键区分:**「预填步」贵(每步重流 40 层 = 136 GB),「纯解码步」不贵(64.8 ms,专家走 CPU)** ⇒ 当年 B273/B277 测的是"更多**预填**步",**没测**"更多**纯解码**步";"没有免费的旋钮"应更正为"**vLLM 现成旋钮里**没有"。
  - 由 B277 两点反解:每-chunk 固定流式成本 **F = 3.93 s**(⇒ 有效聚合 H2D **34.6 GB/s**),每-token 计算 **Rc = 1920 tok/s**;自检 `N=7 × 8.2 s = 57.4 s` vs 实测窗口 **57.6 s(−0.3%)** ✓
  - **同窗口代价(2.314×)下,插入解码步的 ITL 比 LPT=2048 好 44×**;**同 ITL(5.0 s)下,窗口 57.7 s vs 132.8 s**;取可实现 **D=1** 即得 **ITL 4.13 s(2.0×)、窗口 +0.8%** ⇒ 全面优于 LPT 的 5.0 s / +131%。
  - **窗口增长 ≤5% ⇒ D=6 ⇒ ITL 8.2 s → 1.18 s(7.0×)**;**≤10% ⇒ ITL → 0.66 s(12.4×)**。
  - ⚠️ **仍未上机**;三个必须实测的前提:插入步**不重流权重**、`t_dec` 在窗口内仍 64.8 ms、预填吞吐不受插入影响。**实现需改调度语义(vLLM 无此旋钮)** ⇒ 须进 patch 系列并先审。
  - ⭐⭐ **追加(2026-10-08,用户点出业务语义)**:本条收益**前提是"有并发解码在等"** —— 而生产**当前是 `MAXSEQS=1`**(无并发解码)⇒ 那时插入解码步是**纯亏** ✗
    ⇒ ⭐ **本条的真正定位 = 「让【已定案】的 `MAXSEQS` 1→2 安全落地」**([bringup_prod_8070.sh:62](scripts/bringup_prod_8070.sh#L62) 已写 `MAXSEQS=2`,待下次 bringup 生效),**不是**通用预填提速 ✓
    * ⭐ **必须【条件触发】**:只在"有请求正在解码且在等"时插入;否则不插 ✓
    * ⭐ `t_dec` 按并发数取(B273):**64.8 / 103.9 / 170.9 ms**(单流 / C=2 / C=4)⇒ seqs=2 下仍很强:基线 B273 的 C=2 p99 = **8,659 ms**,`f=5% ⇒ ITL ≈ 1.7 s(5×)`、`f=10% ⇒ ≈ 1.0 s(8.6×)` ✓
  - ⭐⭐ **追加(2026-10-08,用户澄清"对象是谁")**:用户说的"不至于显得系统卡死"指的是【**等 decode 输出的那个用户**】,**不是**运维/监控 ⇒ 而那**正是本条的机制** ⇒ ⭐ **两条收益 = 同一机制、不同判据**:① 吞吐/公平性 ② **感知活性** —— 后者要求**弱得多**(只要静默间隔不超人类阈值,**不需要 ITL 很小**)✓
    * ⇒ ⭐ **规格应写成【满足约束】而非优化 Pareto**:先定 `T_alive`,再取满足它的**最小 D** —— `D=1 / 2 / 3` ⇒ 静默 **4.1 / 2.8 / 2.1 s**,窗口只 **+0.8% / +1.6% / +2.4%**(seqs=2,`t_dec`=103.9 ms ⇒ +1.3% / +2.5% / +3.8%)✓
  - ℹ️ **另一个独立问题(别混进本条)**:合法长预填会让**监控心跳**停 45–130 s ⇒ 曾被误判卡死(**B306**:可能 1 次误重启生产,≈40 min),工具是 **GPU 进度心跳**(**B307** 已验证)✓ —— 那是**运维可观测性**,与本条的**用户感知**是两回事 ✓

### R5 (P2) 短 chunk 的 CPU/GPU 门槛扫描与分摊

- **背景**:我们的 GPU 预填门槛 `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS`(默认 **2560**)是**一刀切**:低于门槛全 CPU、高于才 GPU。**512–2560 这个中间区间现在是纯 CPU,GPU 可能闲置**。
- **Strata 证据**:`STRATA_PREFILL_CPU_SHARE` 在 **<1,024 token** 的 chunk 上把专家**分给空闲的 CPU 池**,并**按层实测两侧耗时求"同时结束"的分摊比**:512 token **−26%**、1,000 token **−22%**、**≥2K 无变化**(三台机器一致)。
- **动作**:① 先做**门槛扫描**(纯测量):对 512 / 1024 / 2048 token 的 chunk,量"纯 CPU"vs"GPU 预填"的耗时交叉点,验证 2560 这个默认值;② 只有①显示中间区间确有收益时,才评估"同 forward 内分摊"。
- **可行性**:✅ 高(测量);⚠️ 中(分摊实现)。测量需要起**临时实例** ⇒ 按 R23 **攒批**、并按 AGENTS「同机多实例纪律」用替代模型 + `oom_score_adj=800`。
- **代价**:① 机时(一次实例启动 + 一轮扫描);② 实现中(需在同一 forward 内把不同 token 的专家派到两条路径)。
  * ⭐ **「CPU 用自己的激活格式 ⇒ 最近位不同」不是代价**(R40):门禁是质量验收、不是 bit 相等 ⇒ 本条的可行性只由 ①② 决定 ✓
  * ⚠️ 但**它的对照臂是"GPU-only"**(Strata 证明的是"**小 chunk 时 GPU+CPU > GPU 独自**");**我们的基准是"纯 CPU"** ⇒ 「GPU+CPU > 纯 CPU?」**是另一个对照臂,Strata 的数据回答不了**,必须自己扫 ✓
- **判据**:交叉点落在 2,560 附近则**默认值被证实**,直接关闭这条线;若落在 ~1,000,则门槛可能要下调。**风险**:低(纯测量)。

### R6 (P2) "复制上下文"负载的 prompt-lookup 草稿

- **背景**:我们是 **agent 的推理后端**,负载特征是"**读文件 → 原样写回少量改动**"(README 已记录 `--long-prefill-token-threshold` 与 agent 长 prefill 的痛点)。我们已打通 MiMo MTP(**2.44×**)与 INT8 激活路径。
- **Strata 证据**:`prompt lookup` 在**回复复制上下文**时从早先副本起草最多 5 token,**代码编辑快 6–11%,其他文本不变**,且**输出不变**(草稿只提议);其文档明确标注"**只在实测接受率与成本划算处启用**"。
- **动作**:① 离线测**接受率**(不需要端到端):在我们的真实"改文件"请求上,统计"下一步 token 出现在前文"的比例;② 查当前 vLLM 版本是否已有 n-gram / prompt-lookup proposer 可直接配置。
- **可行性**:✅ 中高。**取决于上游 vLLM 是否提供**;若没有,自研 proposer 要进 patch 系列。
- **代价**:低-中。**无任何数值门槛** —— 草稿只提议、由目标模型 verify ⇒ 输出语义不变(R40:这类差异**本就不计入代价**)✓;显存代价**几乎为 0**(ngram 不需要模型)。⚠️ 唯一真实代价:会占用一部分 **verify 窗口的宽度**,而我们的 MBT/激活 ∝ chunk ⇒ **需确认不会推高激活**;另有对"上游 vLLM 是否提供 ngram proposer"的依赖。
- **⚠️ 口径澄清**:Strata 那句话(原文 *"the output is the same"*)指的是**草稿被 verify ⇒ 语义等同**,**不是**"逐位相同";我们**不需要**逐位相同(R40)⇒ 本条**没有数值侧的否决条件** ✓
- **判据**:真实"改文件"请求的**接受率 + 端到端墙钟**配对 A/B。**风险**:低。

### R7 (P2) 会话边界检查点(治"改写最后一条用户消息")

- **背景**:LMCache 因**长 prompt 崩溃 + 输出退化**被禁 ⇒ 重启要 **~21 min 重预填** ⇒ "重启 = 40 min"成为硬约束。⚠️ 但要注意:**我们的痛点是"跨重启",Strata 的 parking 是"跨请求、在内存里"** ⇒ **它不解决跨进程重启**。真正对应 LMCache 角色的是 SSD 持久化,**Strata 没有这个**。
- **Strata 证据**:`MESSAGE_BOUNDARY_CACHE` 专治"**保留同一段长历史、改写最后一条用户消息**":首次完整编辑 **17.236 → 11.527 s(−33.1%)**,可复用前缀 **22,016 → 32,831 token**;**`PROMPT_CACHE_TAIL`** 则在 prompt 尾部就近存一个检查点(仅一个、优先被驱逐)。
- **动作**:① 先量**本机** `prefix cache` 命中率与重预填的真实构成(是 GPU 预填主导还是 CPU 预填主导);② 评估 **vLLM 原生 prefix caching + 我们的 CED** 能否覆盖"改写末条"这一形态(不必引入第三方)。
- **可行性**:✅ 中高(**先查口径,不动代码**);⚠️ 实现若要动 vLLM 前缀缓存,需上游化或进 patch。
- **代价**:小-中。**风险**:⚠️ 不要重现 LMCache 的老路;**禁止**把"升级/改 LMCache"当解法(AGENTS 第 1 条第 7 项)。
- **判据**:同一条"改写末条消息"的请求,配对 A/B 的**首 token 时间**。

### R8 (P2) 台账加"**证据等级**"列 + 结果的离线收据

- **背景**:我们的台账体系(`docs/EXPERIMENTS.md` B 编号、`IRON_RULES.md` R 编号、`BUG_REGISTRY.md` 可跑判据、QA 错题本)**在"留下可执行判据"上强于 Strata**;差距在**"哪些结论是实测、哪些是读码推算"没有逐条标注**。
- **Strata 证据**:每篇特性文档固定标注 ① 结论来自**读代码**还是**实测** ② 验证环境 ③ 数字 ④ ⭐ **还没验证什么**;并附**可离线复算**的收据(JSON + 一段 SHA256 校验脚本)。
- **动作**(极低成本):① 在新台账条目里加一列 **`证据等级 = 实测 / 读码 / 推算`**;② 对关键 A/B **落一个收据文件**(输入、命令、md5/时间戳),使其**不依赖会话上下文即可复算**。
- **可行性**:✅ **高**;**代价**:极低(改模板)。**判据**:下一条 B 记录即带"证据等级"。

---

## 5. 不建议照搬的部分

| 项 | 为什么不能照搬 |
|---|---|
| **2–3 bit GGUF 量化档** | 我们跑官方 **FP8 稠密 + FP4 专家**;专家字节数差 **2–3 倍** ⇒ "全部专家常驻 RAM + CPU 算 miss"的 RAM 账完全不同 |
| **层切分代替 TP** | 它用层切分是因为**消费卡的显存与 PCIe 拓扑**;**TP=2 是 A100 的既定口径**,两者对"专家驻留"的收益曲线不一样 |
| **`H=2560 / FF=640 / BLOB=1,382,400` 编译期几何** | 它是**单模型专用引擎**,几何不合就**拒绝**;我们必须保持**任意 MoE 架构**与主线 vLLM 兼容 |
| **262K 上下文上限** | 我们要 **1M**,KV 流式与 chunk 定档的约束比它紧 |
| **`POST /v1/vram` 式"可回升"伸缩** | 我们**已定案"不允许回升"**(为消除 OOM 粘滞被 pop 掉后每 forward 重试的风险)⇒ 两者哲学相抵,**但"留白"这一半可以借**(见 R1) |
| **conversation parking 当 LMCache 的替代** | 它是 **in-RAM、跨请求不跨进程**,**不治"重启后重预填"** |

---

## 6. 推进建议(与 R23 阶梯对齐)

```text
第 0 步(零机时、零风险、今天可做)
  R1①  XIAOTU_GP_ACT_RESERVE_GIB 提到实测值 + 启动打印 free 并告警
  R1②  在 SPEC_mbt_budget 里把"留白"写成第一类约束
  R4    用已有 8.7 s/chunk 数据离线估算"chunk 间插入解码"的 Pareto 曲线
  R8    台账加"证据等级"列
  —— 以上都不重启、不改生产代码

第 1 步(离线分析,不重启)
  R3    routing 覆盖率曲线(top-N 专家 vs GiB),用留一口径
  R6①  真实"改文件"请求上的 prompt-lookup 接受率
  R7①  本机 prefix cache 命中率与重预填构成的口径核实

第 2 步(攒批后一次性重启 —— 需用户当次许可)
  R2    SPEC_mbt_budget 改动 ①②③④(≈40 min,须许可)
  R5①  门槛扫描(用临时实例,不占生产)

第 3 步(仅在前面给出正收益才立项)
  R3 → 细粒度专家槽位(实现量大)
  R4 → chunk 间解码窗口(需动上游调度)
  R6 → prompt-lookup proposer
```

⚠️ **R2 是唯一"动了会影响生产"的一项**;其余全部可在它之前、以零生产代价完成。**建议先清空第 0 步**,再决定 R2 的窗口。

---

## 7. 出处与局限

**主要出处**(Strata,均在其 `main`):
- [`docs/DETAILS.md`](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md)(专家缓存预算 / `0 MiB free` 实测 / 低内存模式谱系 / CPU share / 投机三件套)
- [`include/strata/core/expert_cache.hpp`](https://github.com/Niko1221/Strata/blob/main/include/strata/core/expert_cache.hpp)(CPU 专家池 16.2 ms 的立项算式 / `h_expert` 留一口径 / profile 机制)
- [`docs/BATCHING.md`](https://github.com/Niko1221/Strata/blob/main/docs/BATCHING.md)(batch slots / decode share / BYIELD / token 级精确性)
- [`docs/MULTI_GPU.md`](https://github.com/Niko1221/Strata/blob/main/docs/MULTI_GPU.md)(层切分 / pipeline-windows / trim-stage-weights)
- [`docs/VRAM_ELASTIC.md`](https://github.com/Niko1221/Strata/blob/main/docs/VRAM_ELASTIC.md) · [`docs/MESSAGE_BOUNDARY_CACHE.md`](https://github.com/Niko1221/Strata/blob/main/docs/MESSAGE_BOUNDARY_CACHE.md) · [`docs/PROMPT_CACHE_TAIL.md`](https://github.com/Niko1221/Strata/blob/main/docs/PROMPT_CACHE_TAIL.md) · [`docs/EXCHANGE_ROTATION.md`](https://github.com/Niko1221/Strata/blob/main/docs/EXCHANGE_ROTATION.md) · [`docs/BATCHED_DMA.md`](https://github.com/Niko1221/Strata/blob/main/docs/BATCHED_DMA.md) · [`docs/KV_PREFETCH.md`](https://github.com/Niko1221/Strata/blob/main/docs/KV_PREFETCH.md) · [`docs/MODELS.md`](https://github.com/Niko1221/Strata/blob/main/docs/MODELS.md)

**本项目内部出处**:`README.md`(专家层常驻的否决记录 / 显存优先级)、`docs/EXPERIMENTS.md`(B277/B279 长 prefill 饿死解码;B2/B80/B81 MBT 曲线)、`docs/PRODUCTION_8070.md`、`docs/RUNBOOK.md`、`docs/KNOWN_LIMITATIONS.md`、`dev-docs/HANDOFF_DEGEN_AND_MBT_2026-10-06.md`(§4.0 推迟裁决)、`dev-docs/SPEC_mbt_budget.md`、`dev-docs/report/tuning/IRON_RULES.md`(R34/R35)、`dev-docs/BUG_REGISTRY.md`(A58–A65)、`dev-docs/USER_QA_LEDGER.md`(#58–#63)。

**局限**(重复 §0 声明):本报告**未运行** Strata,其全部数字为**自报**;其量化档位与本项目差 2–3 倍,**跨项目的绝对数字不可直接比较**,本报告只把**机理与相对差值**作为参考;R1/R4/R5 的代价评估含**推算成分**,需以本机实测覆盖。
