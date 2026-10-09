# vllm-xtu-moe

> **XTU = X Transformers Unity**(读音:汉语「小兔」)—— 一个**高性能 MoE 推理加速层**。
>
> 这是一个面向**超大 MoE 模型**的 **vLLM 插件**,核心思路是把计算拆开:
>
> * **GPU** 负责注意力、KV cache、embedding 等**非专家部分**;
> * **CPU** 负责**专家权重**与**专家计算**;
> * 对**长 prompt**,还会把**一部分专家计算流式搬到 GPU**。
>
> 也就是说:在**"显存放不下专家权重"**的情况下,仍然让大 MoE 模型跑起来,
> 而且**尽量不改主线 vLLM**。
>
> **项目重点**
>
> * **DeepSeek / GLM / MiMo** 这类**超大 MoE**
> * **CPU + GPU 混合推理** · **低显存需求**
> * **高性能长上下文推理**
> * **兼容主线 vLLM,不需要 fork**
> * 主要面向**大模型部署与工程优化**

[**English**](README_EN.md) · 中文(默认)

> **📌 当前版本：v0.2.7**（2026-10-09）—— 优化引擎：削减 CPU→GPU DMA 缓冲区，节省 11 GB 显存（总），性能无损失
> [发行说明](RELEASE_NOTES_v0.2.7.md) · [工作报告](docs/WORK_REPORT_2026-10-09.md)

---

## 项目特色

1. **支持在单张显卡（>40 GB）且内存充足的设备上运行超大 MoE 模型**（已验证 DeepSeek-V4.1-Flash、GLM-5.3-Flash、MiMo-V2.6-Flash-RL 与 Qwen3.8-Flash-Next），其他 MoE 模型也都支持。
2. **自动 CPU/GPU Prefill 分流**：对长 prompt（阈值可设）使用 DMA 双缓冲把权重流式传入显存，**吃满 PCIe 带宽**（已在 2× Tesla A100 40 GB 上验证：两组 PCIe 4.0 x16 均达到 **25 GB/s** 的上限），最大限度保证 GPU 侧的 Prefill 性能。
3. **支持 SM80 等较老的显卡**，支持 **NVFP4 / MXFP4** 量化权重；硬件不支持时，权重会**反量化后按最优路径计算**。
4. 支持 **AVX-512 VNNI 等多种 ISA**，自动选择最优的 CPU 计算后端。
5. **硬件感知的 NUMA 切片**：按 NUMA 拓扑把进程均匀分布到核心，并把 MoE 权重做对应分片，**每个 node 的计算只访问本地内存**，避免跨 socket 访存；同时**内存总占用仍为单份权重**。

## 实测平台

下文所有性能数字都在这台机器上取得:

| 项 | 配置 |
|---|---|
| CPU | 2× AMD EPYC 9654(192 物理核 / 384 线程,8 NUMA node) |
| 内存 | 1538 GiB DDR5 |
| GPU | 3× NVIDIA A100-PCIE-40GB |
| 系统 | Ubuntu 22.04 · conda env `lvllm` |

---

## 性能

* 口径：**聚合吞吐**——`prefill (tok/s) = 并发数 × prompt_tokens / TTFT`、`decode (tok/s) = (输出 token 数 − 并发数) / (末 token 时刻 − 首 token 时刻)`
* 数据集：**random 随机 token**，`--random-input-len` 固定为短 128 / 长 16384，输出 128；前缀缓存开；每格 8 个请求、**每格不同 seed**。
* 表中「实际 token」= `total_input_tokens / completed`（含 chat template 的少量开销）。
  配置细节、判据与 `MBT` 取舍见 `docs/TUNING_GUIDE.md` §8 与 `docs/EXPERIMENTS.md`。

| 模型(最优配置) | prompt | 实际 token | 并发 | prefill (tok/s) | decode (tok/s) |
|---|---|---|---|---|---|
| **GLM-5.3-Flash**<br>TP=2 · util 0.85 · **MAXLEN 262144**<br>MBT 12288 · MTP 关 | 短 | 140 | 1 | **115.2** | **22.0** |
| | 短 | 140 | 2 | **140.8** | **27.8** |
| | 长 | 16,396 | 1 | **266.3** | **21.4** |
| | 长 | 16,396 | 2 | 待测 | 待测 |
| **DeepSeek-V4.1-Flash**<br>TP=2 · util 0.85 · **MAXLEN 262144**<br>MBT 8192 · **CED 开**（长两行） · 投机解码关 | 短 | 128 | 1 | **254.3** | **19.8** |
| | 短 | 128 | 2 | **350.4** | **28.0** |
| | 长 | 16,384 | 1 | **903.9** | **19.3** |
| | 长 | 16,384 | 2 | **1204.2** | **15.4** |
| **DeepSeek-V4.1-Flash**<br>TP=2 · util 0.90 · **MAXLEN 524288**<br>**MBT 8192** · **seqs 4** · **DSpark k=5**（v0.2.7 生产配置） | 短 | 128 | 1 | **261.1** | **15.3** |
| | 短 | 128 | 2 | **269.8** | **11.1** |
| | 短 | 128 | 4 | **276.2** | **6.6** |
| | 长 | 16,384 | 1 | **1011.0** | **13.2** |
| | 长 | 16,384 | 2 | **1010.1** | **9.7** |
| | 长 | 16,384 | 4 | **1005.0** | **7.0** |
| **MiMo-V2.6-Flash-RL**<br>TP=2 · util 0.85 · **MAXLEN 131072**<br>MBT 8192 · **seqs 4** · **MTP k=1** · **GPU 预填开(门槛 4096)** · 含形状预热 | 短 | 128 | 1 | **232.0** | **30.5** |
| | 短 | 128 | 2 | **367.5** | **40.7** |
| | 长 | 16,384 | 1 | **811.3** | **27.3** |
| | 长 | 16,384 | 2 | **1455.3** | **39.6** |


GLM / DeepSeek-V4.1 未开投机解码（random 数据集对投机是最坏情况，且会挤占长上下文的显存）；**MiMo-V2.6 开了 MTP k=1**。MiMo 的 prefill 数字是 **GPU 预填开启后**的口径（`VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=4096`；该变量不设或为 0 表示**关闭**，会让 prefill 掉到 ~310，见 `docs/MODEL_GUIDES.md` §0.1）。
> ⚠️ **单引擎混跑"长预填 + 解码"会饿死解码**(实测,**B277**/**B279**):正在解码的请求,其 **ITL 下界 = 相邻两次解码步之间排入的预填量 ÷ 预填吞吐**。实测注入一个 32768-token 的 prompt 后,该解码请求的 ITL 从 **中位 64.8 ms 暴涨到 ~8.4 s,并持续整个 57.6 s 预填窗口**(它是每步只拿到 1 个 token,而每步被一个 `MBT` 大小的预填 chunk 撑满)。
>   * 压小 chunk(`--long-prefill-token-threshold`)会**同时毁掉预填效率**(每个 chunk 都要重新流式过 40 层权重:2048-token chunk 仅 ~410 tok/s,8192 的 ~1000)⇒ **净效果更差**(解码吞吐 1.25 → 0.57 tok/s)⇒ **没有可用的调度旋钮** ✗
>   * **单流交付(1M × `seqs=1` 的口径)**:`--max-num-seqs 1` ⇒ 请求串行、不存在混跑 ⇒ 单流 ITL 稳定在 ~65 ms ✓<br>⚠️ 这只是**档位之一** —— 按场景选 **1M×1 / 512K×2 / 256K×4**,以及**生产在用的 `512K`×4**(高天花板 + 短任务多路;⚠️ **满载只有 ≈2 路满长**,其余**排队**、不报错)。判据:`seqs × maxlen ≤ 池容量` ⇒ 满载,超过 ⇒ **只排队不报错**(⚠️ 是**饱和阈值**,不是硬不变量),见 `docs/MODEL_GUIDES.md` §0.1b ✓
>   * **多用户并发**:应做 **prefill/decode 分离(PD disaggregation)**,而不是调 `max-num-seqs` / `max-num-batched-tokens` / `long-prefill-token-threshold` ✓
>   * **根因**:预填吞吐受 **DMA/装配**限制(每个 chunk、每卡、每层都要重流约 **3.84 GB** 权重;设备侧的 **strided 转置**实测仅 **84 GB/s**,而连续布局可达 **1361 GB/s**;装配步骤占 **250 ms/层 = 85%**,而该层的 MoE 计算只要 **5–50 ms**)⇒ 提高预填吞吐要靠**消除 strided 转置 / 融合装配**,不是调度 ✓
> ⚠️ **"`--max-num-seqs` 必须 ≥2"只对【基准口径】成立**:早期用 `seqs=1` 跑 `C=2` 会**串行**、把短 C=2 的聚合 prefill 压到 54(那是**配置问题,不是模型问题**,判据 **B124**)⇒ 做**并发基准**时 `seqs≥2`;做**单流交付**时 `seqs=1` 才是对的 ✓


## 快速开始

```bash
# 1) 主线 vLLM(本插件是插件,不需要 fork)
pip install vllm==2.5.0

# 2) 本插件(发行名 vllm-xtu-moe,**不发 PyPI**;二选一)
# (a) 从 GitHub Release 附件装:wheel 里已含 6 个 ISA 变体,无需本地编译器
pip install ./vllm_xtu_moe-0.2.3-cp312-cp312-manylinux_2_34_x86_64.whl
# (b) 源码安装(需要本地编译器):
# CXX=g++-16 PYTHON=$(which python) bash scripts/build_engine_variants.sh && pip install -e .

# 3) 跑一个专家权重放不进显存的 MoE 模型
VLLM_EXPERTS_LOAD_DEVICE=cpu \
vllm serve <MODEL_DIR> --tensor-parallel-size 2 --enable-expert-parallel \
  --max-model-len 65536 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95
```

逐模型配方、显存核算、自检与排错 → **[`docs/RUNBOOK.md`](docs/RUNBOOK.md)**、
**[`docs/MODEL_GUIDES.md`](docs/MODEL_GUIDES.md)**。

---

## 版本变更(简)

| 版本 | 主题 |
|---|---|
| ~~v0.2.7~~ | **未发布**(2026-10-08)—— 当天无新性能能力;`MBT 6144→8192` 属已知配置收益 ⇒ 决定不发 |
| **v0.2.6** | **自举开发：INT8 激活 × FP4 专家权重 的计算路径** —— 为 **MXFP4** 专家权重增加 **INT8 激活**路径（fp4 码值全含于 int8 ⇒ **权重侧零代价**）：内核级 **24.9 → 15.1–15.7 ms/层（1.54–1.62×）**；端到端（qfn·MXFP4-FP8）GSM8K **两臂 193/200**、逐题一致 194/200（净 0）、256K 四针两臂 4/4 + `stop`、Vision 23 逐项一致 18/23；修复**回退路径回归**（慢 16–22% ⇒ +1.49% 指令）；打通 **qfn 的 MTP**（**2.44×**）；qfn 默认线程数 **120（5 核/CCD）**；路径默认关（`XIAOTU_MOE_INT8=1`）|
| **v0.2.5** | **优化 GPU prefill 内存机制** —— **为 GPU 算子增加对 NUMA 切片权重的处理路径,节省了一层权重空间**(对 DeepSeek-V4.1-Flash **节约 6.33 GiB**;2 个 GPU 时**每卡节约 3.16 GiB**):单层 staging 10.73→**7.56 GiB/rank**、离线同层 364.91→**341.31 ms**、数值逐字节相同;**支持 1M 上下文 + GPU 预填充**(CED 让每 token KV 5437→2106 B,1M 只需 KV ~2.2 GiB);**Engram pinned 浪费修复**(宿主内存 **−75 GiB**);rebase 到上游后 **CED 恢复生效**(16k prefill 527.8→**982.4** tok/s);**TP=2 一律用 GPU1+GPU2**(GPU0 只有 x8);服务日志按 **PID** 命名 |
| **v0.2.4** | **支持 MiMo-V2.6-Flash-RL + 性能优化** —— 新模型接入（MXFP4/TP=2/1M/多模态/MTP k=1，**层内数值门禁 94/94 层通过**、312K 针测试命中、多模态真图通过）；**GPU 预填充「0=关闭」陷阱修复**（长 prefill 313→**811** tok/s）；**`--max-num-seqs` 默认统一为 4**（短 C=2 prefill 236→**367**）；**decode 口径改为 median ITL**（真实争用 1.13–1.77×）；修好 MXFP4 上的层内数值门禁 |
| **v0.2.3** | **跟进上游 + GLM/MiMo 的 MTP** —— 补丁栈 rebase 到上游 `133b71e0b`(11 补丁/40 文件);**GLM-5.3-Flash 的 MTP 落地**(draft 层识别 + GPU 常驻,`SPEC_K=1..4`);**MiMo-V2.5(310B/15B)单卡端到端支持**,MTP k=1 decode +10%;显存契约重标定(GLM `GPU_UTIL` 0.85 → **0.82**) |
| **v0.2.2** | **支持 GLM-5.3-Flash** —— FP8 GPU 预填充接线(4K prompt TTFT 29.3 → 22.8 s)、256K × 2 路并发交付配置;修掉 e4m3 次正规数解码缺陷 + 新增全码字门禁 |
| **v0.2.1** | **针对引擎的显著性能优化** —— CPU MoE 引擎在全部真实形状上**反超 `lk_moe`**;DeepSeek-V4-Flash 同步受益 |
| v0.2 | DeepSeek-V4.1-Flash 全链路可用(1M 上下文 + GPU 预填充 + 投机解码)+ CPU 预填充路径优化 |
| v0.1.0 | 首个公开版:混合模式(CPU 专家 + GPU 其余)、AVX2 / AVX-512 多 ISA、DeepSeek-V4 系列 |

改动清单、性能对照与运行参数变更:
[**v0.2.6**](RELEASE_NOTES_v0.2.6.md) · [**v0.2.5**](RELEASE_NOTES_v0.2.5.md) · [**v0.2.4**](RELEASE_NOTES_v0.2.4.md) · [**v0.2.3**](RELEASE_NOTES_v0.2.3.md) · [**v0.2.2**](RELEASE_NOTES_v0.2.2.md) · [**v0.2.1**](RELEASE_NOTES_v0.2.1.md) · [**v0.2**](RELEASE_NOTES_v0.2.md) · [**v0.1.0**](RELEASE_NOTES_v0.1.0.md)
(归档:[v0.2pre](RELEASE_NOTES_v0.2pre.md))

---

## 文档

**面向使用者 —— [`docs/`](docs/)**

| 文档 | 内容 |
|---|---|
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | **运行手册**:安装、启动参数、显存核算、GPU 预填充配方、JIT 缓存、发布流程 |
| [`docs/MODEL_GUIDES.md`](docs/MODEL_GUIDES.md) | 逐模型的资源需求、启动命令与性能 |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | 实测数据与复现方式 |
| [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md) | 已知限制与不支持的组合 |
| [`docs/INSTALL_MAINLINE.md`](docs/INSTALL_MAINLINE.md) | 主线 vLLM 环境准备 |
| [`docs/FP4_INT8_HARDWARE_OUTLOOK.md`](docs/FP4_INT8_HARDWARE_OUTLOOK.md) | **4-bit 推理格式的公开硬件路线图**(FP4 vs INT4;含出处与"已宣告/传闻"标注)与 **CPU 优先 MoE 服务的设计依据** |
| [`docs/INT8_ACTIVATION_FOR_MXFP4_EXPERTS.md`](docs/INT8_ACTIVATION_FOR_MXFP4_EXPERTS.md) | **INT8 激活 × MXFP4 专家权重** —— 这条计算路径解决什么问题、内核级收益、端到端语义验收口径、启用方法与已知限制 |
| [`docs/AUXILIARY_MODEL_EVALUATION.md`](docs/AUXILIARY_MODEL_EVALUATION.md) | **辅助压缩模型评测(负结果)** —— 为什么不采用小模型做长会话压缩;含 0.6B/1.7B/4B 的真实速度、精度、崩坏率与每 token KV 开销的客观数据 |

**面向开发者** —— 架构与主线集成、GPU 预填充实现、上游漂移、调优记录与全部内部报告,
**属于内部开发文档,不随本仓库发布**(只保留在本地工作副本里)。

---

## 关于作者

独立研究者。曾担任一台 **10 PFLOP/s(双精度)超级计算机**的主要设计师和运营者,此前长期从事高性能计算方向的工作。

这正是本项目关注点的来源:**NUMA 拓扑与内存布局、带宽受限内核、主机侧数据搬运** —— 也正是"把巨型 MoE 模型放到常规服务器上跑"这件事今天真正的战场。

## 致谢

本项目受到 **KTransformers** 和 **Lvllm** 项目启发,尤其是计算引擎 **`xiaotu-moe`** 深度借鉴了 **lk-moe** 思路,
**`xiaotu-moe` 全部代码均为独立编写**,特此致谢。

* KTransformers —— <https://github.com/kvcache-ai/ktransformers>
* Lvllm(及其 CPU MoE 引擎 **lk-moe**;本项目对照的 `lk_moe` 2.4.2 即来自此仓库)—— <https://github.com/guqiong96/Lvllmds4-x>
* 同时感谢 **vLLM**(<https://github.com/vllm-project/vllm>)提供的插件式扩展点,使本项目无需 fork 即可接入。

---

## 许可

Apache-2.0。第三方组件与致谢清单见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) 与 [`NOTICE`](NOTICE)。