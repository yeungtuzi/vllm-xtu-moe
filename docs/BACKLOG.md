# 待办总账 BACKLOG — vllm-xtu-moe

**作者 Author:** 大河马 (dahema@me.com),DeepSeek Harness 辅助

> **本文件是全项目唯一的待办/决策总账(single source of truth)。**
> 用户 2026-09-09 要求:「把讨论过、被列成待办的事都记下来,下次聊到时去召回,
> 并且随项目进度不断更新。」

---

## 0. 怎么用这份文件(召回协议)

1. **每次新会话/新话题开始**,先读本文件;讨论中直接引用条目 ID(如 `T04`、`D9`)。
2. 状态词表:`待办` / `进行中` / `已完成` / `阻塞` / `已决断` / `已放弃`。
   - 已完成、已放弃的条目**保留在表里**(带日期和结论),避免重复讨论。
3. 每次改动本文件:①改条目状态;②在 §7 修订记录追加一行(日期 + 改了什么 + 依据)。
4. 实测数字一律落到 `/home/user/lvllm/results.txt` 和 `report/`,本文件只放结论 + 指针。
5. 本文件不替代 `docs/ROADMAP_GENERALITY.md`(通用化方案细节)与
   `docs/EXPERIMENT_REPORT.md`(论文式报告);它是这两者的**索引 + 待办状态**。

## 0b. 项目关系(务必先读,别把两个项目搞混)

| 项目 | 是什么 | 状态 |
|---|---|---|
| **`vllm-xtu-moe`**(本仓库) | **vLLM 主线插件**:混合推理(专家权重放 CPU、注意力/长 prefill 留 GPU),按量化格式把 CPU 计算后端挂到主线 `FusedMoEFactory` | 对外发布 |
| **`xiaotu-moe`**(独立仓库) | 独立 **CPU MoE 引擎项目**:闭源 `lk_moe` 的开源重实现、在 `Lvllm`/`Lvllmds4-x` fork 里做 drop-in 替换 | **已转私有、仅作参考**;其引擎源码作为本仓库内置内核(`xiaotu_moe/` + `csrc/`) |
| **`Lvllmds4-x`**(第三方 fork) | 别人的 vLLM fork | 仅 PR2/PR3 的代码来源(Apache-2.0 署名);不是运行依赖 |

- 本仓库里 `integration/`、`docs/XIAOTU_MOE_REPORT_*.md`、`docs/THREAD_GEOMETRY.md`、
  `docs/compute_perf_compare.md`、`docs/TODO_LONGTERM.md` 等是 **xiaotu-moe 时期历史材料**,
  已加醒目标记,只作参考。
- 一句话:**引擎来自 xiaotu-moe,项目本身是 vllm-xtu-moe,运行环境是 vLLM 主线(不是 fork)。**

### 快速状态看板(截至 2026-09-09 夜)

| 主题 | 状态 | 一句话 |
|---|---|---|
| 通用 CPU experts 后端(格式无关) | 🟢 主干已通 | `mixed_experts.py` 走主线 router;**BF16/FP8/MXFP4/INT4 四种格式**都被 oracle 选中 |
| 混合模式 oracle 前置(主线补丁) | 🟢 已完成 | `oracle/{fp8,int_wna16,unquantized}.py` 都已改(+跳过 AMX prepack);mxfp4 由 PR1 覆盖;nvfp4/mxfp8 主线无 CPU 后端 |
| 引擎 clamped SwiGLU(`swiglu_limit`) | 🟢 已完成并验证 | `activation_type=1`;BF16/MXFP4/真实 GLM fp8 三路数值验证通过 |
| 引擎 e4m3 subnormal 解码 bug | 🟢 已修 | `2^-7 → 2^-6`,与 `torch.float8_e4m3fn` 逐字节一致 |
| GLM-5.3-Flash 端到端 | 🔴 阻塞(硬件) | A100/SM80 无任何支持其 MLA 维度(256/0/256)的 attention 后端 |
| 通用路径端到端(真实模型) | 🟢 BF16 已过 / 🟡 FP8 在跑 | bf16 tiny-Mixtral:greedy token 48/48 与 GPU 一致;Qwen3-30B-A3B-FP8 加载成功(oracle→CPU FP8) |
| **目标模型 Qwen3.8-Flash-Next-FP8** | 🟡 下载中 | 用户确认的精确目标(185 GB,512 专家 top-10,fp8 block-128);预检:主线支持该架构、无 SM90 硬门槛 |
| 上游 PR(PR1/2/3) | 🧊 **已冻结(D9=A)** | 三个 draft 全部不动,等通用 CPU 后端完成后再统一重排(见 §3 D9) |
| RFC(layerwise GPU prefill / 通用 CPU 后端) | 🔴 未发 | 草稿在 `patches/rfc_layerwise_gpu_prefill.md` |

---

## 1. 进行中(In progress)

| ID | 事项 | 状态 | 备注 |
|---|---|---|---|
| T02 | 通用路径端到端验证(真实模型) | 🟡 进行中 | ①bf16 等价性测试已过(见 T30);②Qwen3-30B-A3B-FP8 已加载成功;③目标模型 Qwen3.8-Flash-Next-FP8 下载中 |
| T31 | 目标模型 Qwen3.8-Flash-Next-FP8 的端到端(下载 185 GB) | 🟡 下载中 | 预计 ~1–2 h;下载完立刻跑 `scripts/fp8_moe_smoke.py`(SMOKE_MODEL=...) |

---

## 2. 待办清单

### A. 通用 CPU experts 后端(主线接口层)

| ID | 待办 | 为什么 | 优先级 |
|---|---|---|---|
| ~~T01~~ | ~~混合模式 oracle 前置推广~~ | ✅ **已完成**:`oracle/{fp8,int_wna16,unquantized}.py` 都加了混合模式分支,**并且**跳过后端的 AMX prepack(`prepare_*_for_cpu`);nvfp4/mxfp8 主线无 CPU 后端,无需改。补丁快照:`patches/mainline_mixed_mode_generic.patch` | 冻结期不入 PR |
| **T02** | 用**能在 A100 跑通**的模型做通用路径端到端(load → 连贯输出 → TTFT/吞吐) | 通用路径至今没在真实模型上端到端跑过(DS-V4 一直走 OOT 覆盖) | **P0** |
| ~~T03~~ | ~~BF16(无量化)CPU 后端~~ | ✅ **已完成**:`XiaotuCPUExpertsBF16(CPUUnquantizedExperts)` + `oracle/unquantized.py` 前置;tiny-Mixtral 端到端验证通过 | – |
| ~~T04~~ | ~~激活守卫~~ | ✅ **已完成**:`_supports_activation` 只允许 `SILU` + `SWIGLUOAI_UNINTERLEAVE`(都是 packed 布局),**拒绝 `SWIGLUOAI`**(gpt-oss 交错布局);4 个后端类全部验证 | – |
| **T05** | 支持专家并行(expert_map / EP) | 现在直接 `raise NotImplementedError`;TP=2 EP 是我们已测过的主力配置 | P1 |
| **T06** | monolithic `apply` 拿不到 `input_ids`(hash routing / DS-V4 部分层需要) | 要么上游加参数,要么改走 modular 路径;当前会报错而不是算错 | P2 |
| **T07** | INT4(WNA16)按 `quant_config` 取真实 group size(32/128)与零点 | 现在 `_group_n/_group_k` 硬编码 128/128 | P1 |
| **T08** | INT8 W8A8 格式(主线 CPU 已有,引擎缺) | 对齐主线 CPU 格式集 | P2 |
| **T09** | 长尾格式:MXFP8 / MXFP6 / FP8 e5m2 / GGUF k-quants / Marlin 布局 | 按用户需求增量 | P3 |
| **T10** | 引擎侧:`apply_router_weight_on_input` 支持(现在 raise) | 少数模型会用 | P3 |
| ~~T28~~ | ~~FP8 块缩放 dtype 归一化~~ | ✅ **已完成**:Qwen3 的 `weight_scale_inv` 是 **bf16**、GLM 是 fp32;`mixed_experts` 现在按格式转成引擎读的 dtype(fp32),MXFP4 保持 uint8 e8m0 | – |
| ~~T33~~ | ~~文档去混淆~~ | ✅ **已完成**:README 重写 + 新增「项目关系」;BACKLOG §0b;6 个 engine-era 文档加历史标记;ref/ 两篇加决策状态;fork 参考代码加头注释;pyproject/`__init__` 描述校正 | – |
| ~~T34~~ | ~~第三方署名文件~~ | ✅ **已完成**:仓库根目录已有 `THIRD_PARTY_NOTICES.md`(vLLM / Lvllmds4-x / ktransformers / lk_moe / 模型数据 5 节),本轮更新了引擎引用路径 | – |
| ~~T35~~ | ~~仓库根目录去混淆~~ | ✅ **已完成**:①`xiaotu-moe/` 目录从本仓库 **untrack**(64 文件,本地保留 + gitignore);②根 `README.md`/`README_CN.md` 重写为「vLLM 主线插件」首页(删掉"一个项目,两个部件"与"lk_moe 开源重实现"的项目定位);③`THIRD_PARTY_NOTICES.md` 引用改为内置引擎路径 | – |
| **T36** | 仓库根目录剩余 legacy 文件处置(见 D13) | 根目录还有 `LK_MOE_*.md`(4)、`HANDOFF.md`、`serve_*/bench_*/silent_*`、`analysis/` 等 xiaotu-moe/fork 时期文件;是否一并 untrack 待用户定 | P1 |
| **T37** | git 历史里仍含 `xiaotu-moe` 内容(见 D13) | untrack 只影响未来提交;要真正从公开仓库消失需 filter-repo + force push | 待定 |
| **T29** | 上游贡献点:主线 fp8/wna16 的 `process_weights_after_loading` **不调用** experts 的钩子(只有 unquantized 调) | 我们在本地补丁里补上了调用(见 L8);冻结解除后可作为 PR1 的配套小改动 | P1 |

### B. 引擎 / 内核

| ID | 待办 | 为什么 | 优先级 |
|---|---|---|---|
| **T11** | 打包全部 ISA 变体 + `xiaotu_moe.self_check()` | 🟡 **部分完成**:`scripts/build_engine_variants.sh` 已内置(5 个变体实测构建成功,~90 s,不依赖外部引擎仓库);还差:把 5 个 `.so` 纳入 wheel + 自检 | P1 |
| **T12** | 格式 × ISA 测试矩阵(用 `scripts/_run_one_variant.py` 驱动) | 保证多 ISA 打包后不回归 | P1 |
| **T13** | AMX 接线(`csrc/moe/amx_gemm.cpp` 3054 行已存在,未接入;需 SPR/EMR 机器) | 性能上限最高,但已实测瓶颈不在算力;需硬件 | P2 |
| **T14** | 决定性实验:同款 batched MXFP4 内核的 "skip-decode" 版本,量化反量化占比 | 目前**反量化占比未测**(见 §4 已证伪结论) | P2 |
| **T15** | 交错 gate/up 布局支持(gpt-oss 的 SWIGLUOAI) | 让 gpt-oss 系模型可用;需要引擎侧改 gate_up 取数 | P2 |

### C. 实验与测量

| ID | 待办 | 为什么 | 优先级 |
|---|---|---|---|
| **T16** | 长 prefill / 并发 / 启动成本 / MoE 内核 的 round-2 收尾项 | 见 `docs/OPTIMIZATION_ROUND2.md`;① ② ④ ⑤ 已有结论,剩余可做项在其中标注 | P2 |
| **T17** | 把今天的新结论写进 `EXPERIMENT_REPORT.md`(通用后端一节 + GLM 阻塞) | 报告是主交付物,目前缺这一段 | **P0(文档)** |
| **T18** | `results.txt` 追加今天全部实测与命令 | 审计/复现 | P1 |
| **T19** | PR3 的 A100 运行时验证(分支 rebase 后未跑) | PR3 描述里已注明待补 | P1 |
| ~~T30~~ | ~~CPU 专家 vs GPU 专家的端到端数值等价性~~ | ✅ **已完成**(`scripts/tiny_moe_equiv.py` + `scripts/make_tiny_mixtral.py`):bf16 tiny-Mixtral,greedy token **48/48 与 GPU 一致**,prompt-logprob max\|Δ\|=0.053 / median 0.016(峰化权重),Δ 始终小于 top1-top2 间距 | – |
| **T31** | 目标模型 `Qwen/Qwen3.8-Flash-Next-FP8` 端到端(185 GB) | 🟡 下载中(~25 MB/s,约 2 h);下载完跑 `SMOKE_MODEL=... scripts/fp8_moe_smoke.py` | **P0** |
| **T32** | Qwen3.8-Flash-Next 的 attention 在 A100 的实测可行性 | 预检:主线支持 `Qwen4ExpForConditionalGeneration`,QSA 是 Triton 稀疏注意力、GDN 是 Triton,未发现 SM90 硬门槛 —— 但**必须实测** | **P0** |

### D. 上游 PR / RFC

| ID | 待办 | 为什么 | 优先级 |
|---|---|---|---|
| **T20** | 冻结解除后重排 PR 集合(触发条件见 §3 D9):PR1 是否扩到 fp8/int4、PR3 是否拆分/降级、T21 是否单独提 | 已决 D9=A:冻结期不动,重排时复核 | 阻塞(等 D9 触发条件) |
| **T21** | 新增两个"上游 bug 修复"小 PR:①`cpu_moe.select_experts` 硬编码 `scoring_func="softmax"`;②CPU 内核忽略 `swiglu_limit`/`alpha`/`beta` | 这两个 bug 让主线自己的 CPU MoE 后端对 GLM/DS-V4 静默算错;小而独立,接受概率高 | P1 |
| **T22** | 发 RFC:①layerwise GPU prefill(草稿已就绪);②通用 CPU experts 后端 + 混合模式设计(需要上游的扩展点,而不是我们 monkey-patch) | D4a 已定"RFC 先行" | P1 |
| **T23** | 上游漂移巡检:`bash scripts/check_upstream_drift.sh` —— **冻结期只记录、不 rebase**;解冻后在 T24 统一 rebase | 上游 ~41 commits/天 | P1(只读) |
| **T24** | 解冻后:同步 `patches/UPSTREAM_PRS.md` + 三个 `pr*_body.md` + rebase 三个分支 | 保持快照与 GitHub 一致 | 阻塞(随 T20) |

### E. 文档与对外

| ID | 待办 | 为什么 | 优先级 |
|---|---|---|---|
| **T17** | 同上(报告补章节) | | P0 |
| **T25** | 对外改名收尾检查:README/docs/PR body 里是否还有该改的对外名(`vllm-xiaotu-moe` → `vllm-xtu-moe`) | 用户 2026-09-09 指令:对外名替换、内部名保留 | P2 |
| **T26** | 把 GLM-5.3-Flash 在 A100 不可用写进 README/报告(含精确报错) | 免得别人重复踩坑 | P1 |
| **T27** | 本文件(§0 召回协议)在每次会话结束时更新 | 用户明确要求 | 持续 |

---

## 3. 已决断 / 已放弃(不要再重新讨论)

| ID | 决定 | 日期 | 依据 |
|---|---|---|---|
| D1a | 先发布仓库,再谈 PR | 2026-09-09 | 用户 |
| D2a | PR3 从 fork 字节级搬过来 + 署名 | 2026-09-09 | 用户 |
| D3a | PR2 拆成独立小 PR | 2026-09-09 | 用户 |
| D4a | RFC issue 先行 | 2026-09-09 | 用户 |
| D5b | MXFP4 内核留在插件里,不上游 | 2026-09-09 | 用户 |
| D6 | 所有 PR 由我生成(draft),用户检查后点提交 | 2026-09-09 | 用户 |
| D7a | 删除全部 `XIAOTU_DEBUG_*` / `XIAOTU_TIMING`,保留内部名前缀 | 2026-09-09 | 用户 |
| D8 | 我们的部分 Apache-2.0,作者「大河马(BigHippo) dahema@me.com」;第三方按其许可 | 2026-09-09 | 用户 |
| **D9** | **A:三个 draft PR(#56118/#56119/#56120)全部冻结**,等通用 CPU experts 后端完成后再统一重排 | 2026-09-09 | 用户拍板 |
| **D10** | 端到端**目标模型** = `Qwen/Qwen3.8-Flash-Next-FP8`(185 GB,fp8 block-128,512 专家 top-10);线上没有叫 `Qwen/Qwen3.8-Flash` 的仓库(HF-mirror + ModelScope 都查过) | 2026-09-09 | 用户确认 |
| **D11** | 误下的 `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`(30 GB)**保留**,作为额外的 fp8 端到端验证 | 2026-09-09 | 用户选择 |
| **D12** | `xiaotu-moe` **转私有**(不再发布,仅作参考);`vllm-xtu-moe` 是**唯一对外项目**,定位=vLLM 主线插件 | 2026-09-09 | 用户决定 |
| **D13** | **待定**:①是否重写 git 历史以彻底移除 `xiaotu-moe` 内容;②根目录其它 legacy 文件是否一并 untrack;③是否把仓库根目录改成插件本身(现在根目录是 `/home/user/lvllm`,插件在 `vllm-xiaotu-moe/` 子目录) | 2026-09-09 | 用户提问(见下) |
| X1 | ~~"CPU 引擎瓶颈是反量化 ALU"~~ **已证伪**:预反量化 BF16 更慢(14×),真实路由下 1.73–2.0 TFLOP/s ≈ 峰值 15–18% | 2026-09-09 | `EXPERIMENT_REPORT.md` §7.2b |
| X2 | ~~"CPU MoE 是带宽敏感型"~~ **已证伪**:权重流量 ~10 GB/s,机器可做 740 GB/s | 2026-09-09 | 同上 |
| X3 | ~~"交叉点 2.5K token"~~ **已更正为 249/550 token**(单/双份权重) | 2026-09-09 | 报告 §5.6d |
| X4 | A2+B0 融合、NUMA page interleave —— 实测回归,不再盲试 | 2026-09-04 | `docs/TODO_LONGTERM.md` |

### D9 已决:三个 draft PR 全部冻结(用户 2026-09-09 拍板:方案 A)

> 用户原话:「当我们把支持大部分 MoE 模型(至少有 deepseek, glm, qwen 等)列为目标的时候,
> 我们对 vllm 要提交的 PR 就要重新考虑了吧,如果是这样,我就先不动这些 draft pr 了,
> 它们可能很快就会失效。」
> **决定:方案 A —— 三个 draft 全部冻结,等通用后端做完再统一重排。**

**冻结的含义(具体动作)**:

| 项 | 冻结期怎么做 |
|---|---|
| GitHub 上的 #56118/#56119/#56120 | **不点 Ready、不改描述、不关**;保持 draft 状态 |
| 三个分支 | **不主动 rebase**(避免把工时花在会再变的东西上);只记录漂移 |
| 漂移巡检 | `bash scripts/check_upstream_drift.sh` 定期跑,结果只记录到 `results.txt` / 本文件,**不修** |
| PR1 的 fp8/int4 扩展 | **不在主线打更多补丁**;扩展内容先落在插件侧 + 本文件的 T01 里,冻结解除时一起做 |
| 新增上游 bug 修复(T21) | 也一并冻结(等重排时决定是否单独提) |
| RFC(T22) | **不受冻结影响**——D4a 已定 RFC 先行,可以照发 |

**解除冻结的触发条件(建议,重排时复核)**:

1. 通用 CPU experts 后端主干稳定:T01(fp8/int4/mxfp4 oracle 推广)、T04(激活守卫)完成;
2. **至少一个非 DS-V4 的 MoE 模型**在 A100 上端到端跑通通用路径(T02);
3. 明确「混合模式 + 第三方 CPU 后端」在上游的设计归属(RFC 有反馈后再定 PR 边界);
4. 届时重排:重新评估 PR1 是否扩到 fp8/int4、PR3 是否拆分/降级、是否新增 T21 两个 bug 修复 PR。

**为什么冻结是安全的**:draft 不占资源、不会自动关闭;上游漂移(~41 commits/天)只影响分支,
而我们已演练两次 rebase(PR2 干净、PR1 一处冲突、PR3 结构性冲突需手工重 port)。
**真正的成本在 PR3**——它每次都要跟着上游 `common/ops/*` 重构重做一遍,这也是重排时优先复核它的原因。

---

## 4. 已知阻塞与限制(诚实记录)

| ID | 限制 | 证据 | 影响 |
|---|---|---|---|
| **L1** | **GLM-5.3-Flash 在 A100/SM80 上完全跑不起来**(与本插件无关) | `docs/evidence/glm53_a100_blocker.txt`、`logs/glm53_smoke.log`(sparse 开:所有 MLA 后端拒绝)、`logs/glm53_smoke2.log`(sparse 关:`MLAPrefillSelectorConfig` 只有 FLASH_ATTN,不支持 qk_nope=256/rope=0/v=256) | 端到端 GLM 测试需要 SM90+(或上游补 256 维 MLA 支持);目前用**层内真实权重**验证代替 |
| **L2** | 主线 CPU MoE 后端硬编码 `scoring_func="softmax"` | `vllm/model_executor/layers/fused_moe/experts/cpu_moe.py` 各 `apply()` | 主线 CPU 路径对 GLM(sigmoid+noaux_tc)、DS-V4(sqrtsoftplus)会选错专家 → 上游 bug(T21) |
| **L3** | 主线 CPU 内核忽略 `swiglu_limit`/`alpha`/`beta` | `cpu_moe.py` 无相关引用;`swiglu_limit` 只在 `activation.py`/`b12x.py`/`utils.py` 使用 | GLM/DS-V4/MiniMax/HY-V4 输出偏差 → 上游 bug(T21) |
| **L4** | 主线 CPU FP8 后端的 quant key 是 `(kFp8Static128BlockSym, kFp8Dynamic128Sym)`(A8),但计算实际是 A16 | `cpu_moe.py:CPUExpertsFp8._supports_quant_scheme` | 我们靠子类继承通过;上游若要支持"任意 fp8 模型"需要重新表述 |
| **L5** | monolithic `apply()` 无 `input_ids` 形参 | `modular_kernel.py:FusedMoEExpertsMonolithic` | hash routing 模型(DS-V4 部分层)无法走通用路径(T06) |
| **L6** | 本机硬件:2×A100-PCIE-40GB(SM80,无 NVLink)、AMD EPYC 9654(**无 AMX**)、1.5 TB DDR5-4800 | 报告 §1 | AMX 路线无法在本机验证;SM90 特性不可用 |
| **L7** | hf-mirror 直连可用(不要走代理),速率**波动大**:8–60 MB/s | 200 MB/25.7 s = 8.1 MB/s;Qwen3-30B(30 GB)实测 ~60 MB/s;Qwen3.8(185 GB)前 20 min ~25 MB/s | 大模型下载时间估计要按 ~25 MB/s 保守算(185 GB ≈ 2 h) |
| **L8** | 主线 fp8/wna16 量化方法的 `process_weights_after_loading` **不调用** experts 的 `process_weights_after_loading`(unquantized 会调) | `quantization/fp8.py:775` 与 `unquantized_fused_moe_method.py:171` 对比 | 依赖该钩子捕获 layer 的 OOT 后端会崩(`apply called before process_weights_after_loading`);已本地补调用(→ T29) |
| **L9** | 模型名要先核实再下载:用户说的「qwen3.8-flash」线上实际只有 `Qwen3.8-Flash-Next(-FP8)` | `Qwen/Qwen3.8-Flash` 在 HF-mirror 与 ModelScope 均 404 | 已按用户确认改下载 `-FP8`(D10);以后下载前先跑 `scripts/probe_oracle.py` 式的"先查后下" |

---

## 5. 关键证据索引(数字/脚本/日志在哪)

### 今天(2026-09-09 夜)新增

| 内容 | 位置 |
|---|---|
| 通用后端实现 | `vllm_xiaotu_moe/mixed_experts.py` |
| 主线 fp8 oracle 本地补丁 | `vllm/model_executor/layers/fused_moe/oracle/fp8.py`(混合模式前置 CPU) |
| 引擎 clamped SwiGLU | `xiaotu_moe/csrc/moe/moe_v2.hpp`(`act::silu_gate_one` / `silu_gate_clamped`,3 个调用点) |
| 引擎 e4m3 subnormal 修复 | `xiaotu_moe/csrc/kernels/fp8_dequant.hpp`(`ldexp(m/8, -6)`) |
| oracle 后端选择探测 | `scripts/probe_oracle.py` |
| 激活数值验证(BF16) | `scripts/test_swiglu_clamp.py` |
| 激活数值验证(MXFP4/真实 DS-V4 权重) | `scripts/test_swiglu_clamp_mxfp4.py` |
| **真实 GLM-5.3 fp8 专家层验证** | `scripts/test_glm53_fp8_layer.py`(layer 3,8 专家,rms_rel **9.3e-5**) |
| GLM 端到端失败日志 | `logs/glm53_smoke.log`、`logs/glm53_smoke2.log`(`*.log` 被 gitignore,已把关键报错摘录到 `docs/evidence/glm53_a100_blocker.txt`) |
| GLM 端到端脚本 | `scripts/glm53_smoke.py`(支持 `GLM_HF_OVERRIDES`) |
| **CPU/GPU 端到端等价性** | `scripts/tiny_moe_equiv.py` + `scripts/make_tiny_mixtral.py`(生成 vLLM 命名兼容的微型 Mixtral) |
| fp8 真实模型端到端 | `scripts/fp8_moe_smoke.py`(`SMOKE_MODEL` 指定模型) |
| 混合模式主线补丁快照 | `patches/mainline_mixed_mode_generic.patch`(3 个 oracle + fp8 experts 钩子) |

### 引擎重建命令(本机,必须记牢)

```bash
cd /home/user/lvllm/vllm-xiaotu-moe/xiaotu_moe
ROOT=$PWD
PY_INC=$(/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python -c "import sysconfig;print(sysconfig.get_paths()['include'])")
PYBIND11_INC=/home/user/lvllm/.search-venv/lib/python3.10/site-packages/pybind11/include
CUD=/home/user/anaconda3/envs/xiaotumoe-vllm/lib/python3.12/site-packages/nvidia/cu13   # 只有这个 env 带 crt/host_config.h
g++ -std=c++17 -shared -fPIC -O3 -ffast-math -fno-finite-math-only \
  -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512bf16 -mfma \
  -DXIAOTU_MOE_MODULE_NAME=_xiaotu_moe_C_avx512_bf16 \
  -I"$PY_INC" -I"$PYBIND11_INC" -I"$ROOT/csrc" -I"$CUD/include" \
  -L"$CUD/lib" -Wl,--no-as-needed -lcudart -Wl,--as-needed -Wl,-rpath,"$CUD/lib" -lcudart \
  "$ROOT/csrc/python_binding/binding.cpp" \
  -o "$ROOT/build/_xiaotu_moe_C_avx512_bf16.cpython-312-x86_64-linux-gnu.so"
# 约 18 s。
# 全部 ISA 变体:PYTHON=<env>/bin/python bash scripts/build_engine_variants.sh
# (引擎源码只在本仓库 xiaotu_moe/csrc;外部 xiaotu-moe 目录已转私有、不再被跟踪)
```

### ⚠️ 主线工作树是"活的",不要 reset

`/home/user/lvllm/process_data/ref/repos/vllm-mainline`(HEAD=`6c73b08`,这是**已安装的 vllm**)
带着一批**未提交**改动:SM80 port + `is_bmm` fp8 修复 + 本轮新增的 `oracle/fp8.py` 混合模式前置。
其中 fp8 那 6 行已单独快照到 `patches/mainline_fp8_oracle_mixed_mode.patch`;
其余仍在工作树里。**任何 `git checkout/clean/reset` 都会丢掉它们**(重装 vllm 亦然)。

### 历史证据(仍在用)

- 论文式报告:`docs/EXPERIMENT_REPORT.md`(rev 12,991 行)
- 第 2 轮优化:`docs/OPTIMIZATION_ROUND2.md`
- 通用化路线图:`docs/ROADMAP_GENERALITY.md`
- 上游 PR 计划与快照:`patches/UPSTREAM_PRS.md`、`patches/upstream/pr{1,2,3}_body.md`
- 全量测量日志:`/home/user/lvllm/results.txt`
- 图表/JSON:`/home/user/lvllm/report/`
- 引擎性能长跑 TODO:`docs/TODO_LONGTERM.md`(与 lk-moe 对齐,另一条线)

---

## 6. 下一步(建议的最近 3 件事)

1. **T04**(激活守卫)+ **T01**(oracle 推广):都是小改动,决定"通用后端"的正确性边界。
2. **T02**:下载 `Qwen3-30B-A3B-Instruct-2507-FP8`(hf-mirror 直连,约 1 小时)做通用路径端到端;
   或先用一个 bf16 MoE(配合 T03)快速验证。
3. **T04 + T01**:做完这两项就满足 D9 解冻条件的第 1 条;端到端(T02)是第 2 条。
   (PR 侧已冻结,见 §3 D9,无需动作。)

---

## 7. 修订记录

- **2026-09-09(第 6 版)** — 仓库根目录去混淆(用户指出):①`xiaotu-moe/` 目录 untrack(本地保留、
  gitignore),本仓库不再包含独立引擎项目;②根 `README.md`/`README_CN.md` 重写为插件首页;
  ③`THIRD_PARTY_NOTICES.md` 引用更新;④新增 T35(已完成)、T36/T37(待定)、D13(待用户拍板);
  ⑤T11 记录 `scripts/build_engine_variants.sh` 已内置。依据:用户 2026-09-09 提问 + 仓库检查。
- **2026-09-09(第 5 版)** — **项目边界澄清(用户指出)**:新增 §0b「项目关系」,记录 D12
  (`xiaotu-moe` 转私有、`vllm-xtu-moe` 为唯一对外项目);新增 T33(文档去混淆,已完成)、
  T34(第三方署名文件,待做)。依据:用户 2026-09-09 澄清。
- **2026-09-09(第 4 版)** — ①T01/T03/T04/T28 **完成**,新增 T29(主线 experts 钩子)、T30(CPU/GPU 端到端等价性,已过)、T31(目标模型端到端)、T32(A100 可行性实测);②新增 D10(目标模型 = `Qwen/Qwen3.8-Flash-Next-FP8`,用户确认)、D11(保留误下的 30 GB 模型作额外验证);③新增 L8(主线 fp8 不调 experts 钩子)、L9(模型名先核实再下载);④看板与 §5 证据索引同步。依据:本轮实测(`scripts/tiny_moe_equiv.py`、`logs/qwen3_fp8_smoke.log`)+ 用户 2026-09-09 澄清。
- **2026-09-09(第 3 版)** — ①T01 补 fp8 oracle 快照指针(`patches/mainline_fp8_oracle_mixed_mode.patch`);
  ②§5 新增「主线工作树是活的,不要 reset」的风险提示。依据:检查 mainline checkout 的 `git status`。
- **2026-09-09(第 2 版)** — **D9 拍板为方案 A**:三个 draft PR 全部冻结,等通用 CPU experts 后端完成
  后再统一重排(附冻结动作清单与 4 条解冻触发条件);同步调整看板、T20/T23/T24 的状态。依据:用户 2026-09-09 选择。
- **2026-09-09(第 1 版)** — 新建。汇总:①今天会话新增(通用 CPU experts 后端重写、
  主线 fp8 oracle 补丁、引擎 clamped SwiGLU、e4m3 subnormal 修复、GLM-5.3 在 A100 的
  阻塞定性、三个新测试脚本);②此前被列为待办但尚未完成的项(T01–T27);③已决断
  D1–D8 与已证伪结论 X1–X4;④阻塞 L1–L7;⑤D9(PR 重排)待用户拍板。
  依据:用户 2026-09-09 要求 + 本会话实测(`logs/`、`scripts/`、`results.txt`)。
