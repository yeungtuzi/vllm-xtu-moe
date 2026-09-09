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

### 快速状态看板(截至 2026-09-09 夜)

| 主题 | 状态 | 一句话 |
|---|---|---|
| 通用 CPU experts 后端(格式无关) | 🟢 主干已通 | `mixed_experts.py` 走主线 router;fp8/mxfp4/int4 三种格式被 oracle 选中 |
| 混合模式 oracle 前置(主线补丁) | 🟡 只做了 fp8 | `oracle/fp8.py` 已改;mxfp4/int4/nvfp4 未改(PR1 只含 mxfp4) |
| 引擎 clamped SwiGLU(`swiglu_limit`) | 🟢 已完成并验证 | `activation_type=1`;BF16/MXFP4/真实 GLM fp8 三路数值验证通过 |
| 引擎 e4m3 subnormal 解码 bug | 🟢 已修 | `2^-7 → 2^-6`,与 `torch.float8_e4m3fn` 逐字节一致 |
| GLM-5.3-Flash 端到端 | 🔴 阻塞(硬件) | A100/SM80 无任何支持其 MLA 维度(256/0/256)的 attention 后端 |
| 通用路径端到端(能跑的模型) | 🟡 待做 | 需要 Qwen3-30B-A3B-FP8 之类可在 A100 跑、格式匹配的模型 |
| 上游 PR(PR1/2/3) | 🧊 **已冻结(D9=A)** | 三个 draft 全部不动,等通用 CPU 后端完成后再统一重排(见 §3 D9) |
| RFC(layerwise GPU prefill / 通用 CPU 后端) | 🔴 未发 | 草稿在 `patches/rfc_layerwise_gpu_prefill.md` |

---

## 1. 进行中(In progress)

| ID | 事项 | 状态 | 备注 |
|---|---|---|---|
| T01 | 把混合模式 oracle 前置推广到全部格式 | 🟡 进行中 | fp8 已改;`int_wna16.py`/`mxfp4.py`/`nvfp4.py`/`mxfp8` 待改,并抽成统一 helper |
| T02 | 通用路径端到端验证(真实模型) | 🟡 进行中 | 目标模型见下;GLM 在 A100 被 attention 挡住 |

---

## 2. 待办清单

### A. 通用 CPU experts 后端(主线接口层)

| ID | 待办 | 为什么 | 优先级 |
|---|---|---|---|
| **T01** | 混合模式下把 CPU 后端前置的改动推广到 `oracle/{mxfp4,int_wna16,nvfp4,mxfp8}.py`,抽成 `oracle/_common.py` 的 `prefer_cpu_backend()` | 现在只有 fp8 生效;PR1 只含 mxfp4。任意格式的模型都要能被选中 | **P0** |
| **T02** | 用**能在 A100 跑通**的模型做通用路径端到端(load → 连贯输出 → TTFT/吞吐) | 通用路径至今没在真实模型上端到端跑过(DS-V4 一直走 OOT 覆盖) | **P0** |
| **T03** | 新增 BF16/FP16(无量化)CPU 后端(`CPUUnquantizedExperts` → 引擎 `MOE_BF16`) | 覆盖 Mixtral / Qwen2-MoE / Qwen3-MoE bf16 等大量模型;实现成本极低 | P1 |
| **T04** | 收紧 `_supports_activation`:只允许 `MoEActivation.SILU`;对 `SWIGLUOAI`(gpt-oss 的**交错** gate/up 布局)必须拒绝而不是静默算错 | 主线 `CPUExpertsMxfp4` 声称支持 SWIGLUOAI,但我们的引擎假设 packed 布局 → 会给出错误结果 | **P0(正确性)** |
| **T05** | 支持专家并行(expert_map / EP) | 现在直接 `raise NotImplementedError`;TP=2 EP 是我们已测过的主力配置 | P1 |
| **T06** | monolithic `apply` 拿不到 `input_ids`(hash routing / DS-V4 部分层需要) | 要么上游加参数,要么改走 modular 路径;当前会报错而不是算错 | P2 |
| **T07** | INT4(WNA16)按 `quant_config` 取真实 group size(32/128)与零点 | 现在 `_group_n/_group_k` 硬编码 128/128 | P1 |
| **T08** | INT8 W8A8 格式(主线 CPU 已有,引擎缺) | 对齐主线 CPU 格式集 | P2 |
| **T09** | 长尾格式:MXFP8 / MXFP6 / FP8 e5m2 / GGUF k-quants / Marlin 布局 | 按用户需求增量 | P3 |
| **T10** | 引擎侧:`apply_router_weight_on_input` 支持(现在 raise) | 少数模型会用 | P3 |

### B. 引擎 / 内核

| ID | 待办 | 为什么 | 优先级 |
|---|---|---|---|
| **T11** | 打包全部 ISA 变体(scalar/avx2/avx512_base/vnni/bf16)+ `xiaotu_moe.self_check()` | 现在插件只带 avx512_bf16 一个 `.so`;AVX2-only 机器用不了 | P1 |
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
| **L7** | hf-mirror 直连可用(不要走代理),实测 ~8.1 MB/s | 200 MB / 25.7 s(2026-09-09) | 31 GB 模型约需 1 小时下载(T02 的成本估算) |

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
# 约 18 s;改完 csrc 要同步一份到 /home/user/lvllm/xiaotu-moe/csrc/(两处都在 git 里)
```

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

- **2026-09-09(第 2 版)** — **D9 拍板为方案 A**:三个 draft PR 全部冻结,等通用 CPU experts 后端完成
  后再统一重排(附冻结动作清单与 4 条解冻触发条件);同步调整看板、T20/T23/T24 的状态。依据:用户 2026-09-09 选择。
- **2026-09-09(第 1 版)** — 新建。汇总:①今天会话新增(通用 CPU experts 后端重写、
  主线 fp8 oracle 补丁、引擎 clamped SwiGLU、e4m3 subnormal 修复、GLM-5.3 在 A100 的
  阻塞定性、三个新测试脚本);②此前被列为待办但尚未完成的项(T01–T27);③已决断
  D1–D8 与已证伪结论 X1–X4;④阻塞 L1–L7;⑤D9(PR 重排)待用户拍板。
  依据:用户 2026-09-09 要求 + 本会话实测(`logs/`、`scripts/`、`results.txt`)。
