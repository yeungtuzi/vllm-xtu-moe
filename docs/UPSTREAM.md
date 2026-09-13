# 需要上游配合的改动

本插件在**原生 vLLM** 上即可运行:混合模式所需的几处主线行为由
`vllm_xiaotu_moe/mainline_shims.py` 以 monkey-patch 方式提供(见
[`ARCHITECTURE.md`](ARCHITECTURE.md) §5)。本文列出这些改动本身,以及可以
上游化的部分——它们对主线同样有价值,不依赖本插件。

---

## 1. 混合模式的基础设施(可上游化)

### 1.1 `VLLM_EXPERTS_LOAD_DEVICE=cpu`(专家权重放 host)

**问题**:MoE 模型的专家权重若超过单卡显存,主线在**构造期**就会 OOM
(例如 DeepSeek-V4-Flash 的专家约 137 GiB)。主线现有的 offload 机制
(`model_executor/offloader/prefetch.py`)只能搬运**已构造**的权重,无法避免
构造期的分配。

**改动**(3 个文件,约 +60 行):

| 文件 | 改动 |
|---|---|
| `vllm/envs.py` | 新增 `VLLM_EXPERTS_LOAD_DEVICE`(默认 `gpu`,`cpu` 为可选) |
| `fused_moe/routed_experts.py` | 开关打开时在 `torch.device("cpu")` 下调用 `create_weights()` |
| `fused_moe/oracle/mxfp4.py` | 开关打开时优先选 CPU 后端,并跳过 AMX 重打包 |

默认行为不变(`gpu` 时走原路径)。

### 1.2 oracle 优先选 CPU 后端(各格式对称)

`oracle/{fp8,int_wna16,unquantized}.py` 需要在混合模式下把 CPU 后端前置
(与 1.1 中 mxfp4 的改动对称)。本插件目前用 shim 实现。

### 1.3 跳过 CPU 后端的 AMX 重打包

`cpu_moe.prepare_{fp8,mxfp4,int4}_moe_layer_for_cpu()` 会把权重重打包成
AMX 内核需要的布局,这会破坏"原始权重直传第三方 CPU 引擎"的用法,并且依赖
`torch.ops._C.convert_weight_packed`(部分部署未编译)。混合模式下应跳过。

### 1.4 量化方法应通知 experts 后端

`Fp8MoEMethod.process_weights_after_loading()` 与 WNA16 对应方法**没有**调用
`experts.process_weights_after_loading(layer)`,而
`UnquantizedFusedMoEMethod` 会调用。依赖该钩子获取 layer 引用的后端
(包括 OOT 后端)因此拿不到上下文。建议统一调用。

## 2. 主线 CPU MoE 后端的两个正确性问题

这两个问题与第三方后端无关,是主线自带 CPU 内核的行为,已在真实模型上复现:

### 2.1 路由硬编码为 softmax

`fused_moe/experts/cpu_moe.py` 的 `apply()` 中:

```python
topk_weights, topk_ids = select_experts(..., scoring_func="softmax", ...)
```

对使用 sigmoid + `noaux_tc`(GLM 系)或 `sqrtsoftplus`(DeepSeek-V4)的模型,
这会**选错专家**。建议改为使用主线已有的 router 对象
(`create_fused_moe_router`),与本插件做法一致。

### 2.2 忽略 `swiglu_limit` / `alpha` / `beta`

`moe_config.swiglu_limit` 只被 `activation.py` / `b12x.py` / `utils.py` 使用,
CPU 内核完全没有应用它。GLM-5.x、DeepSeek-V4、MiniMax、HY-V4 等模型训练时
都带有该 clamp,忽略它会改变专家输出。建议在 CPU 内核的 gated 激活里实现
`clamp(gate,max=L) * sigmoid(alpha*clamp(gate,max=L)) * (clamp(up,±L)+beta)`。

## 3. 已提交的上游 PR(快照见 `patches/`)

| PR | 范围 | 规模 | 说明 |
|---|---|---|---|
| **PR1** | `VLLM_EXPERTS_LOAD_DEVICE=cpu` + MXFP4 oracle | 3 文件 +59/−2 | §1.1 |
| **PR2** | DeepSeek-V4 `o_proj` 在 SM8.x 的 fp8 einsum 修复 | 4 文件 +422/−12 | Ampere 上 Marlin 重打包破坏 `o_proj` 的直读权重;并补可移植 Triton fp8 einsum |
| **PR3** | DeepSeek-V4 在 Ampere/Ada 的可移植 Triton 回退 | 21 文件 +6350/−119 | 稀疏 MLA / DeepGEMM / MQA / `float8e4nv` dtype 等 SM8.x 回退 |

- PR 描述(目标 / 问题 / 设计 / 算法 / 实测 / 测试):`patches/upstream/pr{1,2,3}_body.md`
- 补丁快照:`patches/mainline_sm80_mixed_mode.patch` 等
- RFC 草稿(CPU-offload MoE 的逐层 GPU prefill):
  `patches/rfc_layerwise_gpu_prefill.md`

> PR2/PR3 的 SM80 移植代码来自 vLLM fork
> [`Lvllmds4-x`](https://github.com/guqiong96/Lvllmds4-x)(Apache-2.0),
> 按其许可保留 SPDX 署名;详见 [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)。

## 4. 给维护者的建议顺序

1. §2.1 / §2.2(小而独立、对主线普遍有益的 bug 修复);
2. §1.1 + §1.2 + §1.3(混合模式基础设施,需要一个小设计讨论);
3. §1.4(一行调用,依赖 1.1 的动机);
4. PR2 → PR3(Ampere 上的 DeepSeek-V4 支持)。

---

## 5. 【用户于第 208 轮下达的固定原则】**复用优先级:上游 → lvllm → 自研**

> **只要能参考和复用的功能与代码,一律按 `vLLM upstream` → `lvllm` 的顺序复用;
> 只有当两者都没有时,才允许按搜索/调研结果自行写代码。**

落到本项目就是:

1. **先查 vLLM 上游**(含其 main / PR):有没有现成的开关、后端槽位、模型实现。
   例:`VLLM_EXPERTS_LOAD_DEVICE=cpu`(上游 PR #56118)、
   `Mxfp4MoeBackend.CPU` 后端槽位、`RoutedExperts` 的 `process_weights_after_loading`
   钩子、`DeepSeekV4MTP` / `DSparkDraftModel`(主线**已有**,见 `NOTES.md` §255);
2. **再查 lvllm 系列**(`Lvllm` / `Lvllmds4` / `Lvllmds4-x` / `Lsglang`):
   lk_moe 的接线方式就是 `routed_experts.py` + `moe_runner.py` 两个文件的分派分支;
3. **两者都没有**才自己写,并且要在 `NOTES.md` 里记清"为什么上游和 lvllm 都不适用"。

### 5.1 为什么"照抄 lvllm"在本项目里是**可行且应当优先**的

- lk 的 vLLM 侧集成只有**两个 Python 文件**:
  `vllm/model_executor/layers/fused_moe/routed_experts.py` 与
  `.../runner/moe_runner.py`;它的分派逻辑是
  ```
  if is_gpu_resident_layer:   GPU
  elif is_current_stream_capturing():  _cpu_decode      # 捕获期走 host-func
  elif is_gpu_prefill_layer and enough tokens:  _gpu_prefill
  else:  _cpu_prefill
  ```
  (`Lvllmds4-x` 工作区里那份的改动甚至只是把 `import lk_moe` 换成 `import xiaotu_moe`,
  说明**接口完全一致**,我们的引擎可以直接顶上)。
- **不可上游化的只有"依赖 `lk_moe` 这个二进制包"这一条**:
  上游不会 merge `import lk_moe`,也不会接受核心文件里按 `LVLLM_*` 环境变量分叉。
  但**上游愿意接纳这套基础设施** —— PR #56118 正是把"专家权重放 host + 选 CPU 后端 +
  跳过 AMX 重打包"做成了官方开关,把真正做计算的部分留给任意外部引擎。
- 因此本项目的正确形态就是:**上游的槽位(1.1-1.4)+ lvllm 的分派(§5.1)+ 我们自己的
  计算内核 `xiaotu_moe`**。前两者都不需要我们自研。

### 5.2 ⚠️ 现状里的重复实现(待清理)

`vllm_xiaotu_moe/` 目前**同时**存在两条 DS-V4 集成路径:

| 路径 | 文件 | 性质 |
|---|---|---|
| **lk 同构** | `mixed_experts.py` + `mainline_shims.py` | 主线 `RoutedExperts` 不动,`Mxfp4MoeBackend.CPU` → `XiaotuCPUExperts*`;引擎在 `_ensure_engine` 里按需构造 |
| **自研 OOT 覆盖** | `hybrid_model.py` | 把 `DeepseekV4MoE` 换成 `CpuXiaotuMoE`、并 `ModelRegistry.register_model` 覆盖整个 arch |

对 DeepSeek-V4 生效的是**后者**(启动日志
`registered OOT override of DeepseekV4ForCausalLM`),而它的存在理由是
"主线在构造期就把 137 GiB fp4 专家 materialize 到 GPU" —— **这件事上游 1.1(与我们的
shim 1)已经解决了**。⇒ 按 §5 的原则,应优先验证/迁到 lk 同构那条路,
能省掉整份自研 MoE 模块以及随之而来的所有 cudagraph/prefix-cache/spec-decode 适配。

为此在 `hybrid_model.py:register()` 里加了开关
**`XIAOTU_OOT_OVERRIDE=0`**:关掉 OOT 覆盖,只留 `mixed_experts` + `mainline_shims`,
即纯 lk 同构路径。

---

## 6. 【第 208 轮·用户追问的答案】**把 lk_moe 换成开源的 vllm-xtu-moe 之后,还有没有障碍?**

### 6.1 先确认唯一那条硬障碍(实测事实)
```
https://pypi.org/project/lk-moe/     License: Proprietary
  "lk_moe is provided under a proprietary software license agreement."
  Distribution: only lk_moe-2.4.3-cp312-...-manylinux_2_34_x86_64.whl (binary)
                "No source distribution files available"
```
而本仓是 **Apache-2.0**(`pyproject.toml: license = "Apache-2.0"`),`Lvllmds4-x` 那份 fork
也是 Apache-2.0(它本身是 vLLM 的 fork)。
⇒ **唯一的硬障碍就是"上游不能依赖一个专有二进制包"。**
⇒ 换成开源的 `vllm-xtu-moe` 之后,**这条障碍消失**。

### 6.2 所以答案:**没有架构性障碍,用户的方案成立**;但顺序要按第 5 节的复用原则
正确的形态(每一层都用"已有的"而不是"自研的"):

| 层 | 用谁的 | 具体是什么 |
|---|---|---|
| ① 基础设施 | **vLLM 上游** | `VLLM_EXPERTS_LOAD_DEVICE=cpu` + `Mxfp4MoeBackend.CPU` + 跳过 AMX 重打包 = 上游 PR [#56118](https://github.com/vllm-project/vllm/pull/56118);上游未合并前由本仓 `mainline_shims.py` 等价提供 |
| ② 分派策略 | **lvllm** | `routed_experts.py`/`moe_runner.py` 的四路分派(resident / capture→cpu_decode / gpu_prefill / cpu_prefill)。**但只有在需要 resident 或 gpu-prefill 层时才需要** |
| ③ 计算内核 | **本仓** | `xiaotu_moe`(AVX512,Apache-2.0),通过 `mixed_experts.py` 注册进 ① 的 CPU 后端槽位 |
| ④ 环境变量 | 上游命名 | 上游的那几个用 `VLLM_*`(PR #56118 已经这么命名);插件自身的旋钮用 `XIAOTU_*` |

**关键点:`①②` 合起来意味着"不需要 fork 任何核心文件"。**
我们的 `mixed_experts.XiaotuCPUExperts*` 实现的是主线**已有的**接口
(`mk.FusedMoEExpertsMonolithic`),所以它插进的是**官方后端槽位**,
而不是像 lvllm 那样往 `routed_experts.py` 里加分叉。这比 lvllm 的原方案**更干净**,
也更容易被上游接受(符合 vLLM 的"用扩展点、不加 vendor 分支"取向)。

### 6.3 真正剩下的工作只有一件:**性能持平**
- 已达成:内核级与 lk 同口径 **0.89×/0.80×**(第 1 条结案);
- 未达成:端到端。lk 的 **SM80 分支**(2×3090 / DDR4-3200 16ch,比本机弱)README 写
  `Prefill 1060 t/s @32768 / Decode 26 t/s / Spec 35~47 t/s`;本机 2×A100 + DDR5-4800 24ch
  目前 C=1 ≈ 11.2–13.0 t/s。**⇒ 差距在"引擎之外的编排",不在内核。**
- 因此第 208 轮起的路线是:**先把 DS-V4 从"自研 OOT 覆盖"切到"①+③ 官方槽位路径"**
  (`XIAOTU_OOT_OVERRIDE=0`),再在这条 lk 同构路径上做性能收敛。

---

## 7. 【第 208 轮·追问】**lvllm 的编排方式对 vLLM 上游有什么重大影响,导致其难以接纳?**

结论:**有,而且是上游最在意的那一类 —— "绕过 runner / 绕过后端抽象"。**
这不是猜测,有上游自己的文字作证。

### 7.1 vLLM 已有的、**官方**的 MoE 扩展点(所以根本不需要改核心文件)
| 机制 | 位置 | 作用 |
|---|---|---|
| `FusedMoEExperts` / `FusedMoEExpertsMonolithic` | `fused_moe/modular_kernel.py:481 / :982` | MoE 专家计算的**抽象基类**,外部后端实现它就等于插进官方槽位 |
| `FusedMoEFactory` | `fused_moe/layer.py:88` | 按配置选后端 |
| `oracle/*._get_priority_backends()` | `fused_moe/oracle/{fp8,mxfp4,int_wna16,unquantized}.py` | 后端优先级表(据此选后端) |
| `Mxfp4MoeBackend.CPU` | `fused_moe/oracle/mxfp4.py:133` | **CPU 后端是主线已保留的一等枚举成员** |
| `vllm.general_plugins` | `docs/design/plugin_system.md` | 官方插件入口,文档明言其目的是"**without modifying the vLLM codebase**" |

### 7.2 lvllm 的做法踩到的四条(按严重程度)
1. **绕过 runner(最重的一条)。** lvllm 在 `moe_runner.py` 里直接按环境变量分叉调用
   `self.lk_moe.cpu_decode/_cpu_prefill/_gpu_prefill`,而不是经 `quant_method.apply()`
   走 `FusedMoEExpertsMonolithic`。
   **证据**:同期的上游 PR [#37190](https://github.com/vllm-project/vllm/pull/37190)
   (MoE 专家 CPU offload + GPU 缓存)把这一点当作**卖点**写在摘要里:
   > **"No runner bypass — all paths go through `quant_method.apply()`. EP dispatch,
   > DP chunking, and shared expert overlap work unchanged."**
   ⇒ 反过来读就是:已知的反对意见正是"runner bypass 会让 EP/DP/shared-expert overlap 全部失效"。
2. **专有二进制依赖。** `lk-moe` 在 PyPI 上是 **Proprietary**(只发 manylinux 二进制 wheel,
   无 sdist)——上游不可能 merge `import lk_moe`。
3. **核心文件里的 vendor 环境变量分叉。** `envs.py` 里一批 `LVLLM_*`,
   与插件文档"不改 vLLM 代码库"的取向相反。
4. **把"图内 CPU 计算"的语义押在 `torch.cuda.is_current_stream_capturing()` 上,
   并用类属性 `RoutedExperts.output_gpu` 当固定输出缓冲。**
   这是全局可变状态 + 捕获期语义,属高风险设计。
   **旁证**:上游 PR #37190 在 Limitations 里明确写
   > "`--enforce-eager` required (CUDA graph compat deferred to PR 2)"
   ⇒ 上游对"CPU 工作放进 CUDA graph"是**刻意保守、分两步走**的。

### 7.3 但**"想法"本身上游是欢迎的** —— 正在被两条 PR 接纳
| 上游 PR | 内容 | 状态 |
|---|---|---|
| [#56118](https://github.com/vllm-project/vllm/pull/56118) | `VLLM_EXPERTS_LOAD_DEVICE=cpu` + 选 CPU 后端 + 跳过 AMX 重打包(= 我们 shim 1/2/3/5) | open,需 rebase;只有机器人评论 |
| [#37190](https://github.com/vllm-project/vllm/pull/37190) | 专家权重放 CPU pinned + **GPU LFRU 热专家缓存** `--moe-expert-cache-size`(= 我们的"GPU 常驻专家层"!) | open,59 条讨论、8 条评审意见、12 位 reviewer;明确"不绕过 runner" |
⇒ **判别标准很清楚:上游要的是"实现官方后端接口",不要的是"改核心文件 + 依赖厂商二进制"。**

### 7.4 对本项目的直接结论(按第 5 节复用原则排序)
1. **"GPU 常驻专家层"这个旋钮不该是我们自研的** —— 上游 PR #37190 已经用
   `CachedWeightProvider`(LFRU)在做同一件事,并且挂在 `quant_method.apply()` 下面。
   我们本轮实测"10 层常驻 ⇒ C=1 +16%、预填充 5263 t/s"(NOTES §261),
   正好可以作为该 PR 在 A100/EPYC 上的**独立验证数据**。
   ⇒ 应改为**复用 PR #37190 的 `--moe-expert-cache-size`**,而不是维护
   `XIAOTU_MOE_GPU_RESIDENT_LAYERS`。
2. **CPU 后端槽位用上游 PR #56118 的**(我们先用 shim 补齐,等它合并后 shim 自动变 no-op)。
   本轮就发现并补上了缺失的第 5 处(`mxfp4.convert_weight_to_mxfp4_moe_kernel_format`
   对 CPU 后端原样返回),否则主线 `RoutedExperts` 路径在
   `process_weights_after_loading` 处直接 `ValueError`(NOTES §263)。
3. **不采用 lvllm 的 runner 分叉**,改用我们已有的
   `mixed_experts.XiaotuCPUExperts*`(它实现的正是 `FusedMoEExpertsMonolithic`)。
4. **只有 `xiaotu_moe` 内核本身是我们自研的** —— 这一层上游和 lvllm 都没有开源等价物,
   符合"两者都没有才自己写"。

---

## 8. 【用户第 210 轮强调·硬约束】**投机解码与验证的全序列必须用 vLLM 现成的,不许自己写**

> 用户原话:"记住原则啊,vllm 里面已经有投机解码、验证等全序列,你不许自己写。"

### 8.1 合规审计(本仓全量 grep 的结果)
| 我们自己的代码 | 性质 | 是否违规 |
|---|---|---|
| `--speculative-config '{"method":"dspark",…}'` | **vLLM 的** `SpeculativeConfig`;草稿模型是 vLLM 的 `DSparkDraftModel`,proposer/拒绝采样/验证/奖励 token 全在 `vllm/v1/worker/gpu/spec_decode/dspark/` | ✅ 未自研 |
| `hybrid_model.py: _draft_layer_count()` / `reserve_draft_bytes()` | **显存常驻策略**(draft 的专家层要占多少显存、要不要预留) | ✅ 与投机算法无关 |
| `hybrid_model.py: _instance_index()` / `XIAOTU_MOE_RESIDENT_DRAFT` | 判断"同一 prefix 的第二次构造 = draft 副本",决定其**专家权重放 GPU 还是 CPU** | ✅ 与投机算法无关 |
| `mixed_experts.py: _verify_once()` | **内核数值自检**(把引擎输出与同权重的 torch 参考逐位比对),`XIAOTU_VERIFY_LAYER=1` 才开 | ✅ 不是 token 验证 |
| 全仓其它文件 | `grep -niE "draft|speculat"` 在 `__init__.py`/`mainline_shims.py`/`mixed_experts.py`/`ple_offload.py` 命中数均为 **0** | ✅ |

**⇒ 结论:本插件不含任何自研的起草/接受/拒绝/验证逻辑。**
(历史教训见 `TRIED_AND_REVERTED.md` R100:`method:"mtp"` 那条线曾诱使人自己造 ——
已废止,理由是该 checkpoint 根本没有 MTP 权重,而不是"要自研"。)

### 8.3 【第 211 轮·移植路径(lk port)的合规审计】投机解码全序列的**实际来源**

本轮真正跑起来的是 **lvllm 的编排链 + 我们的引擎**(env `lkxtu`),投机解码**全序列**逐环节
都来自 vLLM 自己(下表均为**被 import 的那棵树** `lkxtu/lib/python3.12/site-packages/vllm` 中的文件):

| 环节 | 上游文件:行 | 我们做了什么 |
|---|---|---|
| 打开投机解码 | `config/speculative.py`(`SpeculativeConfig`) | 只传官方的 `--speculative-config`(脚本里一行字符串) |
| dspark 形态判定 + 草稿 config | `config/speculative.py:810-820` | 无(自动) |
| 草稿模型类 | `models/deepseek_v4/nvidia/dspark.py:306` | 无(自动) |
| 草稿加载/权重映射 | `v1/worker/gpu/spec_decode/dspark/utils.py:36`、`dspark.py:502` | 无(自动) |
| 起草(proposer) | `v1/worker/gpu/spec_decode/dspark/speculator.py:43 DSparkSpeculator` | 无(自动) |
| 验证/拒绝采样 | `v1/worker/gpu/spec_decode/rejection_sampler.py` | 无(自动) |
| 接受率统计 | `v1/spec_decode/metrics.py:120` | 无(自动,直接读日志) |
| 草稿层放 GPU | lvllm 的 `envs.py:2272` + `quantization/mxfp4.py:548` | **只填 lk 现成环境变量** `LVLLM_GPU_RESIDENT_MOE_LAYERS`(层号由 ckpt config 自动算出) |

⇒ 移植路径下我们新增的代码**只有** `scripts/serve_lk_port.sh` 里的:①从 ckpt config 算草稿层号;
②显存护栏(不够就禁用 draft + 警告)。二者都是**策略/参数**,不含任何起草、接受、拒绝、验证算法。

### 8.2 由此推出的一条设计约束(以后照此办理)
> **凡是 vLLM 已有的"全序列"(投机解码/验证、prefix caching、chunked prefill、调度、
> KV 管理、EP/DP、cudagraph),我们的插件只允许做两件事:
> (1) 通过官方开关打开它;(2) 为它提供正确的专家权重放置与计算。
> 一旦发现自己在写"第二个实现",停下来,回到 §5 的复用顺序。**

---

## 9. 【用户第 210 轮·硬约束】**找不到现成代码,就不许新写实质性功能**

> 用户原话:"除非你在 vllm 主线和 lvllm 系列 fork 中找不到代码,否则不允许你再新写实质性功能,
>  免得你又开始自己手搓东西。"

### 9.1 规则(可执行的判据)
写任何**实质性功能**之前,必须先在下面三个来源里找到它:
1. `process_data/ref/repos/vllm-mainline`(我们实际 import 的 fork,含未提交补丁);
2. `Lvllmds4-x`(SM80 分支,lk 的 vLLM 侧集成);
3. `Lvllm` / `Lvllmds4` / `Lsglang`(上游主线分支 / SM120 / sglang 集成)。
**找不到**才允许写,并且必须在 `NOTES.md` 里写明"三个来源都没有"的证据。

### 9.2 据此对本轮已写代码的**回溯审计**(必须处置)
| 我写的东西 | 来源 | 处置 |
|---|---|---|
| `mainline_shims.py` shim 1/2/3 | 上游 PR #56118 的三段 | 保留(上游代码的等价物) |
| `mainline_shims.py` shim 5 | 上游 PR #56118 第三段 | 保留;移植后应可删 |
| `mainline_shims.py` **shim 6**(补 `input_ids`) | **我自己想的** | ❌ **移植 lk 编排后删除** |
| `mainline_shims.py` **shim 7**(补 hash 表到 layer) | **我自己想的** | ❌ **移植 lk 编排后删除** |
| `hybrid_model.py` `_instance_index()` 判 draft | **我自己想的启发式** | ❌ 换成 **lk 现成规则**:`is_lk_moe_mtp_layer(name) = name.startswith("mtp.")`(lk 用它把 draft 层钉在 GPU) |
⇒ **lk 早就实现了"draft 永远在 GPU"**(`is_lk_moe_gpu_resident_layer()` 对 `mtp.` 前缀直接返回 True),
   所以用户第 210 轮那条要求**不需要**我另造机制 —— 直接移植 lk 的规则即可。

### 9.3 移植清单(从 `Lvllmds4-x` 搬,逐字搬 + 机械改名)
| 文件 | 搬什么 | 改名 |
|---|---|---|
| `vllm/envs.py` | `LVLLM_*` 旋钮 + `is_lk_moe_feature_enabled / _use_gpu_prefill / _mtp_layer / _gpu_prefill_layer / _cpu_layer / _gpu_resident_layer` | `LVLLM_` → `VLLM_XIAOTU_` |
| `fused_moe/routed_experts.py` | `is_*` 三个标志、`should_use_gpu_prefill`、`_cpu_decode`、`_cpu_prefill`、`clean_weights_after_loading`、`_initialize_cuda_graph_buffers` | `lk_moe` → `xiaotu_moe` |
| `fused_moe/runner/moe_runner.py` | 4 路分派(resident / 捕获期 `_cpu_decode` / `gpu_prefill` / `cpu_prefill`) | 同上 |

## 10. 【第 211 轮·代码审计】**DS-V4 的 draft(MTP / DSpark)在上游是怎么处理的**

结论:**全部是"权重命名与加载"的处理,没有任何"专家放哪张卡"的逻辑**——设备放置是 lk 的职责,
而 lk 只用两个开关(见 `report/tuning/NOTES.md` §290)。逐处源码:

| 位置 | 作用 |
|---|---|
| `config/speculative.py:326-338` | `model_type == "deepseek_v4"` → 改写成 `deepseek_mtp`,`architectures=["DeepSeekV4MTPModel"]`,`n_predict = num_nextn_predict_layers`(**=1**) |
| `config/speculative.py:810-820` | `method == "dspark"`:复用完整 V4 config,`architectures=["DSparkDraftModel"]` |
| `model_executor/models/registry.py:631` | `DeepSeekV4MTPModel` → `vllm.models.deepseek_v4.DeepSeekV4MTP`(`nvidia/mtp.py:265`) |
| `model_executor/models/registry.py:614` | `DSparkDraftModel` → `vllm.models.deepseek_v4.DSparkDeepseekV4ForCausalLM`(`nvidia/dspark.py:306`) |
| `nvidia/model.py:1250` | target 的 `WeightsMapper`:`"mtp." -> "model.mtp."` |
| `nvidia/model.py:1387/1390` | target **跳过**所有 `mtp.*` 权重(`skip_weight_name_before_load` / `AutoWeightsLoader(skip_substrs=["mtp."])`),留给草稿 |
| `nvidia/model.py:1379` | `get_mtp_target_hidden_states()`:给草稿喂 target 的 hc_head 前残差流 |
| `nvidia/mtp.py:198` | MTP 层键 = `layers.{num_hidden_layers + i}`(=43/44/45),**不是** `mtp.*` |
| `nvidia/mtp.py:368-373` | 权重名 `mtp.{i}.` → `model.layers.{num_hidden_layers+i}.` 重映射 |
| `nvidia/mtp.py:494-524` | 再把 block 张量插到 `.mtp_block.` 之下;head 类张量提到 model 级 |
| `nvidia/dspark.py:132` | draft 层前缀 = `layers.{num_hidden_layers + i}`(=43/44/45) |
| `nvidia/dspark.py:502` | `_remap_dspark_name`:`mtp.{i}.` → `model.layers.{i}.`(**ModuleList 下标**,参数名口径) |
| `v1/worker/gpu_model_runner.py:596` | `use_dspark()` 在 V1 runner 里直接 `raise`:**dspark 必须用 V2 runner**(`v1/worker/gpu/model_runner.py`) |
| `v1/worker/gpu/model_runner.py:284,395,633` | draft 在 `load_model` 内加载 → 再 `initialize_kv_cache` → 再 `profile_run`(常驻显存会被 profile 计入) |

由此得到对本仓库的两条硬结论:
1. `is_lk_moe_mtp_layer()` 的 `mtp.` 前缀规则**对 DS-V4 草稿永不命中**(上游 DS-V4 的 mtp/dspark 都把层建在 `model.layers.43+`),
   它实际服务于"目标模型内嵌 `mtp.` 模块"的架构(qwen3_5 / mimo_v2 / minimax_m3 等);
2. 想让草稿常驻 GPU,**唯一符合"复用优先"的做法**就是设置 lk 现成开关
   `LVLLM_GPU_RESIDENT_MOE_LAYERS`(常驻层由 `quantization/mxfp4.py:548` 决定拿 CUDA 权重),
   不新增任何 vLLM 代码 —— 已实现在 `scripts/serve_lk_port.sh`。
