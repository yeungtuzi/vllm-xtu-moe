# UPSTREAM_DRIFT — fork 与 vLLM 主线的漂移审计(v0.2 基线)

> **目的**:为「让 vllm-xtu-moe 完全支持 vLLM 主线最新版,并提供简化的 patch / 安装方式」提供可核对的数字。
> **方法**:全部用本机三棵树的 git 对象与工作树实测,命令写在每节末尾;**不引用任何估算**。
> **日期**:2026-09-14(审计时主线 HEAD = `6c73b08dec`,2026-09-08)
>
> ⚠️ **本文档的基线已于 2026-09-14 晚更新**:主线基线从 `6c73b08dec` 升到 **`dabc4362b`**
> (346 commits / 6 天)。**第二节起的所有数字仍是旧基线的审计结果,尚未重算**;
> 升级过程、新的漂移形态与复验结果见 `report/tuning/NOTES.md` §368。
> 下次重算请先跑 `scripts/check_upstream_drift.sh --full`,并注意
> **"上游 HEAD"不等于"我们能装的 commit"**(本机不能从源码编译,必须挑有 precompiled wheel 的 commit,
> 见 `report/tuning/IRON_RULES.md` R10.2)。


---

## 一、三棵树与血缘

| 代号 | 路径 | 内容 | git 状态 |
|---|---|---|---|
| **A** | `/tmp/ml_clean`(由 `process_data/ref/repos/vllm-mainline` 的 `git archive HEAD` 导出) | **vLLM 主线 HEAD** `6c73b08dec`(2026-09-08) | 干净提交态 |
| **B** | `/home/user/lvllm/process_data/ref/repos/vllm-mainline`(工作树) | 主线 + **未提交的 DS-V4/SM12x 本地补丁** | 30 文件,`+7044 / −134` |
| **C** | `/home/user/lvllm/Lvllmds4-x` | 参考 fork + **我们的移植提交** `faf95dd5b` | 17,779 提交(含完整 vLLM 史) |

血缘链(由 `C` 的提交作者与信息实测得出):

```
vLLM mainline
  └─ jasl/vllm (deepseek-v4 分支)            ← C 的历史里可见 jasl 的 2026-06 提交
       └─ yhfgyyf/vllm-deepseek-v4-sm89      ← SM80/SM89/SM120 适配(2026-06 ~ 07)
            └─ guqiong96/Lvllmds4-x          ← lk_moe 混合推理(2026-07-19 起)
                 └─ faf95dd5b  ← 【我们的移植】重绑 lk_moe → xiaotu_moe
```

**关键事实**:`A` **已经原生包含 `vllm/models/deepseek_v4/`**(nvidia / amd / xpu 三套后端)。
⇒ "让主线支持 DS-V4"**不需要我们打补丁**,主线已经支持。

```bash
ls /tmp/ml_clean/vllm/models/deepseek_v4/     # __init__ amd attention.py common compressor.py nvidia ...
```

---

## 二、量化漂移

### 2.1 三个方向的文件级差异数

| 比较 | 差异条目数 |
|---|---|
| A → B(主线 + DS-V4 本地补丁) | **52** |
| A → C(主线 → 我们的 fork) | **1631** |
| B → C | 1648 |

> ⚠️ `A → C` 的 1631 **不能**当作"我们要背的补丁量":`C` 基于**较早的主线**,而主线自己往前走了几个月,
> 这 1631 里绝大部分是 **vLLM 自身的演进**(以及 jasl/yhfgyyf 两条 DS-V4 分支的历史),不是 lk 的改动。

### 2.2 真正属于"CPU-GPU 混合推理"的改动(隔离后)

`92c9b76bb^..a9f97ec09`(lk 系列,含后续修复):

```
29 files changed, 1192 insertions(+), 562 deletions(-)
```

按性质拆开:

| 类别 | 文件 | 行数 | 是否必要 |
|---|---|---|---|
| **核心编排** | `vllm/model_executor/layers/fused_moe/routed_experts.py` | **+620** | ✅ 必要(CPU 解码/预填充路径、层分类) |
| | `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` | +74 | ✅(其中我们的移植已改为调用**上游** `forward_monolithic`/`forward_modular`) |
| | `vllm/model_executor/layers/fused_moe/config.py` | +16 | ✅ 混合放置配置 |
| **环境变量登记** | `vllm/envs.py` | **+191** | 🟡 只为了让 `vllm.envs.LVLLM_*` 可读 ⇒ **插件可直接 `os.environ`,非必要** |
| **量化/加载钩子** | `quantization/{fp8,modelopt,mxfp4}.py`、`unquantized_fused_moe_method.py`、`compressed_tensors_moe_*.py`(4 个)、`oracle/fp8.py` | +170(9 文件) | 🟡 目的是**别在 GPU 上构造 CPU 层的权重**;插件已用 `CpuMegaExpertsParams` 从构造期就持 CPU 权重 |
| **NUMA / 系统** | `utils/numa_utils.py`、`utils/system_utils.py`、`platforms/__init__.py`、`utils/import_utils.py` | +80 | 🟡 辅助;我们的引擎自带 per-shard `mbind` |
| **非代码** | `README.md`/`README_EN.md`(468)、`.github/dependabot.yml`(−30)、`config.yaml`(+45)、`pyproject.toml`、`requirements/cuda.txt` | ~490 | ❌ 与运行时无关 |
| **CLI 重命名** | `entrypoints/cli/openai.py → openai_entrypoint.py`(0 行) + `main.py`/`run_batch.py` 改 import | 8 | ❌ 与混合推理无关 |

**结论**:`1192` 行里 **约 490 行是非必要的(docs/CI/CLI/config)**;
剩下的 ~700 行里,**约 380 行(envs + 量化钩子 + NUMA)可以完全由插件在运行时替代**。

### 2.3 我们自己的移植很小

`faf95dd5b`(**唯一**属于本项目的 fork 改动):

```
vllm/.../fused_moe/routed_experts.py | 92 +++++++++++-----------
vllm/.../fused_moe/runner/moe_runner.py | 30 ++++---
2 files changed, 68 insertions(+), 54 deletions(-)
```

它做的是:**把 lk 自研的 `_gpu_prefill`(专有 CUDA kernel)换成上游 vLLM 的
`forward_monolithic` / `forward_modular`** —— 即"复用优先:上游 → lvllm → 自研"的落地。

### 2.4 仓库里已有的补丁清单(审计时实测)

`patches/` 下已经存在一套按"上游 PR"组织的补丁,量化如下:

| 文件 | 涉及文件 | +行 | −行 | 体积 | 性质 |
|---|---|---|---|---|---|
| `patches/upstream/pr1-experts-load-device.patch` | **3** | **+56** | **−5** | 5.4 KB | ✅ **最小使能补丁**:`VLLM_EXPERTS_LOAD_DEVICE=cpu` + MXFP4 oracle 前置 CPU 后端 + `routed_experts` 允许在主机侧构造权重。**不改默认行为** |
| `patches/upstream/pr2-fp8-sm80-o-proj.patch` | — | — | — | 19.8 KB | 🟡 A100(SM80)的 FP8 o_proj 支持 |
| `patches/upstream/pr3-sm80-port.patch` | — | — | — | 251 KB | 🟡 SM80 DS-V4 完整移植(主线若已支持 A100 则不需要) |
| `patches/mainline_mixed_mode_generic.patch` | 4 | +94 | −5 | 8.4 KB | 本地工作树快照(fp8/int_wna16/unquantized 三个 oracle + quantization/fp8.py) |
| `patches/mainline_fp8_oracle_mixed_mode.patch` | 1 | +7 | −1 | 0.8 KB | 局部补充 |
| `patches/mainline_sm80_mixed_mode.patch` | 27 | +7004 | — | 285 KB | **= §1 表 B 的本地快照**,不是给用户的补丁 |

⇒ **"让 CPU 专家后端在 GPU 主机上可被选中"这一件事,最小补丁只有 3 文件 / 61 行改动**,
这就是 v0.2 要发布的 `patches/` 的核心。

---

## 三、插件已经覆盖了多少?(这是 v0.2 的关键)

`vllm_xiaotu_moe` 目前 **3,641 行**(实测 `wc -l`):

| 文件 | 行数 | 作用 |
|---|---|---|
| `hybrid_model.py` | 1190 | `CpuXiaotuMoE`(CPU MoE 模块)+ `HybridDeepseekV4ForCausalLM` + `register()` |
| `gpu_prefill.py` | 970 | 分层 GPU 预填充 |
| `mixed_experts.py` | 892 | `register_mixed_cpu_backend()`:把 `Mxfp4MoeBackend.CPU` 指向 xiaotu 引擎 |
| `mainline_shims.py` | 452 | 主线兼容垫片 |
| `ple_offload.py` | 111 | PLE 卸载 |
| `__init__.py` | 26 | 导出 |

**插件已内置两条"零核心补丁"路径**(由 `XIAOTU_OOT_OVERRIDE` 切换):

| 模式 | 开关 | 机制 | 是否改主线文件 |
|---|---|---|---|
| **A · OOT 模型覆盖** | `XIAOTU_OOT_OVERRIDE=1`(默认) | `ModelRegistry.register_model("DeepseekV4ForCausalLM", …)` + `dv4_nvidia.DeepseekV4MoE = CpuXiaotuMoE`;权重从构造期就在 CPU(`CpuMegaExpertsParams`) | **否** |
| **B · 主线 RoutedExperts + CPU backend** | `XIAOTU_OOT_OVERRIDE=0` | 主线 `RoutedExperts` 不动,`mixed_experts.register_mixed_cpu_backend()` 把 CPU MoE 后端指向 xiaotu(等价上游 `VLLM_EXPERTS_LOAD_DEVICE=cpu` + CPU 后端) | **否** |

⇒ 对照 §2.2:**fork 那 ~700 行必要编排,插件已经用 3,641 行独立实现了一遍**。
所以 v0.2 的合理目标是 **"零核心补丁"**(纯插件),fork 只作为**性能对照与兜底**保留。

**已验证的部分**:
* 装上 wheel 后,对 **mainline vLLM** `import vllm_xiaotu_moe` **成功**(实测,见 `RELEASE_NOTES_v0.1.0.md`);
* 引擎 5 个 ISA 变体随 wheel 分发,`xiaotu_moe.load()` 正常选到 `_avx512_bf16`。

**尚未验证(本目标第 3 条要做的)**:
* 模式 A / B 在**主线最新版**上**真的能起服务并出数**;
* 与 0.1.0(fork 路径)的**同机同协议性能对照**;
* 数值门禁与 greedy 一致性。

---

## 四、最小补丁清单(v0.2 目标形态)

```
【形态 0 · OOT 模型覆盖(默认,零补丁)】
    vLLM mainline(A) + pip install vllm-xtu-moe         补丁行数 = 0
    XIAOTU_OOT_OVERRIDE=1(默认):ModelRegistry 覆盖 DeepseekV4ForCausalLM,
    每层 ffn 直接是 CpuXiaotuMoE(权重从构造期就在 CPU)

【形态 1 · 最小补丁 + 纯插件(推荐给"想用主线原生机制"的用户)】
    vLLM mainline(A) + patches/upstream/pr1-experts-load-device.patch + wheel
    补丁规模 = 3 文件 / +56 −5(实测)
    作用     = 让 routed-expert 权重从构造期落在主机,且 MXFP4 oracle 在 GPU 主机上
               会把 CPU 后端前置 ⇒ 之后完全由插件提供 CPU 计算后端
    额外收益 = 主线原生 cudagraph / prefix caching / chunked prefill / spec decode 全部保留

【形态 2 · A100 专用(仅当主线对 SM80 支持不足时)】
    + pr2-fp8-sm80-o-proj.patch(19.8 KB)+ pr3-sm80-port.patch(251 KB)

明确不进补丁:envs.py 的 LVLLM_* 登记(插件直接读 os.environ)、
              README/CI/CLI 重命名/config.yaml、NUMA 辅助(引擎自带 per-shard mbind)
```

**判定标准(逐块给出"为什么主线做不到")**:凡能通过
`vllm.general_plugins` 入口 / `ModelRegistry.register_model` / `FusedMoEFactory` /
`Mxfp4MoeBackend.CPU` 注册 实现的一律**不进补丁**。

---

## 五、复现命令(审计本身)

```bash
# A:导出干净主线
git -C process_data/ref/repos/vllm-mainline archive HEAD | tar -x -C /tmp/ml_clean

# 三向差异计数
diff -rq /tmp/ml_clean/vllm <tree>/vllm | grep -v __pycache__ | wc -l

# 隔离 lk 混合推理这一系列
git -C Lvllmds4-x diff --stat 92c9b76bb^ a9f97ec09
# 隔离我们的移植
git -C Lvllmds4-x diff --stat faf95dd5b^ faf95dd5b

# 主线是否自带 DS-V4
ls /tmp/ml_clean/vllm/models/deepseek_v4/
```

---

## 六、风险与注意

| 风险 | 说明 | 缓解 |
|---|---|---|
| 主线 API 漂移 | 插件大量 import 主线的内部路径(如 `vllm.models.deepseek_v4.nvidia.model`),主线重构会直接打断 | `mainline_shims.py` 已有 452 行垫片;补一份"最小支持版本 + 冒烟测试" |
| 模式 A 的能力缺口 | OOT 覆盖的是模型类,**cudagraph / prefix caching / chunked prefill / spec decode 是否全部可用未验证** | 本轮之后逐项实测;模式 B 原生吃主线机制,可能是更优形态 |
| 结论只对当前主线 HEAD 成立 | 审计针对 `6c73b08dec` | 每次升主线重跑第五节命令即可 |
| 参考对照会丢 | 一旦不再用 fork,`lk_moe` 同机对照仍需 `lvllmds4-x` env | 保留该 env 不动(0.1.0 已如此) |

---

## 七、补丁对主线 HEAD 的可用性(实测 `patch --dry-run`)

在**干净主线 A**(`6c73b08dec` 的 `git archive` 导出)上逐个试打:

| 补丁 | 检查文件数 | 失败 hunk | 结论 |
|---|---|---|---|
| `pr1-experts-load-device.patch` | 3 | **0** | ✅ 干净可用 |
| `pr2-fp8-sm80-o-proj.patch` | 4 | **0** | ✅ 干净可用 |
| `pr3-sm80-port.patch` | 21 | **0** | ✅ 干净可用 |

```bash
cd /tmp/ml_clean            # 干净主线工作树(无 .git 也可,用 patch(1))
patch -p1 --dry-run < patches/upstream/pr1-experts-load-device.patch
```

⇒ **补丁集对当前主线 HEAD 是 rebase-clean 的**,不需要人工改行。
这直接支撑 v0.2 的第 4 条目标(一条命令安装):`git clone mainline → patch -p1 → pip install wheel`。

---

## 八、新基线 `dabc4362b`(2026-09-14)的补丁清单 —— **当前有效**

> 上表(§2.4 / §7)是 `6c73b08dec` 旧基线的审计结果。主线在 6 天内前进 **346 commits / 300 文件**,
> 已于 2026-09-14 完成升级(过程与复验见 `report/tuning/NOTES.md` §368)。**以下是现行数字。**

### 8.1 补丁规模(重新生成,实测全部干净应用)

| 补丁 | 文件 | 体积 | 作用 | 与旧基线的差异 |
|---|---|---|---|---|
| `pr0-handshake-timeout.patch` | **1** | 17 行 | 握手超时可配置 | 不变(上游仍硬编码 5 分钟) |
| `pr1-experts-load-device.patch` | **6** | 272 行 | 主机侧专家权重 + CPU 后端前置 + 跳过 AMX prepack | **缩小**:`oracle/mxfp4.py` 的 3 个 hunk 已被上游吸收(native `Mxfp4MoeBackend.CPU`) |
| `pr2-fp8-sm80-o-proj.patch` | **2** | 150 行 | A100 的 FP8 o_proj + e4m3 字节 helper | 重写(上游把 `fp8_einsum` 重构了) |
| `pr3-sm80-port.patch` | **21** | 7512 行 | SM80 DS-V4 移植(mHC/sparse-MLA/indexer/rope-quant + 新内核文件) | 重写;**其中 mHC 那处从 43 行冲突简化为 ~20 行门闸**,但整体仍大(含 `sparse_mla_kernels.py` 3517 行等新文件) |

### 8.2 可用性实测(强于 dry-run)

```bash
M=/path/to/vllm && cd $M && git archive dabc4362b | tar -x -C /tmp/ml_base
cd /tmp/ml_base
for p in pr0 pr1 pr2 pr3; do patch -p1 --forward < patches/upstream/$p-*.patch; done
diff -rq /tmp/ml_base/vllm $M/vllm | grep -v __pycache__
```

结果:**4 个补丁全部干净应用(0 failed hunks)**,应用后的源码树与开发树
**逐文件一致** —— `diff` 只剩编译产物(`*.so`、`vllm-rs`、`third_party/` 下由 wheel 解出的目录)。

### 8.3 新增前置约束:**必须挑"有 precompiled wheel 的 commit"**

本机 `nvcc` = CUDA 12.1 而 torch = 2.13.0+cu130,且无 Rust 工具链 ⇒ **不能从源码编译**;
这 346 个 commit 改了 **43 个 csrc 文件 + 5 个 cmake + 126 个 rust 文件** ⇒ 旧 `.so` 不可复用。
所以"升级到上游 HEAD"在本机**做不到**,只能升级到**已发布 cu130 轮子**的 commit:

| commit | 日期 | 有 cu130 轮子? |
|---|---|---|
| `6c73b08dec` | 2026-09-08 | ✅(旧基线) |
| **`dabc4362b`** | 2026-09-14 20:47 | ✅ **采用** |
| `d392ac836` | 2026-09-14 晚 | ✅ **下一次跟进目标** |
| `00972dfd72`(当时 HEAD) | 2026-09-14 21:44 | ❌ 404 |

判定命令见 `report/tuning/IRON_RULES.md` R10.2,或直接跑 `scripts/check_upstream_drift.sh`。
