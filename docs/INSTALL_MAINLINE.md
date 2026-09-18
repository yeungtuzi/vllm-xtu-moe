# 安装指南 v0.2:mainline vLLM + vllm-xtu-moe

> 本文档**就是本项目从零验证的过程记录**:下面每一步都是在本机按顺序真实执行过的命令与输出,
> 你照着做就能得到同样的结果。凡"实测"字样后面都是真实观测,不是预期。
>
> 两种安装方式:
> * **方式 A(推荐,最快)**:装官方 vLLM wheel,再把我们的补丁打到已安装的 `site-packages`
>   —— 补丁是**纯 Python(3 个文件)**,不需要编译任何东西;
> * **方式 B(从源码)**:clone vLLM 主线源码 → 打补丁 → 按官方方式安装(需要编译 CUDA 扩展)。

---

## 0. 前置条件

| 项 | 要求 | 本机实测值 |
|---|---|---|
| OS / 架构 | Linux x86_64 | Ubuntu, x86_64 |
| Python | **3.12** | 3.12 |
| glibc | ≥ 2.34 | ✅ |
| CPU | AVX2 起;最佳 **AVX512-BF16**(AMD Zen4 / Intel SPR 均可) | 2× EPYC 9654(Zen4,**无 AMX**) |
| GPU | 本项目在 **SM80(A100)** 验证;引擎本身与 GPU 型号无关 | 2× A100-40GB |
| 内存 | 需放下**全部专家权重**(本模型 ~137 GiB / 全模型) | 1.5 TB |
| 网络 | 首次安装需要(拉 vLLM / 依赖) | ✅ |

> **与官方配置的差别**:官方推荐 8×H100 级;我们证明 **2×A100-40GB** 也能跑,
> 代价是把专家权重放在主机内存、用 CPU 计算。

---

## 1. 建一个干净环境

```bash
conda create -y -n xtu02 python=3.12
conda activate xtu02
python -m pip install -U pip
```

**实测**:✅(见 `内部调优记录 NOTES.md §334`)

---

## 2. 安装 vLLM

### 方式 A:官方 wheel(推荐)

```bash
pip install vllm==0.29.0          # 以当前 PyPI 最新为准
```

> ⚠️ **GPU 架构**:官方 wheel 默认按当前机器的 CUDA 架构编译。A100 是 SM80,
> 若官方 wheel 未包含 SM80 kernel,需要走方式 B 或用我们预编译的
> `vllm-<ver>+xtu.whl`(见第 3 节)。

### 方式 B:主线源码

```bash
git clone --depth 1 https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip install -e .     # 复用官方预编译二进制,避免数小时编译
```

---

## 3. 应用 vllm-xtu-moe 补丁

补丁位于本仓库 `patches/upstream/`,**按"上游 PR"组织**,并针对 vLLM 主线
**`dabc4362b`(2026-09-14)** 重新生成:

| 补丁 | 文件 | 作用 | 什么时候需要 |
|---|---|---|---|
| `pr0-handshake-timeout.patch` | 1 | 把硬编码的 `HANDSHAKE_TIMEOUT_MINS = 5` 变成可用 `VLLM_HANDSHAKE_TIMEOUT_MINS` 配置(CPU 引擎逐层构造 ~6 min,否则健康加载会被掐断) | **总是需要** |
| `pr1-experts-load-device.patch` | 6 | `VLLM_EXPERTS_LOAD_DEVICE=cpu`:让 routed-expert 权重**从构造期就落在主机**;FP8/INT4/BF16 oracle 在 GPU 主机上把 CPU 后端前置;混合模式下跳过 AMX prepack。**不改默认行为** | **总是需要**(方式 A 与 B 都要) |
| `pr2-fp8-sm80-o-proj.patch` | 2 | A100(SM80)的 FP8 o_proj + e4m3 字节编解码 helper | 只在 A100 上跑 FP8 注意力时需要 |
| `pr3-sm80-port.patch` | 21 | SM80 的 DS-V4 移植(mHC / sparse-MLA / indexer / rope-quant / 新内核文件) | 只在主线对 A100 支持不足时需要 |

> ⚠️ **`mxfp4` 的那部分补丁已被上游吸收** —— 新版主线自带 native
> `Mxfp4MoeBackend.CPU` 与 `prepare_mxfp4_moe_layer_for_cpu`,
> 所以旧 `pr1` 里的 3 个 hunk 只剩 2 个需要打。**每次升级主线都要重问"这条上游做了吗"**,
> 见 `内部纪律 IRON_RULES.md` R10.5。

用仓库自带脚本一键应用(它会自己找 `<tree>/vllm`):

```bash
# 方式 A:打到已安装的 site-packages
TREE=$(python -c "import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))")
bash scripts/apply_xtu_patches.sh "$TREE"

# 方式 B:打到源码树
bash scripts/apply_xtu_patches.sh /path/to/vllm

# 先干跑不落地:
DRY=1 bash scripts/apply_xtu_patches.sh "$TREE"
```

**实测(补丁对主线的可用性)**:在干净主线 `dabc4362b` 的 `git archive` 导出上,
按 `pr0 → pr1 → pr2 → pr3` **顺序全部干净应用,0 个失败 hunk**;应用后的源码树与我们的
开发树 **逐文件一致**(`diff -rq` 只剩编译产物 `.so`)。
复现:`scripts/check_upstream_drift.sh`(会同时报告"当前上游 HEAD 有没有 precompiled wheel")。

---

## 4. 安装 vllm-xtu-moe 插件

```bash
pip install vllm_xtu_moe-0.2.0-<tag>.whl
```

wheel 自带 **5 个 ISA 变体**的引擎 `.so`(scalar / avx2 / avx512_base / avx512_vnni / avx512_bf16),
**不需要编译器,也不需要 nvcc**;导入时按 `/proc/cpuinfo` 自动选最高可用变体。

---

## 5. 自检(不加载模型)

```bash
python -c "import xiaotu_moe; print(xiaotu_moe.load())"          # 引擎变体
python -c "import vllm_xiaotu_moe; print('plugin ok')"           # 插件可导入
python -c "
import importlib.metadata as md
print([e.value for e in md.entry_points().select(group='vllm.general_plugins')])
"                                                                 # 入口点已注册
```

**实测**:✅ 三项通过(见 §334)。

---

## 6. 起服务

```bash
export VLLM_EXPERTS_LOAD_DEVICE=cpu      # 关键:专家权重放主机
TAG=myrun PORT=8071 bash scripts/serve_mainline.sh
```

脚本默认值(与 0.1.0 的最优配置一致,见 `RELEASE_NOTES_v0.1.0.md`):

| 变量 | 默认 | 说明 |
|---|---|---|
| `TP` | 2 | 张量并行度 |
| `GPU_UTIL` | 0.80 | 显存占用上限 |
| `MAXLEN` | 8192 | 最大上下文 |
| `SEQS` | 8 | 最大并发序列 |
| `THREADS` | 60 | 引擎线程 = 每 rank 12 CCD × 5 核 |
| `RESIDENT` | 0-11 | 常驻 GPU 的 MoE 层数(12 层) |
| `XIAOTU_OOT_OVERRIDE` | 1 | 1=OOT 模型覆盖(零补丁);0=主线 RoutedExperts + 我们的 CPU 后端 |

---

## 6.5 已在主线跑通的两条路径(实测,2026-09-14)

| | 形态 A(`XIAOTU_OOT_OVERRIDE=1`,默认) | **形态 B(`=0`)** |
|---|---|---|
| 机制 | `ModelRegistry` 覆盖 `DeepseekV4ForCausalLM` + `dv4_nvidia.DeepseekV4MoE = CpuXiaotuMoE` | 主线 `RoutedExperts` 原样 + `mixed_experts` 把 CPU 后端指向我们的引擎 |
| 启动 | ❌ 加载期被掐(见"失败回退") | ✅ **`Application startup complete`** |
| KV | — | **22.04 GiB / 112,304 token** |
| 输出 | — | ✅ `"The capital of France is"` → `" Paris. The capital of Spain is Madrid"` |
| 引擎 | 逐层构造成功(日志 `[xiaotu] engine built`) | 被调用(日志 `xiaotu MOE_MXFP4 engine: E=256 H=4096 I=1024 …`) |
| 性能 | — | ⚠️ **2.2 s/token(待修)** |

**三个必须的环境修复**(启动器已默认,踩过一次就别再踩):

```bash
--safetensors-load-strategy prefetch          # 分片读取 2.5-4.8 s/片 → 330 s 总加载
                                              # 加这个后 133 s(实测)
--kernel-config '{"enable_jit_warmup": false}' # A100 无 fp8e4nv,Triton 预热会 ValueError
VLLM_USE_FLASHINFER_SAMPLER=0                  # FlashInfer 用 --compress-mode=size(需 CUDA≥12.8)
                                               # 本机 CUDA 12.1 ⇒ nvcc fatal,worker 猝死
```

---

## 7. 验证

```bash
# 延迟/吞吐(固定协议:短输入 + 长输出)
PORT=8071 TAG=myrun L=256 OUT=512 CS="1 2 4" SERVER_TAG=myrun bash scripts/bench_lat.sh

# 数值门禁
XIAOTU_LAYER1_NPZ=fixtures/real_layer1_model.npz python scripts/test_block23_equiv.py

# greedy 一致性
python scripts/probe_greedy.py /tmp/greedy.json 8071
```

---

## 8. 已知限制与失败回退

| 现象 | 原因 | 处理 |
|---|---|---|
| `Free memory on device ... less than desired GPU memory utilization` | 上一次的 EngineCore/Worker 变成孤儿进程占着显存 | `bash scripts/kill_serve.sh`(按 argv[0] 精确匹配;**不要**用 `pkill -f "vllm..."`,会把自己杀掉) |
| `Worker proc VllmWorker-N died unexpectedly` + `nvcc fatal: Unknown option '--compress-mode=size'` | FlashInfer 0.6.18 需要 CUDA ≥ 12.8,本机是 12.1 | `VLLM_USE_FLASHINFER_SAMPLER=0` |
| `ValueError: type fp8e4nv not supported in this architecture` | A100(SM80)没有 fp8e4nv,被 Triton JIT warmup 触发 | `--kernel-config '{"enable_jit_warmup": false}'`(或打 `pr2/pr3`) |
| 启动超时(worker 还在建引擎) | **两个**看门狗都可能先到:① `VLLM_ENGINE_READY_TIMEOUT_S`(默认 600 s,API server 等 EngineCore);② **`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`(默认 300 s,EngineCore 等 worker 响应)** | 启动器已把两者都设为 **3600**。43 层 CPU 引擎逐层构造需要 ~6 分钟,不调大必然在加载中途被杀(实测:05:43:43 worker 初始化 → 05:48:56 被 300 s 那个看门狗掐掉) |
| 显存不足放不下常驻层 | 常驻层每层 1.6 GiB/rank | 调小 `RESIDENT`(如 `0-9`) |
| 长上下文 OOM | 12 层常驻只剩 ~0.6 GiB KV | 用 `RESIDENT=0-10`(11 层)换回 KV |

---

*本文件的每一步都在本机执行并记录;执行日志见 `内部调优记录 logs/` 与 `内部调优记录 NOTES.md §334`。*

---

## 9. 纯插件路径(`LEVEL=0`):能用到什么程度

```bash
LEVEL=0 bash scripts/install_mainline.sh    # 一个补丁都不打,只装插件
```

只走 `vllm.general_plugins` 入口 + OOT 注册表覆盖(即 `内部文档 UPSTREAM_DRIFT.md` 的**形态 0**)。

| 能力 | `LEVEL=0`(纯插件) | `LEVEL=1`(默认,pr0+pr1) |
|---|---|---|
| 插件对主线 **可 import / 可注册 / 可覆盖** | ✅ | ✅ |
| `import vllm_xiaotu_moe` + 入口点被发现 | ✅ | ✅ |
| 数值门禁(`test_block23_equiv.py`,纯引擎) | ✅ | ✅ |
| **真实起服务(DS-V4 / 138 GB 专家)** | ❌ | ✅ |
| 引擎逐层构造不撞握手超时 | ❌ | ✅ |

**为什么 `LEVEL=0` 在本机不能用于真实服务** —— 两件上游还没有的能力,恰好都只能靠补丁:

1. **`VLLM_EXPERTS_LOAD_DEVICE=cpu`(pr1,3 个文件)**
   没有它,138 GB 的 routed-expert 权重无处安放(2×A100-40GB 装不下)。
   这是本方案能成立的前提,主线目前没有等价的公开开关。
2. **可配置的引擎握手超时(pr0,1 个文件)**
   上游把 `HANDSHAKE_TIMEOUT_MINS` 硬编码成 5 分钟;而 CPU 引擎要逐层构造 43 层,
   必然超时。pr0 只是把这个常量变成可配置 —— **不改变任何默认行为**。

⇒ 所以 v0.2 的"最小补丁集"就是 **pr0 + pr1 = 4 个文件**;
`pr2`/`pr3` 是 **A100/SM80 专属**(非 A100 机器不需要),
按 `LEVEL=2/3` 选装。逐块理由见 `patches/upstream/*.patch` 头部注释与 `内部文档 UPSTREAM_DRIFT.md`。

## 10. 装完之后:起服务前先过一遍宿主契约自检

`install_mainline.sh` 结束时已自动调用,也可以随时手动跑:

```bash
bash scripts/check_mainline_env.sh              # 检查当前 shell
TAG=myrun bash scripts/check_mainline_env.sh    # 检查某个已启动实例的 .env 记录
```

它断言的是**那些缺失时不会报错、只会静默变慢(实测可达 30×)的开关**
—— 完整清单与证据见 **`(内部) PLUGIN_INTERFACE.md`**。最要紧的两个:

* `XIAOTU_MOE_SPIN_IDLE_US=0`(缺失 ⇒ 解码从 37 ms 漂到 1225 ms/token)
* `XIAOTU_MOE_NSLICE_SMALL=0`(缺失 ⇒ 60 个 worker 全部参与每一相)

`serve_mainline.sh` **已默认带上这两个**,所以正常路径不用管;
但你自己拼命令行时一定要带上,否则会得到一个"能跑但慢 30×"的服务。
