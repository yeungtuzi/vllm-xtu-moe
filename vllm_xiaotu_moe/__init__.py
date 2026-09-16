# ---------------------------------------------------------------------------
# 环境变量"文件桥"(必须在任何子模块 import 之前执行)
#
# 实测(2026-09-15,vLLM 0.29.1rc1.dev95):**EngineCore 子进程的 environ 与 launcher
# 不是同一份** —— spawn 时部分 `XIAOTU_*` 会被**静默丢弃**。已实测被丢过的有:
#   XIAOTU_MOE_THREADS / XIAOTU_MOE_SPIN_IDLE_US / XIAOTU_MOE_GPU_RESIDENT_LAYERS /
#   XIAOTU_RELEASE_SOURCE / MEMTRACE_INTERVAL / TAG / PORT
# 而这个包在 **launcher 与 EngineCore 两个进程里都会被 import**,所以在这里补一份
# 从文件读入的兜底:真实环境优先,缺失的才用文件补齐。
#
# 已经咬过我们两次(都是"设了开关却没生效、且毫无日志"):
#   * §382:`XIAOTU_RELEASE_SOURCE` 丢了 ⇒ 逐层释放从未执行;
#   * §414:`XIAOTU_MOE_GPU_RESIDENT_LAYERS` 丢了 ⇒ 常驻层实验完全没生效。
# 用 `XIAOTU_ENV_FILE` 指定文件路径(默认 /tmp/xiaotu_env)。
# ---------------------------------------------------------------------------
import os as _os


def _xtu_load_env_file() -> int:
    """把 `$XIAOTU_ENV_FILE` 里的键灌进 os.environ。

    两种模式(**这不是小事,踩过一次大坑**):

    * `XIAOTU_ENV_FILE` **被显式指定** ⇒ **以文件为准(覆盖)**。
      原因:EngineCore 的 environ 是被**重建**的(实测 2711 条 vs launcher 76 条),
      重建时**只保留它认识的名字** —— 于是"已经在环境里"的变量反而会被丢掉,
      而"由本桥新增"的变量能被保留(实测 XIAOTU_MOE_RESIDENT_BUDGET_GB 存活,
      而 XIAOTU_MOE_THREADS/_SPIN_IDLE_US/_GPU_RESIDENT_LAYERS 全被丢)。
      启动脚本(serve_*.sh)都显式导出该变量,所以覆盖不会引入不一致。
    * `XIAOTU_ENV_FILE` **没设** ⇒ 退到默认路径 `/tmp/xiaotu_env`,但**只补缺失的键**
      (setdefault 语义)。因为那个路径是**全局共享**的:2026-09-16 实测 V4 跑在
      V4.1 之后,`/tmp/xiaotu_env` 里还留着 V4.1 的
      `THREADS=184 / SPIN_IDLE_US=5000 / GPU_RESIDENT_LAYERS=`(空)——
      覆盖真实环境后,两 rank 共 368 个自旋线程挤 192 核、12 层 GPU 常驻静默失效,
      服务在 `determine_available_memory` 里被自家 worker 池看门狗 abort(§497),
      而日志上**一个字都没有**。旧值若与显式环境冲突,现在只告警、不覆盖。

    无论哪种模式都**逐行打印实际生效的键**,这个桥再也不允许"设了没生效且无声"。
    """
    explicit = _os.environ.get("XIAOTU_ENV_FILE")
    path = explicit or "/tmp/xiaotu_env"
    n = 0
    try:
        with open(path) as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if not k:
                    continue
                cur = _os.environ.get(k)
                if explicit:
                    _os.environ[k] = v
                    print(f"[vllm-xtu-moe/env] {k}={v}"
                          + ("" if cur is None or cur == v else f"  (覆盖环境里的 {cur})"),
                          flush=True)
                    n += 1
                elif cur is None:
                    _os.environ[k] = v
                    print(f"[vllm-xtu-moe/env] {k}={v}  (默认桥 {path} 补齐)", flush=True)
                    n += 1
                elif cur != v:
                    print(f"[vllm-xtu-moe/env] ⚠️ 忽略默认桥 {path} 的 {k}={v}"
                          f"(显式环境为 {cur});要采用它请显式导出 XIAOTU_ENV_FILE",
                          flush=True)
    except OSError:
        if explicit:
            print(f"[vllm-xtu-moe/env] ⚠️ XIAOTU_ENV_FILE={path} 读不到", flush=True)
    return n


_xtu_env_filled = _xtu_load_env_file()

"""vllm-xtu-moe: vLLM 主线混合推理插件(CPU 专家 + GPU 注意力/长 prefill)。

本包是 vLLM 主线的 OOT 插件,通过官方 `vllm.general_plugins` 入口加载。
内置的 xiaotu CPU 引擎(AVX-512 VNNI/BF16;MXFP4/FP8/INT4/BF16)作为计算内核,
由 `mixed_experts` 注册到主线的 CPU experts 后端槽位;`mainline_shims` 提供
混合模式所需的主线配合(上游尚未合并)。用法见 README 与 docs/RUNBOOK.md。
"""
from . import hybrid_model  # noqa: F401
from . import mixed_experts  # noqa: F401
from . import mainline_shims  # noqa: F401
from . import ple_offload  # noqa: F401

# 混合模式(VLLM_EXPERTS_LOAD_DEVICE=cpu)下,把主线的 CPU experts 后端注册为
# xiaotu 引擎(无 AMX 依赖)。非混合模式时是 no-op。
mixed_experts.register_mixed_cpu_backend()

# 在原生 vLLM 上补齐混合模式所需的 4 处主线行为:
# 专家权重建在 CPU / oracle 优先选 CPU 后端 / 跳过 AMX 重打包 / 通知 experts 后端。
# 已合并相应改动的主线上重复应用等于 no-op(XIAOTU_MAINLINE_SHIMS=0 可关闭)。
mainline_shims.apply_mainline_shims()

# Qwen3.8-Flash-Next 的 PLE n-gram 表(~51 GB)可放到主机内存(UVA 访问),
# 让单卡 40 GB 也能跑起该模型;默认关闭,设 XIAOTU_PLE_CPU=1 打开。
_ple_hooks = ple_offload.install()
if _ple_hooks:
    print(f"[vllm-xtu-moe/ple] PLE 表走主机内存: {', '.join(_ple_hooks)}", flush=True)
