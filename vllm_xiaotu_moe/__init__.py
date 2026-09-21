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


def _xtu_death_debug() -> None:
    """`XIAOTU_DEATH_DEBUG=1`:给 SIGTERM/SIGINT/SIGHUP/SIGUSR1 装栈 dump 处理器。

    为什么要它(2026-09-21,长 prompt worker 静默死亡排查):

    * 长 prompt 走 GPU 预填充之后,worker 会在 ~30–220 s 内**静默消失**;
    * `PYTHONFAULTHANDLER=1` 只为 SIGSEGV/SIGABRT/SIGBUS/SIGFPE/SIGILL 打印,
      而实测**一声不吭** ⇒ 不是这几类致命信号;
    * cgroup `/sys/fs/cgroup/.../memory.events` 的 `oom_kill` 全树为 **0**,
      `MemAvailable` 1.56 TB ⇒ 不是 OOM-kill(本也非 SIGKILL 不可捕获);
    * `Mlocked` 全程 27 MiB(上限 188.93 GiB)⇒ 不是 RLIMIT_MEMLOCK。

    剩下唯一没被排除的解释是 **SIGTERM**:`faulthandler` 默认**不注册**它,
    而 SIGTERM 的默认处置就是静默终止 —— 没有 traceback、没有 core、没有日志。
    这里显式注册,既能看到死亡瞬间的线程栈,也能当作判定实验:
    **若注册后不再死亡,即证明有代码在向 worker 发 SIGTERM。**
    """
    if _os.environ.get("XIAOTU_DEATH_DEBUG") != "1":
        return
    import faulthandler as _fh
    import signal as _sig
    import sys as _sys
    import threading as _th
    import time as _time

    _pid = _os.getpid()
    _path = _os.environ.get("XIAOTU_DEATH_LOG") or f"/tmp/xtu_sig_{_pid}.log"
    try:
        # 专用文件:worker 的 sys.stderr 可能被 vLLM 的日志层重定向/缓冲,
        # 2026-09-21 第一版把 dump 写 stderr,结果「没崩」但也没有 dump,无法分辨
        # 是「信号没来」还是「dump 丢了」。写自有 fd 才能分辨。
        _fp = open(_path, "a", buffering=1)
    except OSError:
        _fp = _sys.stderr

    _fh.enable()

    def _log(_msg: str) -> None:
        print(f"[vllm-xtu-moe/death pid={_pid}] {_msg}", file=_fp, flush=True)

    _log(f"diagnostics ON  ({_time.strftime('%H:%M:%S')})")
    for _name in ("SIGTERM", "SIGINT", "SIGHUP", "SIGUSR1", "SIGUSR2"):
        _s = getattr(_sig, _name, None)
        if _s is None:
            continue
        try:
            _fh.register(_s, file=_fp, all_threads=True, chain=False)
            _log(f"faulthandler registered {_name}")
        except Exception as _e:  # noqa: BLE001
            _log(f"faulthandler register {_name} failed: {_e}")

    # SIGTERM 再叠一个 Python 处理器:faulthandler 的 dump **不写信号编号**,
    # 而我们需要知道到底是哪个信号(以及它到达的准确时刻)。
    # 注意:Python 处理器会**覆盖**同一信号的 faulthandler 注册;这里是有意的。
    def _on_sigterm(_signum, _frame):  # pragma: no cover - 诊断路径
        _log(f"*** 收到 SIGTERM({_signum}) at {_time.strftime('%H:%M:%S')} ***")
        _fh.dump_traceback(file=_fp, all_threads=True)
        _fp.flush()

    try:
        _sig.signal(_sig.SIGTERM, _on_sigterm)
        _log("python handler installed for SIGTERM")
    except Exception as _e:  # noqa: BLE001
        _log(f"python handler install failed: {_e}")

    _t0 = _time.time()

    def _beat() -> None:
        while True:
            _time.sleep(10.0)
            _log(f"alive +{_time.time() - _t0:.0f}s threads={_th.active_count()}")

    if _os.environ.get("XIAOTU_DEATH_BEAT", "1") == "1":
        _th.Thread(target=_beat, daemon=True).start()


_xtu_death_debug()

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
