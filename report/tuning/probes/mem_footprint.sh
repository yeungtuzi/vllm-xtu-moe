#!/usr/bin/env bash
# 常驻内存足迹探针:起一个服务(任意 env/参数),采样到 ready,报**峰值**常驻集。
#
# 为什么需要它(§501/目标 item 5):参考实现(LvLLM + lk_moe)在它自己的文档里写
# "DeepSeek-V4.x peak ≈ 590 GB resident",而我们实测 **851 GB**(v41el/v41dsp2 的 .mem)。
# 要定位那多出来的几百 GB,第一步必须**在同一台机器、同一个模型、同一把尺子**下把参考实现
# 的数字自己量出来 —— 不能拿文档数字跟我们的数字比。
#
# 这把尺子:`ps -eo pid,rss --sort=-rss | head -1` 找最胖进程,读它的
# `/proc/<pid>/smaps_rollup`(Rss/Pss/Anonymous/Private_Dirty)+ `numactl --hardware` 的
# 每 node 空闲。每 `INTERVAL` 秒一次,直到服务 ready(或超时)。
#
# 用法:
#   TAG=ref41 PORT=8200 RUN_ENV="LVLLM_MOE_NUMA_ENABLED=1 ..." bash report/tuning/probes/mem_footprint.sh -- \
#       /home/user/anaconda3/envs/lvllm/bin/python -m vllm.entrypoints.openai.api_server --model ... --port 8200
#   ENV_FROM_AB=abl_a bash ...   # 也可以只给 TAG,用脚本里给的 RUN_ENV
#
# 产物:report/tuning/logs/<TAG>.memfoot(原始采样)+ 终端一行汇总。
# License: Apache-2.0
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTDIR="$ROOT/report/tuning/logs"; mkdir -p "$OUTDIR"
TAG="${TAG:-memfoot_$(date +%m%d_%H%M%S)}"
PORT="${PORT:-8200}"
INTERVAL="${INTERVAL:-15}"
READY_TIMEOUT="${READY_TIMEOUT:-2400}"
OUT="$OUTDIR/$TAG.memfoot"
LOG="$OUTDIR/$TAG.memfoot.log"
RUN_ENV="${RUN_ENV:-}"

echo "[memfoot] TAG=$TAG PORT=$PORT OUT=$OUT"
rm -f "$OUT" "$LOG"
cd /tmp   # ⚠️ 必须在仓库外启动(见 NOTES §499:仓库根的 egg-info 会被 vLLM 插件加载器发现)

# shellcheck disable=SC2086
nohup env $RUN_ENV "$@" > "$LOG" 2>&1 &
PID=$!
echo "[memfoot] pid=$PID log=$LOG"
echo "### tag=$TAG pid=$PID cmd=$* env=$RUN_ENV" >> "$OUT"

python3 - "$PID" "$OUT" "$INTERVAL" "$PORT" "$READY_TIMEOUT" <<'PY'
import json, os, subprocess, sys, time, urllib.request
pid, out, interval, port, ready_timeout = (int(sys.argv[1]), sys.argv[2],
                                           float(sys.argv[3]), int(sys.argv[4]),
                                           float(sys.argv[5]))
def _ppid_map():
    m = {}
    for e in os.listdir("/proc"):
        if not e.isdigit():
            continue
        try:
            with open(f"/proc/{e}/stat", "rb") as fh:
                data = fh.read()
            # comm 可能含空格/括号 ⇒ 取最后一个 ')' 之后再 split
            rest = data[data.rfind(b")") + 2:].split()
            m[int(e)] = int(rest[1])          # 字段 4 = ppid
        except Exception:
            continue
    return m
def _rss_kb(p):
    try:
        with open(f"/proc/{p}/statm") as fh:
            return int(fh.read().split()[1]) * (os.sysconf("SC_PAGE_SIZE") // 1024)
    except Exception:
        return 0
def tree(root):
    """返回 (服务进程树的所有 pid, 总 RSS KB, 树内最胖 pid)。
    为什么必须按**进程树**而不是"全机最胖进程"(§503):TP=2 的服务是
    EngineCore + 2 个 Worker(还有 api server)多个进程,而且机器上别的常驻进程
    (如 dsh web ~1.8 GB)早期可能比正在加载的服务更胖 ⇒ 用"全机最胖"会量错对象。"""
    pm = _ppid_map()
    kids = {}
    for p, pp in pm.items():
        kids.setdefault(pp, []).append(p)
    pids, stack = [], [root]
    while stack:
        p = stack.pop()
        pids.append(p)
        stack.extend(kids.get(p, ()))
    tot, fat, fatr = 0, root, 0
    for p in pids:
        r = _rss_kb(p)
        tot += r
        if r > fatr:
            fat, fatr = p, r
    return pids, tot, fat, fatr
def node_free():
    try:
        t = subprocess.run(["numactl", "--hardware"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return {}
    d = {}
    for line in t.splitlines():
        if line.startswith("node ") and " free:" in line:
            p = line.split()
            d[p[1]] = int(p[3])
    return d

t0 = time.time(); peak = {"rss": 0, "pss": 0, "anon": 0, "pid": 0}
minfree = {}
ready = None
with open(out, "a") as fh:
    while time.time() - t0 < ready_timeout:
        pids, tot, fp, frss = tree(pid)
        rec = {"t": round(time.time() - t0, 1), "nproc_tree": len(pids),
               "tree_rss_kb": tot, "fattest_pid": fp, "pid": fp, "rss_kb": frss}
        if fp and os.path.exists(f"/proc/{fp}/smaps_rollup"):
            try:
                with open(f"/proc/{fp}/smaps_rollup") as g:
                    for line in g:
                        k, _, v = line.partition(":")
                        k = k.strip()
                        if k in ("Rss", "Pss", "Anonymous", "Private_Dirty", "RssAnon", "RssShmem"):
                            rec[k] = int(v.split()[0])
            except OSError:
                pass
        nf = node_free(); rec["node_free_mb"] = nf
        for k, v in nf.items():
            minfree[k] = min(minfree.get(k, v), v)
        # 峰值按 RssAnon(匿名)优先,缺失则用 Rss
        # 峰值按**服务进程树总 RSS**取(这才是"这台服务占了多少内存")
        if tot > peak.get("tree_rss_kb", 0):
            peak["tree_rss_kb"] = tot
            peak.update({k2: rec.get(k2, 0) for k2 in ("rss_kb", "Rss", "Pss", "Anonymous",
                                                      "Private_Dirty", "RssShmem", "nproc_tree")})
            peak["rid"] = os.getpid()
            peak["pid"] = fp
        fh.write(json.dumps(rec) + "\n"); fh.flush()
        # ready?
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as r:
                if r.status == 200:
                    ready = round(time.time() - t0, 1); break
        except Exception:
            pass
        if not os.path.exists(f"/proc/{pid}"):
            break
        time.sleep(interval)
    fh.write(json.dumps({"summary": True, "ready_s": ready, "peak": peak,
                         "min_node_free_mb": minfree}) + "\n")
gb = 1024 * 1024
print(f"[memfoot] ready={ready}s  peak(服务树总RSS)={peak.get('tree_rss_kb',0)/gb:.1f} GiB  "
      f"Rss={peak.get('Rss',0)/gb:.1f} GiB  Pss={peak.get('Pss',0)/gb:.1f} GiB  "
      f"Anonymous={peak.get('Anonymous',0)/gb:.1f} GiB  RssShmem={peak.get('RssShmem',0)/gb:.1f} GiB  pid={peak.get('pid')}")
print(f"[memfoot] min node free (MB) = {minfree}")
PY
