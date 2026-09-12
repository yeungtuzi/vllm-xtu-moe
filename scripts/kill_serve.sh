#!/usr/bin/env bash
# 安全杀掉 vllm 服务进程 —— 专治 TRIED_AND_REVERTED.md 里 M8 的三次复发。
#
# 为什么不能直接用 pkill -f "vllm serve ...":
#   调用它的 shell 自己的命令行里就含有 "vllm serve"(尤其是 heredoc/脚本里),
#   `pkill -f` 匹配整条 cmdline ⇒ 会把调用者一起杀掉(本会话发生过 3 次:
#   第 96 轮、第 102 轮、第 110 轮)。括号技巧 `serv[e]` 也救不了 —— 因为真实调用
#   "vllm serve <path>" 就在同一条命令行里。
#
# 做法:只看 /proc/<pid>/cmdline 的 **argv[0]**(可执行文件路径)是否为本环境的
#   vllm/python,并且 argv 里含 "serve";同时排除自身与所有祖先进程。
# 用法:scripts/kill_serve.sh            # 杀并等显存释放
#       WAIT=30 scripts/kill_serve.sh
set -uo pipefail
WAIT="${WAIT:-22}"
python3 - "$WAIT" <<'PY'
import os, sys, glob, time
wait_s = float(sys.argv[1])
ENV_PY = '/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/'
me, anc = os.getpid(), set()
p = me
while p and p != 1:                      # 收集自身与全部祖先
    anc.add(p)
    try: p = int(open(f'/proc/{p}/stat').read().split()[3])
    except Exception: break
killed = []
for f in glob.glob('/proc/[0-9]*/cmdline'):
    pid = int(f.split('/')[2])
    if pid in anc: continue
    try: argv = open(f, 'rb').read().split(b'\0')
    except Exception: continue
    if not argv or not argv[0]: continue
    a0 = argv[0].decode('utf8', 'ignore')
    rest = b' '.join(argv[1:]).decode('utf8', 'ignore')
    if a0.startswith(ENV_PY) and ('serve' in rest or '/vllm' in a0):
        try: os.kill(pid, 9); killed.append(pid)
        except Exception: pass
print(f'[kill_serve] killed={killed}')
PY
sleep "$WAIT"
echo "[kill_serve] 残留=$(ps -eo args | grep -ac 'vllm serv[e]')"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
rm -f /dev/shm/xiaotu_ep_*.bin
echo "[kill_serve] 完成(显存应已释放;再启动前请确认上面均为 0 MiB —— M9)"
