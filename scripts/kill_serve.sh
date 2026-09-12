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
#
# 【2026-09-12 修复 · M8c】vLLM 的 EngineCore/Worker 子进程会把 **argv[0] 改写成
#   "VLLM::EngineCore" / "VLLM::Worker_TP<n>"**(实测 /proc/<pid>/cmdline 前两段
#   就是这两个字符串)。旧版只匹配 ENV_PY 前缀 ⇒ 只杀掉 API server,
#   **留下孤儿 EngineCore+Worker 各占 ~26 GiB/卡**,下一次启动就报
#   ValueError: Free memory on device cuda:1 (13.16/39.49 GiB) ... less than desired
#   GPU memory utilization (0.9, 35.54 GiB) —— 白等一轮 13 分钟加载。
#   现在把 "VLLM::" 前缀一并匹配。
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
    is_ours = a0.startswith(ENV_PY) or a0.startswith('VLLM::')
    if is_ours and ('serve' in rest or '/vllm' in a0 or a0.startswith('VLLM::')):
        try: os.kill(pid, 9); killed.append((pid, a0))
        except Exception: pass
print(f'[kill_serve] killed={killed}')
PY
sleep "$WAIT"
echo "[kill_serve] 残留=$(ps -eo args | grep -acE 'vllm serv[e]|VLLM::(EngineCore|Worker)')"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
rm -f /dev/shm/xiaotu_ep_*.bin
# 硬校验:任一卡 used >= 1024 MiB 就报错退出,避免又白等一轮 13 分钟加载(M9)
_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if echo "$_used" | awk '$1 >= 1024 {exit 1}'; then
  echo "[kill_serve] 完成(显存已释放,可以启动)"
else
  echo "[kill_serve] **警告:仍有卡占用 >=1 GiB,不要启动**(M9)"; exit 3
fi
