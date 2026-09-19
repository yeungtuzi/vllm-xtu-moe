#!/usr/bin/env bash
# GLM-5.3-Flash 生产服务(8070)一页纸状态:端口 / 进程 / 显存 / 关键日志行。
#
#   bash scripts/glm53_status.sh            # 看 8070 + TAG=glm53_prod
#   PORT=8090 TAG=glm53_t1 bash scripts/glm53_status.sh
#
# 退出码:0 = /v1/models 有响应(健康);1 = 没响应(死了或还在启动)。
# 只读,不会碰服务。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8070}"
TAG="${TAG:-glm53_prod}"
HOST="${HOST:-127.0.0.1}"
OUTDIR="${OUTDIR:-$ROOT/logs}"
LOG="$OUTDIR/$TAG.log"
PIDFILE="$OUTDIR/$TAG.pid"
GPUS="${GPUS:-0,1}"

rc=0
echo "=== GLM-5.3-Flash @$HOST:$PORT (tag=$TAG)"
echo "-- 健康检查"
if curl -sf -m 5 "http://$HOST:$PORT/v1/models" >/tmp/.glm53_models.$$ 2>/dev/null; then
  echo "   HTTP OK: $(python3 -c 'import json,sys;print(",".join(m["id"] for m in json.load(open(sys.argv[1]))["data"]))' /tmp/.glm53_models.$$ 2>/dev/null || echo '?')"
else
  echo "   ❌ 无响应(curl $HOST:$PORT/v1/models 失败)⇒ 服务没在跑,或还在加载(完整启动 ~5 min)"
  rc=1
fi
rm -f /tmp/.glm53_models.$$

echo "-- 进程"
if [ -f "$PIDFILE" ]; then
  pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  echo "   pidfile: $PIDFILE -> ${pid:-<空>}"
  if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
    ps -o pid,ppid,etime,rss,cmd -p "$pid" | sed 's/^/   /' | cut -c1-150
    # 子进程(EngineCore / 引擎管理器)持有真正的显存
    for child in $(pgrep -P "$pid" 2>/dev/null | head -3); do
      ps -o pid,ppid,etime,rss,cmd -p "$child" 2>/dev/null | sed 's/^/   /' | cut -c1-110
    done
    pgrep -af "VLLM::EngineCore|EngineCore" 2>/dev/null | grep -v pgrep | head -2 | sed 's/^/   /' | cut -c1-110
  else
    echo "   ❌ pidfile 里的进程不在了(被 OOM 杀过?看下面日志)"
    rc=1
  fi
else
  echo "   没有 pidfile($PIDFILE)—— 可能是 systemd/手动前台启动的,用 pgrep 找:"
  pgrep -af "vllm.entrypoints.openai.api_server" | sed 's/^/   /' | cut -c1-150
fi

echo "-- 显存(CUDA_VISIBLE_DEVICES=$GPUS)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader \
    | awk -F', ' -v want="$GPUS" 'BEGIN{split(want,w,",");for(i in w)keep[w[i]]=1}
      keep[$1] {printf "   GPU %s: %s / %s, util %s\n", $1, $2, $3, $4}'
fi

echo "-- 关键日志行($LOG)"
if [ -f "$LOG" ]; then
  echo "   $(stat -c 'size=%s bytes  mtime=%y' "$LOG" 2>/dev/null)"
  grep -E "Available KV cache memory|GPU KV cache size|peak activation|GPU prefill (ACTIVE|DISABLED)|startup finished|READY" "$LOG" 2>/dev/null \
    | awk '!seen[$0]++' | tail -5 | sed 's/^/   /' | cut -c1-190
  echo "   最近的错误行:"
  grep -nE "OutOfMemory|Traceback|ERROR|CUDA out of memory" "$LOG" 2>/dev/null | tail -5 | sed 's/^/   /' | cut -c1-190
  [ -z "$(grep -E 'OutOfMemory|Traceback|CUDA out of memory' "$LOG" 2>/dev/null | tail -1)" ] && echo "   (无)"
  echo "   最后 3 行:"
  tail -3 "$LOG" | sed 's/^/   /' | cut -c1-190
else
  echo "   没有日志文件。若用 systemd 启动,看 journal:"
  echo "     journalctl --user -u glm53 -n 100 -f"
fi

echo "-- 怎么起/停"
cat <<'TIP'
   启动(后台,日志进上面那个文件,~5 min 后才 READY):
     cd /home/user/lvllm/vllm-xiaotu-moe && TAG=glm53_prod nohup bash scripts/serve_glm53_mainline.sh >/tmp/glm53_launch.log 2>&1 &
   停(先杀 APIServer,它会带走 EngineCore;确认显存已释放再重启):
     kill "$(cat logs/glm53_prod.pid)"   &&   sleep 20   &&   nvidia-smi
   注意:直接再启动一次会**端口冲突/显存不够**,一定要先停干净。
TIP
exit $rc
