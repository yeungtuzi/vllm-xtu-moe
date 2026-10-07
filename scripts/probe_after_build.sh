#!/usr/bin/env bash
# 等当前编译结束 → 停生产 → 起 8071 测试实例 → 打一个超阈值 prompt → 记录结论
set -u
R=/home/user/lvllm/vllm-xiaotu-moe
LOG="$R/dev-docs/report/tuning/logs"
OUT="$LOG/probe_result.txt"
PY=/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$OUT"; }
say "=========== probe(等编译)开始 ==========="

# ① 等编译结束(最多 90 分钟)
for i in $(seq 1 540); do
  # ⭐ 用退出码判,不用 `status | grep "NOT RUNNING"` 子串匹配(rc≠0 ⇒ 已结束/无 pidfile)✓
  if ! bash "$R/scripts/proc.sh" status vllm_build2 >/dev/null 2>&1; then
    say "① 编译进程已结束(等待 $((i*10)) 秒)"; break
  fi
  sleep 10
done
if grep -qE "error:|Error|Cannot find|失败" "$LOG/vllm_build2.log" 2>/dev/null && ! ls -t "$R"/../process_data/ref/repos/vllm-mainline/vllm/_C_stable_libtorch*.so >/dev/null 2>&1; then
  say "⚠️ 编译日志里有错误关键词"; fi
SO=/home/user/lvllm/process_data/ref/repos/vllm-mainline/vllm/_C_stable_libtorch.abi3.so
say "① .so 时间戳:$(stat -c '%y  %s bytes' "$SO" 2>/dev/null)"
tail -5 "$LOG/vllm_build2.log" 2>/dev/null | sed 's/^/     /' | tee -a "$OUT"

# ② 停生产(⛔ 必须【当次】用户许可 —— 门会对没有许可的生产 PID 返回 rc=4)✓
say "② 停生产"
_stop_ok=1
for _n in v41_8070 dsv41_prod; do
  bash "$R/scripts/proc.sh" stop "$_n" >>"$OUT" 2>&1; _rc=$?
  case "$_rc" in
    0) say "   已停/本就没跑:$_n" ;;
    4) say "⛔ 停 $_n 被【门】拒绝(rc=4)⇒ 说明未取得用户【当次】许可 ⇒ 中止 probe ✗"; _stop_ok=0 ;;
    5) say "⛔ 停 $_n 后仍有存活(rc=5)⇒ 中止 probe ✗"; _stop_ok=0 ;;
    *) say "   ⚠️ 停 $_n 返回 rc=$_rc(按 pidfile 不存在/内容异常处理)" ;;
  esac
done
if [ "$_stop_ok" != "1" ]; then
  say "=========== probe 中止(需用户当次许可后重跑)==========="; exit 4
fi
sleep 8
# ⭐ 2026-10-07 独立审计:停生产后必须【机械 gate】确认 8070 真的没人听,否则绝不许起第二实例 ✗
. "$R/scripts/lib_proc_identity.sh"
pi_port_listeners 8070
if [ "$PI_SS_OK" != "1" ]; then
  say "⛔ 端口证据读不到(ss 失败)⇒ 无法确认生产已停 ⇒ 中止(不许起第二实例)"; exit 4
fi
if [ -n "$PI_PORT_PIDS" ]; then
  say "⛔ 8070 仍有监听(pid=$PI_PORT_PIDS)⇒ 生产未停干净 ⇒ 中止(不许起第二实例)"; exit 4
fi
say "   8070 已确认无人监听 ✓"
say "   GPU:$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
say "   内存可用:$(awk '/MemAvailable/{printf "%.0f GiB",$2/1048576}' /proc/meminfo)"

# ③ 起测试实例(8071)
# ⭐ 审计修:原来用 `nohup … &`(G8 禁止:proc.sh 之外不得裸起后台进程)⇒ 改走 proc.sh spawn ✓
say "③ 起测试实例(8071)"
cd "$R" || exit 1
if ! bash "$R/scripts/proc.sh" spawn probe_8071 env PORT=8071 WITH_MONITORING=0 \
      bash scripts/bringup_prod_8070.sh >>"$OUT" 2>&1; then
  say "❌ spawn probe_8071 被拒/失败 ⇒ 中止"; exit 1
fi
for _ in $(seq 1 90); do
  [ -s "$LOG/probe_8071.pid" ] && break
  sleep 10
done
sleep 15
if ! ss -ltn 2>/dev/null | grep -q ':8071 '; then
  say "❌ 实例未起 ⇒ 尾部日志:"; tail -25 "$LOG/probe_8071.log" 2>/dev/null | sed 's/^/     /' | tee -a "$OUT"
  say "=========== probe 结束(实例未起)==========="; exit 1
fi
say "   ✅ 8071 已就绪"

# ④ 超阈值请求(60万 token)
say "④ 发 60 万 token 请求(原版必崩)"
$PY - >>"$OUT" 2>&1 <<'PYEOF'
import requests,time,subprocess
def build(turns):
    h=[{"role":"system","content":"You are a coding agent. Use absolute paths."}]
    for i in range(turns):
        h.append({"role":"user","content":"继续第 %d 步:查看计划文件与日志。"%i})
        h.append({"role":"assistant","reasoning":("处理第 %d 步,项目在 /home/user/lvllm/vllm-xiaotu-moe。"%i)*8,"content":"执行。"})
        h.append({"role":"tool","tool_call_id":"c%d"%i,"content":"dev-docs/x.md 1234 bytes\n"*25})
    h.append({"role":"user","content":"总结当前状态。"})
    return h
for turns in (400, 520):
    body={"model":"DeepSeek-V4.1-Flash","max_tokens":32,"temperature":0.0,"messages":build(turns)}
    t0=time.time()
    try:
        r=requests.post("http://127.0.0.1:8071/v1/chat/completions",json=body,timeout=2400)
        u=(r.json().get("usage") or {})
        print("  合成 %d 轮 ⇒ prompt=%s  HTTP=%s  用时%.0fs"%(turns,u.get("prompt_tokens"),r.status_code,time.time()-t0))
    except Exception as e:
        print("  合成 %d 轮 ⇒ ⚠️ %s: %s"%(turns,type(e).__name__,str(e)[:100]))
    alive=subprocess.run(["ss","-ltn"],capture_output=True,text=True).stdout.count(":8071 ")
    print("     实例 8071 存活=%d %s"%(alive,"✗ 已死(仍崩)" if alive==0 else "✓ 活着"))
    if alive==0: break
PYEOF

# ⑤ 判定
say "⑤ 判定"
if ss -ltn 2>/dev/null | grep -q ':8071 '; then
  say "   ✅ 【probe 有效】实例存活 ⇒ 因果确认:根因= q_out 按全序列分配 ✓"
  say "   ⇒ 下一步:设计【语义正确】的分块实现(不是这个 min 补丁)"
else
  say "   ❌ 【probe 无效】实例已死 ⇒ 根因不在分配处,需换方向"
  tail -30 "$LOG/probe_8071.log" 2>/dev/null | sed 's/^/     /' | tee -a "$OUT"
fi
say "   (生产仍停止;由用户决定何时恢复 ✓)"
say "=========== probe 结束 ==========="
