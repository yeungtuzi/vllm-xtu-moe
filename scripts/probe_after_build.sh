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
  if ! bash "$R/scripts/proc.sh" status vllm_build2 2>/dev/null | grep -q RUNNING; then
    say "① 编译进程已结束(等待 $((i*10)) 秒)"; break
  fi
  sleep 10
done
if grep -qE "error:|Error|Cannot find|失败" "$LOG/vllm_build2.log" 2>/dev/null && ! ls -t "$R"/../process_data/ref/repos/vllm-mainline/vllm/_C_stable_libtorch*.so >/dev/null 2>&1; then
  say "⚠️ 编译日志里有错误关键词"; fi
SO=/home/user/lvllm/process_data/ref/repos/vllm-mainline/vllm/_C_stable_libtorch.abi3.so
say "① .so 时间戳:$(stat -c '%y  %s bytes' "$SO" 2>/dev/null)"
tail -5 "$LOG/vllm_build2.log" 2>/dev/null | sed 's/^/     /' | tee -a "$OUT"

# ② 停生产(用户已授权 ✓ 无任务在跑 ✓)
say "② 停生产"
bash "$R/scripts/proc.sh" stop v41_8070 >>"$OUT" 2>&1
bash "$R/scripts/proc.sh" stop dsv41_prod >>"$OUT" 2>&1
sleep 8
say "   8070 端口:$(ss -ltn 2>/dev/null | grep -c ':8070 ') (0=已停)"
say "   GPU:$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
say "   内存可用:$(awk '/MemAvailable/{printf "%.0f GiB",$2/1048576}' /proc/meminfo)"

# ③ 起测试实例(8071)
say "③ 起测试实例(8071)"
cd "$R"
PORT=8071 WITH_MONITORING=0 bash -c "cd $R && PORT=8071 nohup bash scripts/bringup_prod_8070.sh >>$LOG/probe_instance.log 2>&1 &"
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ':8071 ' && break; sleep 10; done
sleep 15
if ! ss -ltn 2>/dev/null | grep -q ':8071 '; then
  say "❌ 实例未起 ⇒ 尾部日志:"; tail -25 "$LOG/probe_instance.log" 2>/dev/null | sed 's/^/     /' | tee -a "$OUT"
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
  tail -30 "$LOG/probe_instance.log" 2>/dev/null | sed 's/^/     /' | tee -a "$OUT"
fi
say "   (生产仍停止;由用户决定何时恢复 ✓)"
say "=========== probe 结束 ==========="
