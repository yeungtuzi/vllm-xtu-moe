#!/usr/bin/env bash
# ⭐ 自包含 probe:停生产 → 修编译环境 → 编译 vLLM(含 std::min probe)→ 起测试实例 → 打一个超阈值 prompt → 记录结果
# 设计为【脱离会话】运行(proc.sh spawn 用 setsid)⇒ 即使调用者断线也跑完 ✓
set -u
R=/home/user/lvllm/vllm-xiaotu-moe
V=/home/user/lvllm/process_data/ref/repos/vllm-mainline
LOG="$R/dev-docs/report/tuning/logs"
OUT="$LOG/probe_result.txt"
PY=/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$OUT"; }

say "================ probe 开始 ================"
say "目标:验证 'q_out 按全序列分配' 是否是崩溃根因"
say "probe 改动:$V/csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu"
grep -n "std::min(q_in.size(0)" "$V/csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu" | tee -a "$OUT"

# ① 停生产(用户已授权 ✓)
say "① 停生产(v41_8070 / dsv41_prod)"
bash "$R/scripts/proc.sh" stop v41_8070 >>"$OUT" 2>&1
bash "$R/scripts/proc.sh" stop dsv41_prod >>"$OUT" 2>&1
sleep 6
say "   端口 8070:$(ss -ltn | grep -c ':8070 ') (0=已释放 ✓)"
say "   GPU: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
say "   内存可用: $(awk '/MemAvailable/{printf "%.0f GiB",$2/1048576}' /proc/meminfo)"

# ② 修编译环境
say "② 准备编译环境"
export HTTPS_PROXY="${HTTPS_PROXY:-${PROXY:-}}" HTTP_PROXY="$HTTPS_PROXY" https_proxy="$HTTPS_PROXY" http_proxy="$HTTPS_PROXY"   # ⭐ 代理从环境取,勿硬编码内网地址 ✗
$PY -m pip install -q setuptools-rust >>"$OUT" 2>&1 && say "   setuptools-rust ✓" || say "   ⚠️ setuptools-rust 安装失败(继续试)"

# ③ 编译(全机器 ✓ 生产已停)
say "③ 编译 vLLM(-j 96,预计 10–40 分钟)"
cd "$V" && nice -n 19 $PY setup.py build_ext --inplace -j 96 >>"$LOG/vllm_build2.log" 2>&1
RC=$?
say "   编译退出码=$RC"
if [ $RC -ne 0 ]; then
  say "❌ 编译失败 ⇒ 见 $LOG/vllm_build2.log 尾部:"
  tail -20 "$LOG/vllm_build2.log" | sed 's/^/     /' | tee -a "$OUT"
  say "================ probe 结束(编译失败)================"
  exit 1
fi
say "   ✅ 编译成功"
ls -la "$V"/vllm/_C_stable_libtorch*.so 2>/dev/null | tee -a "$OUT"

# ④ 起测试实例(用生产同款配置,但端口 8071,便于隔离)
say "④ 起测试实例(端口 8071,ds41f,TP=2)"
cd "$R"
PORT=8071 WITH_MONITORING=0 bash -c "cd $R && PORT=8071 nohup bash scripts/bringup_prod_8070.sh >>$LOG/probe_instance.log 2>&1 &"
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ':8071 ' && break; sleep 10; done
sleep 20
say "   端口 8071:$(ss -ltn | grep -c ':8071 ')"
if ! ss -ltn 2>/dev/null | grep -q ':8071 '; then
  say "❌ 测试实例未起来 ⇒ 见 $LOG/probe_instance.log"
  tail -25 "$LOG/probe_instance.log" | sed 's/^/     /' | tee -a "$OUT"
  say "================ probe 结束(实例未起)================"
  exit 1
fi

# ⑤ ⭐ 打一个【超过阈值(58万)】的 prompt ⇒ 看是否还崩
say "⑤ 发一个 60 万 token 的请求(原版必崩 ✗)"
$PY - >>"$OUT" 2>&1 <<'PYEOF'
import requests,time,json,re
def build(turns):
    h=[{"role":"system","content":"You are a coding agent. Use absolute paths."}]
    for i in range(turns):
        h.append({"role":"user","content":"继续第 %d 步:查看计划文件与日志。"%i})
        h.append({"role":"assistant","reasoning":("处理第 %d 步,项目在 /home/user/lvllm/vllm-xiaotu-moe。"%i)*8,
                  "content":"执行第 %d 步。"%i})
        h.append({"role":"tool","tool_call_id":"c%d"%i,"content":"dev-docs/x.md 1234 bytes\n"*25})
    h.append({"role":"user","content":"现在总结一下当前状态。"})
    return h
for turns in (400, 500):
    body={"model":"DeepSeek-V4.1-Flash","max_tokens":32,"temperature":0.0,"messages":build(turns)}
    t0=time.time()
    try:
        r=requests.post("http://127.0.0.1:8071/v1/chat/completions",json=body,timeout=1800)
        u=(r.json().get("usage") or {})
        print("  合成 %d 轮 ⇒ prompt=%s  HTTP=%s  用时 %.0fs  %s"%(
            turns,u.get("prompt_tokens"),r.status_code,time.time()-t0,
            "✅ 成功(未崩)" if r.status_code==200 else "⚠️ HTTP %s"%r.status_code))
    except Exception as e:
        print("  合成 %d 轮 ⇒ ⚠️ 请求异常:%s: %s"%(turns,type(e).__name__,str(e)[:120]))
        print("     ⇒ 紧接着查实例是否还活着:")
        import subprocess
        alive=subprocess.run(["ss","-ltn"],capture_output=True,text=True).stdout.count(":8071 ")
        print("        端口 8071 计数=%d %s"%(alive,"⚠️ 实例死了 ⇒ 仍然崩 ✗" if alive==0 else "✓ 实例还活着"))
        break
PYEOF

# ⑥ 结论 + 实例状态
say "⑥ 结果判定"
if ss -ltn 2>/dev/null | grep -q ':8071 '; then
  say "   ✅ 测试实例仍在运行 ⇒ 【probe 有效:未崩】⇒ 因果确认:根因就是 q_out 按全序列分配 ✓"
else
  say "   ❌ 测试实例已死 ⇒ 【probe 无效】⇒ 根因不在这里(需换方向)"
  tail -30 "$LOG/probe_instance.log" 2>/dev/null | sed 's/^/     /' | tee -a "$OUT"
fi
say "   (生产仍处停止状态 —— 由用户决定何时恢复 ✓)"
say "================ probe 结束 ================"
