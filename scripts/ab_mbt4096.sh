#!/usr/bin/env bash
# ⭐ A/B 测量:MBT 8192 → 4096,验证"峰值 ∝ qlen ⇒ 余量抬升"并重跑 78 万复现
# 设计为【可逆】:结束时把配置改回 8192(除非 AB_KEEP=1)
set -u
R=/home/user/lvllm/vllm-xiaotu-moe
L="$R/dev-docs/report/tuning/logs"
OUT="$L/ab_mbt4096_result.txt"
PY=/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python
B="$R/scripts/bringup_prod_8070.sh"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$OUT"; }

say "=========== A/B:MBT 8192 → 4096 ==========="
say "假设:峰值 ∝ qlen(=MBT)⇒ MBT 减半 ⇒ alloc 峰值显著下降 ⇒ free 抬升 ⇒ 512 MiB 分配必成 ✓"
say "预注册判据:① 启动成功 ② MBT 生效值=4096 ③ free 余量 ≥ 1024 MiB(显著优于 3,134 的对照)"
say "           ④ 78 万 token 请求完成且 EngineCore 不死 ✓ ⑤ 危险关键词 0"

# ① 改 MBT(精确锚定赋值行 ✓)
python3 - <<'PY'
import re
p="/home/user/lvllm/vllm-xiaotu-moe/scripts/bringup_prod_8070.sh"
s=open(p,encoding="utf-8").read()
pat=re.compile(r"^(\s*)MBT=8192\s*$", re.M)
assert len(pat.findall(s))==1, "MBT 赋值行不唯一"
s2=pat.sub(lambda m: m.group(1)+"MBT=4096   # ⭐ A/B 测量(可逆):峰值 ∝ qlen ⇒ 减半抬升余量 ✓", s, count=1)
open(p,"w",encoding="utf-8").write(s2); print("MBT 赋值行已改为 4096")
PY
grep -n "^\s*MBT=" "$B" | sed 's/^/  /'
bash -n "$B" || { say "❌ bash -n 失败"; exit 1; }

# ② 重启
say "② 重启(停 → 起)"
bash "$R/scripts/proc.sh" stop v41_8070 >>"$OUT" 2>&1
bash "$R/scripts/proc.sh" stop dsv41_prod >>"$OUT" 2>&1
sleep 8
cd "$R" && bash scripts/proc.sh spawn dsv41_prod env bash -c "cd $R && WITH_MONITORING=1 bash scripts/bringup_prod_8070.sh" >>"$OUT" 2>&1
for i in $(seq 1 90); do ss -ltn 2>/dev/null | grep -q ':8070 ' && break; sleep 10; done
sleep 15
PORT=$(ss -ltn 2>/dev/null | grep -c ':8070 ')
say "③ 启动结果:8070=$PORT(期望 1)"
[ "$PORT" -eq 1 ] || { say "❌ 启动失败 ⇒ 尾部日志:"; tail -20 "$L/v41_8070.log" 2>/dev/null | sed 's/^/     /' | tee -a "$OUT"; exit 1; }
P=$(tr -dc '0-9' < "$L/v41_8070.pid" 2>/dev/null)
MBT=$(tr '\0' ' ' < /proc/$P/cmdline | grep -oE '\-\-max-num-batched-tokens [0-9]+' | awk '{print $2}')
say "④ MBT 生效值=$MBT(期望 4096)"

# ③ 采样 + 跑 78 万复现
say "⑤ 起采样器并跑 78 万 token 复现"
bash "$R/scripts/proc.sh" stop vram_sampler >>"$OUT" 2>&1
mv "$L/vram_trace.txt" "$L/vram_trace_mbt8192.txt" 2>/dev/null
bash "$R/scripts/proc.sh" spawn vram_sampler env bash -c "/tmp/vram_sampler.sh $L/vram_trace.txt 3600" >>"$OUT" 2>&1
sleep 10
$PY - >>"$OUT" 2>&1 <<'PYEOF'
import requests,time
def build(turns, extra=0):
    h=[{"role":"system","content":"You are a coding agent. Use absolute paths."}]
    for i in range(turns):
        h.append({"role":"user","content":"继续第 %d 步:查看计划文件与日志,汇报差异。"%i})
        h.append({"role":"assistant","reasoning":("处理第 %d 步,项目在 /home/user/lvllm/vllm-xiaotu-moe。"%i)*8,"content":"执行第 %d 步。"%i})
        h.append({"role":"tool","tool_call_id":"c%d"%i,"content":"dev-docs/PLAN_VNNI_SM80.md 1234 bytes\n"*25})
    h.append({"role":"user","content":"总结当前状态。" + ("(第%d轮)"%extra if extra else "")})
    return h
# ⭐ 连发两个:第一个建立长 KV;第二个(稍加一个 token 使之不同)命中前缀缓存 ⇒ 池更紧 ⇒ 逼近原崩溃条件
for turns,extra in ((1612,0),(1612,1)):
    body={"model":"DeepSeek-V4.1-Flash","max_tokens":32,"temperature":0.0,"messages":build(turns,extra)}
    t0=time.time()
    try:
        r=requests.post("http://127.0.0.1:8070/v1/chat/completions",json=body,timeout=7200)
        u=(r.json().get("usage") or {})
        print("[%s] 78万 prompt=%s HTTP=%s 用时%.0fs"%(time.strftime("%H:%M:%S"),u.get("prompt_tokens"),r.status_code,time.time()-t0))
    except Exception as e:
        print("[%s] ⚠️ %s: %s(用时%.0fs)"%(time.strftime("%H:%M:%S"),type(e).__name__,str(e)[:100],time.time()-t0))
PYEOF

# ④ 判据
say "⑥ 判据核对"
$PY - >>"$OUT" 2>&1 <<'PYEOF'
import re
p="/home/user/lvllm/vllm-xiaotu-moe/dev-docs/report/tuning/logs/vram_trace.txt"
rows=[]
for l in open(p,errors="replace"):
    for i,u,f in re.findall(r"(\d+),\s*(\d+),\s*(\d+)", l): rows.append((int(i),int(u),int(f)))
g=[r for r in rows if r[0]==1]
if g:
    mn=min(g,key=lambda x:x[2]); mx=max(g,key=lambda x:x[1])
    print("  GPU1 采样 %d 点:已用最大 %d MiB,空闲最小 %d MiB"%(len(g),mx[1],mn[2]))
    print("  ⇒ 判据③ free ≥1024 MiB:%s"%("✅ 通过" if mn[2]>=1024 else "❌ %d MiB"%mn[2]))
PYEOF
KL=$(ls -t "$R"/dev-docs/report/tuning/logs/v41_8070.*.log 2>/dev/null | head -1)
D=$(grep -acE "aten::new_empty|EngineDeadError" "$KL" 2>/dev/null)
say "  判据⑤ 危险关键词=$D(期望 0)"
say "  判据④ 8070=$(ss -ltn 2>/dev/null | grep -c ':8070 ')(1=EngineCore 未死 ✓)"

# ⑤ 恢复(可逆 ✓)
if [ "${AB_KEEP:-0}" != "1" ]; then
  say "⑦ 恢复 MBT=8192(可逆设计 ✓)"
  python3 - <<'PY'
import re
p="/home/user/lvllm/vllm-xiaotu-moe/scripts/bringup_prod_8070.sh"
s=open(p,encoding="utf-8").read()
s=re.sub(r"^(\s*)MBT=4096.*$", r"\1MBT=8192", s, count=1, flags=re.M)
open(p,"w",encoding="utf-8").write(s); print("MBT 已恢复 8192")
PY
  say "  ⇒ 配置已恢复;如需回到 4096 只需再跑一次(或设 AB_KEEP=1)"
fi
say "=========== A/B 结束 ==========="
