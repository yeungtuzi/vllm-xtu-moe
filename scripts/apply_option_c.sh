#!/usr/bin/env bash
# ⚠️⚠️ 【不要直接运行】选项 C 会把 MAXLEN 从 1M 改回 768K ——
#     而 **1M 是用户 2026-10-03 明确指示要的**(见 bringup_prod_8070.sh 第 6 行注释)✗
#     ⇒ 因此本脚本【必须经用户重新同意】才能执行 ✓
#     ⇒ 它的价值只在于:当用户决定"宁可少 0.65 GiB 余量也要保 1M"时,说明另一条路(MBT↓)是必需的 ✓
# 背景:BUG_REGISTRY A19(池的下界 = maxlen 所需 KV)+ A20(峰值 ∝ qlen)
# ⭐ 选项 C:max_model_len 1M → 768k,并相应把 KV 池 2.5 → 1.85 GiB(留出 ≥512 MiB 激活余量)
# 依据:BUG_REGISTRY A19(池有下界 = maxlen 所需)+ A20(峰值 ∝ qlen)
# ⚠️ 与选项 A(MBT=4096)可叠加;本脚本只改 maxlen + 池 ✓
set -u
R=/home/user/lvllm/vllm-xiaotu-moe
B="$R/scripts/bringup_prod_8070.sh"
echo "=== 选项 C:改配置(精确锚定赋值行 ✓)==="
python3 - <<'PY'
import re
p="/home/user/lvllm/vllm-xiaotu-moe/scripts/bringup_prod_8070.sh"
s=open(p,encoding="utf-8").read()
# max_model_len 的赋值行(脚本里通常叫 MAXLEN 或 MODEL_LEN)
cands=[m for m in re.finditer(r"^(\s*)(MAXLEN|MODEL_LEN|MAX_MODEL_LEN)=(\d+)(\s*)$", s, re.M)]
for c in cands: print("  找到:",c.group(0).strip())
assert cands, "未找到 max_model_len 赋值行 ⇒ 需人工确认变量名"
c=cands[0]
s=s[:c.start()]+f"{c.group(1)}{c.group(2)}=768000   # ⭐ 选项 C:原 1048576;与 DSH contextWindow 一致 ⇒ 池下界降至 ~1.61 GiB ✓"+s[c.end():]
# KV 池
s=re.sub(r"^(\s*)KV_CACHE_BYTES=\d+.*$", r"\g<1>KV_CACHE_BYTES=1986422374   # ⭐ 1.85 GiB(原 2.5)", s, count=1, flags=re.M)
open(p,"w",encoding="utf-8").write(s); print("  ✅ 已改")
PY
grep -nE "^\s*(MAXLEN|MODEL_LEN|MAX_MODEL_LEN)=|^\s*KV_CACHE_BYTES=" "$B" | sed 's/^/  /'
bash -n "$B" && echo "  ✅ bash -n 通过"
echo "  ⇒ 现在【不重启】(等 A/B 结论 ✓);要生效请手动跑:"
echo "     bash scripts/proc.sh stop v41_8070; bash scripts/proc.sh stop dsv41_prod; \\"
echo "     bash scripts/proc.sh spawn dsv41_prod env bash -c 'cd $R && WITH_MONITORING=1 bash scripts/bringup_prod_8070.sh'"
