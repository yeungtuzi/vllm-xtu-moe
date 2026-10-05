#!/usr/bin/env bash
# 检查 DSH 设置里"本地端点能用"所必需的字段是否齐全(B160/B214 的两个坑)
# 用法: bash scripts/check_dsh_settings.sh
python3 - <<'PY'
import yaml,os,sys
p=os.path.expanduser("~/.dsh/settings.yaml")
try: d=yaml.safe_load(open(p,encoding="utf-8"))
except Exception as e: print("  ❌ settings.yaml 解析失败:",e); sys.exit(1)
prov=d.get("llm-pi-ai",{}).get("providers",{}).get("epyc-a100-server",{})
m=(prov.get("models") or [{}])[0]
ok=True
def chk(name,cond,why):
    global ok
    print("  %s %-22s %s"%("✅" if cond else "❌",name,why))
    if not cond: ok=False
chk("models[0].reasoning", m.get("reasoning") is True, "思考强度选择器的开关(丢了就选不了思考强度)")
chk("compat.thinkingFormat", bool(prov.get("compat",{}).get("thinkingFormat")), "告诉客户端按 deepseek 格式读思考")
chk("reasoningEfforts", bool(m.get("reasoningEfforts")), "各档位映射(注意:官方只认 low/high/max,传 medium 会 400)")
chk("多模态字段", bool(m.get("input") or m.get("inputModalities")), "图片输入(注意字段名可能是 input 或 inputModalities)")
print("  ⇒ %s"%("全部通过 ✓" if ok else "有缺失 ✗ ⇒ 按 B160/B214 补齐,DSH 设置是 watch:true 可热生效 ✓"))
sys.exit(0 if ok else 1)
PY
