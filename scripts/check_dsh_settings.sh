#!/usr/bin/env bash
# 检查 DSH 设置里"本地端点 + 官方端点都可用"所必需的字段(B160/B214/B24 几个坑)
# 用法: bash scripts/check_dsh_settings.sh
#
# ⚠️ 2026-10-05 起【权威层变了】:dsh 启动时会把旧的 ~/.dsh/settings.yaml
#    一次性导入 profile patch 层,并把旧文件改名为 settings.yaml.imported
#    (依据:dsh-settings/README.zh.md「早期版本留在 harness home 中的 settings.yaml
#     会被导入一次…文件在第一次写入前改名为 settings.yaml.imported」)
#    ⇒ 真正生效的文件是 ~/.dsh/profiles/web/cordis.patch.yml ✓
#    ⇒ 本脚本优先读 patch 层;只有在 patch 层没有时才回退读旧的 settings.yaml ✓
python3 - <<'PY'
import yaml,os,sys
HOME=os.path.expanduser("~")
PATCH=os.path.join(HOME,".dsh/profiles/web/cordis.patch.yml")
LEGACY=os.path.join(HOME,".dsh/settings.yaml")

def load_entries(path):
    with open(path,encoding="utf-8") as f:
        doc=yaml.safe_load(f)
    if isinstance(doc,list):                      # loader patch 层:顶层数组
        return {e["id"]:e for e in doc if isinstance(e,dict) and e.get("id")}
    if isinstance(doc,dict):                      # 旧 settings.yaml:section -> config
        return {k:{"id":k,"config":v} for k,v in doc.items()}
    return {}

src=None
if os.path.exists(PATCH):
    try:
        entries=load_entries(PATCH); src=PATCH
    except Exception as e:
        print("  ❌ cordis.patch.yml 解析失败:",e); sys.exit(1)
if not entries and os.path.exists(LEGACY):
    try:
        entries=load_entries(LEGACY); src=LEGACY
    except Exception as e:
        print("  ❌ settings.yaml 解析失败:",e); sys.exit(1)
if not entries:
    print("  ❌ 找不到 DSH 设置:既无 %s 也无 %s"%(PATCH,LEGACY)); sys.exit(1)
print("  读取: %s"%src)

def cfg(eid, *path, default=None):
    """按 entry id 取（可能嵌套的）配置值,缺任一层返回 default。"""
    node=(entries.get(eid) or {}).get("config") or {}
    for k in path:
        if not isinstance(node,dict): return default
        node=node.get(k)
        if node is None: return default
    return node

prov=cfg("llm-pi-ai","providers","epyc-a100-server",default={}) or {}
m=(prov.get("models") or [{}])[0]
ld=cfg("llm-deepseek",default={}) or {}
ad=cfg("agent-default-model",default={}) or {}
ok=True
def chk(name,cond,why):
    global ok
    print("  %s %-26s %s"%("✅" if cond else "❌",name,why))
    if not cond: ok=False

print("  --- 本地端点(epyc-a100-server)---")
# ⭐ 2026-10-05 血的教训:我曾把【provider 的协议/路由】改丢 ⇒ DSH 回退到默认 Anthropic 端点
#    ⇒ POST https://api.deepseek.com/v1/messages ⇒ 404 ⇒ 【把我自己的会话打死】✗
chk("provider.api", bool(prov.get("api")), "⭐ 协议(丢了会回退到默认端点 ⇒ 404 打死自己)")
chk("provider.baseURL", bool(prov.get("baseURL")), "⭐ 路由地址(同上,必须与 api 匹配)")
chk("models[0].reasoningEfforts", bool(m.get("reasoningEfforts")), "各档位映射(思考强度选择器就靠它)")
chk("models[0].compat", bool(m.get("compat")), "⚠️ schema 里 compat 属【模型级】(放 provider 级可能不生效)")
chk("compat.thinkingFormat", (m.get("compat") or {}).get("thinkingFormat")=="deepseek", "按 deepseek 格式读写思考(缺失=选不了思考强度 ✗)")
chk("多模态字段", bool(m.get("input") or m.get("inputModalities")), "图片输入(新 schema 用 input)")
chk("无过期字段 reasoning", "reasoning" not in m or m.get("reasoning") is None, "旧 schema 的字段,新版不存在(留着无用)")
chk("agent-default-model", bool(ad.get("provider")) and bool(ad.get("model")), "默认模型指向 %s/%s"%(ad.get("provider"),ad.get("model")))

print("  --- 官方端点(llm-deepseek)---")
b=str(ld.get("baseURL") or "")
chk("baseURL 走 Anthropic 协议", b.rstrip("/").endswith("/anthropic"),
    "B24: dsh 2.0 的 Messages 路径是 <base>/anthropic/v1/messages;裸域名 ⇒ /v1/messages ⇒ 404 ✗  (当前=%s)"%(b or "<未设置:用内置缺省 PUBLIC_BASE_URL>"))
print("  ⇒ %s"%("全部通过 ✓" if ok else "有缺失 ✗ ⇒ 按 B160/B214/B24 补齐;patch 层是 watch 的,外部编辑约 2s 热生效 ✓"))
sys.exit(0 if ok else 1)
PY
