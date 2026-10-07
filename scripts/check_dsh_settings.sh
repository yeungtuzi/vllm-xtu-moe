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
# ⭐⭐ 2026-10-07 【0.2 配置格式 · 思考强度】====================================
# 症状:`本轮运行失败 provider "epyc-a100-server" model "DeepSeek-V4.1-Flash"
#       does not support reasoning effort "high"`(重启生产后暴露)
# 根因:0.2 的档位字段是【模型级】`reasoningEfforts`;迁移到 patch 层时该字段漏掉
#   而自定义 route 在 pi-ai 目录里不存在 ⇒ 能力只能是 `base?.reasoning ?? false` = 无
#   ⇒ dsh-llm `resolveCallWithInfo()` 对任何【显式】档位在发请求前就抛
#     UNSUPPORTED_REASONING_EFFORT ✗
#   (逐行依据:dsh-llm/lib/index.js resolveCallWithInfo 2174-2192 +
#              dsh-llm-pi-ai/lib/index.js resolveModelReasoning 567-590)
# 离线证据:dev-docs/dsh_wire_probe_v41.mjs(假 endpoint,零生产流量)✓
LEVELS = ["off", "minimal", "low", "medium", "high", "xhigh", "max"]
re_map = m.get("reasoningEfforts")
chk("models[0].reasoningEfforts 是 dict", isinstance(re_map, dict) and len(re_map) > 0,
    "⭐ 0.2 的档位字段(键=档位,值=上线拼写);缺失 ⇒ 任何显式档位直接 UNSUPPORTED_REASONING_EFFORT ✗")
if isinstance(re_map, dict) and re_map:
    bad_keys = [k for k in re_map if k not in LEVELS]
    chk("档位键合法", not bad_keys,
        "键只能是 off/minimal/low/medium/high/xhigh/max(实测:非法键会被 schema 拒)" + ("  命中=%s" % bad_keys if bad_keys else ""))
    bad_vals = [k for k, v in re_map.items() if not (v is None or (isinstance(v, str) and v))]
    chk("档位值非空(off 可为 null)", not bad_vals,
        "空串/错值会被 resolveModelReasoning 拒" + ("  命中=%s" % bad_vals if bad_vals else ""))
    chk("含 high 档", "high" in re_map, "⭐ 报错原文就是它:high 必须显式声明(键集合 = UI 可选档)")
mc = m.get("compat") or {}
chk("models[0].compat 在【模型级】", bool(mc), "0.2 里 compat 属模型级(modelFields.compat)")
chk("compat.supportsReasoningEffort=true", mc.get("supportsReasoningEffort") is True,
    "⭐ 0.2 里【是否真的发 reasoning_effort】的开关:false/缺失 ⇒ 选了档位也不改变请求 ✗")
chk("compat.supportsDeveloperRole=false", mc.get("supportsDeveloperRole") is False,
    "⭐ 必须 false:否则 reasoning 模型把 system 改成 developer ⇒ system prompt 被模板静默丢弃 ✗")
chk("compat.thinkingFormat=openai", mc.get("thinkingFormat") == "openai",
    "⭐ vLLM 只认【顶层 reasoning_effort】;deepseek 格式会塞 thinking 对象且不发 off 的 effort ⇒ Off 静默失效 ✗")
_dflt = prov.get("reasoning")
chk("route 级 reasoning 已设", isinstance(_dflt, str) and _dflt in (re_map or {}),
    "⭐ 'Default' 档用哪一档;缺失 ⇒ Default 退化成 off 的拼写(none)= 悄悄关掉思考 ✗ (当前=%s)" % (_dflt,))
# ⭐ 2026-10-05 修正(实测 4 次后):V4.1 编码器【有工具豁免】——
#   `if any(m.get("tools") for m in full_messages): effective_drop_thinking = False`
#   实测:无 tools 时 prompt_tokens=43(思考被丢,纯聊天场景,刻意省 token ✓);
#         带 tools 时 =394(思考【全部保留】✓)⇒ **agent 场景服务端本来就没问题** ✓
#   ⇒ 故此项**不作为失败项**,只作提示 ✓
_ctk = mc.get("chatTemplateKwargs") or {}
print("  %s compat.chatTemplateKwargs.drop_thinking = %s(可选;带 tools 时服务端会自动保留思考 ✓)"%(
    "•", _ctk.get("drop_thinking")))
chk("多模态字段", bool(m.get("input")), "图片输入(0.2 用 input;inputModalities 会被静默忽略)")
chk("无旧字段名(模型条目内)", ("reasoning" not in m) and ("thinkingLevelMap" not in m),
    "`reasoning: true` / `thinkingLevelMap` 是 pi-ai 内部名,0.2 schema 会静默丢弃 ⇒ 写了等于没写 ✗")
chk("agent-default-model", bool(ad.get("provider")) and bool(ad.get("model")), "默认模型指向 %s/%s"%(ad.get("provider"),ad.get("model")))

print("  --- 官方端点(llm-deepseek)---")
b=str(ld.get("baseURL") or "")
chk("baseURL 走 Anthropic 协议", b.rstrip("/").endswith("/anthropic"),
    "B24: dsh 2.0 的 Messages 路径是 <base>/anthropic/v1/messages;裸域名 ⇒ /v1/messages ⇒ 404 ✗  (当前=%s)"%(b or "<未设置:用内置缺省 PUBLIC_BASE_URL>"))
print("  ⇒ %s"%("全部通过 ✓" if ok else "有缺失 ✗ ⇒ 按 B160/B214/B24 补齐;patch 层是 watch 的,外部编辑约 2s 热生效 ✓"))
sys.exit(0 if ok else 1)
PY
