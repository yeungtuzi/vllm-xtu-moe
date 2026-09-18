#!/usr/bin/env python3
"""固定 prompt 的 greedy 探针:用于 TP=1 / TP=2(EP) 输出一致性对比。

用法:  scripts/probe_greedy.py <out.json> [port] [model]
输出:  {"<id>": {"text": ..., "usage": ...}, ...}
说明:  temperature=0 + seed 固定;两边用同一份 prompt 文件
       (dev-docs/report/tuning/correctness_prompts.json),文本一致即认为 MoE 计算路径等价。
"""
import json, sys, urllib.request

out_path = sys.argv[1]
port = sys.argv[2] if len(sys.argv) > 2 else "8090"
model = sys.argv[3] if len(sys.argv) > 3 else "DeepSeek-V4-Flash-xiaotu"
prompts = json.load(open("/home/user/lvllm/vllm-xiaotu-moe/dev-docs/report/tuning/correctness_prompts.json"))

res = {}
for p in prompts:
    body = json.dumps({
        "model": model, "messages": p["messages"],
        "max_tokens": p["max_tokens"], "temperature": 0.0, "seed": 1234,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    _m = d["choices"][0]["message"]
    # 开了 thinking 时,前若干 token 会进 `reasoning` 字段(max_tokens 小的时候 content 会是 None)
    _txt = _m.get("content") or _m.get("reasoning") or ""
    res[p["id"]] = {"text": _txt,
                    "finish_reason": d["choices"][0].get("finish_reason"),
                    "usage": d.get("usage")}
    print(f"[{p['id']}] {res[p['id']]['text']!r}", flush=True)
json.dump(res, open(out_path, "w"), ensure_ascii=False, indent=1)
print("saved", out_path)
