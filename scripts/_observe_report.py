#!/usr/bin/env python3
"""把【会话健康】与【显存曲线】合成一个状态快照(只读 ✓)。"""
import os, re, sys, datetime, glob
import zstandard as z
R = "/home/user/lvllm/vllm-xiaotu-moe"
L = R + "/dev-docs/report/tuning/logs"
SID = sys.argv[1] if len(sys.argv) > 1 else "session-aa2d9063-bbc3-42cc-a458-a2d3edafcfc2"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "本地新会话(重启后)"
OUT = L + "/OBSERVE_STATUS.txt"
META = re.compile(r"\b(I (?:now )?(?:have|understand|fully understand|see)|"
                  r"I have (?:a |the )?(?:full|complete|very complete) picture|"
                  r"Now I (?:have|understand|need|know)|I have enough|I now have)\b", re.I)
CORR = re.compile(r"vxiaotuo|lvmlm|xiaomju|lvlm-|vllm-xiaom|vllm-xiaotuo|lvlml|lvvm|xao-l-e")
base = os.path.expanduser("~/.dsh/sessions/--home-user-lvllm--")
f = os.path.join(base, SID, "session.v4.jsonl.zstd")
lines = ["=== 观察状态 @ %s ===" % datetime.datetime.now().strftime("%m-%d %H:%M:%S"), ""]
lines.append("【会话】%s (%s)" % (LABEL, SID[:24]))
if os.path.exists(f):
    with open(f, "rb") as fh: d = z.ZstdDecompressor().stream_reader(fh).read()
    rr = [__import__("json").loads(l) for l in d.decode("utf-8", "replace").split("\n") if l.strip()]
    A, calls = [], []
    for r in rr:
        t = r.get("type")
        if t == "assistant/message":
            m = (r.get("data") or {}).get("message") or {}
            A.append("".join((p.get("text","") or "") + " " for p in (m.get("content") or []) if isinstance(p, dict)))
        elif t == "tool/call":
            try: a = __import__("json").loads((r.get("data") or {}).get("arguments") or "{}")
            except Exception: a = {}
            c = str(a.get("command",""))
            if "re.compile" in c or "CORRUPT=" in c: continue
            calls.append(c)
    v = [len(META.findall(x)) for x in A]; n = len(v); q = max(1, n // 3)
    pre = sum(v[:q])/q if v else 0; post = sum(v[2*q:])/max(1,len(v[2*q:])) if v else 0
    bad = [i for i, c in enumerate(calls) if CORR.search(c)]
    lines.append("  轮=%d 调用=%d  meta前=%.2f meta后=%.2f  损坏=%d %s" % (
        n, len(calls), pre, post, len(bad),
        ("⭐ 首个损坏在第 %d 次调用 ✗" % bad[0]) if bad else "✅ 尚无损坏"))
    lines.append("  趋势:meta后 %s" % ("🔴 上升(前段→后段 上升)" if post > pre*1.3 and post > 1.0
                                      else ("🟡 上升但未过 1.0" if post > pre*1.3 else "✓ 平稳")))
    lines.append("  参照线:历史样本首坏在【第 313–505 次调用】;当前 %d 次 ⇒ %s" % (
        len(calls), "⚠️ 已进入高发区" if len(calls) >= 300 else "仍在低发区"))
else:
    lines.append("  (会话文件不存在)")
lines.append("")
lines.append("【显存】(只看 running=0 的空闲点 ✓)")
p = L + "/vram_watch.tsv"
if os.path.exists(p):
    rows = [l.split("\t") for l in open(p).read().strip().split("\n")[1:] if l.strip()]
    def g1(s):
        m = re.search(r"1,\s*(\d+)", s)
        return int(m.group(1)) if m else None
    allpts = [(r[0], g1(r[1]), r[2]) for r in rows if g1(r[1]) is not None]
    idle = [(t, v) for t, v, run in allpts if run.startswith("0")]
    if allpts:
        lines.append("  采样总数=%d 空闲点=%d" % (len(allpts), len(idle)))
        t, v, run = allpts[-1]
        lines.append("  最新: %s GPU1 free=%d MiB (running=%s)" % (t, v, run))
        if idle:
            lines.append("  空闲最低=%d MiB @%s" % (min(v for _, v in idle), min(idle, key=lambda x: x[1])[0]))
            if len(idle) >= 2:
                # ⭐ 修正:台阶检测,不用线性拟合(一次性台阶会误导斜率 ✗)
                steps = []
                prev = None
                for t, v in idle:
                    if prev is None or v != prev:
                        steps.append((t, v)); prev = v
                flat_run = 0
                for i in range(len(idle) - 1, 0, -1):
                    if idle[i][1] == idle[i - 1][1]: flat_run += 1
                    else: break
                lines.append("  空闲首=%d → 末=%d MiB(%d 个台阶)" % (idle[0][1], idle[-1][1], len(steps) - 1))
                lines.append("  台阶明细: " + " | ".join("%s:%d" % (t, v) for t, v in steps[-4:]))
                lines.append("  最近持平长度 = %d 个采样(≈%d 分钟)%s" % (
                    flat_run, flat_run, "  ⭐ 已稳定 ✓" if flat_run >= 5 else "  ⚠️ 近期有变化"))
                lines.append("  ⚠️ 判据:【台阶式下降 + 长持平】⇒ 不是持续吃光 ✓;台阶越来越频繁 ⇒ 才是问题 ✗")
        if any(v < 5500 for _, v in idle):
            lines.append("  🔴 告警:空闲 free 已跌破 5,500 MiB(历史退化区 ✗)")
else:
    lines.append("  (尚无显存采样)")
lines.append("")
lines.append("  台阶频率 = %d 个 / %d 采样 ⇒ %s" % (
    len(steps) - 1, len(idle),
    "⚠️ 台阶密集 ⇒ 关注 ✗" if len(idle) > 0 and (len(steps)-1) / max(1,len(idle)) > 0.02 else "✓ 稀疏"))
lines.append("【历史参照】首坏位置:#1 第471次 / #2 第364次 / #3 第328次 / #4 第56次(该进程已有前序消耗 ✗)")
lines.append("【预期】全新进程 ⇒ 若无累积,应能跑到【300–500 次调用】仍健康 ✓;提前退化 ⇒ 累积假设成立 ✗")
open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
