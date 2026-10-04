#!/usr/bin/env python3
"""密集采样分析器(干净重写版;判据=进度心跳,不用 metrics ✗)

用法: python3 scripts/analyze_dense_run.py [采样目录]
"""
import csv, glob, os, re, sys, time

L = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                 "dev-docs", "report", "tuning", "logs")
def latest_run():
    ds = [d for d in glob.glob(os.path.join(L, "dense_20*")) if os.path.isdir(d)]
    return max(ds, key=os.path.getmtime) if ds else None
D = sys.argv[1] if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]) else latest_run()
if not D:
    print("找不到采样目录"); sys.exit(1)
out, rows = [], []
csvf = os.path.join(D, "samples.csv")
if os.path.exists(csvf):
    with open(csvf, newline="", errors="replace") as f:
        rows = [r for r in csv.DictReader(f) if r.get("t")]
pf = os.path.join(D, "pf_progress.txt")
hb = []          # (秒, call, device, layer, qlen, ids0)
if os.path.exists(pf):
    for ln in open(pf, errors="replace"):
        ln = re.sub(r"\x1b\[[0-9;]*m", "", ln)
        m = re.search(r"\[(\d\d):(\d\d):(\d\d)\].*device=([a-z0-9:]+) layer=\S*layers\.(\d+)\S* qlen=(\d+) call=(\d+)", ln)
        if m:
            hb.append((int(m.group(1))*3600+int(m.group(2))*60+int(m.group(3)),
                       int(m.group(7)), m.group(4), int(m.group(5)), int(m.group(6))))
def fnum(x, d=0.0):
    try: return float(x)
    except Exception: return d
out.append("# 密集采样分析报告\n")
out.append("采样目录: `%s`" % os.path.basename(D))
out.append("生成时间: %s\n" % time.strftime("%F %T"))
# 1 规模
out.append("## 1. 规模与判定分布")
out.append("- 样本数: **%d**" % len(rows))
if rows:
    out.append("- 窗口: **%s → %s**" % (rows[0]["t"], rows[-1]["t"]))
    v = {}
    for r in rows: v[r.get("verdict", "?")] = v.get(r.get("verdict", "?"), 0) + 1
    for k, n in sorted(v.items(), key=lambda x: -x[1]): out.append("- verdict `%s`: %d" % (k, n))
out.append("")
# 2 ★ 真卡死判据 = 心跳停推
out.append("## 2. ⭐ 真卡死判据:**进度心跳停推** >180s")
out.append("> ⚠️ **不用** `/metrics` —— 它在长 step 期间是**陈旧值** ✗(2026-10-04 我据此误报 3 次,见 QA #30/#35)")
if not hb:
    out.append("- ⚠️ 本窗口无心跳数据")
else:
    out.append("- 心跳条数: **%d**,`call` 范围 **%d → %d**" % (len(hb), hb[0][1], hb[-1][1]))
    gaps = [(hb[i-1][0], hb[i][0], hb[i][0]-hb[i-1][0]) for i in range(1, len(hb)) if hb[i][0]-hb[i-1][0] > 180]
    if gaps:
        out.append("- ⚠️ **发现 %d 处停推 >180s** ✗(疑似真卡死):" % len(gaps))
        for g in gaps[:5]:
            fm = lambda s: "%02d:%02d:%02d" % (s//3600, s%3600//60, s%60)
            out.append("  - %s → %s(停 **%.0f 秒**)" % (fm(g[0]), fm(g[1]), g[2]))
    else:
        out.append("- ✅ **心跳全程连续(无 >180s 停推)⇒ 未出现真卡死**")
out.append("")
# 3 有请求但计数不推进(参考;注意 metrics 会陈旧)
out.append("## 3. 参考:有请求且 GPU 忙,但计数长时间不变(可能只是长 step ✗)")
cnt = 0; prev = None; start = None
for r in rows:
    if fnum(r.get("running")) > 0:
        tk = (r.get("prompt_tok"), r.get("gen_tok"))
        if tk == prev:
            cnt += 1
            if cnt == 180: out.append("- ⚠️ 自 %s 起计数冻结 (样本 %d 个)" % (start, cnt))
        else:
            prev, cnt, start = tk, 0, r.get("t")
    else: prev, cnt = None, 0
if not any("冻结" in x for x in out[-5:]): out.append("- 无 >180 样本的计数冻结 ✓")
out.append("")
# 4 状态量趋势
out.append("## 4. 状态量趋势(哪个先异常)")
av = [fnum(r.get("avail_gib"), -1) for r in rows]; av = [x for x in av if x > 0]
kv = [fnum(r.get("kv_usage_pct"), -1) for r in rows]; kv = [x for x in kv if x >= 0]
ls = [fnum(r.get("lmc_stored"), -1) for r in rows]; ls = [x for x in ls if x >= 0]
gu = max([fnum(r.get("gpu_max")) for r in rows] or [0]); gp = max([fnum(r.get("power_max")) for r in rows] or [0])
out.append("- **MemAvailable 最低**: %s GiB" % ("%.0f" % min(av) if av else "n/a"))
out.append("- **KV 池使用率峰值**: %s%%" % ("%.1f" % max(kv) if kv else "n/a(本轮未采到)"))
out.append("- **LMCache 已存块数**(末值): %s" % ("%.0f" % ls[-1] if ls else "n/a"))
out.append("- **GPU 峰值**: %.0f%% / **功耗峰值**: %.0fW" % (gu, gp))
out.append("")
# 5 心跳细节
out.append("## 5. 心跳细节(判定"原地重来"还是"推进")")
if hb:
    lay = {}; ql = {}; dev = {}
    for _, _, d, l, q in hb:
        lay[l] = lay.get(l, 0) + 1; ql[q] = ql.get(q, 0) + 1; dev[d] = dev.get(d, 0) + 1
    out.append("- 层号分布(前 8): " + ", ".join("L%d×%d" % (k, n) for k, n in sorted(lay.items(), key=lambda x: -x[1])[:8]))
    out.append("- qlen 分布(前 5): " + ", ".join("%d×%d" % (k, n) for k, n in sorted(ql.items(), key=lambda x: -x[1])[:5]))
    out.append("- 两路: " + ", ".join("%s×%d" % (k, n) for k, n in sorted(dev.items(), key=lambda x: -x[1])))
    out.append("- 最后 5 条:")
    for h in hb[-5:]:
        out.append("  - %02d:%02d:%02d device=%s layer=%d qlen=%d call=%d" % (h[0]//3600, h[0]%3600//60, h[0]%60, h[2], h[3], h[4], h[1]))
else:
    out.append("- 无心跳")
out.append("")
# 6 看门狗
out.append("## 6. 同期看门狗记录")
wl = os.path.join(L, "watch_hang.log")
if os.path.exists(wl):
    tail = [l.rstrip() for l in open(wl, errors="replace")][-400:]
    hits = [l for l in tail if ("★★★" in l or "心跳停推" in l)]
    out.append("- 真卡死告警: **%d** 条" % len(hits))
    for l in hits[-5:]: out.append("  - %s" % l[:150])
    if not hits: out.append("  - 无 ✓")
else: out.append("- 无看门狗日志")
out.append("")
# 7 结论
out.append("## 7. 结论")
if hb:
    big = [g for g in [(hb[i][0]-hb[i-1][0]) for i in range(1, len(hb))] if g > 180]
    out.append("⇒ %s" % ("✗ **出现心跳停推 ⇒ 真卡死**(见第 2 节,用 call/ids0/层号定位 ✓)" if big
                          else "✓ **本窗口未出现真卡死**(心跳全程连续)⇒ 故障未在窗口内复发 ✓"))
else:
    out.append("⇒ 无心跳数据,无法判定")
txt = "\n".join(out)
open(os.path.join(D, "ANALYSIS.md"), "w", encoding="utf-8").write(txt + "\n")
print(txt)
