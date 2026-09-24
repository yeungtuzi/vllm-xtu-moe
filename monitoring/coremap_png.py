#!/usr/bin/env python3
"""Render the CPU core map (NUMA -> CCD -> physical core) to a PNG.

Draws one large block subdivided by NUMA node, each node subdivided by CCD,
each CCD holding one tiny tile per physical core showing its utilization.
Layout is recomputed from the live metric every INTERVAL seconds.
"""
import io, json, os, sys, time, urllib.request
from PIL import Image, ImageDraw, ImageFont

PROM = "http://127.0.0.1:9090/api/v1/query"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/user/lvllm/monitoring/web/coremap.png"
INTERVAL = float(os.environ.get("XTU_MAP_INTERVAL", "5"))
W, H = 1400, 700
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SMALL = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def q(expr):
    try:
        with urllib.request.urlopen(f"{PROM}?query={urllib.parse.quote(expr)}", timeout=6) as r:
            return json.load(r)["data"]["result"]
    except Exception:
        return []


import urllib.parse


def hue(n):
    return (int(n) * 47) % 360


def hsl(h, s, l):
    import colorsys
    r, g, b = colorsys.hls_to_rgb(h / 360.0, l / 100.0, s / 100.0)
    return (int(r * 255), int(g * 255), int(b * 255))


def core_color(v):
    if v < 50:
        return hsl(120 - v * 1.6, 70, 72 - v * 0.18)
    if v < 80:
        return hsl(60 - (v - 50) * 1.8, 85, 60)
    return hsl(max(0, 30 - (v - 80) * 1.5), 80, max(35, 55 - (v - 80) * 0.15))


def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def render():
    rows = q("xtu_core_utilization_percent")
    load = q("node_load1")
    img = Image.new("RGB", (W, H), (14, 17, 22))
    d = ImageDraw.Draw(img)
    f_core = font(FONT_PATH, 14)
    f_head = font(FONT_PATH, 13)
    f_small = font(FONT_SMALL, 9)
    f_foot = font(FONT_SMALL, 12)
    if not rows:
        d.text((10, 10), "no data", fill=(200, 80, 80), font=f_foot)
        img.save(OUT, "PNG")
        return
    nodes = {}
    for r in rows:
        m = r["metric"]
        nodes.setdefault(m.get("node", "?"), {}).setdefault(m.get("ccd", "?"), []).append(
            (int(m["core"]), float(r["value"][1]))
        )
    top, left, right, bottom = 6, 6, W - 6, H - 24
    d.rectangle([left, top, right, bottom], outline=(70, 80, 100), width=2)
    ns = sorted(nodes, key=lambda x: int(x))
    cols = 4
    rowsn = (len(ns) + cols - 1) // cols
    gap = 4
    cw = (right - left - gap * (cols + 1)) // cols
    ch = (bottom - top - gap * (rowsn + 1)) // rowsn
    for i, n in enumerate(ns):
        cx = left + gap + (i % cols) * (cw + gap)
        cy = top + gap + (i // cols) * (ch + gap)
        d.rectangle([cx, cy, cx + cw, cy + ch], fill=hsl(hue(n), 55, 95), outline=(40, 45, 55), width=2)
        vals = [v for lst in nodes[n].values() for _, v in lst]
        avg = sum(vals) / len(vals)
        d.text((cx + 6, cy + 3), f"NUMA {n}", fill=(20, 24, 30), font=f_head)
        t = f"{avg:.0f}%"
        d.text((cx + cw - 6 - d.textlength(t, font=f_head), cy + 3), t, fill=(20, 24, 30), font=f_head)
        ccds = sorted(nodes[n], key=lambda x: int(x))
        inner_top = cy + 17
        ih = (cy + ch - 4 - inner_top) // len(ccds)
        for j, c in enumerate(ccds):
            y0 = inner_top + j * ih
            y1 = y0 + ih - 2
            d.rectangle([cx + 4, y0, cx + cw - 4, y1], fill=hsl(hue(n) + ((int(c) % 3) * 12 - 12), 50, 90),
                        outline=(90, 95, 110), width=1)
            d.text((cx + 7, y0 + 1), f"CCD {c}", fill=(70, 75, 85), font=f_small)
            cores = sorted(nodes[n][c])
            # 每 CCD:4 列 × 2 行
            ccols, crows = 4, 2
            tw = (cw - 7 - (ccols - 1) * 1) // ccols
            th = (y1 - (y0 + 10) - (crows - 1) * 1) // crows
            for k, (core, v) in enumerate(cores):
                tx = cx + 4 + (k % ccols) * (tw + 1)
                ty = y0 + 10 + (k // ccols) * (th + 1)
                d.rectangle([tx, ty, tx + tw, ty + th], fill=core_color(v), outline=(120, 125, 135))
                s = str(int(round(v)))
                tl = d.textlength(s, font=f_core)
                d.text((tx + (tw - tl) / 2, ty + (th - 15) / 2), s, fill=(16, 19, 26), font=f_core)
    busy = sum(1 for r in rows if float(r["value"][1]) > 50)
    avg = sum(float(r["value"][1]) for r in rows) / len(rows)
    l1 = float(load[0]["value"][1]) if load else float("nan")
    d.text((10, H - 20), f"{time.strftime('%H:%M:%S')}   load1 {l1:.2f}   per-core {l1/len(rows):.3f}   "
                         f"busy>50% {busy}/{len(rows)}   avg {avg:.1f}%", fill=(150, 158, 170), font=f_foot)
    tmp = OUT + ".tmp"
    img.save(tmp, "PNG")
    os.replace(tmp, OUT)


while True:
    try:
        render()
    except Exception as e:
        print("render failed:", e, flush=True)
    time.sleep(INTERVAL)
