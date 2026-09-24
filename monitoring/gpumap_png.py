#!/usr/bin/env python3
"""Render the GPU map: one block per card, VRAM split per owning process,
plus a row of indicator tiles. Same visual language as the CPU core map."""
import colorsys, json, os, sys, time, urllib.parse, urllib.request
from PIL import Image, ImageDraw, ImageFont

PROM = "http://127.0.0.1:9090/api/v1/query"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/user/lvllm/monitoring/web/gpumap.png"
INTERVAL = float(os.environ.get("XTU_MAP_INTERVAL", "5"))
W, H = 1400, 430
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def q(expr):
    try:
        with urllib.request.urlopen(f"{PROM}?query={urllib.parse.quote(expr)}", timeout=6) as r:
            return json.load(r)["data"]["result"]
    except Exception:
        return []


def hsl(h, s, l):
    r, g, b = colorsys.hls_to_rgb((h % 360) / 360.0, l / 100.0, s / 100.0)
    return (int(r * 255), int(g * 255), int(b * 255))


def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def vcolor(v, green_below=50, yellow_below=80):
    if v < green_below:
        return hsl(120 - v * 1.6, 70, 74 - v * 0.2)
    if v < yellow_below:
        return hsl(60 - (v - green_below) * 1.8, 85, 62)
    return hsl(max(0, 30 - (v - yellow_below) * 1.5), 80, max(38, 56 - (v - yellow_below) * 0.15))


def gb(x):
    return x / 1024 ** 3


def render():
    used = {m["metric"]["gpu"]: float(m["value"][1]) for m in q("xtu_gpu_memory_used_bytes")}
    total = {m["metric"]["gpu"]: float(m["value"][1]) for m in q("xtu_gpu_memory_total_bytes")}
    name = {m["metric"]["gpu"]: m["metric"].get("name", "") for m in q("xtu_gpu_memory_used_bytes")}
    procs = {}
    for m in q("xtu_gpu_proc_memory_bytes"):
        procs.setdefault(m["metric"]["gpu"], []).append(
            (m["metric"].get("name", "?"), m["metric"].get("pid", "?"), float(m["value"][1])))
    ind = {}
    for metric, key in (("xtu_gpu_utilization_percent", "sm"),
                        ("xtu_gpu_memory_utilization_percent", "memctrl"),
                        ("xtu_gpu_temperature_celsius", "temp"),
                        ("xtu_gpu_power_watts", "power"),
                        ("xtu_gpu_power_limit_watts", "plimit"),
                        ("xtu_gpu_clock_sm_mhz", "clk"),
                        ("xtu_gpu_fan_percent", "fan")):
        for m in q(metric):
            ind.setdefault(m["metric"]["gpu"], {})[key] = float(m["value"][1])
    img = Image.new("RGB", (W, H), (14, 17, 22))
    d = ImageDraw.Draw(img)
    f_head = font(FB, 15)
    f_lab = font(FB, 12)
    f_num = font(FB, 17)
    f_tiny = font(FR, 10)
    f_foot = font(FR, 12)
    gpus = sorted(used, key=lambda x: int(x)) or ["0", "1", "2"]
    gap, top, bottom = 8, 6, H - 26
    bh = (bottom - top - gap * (len(gpus) - 1)) // len(gpus)
    for i, g in enumerate(gpus):
        y0 = top + i * (bh + gap)
        y1 = y0 + bh
        tint = hsl(200 + i * 55, 55, 95)
        d.rectangle([6, y0, W - 6, y1], fill=tint, outline=(45, 50, 62), width=2)
        t = total.get(g, 0)
        u = used.get(g, 0)
        pct = (100.0 * u / t) if t else 0.0
        d.text((14, y0 + 4), f"GPU {g}", fill=(18, 22, 28), font=f_head)
        d.text((78, y0 + 6), name.get(g, "")[:34], fill=(70, 78, 90), font=f_lab)
        s = f"VRAM {pct:.1f}%   {gb(u):.1f}/{gb(t):.1f} GiB   SM {ind.get(g, {}).get('sm', 0):.0f}%"
        d.text((W - 20 - d.textlength(s, font=f_head), y0 + 5), s, fill=(18, 22, 28), font=f_head)
        # ── 显存按占用者切块(宽度 ∝ 占用)────────────────────────────────
        row_y, row_h = y0 + 26, 34
        x = 12
        avail = W - 24
        items = sorted(procs.get(g, []), key=lambda z: -z[2])
        free = max(0.0, t - u)
        parts = [(f"{n.replace('VLLM::','vLLM ')[:16]} #{p}", v) for n, p, v in items]
        if free > 0:
            parts.append(("free", free))
        tot = sum(v for _, v in parts) or 1
        for k, (lab, v) in enumerate(parts):
            w = max(70, int(avail * v / tot))
            if x + w > W - 12:
                w = W - 12 - x
            if w <= 0:
                break
            is_free = lab == "free"
            d.rectangle([x, row_y, x + w, row_y + row_h],
                        fill=(235, 235, 238) if is_free else hsl(210 + k * 40, 60, 88),
                        outline=(150, 155, 165), width=1)
            if w >= 95:
                d.text((x + 6, row_y + 3), lab, fill=(60, 66, 76), font=f_tiny)
            num = f"{gb(v):.1f}G" if not is_free else f"{gb(v):.1f}G free"
            tw_ = d.textlength(num, font=f_lab)
            if w >= 60:
                d.text((x + 6, row_y + 17), num, fill=(25, 30, 38), font=f_lab)
            x += w + 2
        # ── 指标块 ─────────────────────────────────────────────────────
        tiles = [("SM", ind.get(g, {}).get("sm"), "%", "pct"),
                 ("MEMCTRL", ind.get(g, {}).get("memctrl"), "%", "pct"),
                 ("TEMP", ind.get(g, {}).get("temp"), "C", "temp"),
                 ("POWER", ind.get(g, {}).get("power"), "W", "plain"),
                 ("LIMIT", ind.get(g, {}).get("plimit"), "W", "plain"),
                 ("CLK", ind.get(g, {}).get("clk"), "MHz", "plain"),
                 ("FAN", ind.get(g, {}).get("fan"), "%", "pct")]
        tw = (avail - (len(tiles) - 1) * 3) // len(tiles)
        ty0 = row_y + row_h + 5
        th = min(46, y1 - ty0 - 6)
        for k, (lab, val, unit, kind) in enumerate(tiles):
            tx = 12 + k * (tw + 3)
            if val is None:
                fill, txt = (232, 232, 235), "n/a"
            elif kind == "pct":
                fill, txt = vcolor(val), f"{val:.0f}"
            elif kind == "temp":
                fill = vcolor(max(0, (val - 20) * 100 / 70)) if val else (232, 232, 235)
                txt = f"{val:.0f}"
            else:
                fill, txt = (226, 234, 244), f"{val:.0f}"
            d.rectangle([tx, ty0, tx + tw, ty0 + th], fill=fill, outline=(150, 155, 165))
            d.text((tx + 4, ty0 + 2), lab, fill=(70, 76, 86), font=f_tiny)
            tl = d.textlength(txt, font=f_num)
            d.text((tx + (tw - tl) / 2, ty0 + (th - 18) / 2), txt, fill=(18, 22, 28), font=f_num)
            if unit and val is not None:
                ul = d.textlength(unit, font=f_tiny)
                d.text((tx + tw - ul - 3, ty0 + th - 13), unit, fill=(90, 96, 106), font=f_tiny)
    d.text((10, H - 20), f"{time.strftime('%H:%M:%S')}   each VRAM tile = one owning process "
                         f"(width proportional to memory); indicator tiles coloured by value",
           fill=(150, 158, 170), font=f_foot)
    tmp = OUT + ".tmp"
    img.save(tmp, "PNG")
    os.replace(tmp, OUT)


while True:
    try:
        render()
    except Exception as e:
        print("render failed:", e, flush=True)
    time.sleep(INTERVAL)
