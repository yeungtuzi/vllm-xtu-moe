#!/usr/bin/env python3
"""采样生产进程的空闲显存 + 并发数(只读 ✓)。用法:_vram_sample.py <out.tsv>"""
import subprocess, sys, datetime, urllib.request
out = sys.argv[1]
def sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception: return ""
free = sh("nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits").replace("\n", ";")
def metric(name):
    try:
        with urllib.request.urlopen("http://127.0.0.1:8070/metrics", timeout=6) as r:
            for l in r.read().decode("utf-8", "replace").split("\n"):
                if l.startswith(name): return l.rsplit(" ", 1)[-1]
    except Exception: pass
    return ""
run = metric("vllm:num_requests_running")
wait = metric("vllm:num_requests_waiting")
kv = metric("vllm:kv_cache_usage_perc")
pid = sh("pgrep -f 'vllm.entrypoints.openai.api_server' | head -1")
rss = ""
if pid:
    # ⭐ 避免 awk 的 % 与 Python 的 % 冲突:用字符串拼接 ✓
    rss = sh("awk '/VmRSS/{printf \"%.1f\", $2/1048576}' /proc/" + pid + "/status")
open(out, "a").write("%s\t%s\t%s\t%s\t%s\t%s\n" % (
    datetime.datetime.now().strftime("%H:%M:%S"), free, run or "-", wait or "-", kv or "-", rss or "-"))
