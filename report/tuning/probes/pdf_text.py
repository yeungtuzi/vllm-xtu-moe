#!/usr/bin/env python3
"""极简 PDF 文本抽取(无外部依赖):FlateDecode 解压 + 抽 `(...)` 字面量。

为什么需要它:本机 `pdftotext`/`pypdf`/`fitz` 都没有,而 DeepSeek 的技术报告
(`DeepSeek_V41_Tech_Report.pdf`)就在 checkpoint 目录里 —— 核实"层结构/路由方式/草稿模块"
这类问题时,报告原文是最权威的来源之一(§505)。

用法:
    python3 report/tuning/probes/pdf_text.py <file.pdf> [关键词 ...]
不带关键词则把全文写到 /tmp/<basename>.txt 并打印字符数。
License: Apache-2.0
"""
import re, sys, zlib

def extract(path: str) -> str:
    data = open(path, 'rb').read()
    out = []
    for m in re.finditer(rb'stream\r?\n', data):
        s = m.end(); e = data.find(b'endstream', s)
        if e < 0:
            continue
        try:
            dec = zlib.decompress(data[s:e])
        except Exception:
            continue
        for tm in re.finditer(rb'\((?:\\.|[^\\()])*\)', dec):
            t = tm.group(0)[1:-1]
            t = re.sub(rb'\\([()\\])', rb'\1', t)
            out.append(t.decode('latin-1', 'replace'))
    return ' '.join(out)

def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__); return 2
    txt = extract(sys.argv[1])
    base = sys.argv[1].rsplit('/', 1)[-1].rsplit('.', 1)[0]
    dst = f'/tmp/{base}.txt'
    open(dst, 'w').write(txt)
    print(f"[pdf_text] {len(txt)} chars -> {dst}")
    for kw in sys.argv[2:]:
        hits = list(re.finditer(re.escape(kw), txt))
        print(f"\n=== {kw!r}: {len(hits)} hits ===")
        for m in hits[:3]:
            s = max(0, m.start() - 260); e = min(len(txt), m.end() + 260)
            print("  ...", re.sub(r'\s+', ' ', txt[s:e]), "\n")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
