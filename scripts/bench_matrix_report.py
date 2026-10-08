#!/usr/bin/env python
"""把 `bench_matrix.json` 渲染成日期工作报告(口径先写死 ⇒ R15/R25)。"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              cwd=REPO, timeout=60).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="dev-docs/report/tuning/logs/bench_matrix.json")
    ap.add_argument("--out", default="dev-docs/PERF_MATRIX_2026-10-08.md")
    ap.add_argument("--log", default="dev-docs/report/tuning/logs/vllm_prod_8070.log")
    ap.add_argument("--startup-s", type=float, default=0.0)
    args = ap.parse_args()

    d = json.load(open(os.path.join(REPO, args.json)))
    res = d["results"]
    today = datetime.date.today().isoformat()
    gpu = sh("nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader")
    gpu = gpu.replace("\n", " / ")
    md5 = sh("md5sum scripts/bringup_prod_8070.sh | cut -d' ' -f1")
    mem = sh("free -g | awk '/Mem:/{print $2}'")
    caps = int(d.get("max_tokens", 128))

    L = []
    A = L.append
    A("# 性能矩阵报告 —— ds41f 生产配置(c × L,6 个项目)· " + today)
    A("")
    A("> **任务(用户 2026-10-08 下达)**:*启动过后,执行一个 `c=1/2/4 * 128/16384prompt` 总计 **6 个项目**,"
      "每个项目输出 **中位数 TTFT(s)**、**聚合 prefill 速度(tok/s)**、**聚合解码速度(tok/s)** 三项数据,"
      "**事先充分预热**,使用 **random 数据集**进行性能测试,并把测试报告写到今天的工作报告中* ✓")
    A(">")
    A("> 关联:`USER_QA_LEDGER` **#120** · 台账 `docs/EXPERIMENTS.md` **B384** · "
      "原始数据 `" + args.json + "` · 服务日志 `" + args.log + "`")
    A("")
    A("---")
    A("")
    A("## 0. 口径(**先写死,不事后改** ✓ R15/R25)")
    A("")
    A("| 项 | 值 |")
    A("|---|---|")
    A("| 模型 | **ds41f** = DeepSeek-V4.1-Flash(官方量化:FP8 稠密 + **MXFP4** 专家)✓ |")
    A("| 服务 | 生产 **8070** · 受审脚本 `scripts/bringup_prod_8070.sh` · md5 `" + md5 + "` |")
    A("| 配置 | `TP=2`(GPU1+GPU2)· `MAXLEN=524288` · **`MBT=8192`** · `MAXSEQS=4` · "
      "`KV_CACHE_BYTES=2684354560` · `GPU_UTIL=0.90` · `SPEC=1`(DSpark k=5)· `LMCACHE=0` |")
    A("| 硬件 | " + gpu + " |")
    A("| 宿主 | " + mem + " GiB total |")
    A("| 项目数 | **6** = `c ∈ {1,2,4}` × `L ∈ {128, 16384}` |")
    A("| 每项目轮数 | 测量 **" + str(d.get("rounds")) + "** 轮;每轮并发 **c** 个请求 |")
    A("| max_tokens | **" + str(caps) + "**(固定 ⇒ 解码速度可比;`ignore_eos=True`) |")
    A("| 采样 | `temperature=0`;**每请求一份全新随机 token 序列** ⇒ **不触发 prefix cache** ✓ |")
    A("| 数据集 | ⭐ **random**:均匀随机 token id ∈ [1000, 100000)(避开特殊 token)✓ |")
    A("| 预热 | 全局:两种长度各若干轮;每项目:另加若干轮,**全部不计入统计** ✓ |")
    A("| ① 中位数 TTFT | 该项目**全部请求**的 TTFT 的中位数(含 c>1 的各并发请求)✓ |")
    A("| ② 聚合 prefill | `(本轮 prompt token 总数) / max_i(TTFT_i)` —— 并发 c 个请求的预填在墙钟上"
      "≈最后一个首 token 出现时完成,故分母取 **max** ✓ |")
    A("| ③ 聚合 decode | `(本轮 output token 总数 − c) / (max_i 末token时刻 − min_i 首token时刻)` —— "
      "减去各请求的**首 token**(它属 prefill 阶段)✓ |")
    A("| 启动耗时 | " + ("%.0f s" % args.startup_s if args.startup_s else "(未记录)") + " |")
    A("")
    A("---")
    A("")
    A("## 1. 结果(6 个项目)")
    A("")
    A("| c | L | **中位 TTFT (s)** | TTFT 离散 | **聚合 prefill (tok/s)** | 离散 | "
      "**聚合 decode (tok/s)** | 离散 | 请求数 |")
    A("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in res:
        if r.get("err"):
            A("| %s | %s | ⛔ %s | | | | | | |" % (r["c"], r["L"], r["err"]))
            continue
        A("| **%d** | **%d** | **%.2f** | %.1f%% | **%.1f** | %.1f%% | **%.1f** | %.1f%% | %d |"
          % (r["c"], r["L"], r["ttft_med"], r["ttft_spread_pct"], r["prefill_tok_s"],
             r["prefill_spread_pct"], r["decode_tok_s"], r["decode_spread_pct"], r["n_req"]))
    A("")
    A("### 1.1 读法")
    A("")
    A("* **中位 TTFT** 主要反映 **prefill 时长** —— prompt 越长越贵 ✓")
    A("* **聚合 prefill tok/s**:`c` 越大,多个请求的预填**并行分摊**(受 H2D 带宽与 GPU 计算共同约束)✓")
    A("* **聚合 decode tok/s**:`c` 越大,解码阶段的**批处理收益**越明显 ✓")
    A("")

    if res and all(not r.get("err") for r in res):
        A("---")
        A("")
        A("## 2. 随 c 的变化(直接从数据读)")
        A("")
        A("| 长度 L | 指标 | 随并发 c 的变化 |")
        A("|---:|---|---|")
        for Lv in d["lens"]:
            row = {r["c"]: r for r in res if r["L"] == Lv and not r.get("err")}
            if len(row) < 2:
                continue
            cs = sorted(row)
            A("| %d | 中位 TTFT | " % Lv +
              " → ".join("c=%d: **%.2fs**" % (c, row[c]["ttft_med"]) for c in cs) + " |")
            A("| %d | 聚合 prefill | " % Lv +
              " → ".join("c=%d: **%.1f**" % (c, row[c]["prefill_tok_s"]) for c in cs) + " tok/s |")
            A("| %d | 聚合 decode | " % Lv +
              " → ".join("c=%d: **%.1f**" % (c, row[c]["decode_tok_s"]) for c in cs) + " tok/s |")
        A("")

    A("---")
    A("")
    A("## 3. 限制与未做(**不要外推**)")
    A("")
    A("* ⚠️ 只覆盖 **`c ≤ 4` × 两种长度**;更长/更高并发**未测** ✓")
    A("* ⚠️ 请求是 **random token id**,与真实文本的**注意力稀疏性 / 前缀复用**不同 ⇒ "
      "**TTFT 不代表真实业务分布** ✓")
    A("* ⚠️ `MAXSEQS=4` 是**服务上限**;本轮 `c≤4` ⇒ 未触及**排队区** ✓")
    A("* ⚠️ 期间**同一台机器**上还跑着本 agent 自身(其推理后端就是本服务)⇒ 可能存在**轻微自污染** ✓")
    A("")
    A("---")
    A("")
    A("_生成时间:" + datetime.datetime.now().isoformat(timespec="seconds") + "_")

    out = os.path.join(REPO, args.out)
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("✓ 报告已写 → " + out + "(%d 行)" % len(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
