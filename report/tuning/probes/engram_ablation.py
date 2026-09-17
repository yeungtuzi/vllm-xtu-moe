#!/usr/bin/env python3
# Engram **运行时消融**实测(§565):同一进程内只切"是否注入",量它到底买到了什么。
#
# 原理:服务启动时给 `XIAOTU_ENGRAM_ABLATE_FILE=<path>` ⇒ 插件把 `Engram.forward`
# 换成"该文件存在就恒等返回(不注入)"。于是**同一次服务、同一份权重、同一套 kernel**,
# 只差一个开关 ⇒ 差异可归因给 Engram 本身。
#
# 指标:
#   1) **prompt 困惑度(mean NLL)**:给一段文本,让模型预测它,取每 token 的 logprob 平均。
#      消融后 NLL 升高 ⇒ 这些参数确实在"记住"文本里的东西。**配对比较**(同一段文本两条件下都跑)。
#   2) **贪心生成**:几个长尾知识问题,两条件下逐字对照,直观看"知识有没有丢"。
#
# 为什么加 nonce:服务开着 prefix caching,同一 prompt 第二次会被缓存命中、prompt_logprobs
# 可能返回空。所以每次请求前面塞一个唯一的短 nonce,并**跳过前 SKIP 个位置**再统计。
#
# 用法:
#   python3 report/tuning/probes/engram_ablation.py --port 8314 --toggle /tmp/xtu_engram_ablate
#   可选:--rounds 2 --dataset report/tuning/datasets/nat1024.jsonl --nat-count 6
# License: Apache-2.0
import argparse
import json
import math
import os
import statistics
import time
import urllib.request

# 长尾事实段落:Engram 的收益应该主要出现在"稀有实体/具体数字"上
FACT_PASSAGES = [
    "The Antikythera mechanism is an ancient Greek hand-powered orrery, described as the "
    "oldest known example of an analogue computer, used to predict astronomical positions "
    "and eclipses decades in advance. It was recovered in 1901 from the Antikythera wreck "
    "off the Greek island of Antikythera, between Kythera and Crete.",

    "The Tsar Bomba, designated RDS-220, was a Soviet thermonuclear weapon tested on 30 "
    "October 1961 over the Mityushikha Bay near the Matochkin Strait in the Novaya Zemlya "
    "archipelago. It had a yield of approximately 50 megatons of TNT, the largest ever "
    "detonated.",

    "Kolmogorov complexity, named after Andrey Kolmogorov, is the length of the shortest "
    "computer program that produces a given object as output. Andrey Nikolaevich Kolmogorov "
    "published the foundational work in 1965, building partly on earlier ideas of Ray "
    "Solomonoff and Gregory Chaitin.",

    "The Treaty of Tordesillas was signed on 7 June 1494 in Tordesillas, Spain, and divided "
    "the newly discovered lands outside Europe between the Portuguese Empire and the Spanish "
    "Empire along a meridian 370 leagues west of the Cape Verde islands.",

    "Bismuth telluride, Bi2Te3, is a layered semiconductor with a rhombohedral crystal "
    "structure and a band gap of about 0.15 electron volts. It is widely used in "
    "thermoelectric cooling devices because of its high Seebeck coefficient and low thermal "
    "conductivity.",

    "The Huygens probe landed on Titan on 14 January 2005, after separating from the Cassini "
    "spacecraft on 25 December 2004. It transmitted data for about 90 minutes from the "
    "surface, including images of rounded pebbles that suggested flowing liquid methane.",

    "In the Standard Model, the Cabibbo-Kobayashi-Maskawa matrix parametrises the flavour "
    "mixing of quarks. The Wolfenstein parametrisation expresses it in terms of lambda, A, "
    "rho-bar and eta-bar, with lambda approximately 0.22.",

    "The Svalbard Global Seed Vault opened on 26 February 2008, tunnelled into permafrost "
    "on the island of Spitsbergen near Longyearbyen in Norway, and stores duplicate seed "
    "samples from gene banks around the world at minus 18 degrees Celsius.",

    "Hafnium carbide, HfC, has one of the highest melting points of any binary compound, "
    "around 3900 degrees Celsius, and a rock-salt crystal structure. It is used in "
    "ultra-high-temperature ceramics for hypersonic vehicle leading edges.",

    "The Voynich manuscript is a vellum codex carbon-dated to the early fifteenth century, "
    "written in an unidentified script with no known counterpart. It is held at the Beinecke "
    "Rare Book and Manuscript Library at Yale University, catalogued as MS 408.",
]

GREEDY_QUESTIONS = [
    "In what year was the Treaty of Tordesillas signed, and between which two empires?",
    "What is the approximate melting point of hafnium carbide, and what crystal structure does it have?",
    "Who published the foundational work on Kolmogorov complexity, and in what year?",
    "Which spacecraft carried the Huygens probe, and when did Huygens land on Titan?",
]


def _post(port: int, path: str, payload: dict, timeout: float = 300.0):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _set_ablation(toggle: str, on: bool) -> None:
    if on:
        open(toggle, "w").close()
    else:
        try:
            os.remove(toggle)
        except FileNotFoundError:
            pass
    time.sleep(0.4)


def _prompt_nll(port: int, model: str, text: str, skip: int) -> tuple[float, int]:
    """返回 (平均 NLL, 统计到的 token 数)。skip 之前的 token 不计(含 nonce)。"""
    nonce = f"Note {os.urandom(4).hex()}. "
    resp = _post(port, "/v1/completions", {
        "model": model,
        "prompt": nonce + text,
        "max_tokens": 1,
        "temperature": 0.0,
        "prompt_logprobs": 0,
    })
    pls = resp["choices"][0].get("prompt_logprobs")
    if not pls:
        raise RuntimeError(f"prompt_logprobs 为空(prefix cache 命中?): {list(resp['choices'][0])}")
    nlls = []
    for i, entry in enumerate(pls):
        if i < skip or not entry:
            continue
        # entry: {token_id_str: {"logprob": float, ...}}
        lp = None
        for _tid, info in entry.items():
            if isinstance(info, dict) and "logprob" in info:
                lp = info["logprob"]
                break
        if lp is not None and math.isfinite(lp):
            nlls.append(-lp)
    return (sum(nlls) / len(nlls) if nlls else float("nan")), len(nlls)


def _greedy(port: int, model: str, prompt: str, max_tokens: int = 64) -> str:
    resp = _post(port, "/v1/completions", {
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "ignore_eos": True,
    })
    return resp["choices"][0]["text"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="dsv41-xtu")
    ap.add_argument("--toggle", required=True)
    ap.add_argument("--rounds", type=int, default=2, help="A/B 交替轮数(取平均,抵消漂移)")
    ap.add_argument("--skip", type=int, default=16, help="跳过开头 nonce 的 token 数")
    ap.add_argument("--dataset", default="", help="额外的 jsonl(每行 {'prompt': ...})")
    ap.add_argument("--nat-count", type=int, default=4)
    args = ap.parse_args()

    texts = list(FACT_PASSAGES)
    if args.dataset and os.path.exists(args.dataset):
        rows = []
        with open(args.dataset) as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line)["prompt"])
                except Exception:  # noqa: BLE001
                    continue
        # 每段截到 ~1200 字符,避免把 32K 全喂进去
        texts += [r[:1200] for r in rows[: args.nat_count]]
        print(f"[ablate] 数据集 {args.dataset}: 加入 {min(len(rows), args.nat_count)} 段自然文本")

    print(f"[ablate] toggle={args.toggle}  文本 {len(texts)} 段  rounds={args.rounds}")
    res = {"on": [[] for _ in texts], "off": [[] for _ in texts]}
    for r in range(args.rounds):
        for cond, ablate in (("on", False), ("off", True)):   # on=正常(注入), off=消融
            _set_ablation(args.toggle, ablate)
            for i, t in enumerate(texts):
                try:
                    nll, n = _prompt_nll(args.port, args.model, t, args.skip)
                except Exception as exc:  # noqa: BLE001
                    print(f"  [round{r} {cond} #{i}] 失败: {type(exc).__name__}: {exc}")
                    continue
                res[cond][i].append(nll)
            print(f"  [round{r}] {cond} 完成")

    _set_ablation(args.toggle, False)
    print(f"\n{'#':>3} {'Engram开(正常)':>16} {'Engram关(消融)':>16} {'ΔNLL':>9} {'倍数':>7}  文本前 42 字")
    d_all, r_all = [], []
    for i, t in enumerate(texts):
        a = statistics.mean(res["on"][i]) if res["on"][i] else float("nan")
        b = statistics.mean(res["off"][i]) if res["off"][i] else float("nan")
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        d_all.append(b - a)
        r_all.append(math.exp(b - a))
        print(f"{i:>3} {a:16.4f} {b:16.4f} {b - a:+9.4f} {math.exp(b - a):6.3f}x  {t[:42]!r}")
    if d_all:
        print(f"\n平均: NLL 上升 {statistics.mean(d_all):+.4f} nats "
              f"⇒ 困惑度 ×{statistics.mean(r_all):.3f}"
              f"(中位 ×{statistics.median(r_all):.3f})(消融/正常)")
        print("解读:>1 表示拿掉 Engram 后模型更难预测这些文本 ⇒ 这些参数确实在提供知识。")

    print("\n=== 贪心生成对照(同 prompt,两条件)===")
    for q in GREEDY_QUESTIONS:
        _set_ablation(args.toggle, False)
        on = _greedy(args.port, args.model, q)
        _set_ablation(args.toggle, True)
        off = _greedy(args.port, args.model, q)
        same = "相同" if on == off else "**不同**"
        print(f"\nQ: {q}\n  Engram开: {on.strip()[:200]!r}\n  Engram关: {off.strip()[:200]!r}\n  ⇒ {same}")
    _set_ablation(args.toggle, False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
