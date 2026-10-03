#!/usr/bin/env python3
"""GSM8K 准确率评测(走 OpenAI 兼容接口;temperature=0;答案取 #### 之后的数 ✓)
用法:PORT=8090 N=200 bash -c 'python scripts/eval_gsm8k.py'
输出:JSON {n, correct, acc, details[]}(口径:8-shot CoT 提示 ✓,与常见 GSM8K 口径一致)"""
import json, os, re, sys, time, urllib.request

PORT = os.environ.get("PORT", "8090")
N    = int(os.environ.get("N", "200"))
MT   = int(os.environ.get("MAXTOK", "512"))
SRC  = os.environ.get("SRC", "dev-docs/eval/gsm8k_test.jsonl")
OUT  = os.environ.get("OUT", "/tmp/gsm8k_result.json")
FEWSHOT = os.environ.get("FEWSHOT", "1") == "1"

def load(n):
    rows=[]
    for line in open(SRC, encoding="utf-8"):
        d=json.loads(line); rows.append(d)
        if len(rows)>=n: break
    return rows

# 4-shot 示例(标准 GSM8K 风格;少样本数越多越稳,这里取 4 个以省时间 ✓)
FEW = [
 ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
  "There are 15 trees originally. After planting, there are 21 trees. So the workers planted 21 - 15 = 6 trees. #### 6"),
 ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
  "There are 3 cars. 2 more arrive. 3 + 2 = 5. #### 5"),
 ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
  "Originally 32 + 42 = 74. After eating 35, 74 - 35 = 39. #### 39"),
 ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
  "Jason had 20 and now has 12, so he gave 20 - 12 = 8. #### 8"),
]

def prompt(q):
    if not FEWSHOT: return q
    ex="".join("Example:\nQ: %s\nA: %s\n\n" % (a,b) for a,b in FEW)
    return ex + "Now solve this problem:\nQ: %s\nA:" % q

SYS = ("You are a careful math tutor. Solve the problem step by step. "
       "End your answer with a line of exactly the form '#### <number>' where <number> is the final answer.")
def ask(p, retry=True):
    """用 chat 端点 ✓(deepseek_v41 只认 chat 词汇表;completions 会被当作续写 ✗)"""
    body=json.dumps({"model":"DeepSeek-V4.1-Flash",
                     "messages":[{"role":"system","content":SYS},{"role":"user","content":p}],
                     "max_tokens":MT,"temperature":0,"top_p":1}).encode()
    req=urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions"%PORT, data=body,
                               headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        msg = json.loads(r.read())["choices"][0]["message"]
    # ⚠️ V4.1-Flash 是推理模型:content 可能为 None(内容在 reasoning_content ✓)
    out = msg.get("content")
    if not out:
        out = msg.get("reasoning_content") or ""
    return out or ""

def gold(ans):
    m=re.search(r"####\s*([\-\d,\.]+)", ans)
    if not m: return None
    return m.group(1).replace(",","").rstrip(".").strip()

def pred(txt):
    if not isinstance(txt, str): return None          # ⚠️ 守卫:None/非字符串 ✓
    # 1) 优先取 #### 之后 ✓;2) 否则取最后一个数字 ✓
    m=re.findall(r"####\s*([\-\d,\.]+)", txt)
    if m: return m[-1].replace(",","").rstrip(".").strip()
    nums=re.findall(r"\-?\d[\d,]*\.?\d*", txt.replace("$",""))
    return nums[-1].replace(",","").rstrip(".").strip() if nums else None

def main():
    rows=load(N); ok=0; det=[]
    t0=time.time()
    for i,d in enumerate(rows,1):
        try: out=ask(prompt(d["question"]))
        except Exception as e: out=""; print("  [%d] 请求失败: %s" % (i,str(e)[:80]), flush=True)
        g,p=gold(d["answer"]),pred(out)
        good = (g is not None and p is not None and abs(float(g)-float(p))<1e-6) if (g and p) else False
        ok += good
        det.append({"i":i,"good":good,"gold":g,"pred":p,"raw":out[:1200]})   # ⭐ 记 raw ⇒ 可回溯 ✓
        if i%20==0 or i==len(rows):
            print("  [%3d/%3d] 正确 %d ⇒ 当前 %.1f%%  (%.0f s)" % (i,len(rows),ok,ok/i*100,time.time()-t0), flush=True)
    res={"n":len(rows),"correct":ok,"acc":ok/len(rows) if rows else 0,
         "fewshot":FEWSHOT,"maxtok":MT,"secs":time.time()-t0,"details":det}
    json.dump(res, open(OUT,"w"), ensure_ascii=False, indent=1)
    print("  ⇒ 【GSM8K 基线】%d/%d = %.1f%%  ⇒ %s" % (ok,len(rows),ok/len(rows)*100 if rows else 0, OUT))
main()
