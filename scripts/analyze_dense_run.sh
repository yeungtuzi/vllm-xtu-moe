#!/usr/bin/env bash
# 【2026-10-05】等采样结束后自动出分析报告(不依赖 agent 在线 ✓)
# 用法: bash scripts/analyze_dense_run.sh [采样目录] [额外等待秒数]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; L="$HERE/../dev-docs/report/tuning/logs"
D="${1:-$(ls -dt "$L"/dense_* 2>/dev/null | head -1)}"
WAIT="${2:-120}"
[ -d "$D" ] || { echo "找不到采样目录 $D"; exit 1; }
# 1) 等采样器自己结束(DURATION 到点)+ 宽限
for i in $(seq 1 90); do
  bash "$HERE/proc.sh" status dense_sample 2>/dev/null | grep -q RUNNING || break
  sleep 60
done
sleep "$WAIT"
CSV="$D/samples.csv"; PF="$D/pf_progress.txt"; OUT="$D/ANALYSIS.md"
{
echo "# 密集采样分析报告"
echo
echo "采样目录: \`$(basename "$D")\`"
echo "生成时间: $(date '+%F %T')"
echo
echo "## 1. 规模与判定分布"
awk -F, 'NR>1{n++; v[$10]++} END{printf "- 样本数: **%d**\n", n; for(k in v) printf "- verdict \`%s\`: %d\n", k, v[k]}' "$CSV"
echo
echo "## 2. ⭐ 铁证检测:零请求却 GPU 满载(逃逸内核)"
awk -F, 'NR>1 && $3+0==0 && $4+0==0 && $7+0>50 && $8+0>150 {n++; if(n<=5) printf "- %s  gpu=%s%% power=%sW\n", $1,$7,$8} END{printf "\n⇒ 命中 **%d** 个样本 %s\n", n+0, (n+0>0?"✗ 出现逃逸内核":"✓ 未出现")}' "$CSV"
echo
echo "## 3. 停滞检测:有请求但 token 连续不推进(>3 分钟)"
awk -F, 'NR>1{r=$3+0; tk=$5"/"$6; if(r>0){if(tk==prev){cnt++}else{cnt=0;prev=tk;st=$1}
  if(cnt>180 && !done){printf "- 自 %s 起 token 冻结(样本 %d 个)\n", st, cnt; done=1}
 } else {prev=""; cnt=0; done=0}}' "$CSV" | head -5
echo
echo "## 4. 状态量趋势(哪个先异常)"
awk -F, 'NR>1{av=$12+0; kv=$11+0; ls=$13+0;
  if(mn==""||av<mn)mn=av; if(kv>mk)mk=kv; if(ls>ml)ml=ls;
  if($7+0>mu)mu=$7; if($8+0>mp)mp=$8} END{
  printf "- **MemAvailable 最低**: %d GiB(⚠️ <50 需警惕)\n", mn;
  printf "- **KV 池使用率峰值**: %s%%\n", (mk==""?"n/a":mk);
  printf "- **LMCache 已存块数**(末值): %d\n", ml;
  printf "- **GPU 利用率峰值**: %d%%  **功耗峰值**: %dW\n", mu, mp}' "$CSV"
echo
echo "## 5. ⭐ 进度心跳分析(区分"原地重来"与"推进")"
if [ -s "$PF" ]; then
  total=$(wc -l < "$PF")
  echo "- 心跳条数: **$total**"
  echo "- 层号分布(前 8):"
  sed 's/\x1b\[[0-9;]*m//g' "$PF" | grep -oE 'layers\.[0-9]+' | sort | uniq -c | sort -rn | head -8 | sed 's/^/  - /'
  echo "- qlen 分布(前 5):"
  sed 's/\x1b\[[0-9;]*m//g' "$PF" | grep -oE 'qlen=[0-9]+' | sort | uniq -c | sort -rn | head -5 | sed 's/^/  - /'
  echo "- 两路占比: device=cpu $(grep -c 'device=cpu' "$PF") / device=cuda $(grep -c 'device=cuda' "$PF")"
  echo "- **call 序号范围**: $(grep -oE 'call=[0-9]+' "$PF" | head -1) → $(grep -oE 'call=[0-9]+' "$PF" | tail -1)"
  echo "- 最后 5 条:"
  sed 's/\x1b\[[0-9;]*m//g' "$PF" | tail -5 | sed 's/^/  - /'
else
  echo "- ⚠️ 本次没有捕获到进度心跳(说明这段时间没走预填装配,或采样窗口内全在解码)"
fi
echo
echo "## 6. 看门狗告警(同期)"
grep -aE "铁证|可疑|★★★" "$L/watch_hang.log" 2>/dev/null | tail -8 | sed 's/^/  - /' || echo "  - 无"
echo
echo "## 7. 结论与下一步"
awk -F, 'NR>1 && $3+0==0 && $4+0==0 && $7+0>50 && $8+0>150{n++} END{
  if(n+0>0) print "⇒ ✗ **复现了逃逸内核** ⇒ 详见采样目录下的 hang_* 与 watch_hang.log ⇒ 用 call/ids0/层号定位到具体层 ✓"
  else print "⇒ ✓ 本次窗口**未出现逃逸内核**。若期间也无"长时间不推进",则故障未在 60 分钟内复发 ✓ ⇒ 继续拉长观察或按状态相关思路查(哪个量随时间漂移 ✗)"}' "$CSV"
} > "$OUT" 2>&1
echo "报告已生成: $OUT"
