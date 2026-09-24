#!/usr/bin/env bash
# 提交前自检:禁止把内网地址 / 凭据 / 主机名等敏感值写进**会被提交的文件**。
#
# 用法:
#   scripts/check_no_secrets.sh            # 检查**已暂存**的改动(提交前跑 ✓)
#   scripts/check_no_secrets.sh --all      # 检查**已跟踪**的全部文件
#   scripts/check_no_secrets.sh --diff     # 检查工作区未提交改动
#
# 输出:仅 **文件:行号 + 类别**(**绝不打印匹配到的值** ✗ —— 否则等于再次泄露 ✓)
# 退出码:0 = 干净 ✓;1 = 命中(请改成占位符,如 http://<proxy-host>:<port> ✓)
set -uo pipefail
MODE="${1:---staged}"

# 类别:正则(注意不要在这里写任何真实值 ✓)
PATTERNS=(
  'RFC1918 内网 IPv4|(^|[^0-9])(192\.168\.|10\.[0-9]+\.[0-9]+\.[0-9]+|172\.(1[6-9]|2[0-9]|3[01])\.)'
  '带端口的内网地址|(192\.168\.|10\.)[0-9.]+:[0-9]{2,5}'
  '私钥/令牌形态|(sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{12,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)'
  '密钥赋值|(api[_-]?key|apiKey|secret|passwd|password)[[:space:]]*[:=][[:space:]]*["'"'"'][^"'"'"']{8,}'
  '内网域名/主机名|(\.(internal|corp|lan|local)\b|tinyproxy)'
)

case "$MODE" in
  --staged) FILES=$(git diff --cached --name-only --diff-filter=ACM 2>/dev/null); READ=(git diff --cached -U0 --) ;;
  --diff)   FILES=$(git diff --name-only --diff-filter=ACM 2>/dev/null);          READ=(git diff -U0 --) ;;
  --all)    FILES=$(git ls-files 2>/dev/null);                                    READ=(git grep -nI -E) ;;
  *) echo "用法: $0 [--staged|--diff|--all]" >&2; exit 2 ;;
esac
[ -z "${FILES:-}" ] && { echo "[secrets] 无待检文件 ✓"; exit 0; }

rc=0
for entry in "${PATTERNS[@]}"; do
  label="${entry%%|*}"; re="${entry#*|}"
  if [ "$MODE" = "--all" ]; then
    hits=$(git grep -nI -E "$re" -- $FILES 2>/dev/null | cut -d: -f1,2)
  else
    hits=$(git diff ${MODE/--staged/--cached} -U0 -- $FILES 2>/dev/null \
           | awk -v re="$re" 'BEGIN{n=0} /^\+\+\+ b\//{f=substr($0,7)} /^@@/{split($3,a,","); ln=substr(a[1],2)-1} /^\+/ && !/^\+\+\+/{ln++; if (match(substr($0,2), re)) print f":"ln}' 2>/dev/null)
  fi
  if [ -n "$hits" ]; then
    rc=1
    echo "[secrets] ⚠️ 命中类别:$label"
    echo "$hits" | sort -u | head -10 | sed 's/^/    /'
    echo "    ⇒ 请改为占位符(如 http://<proxy-host>:<port>)或删除 ✓"
  fi
done
[ "$rc" -eq 0 ] && echo "[secrets] 干净 ✓" || echo "[secrets] ⛔ 命中,禁止提交 ✗"
exit $rc
