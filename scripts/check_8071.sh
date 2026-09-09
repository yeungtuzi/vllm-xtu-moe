#!/usr/bin/env bash
# Functional check against the 8071 test server (never touches prod 8070).
set -euo pipefail
PORT="${PORT:-8071}"
PROMPT="${PROMPT:-The capital of France is}"
MAXTOK="${MAXTOK:-32}"

echo "== /health =="
curl -s -m 10 "http://127.0.0.1:${PORT}/health" || echo "(health failed)"

echo "== /v1/completions =="
curl -s -m 300 "http://127.0.0.1:${PORT}/v1/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"DeepSeek-V4-Flash-xiaotu\",\"prompt\":\"${PROMPT}\",\"max_tokens\":${MAXTOK},\"temperature\":0}" \
  | python -c "import json,sys; d=json.load(sys.stdin); c=d['choices'][0]; print('text:', repr(c['text'][:200])); print('finish:', c.get('finish_reason')); print('usage:', d.get('usage'))"
