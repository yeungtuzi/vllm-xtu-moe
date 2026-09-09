#!/usr/bin/env bash
# Upstream drift check for vllm-xtu-moe.
#
# Answers two questions every time it runs:
#   1. do our PR branches/patches still apply to the current upstream vLLM main?
#   2. do the upstream modules the plugin imports still exist?
#
# Usage:  scripts/check_upstream_drift.sh [--full]
#   --full  also print the per-file upstream commit counts for the files we patch
#
# Exit code: 0 = no drift, 1 = drift detected.
set -uo pipefail

BASE="${BASE:-6c73b08dec2af5052288169663549687ba61f330}"
UPSTREAM="${UPSTREAM:-https://github.com/vllm-project/vllm.git}"
PLUGIN_DIR="${PLUGIN_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
REPO="${REPO:-$PWD/../vllm-mainline}"   # path to a vLLM mainline checkout
WORK="${WORK:-/tmp/xtu-drift}"
UP_REF=refs/remotes/up/main
BRANCHES=(xtu/pr1-experts-load-device xtu/pr2-fp8-sm80-o-proj xtu/pr3-sm80-port)
STORED=("$PLUGIN_DIR/patches/upstream/pr1-experts-load-device.patch"
        "$PLUGIN_DIR/patches/upstream/pr2-fp8-sm80-o-proj.patch"
        "$PLUGIN_DIR/patches/upstream/pr3-sm80-port.patch")
DRIFT=0

echo "== upstream drift check =="
echo "base     : $BASE"
if [ ! -d "$WORK/.git" ]; then
  git clone --quiet --depth 1 "$UPSTREAM" "$WORK" || exit 2
fi
git -C "$WORK" fetch --quiet --depth 1 "$UPSTREAM" "main:$UP_REF" || exit 2
# fetch the base commit too, so `git apply --3way` can find the pre-image blobs
git -C "$WORK" fetch --quiet --depth 1 "$UPSTREAM" "$BASE" >/dev/null 2>&1 || true
echo "upstream : $(git -C "$WORK" rev-parse --short "$UP_REF") ($(git -C "$WORK" show -s --format=%ci "$UP_REF"))"
BEHIND=$(gh api "repos/vllm-project/vllm/compare/$BASE...main" --jq '.ahead_by' 2>/dev/null || echo "?")
echo "behind   : $BEHIND commits since our base"
echo

echo "== 1. our patches vs current upstream main =="
mkdir -p "$WORK/patches"
for i in "${!BRANCHES[@]}"; do
  b="${BRANCHES[$i]}"
  patch="$WORK/patches/$(echo "$b" | tr '/' '-').patch"
  if git -C "$REPO" rev-parse --verify -q "$b" >/dev/null 2>&1; then
    git -C "$REPO" diff "$BASE..$b" > "$patch"
  elif [ -f "${STORED[$i]}" ]; then
    cp "${STORED[$i]}" "$patch"
  else
    echo "  $b: SKIP (no branch and no stored patch)"
    continue
  fi

  git -C "$WORK" checkout -q --detach "$UP_REF"
  git -C "$WORK" reset -q --hard "$UP_REF"
  if git -C "$WORK" apply --check "$patch" 2>/dev/null; then
    echo "  $b: applies cleanly (exact context)"
  else
    # --3way exits non-zero both on hard failure and on conflicts: inspect the
    # index to tell them apart.
    git -C "$WORK" apply --3way "$patch" >/dev/null 2>&1 || true
    files=$(git -C "$WORK" diff --name-only --diff-filter=U | wc -l)
    if [ "$files" -gt 0 ]; then
      marks=0
      while read -r uf; do
        [ -n "$uf" ] || continue
        c=$(grep -c '^<<<<<<<' "$WORK/$uf" 2>/dev/null || true)
        marks=$((marks + ${c:-0}))
      done < <(git -C "$WORK" diff --name-only --diff-filter=U)
      echo "  $b: DRIFT - $files file(s) need a manual rebase, $marks conflict marker(s)"
      git -C "$WORK" diff --name-only --diff-filter=U | sed 's/^/      /'
      DRIFT=1
    else
      echo "  $b: applies cleanly (3-way merge resolved context drift)"
    fi
  fi
  git -C "$WORK" reset -q --hard "$UP_REF"
done
echo

echo "== 2. plugin's upstream API surface =="
syms=$(grep -rhoE "^\s*from vllm[a-zA-Z0-9_.]*\s+import\s+\(?" "$PLUGIN_DIR"/vllm_xiaotu_moe/*.py \
        | sed 's/^\s*from //; s/ import.*//' | grep -E '^vllm(\.|$)' | sort -u)
missing=0
for mod in $syms; do
  path="$WORK/$(echo "$mod" | tr '.' '/')"
  if [ -f "$path.py" ] || [ -f "$path/__init__.py" ]; then
    printf "  ok      %s\n" "$mod"
  else
    printf "  MISSING %s\n" "$mod"
    missing=$((missing + 1))
    DRIFT=1
  fi
done
echo "  ($(echo "$syms" | wc -l) upstream modules imported, $missing missing)"
echo

if [ "${1:-}" = "--full" ]; then
  echo "== 3. upstream commits in the last 90 days for files we patch =="
  since=$(date -d '90 days ago' +%Y-%m-%dT00:00:00Z)
  for f in $(git -C "$REPO" diff --name-only "$BASE" | head -40); do
    n=$(gh api "repos/vllm-project/vllm/commits?path=$f&since=$since&per_page=1" -i 2>/dev/null \
        | grep -i '^link:' | tail -1 \
        | grep -oE 'page=[0-9]+>; rel="last"' | grep -oE '[0-9]+')
    printf "  %5s  %s\n" "${n:-0}" "$f"
  done
  echo
fi

if [ "$DRIFT" -eq 0 ]; then
  echo "RESULT: no drift - patches apply and every imported module exists."
else
  echo "RESULT: DRIFT detected - see above."
fi
exit "$DRIFT"
