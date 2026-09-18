#!/usr/bin/env bash
# Upstream drift check for vllm-xtu-moe.
#
# Development principle (see dev-docs/report/tuning/IRON_RULES.md R10): the mainline moves
# fast, so we check it on a schedule and follow up, instead of discovering the
# drift months later.  This script answers three questions:
#
#   0. has upstream main moved, and is the newest upstream commit *installable*
#      on this host?  <- the decisive constraint discovered 2026-09-14: this host
#      has nvcc 12.1 + torch cu130 and no Rust toolchain, so a from-source build
#      is impossible.  We can only adopt a commit that publishes a precompiled
#      wheel for our CUDA variant.  "upstream supports X" != "we can run X".
#   1. do our patches still apply to upstream main?
#   2. do the upstream modules the plugin imports still exist?
#
# Usage:  scripts/check_upstream_drift.sh [--full]
#   --full  also print the per-file upstream commit counts for the files we patch
#
# Env:    BASE=<full commit>   REPO=<vllm mainline checkout>   VARIANT=cu130
#
# Exit code: 0 = no actionable drift, 1 = drift detected, 2 = could not check.
set -uo pipefail

# The mainline commit our installed plugin/patch set was last verified against.
BASE="${BASE:-dabc4362b47ad2665b0802b0b42b13f660efe634}"
UPSTREAM="${UPSTREAM:-https://github.com/vllm-project/vllm.git}"
PLUGIN_DIR="${PLUGIN_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
VARIANT="${VARIANT:-cu130}"
WHEEL_INDEX="${WHEEL_INDEX:-https://wheels.vllm.ai}"

# Prefer the tree this project actually develops against.
if [ -z "${REPO:-}" ]; then
  for cand in /home/user/lvllm/process_data/ref/repos/vllm-mainline \
              "$PLUGIN_DIR/../vllm-mainline"; do
    if [ -d "$cand/vllm" ]; then REPO="$cand"; break; fi
  done
fi

WORK="${WORK:-/tmp/xtu-drift}"
UP_REF=refs/remotes/up/main
BRANCHES=(xtu/pr1-experts-load-device xtu/pr2-fp8-sm80-o-proj xtu/pr3-sm80-port)
STORED=("$PLUGIN_DIR/patches/upstream/pr1-experts-load-device.patch"
        "$PLUGIN_DIR/patches/upstream/pr2-fp8-sm80-o-proj.patch"
        "$PLUGIN_DIR/patches/upstream/pr3-sm80-port.patch")
DRIFT=0
HAVE_GH=0; command -v gh >/dev/null 2>&1 && HAVE_GH=1
HAVE_NET=0
curl -sL -m 15 -o /dev/null "$WHEEL_INDEX/nightly/$VARIANT/vllm/metadata.json" && HAVE_NET=1

echo "== upstream drift check =="
echo "base     : ${BASE:0:12}"
echo "plugin   : $PLUGIN_DIR"
echo "mainline : ${REPO:-<not found>}"
echo

# ---------------------------------------------------------------------------
echo "== 0. can we actually adopt a newer upstream? (precompiled wheel) =="
# The wheel filename carries the commit it was built from, e.g.
#   vllm-0.29.1rc1.dev95+gdabc4362b-cp38-abi3-manylinux_2_28_x86_64.whl
if [ "$HAVE_NET" -eq 1 ]; then
  WHEEL_JSON=$(curl -sL -m 30 "$WHEEL_INDEX/nightly/$VARIANT/vllm/metadata.json")
  WHEEL_SHA=$(printf '%s' "$WHEEL_JSON" \
    | grep -oE '\+g[0-9a-f]{7,40}-' | head -1 | tr -d '+g-')
  if [ -n "$WHEEL_SHA" ]; then
    echo "  newest wheel-backed upstream commit : ${WHEEL_SHA:0:12} ($VARIANT)"
  else
    echo "  newest wheel-backed upstream commit : <could not parse nightly index>"
    DRIFT=1
  fi
  if [ "$HAVE_GH" -eq 1 ]; then
    MAIN_SHA=$(gh api repos/vllm-project/vllm/commits/main --jq '.sha' 2>/dev/null)
    MAIN_DATE=$(gh api repos/vllm-project/vllm/commits/main --jq '.commit.author.date' 2>/dev/null)
    if [ -n "$MAIN_SHA" ]; then
      echo "  upstream main HEAD                  : ${MAIN_SHA:0:12} ($MAIN_DATE)"
      if [ "$MAIN_SHA" = "$WHEEL_SHA" ]; then
        echo "  VERDICT: HEAD is installable (wheel published)"
      else
        AHEAD=$(gh api "repos/vllm-project/vllm/compare/${WHEEL_SHA}...main" \
                  --jq '.ahead_by' 2>/dev/null || echo '?')
        echo "  VERDICT: HEAD has NO wheel ($AHEAD commits past the newest wheel)."
        echo "           Upgrade target = ${WHEEL_SHA:0:12}; HEAD is not installable here."
      fi
    fi
    # Is the commit we are pinned to still wheel-backed? (it must be)
    if curl -sL -m 20 -o /dev/null -w '%{http_code}' \
         "$WHEEL_INDEX/$BASE/$VARIANT/vllm/metadata.json" | grep -q '^200$'; then
      echo "  pinned base has a wheel             : yes"
    else
      echo "  pinned base has a wheel             : NO (re-verify install path!)"
      DRIFT=1
    fi
    BEHIND=$(gh api "repos/vllm-project/vllm/compare/$BASE...main" --jq '.ahead_by' 2>/dev/null || echo '?')
    echo "  commits since our base              : $BEHIND"
  fi
else
  echo "  (no network to $WHEEL_INDEX - skipped)"
fi
echo

# ---------------------------------------------------------------------------
echo "== 1. our patches vs current upstream main =="
if [ -n "${REPO:-}" ] && [ -d "$REPO/.git" ]; then
  if [ ! -d "$WORK/.git" ]; then
    git clone --quiet --depth 1 "$UPSTREAM" "$WORK" || { echo "  clone failed"; exit 2; }
  fi
  # `+` forces the mirror ref: a stale shallow work tree may hold a diverged
  # up/main, which makes a plain fetch fail as "non-fast-forward" and would
  # otherwise abort the whole check.
  if ! git -C "$WORK" fetch --quiet --depth 1 "$UPSTREAM" "+main:$UP_REF"; then
    echo "  (stale work tree at $WORK - re-cloning)"
    rm -rf "$WORK"
    git clone --quiet --depth 1 "$UPSTREAM" "$WORK" || { echo "  clone failed"; exit 2; }
    git -C "$WORK" fetch --quiet --depth 1 "$UPSTREAM" "+main:$UP_REF" || exit 2
  fi
  git -C "$WORK" fetch --quiet --depth 1 "$UPSTREAM" "$BASE" >/dev/null 2>&1 || true
  echo "  upstream main: $(git -C "$WORK" rev-parse --short "$UP_REF") ($(git -C "$WORK" show -s --format=%ci "$UP_REF"))"

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
      git -C "$WORK" apply --3way "$patch" >/dev/null 2>&1 || true
      files=$(git -C "$WORK" diff --name-only --diff-filter=U | wc -l)
      if [ "$files" -gt 0 ]; then
        echo "  $b: DRIFT - $files file(s) need a manual rebase"
        git -C "$WORK" diff --name-only --diff-filter=U | sed 's/^/      /'
        DRIFT=1
      else
        echo "  $b: applies cleanly (3-way merge resolved context drift)"
      fi
    fi
    git -C "$WORK" reset -q --hard "$UP_REF"
  done
else
  echo "  SKIP (no mainline checkout; set REPO=...)"
fi
echo

# ---------------------------------------------------------------------------
echo "== 2. plugin's upstream API surface =="
# A module can survive while a name inside it moves - that is exactly how
# `cpu_moe.select_experts` broke on the dabc4362b upgrade (it moved to
# `router/cpu_router.py`). Parse the plugin with `ast`, so aliases, comments and
# multi-line imports are handled correctly, and resolve every `vllm...` import
# against the upstream tree.
if [ -n "${WORK:-}" ] && [ -d "$WORK/.git" ]; then
  api_rc=0
  python3 - "$PLUGIN_DIR" "$WORK" <<'PY' || api_rc=$?
import ast
import pathlib
import re
import sys

plugin_dir, tree = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
mods: set[str] = set()
syms: set[tuple[str, str]] = set()
for f in sorted((plugin_dir / "vllm_xiaotu_moe").glob("*.py")):
    try:
        parsed = ast.parse(f.read_text())
    except SyntaxError as e:
        print(f"  PARSE-FAIL {f.name}: {e}")
        continue
    for n in ast.walk(parsed):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.split(".")[0] == "vllm":
            mods.add(n.module)
            for a in n.names:
                if a.name != "*":
                    syms.add((n.module, a.name))
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] == "vllm":
                    mods.add(a.name)


def resolve(mod: str) -> pathlib.Path | None:
    p = tree / mod.replace(".", "/")
    if p.with_suffix(".py").is_file():
        return p.with_suffix(".py")
    if (p / "__init__.py").is_file():
        return p / "__init__.py"
    return None


missing_mod = 0
for mod in sorted(mods):
    if resolve(mod) is None:
        print(f"  MISSING module {mod}")
        missing_mod += 1

missing_sym: list[str] = []
unverifiable = 0
for mod, name in sorted(syms):
    src = resolve(mod)
    if src is None:
        continue
    base = tree / mod.replace(".", "/")
    if (base / f"{name}.py").is_file() or (base / name / "__init__.py").is_file():
        continue  # submodule import, e.g. `from vllm import envs`
    text = src.read_text()
    if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text):
        continue
    if re.search(r"^\s*from\s+\S+\s+import\s+\*", text, re.M):
        unverifiable += 1  # re-exported via `import *`; not checkable statically
        continue
    missing_sym.append(f"{mod}.{name}")

for m in missing_sym:
    print(f"      MISSING symbol {m}")
print(
    f"  ({len(mods)} upstream modules, {missing_mod} missing; "
    f"{len(syms)} symbols, {len(missing_sym)} missing, "
    f"{unverifiable} unverifiable via `import *`)"
)
sys.exit(1 if (missing_mod or missing_sym) else 0)
PY
  [ "$api_rc" -ne 0 ] && DRIFT=1
else
  echo "  SKIP (no upstream work tree)"
fi
echo

# ---------------------------------------------------------------------------
if [ "${1:-}" = "--full" ]; then
  echo "== 3. upstream commits in the last 90 days for files we patch =="
  since=$(date -d '90 days ago' +%Y-%m-%dT00:00:00Z)
  if [ -n "${REPO:-}" ]; then
    for f in $(git -C "$REPO" diff --name-only "$BASE" 2>/dev/null | head -40); do
      n=$(gh api "repos/vllm-project/vllm/commits?path=$f&since=$since&per_page=1" -i 2>/dev/null \
          | grep -i '^link:' | tail -1 \
          | grep -oE 'page=[0-9]+>; rel="last"' | grep -oE '[0-9]+')
      printf "  %5s  %s\n" "${n:-0}" "$f"
    done
  fi
  echo
fi

if [ "$DRIFT" -eq 0 ]; then
  echo "RESULT: no actionable drift - HEAD installable-or-pinned, patches apply, API surface intact."
else
  echo "RESULT: DRIFT detected - follow up (see dev-docs/report/tuning/IRON_RULES.md R10)."
fi
exit "$DRIFT"
