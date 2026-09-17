#!/usr/bin/env bash
# JIT 固定缓存目录(用户 2026-09-17 要求:"把JIT的固定缓冲目录这个功能也加上,
# 不然每次都重新JIT太可怕了,默认目录就用lk的设定即可")。
#
# ---- 为什么需要它(全部在**安装好的 vLLM 源码**里逐行核实过,不是推测)----------
# 1) vLLM 的 torch.compile 缓存目录 = `$VLLM_CACHE_ROOT/torch_compile_cache/<hash10>`,
#    `hash10 = sha256([env_hash, config_hash, code_hash, compiler_hash])[:10]`
#    (`vllm/compilation/backends.py:1028-1067`),其中
#      * `env_hash`    = **每一个已知的 VLLM_* 环境变量**(`vllm/envs.py:compile_factors()`,
#        只有一张很小的 ignore 白名单,且 `VLLM_CACHE_ROOT` 本身在白名单里);
#      * `config_hash` = `vllm_config.compute_hash()`(max_model_len /
#        max_num_batched_tokens / max_num_seqs / cudagraph 模式 … 全在里面);
#      * `code_hash`   = **被 trace 到的 Python 源码内容**(`traced_files`;
#        我们插件替换的模型类 `hybrid_model.py` / `mixed_experts.py` 就在其中);
#      * `compiler_hash`。
#    ⇒ 改**一个**旗标(例如开关 GPU 预填充用的 `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS`)、
#      或者改**一行**我们自己的代码,目录名就换新的 ⇒ 整份编译缓存作废,
#      **逐形状重新 JIT**(`jit_monitor` 报的 20-60 s/形状 就是它)。
# 2) 更糟的是:`CompilerInterface.initialize_cache()`
#    (`vllm/compilation/compiler_interface.py:470-481`)会把 `TRITON_CACHE_DIR`
#    **重定向**成 `<cache_dir>/triton_cache`,也就是那个 hash 目录
#    (`base_cache_dir = cache_dir[: -len(prefix)]`)。所以 `~/.triton/cache` 里
#    攒下的几千个内核**根本用不上** —— 每次都在新目录里从零编。
#
# ---- 本文件做什么 ------------------------------------------------------------
# 给编译缓存**钉一个稳定目录**(默认沿用 vLLM 自己的根,即参考实现 lk 的设定:
# `~/.cache/vllm/torch_compile_cache`),并把 Triton / Inductor / TileLang 的缓存
# 都指到它下面。目录名里只保留**真正影响产物正确性**的几样东西:
#     <root>/xtu-<模型名>-tp<TP>-<编译模式>-<源码指纹>
# 其中源码指纹 = sha1(我们的 python+引擎 csrc 源码) + vLLM 上游 `git rev-parse --short HEAD`。
# 于是:
#   * **重复启动、换旗标扫描 ⇒ 命中同一个目录**(这才是"固定"要解决的问题);
#   * **我们改了插件/引擎源码、或上游 vLLM 变了 ⇒ 自动换新目录**,不会读到陈旧的计算图;
#   * 想强制重建:`JITCACHE_STAMP=<任意串>`;想完全回到 vLLM 默认的 hash 行为:`JITCACHE=0`。
#
# **只在编译路径上生效**(调用方用 `JITCACHE_COMPILING=1` 声明本次跑 VLLM_COMPILE):
# mode=NONE 时 vLLM 不调 initialize_cache,Triton 用的就是自己的默认目录
# `~/.triton/cache` —— 那本来就固定且持久(本机 1.3 GB / 2814 内核),不该去动它。
#
# ---- 用法 --------------------------------------------------------------------
#   ROOT=<repo>; source scripts/lib_jitcache.sh     # 调用方先设 ROOT
# 先设(可选):JITCACHE(默认 1)、JITCACHE_MODEL、JITCACHE_TP、JITCACHE_MODE、
#             JITCACHE_COMPILING(1 = 本次走编译,才会改那三个 cache env)、
#             JITCACHE_STAMP、XIAOTU_JIT_CACHE_DIR(覆盖根目录)
# 产出:
#   JITCACHE_DIR       固定目录(空 = 未启用;无论是否编译都会算出名字,便于记录)
#   JITCACHE_CC_EXTRA  要拼进 --compilation-config JSON 的片段:`,"cache_dir":"…"`
#   TRITON_CACHE_DIR / TORCHINDUCTOR_CACHE_DIR / TILELANG_CACHE_DIR
#                      **仅当 JITCACHE_COMPILING=1** 时才导出(已导出)
# License: Apache-2.0
# shellcheck shell=bash

_JITCACHE_ROOT_REPO="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
JITCACHE="${JITCACHE:-1}"
JITCACHE_DIR=""
JITCACHE_CC_EXTRA=""

if [ "$JITCACHE" = "1" ]; then
  # 参考实现 lk 用的就是 vLLM 自己的默认根(`$VLLM_CACHE_ROOT` = `~/.cache/vllm`),
  # 我们沿用同一个根,只是**去掉 hash 层**、换成稳定的名字。
  _jit_root="${XIAOTU_JIT_CACHE_DIR:-$HOME/.cache/vllm/torch_compile_cache}"

  # 源码指纹:只覆盖"会进计算图/影响输出"的东西 —— 我们的插件 python + 引擎 C++ 源码。
  # 注意**不要**把 /tmp/xiaotu_env 之类的运行时旗标算进来(那正是要复用的场景)。
  _jit_src="$( { cat "$_JITCACHE_ROOT_REPO"/vllm_xiaotu_moe/*.py \
                     "$_JITCACHE_ROOT_REPO"/xiaotu_moe/csrc/moe/*.hpp \
                     "$_JITCACHE_ROOT_REPO"/xiaotu_moe/csrc/python_binding/*.cpp 2>/dev/null; } \
                 | sha1sum 2>/dev/null | cut -c1-8 )"
  [ -n "$_jit_src" ] || _jit_src="nosrc"
  _jit_vllm="$(git -C /home/user/lvllm/process_data/ref/repos/vllm-mainline rev-parse --short HEAD 2>/dev/null || echo novllm)"

  _jit_name="xtu-$(basename "${JITCACHE_MODEL:-model}")-tp${JITCACHE_TP:-1}-${JITCACHE_MODE:-none}-${_jit_vllm}-${_jit_src}"
  [ -n "${JITCACHE_STAMP:-}" ] && _jit_name="$_jit_name-${JITCACHE_STAMP}"
  JITCACHE_DIR="$_jit_root/$_jit_name"
  JITCACHE_CC_EXTRA=",\"cache_dir\":\"$JITCACHE_DIR\""

  # 只在**真的走编译**时改这三个 env(见下面"为什么不无条件改")。调用方用
  # JITCACHE_COMPILING=1 声明"本次跑 VLLM_COMPILE"。
  #
  # 为什么 mode=NONE 时**不要**动:那条路径下 vLLM 根本不会调 initialize_cache,
  # Triton 用的就是它自己的默认目录 `~/.triton/cache` —— 那个**本来就已经是固定且
  # 持久的**(本机现成 1.3 GB / 2814 个内核)。为了"统一"而去改它,只会白白让这一份
  # 攒了很久的缓存作废一次(一次全量重编),换不来任何东西。
  # 真正坏掉的只有编译路径:Triton 被重定向进 hash 目录。
  if [ "${JITCACHE_COMPILING:-0}" = "1" ]; then
    mkdir -p "$JITCACHE_DIR/triton" "$JITCACHE_DIR/inductor" 2>/dev/null
    export TRITON_CACHE_DIR="$JITCACHE_DIR/triton"
    export TORCHINDUCTOR_CACHE_DIR="$JITCACHE_DIR/inductor"
    export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-$HOME/.tilelang/cache}"
    _jit_used="(已钉住)"
  else
    _jit_used="(本次不编译 ⇒ 不改 Triton 目录,沿用 ~/.triton/cache)"
  fi
  echo "[jitcache] 固定缓存目录 = $JITCACHE_DIR $_jit_used"
  [ "${JITCACHE_COMPILING:-0}" = "1" ] && {
    echo "[jitcache]   TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
    echo "[jitcache]   TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR"
  }
fi
