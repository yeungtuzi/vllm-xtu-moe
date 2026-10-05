// ---------------------------------------------------------------------------
// ISA 选择的 packed4 GEMM 实现（W4A8 集成，2026-10-05）
//
// 为什么这么做：int8 激活路径（`vpdpbusd` + per-32 块 scale）是一项**可选能力**，
// 而它必须活在同一个热路径宏 `XIAOTU_TILE_BODY` 里 ⇒ **无法用 `#if` 从宏体内部剔除**
// （C 预处理器不允许宏体里出现指令 ✗）。若把两段实现混在一个文件里，**非 VNNI 档也会把
// int8 代码编进去** ⇒ 实测让生产默认路径（`avx512_bf16_vbmi`）**慢 25%** ✗
// （输出仍逐字节相同，但耗时退化 ⇒ 不可接受 ✗）。
//
// 处理：按 ISA 在**编译期**选一份实现 ——
//   * 启用 `-mavx512vnni` 的档 ⇒ `moe_v2_packed4_w4a8.inc`（带 int8/ALIGN 路径，门控默认关 ✓）
//   * 其它档（生产在用 `avx512_bf16_vbmi`）⇒ `moe_v2_packed4_base.inc`（**与历史实现逐字节同一份** ✓）
// ⇒ **非 VNNI 档的产物与"未引入 W4A8"时完全一致**（可用 `.so` 的 `sha256` 机械核验 ✓）
// ---------------------------------------------------------------------------
#ifndef XIAOTU_MOE_MOE_V2_PACKED4_HPP_SELECT
#define XIAOTU_MOE_MOE_V2_PACKED4_HPP_SELECT
#if defined(__AVX512VNNI__)
#  include "moe_v2_packed4_w4a8.inc"
#else
#  include "moe_v2_packed4_base.inc"
#endif
#endif
