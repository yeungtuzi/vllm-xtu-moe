#pragma once

// ---------------------------------------------------------------------------
// 【T1.4 格式感知派发】W4A8（int8 激活）路径的 **进程级** 模式 ✓
//
//   0  = 关（**默认** ⇒ 零回归 ✓）
//   1  = ALIGN int8 路径（每 64 MAC 只占 1 个 FP0/1 槽；实测 1.54–1.58× vs 生产默认路径 ✓）
//   2  = 旧 VNNI int8 路径（实测约 1.0×，保留用于对照 ✓）
//   -1 = 尚未设置（此时**只看 env** ⇒ 向后兼容 dev 的 `XIAOTU_MOE_INT8_*` ✓）
//
// 为什么用**普通全局 int** 而不是函数内 `static`：
//   后者会生成**线程安全初始化守卫**（`__cxa_guard_acquire` ⇒ 跨线程原子读 ✗）。
//   本引擎历史上正因同类问题（热路径里的原子/守卫）吃过亏，见
//   `moe_v2_packed4_w4a8.inc` 第 353–360 行的注释 ✓。普通全局 int 的读取就是一次普通 load ✓。
//
// 为什么是"进程级"而不是逐层 cfg：
//   该门控在热路径的静态 helper 里被读取（`xiaotu_int8_align_on()`），逐层透传会改动热路径签名 ✗；
//   而**同一进程内所有 MoE 层的权重格式一致**（同一 checkpoint ✓）⇒ 进程级语义足够且零侵入 ✓。
//
// ⚠️ 只有**带 `-mavx512vnni` 编译的档**才含该路径（见 `moe_v2_packed4.hpp` 的 ISA shim）：
//   非 VNNI 档上把本值设为 1/2 **不报错也不生效**（门控恒为 false ✓）。
// ---------------------------------------------------------------------------
namespace xiaotu_int8 {

inline int g_activation = -1;                      // C++17 inline variable ⇒ 无守卫 ✓

inline void set_activation(int m) { g_activation = m; }

inline int activation() { return g_activation; }

}  // namespace xiaotu_int8
