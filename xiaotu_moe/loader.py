# xiaotu-moe: dynamic ISA loader (Phase 4).
#
# The native extension is built once per ISA level into several coexisting
# modules (_xiaotu_moe_C_scalar / _avx2 / _avx512_base / _avx512_vnni /
# _avx512_bf16). This loader inspects the host CPU at import time and imports
# the best supported variant, exposing the same public API as any variant.
#
# License: Apache-2.0

from __future__ import annotations

import importlib
import os
import sys

# Highest-first ISA ladder. Each entry: (display name, module suffix,
# required /proc/cpuinfo flag set).
# NOTE: `avx512_bf16` implies avx512_base + vnni; modern EPYC (e.g. 9654) has
# avx512_bf16. AMX (avx512_amx) is Intel-only and skipped here.
_LADDER = [
    ("avx512_bf16_vbmi", "_avx512_bf16_vbmi", {"avx512f", "avx512bw", "avx512vl", "avx512dq", "avx512_bf16", "avx512vbmi"}),
    ("avx512_bf16", "_avx512_bf16", {"avx512f", "avx512bw", "avx512vl", "avx512dq", "avx512_bf16"}),
    ("avx512_vnni", "_avx512_vnni", {"avx512f", "avx512bw", "avx512vl", "avx512dq", "avx512_vnni"}),
    ("avx512_base", "_avx512_base", {"avx512f", "avx512bw", "avx512vl", "avx512dq"}),
    ("avx2", "_avx2", {"avx2", "fma"}),
    ("scalar", "_scalar", set()),
]


def _cpu_flags() -> set[str]:
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("flags"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        return set()
    return set()


def _variant_dir() -> str:
    """Directory where variant .so files live.

    Prefers the in-package ``build/`` dir (wheel layout: the native libraries are
    bundled inside ``xiaotu_moe/build/`` so a single wheel is self-contained),
    falling back to the repo-level ``../build`` sibling (development layout,
    ``build_variants.sh`` output) for source checkouts.
    """
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in (os.path.join(pkg_dir, "build"),
                      os.path.join(pkg_dir, "..", "build")):
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(os.path.join(pkg_dir, "..", "build"))


def _available_variants(build_dir: str) -> set[str]:
    """Return set of module suffixes present as .so for this interpreter ABI."""
    suffixes = set()
    for fname in os.listdir(build_dir):
        if fname.startswith("_xiaotu_moe_C") and fname.endswith(".so"):
            # name is _xiaotu_moe_C<SUFFIX>.cpython-<ver>...
            rest = fname[len("_xiaotu_moe_C"):]
            # strip the ABI part: everything after the first '.'
            suffix = rest.split(".", 1)[0]
            suffixes.add(suffix)
    return suffixes


def _best_variant(flags: set[str], available: set[str]) -> str:
    for _name, suffix, required in _LADDER:
        if suffix in available and required.issubset(flags):
            return suffix
    raise RuntimeError(
        "xiaotu-moe: no compatible ISA variant found. Built variants present: "
        f"{sorted(available)}; host requires one of the ladder sets. "
        "Run scripts/build_variants.sh for the target interpreter."
    )


_module = None
_chosen = None


def choose_variant(force: str | None = None) -> str:
    """Return the module suffix that will be imported (best-by-CPU unless forced).

    `force` may be one of the ladder names (e.g. "avx2") or a full module
    suffix (e.g. "_avx2") to override CPU detection.

    `XIAOTU_MOE_VARIANT`（如 `avx512_bf16`）在没有显式 `force` 时优先，
    用于**按 CPU 型号选分支**与 A/B（见 §566）：

    * 变体的 pybind11 类型是**全局注册**的，同一进程**不能**加载两个变体
      （`ImportError: generic_type: type "MOEConfigV2" is already registered!`），
      所以 A/B 必须**一变体一进程** —— 只能在导入前用环境变量选，不能在进程内切换。
    """
    global _chosen
    if force is None:
        force = os.environ.get("XIAOTU_MOE_VARIANT") or None
    if force is not None:
        suffix = force if force.startswith("_") else f"_{force}"
        build_dir = _variant_dir()
        av = _available_variants(build_dir)
        if suffix not in av:
            raise RuntimeError(
                f"xiaotu-moe: forced variant '{suffix}' not built. "
                f"Built: {sorted(av)}"
            )
        _chosen = suffix
        return suffix
    flags = _cpu_flags()
    build_dir = _variant_dir()
    av = _available_variants(build_dir)
    _chosen = _best_variant(flags, av)
    return _chosen


def load(force: str | None = None):
    """Import and return the selected xiaotu-moe native module (cached)."""
    global _module, _chosen
    if _module is not None:
        return _module
    suffix = choose_variant(force)
    build_dir = _variant_dir()
    if build_dir not in sys.path:
        sys.path.insert(0, build_dir)
    mod_name = f"_xiaotu_moe_C{suffix}"
    _module = importlib.import_module(mod_name)
    return _module


def __getattr__(name):
    # Make `import xiaotu_moe as m; m.MOEBF16`, m.MOEV2 etc. work transparently
    # by forwarding attribute access to the loaded module.
    mod = load()
    if hasattr(mod, name):
        return getattr(mod, name)
    raise AttributeError(f"xiaotu_moe has no attribute {name!r}")


def __dir__():
    return sorted(set(globals().keys()) | set(dir(load())))
