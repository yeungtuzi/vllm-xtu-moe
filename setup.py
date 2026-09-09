# vllm-xtu-moe wheel build: force correct platform/ABI-specific wheel tags.
#
# The native extension libraries are prebuilt .so files bundled as package data
# in xiaotu_moe/build/ (see pyproject.toml). Because they are plain data (not
# setuptools Extension objects), setuptools would otherwise tag the wheel
# "py3-none-any", which is wrong: the .so are cp312 / x86_64 specific and linked
# against glibc >= 2.34. We override bdist_wheel to emit a properly-tagged
# platform wheel so pip does not install it on incompatible targets.
#
# License: Apache-2.0

import os
import platform

from setuptools import setup  # noqa: F401  (metadata lives in pyproject.toml)

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # pragma: no cover
    _bdist_wheel = None


def _py_version() -> str:
    try:
        import sysconfig
        ext = sysconfig.get_config_var("EXT_SUFFIX") or ""
        import re
        m = re.search(r"cpython-(\d+)", ext)
        if m:
            return f"cp{m.group(1)}"
    except Exception:
        pass
    return "py3"  # fallback


if _bdist_wheel is not None:

    class bdist_wheel(_bdist_wheel):
        """Force cpXY-cpXY-manylinux_2_34_x86_64 tags for the bundled native libs.

        PyPI rejects the generic "linux_x86_64" tag and requires a PEP 600
        "manylinux_*" tag. Our .so link against glibc <= 2.34 (verified via
        objdump/readelf GLIBC_* symbol deps), so manylinux_2_34_x86_64 is the
        correct, minimal platform tag. Building on a glibc >= 2.34 machine
        (Ubuntu 22.04) satisfies its requirements.
        """

        def finalize_options(self):
            super().finalize_options()
            self.root_is_pure = False

        def get_tag(self):
            py = _py_version()
            return (py, py, "manylinux_2_34_x86_64")


cmdclass = {"bdist_wheel": bdist_wheel} if _bdist_wheel is not None else {}

if __name__ == "__main__":
    setup(cmdclass=cmdclass)
