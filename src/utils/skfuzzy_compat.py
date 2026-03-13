"""
Compatibility shim for scikit-fuzzy on Python 3.12+.

The :mod:`imp` module was removed in Python 3.12, but scikit-fuzzy 0.4.2 still
imports it.  This shim provides a minimal stand‑in implementation that is
good enough for scikit-fuzzy's internal usage patterns.

Usage:
    Import this module **before** any import of :mod:`skfuzzy`, for example::

        import src.utils.skfuzzy_compat  # noqa: F401
        import skfuzzy  # now safe on Python 3.12+
"""

import importlib
import importlib.machinery  # noqa: F401  (parity with legacy imp API)
import importlib.util
import sys
import types

if sys.version_info >= (3, 12) and "imp" not in sys.modules:
    imp_shim = types.ModuleType("imp")

    # Very small subset of the old imp API that scikit-fuzzy may rely on.
    imp_shim.load_source = lambda name, path: importlib.import_module(name)
    imp_shim.find_module = lambda name, path=None: (None, None, None)

    def _load_module(name, file, pathname, description):  # type: ignore[override]
        spec = importlib.util.spec_from_file_location(name, pathname)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader is not None
        spec.loader.exec_module(module)
        return module

    imp_shim.load_module = _load_module  # type: ignore[assignment]
    imp_shim.PKG_DIRECTORY = 5
    imp_shim.PY_SOURCE = 1
    imp_shim.PY_COMPILED = 2
    imp_shim.C_EXTENSION = 3
    imp_shim.get_suffixes = lambda: [(".py", "r", 1)]
    imp_shim.acquire_lock = lambda: None
    imp_shim.release_lock = lambda: None
    imp_shim.NullImporter = type("NullImporter", (), {})

    sys.modules["imp"] = imp_shim
