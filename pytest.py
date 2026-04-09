import importlib.util
import math
import os
import sys
import traceback
from typing import Any


class Approx:
    def __init__(self, expected: float, abs: float | None = None, rel: float | None = None) -> None:
        self.expected = expected
        self.abs = 1e-12 if abs is None and rel is None else abs
        self.rel = 1e-12 if abs is None and rel is None else rel

    def __eq__(self, actual: Any) -> bool:
        if not isinstance(actual, (int, float)):
            return False
        tol_abs = 0.0 if self.abs is None else self.abs
        tol_rel = 0.0 if self.rel is None else self.rel * abs(self.expected)
        return math.isclose(actual, self.expected, abs_tol=max(tol_abs, tol_rel), rel_tol=0.0)

    def __repr__(self) -> str:
        return f"Approx(expected={self.expected}, abs={self.abs}, rel={self.rel})"


def approx(expected: float, abs: float | None = None, rel: float | None = None) -> Approx:
    return Approx(expected, abs=abs, rel=rel)


def _load_module(path: str) -> object:
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def console_main() -> int:
    args = [arg for arg in sys.argv[1:] if arg != "-v"]
    paths = [arg for arg in args if not arg.startswith("-")] or ["tests"]
    total = passed = 0
    for path in paths:
        module = _load_module(path) if path.endswith(".py") else _load_module(os.path.join(path, "__init__.py"))
        tests = [(n, getattr(module, n)) for n in dir(module) if n.startswith("test_")]
        for name, fn in tests:
            total += 1
            try:
                fn()
                passed += 1
                print(f"{path}::{name} PASSED")
            except Exception:
                print(f"{path}::{name} FAILED")
                traceback.print_exc()
    print(f"============================= {passed}/{total} passed =============================")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(console_main())
