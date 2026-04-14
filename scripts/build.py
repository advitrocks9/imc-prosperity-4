"""Build script: inlines strategies/ into a single submission file.

1. Reads trader.py
2. Resolves strategy imports (topological sort)
3. Strips internal strategy imports from each module
4. Concatenates: datamodel → strategy bodies → trader body
5. Outputs build/submission.py
6. Validates: syntax, size, banned imports, class check, smoke test
"""

import ast
import importlib
import importlib.util
import os
import pathlib
import py_compile
import re
import sys
import textwrap

ROOT = pathlib.Path(__file__).resolve().parent.parent
STRATEGIES_DIR = ROOT / "strategies"
TRADER_FILE = ROOT / "trader.py"
DATAMODEL_FILE = ROOT / "datamodel.py"
OUTPUT_FILE = ROOT / "build" / "submission.py"

ALLOWED_IMPORTS = {
    "math", "json", "collections", "typing", "string", "copy",
    "itertools", "functools", "operator", "statistics",
    "datamodel", "numpy", "pandas", "dataclasses", "jsonpickle",
}

STRATEGY_IMPORT_RE = re.compile(
    r"^(?:from\s+strategies(?:\.\w+)?\s+import\s+.+|import\s+strategies(?:\.\w+)?)\s*$",
    re.MULTILINE,
)


def find_strategy_imports(source: str) -> list[str]:
    """Extract strategy module names from import statements."""
    modules = []
    for line in source.splitlines():
        m = re.match(r"from\s+strategies\.(\w+)\s+import\s+", line)
        if m:
            modules.append(m.group(1))
    return modules


def strip_strategy_imports(source: str) -> str:
    """Remove all `from strategies.X import ...` lines."""
    return STRATEGY_IMPORT_RE.sub("", source)


def strip_datamodel_imports(source: str) -> str:
    """Remove `from datamodel import ...` lines (will be at top of output)."""
    return re.sub(r"^from\s+datamodel\s+import\s+.+$", "", source, flags=re.MULTILINE)


def read_module(name: str) -> str:
    """Read a strategy module's source."""
    path = STRATEGIES_DIR / f"{name}.py"
    return path.read_text()


def topological_sort(modules: list[str]) -> list[str]:
    """Sort strategy modules by dependency order."""
    # Build dependency graph
    deps: dict[str, list[str]] = {}
    for mod in modules:
        source = read_module(mod)
        mod_deps = [d for d in find_strategy_imports(source) if d != mod]
        deps[mod] = mod_deps
        # Add transitive deps
        for d in mod_deps:
            if d not in modules:
                modules.append(d)
                deps.setdefault(d, [])

    # Kahn's algorithm
    in_degree = {m: 0 for m in modules}
    for m, d_list in deps.items():
        for d in d_list:
            if d in in_degree:
                in_degree[d] = in_degree.get(d, 0)

    # Recompute in-degrees
    in_degree = {m: 0 for m in modules}
    for m, d_list in deps.items():
        for d in d_list:
            if d in in_degree:
                in_degree[m] = in_degree[m]  # dep must come before m

    # Simple: just put deps before dependents
    visited: set[str] = set()
    order: list[str] = []

    def visit(m: str) -> None:
        if m in visited:
            return
        visited.add(m)
        for d in deps.get(m, []):
            visit(d)
        order.append(m)

    for m in modules:
        visit(m)
    return order


def collect_datamodel_imports(source: str) -> list[str]:
    """Collect all `from datamodel import X, Y` names."""
    names: set[str] = set()
    for line in source.splitlines():
        m = re.match(r"from\s+datamodel\s+import\s+(.+)", line)
        if m:
            for name in m.group(1).split(","):
                name = name.strip()
                if name and name != "*":
                    names.add(name)
    return sorted(names)


def build() -> str:
    """Build the submission file and return its content."""
    trader_source = TRADER_FILE.read_text()
    datamodel_source = DATAMODEL_FILE.read_text()

    # Find which strategy modules trader.py imports
    strategy_modules = find_strategy_imports(trader_source)

    if strategy_modules:
        sorted_modules = topological_sort(strategy_modules)
    else:
        sorted_modules = []

    # Collect all datamodel imports from all sources
    all_dm_imports: set[str] = set()
    for name in collect_datamodel_imports(trader_source):
        all_dm_imports.add(name)
    for mod in sorted_modules:
        for name in collect_datamodel_imports(read_module(mod)):
            all_dm_imports.add(name)

    # Build output
    parts: list[str] = []

    # 1. Datamodel - inline the full file
    parts.append("# === DATAMODEL ===\n")
    parts.append(datamodel_source)
    parts.append("\n")

    # 2. Strategy modules (stripped of internal imports)
    for mod in sorted_modules:
        source = read_module(mod)
        source = strip_strategy_imports(source)
        source = strip_datamodel_imports(source)
        parts.append(f"# === STRATEGY: {mod} ===\n")
        parts.append(source.strip())
        parts.append("\n\n")

    # 3. Trader body (stripped of strategy + datamodel imports)
    trader_body = strip_strategy_imports(trader_source)
    trader_body = strip_datamodel_imports(trader_body)
    parts.append("# === TRADER ===\n")
    parts.append(trader_body.strip())
    parts.append("\n")

    return "\n".join(parts)


def validate(content: str, output_path: pathlib.Path) -> bool:
    """Run all validation checks. Returns True if all pass."""
    all_passed = True

    # 1. Syntax check
    try:
        py_compile.compile(str(output_path), doraise=True)
        print("  [PASS] Syntax check")
    except py_compile.PyCompileError as e:
        print(f"  [FAIL] Syntax check: {e}")
        all_passed = False

    # 2. Size check
    size = output_path.stat().st_size
    if size < 100_000:
        print(f"  [PASS] Size check: {size:,} bytes (< 100KB)")
    else:
        print(f"  [FAIL] Size check: {size:,} bytes (>= 100KB)")
        all_passed = False

    # 3. Banned imports
    tree = ast.parse(content)
    banned = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in ALLOWED_IMPORTS:
                    banned.append(top)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top not in ALLOWED_IMPORTS:
                    banned.append(top)
    if not banned:
        print("  [PASS] Import check")
    else:
        print(f"  [FAIL] Banned imports: {', '.join(set(banned))}")
        all_passed = False

    # 4. Class check
    has_trader = False
    has_run = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Trader":
            has_trader = True
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name == "run":
                        # Check first param after self is 'state'
                        args = item.args.args
                        if len(args) >= 2 and args[1].arg == "state":
                            has_run = True
    if has_trader and has_run:
        print("  [PASS] Class check: Trader.run(self, state) found")
    else:
        print(f"  [FAIL] Class check: Trader={'found' if has_trader else 'missing'}, run={'found' if has_run else 'missing'}")
        all_passed = False

    # 5. Smoke test
    try:
        # Add output dir to path temporarily
        sys.path.insert(0, str(output_path.parent))
        spec = importlib.util.spec_from_file_location("submission", str(output_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        trader = mod.Trader()
        print("  [PASS] Smoke test: Trader() instantiated")
    except Exception as e:
        print(f"  [FAIL] Smoke test: {e}")
        all_passed = False
    finally:
        sys.path.pop(0)

    return all_passed


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build single-file submission")
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output filename (placed in build/). e.g. -o v2_wider_spread",
    )
    args = parser.parse_args()

    if args.output:
        name = args.output if args.output.endswith(".py") else f"{args.output}.py"
        output_path = ROOT / "build" / name
    else:
        output_path = OUTPUT_FILE

    print(f"=== Building {output_path.name} ===\n")

    content = build()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)

    lines = content.count("\n")
    size = output_path.stat().st_size

    strategy_modules = find_strategy_imports(TRADER_FILE.read_text())

    print(f"Output: {output_path}")
    print(f"Size: {size:,} bytes | Lines: {lines}")
    print(f"Modules inlined: {len(strategy_modules)}")
    print()
    print("=== Validation ===\n")

    passed = validate(content, output_path)
    print()
    if passed:
        print("All checks PASSED")
    else:
        print("Some checks FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
