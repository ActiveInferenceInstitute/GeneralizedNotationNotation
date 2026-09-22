#!/usr/bin/env python3
"""Fail when pyproject optional-dependency groups lose their consumers.

Two hygiene checks over ``pyproject.toml``:

Extras consumers. Every group in ``[project.optional-dependencies]`` except
the aggregation group ``all`` and the dev group must have at least one
consumer. Consumer evidence is any import statement (top-level or deferred -
the repo's guarded-probe convention imports optional backends lazily) in
``src/gnn``, ``tests``, or ``scripts``, or a string literal in
``src/gnn/**/*.py`` naming the distribution's import name: dependency
tables, availability probes such as
``importlib.util.find_spec("tensorflow")``, subprocess dispatch tables, and
generated-script templates (the bnlearn generator emits ``import bnlearn``).
Distribution names resolve to import names via
``importlib.metadata.packages_distributions()`` plus a static fallback for
groups a dev-only environment never installs (``scikit-learn`` ->
``sklearn``, ``jupyter-server`` -> ``jupyter_server``, ``cython`` ->
``Cython``). A group with zero consumers and not in ``EXEMPT_EXTRA`` fails
the gate.

``EXEMPT_EXTRA = {"research"}``: pyproject documents the group as
user-facing tooling ("research group targets non-developer end-users who
don't install the dev group"), so no core-caller intent exists for it.

Mypy override staleness. ``[tool.mypy.overrides]`` module patterns that no
``src/gnn`` import references are stale configuration. A module counts as
referenced when an AST import anywhere in ``src/gnn`` names it, or a string
literal is passed directly to dynamic-import machinery
(``import_module`` / ``find_spec`` / ``__import__`` / ``load_module``) in
``src/gnn`` - availability-probe tables that feed
``importlib.import_module`` keep modules such as tensorflow proven this
way. The stale count is ratcheted via
``scripts/dep_hygiene_caps.json["stale_mypy_overrides"]``; removing stale
override entries and then lowering the cap is the deliberate loop.

Fully deterministic: AST + string literals over the working tree plus
locally installed distribution metadata; no network, no clock, no
randomness.

Usage: uv run --extra dev python scripts/check_dep_hygiene.py
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import tomllib
from functools import lru_cache
from importlib.metadata import packages_distributions
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CAP_FILE = ROOT / "scripts" / "dep_hygiene_caps.json"

# Aggregation and dev groups are excluded from the consumer requirement.
EXTRAS_EXCLUDED_GROUPS = frozenset({"all", "dev"})
# Groups whose pyproject comment documents no-core-caller (user-facing)
# intent; orphan status inside them is expected, not a gate failure.
EXEMPT_EXTRA = frozenset({"research"})

# Static fallback for distributions a dev-only venv never installs, so the
# import-name resolution does not depend on the running environment.
STATIC_DIST_IMPORT_NAMES: dict[str, set[str]] = {
    "scikit-learn": {"sklearn"},
    "jupyter-server": {"jupyter_server"},
    "cython": {"Cython"},
}

# Attribute/function names whose string argument is a dynamic-import call.
DYNAMIC_IMPORT_CALL_NAMES = frozenset(
    {"import_module", "find_spec", "__import__", "load_module", "require"}
)


def load_cap() -> int:
    """Read the mypy-staleness ratchet; a broken cap file is a gate error."""
    if not CAP_FILE.exists():
        raise SystemExit(
            f"check_dep_hygiene: missing cap file {CAP_FILE.relative_to(ROOT)} "
            '- create it as {"stale_mypy_overrides": <int>}.'
        )
    try:
        cap = int(
            json.loads(CAP_FILE.read_text(encoding="utf-8"))["stale_mypy_overrides"]
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(
            f"check_dep_hygiene: unreadable cap file "
            f"{CAP_FILE.relative_to(ROOT)} ({exc}) - expected "
            '{"stale_mypy_overrides": <int>}.'
        ) from exc
    if cap < 0:
        raise SystemExit(
            f"check_dep_hygiene: negative cap in {CAP_FILE.relative_to(ROOT)}."
        )
    return cap


@lru_cache(maxsize=1)
def installed_import_names() -> dict[str, set[str]]:
    """Invert packages_distributions(): distribution name -> import names."""
    inverted: dict[str, set[str]] = {}
    for import_name, dists in packages_distributions().items():
        for dist in dists:
            inverted.setdefault(dist, set()).add(import_name)
    return inverted


def import_names_for(dist: str) -> set[str]:
    """Import names a distribution is importable under (static + installed)."""
    names: set[str] = set(STATIC_DIST_IMPORT_NAMES.get(dist, set()))
    names.add(dist.replace("-", "_"))
    names.add(dist)
    names |= installed_import_names().get(dist, set())
    return names


class ImportRoots(ast.NodeVisitor):
    """Collect the root module of every import anywhere in a file."""

    def __init__(self) -> None:
        self.roots: set[str] = set()
        self.dynamic_literals: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.roots.add(alias.name.split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.level == 0:
            self.roots.add(node.module.split(".")[0])
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = getattr(func, "attr", getattr(func, "id", ""))
        if name in DYNAMIC_IMPORT_CALL_NAMES:
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    self.dynamic_literals.add(arg.value.split(".")[0])
        self.generic_visit(node)


def word_pattern(name: str) -> re.Pattern[str]:
    """Standalone-word pattern for an import name (no identifier bleed)."""
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])")


def string_constants(tree: ast.AST) -> list[str]:
    """Every string constant in a module, including f-string parts."""
    constants: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            constants.append(node.value)
        elif isinstance(node, ast.JoinedStr):
            constants.extend(
                v.value
                for v in node.values
                if isinstance(v, ast.Constant) and isinstance(v.value, str)
            )
    return constants


def scan_python_tree() -> tuple[set[str], set[str], set[str], dict[str, list[str]]]:
    """AST import roots and src/gnn strings, with per-scope separation.

    Returns (gnn_import_roots, union_import_roots, gnn_dynamic_literals,
    strings_by_file). The union covers src/gnn + tests + scripts (extras
    consumer evidence); the gnn-scoped sets back the mypy-staleness metric,
    which measures src/gnn references only. strings_by_file maps
    repo-relative src/gnn paths to their string constants.
    """
    gnn_import_roots: set[str] = set()
    union_import_roots: set[str] = set()
    gnn_dynamic_literals: set[str] = set()
    strings_by_file: dict[str, list[str]] = {}

    for base in (ROOT / "src" / "gnn", ROOT / "tests", ROOT / "scripts"):
        for path in sorted(base.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            collector = ImportRoots()
            collector.visit(tree)
            union_import_roots |= collector.roots
            if base.name == "gnn":  # mypy staleness + strings are gnn-scoped
                gnn_import_roots |= collector.roots
                gnn_dynamic_literals |= collector.dynamic_literals
                strings_by_file[path.relative_to(ROOT).as_posix()] = string_constants(
                    tree
                )
    return (
        gnn_import_roots,
        union_import_roots,
        gnn_dynamic_literals,
        strings_by_file,
    )


def string_hit(
    names: set[str], strings_by_file: dict[str, list[str]]
) -> tuple[str, str] | None:
    """First (file, import-name) whose strings contain a name as a word."""
    patterns = {name: word_pattern(name) for name in names}
    for rel in sorted(strings_by_file):
        for name, pattern in patterns.items():
            if any(pattern.search(text) for text in strings_by_file[rel]):
                return rel, name
    return None


def extras_verdicts(
    extras: dict[str, list[str]],
    import_roots: set[str],
    strings_by_file: dict[str, list[str]],
) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Per-extra consumer verdicts; returns (verdicts, orphan_groups)."""
    verdicts: dict[str, dict[str, str]] = {}
    orphan_groups: list[str] = []
    for group in sorted(extras):
        if group in EXTRAS_EXCLUDED_GROUPS:
            continue
        group_verdicts: dict[str, str] = {}
        for requirement in extras[group]:
            dist = re.split(r"[<>=!~;\[ ]", requirement)[0]
            if not dist:
                continue
            names = import_names_for(dist)
            hits = sorted(names & import_roots)
            if hits:
                group_verdicts[dist] = f"import ({hits[0]})"
                continue
            hit = string_hit(names, strings_by_file)
            if hit is not None:
                group_verdicts[dist] = f"string literal ({hit[0]}: {hit[1]})"
            else:
                group_verdicts[dist] = "no consumer found"
        verdicts[group] = group_verdicts
        if all(v == "no consumer found" for v in group_verdicts.values()):
            orphan_groups.append(group)
    return verdicts, orphan_groups


def mypy_override_staleness(
    overrides_modules: list[str],
    import_roots: set[str],
    dynamic_literals: set[str],
) -> list[str]:
    """Mypy override modules no src/gnn import or dynamic-import literal names."""
    stale: list[str] = []
    for module in overrides_modules:
        base = module.removesuffix(".*").removesuffix(".*").rstrip(".")
        if not base or base.startswith("*"):
            continue
        if base in import_roots or base in dynamic_literals:
            continue
        stale.append(base)
    return sorted(stale)


def main() -> int:
    """Run both hygiene checks; exit 1 on orphan extras or a stale cap breach."""
    parser = argparse.ArgumentParser(
        description=(__doc__ or "Dependency hygiene gate").splitlines()[0]
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Compatibility flag; hygiene failures fail by default.",
    )
    args = parser.parse_args()
    del args  # the gate fails on any failure, with or without --strict

    cap = load_cap()
    try:
        pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        print(f"check_dep_hygiene: unreadable pyproject.toml ({exc}).")
        return 1
    extras = pyproject.get("project", {}).get("optional-dependencies", {})
    overrides_modules = [
        module
        for entry in pyproject.get("tool", {}).get("mypy", {}).get("overrides", [])
        for module in entry.get("module", [])
    ]

    (
        gnn_import_roots,
        union_import_roots,
        gnn_dynamic_literals,
        strings_by_file,
    ) = scan_python_tree()
    verdicts, orphan_groups = extras_verdicts(
        extras, union_import_roots, strings_by_file
    )

    print(
        "check_dep_hygiene: extras consumer verdicts "
        f"({len(verdicts)} gated groups; exempt: {sorted(EXEMPT_EXTRA)}):"
    )
    for group, group_verdicts in verdicts.items():
        marker = "ORPHAN" if group in orphan_groups else "consumed"
        detail = "; ".join(f"{dist} -> {v}" for dist, v in group_verdicts.items())
        print(f"  {group}: {marker} [{detail}]")

    failed = False
    non_exempt_orphans = [g for g in orphan_groups if g not in EXEMPT_EXTRA]
    for group in non_exempt_orphans:
        print(
            f"check_dep_hygiene: extra '{group}' has zero consumers and is "
            "not exempt - add a consumer, remove the group, or document "
            "no-core-caller intent in its pyproject comment and add it to "
            "EXEMPT_EXTRA."
        )
        failed = True
    for group in orphan_groups:
        if group in EXEMPT_EXTRA:
            detail = "; ".join(
                f"{dist}"
                for dist, v in verdicts[group].items()
                if v == "no consumer found"
            )
            print(
                f"check_dep_hygiene: exempt extra '{group}' orphans (allowed): "
                f"{detail}."
            )

    stale = mypy_override_staleness(
        overrides_modules, gnn_import_roots, gnn_dynamic_literals
    )
    print(
        f"check_dep_hygiene: {len(stale)} stale mypy override(s) of "
        f"{len(overrides_modules)} entries (cap {cap})."
    )
    if len(stale) > cap:
        for module in stale:
            print(f"  {module}")
        print(
            "Remove the stale [tool.mypy.overrides] entries, then lower "
            "stale_mypy_overrides in "
            f"{CAP_FILE.relative_to(ROOT)} to the new measured value."
        )
        failed = True
    elif len(stale) < cap:
        print(
            f"  note: stale_mypy_overrides cap can be lowered to {len(stale)} "
            f"in {CAP_FILE.relative_to(ROOT)} after this run."
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
