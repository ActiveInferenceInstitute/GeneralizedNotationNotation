#!/usr/bin/env python3
"""Deterministic MAJ-05 benchmark: audit the ``validate_gnn*`` public surface.

Scans ``src/gnn`` for every definition (module function, method, nested
function, alias assignment, or import alias) whose name starts with
``validate_gnn`` and classifies each definition as:

- **canonical**: listed under ``canonical`` in
  ``scripts/validate_surface_manifest.json``;
- **deprecated alias**: listed under ``deprecated`` AND structurally a pure
  deprecation forward (emits ``DeprecationWarning`` and forwards to the
  mapped canonical name);
- **non-canonical**: everything else (the primary metric).

Manifest entries are ``"<file-suffix>:<name>"`` where ``<file-suffix>`` is the
path relative to ``src/gnn`` (e.g. ``"validation/simple.py:validate_gnn_file"``);
they identify a single definition robustly against line shifts.

Additional metrics: internal call sites (``src/`` + ``tests/``) and live-doc
references for every name that has no canonical definition anywhere (names
that are canonical in one module stay uncounted even when a deprecated alias
of the same name exists in another module).

Fully deterministic: AST + regex over the working tree; no network, no clock,
no randomness. Prints ``METRIC <name>=<value>`` lines on stdout and a
human-readable classification table on stderr. Exits 0 on a clean run, 1 on
inconsistent manifest/tree state.

Usage: uv run --extra dev python scripts/audit_validate_surface.py
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = ROOT / "src" / "gnn"
TESTS_DIR = ROOT / "tests"
MANIFEST_PATH = ROOT / "scripts" / "validate_surface_manifest.json"
NAME_RE = re.compile(r"^validate_gnn")

SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "output", "build", "dist",
    "__pycache__", ".ruff_cache", ".mypy_cache", ".pytest_cache",
    "site-packages", ".tox", ".eggs",
}
# Historical ledgers and append-only worker logs keep old names on purpose.
HISTORICAL_DOC_FILES = {"CHANGELOG.md", "VERSION_MAP.md"}
HISTORICAL_DOC_PREFIXES = ("docs/development/fleet-logs/",)


@dataclass
class SurfaceDef:
    name: str
    file: str  # relative to PKG_DIR (posix)
    line: int
    kind: str  # "function" | "method" | "nested" | "assign" | "import-alias"
    forward_target: str | None
    warns: bool

    @property
    def key(self) -> str:
        return f"{self.file}:{self.name}"


def iter_py_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def call_target_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def classify_alias_body(func: ast.AST) -> tuple[str | None, bool]:
    """Return (forward_target, warns) for a deprecation-forwarding body.

    Valid body: optional docstring, then any number of
    ``warnings.warn(..., DeprecationWarning)`` calls and forwarding calls to a
    single target (``return target(...)`` or ``target(...)``), optionally a
    ``pass``. Anything else is not a pure alias.
    """
    body = list(getattr(func, "body", []))
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    target: str | None = None
    warns = False
    for stmt in body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            name = call_target_name(call.func)
            if name == "warn" and "DeprecationWarning" in ast.unparse(stmt):
                warns = True
                continue
            if target is None and name is not None:
                target = name
                continue
            return None, False
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
            name = call_target_name(stmt.value.func)
            if target is None and name is not None:
                target = name
                continue
            return None, False
        if isinstance(stmt, ast.Pass):
            continue
        return None, False
    return target, warns


def collect_defs(tree: ast.Module, rel_file: str) -> list[SurfaceDef]:
    defs: list[SurfaceDef] = []

    def visit(node: ast.AST, cls: str | None, fn: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if NAME_RE.match(child.name):
                    kind = "method" if cls else ("nested" if fn else "function")
                    target, warns = classify_alias_body(child)
                    defs.append(SurfaceDef(child.name, rel_file, child.lineno, kind, target, warns))
                visit(child, None, child.name)
            elif isinstance(child, ast.ClassDef):
                visit(child, child.name, fn)
            elif isinstance(child, ast.Assign):
                for t in child.targets:
                    if isinstance(t, ast.Name) and NAME_RE.match(t.id):
                        val = child.value
                        tgt = call_target_name(val)
                        defs.append(SurfaceDef(t.id, rel_file, child.lineno, "assign", tgt, False))
            elif isinstance(child, ast.AnnAssign):
                t = child.target
                if isinstance(t, ast.Name) and NAME_RE.match(t.id) and child.value is not None:
                    tgt = call_target_name(child.value)
                    defs.append(SurfaceDef(t.id, rel_file, child.lineno, "assign", tgt, False))
            elif isinstance(child, ast.ImportFrom):
                for a in child.names:
                    if a.asname and NAME_RE.match(a.asname):
                        defs.append(
                            SurfaceDef(a.asname, rel_file, child.lineno, "import-alias", a.name, False)
                        )
            else:
                visit(child, cls, fn)

    visit(tree, None, None)
    return defs


def load_manifest() -> tuple[set[str], dict[str, str]]:
    if not MANIFEST_PATH.exists():
        return set(), {}
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    canonical = set(data.get("canonical", []))
    deprecated = {str(k): str(v) for k, v in data.get("deprecated", {}).items()}
    for old, new in deprecated.items():
        if new not in canonical:
            raise SystemExit(f"manifest: deprecated {old!r} maps to non-canonical {new!r}")
    return canonical, deprecated


def count_call_sites(files: list[Path], counted_names: set[str]) -> dict[str, int]:
    counts = {n: 0 for n in sorted(counted_names)}
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError) as exc:
            raise SystemExit(f"unparseable file {path}: {exc}") from exc
        import_aliases: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for a in node.names:
                    if a.asname:
                        import_aliases[a.asname] = a.name
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name):
                cand = import_aliases.get(func.id, func.id)
            elif isinstance(func, ast.Attribute):
                cand = func.attr
            else:
                continue
            if cand in counts:
                counts[cand] += 1
    return counts


def iter_doc_files() -> Iterator[Path]:
    readme = ROOT / "README.md"
    if readme.exists():
        yield readme
    docs = ROOT / "docs"
    if docs.is_dir():
        for path in sorted(docs.rglob("*.md")):
            rel = path.relative_to(ROOT).as_posix()
            if path.name in HISTORICAL_DOC_FILES or rel.startswith(HISTORICAL_DOC_PREFIXES):
                continue
            yield path
    for pattern in ("SKILL.md", "AGENTS.md"):
        for path in sorted(ROOT.rglob(pattern)):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            yield path


def count_doc_refs(names: set[str]) -> dict[str, int]:
    counts = {n: 0 for n in sorted(names)}
    if not names:
        return counts
    patterns = {n: re.compile(rf"\b{re.escape(n)}\b") for n in names}
    for path in iter_doc_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name, pattern in patterns.items():
            counts[name] += len(pattern.findall(text))
    return counts


def main() -> int:
    canonical_keys, deprecated_map = load_manifest()

    all_defs: list[SurfaceDef] = []
    for path in iter_py_files(PKG_DIR):
        rel = path.relative_to(PKG_DIR).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError) as exc:
            raise SystemExit(f"unparseable file {path}: {exc}") from exc
        all_defs.extend(collect_defs(tree, rel))

    defs_by_key = {d.key: d for d in all_defs}
    for key in sorted(canonical_keys):
        if key not in defs_by_key:
            raise SystemExit(f"manifest: canonical {key!r} has no definition in src/gnn")
    for old in sorted(deprecated_map):
        if old not in defs_by_key:
            raise SystemExit(f"manifest: deprecated {old!r} has no definition in src/gnn")

    canonical_names = {key.split(":", 1)[1] for key in canonical_keys}
    deprecated_keys = set(deprecated_map)

    aliases_ok = 0
    aliases_bad = 0
    noncanonical: list[SurfaceDef] = []
    for d in all_defs:
        if d.key in canonical_keys:
            continue
        if d.key in deprecated_keys:
            target_name = deprecated_map[d.key].split(":", 1)[1]
            if d.warns and d.forward_target == target_name:
                aliases_ok += 1
                continue
            aliases_bad += 1
            noncanonical.append(d)
            continue
        noncanonical.append(d)

    noncanonical_names = {d.name for d in noncanonical}
    counted_names = noncanonical_names - canonical_names

    src_files = list(iter_py_files(PKG_DIR))
    test_files = list(iter_py_files(TESTS_DIR)) if TESTS_DIR.is_dir() else []
    caller_counts = count_call_sites(src_files + test_files, counted_names)
    doc_counts = count_doc_refs(counted_names)

    total_noncanonical = len(noncanonical)
    print(f"METRIC validate_noncanonical_defs={total_noncanonical}")
    print(f"METRIC validate_defs_total={len(all_defs)}")
    print(f"METRIC validate_aliases_ok={aliases_ok}")
    print(f"METRIC validate_aliases_bad={aliases_bad}")
    print(f"METRIC internal_callers_noncanonical={sum(caller_counts.values())}")
    print(f"METRIC doc_refs_noncanonical={sum(doc_counts.values())}")

    print("classification (stderr):", file=sys.stderr)
    for d in all_defs:
        state = "canonical"
        if d in noncanonical:
            state = "ALIAS-OK" if (d.key in deprecated_keys and d.warns and d.forward_target) else "NON-CANONICAL"
        elif d.key in deprecated_keys:
            state = "alias-ok"
        print(f"  {state:<14} {d.kind:<12} {d.key} @ line {d.line}", file=sys.stderr)
    if counted_names:
        print("internal callers of non-canonical names (src+tests):", file=sys.stderr)
        for name, n in caller_counts.items():
            if n:
                print(f"  {name}: {n}", file=sys.stderr)
        print("live-doc references of non-canonical names:", file=sys.stderr)
        for name, n in doc_counts.items():
            if n:
                print(f"  {name}: {n}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
