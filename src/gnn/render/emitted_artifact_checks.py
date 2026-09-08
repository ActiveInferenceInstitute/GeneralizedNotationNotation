"""Static hygiene checks for emitted render artifacts.

Two conservative, dependency-free checks:

- ``undefined_names`` reports Load-context names that are never bound anywhere
  in the module (imports, defs, classes, assignments, args, except handlers,
  global/nonlocal) and are not builtins or module-context names. Binding
  anywhere counts, so the check is imprecise about scoping but has near-zero
  false positives - the right trade for a deterministic benchmark gate.
- ``first_party_unresolvable_imports`` reports ``gnn.*`` imports whose full
  dotted path does not resolve from the repository. Third-party import
  availability is an environment concern (optional extras are documented);
  first-party imports, however, must always resolve - a stale template
  referencing a renamed module compiles clean and passes every name-based
  check, yet ImportErrors the moment the emitted script runs.

Consumers: ``scripts/bench_render_backends.py`` (corpus x framework
conformance benchmark) and ``tests/render/test_render_contracts.py``.
"""

from __future__ import annotations

# Module-context names that are never bound by statements but always exist.
MODULE_CONTEXT_NAMES = frozenset(
    {"__name__", "__file__", "__doc__", "__package__", "__spec__", "__loader__"}
)


def undefined_names(code: str) -> tuple[list[tuple[str, int]], bool]:
    """Scan one Python artifact for statically-undefined names.

    Returns ``((name, line) findings, star_import_present)``. When the module
    uses a star import the scan is skipped entirely (names cannot be resolved
    without executing the source module), reported via the second element.
    """
    import ast
    import builtins

    tree = ast.parse(code)
    bound: set[str] = set(MODULE_CONTEXT_NAMES)
    star_import = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    star_import = True
                    continue
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
    if star_import:
        return [], True
    findings = sorted(
        {
            (node.id, node.lineno)
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id not in bound
            and not hasattr(builtins, node.id)
        },
        key=lambda item: (item[1], item[0]),
    )
    return findings, False


def first_party_unresolvable_imports(code: str) -> list[tuple[str, int]]:
    """Scan one Python artifact for unresolvable ``gnn.*`` imports.

    Returns ``(module, line)`` pairs for every first-party ``gnn.*`` import
    (``import`` or ``from ... import``) whose full dotted path does not
    resolve via ``importlib.util.find_spec``.
    """
    import ast
    import importlib.util

    tree = ast.parse(code)
    findings: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        modules: list[tuple[str, int]] = []
        if isinstance(node, ast.Import):
            modules = [
                (alias.name, node.lineno)
                for alias in node.names
                if alias.name.split(".")[0] == "gnn"
            ]
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module
            and node.module.split(".")[0] == "gnn"
        ):
            modules = [(node.module, node.lineno)]
        else:
            continue
        for module, lineno in modules:
            try:
                if importlib.util.find_spec(module) is None:
                    findings.append((module, lineno))
            except (ImportError, AttributeError, ValueError):
                findings.append((module, lineno))
    return sorted(findings, key=lambda item: (item[1], item[0]))
