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


def _literal_shape(node: "object") -> "tuple[int, ...] | None":
    """Return the rectangular shape of an AST list literal, or None.

    Ragged literals, non-numeric leaves, and non-list nodes yield None -
    shapes are never guessed.
    """
    import ast

    if not isinstance(node, ast.List):
        return None
    if not node.elts:
        return (0,)
    first = True
    shape: list[int] = []
    for element in node.elts:
        if isinstance(element, ast.List):
            sub = _literal_shape(element)
            if sub is None:
                return None
            if first:
                shape = [len(node.elts), *sub]
                first = False
            else:
                expected = tuple(shape[1:])
                if sub != expected:
                    return None
        elif (
            isinstance(element, (ast.Constant,))
            and isinstance(element.value, (int, float))
            and not isinstance(element.value, bool)
        ):
            if first:
                shape = [len(node.elts)]
                first = False
            elif len(shape) != 1:
                return None
        elif (
            isinstance(element, ast.UnaryOp)
            and isinstance(element.op, ast.USub)
            and isinstance(element.operand, ast.Constant)
            and isinstance(element.operand.value, (int, float))
        ):
            if first:
                shape = [len(node.elts)]
                first = False
            elif len(shape) != 1:
                return None
        else:
            return None
    return tuple(shape)


def _shape_from_array_call(call: "object") -> "tuple[int, ...] | None":
    """Extract the literal shape from ``jnp.array([...])`` / ``torch.tensor([...])``."""
    import ast

    if not isinstance(call, ast.Call) or not call.args:
        return None
    return _literal_shape(call.args[0])


def matrix_shapes(code: str) -> "dict[str, tuple[int, ...]] | None":
    """Extract canonical A/B/C/D matrix shapes from one emitted Python artifact.

    Understands the four maintained emission conventions:

    - pymdp delegated runners: ``A_data = [...]`` plain list literals;
    - jax: ``'A_matrix': jnp.array([...])`` dict-payload calls;
    - pytorch: ``A = torch.tensor([...])`` plus ``B_slices.append(...)``;
    - numpyro: ``A = jnp.array([...])`` plus ``B_slices.append(...)``.

    Returns None when the module uses a star import (conservative skip - the
    same rule as :func:`undefined_names`). Only letters with a clean
    rectangular literal shape are returned; ragged or call-embedded matrices
    are skipped, never guessed.
    """
    import ast

    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    return None

    names = {
        "A_data": "A",
        "B_data": "B",
        "C_data": "C",
        "D_data": "D",
        "A": "A",
        "B": "B",
        "C": "C",
        "D": "D",
    }
    dict_keys = {
        "A_matrix": "A",
        "B_matrix": "B",
        "C_vector": "C",
        "D_vector": "D",
    }
    shapes: dict[str, tuple[int, ...]] = {}
    b_slices: list[tuple[int, ...]] = []

    def _record(letter: str, shape: tuple[int, ...] | None) -> None:
        if shape is not None:
            shapes.setdefault(letter, shape)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = node.value
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                letter = names.get(target.id)
                if letter is None:
                    continue
                if isinstance(value, ast.List):
                    _record(letter, _literal_shape(value))
                elif isinstance(value, ast.Call):
                    _record(letter, _shape_from_array_call(value))
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and key.value in dict_keys
                    and isinstance(value, ast.Call)
                ):
                    _record(dict_keys[key.value], _shape_from_array_call(value))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "B_slices"
            and node.args
        ):
            shape = _shape_from_array_call(node.args[0])
            if shape is not None:
                b_slices.append(shape)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "B" for target in node.targets
        ):
            # ``B = torch.stack(B_slices, dim=2)`` / ``jnp.stack(..., axis=2)``:
            # the final B is each equal slice shape plus the slice-count axis.
            if (
                isinstance(node.value, ast.Call)
                and b_slices
                and all(s == b_slices[0] for s in b_slices)
                and any(
                    isinstance(arg, ast.Name) and arg.id == "B_slices"
                    for arg in node.value.args
                )
            ):
                _record("B", (*b_slices[0], len(b_slices)))
    if b_slices and all(s == b_slices[0] for s in b_slices):
        _record("B", (*b_slices[0], len(b_slices)))
    return shapes


def julia_dimension_constants(code: str) -> dict[str, int]:
    """Extract ``const NUM_STATES/OBSERVATIONS/ACTIONS = <int>`` from a .jl artifact.

    The Julia backends allocate their matrices from these constants and fill
    values at runtime, so dimension parity against the Python backends' literal
    shapes is checked through them. Returns only the constants present.
    """
    import re

    constants: dict[str, int] = {}
    for match in re.finditer(
        r"const\s+NUM_(STATES|OBSERVATIONS|ACTIONS)\s*=\s*(\d+)", code
    ):
        constants[f"NUM_{match.group(1)}"] = int(match.group(2))
    return constants
