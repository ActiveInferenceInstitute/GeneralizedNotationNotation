"""Tests for the deterministic manuscript-variable producer.

Real objects only: every assertion recomputes the expected value from the live repository
and compares it against :func:`src.manuscript_variables.generate_variables`, so the
test fails if the producer drifts from the source surfaces it claims to read.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

# Import the producer by its canonical top-level name (src/ on path), matching the
# repo convention (`from pipeline.X import ...`). Importing it as `src.manuscript_variables`
# makes mypy (mypy_path=src) resolve the same file under two module names and fail.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from manuscript_variables import (  # noqa: E402
    RepositorySnapshot,
    _capability_clause,
    _registry_specs,
    _render_family_table,
    generate_variables,
    save_variables,
    select_cross_framework_family,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def variables() -> dict[str, str]:
    return generate_variables(_PROJECT_ROOT)


def _registry_flags(snapshot: RepositorySnapshot, flag: str) -> list[str]:
    """Independently parse ``FRAMEWORK_REGISTRY`` for backends whose *flag* is True.

    Parsed here with :mod:`ast` rather than by calling the producer's own
    parser, so an assertion failure means the token really disagrees with the
    registry literal — not that both read the same helper.
    """
    tree = ast.parse(snapshot.read_text("src/render/framework_registry.py"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = (
            [node.target] if isinstance(node, ast.AnnAssign) else list(node.targets)
        )
        if not any(
            isinstance(target, ast.Name) and target.id == "FRAMEWORK_REGISTRY"
            for target in targets
        ):
            continue
        value = node.value
        assert value is not None, "FRAMEWORK_REGISTRY has no assigned value"
        if isinstance(value, ast.Call) and value.args:
            value = value.args[0]
        registry = ast.literal_eval(value)
        return [key for key, spec in registry.items() if spec.get(flag) is True]
    raise AssertionError("FRAMEWORK_REGISTRY literal not found")


@pytest.fixture(scope="module")
def snapshot() -> RepositorySnapshot:
    """The same commit the producer counts.

    Assertions read source surfaces through this snapshot rather than off disk:
    the producer's contract is that every count describes ``GNN_GIT_COMMIT``, so
    comparing against the working tree would make these tests fail for anyone
    with an uncommitted edit instead of catching real drift.
    """
    return RepositorySnapshot(_PROJECT_ROOT)


def test_all_values_are_strings(variables: dict[str, str]) -> None:
    assert variables, "producer returned an empty token map"
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in variables.items())


def test_token_keys_are_injector_compatible(variables: dict[str, str]) -> None:
    # Template injector regex: {{[A-Z][A-Z0-9_]*}}
    key_re = re.compile(r"^[A-Z][A-Z0-9_]*$")
    bad = [k for k in variables if not key_re.match(k)]
    assert not bad, f"keys not matchable by injector regex: {bad}"


def test_minimum_token_coverage(variables: dict[str, str]) -> None:
    # ISC-3: at least 20 distinct tokens.
    assert len(variables) >= 20, f"only {len(variables)} tokens"


def test_determinism(variables: dict[str, str]) -> None:
    # ISC-4: a second invocation yields an identical map.
    again = generate_variables(_PROJECT_ROOT)
    assert again == variables


def test_version_matches_pyproject(
    variables: dict[str, str], snapshot: RepositorySnapshot
) -> None:
    match = re.search(
        r'^\s*version\s*=\s*"([^"]+)"',
        snapshot.read_text("pyproject.toml"),
        re.MULTILINE,
    )
    assert match is not None
    assert variables["GNN_VERSION"] == match.group(1)


def test_step_count_matches_step_modules(
    variables: dict[str, str], snapshot: RepositorySnapshot
) -> None:
    step_modules = [
        rel for rel in snapshot.glob("src", "[0-9]*_*.py") if len(rel.parts) == 2
    ]
    assert variables["GNN_STEP_COUNT"] == str(len(step_modules))
    assert int(variables["GNN_STEP_COUNT"]) >= 1


def test_step_modules_and_packages_are_disjoint_sets(
    variables: dict[str, str], snapshot: RepositorySnapshot
) -> None:
    """GNN_STEP_COUNT and GNN_SRC_PACKAGE_COUNT count sets with no overlap.

    Step modules are top-level ``src/N_*.py`` files; packages are ``src/<name>/``
    directories. No prose may say the step modules "sit inside" the packages.
    """
    step_modules = [
        rel for rel in snapshot.glob("src", "[0-9]*_*.py") if len(rel.parts) == 2
    ]
    packages = {
        rel.parts[1]
        for rel in snapshot.glob("src", "__init__.py")
        if len(rel.parts) == 3 and rel.parts[1] != "tests"
    }
    assert step_modules, "no step modules found"
    assert packages, "no source packages found"
    assert not {rel.name for rel in step_modules} & packages
    assert variables["GNN_SRC_PACKAGE_COUNT"] == str(len(packages))


def test_family_count_matches_manifest(
    variables: dict[str, str], snapshot: RepositorySnapshot
) -> None:
    manifest = json.loads(snapshot.read_text("input/model_family_manifest.json"))
    assert variables["GNN_FAMILY_COUNT"] == str(len(manifest["families"]))


def test_backend_count_matches_registry(
    variables: dict[str, str], snapshot: RepositorySnapshot
) -> None:
    registry = snapshot.read_text("src/render/framework_registry.py")
    names = re.findall(r'"name"\s*:\s*"([^"]+)"', registry)
    assert variables["GNN_BACKEND_COUNT"] == str(len(names))
    assert int(variables["GNN_BACKEND_COUNT"]) >= 2


def test_executable_backend_count_is_the_execution_subset(
    variables: dict[str, str], snapshot: RepositorySnapshot
) -> None:
    """Only backends with ``supports_execution`` execute at Step 12.

    ``GNN_BACKEND_COUNT`` is the registry size and must never label a sentence
    or a figure bar about execution.
    """
    executable = len(_registry_flags(snapshot, "supports_execution"))
    assert variables["GNN_EXECUTABLE_BACKEND_COUNT"] == str(executable)
    assert int(variables["GNN_EXECUTABLE_BACKEND_COUNT"]) < int(
        variables["GNN_BACKEND_COUNT"]
    )
    assert len(variables["GNN_EXECUTABLE_BACKEND_LIST"].split(", ")) == executable


def test_cross_framework_backends_are_all_profiled_by_the_gate(
    variables: dict[str, str], snapshot: RepositorySnapshot
) -> None:
    """The reported cross-framework engines are ones the gate will actually run.

    ``src/pipeline/cross_framework_reliability.py`` raises on any framework
    outside ``MAINTAINED_FRAMEWORKS``, so a declared-but-unprofiled framework
    (``stan``) must not be counted among the reference-comparison engines.
    """
    gate = snapshot.read_text("src/pipeline/cross_framework_reliability.py")
    block = re.search(r"MAINTAINED_FRAMEWORKS = \(([^)]*)\)", gate)
    assert block is not None
    maintained_keys = re.findall(r'"([a-z_]+)"', block.group(1))
    assert variables["GNN_MAINTAINED_FRAMEWORK_COUNT"] == str(len(maintained_keys))

    manifest = json.loads(snapshot.read_text("input/model_family_manifest.json"))
    family = select_cross_framework_family(manifest["families"])
    assert family is not None
    declared = [k.strip() for k in str(family["frameworks"]).split(",") if k.strip()]
    profiled = [k for k in declared if k in maintained_keys]
    assert profiled, "cross-framework family declares nothing the gate profiles"
    assert variables["GNN_CROSS_FRAMEWORK_BACKEND_COUNT"] == str(len(profiled))
    assert variables["GNN_CROSS_FRAMEWORK_DECLARED_BACKEND_COUNT"] == str(len(declared))
    assert len(variables["GNN_CROSS_FRAMEWORK_BACKENDS"].split(", ")) == len(profiled)


def test_manifest_does_not_hand_type_capability_claims() -> None:
    """No manifest description may restate a backend capability in prose.

    The ``discopy``-is-native-on-continuous defect was a free-text capability
    sentence in ``input/model_family_manifest.json`` that contradicted the
    registry's ``supports_continuous`` flag, the acceptance ledger, and
    CLAUDE.md. A family declares which axis it exercises; the split itself is
    generated. Read from the working tree because this is a lint on the
    authored file, not a count.
    """
    manifest = json.loads(
        (_PROJECT_ROOT / "input" / "model_family_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for family in manifest["families"]:
        desc = str(family.get("description", ""))
        assert "Native on" not in desc, (
            f"{family['name']}: capability claim in manifest"
        )
        assert "unsupported" not in desc, (
            f"{family['name']}: capability claim in manifest"
        )
    axes = [f for f in manifest["families"] if f.get("capability_axis")]
    assert axes, "no family declares a capability_axis; the split would vanish"


def test_capability_clause_is_generated_from_the_registry(
    snapshot: RepositorySnapshot,
) -> None:
    """The generated native/unsupported split equals the registry flags."""
    specs = _registry_specs(snapshot)
    native_keys = _registry_flags(snapshot, "supports_continuous")
    clause = _capability_clause("supports_continuous", specs)
    native_part, unsupported_part = clause.split("; ", 1)
    native_names = native_part.removeprefix("Native on ").split(", ")
    unsupported_names = (
        unsupported_part.removeprefix("reported unsupported (not failed) on ")
        .rstrip(".")
        .split(", ")
    )
    assert native_names == [str(specs[k]["name"]) for k in native_keys]
    assert sorted(native_names + unsupported_names) == sorted(
        str(spec["name"]) for spec in specs.values()
    )
    # The registry entry the defect misreported must land on the correct side.
    assert str(specs["discopy"]["name"]) in unsupported_names


def test_family_table_renders_the_split_for_the_declaring_family(
    snapshot: RepositorySnapshot,
) -> None:
    """The rendered family table carries the generated split, not manifest prose.

    Built from the authored manifest and the registry directly, so this holds
    for an edit in progress as well as for the counted commit.
    """
    manifest = json.loads(
        (_PROJECT_ROOT / "input" / "model_family_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    families = manifest["families"]
    declaring = [f for f in families if f.get("capability_axis")]
    assert declaring, "no family declares a capability_axis"
    table = _render_family_table(families, _registry_specs(snapshot))
    for family in declaring:
        row = next(
            line
            for line in table.splitlines()
            if line.startswith(f"| `{family['name']}`")
        )
        assert "Native on" in row
        assert "reported unsupported (not failed) on" in row
    assert "generated from" in table.splitlines()[-1]


def test_mcp_tool_count_matches_audit(
    variables: dict[str, str], snapshot: RepositorySnapshot
) -> None:
    audit = json.loads(snapshot.read_text("src/mcp/audit_report.json"))
    assert variables["GNN_MCP_TOOL_COUNT"] == str(audit["tools_total"])


def test_counts_describe_the_stamped_commit_not_the_working_tree(
    variables: dict[str, str], snapshot: RepositorySnapshot
) -> None:
    """A dirty working tree cannot move a published count.

    The producer's printed contract is that the numbers describe the commit
    ``GNN_GIT_COMMIT`` names. Before this was enforced the PDF shipped 190691
    LOC while the repository's own committed token file said 190943 and neither
    reproduced from the commit the token file stamped.
    """
    # No skip guard: the repo's zero-skip contract (test_zero_skip_contracts.py)
    # forbids one, and a checkout without git metadata cannot honour the
    # producer's reproducibility contract at all — that is a failure, not a skip.
    assert snapshot.from_git, "snapshot is not reading committed blobs"
    assert variables["GNN_GIT_COMMIT"] == snapshot.commit
    assert variables["GNN_GIT_COMMIT"] != "unknown"
    sources = [
        rel
        for rel in snapshot.glob("src", "*.py")
        if Path("src/tests") not in rel.parents
    ]
    snapshot.prefetch(sources)
    total = 0
    for rel in sources:
        text = snapshot.read_text(rel)
        total += text.count("\n") + (0 if not text or text.endswith("\n") else 1)
    assert variables["GNN_SRC_LOC"] == str(total)
    assert variables["GNN_SRC_PY_FILE_COUNT"] == str(len(sources))


def test_tables_are_multiline_markdown(variables: dict[str, str]) -> None:
    for key in ("GNN_STEP_TABLE", "GNN_FAMILY_TABLE", "GNN_BACKEND_TABLE"):
        assert "\n" in variables[key], f"{key} should be a multi-line markdown table"
        assert variables[key].lstrip().startswith("|"), f"{key} should be a pipe table"


def test_license_value_is_resolved(variables: dict[str, str]) -> None:
    # ISC-23 guard: config completion removes the scaffold 'TBD'/empty license.
    assert variables["GNN_LICENSE"] and variables["GNN_LICENSE"] != "TBD"


def test_save_variables_roundtrip(tmp_path: Path, variables: dict[str, str]) -> None:
    out = save_variables(variables, tmp_path / "vars.json")
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded == variables
