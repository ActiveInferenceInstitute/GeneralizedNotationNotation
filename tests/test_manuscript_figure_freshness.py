"""A committed figure may not print a number the token map has moved past.

The prose is re-hydrated from ``output/data/manuscript_variables.json`` on every
render; a figure is a committed PNG that nothing re-runs. So a count can move in
the token map, propagate into the prose, and leave the figure printing the old
value — which is what happened: ``fig:repo_metrics`` shipped "Test files: 365"
on the same PDF page as prose reading 367, under a caption claiming the figure
was "measured from the tracked files at the commit the producer stamps". Four
successive passes each moved that count in the token map without rebuilding the
figure.

``scripts/manuscript_build_figures.py`` now records, per figure, the digest of
the PNG it produced and the ``{key: value}`` pairs the generator actually read
out of the token map — observed through
``scripts/lib/manuscript_figure_tokens.load_tokens``, not declared, so a
generator cannot claim a token it never read. These tests re-check that record
against the committed PNG and the live token map.

Scope, stated exactly: this gate covers figures whose numbers come from the
token map. It does not re-derive counts from the repository (that chain is
``test_manuscript_variables.py``, which pins the producer to the committed
tree), and it does not detect drift in a figure built from some other input —
``fig:pipeline``, ``fig:family_matrix`` and ``fig:backend_matrix`` read the step
index and the family manifest directly, and record no tokens. What it
guarantees is that no committed figure prints a token value that disagrees with
the producer output the prose beside it is hydrated from.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "output" / "figures" / "figure_registry.json"
TOKENS = REPO_ROOT / "output" / "data" / "manuscript_variables.json"
GENERATORS = sorted((REPO_ROOT / "scripts").glob("manuscript_fig_*.py"))
REBUILD = "python -m scripts.manuscript_build_figures"
import sys  # noqa: E402

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))



def _registry() -> list[dict]:
    assert REGISTRY.is_file(), f"figure_registry.json missing — run {REBUILD}"
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    figures: list[dict] = payload["figures"]
    assert figures, "figure registry records no figures"
    return figures


def _tokens() -> dict[str, str]:
    assert TOKENS.is_file(), (
        "manuscript_variables.json missing — run "
        "python scripts/z_generate_manuscript_variables.py"
    )
    return {
        k: str(v) for k, v in json.loads(TOKENS.read_text(encoding="utf-8")).items()
    }


def test_every_registry_entry_carries_build_provenance() -> None:
    """No figure may ship without the record this gate checks."""
    for record in _registry():
        label = record.get("label")
        assert record.get("png_sha256"), f"{label} has no png_sha256 — run {REBUILD}"
        assert "consumed_tokens" in record, (
            f"{label} has no consumed_tokens record — run {REBUILD}"
        )
        assert isinstance(record["consumed_tokens"], dict)


def test_committed_png_is_the_one_the_recorded_build_produced() -> None:
    """A PNG swapped or left behind after a rebuild fails here."""
    for record in _registry():
        png = REGISTRY.parent / record["filename"]
        assert png.is_file(), f"{record['label']}: {png.name} not built — run {REBUILD}"
        digest = hashlib.sha256(png.read_bytes()).hexdigest()
        assert digest == record["png_sha256"], (
            f"{record['label']}: committed {png.name} is not the figure the "
            f"registry records; rebuild and commit both ({REBUILD})"
        )


def test_no_figure_prints_a_token_value_the_producer_has_moved_past() -> None:
    """The regression itself: figure numbers must equal the live token map.

    A figure built before a count moved records the old value here, so the
    disagreement fails the suite instead of shipping inside a PNG that no
    later step re-reads.
    """
    live = _tokens()
    drift: list[str] = []
    for record in _registry():
        for key, recorded in sorted(record["consumed_tokens"].items()):
            assert key in live, (
                f"{record['label']} consumed {key}, absent from the token map"
            )
            if live[key] != recorded:
                drift.append(
                    f"{record['label']} ({record['filename']}) printed "
                    f"{key}={recorded}; the producer now says {live[key]}"
                )
    assert not drift, "\n".join([*drift, f"rebuild the figures: {REBUILD}"])


def test_the_recording_mechanism_is_not_vacuous() -> None:
    """At least one figure must actually record tokens.

    A recorder that silently stops observing would leave every figure with an
    empty ``consumed_tokens`` and make the drift check above pass on anything.
    """
    recorded = {
        record["label"]: record["consumed_tokens"]
        for record in _registry()
        if record["consumed_tokens"]
    }
    assert recorded, "no figure recorded any consumed token — the recorder is dead"
    assert "GNN_TEST_FILE_COUNT" in recorded.get("fig:repo_metrics", {}), (
        "fig:repo_metrics no longer records the test-file count it prints"
    )


def test_no_generator_can_read_the_token_map_outside_the_recorder() -> None:
    """A generator that loads the token map directly bypasses this gate.

    Provenance is only as good as its coverage: a new ``manuscript_fig_*.py``
    that opens ``manuscript_variables.json`` itself would print numbers no test
    compares against anything.
    """
    offenders = [
        path.name
        for path in GENERATORS
        if "manuscript_variables.json" in path.read_text(encoding="utf-8")
        and "load_tokens" not in path.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        "these generators reference the token map without the recording loader "
        f"(scripts/lib/manuscript_figure_tokens.load_tokens): {offenders}"
    )


def test_the_build_records_provenance_for_every_registered_figure() -> None:
    """The build's figure table and the provenance it writes cannot diverge."""
    script = (REPO_ROOT / "scripts" / "manuscript_build_figures.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(script)
    labels: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_FIGURES" for t in node.targets
        ):
            assert isinstance(node.value, (ast.List, ast.Tuple))
            for item in node.value.elts:
                assert isinstance(item, ast.Tuple)
                labels.add(ast.literal_eval(item.elts[0]))
    assert labels, "_FIGURES not found in manuscript_build_figures.py"
    assert {record["label"] for record in _registry()} == labels


# --- SC-21: the sources a figure was built from ------------------------------


def _build_module() -> types.ModuleType:
    """Import the figure build script (module-level snapshot costs a git call)."""
    import importlib.util

    script = REPO_ROOT / "scripts" / "manuscript_build_figures.py"
    spec = importlib.util.spec_from_file_location("manuscript_build_figures", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_registered_figure_records_a_source_digest() -> None:
    """No figure ships without the input digest that pins it to one tree."""
    build = _build_module()
    uncovered = [
        record["label"]
        for record in _registry()
        if not str(record.get("sources_sha256", "")).strip()
    ]
    assert not uncovered, (
        f"registry entries without sources_sha256 (rebuild: {REBUILD}): "
        f"{uncovered}"
    )
    missing = {record["label"] for record in _registry()} - set(
        build._FIGURE_SOURCES
    )
    assert not missing, (
        f"figures with no _FIGURE_SOURCES entry in manuscript_build_figures.py: "
        f"{sorted(missing)}"
    )


def test_source_digests_still_describe_head() -> None:
    """A figure built before one of its inputs moved fails here.

    The build recorded the digest of every generator script and data file it
    reads, computed from the HEAD snapshot at build time. Recomputing at the
    current HEAD catches a figure that silently describes an older tree —
    the STEP_INDEX.md edit that never triggered a PNG rebuild.
    """
    from gnn.manuscript.variables import RepositorySnapshot

    build = _build_module()
    snapshot = RepositorySnapshot(REPO_ROOT)
    assert snapshot.from_git, (
        "snapshot is not reading committed blobs — git metadata unavailable, "
        "so figure source digests cannot be verified"
    )
    stale = []
    for record in _registry():
        label = record["label"]
        fresh = build._sources_digest(label, snapshot)
        if str(record.get("sources_sha256", "")) != fresh:
            stale.append(label)
    assert not stale, (
        f"figures whose recorded sources no longer match HEAD (rebuild: "
        f"{REBUILD}): {stale}"
    )
