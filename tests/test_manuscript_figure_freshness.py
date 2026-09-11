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
tree). ``fig:pipeline``, ``fig:family_matrix`` and ``fig:backend_matrix`` read
the step index and the family manifest directly and record no tokens, so the
build also stamps the working-tree digest of every data surface a generator
reads directly (``source_sha256``); these tests compare those digests against
the files on disk. The HEAD-side comparison lives in
``scripts/check_manuscript_tokens.py --strict``, where a figure built from
uncommitted data the prose does not describe fails the gate.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "output" / "figures" / "figure_registry.json"
TOKENS = REPO_ROOT / "output" / "data" / "manuscript_variables.json"
GENERATORS = sorted((REPO_ROOT / "scripts").glob("manuscript_fig_*.py"))
REBUILD = "python -m scripts.manuscript_build_figures"


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


def test_a_figure_source_edited_after_the_build_fails() -> None:
    """The SC-21 half of the record: digests of the data each generator read.

    ``fig:pipeline``/``fig:family_matrix``/``fig:backend_matrix`` consume no
    tokens, so the consumed-token drift check above is blind to them; their
    staleness shows up as a ``source_sha256`` entry that no longer matches the
    file on disk.
    """
    checked = 0
    for record in _registry():
        for rel, digest in sorted((record.get("source_sha256") or {}).items()):
            source = REPO_ROOT / rel
            assert source.is_file(), (
                f"{record['label']}: source surface {rel} is gone — rebuild ({REBUILD})"
            )
            assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, (
                f"{record['label']}: {rel} changed after the figure was built "
                f"— rebuild ({REBUILD})"
            )
            checked += 1
    assert checked >= 4, (
        "no source digests recorded — the build is not stamping the data "
        "surfaces the direct-reading generators consume"
    )


def test_the_build_declares_every_source_surface_it_stamps() -> None:
    """The build's ``_FIGURE_SOURCES`` table and the registry cannot diverge."""
    script = (REPO_ROOT / "scripts" / "manuscript_build_figures.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(script)
    sources: dict[str, tuple[str, ...]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_FIGURE_SOURCES"
            and node.value is not None
        ):
            sources = ast.literal_eval(node.value)
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_FIGURE_SOURCES" for t in node.targets
        ):
            sources = ast.literal_eval(node.value)
    assert sources, "_FIGURE_SOURCES not found in manuscript_build_figures.py"
    for record in _registry():
        expected = set(sources.get(record["label"], ()))
        assert set(record.get("source_sha256") or {}) == expected, (
            f"{record['label']}: registry source digests do not match the "
            f"build's declared sources {sorted(expected)} — rebuild ({REBUILD})"
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
