"""The committed token map must be exactly what the producer derives at HEAD.

``output/data/manuscript_variables.json`` is the artifact the render pipeline
hydrates prose from and the figure generators read. Nothing in the default
suite checked that the committed copy still describes this commit, so the map
went consistently stale: ``fig:repo_metrics`` shipped 373 test files while a
fresh producer call at the same tip returned 424, and the prose hydrated from
the stale copy. (Real sequence observed 2026-09: the committed JSON named
``043d5b96d``, counted 373 tests — honest at the commit it named — while HEAD
had moved to 424. Staleness of the *stamp* is the detectable signature.)

Two assertions, mirroring the artifacts-commit ritual where the map is
regenerated at tip X and committed at X's child:

1. the committed map is exactly ``generate_variables`` pinned to the commit
   its own ``GNN_GIT_COMMIT`` field names (token-level checksum equality) —
   the map was not tampered with and the producer still reproduces it;
2. every token except the stamp equals a fresh ``HEAD`` generation, and the
   stamp names the tip or its parent — the one-commit lag is the unavoidable
   bootstrap (a JSON inside commit Y can only have been generated at or
   before Y's parent).

``unknown`` as the stamp is a failure (SC-23): a map that cannot name its
commit cannot be audited. Per ``tests/test_zero_skip_contracts.py`` there is
no skip guard — a checkout without git metadata cannot honour the producer's
reproducibility contract, and that is a failure, not a skip (same convention
as ``test_counts_describe_the_stamped_commit_not_the_working_tree``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from gnn.manuscript.variables import (  # noqa: E402
    RepositorySnapshot,
    generate_variables,
    load_variables,
    token_checksum,
)

pytestmark: list[pytest.MarkDecorator] = [pytest.mark.fast, pytest.mark.unit]

TOKENS_PATH = REPO_ROOT / "output" / "data" / "manuscript_variables.json"


def _committed() -> dict[str, str]:
    assert TOKENS_PATH.is_file(), (
        "output/data/manuscript_variables.json is missing — run "
        "python scripts/z_generate_manuscript_variables.py and commit the map"
    )
    return load_variables(TOKENS_PATH)


def _named_commit_is_resolvable(commit: str) -> RepositorySnapshot:
    pinned = RepositorySnapshot(REPO_ROOT, revision=commit)
    assert pinned.commit == commit, (
        f"the committed token map names commit {commit!r}, which git cannot "
        "resolve from this checkout — the artifact is stale beyond its own "
        "provenance"
    )
    return pinned


def test_committed_token_map_reproduces_at_the_commit_it_names() -> None:
    """Token-level checksum equality between the artifact and a pinned rerun."""
    committed = _committed()
    stamp = committed.get("GNN_GIT_COMMIT", "unknown")
    assert stamp != "unknown", (
        "the committed token map reports GNN_GIT_COMMIT='unknown' (git was "
        "unavailable at generation), so no gate can verify it — regenerate "
        "inside the checkout: python scripts/z_generate_manuscript_variables.py"
    )
    pinned = _named_commit_is_resolvable(stamp)
    fresh_at_stamp = generate_variables(REPO_ROOT, snapshot=pinned)
    drift = sorted(
        key
        for key in set(committed) | set(fresh_at_stamp)
        if committed.get(key) != fresh_at_stamp.get(key)
    )
    preview = "\n".join(
        f"  {key}: committed={committed.get(key)!r} fresh={fresh_at_stamp.get(key)!r}"
        for key in drift[:8]
    )
    assert token_checksum(committed) == token_checksum(fresh_at_stamp), (
        f"output/data/manuscript_variables.json does not reproduce at the "
        f"commit it names ({stamp}). Drifting tokens:\n{preview}\n"
        "regenerate and commit: python scripts/z_generate_manuscript_variables.py"
    )


def test_committed_token_map_is_at_most_one_commit_stale() -> None:
    """The stamp must name HEAD or its parent, and fresh content when it does.

    The artifacts-commit bootstrap: the map is generated at tip X and
    committed at X's child, so ``parent(HEAD)`` is the freshest a committed
    map can legitimately name. Anything older is the consistently-stale class
    this gate exists for (the committed JSON said 373 test files while HEAD
    had 424). When the stamp does name HEAD, the whole map — every token,
    not just the stamp — must equal a fresh generation; when it names the
    parent, the pinned-reproduction test above already proves the content.
    """
    committed = _committed()
    stamp = committed.get("GNN_GIT_COMMIT", "unknown")
    assert stamp != "unknown", (
        "committed token map has GNN_GIT_COMMIT='unknown' — regenerate the map "
        "(python scripts/z_generate_manuscript_variables.py) from the checkout"
    )
    fresh_head = generate_variables(REPO_ROOT)
    head = fresh_head["GNN_GIT_COMMIT"]
    parent = RepositorySnapshot(REPO_ROOT, revision="HEAD~1")
    assert stamp in {head, parent.commit}, (
        f"the committed token map was generated at {stamp!r} but HEAD is "
        f"{head!r} — it is stale. Regenerate: "
        "python scripts/z_generate_manuscript_variables.py"
    )
    if stamp != head:
        return  # one-behind bootstrap: content is pinned by the other test
    drift = sorted(
        key
        for key in set(committed) | set(fresh_head)
        if committed.get(key) != fresh_head.get(key)
    )
    preview = "\n".join(
        f"  {key}: committed={committed.get(key)!r} "
        f"fresh@HEAD={fresh_head.get(key)!r}"
        for key in drift[:8]
    )
    assert committed == fresh_head, (
        "output/data/manuscript_variables.json disagrees with a fresh "
        f"generate_variables(HEAD={head}). Drifting tokens:\n{preview}\n"
        "regenerate and commit: python scripts/z_generate_manuscript_variables.py"
    )


def test_committed_map_is_wellformed_sorted_json() -> None:
    """The artifact must stay a sorted, deterministic JSON token map."""
    raw = json.loads(TOKENS_PATH.read_text(encoding="utf-8"))
    keys = list(raw)
    assert keys == sorted(keys), "manuscript_variables.json keys are unsorted"
    committed = _committed()
    assert set(committed) == set(raw)


def test_build_figures_rejects_a_stale_stamp(tmp_path: Path) -> None:
    """manuscript_build_figures.main() must SystemExit on a commit mismatch.

    Behavioral, not source-text: a sandbox tree whose fake producer writes a
    JSON stamped with a foreign commit must fail the build's gate before any
    generator runs. (Running the real main() would rebuild the real tree's
    artifacts as a side effect — this fake-root drive keeps the suite
    deterministic and worktree-clean.)
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "manuscript_build_figures_gate",
        REPO_ROOT / "scripts" / "manuscript_build_figures.py",
    )
    assert spec is not None and spec.loader is not None
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)

    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "output" / "data").mkdir(parents=True)
    # The producer subprocess is faked: it writes a map stamped with a commit
    # no snapshot of this tree can resolve (the tree is not even a git repo,
    # so the snapshot's commit is "unknown" — mismatch either way).
    fake_producer = root / "scripts" / "z_generate_manuscript_variables.py"
    fake_producer.write_text(
        "import json\n"
        "from pathlib import Path\n"
        "out = Path('output/data/manuscript_variables.json')\n"
        "out.write_text(json.dumps({'GNN_GIT_COMMIT': 'deadbeef'}))\n",
        encoding="utf-8",
    )
    build._PROJECT_ROOT = root
    with pytest.raises(SystemExit, match="deadbeef"):
        build.main()


if __name__ == "__main__":  # pragma: no cover - convenience
    raise SystemExit(pytest.main([__file__, "-q"]))
