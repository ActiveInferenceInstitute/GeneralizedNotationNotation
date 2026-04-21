#!/usr/bin/env python3
"""Phase 4.2 regression tests for model_registry (Step 4).

Zero-mock: uses real sample GNN files and real filesystem fixtures.
"""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model_registry.registry import ModelRegistry  # noqa: E402

REPO_ROOT = SRC.parent
SAMPLE_GNN = REPO_ROOT / "input" / "gnn_files" / "basics" / "static_perception.md"


@pytest.mark.skipif(not SAMPLE_GNN.exists(), reason="Sample GNN unavailable")
def test_register_and_lookup_roundtrip(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry = ModelRegistry(registry_path)
    assert registry.register_model(SAMPLE_GNN) is True
    # list must return at least one entry now.
    all_models = registry.list_models()
    assert len(all_models) >= 1
    # Lookup by the registered model_id should succeed for at least one entry.
    ids = [m.model_id for m in all_models]
    found = registry.get_model(ids[0])
    assert found is not None
    assert found.model_id == ids[0]


@pytest.mark.skipif(not SAMPLE_GNN.exists(), reason="Sample GNN unavailable")
def test_registry_persists_across_instances(tmp_path):
    registry_path = tmp_path / "registry.json"
    r1 = ModelRegistry(registry_path)
    r1.register_model(SAMPLE_GNN)
    r1.save()
    # Fresh instance — must see the previously registered entry after load().
    r2 = ModelRegistry(registry_path)
    r2.load()
    assert len(r2.list_models()) >= 1


@pytest.mark.skipif(not SAMPLE_GNN.exists(), reason="Sample GNN unavailable")
def test_search_models_finds_by_name(tmp_path):
    registry = ModelRegistry(tmp_path / "r.json")
    registry.register_model(SAMPLE_GNN)
    # The sample is "static_perception" — search with any substring of that.
    results = registry.search_models("perception")
    assert len(results) >= 1, "search_models failed to find the sample by substring"


def test_registry_handles_empty_lookup(tmp_path):
    registry = ModelRegistry(tmp_path / "empty.json")
    assert registry.list_models() == []
    assert registry.get_model("nonexistent-id") is None


@pytest.mark.skipif(not SAMPLE_GNN.exists(), reason="Sample GNN unavailable")
def test_registry_delete_removes_model(tmp_path):
    registry = ModelRegistry(tmp_path / "r.json")
    registry.register_model(SAMPLE_GNN)
    models = registry.list_models()
    assert len(models) >= 1
    target_id = models[0].model_id
    assert registry.delete_model(target_id) is True
    assert registry.get_model(target_id) is None
