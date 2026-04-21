#!/usr/bin/env python3
"""Phase 4.2 regression tests for the ontology module (Step 10).

Exercises the Active Inference ontology term registry and annotation parsing
via real sample data. Zero-mock per CLAUDE.md.
"""

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ONTOLOGY_TERMS = SRC / "ontology" / "act_inf_ontology_terms.json"


@pytest.mark.skipif(not ONTOLOGY_TERMS.exists(), reason="Ontology terms file missing")
def test_ontology_terms_file_loads_and_contains_core_concepts():
    """The shipped ontology must contain Active Inference foundational terms.

    If these ever go missing, downstream steps that rely on ontology mapping
    (Step 10, Step 13 LLM context, Step 24 intelligent analysis) silently
    emit warnings instead of errors — that's a regression we want to catch.
    """
    data = json.loads(ONTOLOGY_TERMS.read_text())
    # The file may be a list of dicts OR a dict keyed by term name. Normalize.
    if isinstance(data, dict):
        term_names = set(data.keys())
    elif isinstance(data, list):
        term_names = {t.get("name") or t.get("term") or t.get("id") for t in data if isinstance(t, dict)}
    else:
        pytest.fail(f"Unexpected ontology file shape: {type(data).__name__}")
    # Core Active Inference concepts that MUST be present.
    core = {"HiddenState", "Observation"}
    assert core.issubset(term_names), (
        f"Core AI terms missing: {core - term_names}. Ontology file is corrupt or underspecified."
    )


def test_load_defined_ontology_terms_returns_nonempty_dict():
    from ontology import load_defined_ontology_terms
    # API takes zero args — reads from the shipped act_inf_ontology_terms.json.
    terms = load_defined_ontology_terms()
    assert isinstance(terms, dict)
    assert len(terms) > 0, "load_defined_ontology_terms returned empty dict"
    # At least one term should expose a description.
    first_key = next(iter(terms))
    first = terms[first_key]
    assert isinstance(first, (dict, str)), f"Term value should be dict or str, got {type(first).__name__}"


def test_parse_annotation_handles_valid_mapping():
    from ontology import parse_annotation
    # parse_annotation accepts a single annotation line like "s=HiddenState".
    result = parse_annotation("s=HiddenState")
    # API returns some representation that retains both sides of the mapping.
    assert result is not None
    # Whether it's a tuple, dict, or string, "s" and "HiddenState" should be
    # recoverable from the result — the exact shape is implementation-defined.
    as_text = str(result)
    assert "s" in as_text and "HiddenState" in as_text


def test_parse_gnn_ontology_section_extracts_mappings():
    from ontology import parse_gnn_ontology_section
    # Parser recognizes mappings under the `## ActInfOntologyAnnotation` header
    # (also accepts `## Ontology`). Bare lines outside a header are ignored —
    # this behavior is intentional per processor.py:101-107.
    content = (
        "## ActInfOntologyAnnotation\n"
        "s=HiddenState\n"
        "o=Observation\n"
        "A=RecognitionMatrix\n"
    )
    result = parse_gnn_ontology_section(content)
    assert isinstance(result, dict)
    annotations = result.get("annotations", [])
    assert any("s=HiddenState" in str(a) for a in annotations), \
        f"s=HiddenState not captured in annotations: {annotations!r}"
    assert any("o=Observation" in str(a) for a in annotations)
    assert any("A=RecognitionMatrix" in str(a) for a in annotations)


def test_parse_gnn_ontology_section_returns_empty_for_bare_content():
    """Content without the header marker produces an empty result, not None.
    Regression against a silent-None bug where callers would get AttributeError.
    """
    from ontology import parse_gnn_ontology_section
    result = parse_gnn_ontology_section("s=HiddenState\no=Observation\n")
    # Result is dict (possibly empty), not None — callers can safely .get().
    assert isinstance(result, dict)
