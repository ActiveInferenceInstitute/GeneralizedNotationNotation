"""SC-27: shipped grammar and formal-spec payloads resolve via the gnn_root loader."""

from pathlib import Path

import gnn.formal_specs
import gnn.grammars
from gnn.mcp.gnn_root import get_gnn_documentation

SPECS = ["agda", "alloy", "coq", "isabelle", "lean", "maxima", "tla_plus", "z_notation"]


def _spec_path(name: str) -> Path:
    """Resolve a formal_specs payload the way an installed wheel consumer would."""

    suffix = {
        "agda": ".agda",
        "alloy": ".als",
        "coq": ".v",
        "isabelle": ".thy",
        "lean": ".lean",
        "maxima": ".mac",
        "tla_plus": ".tla",
        "z_notation": ".zed",
    }[name]
    path = Path(gnn.formal_specs.__file__).with_name(name + suffix)
    assert path.exists(), f"shipped payload missing: {path}"
    return path


def test_loader_resolves_grammar():
    """The 'grammar' doc resolves the shipped EBNF payload through gnn_root."""
    result = get_gnn_documentation("grammar")
    assert result["success"], result["error"]
    assert "Backus-Naur" in result["content"] or "grammar" in result["content"].lower()


def test_shipped_grammar_files_present():
    """Both grammars/*.bnf and grammars/*.ebnf ship next to the package."""

    base = Path(gnn.grammars.__file__).parent
    assert (base / "bnf.bnf").exists()
    assert (base / "ebnf.ebnf").exists()
    assert (base / "ebnf.ebnf").read_text().strip() != ""


def test_all_formal_specs_resolvable():
    """Every formal_specs payload ships and is non-empty."""
    for name in SPECS:
        path = _spec_path(name)
        assert path.read_text().strip() != "", f"empty payload: {path}"
