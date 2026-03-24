#!/usr/bin/env python3
"""
Rewrite stale doc/gnn/* paths after the gnn doc tree was reorganized into
tutorials/, reference/, operations/, integration/, advanced/.

Run from repo root:
  uv run python doc/development/rewrite_gnn_doc_links.py
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SKIP_NAMES = frozenset({"docs_audit_report.md", "uv.lock"})

# (old_substring, new_substring) — apply in order (longest/specific first)
REPLACEMENTS: list[tuple[str, str]] = [
    ("gnn/gnn_standards.md", "gnn/reference/gnn_standards.md"),
    ("gnn/gnn_troubleshooting.md", "gnn/operations/gnn_troubleshooting.md"),
    ("gnn/gnn_ontology.md", "gnn/advanced/gnn_ontology.md"),
    ("gnn/framework_integration_guide.md", "gnn/integration/framework_integration_guide.md"),
    ("gnn/gnn_llm_neurosymbolic_active_inference.md", "gnn/advanced/gnn_llm_neurosymbolic_active_inference.md"),
    ("gnn/advanced_modeling_patterns.md", "gnn/advanced/advanced_modeling_patterns.md"),
    ("gnn/gnn_multiagent.md", "gnn/advanced/gnn_multiagent.md"),
    ("gnn/ontology_system.md", "gnn/advanced/ontology_system.md"),
    ("gnn/gnn_file_structure_doc.md", "gnn/reference/gnn_file_structure_doc.md"),
    ("gnn/gnn_dsl_manual.md", "gnn/reference/gnn_dsl_manual.md"),
    ("gnn/architecture_reference.md", "gnn/reference/architecture_reference.md"),
    ("gnn/gnn_schema.md", "gnn/reference/gnn_schema.md"),
    ("gnn/gnn_type_system.md", "gnn/reference/gnn_type_system.md"),
    ("gnn/gnn_syntax.md", "gnn/reference/gnn_syntax.md"),
    ("gnn/resource_metrics.md", "gnn/operations/resource_metrics.md"),
    ("gnn/gnn_tools.md", "gnn/operations/gnn_tools.md"),
    ("gnn/gnn_implementation.md", "gnn/integration/gnn_implementation.md"),
    ("gnn/gnn_export.md", "gnn/integration/gnn_export.md"),
    ("gnn/gnn_visualization.md", "gnn/integration/gnn_visualization.md"),
    ("gnn/gnn_examples_doc.md", "gnn/tutorials/gnn_examples_doc.md"),
    ("gnn/quickstart_tutorial.md", "gnn/tutorials/quickstart_tutorial.md"),
]

def should_skip(path: Path) -> bool:
    if path.name in SKIP_NAMES:
        return True
    try:
        rel = path.relative_to(REPO_ROOT)
    except ValueError:
        return True
    # Test harness temp only — not tracked fixtures
    if len(rel.parts) >= 3 and rel.parts[0] == "src" and rel.parts[1] == "tests" and "output" in rel.parts:
        return True
    if "node_modules" in path.parts or ".venv" in path.parts:
        return True
    return False


def main() -> int:
    changed = 0
    for path in REPO_ROOT.rglob("*.md"):
        if should_skip(path):
            continue
        text = path.read_text(encoding="utf-8")
        orig = text
        for old, new in REPLACEMENTS:
            text = text.replace(old, new)
        if text != orig:
            path.write_text(text, encoding="utf-8")
            changed += 1
    print(f"Updated {changed} markdown files (gnn path rewrites).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
