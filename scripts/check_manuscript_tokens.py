#!/usr/bin/env python3
"""Manuscript integrity gate for GeneralizedNotationNotation.

Deterministic checks run before/after rendering the token-injected manuscript:

1. **Unknown tokens** — every ``{{TOKEN}}`` used in a rendered section must be a key
   emitted by ``src.manuscript_variables.generate_variables``. An unknown token would
   survive substitution and render literally in the PDF.
2. **Dangling citations** — every Pandoc ``[@key]`` must resolve to an entry in
   ``manuscript/references.bib``.
3. **Hard-coded counts** — bare literals equal to a high-value producer count (e.g. the
   test-file count, MCP-tool count) are flagged: they should be ``{{TOKEN}}`` instead.

Exit code is non-zero when an unknown token or dangling citation is found (hard gate).
Hard-coded-count findings are reported as warnings unless ``--strict`` is passed.

Usage:
    python scripts/check_manuscript_tokens.py [--strict]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.manuscript_variables import generate_variables  # noqa: E402

# Files in manuscript/ that are documentation, not rendered sections.
_EXCLUDED = {"README.md", "AGENTS.md", "SYNTAX.md", "99_references.md"}

_TOKEN_RE = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
_CITE_RE = re.compile(r"@([A-Za-z][\w:-]+)")
_BIB_KEY_RE = re.compile(r"^@\w+\{([^,]+),", re.MULTILINE)
# Counts small enough to appear coincidentally (step numbers, dims) are not flagged.
_HARDCODE_MIN = 10


def _section_files(manuscript_dir: Path) -> list[Path]:
    return [p for p in sorted(manuscript_dir.glob("*.md")) if p.name not in _EXCLUDED]


def _strip_code(text: str) -> str:
    """Remove fenced and inline code so we do not flag tokens/numbers in code."""
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]*`", "", text)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manuscript token/citation integrity gate"
    )
    parser.add_argument(
        "--strict", action="store_true", help="treat hard-coded counts as failures"
    )
    args = parser.parse_args()

    manuscript_dir = _PROJECT_ROOT / "manuscript"
    variables = generate_variables(_PROJECT_ROOT)
    known_tokens = set(variables)

    bib_text = (manuscript_dir / "references.bib").read_text(encoding="utf-8")
    bib_keys = set(_BIB_KEY_RE.findall(bib_text))

    # High-value counts that should always be tokens, not literals.
    hardcode_targets = {
        variables[k]: k
        for k in (
            "GNN_MCP_TOOL_COUNT",
            "GNN_TEST_FILE_COUNT",
            "GNN_TEST_FUNCTION_COUNT",
            "GNN_DOC_FILE_COUNT",
            "GNN_SRC_PY_FILE_COUNT",
            "GNN_SRC_LOC",
            "GNN_EXAMPLE_COUNT",
            "GNN_SRC_PACKAGE_COUNT",
        )
        if variables.get(k, "").isdigit() and int(variables[k]) >= _HARDCODE_MIN
    }

    # config.yaml title/subtitle are NOT token-substituted (the injector only
    # processes manuscript/*.md), so a count baked into them silently drifts.
    # Scan them against every producer count, including the small ones the body
    # scan skips, so a regression like a hard-coded "25-step" subtitle fails here.
    config_text = (manuscript_dir / "config.yaml").read_text(encoding="utf-8")
    config_counts = {
        variables[k]: k
        for k in (
            "GNN_STEP_COUNT",
            "GNN_FAMILY_COUNT",
            "GNN_BACKEND_COUNT",
            "GNN_MCP_TOOL_COUNT",
            "GNN_EXAMPLE_COUNT",
            "GNN_SRC_PACKAGE_COUNT",
        )
        if variables.get(k, "").isdigit()
    }
    config_hardcoded: list[str] = []
    for line in config_text.splitlines():
        stripped = line.strip()
        if not (stripped.startswith(("title:", "subtitle:"))):
            continue
        for value, key in config_counts.items():
            if re.search(rf"(?<!\d){re.escape(value)}(?!\d)", line):
                config_hardcoded.append(
                    f"config.yaml: {stripped.split(':')[0]} hard-codes {value} (use a description without the number; {{{{{key}}}}} does not resolve in config.yaml)"
                )

    unknown_tokens: list[str] = []
    dangling_cites: list[str] = []
    hardcoded: list[str] = list(config_hardcoded)

    for path in _section_files(manuscript_dir):
        raw = path.read_text(encoding="utf-8")
        body = _strip_code(raw)
        for tok in _TOKEN_RE.findall(raw):
            if tok not in known_tokens:
                unknown_tokens.append(f"{path.name}: {{{{{tok}}}}}")
        for key in _CITE_RE.findall(body):
            # Skip {#sec:...}/{#fig:...} anchors which are not citations.
            if key.startswith(("sec:", "fig:", "tbl:", "eq:")):
                continue
            if key not in bib_keys:
                dangling_cites.append(f"{path.name}: [@{key}]")
        # token-stripped body so {{COUNT}} does not count as a literal
        no_tokens = _TOKEN_RE.sub("", body)
        for value, key in hardcode_targets.items():
            if re.search(rf"(?<!\d){re.escape(value)}(?!\d)", no_tokens):
                hardcoded.append(
                    f"{path.name}: literal {value} should be {{{{{key}}}}}"
                )

    print(f"Sections checked: {len(_section_files(manuscript_dir))}")
    print(f"Known tokens: {len(known_tokens)} | Bib keys: {len(bib_keys)}")

    ok = True
    if unknown_tokens:
        ok = False
        print(f"\nUNKNOWN TOKENS ({len(unknown_tokens)}) — would render literally:")
        for u in unknown_tokens:
            print(f"  ✗ {u}")
    if dangling_cites:
        ok = False
        print(f"\nDANGLING CITATIONS ({len(dangling_cites)}):")
        for d in sorted(set(dangling_cites)):
            print(f"  ✗ {d}")
    if config_hardcoded:
        # config.yaml counts cannot be tokenized away — always a hard failure.
        ok = False
    if hardcoded:
        print(f"\nHARD-CODED COUNTS ({len(hardcoded)}) — prefer tokens:")
        for h in sorted(set(hardcoded)):
            print(f"  ! {h}")
        if args.strict:
            ok = False

    if ok and not hardcoded:
        print("\n✅ Manuscript token/citation integrity: clean")
    elif ok:
        print(
            "\n✅ No unknown tokens or dangling citations (hard-coded warnings above)"
        )
    else:
        print("\n❌ Manuscript integrity gate FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
