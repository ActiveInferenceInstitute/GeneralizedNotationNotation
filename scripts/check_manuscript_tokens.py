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
4. **Hard-coded step numbers** — a bare integer after "step"/"steps" that equals a
   ``GNN_STEP_*`` token value is flagged regardless of size. Step numbers are below the
   ``_HARDCODE_MIN`` floor the count scan uses, so they were invisible to check 3 while
   the producer emitted a token for every one of them.
5. **Malformed cross-references** — a pandoc-crossref marker glued to a stray prefix
   character (``+@fig:x``) still resolves, so no reference gate catches it, but the
   prefix renders literally into the PDF ("+fig. 2 depicts...").
6. **config.yaml drift** — ``manuscript/config.yaml`` is never token-substituted, so its
   ``version:`` / ``date:`` literals are compared against the producer's values.
7. **Figure accessibility registry** — every ``{#fig:...}`` label a section declares must
   have an ``output/figures/figure_registry.json`` entry with alt text and an existing
   image file. The template's own ``validate_figure_registry`` runs at the *validation*
   stage, not the render stage, so a render could (and did) ship a PDF with the registry
   absent entirely. This is the in-repo gate that does not depend on the template.

Exit code is non-zero when an unknown token, dangling citation, malformed cross-reference
or config.yaml drift is found (hard gate). Hard-coded count and step-number findings are
reported as warnings unless ``--strict`` is passed.

Usage:
    python scripts/check_manuscript_tokens.py [--strict]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.manuscript_variables import (  # noqa: E402
    config_metadata_drift,
    generate_variables,
)

# The set of manuscript/*.md files the renderer does NOT substitute, taken from
# the renderer itself so this gate cannot drift from what actually ships. The
# literal fallback mirrors infrastructure.rendering.manuscript_injection's
# EXCLUDED_DOC_FILENAMES for standalone runs (no template checkout on sys.path).
#
# Deriving it replaced a hand-maintained list that exempted 99_references.md —
# a file that IS rendered, as "10 References" in the PDF — while checking
# preamble.md, whose tokens land in the LaTeX header rather than a section.
# Both are substituted, so both are checked.
try:  # pragma: no cover - exercised only with a template checkout present
    from infrastructure.rendering.manuscript_injection import (  # type: ignore
        EXCLUDED_DOC_FILENAMES as _EXCLUDED_FROZEN,
    )

    _EXCLUDED = set(_EXCLUDED_FROZEN)
except ModuleNotFoundError:
    _EXCLUDED = {"AGENTS.md", "MANUSCRIPT_STATUS.md", "README.md", "SYNTAX.md"}

_TOKEN_RE = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
_CITE_RE = re.compile(r"@([A-Za-z][\w:-]+)")
_BIB_KEY_RE = re.compile(r"^@\w+\{([^,]+),", re.MULTILINE)
# Counts small enough to appear coincidentally (step numbers, dims) are not flagged.
_HARDCODE_MIN = 10
# "step 3", "Steps 5 and 6", "steps 11 and 12" — the integers a GNN_STEP_* token owns.
_STEP_PHRASE_RE = re.compile(r"[Ss]teps?\s+(\d+(?:\s*(?:,|and|to|through|–|-)\s*\d+)*)")
_INT_RE = re.compile(r"\d+")
# A crossref marker preceded by a character that is neither "[" nor "!" nor
# whitespace: the marker resolves, so the reference gate stays green, and the
# stray prefix renders literally.
_MALFORMED_XREF_RE = re.compile(r"(?<=[^\[!\s])@(?:fig|tbl|eq|sec):[\w:-]+")


def _section_files(manuscript_dir: Path) -> list[Path]:
    return [p for p in sorted(manuscript_dir.glob("*.md")) if p.name not in _EXCLUDED]


_FIG_LABEL_RE = re.compile(r"\{#(fig:[\w:-]+)")


def _figure_registry_issues(manuscript_dir: Path, sections: list[Path]) -> list[str]:
    """Check every declared figure label against the accessibility registry."""
    declared: dict[str, str] = {}
    for path in sections:
        for label in _FIG_LABEL_RE.findall(path.read_text(encoding="utf-8")):
            declared.setdefault(label, path.name)
    if not declared:
        return []
    registry_path = _PROJECT_ROOT / "output" / "figures" / "figure_registry.json"
    if not registry_path.is_file():
        return [
            f"{registry_path.relative_to(_PROJECT_ROOT)} is missing while "
            f"{len(declared)} figure(s) are referenced — run "
            "python -m scripts.manuscript_build_figures"
        ]
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    records = payload.get("figures", payload) if isinstance(payload, dict) else payload
    by_label = {str(rec.get("label")): rec for rec in records if isinstance(rec, dict)}
    issues: list[str] = []
    for label, section in sorted(declared.items()):
        record = by_label.get(label)
        if record is None:
            issues.append(f"{section}: {label} has no figure_registry.json entry")
            continue
        if not str(record.get("alt_text", "")).strip():
            issues.append(f"figure_registry.json: {label} has no alt_text")
        filename = str(record.get("filename", ""))
        if not filename or not (registry_path.parent / filename).is_file():
            issues.append(
                f"figure_registry.json: {label} image {filename!r} is missing"
            )
    return issues


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
    # version:/date: are producer-owned. scripts/z_generate_manuscript_variables.py
    # writes them; this catches a hand-edit that put them back out of step.
    config_drift = config_metadata_drift(_PROJECT_ROOT, variables)

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

    # Every pipeline step number the producer owns, keyed by its literal value.
    # These sit below _HARDCODE_MIN, so the count scan above cannot see them.
    step_literals = {
        value: key
        for key, value in variables.items()
        if key.startswith("GNN_STEP_") and value.isdigit()
    }

    unknown_tokens: list[str] = []
    dangling_cites: list[str] = []
    malformed_xrefs: list[str] = []
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
        for match in _MALFORMED_XREF_RE.finditer(body):
            malformed_xrefs.append(
                f"{path.name}: {match.group(0)!r} has a stray prefix"
            )
        # token-stripped body so {{COUNT}} does not count as a literal
        no_tokens = _TOKEN_RE.sub("", body)
        for value, key in hardcode_targets.items():
            if re.search(rf"(?<!\d){re.escape(value)}(?!\d)", no_tokens):
                hardcoded.append(
                    f"{path.name}: literal {value} should be {{{{{key}}}}}"
                )
        for phrase in _STEP_PHRASE_RE.finditer(no_tokens):
            for number in _INT_RE.findall(phrase.group(1)):
                step_key = step_literals.get(number)
                if step_key:
                    hardcoded.append(
                        f"{path.name}: step literal {number} in {phrase.group(0)!r} "
                        f"should be {{{{{step_key}}}}}"
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
    if malformed_xrefs:
        ok = False
        print(f"\nMALFORMED CROSS-REFERENCES ({len(malformed_xrefs)}):")
        for m in sorted(set(malformed_xrefs)):
            print(f"  ✗ {m}")
    figure_issues = _figure_registry_issues(
        manuscript_dir, _section_files(manuscript_dir)
    )
    if figure_issues:
        ok = False
        print(f"\nFIGURE REGISTRY ({len(figure_issues)}):")
        for f in figure_issues:
            print(f"  ✗ {f}")
    if config_drift:
        ok = False
        print(f"\nCONFIG.YAML DRIFT ({len(config_drift)}):")
        for d in config_drift:
            print(f"  ✗ {d} — run scripts/z_generate_manuscript_variables.py")
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
