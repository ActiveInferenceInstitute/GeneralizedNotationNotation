#!/usr/bin/env python3
"""Manuscript integrity gate for GeneralizedNotationNotation.

Deterministic checks run before/after rendering the token-injected manuscript:

1. **Unknown tokens** — every ``{{TOKEN}}`` used in a rendered section must be a key
   emitted by ``gnn.manuscript.generate_variables``. An unknown token would
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
8. **Repository path claims** — every ``input/...`` literal must exist on disk, and where
   prose names a manifest family beside such a literal, the literal must match that
   family's declared ``target_dir``. Every other check compares a *number* against the
   producer; a path is not a number, so all of them stayed green while a commit moved a
   family's ``target_dir`` and three prose sites went on naming the old directory.

Exit code is non-zero when an unknown token, dangling citation, malformed
cross-reference, config.yaml drift or contradicted path claim is found (hard gate).
Hard-coded count and step-number findings are reported as warnings unless
``--strict`` is passed.

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
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gnn.manuscript import (  # noqa: E402
    RepositorySnapshot,
    config_metadata_drift,
    generate_variables,
)
from gnn.manuscript.variables import _families  # noqa: E402
from scripts.lib.manuscript_exclusions import AUTHORING_GUIDE_SKIP  # noqa: E402

# The set of manuscript/*.md files the renderer does NOT substitute, taken from
# the renderer itself so this gate cannot drift from what actually ships. The
# literal fallback mirrors infrastructure.rendering.manuscript_injection's
# EXCLUDED_DOC_FILENAMES for standalone runs (no template checkout on sys.path).
#
# Deriving it replaced a hand-maintained list that exempted 99_references.md —
# a file that IS rendered, as "10 References" in the PDF.
#
# preamble.md stays in the checked set, but for the opposite reason: the injector
# substitutes it into output/manuscript/ and then _manuscript_source.py copies the
# RAW file back over that copy, so a token written there NEVER resolves and reaches
# hyperref verbatim (verified 2026-09-05: `Subject: GNNSUBTITLE` in the shipped
# PDF). Scanning it makes that dead end loud instead of silent. The producer owns
# preamble.md's two PDF-metadata values directly — see
# manuscript_variables.sync_preamble_metadata.
try:  # pragma: no cover - exercised only with a template checkout present
    from infrastructure.rendering.manuscript_injection import (  # type: ignore
        EXCLUDED_DOC_FILENAMES as _EXCLUDED_FROZEN,
    )

    _EXCLUDED = set(_EXCLUDED_FROZEN)
except ModuleNotFoundError:
    _EXCLUDED = {"MANUSCRIPT_STATUS.md", *AUTHORING_GUIDE_SKIP}

_TOKEN_RE = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
_CITE_RE = re.compile(r"@([A-Za-z][\w:-]+)")
_BIB_KEY_RE = re.compile(r"^@\w+\{([^,]+),", re.MULTILINE)
_STEP_PHRASE_RE = re.compile(r"[Ss]teps?\s+(\d+(?:\s*(?:,|and|to|through|–|-)\s*\d+)*)")
# Inverted word orders the forward pattern misses: "25 steps", "25-step
# pipeline", "25 to 30 steps". The integers are resolved through the same
# step_literals map (token-backed, no new blocklist).
_INVERTED_STEP_PHRASE_RE = re.compile(
    r"\b(\d+(?:\s*(?:,|and|to|through|–|-)\s*\d+)*)(?:-|\s+)steps?\b"
)
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

_HYDRATED_DIR = _PROJECT_ROOT / "output" / "manuscript"
_DECLARED_LABEL_RE = re.compile(r"\{#((?:fig|tbl|eq|sec):[\w:-]+)")


def _hydrated_token_issues(hydrated_dir: Path = _HYDRATED_DIR) -> list[str]:
    """Find unresolved ``{{TOKEN}}`` placeholders in the hydrated copies.

    ``output/manuscript/*.md`` is what the renderer actually consumes; a token
    that survives hydration reaches the PDF verbatim. Matching is the same
    ``_TOKEN_RE`` on code-stripped text as the source scan, so documented
    literal-brace prose (``{{...}}`` in 05_reproducibility.md) stays green
    while a real ``{{UPPERCASE}}`` placeholder fails the gate.
    """
    if not hydrated_dir.is_dir():
        return []
    issues: list[str] = []
    for path in sorted(hydrated_dir.glob("*.md")):
        body = _strip_code(path.read_text(encoding="utf-8"))
        for token in _TOKEN_RE.findall(body):
            issues.append(
                f"output/manuscript/{path.name}: unresolved {{{{{token}}}}} "
                "in the hydrated copy"
            )
    return issues


def _declared_labels(sections: list[Path], variables: dict[str, str]) -> set[str]:
    """Every ``{#kind:label}`` declaration in sections and producer tokens.

    Producer-emitted tables carry their own captions (``_caption`` writes
    ``{#tbl:pipeline_steps}`` inside ``GNN_STEP_TABLE``), so the declared set
    is the union of the ``.md`` sources and the token values.
    """
    labels: set[str] = set()
    for path in sections:
        labels.update(_DECLARED_LABEL_RE.findall(path.read_text(encoding="utf-8")))
    for value in variables.values():
        labels.update(_DECLARED_LABEL_RE.findall(value))
    return labels


def _dangling_xrefs(referenced: set[str], declared: set[str]) -> list[str]:
    """Crossref markers (@fig:/@tbl:/@eq:/@sec:) with no declared label."""
    return sorted(
        f"{ref} — no {{#{ref}}} declaration in any manuscript section or "
        "producer table"
        for ref in referenced
        if ref.split(":", 1)[0] in {"fig", "tbl", "eq", "sec"} and ref not in declared
    )


def _step_phrase_issues(
    name: str, no_tokens: str, step_literals: dict[str, str]
) -> list[str]:
    """Step literals in either word order that a GNN_STEP_* token owns.

    Both the forward ("step 3", "steps 11 and 12") and the inverted ("25
    steps", "25-step pipeline") orders resolve their integers through the
    same token-backed map — no new blocklist.
    """
    issues: list[str] = []
    for pattern in (_STEP_PHRASE_RE, _INVERTED_STEP_PHRASE_RE):
        for phrase in pattern.finditer(no_tokens):
            for number in _INT_RE.findall(phrase.group(1)):
                step_key = step_literals.get(number)
                if step_key:
                    issues.append(
                        f"{name}: step literal {number} in {phrase.group(0)!r} "
                        f"should be {{{{{step_key}}}}}"
                    )
    return issues




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
        candidate = Path(filename)
        # Reject before joining: an absolute or parent-escaping filename would
        # otherwise probe (or validate) a file outside output/figures/.
        if candidate.is_absolute() or ".." in candidate.parts:
            issues.append(
                f"figure_registry.json: {label} image {filename!r} must be a "
                "bare filename inside output/figures/"
            )
            continue
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


# --- Rule 8: repository path claims -----------------------------------------
#
# Every check above compares a *number* against the producer. A path is not a
# number, so all of them stayed green while a commit repointed the `multiagent`
# family's target_dir from input/multi_agent_models into input/gnn_files/ and
# three prose sites went on naming the old directory. That is the gap this rule
# closes, in the same shape as the rest of the gate: the manifest is the source
# of truth and the manuscript is checked against it.
#
# Two independent checks:
#   (a) existence — an `input/...` literal must name something on disk;
#   (b) family pairing — where prose names a manifest family and an `input/...`
#       literal in the same breath, the literal must be that family's declared
#       target_dir, an ancestor of it (`input/gnn_files` covering a family), or
#       a path inside it.
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_INPUT_PATH_RE = re.compile(r"^input/[\w./-]+$")
# How far from a family mention a path literal is still read as a claim about
# that family. Wide enough to span "the `x` family, whose target directory is
# `input/y`", narrow enough not to reach across a paragraph.
_CLAIM_WINDOW = 160


def _code_spans(text: str) -> list[tuple[int, str]]:
    """``(offset, content)`` for each inline code span, fenced blocks removed.

    Fenced blocks are dropped because they hold reproduction *commands*, where a
    path is an argument rather than a claim about where a family lives.
    """
    without_fences = re.sub(
        r"```.*?```", lambda m: " " * len(m.group(0)), text, flags=re.DOTALL
    )
    return [(m.start(1), m.group(1)) for m in _INLINE_CODE_RE.finditer(without_fences)]


def _path_claim_issues(sections: list[Path], families: list[dict]) -> list[str]:
    """Check `input/...` literals against the model-family manifest."""
    target_dirs = {
        str(f.get("name", "")): str(f.get("target_dir", "")).rstrip("/")
        for f in families
        if f.get("name") and f.get("target_dir")
    }
    issues: list[str] = []
    for path in sections:
        raw = path.read_text(encoding="utf-8")
        spans = _code_spans(raw)
        paths = [
            (offset, content.rstrip("/"))
            for offset, content in spans
            if _INPUT_PATH_RE.match(content.rstrip("/"))
        ]
        for offset, literal in paths:
            if not (_PROJECT_ROOT / literal).exists():
                issues.append(
                    f"{path.name}: `{literal}` does not exist in the repository"
                )
        for offset, content in spans:
            family = content.strip()
            declared = target_dirs.get(family)
            if declared is None:
                continue
            # Only a mention that actually calls it a family is a family claim;
            # `continuous` and `structured` are ordinary words otherwise.
            around = raw[max(0, offset - 40) : offset + len(content) + 40]
            if "famil" not in around.lower():
                continue
            # Only the NEAREST path literal is read as this family's location.
            # Taking every literal in the window swept in unrelated siblings
            # ("`input/recursive_models/` is a reserved directory") that happen
            # to share a sentence with a family mention.
            near = [
                (abs(p_offset - offset), literal)
                for p_offset, literal in paths
                if abs(p_offset - offset) <= _CLAIM_WINDOW
            ]
            if not near:
                continue
            _, literal = min(near)
            related = (
                literal == declared
                or literal.startswith(declared + "/")
                or declared.startswith(literal + "/")
            )
            if not related:
                issues.append(
                    f"{path.name}: the `{family}` family is named beside "
                    f"`{literal}`, but the manifest declares its target_dir "
                    f"as `{declared}`"
                )
    return sorted(set(issues))


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

    # High-value counts that should always be tokens, not literals. Includes
    # the step/family/backend/framework counts and the figure-census families:
    # below _HARDCODE_MIN they are filtered out automatically (a count of 6 or
    # 9 is too coincidental to police), so the list can name every producer
    # count without inventing false positives.
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
            "GNN_STEP_COUNT",
            "GNN_FAMILY_COUNT",
            "GNN_BACKEND_COUNT",
            "GNN_MAINTAINED_FRAMEWORK_COUNT",
            "GNN_OUTPUT_FIGURE_COUNT",
            "GNN_OUTPUT_ARTIFACT_FIGURE_COUNT",
            "GNN_MANUSCRIPT_FIGURE_COUNT",
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
    referenced_labels: set[str] = set()

    for path in _section_files(manuscript_dir):
        raw = path.read_text(encoding="utf-8")
        body = _strip_code(raw)
        for tok in _TOKEN_RE.findall(raw):
            if tok not in known_tokens:
                unknown_tokens.append(f"{path.name}: {{{{{tok}}}}}")
        for key in _CITE_RE.findall(body):
            # Crossref markers are checked against label declarations below.
            if key.startswith(("sec:", "fig:", "tbl:", "eq:")):
                referenced_labels.add(key)
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
        hardcoded.extend(_step_phrase_issues(path.name, no_tokens, step_literals))

    declared = _declared_labels(_section_files(manuscript_dir), variables)
    dangling_xrefs = _dangling_xrefs(referenced_labels, declared)
    hydrated_issues = _hydrated_token_issues()

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
    if dangling_xrefs:
        ok = False
        print(f"\nDANGLING CROSS-REFERENCES ({len(dangling_xrefs)}):")
        for d in dangling_xrefs:
            print(f"  ✗ {d}")
    if hydrated_issues:
        ok = False
        print(f"\nUNRESOLVED TOKENS IN HYDRATED COPIES ({len(hydrated_issues)}):")
        for h in hydrated_issues:
            print(f"  ✗ {h}")
    figure_issues = _figure_registry_issues(
        manuscript_dir, _section_files(manuscript_dir)
    )
    if figure_issues:
        ok = False
        print(f"\nFIGURE REGISTRY ({len(figure_issues)}):")
        for f in figure_issues:
            print(f"  ✗ {f}")
    path_claims = _path_claim_issues(
        _section_files(manuscript_dir),
        _families(RepositorySnapshot(_PROJECT_ROOT)),
    )
    if path_claims:
        ok = False
        print(f"\nREPOSITORY PATH CLAIMS ({len(path_claims)}):")
        for c in path_claims:
            print(f"  \u2717 {c}")
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
