#!/usr/bin/env python3
"""
Markdown documentation audit: relative links, AGENTS→SPEC footers, src/doc coverage,
AGENTS↔README pairing, SPEC coverage, prose layout drift, stale version claims.

Run from repository root:
  uv run --extra dev python docs/development/docs_audit.py
  uv run --extra dev python docs/development/docs_audit.py --strict   # exit 1 if any issue
  uv run --extra dev python docs/development/docs_audit.py --check-anchors  # verify #fragments in .md links (optional)
  uv run --extra dev python docs/development/docs_audit.py --strict --check-anchors --no-write

With ``--strict`` and any findings, a **full per-issue listing** is written to stderr by default
(terminal-friendly fix loop). Use ``--quiet`` to print only counts and the one-line summary.

Writes ``docs/development/docs_audit_report.md`` unless ``--no-write`` is passed or a
custom ``--report-path`` is provided.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Do not scan these path segments
SKIP_PARTS = frozenset(
    {
        "node_modules",
        ".venv",
        "__pycache__",
        ".git",
        ".myp",
        "dist",
        "build",
        ".eggs",
        # Pipeline-generated artifacts — not maintained documentation.
        # LLM output, render output, execute output, etc. may contain code-like
        # fragments that the link regex would otherwise mis-parse.
        "output",
    }
)

# docs/ subtrees excluded from maintained-folder checks (generated or exploratory)
DOC_MAINTAINED_SKIP_PARTS = frozenset(
    {
        "fleet-logs",
        "other",
        "__pycache__",
        ".git",
        "node_modules",
        ".venv",
    }
)

# Pairing report: skip dirs where AGENTS/README policy does not apply
PAIRING_SKIP_PARTS = frozenset(
    {
        "fleet-logs",
        "node_modules",
        ".venv",
        "__pycache__",
        ".git",
        "output",
        "other",
        "build",
        "dist",
        ".claude",
        ".desloppify",
        ".eggs",
    }
)


def should_skip(path: Path) -> bool:
    try:
        rel = path.relative_to(REPO_ROOT)
    except ValueError:
        return True
    return any(p in SKIP_PARTS for p in rel.parts) or _path_is_generated_output(rel)


def _path_is_generated_output(rel: Path) -> bool:
    """Run outputs that are intentionally excluded from maintained-doc audits."""
    parts = rel.parts
    if not parts:
        return False
    if parts[0] == "output":
        return True
    if any(part.startswith("activeinference_outputs_") for part in parts):
        return True
    if any(part.endswith("_outputs") or "_outputs_" in part for part in parts):
        return True
    return "pomdp_gridworld_outputs" in parts


def iter_markdown_files() -> list[Path]:
    out: set[Path] = set()
    for p in REPO_ROOT.rglob("*.md"):
        if should_skip(p):
            continue
        try:
            p.relative_to(REPO_ROOT)
        except ValueError:
            continue
        out.add(p)
    return sorted(out)


# [text](url) — capture path before # or )
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)")

# Image links whose alt text contains one nested bracket level, e.g.
# ![Caption [DAT] tag](Figure_1.png). LINK_RE cannot match these because the
# first closing ] ends the alt text early, so a dedicated pattern extracts the
# path for link validation. Applies to both ! and plain link forms.
NESTED_ALT_LINK_RE = re.compile(
    r"!{0,1}\[(?:[^\]\[]|\[[^\]\[]*\])*\]\(([^)#\s]+)(?:#[^)]*)?\)"
)

# [text](url) — full href including fragment (for anchor checks)
FULL_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

# Fenced code blocks (``` … ``` or ~~~ … ~~~), non-greedy, DOTALL so fences can span lines.
FENCED_CODE_RE = re.compile(r"(?ms)^([ \t]*)(```|~~~)[^\n]*\n.*?^\1\2[ \t]*$")

# Inline code spans: `…` or ``…`` (shortest match wins within a line).
INLINE_CODE_RE = re.compile(r"(`+)(?:(?!\1).)+\1")


def _strip_code(md: str) -> str:
    """Remove fenced code blocks and inline code spans so link regexes don't match
    Python / shell snippets that happen to contain ``[ident](expr)``."""
    no_blocks = FENCED_CODE_RE.sub(
        lambda m: "\n".join("" for _ in m.group(0).splitlines()), md
    )
    return INLINE_CODE_RE.sub(lambda m: " " * len(m.group(0)), no_blocks)


def extract_links(md: str) -> list[str]:
    stripped = _strip_code(md)
    links = [m.group(1).strip() for m in LINK_RE.finditer(stripped)]
    # Second pass: catch links with nested-bracket alt text that LINK_RE skips.
    seen = set(links)
    for m in NESTED_ALT_LINK_RE.finditer(stripped):
        link = m.group(1).strip()
        if link not in seen:
            links.append(link)
            seen.add(link)
    return links


def gfm_slug(heading_line: str) -> str:
    """GitHub-compatible heading slug, per cmark-gfm's `gfm_auto_identifiers`
    extension (github/cmark-gfm, extensions/gfm_auto_identifiers.c):

    1. Take the heading text (inline formatting markers such as backticks are
       removed, as GitHub renders them away before slugging).
    2. Lowercase (Unicode-aware).
    3. Remove every character that is not a Unicode letter, digit, underscore,
       hyphen, or space (this strips emoji and other punctuation).
    4. Replace each space with a hyphen, then strip leading/trailing hyphens
       (GitHub drops hyphens left over from removed punctuation/emoji). Interior
       space runs are NOT collapsed — one hyphen per space.

    >>> gfm_slug("## 🚀 Start here")
    'start-here'
    >>> gfm_slug("## 中文 标题!")
    '中文-标题'
    >>> gfm_slug("## C++ & Python")
    'c--python'
    >>> gfm_slug("## Foo   Bar")
    'foo---bar'
    """
    m = re.match(r"^#{1,6}\s+(.+)$", heading_line.strip())
    text = m.group(1) if m else heading_line
    text = text.strip().replace("`", "").lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    return text.replace(" ", "-").strip("-")


def _heading_slugs_in_markdown(md: str) -> set[str]:
    slugs: set[str] = set()
    for line in md.splitlines():
        if not line.strip().startswith("#"):
            continue
        slugs.add(gfm_slug(line))
    return slugs


def audit_bad_markdown_anchors(files: list[Path]) -> list[tuple[Path, int, str, str]]:
    """Flag relative links to *.md where #fragment does not match any heading slug."""
    issues: list[tuple[Path, int, str, str]] = []
    slug_cache: dict[Path, set[str]] = {}
    for src in files:
        try:
            rel_src = src.relative_to(REPO_ROOT)
        except ValueError:
            continue
        raw = src.read_text(encoding="utf-8", errors="replace")
        lines = _strip_code(raw).splitlines()
        for i, line in enumerate(lines, start=1):
            for m in FULL_LINK_RE.finditer(line):
                href = m.group(1).strip()
                if href.startswith("`"):
                    continue
                href = href.strip("<>")
                if "#" not in href:
                    continue
                path_part, frag = href.split("#", 1)
                if not path_part:
                    continue
                frag = frag.split("?")[0].strip()
                if not frag:
                    continue
                if path_part.startswith(("http://", "https://", "mailto:", "//")):
                    continue
                resolved = resolve_link(src, path_part)
                if resolved is None or not resolved.is_file():
                    continue
                if resolved.suffix.lower() != ".md":
                    continue
                if resolved not in slug_cache:
                    slug_cache[resolved] = _heading_slugs_in_markdown(
                        resolved.read_text(encoding="utf-8", errors="replace")
                    )
                slugs = slug_cache[resolved]
                frag_l = frag.lower()
                if frag_l not in slugs:
                    try:
                        rt = resolved.relative_to(REPO_ROOT)
                    except ValueError:
                        rt = resolved
                    issues.append(
                        (
                            rel_src,
                            i,
                            href,
                            f"anchor #{frag} not found (headings in `{rt}`)",
                        )
                    )
    return issues


def resolve_link(source_file: Path, href: str) -> Path | None:
    if not href or href.startswith(("http://", "https://", "mailto:", "//")):
        return None
    if href.startswith("#"):
        return None
    # Strip angle brackets some authors use
    href = href.strip("<>")
    base = source_file.parent
    target = (base / href).resolve()
    try:
        target.relative_to(REPO_ROOT)
    except ValueError:
        return None
    return target


def audit_broken_links(files: list[Path]) -> list[tuple[Path, int, str, str]]:
    """(source_file, line_no, href, reason)"""
    issues: list[tuple[Path, int, str, str]] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = _strip_code(text).splitlines()
        for i, line in enumerate(lines, start=1):
            for href in extract_links(line):
                if href.startswith("`"):
                    continue
                resolved = resolve_link(path, href)
                if resolved is None:
                    continue
                try:
                    rtarget = resolved.relative_to(REPO_ROOT)
                except ValueError:
                    continue
                if resolved.exists():
                    continue
                rel = path.relative_to(REPO_ROOT)
                issues.append((rel, i, href, f"missing: {rtarget}"))
    return issues


def audit_agents_spec() -> list[tuple[Path, str]]:
    """AGENTS.md that mention sibling SPEC.md link but file missing."""
    issues: list[tuple[Path, str]] = []
    for agents in REPO_ROOT.rglob("AGENTS.md"):
        if should_skip(agents):
            continue
        if "output" in agents.parts:
            continue
        text = agents.read_text(encoding="utf-8", errors="replace")
        if "[SPEC](SPEC.md)" not in text and "](SPEC.md)" not in text:
            continue
        spec = agents.parent / "SPEC.md"
        if not spec.exists():
            rel = agents.relative_to(REPO_ROOT)
            issues.append((rel, "references SPEC.md but sibling SPEC.md missing"))
    return issues


VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)

# Prose-only references to the removed "src/<module>/" layout. The canonical
# surface since v3.3.0 is ``src/gnn/<module>/`` (or ``gnn.<module>``); bare
# ``src/<module>`` and dotted ``src.<module>`` spellings are stale residue of
# the pre-3.3.0 tree. Excluded from matching: comments, fenced code blocks,
# and inline code spans — same treatment as the link/anchor audits.
PROSE_SRC_PATH_RE = re.compile(r"(?<![\w./-])src/[a-zA-Z_][\w-]*/")
PROSE_SRC_IMPORT_RE = re.compile(
    r"(?<![\w./])src\.[a-zA-Z_][\w.]*(?:\.[a-zA-Z_][\w]*)*"
)
PROSE_ALLOWED_SRC_PATHS = frozenset(
    {
        "src/gnn/",
        "src/tests/",
        "src/scripts/",
    }
)

# "### Current Version: X" / "Current Version: X" claims in module AGENTS.md
# files. The product version lives in pyproject.toml; per-tool metadata (the
# external oxdraw CLI version) is out of scope for this check.
CURRENT_VERSION_CLAIM_RE = re.compile(
    r"(?im)^\s*(?:#+\s*)?current version\s*[:=]\s*(\S+)"
)

# Claims already in canonical deferred form are exempt: any version claim
# that references pyproject.toml (the canonical source) instead of a number.
CANONICAL_REF_RE = re.compile(r"pyproject\.toml")

# Per-module metadata: a "Current Version:" block that explicitly states it
# is module-level (independent of the pipeline release) is exempt — the
# claim documents the module/tool, not the pipeline.
PER_MODULE_VERSION_RE = re.compile(
    r"(?i)module|per-module|independent of the pipeline|external tool"
)

# Version claims in docs: either the explicit "Version" line or bare
# "Pipeline Version: X" footers. Only enforced where a module/page states its
# own version, to avoid flagging legitimate historical release notes.
STALE_VERSION_CLAIM_RE = re.compile(
    r"(?im)^(?:\*\*)?\s*(?:pipeline )?version(?:\*\*)?\s*[:=]\s*\[?(\d+\.\d+\.\d+)\]?"
)


def audit_src_spec_coverage() -> list[tuple[Path, str]]:
    """Every module directory under src/gnn/ with .py files must have SPEC.md.

    The AGENTS.md/README.md pairing checks already cover agent-facing docs;
    SPEC.md is the interface contract of a module (public API, data flow) and
    is required for the same module set. ``src/gnn`` itself counts as a module
    directory (it carries AGENTS.md/README.md/SPEC.md at its root).
    """
    missing: list[tuple[Path, str]] = []
    for d in sorted((REPO_ROOT / "src" / "gnn").iterdir()):
        if not d.is_dir():
            continue
        rel_dir = d.relative_to(REPO_ROOT)
        if rel_dir.parts[0] in SKIP_PARTS or any(
            p in SKIP_PARTS for p in rel_dir.parts
        ):
            continue
        if not any(d.glob("*.py")):
            continue
        if not (d / "SPEC.md").is_file():
            missing.append((rel_dir, "module directory has .py files but no SPEC.md"))
    return missing


def _strip_code_and_links(md: str) -> str:
    """Remove fenced blocks, inline code spans, and markdown link targets so
    the prose pattern scans only see free text. Line structure is preserved
    exactly like ``_strip_code`` (fenced blocks become empty lines) so
    line numbers reported by these audits match the raw file."""
    no_blocks = FENCED_CODE_RE.sub(
        lambda m: "\n".join("" for _ in m.group(0).splitlines()), md
    )
    return INLINE_CODE_RE.sub(lambda m: " " * len(m.group(0)), no_blocks)


# Named repo-root docs inside the maintained set (package-version label
# scope: a bare "Version:" here claims the pipeline release).
MAINTAINED_ROOT_DOC_FILES = (
    "AGENTS.md",
    "README.md",
    "ARCHITECTURE.md",
    "TO-DO.md",
    "SETUP_GUIDE.md",
    "CLAUDE.md",
    "SKILL.md",
)

# Historical release-note files — exempt from stale-version/prose checks.
MAINTAINED_HISTORICAL_FILES = frozenset({"CHANGELOG.md", "VERSION_MAP.md"})

# Vendored/third-party source-quote trees whose md files quote upstream code
# (e.g. ActiveInference.jl sources with their own ``src/`` layout). Quoted
# code is exempt from the prose-pattern scan.
MAINTAINED_VENDORED_PREFIXES = (
    "docs/other/",
    "docs/activeinference_jl/",
    "docs/development/fleet-logs/",
)


def maintained_doc_files() -> list[Path]:
    """The maintained-doc scope for the claim-style checks (prose patterns,
    version claims): repo-root navigation docs + docs/** + src/gnn/** minus
    historical/vendored trees. Broader repo-wide scans (agent dispatch
    reports, private-agent trees) are out of scope — they are not
    maintained documentation."""
    keep: list[Path] = []
    for name in MAINTAINED_ROOT_DOC_FILES:
        p = REPO_ROOT / name
        if p.is_file():
            keep.append(p)
    for base in ("docs", "src/gnn"):
        base_dir = REPO_ROOT / base
        if base_dir.is_dir():
            for p in sorted(base_dir.rglob("*.md")):
                keep.append(p)
    out: list[Path] = []
    for p in keep:
        try:
            rel = p.relative_to(REPO_ROOT)
        except ValueError:
            continue
        if any(part in SKIP_PARTS for part in rel.parts):
            continue
        if any(
            str(rel).startswith(prefix) or str(rel) == prefix.rstrip("/")
            for prefix in MAINTAINED_VENDORED_PREFIXES
        ):
            continue
        if rel.name in MAINTAINED_HISTORICAL_FILES:
            continue
        out.append(p)
    return sorted(set(out))


def audit_prose_src_patterns(files: list[Path]) -> list[tuple[Path, int, str]]:
    """Flag prose references to the removed ``src/<module>/`` layout.

    Two stale spellings are flagged in visible prose (code spans and fenced
    blocks are exempt — they quote rather than assert):

    - ``src/<module>/…`` path spellings (e.g. ``src/gui/AGENTS.md``), which
      must read ``src/gnn/<module>/…`` since v3.3.0. ``src/gnn/`` itself is
      the canonical surface and never matches.
    - ``src.<module>`` import spellings (e.g. ``import src.main``), which
      fail at import time — the canonical package root is ``gnn.``.
    """
    issues: list[tuple[Path, int, str]] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        rel = path.relative_to(REPO_ROOT)
        for lineno, line in enumerate(
            _strip_code_and_links(text).splitlines(), start=1
        ):
            for match in PROSE_SRC_PATH_RE.finditer(line):
                span = match.group(0)
                if span.rstrip("/") + "/" in PROSE_ALLOWED_SRC_PATHS:
                    continue
                issues.append(
                    (
                        rel,
                        lineno,
                        f"prose path '{span}' is not the canonical src/gnn/ surface",
                    )
                )
            for match in PROSE_SRC_IMPORT_RE.finditer(line):
                issues.append(
                    (
                        rel,
                        lineno,
                        f"prose import '{match.group(0)}' is not the canonical gnn.* surface",
                    )
                )
    return issues


def audit_version_claims(files: list[Path]) -> list[tuple[Path, int, str, str]]:
    """Flag version claims that contradict the canonical pyproject version.

    The product version lives in ``pyproject.toml``; docs may reference it
    rather than hard-coding a number. Two claim shapes are enforced, with a
    version-axis split:

    - **Pipeline-version axes** (everywhere in the maintained set):
      ``Pipeline Version: X`` labels and ``Current Version: X`` module
      metadata blocks claim the *pipeline release* and must match
      pyproject.toml. ``[canonical · X]`` STEP_INDEX-style references are
      exempt (they already defer to pyproject).
    - **Bare ``Version: X``** lines: enforced only in ``src/gnn/**`` and the
      named repo-root files, where they can only mean the package version
      (``**Version**: 3.3.0`` page headers). Under ``docs/**`` they are
      treated as per-page revision axes (document revision counters, cited
      paper versions, per-tool metadata) and are exempt; a bare revision
      number there does not claim the pipeline release.

    Historical files (CHANGELOG/VERSION_MAP) and vendored quote trees are
    excluded from the maintained set entirely.
    """
    pyproject = REPO_ROOT / "pyproject.toml"
    version_match = VERSION_RE.search(pyproject.read_text(encoding="utf-8"))
    if not version_match:
        return [(Path("pyproject.toml"), 0, "", "cannot read project version")]
    canonical = version_match.group(1)

    package_label_scope: list[Path] = [REPO_ROOT / "src" / "gnn"]
    package_label_scope.extend(REPO_ROOT / name for name in MAINTAINED_ROOT_DOC_FILES)

    issues: list[tuple[Path, int, str, str]] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(REPO_ROOT)
        path_resolved = path.resolve()
        enforce_bare_version = any(
            path_resolved.is_relative_to(root) for root in package_label_scope
        )
        for lineno, line in enumerate(
            _strip_code_and_links(text).splitlines(), start=1
        ):
            stale_match = STALE_VERSION_CLAIM_RE.search(line)
            if (
                stale_match
                and stale_match.group(1) != canonical
                and not CANONICAL_REF_RE.search(line)
            ):
                is_pipeline_label = (
                    stale_match.group(0)
                    .lstrip("*")
                    .startswith(("pipeline", "Pipeline"))
                )
                if is_pipeline_label or enforce_bare_version:
                    issues.append(
                        (
                            rel,
                            lineno,
                            stale_match.group(1),
                            f"version claim != pyproject {canonical}",
                        )
                    )
            current_match = CURRENT_VERSION_CLAIM_RE.search(line)
            if (
                current_match
                and current_match.group(1) != canonical
                and not CANONICAL_REF_RE.search(line)
                and not PER_MODULE_VERSION_RE.search(line)
            ):
                issues.append(
                    (
                        rel,
                        lineno,
                        current_match.group(1),
                        f"'Current Version' claim != pyproject {canonical} (pipeline-versioned modules only)",
                    )
                )
    return issues


def audit_security_supported_version() -> list[str]:
    """SECURITY.md supported-versions table must carry a row for the version
    declared in pyproject.toml, e.g. a line starting with `| 3.3.0`."""
    issues: list[str] = []
    pyproject = REPO_ROOT / "pyproject.toml"
    security = REPO_ROOT / "SECURITY.md"
    m = VERSION_RE.search(pyproject.read_text(encoding="utf-8", errors="replace"))
    if not m:
        return ['pyproject.toml: no `version = "…"` declaration found']
    version = m.group(1)
    row = f"| {version}"
    if row not in security.read_text(encoding="utf-8", errors="replace"):
        return [f"SECURITY.md: no supported-versions row for current version {version}"]
    return issues


def audit_src_agents_coverage() -> list[Path]:
    """Directories under src/ with at least one .py file and no AGENTS.md."""
    missing: list[Path] = []
    src = REPO_ROOT / "src"
    if not src.is_dir():
        return missing
    for d in sorted(src.rglob("*")):
        if not d.is_dir():
            continue
        if should_skip(d) or d.name == "output":
            continue
        if not any(d.glob("*.py")):
            continue
        if (d / "AGENTS.md").exists():
            continue
        # Ignore __pycache__ dirs (already skipped by name in should_skip - __pycache__ is in SKIP_PARTS)
        if "__pycache__" in d.parts:
            continue
        missing.append(d.relative_to(REPO_ROOT))
    return missing


def _doc_path_is_generated_dump(rel: Path) -> bool:
    """Captured outputs / run artifacts — not maintained doc packages."""
    parts = rel.parts
    if "actinf_jl_src" in parts:
        i = parts.index("actinf_jl_src")
        if len(parts) > i + 1:
            return True
    if "meta-aware-2" in parts and "test_output" in parts:
        return True
    if "pomdp_gridworld_outputs" in parts:
        return True
    if "multiagent_trajectory_planning" in parts and "results" in parts:
        return True
    if "SQLite_exports" in parts:
        return True
    return False


def _doc_dir_is_maintained(d: Path) -> bool:
    """docs/ subtree folder expected to carry AGENTS.md and README.md."""
    try:
        rel = d.relative_to(REPO_ROOT)
    except ValueError:
        return False
    if len(rel.parts) < 2 or rel.parts[0] != "docs":
        return False
    if any(p in DOC_MAINTAINED_SKIP_PARTS for p in rel.parts):
        return False
    if _doc_path_is_generated_dump(rel):
        return False
    parts_set = set(rel.parts)
    if "results" in parts_set and "multiagent_trajectory_planning" in parts_set:
        return False
    if "enhanced_exports" in parts_set or "SQLite_exports" in parts_set:
        return False

    md_files = [x for x in d.iterdir() if x.is_file() and x.suffix == ".md"]
    subdirs = [
        x
        for x in d.iterdir()
        if x.is_dir()
        and x.name not in DOC_MAINTAINED_SKIP_PARTS
        and not any(p in SKIP_PARTS for p in x.parts)
    ]
    if not md_files and not subdirs:
        return False
    has_non_nav_md = any(m.name not in ("AGENTS.md", "README.md") for m in md_files)
    if has_non_nav_md:
        return True
    if subdirs:
        return True
    return len(md_files) >= 2


def audit_doc_maintained_missing_agents() -> list[Path]:
    missing: list[Path] = []
    doc_root = REPO_ROOT / "docs"
    if not doc_root.is_dir():
        return missing
    for d in sorted(doc_root.rglob("*")):
        if not d.is_dir():
            continue
        if should_skip(d):
            continue
        if not _doc_dir_is_maintained(d):
            continue
        if (d / "AGENTS.md").exists():
            continue
        missing.append(d.relative_to(REPO_ROOT))
    return missing


def audit_doc_maintained_missing_readme() -> list[Path]:
    missing: list[Path] = []
    doc_root = REPO_ROOT / "docs"
    if not doc_root.is_dir():
        return missing
    for d in sorted(doc_root.rglob("*")):
        if not d.is_dir():
            continue
        if should_skip(d):
            continue
        if not _doc_dir_is_maintained(d):
            continue
        if (d / "README.md").exists():
            continue
        missing.append(d.relative_to(REPO_ROOT))
    return missing


def _dir_eligible_for_pairing(d: Path) -> bool:
    if d.resolve() == REPO_ROOT.resolve():
        return True
    try:
        rel = d.relative_to(REPO_ROOT)
    except ValueError:
        return False
    if not rel.parts or rel.parts[0] not in ("src", "docs", ".github"):
        return False
    if any(p in PAIRING_SKIP_PARTS for p in rel.parts):
        return False
    if rel.parts[0] == "src" and len(rel.parts) >= 2 and rel.parts[1] == "output":
        return False
    if _doc_path_is_generated_dump(rel):
        return False
    return True


def audit_agents_without_readme() -> list[Path]:
    out: list[Path] = []
    for agents in sorted(REPO_ROOT.rglob("AGENTS.md")):
        if should_skip(agents):
            continue
        parent = agents.parent
        if not _dir_eligible_for_pairing(parent):
            continue
        if not (parent / "README.md").exists():
            out.append(parent.relative_to(REPO_ROOT))
    return sorted(set(out))


def audit_readme_without_agents() -> list[Path]:
    out: list[Path] = []
    for readme in sorted(REPO_ROOT.rglob("README.md")):
        if should_skip(readme):
            continue
        parent = readme.parent
        if not _dir_eligible_for_pairing(parent):
            continue
        if not (parent / "AGENTS.md").exists():
            out.append(parent.relative_to(REPO_ROOT))
    return sorted(set(out))


def audit_doc_agents_structure() -> list[tuple[Path, str]]:
    """
    docs/**/AGENTS.md should include a standard orientation section.
    Accept ## Overview, ## Purpose, or ## Directory Identity (GNN subtree manifests).
    If ## Purpose exists, its body (until the next ## heading) should be substantive.
    """
    issues: list[tuple[Path, str]] = []
    doc_root = REPO_ROOT / "docs"
    if not doc_root.is_dir():
        return issues
    orientation = ("## Overview", "## Purpose", "## Directory Identity")
    for agents in sorted(doc_root.rglob("AGENTS.md")):
        if should_skip(agents):
            continue
        try:
            rel = agents.relative_to(REPO_ROOT)
        except ValueError:
            continue
        if any(p in DOC_MAINTAINED_SKIP_PARTS for p in rel.parts):
            continue
        if _doc_path_is_generated_dump(rel):
            continue
        parts_set = set(rel.parts)
        if "results" in parts_set and "multiagent_trajectory_planning" in parts_set:
            continue
        text = agents.read_text(encoding="utf-8", errors="replace")
        if not any(marker in text for marker in orientation):
            issues.append(
                (
                    rel,
                    f"missing orientation section (need one of: {', '.join(orientation)})",
                )
            )
            continue
        m = re.search(r"## Purpose\s*\n(.*?)(?=\n##[^#]|\Z)", text, re.DOTALL)
        if m:
            body = m.group(1).strip()
            if len(body) < 20:
                issues.append(
                    (
                        rel,
                        f"## Purpose section too short ({len(body)} chars, need >= 20)",
                    )
                )
    return issues


def format_strict_issue_detail(
    *,
    link_issues: list[tuple[Path, int, str, str]],
    anchor_issues: list[tuple[Path, int, str, str]],
    anchor_checked: bool,
    spec_issues: list[tuple[Path, str]],
    coverage: list[Path],
    doc_missing_agents: list[Path],
    doc_missing_readme: list[Path],
    agents_no_readme: list[Path],
    readme_no_agents: list[Path],
    doc_agents_structure: list[tuple[Path, str]],
    security_version_issues: list[str],
    spec_coverage_issues: list[tuple[Path, str]],
    prose_src_issues: list[tuple[Path, int, str]],
    version_claim_issues: list[tuple[Path, int, str, str]],
) -> str:
    """Human-readable listing for terminal fix loops (stderr)."""
    chunks: list[str] = []
    chunks.append("Strict mode: full issue list (fix in source order)\n")

    if link_issues:
        chunks.append(f"## Broken relative links ({len(link_issues)})\n")
        for rel, lineno, href, reason in sorted(
            link_issues, key=lambda x: (str(x[0]), x[1])
        ):
            chunks.append(f"  {rel}:{lineno}  `{href}`  → {reason}\n")

    if anchor_checked and anchor_issues:
        chunks.append(f"## Bad markdown anchors ({len(anchor_issues)})\n")
        for rel, lineno, href, reason in sorted(
            anchor_issues, key=lambda x: (str(x[0]), x[1])
        ):
            chunks.append(f"  {rel}:{lineno}  `{href}`  → {reason}\n")

    if spec_issues:
        chunks.append(f"## AGENTS.md → missing SPEC.md ({len(spec_issues)})\n")
        for rel, msg in sorted(spec_issues, key=lambda x: str(x[0])):
            chunks.append(f"  `{rel}`  → {msg}\n")

    if coverage:
        chunks.append(f"## src/ dirs with .py but no AGENTS.md ({len(coverage)})\n")
        for rel in sorted(coverage, key=str):
            chunks.append(f"  `{rel}`\n")

    if doc_missing_agents:
        chunks.append(
            f"## docs/ maintained folders missing AGENTS.md ({len(doc_missing_agents)})\n"
        )
        for rel in sorted(doc_missing_agents, key=str):
            chunks.append(f"  `{rel}`\n")

    if doc_missing_readme:
        chunks.append(
            f"## docs/ maintained folders missing README.md ({len(doc_missing_readme)})\n"
        )
        for rel in sorted(doc_missing_readme, key=str):
            chunks.append(f"  `{rel}`\n")

    if agents_no_readme:
        chunks.append(
            f"## Directories with AGENTS.md but no README.md ({len(agents_no_readme)})\n"
        )
        for rel in sorted(agents_no_readme, key=str):
            chunks.append(f"  `{rel}`\n")

    if readme_no_agents:
        chunks.append(
            f"## Directories with README.md but no AGENTS.md ({len(readme_no_agents)})\n"
        )
        for rel in sorted(readme_no_agents, key=str):
            chunks.append(f"  `{rel}`\n")

    if doc_agents_structure:
        chunks.append(f"## docs/**/AGENTS.md structure ({len(doc_agents_structure)})\n")
        for rel, msg in sorted(doc_agents_structure, key=lambda x: str(x[0])):
            chunks.append(f"  `{rel}`  → {msg}\n")

    if security_version_issues:
        chunks.append(
            f"## SECURITY.md missing supported-versions row ({len(security_version_issues)})\n"
        )
        for msg in security_version_issues:
            chunks.append(f"  `{msg}`\n")

    if spec_coverage_issues:
        chunks.append(f"## src/ dirs missing SPEC.md ({len(spec_coverage_issues)})\n")
        for rel, msg in sorted(spec_coverage_issues, key=lambda x: str(x[0])):
            chunks.append(f"  `{rel}`  → {msg}\n")

    if prose_src_issues:
        chunks.append(f"## Prose src/<module>/ pattern ({len(prose_src_issues)})\n")
        for rel, lineno, reason in sorted(
            prose_src_issues, key=lambda x: (str(x[0]), x[1])
        ):
            chunks.append(f"  {rel}:{lineno}  → {reason}\n")

    if version_claim_issues:
        chunks.append(f"## Stale version claims ({len(version_claim_issues)})\n")
        for rel, lineno, claimed, reason in sorted(
            version_claim_issues, key=lambda x: (str(x[0]), x[1])
        ):
            chunks.append(f"  {rel}:{lineno}  `{claimed}`  → {reason}\n")

    chunks.append("\nTip: full tables also in docs/development/docs_audit_report.md\n")
    return "".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Markdown documentation audit for this repository."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any broken links, SPEC gaps, coverage gaps, AGENTS/README pairing, or doc AGENTS structure issues are found.",
    )
    parser.add_argument(
        "--check-anchors",
        action="store_true",
        help="Also verify that #fragments in relative .md links match a heading slug in the target file.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="With --strict: suppress per-issue detail on stderr (summary counts only).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Log extra diagnostics to stderr (e.g. markdown file count).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run checks without writing docs_audit_report.md.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPO_ROOT / "docs" / "development" / "docs_audit_report.md",
        help="Optional report output path. Ignored when --no-write is set.",
    )
    args = parser.parse_args()

    if not logging.root.handlers:
        logging.basicConfig(
            level=logging.INFO if args.verbose else logging.WARNING,
            format="%(message)s",
            stream=sys.stderr,
            force=True,
        )

    if not REPO_ROOT.joinpath("pyproject.toml").exists():
        print("Run from repo root (pyproject.toml not found).", file=sys.stderr)
        return 1

    md_files = iter_markdown_files()
    if args.verbose:
        logging.info("Markdown files scanned: %d", len(md_files))
    link_issues = audit_broken_links(md_files)
    anchor_issues = audit_bad_markdown_anchors(md_files) if args.check_anchors else []
    spec_issues = audit_agents_spec()
    coverage = audit_src_agents_coverage()
    doc_missing_agents = audit_doc_maintained_missing_agents()
    doc_missing_readme = audit_doc_maintained_missing_readme()
    agents_no_readme = audit_agents_without_readme()
    readme_no_agents = audit_readme_without_agents()
    doc_agents_structure = audit_doc_agents_structure()
    security_version_issues = audit_security_supported_version()
    spec_coverage_issues = audit_src_spec_coverage()
    maintained_files = maintained_doc_files()
    prose_src_issues = audit_prose_src_patterns(maintained_files)
    version_claim_issues = audit_version_claims(maintained_files)

    report_path = args.report_path
    if not report_path.is_absolute():
        report_path = REPO_ROOT / report_path
    lines = [
        "# Documentation audit report",
        "",
        "Generated by `uv run --extra dev python docs/development/docs_audit.py`. Re-run after doc changes.",
        "",
        "## Broken relative Markdown links",
        "",
    ]
    if not link_issues:
        lines.append("None found (scoped scan).")
    else:
        lines.append("| Source | Line | Href | Issue |")
        lines.append("|--------|------|------|-------|")
        for rel, lineno, href, reason in sorted(
            link_issues, key=lambda x: (str(x[0]), x[1])
        ):
            lines.append(f"| `{rel}` | {lineno} | `{href}` | {reason} |")
    lines.extend(
        [
            "",
            "## Suspicious markdown anchors (optional --check-anchors)",
            "",
        ]
    )
    if not args.check_anchors:
        lines.append(
            "Not run (pass `--check-anchors` to validate `#fragments` against heading slugs)."
        )
    elif not anchor_issues:
        lines.append("None found.")
    else:
        lines.append("| Source | Line | Href | Issue |")
        lines.append("|--------|------|------|-------|")
        for rel, lineno, href, reason in sorted(
            anchor_issues, key=lambda x: (str(x[0]), x[1])
        ):
            lines.append(f"| `{rel}` | {lineno} | `{href}` | {reason} |")
    lines.extend(
        [
            "",
            "## AGENTS.md referencing sibling SPEC.md",
            "",
        ]
    )
    if not spec_issues:
        lines.append("None (all SPEC footers have a file).")
    else:
        for rel, msg in spec_issues:
            lines.append(f"- `{rel}`: {msg}")
    lines.extend(
        [
            "",
            "## src/ directories with .py but no AGENTS.md",
            "",
        ]
    )
    if not coverage:
        lines.append("None.")
    else:
        for rel in coverage:
            lines.append(f"- `{rel}`")
    lines.extend(
        [
            "",
            "## docs/ maintained folders missing AGENTS.md",
            "",
        ]
    )
    if not doc_missing_agents:
        lines.append("None.")
    else:
        for rel in doc_missing_agents:
            lines.append(f"- `{rel}`")
    lines.extend(
        [
            "",
            "## docs/ maintained folders missing README.md",
            "",
        ]
    )
    if not doc_missing_readme:
        lines.append("None.")
    else:
        for rel in doc_missing_readme:
            lines.append(f"- `{rel}`")
    lines.extend(
        [
            "",
            "## Directories with AGENTS.md but no README.md (src, doc, .github, repo root)",
            "",
        ]
    )
    if not agents_no_readme:
        lines.append("None.")
    else:
        for rel in agents_no_readme:
            lines.append(f"- `{rel}`")
    lines.extend(
        [
            "",
            "## Directories with README.md but no AGENTS.md (src, doc, .github, repo root)",
            "",
        ]
    )
    if not readme_no_agents:
        lines.append("None.")
    else:
        for rel in readme_no_agents:
            lines.append(f"- `{rel}`")
    lines.extend(
        [
            "",
            "## SECURITY.md supported-versions rows",
            "",
        ]
    )
    if not security_version_issues:
        lines.append("None.")
    else:
        for msg in security_version_issues:
            lines.append(f"- {msg}")
    lines.extend(
        [
            "",
            "## src/ module directories missing SPEC.md",
            "",
        ]
    )
    if not spec_coverage_issues:
        lines.append("None.")
    else:
        for rel, msg in spec_coverage_issues:
            lines.append(f"- `{rel}`: {msg}")
    lines.extend(
        [
            "",
            "## Prose references to the removed src/<module>/ layout",
            "",
        ]
    )
    if not prose_src_issues:
        lines.append("None found.")
    else:
        lines.append("| Source | Line | Issue |")
        lines.append("|--------|------|-------|")
        for rel, lineno, reason in sorted(
            prose_src_issues, key=lambda x: (str(x[0]), x[1])
        ):
            lines.append(f"| `{rel}` | {lineno} | {reason} |")
    lines.extend(
        [
            "",
            "## Version claims contradicting pyproject.toml",
            "",
        ]
    )
    if not version_claim_issues:
        lines.append("None found.")
    else:
        lines.append("| Source | Line | Claimed | Issue |")
        lines.append("|--------|------|---------|-------|")
        for rel, lineno, claimed, reason in sorted(
            version_claim_issues, key=lambda x: (str(x[0]), x[1])
        ):
            lines.append(f"| `{rel}` | {lineno} | {claimed} | {reason} |")
    lines.extend(
        [
            "",
            "## docs/**/AGENTS.md structure (Overview/Purpose)",
            "",
        ]
    )
    if not doc_agents_structure:
        lines.append("None.")
    else:
        for rel, msg in doc_agents_structure:
            lines.append(f"- `{rel}`: {msg}")

    if args.no_write:
        print("Report not written (--no-write).")
    else:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            display_path = report_path.relative_to(REPO_ROOT)
        except ValueError:
            display_path = report_path
        print(f"Wrote {display_path}")
    print(f"Broken links: {len(link_issues)}")
    print(f"Bad markdown anchors: {len(anchor_issues)}")
    print(f"AGENTS/SPEC gaps: {len(spec_issues)}")
    print(f"src dirs missing AGENTS.md: {len(coverage)}")
    print(f"doc maintained missing AGENTS.md: {len(doc_missing_agents)}")
    print(f"doc maintained missing README.md: {len(doc_missing_readme)}")
    print(f"AGENTS without README: {len(agents_no_readme)}")
    print(f"README without AGENTS: {len(readme_no_agents)}")
    print(f"doc AGENTS structure: {len(doc_agents_structure)}")
    print(f"SECURITY.md version row: {len(security_version_issues)}")
    print(f"src dirs missing SPEC.md: {len(spec_coverage_issues)}")
    print(f"prose src/<module> pattern: {len(prose_src_issues)}")
    print(f"stale version claims: {len(version_claim_issues)}")
    total_issues = (
        len(link_issues)
        + len(spec_issues)
        + len(coverage)
        + len(doc_missing_agents)
        + len(doc_missing_readme)
        + len(agents_no_readme)
        + len(readme_no_agents)
        + len(doc_agents_structure)
        + len(security_version_issues)
        + len(spec_coverage_issues)
        + len(prose_src_issues)
        + len(version_claim_issues)
        + (len(anchor_issues) if args.check_anchors else 0)
    )
    if args.strict and total_issues > 0:
        sys.stdout.flush()
        if not args.quiet:
            print(
                format_strict_issue_detail(
                    link_issues=link_issues,
                    anchor_issues=anchor_issues,
                    anchor_checked=args.check_anchors,
                    spec_issues=spec_issues,
                    coverage=coverage,
                    doc_missing_agents=doc_missing_agents,
                    doc_missing_readme=doc_missing_readme,
                    agents_no_readme=agents_no_readme,
                    readme_no_agents=readme_no_agents,
                    doc_agents_structure=doc_agents_structure,
                    security_version_issues=security_version_issues,
                    spec_coverage_issues=spec_coverage_issues,
                    prose_src_issues=prose_src_issues,
                    version_claim_issues=version_claim_issues,
                ),
                file=sys.stderr,
                end="",
            )
        print(f"Strict mode: {total_issues} issue(s); exiting 1.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
