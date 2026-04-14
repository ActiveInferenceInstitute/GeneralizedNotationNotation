#!/usr/bin/env python3
"""
Markdown documentation audit: relative links, AGENTS→SPEC footers, src/doc coverage,
AGENTS↔README pairing.

Run from repository root:
  uv run python doc/development/docs_audit.py
  uv run python doc/development/docs_audit.py --strict   # exit 1 if any issue
  uv run python doc/development/docs_audit.py --check-anchors  # verify #fragments in .md links (optional)

With ``--strict`` and any findings, a **full per-issue listing** is written to stderr by default
(terminal-friendly fix loop). Use ``--quiet`` to print only counts and the one-line summary.

Writes: doc/development/docs_audit_report.md
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
    }
)

# doc/ subtrees excluded from maintained-folder checks (generated or archival)
DOC_MAINTAINED_SKIP_PARTS = frozenset(
    {
        "archive",
        "__pycache__",
        ".git",
        "node_modules",
        ".venv",
    }
)

# Pairing report: skip dirs where AGENTS/README policy does not apply
PAIRING_SKIP_PARTS = frozenset(
    {
        "node_modules",
        ".venv",
        "__pycache__",
        ".git",
        "output",
        "archive",
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
    return any(p in SKIP_PARTS for p in rel.parts)


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

# [text](url) — full href including fragment (for anchor checks)
FULL_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def extract_links(md: str) -> list[str]:
    return [m.group(1).strip() for m in LINK_RE.finditer(md)]


def _gfm_heading_slug(heading_line: str) -> str:
    """Approximate GitHub-style slug from a markdown heading line (with # marks)."""
    m = re.match(r"^#{1,6}\s+(.+)$", heading_line.strip())
    text = m.group(1) if m else heading_line
    text = re.sub(r"`+", "", text.strip()).lower()
    # Keep word chars (Unicode) and spaces; drop punctuation like (), /, emoji
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[-\s]+", "-", text.strip())
    return re.sub(r"-+", "-", text).strip("-")


def _heading_slugs_in_markdown(md: str) -> set[str]:
    slugs: set[str] = set()
    for line in md.splitlines():
        if not line.strip().startswith("#"):
            continue
        slugs.add(_gfm_heading_slug(line))
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
        lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
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
        lines = text.splitlines()
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
    """doc/ subtree folder expected to carry AGENTS.md and README.md."""
    try:
        rel = d.relative_to(REPO_ROOT)
    except ValueError:
        return False
    if len(rel.parts) < 2 or rel.parts[0] != "doc":
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
    doc_root = REPO_ROOT / "doc"
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
    doc_root = REPO_ROOT / "doc"
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
    if not rel.parts or rel.parts[0] not in ("src", "doc", ".github"):
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
    doc/**/AGENTS.md should include a standard orientation section.
    Accept ## Overview, ## Purpose, or ## Directory Identity (GNN subtree manifests).
    If ## Purpose exists, its body (until the next ## heading) should be substantive.
    """
    issues: list[tuple[Path, str]] = []
    doc_root = REPO_ROOT / "doc"
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
                (rel, f"missing orientation section (need one of: {', '.join(orientation)})")
            )
            continue
        m = re.search(r"## Purpose\s*\n(.*?)(?=\n##[^#]|\Z)", text, re.DOTALL)
        if m:
            body = m.group(1).strip()
            if len(body) < 20:
                issues.append((rel, f"## Purpose section too short ({len(body)} chars, need >= 20)"))
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
) -> str:
    """Human-readable listing for terminal fix loops (stderr)."""
    chunks: list[str] = []
    chunks.append("Strict mode: full issue list (fix in source order)\n")

    if link_issues:
        chunks.append(f"## Broken relative links ({len(link_issues)})\n")
        for rel, lineno, href, reason in sorted(link_issues, key=lambda x: (str(x[0]), x[1])):
            chunks.append(f"  {rel}:{lineno}  `{href}`  → {reason}\n")

    if anchor_checked and anchor_issues:
        chunks.append(f"## Bad markdown anchors ({len(anchor_issues)})\n")
        for rel, lineno, href, reason in sorted(anchor_issues, key=lambda x: (str(x[0]), x[1])):
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
            f"## doc/ maintained folders missing AGENTS.md ({len(doc_missing_agents)})\n"
        )
        for rel in sorted(doc_missing_agents, key=str):
            chunks.append(f"  `{rel}`\n")

    if doc_missing_readme:
        chunks.append(
            f"## doc/ maintained folders missing README.md ({len(doc_missing_readme)})\n"
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
        chunks.append(f"## doc/**/AGENTS.md structure ({len(doc_agents_structure)})\n")
        for rel, msg in sorted(doc_agents_structure, key=lambda x: str(x[0])):
            chunks.append(f"  `{rel}`  → {msg}\n")

    chunks.append(
        "\nTip: full tables also in doc/development/docs_audit_report.md\n"
    )
    return "".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description="Markdown documentation audit for this repository.")
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

    report_path = REPO_ROOT / "doc" / "development" / "docs_audit_report.md"
    lines = [
        "# Documentation audit report",
        "",
        "Generated by `uv run python doc/development/docs_audit.py`. Re-run after doc changes.",
        "",
        "## Broken relative Markdown links",
        "",
    ]
    if not link_issues:
        lines.append("None found (scoped scan).")
    else:
        lines.append("| Source | Line | Href | Issue |")
        lines.append("|--------|------|------|-------|")
        for rel, lineno, href, reason in sorted(link_issues, key=lambda x: (str(x[0]), x[1])):
            lines.append(f"| `{rel}` | {lineno} | `{href}` | {reason} |")
    lines.extend(
        [
            "",
            "## Suspicious markdown anchors (optional --check-anchors)",
            "",
        ]
    )
    if not args.check_anchors:
        lines.append("Not run (pass `--check-anchors` to validate `#fragments` against heading slugs).")
    elif not anchor_issues:
        lines.append("None found.")
    else:
        lines.append("| Source | Line | Href | Issue |")
        lines.append("|--------|------|------|-------|")
        for rel, lineno, href, reason in sorted(anchor_issues, key=lambda x: (str(x[0]), x[1])):
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
            "## doc/ maintained folders missing AGENTS.md",
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
            "## doc/ maintained folders missing README.md",
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
            "## doc/**/AGENTS.md structure (Overview/Purpose)",
            "",
        ]
    )
    if not doc_agents_structure:
        lines.append("None.")
    else:
        for rel, msg in doc_agents_structure:
            lines.append(f"- `{rel}`: {msg}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report_path.relative_to(REPO_ROOT)}")
    print(f"Broken links: {len(link_issues)}")
    print(f"Bad markdown anchors: {len(anchor_issues)}")
    print(f"AGENTS/SPEC gaps: {len(spec_issues)}")
    print(f"src dirs missing AGENTS.md: {len(coverage)}")
    print(f"doc maintained missing AGENTS.md: {len(doc_missing_agents)}")
    print(f"doc maintained missing README.md: {len(doc_missing_readme)}")
    print(f"AGENTS without README: {len(agents_no_readme)}")
    print(f"README without AGENTS: {len(readme_no_agents)}")
    print(f"doc AGENTS structure: {len(doc_agents_structure)}")
    total_issues = (
        len(link_issues)
        + len(spec_issues)
        + len(coverage)
        + len(doc_missing_agents)
        + len(doc_missing_readme)
        + len(agents_no_readme)
        + len(readme_no_agents)
        + len(doc_agents_structure)
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
                ),
                file=sys.stderr,
                end="",
            )
        print(f"Strict mode: {total_issues} issue(s); exiting 1.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
