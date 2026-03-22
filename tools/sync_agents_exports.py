#!/usr/bin/env python3
"""
Insert or refresh the ### Export surface (`__all__`) block in src/*/AGENTS.md.

Markers (do not remove):
  <!-- EXPORT_SURFACE_START -->
  <!-- EXPORT_SURFACE_END -->

Run: uv run python tools/sync_agents_exports.py
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SKIP_DIR_NAMES = frozenset({"__pycache__", "output", ".venv"})

START = "<!-- EXPORT_SURFACE_START -->"
END = "<!-- EXPORT_SURFACE_END -->"

SECTION_HEADING = "### Export surface (`__all__`)"


def parse_all(init_path: Path) -> list[str] | None:
    try:
        tree = ast.parse(init_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == "__all__":
                if not isinstance(node.value, (ast.List, ast.Tuple)):
                    return []
                out: list[str] = []
                for elt in node.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        out.append(elt.value)
                    elif isinstance(elt, ast.Str):
                        out.append(elt.s)
                return out
    return []


def format_exports(names: list[str], rel_init: str) -> str:
    if not names:
        body = "_No `__all__` in this package’s `__init__.py`._"
    else:
        # One line per name for readability and diff stability
        lines = [f"- `{n}`" for n in names]
        body = "\n".join(lines)
    return (
        f"{SECTION_HEADING}\n\n"
        f"{START}\n"
        f"_Source of truth: `{rel_init}`. Regenerate:_\n"
        f"`uv run python tools/sync_agents_exports.py`\n\n"
        f"{body}\n"
        f"{END}\n"
    )


def insert_or_replace(agents_text: str, block: str) -> str:
    pattern = re.compile(
        re.escape(SECTION_HEADING) + r"\s*\n.*?" + re.escape(END) + r"\n?",
        re.DOTALL,
    )
    if pattern.search(agents_text):
        return pattern.sub(block.rstrip() + "\n", agents_text, count=1)

    # Insert after first "## API Reference"
    m = re.search(r"^## API Reference\s*\n", agents_text, re.MULTILINE)
    if m:
        pos = m.end()
        return agents_text[:pos] + "\n" + block + "\n" + agents_text[pos:]

    # Append before final --- or at end
    return agents_text.rstrip() + "\n\n" + block + "\n"


def iter_init_with_agents() -> list[Path]:
    out: list[Path] = []
    for init_path in sorted(SRC.rglob("__init__.py")):
        if any(p in SKIP_DIR_NAMES for p in init_path.parts):
            continue
        agents_path = init_path.parent / "AGENTS.md"
        if agents_path.is_file():
            out.append(init_path)
    return out


def main() -> int:
    changed = 0
    for init_path in iter_init_with_agents():
        agents_path = init_path.parent / "AGENTS.md"
        rel_init = init_path.relative_to(ROOT).as_posix()
        exports = parse_all(init_path)
        if exports is None:
            print(f"skip {rel_init}: syntax error")
            continue
        block = format_exports(exports, rel_init)
        old = agents_path.read_text(encoding="utf-8")
        new = insert_or_replace(old, block)
        if new != old:
            agents_path.write_text(new, encoding="utf-8")
            changed += 1
            print(f"updated {agents_path.relative_to(ROOT)}")
    print(f"done: {changed} file(s) updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
