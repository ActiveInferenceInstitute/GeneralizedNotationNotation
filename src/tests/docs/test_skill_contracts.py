from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"


def _frontmatter_fields(text: str) -> dict[str, str]:
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---", 4)
    if end == -1:
        return {}

    fields: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def test_all_source_skills_have_discovery_frontmatter() -> None:
    skill_files = sorted(SRC_DIR.glob("**/SKILL.md"))
    assert skill_files, "No source SKILL.md files found"

    missing: dict[str, list[str]] = {}
    for skill_file in skill_files:
        rel_path = str(skill_file.relative_to(REPO_ROOT))
        fields = _frontmatter_fields(skill_file.read_text(encoding="utf-8"))
        absent = [
            field
            for field in ("name", "description")
            if not fields.get(field, "").strip()
        ]
        if absent:
            missing[rel_path] = absent

    assert missing == {}, f"SKILL.md files missing frontmatter fields: {missing}"


def test_all_source_skills_declare_mcp_tool_surface() -> None:
    skill_files = sorted(SRC_DIR.glob("**/SKILL.md"))
    missing = [
        str(skill_file.relative_to(REPO_ROOT))
        for skill_file in skill_files
        if "\n## MCP Tools\n" not in skill_file.read_text(encoding="utf-8")
    ]

    assert missing == [], f"SKILL.md files missing an MCP Tools section: {missing}"
