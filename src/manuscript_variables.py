"""Deterministic manuscript-variable producer for GeneralizedNotationNotation.

This module introspects the *live* GNN repository and emits a flat
``dict[str, str]`` of ``{{UPPERCASE_KEY}}`` token values consumed by the
docxology/template manuscript renderer
(``infrastructure.rendering.manuscript_injection``).

Design contract
---------------
* **Nothing is hard-coded.** Every quantitative token is computed from a source
  surface in the repository (``pyproject.toml``, ``input/model_family_manifest.json``,
  ``src/render/framework_registry.py``, ``src/mcp/audit_report.json``, ``src/STEP_INDEX.md``,
  and direct filesystem counts). If a source surface changes, re-running this
  producer changes the manuscript.
* **Deterministic.** No timestamps, no wall-clock, no randomness. Two runs over an
  unchanged tree produce byte-identical JSON. The only environment-derived token is
  the current git commit (stable within a checkout).
* **Dependency-light.** Standard library + ``yaml`` only (already a GNN dependency).
  No import of the heavy pipeline packages, so the producer runs even when optional
  simulation backends are absent.

The thin orchestrator ``scripts/z_generate_manuscript_variables.py`` wires
:func:`generate_variables` to the template's manuscript hydration.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

try:  # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - exercised only on <3.11
    _toml = None  # type: ignore[assignment]

try:
    import yaml as _yaml
except ModuleNotFoundError:  # pragma: no cover - yaml is a GNN dependency
    _yaml = None  # type: ignore[assignment]

__all__ = ["generate_variables", "save_variables"]

# Directories that are NOT counted as authored source when walking ``src/``.
_EXCLUDED_DIR_PARTS = {
    "__pycache__",
    ".venv",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
}


def _is_excluded(path: Path) -> bool:
    return any(part in _EXCLUDED_DIR_PARTS for part in path.parts)


def _read_pyproject_version(project_root: Path) -> str:
    pyproject = project_root / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    if _toml is not None:
        data = _toml.loads(text)
        version = data.get("project", {}).get("version")
        if version:
            return str(version)
    # Fallback: regex the [project] version line.
    match = re.search(r'^\s*version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else "0.0.0"


def _load_config(project_root: Path) -> dict:
    config_path = project_root / "manuscript" / "config.yaml"
    if not config_path.is_file() or _yaml is None:
        return {}
    loaded = _yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _count_python(src_dir: Path) -> tuple[int, int]:
    """Return ``(file_count, total_lines)`` for authored ``.py`` under *src_dir*."""
    files = 0
    lines = 0
    for py in src_dir.rglob("*.py"):
        if _is_excluded(py):
            continue
        files += 1
        lines += sum(1 for _ in py.open("r", encoding="utf-8", errors="ignore"))
    return files, lines


def _count_packages(src_dir: Path) -> int:
    return sum(
        1
        for child in src_dir.iterdir()
        if child.is_dir() and not _is_excluded(child) and (child / "__init__.py").is_file()
    )


def _count_test_functions(tests_dir: Path) -> tuple[int, int]:
    """Return ``(test_file_count, test_function_count)`` via static text scan."""
    file_count = 0
    func_count = 0
    pattern = re.compile(r"^\s*(?:async\s+)?def (test_\w+)", re.MULTILINE)
    for py in tests_dir.rglob("test_*.py"):
        if _is_excluded(py):
            continue
        file_count += 1
        func_count += len(pattern.findall(py.read_text(encoding="utf-8", errors="ignore")))
    return file_count, func_count


def _pipeline_steps(src_dir: Path) -> list[tuple[int, str]]:
    """Return sorted ``(step_number, script_name)`` for ``N_*.py`` step modules."""
    steps: list[tuple[int, str]] = []
    for py in src_dir.glob("[0-9]*_*.py"):
        match = re.match(r"(\d+)_", py.name)
        if match:
            steps.append((int(match.group(1)), py.name))
    return sorted(steps)


def _step_purposes(src_dir: Path) -> dict[int, str]:
    """Parse ``src/STEP_INDEX.md`` master table for per-step purposes."""
    index = src_dir / "STEP_INDEX.md"
    purposes: dict[int, str] = {}
    if not index.is_file():
        return purposes
    for line in index.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.split("|")]
        # Master table rows look like: | 0 | `0_template.py` | template/ | Global | Purpose | ...
        if len(cells) >= 7 and cells[1].isdigit():
            purposes[int(cells[1])] = cells[5]
    return purposes


def _humanize_step(script_name: str) -> str:
    stem = script_name.removesuffix(".py")
    stem = re.sub(r"^\d+_", "", stem)
    return stem.replace("_", " ").title()


def _backends(project_root: Path) -> list[tuple[str, str]]:
    """Return ``(key, display_name)`` backend pairs from the framework registry."""
    registry = project_root / "src" / "render" / "framework_registry.py"
    pairs: list[tuple[str, str]] = []
    if registry.is_file():
        text = registry.read_text(encoding="utf-8")
        # Match `    "pymdp": {  ... "name": "PyMDP",` blocks.
        for block in re.finditer(
            r'"(?P<key>[a-z_]+)"\s*:\s*\{[^{}]*?"name"\s*:\s*"(?P<name>[^"]+)"',
            text,
            re.DOTALL,
        ):
            pairs.append((block.group("key"), block.group("name")))
    return pairs


def _mcp_counts(project_root: Path) -> dict[str, int]:
    """MCP tool/module counts.

    ``tools``/``modules`` are read from the project's maintained MCP audit ledger
    (``src/mcp/audit_report.json``), which the test suite regenerates by actually
    loading every MCP module and counting registered tools — a count that cannot be
    reproduced by static text scanning. This is a *source ledger* (one of the
    manuscript's allowed evidence types), as current as the committed ledger. The
    ``files`` count is recomputed live from the filesystem each run.
    """
    audit = project_root / "src" / "mcp" / "audit_report.json"
    counts = {"tools": 0, "modules_total": 0, "modules_loaded": 0, "files": 0}
    if audit.is_file():
        data = json.loads(audit.read_text(encoding="utf-8"))
        counts["tools"] = int(data.get("tools_total", 0))
        counts["modules_total"] = int(data.get("modules_total", 0))
        counts["modules_loaded"] = int(data.get("modules_loaded", 0))
    counts["files"] = sum(
        1 for p in (project_root / "src").rglob("mcp.py") if not _is_excluded(p)
    )
    return counts


def _families(project_root: Path) -> list[dict]:
    manifest = project_root / "input" / "model_family_manifest.json"
    if not manifest.is_file():
        return []
    data = json.loads(manifest.read_text(encoding="utf-8"))
    families = data.get("families", [])
    return families if isinstance(families, list) else []


def _git_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return "unknown"


def _count_files(root: Path, pattern: str) -> int:
    return sum(1 for p in root.rglob(pattern) if not _is_excluded(p))


def generate_variables(project_root: Path) -> dict[str, str]:
    """Compute the manuscript token map by introspecting the live repository.

    Args:
        project_root: Path to the GeneralizedNotationNotation project root.

    Returns:
        Flat ``dict[str, str]`` of ``UPPERCASE_KEY`` -> value. All values are
        strings (the template injector substitutes text). The mapping is
        deterministic for an unchanged working tree.
    """
    project_root = Path(project_root).resolve()
    src_dir = project_root / "src"
    config = _load_config(project_root)
    paper = config.get("paper", {}) if isinstance(config, dict) else {}
    authors = config.get("authors", []) if isinstance(config, dict) else []
    author = authors[0] if authors else {}
    publication = config.get("publication", {}) if isinstance(config, dict) else {}
    metadata = config.get("metadata", {}) if isinstance(config, dict) else {}
    keywords = config.get("keywords", []) if isinstance(config, dict) else []

    version = _read_pyproject_version(project_root)
    steps = _pipeline_steps(src_dir)
    purposes = _step_purposes(src_dir)
    backends = _backends(project_root)
    mcp = _mcp_counts(project_root)
    families = _families(project_root)
    py_files, py_loc = _count_python(src_dir)
    package_count = _count_packages(src_dir)
    test_file_count, test_func_count = _count_test_functions(src_dir / "tests")

    # --- Derived tables (multi-line tokens) ------------------------------------
    step_rows = ["| Step | Module | Purpose |", "|---:|---|---|"]
    for number, script in steps:
        purpose = purposes.get(number) or _humanize_step(script)
        step_rows.append(f"| {number} | `{script}` | {purpose} |")
    step_table = "\n".join(step_rows)

    family_rows = ["| Family | Frameworks | Description |", "|---|---|---|"]
    for fam in families:
        name = fam.get("name", "?")
        frameworks = str(fam.get("frameworks", "")).replace(",", ", ")
        desc = fam.get("description", "")
        family_rows.append(f"| `{name}` | {frameworks} | {desc} |")
    family_table = "\n".join(family_rows)

    backend_rows = ["| Registry key | Backend |", "|---|---|"]
    for key, name in backends:
        backend_rows.append(f"| `{key}` | {name} |")
    backend_table = "\n".join(backend_rows)

    # Cross-framework family = family whose manifest lists >1 framework.
    cross_family = next(
        (f for f in families if "," in str(f.get("frameworks", ""))), None
    )
    cross_backends = ""
    cross_family_name = ""
    if cross_family:
        cross_family_name = cross_family.get("name", "")
        keys = [k.strip() for k in str(cross_family.get("frameworks", "")).split(",")]
        name_by_key = dict(backends)
        cross_backends = ", ".join(name_by_key.get(k, k) for k in keys)

    family_names = [f.get("name", "?") for f in families]
    backend_names = [name for _, name in backends]

    example_count = _count_files(project_root / "input" / "gnn_files", "*.md")
    family_dir_count = sum(
        1
        for c in (project_root / "input" / "gnn_files").iterdir()
        if c.is_dir() and not _is_excluded(c)
    ) if (project_root / "input" / "gnn_files").is_dir() else 0
    figure_count = _count_files(project_root / "output", "*.png")
    doc_file_count = _count_files(project_root / "doc", "*.md")

    variables: dict[str, str] = {
        # Identity / config
        "GNN_TITLE": str(paper.get("title", "GeneralizedNotationNotation")),
        "GNN_SUBTITLE": str(paper.get("subtitle", "")),
        "GNN_VERSION": version,
        "GNN_FIRST_AUTHOR": str(author.get("name", "")),
        "GNN_AUTHOR_ORCID": str(author.get("orcid", "")),
        "GNN_AUTHOR_AFFILIATION": str(author.get("affiliation", "")),
        "GNN_AUTHOR_EMAIL": str(author.get("email", "")),
        "GNN_PUBLICATION_YEAR": str(publication.get("year", "")),
        "GNN_REPO_URL": str(publication.get("github_repository", "")),
        "GNN_LICENSE": str(metadata.get("license", "")),
        "GNN_KEYWORDS": ", ".join(str(k) for k in keywords),
        "GNN_GIT_COMMIT": _git_commit(project_root),
        # Pipeline structure
        "GNN_STEP_COUNT": str(len(steps)),
        "GNN_STEP_FIRST": str(steps[0][0]) if steps else "0",
        "GNN_STEP_LAST": str(steps[-1][0]) if steps else "0",
        "GNN_STEP_RANGE": f"{steps[0][0]}–{steps[-1][0]}" if steps else "0",
        "GNN_STEP_TABLE": step_table,
        # Source surface
        "GNN_SRC_PACKAGE_COUNT": str(package_count),
        "GNN_SRC_PY_FILE_COUNT": str(py_files),
        "GNN_SRC_LOC": str(py_loc),
        "GNN_DOC_FILE_COUNT": str(doc_file_count),
        # MCP
        "GNN_MCP_TOOL_COUNT": str(mcp["tools"]),
        "GNN_MCP_MODULE_COUNT": str(mcp["modules_total"]),
        "GNN_MCP_MODULE_LOADED": str(mcp["modules_loaded"]),
        "GNN_MCP_FILE_COUNT": str(mcp["files"]),
        # Tests
        "GNN_TEST_FILE_COUNT": str(test_file_count),
        "GNN_TEST_FUNCTION_COUNT": str(test_func_count),
        # Model families / corpora
        "GNN_FAMILY_COUNT": str(len(families)),
        "GNN_FAMILY_LIST": ", ".join(family_names),
        "GNN_FAMILY_TABLE": family_table,
        "GNN_EXAMPLE_COUNT": str(example_count),
        "GNN_INPUT_FAMILY_DIR_COUNT": str(family_dir_count),
        # Backends
        "GNN_BACKEND_COUNT": str(len(backends)),
        "GNN_BACKEND_LIST": ", ".join(backend_names),
        "GNN_BACKEND_TABLE": backend_table,
        "GNN_CROSS_FRAMEWORK_FAMILY": cross_family_name,
        "GNN_CROSS_FRAMEWORK_BACKENDS": cross_backends,
        # Generated artifacts
        "GNN_OUTPUT_FIGURE_COUNT": str(figure_count),
    }
    return variables


def save_variables(variables: dict[str, str], out_path: Path) -> Path:
    """Write *variables* to *out_path* as sorted, deterministic JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(variables, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out_path
