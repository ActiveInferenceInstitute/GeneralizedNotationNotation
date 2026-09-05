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
  ``CHANGELOG.md``, and direct filesystem counts). If a source surface changes,
  re-running this producer changes the manuscript.
* **Counts are over tracked files.** Filesystem counters consult ``git ls-files``
  and ignore anything untracked, so a clean clone reproduces the same numbers
  (see :func:`_tracked_files`). ``src/tests/`` is counted once, by the test
  tokens, and is excluded from the source-file and LOC tokens.
* **Deterministic.** No timestamps, no wall-clock, no randomness. Two runs over an
  unchanged tree produce byte-identical JSON. The only environment-derived token is
  the current git commit (stable within a checkout).
* **Dependency-light.** Standard library + ``yaml`` only (already a GNN dependency).
  No import of the heavy pipeline packages, so the producer runs even when optional
  simulation backends are absent.

The thin orchestrator ``scripts/z_generate_manuscript_variables.py`` wires
:func:`generate_variables` to the template's manuscript hydration.

Round-trip helpers: :func:`save_variables` persists a token map,
:func:`load_variables` reads and validates one back, and
:func:`token_checksum` fingerprints a map's canonical JSON so manuscript
tooling can detect token drift without diffing whole documents.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path

try:  # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - exercised only on <3.11
    _toml = None  # type: ignore[assignment]

try:
    import yaml as _yaml
except ModuleNotFoundError:  # pragma: no cover - yaml is a GNN dependency
    _yaml = None

__all__ = [
    "generate_variables",
    "load_variables",
    "save_variables",
    "token_checksum",
]

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


def _tracked_files(project_root: Path) -> set[Path] | None:
    """Return the repo-relative paths git tracks, or ``None`` when unavailable.

    Every filesystem-derived token is counted over *tracked* files rather than
    the working tree, so a clean ``git clone`` reproduces the same numbers a
    developer sees locally. Untracked scratch files, build artifacts and
    half-finished modules must not move a published count. When git is not
    available (a source tarball, a vendored copy) the counters fall back to the
    working tree and the values become checkout-dependent.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - env dependent
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    return {Path(entry) for entry in result.stdout.split("\0") if entry}


def _is_counted(path: Path, project_root: Path, tracked: set[Path] | None) -> bool:
    """True when *path* participates in a published count."""
    if _is_excluded(path):
        return False
    if tracked is None:
        return True
    try:
        rel = path.resolve().relative_to(project_root)
    except ValueError:  # pragma: no cover - defensive
        return False
    return rel in tracked


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


def _count_python(
    src_dir: Path, project_root: Path, tracked: set[Path] | None
) -> tuple[int, int]:
    """Return ``(file_count, total_lines)`` for authored non-test ``.py``.

    ``src/tests/`` is excluded: the test suite has its own tokens
    (``GNN_TEST_FILE_COUNT`` / ``GNN_TEST_FUNCTION_COUNT``) and counting it here
    too would report every test file twice in the manuscript.
    """
    tests_dir = src_dir / "tests"
    files = 0
    lines = 0
    for py in src_dir.rglob("*.py"):
        if not _is_counted(py, project_root, tracked):
            continue
        if tests_dir in py.parents:
            continue
        files += 1
        lines += sum(1 for _ in py.open("r", encoding="utf-8", errors="ignore"))
    return files, lines


def _count_packages(
    src_dir: Path, project_root: Path, tracked: set[Path] | None
) -> int:
    """Count top-level importable packages under ``src/`` (excluding ``tests``)."""
    return sum(
        1
        for child in src_dir.iterdir()
        if child.is_dir()
        and child.name != "tests"
        and _is_counted(child / "__init__.py", project_root, tracked)
        and (child / "__init__.py").is_file()
    )


def _count_test_functions(
    tests_dir: Path, project_root: Path, tracked: set[Path] | None
) -> tuple[int, int]:
    """Return ``(test_file_count, test_function_count)`` via static text scan."""
    file_count = 0
    func_count = 0
    pattern = re.compile(r"^\s*(?:async\s+)?def (test_\w+)", re.MULTILINE)
    for py in tests_dir.rglob("test_*.py"):
        if not _is_counted(py, project_root, tracked):
            continue
        file_count += 1
        func_count += len(
            pattern.findall(py.read_text(encoding="utf-8", errors="ignore"))
        )
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


def _backends(project_root: Path) -> list[tuple[str, str, bool]]:
    """Return ``(key, display_name, supports_execution)`` from the registry.

    The registry literal is parsed with :mod:`ast` rather than imported, so the
    producer keeps working when the heavy render package's dependencies are
    absent. ``supports_execution`` is the registry's own field: it separates the
    render targets from the subset that also has a Step-12 executor.
    """
    registry = project_root / "src" / "render" / "framework_registry.py"
    if not registry.is_file():
        return []
    text = registry.read_text(encoding="utf-8")
    try:
        module = ast.parse(text)
    except SyntaxError:  # pragma: no cover - defensive
        module = None
    if module is not None:
        for node in module.body:
            targets: list[ast.expr]
            if isinstance(node, ast.AnnAssign):
                targets = [node.target]
            elif isinstance(node, ast.Assign):
                targets = list(node.targets)
            else:
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "FRAMEWORK_REGISTRY"
                for target in targets
            ):
                continue
            value = node.value
            if value is None:  # pragma: no cover - bare annotation
                continue
            if isinstance(value, ast.Call) and value.args:
                value = value.args[0]
            try:
                data = ast.literal_eval(value)
            except (ValueError, TypeError):  # pragma: no cover - defensive
                break
            if isinstance(data, dict):
                return [
                    (
                        str(key),
                        str(spec.get("name", key)),
                        bool(spec.get("supports_execution", False)),
                    )
                    for key, spec in data.items()
                    if isinstance(spec, dict)
                ]
            break
    # Fallback: text scan (execution support unknown, reported as False).
    return [
        (block.group("key"), block.group("name"), False)
        for block in re.finditer(
            r'"(?P<key>[a-z_]+)"\s*:\s*\{[^{}]*?"name"\s*:\s*"(?P<name>[^"]+)"',
            text,
            re.DOTALL,
        )
    ]


def _mcp_counts(project_root: Path, tracked: set[Path] | None) -> dict[str, int]:
    """MCP tool/module counts.

    ``tools``/``modules`` are read from the project's maintained MCP audit ledger
    (``src/mcp/audit_report.json``), which is regenerated by
    ``PYTHONPATH=src python src/mcp/validate_tools.py`` (``just mcp-ledger``) by
    actually loading every MCP module and counting registered tools — a count
    that cannot be reproduced by static text scanning. This is a *source ledger*
    (one of the manuscript's allowed evidence types), and it is only as current
    as the committed file: ``src/tests/mcp/test_mcp_audit.py`` fails when the
    ledger drifts from the live registry. The ``files`` count is recomputed live
    from the filesystem each run.
    """
    audit = project_root / "src" / "mcp" / "audit_report.json"
    counts = {"tools": 0, "modules_total": 0, "modules_loaded": 0, "files": 0}
    if audit.is_file():
        data = json.loads(audit.read_text(encoding="utf-8"))
        counts["tools"] = int(data.get("tools_total", 0))
        counts["modules_total"] = int(data.get("modules_total", 0))
        counts["modules_loaded"] = int(data.get("modules_loaded", 0))
    counts["files"] = _count_files(
        project_root / "src", "mcp.py", project_root, tracked
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


def _count_files(
    root: Path, pattern: str, project_root: Path, tracked: set[Path] | None
) -> int:
    return sum(1 for p in root.rglob(pattern) if _is_counted(p, project_root, tracked))


def _count_example_models(project_root: Path, tracked: set[Path] | None) -> int:
    """Count GNN example *models* under ``input/gnn_files``.

    A GNN model file is identified by its mandatory ``## GNNSection`` header, so
    the corpus README/AGENTS/INDEX documents that live alongside the models are
    not counted as models.
    """
    corpus = project_root / "input" / "gnn_files"
    if not corpus.is_dir():
        return 0
    count = 0
    for md in corpus.rglob("*.md"):
        if not _is_counted(md, project_root, tracked):
            continue
        if "## GNNSection" in md.read_text(encoding="utf-8", errors="ignore"):
            count += 1
    return count


def _release_metadata(project_root: Path) -> list[tuple[str, str, str]]:
    """Parse ``CHANGELOG.md`` into ``(version, date, codename)`` triples.

    Each release heading (``## [3.2.0] — 2026-09-02``) is followed by a
    block-quoted summary whose leading bold run is the release codename
    (``> **Exemplar Gold Standard.** ...``). Newest release first.
    """
    changelog = project_root / "CHANGELOG.md"
    if not changelog.is_file():
        return []
    lines = changelog.read_text(encoding="utf-8").splitlines()
    heading = re.compile(r"^##\s+\[(?P<version>[^\]]+)\]\s*[—-]\s*(?P<date>[0-9-]+)")
    codename = re.compile(r"^>\s*\*\*(?P<codename>[^*]+?)\.?\*\*")
    releases: list[tuple[str, str, str]] = []
    for index, line in enumerate(lines):
        match = heading.match(line)
        if not match:
            continue
        name = ""
        for follow in lines[index + 1 : index + 6]:
            found = codename.match(follow)
            if found:
                name = found.group("codename").strip()
                break
        releases.append((match.group("version"), match.group("date"), name))
    return releases


def _caption(text: str, label: str) -> str:
    """Return a pandoc-crossref table caption line.

    The caption must be the line immediately after the table with no blank line
    between them; ``{#tbl:label}`` is what makes the table numbered and
    referenceable as ``[@tbl:label]``.
    """
    return f": {text} {{#tbl:{label}}}"


def _render_step_table(steps: list[tuple[int, str]], purposes: dict[int, str]) -> str:
    """Render the per-step markdown table consumed by the manuscript."""
    rows = ["| Step | Module | Purpose |", "|---:|---|---|"]
    for number, script in steps:
        purpose = purposes.get(number) or _humanize_step(script)
        rows.append(f"| {number} | `{script}` | {purpose} |")
    rows.append(
        _caption(
            "The pipeline steps, their thin orchestrator modules, and their "
            "purposes, read from `src/STEP_INDEX.md`.",
            "pipeline_steps",
        )
    )
    return "\n".join(rows)


def _render_family_table(families: list[dict]) -> str:
    """Render the model-family markdown table."""
    rows = ["| Family | Frameworks | Description |", "|---|---|---|"]
    for fam in families:
        name = fam.get("name", "?")
        frameworks = str(fam.get("frameworks", "")).replace(",", ", ")
        desc = fam.get("description", "")
        rows.append(f"| `{name}` | {frameworks} | {desc} |")
    rows.append(
        _caption(
            "Model families declared in `input/model_family_manifest.json` and "
            "the frameworks each family targets.",
            "model_families",
        )
    )
    return "\n".join(rows)


def _render_backend_table(backends: list[tuple[str, str, bool]]) -> str:
    """Render the backend registry markdown table."""
    rows = ["| Registry key | Backend | Executes |", "|---|---|---|"]
    for key, name, executes in backends:
        rows.append(f"| `{key}` | {name} | {'yes' if executes else 'render-only'} |")
    rows.append(
        _caption(
            "Render targets in `src/render/framework_registry.py`. The "
            "*Executes* column is the registry's own `supports_execution` flag: "
            "a render-only backend has no Step-12 executor.",
            "backend_registry",
        )
    )
    return "\n".join(rows)


def _cross_framework_selection(
    families: list[dict], backends: list[tuple[str, str, bool]]
) -> tuple[str, str, int]:
    """Resolve the cross-framework family, its backend names, and their count.

    The cross-framework family is the one the manifest marks
    ``"cross_framework": true``; when no family declares the flag the widest
    family (most frameworks) is used, so the selection cannot silently move to a
    different family just because the manifest's ordering changed. Returns
    ``("", "", 0)`` when no family lists more than one framework.
    """
    multi = [f for f in families if "," in str(f.get("frameworks", ""))]
    if not multi:
        return "", "", 0
    flagged = [f for f in multi if f.get("cross_framework") is True]
    if flagged:
        cross_family = flagged[0]
    else:
        cross_family = max(
            multi, key=lambda f: len(str(f.get("frameworks", "")).split(","))
        )
    cross_family_name = cross_family.get("name", "")
    keys = [
        k.strip()
        for k in str(cross_family.get("frameworks", "")).split(",")
        if k.strip()
    ]
    name_by_key = {key: name for key, name, _ in backends}
    cross_backends = ", ".join(name_by_key.get(k, k) for k in keys)
    return cross_family_name, cross_backends, len(keys)


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

    tracked = _tracked_files(project_root)
    version = _read_pyproject_version(project_root)
    steps = _pipeline_steps(src_dir)
    purposes = _step_purposes(src_dir)
    backends = _backends(project_root)
    mcp = _mcp_counts(project_root, tracked)
    families = _families(project_root)
    py_files, py_loc = _count_python(src_dir, project_root, tracked)
    package_count = _count_packages(src_dir, project_root, tracked)
    test_file_count, test_func_count = _count_test_functions(
        src_dir / "tests", project_root, tracked
    )

    # --- Derived tables (multi-line tokens) ------------------------------------
    step_table = _render_step_table(steps, purposes)
    family_table = _render_family_table(families)
    backend_table = _render_backend_table(backends)
    (
        cross_family_name,
        cross_backends,
        cross_backend_count,
    ) = _cross_framework_selection(families, backends)

    family_names = [f.get("name", "?") for f in families]
    backend_names = [name for _, name, _ in backends]
    executable_backends = [name for _, name, executes in backends if executes]

    # Per-step scalars keyed by module stem, so prose never hand-types a step
    # number the registry owns (`Parsing ({{GNN_STEP_GNN}})`).
    step_tokens = {
        f"GNN_STEP_{re.sub(r'^[0-9]+_', '', script.removesuffix('.py')).upper()}": str(
            number
        )
        for number, script in steps
    }

    releases = _release_metadata(project_root)
    release_by_version = {ver: (date, name) for ver, date, name in releases}
    release_date, release_codename = release_by_version.get(version, ("", ""))
    # The release that introduced the long-running orchestration contracts is the
    # oldest whose CHANGELOG codename names orchestration.
    orchestration_version = next(
        (ver for ver, _, name in reversed(releases) if "orchestration" in name.lower()),
        version,
    )

    example_count = _count_example_models(project_root, tracked)
    corpus_dir = project_root / "input" / "gnn_files"
    family_dir_count = (
        sum(1 for c in corpus_dir.iterdir() if c.is_dir() and not _is_excluded(c))
        if corpus_dir.is_dir()
        else 0
    )
    family_target_dirs = {
        str(f.get("target_dir", "")) for f in families if f.get("target_dir")
    }
    figure_count = _count_files(project_root / "output", "*.png", project_root, tracked)
    manuscript_figure_count = _count_files(
        project_root / "output" / "figures", "*.png", project_root, tracked
    )
    doc_file_count = _count_files(project_root / "doc", "*.md", project_root, tracked)

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
        "GNN_RELEASE_CODENAME": release_codename,
        "GNN_RELEASE_DATE": release_date,
        "GNN_ORCHESTRATION_VERSION": orchestration_version,
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
        # Directory census under input/gnn_files/ — NOT the same set as the
        # manifest families (some corpus dirs declare no family; some families
        # point outside input/gnn_files/). GNN_FAMILY_TARGET_DIR_COUNT is the
        # manifest-derived figure and is the one that pairs with
        # GNN_FAMILY_COUNT.
        "GNN_INPUT_FAMILY_DIR_COUNT": str(family_dir_count),
        "GNN_FAMILY_TARGET_DIR_COUNT": str(len(family_target_dirs)),
        # Backends
        "GNN_BACKEND_COUNT": str(len(backends)),
        "GNN_BACKEND_LIST": ", ".join(backend_names),
        "GNN_BACKEND_TABLE": backend_table,
        "GNN_EXECUTABLE_BACKEND_COUNT": str(len(executable_backends)),
        "GNN_EXECUTABLE_BACKEND_LIST": ", ".join(executable_backends),
        "GNN_CROSS_FRAMEWORK_FAMILY": cross_family_name,
        "GNN_CROSS_FRAMEWORK_BACKENDS": cross_backends,
        "GNN_CROSS_FRAMEWORK_BACKEND_COUNT": str(cross_backend_count),
        # Generated artifacts. GNN_OUTPUT_ARTIFACT_FIGURE_COUNT is a census of
        # every PNG committed under output/ across all pipeline steps and many
        # runs; GNN_MANUSCRIPT_FIGURE_COUNT is only the manuscript's own figures
        # in output/figures/. GNN_OUTPUT_FIGURE_COUNT is the historical name of
        # the census and is retained for sections that still reference it.
        "GNN_OUTPUT_FIGURE_COUNT": str(figure_count),
        "GNN_OUTPUT_ARTIFACT_FIGURE_COUNT": str(figure_count),
        "GNN_MANUSCRIPT_FIGURE_COUNT": str(manuscript_figure_count),
    }
    variables.update(step_tokens)
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


def load_variables(in_path: Path) -> dict[str, str]:
    """Read back a token map written by :func:`save_variables`.

    Args:
        in_path: Path to a ``manuscript_variables.json`` file.

    Returns:
        The validated flat ``{TOKEN: value}`` mapping.

    Raises:
        ValueError: If the file is not a flat JSON object of string values.
    """
    in_path = Path(in_path)
    data = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in data.items()
    ):
        raise ValueError(
            f"{in_path} is not a flat {{TOKEN: value}} JSON map written by save_variables"
        )
    return data


def token_checksum(variables: Mapping[str, str]) -> str:
    """Return the sha256 checksum of a token map's canonical JSON form.

    The canonical form matches :func:`save_variables` output (sorted keys,
    2-space indent, UTF-8) without the trailing newline, so a checksum over
    :func:`load_variables` output equals one over freshly generated variables
    for an unchanged tree.
    """
    canonical = json.dumps(variables, indent=2, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
