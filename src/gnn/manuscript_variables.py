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
* **Counts describe one commit.** Every source surface is read from the git
  commit named by ``GNN_GIT_COMMIT`` (``HEAD``), not from the working tree: the
  file set comes from ``git ls-tree`` and the bytes come from ``git cat-file``
  (see :class:`RepositorySnapshot`). A tracked-but-modified file therefore
  cannot inflate a published count, a tracked-but-deleted file cannot silently
  vanish from one, and any checkout of that commit reproduces the same numbers.
  When git is unavailable (a source tarball, a vendored copy) the snapshot falls
  back to the working tree and ``GNN_GIT_COMMIT`` reports ``unknown``, so the
  provenance of the numbers is always visible in the token map itself.
  ``src/tests/`` is counted once, by the test tokens, and is excluded from the
  source-file and LOC tokens.
* **Deterministic.** No timestamps, no wall-clock, no randomness. Two runs over an
  unchanged commit produce byte-identical JSON.
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
from collections.abc import Mapping, Sequence
from fnmatch import fnmatch
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
    "RepositorySnapshot",
    "generate_variables",
    "load_variables",
    "save_variables",
    "config_metadata_drift",
    "select_cross_framework_family",
    "sync_config_metadata",
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


class RepositorySnapshot:
    """Read-only view of the repository at one commit.

    Every published count is computed from a snapshot, so the numbers the
    manuscript prints describe exactly the commit ``GNN_GIT_COMMIT`` names
    rather than whatever happens to be on disk. Two properties follow that the
    older ``git ls-files`` + working-tree-read approach did not have:

    * a tracked file with uncommitted edits contributes its *committed* bytes,
      so a dirty tree cannot move a published line count;
    * a tracked file deleted in the working tree is still counted, so a
      half-finished refactor cannot silently shrink one.

    ``commit`` is the resolved short SHA, or ``"unknown"`` when git is not
    available (a source tarball, a vendored copy). In that case the snapshot
    degrades to the working tree and the ``unknown`` commit token is the signal
    that the numbers are checkout-dependent.
    """

    def __init__(self, project_root: Path, revision: str = "HEAD") -> None:
        self.project_root = Path(project_root).resolve()
        self.commit = self._resolve_commit(revision)
        self._paths: frozenset[Path] | None = None
        self._cache: dict[Path, str] = {}
        self._revision = revision if self.commit != "unknown" else ""

    # -- construction helpers ------------------------------------------------
    def _git(self, *args: str) -> subprocess.CompletedProcess[bytes] | None:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=self.project_root,
                capture_output=True,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):  # pragma: no cover - env
            return None

    def _resolve_commit(self, revision: str) -> str:
        result = self._git("rev-parse", "--short", revision)
        if result is None or result.returncode != 0:
            return "unknown"
        return result.stdout.decode("utf-8", "replace").strip() or "unknown"

    # -- public surface ------------------------------------------------------
    @property
    def from_git(self) -> bool:
        """True when the snapshot reads committed blobs rather than the disk."""
        return bool(self._revision)

    def files(self) -> frozenset[Path]:
        """Repo-relative paths present in the snapshot."""
        if self._paths is not None:
            return self._paths
        paths: set[Path] = set()
        if self._revision:
            result = self._git("ls-tree", "-r", "-z", "--name-only", self._revision)
            if result is not None and result.returncode == 0:
                for entry in result.stdout.decode("utf-8", "replace").split("\0"):
                    if entry:
                        paths.add(Path(entry))
        if not paths:
            # Working-tree fallback: walk the checkout, minus build/vcs noise.
            for path in self.project_root.rglob("*"):
                if not path.is_file() or _is_excluded(path):
                    continue
                paths.add(path.relative_to(self.project_root))
        self._paths = frozenset(paths)
        return self._paths

    def exists(self, rel: Path | str) -> bool:
        """True when *rel* is present in the snapshot."""
        return Path(rel) in self.files()

    def prefetch(self, rels: Sequence[Path]) -> None:
        """Read many blobs in one ``git cat-file --batch`` call.

        Counting the source tree one ``git show`` per file costs a subprocess
        per file; the batch protocol makes the whole snapshot one process.
        """
        if not self._revision:
            return
        wanted = [rel for rel in rels if rel not in self._cache]
        if not wanted:
            return
        payload = "".join(f"{self._revision}:{rel.as_posix()}\n" for rel in wanted)
        try:
            result = subprocess.run(
                ["git", "cat-file", "--batch"],
                cwd=self.project_root,
                input=payload.encode("utf-8"),
                capture_output=True,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):  # pragma: no cover - env
            return
        if result.returncode != 0:
            return
        out = result.stdout
        pos = 0
        for rel in wanted:
            newline = out.find(b"\n", pos)
            if newline < 0:
                break
            header = out[pos:newline].decode("utf-8", "replace")
            pos = newline + 1
            parts = header.rsplit(" ", 2)
            if len(parts) != 3 or not parts[2].isdigit():
                # "<name> missing" — record the absence and keep parsing.
                self._cache[rel] = ""
                continue
            size = int(parts[2])
            self._cache[rel] = out[pos : pos + size].decode("utf-8", "replace")
            pos += size + 1

    def read_text(self, rel: Path | str) -> str:
        """Return the snapshot's bytes for *rel* decoded as UTF-8 (lossy)."""
        rel = Path(rel)
        cached = self._cache.get(rel)
        if cached is not None:
            return cached
        data = ""
        if self._revision:
            result = self._git("show", f"{self._revision}:{rel.as_posix()}")
            if result is not None and result.returncode == 0:
                data = result.stdout.decode("utf-8", "replace")
        else:
            candidate = self.project_root / rel
            if candidate.is_file():
                data = candidate.read_text(encoding="utf-8", errors="replace")
        self._cache[rel] = data
        return data

    def glob(self, prefix: str, pattern: str) -> list[Path]:
        """Snapshot equivalent of ``(project_root / prefix).rglob(pattern)``.

        Only the file *name* is matched against *pattern*, which is exactly what
        every call site needs (``*.py``, ``*.png``, ``test_*.py``, ``mcp.py``).
        """
        base = Path(prefix)
        return sorted(
            rel
            for rel in self.files()
            if base in rel.parents and fnmatch(rel.name, pattern)
        )


def _line_count(text: str) -> int:
    """Lines in *text*: newline characters plus a final unterminated line."""
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def _read_pyproject_version(snapshot: RepositorySnapshot) -> str:
    text = snapshot.read_text("pyproject.toml")
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


def _count_python(snapshot: RepositorySnapshot) -> tuple[int, int]:
    """Return ``(file_count, total_lines)`` for package ``.py`` sources.

    The test suite lives outside ``src/`` and has its own tokens
    (``GNN_TEST_FILE_COUNT`` / ``GNN_TEST_FUNCTION_COUNT``), so it is never
    counted here. Both the file set
    and the bytes come from the snapshot's commit, so the pair reproduces from
    that commit alone.
    """
    sources = list(snapshot.glob("src", "*.py"))
    snapshot.prefetch(sources)
    files = 0
    lines = 0
    for py in sources:
        files += 1
        lines += _line_count(snapshot.read_text(py))
    return files, lines


def _count_packages(snapshot: RepositorySnapshot) -> int:
    """Count importable subpackages under ``src/gnn/``."""
    return len(
        {
            rel.parts[2]
            for rel in snapshot.glob("src/gnn", "__init__.py")
            if len(rel.parts) == 4
        }
    )


def _count_test_functions(snapshot: RepositorySnapshot) -> tuple[int, int]:
    """Return ``(test_file_count, test_function_count)`` via static text scan."""
    file_count = 0
    func_count = 0
    pattern = re.compile(r"^\s*(?:async\s+)?def (test_\w+)", re.MULTILINE)
    tests = snapshot.glob("tests", "test_*.py")
    snapshot.prefetch(tests)
    for py in tests:
        file_count += 1
        func_count += len(pattern.findall(snapshot.read_text(py)))
    return file_count, func_count


def _pipeline_steps(snapshot: RepositorySnapshot) -> list[tuple[int, str]]:
    """Return sorted ``(step_number, script_name)`` for ``N_*.py`` step modules.

    Step modules are top-level files directly under ``src/gnn/`` — they are
    siblings of the source packages, never members of one, which is why
    ``GNN_STEP_COUNT`` and ``GNN_SRC_PACKAGE_COUNT`` count disjoint sets.
    """
    steps: list[tuple[int, str]] = []
    for rel in snapshot.glob("src/gnn", "[0-9]*_*.py"):
        if len(rel.parts) != 3:
            continue
        match = re.match(r"(\d+)_", rel.name)
        if match:
            steps.append((int(match.group(1)), rel.name))
    return sorted(steps)


def _step_purposes(snapshot: RepositorySnapshot) -> dict[int, str]:
    """Parse ``src/gnn/STEP_INDEX.md`` master table for per-step purposes."""
    purposes: dict[int, str] = {}
    for line in snapshot.read_text("src/gnn/STEP_INDEX.md").splitlines():
        cells = [c.strip() for c in line.split("|")]
        # Master table rows look like: | 0 | `0_template.py` | template/ | Global | Purpose | ...
        if len(cells) >= 7 and cells[1].isdigit():
            purposes[int(cells[1])] = cells[5]
    return purposes


def _humanize_step(script_name: str) -> str:
    stem = script_name.removesuffix(".py")
    stem = re.sub(r"^\d+_", "", stem)
    return stem.replace("_", " ").title()


def _module_literal(text: str, name: str) -> object | None:
    """Return the literal value assigned to *name* in *text*, or ``None``.

    The value is recovered with :mod:`ast` rather than by importing the module,
    so the producer keeps working when the heavy render/pipeline packages'
    dependencies are absent.
    """
    try:
        module = ast.parse(text)
    except SyntaxError:  # pragma: no cover - defensive
        return None
    for node in module.body:
        targets: list[ast.expr]
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.Assign):
            targets = list(node.targets)
        else:
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == name for target in targets
        ):
            continue
        value = node.value
        if value is None:  # pragma: no cover - bare annotation
            continue
        if isinstance(value, ast.Call) and value.args:
            value = value.args[0]
        try:
            literal: object = ast.literal_eval(value)
        except (ValueError, TypeError):  # pragma: no cover - defensive
            return None
        return literal
    return None


def _registry_specs(snapshot: RepositorySnapshot) -> dict[str, dict]:
    """Return ``FRAMEWORK_REGISTRY`` as ``{key: spec}`` from the snapshot.

    This is the single source of truth for every backend capability the
    manuscript reports: ``supports_execution`` (Step-12 executor present) and
    ``supports_continuous`` (linear-Gaussian semantics) are the registry's own
    fields, never restated by hand anywhere downstream.
    """
    text = snapshot.read_text("src/gnn/render/framework_registry.py")
    if not text:
        return {}
    data = _module_literal(text, "FRAMEWORK_REGISTRY")
    if isinstance(data, dict):
        return {str(key): spec for key, spec in data.items() if isinstance(spec, dict)}
    # Fallback: text scan (capability flags unknown, reported as False).
    return {
        block.group("key"): {"name": block.group("name")}
        for block in re.finditer(
            r'"(?P<key>[a-z_]+)"\s*:\s*\{[^{}]*?"name"\s*:\s*"(?P<name>[^"]+)"',
            text,
            re.DOTALL,
        )
    }


def _backends(snapshot: RepositorySnapshot) -> list[tuple[str, str, bool]]:
    """Return ``(key, display_name, supports_execution)`` from the registry.

    ``supports_execution`` is the registry's own field: it separates the render
    targets from the subset that also has a Step-12 executor.
    """
    return [
        (key, str(spec.get("name", key)), bool(spec.get("supports_execution", False)))
        for key, spec in _registry_specs(snapshot).items()
    ]


def _maintained_frameworks(snapshot: RepositorySnapshot) -> tuple[str, ...]:
    """Return ``MAINTAINED_FRAMEWORKS`` from the cross-framework gate.

    This is the set the reliability gate will actually profile:
    ``src/gnn/pipeline/cross_framework_reliability.py`` raises
    ``ValueError("Unprofiled frameworks: ...")`` for anything outside it. It is
    a strict subset of the registry, so no manuscript sentence may use
    ``GNN_BACKEND_COUNT`` to describe what the gate profiles.
    """
    text = snapshot.read_text("src/gnn/pipeline/cross_framework_reliability.py")
    if not text:
        return ()
    value = _module_literal(text, "MAINTAINED_FRAMEWORKS")
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return ()


def _mcp_counts(snapshot: RepositorySnapshot) -> dict[str, int]:
    """MCP tool/module counts.

    ``tools``/``modules`` are read from the project's maintained MCP audit ledger
    (``src/gnn/mcp/audit_report.json``), which is regenerated by
    ``uv run python src/gnn/mcp/validate_tools.py`` by
    actually loading every MCP module and counting registered tools — a count
    that cannot be reproduced by static text scanning. This is a *source ledger*
    (one of the manuscript's allowed evidence types), and it is only as current
    as the committed file: ``tests/mcp/test_mcp_audit.py`` fails when the
    ledger drifts from the live registry. The ``files`` count is recomputed live
    from the snapshot each run.
    """
    counts = {"tools": 0, "modules_total": 0, "modules_loaded": 0, "files": 0}
    raw = snapshot.read_text("src/gnn/mcp/audit_report.json")
    if raw:
        data = json.loads(raw)
        counts["tools"] = int(data.get("tools_total", 0))
        counts["modules_total"] = int(data.get("modules_total", 0))
        counts["modules_loaded"] = int(data.get("modules_loaded", 0))
    counts["files"] = len(snapshot.glob("src/gnn", "mcp.py"))
    return counts


def _families(snapshot: RepositorySnapshot) -> list[dict]:
    raw = snapshot.read_text("input/model_family_manifest.json")
    if not raw:
        return []
    data = json.loads(raw)
    families = data.get("families", [])
    return families if isinstance(families, list) else []


def _count_files(snapshot: RepositorySnapshot, prefix: str, pattern: str) -> int:
    return len(snapshot.glob(prefix, pattern))


def _example_models(snapshot: RepositorySnapshot) -> list[Path]:
    """Return the GNN example *models* under ``input/gnn_files``.

    A GNN model file is identified by its mandatory ``## GNNSection`` header, so
    the corpus README/AGENTS/INDEX documents that live alongside the models are
    not counted as models. Model corpora that live outside ``input/gnn_files``
    (see ``GNN_UNSCANNED_CORPUS_DIRS``) are deliberately not in this set.
    """
    candidates = snapshot.glob("input/gnn_files", "*.md")
    snapshot.prefetch(candidates)
    return [md for md in candidates if "## GNNSection" in snapshot.read_text(md)]


def _release_metadata(snapshot: RepositorySnapshot) -> list[tuple[str, str, str]]:
    """Parse ``CHANGELOG.md`` into ``(version, date, codename)`` triples.

    Each release heading (``## [3.2.0] — 2026-09-02``) is followed by a
    block-quoted summary whose leading bold run is the release codename
    (``> **Exemplar Gold Standard.** ...``). Newest release first.
    """
    lines = snapshot.read_text("CHANGELOG.md").splitlines()
    if not lines:
        return []
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


def _capability_clause(axis: str, specs: dict[str, dict]) -> str:
    """Render a family's native/unsupported split from the registry flags.

    The manifest declares *which* capability axis a family exercises (e.g.
    ``"capability_axis": "supports_continuous"``); the split itself is read from
    ``src/render/framework_registry.py``. The manifest must never restate the
    split in prose — that is how ``discopy`` came to be listed as natively
    supporting the continuous family while the registry, the acceptance ledger
    and CLAUDE.md all reported it ``unsupported``.
    """
    if not axis or not specs:
        return ""
    native = [
        str(spec.get("name", key))
        for key, spec in specs.items()
        if bool(spec.get(axis, False))
    ]
    unsupported = [
        str(spec.get("name", key))
        for key, spec in specs.items()
        if not bool(spec.get(axis, False))
    ]
    if not native and not unsupported:
        return ""
    parts = []
    if native:
        parts.append(f"Native on {', '.join(native)}")
    if unsupported:
        parts.append(f"reported unsupported (not failed) on {', '.join(unsupported)}")
    return "; ".join(parts) + "."


def _render_family_table(families: list[dict], specs: dict[str, dict]) -> str:
    """Render the model-family markdown table.

    The Description cell is the manifest's authored purpose sentence plus, for
    families that declare a ``capability_axis``, a backend split generated from
    the framework registry.
    """
    rows = ["| Family | Frameworks | Description |", "|---|---|---|"]
    for fam in families:
        name = fam.get("name", "?")
        frameworks = str(fam.get("frameworks", "")).replace(",", ", ")
        desc = str(fam.get("description", "")).strip()
        clause = _capability_clause(str(fam.get("capability_axis", "")), specs)
        if clause:
            desc = f"{desc} {clause}".strip()
        rows.append(f"| `{name}` | {frameworks} | {desc} |")
    rows.append(
        _caption(
            "Model families declared in `input/model_family_manifest.json` and "
            "the frameworks each family targets. Capability splits in the "
            "Description column are generated from "
            "`src/render/framework_registry.py`, not authored in the manifest.",
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


def select_cross_framework_family(families: list[dict]) -> dict | None:
    """Return the manifest family used as the cross-framework reference.

    The cross-framework family is the one the manifest marks
    ``"cross_framework": true``; when no family declares the flag the widest
    family (most frameworks) is used, so the selection cannot silently move to a
    different family just because the manifest's ordering changed. ``None`` when
    no family lists more than one framework.

    Public because every surface that highlights the cross-framework set —
    the token map here and ``scripts/manuscript_fig_backend_matrix.py`` — must
    select the *same* family. Two independent implementations is how the figure
    came to highlight a union of every multi-framework family while the prose
    described one.
    """
    multi = [f for f in families if "," in str(f.get("frameworks", ""))]
    if not multi:
        return None
    flagged = [f for f in multi if f.get("cross_framework") is True]
    if flagged:
        return flagged[0]
    return max(multi, key=lambda f: len(str(f.get("frameworks", "")).split(",")))


def _framework_keys(family: Mapping[str, object] | None) -> list[str]:
    """Split a family's comma-separated ``frameworks`` field into keys."""
    if not family:
        return []
    return [
        k.strip() for k in str(family.get("frameworks", "")).split(",") if k.strip()
    ]


def _cross_framework_selection(
    families: list[dict],
    backends: list[tuple[str, str, bool]],
    maintained: Sequence[str],
) -> tuple[str, list[str], list[str]]:
    """Resolve the cross-framework family and its declared/profiled backends.

    Returns ``(family_name, declared_keys, profiled_keys)``. ``profiled_keys``
    is ``declared_keys`` intersected with the reliability gate's
    ``MAINTAINED_FRAMEWORKS``: a framework the manifest declares but the gate
    refuses to profile (``stan`` today) is *not* one of the engines the
    reference comparison runs on, and the manuscript must never count it as
    one.

    Raises:
        ValueError: If the selected family declares frameworks but none of them
            are profiled by the gate — a manifest/gate contradiction that would
            otherwise ship as an empty backend list in the manuscript.
    """
    family = select_cross_framework_family(families)
    if family is None:
        return "", [], []
    declared = _framework_keys(family)
    if not maintained:  # registry unreadable (tarball); do not silently drop
        return str(family.get("name", "")), declared, declared
    profiled = [key for key in declared if key in set(maintained)]
    if declared and not profiled:
        raise ValueError(
            f"cross-framework family {family.get('name', '?')!r} declares "
            f"{declared} but the reliability gate profiles none of them "
            f"(MAINTAINED_FRAMEWORKS={list(maintained)}); fix "
            "input/model_family_manifest.json or "
            "src/pipeline/cross_framework_reliability.py before rendering"
        )
    return str(family.get("name", "")), declared, profiled


def generate_variables(project_root: Path) -> dict[str, str]:
    """Compute the manuscript token map by introspecting the live repository.

    Args:
        project_root: Path to the GeneralizedNotationNotation project root.

    Returns:
        Flat ``dict[str, str]`` of ``UPPERCASE_KEY`` -> value. All values are
        strings (the template injector substitutes text). Every count is read
        from the commit reported as ``GNN_GIT_COMMIT``, so the mapping is
        deterministic for that commit regardless of working-tree state. The
        authored manuscript config (``manuscript/config.yaml``) is the one
        working-tree read: it is the input a developer edits and re-renders.

    Raises:
        ValueError: If the cross-framework family declares only frameworks the
            reliability gate refuses to profile.
    """
    project_root = Path(project_root).resolve()
    config = _load_config(project_root)
    paper = config.get("paper", {}) if isinstance(config, dict) else {}
    authors = config.get("authors", []) if isinstance(config, dict) else []
    author = authors[0] if authors else {}
    publication = config.get("publication", {}) if isinstance(config, dict) else {}
    metadata = config.get("metadata", {}) if isinstance(config, dict) else {}
    keywords = config.get("keywords", []) if isinstance(config, dict) else []

    snapshot = RepositorySnapshot(project_root)
    version = _read_pyproject_version(snapshot)
    steps = _pipeline_steps(snapshot)
    purposes = _step_purposes(snapshot)
    specs = _registry_specs(snapshot)
    backends = _backends(snapshot)
    maintained = _maintained_frameworks(snapshot)
    mcp = _mcp_counts(snapshot)
    families = _families(snapshot)
    py_files, py_loc = _count_python(snapshot)
    package_count = _count_packages(snapshot)
    test_file_count, test_func_count = _count_test_functions(snapshot)

    # --- Derived tables (multi-line tokens) ------------------------------------
    step_table = _render_step_table(steps, purposes)
    family_table = _render_family_table(families, specs)
    backend_table = _render_backend_table(backends)
    (
        cross_family_name,
        cross_declared_keys,
        cross_profiled_keys,
    ) = _cross_framework_selection(families, backends, maintained)

    family_names = [f.get("name", "?") for f in families]
    backend_names = [name for _, name, _ in backends]
    executable_backends = [name for _, name, executes in backends if executes]
    name_by_key = {key: name for key, name, _ in backends}
    maintained_names = [name_by_key.get(key, key) for key in maintained]
    cross_declared_names = [name_by_key.get(k, k) for k in cross_declared_keys]
    cross_profiled_names = [name_by_key.get(k, k) for k in cross_profiled_keys]

    # Per-step scalars keyed by module stem, so prose never hand-types a step
    # number the registry owns (`Parsing ({{GNN_STEP_GNN}})`).
    step_tokens = {
        f"GNN_STEP_{re.sub(r'^[0-9]+_', '', script.removesuffix('.py')).upper()}": str(
            number
        )
        for number, script in steps
    }

    releases = _release_metadata(snapshot)
    release_by_version = {ver: (date, name) for ver, date, name in releases}
    release_date, release_codename = release_by_version.get(version, ("", ""))
    # The release that introduced the long-running orchestration contracts is the
    # oldest whose CHANGELOG codename names orchestration.
    orchestration_version = next(
        (ver for ver, _, name in reversed(releases) if "orchestration" in name.lower()),
        version,
    )

    example_models = _example_models(snapshot)
    corpus_dirs = sorted(
        {
            rel.parts[2]
            for rel in snapshot.glob("input/gnn_files", "*")
            if len(rel.parts) > 3
        }
    )
    family_target_dirs = {
        str(f.get("target_dir", "")) for f in families if f.get("target_dir")
    }
    # Corpus directories under input/gnn_files/ that no manifest family claims,
    # and family target_dirs that live outside input/gnn_files/. Both sets are
    # emitted so prose can name the gap instead of implying the three sets
    # (families, corpus directories, example models) are coextensive.
    registered_corpus_dirs = {
        Path(d).name for d in family_target_dirs if d.startswith("input/gnn_files/")
    }
    unregistered_corpus_dirs = [
        name for name in corpus_dirs if name not in registered_corpus_dirs
    ]
    unscanned_corpus_dirs = sorted(
        d for d in family_target_dirs if not d.startswith("input/gnn_files/")
    )
    figure_count = _count_files(snapshot, "output", "*.png")
    manuscript_figure_count = _count_files(snapshot, "output/figures", "*.png")
    doc_file_count = _count_files(snapshot, "doc", "*.md")

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
        "GNN_GIT_COMMIT": snapshot.commit,
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
        "GNN_EXAMPLE_COUNT": str(len(example_models)),
        # Directory census under input/gnn_files/ — NOT the same set as the
        # manifest families (some corpus dirs declare no family; some families
        # point outside input/gnn_files/). GNN_FAMILY_TARGET_DIR_COUNT is the
        # manifest-derived figure and is the one that pairs with
        # GNN_FAMILY_COUNT.
        "GNN_INPUT_FAMILY_DIR_COUNT": str(len(corpus_dirs)),
        "GNN_FAMILY_TARGET_DIR_COUNT": str(len(family_target_dirs)),
        "GNN_UNREGISTERED_CORPUS_DIR_COUNT": str(len(unregistered_corpus_dirs)),
        "GNN_UNREGISTERED_CORPUS_DIRS": ", ".join(
            f"`input/gnn_files/{name}`" for name in unregistered_corpus_dirs
        ),
        "GNN_UNSCANNED_CORPUS_DIR_COUNT": str(len(unscanned_corpus_dirs)),
        "GNN_UNSCANNED_CORPUS_DIRS": ", ".join(f"`{d}`" for d in unscanned_corpus_dirs),
        # Backends
        "GNN_BACKEND_COUNT": str(len(backends)),
        "GNN_BACKEND_LIST": ", ".join(backend_names),
        "GNN_BACKEND_TABLE": backend_table,
        "GNN_EXECUTABLE_BACKEND_COUNT": str(len(executable_backends)),
        "GNN_EXECUTABLE_BACKEND_LIST": ", ".join(executable_backends),
        # MAINTAINED_FRAMEWORKS in src/pipeline/cross_framework_reliability.py —
        # the set the reliability gate will profile. A strict subset of the
        # registry: GNN_BACKEND_COUNT is never the right number for a sentence
        # about what the gate does.
        "GNN_MAINTAINED_FRAMEWORK_COUNT": str(len(maintained)),
        "GNN_MAINTAINED_FRAMEWORK_LIST": ", ".join(maintained_names),
        "GNN_CROSS_FRAMEWORK_FAMILY": cross_family_name,
        # GNN_CROSS_FRAMEWORK_BACKENDS is the *profiled* set: the family's
        # declared frameworks intersected with MAINTAINED_FRAMEWORKS. The
        # declared-but-unprofiled remainder (stan today) is exposed separately so
        # a sentence about it cannot silently become a sentence about the gate.
        "GNN_CROSS_FRAMEWORK_BACKENDS": ", ".join(cross_profiled_names),
        "GNN_CROSS_FRAMEWORK_BACKEND_COUNT": str(len(cross_profiled_names)),
        "GNN_CROSS_FRAMEWORK_DECLARED_BACKENDS": ", ".join(cross_declared_names),
        "GNN_CROSS_FRAMEWORK_DECLARED_BACKEND_COUNT": str(len(cross_declared_names)),
        "GNN_CROSS_FRAMEWORK_UNPROFILED_BACKENDS": ", ".join(
            name for name in cross_declared_names if name not in cross_profiled_names
        ),
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


# Fields in manuscript/config.yaml the producer owns, and the token that owns
# each. config.yaml is NOT token-substituted (the injector only processes
# manuscript/*.md), so a literal typed here can only be kept true by writing it.
_CONFIG_OWNED_FIELDS = {
    "version": "GNN_VERSION",
    "date": "GNN_RELEASE_DATE",
}


def config_metadata_drift(
    project_root: Path, variables: Mapping[str, str]
) -> list[str]:
    """Return one message per ``config.yaml`` field that disagrees with a token.

    Empty list means the title page states the same version and date the
    repository does.
    """
    config_path = Path(project_root) / "manuscript" / "config.yaml"
    if not config_path.is_file():
        return []
    drift: list[str] = []
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        for field, token in _CONFIG_OWNED_FIELDS.items():
            expected = variables.get(token, "")
            if not expected or not stripped.startswith(f"{field}:"):
                continue
            actual = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            if actual != expected:
                drift.append(
                    f"config.yaml: {field}: {actual!r} != {token} ({expected!r})"
                )
    return drift


def sync_config_metadata(project_root: Path, variables: Mapping[str, str]) -> list[str]:
    """Rewrite the producer-owned ``manuscript/config.yaml`` fields in place.

    ``version:`` and ``date:`` render onto the PDF title page but sit in a file
    the token injector never touches, so they can only stay true by being
    written from the same source the tokens come from: ``pyproject.toml`` for
    the version and the matching ``CHANGELOG.md`` heading for the date. The
    alternative — typing the correct value once — is what left a 3.2.0 title
    page carrying 3.0.0's release date.

    Returns the list of ``"field: old -> new"`` changes applied (empty when the
    file was already in sync).
    """
    config_path = Path(project_root) / "manuscript" / "config.yaml"
    if not config_path.is_file():
        return []
    lines = config_path.read_text(encoding="utf-8").splitlines(keepends=True)
    changes: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        for field, token in _CONFIG_OWNED_FIELDS.items():
            expected = variables.get(token, "")
            if not expected or not stripped.startswith(f"{field}:"):
                continue
            actual = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            if actual == expected:
                continue
            indent = line[: len(line) - len(line.lstrip())]
            newline = "\n" if line.endswith("\n") else ""
            lines[index] = f'{indent}{field}: "{expected}"{newline}'
            changes.append(f"{field}: {actual} -> {expected}")
    if changes:
        config_path.write_text("".join(lines), encoding="utf-8")
    return changes


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
