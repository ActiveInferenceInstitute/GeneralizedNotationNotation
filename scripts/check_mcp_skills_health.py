#!/usr/bin/env python3
"""Check that every module MCP tool executes and every SKILL.md resolves.

Two independent surface audits:

1. **MCP tool surface** — every registered tool must have a callable handler and
   execute (with schema-minimal arguments) without crashing. Validation-layer
   rejections (MCPInvalidParamsError etc.) and graceful error-result dicts are
   accepted: they prove the tool is wired and the validation layer works.

2. **Skills surface** — every ``src/<module>/SKILL.md`` API import, ``Key
   Exports`` symbol, ``MCP Tools`` bullet claim, and ``Key Commands`` script
   must resolve against the live codebase. A skill that documents a symbol the
   module does not export would fail for an agent following it.

Informational gate (not CI-wired, like ``check_external_links.py``):
    uv run --extra dev python scripts/check_mcp_skills_health.py [--strict]
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
import tempfile
from pathlib import Path

from scripts.lib.shared import add_strict_flag, exit_with_findings, repo_root

REPO = repo_root()
SRC = REPO / "src"
# Make src/ importable when the script is invoked without PYTHONPATH=src.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ---------------------------------------------------------------------------
# Audit 1: MCP tool execution
# ---------------------------------------------------------------------------
_PATH_HINTS = (
    "file",
    "path",
    "dir",
    "target",
    "input",
    "model",
    "gnn",
    "source",
    "out",
)
_OUTPUT_HINTS = ("output", "save", "dest", "result", "destination")

# Seed tool arguments strictly under a temp dir so tool execution can never
# mutate repository files or committed input models.
_TMPDIR = Path(tempfile.mkdtemp(prefix="gnn_mcp_audit_"))
_MODEL_COPY = _TMPDIR / "model.md"
_MODEL_COPY.write_bytes(
    (
        SRC.parent / "input" / "gnn_files" / "basics" / "static_perception.md"
    ).read_bytes()
)
_OUTPUT_PATH = _TMPDIR / "output"
_OUTPUT_PATH.mkdir(exist_ok=True)


def _sample_value(prop_name: str, prop: dict) -> object:
    """Build a plausible minimal value for a schema property (temp paths only)."""
    if "default" in prop:
        return prop["default"]
    if "enum" in prop and prop["enum"]:
        return prop["enum"][0]
    if "const" in prop:
        return prop["const"]
    ptype = prop.get("type")
    low = prop_name.lower()
    if ptype == "string":
        if any(h in low for h in _OUTPUT_HINTS):
            return str(_OUTPUT_PATH)
        if any(h in low for h in _PATH_HINTS):
            return str(_MODEL_COPY)
        return ""
    if ptype == "integer":
        return 1
    if ptype == "number":
        return 1.0
    if ptype == "boolean":
        return False
    if ptype == "array":
        if any(h in low for h in _PATH_HINTS) and not any(
            h in low for h in _OUTPUT_HINTS
        ):
            return [str(_MODEL_COPY)]
        return []
    if ptype == "object":
        return {}
    return None


def _build_args(schema: dict | None) -> dict:
    if not schema or not isinstance(schema, dict):
        return {}
    props = schema.get("properties") or {}
    return {name: _sample_value(name, p) for name, p in props.items()}


def audit_mcp_tools() -> list[str]:
    """Execute every registered MCP tool; return findings for crashes/missing handlers."""
    findings: list[str] = []
    from mcp import initialize, mcp_instance

    # Run the sweep from the temp dir so relative output paths (some pipeline
    # tools default to "output") never write into the repository.
    cwd = Path.cwd()
    os.chdir(_TMPDIR)
    try:
        initialize(
            halt_on_missing_sdk=False, force_proceed_flag=True, force_refresh=True
        )
        tools = dict(mcp_instance.tools)
        summary = {"ok": 0, "validation": 0, "graceful": 0}
        for name, tool in sorted(tools.items()):
            handler = getattr(tool, "func", None)
            if not callable(handler):
                findings.append(f"MCP tool {name!r} has no callable handler")
                continue
            try:
                result = mcp_instance.execute_tool(
                    name, _build_args(getattr(tool, "schema", None))
                )
                if isinstance(result, dict) and (
                    result.get("error") or result.get("success") is False
                ):
                    summary["graceful"] += 1
                else:
                    summary["ok"] += 1
            except Exception as exc:  # noqa: BLE001
                cls = type(exc).__name__
                if cls in (
                    "MCPToolNotFoundError",
                    "MCPInvalidParamsError",
                    "MCPToolExecutionError",
                ):
                    summary["validation"] += 1
                else:
                    findings.append(
                        f"MCP tool {name!r} crashed: {cls}: {str(exc)[:120]}"
                    )
    finally:
        os.chdir(cwd)
    print(
        f"MCP tools: {len(tools)} registered | "
        f"{summary['ok']} ok, {summary['validation']} validation-ok, "
        f"{summary['graceful']} graceful-error"
    )
    return findings


# ---------------------------------------------------------------------------
# Audit 2: SKILL.md resolvability
# ---------------------------------------------------------------------------
_MCP_KNOBS = {
    "performance_mode",
    "enable_caching",
    "enable_rate_limiting",
    "strict_validation",
    "cache_ttl",
    "modules_allowlist",
    "per_module_timeout",
    "overall_timeout",
    "force_refresh",
}


def _py_blocks(text: str) -> list[str]:
    return re.findall(r"```python(.*?)```", text, re.DOTALL)


def _imports_from_block(block: str) -> list[tuple[str, str | None]]:
    pairs: list[tuple[str, str | None]] = []
    for m in re.finditer(
        r"^\s*from\s+([\w.]+)\s+import\s*\(([^)]*)\)", block, re.MULTILINE | re.DOTALL
    ):
        mod = m.group(1)
        for line in m.group(2).splitlines():
            line = line.split("#", 1)[0]
            for name in re.findall(r"[A-Za-z_]\w*", line):
                pairs.append((mod, name))
    for m in re.finditer(
        r"^\s*from\s+([\w.]+)\s+import\s+([A-Za-z_][\w,]*(?:\s+as\s+[\w.]+)?)$",
        block,
        re.MULTILINE,
    ):
        mod = m.group(1)
        for name in m.group(2).split(","):
            name = name.strip().split(" as ")[0]
            if name:
                pairs.append((mod, name))
    for m in re.finditer(r"^\s*import\s+([\w.]+)", block, re.MULTILINE):
        pairs.append((m.group(1), None))
    return pairs


def _section(text: str, header: str) -> str:
    m = re.search(rf"## {re.escape(header)}\n(.*?)(?=\n## |\Z)", text, re.DOTALL)
    return m.group(1) if m else ""


def _resolve(mod: str, symbol: str | None) -> bool:
    candidates = [mod] + ([mod[4:]] if mod.startswith("src.") else [f"src.{mod}"])
    for cand in candidates:
        try:
            module = importlib.import_module(cand)
        except Exception:
            continue
        if symbol is None or hasattr(module, symbol):
            return True
        try:
            importlib.import_module(f"{cand}.{symbol}")
            return True
        except Exception:
            continue
    return False


def _module_package_resolve(module_dir: Path, symbol: str) -> bool:
    for cand in (module_dir.name, f"src.{module_dir.name}"):
        try:
            if hasattr(importlib.import_module(cand), symbol):
                return True
        except Exception:
            continue
    try:
        pkg = importlib.import_module(module_dir.name)
        for sub in pkg.__path__:
            for f in Path(sub).glob("*.py"):
                if f.name.startswith("_"):
                    continue
                try:
                    if hasattr(
                        importlib.import_module(f"{module_dir.name}.{f.stem}"), symbol
                    ):
                        return True
                except Exception:
                    continue
    except Exception:
        pass
    return False


def _resolvability_findings(skill: Path, rel: Path) -> tuple[list[str], int]:
    """Resolver-only checks (API imports, Key Exports, Key Commands), no MCP.

    Returned as a ``(findings, checks)`` pair so it can be unit-tested without
    a live MCP discovery.
    """
    findings: list[str] = []
    checks = 0
    text = skill.read_text(encoding="utf-8")
    module_dir = skill.parent

    for mod, symbol in _imports_from_block("".join(_py_blocks(text))):
        checks += 1
        ok = _resolve(mod, None) if symbol is None else _resolve(mod, symbol)
        if not ok:
            target = (
                f"import {mod}" if symbol is None else f"from {mod} import {symbol}"
            )
            findings.append(f"{rel}: unresolved {target}")

    # Only the bullet's name slot (before the ``—`` description separator)
    # carries resolvability claims; backticked words in prose do not.
    export_symbols = [
        symbol
        for line in _section(text, "Key Exports").splitlines()
        for symbol in re.findall(r"`([A-Za-z_]\w*)`", line.split("\u2014")[0])
    ]
    for symbol in export_symbols:
        checks += 1
        if symbol in _MCP_KNOBS:
            continue  # documented initialize() configuration knobs
        if not _module_package_resolve(module_dir, symbol):
            findings.append(
                f"{rel}: Key Export {symbol!r} not resolvable on {module_dir.name}"
            )

    for m in re.finditer(r"python\s+(src/(?:[0-9]+_)?\w+\.py)", text):
        checks += 1
        if not (SRC / m.group(1).split("/", 1)[1]).exists():
            findings.append(f"{rel}: documented command script {m.group(1)} missing")

    return findings, checks


def audit_skills() -> list[str]:
    """Verify every SKILL.md documents a resolvable surface; return findings."""
    findings: list[str] = []
    from mcp import initialize, mcp_instance

    initialize(halt_on_missing_sdk=False, force_proceed_flag=True, force_refresh=True)
    live_tools: set[str] = set(mcp_instance.tools.keys())

    skill_files = sorted(SRC.glob("*/SKILL.md")) + sorted(
        (SRC / "gui").glob("*/SKILL.md")
    )
    checks = 0
    for skill in skill_files:
        rel = skill.relative_to(REPO)
        local, local_checks = _resolvability_findings(skill, rel)
        findings.extend(local)
        checks += local_checks

        for line in _section(
            skill.read_text(encoding="utf-8"), "MCP Tools"
        ).splitlines():
            if not line.strip().startswith("- "):
                continue  # prose notes (e.g. deliberate exclusions) are not claims
            for tool in re.findall(r"`([A-Za-z_]\w*)`", line):
                checks += 1
                if tool not in live_tools:
                    findings.append(f"{rel}: MCP tool {tool!r} not registered")

    print(f"Skills: {len(skill_files)} SKILL.md files, {checks} checks")
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify MCP tool execution and SKILL.md resolvability"
    )
    parser.add_argument(
        "--mcp-only", action="store_true", help="Run only the MCP tool audit"
    )
    parser.add_argument(
        "--skills-only", action="store_true", help="Run only the skills audit"
    )
    add_strict_flag(parser)
    args = parser.parse_args(argv)

    findings: list[str] = []
    if not args.skills_only:
        findings += audit_mcp_tools()
    if not args.mcp_only:
        findings += audit_skills()

    print(f"\n{len(findings)} finding(s)")
    for f in findings:
        print("  " + f)
    return exit_with_findings(len(findings), args.strict)


if __name__ == "__main__":
    sys.exit(main())
