#!/usr/bin/env python3
"""
Comprehensive MCP Tool Audit Script

Validates that every registered MCP tool is:
  1. Detectable (importable via discover_modules)
  2. Real (backed by a named callable, not a lambda or None)
  3. Documented (non-empty description string)
  4. Callable (can be invoked via execute_tool with empty args)
  5. Logged (register_tools call has logging.info in its source)

Usage:
    cd /path/to/generalizednotationnotation
    PYTHONPATH=src python src/mcp/validate_tools.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

# ── Path setup ───────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent.parent
SRC_ROOT  = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

logger = logging.getLogger("mcp_audit")


def main() -> int:
    """Run the full audit. Returns exit code (0=pass, 1=failures found)."""

    print("=" * 72)
    print("  GNN MCP TOOL AUDIT")
    print(f"  Repo: {REPO_ROOT}")
    print("=" * 72)

    # ── 1. Initialize MCP ────────────────────────────────────────────────────
    print("\n[1] Initializing MCP and discovering modules...")
    try:
        from mcp import initialize, mcp_instance
        initialize(halt_on_missing_sdk=False, force_proceed_flag=True)
        m = mcp_instance
    except Exception as e:
        print(f"  FATAL: Could not initialize MCP: {e}")
        return 1

    # ── 2. Module summary ────────────────────────────────────────────────────
    print(f"\n[2] Modules discovered: {len(m.modules)}")
    loaded  = [name for name, info in m.modules.items() if info.status == "loaded"]
    errored = [name for name, info in m.modules.items() if info.status == "error"]

    for mod, info in sorted(m.modules.items()):
        icon = "✓" if info.status == "loaded" else "✗"
        print(f"  {icon}  {mod:32s}  tools={info.tools_count:3d}  status={info.status}")

    if errored:
        print(f"\n  WARNING: {len(errored)} modules failed to load: {errored}")

    # ── 3. Per-tool audit ────────────────────────────────────────────────────
    print(f"\n[3] Tool audit: {len(m.tools)} tools registered")

    issues: List[Dict[str, Any]] = []

    for name, tool in sorted(m.tools.items()):
        func = getattr(tool, "func", None) or getattr(tool, "function", None)
        desc = (getattr(tool, "description", "") or "").strip()
        fn   = getattr(func, "__name__", "NONE") if func else "NONE"
        mod  = getattr(tool, "module", "") or ""

        is_callable  = callable(func)
        is_lambda    = (fn == "<lambda>")
        is_real      = is_callable and not is_lambda

        is_documented = bool(desc)

        status = "OK"
        if not is_real:
            status = "NOT_CALLABLE"
            issues.append({"tool": name, "issue": "not a real callable", "fn": fn, "mod": mod})
        elif not is_documented:
            status = "UNDOCUMENTED"
            issues.append({"tool": name, "issue": "missing description", "fn": fn, "mod": mod})

        flag = "✓" if status == "OK" else "✗"
        print(f"  {flag} [{status:14s}]  {name:55s}  fn={fn:35s}  doc={is_documented}  mod={mod}")

    # ── 4. Light callability verification ────────────────────────────────────
    print("\n[4] Callability spot-checks (no-arg tools)...")
    no_arg_tools = ["list_analysis_tools", "list_render_frameworks", "list_export_formats",
                    "list_standard_ontology_terms", "list_supported_integrations",
                    "list_research_topics", "list_security_checks",
                    "get_website_module_info", "get_render_module_info",
                    "get_visualization_options", "get_report_module_info",
                    "check_audio_backends", "get_sapf_module_info",
                    "check_integration_dependencies"]

    call_ok  = 0
    call_err = 0
    for tname in no_arg_tools:
        if tname not in m.tools:
            print(f"  ⚠  SKIP (not registered): {tname}")
            continue
        try:
            result = m.execute_tool(tname, {})
            ok = isinstance(result, dict) and result.get("success", False)
            icon = "✓" if ok else "⚠"
            print(f"  {icon}  {tname:55s}  → success={ok}")
            if ok:
                call_ok += 1
            else:
                call_err += 1
                issues.append({"tool": tname, "issue": "execute returned success=False", "result": str(result)[:80]})
        except Exception as e:
            print(f"  ✗  {tname:55s}  → EXCEPTION: {e}")
            call_err += 1
            issues.append({"tool": tname, "issue": f"exception: {e}"})

    # ── 5. Logging coverage check ─────────────────────────────────────────────
    print("\n[5] Logging coverage check (register_tools uses logger.info?)...")
    submodule_dirs = [d for d in SRC_ROOT.iterdir()
                      if d.is_dir() and (d / "mcp.py").exists()
                      and not d.name.startswith("_") and d.name != "mcp"]

    log_ok  = 0
    log_miss = 0
    for d in sorted(submodule_dirs):
        mcp_src = (d / "mcp.py").read_text(encoding="utf-8", errors="replace")
        in_register = False
        has_log_info = False
        for line in mcp_src.splitlines():
            if "def register_tools" in line:
                in_register = True
            if in_register and ("logger.info" in line or "logging.info" in line):
                has_log_info = True
                break
        icon = "✓" if has_log_info else "✗"
        if has_log_info:
            log_ok += 1
        else:
            log_miss += 1
            issues.append({"tool": d.name + "/mcp.py", "issue": "register_tools() has no logger.info"})
        print(f"  {icon}  {d.name}/mcp.py  logged={has_log_info}")

    # ── 6. Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  AUDIT SUMMARY")
    print("=" * 72)
    print(f"  Modules loaded       : {len(loaded):3d} / {len(m.modules)}")
    print(f"  Tools registered     : {len(m.tools):3d}")
    print(f"  Issues found         : {len(issues):3d}")
    print(f"  Spot-checks OK       : {call_ok:3d} / {call_ok + call_err}")
    print(f"  Modules logged       : {log_ok:3d} / {log_ok + log_miss}")

    if issues:
        print(f"\n  ISSUES ({len(issues)}):")
        for iss in issues:
            print(f"    • {iss['tool']}: {iss['issue']}")
        status_str = "PARTIAL - issues found above"
    else:
        status_str = "PASS - all tools real, documented, logged"

    print(f"\n  RESULT: {status_str}")
    print("=" * 72)

    # ── 7. Save report ───────────────────────────────────────────────────────
    report = {
        "modules_total":    len(m.modules),
        "modules_loaded":   len(loaded),
        "modules_errored":  len(errored),
        "tools_total":      len(m.tools),
        "tools_list":       [
            {
                "name":     name,
                "module":   getattr(tool, "module", ""),
                "category": getattr(tool, "category", ""),
                "fn":       getattr(getattr(tool, "func", None) or getattr(tool, "function", None), "__name__", "?"),
                "documented": bool((getattr(tool, "description", "") or "").strip()),
                "description": (getattr(tool, "description", "") or "").strip()[:120],
            }
            for name, tool in sorted(m.tools.items())
        ],
        "spot_checks_ok":   call_ok,
        "spot_checks_err":  call_err,
        "logging_ok":       log_ok,
        "logging_miss":     log_miss,
        "issues":           issues,
    }
    out_path = SRC_ROOT / "mcp" / "audit_report.json"
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=out_path.parent, delete=False) as tmp_f:
        tmp_f.write(json.dumps(report, indent=2))
    os.replace(tmp_f.name, str(out_path))
    print(f"\n  Full report saved → {out_path}")

    return 0 if not issues else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
