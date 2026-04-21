#!/usr/bin/env python3
"""MCP Processor module for GNN Processing Pipeline."""

import logging
from pathlib import Path

from utils.pipeline_template import (
    log_step_error,
    log_step_start,
    log_step_success,
    log_step_warning,
)

logger = logging.getLogger(__name__)

def register_module_tools(module_name: str = None):
    """Register tools for a specific module, or all modules if no name given.

    Args:
        module_name: Name of the module whose MCP tools to register (e.g. ``"gnn"``).
            When *None* all available pipeline modules are auto-discovered and
            registered, and the full list of registered tool dicts is returned.

    Returns:
        *bool* when registering a single named module (True = success).
        *list* of tool dicts when called with no arguments (auto-discover mode).
    """
    import importlib

    from .mcp import mcp_instance

    # --- Auto-discover mode (no module_name given) ---
    if module_name is None:
        # Known pipeline modules that expose an mcp sub-module
        pipeline_modules = [
            "template", "setup", "tests", "gnn", "model_registry", "type_checker",
            "validation", "export", "visualization", "advanced_visualization",
            "ontology", "render", "execute", "llm", "ml_integration", "audio",
            "analysis", "integration", "security", "research", "website", "mcp",
            "gui", "report", "intelligent_analysis",
        ]
        registered: list = []
        for mod_name in pipeline_modules:
            for module_path in (f"src.{mod_name}.mcp", f"{mod_name}.mcp"):
                try:
                    mod = importlib.import_module(module_path)
                    if hasattr(mod, "register_tools") and callable(mod.register_tools):
                        mod.register_tools(mcp_instance)
                        logger.debug(f"Auto-registered tools for module: {mod_name}")
                    break
                except (ImportError, ModuleNotFoundError):
                    continue
                except Exception as e:
                    logger.debug(f"register_tools failed for {mod_name}: {e}")
                    break
        # Return the current tool list so callers can inspect what was registered
        try:
            registered = mcp_instance.list_available_tools()
        except Exception:
            registered = []
        logger.info(f"Auto-discovered registration complete: {len(registered)} tool(s) available")
        return registered

    # --- Single-module mode ---
    try:
        logger.info(f"Registering tools for module: {module_name}")

        module_paths = [f"{module_name}.mcp", f"src.{module_name}.mcp"]
        module = None

        for module_path in module_paths:
            try:
                module = importlib.import_module(module_path)
                break
            except (ImportError, ModuleNotFoundError):
                logger.debug("Could not import %s, trying next path", module_path)
                continue

        if module is None:
            logger.warning(f"Module {module_name}.mcp not importable from paths: {module_paths}")
            return False

        if hasattr(module, "register_tools") and callable(module.register_tools):
            try:
                module.register_tools(mcp_instance)
                logger.info(f"Registered tools for module: {module_name}")
                return True
            except Exception as e:
                logger.error(f"register_tools failed for module {module_name}: {e}")
                return False
        else:
            logger.warning(f"Module src.{module_name}.mcp has no register_tools function")
            return False

    except Exception as e:
        logger.error(f"Failed to register tools for {module_name}: {e}")
        return False

def handle_mcp_request(request: dict) -> dict:
    """Handle an incoming MCP request and return a JSON-RPC response."""
    try:
        logger.info(f"Handling MCP request: {request.get('method', 'unknown')}")

        from .mcp import mcp_instance

        method = request.get("method", "")
        params = request.get("params", {})

        if method == "tools/list":
            tools = mcp_instance.list_available_tools()
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {"tools": tools}
            }
        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_params = params.get("arguments", {})

            result = mcp_instance.execute_tool(tool_name, tool_params)
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": result
            }
        else:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }

    except Exception as e:
        logger.error(f"Failed to handle MCP request: {e}")
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "error": {
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        }

def generate_mcp_report() -> dict:
    """Generate MCP status report with tool and resource counts."""
    try:
        from datetime import datetime

        from . import __version__ as mcp_version
        from .mcp import mcp_instance

        tools = mcp_instance.list_available_tools()
        resources = mcp_instance.list_available_resources()

        report = {
            "timestamp": datetime.now().isoformat(),
            "mcp_version": mcp_version,
            "tools_count": len(tools),
            "resources_count": len(resources),
            "tools": tools,
            "resources": resources,
            "status": "healthy"
        }

        return report

    except Exception as e:
        logger.error(f"Failed to generate MCP report: {e}")
        from datetime import datetime
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "status": "error"
        }

def process_mcp(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False,
    logger=None,
    **kwargs
) -> bool:
    """Initialize MCP, discover modules, register tools, and save reports."""
    import json
    from datetime import datetime

    if logger is None:
        logger = logging.getLogger("mcp")

    if verbose:
        logger.setLevel(logging.DEBUG)

    target_dir = Path(target_dir) if not isinstance(target_dir, Path) else target_dir
    output_dir = Path(output_dir) if not isinstance(output_dir, Path) else output_dir

    try:
        log_step_start(logger, "Processing MCP")

        output_dir.mkdir(parents=True, exist_ok=True)

        from . import __version__ as mcp_version
        from .mcp import initialize, mcp_instance
        perf = kwargs.get("performance_mode", "low")
        pm_timeout = kwargs.get("mcp_per_module_timeout")
        ov_timeout = kwargs.get("mcp_overall_timeout")
        init_kw: dict = {
            "halt_on_missing_sdk": False,
            "force_proceed_flag": True,
            "performance_mode": perf if isinstance(perf, str) else "low",
        }
        if pm_timeout is not None:
            init_kw["per_module_timeout"] = float(pm_timeout)
        if ov_timeout is not None:
            init_kw["overall_timeout"] = float(ov_timeout)
        # Fine-grained overrides: only propagate when the caller set them,
        # so the performance_mode default is respected otherwise.
        # Map both snake-case MCP-prefixed arg names (from the pipeline CLI)
        # and bare names (direct callers) to initialize() kwargs.
        alias_map = {
            "enable_caching": ("enable_caching",),
            "enable_rate_limiting": ("enable_rate_limiting",),
            "strict_validation": ("strict_validation", "mcp_strict_validation"),
            "cache_ttl": ("cache_ttl", "mcp_cache_ttl"),
            "force_refresh": ("force_refresh",),
        }
        for init_key, sources in alias_map.items():
            for src_key in sources:
                if src_key in kwargs and kwargs[src_key] is not None:
                    init_kw[init_key] = kwargs[src_key]
                    break
        allowlist = kwargs.get("modules_allowlist")
        if allowlist is None:
            raw = kwargs.get("mcp_modules_allowlist")
            if isinstance(raw, str) and raw.strip():
                allowlist = [m.strip() for m in raw.split(",") if m.strip()]
        if allowlist:
            init_kw["modules_allowlist"] = list(allowlist)
        initialize(**init_kw)

        # Modules are already discovered by initialize(); just retrieve for reporting
        registered_modules = list(mcp_instance.modules.keys())
        registered_count = len(registered_modules)

        logger.info(f"Auto-discovered {registered_count} modules via initialize()")

        available_tools = get_available_tools()
        tools_count = len(available_tools)

        report = generate_mcp_report()
        report["registered_modules"] = registered_count
        report["registered_module_names"] = registered_modules
        report["timestamp"] = datetime.now().isoformat()
        report["target_dir"] = str(target_dir)
        report["output_dir"] = str(output_dir)

        results_file = output_dir / "mcp_results.json"
        with open(results_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"📋 Detailed MCP report saved to: {results_file}")

        if available_tools:
            tools_file = output_dir / "registered_tools.json"
            with open(tools_file, 'w') as f:
                json.dump(available_tools, f, indent=2)
            logger.info(f"🔧 Registered tools saved to: {tools_file}")

        summary = {
            "timestamp": datetime.now().isoformat(),
            "target_dir": str(target_dir),
            "output_dir": str(output_dir),
            "processing_status": "completed",
            "mcp_version": mcp_version,
            "tools_registered": tools_count,
            "registered_modules_count": registered_count,
            "registered_modules": registered_modules,
            "resources_count": report.get('resources_count', 0),
            "message": f"MCP processing completed - {tools_count} tools registered from {registered_count} modules"
        }

        summary_file = output_dir / "mcp_processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✅ MCP summary saved to: {summary_file}")

        if registered_count > 0:
            log_step_success(logger, f"MCP processing completed successfully - {tools_count} tools from {registered_count} modules registered")
        else:
            log_step_warning(logger, "MCP processing completed with no modules registered")

        return True

    except Exception as e:
        log_step_error(logger, f"MCP processing failed: {e}")
        try:
            from . import __version__ as mcp_version
            summary = {
                "timestamp": datetime.now().isoformat(),
                "target_dir": str(target_dir),
                "output_dir": str(output_dir),
                "processing_status": "failed",
                "mcp_version": mcp_version,
                "tools_registered": 0,
                "error": str(e),
                "message": f"MCP processing failed: {str(e)}"
            }
            summary_file = output_dir / "mcp_processing_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as save_err:
            logger.debug(f"Could not save error summary file (non-fatal): {save_err}")
        return False

def get_available_tools() -> list:
    """Return a list of all registered MCP tools."""
    try:
        from .mcp import mcp_instance
        return mcp_instance.list_available_tools()
    except Exception as e:
        logger.error(f"Failed to get available tools: {e}")
        return []
