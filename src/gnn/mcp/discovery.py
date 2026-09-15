#!/usr/bin/env python3
"""Module discovery for the GNN MCP server.

Mechanical extraction from ``gnn.mcp.mcp`` (MAJ-04 sibling-mixin split):
``MCPDiscoveryMixin`` holds the verbatim discovery/loading methods and
``MCP`` in ``mcp.py`` inherits from it, so every import path, method
resolution, and registered-tool name is unchanged.
"""

from __future__ import annotations

import importlib
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ._late_binding import _MCPModuleRef
from .models import MCPModuleInfo, MCPPerformanceMetrics, MCPResource, MCPTool

# Configure logging
logger = logging.getLogger("mcp")

# Late-bound clock: attribute access resolves through ``gnn.mcp.mcp`` so the
# stub-clock swap in tests/mcp/test_registry_internals.py keeps working.
time = _MCPModuleRef("time")

# Process-wide lock serializing module-body imports across the discovery
# executor (and across MCP instances): several module bodies call
# ``matplotlib.use(...)`` at import time, and a concurrent
# ``matplotlib.use`` while a sibling thread is still executing
# ``matplotlib.pyplot``'s module body raises "partially initialized module
# 'matplotlib.pyplot' has no attribute 'switch_backend'" (flaky CI failure
# on fresh runners, where the first pyplot import is slow enough to open
# the race window).
_MODULE_IMPORT_LOCK = threading.RLock()


class MCPDiscoveryMixin:
    """Verbatim module-discovery methods moved from ``MCP``."""

    if TYPE_CHECKING:
        # Shared ``MCP`` state the moved bodies touch. Declared annotation-
        # only so the mixin type-checks standalone; the real values are
        # created by ``MCP.__init__`` in ``mcp.py``.
        _lock: Any
        _modules_discovered: bool
        modules: Dict[str, MCPModuleInfo]
        tools: Dict[str, MCPTool]
        resources: Dict[str, MCPResource]
        _discovery_cache: Dict[str, Any]
        _discovery_cache_lock: Any
        _executor: Optional[ThreadPoolExecutor]
        _cache_timestamp: float
        _registration_lock: Any
        _registration_context: threading.local
        _performance_metrics: MCPPerformanceMetrics

    def discover_modules(
        self,
        force_refresh: bool = False,
        modules_allowlist: Optional[List[str]] = None,
        per_module_timeout: float = 30.0,
        overall_timeout: float = 120.0,
    ) -> bool:
        """
        Enhanced module discovery with caching, thread safety, and comprehensive error handling.

        This method scans the src/ directory for modules with mcp.py files
        and loads them to register their tools and resources with improved
        caching, thread safety, and error handling.

        Args:
            force_refresh: If True, force refresh of module discovery cache
            modules_allowlist: Optional list of modules to load (others ignored)
            per_module_timeout: Timeout per module in seconds
            overall_timeout: Overall timeout for discovery in seconds

        Returns:
            bool: True if all modules loaded successfully, False otherwise.
        """
        with self._discovery_cache_lock:
            if self._modules_discovered and not force_refresh:
                logger.debug(
                    "MCP modules already discovered. Skipping redundant discovery."
                )
                return True

            # Prevent concurrent discoveries
            self._modules_discovered = True

        root_dir = Path(__file__).parent.parent
        self._configure_local_imports(root_dir)
        logger.info(f"Discovering MCP modules in {root_dir}")
        all_modules_loaded_successfully = True

        # Track discovery performance
        discovery_start = time.time()

        # Clear existing modules if forcing refresh
        if force_refresh:
            with self._lock:
                self.modules.clear()
                self.tools.clear()
                self.resources.clear()
                self._discovery_cache.clear()

        # Get list of directories to scan
        discovery_excluded_dirs: set[Any] = {"tests"}
        directories = [
            d
            for d in root_dir.iterdir()
            if d.is_dir()
            and not d.name.startswith("_")
            and d.name not in discovery_excluded_dirs
        ]
        if modules_allowlist:
            allow = set(modules_allowlist)
            directories = [d for d in directories if d.name in allow or d.name == "mcp"]

        # Use thread pool if available, otherwise load sequentially
        if self._executor is not None:
            module_load_futures: dict[Any, Any] = {}
            for directory in directories:
                mcp_file = directory / "mcp.py"
                if not mcp_file.exists():
                    logger.debug(f"No MCP module found in {directory}")
                    continue
                future = self._executor.submit(self._load_module, directory, mcp_file)
                module_load_futures[directory.name] = future
            # The package root's own MCP module (src/gnn/mcp.py) is keyed
            # "gnn" and carries the gold-standard GNN tools.
            gnn_mcp_file = root_dir / "mcp" / "gnn_root.py"
            if gnn_mcp_file.exists():
                future = self._executor.submit(
                    self._load_module, root_dir, gnn_mcp_file, "gnn"
                )
                module_load_futures["gnn"] = future

            start_wait = time.time()
            from concurrent.futures import TimeoutError as FuturesTimeoutError

            for module_name, future in list(module_load_futures.items()):
                remaining = max(0.0, overall_timeout - (time.time() - start_wait))
                try:
                    success = future.result(
                        timeout=min(per_module_timeout, remaining)
                        if remaining > 0
                        else 0.001
                    )
                    if not success:
                        all_modules_loaded_successfully = False
                except FuturesTimeoutError:
                    # Timeout is a transient issue - module may load later via recovery
                    logger.warning(
                        f"Module {module_name} loading timed out (>{per_module_timeout:.0f}s) — skipped"
                    )
                    all_modules_loaded_successfully = False
                except Exception as e:
                    error_msg = str(e) if str(e) else type(e).__name__
                    logger.error(f"Failed to load module {module_name}: {error_msg}")
                    all_modules_loaded_successfully = False
                    self.modules[module_name] = MCPModuleInfo(
                        name=f"gnn.{module_name}.mcp",
                        path=Path(__file__).parent.parent / module_name / "mcp.py",
                        status="error",
                        error_message=error_msg,
                        last_updated=time.time(),
                    )
        else:
            # Recovery sequential loading
            # Package-root MCP module in sequential (recovery) mode.
            gnn_mcp_file = root_dir / "mcp" / "gnn_root.py"
            if gnn_mcp_file.exists():
                if not self._load_module(root_dir, gnn_mcp_file, "gnn"):
                    all_modules_loaded_successfully = False
            for directory in directories:
                mcp_file = directory / "mcp.py"
                if not mcp_file.exists():
                    logger.debug(f"No MCP module found in {directory}")
                    continue
                success = self._load_module(directory, mcp_file)
                if not success:
                    all_modules_loaded_successfully = False

        # Special handling for core MCP tools in the mcp directory itself
        mcp_dir = Path(__file__).parent
        logger.debug(f"Discovering core MCP tools in {mcp_dir}")

        # Load SymPy MCP integration (special case - located in mcp directory)
        sympy_mcp_file = mcp_dir / "sympy_mcp.py"
        if sympy_mcp_file.exists():
            try:
                # Import directly as gnn.mcp.sympy_mcp since it's in the mcp directory
                import_start = time.time()
                sympy_module = importlib.import_module("gnn.mcp.sympy_mcp")
                import_time = time.time() - import_start

                if hasattr(sympy_module, "register_tools") and callable(
                    sympy_module.register_tools
                ):
                    tools_before = len(self.tools)
                    with self._tool_registration_context(
                        module="gnn.mcp.sympy_mcp", category="sympy_mcp"
                    ):
                        sympy_module.register_tools(self)
                    tools_added = len(self.tools) - tools_before

                    self.modules["sympy_mcp"] = MCPModuleInfo(
                        name="gnn.mcp.sympy_mcp",
                        path=sympy_mcp_file,
                        tools_count=tools_added,
                        status="loaded",
                        load_time=import_time,
                        last_updated=time.time(),
                    )
                    logger.debug(
                        f"Loaded sympy_mcp: {tools_added} tools in {import_time:.3f}s"
                    )
                else:
                    logger.warning("sympy_mcp module has no register_tools function")
            except Exception as e:
                logger.error(
                    f"Failed to load core MCP module gnn.mcp.sympy_mcp: {str(e)}"
                )
                all_modules_loaded_successfully = False

                self.modules["sympy_mcp"] = MCPModuleInfo(
                    name="gnn.mcp.sympy_mcp",
                    path=sympy_mcp_file,
                    status="error",
                    error_message=str(e),
                    last_updated=time.time(),
                )

        discovery_time = time.time() - discovery_start
        logger.info(
            f"Enhanced module discovery completed in {discovery_time:.2f}s: "
            f"{len(self.modules)} modules, {len(self.tools)} tools, {len(self.resources)} resources"
        )

        self._modules_discovered = True
        self._cache_timestamp = time.time()

        return all_modules_loaded_successfully

    def _load_module(
        self, directory: Path, mcp_file: Path, module_name: Optional[str] = None
    ) -> bool:
        """
        Load a single MCP module with enhanced error handling and performance tracking.

        Args:
            directory: Directory containing the module
            mcp_file: Path to the mcp.py file
            module_name: Optional custom module name

        Returns:
            bool: True if module loaded successfully
        """
        if module_name is None:
            module_name = directory.name

        # Import the module under the process-wide import lock: module
        # bodies run ``matplotlib.use(...)`` at import time, and a
        # concurrent ``matplotlib.use`` while another thread is still
        # executing ``matplotlib.pyplot``'s module body raises
        # "partially initialized module 'matplotlib.pyplot' has no
        # attribute 'switch_backend'". The lock also keeps the process-
        # global import cache consistent for the relocated
        # ``gnn.mcp.gnn_root`` tools module (the flat ``gnn/mcp.py`` was
        # shadowed by the ``gnn/mcp/`` package after the v0.5 rename).
        full_module_name = (
            "gnn.mcp.gnn_root" if module_name == "gnn" else f"gnn.{module_name}.mcp"
        )
        module_start = time.time()
        try:
            root_dir = Path(__file__).parent.parent
            if str(root_dir.parent) not in sys.path:
                sys.path.insert(0, str(root_dir.parent))
            if str(root_dir) not in sys.path:
                sys.path.insert(0, str(root_dir))
            with _MODULE_IMPORT_LOCK:
                module = importlib.import_module(full_module_name)
            import_time = time.time() - module_start

            logger.debug(
                f"Loaded MCP module: {full_module_name} (import: {import_time:.3f}s)"
            )

            # Special handling for llm module initialization
            if full_module_name == "gnn.llm.mcp":
                if hasattr(module, "initialize_llm_module") and callable(
                    module.initialize_llm_module
                ):
                    logger.debug(
                        f"Calling initialize_llm_module for {full_module_name}"
                    )
                    module.initialize_llm_module(self)
                else:
                    logger.warning(
                        f"Module {full_module_name} does not have a callable initialize_llm_module function."
                    )

            # Register tools and resources from the module
            register_start = time.time()
            if hasattr(module, "register_tools") and callable(module.register_tools):
                # The tools_added delta must be computed atomically: under the
                # concurrent discovery executor a sibling module can otherwise
                # register between the before/after length reads and have its
                # tools misattributed to this module (e.g. the zero-tool `doc`
                # module spuriously reporting one tool).
                with self._registration_lock:
                    tools_before = len(self.tools)
                    resources_before = len(self.resources)

                    with self._tool_registration_context(
                        module=f"gnn.{module_name}", category=module_name
                    ):
                        module.register_tools(self)

                    tools_added = len(self.tools) - tools_before
                    resources_added = len(self.resources) - resources_before
                register_time = time.time() - register_start

                module_load_time = time.time() - module_start

                # Create module info with enhanced metadata
                self.modules[module_name] = MCPModuleInfo(
                    name=full_module_name,
                    path=mcp_file,
                    tools_count=tools_added,
                    resources_count=resources_added,
                    status="loaded",
                    load_time=module_load_time,
                    import_time=import_time,
                    register_time=register_time,
                    version=getattr(module, "__version__", "1.0.0"),
                    description=getattr(module, "__description__", ""),
                    dependencies=getattr(module, "__dependencies__", []),
                    last_updated=time.time(),
                )

                # Update performance metrics
                self._performance_metrics.module_load_times[module_name] = (
                    module_load_time
                )

                logger.info(
                    f"Successfully loaded module {module_name}: "
                    f"{tools_added} tools, {resources_added} resources "
                    f"(load: {module_load_time:.3f}s, import: {import_time:.3f}s, register: {register_time:.3f}s)"
                )
                return True
            else:
                logger.warning(
                    f"Module {full_module_name} found but has no register_tools function."
                )
                self.modules[module_name] = MCPModuleInfo(
                    name=full_module_name,
                    path=mcp_file,
                    status="no_register_function",
                    import_time=import_time,
                    last_updated=time.time(),
                )
                return False

        except Exception as e:
            module_load_time = time.time() - module_start
            logger.error(f"Failed to load MCP module {full_module_name}: {str(e)}")

            self.modules[module_name] = MCPModuleInfo(
                name=full_module_name,
                path=mcp_file,
                status="error",
                error_message=str(e),
                load_time=module_load_time,
                last_updated=time.time(),
            )
            return False

    @staticmethod
    def _configure_local_imports(root_dir: Path) -> None:
        """Keep local source packages ahead of same-named installed packages."""
        if str(root_dir.parent) not in sys.path:
            sys.path.insert(0, str(root_dir.parent))
        if str(root_dir) not in sys.path:
            sys.path.insert(0, str(root_dir))

        local_names = {
            name
            for name in (
                "advanced_visualization",
                "analysis",
                "gnn",
                "gui",
                "intelligent_analysis",
                "llm",
                "pipeline",
                "type_checker",
                "visualization",
            )
            if (root_dir / name).is_dir()
        }
        if not local_names:
            return

        root_resolved = root_dir.resolve()
        for loaded_name in list(sys.modules):
            top_level_name = loaded_name.split(".", 1)[0]
            if top_level_name not in local_names:
                continue

            module = sys.modules.get(loaded_name)
            module_file = getattr(module, "__file__", None)
            if not module_file:
                sys.modules.pop(loaded_name, None)
                continue

            try:
                module_path = Path(module_file).resolve()
                module_path.relative_to(root_resolved / top_level_name)
            except (OSError, ValueError):
                sys.modules.pop(loaded_name, None)

    @contextmanager
    def _tool_registration_context(self, module: str, category: str) -> Any:
        """Attach default module metadata while a module registers MCP tools."""
        previous = getattr(self._registration_context, "value", None)
        self._registration_context.value = {"module": module, "category": category}
        try:
            yield
        finally:
            if previous is None:
                try:
                    del self._registration_context.value
                except AttributeError:
                    pass
            else:
                self._registration_context.value = previous

    def _default_tool_metadata(self, module: str, category: str) -> Tuple[str, str]:
        """Return explicit metadata or discovery-context defaults."""
        context = getattr(self._registration_context, "value", {}) or {}
        resolved_module = module or str(context.get("module") or "")
        resolved_category = category or str(context.get("category") or "")
        if not resolved_category and resolved_module:
            resolved_category = resolved_module.rsplit(".", 1)[-1]
        return resolved_module, resolved_category
