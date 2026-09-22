"""
GNN Executor Module

This module provides the main execution functionality for GNN models,
including script execution, simulation management, and result collection.
"""

import functools
import json
import logging
import subprocess  # nosec B404
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, cast

from .result_cache import ExecutionResultCache, cache_key_for_script
from .security_gate import check_script_allowed
from .subprocess_envelope import CancelToken, run_subprocess_envelope
from .types import _EXECUTABLE_SUFFIXES, DEFAULT_RUNNER_TIMEOUT_SECONDS


# Lazy runner registry: each backend's runner imports only when a spec is
# resolved or a runner is invoked, keeping module import free of heavy
# optional dependencies (jax, torch, discopy, ...).
@dataclass(frozen=True)
class _RunnerState:
    """Availability verdict plus runner callable for one backend."""

    available: bool
    runner: Any


def _load_pymdp() -> tuple[bool, Any]:
    from .pymdp.pymdp_runner import run_pymdp_scripts

    return True, run_pymdp_scripts


def _load_rxinfer() -> tuple[bool, Any]:
    from .rxinfer.rxinfer_runner import run_rxinfer_scripts

    return True, run_rxinfer_scripts


def _load_discopy() -> tuple[bool, Any]:
    from .discopy.discopy_executor import run_discopy_analysis

    return True, run_discopy_analysis


def _load_activeinference() -> tuple[bool, Any]:
    from .activeinference_jl.activeinference_runner import run_activeinference_analysis

    return True, run_activeinference_analysis


def _load_jax() -> tuple[bool, Any]:
    from .jax.jax_runner import run_jax_scripts

    return True, run_jax_scripts


def _load_numpyro() -> tuple[bool, Any]:
    from .numpyro.numpyro_runner import run_numpyro_scripts

    return True, run_numpyro_scripts


def _load_pytorch() -> tuple[bool, Any]:
    from .pytorch.pytorch_runner import run_pytorch_scripts

    return True, run_pytorch_scripts


def _load_ngclearn() -> tuple[bool, Any]:
    from .ngclearn.ngclearn_runner import is_ngclearn_available, run_ngclearn_scripts

    # ngclearn itself is a marker-gated extra (py3.12+); the probe inside the
    # runner reports importability so an absent runtime yields a SKIPPED
    # record, never a runner failure.
    return is_ngclearn_available(), run_ngclearn_scripts


def _load_lean() -> tuple[bool, Any]:
    from .lean.lean_runner import lean_toolchain_available, run_lean_scripts

    return lean_toolchain_available(), run_lean_scripts


def _load_stan() -> tuple[bool, Any]:
    from .stan.stan_runner import is_stan_available, run_stan_scripts

    return is_stan_available(), run_stan_scripts


_RUNNER_LOADERS: dict[str, Callable[[], tuple[bool, Any]]] = {
    "pymdp": _load_pymdp,
    "rxinfer": _load_rxinfer,
    "discopy": _load_discopy,
    "activeinference_jl": _load_activeinference,
    "jax": _load_jax,
    "numpyro": _load_numpyro,
    "pytorch": _load_pytorch,
    "lean": _load_lean,
    "ngclearn": _load_ngclearn,
    "stan": _load_stan,
}


@functools.cache
def _runner_state(framework_dir_key: str) -> _RunnerState:
    """Resolve one backend's availability and runner, cached per key."""
    loader = _RUNNER_LOADERS.get(framework_dir_key)
    if loader is None:
        return _RunnerState(False, None)
    try:
        available, runner = loader()
    except ImportError:
        return _RunnerState(False, None)
    return _RunnerState(available, runner)


from gnn.pipeline.config import get_output_dir_for_script
from gnn.utils import performance_tracker
from gnn.utils.logging_utils import (
    log_step_error,
    log_step_start,
    log_step_success,
    log_step_warning,
)

logger = logging.getLogger(__name__)

# Shared execution result cache. Off by default (GNN_EXEC_CACHE opt-in);
# GNNExecutor and execute_script_safely fall back to it when no explicit
# cache instance is passed.
_EXECUTION_RESULT_CACHE = ExecutionResultCache()

FRAMEWORK_DIR_NAMES: tuple[str, ...] = (
    "pymdp",
    "rxinfer",
    "discopy",
    "activeinference_jl",
    "jax",
    "numpyro",
    "pytorch",
    "ngclearn",
    "lean",
    "stan",
    "bnlearn",
)


@dataclass(frozen=True)
class ExecutorFrameworkSpec:
    """Runtime wiring for one rendered-simulator execution backend."""

    framework_dir_key: str
    result_key: str
    available: bool
    runner: Any
    operation_name: str
    start_message: str
    success_message: str
    failure_message: str
    unavailable_log: str
    unavailable_message: str
    success_log: str
    warning_log_prefix: str


# Provide a simple hardware detection function used in tests for patching
def get_available_hardware() -> list[str]:
    """Return available hardware."""
    try:
        import jax  # noqa: F401

        return ["cpu", "gpu"]
    except Exception as e:
        logger.debug("jax import failed; falling back to cpu: %s", e)
        return ["cpu"]


class GNNExecutor:
    """
    Main executor for GNN model simulations and scripts.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        cache: Optional[ExecutionResultCache] = None,
    ) -> None:
        """
        Initialize the GNN executor.

        Args:
            output_dir: Directory for execution outputs
            cache: Execution result cache for subprocess dispatches
                (None → shared module-level cache instance)
        """
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Default to a subdirectory within the project root
            self.output_dir = (
                Path(__file__).parent.parent.parent / "output" / "12_execute_output"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.execution_log: list[dict[str, Any]] = []
        self._cache = cache if cache is not None else _EXECUTION_RESULT_CACHE

    def execute_gnn_model(
        self,
        model_path: str,
        execution_type: str = "pymdp",
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        cancel_token: Optional[CancelToken] = None,
    ) -> Dict[str, Any]:
        """
        Execute a GNN model with the specified execution type.

        Args:
            model_path: Path to the GNN model or rendered script
            execution_type: Type of execution (pymdp, rxinfer, discopy, etc.)
            options: Additional execution options
            cancel_token: Optional cooperative cancellation token threaded
                into the subprocess dispatches (lean is not cancellable yet)

        Returns:
            Dictionary with execution results
        """
        try:
            # SC-1: pre-execution security gate — same shared helper as the
            # Step 12 processor path. GNNExecutor runs rendered scripts; the
            # MCP tools (execute_gnn_model_mcp → execute_simulation_from_gnn)
            # reach execution only through this dispatch, so one gate here
            # covers every script GNNExecutor is about to run.
            if Path(model_path).suffix.lower() in _EXECUTABLE_SUFFIXES:
                gate_verdict = check_script_allowed(Path(model_path))
            elif execution_type == "lean" and Path(model_path).suffix.lower() == ".md":
                # Lean dispatch executes .md documents through the fep-lean
                # bridge (lean_runner.verify_document), so they are gate-
                # checked: the shared helper scans their fenced code blocks
                # with the same rendered-script verdict machinery.
                gate_verdict = check_script_allowed(Path(model_path))
            else:
                # Model sources that this dispatch only parses as data (e.g.
                # .md under pymdp/jax) are not executed here; the scanner only
                # accepts executable scripts, so skip the gate.
                gate_verdict = {"ok": True, "overridden": False, "blocked": []}
            if gate_verdict["overridden"]:
                logger.warning(
                    "GNN_ALLOW_UNSAFE_EXEC set: pre-execution security gate "
                    "bypassed for %s (trusted-local use only)",
                    model_path,
                )
            if not gate_verdict["ok"]:
                return {
                    "success": False,
                    "error": (
                        f"Pre-execution security gate blocked {model_path}: "
                        f"{gate_verdict['reason']}"
                    ),
                    "error_type": gate_verdict.get("error_type", "SecurityGateBlocked"),
                    "security_findings": gate_verdict["blocked"],
                    "execution_type": execution_type,
                    "model_path": model_path,
                }

            start_time = time.time()

            if execution_type == "pymdp":
                result = self._execute_pymdp_script(
                    model_path, options, timeout=timeout, cancel_token=cancel_token
                )
            elif execution_type == "rxinfer":
                result = self._execute_rxinfer_config(
                    model_path, options, timeout=timeout, cancel_token=cancel_token
                )
            elif execution_type == "discopy":
                result = self._execute_discopy_diagram(
                    model_path, options, timeout=timeout, cancel_token=cancel_token
                )
            elif execution_type == "jax":
                result = self._execute_jax_script(
                    model_path, options, timeout=timeout, cancel_token=cancel_token
                )
            elif execution_type == "lean":
                # verify_document has no cancellation hook yet; threading a
                # CancelToken through the fep-lean bridge is a follow-up.
                result = self._execute_lean_verification(
                    model_path, options, timeout=timeout
                )
            elif execution_type == "activeinference_jl":
                result = self._execute_activeinference_script(
                    model_path, options, timeout=timeout, cancel_token=cancel_token
                )
            elif execution_type == "numpyro":
                result = self._execute_numpyro_script(
                    model_path, options, timeout=timeout, cancel_token=cancel_token
                )
            elif execution_type == "pytorch":
                result = self._execute_pytorch_script(
                    model_path, options, timeout=timeout, cancel_token=cancel_token
                )
            elif execution_type == "ngclearn":
                result = self._execute_ngclearn_script(
                    model_path, options, timeout=timeout, cancel_token=cancel_token
                )
            elif execution_type == "stan":
                result = self._execute_stan_script(model_path, options, timeout=timeout)
            elif execution_type == "bnlearn":
                result = self._execute_bnlearn_script()
            else:
                result = {
                    "success": False,
                    "error": f"Unsupported execution type: {execution_type}",
                }

            execution_time = time.time() - start_time
            result["execution_time"] = execution_time
            result["execution_type"] = execution_type
            result["model_path"] = model_path
            # Hardware context
            try:
                devices = get_available_hardware()
                result.setdefault("execution_device", devices[0] if devices else "cpu")
            except Exception as e:
                logger.debug("Device discovery failed; falling back to cpu: %s", e)
                result.setdefault("execution_device", "cpu")

            # Log execution
            self.execution_log.append(result)

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_type": execution_type,
                "model_path": model_path,
            }

    def run_simulation(self, simulation_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a simulation based on configuration.

        Args:
            simulation_config: Configuration dictionary for the simulation

        Returns:
            Dictionary with simulation results
        """
        try:
            model_path = simulation_config.get("model_path")
            execution_type = simulation_config.get("execution_type", "pymdp")
            options = simulation_config.get("options", {})

            if not model_path:
                return {
                    "success": False,
                    "error": "No model path specified in simulation config",
                }

            return self.execute_gnn_model(
                model_path,
                execution_type,
                options,
                timeout=simulation_config.get("timeout"),
            )

        except Exception as e:
            return {"success": False, "error": str(e), "error_type": type(e).__name__}

    def generate_execution_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate an execution report from the execution log.

        Args:
            output_file: Path for the output report file

        Returns:
            Path to the generated report
        """
        if not output_file:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"execution_report_{timestamp}.json"
        else:
            output_path = Path(output_file)

        report_data: dict[str, Any] = {
            "execution_summary": {
                "total_executions": len(self.execution_log),
                "successful_executions": sum(
                    1 for r in self.execution_log if r.get("success", False)
                ),
                "failed_executions": sum(
                    1 for r in self.execution_log if not r.get("success", False)
                ),
                "total_execution_time": sum(
                    r.get("execution_time", 0) for r in self.execution_log
                ),
            },
            "execution_details": self.execution_log,
        }

        try:
            with open(output_path, "w") as f:
                json.dump(report_data, f, indent=2)
        except OSError as e:
            raise RuntimeError(
                f"Failed to write execution report to {output_path}: {e}"
            ) from e

        return str(output_path)

    def _execute_lean_verification(
        self,
        model_path: str,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Verify one document via the fep_lean bridge (contract v0.6)."""
        state = _runner_state("lean")
        if not state.available or state.runner is None:
            return {"success": False, "error": "fep_lean unavailable"}

        from .lean.lean_runner import verify_document

        opts = options or {}
        return verify_document(
            model_path,
            opts.get("receipt"),
            model=opts.get("model", "finite"),
            fail_on_warnings=bool(opts.get("fail_on_warnings", True)),
            timeout=opts.get("timeout") or timeout or 1800,
        )

    def _dispatch_cache_lookup(
        self, interpreter: str, content_path: Union[str, Path]
    ) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Consult the execution cache before one subprocess dispatch.

        The key covers the exact argv ``[interpreter, content_path]``: the
        interpreter and the hashed content of the script/config file, with
        no cwd/env overrides and captured output (the dispatch defaults).

        Returns:
            (cache_key, cached_envelope); cache_key is None when the cache
            is disabled, cached_envelope is None on miss.
        """
        if not self._cache.enabled:
            return None, None
        cache_key = cache_key_for_script(content_path, interpreter=interpreter)
        return cache_key, self._cache.lookup(cache_key)

    def _store_dispatch_envelope(
        self, cache_key: Optional[str], envelope: Dict[str, Any]
    ) -> None:
        """Store a successful dispatch envelope under its cache key."""
        if cache_key is not None and envelope.get("success"):
            self._cache.store(cache_key, envelope)

    def _execute_pymdp_script(
        self,
        script_path: str,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        *,
        cancel_token: Optional[CancelToken] = None,
    ) -> Dict[str, Any]:
        """Execute a PyMDP script with graceful recovery for tests."""
        script = Path(script_path)
        if script.suffix.lower() not in {".py"}:
            return {
                "success": True,
                "stdout": f"Input {script.name} treated as source model; render/execute pipeline required for full simulation.",
                "stderr": "",
                "return_code": 0,
            }
        cache_key, cached = self._dispatch_cache_lookup(sys.executable, script_path)
        if cached is not None:
            cached["cache_hit"] = True
            return cached
        envelope = run_subprocess_envelope(
            [sys.executable, script_path],
            timeout=timeout if timeout is not None else DEFAULT_RUNNER_TIMEOUT_SECONDS,
            cancel_token=cancel_token,
        )
        self._store_dispatch_envelope(cache_key, envelope)
        return envelope

    def _execute_rxinfer_config(
        self,
        config_path: str,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        *,
        cancel_token: Optional[CancelToken] = None,
    ) -> Dict[str, Any]:
        """Execute an RxInfer.jl script or TOML config via the committed runner.

        Both .jl and .toml inputs route through ``execute_rxinfer_script``
        (the committed RxInfer.jl project environment); this dispatch
        synthesizes its envelope from the runner's verdict and the evidence
        sidecars it persists beside the script.
        """
        cache_key, cached = self._dispatch_cache_lookup("julia", config_path)
        if cached is not None:
            cached["cache_hit"] = True
            return cached
        from .rxinfer.rxinfer_runner import execute_rxinfer_script

        script = Path(config_path)
        success = execute_rxinfer_script(
            script, timeout=timeout or 300, cancel_token=cancel_token
        )
        envelope = _synthesize_rxinfer_envelope(script, success)
        self._store_dispatch_envelope(cache_key, envelope)
        return envelope

    def _execute_discopy_diagram(
        self,
        diagram_path: str,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        *,
        cancel_token: Optional[CancelToken] = None,
    ) -> Dict[str, Any]:
        """Execute a DisCoPy diagram."""
        cache_key, cached = self._dispatch_cache_lookup(sys.executable, diagram_path)
        if cached is not None:
            cached["cache_hit"] = True
            return cached
        envelope = run_subprocess_envelope(
            [sys.executable, diagram_path],
            timeout=timeout or 300,
            cancel_token=cancel_token,
        )
        self._store_dispatch_envelope(cache_key, envelope)
        return envelope

    def _execute_jax_script(
        self,
        script_path: str,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        *,
        cancel_token: Optional[CancelToken] = None,
    ) -> Dict[str, Any]:
        """Execute a JAX script."""
        cache_key, cached = self._dispatch_cache_lookup(sys.executable, script_path)
        if cached is not None:
            cached["cache_hit"] = True
            return cached
        envelope = run_subprocess_envelope(
            [sys.executable, script_path],
            timeout=timeout or 300,
            cancel_token=cancel_token,
        )
        self._store_dispatch_envelope(cache_key, envelope)
        return envelope

    def _execute_python_script(
        self,
        script_path: str,
        timeout: Optional[int] = None,
        *,
        cancel_token: Optional[CancelToken] = None,
    ) -> Dict[str, Any]:
        """Run one rendered Python framework script under ``sys.executable``.

        Shared by the numpyro, pytorch, and ngclearn dispatches; mirrors the
        JAX envelope contract.
        """
        cache_key, cached = self._dispatch_cache_lookup(sys.executable, script_path)
        if cached is not None:
            cached["cache_hit"] = True
            return cached
        envelope = run_subprocess_envelope(
            [sys.executable, script_path],
            timeout=timeout or 300,
            cancel_token=cancel_token,
        )
        self._store_dispatch_envelope(cache_key, envelope)
        return envelope

    def _execute_numpyro_script(
        self,
        script_path: str,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        *,
        cancel_token: Optional[CancelToken] = None,
    ) -> Dict[str, Any]:
        """Execute a NumPyro script."""
        return self._execute_python_script(
            script_path, timeout, cancel_token=cancel_token
        )

    def _execute_pytorch_script(
        self,
        script_path: str,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        *,
        cancel_token: Optional[CancelToken] = None,
    ) -> Dict[str, Any]:
        """Execute a PyTorch script."""
        return self._execute_python_script(
            script_path, timeout, cancel_token=cancel_token
        )

    def _execute_ngclearn_script(
        self,
        script_path: str,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        *,
        cancel_token: Optional[CancelToken] = None,
    ) -> Dict[str, Any]:
        """Execute an ngc-learn script."""
        return self._execute_python_script(
            script_path, timeout, cancel_token=cancel_token
        )

    def _execute_activeinference_script(
        self,
        script_path: str,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        *,
        cancel_token: Optional[CancelToken] = None,
    ) -> Dict[str, Any]:
        """Execute a rendered ActiveInference.jl script."""
        script = Path(script_path)
        project_dir = Path(__file__).parent / "activeinference_jl"
        cache_key, cached = self._dispatch_cache_lookup("julia", script_path)
        if cached is not None:
            cached["cache_hit"] = True
            return cached
        from .julia_env import julia_subprocess_env

        env = julia_subprocess_env()
        env.setdefault("JULIA_PROJECT", str(project_dir))
        envelope = run_subprocess_envelope(
            ["julia", f"--project={project_dir}", str(script)],  # nosec B607 B603
            timeout=timeout or 600,
            env=env,
            cwd=str(script.parent),
            cancel_token=cancel_token,
        )
        self._store_dispatch_envelope(cache_key, envelope)
        return envelope

    def _execute_stan_script(
        self,
        script_path: str,
        options: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute a rendered Stan driver; skip cleanly without cmdstanpy."""
        state = _runner_state("stan")
        if not state.available:
            return {
                "success": False,
                "skipped": True,
                "status": "skipped",
                "error": "cmdstanpy/CmdStan not installed (uv sync --extra stan)",
            }
        from .stan.stan_runner import execute_stan_script

        opts = options or {}
        output_dir = opts.get("output_dir") or Path(script_path).parent
        return execute_stan_script(script_path, output_dir, timeout=timeout or 1800)

    def _execute_bnlearn_script(self) -> Dict[str, Any]:
        """bnlearn is render-only: the executor registry never runs it."""
        return {
            "success": False,
            "skipped": True,
            "status": "skipped",
            "error": (
                "bnlearn is render-only; rendered bnlearn scripts execute via "
                "the Step 12 script path (BNLEARN_OUTPUT_DIR) with dependency skips"
            ),
        }


    def execute_simulation_from_gnn(
        self,
        gnn_file: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute a simulation from a GNN file path."""
        gnn_path = Path(gnn_file) if not isinstance(gnn_file, Path) else gnn_file
        out_dir = Path(output_dir) if output_dir is not None else self.output_dir
        sim_cfg: dict[str, Any] = {
            "model_path": str(gnn_path),
            "execution_type": "pymdp",
            "options": {"output_dir": str(out_dir)},
        }
        if timeout is not None:
            sim_cfg["timeout"] = timeout
        return self.run_simulation(sim_cfg)


def execute_gnn_model(
    model_path: str,
    execution_type: Union[str, Path] = "pymdp",
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to execute a GNN model.

    Args:
        model_path: Path to the GNN model or rendered script
        execution_type: Type of execution
        options: Additional execution options

    Returns:
        Dictionary with execution results
    """
    normalized_execution_type: str = "pymdp"
    normalized_options = options

    exec_type_obj: object = execution_type
    if isinstance(exec_type_obj, Path):
        normalized_options = dict(options or {})
        normalized_options.setdefault("output_dir", str(exec_type_obj))
    elif isinstance(exec_type_obj, str):
        normalized_execution_type = exec_type_obj
    else:
        normalized_options = dict(options or {})
        normalized_options.setdefault("output_dir", str(exec_type_obj))

    executor = GNNExecutor()
    result = executor.execute_gnn_model(
        model_path, normalized_execution_type, normalized_options
    )
    result.setdefault("status", "SUCCESS" if result.get("success") else "FAILED")
    return result


def run_simulation(simulation_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to run a simulation.

    Args:
        simulation_config: Configuration dictionary for the simulation

    Returns:
        Dictionary with simulation results
    """
    executor = GNNExecutor()
    return executor.run_simulation(simulation_config)


def generate_execution_report(
    execution_log: List[Dict[str, Any]], output_file: Optional[str] = None
) -> str:
    """
    Convenience function to generate an execution report.

    Args:
        execution_log: List of execution results
        output_file: Path for the output report file

    Returns:
        Path to the generated report
    """
    executor = GNNExecutor()
    executor.execution_log = execution_log
    return executor.generate_execution_report(output_file)


def _read_text_sidecar(path: Path) -> str:
    """Read a persisted sidecar file, or return an empty string if absent."""
    try:
        return path.read_text()
    except OSError:
        return ""


def _synthesize_rxinfer_envelope(script_path: Path, success: bool) -> Dict[str, Any]:
    """Build a dispatch envelope from ``execute_rxinfer_script``'s verdict.

    The committed runner persists ``{stem}_stdout.txt`` / ``{stem}_stderr.txt``
    and an ``{stem}_execution_log.json`` beside the script; stdout/stderr are
    read back from those sidecars and the elapsed time from the log when
    present.
    """
    stem = script_path.stem
    envelope: Dict[str, Any] = {
        "success": bool(success),
        "return_code": 0 if success else 1,
        "stdout": _read_text_sidecar(script_path.parent / f"{stem}_stdout.txt"),
        "stderr": _read_text_sidecar(script_path.parent / f"{stem}_stderr.txt"),
    }
    try:
        log = json.loads(
            _read_text_sidecar(script_path.parent / f"{stem}_execution_log.json")
        )
        envelope["elapsed_seconds"] = float(log["elapsed_seconds"])
    except (ValueError, KeyError, TypeError):
        pass
    return envelope


def _run_stan_registry(
    rendered_simulators_dir: Path,
    execution_output_dir: Path,
    recursive_search: bool,  # noqa: ARG001 - uniform registry runner signature
    verbose: bool,
    timeout: Optional[int],
) -> bool:
    """Registry adapter mapping ``run_stan_scripts`` records to the bool contract.

    ``run_stan_scripts`` already emits skip receipts (never FAILED) when
    cmdstanpy is missing; the run succeeds when every record is a success
    or a skip, including the empty-tree case.
    """
    from .stan.stan_runner import run_stan_scripts

    records = run_stan_scripts(
        render_output_dir=rendered_simulators_dir,
        output_dir=execution_output_dir,
        timeout=timeout or 1800,
    )
    return all(
        bool(record.get("success")) or bool(record.get("skipped"))
        for record in records
    )


def _framework_specs() -> tuple[ExecutorFrameworkSpec, ...]:
    """Return framework specs with availability resolved via _runner_state."""
    pymdp_state = _runner_state("pymdp")
    rxinfer_state = _runner_state("rxinfer")
    discopy_state = _runner_state("discopy")
    activeinference_state = _runner_state("activeinference_jl")
    jax_state = _runner_state("jax")
    numpyro_state = _runner_state("numpyro")
    pytorch_state = _runner_state("pytorch")
    ngclearn_state = _runner_state("ngclearn")
    lean_state = _runner_state("lean")
    stan_state = _runner_state("stan")
    return (
        ExecutorFrameworkSpec(
            framework_dir_key="pymdp",
            result_key="pymdp_executions",
            available=pymdp_state.available,
            runner=pymdp_state.runner,
            operation_name="execute_pymdp_scripts",
            start_message="🚀 Executing PyMDP scripts...",
            success_message="PyMDP scripts executed successfully",
            failure_message="PyMDP script execution failed",
            unavailable_log=(
                "ℹ️ PyMDP framework not available - skipping PyMDP execution "
                "(install with: uv pip install inferactively-pymdp)"
            ),
            unavailable_message="PyMDP framework not installed (optional dependency)",
            success_log="PyMDP script execution completed",
            warning_log_prefix="PyMDP script execution failed",
        ),
        ExecutorFrameworkSpec(
            framework_dir_key="rxinfer",
            result_key="rxinfer_executions",
            available=rxinfer_state.available,
            runner=rxinfer_state.runner,
            operation_name="execute_rxinfer_scripts",
            start_message="🚀 Executing RxInfer scripts...",
            success_message="RxInfer scripts executed successfully",
            failure_message="RxInfer script execution failed",
            unavailable_log=(
                "ℹ️ RxInfer framework not available - skipping RxInfer execution "
                "(requires Julia and RxInfer.jl)"
            ),
            unavailable_message=(
                "RxInfer framework not installed (optional dependency - requires Julia)"
            ),
            success_log="RxInfer script execution completed",
            warning_log_prefix="RxInfer script execution failed",
        ),
        ExecutorFrameworkSpec(
            framework_dir_key="discopy",
            result_key="discopy_executions",
            available=discopy_state.available,
            runner=discopy_state.runner,
            operation_name="execute_discopy_analysis",
            start_message="🚀 Executing DisCoPy analysis...",
            success_message="DisCoPy analysis completed successfully",
            failure_message="DisCoPy analysis failed",
            unavailable_log=(
                "ℹ️ DisCoPy framework not available - skipping DisCoPy execution "
                "(install with: uv pip install discopy)"
            ),
            unavailable_message="DisCoPy framework not installed (optional dependency)",
            success_log="DisCoPy analysis completed",
            warning_log_prefix="DisCoPy analysis failed",
        ),
        ExecutorFrameworkSpec(
            framework_dir_key="activeinference_jl",
            result_key="activeinference_executions",
            available=activeinference_state.available,
            runner=activeinference_state.runner,
            operation_name="execute_activeinference_analysis",
            start_message="🚀 Executing ActiveInference.jl analysis...",
            success_message="ActiveInference.jl analysis completed successfully",
            failure_message="ActiveInference.jl analysis failed",
            unavailable_log=(
                "ℹ️ ActiveInference.jl framework not available - skipping "
                "(requires Julia and ActiveInference.jl)"
            ),
            unavailable_message=(
                "ActiveInference.jl framework not installed "
                "(optional dependency - requires Julia)"
            ),
            success_log="ActiveInference.jl analysis completed",
            warning_log_prefix="ActiveInference.jl analysis failed",
        ),
        ExecutorFrameworkSpec(
            framework_dir_key="jax",
            result_key="jax_executions",
            available=jax_state.available,
            runner=jax_state.runner,
            operation_name="execute_jax_scripts",
            start_message="🚀 Executing JAX scripts...",
            success_message="JAX scripts executed successfully",
            failure_message="JAX script execution failed",
            unavailable_log=(
                "ℹ️ JAX framework not available - skipping JAX execution "
                "(install with: uv pip install jax jaxlib)"
            ),
            unavailable_message="JAX framework not installed (optional dependency)",
            success_log="JAX script execution completed",
            warning_log_prefix="JAX script execution failed",
        ),
        ExecutorFrameworkSpec(
            framework_dir_key="numpyro",
            result_key="numpyro_executions",
            available=numpyro_state.available,
            runner=numpyro_state.runner,
            operation_name="execute_numpyro_scripts",
            start_message="🚀 Executing NumPyro scripts...",
            success_message="NumPyro scripts executed successfully",
            failure_message="NumPyro script execution failed",
            unavailable_log=(
                "ℹ️ NumPyro framework not available - skipping NumPyro execution "
                "(install with: uv pip install numpyro jax jaxlib)"
            ),
            unavailable_message="NumPyro framework not installed (optional dependency)",
            success_log="NumPyro script execution completed",
            warning_log_prefix="NumPyro script execution failed",
        ),
        ExecutorFrameworkSpec(
            framework_dir_key="pytorch",
            result_key="pytorch_executions",
            available=pytorch_state.available,
            runner=pytorch_state.runner,
            operation_name="execute_pytorch_scripts",
            start_message="🚀 Executing PyTorch scripts...",
            success_message="PyTorch scripts executed successfully",
            failure_message="PyTorch script execution failed",
            unavailable_log=(
                "ℹ️ PyTorch framework not available - skipping PyTorch execution "
                "(install with: uv pip install torch)"
            ),
            unavailable_message="PyTorch framework not installed (optional dependency)",
            success_log="PyTorch script execution completed",
            warning_log_prefix="PyTorch script execution failed",
        ),
        ExecutorFrameworkSpec(
            framework_dir_key="ngclearn",
            result_key="ngclearn_executions",
            available=ngclearn_state.available,
            runner=ngclearn_state.runner,
            operation_name="execute_ngclearn_scripts",
            start_message="🚀 Executing ngc-learn scripts...",
            success_message="ngc-learn scripts executed successfully",
            failure_message="ngc-learn script execution failed",
            unavailable_log=(
                "ℹ️ ngc-learn framework not available - skipping ngc-learn execution "
                "(install with: uv sync --extra ngclearn)"
            ),
            unavailable_message=(
                "ngc-learn framework not installed "
                "(optional dependency - install with: uv sync --extra ngclearn)"
            ),
            success_log="ngc-learn script execution completed",
            warning_log_prefix="ngc-learn script execution failed",
        ),
        ExecutorFrameworkSpec(
            framework_dir_key="lean",
            result_key="lean_executions",
            available=lean_state.available,
            runner=lean_state.runner,
            operation_name="execute_lean_verification",
            start_message="🚀 Verifying documents with fep_lean (Lean 4)...",
            success_message="Lean verification completed successfully",
            failure_message="Lean verification failed",
            unavailable_log=(
                "ℹ️ fep_lean not available - skipping Lean verification "
                "(set FEP_LEAN_ROOT to the fep_lean checkout)"
            ),
            unavailable_message=(
                "fep_lean not available (optional dependency - set FEP_LEAN_ROOT)"
            ),
            success_log="Lean document verification completed",
            warning_log_prefix="Lean verification failed",
        ),
        ExecutorFrameworkSpec(
            framework_dir_key="stan",
            result_key="stan_executions",
            available=stan_state.available,
            runner=_run_stan_registry,
            operation_name="execute_stan_scripts",
            start_message="🚀 Executing Stan drivers...",
            success_message="Stan scripts executed successfully",
            failure_message="Stan script execution failed",
            unavailable_log=(
                "ℹ️ Stan framework not available - skipping Stan execution "
                "(install with: uv sync --extra stan)"
            ),
            unavailable_message=(
                "Stan framework not installed "
                "(cmdstanpy/CmdStan not installed (uv sync --extra stan))"
            ),
            success_log="Stan script execution completed",
            warning_log_prefix="Stan script execution failed",
        ),
        ExecutorFrameworkSpec(
            framework_dir_key="bnlearn",
            result_key="bnlearn_executions",
            available=False,
            runner=None,
            operation_name="execute_bnlearn_scripts",
            start_message="🚀 Executing bnlearn scripts...",
            success_message="bnlearn scripts executed successfully",
            failure_message="bnlearn script execution failed",
            unavailable_log=(
                "ℹ️ bnlearn is render-only - skipping bnlearn execution "
                "(rendered bnlearn scripts execute via the Step 12 script "
                "path (BNLEARN_OUTPUT_DIR) with dependency skips)"
            ),
            unavailable_message=(
                "bnlearn is render-only; rendered bnlearn scripts execute via "
                "the Step 12 script path (BNLEARN_OUTPUT_DIR) with dependency skips"
            ),
            success_log="bnlearn script execution completed",
            warning_log_prefix="bnlearn script execution failed",
        ),
    )


def list_frameworks() -> list[dict[str, Any]]:
    """Introspect the executor framework registry.

    Returns one record per registered backend with its key, the
    ``*_executions`` result key, and whether the backend's runner is currently
    importable. Useful for CLI/MCP diagnostics and tests that want to assert
    the registry shape without importing the private ``_framework_specs``
    helper.
    """
    return [
        {
            "framework": spec.framework_dir_key,
            "result_key": spec.result_key,
            "available": bool(spec.available),
            "operation": spec.operation_name,
        }
        for spec in _framework_specs()
    ]


def _create_framework_dirs(
    execution_output_dir: Path, logger: logging.Logger
) -> dict[str, Path]:
    """Create and return framework-specific execution directories."""
    framework_dirs = {name: execution_output_dir / name for name in FRAMEWORK_DIR_NAMES}
    for framework_dir in framework_dirs.values():
        framework_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created framework directory: {framework_dir}")
    return framework_dirs


def _initialize_execution_results(
    target_dir: Path, framework_dirs: dict[str, Path]
) -> dict[str, Any]:
    """Build the common execution summary envelope.

    The per-framework ``*_executions`` lists are derived from the
    :func:`_framework_specs` registry so adding a framework is a one-line
    change (the registry is the single source of truth for both the result
    keys and the dispatch wiring).
    """
    result: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_directory": str(target_dir),
        "framework_execution_dirs": {k: str(v) for k, v in framework_dirs.items()},
        "total_successes": 0,
        "total_failures": 0,
        "dependency_issues": [],
        "syntax_errors": [],
        "execution_details": {},
    }
    for spec in _framework_specs():
        result[spec.result_key] = []
    return result


def _check_python_dependencies(
    execution_results: dict[str, Any], logger: logging.Logger
) -> None:
    """Record missing Python dependencies before runner execution starts."""
    missing_python_deps: list[str] = []
    for dep in ["numpy", "pymdp", "flax", "jax", "optax"]:
        try:
            __import__(dep)
            logger.debug(f"✅ Python dependency available: {dep}")
        except ImportError:
            missing_python_deps.append(dep)
            logger.warning(f"⚠️ Python dependency missing: {dep}")

    if missing_python_deps:
        execution_results["dependency_issues"].append(
            f"Missing Python dependencies: {', '.join(missing_python_deps)}"
        )


def _check_julia_availability(
    execution_results: dict[str, Any], logger: logging.Logger
) -> None:
    """Record Julia availability for Julia-backed execution frameworks."""
    try:
        result = subprocess.run(
            ["julia", "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )  # nosec B607 B603
        if result.returncode == 0:
            logger.info(f"✅ Julia available: {result.stdout.strip()}")
        else:
            logger.warning("⚠️ Julia not available or not working properly")
            execution_results["dependency_issues"].append("Julia not available")
    except FileNotFoundError:
        logger.warning("⚠️ Julia not found in PATH")
        execution_results["dependency_issues"].append("Julia not found in PATH")


def _validate_pymdp_script_syntax(
    target_dir: Path, execution_results: dict[str, Any], logger: logging.Logger
) -> None:
    """Compile rendered PyMDP scripts so syntax errors appear in the summary."""
    pymdp_dir = target_dir / "pymdp"
    if not pymdp_dir.exists():
        return

    for script in pymdp_dir.glob("*.py"):
        try:
            with open(script, "r") as f:
                compile(f.read(), script.name, "exec")
            logger.debug(f"✅ PyMDP script syntax valid: {script.name}")
        except SyntaxError as e:
            logger.warning(f"⚠️ PyMDP script syntax error in {script.name}: {e}")
            execution_results["syntax_errors"].append(f"PyMDP: {script.name} - {e}")


def _append_framework_result(
    execution_results: dict[str, Any],
    spec: ExecutorFrameworkSpec,
    status: str,
    message: str,
    output_dir: Path,
) -> None:
    """Append a normalized framework execution record."""
    execution_results[spec.result_key].append(
        {
            "status": status,
            "message": message,
            "output_dir": str(output_dir),
        }
    )


def _execute_framework_spec(
    spec: ExecutorFrameworkSpec,
    target_dir: Path,
    framework_dirs: dict[str, Path],
    execution_results: dict[str, Any],
    logger: logging.Logger,
    recursive: bool,
    verbose: bool,
    timeout: Optional[int] = None,
) -> None:
    """Execute one framework runner and record its status."""
    output_dir = framework_dirs[spec.framework_dir_key]
    if not spec.available:
        logger.info(spec.unavailable_log)
        _append_framework_result(
            execution_results, spec, "SKIPPED", spec.unavailable_message, output_dir
        )
        return

    try:
        with performance_tracker.track_operation(spec.operation_name):
            logger.info(spec.start_message)
            if spec.framework_dir_key == "pymdp":
                _validate_pymdp_script_syntax(target_dir, execution_results, logger)

            success = spec.runner(
                rendered_simulators_dir=target_dir,
                execution_output_dir=output_dir,
                recursive_search=recursive,
                verbose=verbose,
                timeout=timeout,
            )

            if success:
                execution_results["total_successes"] += 1
                _append_framework_result(
                    execution_results, spec, "SUCCESS", spec.success_message, output_dir
                )
                log_step_success(logger, spec.success_log)
            else:
                execution_results["total_failures"] += 1
                _append_framework_result(
                    execution_results, spec, "FAILED", spec.failure_message, output_dir
                )
                log_step_warning(logger, spec.failure_message)
    except Exception as e:
        execution_results["total_failures"] += 1
        _append_framework_result(execution_results, spec, "ERROR", str(e), output_dir)
        log_step_warning(logger, f"{spec.warning_log_prefix}: {e}")


def _execute_configured_frameworks(
    target_dir: Path,
    framework_dirs: dict[str, Path],
    execution_results: dict[str, Any],
    logger: logging.Logger,
    recursive: bool,
    verbose: bool,
    timeout: Optional[int] = None,
) -> None:
    """Execute every supported framework according to current availability."""
    for spec in _framework_specs():
        _execute_framework_spec(
            spec,
            target_dir,
            framework_dirs,
            execution_results,
            logger,
            recursive,
            verbose,
            timeout,
        )


def _write_framework_report_section(
    file_obj: Any,
    title: str,
    executions: list[dict[str, Any]],
    default_script: str,
    include_type: bool = False,
) -> None:
    """Write one framework subsection to the markdown execution report."""
    if not executions:
        return

    file_obj.write(f"## {title}\n\n")
    for exec_info in executions:
        status_icon = "✅" if exec_info.get("status") == "SUCCESS" else "❌"
        type_text = f" ({exec_info.get('type', 'analysis')})" if include_type else ""
        file_obj.write(
            f"- {status_icon} **{exec_info.get('script', default_script)}**{type_text}: {exec_info.get('status', 'Unknown')}\n"
        )
        file_obj.write(f"  - {exec_info.get('message', 'No message')}\n")
        file_obj.write(f"  - Output Directory: {exec_info.get('output_dir', 'N/A')}\n")
        if "scripts_processed" in exec_info:
            file_obj.write(f"  - Scripts processed: {exec_info['scripts_processed']}\n")
    file_obj.write("\n")


def _write_execution_report(
    report_file: Path, execution_results: dict[str, Any]
) -> None:
    """Write the enhanced markdown execution report."""
    with open(report_file, "w") as f:
        f.write("# Enhanced Execution Results Report\n\n")
        f.write(f"**Generated:** {execution_results['timestamp']}\n")
        f.write(f"**Target Directory:** {execution_results['target_directory']}\n")
        f.write(f"**Total Successes:** {execution_results['total_successes']}\n")
        f.write(f"**Total Failures:** {execution_results['total_failures']}\n\n")

        f.write("## Framework-Specific Output Directories\n\n")
        for framework, framework_dir in execution_results[
            "framework_execution_dirs"
        ].items():
            f.write(f"- **{framework.upper()}**: {framework_dir}\n")
        f.write("\n")

        if execution_results["dependency_issues"]:
            f.write("## Dependency Issues\n\n")
            for issue in execution_results["dependency_issues"]:
                f.write(f"- ⚠️ {issue}\n")
            f.write("\n")

        if execution_results["syntax_errors"]:
            f.write("## Syntax Errors\n\n")
            for error in execution_results["syntax_errors"]:
                f.write(f"- ❌ {error}\n")
            f.write("\n")

        _write_framework_report_section(
            f,
            "PyMDP Executions",
            execution_results["pymdp_executions"],
            "PyMDP Scripts",
        )
        _write_framework_report_section(
            f,
            "RxInfer Executions",
            execution_results["rxinfer_executions"],
            "RxInfer Scripts",
        )
        _write_framework_report_section(
            f,
            "DisCoPy Analyses",
            execution_results["discopy_executions"],
            "DisCoPy Analysis",
            include_type=True,
        )
        _write_framework_report_section(
            f,
            "ActiveInference.jl Analyses",
            execution_results["activeinference_executions"],
            "ActiveInference.jl Scripts",
        )
        _write_framework_report_section(
            f, "JAX Executions", execution_results["jax_executions"], "JAX Scripts"
        )
        _write_framework_report_section(
            f,
            "NumPyro Executions",
            execution_results["numpyro_executions"],
            "NumPyro Scripts",
        )
        _write_framework_report_section(
            f,
            "PyTorch Executions",
            execution_results["pytorch_executions"],
            "PyTorch Scripts",
        )
        _write_framework_report_section(
            f,
            "ngc-learn Executions",
            execution_results["ngclearn_executions"],
            "ngc-learn Scripts",
        )
        _write_framework_report_section(
            f,
            "Lean verification",
            execution_results["lean_executions"],
            "Lean Documents",
        )
        _write_framework_report_section(
            f,
            "Stan Executions",
            execution_results["stan_executions"],
            "Stan Drivers",
        )
        _write_framework_report_section(
            f,
            "bnlearn Executions",
            execution_results["bnlearn_executions"],
            "bnlearn Scripts",
        )

        f.write("## Recommendations\n\n")
        if execution_results["dependency_issues"]:
            f.write("### Install Missing Dependencies\n\n")
            for issue in execution_results["dependency_issues"]:
                if "Python dependencies" in issue:
                    f.write(
                        "- Install missing Python packages: `uv pip install <package_name>` or add to pyproject and run `uv sync`\n"
                    )
                elif "Julia" in issue:
                    f.write("- Install Julia from https://julialang.org/downloads/\n")
            f.write("\n")

        if execution_results["syntax_errors"]:
            f.write("### Fix Syntax Errors\n\n")
            f.write("- Review and fix syntax errors in rendered scripts\n")
            f.write("- Check for stray characters or malformed code\n")
            f.write(
                "- Re-run the rendering step (src/gnn/11_render.py) to regenerate scripts\n\n"
            )


def _write_execution_artifacts(
    execution_output_dir: Path, execution_results: dict[str, Any]
) -> None:
    """Write JSON and markdown execution summaries."""
    summaries_dir = execution_output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    with open(summaries_dir / "execution_summary.json", "w") as f:
        json.dump(execution_results, f, indent=2)
    _write_execution_report(summaries_dir / "execution_report.md", execution_results)


def _count_framework_execution_records(execution_results: dict[str, Any]) -> int:
    """Count framework result records across all supported backends."""
    return sum(len(execution_results[spec.result_key]) for spec in _framework_specs())


def _log_execution_outcome(
    execution_results: dict[str, Any], logger: logging.Logger
) -> bool:
    """Log aggregate execution outcome and return success status."""
    total_executions = _count_framework_execution_records(execution_results)
    if total_executions == 0:
        log_step_warning(
            logger, "No simulator scripts or outputs found to execute/analyze"
        )
        return True

    success_rate = execution_results["total_successes"] / total_executions * 100
    log_step_success(
        logger,
        f"Execution completed with framework-specific organization. Success rate: {success_rate:.1f}% ({execution_results['total_successes']}/{total_executions})",
    )

    if execution_results["dependency_issues"]:
        logger.warning(
            f"⚠️ Dependency issues found: {len(execution_results['dependency_issues'])}"
        )
    if execution_results["syntax_errors"]:
        logger.warning(
            f"⚠️ Syntax errors found: {len(execution_results['syntax_errors'])}"
        )

    return cast("bool", execution_results["total_failures"] == 0)


def execute_rendered_simulators(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> bool:
    """
    Execute rendered simulator scripts with enhanced error handling and dependency checking.
    Framework outputs are organized in separate subdirectories.

    Args:
        target_dir: Directory containing rendered simulator scripts
        output_dir: Output directory for results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional execution options

    Returns:
        True if execution succeeded, False otherwise
    """
    timeout: Optional[int] = kwargs.pop("timeout", None)
    log_step_start(
        logger,
        "Executing rendered simulator scripts with framework-specific organization",
    )

    execution_output_dir = get_output_dir_for_script("12_execute.py", output_dir)
    execution_output_dir.mkdir(parents=True, exist_ok=True)
    framework_dirs = _create_framework_dirs(execution_output_dir, logger)

    try:
        execution_results = _initialize_execution_results(target_dir, framework_dirs)
        logger.info("🔍 Pre-execution validation and dependency checking...")
        _check_python_dependencies(execution_results, logger)
        _check_julia_availability(execution_results, logger)

        _execute_configured_frameworks(
            target_dir,
            framework_dirs,
            execution_results,
            logger,
            recursive,
            verbose,
            timeout,
        )
        _write_execution_artifacts(execution_output_dir, execution_results)
        return _log_execution_outcome(execution_results, logger)

    except Exception as e:
        log_step_error(logger, f"Execution failed: {e}")
        return False


def execute_script_safely(
    script_path: Union[str, Path],
    timeout: int = 3600,
    capture_output: bool = True,
    cwd: Optional[Union[str, Path]] = None,
    env: Optional[Dict[str, str]] = None,
    args: Optional[Sequence[str]] = None,
    cancel_token: Optional[CancelToken] = None,
    cache: Optional[ExecutionResultCache] = None,
) -> Dict[str, Any]:
    """Execute a Python script via ``subprocess.run`` with a structured envelope.

    Returns a uniform dict regardless of the failure mode so callers never have
    to distinguish between a missing file, a dependency error, a timeout, and a
    non-zero exit code.

    Args:
        script_path:    Path to the ``.py`` script to execute.
        timeout:        Wall-clock timeout in seconds (default ``3600``).
        capture_output: If True, capture stdout/stderr; otherwise stream to the
                        parent process.
        cwd:            Working directory for the subprocess.
        env:            Environment variables override (merged into ``os.environ``).
        args:           Extra command-line arguments passed to the script.
        cancel_token:   Optional cooperative cancellation token; when cancelled
                        mid-flight the run is group-killed and reported as
                        ``error_type="Cancelled"``.
        cache:          Execution result cache; None falls back to the shared
                        module instance (active only when ``GNN_EXEC_CACHE`` is
                        truthy or the instance was built with ``enabled=True``).
                        Successful envelopes are cached; a cache hit short-
                        circuits before the spawn and is flagged ``cache_hit``.

    Returns:
        Dict with keys:
            - ``success`` (bool): True iff the script exited with return code 0.
            - ``script_path`` (str): Resolved script path.
            - ``return_code`` (int): Subprocess exit code (``-1`` if not started).
            - ``stdout`` (str): Captured stdout (empty if ``capture_output`` is False).
            - ``stderr`` (str): Captured stderr (empty if ``capture_output`` is False).
            - ``duration_seconds`` (float): Wall-clock execution time.
            - ``error`` (str, optional): Populated on failure.
            - ``error_type`` (str, optional): Exception class name on failure
              (``"SecurityGateBlocked"`` when the pre-exec gate denies the
              script; ``"SandboxUnavailable"`` when ``GNN_SANDBOX=require``
              found no backend).
            - ``sandbox_mode``/``sandboxed``: GNN_SANDBOX mode and whether the
              command was actually wrapped (set by the shared envelope).
            - ``cache_hit`` (bool, optional): True when the envelope was served
              from the execution cache instead of a fresh spawn.
            - ``cancelled`` (bool): True when the run was cancelled via the
              cancel token (additive envelope key, False on every other path).
    """
    script = Path(script_path)
    if not script.exists():
        return {
            "success": False,
            "script_path": str(script),
            "return_code": -1,
            "stdout": "",
            "stderr": "",
            "duration_seconds": 0.0,
            "error": f"Script not found: {script}",
            "error_type": "FileNotFoundError",
        }
    if script.suffix.lower() != ".py":
        return {
            "success": False,
            "script_path": str(script),
            "return_code": -1,
            "stdout": "",
            "stderr": "",
            "duration_seconds": 0.0,
            "error": (
                f"execute_script_safely only runs Python scripts; got suffix "
                f"{script.suffix!r}"
            ),
            "error_type": "ValueError",
        }

    # SEC-R2: the registry runners (pymdp/jax/numpyro/pytorch/discopy) funnel
    # every rendered-script execution through this helper, so the
    # pre-execution security gate applies here before anything runs.
    gate_verdict = check_script_allowed(script)
    if gate_verdict["overridden"]:
        logger.warning(
            "GNN_ALLOW_UNSAFE_EXEC set: pre-execution security gate "
            "bypassed for %s (trusted-local use only)",
            script,
        )
    if not gate_verdict["ok"]:
        logger.error(
            "Pre-execution security gate blocked %s: %s",
            script,
            gate_verdict["reason"],
        )
        return {
            "success": False,
            "script_path": str(script),
            "return_code": -1,
            "stdout": "",
            "stderr": "",
            "duration_seconds": 0.0,
            "error": (
                f"Pre-execution security gate blocked {script}: "
                f"{gate_verdict['reason']}"
            ),
            "error_type": gate_verdict.get("error_type", "SecurityGateBlocked"),
            "security_findings": gate_verdict["blocked"],
        }

    # Sandboxing (SEC-R2, Step-12 GNN_SANDBOX semantics) lives in the shared
    # envelope: sandbox=True applies the prefix, emits the
    # sandbox_disabled/active receipts, and refuses on require-without-backend.
    script_args = list(args) if args is not None else []
    effective_cache = cache if cache is not None else _EXECUTION_RESULT_CACHE
    cache_key: Optional[str] = None
    if effective_cache.enabled:
        # Cache consult happens AFTER the security gate, so gate-blocked
        # scripts can never be served from (or written into) the cache.
        cache_key = cache_key_for_script(
            script,
            interpreter=sys.executable,
            args=script_args,
            cwd=str(cwd) if cwd is not None else None,
            env_overrides=env,
            capture_output=capture_output,
        )
        cached_envelope = effective_cache.lookup(cache_key)
        if cached_envelope is not None:
            cached_envelope["cache_hit"] = True
            return cached_envelope

    envelope = run_subprocess_envelope(
        [sys.executable, str(script), *script_args],
        timeout=timeout,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        capture_output=capture_output,
        cancel_token=cancel_token,
    )
    envelope["script_path"] = str(script)
    # Failures, timeouts, and cancels are never stored.
    if cache_key is not None and envelope["success"]:
        effective_cache.store(cache_key, envelope)
    return envelope


def clear_execution_cache() -> int:
    """Invalidate the shared execution result cache; returns entries removed."""
    return _EXECUTION_RESULT_CACHE.invalidate()
