"""
Framework spec registry for the GNN executor: the runtime wiring dataclass,
the framework directory-name tuple, the Stan registry adapter, and the
framework spec table behind ``list_frameworks``.

Split from ``gnn.execute.executor`` (W7-A band split); the moved ranges are
byte-verbatim. ``_runner_state`` resolves through ``gnn.execute.executor`` at
call time so executor-namespace monkeypatches stay observable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from gnn.execute.executor import _RunnerState


def _runner_state(framework_dir_key: str) -> "_RunnerState":
    """Resolve through ``gnn.execute.executor`` (call-time indirection)."""
    from gnn.execute import executor

    return executor._runner_state(framework_dir_key)


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
        bool(record.get("success")) or bool(record.get("skipped")) for record in records
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
