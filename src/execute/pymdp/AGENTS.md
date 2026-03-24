# PyMDP Execute Submodule Agent Guide

## Purpose

Execute PyMDP scripts and simulation runs produced by render outputs, while
producing structured execution artifacts for downstream analysis.

## Canonical Public Surface

Use `src/execute/pymdp/__init__.py` as the source of truth. Exported symbols
include:

- Classes:
  - `PyMDPSimulation`
  - `PyMDPVisualizer` (soft import from analysis package)
- Execution:
  - `execute_pymdp_simulation_from_gnn(...)`
  - `execute_pymdp_simulation(...)`
- Validation:
  - `validate_pymdp_environment(...)`
  - `get_pymdp_health_status(...)`
- Package detection:
  - `detect_pymdp_installation(...)`
  - `is_correct_pymdp_package(...)`
  - `get_pymdp_installation_instructions(...)`
  - `attempt_pymdp_auto_install(...)`
  - `validate_pymdp_for_execution(...)`
- Context:
  - `create_enhanced_pymdp_context(...)`
- Utility exports listed in `__all__`

Do not document non-exported helper names as public contract.

## Execution Boundary

- Execute step owns simulation/runtime artifacts.
- Analysis step owns charting and visualization output.
- Any mention of "real-time execute visualization" is out of scope for this
  folder.

## Dependency Boundary

- Preferred package: `inferactively-pymdp`
- Import style in code/examples: `from pymdp.agent import Agent`
- If unavailable, code may use guarded fallback behavior depending on module.

## Key Internal Files

- `executor.py`
- `pymdp_simulation.py`
- `simple_simulation.py`
- `pymdp_runner.py`
- `validator.py`
- `package_detector.py`
- `context.py`

## Documentation Maintenance Rules

- Keep signatures and examples aligned with implemented functions.
- Keep claims testable by `src/tests/test_execute_pymdp_*.py`, integration tests,
  and `src/tests/test_pymdp_1_0_0_upstream_api.py` (installed `Agent` API).
- Keep wording concise; avoid capability claims not exercised in code.
