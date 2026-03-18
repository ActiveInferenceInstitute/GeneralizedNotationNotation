# PyTorch Execute Backend - Agent Scaffolding

## Module Overview

**Purpose**: Discover and execute PyTorch-rendered simulation scripts produced by Step 11.

**Parent**: `src/execute/` (Step 12: Execute)

---

## Public API

From `src/execute/pytorch/__init__.py`:

- `is_pytorch_available() -> bool`
- `find_pytorch_scripts(base_dir: Union[str, Path], recursive: bool = True) -> List[Path]`
- `execute_pytorch_script(script_path: Path, verbose: bool = False, device: Optional[str] = None, output_dir: Optional[Path] = None, timeout: int = 300) -> bool`
- `run_pytorch_scripts(rendered_simulators_dir: Union[str, Path], execution_output_dir: Optional[Union[str, Path]] = None, recursive_search: bool = True, verbose: bool = False, device: Optional[str] = None) -> bool`

---

## Conventions

- Looks for scripts under `<rendered_simulators_dir>/pytorch/` and for filenames containing `pytorch`.
- Persists `stdout.txt`, `stderr.txt`, and `execution_log.json` under `output_dir` when provided.
- Uses `PYTORCH_OUTPUT_DIR` env var to tell generated scripts where to write `simulation_results.json`.

