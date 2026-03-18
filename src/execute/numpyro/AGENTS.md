# NumPyro Execute Backend - Agent Scaffolding

## Module Overview

**Purpose**: Discover and execute NumPyro-rendered simulation scripts produced by Step 11.

**Parent**: `src/execute/` (Step 12: Execute)

---

## Public API

From `src/execute/numpyro/__init__.py`:

- `is_numpyro_available() -> bool`
- `find_numpyro_scripts(base_dir: Union[str, Path], recursive: bool = True) -> List[Path]`
- `execute_numpyro_script(script_path: Path, verbose: bool = False, output_dir: Optional[Path] = None, timeout: int = 300) -> bool`
- `run_numpyro_scripts(rendered_simulators_dir: Union[str, Path], execution_output_dir: Optional[Union[str, Path]] = None, recursive_search: bool = True, verbose: bool = False) -> bool`

---

## Conventions

- Looks for scripts under `<rendered_simulators_dir>/numpyro/` and for filenames containing `numpyro`.
- Persists `stdout.txt`, `stderr.txt`, and `execution_log.json` under `output_dir` when provided.
- Uses `NUMPYRO_OUTPUT_DIR` env var to tell generated scripts where to write `simulation_results.json`.

---

## Testing guidance

Useful tests:

- dependency check behavior when `numpyro` is missing (skip with clear logs)
- syntax validation failure path
- execution path for a small generated NumPyro script (optional dependency / mark as optional)

