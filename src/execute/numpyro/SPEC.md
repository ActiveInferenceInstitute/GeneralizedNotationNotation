# NumPyro Execute Backend Specification

## Overview

The NumPyro execute backend runs NumPyro-generated Python scripts and persists execution logs and outputs.

## Public API

The module must export:

- `is_numpyro_available() -> bool`
- `find_numpyro_scripts(base_dir: Union[str, Path], recursive: bool = True) -> List[Path]`
- `execute_numpyro_script(script_path: Path, verbose: bool = False, output_dir: Optional[Path] = None, timeout: int = 300) -> bool`
- `run_numpyro_scripts(rendered_simulators_dir: Union[str, Path], execution_output_dir: Optional[Union[str, Path]] = None, recursive_search: bool = True, verbose: bool = False) -> bool`

## Script discovery

`run_numpyro_scripts` searches under:

- `<rendered_simulators_dir>/numpyro/`

and matches either folder name `numpyro` or file name containing `numpyro`.

## Environment contract

When `output_dir` is provided, the runner sets:

- `NUMPYRO_OUTPUT_DIR=<output_dir>`

so that generated scripts write `simulation_results.json` into the execution tree.

