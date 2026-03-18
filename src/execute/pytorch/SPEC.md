# PyTorch Execute Backend Specification

## Overview

The PyTorch execute backend runs PyTorch-generated Python scripts and persists execution logs and outputs.

## Public API

The module must export:

- `is_pytorch_available() -> bool`
- `find_pytorch_scripts(base_dir: Union[str, Path], recursive: bool = True) -> List[Path]`
- `execute_pytorch_script(script_path: Path, verbose: bool = False, device: Optional[str] = None, output_dir: Optional[Path] = None, timeout: int = 300) -> bool`
- `run_pytorch_scripts(rendered_simulators_dir: Union[str, Path], execution_output_dir: Optional[Union[str, Path]] = None, recursive_search: bool = True, verbose: bool = False, device: Optional[str] = None) -> bool`

## Script discovery

`run_pytorch_scripts` searches under:

- `<rendered_simulators_dir>/pytorch/`

and matches either folder name `pytorch` or file name containing `pytorch`.

## Environment contract

When `output_dir` is provided, the runner sets:

- `PYTORCH_OUTPUT_DIR=<output_dir>`

so that generated scripts write `simulation_results.json` into the execution tree.

