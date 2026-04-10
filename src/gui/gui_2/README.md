# GUI 2: Visual Matrix Editor

## Overview
This module represents the second major GUI implementation for the repository: an advanced Visual Matrix Editor dedicated to matrix-level parameter adjustments using intuitive, visual dashboards rather than text forms.

## Key Features
- **Visual Matrix Representation**: High-performance rendering of Active Inference generative matrices via Plotly.
- **Drag-and-Drop State Modification**: Real-time manipulation of the state space tree directly inside the UI.
- **POMDP Bootstrapping**: Features logic (via `get_pomdp_template`) to launch instantly with default Dirichlet matrices populated correctly.

## Architecture Structure
- **`matrix_editor.py`**: Custom grid logic parsing dimensions from standard tensors natively into Plotly maps.
- **`processor.py`**: Execution and endpoint orchestrator.
- **`__init__.py`**: Facade wrapping and integration parameters for the `gui` super-module.

## Usage
GUI 2 is executed via the standard orchestrator flow:
```bash
python src/22_gui.py --gui-types gui_2
```
Access the exposed local port directly in your browser or through standard automated testing suites mapping HTTP actions.
