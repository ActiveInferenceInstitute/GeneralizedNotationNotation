# Execute Module

This module is responsible for running or executing GNN models that have been rendered into specific simulator formats by other parts of the GNN pipeline (e.g., by `11_render.py`).

The execute module is now organized into subfolders for each execution environment:
- `pymdp/` - PyMDP script execution
- `rxinfer/` - RxInfer.jl script execution
- `discopy/` - DisCoPy diagram validation and analysis
- `activeinference_jl/` - ActiveInference.jl script execution and comprehensive analysis

- `activeinference_jl/` - ActiveInference.jl script execution and comprehensive analysis

### Execution Workflow

```mermaid
graph TD
    Pipeline[Main Pipeline] --> Step12[12_execute.py]
    Step12 --> Discovery[Discover Scripts]
    Discovery --> List[Script List]
    
    List --> Loop{For Each Script}
    Loop --> Setup[Env Setup]
    Setup --> Run[Subprocess Exec]
    Run --> Capture[Capture Output]
    Capture --> Report[Execution Report]
    
    subgraph "Execution Environments"
    Run --> PyMDP[PyMDP Env]
    Run --> RxInfer[RxInfer Env]
    Run --> ActInf[ActiveInference.jl Env]
    Run --> JAX[JAX Env]
    Run --> DisCoPy[DisCoPy Env]
    end
```

### Multi-Framework Execution Architecture

```mermaid
flowchart LR
    subgraph "Input"
        RenderedCode[Rendered Code from Step 11]
    end
    
    subgraph "Framework Detection"
        Detect[Framework Detector]
        PyMDPDetect[PyMDP Detector]
        RxInferDetect[RxInfer Detector]
        ActInfDetect[ActiveInference.jl Detector]
        JAXDetect[JAX Detector]
        DisCoPyDetect[DisCoPy Detector]
    end
    
    subgraph "Execution"
        PyMDPExec[PyMDP Executor]
        RxInferExec[RxInfer Executor]
        ActInfExec[ActiveInference.jl Executor]
        JAXExec[JAX Executor]
        DisCoPyExec[DisCoPy Executor]
    end
    
    subgraph "Output"
        Results[Execution Results]
        Logs[Execution Logs]
        Reports[Execution Reports]
    end
    
    RenderedCode --> Detect
    Detect --> PyMDPDetect
    Detect --> RxInferDetect
    Detect --> ActInfDetect
    Detect --> JAXDetect
    Detect --> DisCoPyDetect
    
    PyMDPDetect --> PyMDPExec
    RxInferDetect --> RxInferExec
    ActInfDetect --> ActInfExec
    JAXDetect --> JAXExec
    DisCoPyDetect --> DisCoPyExec
    
    PyMDPExec --> Results
    RxInferExec --> Results
    ActInfExec --> Results
    JAXExec --> Results
    DisCoPyExec --> Results
    
    Results --> Logs
    Results --> Reports
```

### Module Integration Flow

```mermaid
flowchart LR
    subgraph "Pipeline Step 12"
        Step12[12_execute.py Orchestrator]
    end
    
    subgraph "Execute Module"
        Processor[processor.py]
        Executor[executor.py]
        Validator[validator.py]
    end
    
    subgraph "Framework Executors"
        PyMDPExec[pymdp/]
        RxInferExec[rxinfer/]
        ActInfExec[activeinference_jl/]
        JAXExec[jax/]
        DisCoPyExec[discopy/]
    end
    
    subgraph "Downstream Steps"
        Step13[Step 13: LLM]
        Step16[Step 16: Analysis]
    end
    
    Step12 --> Processor
    Processor --> Executor
    Processor --> Validator
    
    Executor --> PyMDPExec
    Executor --> RxInferExec
    Executor --> ActInfExec
    Executor --> JAXExec
    Executor --> DisCoPyExec
    
    Processor -->|Execution Results| Step13
    Processor -->|Execution Results| Step16
```
    RxInferExec --> Results
    ActInfExec --> Results
    JAXExec --> Results
    DisCoPyExec --> Results
    
    Results --> Logs
    Results --> Reports
```

### Execution Sequence Flow

```mermaid
sequenceDiagram
    participant Pipeline as Pipeline Step 12
    participant Executor as Execute Module
    participant Framework as Framework Executor
    participant Subprocess as Subprocess
    participant Monitor as Result Monitor
    
    Pipeline->>Executor: process_execute()
    Executor->>Executor: Discover rendered scripts
    Executor->>Executor: Detect framework type
    
    loop For each script
        Executor->>Framework: Select framework executor
        Framework->>Framework: Setup environment
        Framework->>Subprocess: Execute script
        Subprocess-->>Framework: Return code & output
        Framework->>Monitor: Capture results
        Monitor->>Monitor: Validate output
        Monitor-->>Framework: Execution status
        Framework-->>Executor: Execution result
    end
    
    Executor->>Executor: Aggregate results
    Executor-->>Pipeline: Execution summary
```

## Core Components

### `pymdp_runner.py`

This script is the primary runner for PyMDP-based simulations. Its main responsibilities include:

-   **Discovering Scripts**: It searches for Python scripts (typically `*_pymdp.py`) within `output/11_render_output/`.
-   **Environment Setup**: It attempts to use the Python interpreter from the project's virtual environment (`src/.venv/bin/python`) to ensure that the necessary dependencies (like `inferactively-pymdp`) are available.
-   **Execution**: It executes each discovered PyMDP script as a separate subprocess.
-   **Output Handling**: It captures `stdout` and `stderr` from the executed scripts and logs success or failure for each.
-   **Reporting**: It provides a summary of which scripts were run and their execution status.

### `12_execute.py` (Pipeline Step)

This script integrates the execution capabilities (currently `pymdp_runner.py`) into the main GNN processing pipeline (`main.py`) as "Step 12".
-   It receives pipeline arguments (like output directory, verbosity) from `main.py`.
-   It invokes `pymdp_runner.run_pymdp_scripts` to carry out the execution of rendered PyMDP simulations.
-   It reports the overall success or failure of this execution step back to the main pipeline.

## Usage

The `12_execute.py` script is typically invoked automatically by `main.py` if it's present in the `src/` directory and not skipped.

The `pymdp_runner.py` can also be run standalone for testing purposes, provided that the PyMDP scripts have already been generated in the expected output location (e.g., after running up to Step 11 of the pipeline).

## Future Extensions

-   Support for executing other types of rendered simulators (e.g., RxInfer.jl scripts).
-   More sophisticated control over execution parameters passed to individual simulations. 