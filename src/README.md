# GNN Processing Pipeline (`src/`)

This directory contains the source code for the Generalized Notation Notation (GNN) processing pipeline. It provides a systematic, modular, and extensible way to process, analyze, validate, and visualize GNN files and related project artifacts.

## Pipeline Orchestration (`main.py`)

The entire pipeline is orchestrated by `main.py`, located in this `src/` directory.

**How it Works:**
1.  **Dynamic Script Discovery:** `main.py` automatically discovers all executable pipeline step scripts within the `src/` directory that follow the naming convention `[number]_*.py` (e.g., `1_gnn.py`, `2_setup.py`).
2.  **Sequential Execution:** Scripts are executed in numerical order based on their prefix.
3.  **Argument Propagation:** `main.py` parses command-line arguments (see Options below) and passes them to each individual pipeline script. Each script is responsible for utilizing the arguments relevant to its operation.
4.  **Safe-to-Fail Mechanism:**
    *   Most pipeline steps are designed to be "safe-to-fail." If a non-critical step encounters an error and returns a non-zero exit code, `main.py` will log the failure and continue to the next step.
    *   **Critical Steps:** Certain steps, like `2_setup.py`, are deemed critical. If a critical step fails, the pipeline will halt immediately to prevent further errors due to an improper environment or setup.
    *   **Summary Report:** At the end of its run, `main.py` provides a summary indicating which steps (if any) failed and whether the overall pipeline run is considered a success or failure.

## Pipeline Workflow Diagram

```mermaid
graph TD
    A[Start Pipeline: main.py] --> B(1_gnn.py: GNN Processing);
    B --> C(2_setup.py: Project Setup);
    C -- Critical Step --> D(3_tests.py: Run Tests);
    D --> E(4_gnn_type_checker.py: Type Check GNN);
    E --> F(5_export.py: Export GNNs & Generate Summary);
    F --> G(6_visualization.py: Generate Visualizations);
    G --> H(7_mcp.py: MCP Integration Checks);
    H --> I(8_ontology.py: Ontology Operations);
    I --> J(9_render.py: Render GNN Simulators);
    J --> K[End Pipeline];

    F --> J; // 5_export.py output is input for 9_render.py

    subgraph Modules Called by Pipeline Steps
        E --> E1[gnn_type_checker/cli.py];
        G --> G1[visualization/cli.py];
    end

    style C fill:#f99,stroke:#333,stroke-width:2px;
```

## Pipeline Steps & Corresponding Modules

Below is a detailed description of each pipeline step script (located in `src/`) and its corresponding primary module/folder (also in `src/`).

---

### 1. `1_gnn.py` - GNN Processing
-   **Folder:** `src/gnn/`
-   **What:** Performs initial GNN-specific operations. This includes discovering GNN Markdown (`.md`) files and performing basic parsing to identify key structural elements (like ModelName, StateSpaceBlock, Connections) based on `src/gnn/gnn_file_structure.md` and `src/gnn/gnn_punctuation.md`.
-   **Why:** To get a preliminary understanding of the GNN files being processed and to generate a basic report on their structure. This step can help catch very high-level errors or provide statistics before more intensive processing.
-   **How:**
    -   Scans the `args.target_dir` for `.md` files (recursively if `args.recursive` is set).
    -   For each file, it attempts to parse predefined sections.
    -   Generates a report (`<output_dir>/gnn_processing_step/1_gnn_processing_report.md`) summarizing findings per file.
-   **Output:** A markdown report detailing parsed sections for each GNN file.

---

### 2. `2_setup.py` - Project Setup
-   **Folder:** `src/setup/`
-   **What:** Handles critical initial setup tasks for the project environment. This includes verifying and creating necessary output directories and, importantly, setting up a Python virtual environment (`.venv/` in `src/`) and installing dependencies from `src/requirements.txt`.
-   **Why:** To ensure a consistent and correctly configured environment for the subsequent pipeline steps, preventing issues due to missing dependencies or directories. This step is **critical**; its failure halts the pipeline.
-   **How:**
    -   Calls `verify_directories()` to create standard output subfolders (e.g., for visualizations, type checking) within `args.output_dir`.
    -   Invokes `perform_full_setup()` from `src/setup/setup.py`. This function:
        -   Checks for and creates a virtual environment at `src/.venv/` if one doesn't exist.
        -   Installs/updates dependencies listed in `src/requirements.txt` using `pip` within the virtual environment.
-   **Output:** Created directories, a configured virtual environment.

---

### 3. `3_tests.py` - Run Tests
-   **Folder:** `src/tests/`
-   **What:** Executes automated tests for the project, primarily using the `pytest` framework.
-   **Why:** To verify the correctness and reliability of the codebase, including GNN parsing, type checking logic, and other utilities.
-   **How:**
    -   Invokes `pytest` as a subprocess (`python -m pytest src/tests/`).
    -   The tests are run from the `src/` directory context.
    -   Captures `pytest` output (stdout, stderr) and reports success or failure based on `pytest`'s exit code.
-   **Output:** Test results printed to the console. If `pytest` is configured to produce report files (e.g., HTML, XML), those would be generated in their specified locations (typically managed by `pytest` configuration).

---

### 4. `4_gnn_type_checker.py` - GNN Type Checking
-   **Folder:** `src/gnn_type_checker/`
-   **What:** Performs comprehensive type checking and structural validation of GNN files. It can also estimate computational resources.
-   **Why:** To ensure GNN models adhere to the GNN specification, are internally consistent, and use valid types and connections. This helps maintain model quality and interpretability.
-   **How:**
    -   Imports and calls the `main()` function from `src/gnn_type_checker/cli.py`.
    -   Passes arguments like the target directory/file, output directory (`<pipeline_output_dir>/gnn_type_check/`), and options (`--recursive`, `--strict`, `--estimate-resources`).
    -   The `gnn_type_checker.cli.main()` function then uses `GNNTypeChecker` and `GNNResourceEstimator` classes from `src/gnn_type_checker/` to perform the analysis.
-   **Output:**
    -   Markdown report (`type_check_report.md`) detailing validation results per file.
    -   JSON data file (`resources/type_check_data.json`) with structured validation data.
    -   HTML report for richer viewing.
    -   If `--estimate-resources` is used, additional reports (Markdown, JSON, HTML) for resource estimation are generated in a subfolder.
    -   All outputs are placed within `<pipeline_output_dir>/gnn_type_check/`.

---

### 5. `5_export.py` - Export GNNs & Generate Summary Report
-   **Folder:** `src/export/`
-   **What:** This step has two primary functions:
    1.  **Export GNN Models:** Parses GNN files (typically `.md` source files from `args.target_dir`) and exports them into various intermediate formats (e.g., JSON, XML, GEXF, GraphML, Python Pickle). These exported models are typically saved in a structured way within `<output_dir>/gnn_exports/`. This output is crucial for `9_render.py`.
    2.  **Generate Summary Report:** Generates a high-level summary report (`gnn_processing_summary.md`) of the entire pipeline's run up to this point.
-   **Why:**
    1.  To convert GNN models into standardized formats that can be consumed by other tools or subsequent pipeline steps (like rendering).
    2.  To provide a single document that gives an overview of what was processed and the status of key pipeline stages.
-   **How:**
    -   For GNN model export:
        -   Scans `args.target_dir` for GNN files.
        -   Uses functions from `src/export/format_exporters.py` to convert each GNN model into multiple formats.
        -   Saves these exported files into `<output_dir>/gnn_exports/<model_name>/<model_name>.<format_extension>`.
    -   For summary report generation:
        -   Checks for the existence of output artifacts from previous steps.
        -   Compiles this information into `<output_dir>/gnn_processing_summary.md`.
-   **Output:**
    -   Exported GNN models in various formats within `<output_dir>/gnn_exports/`.
    -   `gnn_processing_summary.md` in `args.output_dir`.

---

### 6. `6_visualization.py` - Generate Visualizations
-   **Folder:** `src/visualization/`
-   **What:** Generates visual representations (e.g., graphs, diagrams) of GNN models.
-   **Why:** To help users understand the structure, connections, and dependencies within their GNN models, aiding in debugging, analysis, and communication.
-   **How:**
    -   Imports and calls the `main()` function from `src/visualization/cli.py`.
    -   Passes arguments like the target directory/file and the output directory (`<pipeline_output_dir>/gnn_examples_visualization/`).
    -   The `visualization.cli.main()` function uses `GNNVisualizer` and other components from `src/visualization/` to parse GNN files and render various visual outputs (e.g., using Graphviz).
-   **Output:** Image files (e.g., PNG, SVG) and potentially HTML files for each processed GNN model, saved in `<pipeline_output_dir>/gnn_examples_visualization/`.

---

### 7. `7_mcp.py` - MCP Integration Checks
-   **Folder:** `src/mcp/` (and scans other modules)
-   **What:** Performs checks related to the project's Model Context Protocol (MCP) integration. It verifies that core MCP files exist and that functional modules (like `export`, `visualization`, etc.) have their necessary `mcp.py` integration files.
-   **Why:** To ensure the project's MCP framework is correctly set up and that all intended modules are ready to expose their functionalities via MCP to external tools (like LLMs).
-   **How:**
    -   Scans `src/mcp/` for essential files (`mcp.py`, `meta_mcp.py`, `cli.py`, etc.).
    -   Scans other primary `src/` subdirectories for the presence of an `mcp.py` file.
    -   Generates a report (`<output_dir>/mcp_processing_step/7_mcp_integration_report.md`) detailing the findings.
-   **Output:** A markdown report on the status of MCP file integrations.

---

### 8. `8_ontology.py` - Ontology Operations
-   **Folder:** `src/ontology/`
-   **What:** Handles ontology-specific operations for GNN files. This includes parsing ontology annotations from GNN files, validating these annotations against a defined set of ontological terms (if a terms file is provided), and generating a report.
-   **Why:** To link variables and components within GNN models to formal ontological terms, enhancing semantic clarity, interoperability, and enabling more advanced model analysis and comparison.
-   **How:**
    -   Uses utility functions from `src/ontology/mcp.py` (note: this `mcp.py` contains helper functions, not full MCP server tools).
    -   Processes GNN files from `args.target_dir`.
    -   Extracts annotations from the `ActInfOntologyAnnotation` section.
    -   If an `--ontology-terms-file` (e.g., a JSON file defining valid terms) is provided, it validates the extracted annotations against these terms.
    -   Generates a consolidated markdown report (`<output_dir>/ontology_processing/ontology_processing_report.md`) summarizing the findings for all processed files.
-   **Output:** A markdown report detailing parsed and validated ontological annotations for each GNN file.

---

### 9. `9_render.py` - Render GNN Simulators
-   **Folder:** `src/render/`
-   **What:** Renders GNN specifications (typically the JSON files exported by `5_export.py` found in `<output_dir>/gnn_exports/`) into executable simulator code or configurations for specific modeling frameworks (e.g., `pymdp`, `rxinfer`).
-   **Why:** To translate abstract GNN models into concrete, runnable simulations or models that can be used for analysis, inference, or further development within supported target frameworks.
-   **How:**
    -   Imports and calls the `main()` function from `src/render/render.py`.
    -   Scans `<output_dir>/gnn_exports/` for GNN specification files (e.g., `.json`).
    -   For each specification and for each supported target format (e.g., "pymdp", "rxinfer"):
        -   Invokes the rendering logic in `src/render/render.py`.
        -   Saves the generated simulator code/configuration into `<output_dir>/gnn_rendered_simulators/<target_format>/...`.
-   **Output:** Generated simulator files (e.g., Python scripts for `pymdp`) in `<output_dir>/gnn_rendered_simulators/`.

---

## Core Utility Modules

Beyond the pipeline step scripts, several folders in `src/` contain core logic and utilities used by these steps:

-   **`src/gnn/`**: Defines GNN file structure, punctuation, and may contain example GNN files.
-   **`src/gnn_type_checker/`**: Contains the `GNNTypeChecker` and `GNNResourceEstimator` classes and their CLI, forming the backbone of the type checking and resource estimation capabilities.
-   **`src/visualization/`**: Contains the `GNNParser`, `GNNVisualizer`, and related components (e.g., `MatrixVisualizer`, `OntologyVisualizer`) along with its CLI, responsible for parsing GNN files and generating various visual outputs.
-   **`src/mcp/`**: Implements the Model Context Protocol server (`server_stdio.py`, `server_http.py`), client (`cli.py`), core MCP class (`mcp.py`), and meta-module (`meta_mcp.py`). Individual modules like `export`, `visualization`, etc., are expected to have their own `mcp.py` to register tools with this MCP framework.
-   **`src/ontology/`**: Provides tools for ontology processing, including parsing annotations from GNN files and validating them. Its `mcp.py` file contains helper functions for these tasks.
-   **`src/export/`**: Contains logic for formatting and exporting data. Currently, `5_export.py` focuses on generating the main pipeline summary report. More specific exporters (e.g., `format_exporters.py`, `report_formatters.py`) exist for potential future use or direct invocation by MCP tools.
-   **`src/setup/`**: Contains the detailed Python environment setup script (`setup.py`) used by `2_setup.py`.
-   **`src/tests/`**: Contains test files (e.g., `test_gnn_type_checker.py`) used by `3_tests.py`.

## Usage

### Running the Full Pipeline

From the `src/` directory:
```bash
python main.py [options]
```

### Running Individual Steps

Each step script can also be run individually (e.g., for debugging or specific tasks). From the `src/` directory:
```bash
python 1_gnn.py [options]
python 2_setup.py [options]
# ... and so on for other steps.
```
When running scripts individually, ensure that any prerequisite steps (especially `2_setup.py`) have been successfully completed or that the environment is otherwise correctly configured. Also, relative paths for default arguments (like `../output` or `gnn/examples`) are typically interpreted from the `src/` directory as the current working directory.

## Options (for `main.py`)

-   `--target-dir DIR`: Target directory for GNN files (default: `gnn/examples`). Some scripts might have more specific targets (e.g., `7_mcp.py` focuses on `src/mcp/` and related module integrations).
-   `--output-dir DIR`: Root directory to save all outputs (default: `../output`, which resolves to `<project_root>/output/` if `main.py` is run from `src/`). Individual steps will create subdirectories within this root output directory.
-   `--recursive`: Recursively process directories (passed to relevant steps like GNN processing, type checking, visualization, ontology).
-   `--skip-steps LIST`: Comma-separated list of step scripts or numbers to skip (e.g., `"1_gnn,7_mcp"` or `"1,7"`).
-   `--only-steps LIST`: Comma-separated list of step scripts or numbers to run exclusively (e.g., `"4_gnn_type_checker,6_visualization"`). This overrides `--skip-steps`.
-   `--verbose`: Enable verbose output for `main.py` and potentially for the individual pipeline steps if they support it.
-   `--strict`: Enable strict type checking mode (passed to `4_gnn_type_checker.py`).
-   `--estimate-resources`: Estimate computational resources (passed to `4_gnn_type_checker.py`).
-   `--ontology-terms-file FILE`: Path to a JSON file defining valid ontological terms (passed to `8_ontology.py`).

## Deprecated Scripts

-   `gnn_type_checker/bin/check_gnn.py`: Functionality is now integrated into the `src/gnn_type_checker/` module and invoked via `4_gnn_type_checker.py`.

---
This README provides a comprehensive guide to the GNN processing pipeline. For details on specific GNN syntax or the Model Context Protocol, refer to the documentation within the respective module directories (e.g., `src/gnn/`, `src/mcp/`). 