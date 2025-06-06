# GeneralizedNotationNotation (GNN)

Generalized Notation Notation (GNN) is a text-based language designed to standardize the representation and communication of Active Inference generative models. It aims to enhance clarity, reproducibility, and interoperability in the field of Active Inference and cognitive modeling. 

Intial publication:
Smékal, J., & Friedman, D. A. (2023). Generalized Notation Notation for Active Inference Models. Active Inference Journal. https://doi.org/10.5281/zenodo.7803328
https://zenodo.org/records/7803328 

## Overview

GNN provides a structured and standardized way to describe complex cognitive models. It is designed to be:

- **Human-readable**: Easy to understand and use for researchers from diverse backgrounds.
- **Machine-parsable**: Can be processed by software tools for analysis, visualization, and code generation.
- **Interoperable**: Facilitates the exchange and reuse of models across different platforms and research groups.
- **Reproducible**: Enables precise replication of model specifications.

GNN addresses the challenge of communicating Active Inference models, which are often described using a mix of natural language, mathematical equations, diagrams, and code. By offering a unified notation, GNN aims to streamline collaboration, improve model understanding, and accelerate research.

## Motivation and Goals

The primary motivation behind GNN is to overcome the limitations arising from the lack of a standardized notation for Active Inference models. This fragmentation can lead to difficulties in:

- **Effective Communication**: Making complex models hard to explain and understand.
- **Reproducibility**: Hindering the ability to replicate research findings.
- **Consistent Implementation**: Leading to variations when translating models into code.
- **Systematic Comparison**: Making it challenging to compare different models.

The goals of GNN are to:

- Facilitate clear communication and understanding of Active Inference models.
- Promote collaboration among researchers.
- Enable the development of tools for model validation, visualization, and automated code generation.
- Support the creation of a shared repository of Active Inference models.
- Bridge the gap between theoretical concepts and practical implementations.

## Key Features

### The Triple Play Approach

GNN supports three complementary modalities for model representation, known as the "Triple Play":

1.  **Text-Based Models**: GNN files are plain text and can be rendered into mathematical notation, pseudocode, or natural language descriptions. This forms the core representation.
2.  **Graphical Models**: The structure defined in GNN (variables and their connections) can be visualized as graphical models (e.g., factor graphs), clarifying dependencies and model architecture.
3.  **Executable Cognitive Models**: GNN specifications can serve as a high-level blueprint or pseudocode for implementing executable simulations in various programming environments. This ensures consistency and aids in the translation from theory to practice.

### Structured File Format

GNN defines a specific file structure, typically using Markdown, to organize model components. This includes sections for:
- Model metadata (name, version, annotations)
- State space (variable definitions)
- Connections (relationships between variables)
- Initial parameterization
- Equations
- Time settings (for dynamic models)
- Mapping to Active Inference Ontology terms

## Project Structure

This project is organized into several key directories. The primary ones are `src/` for all source code and `doc/` for all documentation.

### `src/` Directory Structure

The `src/` directory contains all the Python scripts and modules that constitute the GNN processing pipeline and related tools.

```
src/
├── .pytest_cache/
├── .venv/
├── export/
│   └── (contents related to exporting GNN models)
├── gnn/
│   └── examples/
│       └── archive/
│           └── (archived GNN examples)
├── gnn_type_checker/
│   └── (contents for the GNN type checker)
├── mcp/
│   └── (contents for Model Context Protocol)
├── ontology/
│   └── (contents related to ontology processing)
├── render/
│   └── (contents related to rendering GNN models)
├── setup/
│   └── (contents for project setup and configuration)
├── tests/
│   └── (test scripts)
├── visualization/
│   └── bin/
│       └── (binaries or scripts for visualization)
├── __pycache__/
├── 1_gnn.py
├── 2_setup.py
├── 3_tests.py
├── 4_gnn_type_checker.py
├── 5_export.py
├── 6_visualization.py
├── 7_mcp.py
├── 8_ontology.py
├── 9_render.py
├── main.py
├── README.md
└── requirements.txt
```

### `doc/` Directory Structure

The `doc/` directory contains all supplementary documentation, including conceptual explanations, syntax guides, and examples.

```
doc/
├── cerebrum/
│   └── (contents related to the Cerebrum project integration if any)
├── about_gnn.md
├── gnn_examples_doc.md
├── gnn_file_structure_doc.md
├── gnn_implementation.md
├── gnn_llm_neurosymbolic_active_inference.md
├── gnn_overview.md
├── gnn_paper.md
├── gnn_syntax.md
└── gnn_tools.md
```

## Processing Pipeline Orchestration

The primary way to process GNN files and run various associated tasks (like type checking, visualization, etc.) is through the main pipeline script `src/main.py`. This script discovers and executes a series of numbered Python scripts located in the `src/` directory, each corresponding to a specific stage of the GNN processing workflow.

**Pipeline Stages (Dynamically Discovered and Ordered):**

The `main.py` script will automatically find and run scripts in `src/` that follow the pattern `[number]_*.py`. Based on the current project structure, these include:

*   `1_gnn.py`: Core GNN file processing (details specific to `gnn/` folder tasks).
*   `2_setup.py`: Project setup and configuration tasks (details specific to `setup/` folder tasks). This is a critical step; if it fails, the pipeline halts.
*   `3_tests.py`: Execution of tests (details specific to `tests/` folder tasks).
*   `4_gnn_type_checker.py`: Performs type checking and optionally, resource estimation for GNN files (details specific to `gnn_type_checker/` folder tasks).
*   `5_export.py`: Handles exporting GNN models or related data (details specific to `export/` folder tasks).
*   `6_visualization.py`: Generates visualizations from GNN files (details specific to `visualization/` folder tasks).
*   `7_mcp.py`: Meta-Circular Processing tasks (details specific to `mcp/` folder tasks).
*   `8_ontology.py`: Ontology-related processing (details specific to `ontology/` folder tasks).
*   `9_render.py`: Handles rendering GNN files into different formats (details specific to `render/` folder tasks).

**Running the Pipeline:**

To execute the pipeline, navigate to the project's root directory and run:

```bash
python src/main.py [options]
```

**Key `main.py` Options:**

*   `--target-dir DIR`: Specifies the target directory for GNN files (default: `gnn/examples`). Note that individual scripts might have their own specific target folders (e.g., `src/mcp`).
*   `--output-dir DIR`: Sets the directory where all outputs will be saved (default: `output/`).
*   `--recursive`: Enables recursive processing of directories, passed to relevant pipeline steps.
*   `--skip-steps LIST`: A comma-separated list of step numbers or script names to exclude from the run (e.g., "1,7" or "1_gnn,7_mcp").
*   `--only-steps LIST`: A comma-separated list of step numbers or script names to exclusively run (e.g., "4,6" or "4_gnn_type_checker,6_visualization").
*   `--verbose`: Enables detailed logging output for the pipeline and its steps.
*   `--strict`: Enables strict type checking mode (specifically for the `4_gnn_type_checker` step).
*   `--estimate-resources`: Activates computational resource estimation (specifically for the `4_gnn_type_checker` step).
*   `--ontology-terms-file FILE`: Specifies the path to an ontology terms file (e.g., `ontology/terms.json`), used by the `8_ontology.py` step.
*   `--type-checker-report-name NAME`: Specifies the filename for the main Markdown report generated by the type checker (step 4). Defaults to `type_check_report.md`.

Refer to the `src/main.py --help` command or the script's docstring for a full list of options and their descriptions.
More details on the source code can be found in the [`src/` Directory Structure](#src-directory-structure) section.

## Tools and Utilities

The GNN ecosystem includes several tools to aid in model development, validation, and understanding. These tools are now primarily invoked through the `src/main.py` pipeline script.

### Type Checker and Resource Estimator

The GNN Type Checker (part of the `4_gnn_type_checker.py` step in the main pipeline) helps validate GNN files and can also estimate computational resources.

**Using via `main.py`:**

To run the type checker and resource estimator using the main pipeline, navigate to the project's root directory.

*   **To run only the type checker:**
    ```bash
    python src/main.py --only-steps 4_gnn_type_checker --target-dir path/to/your/gnn_files
    ```
    (Or use step number: `python src/main.py --only-steps 4 ...`)

*   **To include resource estimation with the type checker:**
    ```bash
    python src/main.py --only-steps 4_gnn_type_checker --estimate-resources --target-dir path/to/your/gnn_files
    ```

*   **To run the full pipeline**
    ```bash
    python src/main.py --target-dir path/to/your/gnn_files
    ```

**Options (passed through `main.py`):**

*   `--target-dir <input_path>`: (Required for the step, can be set globally for `main.py`) Path to the GNN file (`.md`) or a directory containing GNN files.
*   `--output-dir <dir_path>`: (Set globally for `main.py`) Specifies the base directory where all output files will be saved. The type checker step will typically save its specific outputs in a subdirectory like `output/gnn_type_check/`.
*   `--report-file <file_name_or_path>`: (Handled internally by the `4_gnn_type_checker.py` step. The main report is typically `type_check_report.md` within the step's output directory.)
*   `--recursive`: (Set globally for `main.py`) Enables recursive processing.
*   `--strict`: (Set globally for `main.py`) Enables strict type checking mode.
*   `--estimate-resources`: (Set globally for `main.py`) Activates the computational resource estimator.

**Examples (run from the project root directory):**

1.  **Check a single GNN file using the pipeline and save reports to the default output directory (`output/`):**
    ```bash
    python src/main.py --only-steps 4 --target-dir doc/gnn_examples_doc.md 
    ```
    *(Note: Adjust path to an actual GNN example file. The output for this step will be within the global output directory, e.g., `output/gnn_type_check/`)*

2.  **Check all GNN files in a directory recursively via the pipeline and save reports to a custom global output directory:**
    ```bash
    python src/main.py --only-steps 4 --target-dir src/gnn/examples --recursive -o custom_output/my_check_results
    ```

3.  **Check files, estimate resources via the pipeline, and save all outputs to a specific global directory:**
    ```bash
    python src/main.py --only-steps 4 --target-dir src/gnn/examples --recursive --estimate-resources -o custom_output/full_analysis
    ```

**Output Structure (when `4_gnn_type_checker` step runs):**

When the `4_gnn_type_checker.py` step is executed via `main.py` with `--output-dir <path>`, it will create its specific output structure within a subdirectory of `<path>` (e.g., `<path>/gnn_type_check/`). This typically includes:
*   `type_check_report.md`: Main Markdown report from the type checker.
*   `html_vis/type_checker_visualization_report.html`: HTML report with visualizations for type checking.
*   `resources/type_check_data.json`: JSON data generated by the type checker.

Features of the type checker include:
- Validation of required sections and structure
- Type checking of variables and dimensions
- Verification of connections and references
- Detailed error reports with suggestions for fixes

### Visualization

GNN files can be visualized to create graphical representations of models. This functionality is handled by the `6_visualization.py` step in the main pipeline.

**Using via `main.py`:**

```bash
# From the project root directory
python src/main.py --only-steps 6_visualization --target-dir path/to/gnn_file.md
```
(Or use step number: `python src/main.py --only-steps 6 ...`)

The `6_visualization.py` script will process the specified GNN file (or files in the target directory if it supports directory processing) and save the visualizations in the designated output directory (e.g., `output/6_visualization_outputs/`).

## Getting Started

-   **Learn the Syntax**: Familiarize yourself with GNN syntax and structure. See `doc/gnn_syntax.md` and `doc/gnn_file_structure_doc.md`.
-   **Explore Examples**: Check out example GNN files in `doc/gnn_examples_doc.md` and `src/gnn/examples/`.
-   **Read the Paper**: For a detailed introduction, refer to "Generalized Notation Notation for Active Inference Models" (see [`doc/gnn_paper.md`](./doc/gnn_paper.md) for details and links).
-   **GitHub Repository**: [https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation)

## Documentation

Comprehensive documentation can be found in the `doc/` directory. See the [`doc/` Directory Structure](#doc-directory-structure) section above for a detailed file listing. Key documents include:

-   [`doc/about_gnn.md`](./doc/about_gnn.md): General information about GNN.
-   [`doc/gnn_overview.md`](./doc/gnn_overview.md): A high-level overview of GNN.
-   [`doc/gnn_syntax.md`](./doc/gnn_syntax.md): Detailed specification of GNN syntax and punctuation.
-   [`doc/gnn_file_structure_doc.md`](./doc/gnn_file_structure_doc.md): Description of the GNN file organization.
-   [`doc/gnn_examples_doc.md`](./doc/gnn_examples_doc.md): Examples and use cases.
-   [`doc/gnn_implementation.md`](./doc/gnn_implementation.md): Guidelines for implementing GNN.
-   [`doc/gnn_tools.md`](./doc/gnn_tools.md): Information on tools and resources.
-   [`doc/gnn_paper.md`](./doc/gnn_paper.md): Details about the GNN academic paper.
-   [`doc/gnn_llm_neurosymbolic_active_inference.md`](./doc/gnn_llm_neurosymbolic_active_inference.md): Information on LLM and Neurosymbolic Active Inference.

## Contributing

GNN is an evolving standard. Contributions are welcome! Please refer to the guidelines on the [GitHub repository](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation) or open an issue to discuss potential changes or improvements.

## License

This project is licensed under the [MIT License](./LICENSE.md). Please see the `LICENSE.md` file for full details.

## Installation and Setup

For comprehensive setup instructions, dependency information, and troubleshooting, please refer to [SETUP.md](doc/SETUP.md).

Quick start:

```bash
# Clone the repository
git clone https://github.com/yourusername/GeneralizedNotationNotation.git
cd GeneralizedNotationNotation

# Run the setup script
cd src
python3 main.py --only-steps 2_setup --dev
```

---

*This README was generated based on documents in the `doc/` folder.*