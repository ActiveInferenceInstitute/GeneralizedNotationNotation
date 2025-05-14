# Ontology Processing

This directory contains the GNN processing pipeline step for ontology-related operations, primarily managed by the `8_ontology.py` script located in the `src/` directory.

## Overview

The core purpose of this module is to enhance the **accessibility, rigor, reproducibility, and flexibility** of GNN models by leveraging explicit ontological annotations. GNN files can include an `## ActInfOntologyAnnotation` section, which provides a mapping from model-specific variable names (defined in `StateSpaceBlock` or other sections) to standardized terms from a relevant ontology (e.g., an Active Inference Ontology).

For example:
```
## ActInfOntologyAnnotation
s_t=HiddenState
A=TransitionMatrix
o_t=Observation
```
This mapping allows a model variable like `s_t` to be formally recognized as a `HiddenState`.

### Benefits of Ontological Annotation:

*   **Accessibility**: By linking potentially opaque variable names (e.g., `A`, `B`, `s1`) to well-defined ontological concepts (e.g., `RecognitionMatrix`, `TransitionMatrix`, `HiddenState`), models become easier to understand for those familiar with the ontology, even if they are new to the specific model's notation.
*   **Rigor**: Annotations can be validated against a defined set of ontological terms. This ensures that terms are used consistently and correctly according to a shared understanding, reducing ambiguity. The `8_ontology.py` script can perform such validation.
*   **Reproducibility & Comparability**: Standardizing the semantic meaning of model components allows for more accurate comparison between different GNN models. It facilitates the development of tools that can operate on models based on their ontological meaning rather than just their syntactic structure, and can aid in translating models to other formalisms.
*   **Flexibility**: GNN allows modelers to use their preferred variable names within the model definition for clarity or convenience specific to that model. The `ActInfOntologyAnnotation` section then bridges this model-specific nomenclature to a common ontological framework without forcing a rigid naming scheme on the model variables themselves.

## `8_ontology.py` Script Functionality

The `8_ontology.py` script is responsible for:
1.  **Parsing**: Reading GNN files (typically `.md` files) and extracting the key-value pairs from their `ActInfOntologyAnnotation` sections.
2.  **Validation (Optional but Recommended)**: Checking if the ontological terms used in the annotations are valid according to a predefined list or a formal ontology definition. This is typically done using a JSON file (e.g., `act_inf_ontology_terms.json` located in this directory) listing accepted terms for the relevant ontology (e.g., an "Active Inference Ontology").
3.  **Reporting**: Generating a summary report that details, for each processed GNN file:
    *   The extracted ontological mappings.
    *   The status of their validation (e.g., which terms are valid, which are unrecognized).
    *   This report is typically saved in the `output/ontology_processing/` directory.

## `mcp.py`

The `mcp.py` file in this directory defines the Model Context Protocol interface for the ontology module. It provides functions to:
- Parse ontology annotations from GNN file content.
- Load defined ontological terms (e.g., from a configuration file).
- Validate GNN annotations against these defined terms.
- Assist in generating reports.

This allows the ontology processing logic to be modular and potentially accessible to other parts of the GNN system or external tools in a standardized way.

## Usage

This step is typically invoked as part of the main GNN pipeline:

```bash
python main.py [--target-dir path/to/gnn_files] [--output-dir path/to/output] [--ontology-terms-file src/ontology/act_inf_ontology_terms.json]
```

If the `--ontology-terms-file` argument is omitted, the pipeline defaults to using `src/ontology/act_inf_ontology_terms.json`.

Or it can be run individually (e.g., from the `src/` directory):

```bash
python 8_ontology.py [--target-dir path/to/gnn_files] [--output-dir path/to/output] [--ontology-terms-file ontology/act_inf_ontology_terms.json] [options]
```

If the `--ontology-terms-file` argument is omitted when running `8_ontology.py` directly from the `src/` directory, it defaults to using `ontology/act_inf_ontology_terms.json` (relative to `src/`).

Refer to `main.py --help` or `8_ontology.py --help` for available options, including how to specify the location of GNN files and the ontology terms definition file. 