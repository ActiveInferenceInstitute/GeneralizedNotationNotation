# GNN Ontology System Documentation

**Version**: v1.1.0  
**Last Updated**: February 9, 2026  
**Status**: ✅ Production Ready  
**Test Count**: 1,127 Tests Passing  

This document describes the ontology system used in conjunction with GNN files, focusing on the Active Inference Ontology.

For implementation details and usage, see:

- **[src/ontology/AGENTS.md](../../src/ontology/AGENTS.md)**: Ontology processing module documentation
- **[src/AGENTS.md](../../src/AGENTS.md)**: Complete pipeline module registry
- **[src/main.py](../../src/main.py)**: Pipeline orchestrator (Step 10 handles ontology processing)

## 1. Ontology Terms Definition File

The core of the ontology system is a JSON file that defines recognized ontological terms.

- **Location:** `src/ontology/act_inf_ontology_terms.json`
- **Structure:** The file is a JSON object where each key is an ontological term (e.g., `HiddenState`, `TransitionMatrix`). The value for each term is another JSON object containing:
  - `description`: A human-readable explanation of the term.
  - `uri`: A Uniform Resource Identifier, often linking to a formal ontology definition (e.g., an OBO Foundry URI like `obo:ACTO_000001`).

**Example Entry from `act_inf_ontology_terms.json`:**

```json
{
    "HiddenState": {
        "description": "A state of the environment or agent that is not directly observable.",
        "uri": "obo:ACTO_000001" 
    },
    // ... other terms
}
```

## 2. Usage in GNN Files

Ontological terms are associated with variables defined in a GNN model using a dedicated section.

- **Section Header:** `## ActInfOntologyAnnotation`
- **Syntax:** Within this section, each line maps a variable name from the GNN model's `StateSpaceBlock` to an ontological term defined in `act_inf_ontology_terms.json`.
  - Format: `VariableName=OntologyTerm`
  - One mapping per line.

**Example `ActInfOntologyAnnotation` section in a GNN file:**

```markdown
## ActInfOntologyAnnotation
A=RecognitionMatrix
B=TransitionMatrix
D=Prior
s=HiddenState
o=Observation
```

In this example, `A`, `B`, `D`, `s`, and `o` are variables defined elsewhere in the GNN file (typically in the `StateSpaceBlock`). `RecognitionMatrix`, `TransitionMatrix`, etc., are terms expected to be present as keys in `act_inf_ontology_terms.json`.

## 3. Validation Process

The GNN processing pipeline includes Step 10 (`src/10_ontology.py`) that validates these annotations.

For detailed implementation, see **[src/ontology/AGENTS.md](../../src/ontology/AGENTS.md)**.

- **Parsing:** The `ActInfOntologyAnnotation` section is parsed to extract all `VariableName=OntologyTerm` mappings.
- **Validation:** Each `OntologyTerm` used in the GNN file is checked for its existence as a key in the loaded `act_inf_ontology_terms.json` file.
  - If the term is found, the annotation is considered **valid**.
  - If the term is not found, the annotation is considered **invalid**.
- **Reporting:** A report is generated (usually `output/10_ontology_output/ontology_processing_report.md`) that summarizes the validation results, including:
  - A list of all processed GNN files.
  - For each file, the parsed annotations and a summary of valid/invalid terms.
  - Overall statistics on the number of annotations found, passed, and failed.

This validation helps ensure that GNN models use consistent and recognized terminology from the specified ontology, aiding in model comparison, interoperability, and semantic clarity.

**Run ontology validation:**

```bash
# Validate ontology annotations (Step 10)
python src/10_ontology.py --target-dir input/gnn_files --verbose

# Or as part of the full pipeline
python src/main.py --only-steps "3,10" --target-dir input/gnn_files
```

## 4. MCP Integration

The ontology processing logic, including loading term definitions, parsing GNN files, and validating annotations, is typically exposed via MCP tools defined in `src/ontology/mcp.py`. This allows other parts of the system or external tools to leverage the ontology functionalities.

Key functions involved (often wrapped as MCP tools):

- `load_defined_ontology_terms`: Loads the `.json` file.
- `parse_gnn_ontology_section`: Extracts annotations from GNN file content.
- `validate_annotations`: Performs the validation against defined terms.
- `generate_ontology_report_for_file`: Creates the per-file report section.
