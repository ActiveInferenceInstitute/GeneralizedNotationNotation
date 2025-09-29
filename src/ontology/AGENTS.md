# Ontology Module - Agent Scaffolding

## Module Overview

**Purpose**: Active Inference Ontology processing, validation, and term mapping for GNN models

**Pipeline Step**: Step 10: Ontology processing (10_ontology.py)

**Category**: Semantic Validation / Ontology Management

---

## Core Functionality

### Primary Responsibilities
1. Extract ontology terms from GNN models
2. Validate against Active Inference Ontology
3. Map GNN components to ontological concepts
4. Generate ontology compliance reports
5. Identify semantic inconsistencies

### Key Capabilities
- Term extraction from GNN specifications
- Ontology compliance validation
- Semantic relationship mapping
- Hierarchical concept analysis
- Cross-reference validation

---

## API Reference

### Public Functions

#### `process_ontology_standardized(target_dir, output_dir, logger, **kwargs) -> bool`
**Description**: Main ontology processing function

**Parameters**:
- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for ontology results
- `logger` (Logger): Logger instance
- `ontology_terms_file` (Path): Path to ontology terms JSON
- `recursive` (bool): Process directories recursively
- `**kwargs`: Additional options

**Returns**: `True` if processing succeeded

#### `extract_ontology_terms(gnn_model: Dict) -> List[str]`
**Description**: Extract ontology terms from parsed GNN model

#### `validate_ontology_compliance(terms: List[str], ontology: Dict) -> Dict`
**Description**: Validate extracted terms against ontology

#### `generate_ontology_mapping(gnn_model: Dict, ontology: Dict) -> Dict`
**Description**: Generate term mapping between GNN and ontology

---

## Ontology Structure

### Active Inference Ontology Terms
- **Core Concepts**: Belief, Preference, Action, Observation
- **Mathematical**: Matrix, Vector, Probability, Distribution
- **Process**: Inference, Learning, Planning, Control
- **Architecture**: Agent, Environment, Model, Policy

---

## Dependencies

### Required Dependencies
- `json` - Ontology file loading
- `pathlib` - File operations

### Internal Dependencies
- `utils.pipeline_template` - Logging utilities
- `pipeline.config` - Configuration management
- `ontology.processor` - Core processing logic

---

## Usage Examples

### Basic Usage
```python
from ontology import process_ontology_standardized

success = process_ontology_standardized(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/10_ontology_output"),
    logger=logger,
    ontology_terms_file=Path("input/ontology_terms.json")
)
```

### Term Extraction
```python
from ontology.processor import extract_ontology_terms

terms = extract_ontology_terms(parsed_gnn_model)
print(f"Extracted terms: {terms}")
```

---

## Output Specification

### Output Products
- `ontology_validation_results.json` - Validation results
- `ontology_term_mapping.json` - Term mappings
- `ontology_compliance_report.json` - Compliance summary

### Output Directory Structure
```
output/10_ontology_output/
├── ontology_validation_results.json
├── ontology_term_mapping.json
└── ontology_compliance_report.json
```

---

## Validation Rules

### Compliance Checks
1. All GNN variables map to ontology terms
2. Connections align with ontological relationships
3. POMDP components match Active Inference structure
4. Terminology is consistent with ontology

### Warning Conditions
- Unrecognized terms (potential new concepts)
- Ambiguous mappings (multiple possible interpretations)
- Missing required ontological components

---

## Performance Characteristics

### Latest Execution
- **Duration**: 55ms
- **Memory**: 28.6 MB
- **Status**: SUCCESS
- **Terms Validated**: 13

---

## Testing

### Test Files
- `src/tests/test_ontology_integration.py`

### Test Coverage
- **Current**: 78%
- **Target**: 85%+

---

**Last Updated**: September 29, 2025  
**Status**: ✅ Production Ready

