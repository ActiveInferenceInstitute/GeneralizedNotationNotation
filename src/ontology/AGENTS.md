# Ontology Module - Agent Scaffolding

## Module Overview

**Purpose**: Active Inference Ontology processing, validation, and term mapping for GNN models

**Pipeline Step**: Step 10: Ontology processing (10_ontology.py)

**Category**: Semantic Validation / Ontology Management

**Status**: ✅ Production Ready

**Version**: 1.0.0

**Last Updated**: 2025-12-30

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

#### `process_ontology(target_dir, output_dir, logger=None, **kwargs) -> bool`
**Description**: Main ontology processing function called by orchestrator (10_ontology.py)

**Parameters**:
- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for ontology results
- `logger` (Logger, optional): Logger instance (default: None)
- `ontology_terms_file` (Path, optional): Path to ontology terms JSON file
- `recursive` (bool, optional): Process directories recursively (default: True)
- `**kwargs`: Additional options

**Returns**: `True` if processing succeeded

**Example**:
```python
from ontology import process_ontology

success = process_ontology(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/10_ontology_output"),
    ontology_terms_file=Path("input/ontology_terms.json")
)
```

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

## Configuration

### Configuration Options

#### Ontology File
- `ontology_terms_file` (Path): Path to ontology terms JSON file (default: `input/ontology_terms.json`)
- `ontology_format` (str): Ontology file format (default: `"json"`)

#### Validation Options
- `strict_validation` (bool): Require all terms to be in ontology (default: `False`)
- `allow_unknown_terms` (bool): Allow unrecognized terms with warnings (default: `True`)
- `validate_relationships` (bool): Validate ontological relationships (default: `True`)

#### Processing Options
- `recursive` (bool): Process directories recursively (default: `True`)
- `extract_all_terms` (bool): Extract all terms, not just variables (default: `False`)
- `generate_mappings` (bool): Generate term mappings (default: `True`)

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

## Error Handling

### Graceful Degradation
- **Ontology File Missing**: Use default ontology, log warning
- **Invalid Ontology Format**: Parse what's possible, log error
- **Invalid GNN Model**: Skip model, log error, continue with others
- **Term Validation Failure**: Log warning, continue processing

### Error Categories
1. **File I/O Errors**: Cannot read ontology file (fallback: use default ontology)
2. **Validation Errors**: Invalid ontology structure (fallback: skip validation)
3. **Term Extraction Errors**: Cannot extract terms from model (fallback: skip model)
4. **Mapping Errors**: Cannot generate term mappings (fallback: partial mapping)

### Error Recovery
- **Default Ontology**: Use built-in ontology if file unavailable
- **Partial Validation**: Validate what's possible, report failures
- **Resource Cleanup**: Proper cleanup of ontology resources on errors

---

## Integration Points

### Pipeline Integration
- **Input**: Receives parsed GNN models from Step 3 (gnn processing)
- **Output**: Generates ontology validation for Step 6 (validation), Step 11 (render), and Step 23 (report generation)
- **Dependencies**: Requires GNN parsing results from `3_gnn.py` output

### Module Dependencies
- **gnn/**: Reads parsed GNN model data for term extraction
- **validation/**: Provides ontology compliance for validation
- **render/**: Uses ontology mappings for code generation
- **report/**: Provides ontology compliance summaries

### External Integration
- **Ontology File**: JSON-based ontology term definitions
- **Active Inference Standards**: Validates against Active Inference ontology

### Data Flow
```
3_gnn.py (GNN parsing)
  ↓
10_ontology.py (Ontology processing)
  ↓
  ├→ 6_validation.py (Ontology compliance)
  ├→ 11_render.py (Term mapping)
  ├→ 23_report.py (Ontology reports)
  └→ output/10_ontology_output/ (Ontology results)
```

---

## Testing

### Test Files
- `src/tests/test_ontology_integration.py`

### Test Coverage
- **Current**: 78%
- **Target**: 85%+

---

