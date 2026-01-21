# Ontology Module - Agent Scaffolding

## Module Overview

**Purpose**: Active Inference Ontology processing, validation, and term mapping for GNN models

**Pipeline Step**: Step 10: Ontology processing (10_ontology.py)

**Category**: Semantic Validation / Ontology Management

**Status**: ✅ Production Ready

**Version**: 1.0.0

**Last Updated**: 2026-01-21

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

#### `process_ontology(target_dir: Path, output_dir: Path, logger: Optional[logging.Logger] = None, **kwargs) -> bool`
**Description**: Main ontology processing function called by orchestrator (10_ontology.py). Processes GNN files for ontology validation and mapping.

**Parameters**:
- `target_dir` (Path): Directory containing GNN files to process
- `output_dir` (Path): Output directory for ontology results
- `logger` (Optional[logging.Logger]): Logger instance (default: None)
- `ontology_terms_file` (Path, optional): Path to ontology terms JSON file (default: `src/ontology/act_inf_ontology_terms.json`)
- `recursive` (bool, optional): Process directories recursively (default: True)
- `strict_validation` (bool, optional): Require all terms to be in ontology (default: False)
- `generate_mapping` (bool, optional): Generate ontology mapping (default: True)
- `generate_enhancements` (bool, optional): Generate enhancement suggestions (default: True)
- `**kwargs`: Additional processing options

**Returns**: `bool` - True if processing succeeded, False otherwise

**Example**:
```python
from ontology import process_ontology
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
success = process_ontology(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/10_ontology_output"),
    ontology_terms_file=Path("src/ontology/act_inf_ontology_terms.json"),
    strict_validation=True
)
```

#### `extract_ontology_terms(gnn_model: Dict[str, Any]) -> List[str]`
**Description**: Extract ontology terms from parsed GNN model.

**Parameters**:
- `gnn_model` (Dict[str, Any]): Parsed GNN model dictionary

**Returns**: `List[str]` - List of extracted ontology terms

#### `validate_ontology_compliance(terms: List[str], ontology: Dict[str, Any], strict: bool = False) -> Dict[str, Any]`
**Description**: Validate extracted terms against Active Inference ontology.

**Parameters**:
- `terms` (List[str]): List of terms to validate
- `ontology` (Dict[str, Any]): Ontology dictionary
- `strict` (bool): Require all terms to be in ontology (default: False)

**Returns**: `Dict[str, Any]` - Validation results with:
- `valid_terms` (List[str]): Valid terms found in ontology
- `invalid_terms` (List[str]): Terms not found in ontology
- `compliance_score` (float): Compliance score (0.0-1.0)
- `suggestions` (List[str]): Suggestions for invalid terms

#### `generate_ontology_mapping(gnn_model: Dict[str, Any], ontology: Dict[str, Any]) -> Dict[str, Any]`
**Description**: Generate term mapping between GNN model and ontology concepts.

**Parameters**:
- `gnn_model` (Dict[str, Any]): Parsed GNN model dictionary
- `ontology` (Dict[str, Any]): Ontology dictionary

**Returns**: `Dict[str, Any]` - Mapping results with:
- `mapped_terms` (Dict[str, str]): GNN term to ontology concept mapping
- `relationships` (List[Dict]): Identified term relationships
- `semantic_clusters` (List[List[str]]): Terms grouped by semantic similarity
- `coverage` (float): Ontology coverage score (0.0-1.0)

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
- `ontology_terms_file` (Path): Path to ontology terms JSON file (default: `src/ontology/act_inf_ontology_terms.json`)
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
    ontology_terms_file=Path("src/ontology/act_inf_ontology_terms.json")
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

### Key Test Scenarios
1. Ontology term extraction
2. Ontology compliance validation
3. Semantic relationship mapping
4. Cross-reference validation

---

## MCP Integration

### Tools Registered
- `ontology.extract_terms` - Extract ontology terms from GNN model
- `ontology.validate_compliance` - Validate ontology compliance
- `ontology.generate_mapping` - Generate ontology mapping
- `ontology.analyze_semantics` - Analyze semantic relationships

### Tool Endpoints
```python
@mcp_tool("ontology.extract_terms")
def extract_ontology_terms_tool(gnn_content: str) -> List[str]:
    """Extract ontology terms from GNN content"""
    # Implementation
```

### MCP File Location
- `src/ontology/mcp.py` - MCP tool registrations

---

## Troubleshooting

### Common Issues

#### Issue 1: Ontology validation fails
**Symptom**: Validation reports errors even for valid terms  
**Cause**: Ontology terms file missing or outdated  
**Solution**: 
- Check that `src/ontology/act_inf_ontology_terms.json` exists
- Verify ontology terms file format is valid JSON
- Update ontology terms file if needed
- Use `--verbose` flag for detailed validation messages

#### Issue 2: Terms not found in ontology
**Symptom**: Valid Active Inference terms reported as invalid  
**Cause**: Ontology terms file incomplete or term naming mismatch  
**Solution**:
- Check term spelling and case sensitivity
- Verify ontology terms file includes all required terms
- Use `--strict-validation=False` for lenient validation
- Review ontology terms file structure

#### Issue 3: Semantic mapping incomplete
**Symptom**: Mapping generation produces incomplete results  
**Cause**: Missing ontology relationships or incomplete GNN model  
**Solution**:
- Ensure GNN model has complete ontology annotations
- Verify ontology terms file includes relationship data
- Check that GNN processing (step 3) completed successfully

---

## Version History

### Current Version: 1.0.0

**Features**:
- Ontology term extraction
- Ontology compliance validation
- Semantic relationship mapping
- Hierarchical concept analysis
- Cross-reference validation

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Enhanced semantic relationship detection
- **Future**: Automated ontology term suggestion

---

## References

### Related Documentation
- [Pipeline Overview](../../README.md)
- [Architecture Guide](../../ARCHITECTURE.md)
- [Active Inference Ontology](https://activeinference.org)
- [GNN Ontology Guide](../../doc/gnn/gnn_ontology.md)

### External Resources
- [Active Inference Institute](https://activeinference.institute/)
- [Active Inference Ontology Documentation](https://activeinference.org/ontology)

---

**Last Updated**: 2026-01-21
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Version**: 1.0.0
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern

