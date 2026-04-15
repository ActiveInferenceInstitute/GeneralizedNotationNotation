# Ontology Module - Agent Scaffolding

## Module Overview

**Purpose**: Active Inference Ontology processing, validation, and term mapping for GNN models

**Pipeline Step**: Step 10: Ontology processing (10_ontology.py)

**Category**: Semantic Validation / Ontology Management

**Status**: ✅ Production Ready

**Version**: 1.5.0

**Last Updated**: 2026-04-15

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

#### `parse_gnn_ontology_section(content: str) -> Dict[str, Any]`
**Description**: Extract the ontology annotation section (e.g. `## ActInfOntologyAnnotation`) from GNN Markdown content.

**Parameters**:
- `content` (str): Raw GNN Markdown content

**Returns**: `Dict[str, Any]` - Parsed ontology content (annotations + any extracted fields)

#### `load_defined_ontology_terms() -> Dict[str, Any]`
**Description**: Load the Active Inference ontology term dictionary used for validation (default: `src/ontology/act_inf_ontology_terms.json`).

**Returns**: `Dict[str, Any]` - Term definitions (format depends on the JSON file)

#### `validate_annotations(annotations: List[str], ontology_terms: Dict[str, Any] | None = None) -> Dict[str, Any]`
**Description**: Validate ontology annotations (e.g. `A=LikelihoodMatrix`) against the known ontology term set.

**Parameters**:
- `annotations` (List[str]): Annotation strings
- `ontology_terms` (Dict[str, Any] | None): Optional explicit ontology term set (default: module’s loaded terms)

**Returns**: `Dict[str, Any]` - Validation details (valid/invalid annotations, suggestions when available)

#### `process_gnn_ontology(gnn_file: str) -> Dict[str, Any]`
**Description**: Process ontology annotations for a single GNN file path (reads file, parses ontology section, validates annotations).

#### `generate_ontology_report_for_file(gnn_file: Path, output_dir: Path) -> Dict[str, Any]`
**Description**: Generate and write a per-file ontology report JSON for a single GNN file.

#### `validate_ontology_terms(terms: List[str] | str = None) -> bool`
**Description**: Convenience validator for terms/annotations (returns boolean validity; used by integrations/tests).

### Public Classes

#### `OntologyProcessor`
**Description**: Convenience wrapper exposing `process_ontology()` and `validate_terms()` around the functional API.

#### `OntologyValidator`
**Description**: Validator exposing `validate_ontology()` and `check_consistency()` for quick boolean checks.

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
from ontology import process_ontology

success = process_ontology(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/10_ontology_output"),
    ontology_terms_file=Path("src/ontology/act_inf_ontology_terms.json")
)
```

### Single-file processing
```python
from ontology import process_gnn_ontology

result = process_gnn_ontology("input/gnn_files/discrete/simple_mdp.md")
print(result["success"], result.get("validation_result", {}))
```

---

## Output Specification

### Output Products
- `ontology_results.json` - Aggregate summary across processed files
- `*_ontology_report.json` - Per-model ontology validation reports

### Output Directory Structure
```
output/10_ontology_output/
├── ontology_results.json
├── actinf_pomdp_agent_ontology_report.json
├── deep_planning_horizon_ontology_report.json
└── ... (one per processed model)
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
1. **File I/O Errors**: Cannot read ontology file (recovery: use default ontology)
2. **Validation Errors**: Invalid ontology structure (recovery: skip validation)
3. **Term Extraction Errors**: Cannot extract terms from model (recovery: skip model)
4. **Mapping Errors**: Cannot generate term mappings (recovery: partial mapping)

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
- `src/tests/test_ontology_overall.py`

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
- [GNN Ontology Guide](../../doc/gnn/advanced/gnn_ontology.md)

### External Resources
- [Active Inference Institute](https://activeinference.institute/)
- [Active Inference Ontology Documentation](https://activeinference.org/ontology)

---

**Last Updated**: 2026-04-15
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Version**: 1.5.0
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern



---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
