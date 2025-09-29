# GNN Processing - Agent Scaffolding

## Module Overview

**Purpose**: Core GNN file discovery, parsing, multi-format serialization, and validation for Generalized Notation Notation specifications

**Pipeline Step**: Step 3: GNN file processing (3_gnn.py)

**Category**: Core Processing

---

## Core Functionality

### Primary Responsibilities
1. Discover GNN specification files in target directories
2. Parse GNN markdown specifications into structured data
3. Serialize parsed models to 22 different formats
4. Validate GNN syntax and semantic correctness

### Key Capabilities
- Multi-format GNN parsing (markdown, JSON, YAML, etc.)
- 22 format serialization (Scala, Lean, Coq, Python, BNF, EBNF, Isabelle, Maxima, XML, JSON, Protobuf, YAML, XSD, ASN.1, PKL, Alloy, Z-notation, TLA+, Agda, Haskell, Pickle)
- Round-trip validation (parse → serialize → parse)
- Cross-format consistency checking

---

## API Reference

### Public Functions

#### `process_gnn_multi_format(target_dir, output_dir, logger, **kwargs) -> bool`
**Description**: Main processing function that discovers, parses, and serializes GNN files

**Parameters**:
- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for processed files
- `logger` (Logger): Logger instance
- `**kwargs`: Additional options (recursive, enable_round_trip, enable_cross_format)

**Returns**: `True` if processing succeeded, `False` otherwise

**Example**:
```python
from gnn.multi_format_processor import process_gnn_multi_format

success = process_gnn_multi_format(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/3_gnn_output"),
    logger=logger,
    recursive=True,
    enable_round_trip=True
)
```

#### `discover_gnn_files(target_dir, recursive=True) -> List[Path]`
**Description**: Discovers all GNN specification files in target directory

**Parameters**:
- `target_dir` (Path): Directory to search
- `recursive` (bool): Whether to search subdirectories

**Returns**: List of Path objects for discovered GNN files

---

### Public Classes

#### `GNNParser`
**Description**: Main parser for GNN specifications

**Methods**:
- `parse(file_path: Path) -> GNNModel` - Parse GNN file to model
- `validate(model: GNNModel) -> ValidationResult` - Validate model
- `serialize(model: GNNModel, format: GNNFormat) -> str` - Serialize to format

**Example**:
```python
parser = GNNParser()
model = parser.parse(Path("model.md"))
json_output = parser.serialize(model, GNNFormat.JSON)
```

#### `GNNFormat` (Enum)
**Description**: Enumeration of supported GNN formats

**Values**:
- `MARKDOWN`, `JSON`, `XML`, `YAML`, `SCALA`, `PROTOBUF`, `PKL`, `XSD`, `ASN1`, `LEAN`, `COQ`, `PYTHON`, `BNF`, `EBNF`, `ISABELLE`, `MAXIMA`, `ALLOY`, `Z_NOTATION`, `TLA_PLUS`, `AGDA`, `HASKELL`, `PICKLE`

---

## Dependencies

### Required Dependencies
- `pathlib` - File path manipulation
- `typing` - Type annotations
- `re` - Regular expression parsing
- `json` - JSON serialization

### Optional Dependencies
- `yaml` - YAML format support (fallback: skip YAML generation)
- `protobuf` - Protocol buffer support (fallback: skip Protobuf generation)

### Internal Dependencies
- `utils.pipeline_template` - Logging and pipeline utilities
- `pipeline.config` - Configuration management

---

## Configuration

### Environment Variables
- `GNN_MAX_FILE_SIZE` - Maximum GNN file size in bytes (default: 10MB)
- `GNN_ENABLE_VALIDATION` - Enable strict validation (default: True)

### Default Settings
```python
DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
DEFAULT_SUPPORTED_FORMATS = 22
DEFAULT_ENABLE_ROUND_TRIP = False
DEFAULT_ENABLE_CROSS_FORMAT = False
```

---

## Usage Examples

### Basic Usage
```python
from gnn.multi_format_processor import process_gnn_multi_format
from pathlib import Path

success = process_gnn_multi_format(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/3_gnn_output"),
    logger=logger
)
```

### Advanced Usage with Round-Trip Validation
```python
success = process_gnn_multi_format(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/3_gnn_output"),
    logger=logger,
    recursive=True,
    enable_round_trip=True,
    enable_cross_format=True
)
```

### Pipeline Integration
```python
# Called from 3_gnn.py
from gnn.multi_format_processor import process_gnn_multi_format

run_script = create_standardized_pipeline_script(
    "3_gnn.py",
    process_gnn_multi_format,
    "GNN discovery, parsing, and multi-format serialization"
)
```

---

## Input/Output Specification

### Input Requirements
- **File Formats**: `.md` files containing GNN specifications
- **Directory Structure**: Any directory structure (recursive search supported)
- **Prerequisites**: None (first processing step after template/setup)

### Output Products
- **Primary Outputs**: 
  - Parsed model JSON files (`*_parsed.json`)
  - 22 format serializations per model
- **Metadata Files**: 
  - `gnn_processing_results.json` - Processing summary
  - `gnn_processing_summary.json` - Detailed statistics
- **Artifacts**: Format-specific files in subdirectories

### Output Directory Structure
```
output/3_gnn_output/
├── model_name/
│   ├── model_name_parsed.json
│   ├── model_name.scala
│   ├── model_name.lean
│   ├── model_name.coq
│   ├── model_name.py
│   ├── ... (18 more formats)
├── gnn_processing_results.json
└── gnn_processing_summary.json
```

---

## Error Handling

### Error Categories
1. **File Not Found**: Log warning, continue to next file
2. **Parse Errors**: Log error with line number, mark file as failed
3. **Serialization Errors**: Log warning, skip problematic format
4. **Validation Errors**: Log error, optionally continue based on strict mode

### Fallback Strategies
- **Primary**: Parse all formats successfully
- **Fallback 1**: Skip problematic format, continue with others
- **Fallback 2**: Generate minimal JSON representation
- **Final**: Log error, continue pipeline (non-blocking)

### Error Reporting
- **Logging Level**: ERROR for parse failures, WARNING for format skips
- **User Messages**: "Failed to parse {file}: {specific_error}"
- **Recovery Suggestions**: "Check GNN syntax at line {N}" or "Install {dependency} for {format} support"

---

## Integration Points

### Orchestrated By
- **Script**: `3_gnn.py`
- **Function**: `run_script()` wrapper

### Imports From
- `utils.pipeline_template` - Standardized logging and error handling
- `pipeline.config` - Output directory management

### Imported By
- `5_type_checker.py` - Uses parsed model data
- `6_validation.py` - Uses validation results
- `7_export.py` - Uses parsed models for export
- `8_visualization.py` - Uses model structure for visualization
- `10_ontology.py` - Uses ontology terms from models
- `11_render.py` - Uses models for code generation

### Data Flow
```
input/gnn_files/*.md → GNN Parser → Multi-Format Serializer → output/3_gnn_output/
                            ↓
                    Parsed Model JSON
                            ↓
                  [Downstream Steps 5-23]
```

---

## Testing

### Test Files
- `src/tests/test_gnn_overall.py` - Integration tests
- `src/tests/test_gnn_parser.py` - Parser unit tests
- `src/tests/test_gnn_serialization.py` - Serialization tests
- `src/tests/test_gnn_validation.py` - Validation tests

### Test Coverage
- **Current**: 85%
- **Target**: 90%+

### Key Test Scenarios
1. Parse valid GNN markdown files
2. Handle malformed GNN syntax gracefully
3. Serialize to all 22 formats
4. Round-trip validation (parse → serialize → parse)
5. Cross-format consistency checking

### Test Commands
```bash
# Run GNN-specific tests
pytest src/tests/test_gnn*.py -v

# Run with coverage
pytest src/tests/test_gnn*.py --cov=src/gnn --cov-report=term-missing

# Run only parser tests
pytest src/tests/test_gnn_parser.py -v
```

---

## MCP Integration

### Tools Registered
- `gnn_parse` - Parse GNN file and return structured model
- `gnn_validate` - Validate GNN syntax
- `gnn_serialize` - Serialize model to specified format

### Tool Endpoints
```python
@mcp_tool("gnn_parse")
def parse_tool(file_path: str):
    """Parse a GNN specification file"""
    parser = GNNParser()
    return parser.parse(Path(file_path))
```

### MCP File Location
- `src/gnn/mcp.py` - MCP tool registrations

---

## Performance Characteristics

### Resource Requirements
- **Memory**: ~5MB per GNN file + 2MB per format
- **CPU**: Low (primarily I/O bound)
- **Disk**: ~150KB per format × 22 formats = ~3.3MB per model

### Execution Time
- **Fast Path**: <100ms for typical GNN file (13 variables, 11 connections)
- **Slow Path**: ~2-3s for large models (>100 variables, >50 connections)
- **Timeout**: None (synchronous processing)

### Scalability
- **Input Size Limits**: 10MB per file (configurable)
- **Parallelization**: Not currently parallelized (could process multiple files in parallel)

---

## Development Guidelines

### Adding New Formats
1. Add format to `GNNFormat` enum
2. Implement serializer in `src/gnn/serializers/`
3. Add format tests
4. Update documentation

### Code Style
- Follow PEP 8
- Use type hints for all public functions
- Document all public classes and methods
- Include docstring examples

### Testing Requirements
- All new formats must have serialization tests
- Round-trip tests for all formats
- Coverage must remain >85%

---

## Troubleshooting

### Common Issues

#### Issue 1: "Failed to parse GNN file at line X"
**Symptom**: Parser error with line number  
**Cause**: Invalid GNN syntax (missing delimiter, incorrect format)  
**Solution**: Check GNN syntax at specified line, ensure proper markdown formatting

#### Issue 2: "Format {X} not supported"
**Symptom**: Warning about missing format support  
**Cause**: Optional dependency not installed  
**Solution**: Install missing dependency or accept format will be skipped

#### Issue 3: "Round-trip validation failed"
**Symptom**: Parsed model differs after serialize/parse cycle  
**Cause**: Lossy serialization format or parser inconsistency  
**Solution**: Check format specification, report bug if parser issue

### Debug Mode
```bash
# Run with verbose logging
python src/3_gnn.py --verbose

# Check output directory
ls -la output/3_gnn_output/

# View processing summary
cat output/3_gnn_output/gnn_processing_summary.json | python -m json.tool
```

---

## Version History

### Current Version: 2.0.0

**Features**:
- 22 format serialization support
- Round-trip validation
- Cross-format consistency checking
- Comprehensive error handling

**Known Issues**:
- Some formats (Protobuf, ASN.1) require optional dependencies
- Large models (>100 variables) may be slow to serialize

### Roadmap
- **Next Version**: Parallel processing for multiple files
- **Future**: Incremental parsing, lazy serialization

---

## References

### Related Documentation
- [GNN Syntax Guide](../../doc/gnn/gnn_syntax.md)
- [Pipeline Overview](../../README.md)
- [.cursorrules](../../.cursorrules)

### External Resources
- [GNN Specification](doc/gnn/SPECIFICATION.md)
- [Active Inference Papers](https://en.wikipedia.org/wiki/Active_inference)

---

**Last Updated**: September 29, 2025  
**Maintainer**: GNN Pipeline Team  
**Status**: ✅ Production Ready



