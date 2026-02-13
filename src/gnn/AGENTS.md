# GNN Processing - Agent Scaffolding

## Module Overview

**Purpose**: Core GNN file discovery, parsing, multi-format serialization, and validation for Generalized Notation Notation specifications

**Pipeline Step**: Step 3: GNN file processing (3_gnn.py)

**Category**: Core Processing

**Status**: ✅ Production Ready

**Version**: 1.1.3

**Last Updated**: 2026-01-21

---

## Core Functionality

### Primary Responsibilities

1. Discover GNN specification files in target directories
2. Parse GNN markdown specifications into structured data
3. Serialize parsed models to 23 different formats
4. Validate GNN syntax and semantic correctness

### Key Capabilities

- Multi-format GNN parsing (markdown, JSON, YAML, etc.)
- 23 format serialization (Scala, Lean, Coq, Python, BNF, EBNF, Isabelle, Maxima, XML, JSON, Protobuf, YAML, XSD, ASN.1, PKL, Alloy, Z-notation, TLA+, Agda, Haskell, Pickle, Markdown, PNML)
- Round-trip validation (parse → serialize → parse)
- Cross-format consistency checking

---

## API Reference

### Pipeline Processing Function

#### `process_gnn_multi_format(target_dir: Path, output_dir: Path, logger: logging.Logger, recursive: bool = True, verbose: bool = False, **kwargs: Any) -> bool`

**Description**: Main processing function used by pipeline orchestrator (3_gnn.py). Discovers, parses, and serializes GNN files to all supported formats.

**Parameters**:

- `target_dir` (Path): Directory containing GNN files to process
- `output_dir` (Path): Base output directory (step-specific directory will be created)
- `logger` (logging.Logger): Logger instance
- `recursive` (bool): Whether to recurse into subdirectories (default: True)
- `verbose` (bool): Enable verbose logs (default: False)
- `**kwargs` (Any): Additional processing options

**Returns**: `bool` - True on success, False otherwise

**Location**: `src/gnn/multi_format_processor.py`

**Example**:

```python
from gnn.multi_format_processor import process_gnn_multi_format
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
success = process_gnn_multi_format(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output"),
    logger=logger,
    recursive=True,
    verbose=True
)
```

### Exported Functions from `__init__.py`

#### `process_gnn_directory(directory: Union[str, Path], output_dir: Union[str, Path, None] = None, recursive: bool = True, parallel: bool = False) -> Dict[str, Any]`

**Description**: Process all GNN files in a directory. Returns processing results dictionary.

**Parameters**:

- `directory` (Union[str, Path]): Directory to process
- `output_dir` (Union[str, Path, None]): Optional output directory for results (default: None)
- `recursive` (bool): Whether to process subdirectories (default: True)
- `parallel` (bool): Whether to use parallel processing (not implemented, default: False)

**Returns**: `Dict[str, Any]` - Dictionary with processing results containing:

- `status` (str): Processing status ("SUCCESS" or "FAILED")
- `files` (List[str]): List of processed file paths
- `processed_files` (List[str]): List of successfully processed files

**Location**: `src/gnn/processor.py`

#### `process_gnn_directory_lightweight(target_dir: Path, output_dir: Path = None, recursive: bool = False) -> Dict[str, Any]`

**Description**: Lightweight GNN directory processing without heavy dependencies and faster execution.

**Parameters**:

- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path, optional): Directory to save results (default: None)
- `recursive` (bool): Whether to process subdirectories (default: False)

**Returns**: `Dict[str, Any]` - Dictionary with processing results containing:

- `timestamp` (str): Processing timestamp
- `target_directory` (str): Source directory path
- `files_found` (int): Number of files discovered
- `files_processed` (int): Number of files successfully processed
- `success` (bool): Overall success status
- `errors` (List[Dict]): List of error information
- `parsed_files` (List[Dict]): List of parsed file information
- `validation_results` (List[Dict]): List of validation results

**Location**: `src/gnn/processor.py`

#### `discover_gnn_files(directory: Union[str, Path], recursive: bool = True) -> List[Path]`

**Description**: Discovers all GNN specification files in target directory with support for multiple formats.

**Parameters**:

- `directory` (Union[str, Path]): Directory to search
- `recursive` (bool): Whether to search subdirectories (default: True)

**Returns**: `List[Path]` - List of Path objects for discovered GNN files

**Supported Extensions**: `.md`, `.markdown`, `.json`, `.xml`, `.yaml`, `.yml`, `.py`, `.jl`, `.hs`, `.lean`, `.v`, `.thy`, `.max`, `.proto`, `.xsd`, `.asn1`, `.pkl`, `.als`, `.z`, `.tla`, `.agda`, `.bnf`, `.ebnf`

**Location**: `src/gnn/processor.py`

#### `parse_gnn_file(file_path: Union[str, Path]) -> Dict[str, Any]`

**Description**: Parse a single GNN file and extract basic information.

**Parameters**:

- `file_path` (Union[str, Path]): Path to the GNN file

**Returns**: `Dict[str, Any]` - Dictionary with parsed information containing:

- `file_path` (str): Path to the file
- `file_name` (str): Name of the file
- `file_size` (int): Size of the file in bytes
- `sections` (List[str]): List of extracted sections
- `variables` (List[str]): List of extracted variables
- `structure_info` (Dict): Structure analysis information
- `parse_timestamp` (str): Timestamp of parsing

**Location**: `src/gnn/processor.py`

#### `validate_gnn_structure(file_path: Union[str, Path]) -> Dict[str, Any]`

**Description**: Validate the structure of a GNN file.

**Parameters**:

- `file_path` (Union[str, Path]): Path to the GNN file

**Returns**: `Dict[str, Any]` - Dictionary with validation results containing:

- `file_path` (str): Path to the file
- `file_name` (str): Name of the file
- `valid` (bool): Whether the file structure is valid
- `errors` (List[str]): List of validation errors
- `warnings` (List[str]): List of validation warnings
- `validation_timestamp` (str): Timestamp of validation

**Location**: `src/gnn/processor.py`

#### `generate_gnn_report(processing_results: Dict[str, Any], output_path: Union[str, Path] = None) -> str`

**Description**: Generate a report from GNN processing results.

**Parameters**:

- `processing_results` (Dict[str, Any]): Results from GNN processing
- `output_path` (Union[str, Path, None]): Optional path to save the report (default: None)

**Returns**: `str` - Report content as markdown string

**Location**: `src/gnn/processor.py`

#### `get_module_info() -> Dict[str, Any]`

**Description**: Get information about the GNN module.

**Returns**: `Dict[str, Any]` - Dictionary with module information containing:

- `name` (str): Module name
- `version` (str): Module version
- `description` (str): Module description
- `features` (List[str]): List of available features
- `available_validators` (List[str]): List of available validators
- `available_parsers` (List[str]): List of available parsers
- `schema_formats` (List[str]): List of supported schema formats
- `supported_formats` (List[str]): List of supported file formats
- `capabilities` (Dict): Dictionary of capability flags

**Location**: `src/gnn/processor.py`

#### `process_gnn(*args, **kwargs) -> Dict[str, Any]`

**Description**: Alias for `process_gnn_directory`. Provides backward compatibility.

**Parameters**: Same as `process_gnn_directory`

**Returns**: Same as `process_gnn_directory`

**Location**: `src/gnn/__init__.py`

#### `validate_gnn_file(content: str) -> Dict[str, Any]`

**Description**: Validate GNN file content string.

**Parameters**:

- `content` (str): GNN file content as string

**Returns**: `Dict[str, Any]` - Dictionary with validation results:

- `is_valid` (bool): Whether content is valid
- `errors` (List[str]): List of validation errors

**Location**: `src/gnn/__init__.py`

#### `validate_gnn(file_path_or_content: str, validation_level: ValidationLevel = ValidationLevel.STANDARD, **kwargs) -> Tuple[bool, List[str]]`

**Description**: Validate a GNN file or content string.

**Parameters**:

- `file_path_or_content` (str): Path to a GNN file or GNN content string
- `validation_level` (ValidationLevel): Level of validation to perform (default: STANDARD)
- `**kwargs`: Additional validation options

**Returns**: `Tuple[bool, List[str]]` - Tuple of (is_valid, list_of_errors)

**Location**: `src/gnn/parser.py`

### Helper Functions (Internal but Exported)

#### `_extract_sections_lightweight(content: str) -> List[str]`

**Description**: Extract sections from GNN content using lightweight parsing.

**Parameters**:

- `content` (str): GNN file content

**Returns**: `List[str]` - List of extracted section names

**Location**: `src/gnn/processor.py`

#### `_extract_variables_lightweight(content: str) -> List[str]`

**Description**: Extract variables from GNN content using lightweight parsing.

**Parameters**:

- `content` (str): GNN file content

**Returns**: `List[str]` - List of extracted variable names

**Location**: `src/gnn/processor.py`

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

- `MARKDOWN`, `JSON`, `XML`, `YAML`, `SCALA`, `PROTOBUF`, `PKL`, `XSD`, `ASN1`, `PNML`, `LEAN`, `COQ`, `PYTHON`, `BNF`, `EBNF`, `ISABELLE`, `MAXIMA`, `ALLOY`, `Z_NOTATION`, `TLA_PLUS`, `AGDA`, `HASKELL`, `PICKLE`

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
DEFAULT_SUPPORTED_FORMATS = 23
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
  - 23 format serializations per model
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
│   ├── ... (19 more formats)
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
3. Serialize to all 23 formats
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

### Current Version: 1.1.3

**Features**:

- 23 format serialization support
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

- [GNN Specification](../../doc/gnn/gnn_dsl_manual.md)
- [Active Inference Papers](https://en.wikipedia.org/wiki/Active_inference)

---
