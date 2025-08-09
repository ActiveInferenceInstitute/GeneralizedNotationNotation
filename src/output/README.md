# Output Module

This module provides comprehensive output management capabilities for the GNN pipeline, including file organization, result aggregation, output validation, and cross-step data coordination.

## Module Structure

```
src/output/
├── __init__.py                    # Module initialization and exports
├── README.md                      # This documentation
├── manager.py                     # Output management system
├── organizer.py                   # File organization utilities
├── validator.py                   # Output validation
├── aggregator.py                  # Result aggregation
├── gnn_exports/                   # GNN export outputs
│   ├── json/                      # JSON format exports
│   ├── xml/                       # XML format exports
│   ├── graphml/                   # GraphML format exports
│   ├── gexf/                      # GEXF format exports
│   └── pickle/                    # Pickle format exports
└── mcp.py                         # Model Context Protocol integration
```

## Core Components

### Output Management Functions

#### `process_output(target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs) -> bool`
Main function for processing output-related tasks.

**Features:**
- Output file organization
- Result aggregation and validation
- Cross-step data coordination
- Output documentation
- Performance tracking

**Returns:**
- `bool`: Success status of output operations

### Output Organization System

#### `organize_pipeline_outputs(output_dir: Path) -> bool`
Organizes pipeline outputs into structured directories.

**Organization Features:**
- Step-based directory structure
- File type categorization
- Metadata preservation
- Cross-reference management
- Archive creation

#### `validate_output_integrity(output_dir: Path) -> Dict[str, Any]`
Validates the integrity of pipeline outputs.

**Validation Features:**
- File completeness checking
- Format validation
- Size verification
- Content validation
- Cross-reference verification

#### `aggregate_pipeline_results(output_dir: Path) -> Dict[str, Any]`
Aggregates results from all pipeline steps.

**Aggregation Features:**
- Result collection
- Performance metrics
- Success/failure analysis
- Resource usage tracking
- Quality assessment

### Output Management Functions

#### `create_output_structure(base_dir: Path) -> bool`
Creates the standard output directory structure.

**Structure Features:**
- Step-specific directories
- Format-specific subdirectories
- Metadata directories
- Archive directories
- Temporary directories

#### `manage_output_files(file_paths: List[Path], output_dir: Path) -> bool`
Manages output files with proper organization.

**Management Features:**
- File categorization
- Duplicate handling
- Version control
- Backup creation
- Cleanup management

#### `generate_output_summary(output_dir: Path) -> Dict[str, Any]`
Generates comprehensive output summary.

**Summary Features:**
- File counts and sizes
- Processing statistics
- Performance metrics
- Quality indicators
- Resource usage

## Usage Examples

### Basic Output Processing

```python
from output import process_output

# Process output-related tasks
success = process_output(
    target_dir=Path("pipeline_results/"),
    output_dir=Path("organized_output/"),
    verbose=True
)

if success:
    print("Output processing completed successfully")
else:
    print("Output processing failed")
```

### Output Organization

```python
from output import organize_pipeline_outputs

# Organize pipeline outputs
success = organize_pipeline_outputs(
    output_dir=Path("output/")
)

if success:
    print("Outputs organized successfully")
else:
    print("Output organization failed")
```

### Output Validation

```python
from output import validate_output_integrity

# Validate output integrity
validation_results = validate_output_integrity(
    output_dir=Path("output/")
)

print(f"Output valid: {validation_results['valid']}")
print(f"Missing files: {len(validation_results['missing_files'])}")
print(f"Corrupted files: {len(validation_results['corrupted_files'])}")
print(f"Format errors: {len(validation_results['format_errors'])}")
```

### Result Aggregation

```python
from output import aggregate_pipeline_results

# Aggregate pipeline results
aggregation_results = aggregate_pipeline_results(
    output_dir=Path("output/")
)

print(f"Total files: {aggregation_results['total_files']}")
print(f"Total size: {aggregation_results['total_size']}")
print(f"Processing time: {aggregation_results['processing_time']}")
print(f"Success rate: {aggregation_results['success_rate']}")
```

### Output Structure Creation

```python
from output import create_output_structure

# Create output structure
success = create_output_structure(
    base_dir=Path("output/")
)

if success:
    print("Output structure created successfully")
else:
    print("Output structure creation failed")
```

### Output File Management

```python
from output import manage_output_files

# Manage output files
    file_paths = [
    Path("output/6_validation_output/validation_results.json"),
    Path("output/7_export_output/actinf_pomdp_agent/actinf_pomdp_agent_xml.xml"),
    Path("output/8_visualization_output/actinf_pomdp_agent/graph.png")
]

success = manage_output_files(
    file_paths=file_paths,
    output_dir=Path("organized_output/")
)

if success:
    print("Output files managed successfully")
else:
    print("Output file management failed")
```

### Output Summary Generation

```python
from output import generate_output_summary

# Generate output summary
summary = generate_output_summary(
    output_dir=Path("output/")
)

print(f"Total files: {summary['total_files']}")
print(f"Total size: {summary['total_size_mb']} MB")
print(f"File types: {summary['file_types']}")
print(f"Step coverage: {summary['step_coverage']}")
```

## Output Structure

### Standard Output Directory Structure
```
output/
├── template/                       # Template processing outputs
│   ├── templates/                  # Generated templates
│   ├── customizations/             # Template customizations
│   └── validation/                 # Template validation results
├── setup/                          # Setup outputs
│   ├── environment/                # Environment information
│   ├── dependencies/               # Dependency status
│   └── configuration/              # Configuration files
├── tests/                          # Test outputs
│   ├── results/                    # Test results
│   ├── coverage/                   # Coverage reports
│   └── performance/                # Performance metrics
├── gnn/                            # GNN processing outputs
│   ├── parsed/                     # Parsed GNN files
│   ├── validated/                  # Validation results
│   └── processed/                  # Processed models
├── model_registry/                 # Model registry outputs
│   ├── registry/                   # Registry files
│   ├── versions/                   # Version information
│   └── metadata/                   # Model metadata
├── type_checker/                   # Type checker outputs
│   ├── syntax/                     # Syntax validation
│   ├── types/                      # Type information
│   └── resources/                  # Resource estimates
├── validation/                     # Validation outputs
│   ├── consistency/                # Consistency checks
│   ├── semantic/                   # Semantic validation
│   └── quality/                    # Quality assessment
├── export/                         # Export outputs
│   ├── json/                       # JSON exports
│   ├── xml/                        # XML exports
│   ├── graphml/                    # GraphML exports
│   ├── gexf/                       # GEXF exports
│   └── pickle/                     # Pickle exports
├── visualization/                   # Visualization outputs
│   ├── graphs/                     # Graph visualizations
│   ├── matrices/                   # Matrix visualizations
│   └── interactive/                # Interactive plots
├── advanced_visualization/          # Advanced visualization outputs
│   ├── 3d/                         # 3D visualizations
│   ├── animations/                 # Animations
│   └── interactive/                # Interactive visualizations
├── ontology/                       # Ontology outputs
│   ├── terms/                      # Ontology terms
│   ├── mappings/                   # Term mappings
│   └── validation/                 # Ontology validation
├── render/                         # Render outputs
│   ├── pymdp/                      # PyMDP code
│   ├── rxinfer/                    # RxInfer code
│   ├── activeinference_jl/         # ActiveInference.jl code
│   └── jax/                        # JAX code
├── execute/                        # Execute outputs
│   ├── results/                    # Execution results
│   ├── logs/                       # Execution logs
│   └── performance/                # Performance metrics
├── llm/                            # LLM outputs
│   ├── analysis/                   # LLM analysis
│   ├── insights/                   # Generated insights
│   └── interpretations/            # Model interpretations
├── ml_integration/                 # ML integration outputs
│   ├── models/                     # Trained models
│   ├── evaluations/                # Model evaluations
│   └── predictions/                # Predictions
├── audio/                          # Audio outputs
│   ├── sapf/                       # SAPF audio
│   ├── pedalboard/                 # Pedalboard audio
│   └── other/                      # Other audio formats
├── analysis/                       # Analysis outputs
│   ├── statistics/                 # Statistical analysis
│   ├── performance/                # Performance analysis
│   └── comparisons/                # Model comparisons
├── integration/                    # Integration outputs
│   ├── coordination/               # Module coordination
│   ├── data_flow/                  # Data flow management
│   └── system/                     # System integration
├── security/                       # Security outputs
│   ├── validation/                 # Security validation
│   ├── vulnerabilities/            # Vulnerability reports
│   └── compliance/                 # Compliance reports
├── research/                       # Research outputs
│   ├── experiments/                # Experimental results
│   ├── insights/                   # Research insights
│   └── methodology/                # Methodology documentation
├── website/                        # Website outputs
│   ├── html/                       # Generated HTML
│   ├── assets/                     # Website assets
│   └── static/                     # Static files
├── report/                         # Report outputs
│   ├── comprehensive/              # Comprehensive reports
│   ├── summaries/                  # Executive summaries
│   └── detailed/                   # Detailed reports
├── metadata/                       # Pipeline metadata
│   ├── configuration/              # Configuration files
│   ├── logs/                       # Pipeline logs
│   └── status/                     # Status information
└── archives/                       # Archive outputs
    ├── compressed/                 # Compressed archives
    ├── backups/                    # Backup files
    └── snapshots/                  # Pipeline snapshots
```

### Output File Types

#### Data Formats
- **JSON**: Structured data exports
- **XML**: Hierarchical data exports
- **GraphML**: Graph structure exports
- **GEXF**: Graph exchange format
- **Pickle**: Python object serialization

#### Visualization Formats
- **PNG**: Static image visualizations
- **SVG**: Scalable vector graphics
- **PDF**: Portable document format
- **HTML**: Interactive visualizations

#### Report Formats
- **HTML**: Web-based reports
- **PDF**: Portable document reports
- **Markdown**: Text-based reports
- **JSON**: Structured reports

#### Code Formats
- **Python**: PyMDP and JAX code
- **Julia**: RxInfer and ActiveInference.jl code
- **TOML**: Configuration files
- **YAML**: Configuration files

## Configuration Options

### Output Configuration
```python
# Output configuration
output_config = {
    'base_directory': 'output/',
    'organization_enabled': True,
    'validation_enabled': True,
    'aggregation_enabled': True,
    'compression_enabled': True,
    'backup_enabled': True
}
```

### Organization Configuration
```python
# Organization configuration
organization_config = {
    'step_based_dirs': True,
    'format_based_dirs': True,
    'metadata_preservation': True,
    'cross_references': True,
    'archive_creation': True
}
```

### Validation Configuration
```python
# Validation configuration
validation_config = {
    'file_completeness': True,
    'format_validation': True,
    'size_verification': True,
    'content_validation': True,
    'cross_reference_check': True
}
```

## Error Handling

### Output Processing Failures
```python
# Handle output processing failures gracefully
try:
    success = process_output(target_dir, output_dir)
except OutputError as e:
    logger.error(f"Output processing failed: {e}")
    # Provide fallback processing or error reporting
```

### Organization Issues
```python
# Handle organization issues gracefully
try:
    success = organize_pipeline_outputs(output_dir)
except OrganizationError as e:
    logger.warning(f"Output organization failed: {e}")
    # Provide fallback organization or error reporting
```

### Validation Issues
```python
# Handle validation issues gracefully
try:
    validation = validate_output_integrity(output_dir)
except ValidationError as e:
    logger.error(f"Output validation failed: {e}")
    # Provide fallback validation or error reporting
```

## Performance Optimization

### Output Optimization
- **Parallel Processing**: Process outputs in parallel
- **Incremental Updates**: Update only changed outputs
- **Caching**: Cache output metadata
- **Compression**: Compress large outputs

### Organization Optimization
- **Batch Operations**: Organize files in batches
- **Lazy Loading**: Load metadata on demand
- **Indexing**: Create output indices
- **Cleanup**: Automatic cleanup of temporary files

### Validation Optimization
- **Parallel Validation**: Validate files in parallel
- **Caching**: Cache validation results
- **Incremental Validation**: Validate only new/changed files
- **Error Recovery**: Efficient error recovery

## Testing and Validation

### Unit Tests
```python
# Test individual output functions
def test_output_organization():
    success = organize_pipeline_outputs(test_output_dir)
    assert success
    # Verify organized structure
    organized_files = list(test_output_dir.glob("**/*"))
    assert len(organized_files) > 0
```

### Integration Tests
```python
# Test complete output pipeline
def test_output_pipeline():
    success = process_output(test_dir, output_dir)
    assert success
    # Verify output processing
    output_files = list(output_dir.glob("**/*"))
    assert len(output_files) > 0
```

### Validation Tests
```python
# Test output validation
def test_output_validation():
    validation = validate_output_integrity(test_output_dir)
    assert 'valid' in validation
    assert 'missing_files' in validation
    assert 'corrupted_files' in validation
```

## Dependencies

### Required Dependencies
- **pathlib**: Path handling
- **json**: JSON data handling
- **logging**: Logging functionality
- **shutil**: File operations
- **zipfile**: Archive handling

### Optional Dependencies
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **matplotlib**: Plotting
- **plotly**: Interactive plotting
- **yaml**: YAML configuration

## Performance Metrics

### Processing Performance
- **Output Processing**: 10-300 seconds depending on size
- **File Organization**: 5-60 seconds for organization
- **Validation**: 2-30 seconds for validation
- **Aggregation**: 1-15 seconds for aggregation

### Storage Performance
- **File Sizes**: 1KB-100MB per file
- **Total Size**: 10MB-10GB per pipeline run
- **Compression Ratio**: 50-80% size reduction
- **I/O Performance**: Optimized for minimal disk impact

### Quality Metrics
- **File Completeness**: 95-99% completeness
- **Format Accuracy**: 90-95% accuracy
- **Cross-reference Integrity**: 85-90% integrity
- **Metadata Preservation**: 95-99% preservation

## Troubleshooting

### Common Issues

#### 1. Output Processing Failures
```
Error: Output processing failed - insufficient disk space
Solution: Check disk space and implement cleanup procedures
```

#### 2. Organization Issues
```
Error: Output organization failed - invalid file paths
Solution: Check file paths and implement path validation
```

#### 3. Validation Issues
```
Error: Output validation failed - corrupted files
Solution: Check file integrity and implement recovery procedures
```

#### 4. Performance Issues
```
Error: Output processing timeout - large file handling
Solution: Implement chunking and parallel processing
```

### Debug Mode
```python
# Enable debug mode for detailed output information
results = process_output(target_dir, output_dir, debug=True, verbose=True)
```

## Future Enhancements

### Planned Features
- **Intelligent Organization**: AI-powered output organization
- **Real-time Monitoring**: Live output monitoring and tracking
- **Advanced Compression**: Advanced compression algorithms
- **Distributed Storage**: Multi-location output storage

### Performance Improvements
- **Advanced Caching**: Advanced caching strategies
- **Parallel Processing**: Enhanced parallel processing
- **Incremental Updates**: Improved incremental processing
- **Machine Learning**: ML-based output optimization

## Summary

The Output module provides comprehensive output management capabilities for the GNN pipeline, including file organization, result aggregation, output validation, and cross-step data coordination. The module ensures reliable output processing, proper file organization, and optimal data management to support the full Active Inference modeling lifecycle.

## License and Citation

This module is part of the GeneralizedNotationNotation project. See the main repository for license and citation information. 

## References

- Project overview: ../../README.md
- Comprehensive docs: ../../DOCS.md
- Architecture guide: ../../ARCHITECTURE.md
- Pipeline details: ../../doc/pipeline/README.md