# Model Registry Module

This module provides comprehensive model registry capabilities for GNN models, including versioning, metadata management, model discovery, and lifecycle management.

## Module Structure

```
src/model_registry/
├── __init__.py                    # Module initialization and exports
├── README.md                      # This documentation
├── mcp.py                         # Model Context Protocol integration
└── registry.py                    # Core registry functionality
```

## Core Components

### Model Registry Functions

#### `process_model_registry(target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs) -> bool`
Main function for processing model registry tasks.

**Features:**
- Model registration and versioning
- Metadata management and indexing
- Model discovery and search
- Lifecycle management
- Registry maintenance and cleanup

**Returns:**
- `bool`: Success status of registry operations

### Registry Management Functions

#### `register_model(model_path: Path, metadata: Dict[str, Any] = None) -> str`
Registers a GNN model in the registry with metadata.

**Features:**
- Automatic model ID generation
- Metadata extraction and validation
- Version tracking
- Dependency management
- Registry indexing

#### `get_model_info(model_id: str) -> Dict[str, Any]`
Retrieves comprehensive information about a registered model.

**Information:**
- Model metadata and properties
- Version history
- Dependencies and requirements
- Performance metrics
- Usage statistics

#### `search_models(query: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]`
Searches for models in the registry based on query and filters.

**Search Capabilities:**
- Text-based search
- Metadata filtering
- Version filtering
- Performance filtering
- Tag-based search

#### `update_model_metadata(model_id: str, metadata: Dict[str, Any]) -> bool`
Updates metadata for a registered model.

**Update Features:**
- Metadata validation
- Version tracking
- Change history
- Dependency updates
- Performance updates

### Version Management

#### `create_model_version(model_id: str, version: str = None) -> str`
Creates a new version of a registered model.

**Version Features:**
- Semantic versioning support
- Automatic version numbering
- Change tracking
- Dependency updates
- Migration support

#### `get_model_versions(model_id: str) -> List[Dict[str, Any]]`
Retrieves version history for a model.

**Version Information:**
- Version numbers and timestamps
- Change descriptions
- Performance comparisons
- Dependency changes
- Migration notes

#### `compare_model_versions(model_id: str, version1: str, version2: str) -> Dict[str, Any]`
Compares two versions of a model.

**Comparison Features:**
- Structural differences
- Performance differences
- Dependency changes
- Metadata changes
- Migration requirements

### Lifecycle Management

#### `deprecate_model(model_id: str, reason: str = None) -> bool`
Marks a model as deprecated.

**Deprecation Features:**
- Deprecation notice
- Migration guidance
- Alternative suggestions
- Timeline management
- Cleanup scheduling

#### `archive_model(model_id: str, reason: str = None) -> bool`
Archives a model in the registry.

**Archive Features:**
- Archive metadata
- Storage optimization
- Access control
- Recovery options
- Cleanup procedures

#### `delete_model(model_id: str, force: bool = False) -> bool`
Deletes a model from the registry.

**Deletion Features:**
- Dependency checking
- Backup creation
- Cleanup procedures
- Audit trail
- Recovery options

## Usage Examples

### Basic Model Registration

```python
from model_registry import register_model

# Register a GNN model
model_id = register_model(
    model_path=Path("models/my_model.md"),
    metadata={
        "name": "My Active Inference Model",
        "description": "A comprehensive Active Inference model",
        "author": "John Doe",
        "tags": ["active-inference", "pomdp", "research"]
    }
)

print(f"Model registered with ID: {model_id}")
```

### Model Information Retrieval

```python
from model_registry import get_model_info

# Get comprehensive model information
model_info = get_model_info("model_123")

print(f"Model name: {model_info['name']}")
print(f"Current version: {model_info['current_version']}")
print(f"Total versions: {len(model_info['versions'])}")
print(f"Performance metrics: {model_info['performance']}")
```

### Model Search

```python
from model_registry import search_models

# Search for models
results = search_models(
    query="active inference",
    filters={
        "tags": ["pomdp"],
        "min_performance": 0.8,
        "author": "John Doe"
    }
)

for model in results:
    print(f"Found model: {model['name']} (ID: {model['id']})")
```

### Version Management

```python
from model_registry import create_model_version, compare_model_versions

# Create new version
new_version = create_model_version(
    model_id="model_123",
    version="2.0.0"
)

# Compare versions
comparison = compare_model_versions(
    model_id="model_123",
    version1="1.0.0",
    version2="2.0.0"
)

print(f"Performance improvement: {comparison['performance_delta']}")
print(f"Structural changes: {len(comparison['structural_changes'])}")
```

### Lifecycle Management

```python
from model_registry import deprecate_model, archive_model

# Deprecate a model
deprecate_model(
    model_id="model_123",
    reason="Superseded by improved version"
)

# Archive a model
archive_model(
    model_id="model_456",
    reason="No longer maintained"
)
```

## Registry Pipeline

### 1. Model Discovery
```python
# Discover models in target directory
models = discover_models(target_dir)
model_paths = [model['path'] for model in models]
```

### 2. Metadata Extraction
```python
# Extract metadata from models
for model_path in model_paths:
    metadata = extract_model_metadata(model_path)
    validate_metadata(metadata)
```

### 3. Model Registration
```python
# Register models with metadata
for model_path, metadata in zip(model_paths, metadata_list):
    model_id = register_model(model_path, metadata)
    index_model(model_id, metadata)
```

### 4. Registry Maintenance
```python
# Maintain registry integrity
cleanup_registry()
update_indexes()
validate_registry()
```

### 5. Registry Reporting
```python
# Generate registry reports
registry_stats = generate_registry_statistics()
registry_report = create_registry_report(registry_stats)
```

## Integration with Pipeline

### Pipeline Step 4: Model Registry
```python
# Called from 4_model_registry.py
def process_model_registry(target_dir, output_dir, verbose=False, **kwargs):
    # Discover and register models
    registered_models = register_discovered_models(target_dir)
    
    # Generate registry reports
    registry_stats = generate_registry_statistics(registered_models)
    
    # Create registry documentation
    registry_docs = create_registry_documentation(registered_models)
    
    return True
```

### Output Structure
```
output/model_registry/
├── registry.json                   # Complete registry data
├── model_index.json               # Searchable model index
├── version_history.json           # Version history data
├── metadata_index.json            # Metadata index
├── registry_statistics.json       # Registry statistics
├── registry_report.md             # Registry report
└── registry_summary.md            # Registry summary
```

## Registry Features

### Model Discovery
- **Automatic Discovery**: Automatically discover GNN models
- **Metadata Extraction**: Extract metadata from model files
- **Validation**: Validate model structure and metadata
- **Indexing**: Create searchable indexes

### Version Control
- **Semantic Versioning**: Support for semantic versioning
- **Change Tracking**: Track changes between versions
- **Migration Support**: Support for model migrations
- **Rollback Capability**: Rollback to previous versions

### Search and Discovery
- **Text Search**: Full-text search across model metadata
- **Filtered Search**: Search with multiple filters
- **Tag-based Search**: Search by model tags
- **Performance Search**: Search by performance metrics

### Lifecycle Management
- **Model States**: Active, deprecated, archived, deleted
- **State Transitions**: Controlled state transitions
- **Audit Trail**: Complete audit trail for all changes
- **Cleanup Procedures**: Automated cleanup procedures

## Configuration Options

### Registry Settings
```python
# Registry configuration
config = {
    'registry_path': 'models/registry',  # Registry storage path
    'index_enabled': True,              # Enable search indexing
    'versioning_enabled': True,         # Enable version control
    'backup_enabled': True,             # Enable automatic backups
    'cleanup_enabled': True,            # Enable automatic cleanup
    'max_versions': 10,                 # Maximum versions per model
    'archive_after_days': 365           # Archive models after days
}
```

### Metadata Settings
```python
# Metadata configuration
metadata_config = {
    'required_fields': ['name', 'description', 'author'],
    'optional_fields': ['tags', 'performance', 'dependencies'],
    'validation_rules': {
        'name': 'min_length:3',
        'description': 'min_length:10',
        'performance': 'range:0-1'
    },
    'auto_extraction': True,            # Auto-extract metadata
    'validation_enabled': True          # Enable metadata validation
}
```

## Error Handling

### Registration Failures
```python
# Handle registration failures gracefully
try:
    model_id = register_model(model_path, metadata)
except RegistrationError as e:
    logger.error(f"Model registration failed: {e}")
    # Provide fallback registration or error reporting
```

### Search Failures
```python
# Handle search failures gracefully
try:
    results = search_models(query, filters)
except SearchError as e:
    logger.warning(f"Search failed: {e}")
    # Provide fallback search or error reporting
```

### Version Management Issues
```python
# Handle version management issues
try:
    new_version = create_model_version(model_id, version)
except VersionError as e:
    logger.error(f"Version creation failed: {e}")
    # Provide fallback versioning or error reporting
```

## Performance Optimization

### Indexing Optimization
- **Incremental Indexing**: Update indexes incrementally
- **Background Indexing**: Index in background processes
- **Index Compression**: Compress indexes for storage efficiency
- **Search Optimization**: Optimize search algorithms

### Storage Optimization
- **Metadata Compression**: Compress metadata storage
- **Version Storage**: Efficient version storage
- **Cleanup Procedures**: Automated cleanup procedures
- **Backup Optimization**: Optimize backup procedures

### Search Optimization
- **Full-text Search**: Optimize full-text search
- **Filtered Search**: Optimize filtered search
- **Caching**: Cache search results
- **Pagination**: Efficient pagination for large result sets

## Testing and Validation

### Unit Tests
```python
# Test individual registry functions
def test_model_registration():
    model_id = register_model(test_model_path, test_metadata)
    assert model_id is not None
    model_info = get_model_info(model_id)
    assert model_info['name'] == test_metadata['name']
```

### Integration Tests
```python
# Test complete registry pipeline
def test_registry_pipeline():
    success = process_model_registry(test_dir, output_dir)
    assert success
    # Verify registry outputs
    registry_files = list(output_dir.glob("**/*"))
    assert len(registry_files) > 0
```

### Performance Tests
```python
# Test registry performance
def test_registry_performance():
    start_time = time.time()
    results = search_models("test query")
    end_time = time.time()
    
    assert len(results) > 0
    assert (end_time - start_time) < 1.0  # Should complete within 1 second
```

## Dependencies

### Required Dependencies
- **pathlib**: Path handling
- **json**: JSON data handling
- **sqlite3**: Database storage (optional)
- **hashlib**: Hash generation for model IDs

### Optional Dependencies
- **elasticsearch**: Advanced search capabilities
- **redis**: Caching and session management
- **sqlalchemy**: Database ORM
- **whoosh**: Full-text search engine

## Performance Metrics

### Processing Times
- **Model Registration**: < 1 second per model
- **Metadata Extraction**: < 0.5 seconds per model
- **Search Operations**: < 0.1 seconds per query
- **Version Creation**: < 0.5 seconds per version

### Memory Usage
- **Base Memory**: ~20MB
- **Per Model**: ~1-5MB depending on metadata size
- **Peak Memory**: 1.5-2x base usage during operations

### Storage Requirements
- **Metadata Storage**: ~1-10KB per model
- **Index Storage**: ~5-50KB per model
- **Version Storage**: ~1-5KB per version
- **Backup Storage**: 2-3x base storage

## Troubleshooting

### Common Issues

#### 1. Registration Failures
```
Error: Model registration failed - invalid metadata
Solution: Validate metadata format and required fields
```

#### 2. Search Issues
```
Error: Search operation failed - index corruption
Solution: Rebuild search indexes or restore from backup
```

#### 3. Version Management Issues
```
Error: Version creation failed - dependency conflict
Solution: Resolve dependency conflicts or use force option
```

#### 4. Performance Issues
```
Error: Registry operations taking too long
Solution: Optimize indexes or increase system resources
```

### Debug Mode
```python
# Enable debug mode for detailed registry information
results = register_model(model_path, metadata, debug=True, verbose=True)
```

## Future Enhancements

### Planned Features
- **Distributed Registry**: Distributed registry across multiple nodes
- **Advanced Search**: Advanced search with ML-based ranking
- **Model Marketplace**: Model marketplace and sharing capabilities
- **Automated Testing**: Automated model testing and validation

### Performance Improvements
- **Advanced Indexing**: Advanced indexing strategies
- **Distributed Storage**: Distributed storage capabilities
- **Real-time Updates**: Real-time registry updates
- **Advanced Caching**: Advanced caching strategies

## Summary

The Model Registry module provides comprehensive model registry capabilities for GNN models, including versioning, metadata management, model discovery, and lifecycle management. The module ensures efficient model organization, search capabilities, and lifecycle management to support Active Inference research and development.

## License and Citation

This module is part of the GeneralizedNotationNotation project. See the main repository for license and citation information. 

## References

- Project overview: ../../README.md
- Comprehensive docs: ../../DOCS.md
- Architecture guide: ../../ARCHITECTURE.md
- Pipeline details: ../../doc/pipeline/README.md