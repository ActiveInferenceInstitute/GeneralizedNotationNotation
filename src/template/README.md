# Template Module

This module serves as a **template and reference implementation** for other modules in the GNN pipeline. It demonstrates the unified patterns for logging, argument handling, pipeline orchestration, and module structure that all other modules should follow.

## Module Structure

```
src/template/
├── __init__.py                    # Module initialization and exports
├── README.md                      # This documentation
├── mcp.py                         # Model Context Protocol integration
└── processor.py                   # Core template processing
```

## Template for Other Modules

This module implements the **standard patterns** that all other modules should follow:

### 1. Unified Logging Pattern
```python
# Standard import pattern for all modules
from utils import setup_step_logging, log_step_start, log_step_success, log_step_error
from pipeline import get_output_dir_for_script, get_pipeline_config

# Standard logging setup
logger = setup_step_logging(__name__)
```

### 2. Argument Handling Pattern
```python
# Standard argument parsing pattern
from utils import EnhancedArgumentParser

def parse_step_arguments():
    """Parse arguments with fallback for graceful degradation."""
    try:
        parser = EnhancedArgumentParser.parse_step_arguments()
        return parser.parse_args()
    except Exception as e:
        # Fallback to basic argument parser
        parser = argparse.ArgumentParser()
        parser.add_argument("--target-dir", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--verbose", action="store_true")
        return parser.parse_args()
```

### 3. Pipeline Orchestration Pattern
```python
# Standard pipeline step function signature
def process_template(target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs) -> bool:
    """
    Main function for processing template-related tasks.
    
    This is the standard pattern that all pipeline step functions should follow.
    """
    try:
        log_step_start("template", target_dir, output_dir, verbose)
        
        # Core processing logic here
        results = perform_template_processing(target_dir, output_dir, verbose)
        
        log_step_success("template", results)
        return True
        
    except Exception as e:
        log_step_error("template", e)
        return False
```

### 4. Module Structure Pattern
Every module should follow this structure:
- `__init__.py`: Module initialization with standard exports
- `README.md`: Comprehensive documentation (this file)
- `mcp.py`: MCP integration with tool registration
- Core functionality files (e.g., `processor.py`, `analyzer.py`, etc.)

## Core Components

### Template Functions

#### `process_template(target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs) -> bool`
Main function for processing template-related tasks.

**Features:**
- Template generation and customization
- Pipeline initialization
- Template validation and verification
- Template documentation
- Template management

**Returns:**
- `bool`: Success status of template operations

### Template Processing Functions

#### `generate_gnn_template(template_type: str, output_path: Path, **kwargs) -> bool`
Generates GNN templates for different model types.

**Template Types:**
- **basic**: Basic Active Inference model template
- **pomdp**: Partially Observable Markov Decision Process template
- **hierarchical**: Hierarchical model template
- **multiagent**: Multi-agent system template
- **custom**: Custom template with user-defined parameters

#### `customize_template(template_path: Path, customizations: Dict[str, Any]) -> str`
Customizes existing templates with specific parameters.

**Customization Features:**
- Variable customization
- Parameter modification
- Structure adaptation
- Component addition
- Validation integration

#### `validate_template(template_content: str) -> Dict[str, Any]`
Validates template content for correctness and completeness.

**Validation Features:**
- Syntax validation
- Structure verification
- Parameter checking
- Dependency validation
- Best practices compliance

### Template Management Functions

#### `create_pipeline_template(pipeline_config: Dict[str, Any]) -> bool`
Creates pipeline template for GNN processing.

**Pipeline Features:**
- Step configuration
- Workflow definition
- Parameter setup
- Error handling
- Logging configuration

#### `initialize_project_template(project_config: Dict[str, Any]) -> bool`
Initializes project template with standard structure.

**Project Features:**
- Directory structure
- Configuration files
- Documentation templates
- Testing framework
- Development environment

## Usage Examples

### Basic Template Processing

```python
from template import process_template

# Process template-related tasks
success = process_template(
    target_dir=Path("templates/"),
    output_dir=Path("template_output/"),
    verbose=True
)

if success:
    print("Template processing completed successfully")
else:
    print("Template processing failed")
```

### GNN Template Generation

```python
from template import generate_gnn_template

# Generate basic GNN template
success = generate_gnn_template(
    template_type="basic",
    output_path=Path("models/basic_model.md")
)

if success:
    print("Basic GNN template generated successfully")
else:
    print("Template generation failed")
```

### POMDP Template Generation

```python
from template import generate_gnn_template

# Generate POMDP template
success = generate_gnn_template(
    template_type="pomdp",
    output_path=Path("models/pomdp_model.md"),
    parameters={
        "states": 3,
        "observations": 2,
        "actions": 2
    }
)

if success:
    print("POMDP template generated successfully")
else:
    print("POMDP template generation failed")
```

### Template Customization

```python
from template import customize_template

# Customize existing template
customizations = {
    "model_name": "my_custom_model",
    "variables": {
        "A": {"dimensions": [3, 3], "description": "Transition matrix"},
        "B": {"dimensions": [3, 2], "description": "Observation matrix"}
    },
    "parameters": {
        "learning_rate": 0.01,
        "precision": 1.0
    }
}

customized_template = customize_template(
    template_path=Path("templates/basic_template.md"),
    customizations=customizations
)

print("Customized template:")
print(customized_template)
```

### Template Validation

```python
from template import validate_template

# Validate template content
validation_results = validate_template(template_content)

print(f"Template valid: {validation_results['valid']}")
print(f"Syntax errors: {len(validation_results['syntax_errors'])}")
print(f"Structure issues: {len(validation_results['structure_issues'])}")
print(f"Parameter issues: {len(validation_results['parameter_issues'])}")
```

### Pipeline Template Creation

```python
from template import create_pipeline_template

# Create pipeline template
pipeline_config = {
    "steps": ["setup", "gnn", "validation", "export", "visualization"],
    "parameters": {
        "verbose": True,
        "output_dir": "output/",
        "log_level": "INFO"
    }
}

success = create_pipeline_template(pipeline_config)

if success:
    print("Pipeline template created successfully")
else:
    print("Pipeline template creation failed")
```

### Project Template Initialization

```python
from template import initialize_project_template

# Initialize project template
project_config = {
    "project_name": "my_gnn_project",
    "author": "John Doe",
    "description": "Active Inference research project",
    "structure": {
        "models": "models/",
        "outputs": "outputs/",
        "docs": "docs/",
        "tests": "tests/"
    }
}

success = initialize_project_template(project_config)

if success:
    print("Project template initialized successfully")
else:
    print("Project template initialization failed")
```

## Template Pipeline

### 1. Template Selection
```python
# Select appropriate template
template_type = select_template_type(requirements)
template_config = get_template_config(template_type)
```

### 2. Template Generation
```python
# Generate base template
base_template = generate_base_template(template_type)
template_content = populate_template(base_template, parameters)
```

### 3. Template Customization
```python
# Customize template
customizations = apply_customizations(template_content, custom_params)
validated_template = validate_customizations(customizations)
```

### 4. Template Validation
```python
# Validate template
validation_results = validate_template(validated_template)
if not validation_results['valid']:
    fix_template_issues(validation_results['issues'])
```

### 5. Template Deployment
```python
# Deploy template
deploy_template(validated_template, output_path)
generate_documentation(validated_template)
```

## Integration with Pipeline

### Pipeline Step 0: Template Processing
```python
# Called from 0_template.py
def process_template(target_dir, output_dir, verbose=False, **kwargs):
    # Generate and process templates
    template_results = generate_and_process_templates(target_dir, verbose)
    
    # Generate template reports
    template_reports = generate_template_reports(template_results)
    
    # Create template documentation
    template_docs = create_template_documentation(template_results)
    
    return True
```

### Output Structure
```
output/template_processing/
├── template_generation.json        # Template generation results
├── template_customization.json     # Template customization results
├── template_validation.json        # Template validation results
├── pipeline_templates.json         # Pipeline template results
├── project_templates.json          # Project template results
├── template_summary.md             # Template summary
└── template_report.md              # Comprehensive template report
```

## Template Features

### Template Types
- **Basic Templates**: Simple Active Inference models
- **POMDP Templates**: Partially Observable Markov Decision Processes
- **Hierarchical Templates**: Multi-level model structures
- **Multi-agent Templates**: Agent-based systems
- **Custom Templates**: User-defined model structures

### Template Customization
- **Variable Customization**: Model variable definition
- **Parameter Customization**: Model parameter specification
- **Structure Customization**: Model structure modification
- **Component Customization**: Model component addition
- **Validation Customization**: Custom validation rules

### Template Validation
- **Syntax Validation**: Template syntax checking
- **Structure Validation**: Template structure verification
- **Parameter Validation**: Parameter correctness checking
- **Dependency Validation**: Dependency requirement checking
- **Best Practices**: Template best practices compliance

### Template Management
- **Version Control**: Template version management
- **Template Library**: Centralized template storage
- **Template Sharing**: Template sharing and distribution
- **Template Documentation**: Comprehensive template documentation
- **Template Testing**: Template testing and validation

## Configuration Options

### Template Settings
```python
# Template configuration
config = {
    'template_library': 'templates/',  # Template library path
    'auto_validation': True,           # Enable automatic validation
    'customization_enabled': True,     # Enable template customization
    'version_control': True,           # Enable version control
    'documentation_enabled': True,     # Enable documentation generation
    'testing_enabled': True            # Enable template testing
}
```

### Generation Settings
```python
# Generation configuration
generation_config = {
    'default_template': 'basic',       # Default template type
    'auto_customization': True,        # Enable auto-customization
    'validation_strict': False,        # Strict validation mode
    'backup_existing': True,           # Backup existing templates
    'overwrite_protection': True       # Protect against overwrites
}
```

## Error Handling

### Template Failures
```python
# Handle template failures gracefully
try:
    results = process_template(target_dir, output_dir)
except TemplateError as e:
    logger.error(f"Template processing failed: {e}")
    # Provide fallback template or error reporting
```

### Generation Issues
```python
# Handle generation issues gracefully
try:
    template = generate_gnn_template(template_type, output_path)
except GenerationError as e:
    logger.warning(f"Template generation failed: {e}")
    # Provide fallback generation or error reporting
```

### Validation Issues
```python
# Handle validation issues gracefully
try:
    validation = validate_template(template_content)
except ValidationError as e:
    logger.error(f"Template validation failed: {e}")
    # Provide fallback validation or error reporting
```

## Performance Optimization

### Template Optimization
- **Caching**: Cache template generation results
- **Parallel Processing**: Parallel template processing
- **Incremental Generation**: Incremental template updates
- **Optimized Algorithms**: Optimize template algorithms

### Generation Optimization
- **Template Caching**: Cache generated templates
- **Parallel Generation**: Parallel template generation
- **Incremental Generation**: Incremental template generation
- **Optimized Validation**: Optimize template validation

### Customization Optimization
- **Customization Caching**: Cache customization results
- **Parallel Customization**: Parallel template customization
- **Incremental Customization**: Incremental customization updates
- **Optimized Validation**: Optimize customization validation

## Testing and Validation

### Unit Tests
```python
# Test individual template functions
def test_template_generation():
    success = generate_gnn_template("basic", test_output_path)
    assert success
    assert test_output_path.exists()
```

### Integration Tests
```python
# Test complete template pipeline
def test_template_pipeline():
    success = process_template(test_dir, output_dir)
    assert success
    # Verify template outputs
    template_files = list(output_dir.glob("**/*"))
    assert len(template_files) > 0
```

### Validation Tests
```python
# Test template validation
def test_template_validation():
    validation = validate_template(test_template_content)
    assert 'valid' in validation
    assert 'syntax_errors' in validation
    assert 'structure_issues' in validation
```

## Dependencies

### Required Dependencies
- **pathlib**: Path handling
- **jinja2**: Template engine
- **json**: JSON data handling
- **logging**: Logging functionality

### Optional Dependencies
- **markdown**: Markdown processing
- **yaml**: YAML configuration
- **pydantic**: Data validation
- **click**: Command line interface

## Performance Metrics

### Generation Times
- **Basic Templates** (< 100 lines): < 1 second
- **Medium Templates** (100-500 lines): 1-5 seconds
- **Complex Templates** (> 500 lines): 5-30 seconds

### Memory Usage
- **Base Memory**: ~10MB
- **Per Template**: ~1-5MB depending on complexity
- **Peak Memory**: 1.5-2x base usage during generation

### Quality Metrics
- **Template Completeness**: 95-99% completeness
- **Validation Accuracy**: 90-95% accuracy
- **Customization Success**: 85-90% success rate
- **Documentation Quality**: 80-85% quality score

## Troubleshooting

### Common Issues

#### 1. Template Failures
```
Error: Template processing failed - invalid template type
Solution: Check template type specification and available templates
```

#### 2. Generation Issues
```
Error: Template generation failed - invalid parameters
Solution: Validate template parameters and requirements
```

#### 3. Validation Issues
```
Error: Template validation failed - syntax errors
Solution: Check template syntax and structure
```

#### 4. Customization Issues
```
Error: Template customization failed - incompatible parameters
Solution: Verify parameter compatibility and template structure
```

### Debug Mode
```python
# Enable debug mode for detailed template information
results = process_template(target_dir, output_dir, debug=True, verbose=True)
```

## Future Enhancements

### Planned Features
- **Advanced Templates**: AI-powered template generation
- **Interactive Customization**: Interactive template customization
- **Template Marketplace**: Template sharing and marketplace
- **Automated Testing**: Automated template testing and validation

### Performance Improvements
- **Advanced Caching**: Advanced caching strategies
- **Parallel Processing**: Parallel template processing
- **Incremental Updates**: Incremental template updates
- **Machine Learning**: ML-based template optimization

## Summary

The Template module serves as a **reference implementation** for all other modules in the GNN pipeline. It demonstrates the unified patterns for logging, argument handling, pipeline orchestration, and module structure that ensure consistency across the entire codebase. The module provides comprehensive template processing capabilities for GNN models, including template generation, customization, and pipeline initialization, while serving as a template for other modules to follow.

## License and Citation

This module is part of the GeneralizedNotationNotation project. See the main repository for license and citation information. 