# Template Module - Architectural Pattern Reference

This module serves as the **reference implementation** for the GNN pipeline's architectural pattern. It demonstrates the complete flow from `main.py` through numbered scripts as thin orchestrators to modular scripts in dedicated folders.

## Architectural Pattern Overview

The GNN pipeline follows a **three-tier architectural pattern**:

```
main.py → Numbered Scripts (Thin Orchestrators) → Modular Scripts in Folders
```

### 1. Main Pipeline Orchestrator (`main.py`)
- **Role**: Central pipeline coordinator
- **Responsibilities**: 
  - Parse command-line arguments
  - Execute numbered scripts in sequence
  - Track pipeline progress and results
  - Handle step filtering (`--only-steps`, `--skip-steps`)
  - Generate comprehensive pipeline summaries

### 2. Numbered Scripts (Thin Orchestrators)
- **Location**: `src/0_template.py`, `src/1_setup.py`, etc.
- **Role**: Minimal orchestrators that delegate to modules
- **Responsibilities**:
  - Import core functions from corresponding modules
  - Handle argument parsing and logging setup
  - Call modular functions with proper parameters
  - Provide fallback implementations if modules unavailable
  - Return standardized exit codes (0=success, 1=error)

### 3. Modular Scripts in Folders
- **Location**: `src/template/`, `src/setup/`, `src/validation/`, etc.
- **Role**: Core functionality implementation
- **Responsibilities**:
  - Implement domain-specific logic
  - Provide comprehensive functionality
  - Handle detailed error cases
  - Generate detailed outputs and reports

## Template Module Structure

```
src/
├── main.py                           # Main pipeline orchestrator
├── 0_template.py                     # Thin orchestrator for template step
└── template/                         # Modular template implementation
    ├── __init__.py                   # Module exports and initialization
    ├── README.md                     # This documentation
    ├── processor.py                  # Core template processing logic
    └── mcp.py                        # Model Context Protocol integration
```

## Implementation Pattern

### Step 1: Main Pipeline (`main.py`)

```python
# main.py - Central pipeline orchestrator
def main():
    """Main pipeline orchestration function."""
    args = EnhancedArgumentParser.parse_step_arguments()
    
    # Define pipeline steps
    pipeline_steps = [
        ("0_template.py", "Template initialization"),
        ("1_setup.py", "Environment setup"),
        # ... other steps
    ]
    
    # Execute each step
    for step_number, (script_name, description) in enumerate(pipeline_steps, 1):
        step_result = execute_pipeline_step(script_name, args, logger)
        # Track results and continue

def execute_pipeline_step(script_name: str, args, logger):
    """Execute a single pipeline step."""
    script_path = Path(__file__).parent / script_name
    process = subprocess.Popen([sys.executable, str(script_path), ...])
    # Return standardized result
```

### Step 2: Thin Orchestrator (`0_template.py`)

```python
# 0_template.py - Thin orchestrator
#!/usr/bin/env python3
"""
Step 0: Template Processing (Thin Orchestrator)

This step demonstrates the thin orchestrator pattern.
"""

# Import core functions from template module
try:
    from template import (
        process_template_standardized,
        generate_correlation_id,
        safe_template_execution,
        demonstrate_utility_patterns
    )
    TEMPLATE_AVAILABLE = True
except ImportError:
    TEMPLATE_AVAILABLE = False
    # Fallback function definitions if template module is not available
    def process_template_standardized(*args, **kwargs):
        return False
    # ... other fallbacks

def process_template_standardized_wrapper(
    target_dir: Path,
    output_dir: Path,
    logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """Standardized template processing function."""
    try:
        # Check if template module is available
        if not TEMPLATE_AVAILABLE:
            log_step_warning(logger, "Template module not available, using fallback functions")
        
        # Get pipeline configuration
        config = get_pipeline_config()
        step_output_dir = get_output_dir_for_script("0_template.py", output_dir)
        step_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Call modular function
        success = process_template_standardized(
            target_dir=target_dir,
            output_dir=step_output_dir,
            logger=logger,
            recursive=recursive,
            verbose=verbose,
            **kwargs
        )
        
        return success
        
    except Exception as e:
        log_step_error(logger, f"Template processing failed: {e}")
        return False

def main():
    """Main template processing function."""
    args = EnhancedArgumentParser.parse_step_arguments("0_template.py")
    logger = setup_step_logging("template", args)
    
    success = process_template_standardized_wrapper(
        target_dir=args.target_dir,
        output_dir=args.output_dir,
        logger=logger,
        recursive=args.recursive,
        verbose=args.verbose
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
```

### Step 3: Modular Implementation (`template/`)

```python
# template/__init__.py - Module exports
"""
Template Step Module

This module provides the core template processing functionality.
"""

# Export main functionality
from .processor import (
    process_template_standardized,
    process_single_file,
    validate_file,
    generate_correlation_id,
    safe_template_execution,
    demonstrate_utility_patterns
)

# Version information
VERSION_INFO = {
    "version": "1.0.0",
    "name": "Template Step",
    "description": "Standardized template for GNN pipeline steps",
    "author": "GNN Pipeline Team"
}
```

```python
# template/processor.py - Core functionality
"""
Template Step Processor

This module contains the core functionality for the template step.
"""

def process_template_standardized(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Process files in a directory using the template processor.
    
    This is the core function that implements the actual template processing logic.
    """
    try:
        # Start performance tracking
        with performance_tracker.track_operation("template_processing", {"verbose": verbose, "recursive": recursive}):
            # Update logger verbosity if needed
            if verbose:
                logger.setLevel(logging.DEBUG)
            
            # Set up output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Log processing parameters
            logger.info(f"Processing files from: {target_dir}")
            logger.info(f"Output directory: {output_dir}")
            logger.info(f"Recursive processing: {recursive}")
            
            # Find files to process
            pattern = "**/*.*" if recursive else "*.*"
            input_files = list(target_dir.glob(pattern))
            
            if not input_files:
                log_step_warning(logger, f"No files found in {target_dir}")
                return True  # Not an error, just no files to process
            
            logger.info(f"Found {len(input_files)} files to process")
            
            # Process files
            successful_files = 0
            failed_files = 0
            
            for input_file in input_files:
                try:
                    success = process_single_file(input_file, output_dir, options)
                    if success:
                        successful_files += 1
                    else:
                        failed_files += 1
                except Exception as e:
                    log_step_error(logger, f"Unexpected error processing {input_file}: {e}")
                    failed_files += 1
            
            # Generate summary report
            summary_file = output_dir / "template_processing_summary.json"
            summary = {
                "timestamp": datetime.datetime.now().isoformat(),
                "step_name": "template",
                "input_directory": str(target_dir),
                "output_directory": str(output_dir),
                "total_files": len(input_files),
                "successful_files": successful_files,
                "failed_files": failed_files,
                "performance_metrics": performance_tracker.get_summary()
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Determine success
            if failed_files == 0:
                log_step_success(logger, f"Successfully processed {successful_files} files")
                return True
            elif successful_files > 0:
                log_step_warning(logger, f"Partially successful: {failed_files} files failed")
                return True  # Still consider successful for pipeline continuation
            else:
                log_step_error(logger, "All files failed to process")
                return False
            
    except Exception as e:
        log_step_error(logger, f"Template processing failed: {e}")
        return False

def process_single_file(input_file: Path, output_dir: Path, options: Dict[str, Any]) -> bool:
    """Process a single file."""
    # Implementation of single file processing
    pass

def validate_file(input_file: Path) -> Dict[str, Any]:
    """Validate a file for processing."""
    # Implementation of file validation
    pass
```

## Key Architectural Principles

### 1. Separation of Concerns
- **Main Pipeline**: Orchestration and coordination
- **Thin Orchestrators**: Argument handling and module delegation
- **Modular Scripts**: Domain-specific logic implementation

### 2. Graceful Degradation
- Thin orchestrators provide fallback implementations
- Pipeline continues even if individual modules fail
- Comprehensive error reporting and logging

### 3. Standardized Interfaces
- All numbered scripts follow the same pattern
- Consistent function signatures across modules
- Standardized exit codes and error handling

### 4. Modular Design
- Each step has its own dedicated folder
- Clear separation between orchestration and implementation
- Easy to test, maintain, and extend

## Pipeline Execution Flow

### 1. Pipeline Start (`main.py`)
```bash
python src/main.py --target-dir input/gnn_files --output-dir output --verbose
```

### 2. Step Execution (Numbered Scripts)
```bash
# main.py calls each numbered script in sequence
python src/0_template.py --target-dir input/gnn_files --output-dir output --verbose
python src/1_setup.py --target-dir input/gnn_files --output-dir output --verbose
# ... continues for all steps
```

### 3. Module Delegation (Modular Scripts)
```python
# Each numbered script imports and calls functions from its module
from template import process_template_standardized
success = process_template_standardized(target_dir, output_dir, logger, ...)
```

## Benefits of This Architecture

### 1. Maintainability
- Clear separation between orchestration and implementation
- Easy to modify individual steps without affecting others
- Consistent patterns across all pipeline steps

### 2. Testability
- Each component can be tested independently
- Modular functions can be unit tested
- Integration tests can focus on orchestration

### 3. Extensibility
- New steps can be added by following the pattern
- Existing steps can be enhanced without breaking changes
- Modules can be reused across different contexts

### 4. Reliability
- Graceful degradation when modules are unavailable
- Comprehensive error handling and reporting
- Standardized logging and monitoring

## Template for New Pipeline Steps

### Creating a New Step

1. **Create the module folder**:
   ```
   src/new_step/
   ├── __init__.py
   ├── processor.py
   ├── mcp.py
   └── README.md
   ```

2. **Implement core functionality** in `processor.py`:
   ```python
   def process_new_step_standardized(
       target_dir: Path,
       output_dir: Path,
       logger: logging.Logger,
       recursive: bool = False,
       verbose: bool = False,
       **kwargs
   ) -> bool:
       """Standardized new step processing function."""
       # Implementation here
       pass
   ```

3. **Export functions** in `__init__.py`:
   ```python
   from .processor import process_new_step_standardized
   ```

4. **Create thin orchestrator** `src/23_new_step.py`:
   ```python
   # Import core functions from new_step module
   try:
       from new_step import process_new_step_standardized
       NEW_STEP_AVAILABLE = True
   except ImportError:
       NEW_STEP_AVAILABLE = False
       def process_new_step_standardized(*args, **kwargs):
           return False

   def process_new_step_standardized_wrapper(...):
       # Orchestration logic here
       pass

   def main():
       # Main function implementation
       pass
   ```

5. **Add to main pipeline** in `main.py`:
   ```python
   pipeline_steps = [
       # ... existing steps
       ("23_new_step.py", "New step processing"),
   ]
   ```

## Testing the Architecture

### Unit Testing
```python
# Test modular functions directly
def test_process_template_standardized():
    result = process_template_standardized(test_dir, output_dir, logger)
    assert result == True
```

### Integration Testing
```python
# Test thin orchestrator
def test_template_orchestrator():
    result = subprocess.run([sys.executable, "src/0_template.py", ...])
    assert result.returncode == 0
```

### Pipeline Testing
```python
# Test complete pipeline
def test_main_pipeline():
    result = subprocess.run([sys.executable, "src/main.py", ...])
    assert result.returncode == 0
```

## Summary

This architectural pattern provides a clear, maintainable, and extensible foundation for the GNN pipeline. The three-tier approach ensures that:

1. **Main pipeline** (`main.py`) handles orchestration and coordination
2. **Thin orchestrators** (numbered scripts) handle argument parsing and module delegation
3. **Modular scripts** (folder implementations) handle domain-specific logic

This pattern is demonstrated by the template module and should be followed by all other pipeline steps to ensure consistency and maintainability across the entire codebase.

## License and Citation

This module is part of the GeneralizedNotationNotation project. See the main repository for license and citation information. 

## References

- Project overview: ../../README.md
- Comprehensive docs: ../../DOCS.md
- Architecture guide: ../../ARCHITECTURE.md
- Pipeline details: ../../doc/pipeline/README.md