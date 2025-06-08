# GNN Processing Pipeline - Source Code

This directory contains the complete GNN (Generalized Notation Notation) processing pipeline implementation, featuring a streamlined, correlation-aware logging system and modular architecture.

## 🚀 Recent Improvements: Streamlined Logging & Utilities

### Overview
The pipeline has been comprehensively updated with a unified logging and utility system that provides:

- **Consistent logging patterns** across all 14 pipeline steps
- **Correlation-aware logging** for tracing operations across modules
- **Graceful fallback mechanisms** when enhanced utilities are unavailable
- **Standardized argument parsing** and configuration management
- **Performance tracking** and comprehensive error handling

### Key Components

#### 1. Centralized Utilities (`utils/`)
- **`logging_utils.py`**: Correlation-aware logging with performance tracking
- **`argument_utils.py`**: Streamlined argument parsing and validation
- **`dependency_validator.py`**: Comprehensive dependency validation
- **`__init__.py`**: Unified imports and fallback mechanisms

#### 2. Streamlined Import Pattern
All pipeline modules now use a consistent import pattern:

```python
# Import streamlined utilities
try:
    from utils import (
        setup_step_logging,
        log_step_start,
        log_step_success, 
        log_step_warning,
        log_step_error,
        UTILS_AVAILABLE
    )
    
    # Initialize logger for this step  
    logger = setup_step_logging("step_name", verbose=False)
    
except ImportError:
    # Graceful fallback to basic logging
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    UTILS_AVAILABLE = False
    
    # Define fallback stub functions with consistent signature
    def log_step_start(logger, message: str, **kwargs):
        logger.info(f"🚀 {message}")
    def log_step_success(logger, message: str, **kwargs):
        logger.info(f"✅ {message}")
    def log_step_warning(logger, message: str, **kwargs):
        logger.warning(f"⚠️ {message}")
    def log_step_error(logger, message: str, **kwargs):
        logger.error(f"❌ {message}")
```

#### 3. Consistent Logging Function Signatures
All logging functions now use a standardized signature:

```python
# ✅ Correct usage
log_step_start(logger, "Starting operation", metadata={"key": "value"})
log_step_success(logger, "Operation completed successfully")
log_step_warning(logger, "Non-critical issue detected")
log_step_error(logger, "Critical error occurred")

# ❌ Old inconsistent patterns (now fixed)
log_step_start("step_name", "message")  # Missing logger parameter
log_step_success("step_name", "message")  # Inconsistent signature
```

#### 4. Enhanced Verbosity Handling
Verbosity is now handled consistently across all modules:

```python
def main(parsed_args: argparse.Namespace):
    # Update logger verbosity based on args
    if UTILS_AVAILABLE and hasattr(parsed_args, 'verbose') and parsed_args.verbose:
        from utils import PipelineLogger
        PipelineLogger.set_verbosity(True)
    
    log_step_start(logger, "Starting pipeline step")
    # ... rest of function
```

### Updated Modules

#### ✅ Fully Updated
- **`main.py`**: Main pipeline orchestrator with enhanced utilities
- **`1_gnn.py`**: GNN file discovery and parsing
- **`2_setup.py`**: Project setup and environment configuration
- **`12_discopy.py`**: DisCoPy diagram generation
- **`utils/__init__.py`**: Centralized utility exports
- **`utils/dependency_validator.py`**: Fixed logger references

#### 🔄 Partially Updated
- **`13_discopy_jax_eval.py`**: Needs logging function signature updates
- **`14_site.py`**: Needs import pattern updates

#### ⏳ Pending Updates
The following modules need to be updated with the new logging pattern:
- **`3_tests.py`**: Test execution and validation
- **`4_gnn_type_checker.py`**: GNN file validation
- **`5_export.py`**: Model export functionality
- **`6_visualization.py`**: Graphical model visualization
- **`7_mcp.py`**: Model Context Protocol tasks
- **`8_ontology.py`**: Ontology processing
- **`9_render.py`**: Code generation for simulators
- **`10_execute.py`**: Execute rendered simulator scripts
- **`11_llm.py`**: LLM-enhanced analysis

### Migration Guide for Remaining Modules

To update a pipeline module to use the new streamlined logging:

1. **Replace the import section** with the standardized pattern shown above
2. **Update all logging function calls** to use the correct signature:
   ```python
   # Change from:
   log_step_start("step_name", "message")
   
   # To:
   log_step_start(logger, "message")
   ```
3. **Add verbosity handling** in the main function
4. **Remove redundant logging setup** code
5. **Test both enhanced and fallback modes**

### Benefits of the New System

#### 🔍 **Correlation Tracking**
- Each pipeline step gets a unique correlation ID
- Operations can be traced across the entire pipeline
- Enhanced debugging and monitoring capabilities

#### 📊 **Performance Monitoring**
- Automatic timing of operations
- Memory usage tracking
- Performance summaries and bottleneck identification

#### 🛡️ **Robust Error Handling**
- Graceful degradation when utilities are unavailable
- Consistent error reporting across all modules
- Structured logging with metadata support

#### 🔧 **Developer Experience**
- Consistent patterns reduce cognitive load
- Clear separation of concerns
- Comprehensive documentation and examples

### Configuration

The logging system can be configured through environment variables or programmatically:

```python
# Set global verbosity
PipelineLogger.set_verbosity(True)

# Configure correlation context
correlation_id = PipelineLogger.set_correlation_context("custom_step")

# Access performance metrics
summary = get_performance_summary()
```

### Testing

The streamlined utilities include comprehensive fallback mechanisms that ensure the pipeline continues to function even when enhanced features are unavailable. This provides:

- **Backward compatibility** with existing deployments
- **Graceful degradation** in constrained environments
- **Consistent behavior** across different execution contexts

## Pipeline Architecture

### 14-Step Processing Pipeline

1. **`1_gnn.py`** - GNN file discovery and basic parsing
2. **`2_setup.py`** - Project setup and environment configuration ⚠️ **Critical Step**
3. **`3_tests.py`** - Test execution and validation
4. **`4_gnn_type_checker.py`** - GNN file validation and resource estimation
5. **`5_export.py`** - Model export to various formats
6. **`6_visualization.py`** - Graphical model visualization
7. **`7_mcp.py`** - Model Context Protocol tasks and API integration
8. **`8_ontology.py`** - Ontology processing and validation
9. **`9_render.py`** - Code generation for simulation environments
10. **`10_execute.py`** - Execute rendered simulator scripts
11. **`11_llm.py`** - LLM-enhanced analysis and processing
12. **`12_discopy.py`** - DisCoPy categorical diagram translation
13. **`13_discopy_jax_eval.py`** - JAX-based evaluation of DisCoPy diagrams
14. **`14_site.py`** - Static site generation for documentation/reports

### Module Structure

Each pipeline module follows a consistent structure:
- **Imports**: Streamlined utilities and module-specific dependencies
- **Configuration**: Constants and default values
- **Core Functions**: Main processing logic
- **Main Function**: Argument handling and orchestration
- **CLI Support**: Standalone execution capability

### Dependencies

Core dependencies are managed through:
- **`requirements.txt`**: Production dependencies
- **`requirements-dev.txt`**: Development and testing dependencies
- **Dependency validation**: Automatic checking and reporting

## Usage

### Running the Full Pipeline
```bash
python main.py --target-dir src/gnn/examples --output-dir ../output --verbose
```

### Running Individual Steps
```bash
python 1_gnn.py --target-dir src/gnn/examples --output-dir ../output --verbose
python 2_setup.py --target-dir src/gnn/examples --output-dir ../output --recreate-venv
```

### Development Mode
```bash
python main.py --verbose --dev --only-steps "1,2,3,4"
```

## Contributing

When contributing to the pipeline:

1. **Follow the logging patterns** established in updated modules
2. **Use the streamlined utilities** for consistency
3. **Include comprehensive error handling** with appropriate logging
4. **Test both enhanced and fallback modes**
5. **Update documentation** for any new features or changes

## Future Enhancements

Planned improvements include:
- **Distributed processing** support for large-scale GNN models
- **Real-time monitoring** dashboard for pipeline execution
- **Advanced caching** mechanisms for improved performance
- **Plugin architecture** for custom processing steps
- **Integration testing** framework for end-to-end validation

## Status of Pipeline Modules

### ✅ Fully Updated Modules (Standardized Logging)
- **`main.py`**: Enhanced with streamlined utilities, improved command building with `build_enhanced_step_command_args`
- **`1_gnn.py`**: Complete overhaul - new import pattern, consistent logging calls, enhanced argument parsing with EnhancedArgumentParser
- **`2_setup.py`**: Comprehensive update - all 20+ logging function calls updated to correct signature, verbosity handling added
- **`3_tests.py`**: Updated imports, logging consistency, enhanced error handling and test result reporting
- **`4_gnn_type_checker.py`**: Complete logging standardization, improved error handling for type checker CLI integration
- **`5_export.py`**: Full logging update, enhanced export format handling and summary report generation
- **`6_visualization.py`**: Complete logging standardization, improved visualization module integration
- **`7_mcp.py`**: Full logging update, enhanced MCP integration reporting and error handling
- **`12_discopy.py`**: Updated imports, logging consistency, added translator function import handling
- **`13_discopy_jax_eval.py`**: Complete logging standardization, enhanced JAX evaluation and error handling

### 🔄 Remaining Modules (Need Updates)
- **`8_ontology.py`**: Pending logging standardization
- **`9_render.py`**: Pending logging standardization  
- **`10_execute.py`**: Pending logging standardization
- **`11_llm.py`**: Pending logging standardization
- **`14_site.py`**: Pending logging standardization

### 📊 Progress Summary
- **Completed**: 10/15 modules (67% complete)
- **Remaining**: 5/15 modules (33% remaining)
- **Critical modules**: All high-priority modules completed
- **Infrastructure**: Enhanced utilities fully operational

### 🎯 Key Achievements
- **Unified Import Pattern**: All updated modules use consistent `from utils import (...)` pattern
- **Standardized Function Signatures**: All `log_step_*` functions now use `(logger, message, **kwargs)` pattern
- **Enhanced Error Handling**: Graceful fallbacks ensure pipeline continues functioning even without enhanced utilities
- **Correlation Tracking**: Unique IDs for tracing operations across modules (where enhanced utilities available)
- **Performance Monitoring**: Automatic timing and resource tracking capabilities
- **Visual Indicators**: 🚀 ✅ ⚠️ ❌ emojis for better log readability
- **Structured Logging**: Metadata support for better observability and debugging

---

For detailed information about specific modules or advanced configuration options, refer to the individual module documentation and the comprehensive pipeline documentation in the `doc/` directory. 