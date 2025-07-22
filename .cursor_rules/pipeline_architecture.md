# Pipeline Architecture & Implementation Paradigm

### Core Pipeline Orchestration
- **Main Orchestrator**: `src/main.py` dynamically discovers and executes numbered pipeline scripts (1-13)
- **Script Discovery**: Uses glob patterns to find `*_*.py` files, sorts by number and basename
- **Execution Model**: Each script runs as subprocess with centralized argument passing
- **Virtual Environment**: Automatic detection and use of project virtual environment
- **Dependency Validation**: Comprehensive dependency checking before pipeline execution
- **Performance Tracking**: Built-in performance monitoring and resource usage tracking
- **Configuration Management**: YAML-based configuration with CLI argument overrides

### Pipeline Steps (13 Steps - Current Order)
Each numbered script corresponds to a specific module folder and implements real functionality:

1. **1_setup.py** → `src/setup/` - Environment setup, virtual environment management, dependency installation (CRITICAL STEP)
2. **2_gnn.py** → `src/gnn/` - GNN file discovery, multi-format parsing, and comprehensive validation
3. **3_tests.py** → `src/tests/` - Test suite execution with pytest integration and coverage reporting
4. **4_type_checker.py** → `src/type_checker/` - GNN syntax validation, type checking, and resource estimation
5. **5_export.py** → `src/export/` - Multi-format export (JSON, XML, GraphML, GEXF, Pickle, PKL)
6. **6_visualization.py** → `src/visualization/` - Graph visualization, matrix visualization, and ontology diagrams
7. **7_mcp.py** → `src/mcp/` - Model Context Protocol operations, tool registration, and MCP system integration
8. **8_ontology.py** → `src/ontology/` - Active Inference Ontology processing, validation, and semantic mapping
9. **9_render.py** → `src/render/` - Code generation for PyMDP, RxInfer.jl, ActiveInference.jl, JAX, and DisCoPy environments
10. **10_execute.py** → `src/execute/` - Execute rendered simulation scripts with comprehensive result capture
11. **11_llm.py** → `src/llm/` - LLM-enhanced analysis, model interpretation, natural language explanations
12. **12_website.py** → `src/website/` - Static HTML website generation from pipeline artifacts with interactive elements
13. **13_sapf.py** → `src/sapf/` - SAPF (Sound As Pure Form) audio generation and model sonification

### Centralized Infrastructure

#### Utilities (`src/utils/`)
- **EnhancedArgumentParser**: Centralized argument parsing with step-specific configuration and fallback handling
- **PipelineLogger**: Structured logging with correlation contexts, step tracking, and performance monitoring
- **performance_tracker**: Performance monitoring, resource usage tracking, and execution profiling
- **validate_pipeline_dependencies**: Comprehensive dependency validation with detailed reporting
- **setup_step_logging**: Standardized logging setup with dynamic verbosity control for each pipeline step
- **config_loader**: YAML configuration loading with validation and CLI argument override support

#### Pipeline Configuration (`src/pipeline/`)
- **STEP_METADATA**: Centralized metadata for all 13 pipeline steps with dependencies, timeouts, and requirements
- **get_pipeline_config**: Pipeline configuration management with step-specific settings
- **get_output_dir_for_script**: Standardized output directory structure with automatic creation
- **execute_pipeline_step**: Centralized step execution with enhanced error handling and performance tracking
- **StepExecutionResult**: Comprehensive execution result tracking with metrics and status reporting

#### Module Structure Pattern
Each module follows this structure:
```
src/[module_name]/
├── __init__.py          # Module initialization with version and feature flags
├── [main_module].py     # Core functionality with comprehensive implementation
├── mcp.py              # MCP integration with functional tool registration (if applicable)
└── [additional_files]   # Specialized components and utilities
```

### Implementation Standards

#### Script Structure Pattern
Every numbered script follows this standardized pattern:
1. **Import centralized utilities** (`utils`, `pipeline`) with graceful fallback handling
2. **Initialize step-specific logger** via `setup_step_logging` with correlation context
3. **Import module-specific functionality** from corresponding folder with dependency validation
4. **Define main() function** with centralized argument handling using `EnhancedArgumentParser`
5. **Use EnhancedArgumentParser.parse_step_arguments()** for argument parsing with fallback parser
6. **Implement comprehensive error handling** with step logging functions and proper exit codes
7. **Return appropriate exit codes** (0=success, 1=critical error, 2=success with warnings)

#### Argument Parsing Standards
- **Centralized Parsing**: Use `EnhancedArgumentParser.parse_step_arguments()` for all pipeline steps with step-specific configuration
- **Fallback Handling**: Always provide fallback argument parser for graceful degradation when utilities unavailable
- **Common Arguments**: All steps support `--target-dir`, `--output-dir`, `--verbose`, `--recursive` with consistent behavior
- **Step-Specific Arguments**: Additional arguments based on step functionality (e.g., `--strict`, `--duration`, `--estimate-resources`)
- **Path Conversion**: Defensive conversion of string arguments to `pathlib.Path` objects with validation
- **Configuration Override**: CLI arguments override YAML configuration settings

#### Logging Standards
- **Structured Logging**: Use `log_step_start`, `log_step_success`, `log_step_warning`, `log_step_error` with correlation IDs
- **Correlation Context**: Each step sets unique correlation context for end-to-end traceability
- **Dynamic Verbosity**: Verbosity adjustment based on arguments and configuration
- **Performance Integration**: Use `performance_tracker.track_operation()` for timed operations and resource monitoring
- **Event-Driven Logging**: Structured logging with event types for pipeline monitoring and analysis

#### Error Handling Standards
- **Graceful Degradation**: Steps continue pipeline unless critical failure occurs
- **Critical vs Non-Critical**: Steps marked as `required=True` in configuration halt pipeline on failure
- **Comprehensive Reporting**: All errors logged with context, suggestions, and actionable information
- **Exit Code Conventions**: 0=success, 1=critical error (stops pipeline), 2=success with warnings (continues pipeline)
- **Dependency Validation**: Pre-flight dependency checks with clear error messages and installation guidance

### Module-Specific Implementation Details

#### Setup Management (`src/setup/`) - Step 1
- **Virtual Environment**: Detection, creation, and management with Python version validation
- **Dependency Installation**: Real pip installation with requirements.txt and requirements-dev.txt support
- **System Validation**: Comprehensive system dependency checking with detailed environment reporting
- **Critical Step**: Pipeline execution depends on successful setup completion

#### GNN Processing (`src/gnn/`) - Step 2  
- **Multi-Format Support**: Comprehensive parsing for Markdown, JSON, YAML, XML, Binary (PKL), Protobuf, Maxima, and more
- **Validation Levels**: BASIC, STANDARD, STRICT, RESEARCH, and ROUND_TRIP validation with configurable depth
- **Round-Trip Testing**: Semantic preservation testing across format conversions
- **Cross-Format Validation**: Consistency checking across different format representations
- **Real Implementation**: Full GNN specification parsing with sophisticated error reporting and recovery

#### Type Checking (`src/type_checker/`) - Step 4
- **Syntax Validation**: Complete GNN syntax validation with detailed error reporting and suggestions
- **Resource Estimation**: Computational resource estimation for model execution planning
- **Type Consistency**: Variable type checking, dimension validation, and Active Inference conformance
- **Performance Analysis**: Complexity analysis and optimization recommendations

#### Export Systems (`src/export/`) - Step 5
- **Multi-Format Support**: JSON, XML, GraphML, GEXF, Pickle, PKL formats with format-specific optimizations
- **Network Graph Export**: NetworkX-based graph format exports with rich metadata preservation
- **Structured Data**: Comprehensive model metadata preservation across format boundaries
- **Round-Trip Compatibility**: Format exports designed for semantic preservation in round-trip scenarios

#### Visualization (`src/visualization/`) - Step 6
- **Graph Visualization**: Model structure visualization with matplotlib, networkx, and interactive elements
- **Matrix Visualization**: A, B, C, D matrix heatmaps, correlation matrices, and statistical visualizations
- **Ontology Visualization**: Active Inference ontology relationship diagrams with semantic annotations
- **Interactive Elements**: Support for interactive visualization with export to multiple formats

#### Simulation Rendering (`src/render/`) - Step 9
- **PyMDP Integration**: Complete PyMDP agent and environment code generation with Active Inference semantics
- **RxInfer Integration**: Full RxInfer.jl model translation with Bayesian inference implementation
- **ActiveInference.jl**: Native Julia implementation generation for high-performance simulation
- **Template System**: Comprehensive template-based code generation with customizable patterns
- **Real Code Generation**: Executable simulation code with proper imports, initialization, and execution logic

#### Execution Engine (`src/execute/`) - Step 10
- **Multi-Backend Support**: PyMDP, RxInfer.jl, ActiveInference.jl, JAX, and custom execution backends
- **Script Execution**: Real Python and Julia script execution with subprocess management and result capture
- **Result Analysis**: Output processing, error handling, and comprehensive result reporting
- **Performance Monitoring**: Execution time tracking, memory usage, and convergence analysis

#### LLM Integration (`src/llm/`) - Step 11
- **Model Analysis**: AI-powered GNN model interpretation, validation, and enhancement suggestions
- **Natural Language**: GNN to natural language explanation generation with technical accuracy
- **Enhancement Recommendations**: Automated model improvement suggestions based on best practices
- **Real AI Integration**: Functional LLM processing with multiple provider support and robust error handling

#### Website Generation (`src/website/`) - Step 12
- **HTML Generation**: Static HTML site generation from pipeline artifacts with modern responsive design
- **Report Aggregation**: Comprehensive pipeline summary and results presentation with interactive navigation
- **Interactive Elements**: Dynamic content, visualization embedding, and user-friendly interfaces
- **Multi-Format Output**: Support for different website templates and customization options

#### SAPF Audio Generation (`src/sapf/`) - Step 13
- **Sound As Pure Form**: Advanced audio representation and sonification of mathematical GNN models
- **Model Sonification**: Converting mathematical structures, matrices, and relationships to audio patterns
- **Audio Synthesis**: Real-time audio generation, processing, and export with multiple format support
- **Scientific Sonification**: Mathematically grounded audio representations for model analysis and understanding

### Performance and Quality Standards
- **Real-Time Monitoring**: Comprehensive performance tracking across all pipeline steps
- **Resource Optimization**: Memory usage optimization and computational efficiency analysis
- **Scientific Reproducibility**: Deterministic behavior with full audit trails and version control
- **Comprehensive Testing**: Unit tests, integration tests, and end-to-end pipeline validation
- **Error Recovery**: Robust error handling with graceful recovery and detailed diagnostics 