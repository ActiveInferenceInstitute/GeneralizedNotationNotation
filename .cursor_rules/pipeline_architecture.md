# Pipeline Architecture & Implementation Paradigm

### Core Pipeline Orchestration
- **Main Orchestrator**: `src/main.py` dynamically executes numbered pipeline scripts (0-23)
- **Script Discovery**: Uses predefined pipeline_steps list to execute 24 steps in order
- **Execution Model**: Each script runs as subprocess with centralized argument passing
- **Virtual Environment**: Automatic detection and use of project virtual environment
- **Dependency Validation**: Comprehensive dependency checking before pipeline execution
- **Performance Tracking**: Built-in performance monitoring and resource usage tracking
- **Configuration Management**: YAML-based configuration with CLI argument overrides

### Pipeline Steps (24 Steps - Current Order 0-23)
Each numbered script corresponds to a specific module folder and implements real functionality:

0. **0_template.py** → `src/template/` - Pipeline template and initialization (thin orchestrator)
1. **1_setup.py** → `src/setup/` - Environment setup, virtual environment management, dependency installation (CRITICAL STEP)
2. **2_tests.py** → `src/tests/` - Test suite execution with pytest integration and coverage reporting
3. **3_gnn.py** → `src/gnn/` - GNN file discovery, multi-format parsing, and comprehensive validation (CRITICAL STEP)
4. **4_model_registry.py** → `src/model_registry/` - Model registry management and versioning (thin orchestrator)
5. **5_type_checker.py** → `src/type_checker/` - GNN syntax validation, type checking, and resource estimation
6. **6_validation.py** → `src/validation/` - Advanced validation and consistency checking
7. **7_export.py** → `src/export/` - Multi-format export (JSON, XML, GraphML, GEXF, Pickle, PKL)
8. **8_visualization.py** → `src/visualization/` - Graph visualization, matrix visualization with comprehensive safe-to-fail patterns
9. **9_advanced_viz.py** → `src/advanced_visualization/` - Advanced visualization and interactive plots with robust safety patterns
10. **10_ontology.py** → `src/ontology/` - Active Inference Ontology processing, validation, and semantic mapping (thin orchestrator)
11. **11_render.py** → `src/render/` - Code generation for PyMDP, RxInfer.jl, ActiveInference.jl, JAX, and DisCoPy environments
12. **12_execute.py** → `src/execute/` - Execute rendered simulation scripts with comprehensive safety patterns and result capture
13. **13_llm.py** → `src/llm/` - LLM-enhanced analysis, model interpretation, natural language explanations (thin orchestrator)
14. **14_ml_integration.py** → `src/ml_integration/` - Machine learning integration and model training (thin orchestrator)
15. **15_audio.py** → `src/audio/` - Audio generation (SAPF, Pedalboard, and other backends) (thin orchestrator)
16. **16_analysis.py** → `src/analysis/` - Advanced analysis and statistical processing (thin orchestrator)
17. **17_integration.py** → `src/integration/` - System integration and cross-module coordination (thin orchestrator)
18. **18_security.py** → `src/security/` - Security validation and access control (thin orchestrator)
19. **19_research.py** → `src/research/` - Research tools and experimental features (thin orchestrator)
20. **20_website.py** → `src/website/` - Static HTML website generation from pipeline artifacts (thin orchestrator)
21. **21_mcp.py** → `src/mcp/` - Model Context Protocol processing and tool registration (thin orchestrator)
22. **22_gui.py** → `src/gui/` - Interactive GUI for constructing and editing GNN models (thin orchestrator)
23. **23_report.py** → `src/report/` - Comprehensive analysis report generation (thin orchestrator)

### Architectural Pattern: Thin Orchestrator Scripts
**CRITICAL**: Numbered pipeline scripts (especially steps 10, 13-23) must be thin orchestrators that:
1. **Import and invoke methods** from their corresponding modules (e.g., `src/ontology/`, `src/llm/`)
2. **NEVER contain long method definitions** - all core logic belongs in the module
3. **Handle pipeline orchestration** - argument parsing, logging, output directory management, result aggregation
4. **Delegate core functionality** to module classes and functions
5. **Maintain separation of concerns** - scripts handle pipeline flow, modules handle domain logic

**Current Implementation Status:**
- **Thin Orchestrators**: Steps 0, 3, 4, 10, 11, 13-23 (correctly delegate to modules)
- **Hybrid Implementation**: Steps 1-2, 5-9, 12 (mix of orchestration and implementation)
- **Legacy Pattern**: Some steps contain substantial logic but are being migrated to thin orchestrator pattern

**Target Architecture**: All steps follow thin orchestrator pattern while maintaining current functionality. The pattern is:
1. **Numbered Scripts** (e.g., `11_render.py`, `10_ontology.py`): Thin orchestrators that handle pipeline orchestration, argument parsing, logging, and result aggregation
2. **Module `__init__.py`**: Imports and exposes functions from modular files within the module folder  
3. **Modular Files** (e.g., `src/render/renderer.py`, `src/ontology/processor.py`): Contain the actual implementation of core methods
4. **Tests**: All methods are tested in `src/tests/` with comprehensive test coverage

### Centralized Infrastructure

#### Utilities (`src/utils/`)
- **EnhancedArgumentParser**: Centralized argument parsing with step-specific configuration and fallback handling
- **PipelineLogger**: Structured logging with correlation contexts, step tracking, and performance monitoring
- **performance_tracker**: Performance monitoring, resource usage tracking, and execution profiling
- **validate_pipeline_dependencies**: Comprehensive dependency validation with detailed reporting
- **setup_step_logging**: Standardized logging setup with dynamic verbosity control for each pipeline step
- **config_loader**: YAML configuration loading with validation and CLI argument override support

#### Pipeline Configuration (`src/pipeline/`)
- **STEP_METADATA**: Centralized metadata for all 24 pipeline steps with dependencies, timeouts, and requirements
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
- **Critical vs Non-Critical**: Steps 1 and 3 marked as `required=True` in configuration halt pipeline on failure
- **Comprehensive Reporting**: All errors logged with context, suggestions, and actionable information
- **Exit Code Conventions**: 0=success, 1=critical error (stops pipeline), 2=success with warnings (continues pipeline)
- **Dependency Validation**: Pre-flight dependency checks with clear error messages and installation guidance

#### Safe-to-Fail Implementation
- **Steps 8, 9, 12**: Comprehensive safe-to-fail patterns with multiple fallback levels
- **All Steps**: Must return 0 to ensure pipeline continuation regardless of internal failures
- **Error Recovery**: Multiple fallback systems with detailed diagnostics and recovery suggestions
- **Output Generation**: All steps must produce outputs even in failure modes

### Module-Specific Implementation Details

#### Setup Management (`src/setup/`) - Step 1
- **Virtual Environment**: Detection, creation, and management with Python version validation
- **Dependency Installation**: Real pip installation with requirements.txt and requirements-dev.txt support
- **System Validation**: Comprehensive system dependency checking with detailed environment reporting
- **Critical Step**: Pipeline execution depends on successful setup completion

#### GNN Processing (`src/gnn/`) - Step 3
- **Multi-Format Support**: Comprehensive parsing for 21+ formats (Markdown, JSON, YAML, XML, Binary, Protobuf, Maxima, etc.)
- **Validation Levels**: BASIC, STANDARD, STRICT, RESEARCH, and ROUND_TRIP validation with configurable depth
- **Round-Trip Testing**: Semantic preservation testing across format conversions
- **Cross-Format Validation**: Consistency checking across different format representations
- **Real Implementation**: Full GNN specification parsing with sophisticated error reporting and recovery

#### Type Checking (`src/type_checker/`) - Step 5
- **Syntax Validation**: Complete GNN syntax validation with detailed error reporting and suggestions
- **Resource Estimation**: Computational resource estimation for model execution planning
- **Type Consistency**: Variable type checking, dimension validation, and Active Inference conformance
- **Performance Analysis**: Complexity analysis and optimization recommendations

#### Export Systems (`src/export/`) - Step 7
- **Multi-Format Support**: JSON, XML, GraphML, GEXF, Pickle, PKL formats with format-specific optimizations
- **Network Graph Export**: NetworkX-based graph format exports with rich metadata preservation
- **Structured Data**: Comprehensive model metadata preservation across format boundaries
- **Round-Trip Compatibility**: Format exports designed for semantic preservation in round-trip scenarios

#### Visualization (`src/visualization/`) - Step 8
- **Safe-to-Fail Patterns**: Comprehensive fallback systems from full visualization to basic HTML reports
- **Graph Visualization**: Model structure visualization with matplotlib, networkx, and interactive elements
- **Matrix Visualization**: A, B, C, D matrix heatmaps, correlation matrices, and statistical visualizations
- **Correlation ID Tracking**: Complete traceability with unique correlation IDs for debugging
- **Pipeline Continuation**: Always returns 0 to ensure pipeline never stops on visualization failures

#### Advanced Visualization (`src/advanced_visualization/`) - Step 9
- **Comprehensive Safety Patterns**: Multiple fallback levels with detailed HTML reports and dependency status
- **Interactive Visualization**: Advanced 3D plots, interactive dashboards, and dynamic content
- **Resource Management**: Safe processing contexts with automatic cleanup and timeout handling
- **Error Recovery**: Beautiful fallback HTML with recovery suggestions and diagnostics
- **Performance Tracking**: Detailed timing and resource usage tracking for all attempts

#### Simulation Rendering (`src/render/`) - Step 11
- **PyMDP Integration**: Complete PyMDP agent and environment code generation with Active Inference semantics
- **RxInfer Integration**: Full RxInfer.jl model translation with Bayesian inference implementation
- **ActiveInference.jl**: Native Julia implementation generation for high-performance simulation
- **DisCoPy Integration**: Categorical diagram generation for theoretical analysis
- **Template System**: Comprehensive template-based code generation with customizable patterns

#### Execution Engine (`src/execute/`) - Step 12
- **Safe-to-Fail Execution**: Comprehensive safety patterns with circuit breaker implementation
- **Multi-Backend Support**: PyMDP, RxInfer.jl, ActiveInference.jl, JAX, and custom execution backends
- **Error Classification**: Detailed error classification (dependency, syntax, resource, timeout, etc.)
- **Retry Logic**: Exponential backoff retry with intelligent error recovery
- **Performance Monitoring**: Execution time tracking, memory usage, and convergence analysis
- **Pipeline Continuation**: Always returns 0 to ensure pipeline continues even on complete execution failure

#### AI and ML Integration (`src/llm/`, `src/ml_integration/`) - Steps 13-14
- **LLM Processing**: AI-powered GNN model interpretation, validation, and enhancement suggestions
- **Natural Language**: GNN to natural language explanation generation with technical accuracy
- **ML Integration**: Machine learning pipeline integration with model training capabilities
- **Thin Orchestrator Pattern**: Delegate to module implementations for core functionality

#### Audio Generation (`src/audio/`) - Step 15
- **Multi-Backend Support**: SAPF, Pedalboard, and other audio generation backends
- **Model Sonification**: Converting mathematical structures, matrices, and relationships to audio patterns
- **Audio Synthesis**: Real-time audio generation, processing, and export with multiple format support
- **Scientific Sonification**: Mathematically grounded audio representations for model analysis

#### Final Processing Chain (Steps 16-23)
- **Analysis**: Comprehensive statistical analysis and performance metrics aggregation
- **Integration**: Cross-module coordination and system integration
- **Security**: Security validation and access control
- **Research**: Experimental features and advanced research tools
- **Website**: Static HTML site generation with interactive elements
- **MCP**: Model Context Protocol processing and tool registration
- **GUI**: Interactive GNN model construction and editing interface
- **Report**: Final comprehensive reporting and documentation generation

### Performance and Quality Standards
- **Real-Time Monitoring**: Comprehensive performance tracking across all pipeline steps
- **Resource Optimization**: Memory usage optimization and computational efficiency analysis
- **Scientific Reproducibility**: Deterministic behavior with full audit trails and version control
- **Comprehensive Testing**: Unit tests, integration tests, and end-to-end pipeline validation
- **Error Recovery**: Robust error handling with graceful recovery and detailed diagnostics
- **Safe-to-Fail Implementation**: All steps designed to continue pipeline execution regardless of internal failures

### Documentation and Communication Standards
- **Direct Documentation Updates**: Update existing README.md, docstrings, and documentation files directly
- **Functional Improvements**: Focus on making smart functional improvements to code and documentation
- **Inline Updates**: Add documentation directly to relevant files rather than creating separate report files
- **Concrete Demonstrations**: Show functionality through working code, real outputs, and measurable results
- **Understated Communication**: Use specific examples and functional demonstrations over promotional language