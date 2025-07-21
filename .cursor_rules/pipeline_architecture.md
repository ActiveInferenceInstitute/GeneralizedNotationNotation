# Pipeline Architecture & Implementation Paradigm

### Core Pipeline Orchestration
- **Main Orchestrator**: `src/main.py` dynamically discovers and executes numbered pipeline scripts
- **Script Discovery**: Uses glob patterns to find `*_*.py` files, sorts by number and basename
- **Execution Model**: Each script runs as subprocess with centralized argument passing
- **Virtual Environment**: Automatic detection and use of project virtual environment
- **Dependency Validation**: Comprehensive dependency checking before pipeline execution
- **Performance Tracking**: Built-in performance monitoring and resource usage tracking

### Pipeline Steps (Dynamically Discovered, 1-13)
Each numbered script corresponds to a specific module folder and implements real functionality:

1. **1_setup.py** → `src/setup/` - Environment setup, virtual environment management, dependency installation (CRITICAL STEP)
2. **2_gnn.py** → `src/gnn/` - GNN file discovery, parsing, and basic validation
3. **3_tests.py** → `src/tests/` - Test suite execution with pytest integration
4. **4_type_checker.py** → `src/type_checker/` - GNN syntax validation and resource estimation
5. **5_export.py** → `src/export/` - Multi-format export (JSON, XML, GraphML, GEXF, Pickle)
6. **6_visualization.py** → `src/visualization/` - Graph visualization and matrix visualization
7. **7_mcp.py** → `src/mcp/` - Model Context Protocol operations and tool registration
8. **8_ontology.py** → `src/ontology/` - Active Inference Ontology processing and validation
9. **9_render.py** → `src/render/` - Code generation for PyMDP, RxInfer simulation environments
10. **10_execute.py** → `src/execute/` - Execute rendered simulation scripts
11. **11_llm.py** → `src/llm/` - LLM-enhanced analysis, model interpretation, and AI assistance
12. **12_website.py** → `src/website/` - Static HTML website generation from pipeline artifacts
13. **13_sapf.py** → `src/sapf/` - SAPF (Sound As Pure Form) audio generation and sonification

### Centralized Infrastructure

#### Utilities (`src/utils/`)
- **EnhancedArgumentParser**: Centralized argument parsing for all pipeline steps
- **PipelineLogger**: Structured logging with correlation contexts and step tracking
- **performance_tracker**: Performance monitoring and resource usage tracking
- **validate_pipeline_dependencies**: Comprehensive dependency validation
- **setup_step_logging**: Standardized logging setup for each pipeline step

#### Pipeline Configuration (`src/pipeline/`)
- **STEP_METADATA**: Centralized metadata for all pipeline steps
- **get_pipeline_config**: Pipeline configuration management
- **get_output_dir_for_script**: Standardized output directory structure
- **execute_pipeline_step**: Centralized step execution with error handling

#### Module Structure Pattern
Each module follows this structure:
```
src/[module_name]/
├── __init__.py          # Module initialization
├── [main_module].py     # Core functionality
├── mcp.py              # MCP integration (if applicable)
└── [additional_files]   # Specialized components
```

### Implementation Standards

#### Script Structure Pattern
Every numbered script follows this pattern:
1. **Import centralized utilities** (`utils`, `pipeline`)
2. **Initialize step-specific logger** via `setup_step_logging`
3. **Import module-specific functionality** from corresponding folder
4. **Define main() function** with centralized argument handling
5. **Use EnhancedArgumentParser** for argument parsing with fallback
6. **Implement proper error handling** with step logging functions
7. **Return appropriate exit codes** (0=success, 1=error, 2=warnings)

#### Argument Parsing Standards
- **Centralized Parsing**: Use `EnhancedArgumentParser.parse_step_arguments()` for all pipeline steps
- **Fallback Handling**: Always provide fallback argument parser for graceful degradation
- **Common Arguments**: All steps support `--target-dir`, `--output-dir`, `--verbose`, `--recursive`
- **Step-Specific Arguments**: Additional arguments based on step functionality (e.g., `--strict`, `--duration`)
- **Path Conversion**: Defensive conversion of string arguments to `pathlib.Path` objects

#### Logging Standards
- **Step Logging Functions**: Use `log_step_start`, `log_step_success`, `log_step_warning`, `log_step_error`
- **Correlation Context**: Each step sets correlation context for traceability
- **Verbosity Control**: Dynamic verbosity adjustment based on arguments
- **Performance Tracking**: Use `performance_tracker.track_operation()` for timed operations

#### Error Handling Standards
- **Graceful Degradation**: Steps continue pipeline unless critical failure occurs
- **Critical vs Non-Critical**: Steps marked as `required=True` halt pipeline on failure
- **Comprehensive Reporting**: All errors logged with context and suggestions
- **Exit Code Conventions**: 0=success, 1=critical error, 2=success with warnings

### Module-Specific Implementation Details

#### GNN Processing (`src/gnn/`)
- **File Discovery**: Recursive and non-recursive GNN file discovery
- **Basic Parsing**: ModelName, StateSpaceBlock, Connections extraction
- **Parameter Extraction**: ModelParameters section parsing with AST evaluation
- **Real Implementation**: Full GNN file structure parsing, no mocks

#### Setup Management (`src/setup/`)
- **Virtual Environment**: Detection, creation, and management
- **Dependency Installation**: Real pip installation with requirements.txt
- **System Validation**: Comprehensive system dependency checking
- **Environment Info**: Detailed environment reporting and validation

#### Type Checking (`src/type_checker/`)
- **Syntax Validation**: Full GNN syntax validation with detailed error reporting
- **Resource Estimation**: Computational resource estimation for models
- **Type Consistency**: Variable type checking and dimension validation
- **Real Validation**: Complete GNN specification compliance checking

#### Export Systems (`src/export/`)
- **Multi-Format Support**: JSON, XML, GraphML, GEXF, Pickle formats
- **Network Graph Export**: NetworkX-based graph format exports
- **Structured Data**: Comprehensive model metadata preservation
- **Real Exporters**: Fully functional format conversion, no stubs

#### Visualization (`src/visualization/`)
- **Graph Visualization**: Model structure visualization with matplotlib/networkx
- **Matrix Visualization**: A, B, C, D matrix heatmaps and visualizations
- **Ontology Visualization**: Active Inference ontology relationship diagrams
- **Real Rendering**: Complete visualization pipeline with multiple output formats

#### Simulation Rendering (`src/render/`)
- **PyMDP Integration**: Full PyMDP agent and environment code generation
- **RxInfer Integration**: Complete RxInfer.jl model translation
- **Template System**: Comprehensive template-based code generation
- **Real Code Generation**: Executable simulation code, not pseudocode

#### Execution Engine (`src/execute/`)
- **Script Execution**: Real Python script execution with subprocess management
- **Result Capture**: Output capture, error handling, and result reporting
- **Multi-Backend**: Support for PyMDP, RxInfer, and custom execution engines
- **Real Execution**: Actual simulation execution with performance monitoring

#### LLM Integration (`src/llm/`)
- **Model Analysis**: AI-powered GNN model interpretation and validation
- **Enhancement Suggestions**: Automated model improvement recommendations
- **Natural Language**: GNN to natural language explanation generation
- **Real AI Integration**: Functional LLM processing, not mock responses

#### Site Generation (`src/site/`)
- **HTML Generation**: Static HTML site generation from pipeline artifacts
- **Report Aggregation**: Comprehensive pipeline summary and results presentation
- **Interactive Elements**: Dynamic content and navigation
- **Real Generation**: Complete HTML site with all pipeline outputs

#### SAPF Audio Generation (`src/sapf/`)
- **Sound As Pure Form**: Audio representation and sonification of GNN models
- **Model Sonification**: Converting mathematical structures to audio patterns
- **Audio Synthesis**: Real-time audio generation and processing
- **Real Audio**: Functional audio generation, not placeholder implementations 