# Quality Assurance and Development Standards

> **Environment Note**: Standardize all tooling through `uv` (e.g., `uv pip install`, `uv run python src/main.py`, `uv pytest`). This keeps dependency, environment, and lockfile alignment consistent across developers and CI.

## Quality Assurance Standards

### Implementation Requirements
- **No Mocks or Stubs**: All implementations must be fully functional with real data processing
- **Real Data Processing**: Use actual GNN files, comprehensive format support, and genuine computation
- **Complete Pipelines**: End-to-end functionality from multi-format input to executable simulation output
- **Scientific Accuracy**: Mathematically correct Active Inference implementations with validation
- **Performance Monitoring**: Real-time resource tracking and optimization across all pipeline steps

### Comprehensive Testing Infrastructure
- **Pytest Framework**: Complete test suite with real GNN file testing across all supported formats
- **Integration Tests**: Full pipeline testing with actual data and cross-format validation
- **Performance Tests**: Resource usage validation, timing analysis, and scalability testing
- **Regression Tests**: Ensure changes don't break existing functionality with automated CI/CD
- **Round-Trip Testing**: Semantic preservation validation across format conversions
- **Cross-Format Validation**: Consistency testing across Markdown, JSON, YAML, XML, PKL, and Protobuf formats

### Documentation and Type Safety Standards
- **Comprehensive Docstrings**: Full documentation for all functions, classes, and modules with examples
- **Complete Type Hints**: Type annotations throughout codebase with proper generics and union types
- **Usage Examples**: Real examples using actual GNN files and multi-format scenarios
- **API Documentation**: Auto-generated documentation from docstrings and type hints
- **Validation Documentation**: Clear documentation of all validation levels and their requirements

## Development Guidelines

### Advanced Code Organization
- **Module Separation**: Clear separation between pipeline steps, core modules, and infrastructure
- **Import Hierarchy**: Centralized utilities → pipeline config → module-specific with fallback handling
- **Dependency Management**: Explicit dependency declarations, validation, and graceful degradation
- **Path Management**: Consistent path handling with pathlib.Path and centralized output management
- **Configuration Management**: YAML-based configuration with CLI override support

### Documentation and Communication Standards
- **Direct Documentation**: Update existing documentation files directly rather than creating separate report files
- **Functional Improvements**: Focus on making smart functional improvements to code and documentation
- **Inline Updates**: Add documentation directly to relevant files (README.md, docstrings, etc.)
- **Show Not Tell**: Demonstrate functionality through working code and real outputs rather than separate reports
- **Understated Communication**: Use concrete examples and functional demonstrations over promotional language

### Enhanced Error Handling Philosophy
- **Fail Fast with Recovery**: Early detection of issues with intelligent recovery mechanisms
- **Informative Diagnostics**: Clear error messages with actionable suggestions and context
- **Graceful Degradation**: Attempt to continue pipeline processing when possible
- **Comprehensive Audit Trails**: Full logging of all operations with correlation IDs and performance metrics
- **Multi-Level Validation**: Different validation levels (BASIC, STANDARD, STRICT, RESEARCH, ROUND_TRIP)

### Performance and Resource Management
- **Real-Time Monitoring**: Track memory usage, execution time, and system resources across all steps
- **Parallel Processing**: Use parallel operations when appropriate and safe with proper resource management
- **Intelligent Caching**: Cache parsed results, validation outcomes, and expensive computations
- **Performance Optimization**: Profile and optimize critical paths with data-driven improvements
- **Resource Estimation**: Computational resource estimation for model execution planning

## Advanced Features and Architecture

### Extensibility and Plugin Architecture
- **Modular Pipeline System**: Support for custom pipeline steps and dynamic module discovery
- **Multi-Backend Flexibility**: Multiple simulation backends (PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy)
- **Format Extension System**: Easy addition of new export/import formats with round-trip support
- **Tool Integration Ecosystem**: MCP-based tool ecosystem for external integrations and AI assistance
- **Template-Based Code Generation**: Extensible template system for multiple target environments

### Scientific Reproducibility and Validation
- **Deterministic Behavior**: Consistent outputs for identical inputs across all processing steps
- **Comprehensive Version Control**: Full provenance tracking, version management, and audit trails
- **Environment Capture**: Complete environment specification and reproduction capabilities
- **Multi-Level Validation**: Configurable validation depth from basic syntax to research-grade validation
- **Cross-Format Consistency**: Semantic preservation and consistency across all supported formats

## Current Module Structure and Naming Conventions

### 14-Step Pipeline Structure
- **Core Processing Modules**: `setup/`, `gnn/`, `tests/`, `type_checker/`, `export/`, `visualization/`, `mcp/`, `ontology/`, `render/`, `execute/`, `llm/`, `website/`, `sapf/`
- **Infrastructure Modules**: `utils/`, `pipeline/` (centralized configuration and execution), `tests/` (comprehensive testing)
- **Pipeline Scripts**: `1_setup.py`, `3_gnn.py`, `2_tests.py`, `5_type_checker.py`, `7_export.py`, `6_visualization.py`, `21_mcp.py`, `10_ontology.py`, `11_render.py`, `12_execute.py`, `11_llm.py`, `12_audio.py`, `13_website.py`, `14_report.py`

### Standardized Output Directory Structure
- **Setup Artifacts**: `output/setup_artifacts/` (environment info, dependency reports)
- **GNN Processing**: `output/gnn_processing_step/` (discovery reports, parsing results)
- **Test Reports**: `output/test_reports/` (pytest results, coverage reports)
- **Type Checking**: `output/type_check/` (validation reports, resource estimates)
- **Multi-Format Exports**: `output/gnn_exports/` (JSON, XML, GraphML, GEXF, PKL, Protobuf)
- **Visualizations**: `output/visualization/` (graphs, matrices, ontology diagrams)
- **MCP Processing**: `output/mcp_processing_step/` (tool registrations, MCP operations)
- **Ontology Processing**: `output/ontology_processing/` (semantic mappings, validation)
- **Rendered Simulators**: `output/gnn_rendered_simulators/` (PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy)
- **Execution Results**: `output/execution_results/` (simulation outputs, performance metrics)
- **LLM Processing**: `output/llm_processing_step/` (AI analysis, natural language explanations)
- **Website Generation**: `output/website/` (HTML reports, interactive elements)
- **SAPF Processing**: `output/sapf_processing_step/` (audio representations, sonifications)
- **Pipeline Logs**: `output/logs/` (correlation IDs, performance tracking, audit trails)

## Development Workflow and Tools

### Pipeline Development Tools
- **Pipeline Step Template**: Use `src/pipeline/pipeline_step_template.py` for consistent new step creation
- **Pipeline Validation**: Use `src/pipeline/pipeline_validation.py` to validate implementation consistency
- **Module Template**: Follow `src/utils/pipeline_template.py` patterns for module structure
- **Configuration System**: Leverage `src/pipeline/config.py` for centralized configuration management
- **Testing Infrastructure**: Use comprehensive testing in `src/tests/` with real data validation

### Code Quality and Consistency Tools
- **EnhancedArgumentParser**: Standardized argument parsing with step-specific configuration
- **Centralized Logging**: Structured logging with correlation IDs and performance tracking
- **Performance Tracking**: Built-in performance monitoring and resource usage analysis
- **Dependency Validation**: Comprehensive dependency checking with clear error messages
- **Multi-Format Support**: Robust parsing and validation across all supported GNN formats

### Advanced Quality Assurance
- **Round-Trip Validation**: Semantic preservation testing across format conversions
- **Cross-Format Consistency**: Validation across Markdown, JSON, YAML, XML, PKL, and Protobuf
- **Performance Benchmarking**: Automated performance testing and regression detection
- **Scientific Validation**: Active Inference model compliance and mathematical accuracy checking
- **Comprehensive Error Reporting**: Detailed diagnostics with actionable suggestions and context 