# .cursor_rules - Complete Documentation Index

This directory contains comprehensive architectural and implementation guidelines for the GNN Processing Pipeline. Each file covers specific aspects of the system to ensure consistency, quality, and maintainability.

## File Organization

### 1. **pipeline_architecture.md** (215 lines, 16.1 KB)
**Comprehensive pipeline architecture and design patterns**

- **24-Step Pipeline Structure**: Complete 0-23 step mapping with module relationships
- **Thin Orchestrator Pattern**: Design pattern for numbered scripts and module delegation
- **Centralized Infrastructure**: Utils, pipeline config, and module patterns
- **Module-Specific Details**: Implementation details for each major module
- **Performance Standards**: Execution time, memory, and reliability metrics
- **Documentation Standards**: Communication and writing conventions
- **Step Dependencies**: Data flow and dependencies between pipeline steps

**Key Sections:**
- Core Pipeline Orchestration
- Architectural Pattern (Thin Orchestrator)
- Centralized Infrastructure
- Implementation Standards
- Module-Specific Implementation Details
- Performance and Quality Standards

**Best For**: Understanding overall pipeline structure, step execution flow, and architectural patterns.

### 2. **implementation_patterns.md** (750+ lines, 20+ KB)
**Detailed code patterns and infrastructure usage**

- **Modern Standardized Pattern**: `create_standardized_pipeline_script` pattern (PREFERRED for all new scripts)
- **Pipeline Script Implementation**: Standard patterns for numbered scripts with thin orchestrator pattern
- **Argument Parsing**: Enhanced argument parser usage and fallback handling
- **Module Structure**: Organization of module files and __init__.py patterns (including package-level discovery)
- **Module Function Signatures**: Standard interface for all module functions called by orchestrators
- **Output Directory Management**: Standardized output directory structure and naming conventions
- **Enhanced Visual Logging**: Integration with visual logging system for progress indicators and status
- **MCP Integration**: Standard MCP module structure and tool registration
- **Error Handling**: Structured error handling and logging patterns
- **Configuration Management**: Configuration access and CLI override patterns
- **Testing Patterns**: Comprehensive testing structure and fixtures
- **Quality Assurance**: Input validation, performance monitoring patterns

**Key Sections:**
- Modern Standardized Pattern (PREFERRED)
- Legacy Pattern (For Reference Only)
- Module Structure Pattern (including package-level __init__.py)
- Module Function Signature Pattern
- Output Directory Management Pattern
- Enhanced Visual Logging Pattern
- MCP Integration Pattern
- Error Handling and Logging Patterns
- Configuration Management Pattern
- Testing and Validation Patterns
- Quality Assurance Patterns

**Best For**: Writing new pipeline scripts, modules, and understanding coding standards. **CRITICAL**: Use the modern standardized pattern for all new scripts.

### 3. **testing_framework.md** (926 lines, 28.2 KB)
**Comprehensive testing guidelines and infrastructure**

- **Test Architecture**: Module-based organization and test categories
- **Test File Naming**: Standardized naming conventions
- **Import Patterns**: Correct and incorrect import patterns (CRITICAL)
- **Test File Structure**: Standard test file templates and fixtures
- **Test Execution**: Pytest, runner, and fast test suite execution
- **Test Markers**: Unit, integration, performance, safe-to-fail markers
- **Test Fixtures**: Core fixtures and mock fixtures
- **Test Utilities**: Shared test utilities and helpers
- **Error Handling**: Safe-to-fail patterns and graceful degradation
- **23 Test Categories**: Coverage for all major modules
- **Performance Testing**: Memory tracking and resource limits
- **Coverage Analysis**: Running and analyzing test coverage

**Key Sections:**
- Module-Based Test Organization
- Test File Naming Convention
- Import Patterns (CRITICAL - correct vs incorrect)
- Standard Test File Template
- Test Execution Patterns
- Test Markers
- Test Fixtures
- Test Utilities
- Error Handling Patterns
- Current Status
- Troubleshooting Guide

**Best For**: Writing tests, understanding test infrastructure, setting up new test categories.

### 4. **pipeline_architecture.md** (300+ lines, 20+ KB)
**Pipeline structure and orchestration patterns**

- **Main Orchestrator**: `src/main.py` execution model with comprehensive details
- **Script Discovery**: 24-step pipeline structure (0-23)
- **Execution Model**: Subprocess execution with centralized arguments
- **Data Dependencies**: Complete dependency graph with automatic resolution
- **Step Relationships**: Detailed step dependencies and data flow
- **Virtual Environment**: Automatic venv detection and usage
- **Dependency Validation**: Pre-flight checks
- **Performance Tracking**: Built-in monitoring with memory and timing
- **Enhanced Visual Logging**: Real-time progress indicators and status tracking
- **Configuration Management**: YAML-based configuration
- **Automatic Dependency Resolution**: Intelligent dependency inclusion with `--only-steps`

**Key Sections:**
- Core Pipeline Orchestration
- Pipeline Steps (24 steps - 0-23)
- Architectural Pattern (Thin Orchestrator)
- Main Orchestrator (`src/main.py`) - Detailed execution flow
- Data Dependencies and Step Relationships
- Centralized Infrastructure
- Implementation Standards
- Module-Specific Implementation Details
- Performance Standards

**Best For**: Understanding overall pipeline flow, execution model, and step dependencies.

### 5. **gnn_standards.md** (326 lines, 16.4 KB)
**GNN domain knowledge and processing standards**

- **Project Overview**: GNN ecosystem description
- **Core Architecture**: Validation levels and processing pipeline
- **GNN File Structure**: Markdown format with specific sections
- **Variable Definitions**: Pattern and examples
- **Connection Types**: Directed, undirected, conditional relationships
- **GNN Syntax**: Mathematical notation and Unicode support
- **Multi-Format Support**: 21+ format specifications
- **Validation Levels**: BASIC, STANDARD, STRICT, RESEARCH, ROUND_TRIP
- **Active Inference Standards**: Standard variables and POMDP structure
- **Format Ecosystem**: Schema, language, and formal specification formats
- **Testing Infrastructure**: Round-trip testing and validation

**Key Sections:**
- Project Overview
- Core Architecture
- GNN File Structure Understanding
- GNN Syntax and Punctuation
- Multi-Format GNN Support
- Validation Levels and Standards
- Active Inference Model Standards
- Processing Standards
- Format Ecosystem
- Testing and Quality Assurance

**Best For**: Understanding GNN specifications, validation requirements, and supported formats.

### 6. **quality_and_dev.md** (119 lines, 8.6 KB)
**Quality assurance and development standards**

- **Quality Assurance Standards**: No mocks, real data, complete pipelines
- **Comprehensive Testing**: Pytest, integration, performance, regression tests
- **Documentation Standards**: Docstrings, type hints, examples, API documentation
- **Development Guidelines**: Module separation, import hierarchy, path management
- **Error Handling Philosophy**: Fail fast, informative diagnostics, graceful degradation
- **Performance Management**: Real-time monitoring, parallel processing, optimization
- **Advanced Features**: Extensibility, plugin architecture, reproducibility
- **Module Structure**: 14-step pipeline, output directory structure
- **Development Workflow**: Tools and integration
- **Code Quality**: Advanced organization and consistency

**Key Sections:**
- Quality Assurance Standards
- Development Guidelines
- Advanced Code Organization
- Documentation and Communication Standards
- Enhanced Error Handling Philosophy
- Performance and Resource Management
- Advanced Features and Architecture
- Development Workflow Integration
- Code Quality and Consistency Tools

**Best For**: Understanding quality standards, development workflow, and best practices.

### 7. **mcp_integration.md** (260+ lines)
**Model Context Protocol integration and tool registration**

- **Overview**: MCP enables standardized tool registration and discovery
- **Universal MCP Pattern**: Tool registration across modules
- **MCP System Architecture**: Central registry, module integration, tool discovery
- **Tool Categories**: By module with complete tool listings
- **MCP Tool Implementation**: Standard tool structure and patterns
- **Error Handling**: Error types and response formats
- **Server Configuration**: Setup, registration, and management
- **Usage Examples**: Querying tools, calling tools, response formats
- **Protocol Compliance**: Version, standard features, extensions
- **Performance Considerations**: Execution, monitoring, resource limits
- **Security**: Access control, input validation, authentication

**Key Sections:**
- Overview
- Universal MCP Pattern
- MCP System Architecture
- MCP Tool Categories
- MCP Tool Implementation Pattern
- MCP Error Handling
- MCP Server Configuration
- Usage Examples
- Protocol Compliance
- Performance Considerations
- Security and Access Control

**Best For**: Understanding MCP integration, registering new tools, working with external systems.

## Navigation Guide

### By Task

#### "I'm writing a new pipeline script"
1. Read: **pipeline_architecture.md** (Step definitions and patterns)
2. Read: **implementation_patterns.md** (Script patterns and argument parsing)
3. Reference: **quality_and_dev.md** (Quality standards)

#### "I'm creating a new module"
1. Read: **implementation_patterns.md** (Module structure pattern)
2. Read: **quality_and_dev.md** (Development guidelines)
3. Read: **pipeline_architecture.md** (Thin orchestrator pattern)

#### "I'm writing tests"
1. Read: **testing_framework.md** (Comprehensive testing guidelines)
2. Reference: **implementation_patterns.md** (Testing patterns)
3. Reference: **quality_and_dev.md** (Quality standards)

#### "I'm adding MCP tools"
1. Read: **mcp_integration.md** (MCP patterns and tool registration)
2. Reference: **implementation_patterns.md** (MCP integration pattern)
3. Reference: **quality_and_dev.md** (Quality standards)

#### "I need to understand GNN specifications"
1. Read: **gnn_standards.md** (Complete GNN documentation)
2. Reference: **pipeline_architecture.md** (GNN processing step)

#### "I'm working on code quality"
1. Read: **quality_and_dev.md** (Complete QA standards)
2. Read: **implementation_patterns.md** (Code patterns)
3. Reference: **testing_framework.md** (Testing patterns)

### By Module Area

#### Pipeline Orchestration
- **pipeline_architecture.md**: Overall structure and flow
- **implementation_patterns.md**: Script patterns and orchestration
- **quality_and_dev.md**: Quality and workflow standards

#### Code Implementation
- **implementation_patterns.md**: All coding patterns
- **quality_and_dev.md**: Development guidelines
- **testing_framework.md**: Testing patterns

#### Testing
- **testing_framework.md**: Comprehensive testing guide
- **implementation_patterns.md**: Testing patterns
- **quality_and_dev.md**: Quality standards

#### GNN Processing
- **gnn_standards.md**: Complete GNN reference
- **pipeline_architecture.md**: Step 3 details
- **quality_and_dev.md**: Quality standards

#### MCP Integration
- **mcp_integration.md**: Complete MCP reference
- **implementation_patterns.md**: MCP patterns
- **quality_and_dev.md**: Quality standards

#### Quality Assurance
- **quality_and_dev.md**: Complete QA reference
- **testing_framework.md**: Testing infrastructure
- **implementation_patterns.md**: Code patterns

## Cross-References

### Key Concepts Indexed

**Thin Orchestrator Pattern**
- Defined in: **pipeline_architecture.md** (Architectural Pattern section)
- Implementation: **implementation_patterns.md** (Pipeline Script Implementation)
- Examples: **pipeline_architecture.md** (Step implementations)

**Test Infrastructure**
- Defined in: **testing_framework.md** (Test Architecture)
- Import Patterns: **testing_framework.md** (Import Patterns - CRITICAL)
- Execution: **testing_framework.md** (Test Execution Patterns)

**Error Handling**
- Defined in: **implementation_patterns.md** (Error Handling Patterns)
- Standards: **quality_and_dev.md** (Error Handling Philosophy)
- Testing: **testing_framework.md** (Error Handling Patterns)

**Module Structure**
- Pattern: **implementation_patterns.md** (Module Structure Pattern)
- Organization: **quality_and_dev.md** (Module Structure)
- Guidelines: **quality_and_dev.md** (Development Guidelines)

**MCP Integration**
- Complete Guide: **mcp_integration.md** (All sections)
- Implementation Pattern: **implementation_patterns.md** (MCP Integration Pattern)
- Tool Examples: **pipeline_architecture.md** (Module-specific details)

**GNN Processing**
- Complete Reference: **gnn_standards.md** (All sections)
- Pipeline Step: **pipeline_architecture.md** (Step 3)
- Implementation: **implementation_patterns.md** (Error handling, logging)

## File Statistics

| File | Lines | Size | Sections | Focus |
|------|-------|------|----------|-------|
| pipeline_architecture.md | 300+ | 20+ KB | 10+ | Overall structure + dependencies |
| implementation_patterns.md | 750+ | 20+ KB | 12+ | Code patterns + standardized script |
| testing_framework.md | 926 | 28.2 KB | 22 | Testing |
| gnn_standards.md | 326 | 16.4 KB | 14 | GNN specs |
| quality_and_dev.md | 119 | 8.6 KB | 8 | QA standards |
| mcp_integration.md | 260+ | ~12 KB | 11 | MCP tools |
| **TOTAL** | **2680+** | **~112 KB** | **77+** | Complete guide |

## Quick Reference

### Critical Concepts

1. **Thin Orchestrator Pattern** (CRITICAL)
   - Numbered scripts (N_module.py) are thin orchestrators
   - All core logic must be in module folder
   - See: **pipeline_architecture.md** + **implementation_patterns.md**

2. **Import Patterns** (CRITICAL)
   - Direct module imports (not `src.module`)
   - See: **testing_framework.md** (Import Patterns section)

3. **No Mocks** (CRITICAL)
   - All implementations must be functional
   - See: **quality_and_dev.md** + **testing_framework.md**

4. **MCP Integration**
   - Every module should expose MCP tools
   - See: **mcp_integration.md**

5. **GNN Processing**
   - 21+ format support with semantic preservation
   - See: **gnn_standards.md**

## Maintenance

### When to Update
- Adding new pipeline steps
- Changing coding standards
- Modifying testing framework
- Adding MCP tools
- Updating GNN specifications
- Changing quality standards

### Update Procedure
1. Identify which file(s) need updates
2. Make targeted updates to specific sections
3. Update cross-references if needed
4. Add entries to this INDEX if new sections added
5. Update statistics table

### Consistency Checks
- All 24 pipeline steps documented (Steps 0-23)
- All modules have corresponding guidance
- Testing framework covers all modules
- MCP integration examples provided
- GNN standards current with implementations

---

**Last Updated**: December 2024  
**Total Coverage**: 77+ sections across 7 files  
**Status**: ✅ Complete and Production Ready

**Recent Enhancements (December 2024):**
- ✅ Added modern standardized pipeline script pattern using `create_standardized_pipeline_script`
- ✅ Expanded module structure patterns including package-level `__init__.py` discovery
- ✅ Added module function signature requirements and output directory management patterns
- ✅ Enhanced visual logging integration patterns
- ✅ Added comprehensive main orchestrator documentation with execution flow
- ✅ Added complete data dependencies and step relationships documentation
- ✅ Updated all cross-references and examples to reflect current implementation


