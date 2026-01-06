# .cursor_rules - Complete Documentation Index

This directory contains comprehensive architectural and implementation guidelines for the GNN Processing Pipeline. Each file covers specific aspects of the system to ensure consistency, quality, and maintainability.

---

## File Organization

### Core Architecture Files

#### 1. **pipeline_architecture.md** (~300 lines)
**Pipeline structure and orchestration patterns**

- **24-Step Pipeline Structure**: Complete 0-23 step mapping with module relationships
- **Thin Orchestrator Pattern**: Design pattern for numbered scripts and module delegation
- **Main Orchestrator**: `src/main.py` execution model with comprehensive details
- **Data Dependencies**: Complete dependency graph with automatic resolution
- **Performance Standards**: Execution time, memory, and reliability metrics

**Best For**: Understanding overall pipeline structure, step execution flow, and architectural patterns.

#### 2. **implementation_patterns.md** (~750 lines)
**Detailed code patterns and infrastructure usage**

- **Modern Standardized Pattern**: `create_standardized_pipeline_script` (PREFERRED)
- **Module Structure**: Organization and `__init__.py` patterns
- **Module Function Signatures**: Standard interface for module functions
- **MCP Integration**: Standard MCP module structure
- **Error Handling**: Structured error handling and logging patterns

**Best For**: Writing new pipeline scripts, modules, and understanding coding standards.

#### 3. **module_patterns.md** (~350 lines) ⭐ NEW
**Advanced module architecture patterns**

- **Module Directory Structure**: Standard layout for all modules
- **Thin Orchestrator Pattern**: Script vs module responsibilities
- **MCP Integration Pattern**: Standard mcp.py structure
- **Dependency Injection**: Configuration and factory patterns
- **Testing Patterns**: Module test structure

**Best For**: Creating new modules, understanding module architecture.

---

### Quality & Standards Files

#### 4. **code_quality.md** (~320 lines) ⭐ NEW
**Detailed code quality standards**

- **Type Safety**: Complete type hints requirements
- **Documentation Standards**: Docstrings, examples
- **Testing Standards**: No mocks policy, coverage requirements
- **Linting and Formatting**: Black, Ruff standards
- **Review Checklist**: Pre-submission requirements

**Best For**: Understanding quality requirements, code review criteria.

#### 5. **quality_and_dev.md** (~120 lines)
**Quality assurance and development workflow**

- **Quality Assurance Standards**: No mocks, real data
- **Development Guidelines**: Module separation, path management
- **Error Handling Philosophy**: Fail fast, graceful degradation
- **Development Workflow**: Tools and integration

**Best For**: Development workflow, quick QA reference.

#### 6. **testing_framework.md** (~930 lines)
**Comprehensive testing guidelines**

- **Test Architecture**: Module-based organization
- **Import Patterns**: Correct patterns (CRITICAL)
- **Test Markers**: Unit, integration, safe-to-fail
- **23 Test Categories**: Coverage for all modules
- **Troubleshooting**: Common test issues

**Best For**: Writing tests, test infrastructure setup.

---

### Error & Performance Files

#### 7. **error_handling.md** (~400 lines) ⭐ NEW
**Comprehensive error handling strategies**

- **Exit Code Conventions**: 0/1/2 meanings
- **Safe-to-Fail Pattern**: Steps 8, 9, 12 implementation
- **Graceful Degradation**: Tiered functionality
- **Error Classification**: Types and handling
- **Recovery Patterns**: Retry, circuit breaker

**Best For**: Implementing error handling, understanding safe-to-fail.

#### 8. **performance_optimization.md** (~350 lines) ⭐ NEW
**Performance tuning guide**

- **Performance Baselines**: Step-specific benchmarks
- **Memory Optimization**: Tracking, cleanup patterns
- **Execution Time**: Parallel processing, caching
- **Timeout Management**: Step timeouts
- **Profiling Guidelines**: CPU, memory profiling

**Best For**: Performance optimization, profiling, benchmarking.

---

### Domain Knowledge Files

#### 9. **gnn_standards.md** (~330 lines)
**GNN domain knowledge and processing**

- **GNN File Structure**: Markdown format, sections
- **Variable Definitions**: Patterns and examples
- **Multi-Format Support**: 21+ format specifications
- **Validation Levels**: BASIC to ROUND_TRIP
- **Active Inference Standards**: POMDP structure

**Best For**: Understanding GNN specifications, validation requirements.

#### 10. **render_frameworks.md** (~400 lines) ⭐ NEW
**Framework-specific rendering documentation**

- **PyMDP**: Python Active Inference
- **JAX**: Pure JAX (no Flax required)
- **RxInfer.jl**: Julia probabilistic programming
- **ActiveInference.jl**: Native Julia AI
- **DisCoPy**: Categorical diagrams
- **Framework Selection Guide**: Decision matrix

**Best For**: Understanding render frameworks, code generation patterns.

#### 11. **optional_dependencies.md** (~280 lines) ⭐ NEW
**Guide for optional packages**

- **Dependency Categories**: Core vs optional
- **Detection Patterns**: Python, Julia, system
- **Graceful Fallback**: Import patterns
- **Framework-Specific Handling**: JAX, PyMDP, Julia
- **Installation Helpers**: Commands and scripts

**Best For**: Handling optional dependencies, graceful degradation.

---

### Integration Files

#### 12. **mcp_integration.md** (~360 lines)
**Model Context Protocol integration**

- **MCP System Architecture**: Registry, discovery
- **Tool Categories**: By module
- **Tool Implementation Pattern**: Standard structure
- **Error Handling**: Response formats
- **Security**: Access control

**Best For**: MCP integration, registering new tools.

#### 13. **troubleshooting.md** (~300 lines) ⭐ NEW
**Common issues and solutions**

- **Quick Diagnostics**: Health check commands
- **Common Issues**: By category
- **Error Message Reference**: Meanings and actions
- **Recovery Procedures**: Reset, rebuild, resume
- **Debug Information**: What to collect

**Best For**: Debugging issues, quick problem resolution.

---

### Reference Files

#### 14. **AGENTS.md** (~550 lines)
**28 Agent capabilities summary**

- **Agent Index**: All 26 modules (24 pipeline + 2 infrastructure)
- **Capabilities by Category**: Core, simulation, integration
- **Agent Performance Metrics**: Execution times
- **Agent Development Guidelines**: Standards

**Best For**: Understanding agent capabilities, module overview.

#### 15. **README.md** (~430 lines)
**Navigation guide**

- **Quick Navigation**: By task, by topic
- **Key Concepts**: Critical patterns
- **Getting Started Workflows**: New developers
- **Environment Setup**: uv commands

**Best For**: Finding documentation, onboarding.

---

## Navigation Guide

### By Task

| Task | Primary Files | Secondary Files |
|------|--------------|-----------------|
| Writing pipeline script | implementation_patterns.md | pipeline_architecture.md |
| Creating new module | module_patterns.md | implementation_patterns.md |
| Writing tests | testing_framework.md | code_quality.md |
| Adding MCP tools | mcp_integration.md | module_patterns.md |
| Understanding GNN | gnn_standards.md | render_frameworks.md |
| Handling errors | error_handling.md | troubleshooting.md |
| Performance tuning | performance_optimization.md | pipeline_architecture.md |
| Optional deps | optional_dependencies.md | troubleshooting.md |
| Framework rendering | render_frameworks.md | optional_dependencies.md |

### By Module Area

| Area | Files |
|------|-------|
| Pipeline Orchestration | pipeline_architecture.md, implementation_patterns.md |
| Code Implementation | implementation_patterns.md, module_patterns.md, code_quality.md |
| Testing | testing_framework.md, code_quality.md |
| GNN Processing | gnn_standards.md, render_frameworks.md |
| MCP Integration | mcp_integration.md, module_patterns.md |
| Error Handling | error_handling.md, troubleshooting.md |
| Performance | performance_optimization.md |
| Dependencies | optional_dependencies.md |

---

## Cross-References

### Key Concepts

| Concept | Defined In | Related Files |
|---------|-----------|---------------|
| Thin Orchestrator | pipeline_architecture.md | implementation_patterns.md, module_patterns.md |
| Safe-to-Fail | error_handling.md | troubleshooting.md |
| No Mocks Policy | code_quality.md | testing_framework.md |
| Module Function Signature | implementation_patterns.md | module_patterns.md |
| Optional Dependencies | optional_dependencies.md | render_frameworks.md |
| Exit Codes | error_handling.md | pipeline_architecture.md |
| JAX Rendering | render_frameworks.md | optional_dependencies.md |

---

## File Statistics

| File | Lines | Focus | Status |
|------|-------|-------|--------|
| pipeline_architecture.md | ~300 | Pipeline structure | ✅ |
| implementation_patterns.md | ~750 | Code patterns | ✅ |
| module_patterns.md | ~350 | Module architecture | ⭐ NEW |
| code_quality.md | ~320 | Quality standards | ⭐ NEW |
| quality_and_dev.md | ~120 | QA overview | ✅ |
| testing_framework.md | ~930 | Testing | ✅ |
| error_handling.md | ~400 | Error strategies | ⭐ NEW |
| performance_optimization.md | ~350 | Performance | ⭐ NEW |
| gnn_standards.md | ~330 | GNN domain | ✅ |
| render_frameworks.md | ~400 | Frameworks | ⭐ NEW |
| optional_dependencies.md | ~280 | Dependencies | ⭐ NEW |
| mcp_integration.md | ~360 | MCP tools | ✅ |
| troubleshooting.md | ~300 | Debugging | ⭐ NEW |
| AGENTS.md | ~550 | Agent capabilities | ✅ |
| README.md | ~430 | Navigation | ✅ |
| **TOTAL** | **~6170** | **Complete guide** | ✅ |

---

## Quick Reference

### Critical Concepts

1. **Thin Orchestrator Pattern** (CRITICAL)
   - Numbered scripts are thin orchestrators
   - All core logic in module folder
   - See: pipeline_architecture.md, module_patterns.md

2. **No Mocks Policy** (CRITICAL)
   - All tests execute real code
   - Skip when deps unavailable
   - See: code_quality.md, testing_framework.md

3. **Safe-to-Fail Pattern** (CRITICAL)
   - Steps 8, 9, 12 never stop pipeline
   - See: error_handling.md

4. **Optional Dependencies** (IMPORTANT)
   - PyMDP, Flax, Julia are optional
   - JAX works without Flax
   - See: optional_dependencies.md, render_frameworks.md

5. **Exit Codes**
   - 0 = Success, 1 = Critical, 2 = Warnings
   - See: error_handling.md

---

## Maintenance

### When to Update
- Adding new pipeline steps
- Changing coding standards
- Adding new frameworks
- Modifying error handling
- Updating dependencies

### Update Procedure
1. Update specific documentation file
2. Update cross-references in INDEX.md
3. Update file statistics
4. Update README.md navigation if needed

---

**Last Updated**: December 2025  
**Total Files**: 16 (9 original + 7 new)  
**Total Lines**: ~6170  
**Status**: ✅ Complete and Production Ready

**December 2025 Updates:**
- ✅ Removed all legacy compatibility code and backwards-compatibility wrappers
- ✅ Updated terminology: "legacy pattern" → "older implementation pattern"
- ✅ code_quality.md - Detailed quality standards
- ✅ module_patterns.md - Advanced module patterns
- ✅ error_handling.md - Error strategies and safe-to-fail
- ✅ performance_optimization.md - Performance tuning
- ✅ optional_dependencies.md - Optional package handling
- ✅ render_frameworks.md - Framework documentation
- ✅ troubleshooting.md - Common issues guide
