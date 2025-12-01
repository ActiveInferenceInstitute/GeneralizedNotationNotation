# .cursor_rules - Comprehensive AI Development Guidelines

**Version**: 2.2.0  
**Status**: ✅ Production Ready  
**Last Updated**: December 2025  
**Total Documentation**: 6100+ lines across 16 files

---

## Quick Navigation

This directory contains comprehensive architectural, implementation, and quality guidelines for the GNN Processing Pipeline. Use this README to understand what's documented where.

### By Task

| Task | Start Here |
|------|-----------|
| **Writing a new pipeline script** | [implementation_patterns.md](implementation_patterns.md) |
| **Creating a new module** | [module_patterns.md](module_patterns.md) |
| **Writing tests** | [testing_framework.md](testing_framework.md) |
| **Adding MCP tools** | [mcp_integration.md](mcp_integration.md) |
| **Understanding GNN specs** | [gnn_standards.md](gnn_standards.md) |
| **Improving code quality** | [code_quality.md](code_quality.md) |
| **Handling errors** | [error_handling.md](error_handling.md) |
| **Performance optimization** | [performance_optimization.md](performance_optimization.md) |
| **Optional dependencies** | [optional_dependencies.md](optional_dependencies.md) |
| **Render frameworks** | [render_frameworks.md](render_frameworks.md) |
| **Troubleshooting issues** | [troubleshooting.md](troubleshooting.md) |

### By Topic

**Architecture & Design**
- [pipeline_architecture.md](pipeline_architecture.md) - 24-step pipeline, thin orchestrator pattern
- [implementation_patterns.md](implementation_patterns.md) - Code patterns, standardized scripts
- [module_patterns.md](module_patterns.md) - Advanced module patterns, dependencies

**Quality & Testing**
- [code_quality.md](code_quality.md) - Type safety, documentation, testing standards
- [testing_framework.md](testing_framework.md) - Test architecture, fixtures, execution
- [quality_and_dev.md](quality_and_dev.md) - Development guidelines, QA standards

**Error Handling & Performance**
- [error_handling.md](error_handling.md) - Error strategies, safe-to-fail, recovery
- [performance_optimization.md](performance_optimization.md) - Performance tuning, profiling

**Domain Knowledge**
- [gnn_standards.md](gnn_standards.md) - GNN specifications, Active Inference
- [render_frameworks.md](render_frameworks.md) - PyMDP, JAX, RxInfer, DisCoPy
- [optional_dependencies.md](optional_dependencies.md) - Handling optional packages

**Operations**
- [mcp_integration.md](mcp_integration.md) - Model Context Protocol, tool registration
- [troubleshooting.md](troubleshooting.md) - Common issues and solutions

**Reference**
- [AGENTS.md](AGENTS.md) - 28 agent capabilities summary
- [INDEX.md](INDEX.md) - Complete documentation index

---

## File Organization

```
.cursor_rules/
├── README.md                     # This navigation guide
├── INDEX.md                      # Master index with cross-references
│
├── # Architecture
├── pipeline_architecture.md      # 24-step pipeline structure
├── implementation_patterns.md    # Code patterns, standardized scripts
├── module_patterns.md            # Module architecture patterns ⭐ NEW
│
├── # Quality
├── code_quality.md               # Quality standards ⭐ NEW
├── quality_and_dev.md            # QA overview
├── testing_framework.md          # Testing infrastructure
│
├── # Error & Performance
├── error_handling.md             # Error strategies ⭐ NEW
├── performance_optimization.md   # Performance tuning ⭐ NEW
│
├── # Domain
├── gnn_standards.md              # GNN specifications
├── render_frameworks.md          # Framework documentation ⭐ NEW
├── optional_dependencies.md      # Optional packages ⭐ NEW
│
├── # Integration
├── mcp_integration.md            # MCP tools
├── troubleshooting.md            # Common issues ⭐ NEW
│
└── # Reference
    └── AGENTS.md                 # Agent capabilities
```

---

## Key Concepts Quick Reference

### Thin Orchestrator Pattern (CRITICAL)

Numbered scripts (`N_module.py`) are thin orchestrators:
- All core logic in module folders (`src/module_name/`)
- Scripts handle: argument parsing, logging, output management
- Modules handle: domain logic, processing, algorithms

**Reference**: [pipeline_architecture.md](pipeline_architecture.md), [module_patterns.md](module_patterns.md)

### No Mocks Policy (CRITICAL)

All tests execute real code paths:
- No `unittest.mock` or monkeypatching
- Skip gracefully when dependencies unavailable
- Use `@pytest.mark.safe_to_fail` for optional features

**Reference**: [code_quality.md](code_quality.md), [testing_framework.md](testing_framework.md)

### Safe-to-Fail Pattern (CRITICAL)

Steps 8, 9, 12 never stop the pipeline:
- Multiple fallback levels
- Always return exit code 0
- Generate fallback reports on failure

**Reference**: [error_handling.md](error_handling.md)

### Exit Code Conventions

| Code | Meaning | Pipeline Behavior |
|------|---------|-------------------|
| 0 | Success | Continue |
| 1 | Critical Error | Stop pipeline |
| 2 | Success with Warnings | Continue |

**Reference**: [error_handling.md](error_handling.md)

### JAX Without Flax

JAX renderer generates pure JAX code:
- No Flax dependency required
- Only needs: `jax`, `jaxlib`, `optax`
- Pure functional approach

**Reference**: [render_frameworks.md](render_frameworks.md), [optional_dependencies.md](optional_dependencies.md)

---

## Getting Started Workflows

### New Developer
1. Read [AGENTS.md](AGENTS.md) - Understand 28 modules
2. Read [pipeline_architecture.md](pipeline_architecture.md) - Understand flow
3. Read [gnn_standards.md](gnn_standards.md) - Understand domain
4. Ready to fix bugs and write tests

### Adding Features
1. Read [module_patterns.md](module_patterns.md) - Module architecture
2. Read [implementation_patterns.md](implementation_patterns.md) - Code patterns
3. Read [code_quality.md](code_quality.md) - Quality requirements
4. Read [testing_framework.md](testing_framework.md) - Testing patterns

### Debugging Issues
1. Check [troubleshooting.md](troubleshooting.md) - Common issues
2. Check [error_handling.md](error_handling.md) - Error patterns
3. Check [optional_dependencies.md](optional_dependencies.md) - Dependency issues

---

## Environment Setup

### Using uv (Recommended)

```bash
# Install dependencies
uv pip install -e .

# Run pipeline
uv run python src/main.py --verbose

# Run tests
uv run pytest src/tests/ -v

# Run specific step
uv run python src/11_render.py --target-dir input/gnn_files
```

### Quick Commands

```bash
# Full pipeline
python src/main.py --verbose

# Specific steps
python src/main.py --only-steps "3,11,12"

# Skip steps
python src/main.py --skip-steps "2"

# Run tests
python src/2_tests.py --fast-only

# Check environment
python src/1_setup.py --verbose
```

---

## Documentation Statistics

| File | Lines | Focus |
|------|-------|-------|
| pipeline_architecture.md | ~300 | Pipeline structure |
| implementation_patterns.md | ~750 | Code patterns |
| module_patterns.md | ~350 | Module architecture |
| code_quality.md | ~320 | Quality standards |
| quality_and_dev.md | ~120 | QA overview |
| testing_framework.md | ~930 | Testing |
| error_handling.md | ~400 | Error strategies |
| performance_optimization.md | ~350 | Performance |
| gnn_standards.md | ~330 | GNN domain |
| render_frameworks.md | ~400 | Frameworks |
| optional_dependencies.md | ~280 | Dependencies |
| mcp_integration.md | ~360 | MCP tools |
| troubleshooting.md | ~300 | Debugging |
| AGENTS.md | ~550 | Agent capabilities |
| **TOTAL** | **~6100+** | **Complete guide** |

---

## Critical Standards Summary

### Type Safety
- Complete type hints on all public functions
- Generic types for containers (`Dict[str, Any]`, `List[str]`)
- Reference: [code_quality.md](code_quality.md)

### Documentation
- Comprehensive docstrings with examples
- Argument and return documentation
- Reference: [code_quality.md](code_quality.md)

### Testing
- >95% coverage for critical paths
- Real implementations, no mocks
- Reference: [testing_framework.md](testing_framework.md)

### Performance
- Full pipeline: <5 minutes typical
- Peak memory: <500MB typical
- Reference: [performance_optimization.md](performance_optimization.md)

### Error Handling
- Graceful degradation for optional dependencies
- Actionable error messages
- Reference: [error_handling.md](error_handling.md)

---

## Recent Enhancements (December 2025)

### New Documentation Files
- **code_quality.md** - Detailed code quality standards
- **module_patterns.md** - Advanced module architecture
- **error_handling.md** - Error strategies and safe-to-fail
- **performance_optimization.md** - Performance tuning guide
- **optional_dependencies.md** - Optional package handling
- **render_frameworks.md** - Framework-specific documentation
- **troubleshooting.md** - Common issues and solutions

### Key Updates
- JAX renderer now generates pure JAX (no Flax)
- RxInfer setup script added
- Comprehensive safe-to-fail documentation
- Updated performance baselines from latest run

---

## Version History

- **v2.2.0** (Dec 2025) - Added 7 new documentation files, comprehensive coverage
- **v2.1.0** (Nov 2025) - Enhanced documentation with performance & MCP
- **v2.0.0** (Oct 2025) - Major restructuring with agent documentation
- **v1.0.0** (Sep 2025) - Initial comprehensive guidelines

---

**Status**: ✅ Complete and Production Ready  
**Total Files**: 16  
**Total Lines**: 6100+  
**Last Updated**: December 2025
