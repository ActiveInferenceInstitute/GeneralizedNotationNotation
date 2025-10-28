# GNN Error Taxonomy

> **📋 Document Metadata**  
> **Type**: Reference Guide | **Audience**: Users, Developers | **Complexity**: Intermediate  
> **Last Updated: October 2025 | **Status**: Production-Ready  
> **Cross-References**: [Common Errors](common_errors.md) | [Debugging Workflows](debugging_workflows.md) | [API Reference](../api/README.md)

## Overview

This document provides a systematic classification of all error types in the GNN (Generalized Notation Notation) system, enabling rapid diagnosis and resolution of issues.

## Error Classification System

### 1. **Syntax Errors** (Code: SYN-xxx)

**Definition**: Errors in GNN file parsing and syntax validation

#### 1.1 File Structure Errors (SYN-100-199)
| Error Code | Error Type | Description | Common Causes |
|------------|------------|-------------|---------------|
| SYN-101 | Missing Required Section | Required GNN section not found | Missing `## ModelName`, `## StateSpaceBlock` |
| SYN-102 | Invalid Section Header | Section header format incorrect | Wrong `#` count, typos in section names |
| SYN-103 | Malformed Metadata Block | Document metadata syntax error | Missing `>`, incorrect YAML format |
| SYN-104 | Encoding Issues | File encoding problems | Non-UTF-8 characters, BOM presence |

**Example**:
```bash
# Error SYN-101
ERROR: Missing required section 'ModelName' in file: my_model.md
LINE: N/A
SOLUTION: Add ## ModelName section with model identifier
```

#### 1.2 Variable Declaration Errors (SYN-200-299)
| Error Code | Error Type | Description | Common Causes |
|------------|------------|-------------|---------------|
| SYN-201 | Invalid Variable Name | Variable naming convention violation | Spaces, special chars, reserved words |
| SYN-202 | Malformed Dimensions | Dimension specification syntax error | Missing brackets, non-numeric values |
| SYN-203 | Invalid Variable Type | Unsupported variable type | Typos in `categorical`, `continuous`, `binary` |
| SYN-204 | Duplicate Variable | Variable declared multiple times | Copy-paste errors, naming conflicts |

**Example**:
```bash
# Error SYN-201
ERROR: Invalid variable name 's_factor 0' contains spaces
LINE: 15: s_factor 0[2,1,type=categorical]
SOLUTION: Use underscores: s_factor_0[2,1,type=categorical]
```

#### 1.3 Connection Specification Errors (SYN-300-399)
| Error Code | Error Type | Description | Common Causes |
|------------|------------|-------------|---------------|
| SYN-301 | Invalid Connection Syntax | Connection notation malformed | Wrong arrows, missing variables |
| SYN-302 | Undefined Variable Reference | Connection references unknown variable | Typos, undeclared variables |
| SYN-303 | Circular Dependency | Self-referential connections | Poor model design |
| SYN-304 | Mixed Connection Types | Inconsistent connection notation | Mixing `>`, `->`, `-` incorrectly |

### 2. **Validation Errors** (Code: VAL-xxx)

**Definition**: Errors in model consistency and mathematical validation

#### 2.1 Dimension Mismatch Errors (VAL-100-199)
| Error Code | Error Type | Description | Common Causes |
|------------|------------|-------------|---------------|
| VAL-101 | Matrix Dimension Mismatch | A/B/C/D matrix dimensions incorrect | Wrong state/action/observation counts |
| VAL-102 | Incompatible Variable Dimensions | Variables with incompatible sizes | Inconsistent factor declarations |
| VAL-103 | Missing Matrix Elements | Required matrix elements not specified | Incomplete parameterization |
| VAL-104 | Non-Stochastic Matrix | Probability matrix doesn't sum to 1 | Normalization errors |

**Example**:
```bash
# Error VAL-101
ERROR: A matrix dimension mismatch
EXPECTED: A_m0[2,3] for 2 observations × 3 states
ACTUAL: A_m0[2,2] specified in InitialParameterization
LINE: 45: A_m0 = [[0.9, 0.1], [0.1, 0.9]]
SOLUTION: Add third column or reduce state space to 2
```

#### 2.2 Mathematical Consistency Errors (VAL-200-299)
| Error Code | Error Type | Description | Common Causes |
|------------|------------|-------------|---------------|
| VAL-201 | Probability Violation | Probabilities outside [0,1] range | Negative values, values > 1 |
| VAL-202 | Normalization Error | Probability distributions don't sum to 1 | Manual calculation errors |
| VAL-203 | Incompatible Time Settings | Time specification conflicts | Static model with dynamics |
| VAL-204 | Ontology Mapping Error | Invalid Active Inference ontology terms | Typos, deprecated terms |

### 3. **Runtime Errors** (Code: RUN-xxx)

**Definition**: Errors during pipeline execution and processing

#### 3.1 Pipeline Execution Errors (RUN-100-199)
| Error Code | Error Type | Description | Common Causes |
|------------|------------|-------------|---------------|
| RUN-101 | Step Timeout | Pipeline step exceeded time limit | Large models, resource constraints |
| RUN-102 | Memory Exhaustion | Insufficient memory for processing | Large state spaces, memory leaks |
| RUN-103 | Dependency Missing | Required software not installed | Missing Python packages, Julia |
| RUN-104 | Permission Denied | File system access denied | Wrong file permissions, protected dirs |

#### 3.2 Resource Errors (RUN-200-299)
| Error Code | Error Type | Description | Common Causes |
|------------|------------|-------------|---------------|
| RUN-201 | Disk Space Insufficient | Not enough disk space for outputs | Large visualization files |
| RUN-202 | CPU Limit Exceeded | Computation too intensive | Exponential complexity models |
| RUN-203 | Network Timeout | External service unavailable | LLM API calls, package downloads |
| RUN-204 | Environment Setup Failed | Virtual environment creation failed | Python version conflicts |

### 4. **Integration Errors** (Code: INT-xxx)

**Definition**: Framework-specific integration and code generation errors

#### 4.1 PyMDP Integration Errors (INT-100-199)
| Error Code | Error Type | Description | Common Causes |
|------------|------------|-------------|---------------|
| INT-101 | PyMDP Import Error | Cannot import pymdp modules | Package not installed, version mismatch |
| INT-102 | Matrix Conversion Error | GNN to PyMDP matrix conversion failed | Incompatible dimensions |
| INT-103 | Agent Creation Error | PyMDP agent instantiation failed | Invalid parameters |
| INT-104 | Simulation Runtime Error | Generated PyMDP code fails to run | Logic errors, parameter issues |

#### 4.2 RxInfer Integration Errors (INT-200-299)
| Error Code | Error Type | Description | Common Causes |
|------------|------------|-------------|---------------|
| INT-201 | Julia Not Found | Julia executable not in PATH | Julia not installed |
| INT-202 | RxInfer Package Error | RxInfer.jl package issues | Package not installed, version conflict |
| INT-203 | TOML Generation Error | Configuration file creation failed | Invalid parameter values |
| INT-204 | Julia Execution Error | Generated Julia code fails | Syntax errors, runtime issues |

#### 4.3 DisCoPy Integration Errors (INT-300-399)
| Error Code | Error Type | Description | Common Causes |
|------------|------------|-------------|---------------|
| INT-301 | DisCoPy Import Error | Cannot import discopy modules | Package installation issues |
| INT-302 | Diagram Creation Error | Categorical diagram generation failed | Unsupported model structures |
| INT-303 | JAX Backend Error | JAX evaluation backend issues | JAX/JAXlib installation problems |
| INT-304 | Composition Error | Diagram composition failed | Incompatible diagram types |

### 5. **Performance Issues** (Code: PERF-xxx)

**Definition**: Performance degradation and optimization opportunities

#### 5.1 Memory Performance (PERF-100-199)
| Error Code | Issue Type | Description | Common Causes |
|------------|------------|-------------|---------------|
| PERF-101 | High Memory Usage | Memory consumption above threshold | Large state spaces, memory leaks |
| PERF-102 | Memory Growth | Increasing memory usage over time | Accumulating intermediate results |
| PERF-103 | Swap Usage | System using swap memory | Insufficient RAM for model size |
| PERF-104 | Garbage Collection | Frequent GC causing slowdown | Object lifecycle management |

#### 5.2 Computational Performance (PERF-200-299)
| Error Code | Issue Type | Description | Common Causes |
|------------|------------|-------------|---------------|
| PERF-201 | Slow Matrix Operations | Matrix computations taking too long | Large matrices, inefficient algorithms |
| PERF-202 | Exponential Complexity | Processing time growing exponentially | Combinatorial explosion |
| PERF-203 | I/O Bottleneck | File operations causing delays | Large files, slow storage |
| PERF-204 | Network Latency | Remote API calls causing delays | LLM services, package downloads |

## Error Resolution Workflow

### Quick Diagnosis Steps
1. **Identify Error Category**: Check error code prefix (SYN, VAL, RUN, INT, PERF)
2. **Locate Error Context**: Find file, line number, and section
3. **Check Prerequisites**: Verify dependencies and environment
4. **Review Documentation**: Consult specific error guide
5. **Apply Fix**: Implement recommended solution
6. **Validate Fix**: Re-run pipeline to confirm resolution

### Error Priority Matrix
| Severity | Impact | Example Errors | Response Time |
|----------|--------|----------------|---------------|
| **Critical** | Pipeline halts | SYN-101, RUN-103, INT-101 | Immediate |
| **High** | Major functionality affected | VAL-101, RUN-101, INT-201 | < 1 hour |
| **Medium** | Minor functionality affected | SYN-204, VAL-204, PERF-101 | < 1 day |
| **Low** | Performance/cosmetic | PERF-204, SYN-103 | Next release |

## Advanced Debugging Techniques

### Error Context Collection
```bash
# Collect comprehensive error context
python src/main.py --verbose --debug-mode \
  --save-error-state \
  --target-dir problematic_model.md \
  > debug_output.log 2>&1
```

### Interactive Debugging
```python
# Python debugging for complex errors
import pdb; pdb.set_trace()

# Inspect model state
from src.gnn import GNNModel
model = GNNModel.from_file("problematic_model.md")
print(f"Variables: {model.state_space}")
print(f"Connections: {model.connections}")
```

### Performance Profiling
```bash
# Profile memory usage
python -m memory_profiler src/main.py --target-dir large_model.md

# Profile CPU usage  
python -m cProfile -o profile_output.prof src/main.py
python -m pstats profile_output.prof
```

## Prevention Strategies

### Development Best Practices
1. **Use Templates**: Start with validated templates
2. **Incremental Development**: Add complexity gradually
3. **Regular Validation**: Run type checker frequently
4. **Version Control**: Track changes systematically
5. **Testing**: Validate with multiple examples

### Automated Error Prevention
```bash
# Pre-commit validation
git add model.md
python src/main.py --only-steps 4 --strict --target-dir model.md

# Continuous integration
python scripts/validate_all_examples.py
```

---

## Related Documentation

- **[Common Errors](common_errors.md)**: Specific error scenarios with solutions
- **[Debugging Workflows](debugging_workflows.md)**: Step-by-step debugging procedures
- **[Performance Guide](../performance/README.md)**: Performance optimization strategies
- **[API Error Reference](api_error_reference.md)**: Programmatic error handling

---

**Last Updated: October 2025  
**Error Taxonomy Version**: 1.0  
**Coverage**: 45 error types across 5 categories  
**Status**: Production-Ready 