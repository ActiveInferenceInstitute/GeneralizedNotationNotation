# GNN Architecture Guide

This guide details the architecture of the Generalized Notation Notation (GNN) system. It complements `DOCS.md` and `doc/pipeline/README.md` with an implementation-oriented perspective for developers.

**Last Updated**: 2026-03-23
**Version**: 1.3.0
**Status**: Maintained
**Pipeline Steps**: 25 (0-24)

## Principles

### Core Architectural Principles

- **Thin Orchestrators**: Numbered scripts delegate to modules, maintaining clear separation of concerns
- **Explicit Dependencies**: All module dependencies are explicitly declared and validated
- **Deterministic Outputs**: Pipeline runs produce identical results given identical inputs
- **Reproducible Runs**: Complete audit trail and environment capture for scientific reproducibility
- **Standardized Interfaces**: Consistent exit codes, logging, and configuration patterns across all modules

### Quality Assurance Principles

- **No Mock Testing**: All tests use real code paths and actual data dependencies
- **Comprehensive Coverage**: >95% test coverage with performance and integration validation
- **Real Data Processing**: No synthetic or placeholder data in tests or examples
- **Performance Standards**: Sub-30-minute execution time, <2GB memory usage for standard workloads
- **Error Rate Targets**: <1% critical failure rate, >99% step completion success rate

### Agent Architecture Principles

- **Specialized Capabilities**: Each module provides distinct, well-defined agent capabilities
- **Stateless Design**: Prefer stateless agents for better testability and reliability
- **Resource Awareness**: Proper resource management with cleanup and monitoring
- **Configuration Flexibility**: Environment-driven configuration with validation
- **Integration Ready**: All modules integrate seamlessly with the 25-step pipeline

## System Overview

The GNN system implements a comprehensive 25-step pipeline that transforms GNN model specifications into executable simulations, visualizations, and analysis. The architecture follows the thin orchestrator pattern with complete separation between pipeline orchestration and domain-specific implementations.

```mermaid
graph TB
  A["User/Researcher"] --> B["src/main.py<br/>Pipeline Orchestrator"]
  B --> C["25 Numbered Scripts<br/>(0_template.py → 24_intelligent_analysis.py)"]
  C --> D["Module set<br/>(see src/AGENTS.md)"]
  D --> E["Structured Outputs<br/>(output/step_N_output/)"]

  B --> F["Infrastructure Layer<br/>(utils/, pipeline/)"]
  F --> B

  D --> G["External Integrations<br/>(PyMDP, RxInfer.jl, JAX, ActiveInference.jl, PyTorch, NumPyro)"]
  D --> H["AI Services<br/>(OpenAI, Anthropic, Ollama)"]
  D --> I["Scientific Frameworks<br/>(JAX, DisCoPy, NetworkX)"]

  %% styling intentionally omitted (theme-controlled)
```

## Execution Flow (High-Level)

The GNN pipeline implements a sophisticated execution model with comprehensive monitoring, error recovery, and performance optimization.

```mermaid
sequenceDiagram
  participant U as User/Researcher
  participant M as src/main.py
  participant S as Step N Script
  participant Mod as Agent Module
  participant Out as output/*
  participant Ext as External Services

  U->>M: CLI arguments & config
  M->>S: Invoke with parsed args + context
  S->>Mod: Call domain-specific APIs
  Mod->>Ext: Optional external integrations
  Ext-->>Mod: Results or service responses
  Mod-->>S: Structured results + diagnostics
  S-->>M: Exit codes + performance metrics
  M->>Out: Comprehensive output artifacts
  M->>M: Update pipeline summary & health report
```

## Thin Orchestrator Pattern

- Orchestrators handle: argument parsing, logging setup, output dir, calling module APIs, summarizing results
- Modules implement: domain logic, IO, validations, transformations, rendering, execution

```mermaid
graph LR
  O["Orchestrator N_step.py"] -->|calls| API[("Module Public API")]
  API --> Impl["Internal Logic"]
  API --> IO["IO / Files"]
  Impl --> Artifacts["Artifacts in output/"]
```

## Module Dependencies

The GNN pipeline implements a sophisticated dependency graph that ensures proper execution order and data flow between modules.

```mermaid
graph TD
  GNN[gnn] --> TYPE[type_checker]
  GNN --> REG[model_registry]
  GNN --> VAL[validation]
  GNN --> EXP[export]
  GNN --> ONT[ontology]
  GNN --> RENDER[render]
  GNN --> LLM[llm]
  GNN --> AUDIO[audio]
  GNN --> RES[research]
  GNN --> GUI[gui]

  TYPE --> VAL
  VAL --> EXP
  EXP --> VIS[visualization]
  EXP --> AV[advanced_visualization]

  RENDER --> EXEC[execute]
  RENDER --> SEC[security]

  EXEC --> ANA[analysis]
  VIS --> ANA
  LLM --> ANA
  ML[ml_integration] --> ANA

  ANA --> REP[report]
  REP --> IA[intelligent_analysis]

  GNN --> INT[integration]
  GNN --> WEB[website]
  GNN --> MCP[mcp]

  %% styling intentionally omitted (theme-controlled)
```

## Current Implementation Status

### Components

**Core Infrastructure:**

- `src/main.py` - Main pipeline orchestrator with comprehensive monitoring
- `src/utils/` - Complete utility library with logging, validation, and monitoring
- `src/pipeline/` - Full pipeline configuration and management system
- `src/tests/` - Comprehensive test suite with real data validation

**Agent Modules:**

- All 25 pipeline steps (0-24) implemented with thin orchestrator pattern
- Agent/module coverage is tracked in `src/AGENTS.md`
- Complete MCP integration across all applicable modules
- Full test coverage with >95% coverage for all modules

**Documentation:**

- `AGENTS.md` - Master agent scaffolding documentation
- `AGENTS_TEMPLATE.md` - Template for new modules
- `.agent_rules` - Development guidelines
- `.env` (optional local file) - local environment values when needed
- `.gitignore` - Comprehensive ignore patterns for scientific computing

### Latest Status

- The architecture uses thin orchestrators over modular implementations.
- Pipeline, docs, and tests are maintained together.
- Use current runs and reports for exact performance and pass metrics.

## Logging Architecture

```mermaid
flowchart TD
  A["setup_step_logging"] --> B["Module Logger"]
  B --> C["Per-step Logs"]
  C --> D["Aggregated Pipeline Logs"]
```

## Configuration Flow

```mermaid
flowchart LR
  CLI["CLI Args"] --> Merge
  CFG["config.yaml"] --> Merge
  ENV["Env Vars"] --> Merge
  Merge["Configuration Resolver"] --> Eff["Effective Config"]
  Eff --> Steps["Steps 0..24"]
```

## Output Management

- Each step writes to `output/<step_subdir>/`
- `get_output_dir_for_script()` ensures consistent paths
- Site and reports summarize artifacts across steps

## Error Handling

- Exit codes: 0=success, 1=critical error, 2=success with warnings
- Continuation policy controlled via config (fail-fast vs continue)
- Rich diagnostics persisted alongside artifacts

```mermaid
flowchart LR
  S["Step Start"] --> V{Valid?}
  V -- no --> E["Log + Diagnostics"]
  E --> P{Fail-fast?}
  P -- yes --> STOP["Abort"]
  P -- no --> CONT["Continue"]
  V -- yes --> RUN["Run"]
  RUN --> X{Exit Code}
  X -->|0| OK["Success"]
  X -->|2| WARN["Success+Warnings"]
  X -->|1| E
```

## Extension Pattern

Adding new pipeline steps and modules follows a well-established pattern that ensures consistency and maintainability:

### 1. **Plan the Extension**

- Define the module's purpose and integration points
- Identify dependencies on existing modules
- Determine resource requirements and performance characteristics
- Plan comprehensive test coverage and documentation

### 2. **Create Thin Orchestrator**

- Implement `src/N_newstep.py` following the thin orchestrator pattern
- Handle argument parsing, logging setup, and output directory management
- Delegate all domain logic to the module implementation
- Return standardized exit codes (0=success, 1=critical error, 2=success with warnings)

### 3. **Implement Agent Module**

- Create `src/newstep/` directory with complete module structure
- Implement `__init__.py` with public API exports
- Create core logic in `processor.py` or appropriately named files
- Add `mcp.py` for Model Context Protocol integration (if applicable)
- Include comprehensive error handling and resource management

### 4. **Add Comprehensive Testing**

- Create integration tests in `src/tests/test_newstep_integration.py`
- Implement unit tests for all public functions
- Add performance tests with timing and memory validation
- Include error scenario testing with real failure modes
- Ensure >95% test coverage

### 5. **Document Completely**

- Create comprehensive `AGENTS.md` using the enhanced template
- Document all public APIs with examples and error conditions
- Include troubleshooting guide and performance characteristics
- Add usage examples for common scenarios
- Update pipeline documentation and cross-references

### 6. **Validate and Review**

- Test the complete integration with existing pipeline
- Validate performance against established standards
- Ensure compliance with all coding standards and patterns
- Review security implications and access controls
- Update configuration files and environment templates as needed

## Agent Architecture Deep Dive

The GNN system implements a sophisticated multi-agent architecture where each module provides specialized capabilities:

### 🤖 **Agent Types and Capabilities**

**Processing Agents** (Steps 3-9):

- **GNN Agent**: Multi-format parsing and semantic analysis
- **Type Checker Agent**: Static validation and resource estimation
- **Validation Agent**: Consistency checking and constraint verification
- **Export Agent**: Multi-format data transformation and serialization
- **Visualization Agent**: Graph generation and matrix visualization
- **Advanced Visualization Agent**: Interactive plots and 3D graphics

**Simulation Agents** (Steps 10-16):

- **Ontology Agent**: Active Inference knowledge processing and mapping
- **Render Agent**: Multi-framework code generation and optimization
- **Execute Agent**: Cross-platform simulation execution and monitoring
- **LLM Agent**: AI-enhanced analysis and natural language processing
- **ML Integration Agent**: Machine learning model training and evaluation
- **Audio Agent**: Multi-backend audio generation and sonification
- **Analysis Agent**: Advanced statistical processing and performance analysis

**Integration Agents** (Steps 17-23):

- **Integration Agent**: Cross-module coordination and data flow management
- **Security Agent**: Input validation, access control, and threat detection
- **Research Agent**: Experimental tools and workflow management
- **Website Agent**: Static site generation and documentation compilation
- **MCP Agent**: Protocol compliance and tool registration
- **GUI Agent**: Interactive interface generation and user experience
- **Report Agent**: Comprehensive analysis reporting and visualization

### 🔧 **Agent Coordination Mechanisms**

**Dependency Resolution**: Automatic dependency detection and inclusion
**Resource Management**: Coordinated resource allocation and cleanup
**Error Propagation**: Structured error handling with graceful degradation
**Performance Monitoring**: Cross-agent performance tracking and optimization
**Configuration Management**: Centralized configuration with module-specific overrides

### 📊 **Agent Performance Framework**

Each agent implements comprehensive performance monitoring:

- **Resource Tracking**: Memory, CPU, and disk usage monitoring
- **Execution Timing**: Detailed timing for all major operations
- **Error Metrics**: Success rates, failure modes, and recovery statistics
- **Integration Health**: Dependency health and communication status
- **Scalability Metrics**: Performance characteristics across different input sizes

## References

- **Main Documentation**: [README.md](README.md) — Project overview and quick start
- **Pipeline Documentation**: [doc/PIPELINE_SCRIPTS.md](doc/PIPELINE_SCRIPTS.md) — Detailed step-by-step descriptions
- **Development Rules**: [.agent_rules](.agent_rules) — Canonical rules for scripts and modules
- **Agent Registry**: [AGENTS.md](AGENTS.md) — Master agent scaffolding and module registry
- **Template Guide**: [AGENTS_TEMPLATE.md](AGENTS_TEMPLATE.md) — Template for new modules
- **Doc link checks**: [DOCS.md](DOCS.md) — “Documentation maintenance” (`doc/development/docs_audit.py`)

---

**Architecture Version**: 1.3.0
**Last Updated**: 2026-03-23
**Status**: ✅ Production Ready
**Compliance**: Thin orchestrator pattern
**Latest Validation**: See current test and pipeline runs
