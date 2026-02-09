# GNN Technical Documentation Agents

**Version**: v1.1.0  
**Last Updated**: February 9, 2026  
**Status**: ✅ Production Ready  
**Test Count**: 1,127 Tests Passing  

---

## 📚 Documentation Agent Registry

The GNN documentation system consists of 24 specialized agents, each providing focused expertise:

### Core Foundation Agents (4 modules)

#### 1. **GNN Overview Agent** (`gnn_overview.md`)

**Purpose**: High-level introduction to GNN and its role in Active Inference  
**Audience**: Everyone (entry point for new users)  
**Key Topics**:

- What is GNN and why it matters
- The Triple Play approach (text, graphical, executable models)
- GNN in the Active Inference ecosystem
- Links to 25-step pipeline

**Entry Point Usage**:

```bash
# Start here for complete overview
python src/main.py --target-dir input/gnn_files --verbose
```

#### 2. **Quick Start Tutorial Agent** (`quickstart_tutorial.md`)

**Purpose**: Get new users to build their first GNN model in 15 minutes  
**Audience**: Beginners, students  
**Key Topics**:

- Step-by-step first model creation
- Navigation agent in a 2x2 grid
- Running through the pipeline
- Common first questions

#### 3. **GNN Overview Agent (About)** (`about_gnn.md`)

**Purpose**: Detailed GNN specification and motivation  
**Audience**: Researchers, implementers  
**Key Topics**:

- Detailed motivation for GNN
- Complete file structure specification
- GNN processing pipeline
- Progressive model development

#### 4. **README Index Agent** (`README.md`)

**Purpose**: Navigation hub for all GNN documentation  
**Audience**: All users  
**Key Topics**:

- Quick start (3 entry points)
- Complete module registry (28 modules)
- Search by task or audience
- Documentation standards

---

### Language Specification Agents (4 modules)

#### 5. **GNN Syntax Reference Agent** (`gnn_syntax.md`)

**Purpose**: Quick reference for GNN syntax with working examples  
**Audience**: All users (reference)  
**Key Topics**:

- Variable declarations
- Subscripts and superscripts
- Connection syntax
- Dimensions and types
- Time specifications

**Related Pipeline Steps**:

- Step 3: `src/3_gnn.py` - GNN parsing
- Step 5: `src/5_type_checker.py` - Syntax validation

#### 6. **GNN DSL Manual Agent** (`gnn_dsl_manual.md`)

**Purpose**: Complete Domain-Specific Language specification  
**Audience**: Developers, power users  
**Key Topics**:

- File structure specification
- Section descriptions (all 10 sections)
- Detailed syntax for each section
- Parser behavior documentation

**Implementation Reference**: `src/gnn/parser.py`

#### 7. **GNN Schema Agent** (`gnn_schema.md`)

**Purpose**: Parsing and validation schemas for GNN processing  
**Audience**: Developers, pipeline maintainers  
**Key Topics**:

- Core schema components
- Round-trip data flow (GNN → JSON → Formats → Code)
- Core method locations
- Cross-module data dependencies
- Validation schema definitions

**Implementation**:

- Parsing: `src/gnn/multi_format_processor.py`
- Validation: `src/type_checker/analysis_utils.py`

#### 8. **GNN File Structure Agent** (`gnn_file_structure_doc.md`)

**Purpose**: Complete guide to GNN file organization and sections  
**Audience**: Model creators, beginners  
**Key Topics**:

- 11 required sections (detailed)
- File structure visualization
- Progressive model development patterns
- Best practices for GNN files
- Complete example GNN file

**Related Documentation**: Section 1 of DSL Manual

---

### Modeling and Learning Agents (4 modules)

#### 9. **GNN Examples Agent** (`gnn_examples_doc.md`)

**Purpose**: Model progression from simple to complex with step-by-step examples  
**Audience**: Learners, researchers  
**Key Topics**:

- Static perception model (simplest)
- Dynamic perception model
- Dynamic with policy selection
- Dynamic with flexible policy (most complex)
- Comparative analysis tables

**Example Usage**:

```bash
python src/main.py --target-dir doc/gnn/examples/ --verbose
```

#### 10. **Advanced Modeling Patterns Agent** (`advanced_modeling_patterns.md`)

**Purpose**: Sophisticated Active Inference modeling techniques  
**Audience**: Researchers, advanced users  
**Key Topics**:

- Hierarchical modeling (temporal and spatial)
- Multi-agent systems (distributed, leader-follower)
- Learning and adaptation patterns
- Temporal dynamics (predictive coding, memory)
- Uncertainty and robustness
- Compositional modeling
- Domain-specific patterns (social, language)

**Prerequisites**: Completion of basic examples

#### 11. **Multi-Agent Systems Agent** (`gnn_multiagent.md`)

**Purpose**: Multi-agent system specification and simulation  
**Audience**: Researchers in multi-agent systems  
**Key Topics**:

- Agent definition and heterogeneity
- AgentsBlock specification
- Communication mechanisms (broadcast, P2P, shared memory)
- Simulation semantics and execution models
- Ontology integration for MAS
- Example: Two Greeter Agents

**Pipeline Integration**: Steps 3, 11, 12, 16

#### 12. **GNN Standards Agent** (`gnn_standards.md`)

**Purpose**: Domain knowledge and coding standards  
**Audience**: Developers, maintainers  
**Key Topics**:

- Pipeline processing standards
- Thin orchestrator pattern
- GNN file structure understanding
- GNN syntax and punctuation
- Module architecture requirements

---

### Implementation and Integration Agents (5 modules)

#### 13. **GNN Implementation Guide Agent** (`gnn_implementation.md`)

**Purpose**: Guidelines for implementing GNN models in computational environments  
**Audience**: Developers, practitioners  
**Key Topics**:

- Implementation workflow (6 steps)
- Parsing GNN files (Python example)
- Setting up variables (NumPy)
- Establishing model structure
- Implementing equations
- Testing and validation
- Model refinement
- Framework-specific examples (PyTorch, Julia)
- Visualization techniques

**Example Implementations**:

- Static perception model
- Dynamic perception model
- PyTorch implementation
- Julia implementation

#### 14. **Framework Integration Guide Agent** (`framework_integration_guide.md`)

**Purpose**: Integration patterns for Active Inference frameworks  
**Audience**: Framework developers, advanced users  
**Key Topics**:

- Integration architecture overview
- Common configuration interface
- Unified result format
- PyMDP integration (detailed with templates)
- RxInfer integration (Julia models)
- DisCoPy integration (category theory)
- Performance benchmarking framework
- Quality assurance and integration testing

**Pipeline Integration**: Steps 11 and 12

#### 15. **Architecture Reference Agent** (`architecture_reference.md`)

**Purpose**: Implementation details of thin orchestrator pattern and cross-module integration  
**Audience**: Architects, maintainers  
**Key Topics**:

- Complete 25-step pipeline mapping
- Thin orchestrator pattern definition
- Pattern implementation examples
- Cross-module data flow details
- Module structure analysis (actual locations)
- Framework integration points (PyMDP, RxInfer, DisCoPy)
- Pipeline orchestration details
- Dependency resolution patterns
- MCP integration patterns

**Key Example**: Step 8 Visualization implementation with cross-references

#### 16. **Technical Reference Agent** (`technical_reference.md`)

**Purpose**: Round-trip data flow and complete entry points  
**Audience**: Developers, pipeline maintainers  
**Key Topics**:

- All 25 pipeline entry points documented
- Stage 1: GNN → Parsed JSON (Step 3)
- Stage 2: JSON → Typed JSON (Step 5)
- Stage 3: JSON → Multiple Formats (Step 7)
- Stage 4: JSON → Framework Code (Step 11)
- Stage 5: Code → Execution Results (Step 12)
- Cross-module communication patterns
- Output directory management

#### 17. **Resource Metrics Agent** (`resource_metrics.md`)

**Purpose**: Computational resource estimation and metrics  
**Audience**: Researchers, system designers  
**Key Topics**:

- General metrics (memory, inference cost, storage)
- Complexity metrics
- Dimensional complexity scores
- Variable count impact
- Connection density analysis
- Scalability predictions

**Pipeline Integration**: Step 5 (Type Checker)  
**Output**: `resource_metrics.json` and `resource_report.md`

---

### Advanced Topics Agents (4 modules)

#### 18. **LLM and Neurosymbolic Active Inference Agent** (`gnn_llm_neurosymbolic_active_inference.md`)

**Purpose**: Integration of LLMs with GNN for neurosymbolic cognitive agents  
**Audience**: Researchers in AI, neuroscience, cognitive science  
**Key Topics**:

- GNN as symbolic backbone
- LLM for semantic richness and processing
- Neurosymbolic AI architecture
- Free Energy Principle in detail
- Active Inference loop (perception, learning, action)
- Detailed neurosymbolic integration
- Use case examples (robotics, decision support, education)
- Advantages and challenges
- Future research directions

**Pipeline Integration**: Step 13 (LLM analysis)

**Theoretical Grounding**:

- Free Energy Principle (FEP)
- Active Inference framework
- Variational Free Energy (VFE) mathematics
- Expected Free Energy (EFE)

#### 19. **Ontology System Agent** (`ontology_system.md`)

**Purpose**: Active Inference Ontology integration and annotation  
**Audience**: Researchers, knowledge engineers  
**Key Topics**:

- Ontology terms definition file
- Usage in GNN files (ActInfOntologyAnnotation)
- Validation process and reporting
- MCP integration for ontology functions
- Example ontology mappings

**Pipeline Integration**: Step 10 (Ontology processing)  
**Output**: `ontology_processing_report.md`

#### 20. **Improvement Analysis Agent** (`improvement_analysis.md`)

**Purpose**: Identified improvement areas and enhancement opportunities  
**Audience**: Maintainers, architects  
**Key Topics**:

- Dependency management inconsistencies
- Error handling pattern fragmentation
- Cross-module communication standards
- Import strategy standardization
- MCP integration gaps
- Specific recommendations for each area
- Implementation priority assessment

**Current Status**: 5 critical improvement areas identified

#### 21. **REPO Coherence Check Agent** (`REPO_COHERENCE_CHECK.md`)

**Purpose**: Comprehensive validation of entire GNN codebase  
**Audience**: QA, maintainers, architects  
**Key Topics**:

- High-level coherence metrics (25 steps, 28 modules, 41 AGENTS.md files)
- Validation scope (10 dimensions)
- Detailed analysis of each pipeline step
- Module consistency checks
- Documentation completeness verification
- Testing standards verification
- Performance metrics validation
- Security and compliance checks

**Scope**: All 25 pipeline steps + 28 specialized modules

---

### Integration Agents (2 modules)

#### 22. **GNN Paper Agent** (`gnn_paper.md`)

**Purpose**: Academic paper introducing GNN to the research community  
**Audience**: Academics, researchers  
**Key Topics**:

- Introduction and motivation
- Active Inference linguistics
- GNN specification
- The Triple Play approach
- Implications and future directions

**Citation**:

```
Smékal, J., & Friedman, D. A. (2023). Generalized Notation Notation for 
Active Inference Models. Active Inference Institute. 
https://doi.org/10.5281/zenodo.7803328
```

#### 23. **GNN Tools and Resources Agent** (`gnn_tools.md`)

**Purpose**: Complete ecosystem of tools, libraries, and resources  
**Audience**: Developers, practitioners  
**Key Topics**:

- Parsing tools (Python, JavaScript)
- Visualization tools (static and interactive)
- Conversion tools (code generation)
- Validation tools
- Editing tools (IDE plugins, web editors)
- Integration frameworks
- Performance optimization tools
- Community resources

#### 24. **GNN Troubleshooting Agent** (`gnn_troubleshooting.md`)

**Purpose**: Comprehensive guide for diagnosing and resolving GNN issues  
**Audience**: All users  
**Key Topics**:

- Parsing errors and solutions
- Type checking failures
- Framework integration issues
- Visualization problems
- Pipeline recovery strategies

---

## 🔄 Documentation Workflow

### For New Users

```
1. README.md (navigation hub)
   ↓
2. gnn_overview.md (what is GNN?)
   ↓
3. quickstart_tutorial.md (build first model)
   ↓
4. gnn_examples_doc.md (see more examples)
   ↓
5. gnn_syntax.md (learn syntax details)
```

### For Researchers

```
1. about_gnn.md (detailed spec)
   ↓
2. advanced_modeling_patterns.md (sophisticated techniques)
   ↓
3. gnn_multiagent.md (multi-agent systems)
   ↓
4. gnn_llm_neurosymbolic_active_inference.md (LLM integration)
   ↓
5. ontology_system.md (semantic grounding)
```

### For Developers

```
1. src/AGENTS.md (pipeline overview)
   ↓
2. architecture_reference.md (implementation patterns)
   ↓
3. technical_reference.md (entry points and data flow)
   ↓
4. gnn_implementation.md (implementation examples)
   ↓
5. framework_integration_guide.md (framework integration)
```

### For System Architects

```
1. src/README.md (pipeline architecture)
   ↓
2. architecture_reference.md (detailed patterns)
   ↓
3. REPO_COHERENCE_CHECK.md (validation framework)
   ↓
4. improvement_analysis.md (enhancement opportunities)
   ↓
5. resource_metrics.md (scalability planning)
```

---

## 📊 Cross-Reference Matrix

### By Pipeline Step

| Step | Primary Agent | Supporting Agents |
|------|---------------|-------------------|
| 0 | README | gnn_standards |
| 1 | README | gnn_standards |
| 2 | README | gnn_standards |
| 3 | gnn_schema, technical_reference | gnn_dsl_manual, gnn_syntax |
| 4 | README | resource_metrics |
| 5 | resource_metrics, gnn_schema | gnn_standards, technical_reference |
| 6 | REPO_COHERENCE_CHECK | gnn_standards |
| 7 | technical_reference | gnn_schema |
| 8-9 | architecture_reference | technical_reference |
| 10 | ontology_system | gnn_standards |
| 11 | framework_integration_guide | gnn_implementation, technical_reference |
| 12 | gnn_implementation, framework_integration_guide | technical_reference |
| 13 | gnn_llm_neurosymbolic_active_inference | gnn_standards |
| 14-16 | technical_reference | resource_metrics, gnn_implementation |
| 17-23 | architecture_reference | src/AGENTS.md |

### By Topic

| Topic | Primary Agent | Supporting Agents |
|-------|---------------|-------------------|
| Syntax | gnn_syntax, gnn_dsl_manual | gnn_file_structure_doc |
| File Structure | gnn_file_structure_doc | gnn_dsl_manual, gnn_examples_doc |
| Examples | gnn_examples_doc | gnn_syntax, quickstart_tutorial |
| Advanced Patterns | advanced_modeling_patterns | gnn_multiagent, gnn_llm_neurosymbolic |
| Implementation | gnn_implementation | framework_integration_guide |
| Framework Integration | framework_integration_guide | technical_reference, gnn_implementation |
| Architecture | architecture_reference | technical_reference, src/README.md |
| Quality | REPO_COHERENCE_CHECK | improvement_analysis, gnn_standards |

---

## 🎯 Key Metrics

### Documentation Coverage

- **Total Documentation Modules**: 24
- **Total Lines of Documentation**: ~12,000+
- **Code Examples Provided**: 100+
- **Diagrams/Flowcharts**: 50+
- **Cross-References**: 200+

### Documentation Organization

- **By Audience**: 4 clear paths (Beginners, Researchers, Developers, Architects)
- **By Task**: 8+ specific workflows documented
- **By Pipeline Step**: All 25 steps covered
- **By Module**: All 28 pipeline modules referenced

### Coverage Standards

- **Functional Completeness**: ✅ 100% (all documentation files present)
- **Cross-Referencing**: ✅ Comprehensive (all related docs linked)
- **Examples**: ✅ Working examples for all major features
- **Standards Compliance**: ✅ Professional, technical, evidence-based

---

## 🔗 Integration with Pipeline Modules

All GNN documentation agents integrate with the 25-step processing pipeline:

```
Documentation Layer (23 agents)
          ↓
Pipeline Infrastructure (25 steps, 28 modules)
          ↓
Framework Integration (PyMDP, RxInfer, DisCoPy, JAX)
          ↓
Active Inference Ecosystem
```

### Key Integration Points

1. **Step 3** (GNN Parsing) ← `gnn_schema`, `gnn_dsl_manual`
2. **Step 5** (Type Checking) ← `resource_metrics`, `gnn_standards`
3. **Step 10** (Ontology) ← `ontology_system`
4. **Step 11** (Rendering) ← `framework_integration_guide`, `gnn_implementation`
5. **Step 13** (LLM Analysis) ← `gnn_llm_neurosymbolic_active_inference`

---

## 📝 Documentation Standards

All GNN documentation follows these principles:

### 1. **Clarity and Precision**

- Concrete examples over abstract descriptions
- Technical accuracy prioritized
- Clear terminology consistent across documents

### 2. **Functionality-Focused**

- Show what code actually does, not what it might do
- Specific metrics and measurable results
- Real data and real execution examples

### 3. **Cross-Referencing**

- Every major concept links to related documentation
- Pipeline steps referenced in relevant docs
- Audience-specific navigation paths provided

### 4. **Evidence-Based**

- Specific file locations and line numbers
- Real execution times and memory usage
- Actual code examples from the codebase

### 5. **Professional Tone**

- Academic rigor where appropriate
- Clear, structured organization
- No promotional language or hyperbole

---

## 🚀 Usage Examples

### Running with Full Documentation

```bash
# Run pipeline with documentation reference
python src/main.py --target-dir input/gnn_files --verbose

# See specific step documentation
cat doc/gnn/technical_reference.md | grep "Step 3"

# Follow learning path
echo "Learning path: README.md → gnn_overview.md → quickstart_tutorial.md"
```

### Finding Information

```bash
# Find information about a specific topic
grep -l "multi-agent" doc/gnn/*.md

# Check syntax reference
less doc/gnn/gnn_syntax.md

# Understand a pipeline step
cat doc/gnn/architecture_reference.md | grep "Step 11"
```

---

## ✅ Validation Status

- ✅ All 24 documentation agents complete
- ✅ Complete cross-referencing
- ✅ All pipeline steps documented
- ✅ All 28 modules documented
- ✅ Professional standards compliance
- ✅ Working examples for all major features

**Recent Improvements**:

- Enhanced README navigation
- Comprehensive agent registry
- Clear audience-based learning paths
- Technical reference expansion
- Cross-module data flow documentation

---

## 📞 Support and Navigation

### Quick Navigation

- **Get Started**: `README.md` → `gnn_overview.md`
- **Learn Syntax**: `gnn_syntax.md` → `gnn_dsl_manual.md`
- **Build Models**: `quickstart_tutorial.md` → `gnn_examples_doc.md`
- **Understand Pipeline**: `src/AGENTS.md` → `architecture_reference.md`
- **Implement Features**: `gnn_implementation.md` → `framework_integration_guide.md`

### Documentation Maintenance

All documentation is maintained with the codebase and updated whenever pipeline functionality changes. Each module's AGENTS.md should reference relevant documentation files.

---

**Documentation Version**: v1.1.0
**Pipeline Version**: v1.1.0
**Compliance Status**: ✅ Production Ready
