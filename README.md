# GeneralizedNotationNotation (GNN)

**Last Updated**: 2026-03-24

<div align="center">

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](./LICENSE.md)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Active Inference](https://img.shields.io/badge/Active%20Inference-Research-brightgreen.svg)](https://activeinference.org/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.7803328-blue.svg)](https://doi.org/10.5281/zenodo.7803328)
[![Pipeline Steps](https://img.shields.io/badge/Pipeline%20Steps-25-blue.svg)](#%EF%B8%8F-processing-pipeline)
[![Documentation](https://img.shields.io/badge/Documentation-Comprehensive-success.svg)](#-documentation)

**A standardized text-based language for Active Inference generative models**

[🚀 Quick Start](#-quick-start) •
[📖 Documentation](#-documentation) •
[🎯 Examples](#-examples) •
[🛠️ Tools](#%EF%B8%8F-tools-and-utilities) •
[🤝 Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [🌟 Overview](#-overview)
- [📁 Key Files](#-key-files)
- [🎯 Motivation and Goals](#-motivation-and-goals)
- [✨ Key Features](#-key-features)
- [🏗️ Project Architecture](#%EF%B8%8F-project-architecture)
- [⚙️ Processing Pipeline](#%EF%B8%8F-processing-pipeline)
- [🛠️ Tools and Utilities](#%EF%B8%8F-tools-and-utilities)
- [🚀 Quick Start](#-quick-start)
- [📖 Documentation](#-documentation)
- [🎯 Examples](#-examples)
- [💚 Repository Health](#-repository-health)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🌟 Overview

**Generalized Notation Notation (GNN)** is a text-based language designed to standardize the representation and communication of [Active Inference](https://activeinference.org/) generative models. It aims to enhance clarity, reproducibility, and interoperability in the field of Active Inference and cognitive modeling.

### 📚 Initial Publication

**Smékal, J., & Friedman, D. A. (2023)**. *Generalized Notation Notation for Active Inference Models*. Active Inference Journal.  
**Last Updated**: 2026-03-24  
**Version**: 1.3.0  
**Status**: ✅ Production Ready (Active Inference Institute)  
**Test Suite (latest local `uv run pytest src/tests/ -q --tb=no --ignore=src/tests/test_llm_ollama.py --ignore=src/tests/test_llm_ollama_integration.py`)**: 1,906 passed, 30 skipped (2026-03-24). Including `test_llm_ollama*.py` adds integration cases that call the local `ollama` CLI (default tag overridable via `OLLAMA_MODEL`); without a responsive daemon they may time out or fail.  
📖 **DOI:** [10.5281/zenodo.7803328](https://doi.org/10.5281/zenodo.7803328)  
📁 **Archive:** [zenodo.org/records/7803328](https://zenodo.org/records/7803328)

### 🎯 Core Design Principles

GNN provides a structured and standardized way to describe complex cognitive models. It is designed to be:

- **🧑‍💻 Human-readable**: Easy to understand and use for researchers from diverse backgrounds
- **🤖 Machine-parsable**: Can be processed by software tools for analysis, visualization, and code generation
- **🔄 Interoperable**: Facilitates the exchange and reuse of models across different platforms and research groups
- **🔬 Reproducible**: Enables precise replication of model specifications

GNN addresses the challenge of communicating Active Inference models, which are often described using a mix of natural language, mathematical equations, diagrams, and code. By offering a unified notation, GNN aims to streamline collaboration, improve model understanding, and accelerate research.

---

## 📁 Key Files

> **🚀 Start here** to understand the repository structure and find what you need quickly.

| File | Purpose | Start Here If... |
|------|---------|------------------|
| **[README.md](./README.md)** | Main entry point and overview | You're new to GNN |
| **[AGENTS.md](./AGENTS.md)** | Master agent scaffolding - all 25 modules documented | You want to understand the pipeline architecture |
| **[DOCS.md](./DOCS.md)** | Comprehensive documentation with all diagrams | You need the complete system overview |
| **[ARCHITECTURE.md](./ARCHITECTURE.md)** | Implementation patterns and extension guides | You're developing or extending GNN |
| **[doc/quickstart.md](./doc/quickstart.md)** | Step-by-step getting started guide | You want to run your first pipeline |
| **[doc/gnn/reference/gnn_syntax.md](./doc/gnn/reference/gnn_syntax.md)** | Complete GNN syntax specification | You're writing GNN model files |
| **[pyproject.toml](./pyproject.toml)** | Project dependencies and configuration | You're setting up the environment |
| **[SETUP_GUIDE.md](./SETUP_GUIDE.md)** | Detailed installation instructions | You're having setup issues |
| **[SECURITY.md](./SECURITY.md)** | Security policy and vulnerability reporting | You found a security issue |
| **[SUPPORT.md](./SUPPORT.md)** | Getting help and community resources | You need assistance |
| **[CITATION.cff](./CITATION.cff)** | Citation information for academic use | You're citing GNN in research |
| **[CHANGELOG.md](./CHANGELOG.md)** | Release history and version changes | You want to see what changed |
| **[.github/README.md](./.github/README.md)** | GitHub-facing hub: deep links, repo map, CI, Dependabot | You want navigation from GitHub UI or you're changing automation |

### 📂 Directory Overview

```text
GeneralizedNotationNotation/
├── 📄 README.md, AGENTS.md, DOCS.md, ARCHITECTURE.md  # Core documentation
├── 📁 src/                    # 25-step pipeline + 31 modules
│   ├── main.py               # 🎯 Main orchestrator - run this!
│   ├── 0_template.py → 24_intelligent_analysis.py  # Numbered pipeline scripts
│   ├── gnn/, render/, execute/, llm/, ...  # Agent modules
│   └── tests/                # Comprehensive test suite
├── 📁 doc/                    # ~450 Markdown files under doc/ (plus assets; see doc/README.md)
│   ├── gnn/                  # GNN language specification
│   ├── pymdp/, rxinfer/      # Framework integration guides
│   └── cognitive_phenomena/  # Example cognitive models
├── 📁 input/                  # Input GNN files and configuration
│   └── gnn_files/            # Sample GNN model files
├── 📁 output/                 # Pipeline outputs (tracked; see .gitignore for temp/audio/cache)
└── 📄 pyproject.toml, pytest.ini  # Configuration files
```

---

## 🧩 Modules & Agents

The GNN pipeline is composed of **25 specialized modules**, each acting as an agent with specific responsibilities. This "Thin Orchestrator" architecture ensures modularity and testability.

| Step | Agent / Module | Description | Links |
| :--- | :--- | :--- | :--- |
| **0** | **[Template](src/template/)** | Initial project scaffolding and configuration. | [🤖 Agent](src/template/AGENTS.md) • [📝 Code](src/0_template.py) |
| **1** | **[Setup](src/setup/)** | Environment, dependencies, and UV package management. | [🤖 Agent](src/setup/AGENTS.md) • [📝 Code](src/1_setup.py) |
| **2** | **[Tests](src/tests/)** | Comprehensive suite orchestration and coverage analysis. | [🤖 Agent](src/tests/AGENTS.md) • [📝 Code](src/2_tests.py) |
| **3** | **[GNN](src/gnn/)** | Core parsing, discovery, and semantic object model. | [🤖 Agent](src/gnn/AGENTS.md) • [📝 Code](src/3_gnn.py) |
| **4** | **[Registry](src/model_registry/)** | Validation and versioning of model artifacts. | [🤖 Agent](src/model_registry/AGENTS.md) • [📝 Code](src/4_model_registry.py) |
| **5** | **[TypeCheck](src/type_checker/)** | Static analysis, dimension validation, resource estimation. | [🤖 Agent](src/type_checker/AGENTS.md) • [📝 Code](src/5_type_checker.py) |
| **6** | **[Validate](src/validation/)** | Logical consistency and ontology compliance. | [🤖 Agent](src/validation/AGENTS.md) • [📝 Code](src/6_validation.py) |
| **7** | **[Export](src/export/)** | Serialization to JSON, XML, GraphML, etc. | [🤖 Agent](src/export/AGENTS.md) • [📝 Code](src/7_export.py) |
| **8** | **[Viz](src/visualization/)** | Static visualization of matrices and network logic. | [🤖 Agent](src/visualization/AGENTS.md) • [📝 Code](src/8_visualization.py) |
| **9** | **[Adv. Viz](src/advanced_visualization/)** | Interactive diagrams and complex visual analysis. | [🤖 Agent](src/advanced_visualization/AGENTS.md) • [📝 Code](src/9_advanced_viz.py) |
| **10** | **[Ontology](src/ontology/)** | Semantic mapping to Active Inference definitions. | [🤖 Agent](src/ontology/AGENTS.md) • [📝 Code](src/10_ontology.py) |
| **11** | **[Render](src/render/)** | Code generation for PyMDP, RxInfer, ActiveInference.jl, DisCoPy, JAX, Stan, PyTorch, NumPyro | [🤖 Agent](src/render/AGENTS.md) • [📝 Code](src/11_render.py) |
| **12** | **[Execute](src/execute/)** | Simulation runner and runtime management. | [🤖 Agent](src/execute/AGENTS.md) • [📝 Code](src/12_execute.py) |
| **13** | **[LLM](src/llm/)** | Neurosymbolic analysis and text generation. | [🤖 Agent](src/llm/AGENTS.md) • [📝 Code](src/13_llm.py) |
| **14** | **[ML](src/ml_integration/)** | Integration with external ML frameworks. | [🤖 Agent](src/ml_integration/AGENTS.md) • [📝 Code](src/14_ml_integration.py) |
| **15** | **[Audio](src/audio/)** | Sonification of model dynamics. | [🤖 Agent](src/audio/AGENTS.md) • [📝 Code](src/15_audio.py) |
| **16** | **[Analysis](src/analysis/)** | Statistical post-processing of simulation results. | [🤖 Agent](src/analysis/AGENTS.md) • [📝 Code](src/16_analysis.py) |
| **17** | **[Integrate](src/integration/)** | Cross-module synthesis and coordination. | [🤖 Agent](src/integration/AGENTS.md) • [📝 Code](src/17_integration.py) |
| **18** | **[Security](src/security/)** | Safety checks and sandboxing. | [🤖 Agent](src/security/AGENTS.md) • [📝 Code](src/18_security.py) |
| **19** | **[Research](src/research/)** | Experimental features and benchmarking. | [🤖 Agent](src/research/AGENTS.md) • [📝 Code](src/19_research.py) |
| **20** | **[Website](src/website/)** | Static site generation for reports/docs. | [🤖 Agent](src/website/AGENTS.md) • [📝 Code](src/20_website.py) |
| **21** | **[MCP](src/mcp/)** | Model Context Protocol server and tools. | [🤖 Agent](src/mcp/AGENTS.md) • [📝 Code](src/21_mcp.py) |
| **22** | **[GUI](src/gui/)** | Interactive visual editors (Web & Local). | [🤖 Agent](src/gui/AGENTS.md) • [📝 Code](src/22_gui.py) |
| **23** | **[Report](src/report/)** | Final comprehensive report assembly. | [🤖 Agent](src/report/AGENTS.md) • [📝 Code](src/23_report.py) |
| **24** | **[Intelligent Analysis](src/intelligent_analysis/)** | AI-powered pipeline analysis and executive reports. | [🤖 Agent](src/intelligent_analysis/AGENTS.md) • [📝 Code](src/24_intelligent_analysis.py) |

---

## 🎯 Motivation and Goals

### 🚫 Current Challenges

The primary motivation behind GNN is to overcome the limitations arising from the lack of a standardized notation for Active Inference models. This fragmentation can lead to difficulties in:

- **💬 Effective Communication**: Making complex models hard to explain and understand
- **🔄 Reproducibility**: Hindering the ability to replicate research findings  
- **⚙️ Consistent Implementation**: Leading to variations when translating models into code
- **📊 Systematic Comparison**: Making it challenging to compare different models

### 🎯 Our Goals

The goals of GNN are to:

- ✅ Facilitate clear communication and understanding of Active Inference models
- ✅ Promote collaboration among researchers
- ✅ Enable the development of tools for model validation, visualization, and automated code generation
- ✅ Support the creation of a shared repository of Active Inference models
- ✅ Bridge the gap between theoretical concepts and practical implementations

---

## ✨ Key Features

### 🎭 The Triple Play Approach

GNN supports three complementary modalities for model representation, known as the **"Triple Play"**:

```mermaid
graph LR
    subgraph "🎭 Triple Play Approach"
        A["📝 Text-Based Models<br/>• GNN Markdown files<br/>• Human-readable syntax<br/>• Mathematical notation<br/>• Natural language descriptions"]
        
        B["📊 Graphical Models<br/>• Factor graphs<br/>• Network visualizations<br/>• Dependency diagrams<br/>• Interactive visualizations"]
        
        C["⚙️ Executable Models<br/>• PyMDP simulations<br/>• RxInfer.jl implementations<br/>• JAX computations<br/>• PyTorch inference<br/>• NumPyro probabilistic<br/>• DisCoPy diagrams"]
    end
    
    A -->|Parse & Extract| B
    A -->|Generate Code| C
    B -->|Validate Structure| C
    C -->|Simulate & Test| A
    B -->|Visual Feedback| A
    C -->|Results Analysis| B
    
    %% styling intentionally omitted (theme-controlled)
```

1. **📝 Text-Based Models**: GNN files are plain text and can be rendered into mathematical notation, pseudocode, or natural language descriptions. This forms the core representation.

2. **📊 Graphical Models**: The structure defined in GNN (variables and their connections) can be visualized as graphical models (e.g., factor graphs), clarifying dependencies and model architecture.

3. **⚙️ Executable Cognitive Models**: GNN specifications can serve as a high-level blueprint or pseudocode for implementing executable simulations in various programming environments. This ensures consistency and aids in the translation from theory to practice.

### 📋 Structured File Format

GNN defines a specific file structure, typically using Markdown, to organize model components. This includes sections for:

- 🏷️ Model metadata (name, version, annotations)
- 🌐 State space (variable definitions)  
- 🔗 Connections (relationships between variables)
- ⚙️ Initial parameterization
- 📐 Equations
- ⏰ Time settings (for dynamic models)
- 🧠 Mapping to Active Inference Ontology terms

### 📝 GNN Syntax Preview

Here's a glimpse of what a GNN model file looks like (from [`input/gnn_files/discrete/actinf_pomdp_agent.md`](./input/gnn_files/discrete/actinf_pomdp_agent.md)):

<details>
<summary><strong>📄 View GNN File Example</strong></summary>

```markdown
# GNN Example: Active Inference POMDP Agent
# GNN Version: 1.0

## GNNSection
ActInfPOMDP

## GNNVersionAndFlags
GNN v1

## ModelName
Active Inference POMDP Agent

## StateSpaceBlock
# Likelihood matrix: A[observation_outcomes, hidden_states]
A[3,3,type=float]   # Likelihood mapping hidden states to observations

# Transition matrix: B[states_next, states_previous, actions]
B[3,3,3,type=float]   # State transitions given previous state and action

# Preference vector: C[observation_outcomes]
C[3,type=float]       # Log-preferences over observations

# Hidden State
s[3,1,type=float]     # Current hidden state distribution

## Connections
D>s
s-A
s>s_prime
A-o
π>u

## InitialParameterization
A={(0.9,0.05,0.05), (0.05,0.9,0.05), (0.05,0.05,0.9)}
C={(0.1, 0.1, 1.0)}
D={(0.33333, 0.33333, 0.33333)}

## ActInfOntologyAnnotation
A=LikelihoodMatrix
B=TransitionMatrix
s=HiddenState
o=Observation
```

</details>

**Connection Syntax:**

- `D>s` — D feeds into s (directed)
- `s-A` — s connects to A (undirected/bidirectional)
- `π>u` — Policy determines action

📖 **Full syntax reference**: [doc/gnn/reference/gnn_syntax.md](./doc/gnn/reference/gnn_syntax.md)

---

## 🏗️ Project Architecture

```mermaid
graph TB
    subgraph "🏗️ GNN Project Architecture"
        subgraph "📁 Source Code (src/)"
            A[⚙️ Pipeline Scripts<br/>0_template.py → 24_intelligent_analysis.py]
            B[🧠 Core Modules<br/>gnn/, render/, llm/]
            C[🔧 Utilities<br/>utils/, pipeline/]
            D[🧪 Testing<br/>tests/]
        end
        
        subgraph "📚 Documentation (doc/)"
            E[📖 Core Docs<br/>gnn/, syntax, examples]
            F[🎯 Specialized<br/>pymdp/, rxinfer/, mcp/]
            G[🧩 Applications<br/>cognitive_phenomena/]
        end
        
        subgraph "🎯 Outputs (output/)"
            H[📊 Reports<br/>Type checking, analysis]
            I[🎨 Visualizations<br/>Graphs, matrices]
            J[💻 Generated Code<br/>PyMDP, RxInfer, PyTorch, NumPyro]
            K[🌐 Static Site<br/>HTML summaries]
        end
    end
    
    A --> H
    B --> I
    B --> J
    E --> A
    F --> B
    G --> B
    
    %% styling intentionally omitted (theme-controlled)
```

### Module Dependency Graph

```mermaid
graph LR
    subgraph "Infrastructure Layer"
        Utils[utils/]
        Pipeline[pipeline/]
    end
    
    subgraph "Core Processing"
        GNN[gnn/]
        TypeChecker[type_checker/]
        Validation[validation/]
        Export[export/]
    end
    
    subgraph "Code Generation"
        Render[render/]
        Execute[execute/]
    end
    
    subgraph "Analysis & Output"
        LLM[llm/]
        Analysis[analysis/]
        Report[report/]
    end
    
    Utils --> GNN
    Utils --> TypeChecker
    Utils --> Render
    Pipeline --> GNN
    Pipeline --> Render
    
    GNN --> TypeChecker
    GNN --> Validation
    GNN --> Export
    GNN --> Render
    
    Render --> Execute
    Execute --> Analysis
    LLM --> Analysis
    Analysis --> Report
```

### Framework Integration Architecture

```mermaid
graph TB
    subgraph "GNN Input"
        GNNFile[GNN Markdown File]
    end
    
    subgraph "Parsing & Validation"
        Parser[GNN Parser]
        Validator[Type Checker]
    end
    
    subgraph "Code Generation"
        Renderer[Render Module]
        PyMDP[PyMDP Generator]
        RxInfer[RxInfer.jl Generator]
        ActInf[ActiveInference.jl Generator]
        JAX[JAX Generator]
        DisCoPy[DisCoPy Generator]
        Stan[Stan Generator]
        PyTorch[PyTorch Generator]
        NumPyro[NumPyro Generator]
    end
    
    subgraph "Execution"
        Executor[Execute Module]
        PyMDPExec[PyMDP Runner]
        RxInferExec[RxInfer Runner]
        ActInfExec[ActiveInference Runner]
        JAXExec[JAX Runner]
        PyTorchExec[PyTorch Runner]
        NumPyroExec[NumPyro Runner]
    end
    
    subgraph "Analysis"
        Analyzer[Analysis Module]
        Results[Simulation Results]
    end
    
    GNNFile --> Parser
    Parser --> Validator
    Validator --> Renderer
    
    Renderer --> PyMDP
    Renderer --> RxInfer
    Renderer --> ActInf
    Renderer --> JAX
    Renderer --> DisCoPy
    Renderer --> Stan
    Renderer --> PyTorch
    Renderer --> NumPyro
    
    PyMDP --> Executor
    RxInfer --> Executor
    ActInf --> Executor
    JAX --> Executor
    PyTorch --> Executor
    NumPyro --> Executor
    
    Executor --> PyMDPExec
    Executor --> RxInferExec
    Executor --> ActInfExec
    Executor --> JAXExec
    Executor --> PyTorchExec
    Executor --> NumPyroExec
    
    PyMDPExec --> Analyzer
    RxInferExec --> Analyzer
    ActInfExec --> Analyzer
    JAXExec --> Analyzer
    PyTorchExec --> Analyzer
    NumPyroExec --> Analyzer
    
    Analyzer --> Results
```

### 📁 Directory Structure

<details>
<summary><strong>📂 src/ Directory Structure</strong></summary>

The `src/` directory contains the 25-step pipeline scripts (`0_template.py` → `24_intelligent_analysis.py`), their corresponding modules, and shared infrastructure. See `DOCS.md` and `doc/pipeline/README.md` for the full step-by-step mapping.

```text
src/
├── 📜 Pipeline Scripts (0-24)
│   ├── 0_template.py … 24_intelligent_analysis.py   # Thin orchestrators (0–24)
├── 🧠 Core Modules
│   ├── gnn/ render/ execute/ llm/ visualization/ export/ type_checker/ ontology/ mcp/
│   ├── setup/ tests/ website/ audio/ analysis/ integration/ security/ research/ report/
├── 🔧 Infrastructure: utils/ pipeline/
└── 🗂️ Orchestrator: main.py
```

</details>

<details>
<summary><strong>📂 doc/ Directory Structure</strong></summary>

The `doc/` directory contains all supplementary documentation, including conceptual explanations, syntax guides, and examples.

```text
doc/
├── 📖 Core Documentation
│   ├── gnn/                       # GNN specifications
│   ├── quickstart.md             # Getting started guide
│   ├── SETUP.md                  # Installation instructions
│   └── README.md                 # Documentation overview
├── 🎯 Domain Applications
│   ├── cognitive_phenomena/       # Cognitive modeling examples
│   ├── pymdp/                    # PyMDP integration
│   ├── rxinfer/                  # RxInfer.jl integration
│   └── templates/                # Model templates
├── 🛠️ Technical Integration
│   ├── mcp/                      # Model Context Protocol
│   ├── llm/                      # LLM integration
│   ├── discopy/                  # DisCoPy categorical diagrams
│   ├── sympy/                    # SymPy mathematical processing
└── 📚 Resources
    ├── troubleshooting/          # Common issues & solutions
    ├── testing/                  # Testing documentation
    └── security/                 # Security guidelines
```

</details>

---

## ⚙️ Processing Pipeline

The GNN processing pipeline consists of **25 comprehensive steps (0-24)**, each handling a specific aspect of model processing from parsing to final report generation. The pipeline follows a **thin orchestrator pattern** where numbered scripts orchestrate execution while delegating core functionality to modular components.

```mermaid
flowchart TD
    A["🚀 Start Pipeline"] --> B["0️⃣ Template Init<br/>src/template/"]
    B --> C["1️⃣ Setup & Dependencies<br/>src/setup/"]
    C --> D["2️⃣ Tests<br/>src/tests/"]
    D --> E["3️⃣ GNN Discovery & Parsing<br/>src/gnn/"]
    E --> F["4️⃣ Model Registry<br/>src/model_registry/"]
    F --> G["5️⃣ Type Checking<br/>src/type_checker/"]
    G --> H["6️⃣ Validation<br/>src/validation/"]
    H --> I["7️⃣ Export<br/>src/export/"]
    I --> J["8️⃣ Visualization<br/>src/visualization/"]
    J --> K["9️⃣ Advanced Viz<br/>src/advanced_visualization/"]
    K --> L["1️⃣0️⃣ Ontology<br/>src/ontology/"]
    L --> M["1️⃣1️⃣ Rendering<br/>src/render/"]
    M --> N["1️⃣2️⃣ Execution<br/>src/execute/"]
    N --> O["1️⃣3️⃣ LLM Analysis<br/>src/llm/"]
    O --> P["1️⃣4️⃣ ML Integration<br/>src/ml_integration/"]
    P --> Q["1️⃣5️⃣ Audio<br/>src/audio/"]
    Q --> R["1️⃣6️⃣ Analysis<br/>src/analysis/"]
    R --> S["1️⃣7️⃣ Integration<br/>src/integration/"]
    S --> T["1️⃣8️⃣ Security<br/>src/security/"]
    T --> U["1️⃣9️⃣ Research<br/>src/research/"]
    U --> V["2️⃣0️⃣ Website<br/>src/website/"]
    V --> W["2️⃣1️⃣ MCP<br/>src/mcp/"]
    W --> X["2️⃣2️⃣ GUI<br/>src/gui/"]
    X --> Y["2️⃣3️⃣ Report<br/>src/report/"]
    Y --> Y2["2️⃣4️⃣ Intelligent Analysis<br/>src/intelligent_analysis/"]
    Y2 --> Z["✅ Complete"]

    %% styling intentionally omitted (theme-controlled)
```

### 🎯 GNN Processing Workflow

```mermaid
flowchart TD
    subgraph "🧠 GNN Processing Workflow"
        A["📄 GNN File Input<br/>.md format"] --> B["🔍 Discovery & Parsing<br/>Extract sections"]
        B --> C["✅ Type Checking<br/>Validate syntax & structure"]
        C --> D["📊 Multi-Format Export<br/>JSON, XML, GraphML, Pickle"]
        
        C --> E["🎨 Visualization<br/>Generate graphs & matrices"]
        C --> F["🔄 Code Generation<br/>PyMDP, RxInfer, JAX, PyTorch, NumPyro, DisCoPy"]
        
        F --> G["▶️ Simulation Execution<br/>Run generated code"]
        E --> H["🧠 LLM Analysis<br/>AI-powered insights"]
        
        D --> I["📚 Ontology Mapping<br/>Active Inference terms"]
        G --> J["📈 Results Analysis<br/>Performance metrics"]
        H --> K["🎵 Audio Generation<br/>SAPF, Pedalboard backends"]
        
        I --> L["🌐 Site Generation<br/>Static HTML reports"]
        J --> L
        K --> M["📊 Report Generation<br/>Comprehensive analysis"]
        
        L --> M
        M --> N["✨ Complete Analysis<br/>Multi-modal outputs"]
    end
    
    %% styling intentionally omitted (theme-controlled)
```

### Data Flow Between Pipeline Steps

```mermaid
flowchart LR
    subgraph "Input Stage"
        Input[GNN Files]
    end
    
    subgraph "Processing Stage"
        Step3[Step 3: GNN Parse]
        Step5[Step 5: Type Check]
        Step6[Step 6: Validation]
        Step7[Step 7: Export]
    end
    
    subgraph "Generation Stage"
        Step8[Step 8: Visualization]
        Step11[Step 11: Render]
        Step12[Step 12: Execute]
    end
    
    subgraph "Analysis Stage"
        Step13[Step 13: LLM]
        Step16[Step 16: Analysis]
        Step23[Step 23: Report]
    end
    
    Input --> Step3
    Step3 --> Step5
    Step3 --> Step6
    Step3 --> Step7
    Step3 --> Step8
    Step3 --> Step11
    
    Step5 --> Step6
    Step6 --> Step7
    Step7 --> Step8
    
    Step11 --> Step12
    Step12 --> Step13
    Step12 --> Step16
    
    Step13 --> Step16
    Step8 --> Step16
    Step16 --> Step23
```

### 🧠 Modules & Agents

The GNN framework is built around a modular architecture, where each pipeline step corresponds to a dedicated module. These modules encapsulate specific functionalities and interact with various "agents" (e.g., LLMs, external APIs) to perform their tasks.

| Module Name | Pipeline Step | Description | Key Agents/Integrations |
| :---------- | :------------ | :---------- | :---------------------- |
| `template` | 0 | Initializes pipeline, sets up environment, and loads configuration. | Configuration Manager, Environment Setup |
| `setup` | 1 | Manages dependencies, virtual environments, and system checks. | `uv`, `pip`, System Utilities |
| `tests` | 2 | Executes unit, integration, and end-to-end tests. | `pytest`, `coverage.py` |
| `gnn` | 3 | Discovers, parses, and validates GNN markdown files. | Markdown Parser, Schema Validator |
| `model_registry` | 4 | Manages a central registry of GNN models and their metadata. | Database (SQLite), Model Metadata |
| `type_checker` | 5 | Performs static analysis, type checking, and resource estimation. | Type Inference Engine, Resource Estimator |
| `validation` | 6 | Validates GNN models against predefined rules and constraints. | Constraint Solver, Logic Validator |
| `export` | 7 | Exports GNN models to various formats (JSON, XML, GraphML). | JSON/XML/GraphML Serializers |
| `visualization` | 8 | Generates graphical representations of GNN models. | `matplotlib`, `plotly`, `graphviz` |
| `advanced_visualization` | 9 | Provides advanced, interactive 3D and dynamic visualizations. | `vedo`, `pyvista`, `dash` |
| `ontology` | 10 | Maps GNN concepts to Active Inference ontology terms. | Ontology Mapper, Knowledge Graph |
| `render` | 11 | Renders GNN models into executable code for various backends. | Code Generators (PyMDP, RxInfer, JAX, ActInf.jl, PyTorch, NumPyro, DisCoPy) |
| `execute` | 12 | Executes the rendered code using the specified backend. | PyMDP, RxInfer.jl, JAX, ActiveInference.jl, PyTorch, NumPyro |
| `llm` | 13 | Integrates Large Language Models for analysis, generation, and insights. | OpenAI GPT, Anthropic Claude, Local LLMs |
| `ml_integration` | 14 | Integrates with machine learning frameworks for advanced analysis. | `scikit-learn`, `tensorflow`, `pytorch` |
| `audio` | 15 | Generates audio representations of GNN model dynamics. | `SAPF`, `Pedalboard`, Audio Synthesis Engines |
| `analysis` | 16 | Performs statistical and qualitative analysis of execution results. | `pandas`, `scipy`, Statistical Models |
| `integration` | 17 | Manages external system integrations and data exchange. | REST APIs, Message Queues |
| `security` | 18 | Implements security checks, vulnerability scanning, and access control. | SAST Tools, Security Scanners |
| `research` | 19 | Supports research-specific tasks like hypothesis testing and data collection. | Experimentation Frameworks |
| `website` | 20 | Generates static websites or documentation portals from GNN outputs. | Static Site Generators (e.g., MkDocs) |
| `mcp` | 21 | Implements the Model Context Protocol for inter-model communication. | Protocol Handlers, Message Brokers |
| `gui` | 22 | Provides interactive graphical user interfaces for model creation and editing. | `gradio`, `streamlit`, `dash` |
| `report` | 23 | Generates comprehensive reports summarizing the pipeline execution. | Report Generators (PDF, HTML) |
| `intelligent_analysis` | 24 | AI-powered pipeline analysis and executive reports. | LLM analysis, remediation, pipeline summaries |

### 🏗️ Pipeline Architecture: Three-Tier Pattern

The pipeline follows a **three-tier architectural pattern** (`Script -> Interface -> Processor`) for high maintainability and testability.

```mermaid
graph TB
    subgraph "Tier 1: Orchestrator Layer"
        Script[N_Module.py<br/>Thin Orchestrator]
    end
    
    subgraph "Tier 2: Interface Layer"
        Init[__init__.py<br/>Public API]
    end
    
    subgraph "Tier 3: Implementation Layer"
        Processor[processor.py<br/>Core Logic]
        Framework[framework/<br/>Adapters]
        MCP[mcp.py<br/>Tools]
    end
    
    Script -->|Calls| Init
    Init -->|Delegates| Processor
    Processor -->|Uses| Framework
    Processor -->|Registers| MCP
    
    %% styling intentionally omitted (theme-controlled)
```

#### 🏛️ Architectural Components

1. **Main Pipeline Orchestrator** (`src/main.py`): Central coordinator that executes numbered scripts in sequence.
2. **Thin Orchestrators** (`src/0_template.py`, `src/1_setup.py`, etc.): Minimal scripts (<150 lines) that handle CLI args and logging, then delegate immediately.
3. **Modular Scripts** (`src/template/`, `src/setup/`, etc.): The actual "brains" of the operation, containing `processor.py`, logic, and specialized tests.

#### 📋 Current Status

**✅ All Scripts Compliant (25/25):**

All 25 pipeline scripts now follow the thin orchestrator pattern with complete delegation to their respective modules. Each script handles argument parsing, logging setup, and output directory management, then delegates all domain logic to the module implementation.

#### 📁 Example Structure

```text
src/
├── main.py                          # Main pipeline orchestrator
├── 0_template.py                    # Thin orchestrator - imports from template/
├── template/                        # Modular template implementation
│   ├── __init__.py                 # Module exports and initialization
│   ├── processor.py                # Core template processing logic
│   └── mcp.py                      # Model Context Protocol integration
└── tests/
    └── test_template_integration.py # Tests for template module
```

#### 📚 Documentation

For comprehensive architectural documentation, see:

- `src/template/README.md`: Reference implementation and pattern documentation
- `ARCHITECTURE.md`: Complete architectural guide

### 🚀 Running the Pipeline

Navigate to the project's root directory and execute:

```bash
python src/main.py [options]
```

#### 🛠️ Key Pipeline Options

| Option | Description | Default |
|--------|-------------|---------|
| `--target-dir DIR` | Target directory for GNN files | `input/gnn_files` |
| `--output-dir DIR` | Directory to save outputs | `output/` |
| `--recursive` | Recursively process directories | `True` |
| `--skip-steps LIST` | Skip specific steps (e.g., "1,7") | None |
| `--only-steps LIST` | Run only specific steps | None |
| `--verbose` | Enable detailed logging | `False` |
| `--strict` | Enable strict type checking | `False` |
| `--estimate-resources` | Estimate computational resources | `True` |

<details>
<summary><strong>📋 View All Pipeline Options</strong></summary>

```bash
python src/main.py --help
```

**Additional specialized options:**

- `--ontology-terms-file FILE`: Path to ontology terms file
- `--llm-tasks LIST`: Comma-separated LLM tasks
- `--llm-timeout`: LLM processing timeout
- `--pipeline-summary-file FILE`: Pipeline summary report path
- `--site-html-filename NAME`: Generated HTML site filename
- `--duration`: Audio duration for audio generation (default: 30.0)
- `--audio-backend`: Audio backend to use (auto, sapf, pedalboard, default: auto)
- `--recreate-uv-env`: Recreate UV environment
- `--dev`: Install development dependencies

</details>

---

## 🛠️ Tools and Utilities

The GNN ecosystem includes several sophisticated tools to aid in model development, validation, and understanding. These tools are primarily invoked through the `src/main.py` pipeline script. The project also provides a **CLI** (`gnn` command), **LSP** (Language Server Protocol) for editor support, and a **REST API** (FastAPI Pipeline-as-a-Service); **131 MCP tools** are registered for model context integration. See [AGENTS.md](./AGENTS.md) and [doc/gnn/](doc/gnn/) for details.

### ✅ Type Checker and Resource Estimator

The **GNN Type Checker** (pipeline step 5) helps validate GNN files and estimates computational resources.

#### 🎯 Quick Usage

```bash
# Run only type checker
python src/main.py --only-steps 5 --target-dir path/to/gnn_files

# Include resource estimation
python src/main.py --only-steps 5 --estimate-resources --target-dir path/to/gnn_files

# Run full pipeline
python src/main.py --target-dir path/to/gnn_files
```

#### 📊 Features

- ✅ Validation of required sections and structure
- 🔍 Type checking of variables and dimensions  
- 🔗 Verification of connections and references
- 📋 Detailed error reports with suggestions for fixes
- 💾 Resource usage estimation and optimization recommendations

#### 📁 Output Structure

When executed, the type checker writes to `output/5_type_checker_output/`:

```text
output/5_type_checker_output/
├── type_check_results.json
├── type_check_summary.json
└── global_type_analysis.json
```

### 🎨 Visualization

GNN files can be visualized to create comprehensive graphical representations of models (pipeline step 8).

#### 🎯 Usage

```bash
# Generate visualizations (target-dir is a folder of GNN files)
python src/main.py --only-steps 8 --target-dir path/to/gnn_models/
```

#### 🖼️ Visualization Types

- 📊 **Network Graphs**: Model structure and dependencies
- 🎨 **Matrix Heatmaps**: A, B, C, D matrix visualizations  
- 🧠 **Ontology Diagrams**: Active Inference relationship maps
- 📈 **Performance Plots**: Resource usage and timing analysis

### 🎮 Interactive GUI Interfaces

GNN provides **three distinct interactive GUI interfaces** for visual model construction and editing (pipeline step 22).

#### 🎯 GUI Overview

| **GUI** | **Focus** | **Port** | **Key Features** |
|---------|-----------|----------|------------------|
| **GUI 1** | Form-based Constructor | `:7860` | Step-by-step model building, component management |
| **GUI 2** | Visual Matrix Editor | `:7861` | Real-time heatmaps, interactive matrix editing |
| **GUI 3** | Design Studio | `:7862` | State space design, ontology editing, low-dependency |
| **oxdraw** | Diagram-as-Code | `:5151` | Mermaid-based visual editing, GNN ↔ diagram sync |

#### 🚀 Quick Launch

```bash
# Launch all GUIs (recommended)
python src/22_gui.py --target-dir input/gnn_files --output-dir output --gui-types "gui_1,gui_2,gui_3,oxdraw" --interactive --verbose

# Launch specific GUI
python src/22_gui.py --gui-types "gui_3" --interactive --verbose  # Design Studio only
python src/22_gui.py --gui-types "oxdraw" --interactive --verbose  # oxdraw only

# Launch via main pipeline (headless mode)
python src/main.py --only-steps 22 --verbose
```

#### 🏗️ GUI Details

**🔧 GUI 1: Form-based Constructor** (`http://localhost:7860`)

- Interactive two-pane editor for systematic GNN model construction
- Component management (observation/hidden/action/policy variables)  
- State space entry management with live validation
- Synchronized plaintext GNN markdown editor

**📊 GUI 2: Visual Matrix Editor** (`http://localhost:7861`)

- Real-time matrix heatmap visualizations with Plotly
- Interactive DataFrame editing with +/- dimension controls
- Vector bar chart displays for C & D vectors
- Live matrix statistics (min, max, mean, sum)
- Auto-update functionality and matrix validation

**🎨 GUI 3: State Space Design Studio** (`http://localhost:7862`)

- Visual state space architecture designer with SVG diagrams
- Ontology term editor for Active Inference concept mapping
- Interactive connection graph interface (D>s, s-A, A-o format)
- Parameter tuning controls (states, observations, actions, horizons)
- Real-time GNN export and preview with low-dependency approach

#### 📁 GUI Output Structure

```text
output/22_gui_output/
├── gui_1_output/           # Form-based constructor outputs
│   ├── constructed_model_gui_1.md
│   └── gui_status.json
├── gui_2_output/           # Visual matrix editor outputs  
│   ├── visual_model_gui_2.md
│   └── visual_matrices.json
├── gui_3_output/           # Design studio outputs
│   ├── designed_model_gui_3.md
│   └── design_analysis.json
└── gui_processing_summary.json
```

---

## 🚀 Quick Start

If you use [uv](https://github.com/astral-sh/uv) (`uv sync` / `uv run`), prefer **`uv run python src/...`** instead of **`python src/...`** for pipeline commands so they run in the project environment. Many examples below use bare `python` for a minimal local setup; substitute `uv run python` when you work with uv (see [CONTRIBUTING.md](CONTRIBUTING.md)).

**New to GNN?** Choose your learning path:

### 🎯 **Choose Your Journey**

- **⚡ Quick Demo (5 min)**: See GNN in action → [5-Minute Demo](doc/quickstart.md#5-minute-demo)
- **🔬 I'm a Researcher**: Theory-first approach → [Research Path](doc/learning_paths.md#research-focused-path)  
- **💻 I'm a Developer**: Code-first approach → [Developer Path](doc/learning_paths.md#developer-focused-path)
- **🎓 I'm Learning**: Structured curriculum → [Academic Path](doc/learning_paths.md#academic-learning-path)

**📚 Need guidance choosing?** → [Complete Learning Paths Guide](doc/learning_paths.md)

### 🛠️ **Direct Installation** (if you know what you want)

**1️⃣ Prerequisites**

Ensure you have **Python 3.11+** installed:

```bash
python --version  # Should show 3.11 or higher
```

**2️⃣ Clone Repository**

```bash
git clone https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation.git
cd GeneralizedNotationNotation
```

**3️⃣ Setup Environment**

Run the setup pipeline step to configure dependencies:

```bash
python src/main.py --only-steps 1 --dev
```

This will:

- ✅ Create and configure virtual environment
- 📦 Install all required dependencies
- 🧪 Install development dependencies (with `--dev`)
- ✅ Validate system requirements

**4️⃣ Enhanced Visual Output** *(Optional)*

The pipeline includes enhanced visual logging for better accessibility:

```bash
# Run with visual enhancements (recommended)
python src/main.py --verbose

# Run with compact visual mode for narrow terminals
python src/main.py --verbose --compact

# Run with timestamps for detailed logging
python src/main.py --verbose --timestamps
```

**Visual Features:**

- 🎨 **Color-coded status indicators** (green=success, yellow=warning, red=error)
- 📊 **Progress bars** and completion indicators
- 🔢 **Step-by-step visual progress** with correlation IDs
- 📋 **Structured summary tables** with key metrics
- ♿ **Screen reader friendly** output (emoji can be disabled)
- ⏱️ **Performance timing** and memory usage tracking

**5️⃣ Run Your First Pipeline (using `uv`)**

Use `uv` to run the pipeline inside the managed project environment:

```bash
uv sync                # ensure dependencies from pyproject.toml are installed
uv run python src/main.py -- --target-dir input/gnn_files --verbose
```

You can also run individual commands under `uv` (recommended):

```bash
uv run pytest          # run tests inside uv-managed venv
uv run python -m pytest
```

**6️⃣ Explore Results**

Check the generated outputs in the `output/` directory. The static site is under the numbered website folder:

```bash
ls -la output/
open output/20_website_output/website/index.html  # macOS
# or
xdg-open output/20_website_output/website/index.html  # Linux
```

### 🆘 Need Help?

<details>
<summary><strong>🔍 Common Issues & Solutions</strong></summary>

**🐍 Python Version Issues**

```bash
# Check Python version
python --version
# If < 3.11, install Python 3.11+ from python.org
```

**📦 Dependency Issues**

```bash
# Force reinstall dependencies
uv run python src/main.py --only-steps 2 --recreate-uv-env --dev
```

**🔧 Pipeline Failures**

```bash
# Run with verbose logging
python src/main.py --verbose
# Check specific step
python src/main.py --only-steps 5 --verbose
```

**💾 Disk Space Issues**

```bash
# Check available space
df -h
# Clean output directory
rm -rf output/*
```

</details>

**🔗 Get Support:**

- 📖 **Documentation**: See [Documentation](#-documentation) section below
- 🐛 **Known Issues**: Check [troubleshooting guide](./doc/troubleshooting/)
- 💬 **Community**: Open an issue on [GitHub](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/issues)
- 🚀 **Quick Fix**: Try `python src/main.py --only-steps 2 --dev` first

---

## 📖 Documentation

Comprehensive documentation is organized in the `doc/` directory.

> [!TIP]
> **Start Here for Architecture**:
>
> - **[AGENTS.md](./AGENTS.md)**: **Master Agent Scaffolding** - Detailed breakdown of every pipeline module and its agentic responsibilities.
> - **[DOCS.md](./DOCS.md)**: **Comprehensive Project Docs** - Full architecture, high-level diagrams, and integration points.

### 📚 Core Documentation

| Document | Description |
|----------|-------------|
| [**AGENTS.md**](./AGENTS.md) | **MUST READ**: The master guide to all pipeline agents and modules. |
| [**DOCS.md**](./DOCS.md) | High-level system architecture and comprehensive documentation index. |
| [**GNN Overview**](./doc/gnn/gnn_overview.md) | High-level introduction to the GNN language. |
| [**Syntax Guide**](./doc/gnn/reference/gnn_syntax.md) | Detailed GNN syntax specification. |
| [**File Structure**](./doc/gnn/reference/gnn_file_structure_doc.md) | Guide to GNN file organization. |
| [**Quick Start Tutorial**](./doc/gnn/tutorials/quickstart_tutorial.md) | Step-by-step beginner guide. |
| [**Architecture Guide**](./ARCHITECTURE.md) | Implementation, extension patterns, and system design. |
| [**Machine-Readable Indices**](./doc/api/README.md) | API index and generator. |

### 🎯 Specialized Guides

| Topic | Documentation |
|-------|---------------|
| **🧠 Active Inference** | [About GNN](./doc/gnn/about_gnn.md) |
| **🤖 LLM Integration** | [LLM & Neurosymbolic AI](./doc/gnn/advanced/gnn_llm_neurosymbolic_active_inference.md) |
| **📊 Implementation** | [Implementation Guide](./doc/gnn/integration/gnn_implementation.md) |
| **🛠️ Tools** | [Tools & Resources](./doc/gnn/operations/gnn_tools.md) |
| **📄 Research Paper** | [Academic Paper Details](./doc/gnn/gnn_paper.md) |

### 🎯 Integration Guides

| Platform | Documentation |
|----------|---------------|
| **🐍 PyMDP** | [PyMDP Integration](./doc/pymdp/) |
| **🔬 RxInfer.jl** | [RxInfer Integration](./doc/rxinfer/) |
| **🧠 ActiveInference.jl** | [ActiveInference.jl Integration](./doc/activeinference_jl/) |
| **📡 MCP** | [Model Context Protocol](./doc/mcp/) |
| **🧮 SymPy** | [Mathematical Processing](./doc/sympy/) |
| **🔄 DisCoPy** | [Categorical Diagrams](./doc/discopy/) |

### 🧩 Application Examples

| Domain | Examples |
|--------|----------|
| **🧠 Cognitive Phenomena** | [Cognitive Models](./doc/cognitive_phenomena/) |
| **🎯 Templates** | [Model Templates](./doc/templates/) |
| **📋 Configuration** | [Configuration Examples](./doc/configuration/) |

---

## 🎯 Examples

Explore practical GNN implementations and use cases:

### 📂 Example Files Location

- **📁 Primary Examples**: [`src/gnn/gnn_examples/`](./src/gnn/gnn_examples/)
- **📁 Cognitive Models**: [`doc/cognitive_phenomena/`](./doc/cognitive_phenomena/)
- **📁 Templates**: [`doc/templates/`](./doc/templates/)

### 🔥 Featured Examples

| Example | Description | Location |
|---------|-------------|----------|
| **🎯 PyMDP POMDP Agent** | Complete POMDP implementation | [`src/gnn/gnn_examples/actinf_pomdp_agent.md`](src/gnn/gnn_examples/actinf_pomdp_agent.md) |
| **🔬 RxInfer Hidden Markov Model** | Probabilistic sequence modeling | [`doc/archive/rxinfer_hidden_markov_model.md`](doc/archive/rxinfer_hidden_markov_model.md) |
| **🧠 ActiveInference.jl Examples** | Julia-based Active Inference models | [`doc/activeinference_jl/actinf_jl_src/`](doc/activeinference_jl/actinf_jl_src/) |
| **🤝 Multi-Agent System** | Collaborative agent modeling | [`doc/archive/rxinfer_multiagent_gnn.md`](doc/archive/rxinfer_multiagent_gnn.md) |

### 🧠 Cognitive Phenomena Examples

| Phenomenon | Model | Documentation |
|------------|-------|---------------|
| **🎯 Attention** | Attention mechanisms | [`doc/cognitive_phenomena/attention/`](./doc/cognitive_phenomena/attention/) |
| **🧠 Consciousness** | Global workspace theory | [`doc/cognitive_phenomena/consciousness/`](./doc/cognitive_phenomena/consciousness/) |
| **💪 Cognitive Effort** | Effort and control | [`doc/cognitive_phenomena/effort/`](./doc/cognitive_phenomena/effort/) |
| **❤️ Emotion & Affect** | Interoceptive emotion | [`doc/cognitive_phenomena/emotion_affect/`](./doc/cognitive_phenomena/emotion_affect/) |
| **🎮 Executive Control** | Task switching | [`doc/cognitive_phenomena/executive_control/`](./doc/cognitive_phenomena/executive_control/) |

### 🏃‍♂️ Running Examples

```bash
# Process all packaged examples (--target-dir is always a directory)
python src/main.py --target-dir src/gnn/gnn_examples

# Default pipeline input directory (includes actinf_pomdp_agent.md)
python src/main.py --target-dir input/gnn_files

# Process with full analysis
python src/main.py --target-dir src/gnn/gnn_examples --estimate-resources --verbose
```

### ⚡ Power User Tips

<details>
<summary><strong>🚀 Advanced Usage Patterns</strong></summary>

**🔥 Quick Commands**

```bash
# Full pipeline with all features
python src/main.py --verbose --estimate-resources --dev

# Type check only (fastest validation)  
python src/main.py --only-steps 5 --strict

# Visualization only (quick preview)
python src/main.py --only-steps 8

# Complete analysis for a directory of models
python src/main.py --target-dir path/to/gnn_models/ --verbose
```

**🎯 Pipeline Optimization**

```bash
# Skip time-consuming steps for quick iteration
python src/main.py --skip-steps "11,12,13"

# Focus on core processing
python src/main.py --only-steps "1,4,5,6"

# Development workflow
python src/main.py --only-steps "2,3" --dev
```

**📊 Output Management**

```bash
# Custom output directory
python src/main.py -o /path/to/custom/output

# Timestamped outputs
python src/main.py -o "output/run_$(date +%Y%m%d_%H%M%S)"
```

</details>

---

## 💚 Repository Health

The GNN project maintains high standards for code quality, testing, and documentation.

### Repository Health

- Pipeline orchestration, module docs, and tests are maintained together.
- Use current test and pipeline runs as the source of truth for operational status.
- See `src/tests/` and step-specific outputs in `output/` for current validation artifacts.

### 🧪 Testing Infrastructure

```bash
# Run comprehensive test suite
python src/2_tests.py --comprehensive

# Run fast pipeline tests (default)
python src/2_tests.py

# Check test coverage
pytest --cov=src --cov-report=term-missing

# Run specific module tests
pytest src/tests/test_[module]*.py -v
```

**Test Configuration:** See [pytest.ini](./pytest.ini) for complete test settings.

### 📋 Configuration Files

| File | Purpose |
|------|---------|
| **[pyproject.toml](./pyproject.toml)** | Project metadata, dependencies, and build configuration |
| **[pytest.ini](./pytest.ini)** | Test suite configuration with markers and coverage settings |
| **[input/config.yaml](./input/config.yaml)** | Pipeline default configuration |
| **[Dockerfile](./Dockerfile)** | Container image definition |

### 🔒 Security & Quality

- **Security Policy**: See [SECURITY.md](./SECURITY.md) for vulnerability reporting
- **Code of Conduct**: See [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md)
- **Contributing Guidelines**: See [CONTRIBUTING.md](./CONTRIBUTING.md)
- **GitHub automation** (CI, Dependabot, workflows): See [.github/README.md](./.github/README.md)

---

## 🤝 Contributing

GNN is an evolving standard, and **contributions are welcome**! Here's how you can get involved:

### 🎯 Ways to Contribute

- 🐛 **Report Issues**: Found a bug? [Open an issue](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/issues)
- 💡 **Suggest Features**: Have ideas? [Start a discussion](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/discussions)  
- 📝 **Improve Documentation**: Help make our docs better
- 🧪 **Add Examples**: Share your GNN models
- 🔧 **Code Contributions**: Submit pull requests

### 📋 Contribution Guidelines

1. **🍴 Fork** the repository
2. **🌿 Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **✅ Test** your changes thoroughly
4. **📝 Document** your changes
5. **💾 Commit** with clear messages (`git commit -m 'Add amazing feature'`)
6. **📤 Push** to your branch (`git push origin feature/amazing-feature`)
7. **🔄 Submit** a Pull Request

### 🛡️ Code of Conduct

Please read our [Code of Conduct](./CODE_OF_CONDUCT.md) to understand the standards we maintain for our community.

### 📞 Getting Help

- 📖 **Documentation**: Check the [docs](./doc/) first
- 💬 **Discussions**: Use [GitHub Discussions](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/discussions)
- 🐛 **Issues**: For bugs, use [GitHub Issues](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/issues)
- 📧 **Contact**: Reach out to the maintainers

### 🙌 Recognition

All contributors will be recognized in our [contributors list](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/graphs/contributors) and release notes.

---

## 📄 License

This project is licensed under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**. See [LICENSE.md](./LICENSE.md) for full terms.

### 📋 License Summary

- ✅ **Attribution** required
- ✅ **Adaptation/Redistribution** permitted under the same license
- ✅ **Private use** permitted
- ❌ **Commercial use** not permitted without explicit permission
- ❗ **Include license and changes notice** in redistributions

---

## 🔗 External Resources & Links

### 🌐 Active Inference Community

- 🏠 **Active Inference Institute**: [activeinference.institute](https://activeinference.institute/)
- 💬 **Community Discussions**: [Active Inference Discord](https://discord.activeinference.institute/)

### 🛠️ Technical Resources

- 🐍 **PyMDP Framework**: [github.com/infer-actively/pymdp](https://github.com/infer-actively/pymdp)
- 🔬 **RxInfer.jl**: [github.com/biaslab/RxInfer.jl](https://github.com/biaslab/RxInfer.jl)
- 🧠 **ActiveInference.jl**: [github.com/ComputationalPsychiatry/ActiveInference.jl](https://github.com/ComputationalPsychiatry/ActiveInference.jl)
- 📊 **DisCoPy**: [github.com/oxford-quantum-group/discopy](https://github.com/oxford-quantum-group/discopy)
- 📚 **Research Papers**: [Active Inference on arXiv](https://arxiv.org/search/?query=active+inference&searchtype=all)

<div align="center">

---

**Built by the Active Inference community**

[⬆️ Back to top](#generalizednotationnotation-gnn)

</div>
