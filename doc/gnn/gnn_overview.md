# Generalized Notation Notation (GNN): Overview

## What is GNN?

Generalized Notation Notation (GNN) is a standardized text-based language designed to express Active Inference generative models with precision and clarity. It provides a structured framework for representing cognitive models in a way that is:

- **Human-readable**: Easy to understand and communicate
- **Machine-parsable**: Can be processed by software tools
- **Interoperable**: Works across different platforms and frameworks
- **Reproducible**: Enables exact replication of model specifications

GNN bridges the gap between theoretical models and their implementation by defining a consistent syntax and file structure for representing Active Inference models.

```mermaid
graph TD
    GNN[Generalized Notation Notation]
    AI[Active Inference Theory]
    RM[Research Models]
    SIM[Simulations]
    VIS[Visualizations]
    CODE[Code Implementation]
    DOC[Documentation]
    ONTO[Ontology]
    
    AI --> GNN
    GNN --> RM
    GNN --> SIM
    GNN --> VIS
    GNN --> CODE
    GNN --> DOC
    ONTO <--> GNN
    
    classDef core fill:#f96,stroke:#333,stroke-width:2px;
    classDef output fill:#9cf,stroke:#333,stroke-width:1px;
    classDef input fill:#fcf,stroke:#333,stroke-width:1px;
    
    class GNN core;
    class AI,ONTO input;
    class RM,SIM,VIS,CODE,DOC output;
```

## Motivation and Goals

Active Inference research has historically been challenged by the lack of a unified notation system. Models are often described through a mixture of:

- Natural language explanations
- Mathematical formulas
- Diagrams and schematics 
- Programming code
- Pseudocode implementations

This fragmentation leads to difficulties in:
- **Communication**: Complex models become hard to explain
- **Replication**: Research findings are difficult to reproduce
- **Implementation**: Translating models to code introduces inconsistencies
- **Comparison**: Different models are hard to contrast systematically

GNN addresses these challenges by providing a standardized, structured approach to cognitive model representation.

```mermaid
flowchart LR
    subgraph Problems
        P1[Communication Barriers]
        P2[Implementation Inconsistencies]
        P3[Replication Challenges]
        P4[Limited Interoperability]
    end
    
    subgraph GNN
        S1[Standardized Syntax]
        S2[Structured File Format]
        S3[Triple Play Approach]
        S4[Machine Readability]
    end
    
    subgraph Benefits
        B1[Clearer Communication]
        B2[Consistent Implementation]
        B3[Improved Reproducibility]
        B4[Enhanced Collaboration]
    end
    
    Problems --> GNN
    GNN --> Benefits
```

## Core Objectives

GNN aims to:

1. **Streamline collaboration** across research groups by providing a common language
2. **Enable automated rendering** of models into mathematical notation, figures, and code
3. **Bridge disciplinary gaps** between theoretical, computational, and experimental approaches
4. **Ensure reproducibility** of computational cognitive models
5. **Standardize documentation** for model implementations
6. **Facilitate knowledge transfer** between researchers at different expertise levels

## The Triple Play Approach

A key feature of GNN is its support for three complementary modalities of representation:

```mermaid
graph TB
    GNN[GNN Specification] --> TXT[Text-Based Models]
    GNN --> GRAPH[Graphical Models]
    GNN --> EXEC[Executable Models]
    
    TXT --> DOCS[Documentation/Papers]
    TXT --> MATH[Mathematical Formulas]
    
    GRAPH --> FG[Factor Graphs]
    GRAPH --> DEP[Dependency Networks]
    
    EXEC --> SIM[Simulations]
    EXEC --> CODE[Program Code]
    
    classDef primary fill:#f96,stroke:#333,stroke-width:2px;
    classDef secondary fill:#9cf,stroke:#333,stroke-width:1px;
    classDef tertiary fill:#fcf,stroke:#333,stroke-width:1px;
    
    class GNN primary;
    class TXT,GRAPH,EXEC secondary;
    class DOCS,MATH,FG,DEP,SIM,CODE tertiary;
```

1. **Text-Based Models**: Plain-text GNN files render directly into formulas, pseudocode, or prose, making them ideal for documentation, papers, and educational materials.

2. **Graphical Models**: GNN specifies nodes and edges that visualize as factor graphs, clarifying model architecture and causal relationships between variables.

3. **Executable Models**: GNN serves as pseudocode for generating simulations in any programming environment, ensuring consistent implementation across platforms.

This three-fold approach enables GNN to serve diverse needs within the scientific community while maintaining consistency and interoperability.

## GNN in the Active Inference Ecosystem

GNN is part of a broader ecosystem of tools and concepts in Active Inference research:

```mermaid
graph LR
    AI[Active Inference]
    FEP[Free Energy Principle]
    AIO[Active Inference Ontology]
    GNN[Generalized Notation Notation]
    AIMOD[AI Models Repository]
    SOFT[Software Implementations]
    
    FEP --> AI
    AI --> AIO
    AIO <--> GNN
    GNN --> AIMOD
    GNN --> SOFT
    
    classDef theory fill:#fcf,stroke:#333,stroke-width:1px;
    classDef tools fill:#9cf,stroke:#333,stroke-width:1px;
    classDef impl fill:#cfc,stroke:#333,stroke-width:1px;
    
    class FEP,AI theory;
    class AIO,GNN tools;
    class AIMOD,SOFT impl;
```

- **Active Inference Theory**: Provides the theoretical foundation for GNN models
- **Active Inference Ontology**: Defines standardized terms that GNN variables map to
- **Model Repositories**: Collections of GNN models for reference and reuse
- **Software Tools**: Implementations that can parse, visualize, and execute GNN models. In this project, these tools are primarily accessed and orchestrated via the `src/main.py` processing pipeline.

## Related Documentation

For more detailed information about specific aspects of GNN, refer to:

- [GNN Syntax and Punctuation](gnn_syntax.md): Detailed specification of the notation symbols
- [GNN File Structure](gnn_file_structure_doc.md): Comprehensive guide to GNN file organization
- [GNN Examples](gnn_examples_doc.md): Step-by-step examples and use cases
- [GNN Implementation Guide](gnn_implementation.md): Guidelines for implementing GNN in practice
- [GNN Tools and Resources](gnn_tools.md): Available tools and resources for working with GNN
- [GNN Future Directions](gnn_future.md): Roadmap and future development plans

## References

1. Smékal, J., & Friedman, D. A. (2023). Generalized Notation Notation for Active Inference Models. Active Inference Institute. https://doi.org/10.5281/zenodo.7803328
2. Active Inference Institute: Generalized Notation Notation (GNN) Github repo: https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation
3. Friston, K. J., Parr, T., & de Vries, B. (2017). The graphical brain: belief propagation and active inference. Network Neuroscience, 1(4), 381-414. 