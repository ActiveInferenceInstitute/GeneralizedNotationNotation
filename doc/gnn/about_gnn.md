# About GNN

**Version**: v1.1.0  
**Last Updated**: February 9, 2026  
**Status**: ✅ Production Ready  
**Test Count**: 1,083 Tests Passing  

Generalized Notation Notation (GNN) is a text-based language designed to standardize the representation of Active Inference generative models, improving clarity, reproducibility, and interoperability across domains. It defines a concise ASCII syntax for model components, a structured file format, and supports multiple modalities—textual, graphical, and executable—facilitating seamless communication among researchers and tools.

## Motivation

### GNN Structure and Sections

GNN is centered around several core objects that define the structure and behavior of Active Inference models:

### 1. Variables

Variables represent states, observations, actions, or other quantities in the model. They can be categorical, continuous, or discrete.

### 2. Edges

Edges represent the relationships between variables, such as dependencies, transitions, or connections. They define how information flows through the model.

GNN addresses the lack of a unified representation for Active Inference models, which are often described using disparate mixtures of natural language, pseudocode, mathematical formulas, and diagrams. By providing a formalized syntax and file structure, GNN aims to:

- Streamline collaboration and model sharing across research groups
- Enable automated rendering into mathematical notation, figures, and code
- Bridge gaps between ontologies, morphology, grammar, and pragmatics in Active Inference modeling
- Facilitate interdisciplinary research and application
- Ensure reproducibility of computational cognitive models
- Create a standardized way to document model implementations

## GNN Syntax and Logic Rules

GNN leverages standard ASCII symbols to denote variables, operations, and causal relationships in generative models. The complete specification is maintained in machine-readable format at `src/gnn/documentation/punctuation.md`.

The syntax is designed to be both human-readable and machine-parsable, making it suitable for documentation and automated processing. Key symbols include:

| Symbol       | Meaning                         | Example   | Interpretation                      |
|:-------------|:--------------------------------|:----------|:------------------------------------|
| ,            | List separator                  | X,Y       | Elements X and Y                    |
| _            | Subscript                       | X_2       | Variable X with subscript 2         |
| ^            | Superscript                     | X^Y       | Variable X with superscript Y       |
| =            | Equality or assignment          | X=5       | X is set to 5                       |
| >            | Directed causal edge            | X>Y       | Causal influence from X to Y        |
| -            | Undirected causal edge          | X-Y       | Undirected relation between X and Y |
| ()           | Grouping                        | (X+Y)     | Parenthesized expression            |
| {}           | Exact value specification       | X{1}      | X equals 1                          |
| []           | Dimensionality or indexing      | X[2,3]    | X is a 2×3 matrix                   |
| # / ## / ### | Markdown headings and comments  | ## Title  | Section header in GNN source file   |
| +            | Addition                        | X+Y       | Sum of X and Y                      |
| *            | Multiplication                  | X*Y       | Product of X and Y                  |
| /            | Division                        | X/Y       | X divided by Y                      |
| \|           | Conditional probability         | P(X\|Y)   | Probability of X given Y            |

### Syntax Guidelines

When writing GNN expressions:

1. **Variables** should be concise, meaningful identifiers
2. **Dimensionality** should be specified using square brackets, e.g., `X[2,3]` for a 2×3 matrix
3. **Causal relationships** should be denoted with directed (>) or undirected (-) edges
4. **Mathematical operations** should use standard operators (+, -, *, /)
5. **Comments** can be added using triple hashtags (###)

## GNN Source File Structure

A GNN source file follows a Markdown-like organization that segments model metadata, variables, connections, and equations. The complete specification is maintained in machine-readable format at `src/gnn/documentation/file_structure.md`.

Each GNN file is organized into the following sections:

```gnn
## GNNVersionAndFlags
GNN v1
Specification of GNN release and optional flags that govern the file's interpretation.
```

1. **Model Name**: Descriptive identifier for the model, providing a concise label for reference.

2. **Model Annotation**: Free-text caption explaining the model's purpose, context, and key features. This section allows for more detailed description than the name alone.

3. **State Space Block**: Definitions of variables and their dimensions/types. This section specifies all variables used in the model, including their dimensionality (e.g., scalar, vector, matrix) and data types.

4. **Connections**: Directed or undirected edges specifying dependencies between variables. This section defines the graphical structure of the model, showing how variables influence each other.

5. **Initial Parameterization**: Starting values for parameters and variables, which may include constants, distributions, or specific values for model initialization.

6. **Equations**: LaTeX-rendered formulas defining model dynamics and the mathematical relationships between variables. These equations specify how variables change over time or in response to inputs.

7. **Time**: Discrete or continuous time settings and horizons, including whether the model is static or dynamic, how time is represented, and the time horizon for simulation.

8. **ActInf Ontology Annotation**: Mapping of variables to Active Inference Ontology terms, which standardizes the interpretation of variables and facilitates cross-model comparison.

9. **Footer and Signature**: File closure and provenance information, potentially including cryptographic signatures for verification.

### Machine-Readable Format

The GNN file structure is designed to be machine-readable, with each section clearly delineated by Markdown headers. This allows for automated parsing and processing of GNN files, enabling:

- Validation of GNN syntax and structure  
- Automatic conversion to computational implementations
- Visualization of model structure

### Cross-model comparison and analysis

### Punctuation and Representation

### GNN Processing Pipeline

GNN files are processed through a comprehensive 25-step pipeline orchestrated by **`src/main.py`**. The pipeline handles:

### Parsing and Validation (Steps 3, 5, 6)

- `3_gnn.py`: GNN file discovery and multi-format parsing
- `5_type_checker.py`: Type checking and resource estimation
- `6_validation.py`: Advanced validation and consistency checking

### Rendering and Execution (Steps 11, 12)

- `11_render.py`: Code generation for PyMDP, RxInfer, ActiveInference.jl, DisCoPy, JAX
- `12_execute.py`: Execution of rendered simulation scripts

### Analysis and Reporting (Steps 13, 16, 23)

- `13_llm.py`: LLM-enhanced analysis and model interpretation
- `16_analysis.py`: Advanced statistical analysis
- `23_report.py`: Comprehensive report generation

For complete pipeline documentation, see:

- **[src/AGENTS.md](../../src/AGENTS.md)**: Master agent scaffolding and module registry
- **[src/README.md](../../src/README.md)**: Pipeline architecture and safety documentation
- **[GNN Tools and Resources](gnn_tools.md)**: Detailed pipeline usage examples

**Quick Start:**

```bash
# Process a GNN model through the full pipeline
python src/main.py --target-dir input/gnn_files --verbose
```

```bash
# Parse a GNN file using the dedicated parser
python src/gnn/gnn_parser.py --input input/gnn_files/model.gnn --output output/gnn.json
```

```bash
# Run specific steps
python src/main.py --only-steps "3,5,11,12" --target-dir input/gnn_files
```

```bash
# Step 3: GNN Core Processing
python src/main.py --target-dir input/gnn_files
 --verbose
```

```bash
# Step 1: Environment Setup
python src/1_setup.py
```

## Progressive Model Development with GNN

GNN supports an incremental approach to model development, allowing practitioners to start with simple models and progressively add complexity. Models can be extended by:

1. Adding new variables to the state space
2. Introducing additional connections between variables
3. Refining equations to capture more complex dynamics
4. Incorporating temporal components for dynamic models
5. Adding policy selection mechanisms for active inference

The examples in `src/gnn/gnn_examples/` demonstrate this progression from simple to complex models, following the tutorial by Smith et al. (2022).

## Step-by-Step Example: Static to Dynamic Models

The progression from static to dynamic models in GNN typically follows these steps:

1. **Static Perception Model**: Basic model with hidden states, observations, and recognition matrix.

2. **Dynamic Perception Model**: Adds time dimension and transition matrices to model state changes over time.

3. **Dynamic Perception with Policy Selection**: Incorporates action selection through policy variables and expected free energy.

4. **Dynamic Perception with Flexible Policy Selection**: Adds preference learning and adaptive policy selection mechanisms.

### Example: Simple Perception Model in GNN

```gnn
# Simple Perception Model
## Model Annotation
A basic model of perceptual inference with hidden states and observations.

## State Space Block
s[2]    # Hidden state with 2 possible values
o[2]    # Observation with 2 possible values

## Connections
s>o     # Hidden states cause observations

## Initial Parameterization
A=[[0.7,0.3],[0.3,0.7]]  # State transition matrix
B=[[0.9,0.1],[0.1,0.9]]  # Observation likelihood matrix

## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon=10

## ActInf Ontology Annotation
s=HiddenState
o=Observation
A=StateTransitionMatrix
B=ObservationLikelihoodMatrix
```

## The Triple Play: Modalities

GNN expressions support three complementary modalities for model representation, which can be used together for comprehensive model documentation:

1. **Text-Based Models**: Plain-text GNN files render directly into formulas, pseudocode, or prose using tools like regex or LLMs. This modality provides a human-readable specification that can be shared in papers, documentation, and educational materials.

2. **Graphical Models**: GNN specifies nodes and edges that can be visualized as factor graphs, clarifying dependencies and causal structure. The graphical representation enhances understanding of model architecture and facilitates communication of complex relationships.

3. **Executable Cognitive Models**: GNN serves as pseudocode to generate simulations in any programming environment, ensuring backwards- and forwards-compatibility across software implementations. This modality enables computational verification of model behavior and supports empirical research.

### Benefits of the Triple Play Approach

- **Completeness**: Different aspects of the model are captured in complementary ways
- **Accessibility**: Various stakeholders can engage with the representation that suits their needs
- **Verification**: Multiple representations allow cross-checking for consistency
- **Interoperability**: Models can be shared and implemented across different platforms

## Implementation Guidelines

### Translating GNN to Code

When implementing a GNN model in code:

1. **Variable Declaration**: Define variables according to their dimensionality in the State Space Block
2. **Initialization**: Use the Initial Parameterization section to set starting values
3. **Function Definition**: Implement equations as functions in your target language
4. **Temporal Logic**: For dynamic models, implement time steps according to the Time section
5. **Connections**: Ensure that variable dependencies follow the structure defined in the Connections section

### Visualization Tools

Visualization of GNN models can be implemented using:

- **Graph libraries**: NetworkX, Graphviz, or D3.js for rendering connection structures
- **LaTeX renderers**: For equation display in documentation
- **Custom renderers**: For specialized visualizations of model components

### Validation Process

To validate a GNN implementation:

1. Check that all variables have the correct dimensions
2. Verify that all connections are properly implemented
3. Compare equation implementations against the LaTeX definitions
4. Test temporal behavior for dynamic models
5. Validate against expected outcomes or benchmark datasets

## Tools and Resources

To work with GNN, several tools and approaches can be used:

- **Parsing**: Standard CSV parsers can process the machine-readable specifications
- **Visualization**: Graph visualization libraries can render GNN connections
- **Simulation**: The model structure can be translated to executable code
- **Verification**: GNN files can be validated against the specification
- **GNN Repository**: Access examples and tools at the [GNN GitHub repository](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation)
- **Active Inference Ontology**: Map variables to standardized terms using the [Active Inference Ontology](https://doi.org/10.5281/zenodo.7430333)

## Contributing to GNN

GNN is an evolving standard maintained by the Active Inference Institute. Contributions are welcome through the official repository:

1. **Documentation**: Improve explanations, add examples, or clarify specifications
2. **Tools**: Develop parsers, renderers, or validators for GNN files
3. **Examples**: Contribute model implementations that showcase GNN features
4. **Extensions**: Propose enhancements to the GNN specification for new model types

## Future Directions

GNN's development opens avenues for enhancing Active Inference modeling:

- **Automated rendering tools**: Linking GNN with mathematical typesetting and diagram generators
- **Ontology integration**: Deeper integration with the Active Inference Ontology for semantic consistency
- **Linguistic analysis**: Application of formal linguistics and semiotic analysis to evolve GNN grammar and dialects
- **Systems engineering**: Use of frameworks (e.g., cadCAD) for specifying execution order and performing parameter sweeps
- **Model repositories**: Standardized collections of GNN models for benchmarking and educational purposes
- **Interoperability**: Translations between GNN and other modeling frameworks
- **Educational resources**: Tutorials and courses on using GNN for cognitive modeling

By providing a rigorous yet flexible notation, GNN fosters reproducible, accessible, and interoperable Active Inference models, advancing interdisciplinary research in cognitive science.

## See Also

### Core Documentation

- **[GNN Overview](gnn_overview.md)**: High-level concepts and ecosystem overview
- **[GNN Syntax](gnn_syntax.md)**: Complete syntax reference
- **[GNN Examples](gnn_examples_doc.md)**: Example models from simple to complex
- **[Quickstart Tutorial](quickstart_tutorial.md)**: Step-by-step getting started guide

### Advanced Topics

- **[Advanced Modeling Patterns](advanced_modeling_patterns.md)**: Hierarchical and sophisticated techniques
- **[Multi-agent Systems](gnn_multiagent.md)**: Multi-agent modeling specification
- **[LLM Integration](gnn_llm_neurosymbolic_active_inference.md)**: AI-assisted modeling

### Framework Integration

- **[PyMDP Integration](../pymdp/gnn_pymdp.md)**: Python Active Inference framework
- **[RxInfer Integration](../rxinfer/gnn_rxinfer.md)**: Julia Bayesian inference
- **[DisCoPy Integration](../discopy/gnn_discopy.md)**: Category theory integration

### Pipeline and Tools

- **[Pipeline Documentation](../gnn/gnn_tools.md)**: Complete 25-step pipeline guide
- **[GNN Tools](gnn_tools.md)**: Available software and utilities
- **[Ontology System](ontology_system.md)**: Active Inference ontology processing

### Navigation

- **[Main Documentation Index](../README.md)**: Return to main documentation hub
- **[Cross-Reference Index](../CROSS_REFERENCE_INDEX.md)**: Complete cross-reference system
- **[Learning Paths](../learning_paths.md)**: Structured learning pathways

## References

1. Smékal, J., & Friedman, D. A. (2023). Generalized Notation Notation for Active Inference Models. Active Inference Institute. <https://doi.org/10.5281/zenodo.7803328>
2. Active Inference Institute: Generalized Notation Notation (GNN) Github repo: <https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation>
3. Active Inference Institute: Generalized Notation Notation (GNN) Coda: <https://coda.io/@active-inference-institute/generalized-notation-notation>
4. Smith, R., Friston, K.J., & Whyte, C.J. (2022). A step-by-step tutorial on active inference and its application to empirical data. Journal of Mathematical Psychology, 107, 102632.
5. Friston, K. J., Parr, T., & de Vries, B. (2017). The graphical brain: belief propagation and active inference. Network Neuroscience, 1(4), 381-414.
6. Parr, T., Pezzulo, G., & Friston, K. J. (2022). Active Inference: The Free Energy Principle in Mind, Brain, and Behavior. MIT Press.
