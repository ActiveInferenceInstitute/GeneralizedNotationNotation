# GNN Domain Knowledge and Standards

## GNN File Structure Understanding
- **GNN Files**: Multi-format support including Markdown (.md), JSON (.json), YAML (.yml/.yaml), XML (.xml), Binary Pickle (.pkl), Protobuf (.proto), Maxima (.max), and more
- **Markdown Format** (Primary): Markdown-based (.md) with specific sections:
  - `GNNVersionAndFlags`: Version specification and processing flags
  - `ModelName`: Descriptive model identifier
  - `ModelAnnotation`: Free-text explanation of model purpose and features
  - `StateSpaceBlock`: Variable definitions with dimensions/types (s_fX[dims,type])
  - `Connections`: Directed/undirected edges showing dependencies (>, -, ->)
  - `InitialParameterization`: Starting values, matrices (A, B, C, D), priors
  - `Equations`: LaTeX-rendered mathematical relationships
  - `Time`: Temporal settings (Dynamic/Static, DiscreteTime, ModelTimeHorizon)
  - `ActInfOntologyAnnotation`: Mapping to Active Inference Ontology terms
  - `Footer` and `Signature`: Provenance information

## GNN Syntax and Punctuation
- **Variables**: Use underscore for subscripts (X_2), caret for superscripts (X^Y)
- **Dimensions**: Square brackets for array dimensions [2,3] = 2x3 matrix
- **Causality**: `>` for directed edges (X>Y), `-` for undirected (X-Y)
- **Operations**: Standard math operators (+, -, *, /, |)
- **Grouping**: Parentheses (), exact values {1}, indexing/dimensions [2,3]
- **Comments**: Triple hashtags (###) for inline comments
- **Probability**: Conditional probability notation P(X|Y) using pipe |
- **Unicode Support**: Proper handling of mathematical symbols like π, α, β

## Multi-Format GNN Support
- **Format Detection**: Automatic content type detection and format-appropriate parsing
- **Round-Trip Compatibility**: All formats support embedding complete JSON representation for semantic preservation
- **Cross-Format Validation**: Consistency checking across different format representations
- **Supported Formats**:
  - **Markdown (.md)**: Primary human-readable format with structured sections
  - **JSON (.json)**: Structured data format with complete metadata preservation
  - **YAML (.yml/.yaml)**: Human-readable structured format with inline comments
  - **XML (.xml)**: Structured markup with schema validation support
  - **Binary Pickle (.pkl)**: Python-native serialization for performance-critical applications
  - **Protobuf (.proto)**: Language-neutral serialization with schema evolution
  - **Maxima (.max)**: Computer algebra system integration for symbolic computation

## Validation Levels and Standards
The GNN system supports multiple validation levels for different use cases:

### BASIC Validation
- File format detection and basic syntax checking
- Required section presence validation
- Basic variable name consistency

### STANDARD Validation  
- Complete syntax validation with detailed error reporting
- Variable type checking and dimension validation
- Connection consistency and graph structure validation
- Active Inference ontology compliance checking

### STRICT Validation
- Enhanced type checking with mathematical constraint validation
- Resource estimation and computational complexity analysis
- Performance optimization recommendations
- Advanced syntax pattern validation

### RESEARCH Validation
- Scientific reproducibility validation
- Comprehensive Active Inference model structure validation
- Advanced mathematical relationship checking
- Publication-ready model validation

### ROUND_TRIP Validation
- Semantic preservation testing across format conversions
- Cross-format consistency validation
- Data integrity verification after format transformations
- Complete fidelity testing for all supported formats

## Active Inference Model Standards
- **Standard Variables**: Recognize and validate standard Active Inference variables (A, B, C, D, π, G, s, o, u, E, F)
- **POMDP Structure**: Validate Partially Observable Markov Decision Process components
- **Bayesian Mechanics**: Ensure proper prior and posterior specifications
- **Temporal Dynamics**: Validate time-dependent relationships and update rules
- **Ontology Mapping**: Map model components to Active Inference Ontology terms
- **Habit Vectors**: Support for habit priors (E_c) and policy bias mechanisms

## Processing Standards
- **Real Implementation**: All parsing and validation must be fully functional, no mock or stub implementations
- **Performance Tracking**: All operations should be monitored for performance and resource usage
- **Error Recovery**: Graceful handling of parsing errors with detailed diagnostic information
- **Semantic Preservation**: Maintain model semantics across all format transformations
- **Scientific Rigor**: Ensure mathematical accuracy and Active Inference theoretical compliance

## Quality Assurance Standards
- **Comprehensive Testing**: Unit tests, integration tests, and round-trip validation
- **Documentation**: Complete docstrings and type hints for all GNN processing functions
- **Reproducibility**: Deterministic parsing and validation results
- **Extensibility**: Modular design allowing easy addition of new formats and validation rules 