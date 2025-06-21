/*
 * GNN (Generalized Notation Notation) Alloy Model
 * This file provides an Alloy specification for GNN models, enabling
 * formal verification and constraint checking of model properties.
 */

module GNN

// Basic signature for identifiers
abstract sig Identifier {}

// Version specification
sig Version {
    major: one Int,
    minor: one Int,
    patch: one Int
} {
    major >= 1 and major <= 10
    minor >= 0 and minor <= 99
    patch >= 0 and patch <= 999
}

// Data types supported in GNN
enum DataType {
    Categorical, Continuous, Binary, Integer, Float, Complex
}

// Variable types in Active Inference
abstract sig VariableType {}
sig HiddenState extends VariableType { factor: one Int }
sig Observation extends VariableType { modality: one Int }
sig Action extends VariableType { control: one Int }
sig Policy extends VariableType { control: one Int }

// Processing flags
sig ProcessingFlags {
    strictValidation: one Bool,
    allowExperimental: one Bool,
    enableOptimizations: one Bool,
    debugMode: one Bool,
    verboseLogging: one Bool
}

// Default processing flags
one sig DefaultFlags extends ProcessingFlags {} {
    strictValidation = True
    allowExperimental = False
    enableOptimizations = True
    debugMode = False
    verboseLogging = False
}

// Dimension specification
sig Dimension {
    size: one Int
} {
    size > 0
}

// Variable definition
sig Variable {
    name: one Identifier,
    varType: one VariableType,
    dimensions: some Dimension,
    dataType: one DataType,
    description: lone String
}

// Valid variable naming convention
pred validVariableName[v: Variable] {
    (v.varType in HiddenState implies v.name in StateVariableId) or
    (v.varType in Observation implies v.name in ObservationVariableId) or
    (v.varType in Action implies v.name in ActionVariableId) or
    (v.varType in Policy implies v.name in PolicyVariableId)
}

// State variable identifiers (s_f0, s_f1, ...)
abstract sig StateVariableId extends Identifier {}
abstract sig ObservationVariableId extends Identifier {}
abstract sig ActionVariableId extends Identifier {}
abstract sig PolicyVariableId extends Identifier {}

// Connection types
enum ConnectionType {
    Directed, Undirected, Conditional, Bidirectional
}

// Connection symbols
enum ConnectionSymbol {
    Greater, Arrow, Dash, Pipe, BiArrow
}

// Connection between variables
sig Connection {
    sourceVars: some Variable,
    targetVars: some Variable,
    connType: one ConnectionType,
    symbol: one ConnectionSymbol,
    description: lone String,
    weight: lone Int
}

// Symbol-type consistency
pred validConnectionSymbol[c: Connection] {
    (c.connType = Directed implies c.symbol in (Greater + Arrow)) and
    (c.connType = Undirected implies c.symbol = Dash) and
    (c.connType = Conditional implies c.symbol = Pipe) and
    (c.connType = Bidirectional implies c.symbol = BiArrow)
}

// Mathematical constraints
enum Constraint {
    Stochastic, NonNegative, Symmetric, Orthogonal, Unitary, Normalized
}

// Parameter values (simplified)
abstract sig ParameterValue {}
sig ScalarValue extends ParameterValue { value: one Int }
sig VectorValue extends ParameterValue { values: some Int }
sig MatrixValue extends ParameterValue { values: some Int }

// Parameter assignment
sig ParameterAssignment {
    paramName: one Identifier,
    value: one ParameterValue,
    constraints: set Constraint,
    description: lone String
}

// Mathematical equation
sig Equation {
    latexExpr: one String,
    plaintextForm: lone String,
    variables: set Variable,
    description: lone String
}

// Time configuration
sig TimeConfiguration {
    timeType: one TimeType,
    discretization: one Discretization,
    horizon: one Int,
    timeStep: one Int
} {
    horizon > 0
    timeStep > 0
    (timeType = Static implies horizon = 1)
    (timeType = Dynamic implies horizon > 1)
}

enum TimeType { Static, Dynamic }
enum Discretization { DiscreteTime, ContinuousTime }

// Ontology mapping
sig OntologyMapping {
    variable: one Variable,
    ontologyTerm: one Identifier,
    namespace: one String,
    confidence: one Int
} {
    confidence >= 0 and confidence <= 100
    namespace = "ActInfOntology"
}

// State space definition
sig StateSpace {
    factors: some Variable,
    jointDimension: one Int,
    description: lone String
} {
    all v: factors | v.varType in HiddenState
    jointDimension > 0
}

// Observation space definition
sig ObservationSpace {
    modalities: some Variable,
    jointDimension: one Int,
    description: lone String
} {
    all v: modalities | v.varType in Observation
    jointDimension > 0
}

// Action space definition
sig ActionSpace {
    controls: set Variable,
    jointDimension: one Int,
    description: lone String
} {
    all v: controls | v.varType in (Action + Policy)
    jointDimension > 0
}

// Active Inference matrices
abstract sig AIMatrix {
    name: one Identifier,
    dimensions: some Dimension,
    constraints: set Constraint,
    description: lone String
}

sig LikelihoodMatrix extends AIMatrix {
    values: set Int
} {
    name in AMatrixId
    Stochastic in constraints
    NonNegative in constraints
}

sig TransitionMatrix extends AIMatrix {
    values: set Int
} {
    name in BMatrixId
    Stochastic in constraints
    NonNegative in constraints
}

sig PreferenceVector extends AIMatrix {
    values: set Int
} {
    name in CMatrixId
}

sig PriorVector extends AIMatrix {
    values: set Int
} {
    name in DMatrixId
    Stochastic in constraints
    NonNegative in constraints
}

// Matrix identifiers
abstract sig AMatrixId extends Identifier {}
abstract sig BMatrixId extends Identifier {}
abstract sig CMatrixId extends Identifier {}
abstract sig DMatrixId extends Identifier {}

// Active Inference model structure
sig ActiveInferenceModel {
    stateSpace: one StateSpace,
    observationSpace: one ObservationSpace,
    actionSpace: lone ActionSpace,
    likelihoodMatrices: some LikelihoodMatrix,
    transitionMatrices: some TransitionMatrix,
    preferenceVectors: set PreferenceVector,
    priorVectors: some PriorVector,
    timeHorizon: one Int,
    description: lone String
} {
    timeHorizon > 0
}

// Validation levels
enum ValidationLevel {
    Basic, Standard, Strict, Research
}

// Validation result
sig ValidationResult {
    isValid: one Bool,
    errors: set String,
    warnings: set String,
    suggestions: set String
}

// Complete GNN model specification
sig GNNModel {
    gnnSection: one String,
    version: one Version,
    processingFlags: one ProcessingFlags,
    modelName: one String,
    modelAnnotation: one String,
    variables: some Variable,
    connections: set Connection,
    aiModel: one ActiveInferenceModel,
    equations: set Equation,
    timeConfig: one TimeConfiguration,
    initialParams: set ParameterAssignment,
    modelParams: set ParameterAssignment,
    ontologyMappings: set OntologyMapping,
    footer: lone String
} {
    gnnSection = "GNN"
    // Variable name uniqueness
    all disj v1, v2: variables | v1.name != v2.name
    // All variables in connections must be defined
    all c: connections | c.sourceVars + c.targetVars in variables
    // All ontology mappings must reference defined variables
    all om: ontologyMappings | om.variable in variables
}

// Well-formed GNN model
sig WellFormedGNNModel extends GNNModel {} {
    // All variables have valid names
    all v: variables | validVariableName[v]
    // All connections have valid symbols
    all c: connections | validConnectionSymbol[c]
    // State space factors are subset of variables
    aiModel.stateSpace.factors in variables
    // Observation space modalities are subset of variables
    aiModel.observationSpace.modalities in variables
    // Action space controls are subset of variables (if defined)
    some aiModel.actionSpace implies aiModel.actionSpace.controls in variables
}

// Model morphism
sig ModelMorphism {
    source: one ActiveInferenceModel,
    target: one ActiveInferenceModel,
    stateMapping: StateSpace -> StateSpace,
    obsMapping: ObservationSpace -> ObservationSpace,
    actionMapping: ActionSpace -> ActionSpace
}

// Identity morphism
pred identityMorphism[m: ModelMorphism, model: ActiveInferenceModel] {
    m.source = model
    m.target = model
    m.stateMapping = (model.stateSpace -> model.stateSpace)
    m.obsMapping = (model.observationSpace -> model.observationSpace)
    some model.actionSpace implies 
        m.actionMapping = (model.actionSpace -> model.actionSpace)
}

// Morphism composition
pred composeMorphisms[result: ModelMorphism, f: ModelMorphism, g: ModelMorphism] {
    f.target = g.source
    result.source = f.source
    result.target = g.target
    // Composition of mappings (simplified)
    result.stateMapping = f.stateMapping.g.stateMapping
    result.obsMapping = f.obsMapping.g.obsMapping
    result.actionMapping = f.actionMapping.g.actionMapping
}

// Categorical structure properties
pred categoryAxioms {
    // Identity laws
    all m: ModelMorphism, model: ActiveInferenceModel |
        m.source = model implies
        (some id: ModelMorphism | 
            identityMorphism[id, model] and
            (some comp: ModelMorphism | composeMorphisms[comp, m, id] and comp = m))
    
    // Associativity (simplified)
    all f, g, h: ModelMorphism |
        (f.target = g.source and g.target = h.source) implies
        (some comp1, comp2, comp3, comp4: ModelMorphism |
            composeMorphisms[comp1, f, g] and
            composeMorphisms[comp2, comp1, h] and
            composeMorphisms[comp3, g, h] and
            composeMorphisms[comp4, f, comp3] and
            comp2 = comp4)
}

// Coherence conditions for Active Inference
pred coherentAIModel[model: ActiveInferenceModel] {
    // Dimension compatibility between likelihood and state space
    all A: model.likelihoodMatrices |
        #A.dimensions = 2
    
    // Transition matrices have appropriate dimensions
    all B: model.transitionMatrices |
        #B.dimensions >= 2
    
    // Prior vectors are one-dimensional
    all D: model.priorVectors |
        #D.dimensions = 1
}

// Export configuration
sig ExportConfiguration {
    formats: set ExportFormat,
    includeMetadata: one Bool,
    prettify: one Bool,
    compression: one Bool
}

enum ExportFormat {
    JSON, XML, YAML, GraphML, Pickle
}

// Model collection
sig ModelCollection {
    models: some WellFormedGNNModel,
    collectionName: lone String,
    description: lone String
}

// Performance metrics
sig PerformanceMetrics {
    parseTime: one Int,
    validationTime: one Int,
    exportTime: one Int,
    memoryUsage: one Int,
    modelComplexity: one Int
} {
    parseTime >= 0
    validationTime >= 0
    exportTime >= 0
    memoryUsage >= 0
    modelComplexity >= 0
}

// Complexity estimation
fun estimateComplexity[model: GNNModel]: Int {
    model.aiModel.stateSpace.jointDimension.mul[
        model.aiModel.observationSpace.jointDimension].mul[
        model.timeConfig.horizon]
}

// Well-formedness predicates
pred wellFormedVariable[v: Variable] {
    validVariableName[v]
    #v.dimensions > 0
}

pred wellFormedConnection[c: Connection] {
    validConnectionSymbol[c]
    #c.sourceVars > 0
    #c.targetVars > 0
}

pred wellFormedGNNModel[model: GNNModel] {
    all v: model.variables | wellFormedVariable[v]
    all c: model.connections | wellFormedConnection[c]
    coherentAIModel[model.aiModel]
}

// Facts and constraints
fact {
    // Every well-formed GNN model satisfies coherence conditions
    all model: WellFormedGNNModel | wellFormedGNNModel[model]
    
    // Category theory axioms hold
    categoryAxioms
    
    // Variable types are consistent with their names
    all v: Variable | validVariableName[v]
}

// Assertions for verification
assert VariableNameUniqueness {
    all model: WellFormedGNNModel |
        all disj v1, v2: model.variables | v1.name != v2.name
}

assert ConnectionValidity {
    all model: WellFormedGNNModel |
        all c: model.connections |
            c.sourceVars + c.targetVars in model.variables
}

assert AIModelCoherence {
    all model: WellFormedGNNModel |
        coherentAIModel[model.aiModel]
}

assert CategoryLaws {
    categoryAxioms
}

// Predicates for model checking
pred someWellFormedModel {
    some WellFormedGNNModel
}

pred modelWithConnections {
    some model: WellFormedGNNModel |
        #model.connections > 0
}

pred modelWithMultipleFactors {
    some model: WellFormedGNNModel |
        #model.aiModel.stateSpace.factors > 1
}

pred modelWithActions {
    some model: WellFormedGNNModel |
        some model.aiModel.actionSpace
}

// Example instances
pred exampleTwoStateModel {
    some model: WellFormedGNNModel |
        #model.aiModel.stateSpace.factors = 2 and
        #model.aiModel.observationSpace.modalities = 1 and
        some model.aiModel.actionSpace and
        #model.aiModel.actionSpace.controls = 1
}

// Commands for model finding
run someWellFormedModel for 5
run modelWithConnections for 5
run modelWithMultipleFactors for 5
run modelWithActions for 5
run exampleTwoStateModel for 5

// Verification commands
check VariableNameUniqueness for 5
check ConnectionValidity for 5
check AIModelCoherence for 5
check CategoryLaws for 3 