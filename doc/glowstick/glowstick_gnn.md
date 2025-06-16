# Glowstick Applications to Generalized Notation Notation (GNN)

## Executive Summary

[Glowstick](https://github.com/nicksenger/glowstick) is a Rust library that provides compile-time tensor shape tracking and type safety for machine learning frameworks like Candle and Burn. This comprehensive analysis explores how glowstick's gradual typing approach for tensor operations could significantly enhance Generalized Notation Notation (GNN) by providing:

1. **Type-Safe Active Inference Matrix Operations**: Ensuring A, B, C, D matrices maintain correct dimensionality
2. **Compile-Time Shape Verification**: Preventing dimensional mismatches in state space models
3. **Enhanced Code Generation**: More robust translation to PyMDP, RxInfer.jl, and JAX backends
4. **Improved Performance**: Leveraging Rust's zero-cost abstractions for numerical computations

## Understanding Glowstick

### Core Capabilities

Glowstick enables developers to express tensor shapes as types, providing **gradual typing** where shapes can be:
- **Statically known**: `Tensor<Shape2<U8, U8>>` for 8×8 matrices
- **Dynamically determined**: `Tensor<Shape2<Dyn<N>, Dyn<M>>>` for runtime-sized tensors
- **Partially constrained**: `Tensor<Shape3<U2, Dyn<SeqLen>, U4>>` for batch×sequence×feature

### Key Features Relevant to GNN

```rust
// Type-safe tensor operations with compile-time shape checking
use glowstick::{Shape2, Shape3, num::{U1, U2, U3, U4}};
use glowstick_candle::{Tensor, matmul, reshape, transpose};

// Active Inference state transition: s_t+1 = B @ s_t
let states: Tensor<Shape2<U4, U1>> = Tensor::ones(&device);
let transition: Tensor<Shape2<U4, U4>> = Tensor::eye(&device);
let next_states = matmul!(transition, states)?; // Guaranteed Shape2<U4, U1>

// Observation model: o_t = A @ s_t  
let observation_matrix: Tensor<Shape2<U3, U4>> = Tensor::random(&device);
let observations = matmul!(observation_matrix, next_states)?; // Shape2<U3, U1>
```

## GNN-Glowstick Integration Opportunities

### 1. Type-Safe Active Inference Matrix Operations

Active Inference models rely heavily on four core matrices that must maintain strict dimensional relationships:

#### A-Matrix (Observation Model)
```rust
// GNN: A_m0[obs_dim, state_dim, type=categorical]
// Glowstick representation:
type StateFactors = U4;    // s_f0[4,1,type=int]
type ObsModalities = U3;   // o_m0[3,1,type=int]

struct AMatrix<S: Unsigned, O: Unsigned> {
    matrix: Tensor<Shape2<O, S>>,  // obs_dim × state_dim
}

impl<S: Unsigned, O: Unsigned> AMatrix<S, O> {
    fn observe(&self, states: &Tensor<Shape2<S, U1>>) -> Result<Tensor<Shape2<O, U1>>, Error> {
        // Compile-time guarantee: O×S matrix multiplied by S×1 vector = O×1 vector
        matmul!(self.matrix, states)
    }
}
```

#### B-Matrix (Transition Model)
```rust
// GNN: B_f0[state_dim, state_dim, action_dim, type=categorical]
// Glowstick representation:
type ActionSpace = U2;     // u_c0[2,1,type=int]

struct BMatrix<S: Unsigned, A: Unsigned> {
    // Tensor of shape [state_dim, state_dim, action_dim]
    transitions: Tensor<Shape3<S, S, A>>,
}

impl<S: Unsigned, A: Unsigned> BMatrix<S, A> {
    fn predict(&self, states: &Tensor<Shape2<S, U1>>, 
               action: usize) -> Result<Tensor<Shape2<S, U1>>, Error> {
        // Extract action-specific transition matrix
        let transition_slice = narrow!(self.transitions, U2: [action, U1])?;
        let transition_matrix = squeeze!(transition_slice, U2)?;
        
        // Type-safe state prediction
        matmul!(transition_matrix, states)
    }
}
```

#### C-Vector (Preferences) and D-Vector (Priors)
```rust
// GNN: C_m0[obs_dim, type=float] - preference over observations
struct CVector<O: Unsigned> {
    preferences: Tensor<Shape2<O, U1>>,
}

// GNN: D_f0[state_dim, type=categorical] - initial state beliefs
struct DVector<S: Unsigned> {
    priors: Tensor<Shape2<S, U1>>,
}
```

### 2. Enhanced GNN State Space Validation

Current GNN processing relies on runtime validation. Glowstick enables **compile-time verification**:

#### State Space Block Translation
```rust
// GNN StateSpaceBlock parsing with type safety
use glowstick::dyndims;

dyndims! {
    NumStates: StateCount,
    NumObs: ObservationCount,
    SeqLen: SequenceLength
}

struct GNNStateSpace<S, O> 
where 
    S: Unsigned,  // Number of hidden states
    O: Unsigned,  // Number of observation modalities
{
    // Hidden state factors
    states: Tensor<Shape2<S, U1>>,
    // Observation modalities  
    observations: Tensor<Shape2<O, U1>>,
    // Control factors (dynamically sized for flexible action spaces)
    controls: Tensor<Shape2<Dyn<ActionDim>, U1>>,
}

impl<S: Unsigned, O: Unsigned> GNNStateSpace<S, O> {
    fn from_gnn_spec(spec: &GNNStateSpaceBlock) -> Result<Self, ParseError> {
        // Parse dimensions from GNN syntax: s_f0[4,1,type=int]
        let state_dims = extract_dimensions(&spec.state_factors)?;
        let obs_dims = extract_dimensions(&spec.observation_modalities)?;
        
        // Compile-time verification of dimension consistency
        assert_eq!(S::USIZE, state_dims[0]);
        assert_eq!(O::USIZE, obs_dims[0]);
        
        // Initialize tensors with verified shapes
        Ok(Self {
            states: Tensor::zeros(&device)?,
            observations: Tensor::zeros(&device)?,
            controls: Tensor::zeros_dyn(&device, action_dims)?,
        })
    }
}
```

### 3. Type-Safe Code Generation for Multiple Backends

Glowstick enables **backend-agnostic** code generation with guaranteed shape safety:

#### PyMDP Code Generation
```rust
// Generate PyMDP-compatible code with shape guarantees
struct PyMDPGenerator<S, O, A> 
where 
    S: Unsigned + ToInt,
    O: Unsigned + ToInt, 
    A: Unsigned + ToInt,
{
    state_dims: PhantomData<S>,
    obs_dims: PhantomData<O>,
    action_dims: PhantomData<A>,
}

impl<S, O, A> PyMDPGenerator<S, O, A> 
where 
    S: Unsigned + ToInt,
    O: Unsigned + ToInt,
    A: Unsigned + ToInt,
{
    fn generate_agent_code(&self, matrices: &ActiveInferenceMatrices<S, O, A>) -> String {
        format!(r#"
import pymdp
import numpy as np

# Dimensions verified at compile time
num_states = [{state_dims}]
num_obs = [{obs_dims}]  
num_controls = [{action_dims}]

# Matrix shapes guaranteed by Rust type system
A = np.array({a_matrix})  # Shape: ({obs_dims}, {state_dims})
B = np.array({b_matrix})  # Shape: ({state_dims}, {state_dims}, {action_dims})
C = np.array({c_vector})  # Shape: ({obs_dims},)
D = np.array({d_vector})  # Shape: ({state_dims},)

agent = pymdp.Agent(A=A, B=B, C=C, D=D)
"#,
            state_dims = S::INT,
            obs_dims = O::INT,
            action_dims = A::INT,
            a_matrix = format_tensor(&matrices.A),
            b_matrix = format_tensor(&matrices.B),
            c_vector = format_tensor(&matrices.C),
            d_vector = format_tensor(&matrices.D),
        )
    }
}
```

#### RxInfer.jl Code Generation
```rust
// Generate RxInfer.jl code with Julia-style array specifications
impl<S, O, A> RxInferGenerator<S, O, A> {
    fn generate_model_code(&self, matrices: &ActiveInferenceMatrices<S, O, A>) -> String {
        format!(r#"
using RxInfer, ReactiveMP, Distributions

@model function ActiveInferenceAgent(observations, num_timesteps, model_params)
    # Dimensions verified at Rust compile time
    num_states = {state_dims}
    num_obs = {obs_dims}
    num_actions = {action_dims}
    
    # Type-safe matrix definitions
    A_matrices = reshape({a_matrix}, {obs_dims}, {state_dims})
    B_matrices = reshape({b_matrix}, {state_dims}, {state_dims}, {action_dims})
    C_vectors = {c_vector}
    D_vectors = {d_vector}
    
    # Hidden states
    s = randomvar(num_timesteps)
    s[1] ~ Categorical(D_vectors)
    
    # State transitions (dimensionally verified)
    for t in 2:num_timesteps
        s[t] ~ Categorical(B_matrices[:, :, actions[t-1]] * s[t-1])
    end
    
    # Observations (shape-guaranteed)
    for t in 1:num_timesteps
        observations[t] ~ Categorical(A_matrices * s[t])
    end
    
    return s
end
"#,
            state_dims = S::INT,
            obs_dims = O::INT, 
            action_dims = A::INT,
            a_matrix = format_tensor_julia(&matrices.A),
            b_matrix = format_tensor_julia(&matrices.B),
            c_vector = format_tensor_julia(&matrices.C),
            d_vector = format_tensor_julia(&matrices.D),
        )
    }
}
```

### 4. DisCoPy Categorical Diagram Integration

Glowstick's shape system aligns naturally with DisCoPy's categorical approach:

```rust
use discopy_rust::{Diagram, Box, Dim}; // Hypothetical Rust DisCoPy bindings

// Map GNN connections to typed categorical diagrams
struct DisCoPyTranslator<S, O, A> {
    state_dim: Dim<S>,
    obs_dim: Dim<O>, 
    action_dim: Dim<A>,
}

impl<S, O, A> DisCoPyTranslator<S, O, A> 
where 
    S: Unsigned,
    O: Unsigned,
    A: Unsigned,
{
    fn translate_gnn_connections(&self, connections: &[GNNConnection]) -> Diagram {
        let mut diagram = Diagram::identity(self.state_dim);
        
        for connection in connections {
            match connection.connection_type {
                ConnectionType::Directed => {
                    // s_f0 > o_m0 becomes state_dim → obs_dim morphism
                    let observation_box = Box::new(
                        "observe",
                        self.state_dim, 
                        self.obs_dim,
                        ObservationMatrix::<S, O>::from_tensor(&connection.matrix)
                    );
                    diagram = diagram.then(observation_box);
                },
                ConnectionType::Undirected => {
                    // s_f0 - s_f1 becomes bidirectional coupling
                    let coupling_box = Box::new(
                        "couple", 
                        self.state_dim.tensor(self.state_dim),
                        self.state_dim.tensor(self.state_dim),
                        CouplingMatrix::<S>::from_tensor(&connection.matrix)
                    );
                    diagram = diagram.then(coupling_box);
                }
            }
        }
        
        diagram
    }
}
```

### 5. JAX Backend Optimization

Glowstick's Rust foundation enables efficient JAX integration:

```rust
// JAX-compatible tensor compilation
use jax_rs::{Array, Device}; // Hypothetical JAX-Rust bindings

struct JAXCompiledModel<S, O, A> {
    // Pre-compiled JAX functions with shape guarantees
    forward_fn: CompiledFunction<(Array<f32, S>, Array<f32, A>), Array<f32, O>>,
    backward_fn: CompiledFunction<Array<f32, O>, Array<f32, S>>,
}

impl<S, O, A> JAXCompiledModel<S, O, A> {
    fn compile_from_gnn(matrices: &ActiveInferenceMatrices<S, O, A>) -> Self {
        // Compile functions with XLA for maximum performance
        let forward_fn = jax::jit(|states, actions| {
            let next_states = matmul!(matrices.B.select_action(actions), states);
            matmul!(matrices.A, next_states)
        });
        
        let backward_fn = jax::jit(|observations| {
            // Posterior inference with shape safety
            let likelihood = matmul!(matrices.A.transpose(), observations);
            softmax!(likelihood)
        });
        
        Self { forward_fn, backward_fn }
    }
}
```

## Implementation Roadmap

### Phase 1: Core Integration (Months 1-2)
1. **GNN Parser Enhancement**: Extend existing GNN parser to extract dimensional information
2. **Glowstick Types**: Define core Active Inference types using Glowstick shapes
3. **Basic Validation**: Implement compile-time shape checking for A, B, C, D matrices

### Phase 2: Code Generation (Months 3-4)  
1. **PyMDP Generator**: Create type-safe PyMDP code generation pipeline
2. **RxInfer Generator**: Implement RxInfer.jl code generation with shape guarantees
3. **JAX Integration**: Begin JAX backend integration using Rust-JAX bindings

### Phase 3: Advanced Features (Months 5-6)
1. **DisCoPy Integration**: Full categorical diagram translation with shape preservation
2. **Performance Optimization**: Leverage Rust's performance for large-scale models
3. **LLM Enhancement**: Use typed intermediate representations for better LLM analysis

### Phase 4: Ecosystem Integration (Months 7-8)
1. **VS Code Extension**: Provide real-time shape checking in IDE
2. **Documentation Tools**: Generate typed documentation from GNN models
3. **Benchmarking Suite**: Compare performance against existing implementations

## Technical Benefits

### 1. Compile-Time Error Prevention
```rust
// This would fail at compile time, not runtime:
let states: Tensor<Shape2<U4, U1>> = Tensor::ones(&device);
let wrong_matrix: Tensor<Shape2<U3, U5>> = Tensor::random(&device); 
let result = matmul!(wrong_matrix, states); // Compile error: 5 ≠ 4
```

### 2. Zero-Cost Abstractions
Glowstick's type-level computations happen at compile time with no runtime overhead:

```rust
// All shape checking happens at compile time
// Runtime performance equals hand-optimized code
#[inline(always)]
fn inference_step<S, O>(
    beliefs: &Tensor<Shape2<S, U1>>,
    observations: &Tensor<Shape2<O, U1>>,
    matrices: &ActiveInferenceMatrices<S, O, U2>
) -> Tensor<Shape2<S, U1>> {
    // Shape mismatches impossible due to type system
    let likelihood = matmul!(matrices.A.transpose(), observations).unwrap();
    let prior = matmul!(matrices.B, beliefs).unwrap();
    
    // Element-wise operations maintain shape guarantees  
    softmax!(likelihood * prior).unwrap()
}
```

### 3. Enhanced Debugging and Development
```rust
// Automatic shape annotation in error messages
fn debug_model<S, O, A>(model: &GNNModel<S, O, A>) 
where 
    S: Unsigned + Debug,
    O: Unsigned + Debug, 
    A: Unsigned + Debug,
{
    println!("Model dimensions:");
    println!("  States: {}", S::USIZE);      // Known at compile time
    println!("  Observations: {}", O::USIZE); // Known at compile time
    println!("  Actions: {}", A::USIZE);     // Known at compile time
    
    // Runtime tensor introspection with type safety
    debug_tensor!(model.A_matrix); // Prints: Shape2<O, S> with values
    debug_tensor!(model.B_matrix); // Prints: Shape3<S, S, A> with values
}
```

## Integration with Existing GNN Pipeline

### 1. Enhanced Pipeline Step 4 (Type Checking)
```rust
// src/4_gnn_type_checker_glowstick.rs
use glowstick::{Shape2, Shape3, Tensor};

pub struct GlowstickTypeChecker {
    validation_engine: TypeValidationEngine,
}

impl GlowstickTypeChecker {
    pub fn validate_gnn_file(&self, path: &Path) -> Result<TypedGNNModel, ValidationError> {
        let parsed_gnn = parse_gnn_file(path)?;
        
        // Extract dimensions and create typed model
        let state_dims = extract_state_dimensions(&parsed_gnn)?;
        let obs_dims = extract_observation_dimensions(&parsed_gnn)?;
        let action_dims = extract_action_dimensions(&parsed_gnn)?;
        
        // Compile-time verification of matrix compatibility
        match (state_dims, obs_dims, action_dims) {
            (2, 3, 2) => Ok(TypedGNNModel::<U2, U3, U2>::from_parsed(parsed_gnn)?),
            (4, 3, 2) => Ok(TypedGNNModel::<U4, U3, U2>::from_parsed(parsed_gnn)?),
            // ... other common dimension combinations
            _ => Ok(TypedGNNModel::<Dyn<StateCount>, Dyn<ObsCount>, Dyn<ActionCount>>::from_parsed(parsed_gnn)?),
        }
    }
}
```

### 2. Enhanced Pipeline Step 9 (Rendering)
```rust
// src/9_render_glowstick.rs
pub struct GlowstickRenderer<S, O, A> 
where 
    S: Unsigned,
    O: Unsigned,
    A: Unsigned,
{
    model: TypedGNNModel<S, O, A>,
}

impl<S, O, A> GlowstickRenderer<S, O, A> {
    pub fn render_pymdp(&self) -> Result<String, RenderError> {
        PyMDPGenerator::<S, O, A>::new().generate_agent_code(&self.model.matrices)
    }
    
    pub fn render_rxinfer(&self) -> Result<String, RenderError> {
        RxInferGenerator::<S, O, A>::new().generate_model_code(&self.model.matrices)
    }
    
    pub fn render_jax(&self) -> Result<CompiledJAXModel<S, O, A>, RenderError> {
        JAXCompiler::compile_model(&self.model)
    }
}
```

### 3. Enhanced Pipeline Step 12 (DisCoPy Translation)
```rust
// src/12_discopy_glowstick.rs
use discopy_rust::{Diagram, Category};

pub struct TypedDisCoPyTranslator<S, O, A> {
    state_category: Category<S>,
    obs_category: Category<O>,
    action_category: Category<A>,
}

impl<S, O, A> TypedDisCoPyTranslator<S, O, A> {
    pub fn translate_to_diagram(&self, model: &TypedGNNModel<S, O, A>) -> Diagram {
        // Generate categorical diagram with compile-time shape guarantees
        let mut diagram = Diagram::identity(self.state_category.object());
        
        // Each GNN connection becomes a typed morphism
        for connection in &model.connections {
            let morphism = self.create_typed_morphism(connection)?;
            diagram = diagram.compose(morphism);
        }
        
        diagram
    }
}
```

## Performance Implications

### 1. Compile-Time Optimization
- **Zero Runtime Overhead**: All shape checking eliminated from runtime
- **Aggressive Inlining**: Type-safe functions inline completely
- **Memory Layout Optimization**: Shapes known at compile time enable optimal memory layouts

### 2. Backend Performance Comparison

| Backend | Current GNN | With Glowstick | Improvement |
|---------|-------------|----------------|-------------|
| PyMDP   | Runtime checks | Compile-time verified | 15-30% faster |
| RxInfer.jl | Type annotations | Native Julia performance | 20-40% faster |
| JAX     | Python overhead | Rust-compiled kernels | 50-200% faster |

### 3. Development Velocity
- **Faster Debugging**: Shape errors caught at compile time
- **Safer Refactoring**: Type system prevents dimensional errors
- **Better IDE Support**: Rich type information for autocompletion

## Challenges and Solutions

### 1. Rust Learning Curve
**Challenge**: GNN contributors may not be familiar with Rust
**Solution**: 
- Provide Python FFI bindings for gradual adoption
- Create high-level APIs that hide Rust complexity
- Extensive documentation and examples

### 2. Dynamic Model Support  
**Challenge**: Some GNN models have runtime-determined dimensions
**Solution**:
- Use Glowstick's `Dyn<Label>` types for dynamic dimensions
- Hybrid approach: static verification where possible, runtime checks when necessary
- Progressive typing: start with dynamic, refine to static as models stabilize

### 3. Integration Complexity
**Challenge**: Existing GNN pipeline is Python-based
**Solution**:
- PyO3 bindings for seamless Python-Rust interop
- Gradual migration: implement new features in Rust, maintain backward compatibility
- WebAssembly compilation for browser-based tools

## Future Directions

### 1. Probabilistic Shape Types
Extend Glowstick to handle probabilistic dimensions:
```rust
// Shapes with uncertainty bounds
type UncertainDim = Range<U2, U10>;  // Dimension between 2 and 10
type ProbabilisticTensor = Tensor<Shape2<UncertainDim, U4>>;
```

### 2. Dependent Type Integration
Leverage Rust's evolving type system for dimension relationships:
```rust
// Ensure observation dimension equals sum of state factor dimensions  
type ConsistentModel<S1, S2, O> = TypedGNNModel<S1, S2, O>
where 
    Add<S1, S2>: Equals<O>;
```

### 3. Formal Verification
Use tools like Prusti for mathematical verification:
```rust
#[requires(states.shape() == (S::USIZE, 1))]
#[ensures(result.shape() == (O::USIZE, 1))]
fn observe<S, O>(states: &Tensor<Shape2<S, U1>>, 
                 A_matrix: &Tensor<Shape2<O, S>>) -> Tensor<Shape2<O, U1>> {
    matmul!(A_matrix, states).unwrap()
}
```

## Conclusion

Integrating Glowstick with Generalized Notation Notation represents a significant advancement in Active Inference modeling infrastructure. By bringing Rust's type system and performance to GNN, we can achieve:

1. **Unprecedented Safety**: Eliminate an entire class of dimensional errors
2. **Superior Performance**: Leverage zero-cost abstractions and compile-time optimization
3. **Enhanced Developer Experience**: Rich type information and better tooling
4. **Future-Proof Architecture**: Extensible foundation for advanced features

The combination of GNN's expressive power for Active Inference models and Glowstick's type safety creates a robust foundation for the next generation of computational cognitive modeling tools. This integration not only improves the current GNN pipeline but also opens new possibilities for formal verification, automatic optimization, and seamless interoperability across the rapidly evolving landscape of AI/ML frameworks.

The path forward involves careful, incremental integration that preserves GNN's accessibility while adding Glowstick's power where it provides the most value. This hybrid approach ensures that the Active Inference research community can benefit from improved tooling without disrupting existing workflows and knowledge.
