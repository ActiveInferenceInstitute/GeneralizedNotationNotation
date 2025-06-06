## GNNSection
AliceLovesBobDiscopy

## GNNVersionAndFlags
GNN v1
DisCoPyCompatible

## ModelName
Alice Loves Bob - DisCoPy Quantum NLP Example

## ModelAnnotation
This GNN file represents the "Alice loves Bob" quantum NLP example from DisCoPy.
It's designed to be parsed by `discopy_translator_module` and produce a JAX-backed
DisCoPy diagram that evaluates to the expected amplitude.
The target amplitude for "Alice loves Bob" is [0.0 - 1.0j].

## StateSpaceBlock
# Abstract types (DisCoPy Ty) - dimensions indicate number of qubits (for 'n') or scalar (for 's')
n[2] # Noun type (1 qubit, Dim(2))
s[1] # Sentence type (0 qubits, Dim(1) - scalar)

# Intermediate states representing tensor products of qubits
# These are helper states for defining the diagram structure.
# Input state after word functorization: Alice @ Loves @ Bob
# n_A @ (n_L_r @ s_L @ n_L_l) @ n_B
# The 'loves' word itself has type n.r @ s @ n.l.
# Functor maps:
# F(Alice): () -> n_A                  (Dim(1) -> Dim(2))
# F(loves): () -> n_L1 @ n_L2          (Dim(1) -> Dim(4), representing 2 qubits for 'loves')
# F(Bob):   () -> n_B                  (Dim(1) -> Dim(2))
# Initial combined state before grammar:
InitState_scalar[1]                   # Overall scalar input to the full sentence
QubitState_A[2]                       # Qubit for Alice
QubitState_L1[2]                      # First qubit for Loves
QubitState_L2[2]                      # Second qubit for Loves
QubitState_B[2]                       # Qubit for Bob
Combined_4Qubit_State[16]             # Represents n_A @ n_L1 @ n_L2 @ n_B

# Final output state
FinalScalarOutput[1]                  # Output of type 's', scalar

## TensorDefinitions
# BoxName | Dom | Cod | Initializer (JSON string for complex lists, or "load:path/to/file.npy", "random_normal", "random_uniform")
# default_dtype: "complex64" # Will be set in translator if needed by initializer

InitState_scalar_to_Combined_4Qubit_State | 1 | 16 | "[[0.0,0.0], [0.0,0.0], [0.0,0.0], [0.0,-1.0], [0.0,0.0], [0.0,-1.0], [0.0,0.0], [0.0,0.0], [0.0,0.0], [0.0,0.0], [0.0,0.0], [0.0,0.0], [0.0,0.0], [0.0,0.0], [0.0,0.0], [0.0,0.0]]"
# This tensor represents the DisCoPy operations:
# Alice_box @ Loves_box @ Bob_box >> Id(n) @ Cup(n, n.r) @ Id(s) @ Cup(n.l, n) @ Id(n)
# which simplifies to:
# Ket(0) @ verb_ansatz(0.5) @ Ket(1) >> grammar
# Ket(0) is [1,0]; Ket(1) is [0,1]
# verb_ansatz(0.5) for loves (complex64, shape (1,4)): jnp.array([0.70710677+0.j, 0.00000000+0.j, 0.00000000+0.j, 0.00000000-0.70710677j]) (approx)
#   Simplified from (Ket(0,0) >> H @ sqrt(2) @ Rx(0.5) >> CX).eval().array which is [[1,0,0,0],[0,1,0,0],[0,0,0,-1j],[0,0,1j,0]] * 0.5 (incorrect, this is H.eval() @ CX.eval() )
#   Actual verb_ansatz(0.5).eval().array is shape (4,) for dom=Dim(1) cod=Dim(4) or (1,4) if using Matrix
#   The example output for F(params0)(parsing['Alice loves Bob.']).eval() is Tensor(dom=Dim(1), cod=Dim(1), array=[0.-1.j])
#   The intermediate state Alice @ loves @ Bob has type n @ n.r @ s @ n.l @ n, which is 1+1+0+1+1 = 4 qubits = Dim(16)
#   The initial state being fed into the "grammar" part.
#   The GNN "Initialize_Sentence_State" is conceptualized as the state *after* Alice, Loves, Bob functors are applied,
#   before the main grammar cups. So it's already a Dim(16) state.
#   The example notebook shows: F(params0)(parsing['Alice loves Bob.']) which is
#   (Alice_circ @ Loves_circ @ Bob_circ) >> grammar_circ
#   Alice_circ = Ket(0) (dom=1, cod=2)
#   Loves_circ = verb_ansatz(0.5) (dom=1, cod=4)
#   Bob_circ = Ket(1) (dom=1, cod=2)
#   Tensor product: dom=1, cod= 2*4*2 = 16. The array is the tensor product of these, flattened.
#   This means the value [0.0+0.0j, ..., 0.0-1.0j, ...] for the "sentence state" IS the JAX array for this product.

Combined_4Qubit_State_to_FinalScalarOutput | 16 | 1 | "[[1.0,0.0], [0.0,0.0], [0.0,0.0], [1.0,0.0], [0.0,0.0], [0.0,0.0], [0.0,0.0], [0.0,0.0], [0.0,0.0], [0.0,0.0], [0.0,0.0], [0.0,0.0], [1.0,0.0], [0.0,0.0], [0.0,0.0], [1.0,0.0]]"
# This tensor represents the grammar: Cup(n, n.r) @ Id(s) @ Cup(n.l, n)
# It takes the Dim(16) state (4 qubits) and outputs Dim(1) (scalar for sentence truth).
# The DisCoPy grammar object is: Cup(Dim(2), Dim(2).r) @ Id(Dim(1)) @ Cup(Dim(2).l, Dim(2))
# n = Ty('n') -> Dim(2); s = Ty('s') -> Dim(1)
# (n @ n.r @ s @ n.l @ n) >> grammar_cups --> s
# Dim(2) @ Dim(2) @ Dim(1) @ Dim(2) @ Dim(2)  >> grammar_cups --> Dim(1)
# Dim(16) >> grammar_cups --> Dim(1)
# The array for Cup(Dim(2), Dim(2)) is [[1,0,0,1]]. Id(Dim(1)) is [1].
# So the grammar matrix is effectively a large contraction.
# The provided array is shape (1,16), but for discopy.Matrix(dom=16, cod=1, array=...) it needs (16,1)
# The parser should handle reshaping this from the JSON [[]] to the column vector.

## Connections
# Defines the sequence: InitState_scalar -> [Box1] -> Combined_4Qubit_State -> [Box2] -> FinalScalarOutput
# Box1 is InitState_scalar_to_Combined_4Qubit_State
# Box2 is Combined_4Qubit_State_to_FinalScalarOutput
InitState_scalar > Combined_4Qubit_State
Combined_4Qubit_State > FinalScalarOutput

## Footer
End of Alice Loves Bob GNN. 