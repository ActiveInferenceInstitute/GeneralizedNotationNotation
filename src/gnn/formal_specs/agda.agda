{-
GNN (Generalized Notation Notation) Type Theory Specification in Agda

This file provides a type-theoretic specification of GNN models using
dependent types, enabling constructive proofs and formal verification
of Active Inference model properties.
-}

{-# OPTIONS --without-K --exact-split #-}

module GNN where

open import Data.Nat using (ℕ; zero; suc; _+_; _*_; _≤_; _<_)
open import Data.String using (String; _++_)
open import Data.Bool using (Bool; true; false)
open import Data.Product using (_×_; _,_; proj₁; proj₂; Σ; ∃)
open import Data.List using (List; []; _∷_)
open import Data.Vec using (Vec; []; _∷_; lookup)
open import Data.Fin using (Fin; zero; suc)
open import Relation.Binary.PropositionalEquality using (_≡_; refl; _≢_)
open import Function using (_∘_; id)

-- Basic types
Identifier : Set
Identifier = String

-- Version specification
record Version : Set where
  constructor mkVersion
  field
    major : ℕ
    minor : ℕ  
    patch : ℕ

-- Data types supported in GNN
data DataType : Set where
  categorical : DataType
  continuous  : DataType
  binary      : DataType
  integer     : DataType
  float       : DataType
  complex     : DataType

-- Variable types with indices
data VariableType : Set where
  hidden-state  : ℕ → VariableType
  observation   : ℕ → VariableType
  action        : ℕ → VariableType
  policy        : ℕ → VariableType

-- Processing flags
record ProcessingFlags : Set where
  constructor mkFlags
  field
    strict-validation     : Bool
    allow-experimental    : Bool
    enable-optimizations  : Bool
    debug-mode           : Bool
    verbose-logging      : Bool

-- Variable definition
record Variable : Set where
  constructor mkVariable
  field
    name        : Identifier
    var-type    : VariableType
    dimensions  : List ℕ
    data-type   : DataType
    description : String

-- Valid variable naming
data ValidVariableName : Identifier → VariableType → Set where
  valid-state  : (i : ℕ) → ValidVariableName ("s_f" ++ show i) (hidden-state i)
  valid-obs    : (i : ℕ) → ValidVariableName ("o_m" ++ show i) (observation i)
  valid-action : (i : ℕ) → ValidVariableName ("u_c" ++ show i) (action i)
  valid-policy : (i : ℕ) → ValidVariableName ("pi_c" ++ show i) (policy i)

WellFormedVariable : Set
WellFormedVariable = Σ Variable λ v → ValidVariableName (Variable.name v) (Variable.var-type v)

-- Connection types
data ConnectionType : Set where
  directed      : ConnectionType
  undirected    : ConnectionType
  conditional   : ConnectionType
  bidirectional : ConnectionType

-- Connection definition
record Connection : Set where
  constructor mkConnection
  field
    source-vars : List Variable
    target-vars : List Variable
    conn-type   : ConnectionType
    symbol      : String
    description : String

-- Mathematical constraints
data Constraint : Set where
  stochastic     : Constraint
  non-negative   : Constraint
  symmetric      : Constraint
  orthogonal     : Constraint

-- Parameter values
data ParameterValue : Set where
  scalar-value : ℕ → ParameterValue
  vector-value : List ℕ → ParameterValue
  matrix-value : List (List ℕ) → ParameterValue

-- Time configuration
record TimeConfiguration : Set where
  constructor mkTimeConfig
  field
    time-type       : Bool  -- true = dynamic, false = static
    discretization  : Bool  -- true = discrete, false = continuous
    horizon         : ℕ
    time-step       : ℕ

-- State space
record StateSpace : Set where
  constructor mkStateSpace
  field
    factors         : List Variable
    joint-dimension : ℕ
    description     : String

-- Active Inference model
record ActiveInferenceModel : Set where
  constructor mkAIModel
  field
    state-space          : StateSpace
    observation-space    : StateSpace  -- Reusing StateSpace structure
    time-horizon        : ℕ
    description         : String

-- Complete GNN model
record GNNModel : Set where
  constructor mkGNNModel
  field
    gnn-section       : String
    version           : Version
    processing-flags  : ProcessingFlags
    model-name        : String
    model-annotation  : String
    variables         : List WellFormedVariable
    connections       : List Connection
    ai-model          : ActiveInferenceModel
    time-config       : TimeConfiguration
    footer            : String

-- Well-formed GNN model predicate
data WellFormedGNN : GNNModel → Set where
  well-formed : (model : GNNModel) →
    GNNModel.gnn-section model ≡ "GNN" →
    GNNModel.model-name model ≢ "" →
    WellFormedGNN model

-- Example construction
example-model : GNNModel
example-model = mkGNNModel
  "GNN"
  (mkVersion 2 0 0)
  (mkFlags true false true false false)
  "Example Model"
  "Simple example"
  []
  []
  (mkAIModel 
    (mkStateSpace [] 1 "")
    (mkStateSpace [] 1 "")
    10
    "")
  (mkTimeConfig true true 10 1)
  ""
``` 