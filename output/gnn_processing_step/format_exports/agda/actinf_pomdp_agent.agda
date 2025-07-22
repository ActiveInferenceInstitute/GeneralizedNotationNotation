-- GNN Model: Classic Active Inference POMDP Agent v1
-- Generated from GNN markdown

module GNNModel where

open import Data.String
open import Data.List
open import Data.Nat
open import Data.Product

record Variable : Set where
  constructor variable
  field
    name : String
    type : String
    dimensions : List ℕ
    description : String

record Connection : Set where
  constructor connection
  field
    source : String
    target : String
    type : String
    description : String

record Parameter : Set where
  constructor parameter
  field
    name : String
    value : String
    description : String

record GNNModel : Set where
  constructor gnnModel
  field
    modelName : String
    version : String
    annotation : String
    variables : List Variable
    connections : List Connection
    parameters : List Parameter

model : GNNModel
model = gnnModel
  "Classic Active Inference POMDP Agent v1"
  "1.0"
  "This model describes a classic Active Inference agent for a discrete POMDP:
- One observation modality ("state_observation") with 3 possible outcomes.
- One hidden state factor ("location") with 3 possible states.
- The hidden state is fully controllable via 3 discrete actions.
- The agent's preferences are encoded as log-probabilities over observations.
- The agent has an initial policy prior (habit) encoded as log-probabilities over actions."
  [
    variable "A" "parameter_matrix" [3, 3] "Likelihood mapping hidden states to observations"
    variable "B" "parameter_matrix" [3, 3, 3] "State transitions given previous state and action"
    variable "C" "parameter_matrix" [3] "Log-preferences over observations"
    variable "D" "parameter_matrix" [3] "Prior over initial hidden states"
    variable "E" "parameter_matrix" [3] "Initial policy prior (habit) over actions"
    variable "s" "hidden_state" [3, 1] "Current hidden state distribution"
    variable "s_prime" "hidden_state" [3, 1] "Next hidden state distribution"
    variable "o" "observation" [3, 1] "Current observation (integer index)"
    variable "π" "hidden_state" [3] "Policy (distribution over actions), no planning"
    variable "u" "action" [1] "Action taken"
    variable "G" "hidden_state" [] "Expected Free Energy (per policy)"
    variable "t" "hidden_state" [1] "Discrete time step"
  ]
  [
    connection "D" "s" "directed" ""
    connection "s" "A" "undirected" ""
    connection "s" "s_prime" "directed" ""
    connection "A" "o" "undirected" ""
    connection "s" "B" "undirected" ""
    connection "C" "G" "directed" ""
    connection "E" "π" "directed" ""
    connection "G" "π" "directed" ""
    connection "π" "u" "directed" ""
    connection "B" "u" "directed" ""
    connection "u" "s_prime" "directed" ""
  ]
  [
    parameter "A" "{" ""
    parameter "B" "{" ""
    parameter "C" "(0.1, 0.1, 1.0)" ""
    parameter "D" "(0.33333, 0.33333, 0.33333)" ""
    parameter "E" "(0.33333, 0.33333, 0.33333)" ""
  ]
