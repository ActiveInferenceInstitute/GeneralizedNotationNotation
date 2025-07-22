-- GNN Model: Classic Active Inference POMDP Agent v1
-- Generated from GNN markdown

structure Variable :=
  (name : string)
  (type : string)
  (dimensions : list ℕ)
  (description : string)

structure Connection :=
  (source : string)
  (target : string)
  (type : string)
  (description : string)

structure Parameter :=
  (name : string)
  (value : string)
  (description : string)

structure GNNModel :=
  (model_name : string)
  (version : string)
  (annotation : string)
  (variables : list Variable)
  (connections : list Connection)
  (parameters : list Parameter)

def model : GNNModel := {
  model_name := "Classic Active Inference POMDP Agent v1",
  version := "1.0",
  annotation := "This model describes a classic Active Inference agent for a discrete POMDP:
- One observation modality ("state_observation") with 3 possible outcomes.
- One hidden state factor ("location") with 3 possible states.
- The hidden state is fully controllable via 3 discrete actions.
- The agent's preferences are encoded as log-probabilities over observations.
- The agent has an initial policy prior (habit) encoded as log-probabilities over actions.",
  variables := [
    { name := "A", type := "parameter_matrix", dimensions := [3, 3], description := "Likelihood mapping hidden states to observations" }
    { name := "B", type := "parameter_matrix", dimensions := [3, 3, 3], description := "State transitions given previous state and action" }
    { name := "C", type := "parameter_matrix", dimensions := [3], description := "Log-preferences over observations" }
    { name := "D", type := "parameter_matrix", dimensions := [3], description := "Prior over initial hidden states" }
    { name := "E", type := "parameter_matrix", dimensions := [3], description := "Initial policy prior (habit) over actions" }
    { name := "s", type := "hidden_state", dimensions := [3, 1], description := "Current hidden state distribution" }
    { name := "s_prime", type := "hidden_state", dimensions := [3, 1], description := "Next hidden state distribution" }
    { name := "o", type := "observation", dimensions := [3, 1], description := "Current observation (integer index)" }
    { name := "π", type := "hidden_state", dimensions := [3], description := "Policy (distribution over actions), no planning" }
    { name := "u", type := "action", dimensions := [1], description := "Action taken" }
    { name := "G", type := "hidden_state", dimensions := [], description := "Expected Free Energy (per policy)" }
    { name := "t", type := "hidden_state", dimensions := [1], description := "Discrete time step" }
  ],
  connections := [
    { source := "D", target := "s", type := "directed", description := "" }
    { source := "s", target := "A", type := "undirected", description := "" }
    { source := "s", target := "s_prime", type := "directed", description := "" }
    { source := "A", target := "o", type := "undirected", description := "" }
    { source := "s", target := "B", type := "undirected", description := "" }
    { source := "C", target := "G", type := "directed", description := "" }
    { source := "E", target := "π", type := "directed", description := "" }
    { source := "G", target := "π", type := "directed", description := "" }
    { source := "π", target := "u", type := "directed", description := "" }
    { source := "B", target := "u", type := "directed", description := "" }
    { source := "u", target := "s_prime", type := "directed", description := "" }
  ],
  parameters := [
    { name := "A", value := "{", description := "" }
    { name := "B", value := "{", description := "" }
    { name := "C", value := "(0.1, 0.1, 1.0)", description := "" }
    { name := "D", value := "(0.33333, 0.33333, 0.33333)", description := "" }
    { name := "E", value := "(0.33333, 0.33333, 0.33333)", description := "" }
  ]
}
