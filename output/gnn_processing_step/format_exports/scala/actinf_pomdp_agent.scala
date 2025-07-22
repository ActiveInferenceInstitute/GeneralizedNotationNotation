// GNN Model: Classic Active Inference POMDP Agent v1
// Generated from GNN markdown

case class Variable(name: String, `type`: String, dimensions: List[Int], description: String)
case class Connection(source: String, target: String, `type`: String, description: String)
case class Parameter(name: String, value: String, description: String)

case class GNNModel(
  modelName: String,
  version: String,
  annotation: String,
  variables: List[Variable],
  connections: List[Connection],
  parameters: Map[String, Parameter]
)

object GNNModel {
  val model = GNNModel(
    modelName = "Classic Active Inference POMDP Agent v1",
    version = "1.0",
    annotation = """This model describes a classic Active Inference agent for a discrete POMDP:
- One observation modality ("state_observation") with 3 possible outcomes.
- One hidden state factor ("location") with 3 possible states.
- The hidden state is fully controllable via 3 discrete actions.
- The agent's preferences are encoded as log-probabilities over observations.
- The agent has an initial policy prior (habit) encoded as log-probabilities over actions.""",
    variables = List(
      Variable("A", "parameter_matrix", List(3, 3), "Likelihood mapping hidden states to observations")
      Variable("B", "parameter_matrix", List(3, 3, 3), "State transitions given previous state and action")
      Variable("C", "parameter_matrix", List(3), "Log-preferences over observations")
      Variable("D", "parameter_matrix", List(3), "Prior over initial hidden states")
      Variable("E", "parameter_matrix", List(3), "Initial policy prior (habit) over actions")
      Variable("s", "hidden_state", List(3, 1), "Current hidden state distribution")
      Variable("s_prime", "hidden_state", List(3, 1), "Next hidden state distribution")
      Variable("o", "observation", List(3, 1), "Current observation (integer index)")
      Variable("π", "hidden_state", List(3), "Policy (distribution over actions), no planning")
      Variable("u", "action", List(1), "Action taken")
      Variable("G", "hidden_state", List(), "Expected Free Energy (per policy)")
      Variable("t", "hidden_state", List(1), "Discrete time step")
    ),
    connections = List(
      Connection("D", "s", "directed", "")
      Connection("s", "A", "undirected", "")
      Connection("s", "s_prime", "directed", "")
      Connection("A", "o", "undirected", "")
      Connection("s", "B", "undirected", "")
      Connection("C", "G", "directed", "")
      Connection("E", "π", "directed", "")
      Connection("G", "π", "directed", "")
      Connection("π", "u", "directed", "")
      Connection("B", "u", "directed", "")
      Connection("u", "s_prime", "directed", "")
    ),
    parameters = Map(
      "A" -> Parameter("A", "{", "")
      "B" -> Parameter("B", "{", "")
      "C" -> Parameter("C", "(0.1, 0.1, 1.0)", "")
      "D" -> Parameter("D", "(0.33333, 0.33333, 0.33333)", "")
      "E" -> Parameter("E", "(0.33333, 0.33333, 0.33333)", "")
    )
  )
}
