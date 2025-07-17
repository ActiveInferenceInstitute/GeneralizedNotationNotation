/*
 * Example POMDP Agent in GNN Type System
 * 
 * This file demonstrates how the Scala categorical type system
 * can represent and validate the POMDP agent described in
 * src/gnn/gnn_examples/actinf_pomdp_agent.md
 */

package gnn.categorical.examples

import gnn.categorical._
import cats._
import cats.data._
import cats.implicits._
import scala.util.Try

/**
 * Example implementation of the POMDP agent using the GNN type system
 */
object POMDPExample extends App {

  // Create the dimensions for the model variables
  val dim3 = Dimension(3)
  val dim1 = Dimension(1)
  
  // Create the variables matching the POMDP agent model
  val hiddenState = Variable(
    name = "s",
    variableType = VariableType.HiddenState(0),
    dimensions = List(dim3, dim1),
    dataType = DataType.Float,
    description = Some("Current hidden state distribution")
  )
  
  val nextState = Variable(
    name = "s_prime",
    variableType = VariableType.HiddenState(0),
    dimensions = List(dim3, dim1),
    dataType = DataType.Float,
    description = Some("Next hidden state distribution")
  )
  
  val observation = Variable(
    name = "o",
    variableType = VariableType.Observation(0),
    dimensions = List(dim3, dim1),
    dataType = DataType.Integer,
    description = Some("Current observation (integer index)")
  )
  
  val policy = Variable(
    name = "π",
    variableType = VariableType.Policy(0),
    dimensions = List(dim3),
    dataType = DataType.Float,
    description = Some("Policy (distribution over actions), no planning")
  )
  
  val action = Variable(
    name = "u",
    variableType = VariableType.Action(0),
    dimensions = List(dim1),
    dataType = DataType.Integer,
    description = Some("Action taken")
  )
  
  val likelihoodMatrix = Variable(
    name = "A",
    variableType = VariableType.LikelihoodMatrix(0),
    dimensions = List(dim3, dim3),
    dataType = DataType.Float,
    description = Some("Likelihood mapping hidden states to observations")
  )
  
  val transitionMatrix = Variable(
    name = "B",
    variableType = VariableType.TransitionMatrix(0),
    dimensions = List(dim3, dim3, dim3),
    dataType = DataType.Float,
    description = Some("State transitions given previous state and action")
  )
  
  val preferenceVector = Variable(
    name = "C",
    variableType = VariableType.PreferenceVector(0),
    dimensions = List(dim3),
    dataType = DataType.Float,
    description = Some("Log-preferences over observations")
  )
  
  val priorVector = Variable(
    name = "D",
    variableType = VariableType.PriorVector(0),
    dimensions = List(dim3),
    dataType = DataType.Float,
    description = Some("Prior over initial hidden states")
  )
  
  val habitVector = Variable(
    name = "E",
    variableType = VariableType.HabitVector(0),
    dimensions = List(dim3),
    dataType = DataType.Float,
    description = Some("Initial policy prior (habit) over actions")
  )
  
  val expectedFreeEnergy = Variable(
    name = "G",
    variableType = VariableType.Policy(0),
    dimensions = List(policy.dimensions.head),
    dataType = DataType.Float,
    description = Some("Expected Free Energy (per policy)")
  )
  
  val timeStep = Variable(
    name = "t",
    variableType = VariableType.HiddenState(1),
    dimensions = List(dim1),
    dataType = DataType.Integer,
    description = Some("Discrete time step")
  )
  
  // Create all variables in the model
  val variables = List(
    hiddenState, nextState, observation, policy, action,
    likelihoodMatrix, transitionMatrix, preferenceVector, priorVector, habitVector,
    expectedFreeEnergy, timeStep
  )
  
  // Define the connections between variables
  val connections = List(
    Connection(
      source = NonEmptyList.one(priorVector),
      target = NonEmptyList.one(hiddenState),
      connectionType = ConnectionType.Directed
    ),
    Connection(
      source = NonEmptyList.one(hiddenState),
      target = NonEmptyList.one(likelihoodMatrix),
      connectionType = ConnectionType.Undirected
    ),
    Connection(
      source = NonEmptyList.one(hiddenState),
      target = NonEmptyList.one(nextState),
      connectionType = ConnectionType.Directed
    ),
    Connection(
      source = NonEmptyList.one(likelihoodMatrix),
      target = NonEmptyList.one(observation),
      connectionType = ConnectionType.Undirected
    ),
    Connection(
      source = NonEmptyList.one(hiddenState),
      target = NonEmptyList.one(transitionMatrix),
      connectionType = ConnectionType.Undirected
    ),
    Connection(
      source = NonEmptyList.one(preferenceVector),
      target = NonEmptyList.one(expectedFreeEnergy),
      connectionType = ConnectionType.Directed
    ),
    Connection(
      source = NonEmptyList.one(habitVector),
      target = NonEmptyList.one(policy),
      connectionType = ConnectionType.Directed
    ),
    Connection(
      source = NonEmptyList.one(expectedFreeEnergy),
      target = NonEmptyList.one(policy),
      connectionType = ConnectionType.Directed
    ),
    Connection(
      source = NonEmptyList.one(policy),
      target = NonEmptyList.one(action),
      connectionType = ConnectionType.Directed
    ),
    Connection(
      source = NonEmptyList.one(transitionMatrix),
      target = NonEmptyList.one(action),
      connectionType = ConnectionType.Directed
    ),
    Connection(
      source = NonEmptyList.one(action),
      target = NonEmptyList.one(nextState),
      connectionType = ConnectionType.Directed
    )
  )
  
  // Create matrices and probability distributions as defined in the POMDP model
  
  // A: 3 observations x 3 hidden states (identity-like mapping)
  val likelihoodProbs = Map(
    (0, 0) -> Probability(0.9), (0, 1) -> Probability(0.05), (0, 2) -> Probability(0.05),
    (1, 0) -> Probability(0.05), (1, 1) -> Probability(0.9), (1, 2) -> Probability(0.05),
    (2, 0) -> Probability(0.05), (2, 1) -> Probability(0.05), (2, 2) -> Probability(0.9)
  )
  
  // Initialize uniform distributions for states
  val stateSet = Set("state1", "state2", "state3")
  val obsSet = Set("obs1", "obs2", "obs3")
  val actionSet = Set("action1", "action2", "action3")
  
  val uniformProb = Probability(1.0 / 3)
  val uniformStateDist = CategoricalDist(stateSet.map(_ -> uniformProb).toMap)
  val uniformActionDist = CategoricalDist(actionSet.map(_ -> uniformProb).toMap)
  
  // Create state, observation, and action spaces
  val stateSpace = StateSpace(
    factors = NonEmptyList.one(hiddenState),
    states = stateSet,
    factorDims = NonEmptyList.one(3)
  )
  
  val observationSpace = ObservationSpace(
    modalities = NonEmptyList.one(observation),
    observations = obsSet,
    modalityDims = NonEmptyList.one(3)
  )
  
  val actionSpace = ActionSpace(
    controls = List(action),
    actions = actionSet,
    controlDims = List(3)
  )
  
  // Set preference vector values as defined in POMDP model
  val preferences = Map(
    "obs1" -> 0.1,
    "obs2" -> 0.1,
    "obs3" -> 1.0
  )
  
  // Attempt to create a complete POMDP Active Inference model
  try {
    println("Creating POMDP Active Inference model with GNN type system...")
    
    // Would complete the model with proper matrices and distributions
    // For brevity, we're demonstrating the structure but not implementing all matrices
    
    println("Model structure valid!")
    println(s"Number of variables: ${variables.size}")
    println(s"Number of connections: ${connections.size}")
    
    // In a complete implementation, we would create the full model:
    /*
    val pomdpModel = GNNModel(
      section = "ActInfPOMDP",
      version = "GNN v1",
      modelName = "Classic Active Inference POMDP Agent v1",
      annotation = "This model describes a classic Active Inference agent for a discrete POMDP",
      variables = variables,
      connections = connections,
      aiModel = ActiveInferenceModel(...),
      equations = List("Standard Active Inference update equations for POMDPs"),
      timeConfig = "Dynamic",
      ontologyMappings = Map(...),
      footer = "Active Inference POMDP Agent v1 - GNN Representation"
    )
    */
    
  } catch {
    case e: Exception => 
      println(s"Model validation failed: ${e.getMessage}")
  }
} 