/*
 * GNN (Generalized Notation Notation) Categorical Specification in Scala
 * 
 * This file provides a complete categorical specification of GNN models using
 * Scala and the Cats library, emphasizing category theory foundations and
 * functional programming principles for Active Inference systems.
 */

package gnn.categorical

import cats._
import cats.data._
import cats.effect._
import cats.implicits._
import cats.kernel.Monoid
import cats.arrow.Category
import spire.algebra._
import spire.math._
import spire.implicits._

import scala.language.higherKinds

// Basic Types and Algebraic Structures

/** Dimension with positive constraint */
case class Dimension(value: Int) {
  require(value > 0, "Dimension must be positive")
}

object Dimension {
  implicit val dimensionShow: Show[Dimension] = Show.show(_.value.toString)
  implicit val dimensionEq: Eq[Dimension] = Eq.by(_.value)
  implicit val dimensionOrder: Order[Dimension] = Order.by(_.value)
}

/** Variable types in Active Inference */
sealed trait VariableType extends Product with Serializable

object VariableType {
  case class HiddenState(factor: Int) extends VariableType
  case class Observation(modality: Int) extends VariableType  
  case class Action(control: Int) extends VariableType
  case class Policy(control: Int) extends VariableType
  case class LikelihoodMatrix(modality: Int) extends VariableType
  case class TransitionMatrix(factor: Int) extends VariableType
  case class PreferenceVector(modality: Int) extends VariableType
  case class PriorVector(factor: Int) extends VariableType
  case class HabitVector(control: Int) extends VariableType  // Added habit vector type
  
  implicit val variableTypeShow: Show[VariableType] = Show.show {
    case HiddenState(f) => s"s_f$f"
    case Observation(m) => s"o_m$m"
    case Action(c) => s"u_c$c"
    case Policy(c) => s"π_c$c"
    case LikelihoodMatrix(m) => s"A_m$m"
    case TransitionMatrix(f) => s"B_f$f"
    case PreferenceVector(m) => s"C_m$m"
    case PriorVector(f) => s"D_f$f"
    case HabitVector(c) => s"E_c$c"  // Show instance for habit vector
  }
  
  implicit val variableTypeEq: Eq[VariableType] = Eq.fromUniversalEquals
}

/** Data types supported in GNN */
sealed trait DataType extends Product with Serializable

object DataType {
  case object Categorical extends DataType
  case object Continuous extends DataType
  case object Binary extends DataType
  case object Integer extends DataType
  case object Float extends DataType
  case object Complex extends DataType
  
  implicit val dataTypeShow: Show[DataType] = Show.fromToString
  implicit val dataTypeEq: Eq[DataType] = Eq.fromUniversalEquals
}

/** GNN Variable with categorical structure */
case class Variable(
  name: String,
  variableType: VariableType,
  dimensions: List[Dimension],
  dataType: DataType,
  description: Option[String] = None
)

object Variable {
  implicit val variableShow: Show[Variable] = Show.show { v =>
    val dims = v.dimensions.map(_.value).mkString("[", ",", "]")
    s"${v.name}$dims"
  }
  
  implicit val variableEq: Eq[Variable] = Eq.by(v => (v.name, v.variableType, v.dimensions, v.dataType))
}

/** Connection types with categorical interpretation */
sealed trait ConnectionType extends Product with Serializable

object ConnectionType {
  case object Directed extends ConnectionType
  case object Undirected extends ConnectionType
  case object Conditional extends ConnectionType
  case object Bidirectional extends ConnectionType
  
  implicit val connectionTypeShow: Show[ConnectionType] = Show.fromToString
  implicit val connectionTypeEq: Eq[ConnectionType] = Eq.fromUniversalEquals
}

/** Morphism between variables in the GNN category */
case class Connection(
  source: NonEmptyList[Variable],
  target: NonEmptyList[Variable],
  connectionType: ConnectionType,
  weight: Option[Double] = None,
  description: Option[String] = None
)

object Connection {
  implicit val connectionShow: Show[Connection] = Show.show { c =>
    val sourceNames = c.source.map(_.name).toList.mkString("(", ",", ")")
    val targetNames = c.target.map(_.name).toList.mkString("(", ",", ")")
    val symbol = c.connectionType match {
      case ConnectionType.Directed => ">"
      case ConnectionType.Undirected => "-"
      case ConnectionType.Conditional => "|"
      case ConnectionType.Bidirectional => "<->"
    }
    s"$sourceNames$symbol$targetNames"
  }
}

// Probability Theory with Category Theory

/** Probability values as a monoid */
case class Probability(value: Double) {
  require(value >= 0.0 && value <= 1.0, "Probability must be between 0 and 1")
}

object Probability {
  implicit val probabilityShow: Show[Probability] = Show.show(_.value.toString)
  implicit val probabilityEq: Eq[Probability] = Eq.by(_.value)
  implicit val probabilityOrder: Order[Probability] = Order.by(_.value)
  
  // Probability forms a bounded semilattice under max
  implicit val probabilityBoundedSemilattice: BoundedSemilattice[Probability] =
    new BoundedSemilattice[Probability] {
      def empty: Probability = Probability(0.0)
      def combine(x: Probability, y: Probability): Probability = 
        Probability(math.max(x.value, y.value))
    }
  
  val zero: Probability = Probability(0.0)
  val one: Probability = Probability(1.0)
}

/** Categorical distribution as a functor */
case class CategoricalDist[A](probabilities: Map[A, Probability]) {
  require(
    math.abs(probabilities.values.map(_.value).sum - 1.0) < 1e-10,
    "Probabilities must sum to 1"
  )
  require(
    probabilities.values.forall(_.value >= 0.0),
    "All probabilities must be non-negative"
  )
}

object CategoricalDist {
  implicit def categoricalDistFunctor: Functor[CategoricalDist] = 
    new Functor[CategoricalDist] {
      def map[A, B](fa: CategoricalDist[A])(f: A => B): CategoricalDist[B] =
        CategoricalDist(fa.probabilities.map { case (a, p) => f(a) -> p })
    }
  
  implicit def categoricalDistMonad: Monad[CategoricalDist] = 
    new Monad[CategoricalDist] {
      def pure[A](a: A): CategoricalDist[A] = 
        CategoricalDist(Map(a -> Probability.one))
      
      def flatMap[A, B](fa: CategoricalDist[A])(f: A => CategoricalDist[B]): CategoricalDist[B] = {
        val combined = for {
          (a, pa) <- fa.probabilities.toList
          (b, pb) <- f(a).probabilities.toList
        } yield b -> Probability(pa.value * pb.value)
        
        val grouped = combined.groupBy(_._1).view.mapValues { pairs =>
          Probability(pairs.map(_._2.value).sum)
        }.toMap
        
        CategoricalDist(grouped)
      }
      
      def tailRecM[A, B](a: A)(f: A => CategoricalDist[Either[A, B]]): CategoricalDist[B] = {
        // Simplified implementation for finite distributions
        def loop(current: CategoricalDist[Either[A, B]]): CategoricalDist[B] = {
          val (lefts, rights) = current.probabilities.toList.partition(_._1.isLeft)
          val rightDist = CategoricalDist(rights.collect {
            case (Right(b), p) => b -> p
          }.toMap)
          
          if (lefts.isEmpty) rightDist
          else {
            val leftDist = CategoricalDist(lefts.collect {
              case (Left(a), p) => a -> p
            }.toMap)
            val nextDist = leftDist.flatMap(f)
            rightDist.combine(loop(nextDist))
          }
        }
        
        loop(f(a))
      }
    }
  
  implicit def categoricalDistShow[A: Show]: Show[CategoricalDist[A]] = 
    Show.show { dist =>
      dist.probabilities.map { case (a, p) => 
        s"${a.show} -> ${p.value}"
      }.mkString("CategoricalDist(", ", ", ")")
    }
  
  // Helper method for combining distributions
  implicit class CategoricalDistOps[A](dist: CategoricalDist[A]) {
    def combine(other: CategoricalDist[A])(implicit S: Semigroup[Probability]): CategoricalDist[A] = {
      val allKeys = dist.probabilities.keySet ++ other.probabilities.keySet
      val combined = allKeys.map { key =>
        val p1 = dist.probabilities.getOrElse(key, Probability.zero)
        val p2 = other.probabilities.getOrElse(key, Probability.zero)
        key -> S.combine(p1, p2)
      }.toMap
      
      // Renormalize
      val total = combined.values.map(_.value).sum
      if (total > 0) {
        CategoricalDist(combined.view.mapValues(p => Probability(p.value / total)).toMap)
      } else {
        dist
      }
    }
  }
}

/** Stochastic matrix as a natural transformation */
case class StochasticMatrix[A, B](matrix: Map[(A, B), Probability]) {
  def apply(a: A): CategoricalDist[B] = {
    val row = matrix.collect { case ((a2, b), p) if a2 == a => b -> p }
    if (row.nonEmpty) CategoricalDist(row) else sys.error(s"No row for $a")
  }
}

object StochasticMatrix {
  implicit def stochasticMatrixCategory[A]: Category[StochasticMatrix] = 
    new Category[StochasticMatrix] {
      def id[A]: StochasticMatrix[A, A] = ???  // Would need finite domain constraint
      
      def compose[A, B, C](f: StochasticMatrix[B, C], g: StochasticMatrix[A, B]): StochasticMatrix[A, C] = {
        // Matrix multiplication for stochastic matrices
        ???  // Implementation would require finite domains
      }
    }
}

// Active Inference Structures

/** State space as an object in the category of measurable spaces */
case class StateSpace[S](
  factors: NonEmptyList[Variable],
  states: Set[S],
  factorDims: NonEmptyList[Int]
) {
  require(
    states.size == factorDims.toList.product,
    "Joint dimension must equal product of factor dimensions"
  )
}

object StateSpace {
  implicit def stateSpaceFunctor: Functor[StateSpace] = 
    new Functor[StateSpace] {
      def map[A, B](fa: StateSpace[A])(f: A => B): StateSpace[B] =
        StateSpace(fa.factors, fa.states.map(f), fa.factorDims)
    }
}

/** Observation space */
case class ObservationSpace[O](
  modalities: NonEmptyList[Variable],
  observations: Set[O],
  modalityDims: NonEmptyList[Int]
)

/** Action space */
case class ActionSpace[U](
  controls: List[Variable],
  actions: Set[U],
  controlDims: List[Int]
)

/** Likelihood mapping as a natural transformation */
case class LikelihoodMapping[S, O](
  stateSpace: StateSpace[S],
  observationSpace: ObservationSpace[O],
  matrix: StochasticMatrix[S, O]
)

/** Transition mapping with actions */
case class TransitionMapping[S, U](
  stateSpace: StateSpace[S],
  actionSpace: ActionSpace[U],
  matrix: Map[U, StochasticMatrix[S, S]]
)

/** Preference mapping as a functor */
case class PreferenceMapping[O](
  observationSpace: ObservationSpace[O],
  preferences: Map[O, Double]  // Log preferences
)

/** Prior mapping */
case class PriorMapping[S](
  stateSpace: StateSpace[S],
  prior: CategoricalDist[S]
)

/** Habit mapping as a functor */
case class HabitMapping[U](
  actionSpace: ActionSpace[U],
  habits: CategoricalDist[U]  // Initial policy prior over actions
)

/** Complete Active Inference model as a structured category */
case class ActiveInferenceModel[S, O, U](
  stateSpace: StateSpace[S],
  observationSpace: ObservationSpace[O],
  actionSpace: ActionSpace[U],
  likelihood: LikelihoodMapping[S, O],
  transition: TransitionMapping[S, U],
  preferences: PreferenceMapping[O],
  priors: PriorMapping[S],
  habits: HabitMapping[U],  // Added habit mapping
  timeHorizon: Int
)

object ActiveInferenceModel {
  
  /** State inference as a natural transformation */
  def stateInference[S, O](
    model: ActiveInferenceModel[S, O, _],
    observation: O,
    priorBeliefs: CategoricalDist[S]
  ): CategoricalDist[S] = {
    // Bayes' rule: P(s|o) ∝ P(o|s) * P(s)
    val likelihood = model.likelihood.matrix.matrix.collect {
      case ((s, o2), p) if o2 == observation => s -> p
    }
    
    val unnormalized = priorBeliefs.probabilities.map { case (s, prior) =>
      val likelihoodProb = likelihood.getOrElse(s, Probability.zero)
      s -> Probability(prior.value * likelihoodProb.value)
    }
    
    val total = unnormalized.values.map(_.value).sum
    if (total > 0) {
      CategoricalDist(unnormalized.view.mapValues(p => Probability(p.value / total)).toMap)
    } else {
      priorBeliefs
    }
  }
  
  /** Expected free energy computation */
  def expectedFreeEnergy[S, O, U](
    model: ActiveInferenceModel[S, O, U],
    beliefs: CategoricalDist[S],
    action: U
  ): Double = {
    // Simplified EFE: negative expected reward
    beliefs.probabilities.toList.map { case (s, pS) =>
      model.observationSpace.observations.toList.map { o =>
        val likelihoodProb = model.likelihood.matrix.matrix.getOrElse((s, o), Probability.zero)
        val preference = model.preferences.preferences.getOrElse(o, 0.0)
        pS.value * likelihoodProb.value * preference
      }.sum
    }.sum
  }
  
  /** Policy inference via softmax over expected free energy, biased by habit */
  def policyInference[S, O, U](
    model: ActiveInferenceModel[S, O, U],
    beliefs: CategoricalDist[S]
  ): CategoricalDist[U] = {
    val efeValues = model.actionSpace.actions.map { action =>
      action -> expectedFreeEnergy(model, beliefs, action)
    }.toMap
    
    // Get habit prior
    val habitPrior = model.habits.habits
    
    // Softmax with inverted EFE (lower EFE = higher probability)
    // Modified to incorporate habit bias
    val expValues = efeValues.map { case (action, efe) => 
      val habitStrength = habitPrior.probabilities.getOrElse(action, Probability.zero).value
      action -> math.exp(-efe) * (1.0 + habitStrength)  // Habit modulates policy selection
    }
    val total = expValues.values.sum
    
    if (total > 0) {
      val normalized = expValues.view.mapValues(exp => Probability(exp / total)).toMap
      CategoricalDist(normalized)
    } else {
      // Uniform distribution
      val uniform = Probability(1.0 / model.actionSpace.actions.size)
      CategoricalDist(model.actionSpace.actions.map(_ -> uniform).toMap)
    }
  }
  
  /** One step of Active Inference as a state transition */
  def aiStep[S, O, U](
    model: ActiveInferenceModel[S, O, U],
    beliefs: CategoricalDist[S],
    observation: O
  ): (CategoricalDist[S], CategoricalDist[U]) = {
    val newBeliefs = stateInference(model, observation, beliefs)
    val policy = policyInference(model, newBeliefs)
    (newBeliefs, policy)
  }
}

// GNN Model Structure

/** Complete GNN model with categorical structure */
case class GNNModel[S, O, U](
  section: String,
  version: String,
  modelName: String,
  annotation: String,
  variables: List[Variable],
  connections: List[Connection],
  aiModel: ActiveInferenceModel[S, O, U],
  equations: List[String],
  timeConfig: String,
  ontologyMappings: Map[String, String],
  footer: String,
  signature: Option[String] = None
)

object GNNModel {
  
  /** Functor instance for GNN models */
  implicit def gnnModelFunctor[O, U]: Functor[GNNModel[*, O, U]] = 
    new Functor[GNNModel[*, O, U]] {
      def map[A, B](fa: GNNModel[A, O, U])(f: A => B): GNNModel[B, O, U] =
        fa.copy(aiModel = fa.aiModel.copy(
          stateSpace = fa.aiModel.stateSpace.map(f),
          likelihood = fa.aiModel.likelihood.copy(
            stateSpace = fa.aiModel.likelihood.stateSpace.map(f)
          ),
          transition = fa.aiModel.transition.copy(
            stateSpace = fa.aiModel.transition.stateSpace.map(f)
          ),
          priors = fa.aiModel.priors.copy(
            stateSpace = fa.aiModel.priors.stateSpace.map(f)
          )
        ))
    }
  
  /** Validation using categorical constraints */
  def validate[S, O, U](model: GNNModel[S, O, U]): ValidatedNel[String, Unit] = {
    val validations = List(
      validateSection(model.section),
      validateModelName(model.modelName),
      validateVariables(model.variables),
      validateConnections(model.connections, model.variables),
      validateAIModel(model.aiModel)
    )
    
    validations.sequence_
  }
  
  private def validateSection(section: String): ValidatedNel[String, Unit] =
    if (section.nonEmpty) ().validNel
    else "GNN section cannot be empty".invalidNel
  
  private def validateModelName(name: String): ValidatedNel[String, Unit] =
    if (name.nonEmpty) ().validNel
    else "Model name cannot be empty".invalidNel
  
  private def validateVariables(variables: List[Variable]): ValidatedNel[String, Unit] =
    if (variables.nonEmpty) ().validNel
    else "Variables list cannot be empty".invalidNel
  
  private def validateConnections(
    connections: List[Connection], 
    variables: List[Variable]
  ): ValidatedNel[String, Unit] = {
    val variableNames = variables.map(_.name).toSet
    val invalidConnections = connections.filter { conn =>
      val sourceNames = conn.source.toList.map(_.name).toSet
      val targetNames = conn.target.toList.map(_.name).toSet
      !(sourceNames.subsetOf(variableNames) && targetNames.subsetOf(variableNames))
    }
    
    if (invalidConnections.isEmpty) ().validNel
    else s"Invalid connections found: ${invalidConnections.map(_.show).mkString(", ")}".invalidNel
  }
  
  private def validateAIModel[S, O, U](model: ActiveInferenceModel[S, O, U]): ValidatedNel[String, Unit] =
    if (model.timeHorizon > 0) ().validNel
    else "Time horizon must be positive".invalidNel
}

// Model Transformations as Functors

/** Model transformation as a natural transformation */
trait ModelTransformation[F[_], G[_]] {
  def transform[S, O, U](model: F[GNNModel[S, O, U]]): G[GNNModel[S, O, U]]
}

/** Identity transformation */
object IdentityTransformation extends ModelTransformation[Id, Id] {
  def transform[S, O, U](model: GNNModel[S, O, U]): GNNModel[S, O, U] = model
}

/** Validation transformation */
object ValidationTransformation extends ModelTransformation[Id, ValidatedNel[String, *]] {
  def transform[S, O, U](model: GNNModel[S, O, U]): ValidatedNel[String, GNNModel[S, O, U]] =
    GNNModel.validate(model).map(_ => model)
}

// Export and Serialization

/** Type class for GNN serialization */
trait GNNSerializer[F[_]] {
  def serialize[S, O, U](model: GNNModel[S, O, U]): F[String]
  def deserialize[S, O, U](content: String): F[GNNModel[S, O, U]]
}

/** JSON serialization instance */
implicit object JsonGNNSerializer extends GNNSerializer[Either[String, *]] {
  def serialize[S, O, U](model: GNNModel[S, O, U]): Either[String, String] = {
    // Simplified JSON serialization
    Right(s"""{"section": "${model.section}", "modelName": "${model.modelName}"}""")
  }
  
  def deserialize[S, O, U](content: String): Either[String, GNNModel[S, O, U]] = {
    // Simplified JSON deserialization
    Left("Deserialization not implemented")
  }
}

// Example usage and testing
object GNNExample {
  
  // Create example state space
  val stateVar = Variable("s_f0", VariableType.HiddenState(0), List(Dimension(2)), DataType.Categorical)
  val obsVar = Variable("o_m0", VariableType.Observation(0), List(Dimension(3)), DataType.Categorical)
  
  val stateSpace = StateSpace(
    factors = NonEmptyList.one(stateVar),
    states = Set("state1", "state2"),
    factorDims = NonEmptyList.one(2)
  )
  
  val observationSpace = ObservationSpace(
    modalities = NonEmptyList.one(obsVar),
    observations = Set("obs1", "obs2", "obs3"),
    modalityDims = NonEmptyList.one(3)
  )
  
  val actionSpace = ActionSpace(
    controls = List.empty,
    actions = Set("action1"),
    controlDims = List.empty
  )
  
  // Example GNN model
  val exampleModel = GNNModel(
    section = "ExampleModel",
    version = "GNN v1",
    modelName = "Categorical Example",
    annotation = "Example GNN model using categorical structures",
    variables = List(stateVar, obsVar),
    connections = List.empty,
    aiModel = ActiveInferenceModel(
      stateSpace = stateSpace,
      observationSpace = observationSpace,
      actionSpace = actionSpace,
      likelihood = ??? // Would need proper likelihood matrix
      transition = ??? // Would need proper transition matrix
      preferences = ??? // Would need preferences
      priors = ??? // Would need priors
      habits = ??? // Would need habits
      timeHorizon = 10
    ),
    equations = List.empty,
    timeConfig = "Dynamic",
    ontologyMappings = Map.empty,
    footer = "End of example model"
  )
} 