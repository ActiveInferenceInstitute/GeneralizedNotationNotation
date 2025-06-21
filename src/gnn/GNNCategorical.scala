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

// Basic Types and Algebraic Structures

case class Dimension(value: Int) {
  require(value > 0, "Dimension must be positive")
}

object Dimension {
  implicit val dimensionShow: Show[Dimension] = Show.show(_.value.toString)
  implicit val dimensionEq: Eq[Dimension] = Eq.by(_.value)
}

sealed trait VariableType

object VariableType {
  case class HiddenState(factor: Int) extends VariableType
  case class Observation(modality: Int) extends VariableType  
  case class LikelihoodMatrix(modality: Int) extends VariableType
  case class TransitionMatrix(factor: Int) extends VariableType
  
  implicit val variableTypeShow: Show[VariableType] = Show.show {
    case HiddenState(f) => s"s_f$f"
    case Observation(m) => s"o_m$m"
    case LikelihoodMatrix(m) => s"A_m$m"
    case TransitionMatrix(f) => s"B_f$f"
  }
}

case class Variable(
  name: String,
  variableType: VariableType,
  dimensions: List[Dimension],
  dataType: String
)

case class Probability(value: Double) {
  require(value >= 0.0 && value <= 1.0, "Probability must be between 0 and 1")
}

case class CategoricalDist[A](probabilities: Map[A, Probability]) {
  require(
    math.abs(probabilities.values.map(_.value).sum - 1.0) < 1e-10,
    "Probabilities must sum to 1"
  )
}

object CategoricalDist {
  implicit def categoricalDistFunctor: Functor[CategoricalDist] = 
    new Functor[CategoricalDist] {
      def map[A, B](fa: CategoricalDist[A])(f: A => B): CategoricalDist[B] =
        CategoricalDist(fa.probabilities.map { case (a, p) => f(a) -> p })
    }
}

case class ActiveInferenceModel[S, O, U](
  stateSpace: Set[S],
  observationSpace: Set[O],
  actionSpace: Set[U],
  timeHorizon: Int
)

case class GNNModel[S, O, U](
  section: String,
  version: String,
  modelName: String,
  variables: List[Variable],
  aiModel: ActiveInferenceModel[S, O, U]
)

object GNNModel {
  def validate[S, O, U](model: GNNModel[S, O, U]): ValidatedNel[String, Unit] = {
    if (model.section.nonEmpty && model.modelName.nonEmpty) ().validNel
    else "Invalid model structure".invalidNel
  }
} 