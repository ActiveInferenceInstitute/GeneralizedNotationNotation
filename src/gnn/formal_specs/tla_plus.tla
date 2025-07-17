---- MODULE GNNTemporalLogic ----
(*
  GNN (Generalized Notation Notation) Temporal Logic Specification in TLA+
  
  This specification models the temporal behavior and dynamic properties
  of Active Inference systems using TLA+ temporal logic, enabling
  verification of liveness and safety properties.
*)

EXTENDS Naturals, Reals, Sequences, FiniteSets, TLC

CONSTANTS
  MaxStates,        \* Maximum number of hidden states
  MaxObservations,  \* Maximum number of observations  
  MaxActions,       \* Maximum number of actions
  MaxTimeSteps,     \* Maximum simulation time steps
  Precision         \* Numerical precision for real comparisons

ASSUME /\ MaxStates \in Nat \ {0}
       /\ MaxObservations \in Nat \ {0}
       /\ MaxActions \in Nat \ {0}
       /\ MaxTimeSteps \in Nat \ {0}
       /\ Precision \in Real
       /\ Precision > 0

---- Variable Types and Domains ----

\* State space indices
States == 1..MaxStates

\* Observation space indices  
Observations == 1..MaxObservations

\* Action space indices
Actions == 1..MaxActions

\* Time steps
TimeSteps == 0..MaxTimeSteps

\* Probability values (approximated as rationals)
Probabilities == {r \in Real : r >= 0 /\ r <= 1}

\* Probability distributions (must sum to 1)
ProbabilityDistributions == {f \in [States -> Probabilities] : 
                            LET sum == CHOOSE s \in Real : 
                                       s = (+ {f[i] : i \in States})
                            IN Abs(sum - 1) < Precision}

\* Stochastic matrices
StochasticMatrices == {M \in [States \X Observations -> Probabilities] :
                      \A s \in States : 
                        LET rowSum == CHOOSE r \in Real :
                                      r = (+ {M[s, o] : o \in Observations})
                        IN Abs(rowSum - 1) < Precision}

---- GNN Model Structure ----

\* Variable types in Active Inference
VariableTypes == {"HiddenState", "Observation", "Action", "Policy", 
                 "LikelihoodMatrix", "TransitionMatrix", 
                 "PreferenceVector", "PriorVector"}

\* GNN Variable definition
GNNVariable == [name : STRING,
                type : VariableTypes,
                dimensions : Seq(Nat),
                dataType : STRING]

\* Connection types
ConnectionTypes == {"Directed", "Undirected", "Conditional", "Bidirectional"}

\* GNN Connection definition  
GNNConnection == [source : Seq(STRING),
                  target : Seq(STRING),
                  type : ConnectionTypes,
                  weight : Real]

\* Active Inference Model components
ActiveInferenceModel == [
  \* Likelihood matrices A: P(o|s)
  A : StochasticMatrices,
  
  \* Transition matrices B: P(s'|s,u) 
  B : [States \X States \X Actions -> Probabilities],
  
  \* Preference vectors C: log preferences over observations
  C : [Observations -> Real],
  
  \* Prior vectors D: initial state distributions
  D : ProbabilityDistributions,
  
  \* Time horizon
  T : Nat
]

\* Complete GNN Model
GNNModel == [
  section : STRING,
  version : STRING,
  modelName : STRING,
  annotation : STRING,
  variables : Seq(GNNVariable),
  connections : Seq(GNNConnection),
  aiModel : ActiveInferenceModel,
  equations : Seq(STRING),
  timeConfig : STRING,
  ontologyMappings : Seq([variable : STRING, ontologyTerm : STRING]),
  footer : STRING
]

---- State Variables ----

VARIABLES
  \* Current GNN model being executed
  model,
  
  \* Current time step
  time,
  
  \* Current beliefs over hidden states
  beliefs,
  
  \* Current observations
  observations,
  
  \* Current policy (distribution over actions)
  policy,
  
  \* Action taken at current step
  action,
  
  \* History of beliefs
  beliefHistory,
  
  \* History of observations
  observationHistory,
  
  \* History of actions
  actionHistory,
  
  \* System state (running, stopped, error)
  systemState

---- Helper Functions ----

\* Check if a value is approximately equal to another (within precision)
ApproxEqual(x, y) == Abs(x - y) < Precision

\* Sum of a set of real numbers
Sum(S) == CHOOSE s \in Real : s = (+ S)

\* Normalize a probability distribution
Normalize(dist) == 
  LET total == Sum({dist[i] : i \in DOMAIN dist})
  IN IF total > 0 
     THEN [i \in DOMAIN dist |-> dist[i] / total]
     ELSE [i \in DOMAIN dist |-> 1 / Cardinality(DOMAIN dist)]

\* Matrix-vector multiplication for belief update
MatrixVectorMult(matrix, vector) ==
  [i \in DOMAIN vector |-> 
    Sum({matrix[i, j] * vector[j] : j \in DOMAIN vector})]

\* Softmax function for policy computation
Softmax(values) ==
  LET expValues == [i \in DOMAIN values |-> Exp(values[i])]
      total == Sum({expValues[i] : i \in DOMAIN expValues})
  IN [i \in DOMAIN values |-> expValues[i] / total]

\* Expected free energy computation (simplified)
ExpectedFreeEnergy(aiModel, beliefs, actionIndex) ==
  \* Simplified EFE: negative expected reward
  -Sum({beliefs[s] * aiModel.C[o] * aiModel.A[s, o] : 
        s \in States, o \in Observations})

---- Active Inference Dynamics ----

\* State inference using Bayes' rule: P(s|o) ∝ P(o|s) * P(s)
StateInference(aiModel, obs, priorBeliefs) ==
  LET likelihood == [s \in States |-> aiModel.A[s, obs]]
      unnormalized == [s \in States |-> likelihood[s] * priorBeliefs[s]]
  IN Normalize(unnormalized)

\* Policy inference via minimizing expected free energy
PolicyInference(aiModel, currentBeliefs) ==
  LET efeValues == [a \in Actions |-> 
                     ExpectedFreeEnergy(aiModel, currentBeliefs, a)]
      \* Invert EFE for softmax (lower EFE = higher probability)
      invertedEFE == [a \in Actions |-> -efeValues[a]]
  IN Softmax(invertedEFE)

\* Action sampling from policy
ActionSampling(policyDist) ==
  \* Simplified: choose action with highest probability
  CHOOSE a \in Actions : \A a2 \in Actions : policyDist[a] >= policyDist[a2]

---- Temporal Behavior Specification ----

\* Initial state predicate
Init == 
  /\ model \in GNNModel
  /\ time = 0
  /\ beliefs = model.aiModel.D  \* Start with prior beliefs
  /\ observations \in Observations
  /\ policy = [a \in Actions |-> 1 / Cardinality(Actions)]  \* Uniform initial policy
  /\ action \in Actions
  /\ beliefHistory = <<model.aiModel.D>>
  /\ observationHistory = <<>>
  /\ actionHistory = <<>>
  /\ systemState = "running"

\* One step of Active Inference
ActiveInferenceStep ==
  /\ systemState = "running"
  /\ time < MaxTimeSteps
  /\ \E newObs \in Observations :
       \* Update beliefs based on new observation
       /\ LET newBeliefs == StateInference(model.aiModel, newObs, beliefs)
              newPolicy == PolicyInference(model.aiModel, newBeliefs)
              newAction == ActionSampling(newPolicy)
          IN /\ beliefs' = newBeliefs
             /\ observations' = newObs
             /\ policy' = newPolicy
             /\ action' = newAction
             /\ time' = time + 1
             /\ beliefHistory' = Append(beliefHistory, newBeliefs)
             /\ observationHistory' = Append(observationHistory, newObs)
             /\ actionHistory' = Append(actionHistory, newAction)
             /\ systemState' = systemState
             /\ UNCHANGED model

\* System termination
Terminate ==
  /\ systemState = "running"
  /\ time >= MaxTimeSteps
  /\ systemState' = "stopped"
  /\ UNCHANGED <<model, time, beliefs, observations, policy, action,
                beliefHistory, observationHistory, actionHistory>>

\* Error state (for robustness)
ErrorState ==
  /\ systemState' = "error"
  /\ UNCHANGED <<model, time, beliefs, observations, policy, action,
                beliefHistory, observationHistory, actionHistory>>

\* Next state relation
Next == ActiveInferenceStep \/ Terminate \/ ErrorState

\* Complete specification
Spec == Init /\ [][Next]_<<model, time, beliefs, observations, policy, action,
                          beliefHistory, observationHistory, actionHistory,
                          systemState>>

---- Safety Properties ----

\* Type invariant: all variables maintain their types
TypeInvariant ==
  /\ model \in GNNModel
  /\ time \in TimeSteps
  /\ beliefs \in ProbabilityDistributions
  /\ observations \in Observations
  /\ policy \in [Actions -> Probabilities]
  /\ action \in Actions
  /\ beliefHistory \in Seq(ProbabilityDistributions)
  /\ observationHistory \in Seq(Observations)
  /\ actionHistory \in Seq(Actions)
  /\ systemState \in {"running", "stopped", "error"}

\* Beliefs always form a valid probability distribution
BeliefsValid == 
  /\ \A s \in States : beliefs[s] >= 0
  /\ ApproxEqual(Sum({beliefs[s] : s \in States}), 1)

\* Policy always forms a valid probability distribution
PolicyValid ==
  /\ \A a \in Actions : policy[a] >= 0
  /\ ApproxEqual(Sum({policy[a] : a \in Actions}), 1)

\* History lengths are consistent
HistoryConsistent ==
  /\ Len(beliefHistory) = time + 1
  /\ Len(observationHistory) = time
  /\ Len(actionHistory) = time

\* Safety: system never enters invalid state
Safety == TypeInvariant /\ BeliefsValid /\ PolicyValid /\ HistoryConsistent

---- Liveness Properties ----

\* Eventually the system terminates or continues making progress
Progress == 
  \/ systemState = "stopped"
  \/ systemState = "error"
  \/ WF_<<time>>(ActiveInferenceStep)

\* If running, eventually either terminate or take another step
EventualProgress ==
  systemState = "running" ~> (systemState = "stopped" \/ time' > time)

\* Beliefs eventually converge (simplified convergence criterion)
BeliefConvergence ==
  \E n \in Nat : 
    \A m \in Nat : m > n =>
      (m < Len(beliefHistory) /\ m-1 < Len(beliefHistory)) =>
      \A s \in States : 
        Abs(beliefHistory[m][s] - beliefHistory[m-1][s]) < Precision

---- Fairness Properties ----

\* All actions are eventually explored (under suitable conditions)
ActionExploration ==
  \A a \in Actions : 
    <>[](\E t \in 1..Len(actionHistory) : actionHistory[t] = a)

\* All states are eventually believed possible (ergodicity)
StateExploration ==
  \A s \in States :
    <>[](\E t \in 1..Len(beliefHistory) : beliefHistory[t][s] > Precision)

---- Model Validation Properties ----

\* GNN model structure is well-formed
ModelWellFormed ==
  /\ model.aiModel.A \in StochasticMatrices
  /\ model.aiModel.D \in ProbabilityDistributions
  /\ \A s1, s2 \in States, a \in Actions : 
       model.aiModel.B[s1, s2, a] \in Probabilities
  /\ \A s \in States, a \in Actions :
       ApproxEqual(Sum({model.aiModel.B[s1, s, a] : s1 \in States}), 1)

\* Active Inference semantics are preserved
AISemantics ==
  /\ BeliefsValid
  /\ PolicyValid
  /\ \A t \in 1..Len(beliefHistory) :
       beliefHistory[t] \in ProbabilityDistributions

---- Temporal Logic Formulas ----

\* Always maintain safety
AlwaysSafe == []Safety

\* Eventually make progress
EventuallyProgress == <>Progress

\* Infinitely often explore actions (under fairness assumptions)
InfinitelyOftenExplore == 
  \A a \in Actions : []<>(action = a)

\* Beliefs stabilize eventually
BeliefStabilization ==
  <>[]\A s \in States, t1, t2 \in Nat :
    (t1 < Len(beliefHistory) /\ t2 < Len(beliefHistory) /\ t1 > time - 10 /\ t2 > time - 10) =>
    Abs(beliefHistory[t1][s] - beliefHistory[t2][s]) < Precision

---- Verification Goals ----

\* Main theorem: GNN models preserve Active Inference semantics over time
THEOREM GNNPreservesAI == Spec => []AISemantics

\* Safety theorem: system always maintains valid probability distributions
THEOREM AlwaysValid == Spec => AlwaysSafe

\* Liveness theorem: system makes progress
THEOREM MakesProgress == Spec => EventuallyProgress

\* Convergence theorem: beliefs eventually stabilize
THEOREM BeliefsConverge == Spec => BeliefStabilization

---- Model Checking Configuration ----

\* State space constraints for model checking
StateConstraint == 
  /\ time <= 10  \* Limit time steps for finite model checking
  /\ Cardinality({s \in States : beliefs[s] > 0}) <= 5  \* Limit active states

\* Symmetry reductions
Symmetry == Permutations(States) \cup Permutations(Actions)

\* Properties to check
PROPERTY SafetyProperty == AlwaysSafe
PROPERTY LivenessProperty == EventuallyProgress  
PROPERTY ConvergenceProperty == BeliefStabilization

==== 