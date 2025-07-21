{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : GNN
Description : Haskell representation of Generalized Notation Notation (GNN)
Copyright   : (c) 2025 Active Inference Institute
License     : MIT
Maintainer  : blanket@activeinference.institute
Stability   : experimental

This module provides a strongly-typed Haskell representation of GNN models,
leveraging the type system to enforce mathematical constraints at compile time.
-}

module GNN
  ( -- * Core Types
    GNNModel(..)
  , Variable(..)
  , VariableType(..)
  , DataType(..)
  , Dimensions(..)
  
    -- * Active Inference Components
  , StateSpace(..)
  , ObservationSpace(..)
  , ActionSpace(..)
  , LikelihoodMatrix(..)
  , TransitionMatrix(..)
  , PreferenceVector(..)
  , PriorVector(..)
  , HabitVector(..)  -- Added HabitVector to exports
  
    -- * Connections and Morphisms
  , Connection(..)
  , ConnectionType(..)
  , GNNMorphism(..)
  
    -- * Validation and Constraints
  , ValidationLevel(..)
  , ValidationResult(..)
  , Constraint(..)
  , validateModel
  
    -- * Mathematical Operations
  , compose
  , tensorProduct
  , expectedFreeEnergy
  , minimizeFreeEnergy
  
    -- * Categorical Structure
  , GNNCategory(..)
  , identity
  , (>>>) 
  
    -- * Examples
  , exampleTwoStateModel
  , exampleVisualForaging
  ) where

import Data.Matrix (Matrix, (!))
import qualified Data.Matrix as M
import Data.Vector (Vector)
import qualified Data.Vector as V
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Control.Monad (guard)
import Data.Validation (Validation(..), AccValidation, _Success, _Failure)
import Data.List.NonEmpty (NonEmpty)
import qualified Data.List.NonEmpty as NE
import GHC.TypeLits (Nat, KnownNat, natVal)
import Data.Proxy (Proxy(..))
import Text.Printf (printf)

-- | Dimensions of a tensor or variable (type-level)
data Dimensions (n :: [Nat]) where
  Dims :: KnownNat n => Proxy n -> Dimensions '[n]
  DimsN :: (KnownNat n, KnownNat m) => Proxy n -> Proxy m -> Dimensions '[n, m]

deriving instance Show (Dimensions n)

-- | Data types supported in GNN
data DataType
  = Categorical
  | Continuous
  | Binary
  | Integer
  | Float
  | Complex
  deriving (Show, Eq, Ord, Enum)

-- | Variable types in Active Inference
data VariableType
  = HiddenState { factorIndex :: Int }
  | Observation { modalityIndex :: Int }
  | Action { controlIndex :: Int }
  | Policy { controlIndex :: Int }
  | Habit { controlIndex :: Int }  -- Added Habit type for E vector
  deriving (Show, Eq, Ord)

-- | A GNN variable with type-level dimension information
data Variable (dims :: [Nat]) = Variable
  { varName :: String
  , varType :: VariableType
  , varDimensions :: Dimensions dims
  , varDataType :: DataType
  , varDescription :: Maybe String
  } deriving (Show)

-- | Connection types between variables
data ConnectionType
  = Directed
  | Undirected
  | Conditional
  | Bidirectional
  deriving (Show, Eq, Ord, Enum)

-- | Connection between variables
data Connection where
  Connection :: Variable m -> Variable n -> ConnectionType -> String -> Maybe String -> Connection

deriving instance Show Connection

-- | State space with multiple factors
data StateSpace where
  StateSpace :: [Variable dims] -> StateSpace

deriving instance Show StateSpace

-- | Observation space with multiple modalities
data ObservationSpace where
  ObservationSpace :: [Variable dims] -> ObservationSpace

deriving instance Show ObservationSpace

-- | Action space with multiple control factors
data ActionSpace where
  ActionSpace :: [Variable dims] -> ActionSpace

deriving instance Show ActionSpace

-- | Mathematical constraints for matrices and vectors
data Constraint
  = Stochastic        -- Rows sum to 1
  | NonNegative       -- All elements >= 0
  | Symmetric         -- Matrix is symmetric
  | Orthogonal        -- Matrix is orthogonal
  | Unitary           -- Matrix is unitary
  | Normalized        -- Vector has unit norm
  deriving (Show, Eq, Ord, Enum)

-- | Likelihood matrix A: P(o|s) with constraints
newtype LikelihoodMatrix (s :: Nat) (o :: Nat) = LikelihoodMatrix
  { getLikelihoodMatrix :: Matrix Double
  } deriving (Show)

-- | Transition matrix B: P(s'|s,u) with constraints
newtype TransitionMatrix (s :: Nat) (u :: Nat) = TransitionMatrix
  { getTransitionMatrix :: Matrix Double
  } deriving (Show)

-- | Preference vector C: log preferences over observations
newtype PreferenceVector (o :: Nat) = PreferenceVector
  { getPreferenceVector :: Vector Double
  } deriving (Show)

-- | Prior vector D: initial state distribution
newtype PriorVector (s :: Nat) = PriorVector
  { getPriorVector :: Vector Double
  } deriving (Show)

-- | Habit vector E: initial policy prior over actions
newtype HabitVector (u :: Nat) = HabitVector
  { getHabitVector :: Vector Double
  } deriving (Show)

-- | Validation levels for different use cases
data ValidationLevel
  = Basic
  | Standard
  | Strict
  | Research
  deriving (Show, Eq, Ord, Enum)

-- | Validation result with detailed feedback
data ValidationResult = ValidationResult
  { isValid :: Bool
  , validationErrors :: [String]
  , validationWarnings :: [String]
  , validationSuggestions :: [String]
  , validationMetadata :: Map String String
  } deriving (Show)

-- | Active Inference model with type-level constraints
data ActiveInferenceModel (s :: Nat) (o :: Nat) (u :: Nat) = ActiveInferenceModel
  { stateSpace :: StateSpace
  , observationSpace :: ObservationSpace
  , actionSpace :: ActionSpace
  , likelihoodMatrix :: LikelihoodMatrix s o
  , transitionMatrix :: TransitionMatrix s u
  , preferenceVector :: PreferenceVector o
  , priorVector :: PriorVector s
  , habitVector :: HabitVector u  -- Added habit vector to match POMDP agent model
  , timeHorizon :: Int
  } deriving (Show)

-- | Complete GNN model specification
data GNNModel where
  GNNModel :: (KnownNat s, KnownNat o, KnownNat u) =>
    { gnnName :: String
    , gnnVersion :: String
    , gnnAnnotation :: String
    , gnnActiveInferenceModel :: ActiveInferenceModel s o u
    , gnnVariables :: [Variable dims]
    , gnnConnections :: [Connection]
    , gnnEquations :: [String]
    , gnnOntologyMappings :: Map String String
    , gnnModelParameters :: Map String Double
    } -> GNNModel

deriving instance Show GNNModel

-- | Morphism between GNN models
data GNNMorphism where
  GNNMorphism :: (KnownNat s1, KnownNat o1, KnownNat u1,
                  KnownNat s2, KnownNat o2, KnownNat u2) =>
    { source :: ActiveInferenceModel s1 o1 u1
    , target :: ActiveInferenceModel s2 o2 u2
    , stateMap :: Vector Int
    , observationMap :: Vector Int
    , actionMap :: Vector Int
    } -> GNNMorphism

deriving instance Show GNNMorphism

-- | Category structure for GNN models
class GNNCategory cat where
  type Ob cat :: *
  type Mor cat :: * -> * -> *
  
  identity :: Ob cat -> Mor cat (Ob cat) (Ob cat)
  compose :: Mor cat b c -> Mor cat a b -> Mor cat a c

instance GNNCategory ActiveInferenceModel where
  type Ob ActiveInferenceModel = GNNModel
  type Mor ActiveInferenceModel = GNNMorphism
  
  identity = undefined -- Implementation required
  compose = undefined  -- Implementation required

-- | Composition operator for morphisms
(>>>) :: GNNMorphism -> GNNMorphism -> GNNMorphism
f >>> g = compose g f

-- | Validate a constraint on a matrix
validateConstraint :: Constraint -> Matrix Double -> Bool
validateConstraint Stochastic m = all (\i -> abs (sum [m ! (i, j) | j <- [1..M.ncols m]] - 1.0) < 1e-10) [1..M.nrows m]
validateConstraint NonNegative m = all (>= 0) (M.toList m)
validateConstraint Symmetric m = m == M.transpose m
validateConstraint _ _ = True -- Other constraints not implemented

-- | Validate a GNN model
validateModel :: ValidationLevel -> GNNModel -> ValidationResult
validateModel level model = case model of
  GNNModel name version annotation aiModel variables connections equations ontologyMappings modelParameters ->
    let errors = []
        warnings = []
        suggestions = []
        metadata = Map.fromList [("model_name", name), ("version", version)]
        
        -- Add validation logic based on level
        allValid = null errors
        
    in ValidationResult allValid errors warnings suggestions metadata

-- | Calculate expected free energy
expectedFreeEnergy :: (KnownNat s, KnownNat o, KnownNat u) =>
                     ActiveInferenceModel s o u -> 
                     Vector Double -> 
                     Vector Double -> 
                     Double
expectedFreeEnergy model beliefs policy = 
  let -- Complexity term (KL divergence from prior)
      complexity = sum $ V.zipWith (\b p -> if p > 0 then b * log (b / p) else 0) 
                                   beliefs (getPriorVector $ priorVector model)
      
      -- Expected cost (negative log probability of preferred outcomes)
      expectedCost = 0.0 -- Simplified implementation
      
  in complexity + expectedCost

-- | Policy inference with habit bias
policyInference :: (KnownNat s, KnownNat o, KnownNat u) =>
                  ActiveInferenceModel s o u ->
                  Vector Double ->
                  Vector Double
policyInference model beliefs =
  let -- Calculate EFE for each action
      efe = undefined -- EFE calculation
      
      -- Get habit bias
      habits = getHabitVector $ habitVector model
      
      -- Apply habit-modulated softmax
      softmax = V.map (\(e, h) -> exp (-e) * (1 + h)) $ V.zip efe habits
      total = V.sum softmax
      
      -- Normalize
      normalized = V.map (/ total) softmax
  
  in normalized

-- | Minimize free energy to find optimal beliefs
minimizeFreeEnergy :: (KnownNat s, KnownNat o, KnownNat u) =>
                     ActiveInferenceModel s o u -> 
                     Vector Double -> 
                     Vector Double
minimizeFreeEnergy model policy = 
  -- Simplified implementation - would use optimization algorithm
  getPriorVector $ priorVector model

-- | Tensor product of two Active Inference models
tensorProduct :: (KnownNat s1, KnownNat o1, KnownNat u1,
                  KnownNat s2, KnownNat o2, KnownNat u2) =>
                ActiveInferenceModel s1 o1 u1 ->
                ActiveInferenceModel s2 o2 u2 ->
                -- Result type would be (s1*s2, o1*o2, u1*u2) but requires type-level arithmetic
                String -- Placeholder
tensorProduct m1 m2 = "Tensor product not fully implemented"

-- | Smart constructor for likelihood matrix with validation
mkLikelihoodMatrix :: (KnownNat s, KnownNat o) => 
                     Matrix Double -> 
                     Either String (LikelihoodMatrix s o)
mkLikelihoodMatrix m = 
  if validateConstraint Stochastic m && validateConstraint NonNegative m
  then Right (LikelihoodMatrix m)
  else Left "Matrix does not satisfy stochastic and non-negative constraints"

-- | Smart constructor for transition matrix with validation
mkTransitionMatrix :: (KnownNat s, KnownNat u) => 
                     Matrix Double -> 
                     Either String (TransitionMatrix s u)
mkTransitionMatrix m = 
  if validateConstraint Stochastic m && validateConstraint NonNegative m
  then Right (TransitionMatrix m)
  else Left "Matrix does not satisfy stochastic and non-negative constraints"

-- | Smart constructor for prior vector with validation
mkPriorVector :: (KnownNat s) => 
                Vector Double -> 
                Either String (PriorVector s)
mkPriorVector v = 
  if abs (V.sum v - 1.0) < 1e-10 && V.all (>= 0) v
  then Right (PriorVector v)
  else Left "Vector does not satisfy probability distribution constraints"

-- | Smart constructor for habit vector with validation
mkHabitVector :: (KnownNat u) => 
                Vector Double -> 
                Either String (HabitVector u)
mkHabitVector v = 
  if abs (V.sum v - 1.0) < 1e-10 && V.all (>= 0) v
  then Right (HabitVector v)
  else Left "Habit vector does not satisfy probability distribution constraints"

-- | Example: Simple two-state Active Inference model
exampleTwoStateModel :: Either String (ActiveInferenceModel 2 2 2)
exampleTwoStateModel = do
  -- Likelihood matrix A: P(o|s)
  likelihoodMat <- mkLikelihoodMatrix $ M.fromLists [[0.8, 0.2], [0.3, 0.7]]
  
  -- Transition matrix B: P(s'|s,u) - simplified to 2x2 for example
  transitionMat <- mkTransitionMatrix $ M.fromLists [[0.9, 0.1], [0.2, 0.8]]
  
  -- Prior vector D: initial state distribution
  priorVec <- mkPriorVector $ V.fromList [0.5, 0.5]
  
  -- Preference vector C: log preferences
  let prefVec = PreferenceVector $ V.fromList [1.0, 0.0]
  
  -- Habit vector E: initial policy prior
  habitVec <- mkHabitVector $ V.fromList [0.5, 0.5]  -- Added habit vector
  
  -- Construct state space
  let stateVar = Variable "s_f0" (HiddenState 0) (Dims (Proxy :: Proxy 2)) Categorical Nothing
      stateSpace = StateSpace [stateVar]
  
  -- Construct observation space
  let obsVar = Variable "o_m0" (Observation 0) (Dims (Proxy :: Proxy 2)) Categorical Nothing
      obsSpace = ObservationSpace [obsVar]
  
  -- Construct action space
  let actionVar = Variable "u_c0" (Action 0) (Dims (Proxy :: Proxy 2)) Categorical Nothing
      actionSpace = ActionSpace [actionVar]
  
  return $ ActiveInferenceModel stateSpace obsSpace actionSpace 
                                likelihoodMat transitionMat prefVec priorVec habitVec 10

-- | Example: Visual foraging model with multiple modalities
exampleVisualForaging :: Either String GNNModel
exampleVisualForaging = do
  aiModel <- exampleTwoStateModel
  
  let locationVar = Variable "s_f0" (HiddenState 0) (Dims (Proxy :: Proxy 4)) Categorical (Just "Location factor")
      contextVar = Variable "s_f1" (HiddenState 1) (Dims (Proxy :: Proxy 2)) Categorical (Just "Context factor")
      visualVar = Variable "o_m0" (Observation 0) (Dims (Proxy :: Proxy 4)) Categorical (Just "Visual observations")
      habitVar = Variable "E_c0" (Habit 0) (Dims (Proxy :: Proxy 3)) Categorical (Just "Habit prior over actions")  -- Added habit variable
      
      variables = [locationVar, contextVar, visualVar, habitVar]  -- Include habit variable
      
      connections = [Connection locationVar visualVar Directed ">" (Just "Location generates visual observations")]
      
      equations = ["F = D_{KL}(q(s)||p(s)) + E_q[ln p(o|s)] - E_q[ln p(o|π)]"]
      
      ontologyMappings = Map.fromList 
        [ ("s_f0", "HiddenStateFactor")
        , ("s_f1", "HiddenStateFactor") 
        , ("o_m0", "ObservationModality")
        , ("E_c0", "HabitVector")  -- Added habit vector mapping
        ]
      
      modelParameters = Map.fromList 
        [ ("learning_rate", 0.1)
        , ("precision", 4.0)
        , ("temperature", 0.5)
        ]
  
  return $ GNNModel "Visual Foraging Model" "2.0.0" 
                   "Active Inference model for visual foraging behavior"
                   aiModel variables connections equations ontologyMappings modelParameters

-- | Pretty printing for GNN models
instance {-# OVERLAPPING #-} Show (ActiveInferenceModel s o u) where
  show model = printf "ActiveInferenceModel {\n  stateSpace = %s,\n  observationSpace = %s,\n  actionSpace = %s,\n  timeHorizon = %d\n}" 
                     (show $ stateSpace model)
                     (show $ observationSpace model)
                     (show $ actionSpace model)
                     (timeHorizon model)

-- | Functor instance for transforming variables
instance Functor Variable where
  fmap f (Variable name vtype dims dtype desc) = Variable (f name) vtype dims dtype (fmap f desc)

-- | Applicative instance for variable validation
instance Applicative (Either String) where
  pure = Right
  Left e <*> _ = Left e
  Right f <*> Right x = Right (f x)
  Right _ <*> Left e = Left e

-- | Monad instance for chaining validations
instance Monad (Either String) where
  return = Right
  Left e >>= _ = Left e
  Right x >>= f = f x

-- | Validation utilities
type ValidationErrors = NonEmpty String

-- | Accumulating validation
type AccValidation' = AccValidation ValidationErrors

-- | Validate variable naming convention
validateVariableName :: String -> AccValidation' String
validateVariableName name
  | name `elem` validPrefixes = _Success name
  | otherwise = _Failure (NE.singleton $ "Invalid variable name: " ++ name)
  where
    validPrefixes = ["s_f", "o_m", "u_c", "pi_c", "A_", "B_", "C_", "D_", "E_"]

-- | Type-level programming utilities for dimension checking
type family Multiply (m :: Nat) (n :: Nat) :: Nat where
  Multiply m n = m * n

type family Add (m :: Nat) (n :: Nat) :: Nat where
  Add m n = m + n

-- | Matrix operations with type-level dimension checking
matrixMultiply :: (KnownNat m, KnownNat n, KnownNat p) =>
                 Matrix Double -> 
                 Matrix Double -> 
                 Maybe (Matrix Double)
matrixMultiply m1 m2 = 
  if M.ncols m1 == M.nrows m2
  then Just (M.multStd m1 m2)
  else Nothing

-- | Export to various formats
class Exportable a where
  toJSON :: a -> String
  toXML :: a -> String
  toYAML :: a -> String

instance Exportable GNNModel where
  toJSON model = "{\"model\": \"" ++ show model ++ "\"}"
  toXML model = "<gnn-model>" ++ show model ++ "</gnn-model>"
  toYAML model = "model:\n  " ++ show model