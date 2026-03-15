module ThreeLevelTemporalHierarchyAgent where

import Data.List (sort)
import Numeric.LinearAlgebra ()

-- Variable Types
data A0 = A0 Double
data A1 = A1 Double
data A2 = A2 Double
data B0 = B0 Double
data B1 = B1 Double
data B2 = B2 Double
data C0 = C0 Double
data C1 = C1 Double
data C2 = C2 Double
data D0 = D0 Double
data D1 = D1 Double
data D2 = D2 Double
data G0 = G0 Double
data G1 = G1 Double
data G2 = G2 Double
data o0 = o0 Int
data o1 = o1 Double
data o2 = o2 Double
data pi0 = pi0 Double
data pi1 = pi1 Double
data pi2 = pi2 Double
data s0 = s0 Double
data s1 = s1 Double
data s2 = s2 Double
data t = t Int
data tau0 = tau0 Double
data tau1 = tau1 Double
data tau2 = tau2 Double
data u0 = u0 Int
data u1 = u1 Int
data u2 = u2 Int

-- Connections as Functions
D0Tos0 :: D0 -> s0
D0Tos0 x = undefined
s0ToA0 :: s0 -> A0
s0ToA0 x = undefined
A0Too0 :: A0 -> o0
A0Too0 x = undefined
C0ToG0 :: C0 -> G0
C0ToG0 x = undefined
G0Topi0 :: G0 -> pi0
G0Topi0 x = undefined
pi0Tou0 :: pi0 -> u0
pi0Tou0 x = undefined
B0Tou0 :: B0 -> u0
B0Tou0 x = undefined
D1Tos1 :: D1 -> s1
D1Tos1 x = undefined
s1ToA1 :: s1 -> A1
s1ToA1 x = undefined
A1Too1 :: A1 -> o1
A1Too1 x = undefined
C1ToG1 :: C1 -> G1
C1ToG1 x = undefined
G1Topi1 :: G1 -> pi1
G1Topi1 x = undefined
pi1Tou1 :: pi1 -> u1
pi1Tou1 x = undefined
B1Tou1 :: B1 -> u1
B1Tou1 x = undefined
D2Tos2 :: D2 -> s2
D2Tos2 x = undefined
s2ToA2 :: s2 -> A2
s2ToA2 x = undefined
A2Too2 :: A2 -> o2
A2Too2 x = undefined
C2ToG2 :: C2 -> G2
C2ToG2 x = undefined
G2Topi2 :: G2 -> pi2
G2Topi2 x = undefined
pi2Tou2 :: pi2 -> u2
pi2Tou2 x = undefined
B2Tou2 :: B2 -> u2
B2Tou2 x = undefined
s2ToC1 :: s2 -> C1
s2ToC1 x = undefined
s1ToC0 :: s1 -> C0
s1ToC0 x = undefined
s2ToD1 :: s2 -> D1
s2ToD1 x = undefined
s0Too1 :: s0 -> o1
s0Too1 x = undefined
s1Too2 :: s1 -> o2
s1Too2 x = undefined

-- MODEL_DATA: {"model_name":"Three-Level Temporal Hierarchy Agent","annotation":"A three-level hierarchical Active Inference agent with distinct temporal scales:\n\n- Level 0 (fast, 100ms): Sensorimotor control \u2014 immediate reflexive responses\n- Level 1 (medium, 1s): Tactical planning \u2014 goal-directed behavior sequences\n- Level 2 (slow, 10s): Strategic planning \u2014 long-term objective management\n- Top-down flow: Strategy sets tactical goals, tactics set sensorimotor preferences\n- Bottom-up flow: Sensorimotor observations inform tactical beliefs, tactical outcomes inform strategy\n- Each level maintains its own generative model with A, B, C, D matrices\n- Timescale separation encoded via update ratios (Level 2 updates every 10 Level 0 steps)\n- Demonstrates deep temporal models from Friston et al. hierarchical Active Inference","variables":[{"name":"A0","var_type":"action","data_type":"float","dimensions":[3,4]},{"name":"B0","var_type":"hidden_state","data_type":"float","dimensions":[4,4,3]},{"name":"C0","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"D0","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"s0","var_type":"hidden_state","data_type":"float","dimensions":[4,1]},{"name":"o0","var_type":"observation","data_type":"integer","dimensions":[3,1]},{"name":"pi0","var_type":"policy","data_type":"float","dimensions":[3]},{"name":"u0","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G0","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"A1","var_type":"action","data_type":"float","dimensions":[4,3]},{"name":"B1","var_type":"hidden_state","data_type":"float","dimensions":[3,3,3]},{"name":"C1","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"D1","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"s1","var_type":"hidden_state","data_type":"float","dimensions":[3,1]},{"name":"o1","var_type":"observation","data_type":"float","dimensions":[4,1]},{"name":"pi1","var_type":"policy","data_type":"float","dimensions":[3]},{"name":"u1","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G1","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"A2","var_type":"action","data_type":"float","dimensions":[3,2]},{"name":"B2","var_type":"hidden_state","data_type":"float","dimensions":[2,2,2]},{"name":"C2","var_type":"hidden_state","data_type":"float","dimensions":[3]},{"name":"D2","var_type":"hidden_state","data_type":"float","dimensions":[2]},{"name":"s2","var_type":"hidden_state","data_type":"float","dimensions":[2,1]},{"name":"o2","var_type":"observation","data_type":"float","dimensions":[3,1]},{"name":"pi2","var_type":"policy","data_type":"float","dimensions":[2]},{"name":"u2","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G2","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"tau0","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"tau1","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"tau2","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"source_variables":["D0"],"target_variables":["s0"],"connection_type":"directed"},{"source_variables":["s0"],"target_variables":["A0"],"connection_type":"undirected"},{"source_variables":["A0"],"target_variables":["o0"],"connection_type":"undirected"},{"source_variables":["C0"],"target_variables":["G0"],"connection_type":"directed"},{"source_variables":["G0"],"target_variables":["pi0"],"connection_type":"directed"},{"source_variables":["pi0"],"target_variables":["u0"],"connection_type":"directed"},{"source_variables":["B0"],"target_variables":["u0"],"connection_type":"directed"},{"source_variables":["D1"],"target_variables":["s1"],"connection_type":"directed"},{"source_variables":["s1"],"target_variables":["A1"],"connection_type":"undirected"},{"source_variables":["A1"],"target_variables":["o1"],"connection_type":"undirected"},{"source_variables":["C1"],"target_variables":["G1"],"connection_type":"directed"},{"source_variables":["G1"],"target_variables":["pi1"],"connection_type":"directed"},{"source_variables":["pi1"],"target_variables":["u1"],"connection_type":"directed"},{"source_variables":["B1"],"target_variables":["u1"],"connection_type":"directed"},{"source_variables":["D2"],"target_variables":["s2"],"connection_type":"directed"},{"source_variables":["s2"],"target_variables":["A2"],"connection_type":"undirected"},{"source_variables":["A2"],"target_variables":["o2"],"connection_type":"undirected"},{"source_variables":["C2"],"target_variables":["G2"],"connection_type":"directed"},{"source_variables":["G2"],"target_variables":["pi2"],"connection_type":"directed"},{"source_variables":["pi2"],"target_variables":["u2"],"connection_type":"directed"},{"source_variables":["B2"],"target_variables":["u2"],"connection_type":"directed"},{"source_variables":["s2"],"target_variables":["C1"],"connection_type":"directed"},{"source_variables":["s1"],"target_variables":["C0"],"connection_type":"directed"},{"source_variables":["s2"],"target_variables":["D1"],"connection_type":"directed"},{"source_variables":["s0"],"target_variables":["o1"],"connection_type":"directed"},{"source_variables":["s1"],"target_variables":["o2"],"connection_type":"directed"}],"parameters":[{"name":"A0","value":[[0.85,0.05,0.05,0.05],[0.05,0.85,0.05,0.05],[0.05,0.05,0.85,0.05]],"param_type":"constant"},{"name":"C0","value":[[0.0,-1.0,1.0]],"param_type":"constant"},{"name":"D0","value":[[0.25,0.25,0.25,0.25]],"param_type":"constant"},{"name":"A1","value":[[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8],[0.1,0.1,0.1]],"param_type":"constant"},{"name":"C1","value":[[-0.5,1.0,1.5,-1.0]],"param_type":"constant"},{"name":"D1","value":[[0.33,0.33,0.34]],"param_type":"constant"},{"name":"A2","value":[[0.9,0.1],[0.1,0.9],[0.1,0.1]],"param_type":"constant"},{"name":"C2","value":[[-1.0,2.0,0.5]],"param_type":"constant"},{"name":"D2","value":[[0.5,0.5]],"param_type":"constant"},{"name":"tau0","value":[[0.1]],"param_type":"constant"},{"name":"tau1","value":[[1.0]],"param_type":"constant"},{"name":"tau2","value":[[10.0]],"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":100,"step_size":null},"ontology_mappings":[{"variable_name":"A0","ontology_term":"FastLikelihoodMatrix","description":null},{"variable_name":"B0","ontology_term":"FastTransitionMatrix","description":null},{"variable_name":"C0","ontology_term":"FastPreferenceVector","description":null},{"variable_name":"D0","ontology_term":"FastPrior","description":null},{"variable_name":"s0","ontology_term":"FastHiddenState","description":null},{"variable_name":"o0","ontology_term":"FastObservation","description":null},{"variable_name":"pi0","ontology_term":"FastPolicyVector","description":null},{"variable_name":"u0","ontology_term":"FastAction","description":null},{"variable_name":"G0","ontology_term":"FastExpectedFreeEnergy","description":null},{"variable_name":"A1","ontology_term":"TacticalLikelihoodMatrix","description":null},{"variable_name":"B1","ontology_term":"TacticalTransitionMatrix","description":null},{"variable_name":"C1","ontology_term":"TacticalPreferenceVector","description":null},{"variable_name":"D1","ontology_term":"TacticalPrior","description":null},{"variable_name":"s1","ontology_term":"TacticalHiddenState","description":null},{"variable_name":"o1","ontology_term":"TacticalObservation","description":null},{"variable_name":"pi1","ontology_term":"TacticalPolicyVector","description":null},{"variable_name":"u1","ontology_term":"TacticalAction","description":null},{"variable_name":"G1","ontology_term":"TacticalExpectedFreeEnergy","description":null},{"variable_name":"A2","ontology_term":"StrategicLikelihoodMatrix","description":null},{"variable_name":"B2","ontology_term":"StrategicTransitionMatrix","description":null},{"variable_name":"C2","ontology_term":"StrategicPreferenceVector","description":null},{"variable_name":"D2","ontology_term":"StrategicPrior","description":null},{"variable_name":"s2","ontology_term":"StrategicHiddenState","description":null},{"variable_name":"o2","ontology_term":"StrategicObservation","description":null},{"variable_name":"pi2","ontology_term":"StrategicPolicyVector","description":null},{"variable_name":"u2","ontology_term":"StrategicAction","description":null},{"variable_name":"G2","ontology_term":"StrategicExpectedFreeEnergy","description":null},{"variable_name":"tau0","ontology_term":"FastTimeConstant","description":null},{"variable_name":"tau1","ontology_term":"TacticalTimeConstant","description":null},{"variable_name":"tau2","ontology_term":"StrategicTimeConstant","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
