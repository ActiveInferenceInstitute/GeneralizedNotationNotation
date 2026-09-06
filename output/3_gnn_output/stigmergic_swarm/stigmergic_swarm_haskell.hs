module StigmergicSwarmActiveInference where

import Data.List (sort)
import Numeric.LinearAlgebra ()

-- Variable Types
data A_agent1 = A_agent1 Double
data A_agent2 = A_agent2 Double
data A_agent3 = A_agent3 Double
data B_agent1 = B_agent1 Double
data B_agent2 = B_agent2 Double
data B_agent3 = B_agent3 Double
data C_agent1 = C_agent1 Double
data C_agent2 = C_agent2 Double
data C_agent3 = C_agent3 Double
data D_agent1 = D_agent1 Double
data D_agent2 = D_agent2 Double
data D_agent3 = D_agent3 Double
data G1 = G1 Double
data G2 = G2 Double
data G3 = G3 Double
data env_obs_likelihood = env_obs_likelihood Double
data env_signal = env_signal Double
data env_signal_prior = env_signal_prior Double
data o_agent1 = o_agent1 Int
data o_agent2 = o_agent2 Int
data o_agent3 = o_agent3 Int
data pi1 = pi1 Double
data pi2 = pi2 Double
data pi3 = pi3 Double
data s_agent1 = s_agent1 Double
data s_agent2 = s_agent2 Double
data s_agent3 = s_agent3 Double
data signal_decay = signal_decay Double
data signal_seek = signal_seek Double
data t = t Int
data u_agent1 = u_agent1 Int
data u_agent2 = u_agent2 Int
data u_agent3 = u_agent3 Int

-- Connections as Functions
D_agent1Tos_agent1 :: D_agent1 -> s_agent1
D_agent1Tos_agent1 x = undefined
s_agent1ToA_agent1 :: s_agent1 -> A_agent1
s_agent1ToA_agent1 x = undefined
A_agent1Too_agent1 :: A_agent1 -> o_agent1
A_agent1Too_agent1 x = undefined
C_agent1ToG1 :: C_agent1 -> G1
C_agent1ToG1 x = undefined
G1Topi1 :: G1 -> pi1
G1Topi1 x = undefined
pi1Tou_agent1 :: pi1 -> u_agent1
pi1Tou_agent1 x = undefined
B_agent1Tou_agent1 :: B_agent1 -> u_agent1
B_agent1Tou_agent1 x = undefined
D_agent2Tos_agent2 :: D_agent2 -> s_agent2
D_agent2Tos_agent2 x = undefined
s_agent2ToA_agent2 :: s_agent2 -> A_agent2
s_agent2ToA_agent2 x = undefined
A_agent2Too_agent2 :: A_agent2 -> o_agent2
A_agent2Too_agent2 x = undefined
C_agent2ToG2 :: C_agent2 -> G2
C_agent2ToG2 x = undefined
G2Topi2 :: G2 -> pi2
G2Topi2 x = undefined
pi2Tou_agent2 :: pi2 -> u_agent2
pi2Tou_agent2 x = undefined
B_agent2Tou_agent2 :: B_agent2 -> u_agent2
B_agent2Tou_agent2 x = undefined
D_agent3Tos_agent3 :: D_agent3 -> s_agent3
D_agent3Tos_agent3 x = undefined
s_agent3ToA_agent3 :: s_agent3 -> A_agent3
s_agent3ToA_agent3 x = undefined
A_agent3Too_agent3 :: A_agent3 -> o_agent3
A_agent3Too_agent3 x = undefined
C_agent3ToG3 :: C_agent3 -> G3
C_agent3ToG3 x = undefined
G3Topi3 :: G3 -> pi3
G3Topi3 x = undefined
pi3Tou_agent3 :: pi3 -> u_agent3
pi3Tou_agent3 x = undefined
B_agent3Tou_agent3 :: B_agent3 -> u_agent3
B_agent3Tou_agent3 x = undefined
u_agent1Toenv_signal :: u_agent1 -> env_signal
u_agent1Toenv_signal x = undefined
u_agent2Toenv_signal :: u_agent2 -> env_signal
u_agent2Toenv_signal x = undefined
u_agent3Toenv_signal :: u_agent3 -> env_signal
u_agent3Toenv_signal x = undefined
env_signalToA_agent1 :: env_signal -> A_agent1
env_signalToA_agent1 x = undefined
env_signalToA_agent2 :: env_signal -> A_agent2
env_signalToA_agent2 x = undefined
env_signalToA_agent3 :: env_signal -> A_agent3
env_signalToA_agent3 x = undefined
signal_decayToenv_signal :: signal_decay -> env_signal
signal_decayToenv_signal x = undefined

-- MODEL_DATA: {"model_name":"Stigmergic Swarm Active Inference","annotation":"Three Active Inference agents coordinating via stigmergy (environmental traces):\n\n- No direct communication between agents \u2014 coordination emerges from environment\n- Agents deposit and sense environmental signals (pheromone analogy)\n- Shared 3x3 grid environment with signal intensity at each cell\n- Each agent navigates independently while responding to accumulated signals\n- Signal deposition: actions leave traces that other agents can observe\n- Signal decay: environmental signals decay over time (volatility)\n- Demonstrates emergent collective behavior from individual free energy minimization\n- Models ant colony foraging, distributed robotics, and decentralized coordination","variables":[{"name":"A_agent1","var_type":"action","data_type":"float","dimensions":[4,9]},{"name":"B_agent1","var_type":"hidden_state","data_type":"float","dimensions":[9,9,4]},{"name":"C_agent1","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"D_agent1","var_type":"hidden_state","data_type":"float","dimensions":[9]},{"name":"s_agent1","var_type":"hidden_state","data_type":"float","dimensions":[9,1]},{"name":"o_agent1","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"pi1","var_type":"policy","data_type":"float","dimensions":[4]},{"name":"u_agent1","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G1","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"A_agent2","var_type":"action","data_type":"float","dimensions":[4,9]},{"name":"B_agent2","var_type":"hidden_state","data_type":"float","dimensions":[9,9,4]},{"name":"C_agent2","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"D_agent2","var_type":"hidden_state","data_type":"float","dimensions":[9]},{"name":"s_agent2","var_type":"hidden_state","data_type":"float","dimensions":[9,1]},{"name":"o_agent2","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"pi2","var_type":"policy","data_type":"float","dimensions":[4]},{"name":"u_agent2","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G2","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"A_agent3","var_type":"action","data_type":"float","dimensions":[4,9]},{"name":"B_agent3","var_type":"hidden_state","data_type":"float","dimensions":[9,9,4]},{"name":"C_agent3","var_type":"hidden_state","data_type":"float","dimensions":[4]},{"name":"D_agent3","var_type":"hidden_state","data_type":"float","dimensions":[9]},{"name":"s_agent3","var_type":"hidden_state","data_type":"float","dimensions":[9,1]},{"name":"o_agent3","var_type":"observation","data_type":"integer","dimensions":[4,1]},{"name":"pi3","var_type":"policy","data_type":"float","dimensions":[4]},{"name":"u_agent3","var_type":"action","data_type":"integer","dimensions":[1]},{"name":"G3","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"env_signal","var_type":"hidden_state","data_type":"float","dimensions":[9,1]},{"name":"signal_decay","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"env_obs_likelihood","var_type":"observation","data_type":"float","dimensions":[4,3]},{"name":"env_signal_prior","var_type":"prior_vector","data_type":"float","dimensions":[3]},{"name":"signal_seek","var_type":"hidden_state","data_type":"float","dimensions":[1]},{"name":"t","var_type":"hidden_state","data_type":"integer","dimensions":[1]}],"connections":[{"annotation":null,"source_variables":["D_agent1"],"target_variables":["s_agent1"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_agent1"],"target_variables":["A_agent1"],"connection_type":"undirected"},{"annotation":null,"source_variables":["A_agent1"],"target_variables":["o_agent1"],"connection_type":"undirected"},{"annotation":null,"source_variables":["C_agent1"],"target_variables":["G1"],"connection_type":"directed"},{"annotation":null,"source_variables":["G1"],"target_variables":["pi1"],"connection_type":"directed"},{"annotation":null,"source_variables":["pi1"],"target_variables":["u_agent1"],"connection_type":"directed"},{"annotation":null,"source_variables":["B_agent1"],"target_variables":["u_agent1"],"connection_type":"directed"},{"annotation":null,"source_variables":["D_agent2"],"target_variables":["s_agent2"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_agent2"],"target_variables":["A_agent2"],"connection_type":"undirected"},{"annotation":null,"source_variables":["A_agent2"],"target_variables":["o_agent2"],"connection_type":"undirected"},{"annotation":null,"source_variables":["C_agent2"],"target_variables":["G2"],"connection_type":"directed"},{"annotation":null,"source_variables":["G2"],"target_variables":["pi2"],"connection_type":"directed"},{"annotation":null,"source_variables":["pi2"],"target_variables":["u_agent2"],"connection_type":"directed"},{"annotation":null,"source_variables":["B_agent2"],"target_variables":["u_agent2"],"connection_type":"directed"},{"annotation":null,"source_variables":["D_agent3"],"target_variables":["s_agent3"],"connection_type":"directed"},{"annotation":null,"source_variables":["s_agent3"],"target_variables":["A_agent3"],"connection_type":"undirected"},{"annotation":null,"source_variables":["A_agent3"],"target_variables":["o_agent3"],"connection_type":"undirected"},{"annotation":null,"source_variables":["C_agent3"],"target_variables":["G3"],"connection_type":"directed"},{"annotation":null,"source_variables":["G3"],"target_variables":["pi3"],"connection_type":"directed"},{"annotation":null,"source_variables":["pi3"],"target_variables":["u_agent3"],"connection_type":"directed"},{"annotation":null,"source_variables":["B_agent3"],"target_variables":["u_agent3"],"connection_type":"directed"},{"annotation":null,"source_variables":["u_agent1"],"target_variables":["env_signal"],"connection_type":"directed"},{"annotation":null,"source_variables":["u_agent2"],"target_variables":["env_signal"],"connection_type":"directed"},{"annotation":null,"source_variables":["u_agent3"],"target_variables":["env_signal"],"connection_type":"directed"},{"annotation":null,"source_variables":["env_signal"],"target_variables":["A_agent1"],"connection_type":"undirected"},{"annotation":null,"source_variables":["env_signal"],"target_variables":["A_agent2"],"connection_type":"undirected"},{"annotation":null,"source_variables":["env_signal"],"target_variables":["A_agent3"],"connection_type":"undirected"},{"annotation":null,"source_variables":["signal_decay"],"target_variables":["env_signal"],"connection_type":"directed"}],"parameters":[{"name":"A_agent1","value":[[0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.1],[0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.7]],"param_type":"constant"},{"name":"A_agent2","value":[[0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.1],[0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.7]],"param_type":"constant"},{"name":"A_agent3","value":[[0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.1],[0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.7]],"param_type":"constant"},{"name":"C_agent1","value":[[-0.5,0.5,1.5,3.0]],"param_type":"constant"},{"name":"C_agent2","value":[[-0.5,0.5,1.5,3.0]],"param_type":"constant"},{"name":"C_agent3","value":[[-0.5,0.5,1.5,3.0]],"param_type":"constant"},{"name":"D_agent1","value":[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],"param_type":"constant"},{"name":"D_agent2","value":[[0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]],"param_type":"constant"},{"name":"D_agent3","value":[[0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]],"param_type":"constant"},{"name":"B_agent1","value":[[[1.0,0.0,0.0,0.9,0.0,0.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0,0.9,0.0,0.0,0.0,0.0],[0.0,0.0,1.0,0.0,0.0,0.9,0.0,0.0,0.0],[0.0,0.0,0.0,0.1,0.0,0.0,0.9,0.0,0.0],[0.0,0.0,0.0,0.0,0.1,0.0,0.0,0.9,0.0],[0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.0,0.9],[0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1]],[[0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0,0.0],[0.9,0.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0],[0.0,0.9,0.0,0.0,0.1,0.0,0.0,0.0,0.0],[0.0,0.0,0.9,0.0,0.0,0.1,0.0,0.0,0.0],[0.0,0.0,0.0,0.9,0.0,0.0,1.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.9,0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.9,0.0,0.0,1.0]],[[0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.9,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.9,1.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.9,0.1,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.9,1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.9,0.1,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.9,1.0]],[[1.0,0.9,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.1,0.9,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0,0.9,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.1,0.9,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.9,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.9],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1]]],"param_type":"constant"},{"name":"B_agent2","value":[[[1.0,0.0,0.0,0.9,0.0,0.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0,0.9,0.0,0.0,0.0,0.0],[0.0,0.0,1.0,0.0,0.0,0.9,0.0,0.0,0.0],[0.0,0.0,0.0,0.1,0.0,0.0,0.9,0.0,0.0],[0.0,0.0,0.0,0.0,0.1,0.0,0.0,0.9,0.0],[0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.0,0.9],[0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1]],[[0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0,0.0],[0.9,0.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0],[0.0,0.9,0.0,0.0,0.1,0.0,0.0,0.0,0.0],[0.0,0.0,0.9,0.0,0.0,0.1,0.0,0.0,0.0],[0.0,0.0,0.0,0.9,0.0,0.0,1.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.9,0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.9,0.0,0.0,1.0]],[[0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.9,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.9,1.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.9,0.1,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.9,1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.9,0.1,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.9,1.0]],[[1.0,0.9,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.1,0.9,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0,0.9,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.1,0.9,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.9,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.9],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1]]],"param_type":"constant"},{"name":"B_agent3","value":[[[1.0,0.0,0.0,0.9,0.0,0.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0,0.9,0.0,0.0,0.0,0.0],[0.0,0.0,1.0,0.0,0.0,0.9,0.0,0.0,0.0],[0.0,0.0,0.0,0.1,0.0,0.0,0.9,0.0,0.0],[0.0,0.0,0.0,0.0,0.1,0.0,0.0,0.9,0.0],[0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.0,0.9],[0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1]],[[0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0,0.0],[0.9,0.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0],[0.0,0.9,0.0,0.0,0.1,0.0,0.0,0.0,0.0],[0.0,0.0,0.9,0.0,0.0,0.1,0.0,0.0,0.0],[0.0,0.0,0.0,0.9,0.0,0.0,1.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.9,0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.9,0.0,0.0,1.0]],[[0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.9,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.9,1.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.9,0.1,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.9,1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.9,0.1,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.9,1.0]],[[1.0,0.9,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.1,0.9,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.1,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0,0.9,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.1,0.9,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.9,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.9],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1]]],"param_type":"constant"},{"name":"env_signal","value":[[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],"param_type":"constant"},{"name":"signal_decay","value":[[0.9]],"param_type":"constant"},{"name":"env_obs_likelihood","value":[[0.7,0.1,0.05],[0.15,0.7,0.15],[0.1,0.15,0.75],[0.05,0.05,0.05]],"param_type":"constant"},{"name":"env_signal_prior","value":[[0.7,0.2,0.1]],"param_type":"constant"},{"name":"signal_seek","value":[[2.0]],"param_type":"constant"},{"name":"num_hidden_states","value":729,"param_type":"constant"},{"name":"num_obs","value":64,"param_type":"constant"},{"name":"num_actions","value":4,"param_type":"constant"},{"name":"num_agents","value":3,"param_type":"constant"},{"name":"grid_size","value":9,"param_type":"constant"},{"name":"num_obs","value":4,"param_type":"constant"},{"name":"num_actions","value":4,"param_type":"constant"},{"name":"signal_decay_rate","value":0.9,"param_type":"constant"},{"name":"signal_deposit_rate","value":0.3,"param_type":"constant"},{"name":"num_timesteps","value":30,"param_type":"constant"}],"equations":[],"time_specification":{"time_type":"Dynamic","discretization":null,"horizon":30,"step_size":null},"ontology_mappings":[{"variable_name":"A_agent1","ontology_term":"Agent1LikelihoodMatrix","description":null},{"variable_name":"C_agent1","ontology_term":"Agent1PreferenceVector","description":null},{"variable_name":"D_agent1","ontology_term":"Agent1PositionPrior","description":null},{"variable_name":"s_agent1","ontology_term":"Agent1PositionState","description":null},{"variable_name":"o_agent1","ontology_term":"Agent1Observation","description":null},{"variable_name":"pi1","ontology_term":"Agent1PolicyVector","description":null},{"variable_name":"u_agent1","ontology_term":"Agent1Action","description":null},{"variable_name":"G1","ontology_term":"Agent1ExpectedFreeEnergy","description":null},{"variable_name":"A_agent2","ontology_term":"Agent2LikelihoodMatrix","description":null},{"variable_name":"C_agent2","ontology_term":"Agent2PreferenceVector","description":null},{"variable_name":"D_agent2","ontology_term":"Agent2PositionPrior","description":null},{"variable_name":"s_agent2","ontology_term":"Agent2PositionState","description":null},{"variable_name":"o_agent2","ontology_term":"Agent2Observation","description":null},{"variable_name":"pi2","ontology_term":"Agent2PolicyVector","description":null},{"variable_name":"u_agent2","ontology_term":"Agent2Action","description":null},{"variable_name":"G2","ontology_term":"Agent2ExpectedFreeEnergy","description":null},{"variable_name":"A_agent3","ontology_term":"Agent3LikelihoodMatrix","description":null},{"variable_name":"C_agent3","ontology_term":"Agent3PreferenceVector","description":null},{"variable_name":"D_agent3","ontology_term":"Agent3PositionPrior","description":null},{"variable_name":"s_agent3","ontology_term":"Agent3PositionState","description":null},{"variable_name":"o_agent3","ontology_term":"Agent3Observation","description":null},{"variable_name":"pi3","ontology_term":"Agent3PolicyVector","description":null},{"variable_name":"u_agent3","ontology_term":"Agent3Action","description":null},{"variable_name":"G3","ontology_term":"Agent3ExpectedFreeEnergy","description":null},{"variable_name":"env_signal","ontology_term":"EnvironmentalSignal","description":null},{"variable_name":"signal_decay","ontology_term":"SignalDecayRate","description":null},{"variable_name":"t","ontology_term":"Time","description":null}]}
