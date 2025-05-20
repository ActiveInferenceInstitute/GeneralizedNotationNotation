import numpy as np
from pymdp.agent import Agent
from pymdp import utils
from pymdp import maths
import copy
import sys
import inspect
import traceback

# --- GNN to PyMDP Conversion Summary ---
# INFO: Starting GNN data extraction.
# DEBUG: Successfully parsed stringified direct_num_obs: [3, 3, 3]
# DEBUG: Successfully parsed stringified direct_num_states: [2, 3]
# INFO: Observation dimensions (num_obs) derived directly from gnn_spec.num_obs_modalities: [3, 3, 3]
# INFO: Observation names generated as defaults (gnn_spec.obs_modality_names not found or mismatched).
# INFO: Hidden state dimensions (num_states) derived directly from gnn_spec.num_hidden_states_factors: [2, 3]
# INFO: Hidden state names generated as defaults (gnn_spec.hidden_state_factor_names not found or mismatched).
# WARNING: Could not definitively determine control structure from gnn_spec, ModelParameters or StateSpaceBlock. control_fac_idx might be empty.
# INFO: A_spec: No 'A_Matrix' or 'A_m<idx>' keys found in InitialParameterization.
# INFO: B_spec: No 'B_Matrix' or 'B_f<idx>' keys found in InitialParameterization.
# INFO: C_spec: No 'C_Vector' or 'C_m<idx>' keys found in InitialParameterization.
# INFO: D_spec: No 'D_Vector' or 'D_f<idx>' keys found in InitialParameterization.
# INFO: E_spec: No 'E_Vector' or 'E' key found.
# INFO: Finished GNN data extraction.
# INFO: A_matrix: No A_spec provided in GNN. All modalities of A defaulted to uniform.
# DEBUG: B_matrix (factor factor_0): Initialized with default identity matrix for uncontrolled factor.
# DEBUG: B_matrix (factor factor_1): Initialized with default identity matrix for uncontrolled factor.
# INFO: B_matrix: No B_spec provided in GNN. B slices will use default initializations (identities).
# INFO: C_vector: No C_spec. C will be initialized to zeros by obj_array_zeros.
# INFO: D_vector: No D_spec. All factors of D defaulted to uniform.
# INFO: AgentHyperparameters: Extracting learning and algorithm parameter dicts.
# INFO: Simulation: No 'initial_observations' in GNN, using default.
# INFO: Simulation: No 'initial_true_states' in GNN, using default.
# --- End of GNN to PyMDP Conversion Summary ---


# --- GNN Model: Multifactor_PyMDP_Agent_v1 ---

obs_names = ['modality_0', 'modality_1', 'modality_2']
num_obs = [3, 3, 3]
num_modalities = 3
state_names = ['factor_0', 'factor_1']
num_states = [2, 3]
num_factors = 2
control_fac_idx = []
num_controls = [1, 1]

# --- Matrix Definitions ---
A = utils.obj_array(3)
A[0] = utils.norm_dist(np.ones((3, 2, 3))) # Defaulted to uniform
A[1] = utils.norm_dist(np.ones((3, 2, 3))) # Defaulted to uniform
A[2] = utils.norm_dist(np.ones((3, 2, 3))) # Defaulted to uniform
B = utils.obj_array(num_factors)
B[0] = np.eye(2)[:, :, np.newaxis] # Default for uncontrolled (identity)
B[1] = np.eye(3)[:, :, np.newaxis] # Default for uncontrolled (identity)
C = utils.obj_array_zeros(num_obs)
D = utils.obj_array(2)
D[0] = utils.norm_dist(np.ones(2)) # Default: uniform D for factor 0
D[1] = utils.norm_dist(np.ones(3)) # Default: uniform D for factor 1
E = None

# --- Agent Instantiation ---
Multifactor_PyMDP_Agent_v1 = Agent(
    A=A,
    B=B,
    C=C,
    D=D
)

agent_params_for_debug = {
    'A': A if 'A' in globals() else None,
    'B': B if 'B' in globals() else None,
    'C': C if 'C' in globals() else None,
    'D': D if 'D' in globals() else None,
    'control_fac_idx': (control_fac_idx if control_fac_idx else None) if 'control_fac_idx' in globals() else None
}


# --- Example Usage ---
if __name__ == '__main__':
    # Initialize agent (already done above)
    agent = Multifactor_PyMDP_Agent_v1
    print(f"Agent 'Multifactor_PyMDP_Agent_v1' initialized with {agent.num_factors if hasattr(agent, 'num_factors') else 'N/A'} factors and {agent.num_modalities if hasattr(agent, 'num_modalities') else 'N/A'} modalities.")
    o_current = [0, 0, 0] # Example initial observation (e.g. first outcome for each modality)
    s_current = [0, 0] # Example initial true states for simulation
    T = 5 # Number of timesteps
    A_gen_process = copy.deepcopy(A)
    B_gen_process = copy.deepcopy(B)

    for t_step in range(T):
        print(f"\n--- Timestep {t_step + 1} ---")
        if o_current is not None:
            for g_idx, o_val in enumerate(o_current):
                print(f"Observation ({obs_names[g_idx] if obs_names else f'Modality {g_idx}'}): {o_val}")
        # Infer states
        qs_current = agent.infer_states(o_current)
        if qs_current is not None:
            for f_idx, q_val in enumerate(qs_current):
                print(f"Beliefs about {state_names[f_idx] if state_names else f'Factor {f_idx}'}: {q_val}")

        # Infer policies and sample action
        q_pi_current, efe_current = agent.infer_policies()
        if hasattr(agent, 'q_pi') and agent.q_pi is not None:
            print(f"Posterior over policies (q_pi): {agent.q_pi}")
        if efe_current is not None:
            print(f"Expected Free Energy (EFE): {efe_current}")
        action_agent = agent.sample_action()
        # Map agent's action (on control factors) to full environment action vector
        action_env = np.zeros(num_factors, dtype=int)
        if control_fac_idx and action_agent is not None:
            for i, cf_idx in enumerate(control_fac_idx):
                action_env[cf_idx] = int(action_agent[i])
        # Construct action names for printing
        action_names_str_list = []
        if control_fac_idx and action_agent is not None:
            for i, cf_idx in enumerate(control_fac_idx):
                factor_action_name_list = agent.action_names.get(cf_idx, []) if hasattr(agent, 'action_names') and isinstance(agent.action_names, dict) else []
                action_idx_on_factor = int(action_agent[i])
                if factor_action_name_list and action_idx_on_factor < len(factor_action_name_list):
                    action_names_str_list.append(f"{state_names[cf_idx] if state_names else f'Factor {cf_idx}'}: {factor_action_name_list[action_idx_on_factor]} (idx {action_idx_on_factor})")
                else:
                    action_names_str_list.append(f"{state_names[cf_idx] if state_names else f'Factor {cf_idx}'}: Action idx {action_idx_on_factor}")
        print(f"Action taken: {', '.join(action_names_str_list) if action_names_str_list else 'No controllable actions or names not found'}")
        # Raw sampled action_agent: {action_agent}
        # Mapped action_env for B matrix: {action_env}

        # Update true states of the environment based on action
        s_next = np.zeros(num_factors, dtype=int)
        if s_current is not None and B_gen_process is not None:
            for f_idx in range(num_factors):
                # B_gen_process[f_idx] shape: (num_states[f_idx], num_states[f_idx], num_actions_for_this_factor_or_1)
                action_for_factor = action_env[f_idx] if f_idx in control_fac_idx else 0
                s_next[f_idx] = utils.sample(B_gen_process[f_idx][:, s_current[f_idx], action_for_factor])
        s_current = s_next.tolist()
        if s_current is not None:
            for f_idx, s_val in enumerate(s_current):
                print(f"New true state ({state_names[f_idx] if state_names else f'Factor {f_idx}'}): {s_val}")

        # Generate next observation based on new true states
        o_next = np.zeros(num_modalities, dtype=int)
        if s_current is not None and A_gen_process is not None:
            for g_idx in range(num_modalities):
                # A_gen_process[g_idx] shape: (num_obs[g_idx], num_states[0], num_states[1], ...)
                # Construct index for A matrix: (outcome_idx, s_f0, s_f1, ...)
                prob_vector = A_gen_process[g_idx][:, s_current[0], s_current[1]]
                o_next[g_idx] = utils.sample(prob_vector)
        o_current = o_next.tolist()

    print(f"\nSimulation finished after {T} timesteps.")
print('--- PyMDP Runtime Debug ---')
try:
    import pymdp
    try:
        print(f'AGENT_SCRIPT: Imported pymdp version: {pymdp.__version__}')
    except AttributeError:
        print('AGENT_SCRIPT: pymdp.__version__ attribute not found.')
    print(f'AGENT_SCRIPT: pymdp module location: {pymdp.__file__}')
    from pymdp.agent import Agent
    print(f'AGENT_SCRIPT: Imported Agent: {Agent}')
    print(f'AGENT_SCRIPT: Agent module location: {inspect.getfile(Agent)}')
    print('AGENT_SCRIPT: Checking for required variables in global scope:')
    # Check defined parameters for the main agent
    print(f"  AGENT_SCRIPT: A = {{A if 'A' in globals() else 'Not Defined'}}")
    print(f"  AGENT_SCRIPT: B = {{B if 'B' in globals() else 'Not Defined'}}")
    print(f"  AGENT_SCRIPT: C = {{C if 'C' in globals() else 'Not Defined'}}")
    print(f"  AGENT_SCRIPT: D = {{D if 'D' in globals() else 'Not Defined'}}")
    print(f"  AGENT_SCRIPT: E = {{E if 'E' in globals() else 'Not Defined'}}")
    print(f"  AGENT_SCRIPT: control_fac_idx = {{control_fac_idx if 'control_fac_idx' in globals() else 'Not Defined'}}")
    print(f'  AGENT_SCRIPT: action_names = {action_names_dict_str if action_names_dict_str != "{{}}" else "Not Defined"}')
    print(f'  AGENT_SCRIPT: qs_initial = {qs_initial_str if qs_initial_str != "None" else "Not Defined"}')
    print(f'  AGENT_SCRIPT: agent_hyperparams = {agent_hyperparams_dict_str}')
    print('AGENT_SCRIPT: Attempting to instantiate agent with defined parameters for debug...')
    # Filter out None hyperparams from agent_params_for_debug if it was originally None
    # The ** unpacking handles empty dicts correctly if agent_hyperparams_dict_str was "{}"
    debug_params_copy = {k: v for k, v in agent_params_for_debug.items() if not (isinstance(v, str) and v == 'None')}
    temp_agent = Agent(**debug_params_copy)
    print(f'AGENT_SCRIPT: Debug agent successfully instantiated: {temp_agent}')
except Exception as e_debug:
    print(f'AGENT_SCRIPT: Error during PyMDP runtime debug: {e_debug}')
    print(f"AGENT_SCRIPT: Traceback:\n{traceback.format_exc()}")
print('--- End PyMDP Runtime Debug ---')
# --- GNN Model: Multifactor_PyMDP_Agent_v1 ---\n