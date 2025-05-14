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
# WARNING: No 'ObservationModalities' defined in StateSpaceBlock.
# WARNING: No 'HiddenStateFactors' defined in StateSpaceBlock.
# INFO: Finished GNN data extraction.
# WARNING: A_matrix: No observation modalities defined. 'A' will be [].
# WARNING: B_matrix: No hidden state factors defined. 'B' will be [].
# INFO: C_vector: No observation modalities defined. 'C' will be None.
# ERROR: D_vector: No hidden state factors defined. 'D' will be [].
# INFO: AgentHyperparameters: Extracting learning and algorithm parameter dicts.
# --- End of GNN to PyMDP Conversion Summary ---


# --- GNN Model: pymdp_agent_model ---

obs_names = []
num_obs = []
num_modalities = 0
state_names = []
num_states = []
num_factors = 0
control_fac_idx = []
num_controls = []

# --- Matrix Definitions ---
A = []
B = []
C = None
D = []
E = None

# --- Agent Instantiation ---
pymdp_agent_model = Agent(
    A=A,
    B=B,
    C=C,
    control_fac_idx=control_fac_idx
)


# --- Example Usage ---
if __name__ == '__main__':
    # Initialize agent (already done above)
    agent = pymdp_agent_model
    print(f"Agent 'pymdp_agent_model' initialized with 0 factors and 0 modalities.")
    o_current = None # Example initial observation (e.g. first outcome for each modality)
    s_current = None # Example initial true states for simulation
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
                prob_vector = A_gen_process[g_idx][:, ]
                o_next[g_idx] = utils.sample(prob_vector)
        o_current = o_next.tolist()

    print(f"\nSimulation finished after {T} timesteps.")
print('--- PyMDP Runtime Debug ---')
try:
    import pymdp
    print(f'AGENT_SCRIPT: Imported pymdp version: {pymdp.__version__}')
    print(f'AGENT_SCRIPT: pymdp module location: {pymdp.__file__}')
    from pymdp.agent import Agent
    print(f'AGENT_SCRIPT: Imported Agent: {Agent}')
    print(f'AGENT_SCRIPT: Agent module location: {inspect.getfile(Agent)}')
    # Check if required variables are in global scope
    required_vars = ['A', 'B', 'C', 'D', 'num_obs', 'num_states', 'num_controls', 'control_factor_idx', 'agent_params']
    print('AGENT_SCRIPT: Checking for required variables in global scope:')
    for var_name in required_vars:
        if var_name in globals():
            print(f'  AGENT_SCRIPT: {var_name} is defined. Value (first 100 chars): {str(globals()[var_name])[:100]}')
        else:
            print(f'  AGENT_SCRIPT: {var_name} is NOT defined.')
    # Instantiate agent to catch initialization errors
    print('AGENT_SCRIPT: Attempting to instantiate agent with defined parameters...')
    temp_agent = Agent(**agent_params)
    print(f'AGENT_SCRIPT: Agent successfully instantiated: {temp_agent}')
except Exception as e_debug:
    print(f'AGENT_SCRIPT: Error during PyMDP runtime debug: {e_debug}')
    print(f'AGENT_SCRIPT: Traceback:\n{traceback.format_exc()}')
print('--- End PyMDP Runtime Debug ---')
# --- GNN Model: pymdp_agent_model ---
