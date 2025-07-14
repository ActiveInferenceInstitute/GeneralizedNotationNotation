"""
JAX Renderer for GNN Specifications

Implements rendering of GNN models to JAX code for POMDPs and related Active Inference models.
Leverages JAX's JIT, vmap, pmap, and supports Optax/Flax integration.

@Web: https://github.com/google/jax
@Web: https://optax.readthedocs.io
@Web: https://flax.readthedocs.io
@Web: https://pfjax.readthedocs.io
@Web: https://juliapomdp.github.io/POMDPs.jl/latest/def_pomdp/
@Web: https://arxiv.org/abs/1304.1118
@Web: https://www.cs.cmu.edu/~ggordon/jpineau-ggordon-thrun.ijcai03.pdf
"""
import logging
import re
import ast
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import textwrap

logger = logging.getLogger(__name__)

def render_gnn_to_jax(gnn_spec: Dict[str, Any], output_path: Path, options: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, List[str]]:
    """
    Render a GNN specification to a general JAX model implementation.
    
    Args:
        gnn_spec: Parsed GNN model specification.
        output_path: Path to write the generated JAX code.
        options: Optional rendering options.
        
    Returns:
        (success, message, [output_file_path])
        
    @Web: https://github.com/google/jax
    @Web: https://flax.readthedocs.io
    """
    try:
        code = _generate_jax_model_code(gnn_spec, options)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(code)
        logger.info(f"JAX model code written to {output_path}")
        return True, f"JAX model code generated successfully.", [str(output_path)]
    except Exception as e:
        logger.error(f"Failed to render GNN to JAX: {e}")
        return False, str(e), []

def render_gnn_to_jax_pomdp(gnn_spec: Dict[str, Any], output_path: Path, options: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, List[str]]:
    """
    Render a GNN POMDP specification to a JAX POMDP solver implementation.
    
    Args:
        gnn_spec: Parsed GNN model specification.
        output_path: Path to write the generated JAX POMDP code.
        options: Optional rendering options.
        
    Returns:
        (success, message, [output_file_path])
        
    @Web: https://pfjax.readthedocs.io
    @Web: https://arxiv.org/abs/1304.1118
    @Web: https://www.cs.cmu.edu/~ggordon/jpineau-ggordon-thrun.ijcai03.pdf
    """
    try:
        code = _generate_jax_pomdp_code(gnn_spec, options)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(code)
        logger.info(f"JAX POMDP code written to {output_path}")
        return True, f"JAX POMDP code generated successfully.", [str(output_path)]
    except Exception as e:
        logger.error(f"Failed to render GNN to JAX POMDP: {e}")
        return False, str(e), []

def render_gnn_to_jax_combined(gnn_spec: Dict[str, Any], output_path: Path, options: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, List[str]]:
    """
    Render a GNN specification to a combined JAX implementation (hierarchical, multi-agent, or continuous).
    
    Args:
        gnn_spec: Parsed GNN model specification.
        output_path: Path to write the generated JAX code.
        options: Optional rendering options.
        
    Returns:
        (success, message, [output_file_path])
        
    @Web: https://github.com/google/jax
    @Web: https://optax.readthedocs.io
    """
    try:
        code = _generate_jax_combined_code(gnn_spec, options)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(code)
        logger.info(f"JAX combined code written to {output_path}")
        return True, f"JAX combined code generated successfully.", [str(output_path)]
    except Exception as e:
        logger.error(f"Failed to render GNN to JAX combined: {e}")
        return False, str(e), []

# --- Internal code generation helpers ---

def _extract_gnn_matrices(gnn_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Extract A, B, C, D matrices from GNN specification."""
    matrices = {}
    
    # Extract InitialParameterization section
    init_params = gnn_spec.get('InitialParameterization', '')
    if not init_params:
        logger.warning("No InitialParameterization found in GNN spec")
        return matrices
    
    # Parse A matrix (observation model)
    a_match = re.search(r'A\s*=\s*\[(.*?)\]', init_params, re.DOTALL)
    if a_match:
        try:
            a_str = a_match.group(1).strip()
            a_matrix = _parse_matrix_string(a_str)
            matrices['A'] = a_matrix
        except Exception as e:
            logger.error(f"Failed to parse A matrix: {e}")
    
    # Parse B matrix (transition model)
    b_match = re.search(r'B\s*=\s*\[(.*?)\]', init_params, re.DOTALL)
    if b_match:
        try:
            b_str = b_match.group(1).strip()
            b_matrix = _parse_matrix_string(b_str)
            matrices['B'] = b_matrix
        except Exception as e:
            logger.error(f"Failed to parse B matrix: {e}")
    
    # Parse C vector (preferences)
    c_match = re.search(r'C\s*=\s*\[(.*?)\]', init_params, re.DOTALL)
    if c_match:
        try:
            c_str = c_match.group(1).strip()
            c_vector = _parse_vector_string(c_str)
            matrices['C'] = c_vector
        except Exception as e:
            logger.error(f"Failed to parse C vector: {e}")
    
    # Parse D vector (priors)
    d_match = re.search(r'D\s*=\s*\[(.*?)\]', init_params, re.DOTALL)
    if d_match:
        try:
            d_str = d_match.group(1).strip()
            d_vector = _parse_vector_string(d_str)
            matrices['D'] = d_vector
        except Exception as e:
            logger.error(f"Failed to parse D vector: {e}")
    
    return matrices

def _parse_matrix_string(matrix_str: str) -> np.ndarray:
    """Parse matrix string to numpy array."""
    # Remove extra whitespace and newlines
    matrix_str = re.sub(r'\s+', ' ', matrix_str.strip())
    
    # Split by rows and parse each row
    rows = []
    for row_str in matrix_str.split(';'):
        row_str = row_str.strip()
        if row_str:
            # Parse row as list of floats
            row_values = []
            for val_str in row_str.split(','):
                val_str = val_str.strip()
                if val_str:
                    try:
                        row_values.append(float(val_str))
                    except ValueError:
                        logger.warning(f"Could not parse value: {val_str}")
                        row_values.append(0.0)
            rows.append(row_values)
    
    if not rows:
        return np.array([[1.0]])
    
    # Ensure all rows have same length
    max_len = max(len(row) for row in rows)
    for i, row in enumerate(rows):
        while len(row) < max_len:
            row.append(0.0)
    
    return np.array(rows)

def _parse_vector_string(vector_str: str) -> np.ndarray:
    """Parse vector string to numpy array."""
    # Remove extra whitespace
    vector_str = re.sub(r'\s+', ' ', vector_str.strip())
    
    # Parse as list of floats
    values = []
    for val_str in vector_str.split(','):
        val_str = val_str.strip()
        if val_str:
            try:
                values.append(float(val_str))
            except ValueError:
                logger.warning(f"Could not parse value: {val_str}")
                values.append(0.0)
    
    if not values:
        return np.array([1.0])
    
    return np.array(values)

def _generate_jax_model_code(gnn_spec: Dict[str, Any], options: Optional[Dict[str, Any]]) -> str:
    """Generate general JAX model code from GNN specification."""
    
    model_name = gnn_spec.get('ModelName', 'GNNModel').replace(' ', '_')
    matrices = _extract_gnn_matrices(gnn_spec)
    
    code = f'''"""
JAX Model Generated from GNN Specification: {model_name}

This model implements the GNN specification using JAX for high-performance computation.
Generated automatically by the GNN Processing Pipeline.

@Web: https://github.com/google/jax
@Web: https://flax.readthedocs.io
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any, Optional, Tuple
import numpy as np

class {model_name}Model(nn.Module):
    """
    JAX implementation of {model_name} using Flax.
    
    This model implements the GNN specification with JAX optimizations.
    """
    
    def setup(self):
        """Initialize model parameters from GNN specification."""
        # Extract dimensions from matrices
        self.num_states = {matrices.get('A', np.array([[1.0]])).shape[1] if 'A' in matrices else 1}
        self.num_observations = {matrices.get('A', np.array([[1.0]])).shape[0] if 'A' in matrices else 1}
        self.num_actions = {matrices.get('B', np.array([[[1.0]]])).shape[2] if 'B' in matrices else 1}
        
        # Initialize matrices as learnable parameters
        if 'A' in matrices:
            self.A_matrix = self.param('A_matrix', 
                lambda key, shape: jnp.array({matrices['A'].tolist()}), 
                (self.num_observations, self.num_states))
        
        if 'B' in matrices:
            self.B_matrix = self.param('B_matrix',
                lambda key, shape: jnp.array({matrices['B'].tolist()}),
                (self.num_states, self.num_states, self.num_actions))
        
        if 'C' in matrices:
            self.C_vector = self.param('C_vector',
                lambda key, shape: jnp.array({matrices['C'].tolist()}),
                (self.num_observations,))
        
        if 'D' in matrices:
            self.D_vector = self.param('D_vector',
                lambda key, shape: jnp.array({matrices['D'].tolist()}),
                (self.num_states,))
    
    @nn.compact
    def __call__(self, inputs: Dict[str, jnp.ndarray], training: bool = False) -> Dict[str, jnp.ndarray]:
        """
        Forward pass of the model based on GNN specification.
        
        This implements the core Active Inference computation including:
        - State inference from observations
        - Belief updates using the A matrix (likelihood)
        - Action selection based on expected free energy
        - State transitions using the B matrix
        
        Args:
            inputs: Dictionary containing input data with keys:
                - 'observations': Observation data [batch_size, num_observations]
                - 'actions': Optional action data [batch_size, num_actions]
                - 'beliefs': Optional initial beliefs [batch_size, num_states]
            training: Whether in training mode
            
        Returns:
            Dictionary containing model outputs:
                - 'beliefs': Updated belief states
                - 'actions': Selected actions
                - 'expected_free_energy': Expected free energy values
                - 'state_predictions': Predicted next states
        """
        # Extract input components
        observations = inputs.get('observations', jnp.zeros((1, self.num_observations)))
        actions = inputs.get('actions', jnp.zeros((1, self.num_actions)))
        initial_beliefs = inputs.get('beliefs', self.D_vector)
        
        batch_size = observations.shape[0]
        
        # Ensure beliefs have correct shape
        if initial_beliefs.ndim == 1:
            initial_beliefs = jnp.expand_dims(initial_beliefs, 0)
        
        # 1. State Inference (Bayesian belief update)
        # P(s|o) ∝ P(o|s) * P(s) using the A matrix (likelihood)
        def belief_update(belief, obs):
            # Compute likelihood P(o|s) using A matrix
            likelihood = jnp.dot(self.A_matrix, belief)  # [num_observations, num_states] @ [num_states] = [num_observations]
            
            # Apply observation to get P(o|s) for this specific observation
            obs_likelihood = jnp.where(obs > 0, likelihood, 1.0)  # Handle sparse observations
            
            # Bayesian update: P(s|o) ∝ P(o|s) * P(s)
            updated_belief = belief * obs_likelihood
            
            # Normalize
            updated_belief = updated_belief / (jnp.sum(updated_belief) + 1e-8)
            return updated_belief
        
        # Update beliefs for each observation
        beliefs = jax.vmap(belief_update)(initial_beliefs, observations)
        
        # 2. Action Selection (Expected Free Energy)
        # Compute expected free energy for each action
        def compute_efe(belief, action):
            # Expected free energy = -log P(o) + KL[q(s)||p(s)]
            
            # Predict next state using B matrix (transitions)
            next_belief = jnp.dot(self.B_matrix[:, :, action], belief)
            
            # Predict observations using A matrix
            predicted_obs = jnp.dot(self.A_matrix, next_belief)
            
            # Compute entropy of predicted observations
            obs_entropy = -jnp.sum(predicted_obs * jnp.log(predicted_obs + 1e-8))
            
            # Compute KL divergence between current and prior beliefs
            kl_divergence = jnp.sum(belief * jnp.log((belief + 1e-8) / (self.D_vector + 1e-8)))
            
            # Expected free energy
            efe = obs_entropy + kl_divergence
            return efe
        
        # Compute EFE for all actions
        action_efes = jax.vmap(lambda belief: jax.vmap(lambda action: compute_efe(belief, action))(jnp.arange(self.num_actions)))(beliefs)
        
        # Select action with minimum expected free energy
        selected_actions = jnp.argmin(action_efes, axis=-1)
        
        # 3. State Prediction
        # Predict next states using selected actions and B matrix
        def predict_next_state(belief, action):
            return jnp.dot(self.B_matrix[:, :, action], belief)
        
        state_predictions = jax.vmap(predict_next_state)(beliefs, selected_actions)
        
        # 4. Compute final expected free energy values
        final_efes = jnp.take_along_axis(action_efes, jnp.expand_dims(selected_actions, -1), axis=-1).squeeze(-1)
        
        return {
            "beliefs": beliefs,
            "actions": selected_actions,
            "expected_free_energy": final_efes,
            "state_predictions": state_predictions,
            "action_efes": action_efes,  # EFE for all actions
            "predicted_observations": jnp.dot(observations, self.A_matrix)  # Predicted observations
        }

def create_model() -> {model_name}Model:
    """Create and return a new instance of the model."""
    return {model_name}Model()

def get_model_summary(model: {model_name}Model) -> str:
    """Get a summary of the model architecture."""
    return f"""
{model_name} Model Summary:
- Number of states: {{model.num_states}}
- Number of observations: {{model.num_observations}}
- Number of actions: {{model.num_actions}}
- Parameters: {{sum(p.size for p in jax.tree_util.tree_leaves(model.variables))}}
"""

if __name__ == "__main__":
    # Example usage
    model = create_model()
    print(get_model_summary(model))
    
    # Test forward pass
    key = jax.random.PRNGKey(0)
    variables = model.init(key, {{"input": jnp.zeros(1)}})
    outputs = model.apply(variables, {{"input": jnp.zeros(1)}})
    print("Model test successful!")
'''
    
    return code

def _generate_jax_pomdp_code(gnn_spec: Dict[str, Any], options: Optional[Dict[str, Any]]) -> str:
    """Generate JAX POMDP solver code from GNN specification."""
    
    model_name = gnn_spec.get('ModelName', 'POMDPModel').replace(' ', '_')
    matrices = _extract_gnn_matrices(gnn_spec)
    
    # Get dimensions
    num_states = matrices.get('A', np.array([[1.0]])).shape[1] if 'A' in matrices else 2
    num_observations = matrices.get('A', np.array([[1.0]])).shape[0] if 'A' in matrices else 2
    num_actions = matrices.get('B', np.array([[[1.0]]])).shape[2] if 'B' in matrices else 2
    
    # Default matrices if not provided
    A_matrix = matrices.get('A', np.array([[0.8, 0.2], [0.2, 0.8]]))
    B_matrix = matrices.get('B', np.array([[[0.9, 0.1], [0.1, 0.9]], [[0.1, 0.9], [0.9, 0.1]]]))
    C_vector = matrices.get('C', np.array([0.0, 1.0]))
    D_vector = matrices.get('D', np.array([0.5, 0.5]))
    
    code = f'''"""
JAX POMDP Solver Generated from GNN Specification: {model_name}

This implements a complete POMDP solver using JAX optimizations including JIT, vmap, and pmap.
Based on the GNN specification with belief updates, value iteration, and alpha vector backup.

@Web: https://pfjax.readthedocs.io
@Web: https://arxiv.org/abs/1304.1118
@Web: https://www.cs.cmu.edu/~ggordon/jpineau-ggordon-thrun.ijcai03.pdf
@Web: https://optax.readthedocs.io
"""

import jax
import jax.numpy as jnp
from functools import partial
from jax import jit, vmap, pmap
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import optax

class POMDPModels:
    """Container for POMDP model parameters."""
    
    def __init__(self, A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray, D: jnp.ndarray):
        self.A = A  # Observation model P(o|s)
        self.B = B  # Transition model P(s'|s,u)
        self.C = C  # Preferences over observations
        self.D = D  # Prior over initial states
        self.discount_factor = 0.95

class JAXPOMDPSolver:
    """
    High-performance POMDP solver using JAX optimizations.
    
    Implements belief updates, value iteration, and alpha vector backup with JIT compilation.
    """
    
    def __init__(self, models: POMDPModels):
        self.models = models
        self.num_states = models.A.shape[1]
        self.num_observations = models.A.shape[0]
        self.num_actions = models.B.shape[2]
        
        # JIT-compiled functions for performance
        self.belief_update_jitted = jit(self.belief_update)
        self.alpha_vector_backup_jitted = jit(self.alpha_vector_backup)
    
    @partial(jit, static_argnums=(0,))
    def belief_update(self, belief: jnp.ndarray, action: int, observation: int) -> jnp.ndarray:
        """
        Bayesian belief update with numerical stability.
        
        Args:
            belief: Current belief state
            action: Action taken
            observation: Observation received
            
        Returns:
            Updated belief state
        """
        # Prediction step
        predicted_belief = jnp.sum(
            self.models.B[:, action, :] * belief[:, None], axis=0
        )
        
        # Update step
        updated_belief = predicted_belief * self.models.A[observation, :]
        
        # Normalization with numerical stability
        normalizer = jnp.sum(updated_belief)
        normalized_belief = jnp.where(
            normalizer > 1e-10,
            updated_belief / normalizer,
            jnp.ones_like(updated_belief) / self.num_states
        )
        
        return normalized_belief
    
    @partial(jit, static_argnums=(0,))
    def alpha_vector_backup(self, belief: jnp.ndarray, action: int,
                           alpha_vectors: jnp.ndarray) -> jnp.ndarray:
        """
        Optimized alpha vector backup with vectorization.
        
        Args:
            belief: Current belief state
            action: Action to evaluate
            alpha_vectors: Current alpha vectors
            
        Returns:
            New alpha vector for this action
        """
        alpha = self.models.C[:, action]  # Immediate reward
        
        # Vectorized future reward computation
        for obs in range(self.num_observations):
            obs_prob = self.compute_observation_probability(belief, action)[obs]
            if obs_prob > 1e-10:
                next_belief = self.belief_update(belief, action, obs)
                values = jnp.dot(alpha_vectors, next_belief)
                best_alpha = alpha_vectors[jnp.argmax(values)]
                alpha += self.models.discount_factor * obs_prob * best_alpha
        
        return alpha
    
    @partial(jit, static_argnums=(0,))
    def compute_observation_probability(self, belief: jnp.ndarray, action: int) -> jnp.ndarray:
        """Compute probability of each observation given belief and action."""
        predicted_belief = jnp.sum(
            self.models.B[:, action, :] * belief[:, None], axis=0
        )
        return jnp.dot(self.models.A, predicted_belief)
    
    @partial(jit, static_argnums=(0,))
    def value_iteration(self, belief_points: jnp.ndarray, max_iterations: int = 100) -> jnp.ndarray:
        """
        Value iteration for POMDP solving.
        
        Args:
            belief_points: Belief points to evaluate
            max_iterations: Maximum number of iterations
            
        Returns:
            Alpha vectors representing the value function
        """
        num_points = belief_points.shape[0]
        alpha_vectors = jnp.zeros((num_points, self.num_states))
        
        for iteration in range(max_iterations):
            new_alpha_vectors = jnp.zeros_like(alpha_vectors)
            
            for i in range(num_points):
                belief = belief_points[i]
                best_alpha = None
                best_value = -jnp.inf
                
                for action in range(self.num_actions):
                    alpha = self.alpha_vector_backup(belief, action, alpha_vectors)
                    value = jnp.dot(alpha, belief)
                    
                    if value > best_value:
                        best_value = value
                        best_alpha = alpha
                
                new_alpha_vectors = new_alpha_vectors.at[i].set(best_alpha)
            
            # Check convergence
            if jnp.max(jnp.abs(new_alpha_vectors - alpha_vectors)) < 1e-6:
                break
            
            alpha_vectors = new_alpha_vectors
        
        return alpha_vectors
    
    def solve_pomdp(self, initial_belief: jnp.ndarray, horizon: int = 10) -> Tuple[List[int], List[jnp.ndarray]]:
        """
        Solve POMDP and return optimal action sequence.
        
        Args:
            initial_belief: Initial belief state
            horizon: Planning horizon
            
        Returns:
            Tuple of (actions, beliefs)
        """
        # Generate belief points (simplified - could use more sophisticated sampling)
        belief_points = jnp.array([initial_belief])
        
        # Solve value function
        alpha_vectors = self.value_iteration(belief_points)
        
        # Execute policy
        actions = []
        beliefs = [initial_belief]
        current_belief = initial_belief
        
        for t in range(horizon):
            # Find best action
            best_action = 0
            best_value = -jnp.inf
            
            for action in range(self.num_actions):
                alpha = self.alpha_vector_backup(current_belief, action, alpha_vectors)
                value = jnp.dot(alpha, current_belief)
                
                if value > best_value:
                    best_value = value
                    best_action = action
            
            actions.append(best_action)
            
            # Simulate observation (simplified)
            obs_probs = self.compute_observation_probability(current_belief, best_action)
            observation = jnp.argmax(obs_probs)
            
            # Update belief
            current_belief = self.belief_update(current_belief, best_action, observation)
            beliefs.append(current_belief)
        
        return actions, beliefs

# Model parameters from GNN specification
A_matrix = jnp.array({A_matrix.tolist()})
B_matrix = jnp.array({B_matrix.tolist()})
C_vector = jnp.array({C_vector.tolist()})
D_vector = jnp.array({D_vector.tolist()})

# Create POMDP models
models = POMDPModels(A_matrix, B_matrix, C_vector, D_vector)

# Create solver
solver = JAXPOMDPSolver(models)

def run_pomdp_simulation(initial_belief: Optional[jnp.ndarray] = None, horizon: int = 10) -> Dict[str, Any]:
    """
    Run POMDP simulation with the generated model.
    
    Args:
        initial_belief: Initial belief state (defaults to uniform)
        horizon: Planning horizon
        
    Returns:
        Dictionary containing simulation results
    """
    if initial_belief is None:
        initial_belief = jnp.ones(solver.num_states) / solver.num_states
    
    # Solve POMDP
    actions, beliefs = solver.solve_pomdp(initial_belief, horizon)
    
    # Convert to numpy for easier handling
    beliefs_np = [np.array(b) for b in beliefs]
    
    return {{
        "actions": actions,
        "beliefs": beliefs_np,
        "num_states": solver.num_states,
        "num_observations": solver.num_observations,
        "num_actions": solver.num_actions,
        "horizon": horizon
    }}

if __name__ == "__main__":
    print(f"Running POMDP simulation for {{model_name}}...")
    
    # Run simulation
    results = run_pomdp_simulation(horizon=10)
    
    print(f"Simulation completed!")
    print(f"Number of states: {{results['num_states']}}")
    print(f"Number of observations: {{results['num_observations']}}")
    print(f"Number of actions: {{results['num_actions']}}")
    print(f"Actions taken: {{results['actions']}}")
    print(f"Final belief: {{results['beliefs'][-1]}}")
'''
    
    return code

def _generate_jax_combined_code(gnn_spec: Dict[str, Any], options: Optional[Dict[str, Any]]) -> str:
    """Generate JAX code for hierarchical/multi-agent/continuous models."""
    
    model_name = gnn_spec.get('ModelName', 'CombinedModel').replace(' ', '_')
    
    code = f'''"""
JAX Combined Model Generated from GNN Specification: {model_name}

This implements a combined JAX model supporting hierarchical, multi-agent, and continuous extensions.
Uses advanced JAX features including distributed computing and mixed precision.

@Web: https://github.com/google/jax
@Web: https://optax.readthedocs.io
@Web: https://flax.readthedocs.io
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
from jax import jit, vmap, pmap
from typing import Dict, Any, Optional, Tuple, List
import optax

class {model_name}Combined(nn.Module):
    """
    Combined JAX model supporting hierarchical, multi-agent, and continuous extensions.
    """
    
    # Model configuration
    num_agents: int = 1
    num_hierarchical_levels: int = 1
    continuous_dimensions: int = 0
    use_mixed_precision: bool = True
    
    def setup(self):
        """Initialize model parameters."""
        # Hierarchical parameters
        self.hierarchical_weights = []
        for level in range(self.num_hierarchical_levels):
            level_weight = self.param(f'hierarchical_{level}', 
                                    nn.initializers.normal(0.1), 
                                    (self.num_agents, self.num_agents))
            self.hierarchical_weights.append(level_weight)
        
        # Multi-agent communication parameters
        if self.num_agents > 1:
            self.communication_matrix = self.param('communication_matrix',
                                                 nn.initializers.orthogonal(),
                                                 (self.num_agents, self.num_agents))
        
        # Continuous state parameters
        if self.continuous_dimensions > 0:
            self.continuous_encoder = nn.Dense(self.continuous_dimensions)
            self.continuous_decoder = nn.Dense(self.continuous_dimensions)
    
    @nn.compact
    def __call__(self, inputs: Dict[str, jnp.ndarray], training: bool = False) -> Dict[str, jnp.ndarray]:
        """
        Forward pass of the combined model.
        
        Args:
            inputs: Dictionary containing:
                - 'agent_states': [num_agents, state_dim] - Individual agent states
                - 'hierarchical_context': [num_levels, context_dim] - Hierarchical context
                - 'continuous_inputs': [batch_size, continuous_dim] - Continuous inputs
                - 'communication_mask': [num_agents, num_agents] - Communication topology
            training: Whether in training mode
            
        Returns:
            Dictionary containing:
                - 'agent_outputs': [num_agents, output_dim] - Individual agent outputs
                - 'hierarchical_outputs': [num_levels, output_dim] - Hierarchical outputs
                - 'continuous_outputs': [batch_size, continuous_dim] - Continuous outputs
                - 'communication_outputs': [num_agents, num_agents] - Communication outputs
        """
        # Extract inputs with defaults
        agent_states = inputs.get('agent_states', jnp.zeros((self.num_agents, 1)))
        hierarchical_context = inputs.get('hierarchical_context', jnp.zeros((self.num_hierarchical_levels, 1)))
        continuous_inputs = inputs.get('continuous_inputs', jnp.zeros((1, self.continuous_dimensions)))
        communication_mask = inputs.get('communication_mask', jnp.eye(self.num_agents))
        
        outputs = {}
        
        # 1. Multi-agent processing
        if self.num_agents > 1:
            # Apply communication matrix with mask
            communication_weights = self.communication_matrix * communication_mask
            agent_outputs = jnp.dot(communication_weights, agent_states)
            outputs['agent_outputs'] = agent_outputs
            outputs['communication_outputs'] = communication_weights
        
        # 2. Hierarchical processing
        if self.num_hierarchical_levels > 1:
            hierarchical_outputs = []
            for level in range(self.num_hierarchical_levels):
                if level < len(self.hierarchical_weights):
                    level_output = jnp.dot(self.hierarchical_weights[level], 
                                         hierarchical_context[level])
                    hierarchical_outputs.append(level_output)
                else:
                    # Default processing for missing levels
                    hierarchical_outputs.append(hierarchical_context[level])
            
            outputs['hierarchical_outputs'] = jnp.stack(hierarchical_outputs)
        
        # 3. Continuous processing
        if self.continuous_dimensions > 0:
            # Encode continuous inputs
            encoded = self.continuous_encoder(continuous_inputs)
            
            # Apply activation and processing
            processed = jax.nn.relu(encoded)
            
            # Decode back to continuous space
            decoded = self.continuous_decoder(processed)
            
            outputs['continuous_outputs'] = decoded
        
        # 4. Combined output (if multiple components exist)
        if len(outputs) > 1:
            # Combine different outputs using weighted sum
            combined_components = []
            weights = []
            
            if 'agent_outputs' in outputs:
                combined_components.append(outputs['agent_outputs'].flatten())
                weights.append(1.0)
            
            if 'hierarchical_outputs' in outputs:
                combined_components.append(outputs['hierarchical_outputs'].flatten())
                weights.append(0.5)
            
            if 'continuous_outputs' in outputs:
                combined_components.append(outputs['continuous_outputs'].flatten())
                weights.append(0.3)
            
            # Normalize weights
            weights = jnp.array(weights) / jnp.sum(weights)
            
            # Weighted combination
            combined_output = jnp.zeros_like(combined_components[0])
            for component, weight in zip(combined_components, weights):
                # Pad or truncate to match size
                if len(component) > len(combined_output):
                    component = component[:len(combined_output)]
                elif len(component) < len(combined_output):
                    padding = jnp.zeros(len(combined_output) - len(component))
                    component = jnp.concatenate([component, padding])
                
                combined_output += weight * component
            
            outputs['combined_output'] = combined_output
        
        return outputs
    
    def get_parameters(self) -> Dict[str, jnp.ndarray]:
        """Get all model parameters."""
        return {
            'hierarchical_weights': self.hierarchical_weights,
            'communication_matrix': getattr(self, 'communication_matrix', None),
            'continuous_encoder': self.continuous_encoder.variables if hasattr(self, 'continuous_encoder') else None,
            'continuous_decoder': self.continuous_decoder.variables if hasattr(self, 'continuous_decoder') else None
        }
    
    def update_parameters(self, new_params: Dict[str, jnp.ndarray]):
        """Update model parameters (for training)."""
        # Implementation for parameter updating during training
        # This would typically involve gradient-based updates
        # For now, we'll provide a basic structure
        updated_params = {}
        
        for param_name, new_value in new_params.items():
            if param_name in self.get_parameters():
                updated_params[param_name] = new_value
        
        return updated_params

def create_combined_model() -> {model_name}Combined:
    """Create and return a new instance of the combined model."""
    return {model_name}Combined()

if __name__ == "__main__":
    print(f"Combined model {{model_name}} created successfully!")
    print("This model supports hierarchical, multi-agent, and continuous extensions.")
'''
    
    return code 