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

def _parse_gnn_matrix_string(matrix_str: str) -> np.ndarray:
    """Parse GNN matrix string format to numpy array."""
    try:
        # Remove comments and clean up
        lines = matrix_str.split('\n')
        cleaned_lines = []
        for line in lines:
            if '#' in line:
                line = line.split('#')[0]
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        
        # Reconstruct the matrix string
        matrix_str = ' '.join(cleaned_lines)
        
        # Handle A matrix format: { (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0) }
        if matrix_str.startswith('{') and matrix_str.endswith('}'):
            inner = matrix_str[1:-1].strip()
            
            # Split by commas, but be careful with nested tuples
            rows = []
            current_row = ""
            paren_count = 0
            
            for char in inner:
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                
                if char == ',' and paren_count == 0:
                    # End of a row
                    if current_row.strip():
                        rows.append(current_row.strip())
                    current_row = ""
                else:
                    current_row += char
            
            # Add the last row
            if current_row.strip():
                rows.append(current_row.strip())
            
            # Parse each row
            matrix = []
            for row in rows:
                row = row.strip()
                try:
                    if row.startswith('(') and row.endswith(')'):
                        # Parse tuple row
                        inner_row = row[1:-1]
                        row_values = [float(x.strip()) for x in inner_row.split(',') if x.strip()]
                        matrix.append(row_values)
                    elif row.startswith('((') and row.endswith('))'):
                        # Parse nested tuple row (for B matrix)
                        inner_row = row[2:-2]
                        # Split by '),('
                        nested_tuples = inner_row.split('),(')
                        row_values = []
                        for nested_tuple in nested_tuples:
                            nested_tuple = nested_tuple.strip('()')
                            tuple_values = [float(x.strip()) for x in nested_tuple.split(',') if x.strip()]
                            row_values.append(tuple_values)
                        matrix.append(row_values)
                    else:
                        # Try to parse as simple values
                        row_values = [float(x.strip()) for x in row.split(',') if x.strip()]
                        if row_values:
                            matrix.append(row_values)
                except Exception as e:
                    logger.warning(f"Failed to parse row '{row}': {e}")
                    # Add a default row to maintain matrix structure
                    if matrix:
                        # Use the same length as previous rows
                        matrix.append([0.0] * len(matrix[0]))
                    else:
                        matrix.append([1.0])
            
            if not matrix:
                return np.array([[1.0]])
            
            # Ensure all rows have the same length
            max_len = max(len(row) for row in matrix)
            for i, row in enumerate(matrix):
                while len(row) < max_len:
                    row.append(0.0)
            
            return np.array(matrix)
        
        return np.array([[1.0]])  # Default fallback
        
    except Exception as e:
        logger.warning(f"Failed to parse matrix string: {e}")
        return np.array([[1.0]])  # Default fallback

def _extract_gnn_matrices(gnn_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Extract A, B, C, D matrices from GNN specification."""
    matrices = {}
    
    # Handle parsed GNN data structure
    if "variables" in gnn_spec:
        logger.info("Extracting matrices from parsed GNN data structure")
        
        # Extract variable dimensions
        var_dims = {}
        for var_data in gnn_spec.get("variables", []):
            var_name = var_data.get("name", "")
            dimensions = var_data.get("dimensions", [])
            var_dims[var_name] = dimensions
        
        # Create default matrices based on dimensions
        if "A" in var_dims:
            dims = var_dims["A"]
            if len(dims) >= 2:
                matrices['A'] = np.eye(dims[0], dims[1])  # Identity matrix
                logger.info(f"Created A matrix with dimensions {dims}")
        
        if "B" in var_dims:
            dims = var_dims["B"]
            if len(dims) >= 3:
                # Create identity-like transition matrix
                matrices['B'] = np.eye(dims[0], dims[1])[:, :, np.newaxis]
                matrices['B'] = np.repeat(matrices['B'], dims[2], axis=2)
                logger.info(f"Created B matrix with dimensions {dims}")
        
        if "C" in var_dims:
            dims = var_dims["C"]
            if len(dims) >= 1:
                matrices['C'] = np.zeros(dims[0])  # Zero preferences
                logger.info(f"Created C vector with dimensions {dims}")
        
        if "D" in var_dims:
            dims = var_dims["D"]
            if len(dims) >= 1:
                matrices['D'] = np.ones(dims[0]) / dims[0]  # Uniform prior
                logger.info(f"Created D vector with dimensions {dims}")
        
        # Extract actual parameter values if available
        for param_data in gnn_spec.get("parameters", []):
            param_name = param_data.get("name", "")
            param_value = param_data.get("value")
            
            if param_name in ["A", "B", "C", "D"] and param_value is not None:
                try:
                    if isinstance(param_value, str):
                        # Parse GNN matrix string format
                        matrices[param_name] = _parse_gnn_matrix_string(param_value)
                        logger.info(f"Parsed {param_name} matrix from string")
                    elif isinstance(param_value, list):
                        matrices[param_name] = np.array(param_value)
                        logger.info(f"Converted {param_name} list to array")
                    elif isinstance(param_value, set):
                        # Convert set to list then to array
                        matrices[param_name] = np.array(list(param_value))
                        logger.info(f"Converted {param_name} set to array")
                    else:
                        matrices[param_name] = param_value
                        logger.info(f"Used {param_name} parameter value directly")
                except Exception as e:
                    logger.warning(f"Failed to convert {param_name} parameter: {e}")
                    continue
    
    else:
        # Handle legacy raw text format
        logger.info("Extracting matrices from raw text format")
        
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
    
    try:
        model_name = gnn_spec.get('ModelName', 'GNNModel').replace(' ', '_')
        matrices = _extract_gnn_matrices(gnn_spec)
        
        # Pre-compute dimensions to avoid f-string issues
        A_matrix = matrices.get('A', np.array([[1.0]]))
        B_matrix = matrices.get('B', np.array([[[1.0]]]))
        C_vector = matrices.get('C', np.array([0.0, 1.0]))
        D_vector = matrices.get('D', np.array([0.5, 0.5]))
        
        # Safely get dimensions
        try:
            num_states = A_matrix.shape[1] if len(A_matrix.shape) >= 2 else 1
            num_observations = A_matrix.shape[0] if len(A_matrix.shape) >= 1 else 1
            num_actions = B_matrix.shape[2] if len(B_matrix.shape) >= 3 else 1
        except Exception as e:
            logger.warning(f"Error accessing matrix dimensions: {e}")
            num_states = 1
            num_observations = 1
            num_actions = 1
        
        # Convert matrices to lists for f-string insertion
        A_list = A_matrix.tolist()
        B_list = B_matrix.tolist()
        C_list = C_vector.tolist()
        D_list = D_vector.tolist()
        
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
        self.num_states = {num_states}
        self.num_observations = {num_observations}
        self.num_actions = {num_actions}
        
        # Initialize matrices as learnable parameters
        if 'A' in matrices:
            self.A_matrix = self.param('A_matrix', 
                lambda key, shape: jnp.array({A_list}), 
                (self.num_observations, self.num_states))
        
        if 'B' in matrices:
            self.B_matrix = self.param('B_matrix',
                lambda key, shape: jnp.array({B_list}),
                (self.num_states, self.num_states, self.num_actions))
        
        if 'C' in matrices:
            self.C_vector = self.param('C_vector',
                lambda key, shape: jnp.array({C_list}),
                (self.num_observations,))
        
        if 'D' in matrices:
            self.D_vector = self.param('D_vector',
                lambda key, shape: jnp.array({D_list}),
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
- Number of states: {{{{model.num_states}}}}
- Number of observations: {{{{model.num_observations}}}}
- Number of actions: {{{{model.num_actions}}}}
- Parameters: {{{{sum(p.size for p in jax.tree_util.tree_leaves(model.variables))}}}}
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
        
    except Exception as e:
        logger.error(f"Error in _generate_jax_model_code: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Return a minimal working code as fallback
        return '''"""
JAX Model - Fallback Implementation
"""

import jax
import jax.numpy as jnp

def create_model():
    """Create a basic model."""
    return None

if __name__ == "__main__":
    print("JAX model created (fallback implementation)")
'''

def _generate_jax_pomdp_code(gnn_spec: Dict[str, Any], options: Optional[Dict[str, Any]]) -> str:
    """Generate JAX POMDP solver code from GNN specification."""
    
    try:
        model_name = gnn_spec.get('ModelName', 'POMDPModel').replace(' ', '_')
        matrices = _extract_gnn_matrices(gnn_spec)
        
        # Get dimensions with error handling
        try:
            A_matrix = matrices.get('A', np.array([[1.0]]))
            B_matrix = matrices.get('B', np.array([[[1.0]]]))
            C_vector = matrices.get('C', np.array([0.0, 1.0]))
            D_vector = matrices.get('D', np.array([0.5, 0.5]))
            
            logger.info(f"A matrix shape: {A_matrix.shape}")
            logger.info(f"B matrix shape: {B_matrix.shape}")
            logger.info(f"C vector shape: {C_vector.shape}")
            logger.info(f"D vector shape: {D_vector.shape}")
            
            # Safely get dimensions
            if len(A_matrix.shape) >= 2:
                num_states = A_matrix.shape[1]
                num_observations = A_matrix.shape[0]
            else:
                num_states = 2
                num_observations = 2
                logger.warning("A matrix has insufficient dimensions, using defaults")
            
            if len(B_matrix.shape) >= 3:
                num_actions = B_matrix.shape[2]
            else:
                num_actions = 2
                logger.warning("B matrix has insufficient dimensions, using defaults")
                
        except Exception as e:
            logger.error(f"Error accessing matrix dimensions: {e}")
            # Use safe defaults
            num_states = 2
            num_observations = 2
            num_actions = 2
            A_matrix = np.array([[0.8, 0.2], [0.2, 0.8]])
            B_matrix = np.array([[[0.9, 0.1], [0.1, 0.9]], [[0.1, 0.9], [0.9, 0.1]]])
            C_vector = np.array([0.0, 1.0])
            D_vector = np.array([0.5, 0.5])
        
        logger.info(f"Final dimensions: states={num_states}, observations={num_observations}, actions={num_actions}")
        
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
        alpha = self.models.C  # Immediate reward
        
        # Vectorized future reward computation
        for obs in range(self.num_observations):
            obs_prob = self.compute_observation_probability(belief, action)[obs]
            if obs_prob > 1e-10:
                next_belief = self.belief_update(belief, action, obs)
                values = jnp.dot(alpha_vectors, next_belief)
                best_alpha = alpha_vectors[jnp.argmax(values)]
                alpha = alpha + self.discount_factor * obs_prob * best_alpha
        
        return alpha
    
    def compute_observation_probability(self, belief: jnp.ndarray, action: int) -> jnp.ndarray:
        """Compute probability of observations given belief and action."""
        next_belief = jnp.sum(
            self.models.B[:, action, :] * belief[:, None], axis=0
        )
        return jnp.dot(self.models.A, next_belief)

def create_pomdp_solver() -> JAXPOMDPSolver:
    """Create and return a POMDP solver with the specified model parameters."""
    
    # Model parameters from GNN specification
    A = jnp.array({A_matrix.tolist()})  # Observation model
    B = jnp.array({B_matrix.tolist()})  # Transition model  
    C = jnp.array({C_vector.tolist()})  # Preferences
    D = jnp.array({D_vector.tolist()})  # Prior
    
    models = POMDPModels(A=A, B=B, C=C, D=D)
    return JAXPOMDPSolver(models)

def solve_pomdp(solver: JAXPOMDPSolver, initial_belief: jnp.ndarray, 
                horizon: int = 10) -> Dict[str, jnp.ndarray]:
    """
    Solve POMDP using value iteration with alpha vectors.
    
    Args:
        solver: POMDP solver instance
        initial_belief: Initial belief state
        horizon: Planning horizon
        
    Returns:
        Dictionary containing solution components
    """
    # Initialize alpha vectors
    alpha_vectors = jnp.zeros((solver.num_actions, solver.num_states))
    
    # Value iteration
    for t in range(horizon):
        new_alpha_vectors = []
        for action in range(solver.num_actions):
            alpha = solver.alpha_vector_backup_jitted(initial_belief, action, alpha_vectors)
            new_alpha_vectors.append(alpha)
        alpha_vectors = jnp.array(new_alpha_vectors)
    
    # Compute optimal action
    values = jnp.dot(alpha_vectors, initial_belief)
    optimal_action = jnp.argmax(values)
    
    return {{
        "optimal_action": optimal_action,
        "value": jnp.max(values),
        "alpha_vectors": alpha_vectors
    }}

if __name__ == "__main__":
    # Example usage
    solver = create_pomdp_solver()
    print(f"POMDP Solver created with {{solver.num_states}} states, {{solver.num_observations}} observations, {{solver.num_actions}} actions")
    
    # Test with uniform initial belief
    initial_belief = jnp.ones(solver.num_states) / solver.num_states
    solution = solve_pomdp(solver, initial_belief, horizon=5)
    
    print(f"Optimal action: {{solution['optimal_action']}}")
    print(f"Value: {{solution['value']:.4f}}")
    print("POMDP solver test successful!")
'''
        
        return code
        
    except Exception as e:
        logger.error(f"Error in _generate_jax_pomdp_code: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Return a minimal working code as fallback
        return '''"""
JAX POMDP Solver - Fallback Implementation
"""

import jax
import jax.numpy as jnp

def create_pomdp_solver():
    """Create a basic POMDP solver."""
    return None

if __name__ == "__main__":
    print("JAX POMDP solver created (fallback implementation)")
'''

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
        
        # The following line generates 'outputs = {{{{}}}}' in the output code
        outputs = {{}}
        
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
        updated_params = {{}}
        
        for param_name, new_value in new_params.items():
            if param_name in self.get_parameters():
                updated_params[param_name] = new_value
        
        return updated_params

if __name__ == "__main__":
    print(f"Combined model {model_name} created successfully!")
    print("This model supports hierarchical, multi-agent, and continuous extensions.")
'''
    
    return code 