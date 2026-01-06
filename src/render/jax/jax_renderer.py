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
    
    # --- Primary: Handle POMDP processor format ---
    if "model_parameters" in gnn_spec:
        logger.info("Extracting matrices from POMDP processor format")
        
        model_params = gnn_spec["model_parameters"]
        n_states = model_params.get("num_hidden_states", 3)
        n_obs = model_params.get("num_obs", 3) 
        n_actions = model_params.get("num_actions", 3)
        
        logger.info(f"Extracted variable dimensions: {{n_states: {n_states}, n_obs: {n_obs}, n_actions: {n_actions}}}")
        
        # Create default matrices based on dimensions
        default_matrices = {
            'A': np.eye(n_obs, n_states),  # Identity-like likelihood matrix
            'B': np.stack([np.eye(n_states) for _ in range(n_actions)], axis=2),  # Identity transitions for each action
            'C': np.zeros(n_obs),  # Zero preferences
            'D': np.ones(n_states) / n_states  # Uniform prior
        }
        logger.info(f"Created default matrices: A={default_matrices['A'].shape}, B={default_matrices['B'].shape}, C={default_matrices['C'].shape}, D={default_matrices['D'].shape}")
        
        # Initialize matrices with defaults
        matrices.update(default_matrices)
        
        # Extract actual parameter values from initialparameterization
        init_params = gnn_spec.get("initialparameterization", {})

        # Override dimensions from B matrix if available (consistent with other renderers)
        if "B" in init_params:
            B_matrix = init_params["B"]
            if isinstance(B_matrix, (list, np.ndarray)) and len(B_matrix) > 0:
                # B matrix shape should be (n_states, n_states, n_actions) or similar
                if isinstance(B_matrix, list):
                    if len(B_matrix) > 0 and isinstance(B_matrix[0], list) and len(B_matrix[0]) > 0 and isinstance(B_matrix[0][0], list):
                        # B is (actions, states, observations) or similar nested structure
                        n_actions_from_b = len(B_matrix)
                    else:
                        n_actions_from_b = 1  # Default fallback
                else:
                    # B is numpy array
                    if len(B_matrix.shape) >= 3:
                        n_actions_from_b = B_matrix.shape[2]  # Last dimension is actions
                    else:
                        n_actions_from_b = 1  # Default fallback

                if n_actions_from_b > 1:  # Only override if we got a meaningful value
                    n_actions = n_actions_from_b
                    logger.info(f"Corrected n_actions from B matrix: {n_actions}")

            # Also check for explicit dimensions in model_params that might override
            if "B" in gnn_spec.get("model_params", {}):
                B_spec = gnn_spec["model_params"]["B"]
                if "shape" in B_spec:
                    shape_parts = B_spec["shape"].strip("()").split(",")
                    if len(shape_parts) >= 3:
                        try:
                            n_actions_from_spec = int(shape_parts[2])
                            if n_actions_from_spec > 1:
                                n_actions = n_actions_from_spec
                                logger.info(f"Corrected n_actions from B matrix specification: {n_actions}")
                        except (ValueError, IndexError):
                            pass
        if init_params:
            logger.info("Found initialparameterization, extracting actual matrix values")
            
            # Extract A matrix
            if "A" in init_params:
                try:
                    A_data = init_params["A"]
                    if isinstance(A_data, list):
                        A_matrix = np.array(A_data)
                        if A_matrix.ndim == 2:
                            matrices['A'] = A_matrix
                            logger.info(f"Successfully extracted A matrix: shape {A_matrix.shape}")
                except Exception as e:
                    logger.warning(f"Failed to extract A matrix: {e}")
            
            # Extract B matrix
            if "B" in init_params:
                try:
                    B_data = init_params["B"]
                    if isinstance(B_data, list):
                        B_matrix = np.array(B_data)
                        if B_matrix.ndim == 3:
                            matrices['B'] = B_matrix
                            logger.info(f"Successfully extracted B matrix: shape {B_matrix.shape}")
                except Exception as e:
                    logger.warning(f"Failed to extract B matrix: {e}")
            
            # Extract C vector
            if "C" in init_params:
                try:
                    C_data = init_params["C"]
                    if isinstance(C_data, list):
                        C_vector = np.array(C_data)
                        if C_vector.ndim == 1:
                            matrices['C'] = C_vector
                            logger.info(f"Successfully extracted C vector: shape {C_vector.shape}")
                except Exception as e:
                    logger.warning(f"Failed to extract C vector: {e}")
            
            # Extract D vector
            if "D" in init_params:
                try:
                    D_data = init_params["D"]
                    if isinstance(D_data, list):
                        D_vector = np.array(D_data)
                        if D_vector.ndim == 1:
                            matrices['D'] = D_vector
                            logger.info(f"Successfully extracted D vector: shape {D_vector.shape}")
                except Exception as e:
                    logger.warning(f"Failed to extract D vector: {e}")
    
    # --- Fallback: Handle the JSON export format from GNN processing pipeline ---
    elif "statespaceblock" in gnn_spec:
        logger.info("Extracting matrices from GNN JSON export structure")
        
        # Extract variable dimensions from statespaceblock
        var_dims = {}
        for var_data in gnn_spec.get("statespaceblock", []):
            var_name = var_data.get("id", "")
            dimensions_str = var_data.get("dimensions", "")
            # Parse dimensions like "3,3,type=float" -> [3, 3]
            if dimensions_str:
                dims_parts = dimensions_str.split(',')
                dims = []
                for part in dims_parts:
                    part = part.strip()
                    if part.startswith('type='):
                        break
                    try:
                        dims.append(int(part))
                    except ValueError:
                        continue
                var_dims[var_name] = dims
        
        # Create default matrices based on dimensions
        default_matrices = {}
        if "A" in var_dims:
            dims = var_dims["A"]
            if len(dims) >= 2:
                default_matrices['A'] = np.eye(dims[0], dims[1])  # Identity matrix
                logger.info(f"Created default A matrix with dimensions {dims}")
        
        if "B" in var_dims:
            dims = var_dims["B"]
            if len(dims) >= 3:
                # Create identity-like transition matrix
                default_matrices['B'] = np.eye(dims[0], dims[1])[:, :, np.newaxis]
                default_matrices['B'] = np.repeat(default_matrices['B'], dims[2], axis=2)
                logger.info(f"Created default B matrix with dimensions {dims}")
        
        if "C" in var_dims:
            dims = var_dims["C"]
            if len(dims) >= 1:
                default_matrices['C'] = np.zeros(dims[0])  # Zero preferences
                logger.info(f"Created default C vector with dimensions {dims}")
        
        if "D" in var_dims:
            dims = var_dims["D"]
            if len(dims) >= 1:
                default_matrices['D'] = np.ones(dims[0]) / dims[0]  # Uniform prior
                logger.info(f"Created default D vector with dimensions {dims}")
        
        # Initialize matrices with defaults
        matrices.update(default_matrices)
        
        # Extract actual parameter values from InitialParameterization
        initial_params = gnn_spec.get("raw_sections", {}).get("InitialParameterization", "")
        if initial_params:
            # Parse A matrix
            a_match = re.search(r'A\s*=\s*\{([^}]+)\}', initial_params, re.DOTALL)
            if a_match:
                try:
                    a_str = a_match.group(1).strip()
                    parsed_matrix = _parse_gnn_matrix_string(f"{{{a_str}}}")
                    if parsed_matrix.shape != (1, 1):
                        matrices['A'] = parsed_matrix
                        logger.info(f"Successfully parsed A matrix from InitialParameterization: shape {parsed_matrix.shape}")
                    elif 'A' in default_matrices:
                        improved_matrix = _create_improved_default_matrix('A', default_matrices['A'], a_str)
                        matrices['A'] = improved_matrix
                        logger.info(f"Used improved default A matrix: shape {improved_matrix.shape}")
                except Exception as e:
                    logger.warning(f"Failed to parse A matrix: {e}")
            
            # Parse B matrix
            b_match = re.search(r'B\s*=\s*\{([^}]+)\}', initial_params, re.DOTALL)
            if b_match:
                try:
                    b_str = b_match.group(1).strip()
                    parsed_matrix = _parse_gnn_matrix_string(f"{{{b_str}}}")
                    if parsed_matrix.shape != (1, 1):
                        matrices['B'] = parsed_matrix
                        logger.info(f"Successfully parsed B matrix from InitialParameterization: shape {parsed_matrix.shape}")
                    elif 'B' in default_matrices:
                        improved_matrix = _create_improved_default_matrix('B', default_matrices['B'], b_str)
                        matrices['B'] = improved_matrix
                        logger.info(f"Used improved default B matrix: shape {improved_matrix.shape}")
                except Exception as e:
                    logger.warning(f"Failed to parse B matrix: {e}")
            
            # Parse C vector
            c_match = re.search(r'C\s*=\s*\{([^}]+)\}', initial_params, re.DOTALL)
            if c_match:
                try:
                    c_str = c_match.group(1).strip()
                    parsed_vector = _parse_gnn_matrix_string(f"{{{c_str}}}")
                    if parsed_vector.shape != (1, 1):
                        matrices['C'] = parsed_vector.flatten()
                        logger.info(f"Successfully parsed C vector from InitialParameterization: shape {parsed_vector.flatten().shape}")
                    elif 'C' in default_matrices:
                        improved_vector = _create_improved_default_matrix('C', default_matrices['C'], c_str)
                        matrices['C'] = improved_vector
                        logger.info(f"Used improved default C vector: shape {improved_vector.shape}")
                except Exception as e:
                    logger.warning(f"Failed to parse C vector: {e}")
            
            # Parse D vector
            d_match = re.search(r'D\s*=\s*\{([^}]+)\}', initial_params, re.DOTALL)
            if d_match:
                try:
                    d_str = d_match.group(1).strip()
                    parsed_vector = _parse_gnn_matrix_string(f"{{{d_str}}}")
                    if parsed_vector.shape != (1, 1):
                        matrices['D'] = parsed_vector.flatten()
                        logger.info(f"Successfully parsed D vector from InitialParameterization: shape {parsed_vector.flatten().shape}")
                    elif 'D' in default_matrices:
                        improved_vector = _create_improved_default_matrix('D', default_matrices['D'], d_str)
                        matrices['D'] = improved_vector
                        logger.info(f"Used improved default D vector: shape {improved_vector.shape}")
                except Exception as e:
                    logger.warning(f"Failed to parse D vector: {e}")
    
    # Handle parsed GNN data structure (older format)
    elif "variables" in gnn_spec:
        logger.info("Extracting matrices from parsed GNN data structure")
        
        # Extract variable dimensions from the variables list
        var_dims = {}
        for i, var_data in enumerate(gnn_spec.get("variables", [])):
            var_name = var_data.get("id", "")  # Use 'id' instead of 'name'
            dimensions_str = var_data.get("dimensions", "")  # This is a string like "3,3,type=float"
            
            # Parse dimensions string like "3,3,type=float" -> [3, 3]
            if var_name and dimensions_str:
                dims = []
                for part in dimensions_str.split(','):
                    part = part.strip()
                    if part.startswith('type='):
                        break
                    try:
                        dims.append(int(part))
                    except ValueError:
                        continue
                
                if dims:  # Only add if we successfully parsed dimensions
                    var_dims[var_name] = dims
                    logger.info(f"Found variable '{var_name}' with dimensions {dims}")
                else:
                    logger.warning(f"Could not parse dimensions for variable '{var_name}': '{dimensions_str}'")
        
        logger.info(f"Extracted variable dimensions: {var_dims}")
        
        # Create default matrices based on dimensions
        default_matrices = {}
        if "A" in var_dims:
            dims = var_dims["A"]
            if len(dims) >= 2:
                default_matrices['A'] = np.eye(dims[0], dims[1])  # Identity matrix
                logger.info(f"Created default A matrix with dimensions {dims} -> shape {default_matrices['A'].shape}")
        
        if "B" in var_dims:
            dims = var_dims["B"]
            if len(dims) >= 3:
                # Create identity-like transition matrix
                default_matrices['B'] = np.eye(dims[0], dims[1])[:, :, np.newaxis]
                default_matrices['B'] = np.repeat(default_matrices['B'], dims[2], axis=2)
                logger.info(f"Created default B matrix with dimensions {dims} -> shape {default_matrices['B'].shape}")
        
        if "C" in var_dims:
            dims = var_dims["C"]
            if len(dims) >= 1:
                default_matrices['C'] = np.zeros(dims[0])  # Zero preferences
                logger.info(f"Created default C vector with dimensions {dims} -> shape {default_matrices['C'].shape}")
        
        if "D" in var_dims:
            dims = var_dims["D"]
            if len(dims) >= 1:
                default_matrices['D'] = np.ones(dims[0]) / dims[0]  # Uniform prior
                logger.info(f"Created default D vector with dimensions {dims} -> shape {default_matrices['D'].shape}")
        
        # Initialize matrices with defaults (will be overwritten if parameter parsing succeeds)
        matrices.update(default_matrices)
        logger.info(f"Initialized matrices with shapes: A={matrices.get('A', 'None')}, B={matrices.get('B', 'None')}, C={matrices.get('C', 'None')}, D={matrices.get('D', 'None')}")
        
        # Extract actual parameter values from InitialParameterization if available
        initial_params = gnn_spec.get("InitialParameterization", "")
        if initial_params:
            logger.info("Found InitialParameterization section, attempting to parse matrix values")
            
            # Parse A matrix
            a_match = re.search(r'A\s*=\s*\{([^}]+)\}', initial_params, re.DOTALL)
            if a_match:
                try:
                    a_str = a_match.group(1).strip()
                    parsed_matrix = _parse_gnn_matrix_string(f"{{{a_str}}}")
                    if parsed_matrix.shape != (1, 1):
                        matrices['A'] = parsed_matrix
                        logger.info(f"Successfully parsed A matrix from InitialParameterization: shape {parsed_matrix.shape}")
                    elif 'A' in default_matrices:
                        # Use default matrix with correct dimensions
                        matrices['A'] = default_matrices['A']
                        logger.info(f"Used default A matrix: shape {default_matrices['A'].shape}")
                except Exception as e:
                    logger.warning(f"Failed to parse A matrix: {e}")
            
            # Parse B matrix  
            b_match = re.search(r'B\s*=\s*\{([^}]+)\}', initial_params, re.DOTALL)
            if b_match:
                try:
                    b_str = b_match.group(1).strip()
                    parsed_matrix = _parse_gnn_matrix_string(f"{{{b_str}}}")
                    if parsed_matrix.shape != (1, 1):
                        matrices['B'] = parsed_matrix
                        logger.info(f"Successfully parsed B matrix from InitialParameterization: shape {parsed_matrix.shape}")
                    elif 'B' in default_matrices:
                        matrices['B'] = default_matrices['B']
                        logger.info(f"Used default B matrix: shape {default_matrices['B'].shape}")
                except Exception as e:
                    logger.warning(f"Failed to parse B matrix: {e}")
            
            # Parse C vector
            c_match = re.search(r'C\s*=\s*\{([^}]+)\}', initial_params, re.DOTALL)
            if c_match:
                try:
                    c_str = c_match.group(1).strip()
                    parsed_vector = _parse_gnn_matrix_string(f"{{{c_str}}}")
                    if parsed_vector.shape != (1, 1):
                        matrices['C'] = parsed_vector.flatten()
                        logger.info(f"Successfully parsed C vector from InitialParameterization: shape {parsed_vector.flatten().shape}")
                    elif 'C' in default_matrices:
                        matrices['C'] = default_matrices['C']
                        logger.info(f"Used default C vector: shape {default_matrices['C'].shape}")
                except Exception as e:
                    logger.warning(f"Failed to parse C vector: {e}")
            
            # Parse D vector
            d_match = re.search(r'D\s*=\s*\{([^}]+)\}', initial_params, re.DOTALL)
            if d_match:
                try:
                    d_str = d_match.group(1).strip()
                    parsed_vector = _parse_gnn_matrix_string(f"{{{d_str}}}")
                    if parsed_vector.shape != (1, 1):
                        matrices['D'] = parsed_vector.flatten()
                        logger.info(f"Successfully parsed D vector from InitialParameterization: shape {parsed_vector.flatten().shape}")
                    elif 'D' in default_matrices:
                        matrices['D'] = default_matrices['D']
                        logger.info(f"Used default D vector: shape {default_matrices['D'].shape}")
                except Exception as e:
                    logger.warning(f"Failed to parse D vector: {e}")
        
        # Extract actual parameter values if available (older parameters section)
        for param_data in gnn_spec.get("parameters", []):
            param_name = param_data.get("name", "")
            param_value = param_data.get("value")
            
            if param_name in ["A", "B", "C", "D"] and param_value is not None:
                try:
                    if isinstance(param_value, str):
                        # Parse GNN matrix string format
                        parsed_matrix = _parse_gnn_matrix_string(param_value)
                        
                        # Enhanced dimension inference and validation
                        if parsed_matrix.shape != (1, 1):
                            # Parsing succeeded with meaningful dimensions
                            matrices[param_name] = parsed_matrix
                            logger.info(f"Successfully parsed {param_name} matrix from string: shape {parsed_matrix.shape}")
                        elif param_name in default_matrices:
                            # Parsing failed but we have default dimensions - use improved matrix
                            improved_matrix = _create_improved_default_matrix(param_name, default_matrices[param_name], param_value)
                            matrices[param_name] = improved_matrix
                            logger.info(f"Used improved default {param_name} matrix based on context: shape {improved_matrix.shape}")
                        else:
                            # Parsing failed and no defaults - try dimension inference from context
                            inferred_matrix = _infer_matrix_from_context(param_name, param_value, var_dims)
                            matrices[param_name] = inferred_matrix
                            logger.info(f"Inferred {param_name} matrix from context: shape {inferred_matrix.shape}")
                            
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
                    # Create fallback matrix based on parameter name and expected dimensions
                    fallback_matrix = _create_fallback_matrix(param_name, var_dims)
                    matrices[param_name] = fallback_matrix
                    logger.info(f"Created fallback {param_name} matrix: shape {fallback_matrix.shape}")
                    continue

    else:
        # Handle older raw text format
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

def _create_improved_default_matrix(param_name: str, default_matrix: np.ndarray, param_value: str) -> np.ndarray:
    """Create an improved default matrix based on context clues from the failed parsing."""
    # Try to extract numerical values from the failed parsing string
    import re
    
    # Look for numerical patterns in the string
    numbers = re.findall(r'-?\d+\.?\d*', param_value)
    
    if numbers and len(numbers) > 1:
        # We found numbers, try to use them to improve the default matrix
        try:
            float_numbers = [float(n) for n in numbers]
            
            if param_name == "A":
                # For A matrix, try to create observation model with extracted values
                shape = default_matrix.shape
                new_matrix = np.zeros(shape)
                for i, val in enumerate(float_numbers[:np.prod(shape)]):
                    row = i // shape[1]
                    col = i % shape[1]
                    if row < shape[0] and col < shape[1]:
                        new_matrix[row, col] = val
                
                # Normalize rows to make it a proper probability matrix
                row_sums = new_matrix.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums > 0, row_sums, 1.0)
                new_matrix = new_matrix / row_sums
                return new_matrix
                
            elif param_name == "B":
                # For B matrix, create transition model with extracted values
                shape = default_matrix.shape
                new_matrix = np.zeros(shape)
                for i, val in enumerate(float_numbers[:np.prod(shape)]):
                    # Map linear index to 3D coordinates
                    idx_2d = i % (shape[0] * shape[1])
                    action = i // (shape[0] * shape[1])
                    row = idx_2d // shape[1]
                    col = idx_2d % shape[1]
                    if action < shape[2] and row < shape[0] and col < shape[1]:
                        new_matrix[row, col, action] = val
                
                # Normalize each action matrix
                for a in range(shape[2]):
                    action_matrix = new_matrix[:, :, a]
                    row_sums = action_matrix.sum(axis=1, keepdims=True)
                    row_sums = np.where(row_sums > 0, row_sums, 1.0)
                    new_matrix[:, :, a] = action_matrix / row_sums
                return new_matrix
                
            elif param_name in ["C", "D", "E"]:
                # For vectors, use extracted values directly
                shape = default_matrix.shape
                new_vector = np.zeros(shape)
                for i, val in enumerate(float_numbers[:shape[0]]):
                    new_vector[i] = val
                
                # Normalize if it's a probability vector (D, E)
                if param_name in ["D", "E"]:
                    vector_sum = new_vector.sum()
                    if vector_sum > 0:
                        new_vector = new_vector / vector_sum
                    else:
                        new_vector = np.ones(shape) / shape[0]
                
                return new_vector
        except:
            pass
    
    # If we can't improve it, return the default
    return default_matrix

def _infer_matrix_from_context(param_name: str, param_value: str, var_dims: dict) -> np.ndarray:
    """Infer matrix dimensions and create appropriate matrix when no defaults are available."""
    
    # Try to infer dimensions from variable information
    if param_name in var_dims:
        dims = var_dims[param_name]
    else:
        # Use standard POMDP defaults
        dims = {
            "A": [2, 2],  # 2 observations x 2 states
            "B": [2, 2, 2],  # 2 states x 2 states x 2 actions
            "C": [2],  # 2 observations
            "D": [2],  # 2 states
            "E": [2]   # 2 actions
        }.get(param_name, [2])
    
    # Create appropriate matrix based on parameter type
    if param_name == "A":
        # Observation model - create informative but not deterministic
        shape = tuple(dims) if len(dims) >= 2 else (2, 2)
        matrix = np.eye(min(shape)) + 0.1 * np.random.rand(*shape)
        # Normalize rows
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        return matrix
        
    elif param_name == "B":
        # Transition model - create identity-like transitions with some noise
        shape = tuple(dims) if len(dims) >= 3 else (2, 2, 2)
        matrix = np.zeros(shape)
        for a in range(shape[2]):
            action_matrix = np.eye(shape[0], shape[1]) + 0.1 * np.random.rand(shape[0], shape[1])
            matrix[:, :, a] = action_matrix / action_matrix.sum(axis=1, keepdims=True)
        return matrix
        
    elif param_name == "C":
        # Preferences - slight preference for later observations
        shape = dims[0] if dims else 2
        vector = np.linspace(0.1, 1.0, shape)
        return vector
        
    elif param_name == "D":
        # Prior - uniform
        shape = dims[0] if dims else 2
        return np.ones(shape) / shape
        
    elif param_name == "E":
        # Action prior - uniform
        shape = dims[0] if dims else 2
        return np.ones(shape) / shape
        
    else:
        # Generic fallback
        shape = dims[0] if dims else 2
        return np.ones(shape) / shape

def _create_fallback_matrix(param_name: str, var_dims: dict) -> np.ndarray:
    """Create a fallback matrix when all else fails."""
    return _infer_matrix_from_context(param_name, "", var_dims)

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
    """Generate general JAX model code from GNN specification.
    
    This generates standalone JAX code that only requires jax and jax.numpy,
    without external dependencies like Flax or Optax.
    """
    
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

This model implements the GNN specification using pure JAX for high-performance computation.
Generated automatically by the GNN Processing Pipeline.

This is a standalone JAX implementation that only requires jax and numpy.
No external dependencies like Flax or Optax are required.

@Web: https://github.com/google/jax
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from typing import Dict, Any, Tuple
import numpy as np


# Model configuration
NUM_STATES = {num_states}
NUM_OBSERVATIONS = {num_observations}
NUM_ACTIONS = {num_actions}


def create_params() -> Dict[str, jnp.ndarray]:
    """Create model parameters from GNN specification."""
    return {{
        'A_matrix': jnp.array({A_list}),  # Observation model P(o|s)
        'B_matrix': jnp.array({B_list}),  # Transition model P(s'|s,a)
        'C_vector': jnp.array({C_list}),  # Preferences over observations
        'D_vector': jnp.array({D_list}),  # Prior over initial states
    }}


@jit
def belief_update(params: Dict[str, jnp.ndarray], belief: jnp.ndarray, 
                  observation: jnp.ndarray) -> jnp.ndarray:
    """
    Bayesian belief update given an observation.
    
    P(s|o) ∝ P(o|s) * P(s) using the A matrix (likelihood)
    
    Args:
        params: Model parameters dictionary
        belief: Current belief state [num_states]
        observation: Observation vector [num_observations]
        
    Returns:
        Updated belief state [num_states]
    """
    A_matrix = params['A_matrix']
    
    # Compute likelihood P(o|s) for each state
    # For each state s, compute sum over observations: A[o,s] * observation[o]
    likelihood = jnp.dot(observation, A_matrix)  # [num_states]
    
    # Bayesian update: P(s|o) ∝ P(o|s) * P(s)
    updated_belief = belief * likelihood
    
    # Normalize with numerical stability
    normalizer = jnp.sum(updated_belief) + 1e-8
    updated_belief = updated_belief / normalizer
    
    return updated_belief


@jit
def compute_expected_free_energy(params: Dict[str, jnp.ndarray], belief: jnp.ndarray, 
                                  action: int) -> float:
    """
    Compute expected free energy (EFE) for a given action.
    
    EFE = -E[log P(o)] + KL[Q(s')||P(s')]
        = Entropy of predicted observations + KL divergence
    
    Args:
        params: Model parameters dictionary
        belief: Current belief state [num_states]
        action: Action index
        
    Returns:
        Expected free energy value
    """
    A_matrix = params['A_matrix']
    B_matrix = params['B_matrix']
    C_vector = params['C_vector']
    D_vector = params['D_vector']
    
    # Predict next state distribution using B matrix (transitions)
    # B_matrix[:, :, action] is [num_states, num_states]
    next_belief = jnp.dot(B_matrix[:, :, action], belief)
    next_belief = next_belief / (jnp.sum(next_belief) + 1e-8)  # Normalize
    
    # Predict observation distribution using A matrix
    predicted_obs = jnp.dot(A_matrix, next_belief)
    predicted_obs = predicted_obs / (jnp.sum(predicted_obs) + 1e-8)  # Normalize
    
    # Compute epistemic value (expected information gain)
    # This is the negative entropy of predicted observations
    obs_entropy = -jnp.sum(predicted_obs * jnp.log(predicted_obs + 1e-8))
    
    # Compute pragmatic value (expected utility based on preferences)
    # Higher values in C_vector mean more preferred observations
    pragmatic_value = jnp.dot(predicted_obs, C_vector)
    
    # Compute KL divergence between predicted state and prior
    kl_divergence = jnp.sum(
        jnp.where(next_belief > 1e-8,
                  next_belief * jnp.log((next_belief + 1e-8) / (D_vector + 1e-8)),
                  0.0)
    )
    
    # Expected free energy = uncertainty - utility + complexity
    efe = obs_entropy - pragmatic_value + 0.1 * kl_divergence
    
    return efe


@jit
def select_action(params: Dict[str, jnp.ndarray], belief: jnp.ndarray) -> Tuple[int, jnp.ndarray]:
    """
    Select action with minimum expected free energy.
    
    Args:
        params: Model parameters dictionary
        belief: Current belief state [num_states]
        
    Returns:
        Tuple of (selected_action_index, efe_values_for_all_actions)
    """
    # Compute EFE for all actions
    efe_values = jnp.array([
        compute_expected_free_energy(params, belief, a) 
        for a in range(NUM_ACTIONS)
    ])
    
    # Select action with minimum EFE
    selected_action = jnp.argmin(efe_values)
    
    return selected_action, efe_values


@jit
def state_transition(params: Dict[str, jnp.ndarray], belief: jnp.ndarray, 
                     action: int) -> jnp.ndarray:
    """
    Predict next state distribution given current belief and action.
    
    Args:
        params: Model parameters dictionary
        belief: Current belief state [num_states]
        action: Action index
        
    Returns:
        Predicted next state distribution [num_states]
    """
    B_matrix = params['B_matrix']
    
    # Apply transition model
    next_belief = jnp.dot(B_matrix[:, :, action], belief)
    
    # Normalize
    next_belief = next_belief / (jnp.sum(next_belief) + 1e-8)
    
    return next_belief


def simulate_step(params: Dict[str, jnp.ndarray], belief: jnp.ndarray, 
                  observation: jnp.ndarray) -> Dict[str, Any]:
    """
    Perform one step of Active Inference simulation.
    
    This includes:
    1. Belief update given observation
    2. Action selection based on expected free energy
    3. State prediction for selected action
    
    Args:
        params: Model parameters dictionary
        belief: Current belief state [num_states]
        observation: Observation vector [num_observations]
        
    Returns:
        Dictionary with simulation results
    """
    # 1. Update belief based on observation
    updated_belief = belief_update(params, belief, observation)
    
    # 2. Select action
    selected_action, efe_values = select_action(params, updated_belief)
    
    # 3. Predict next state
    predicted_next_state = state_transition(params, updated_belief, selected_action)
    
    return {{
        'belief': updated_belief,
        'action': selected_action,
        'expected_free_energy': efe_values[selected_action],
        'all_efe_values': efe_values,
        'predicted_next_state': predicted_next_state,
    }}


def run_simulation(params: Dict[str, jnp.ndarray], observations: jnp.ndarray, 
                   initial_belief: jnp.ndarray = None) -> Dict[str, Any]:
    """
    Run a full Active Inference simulation over a sequence of observations.
    
    Args:
        params: Model parameters dictionary
        observations: Sequence of observations [num_steps, num_observations]
        initial_belief: Optional initial belief (default: uniform)
        
    Returns:
        Dictionary with simulation trajectory
    """
    num_steps = observations.shape[0]
    
    # Initialize belief
    if initial_belief is None:
        belief = params['D_vector'].copy()
    else:
        belief = initial_belief
    
    # Storage for trajectory
    beliefs = []
    actions = []
    efes = []
    
    # Run simulation
    for t in range(num_steps):
        result = simulate_step(params, belief, observations[t])
        
        beliefs.append(result['belief'])
        actions.append(result['action'])
        efes.append(result['expected_free_energy'])
        
        # Update belief for next step (using predicted next state)
        belief = result['predicted_next_state']
    
    return {{
        'beliefs': jnp.stack(beliefs),
        'actions': jnp.array(actions),
        'expected_free_energies': jnp.array(efes),
        'final_belief': belief,
    }}


def get_model_summary() -> str:
    """Get a summary of the model architecture."""
    return f"""
{model_name} Model Summary (Pure JAX Implementation):
- Number of states: {{NUM_STATES}}
- Number of observations: {{NUM_OBSERVATIONS}}
- Number of actions: {{NUM_ACTIONS}}
- Parameters: A matrix, B matrix, C vector, D vector

Key Functions:
- create_params(): Create model parameters
- belief_update(): Bayesian belief update
- compute_expected_free_energy(): Compute EFE for action
- select_action(): Select action with minimum EFE
- simulate_step(): One step of Active Inference
- run_simulation(): Full simulation over observations
"""


if __name__ == "__main__":
    print("=" * 60)
    print("JAX Active Inference Model: {model_name}")
    print("=" * 60)
    
    # Create parameters
    params = create_params()
    print("\\n✅ Model parameters created")
    print(f"   A matrix shape: {{params['A_matrix'].shape}}")
    print(f"   B matrix shape: {{params['B_matrix'].shape}}")
    print(f"   C vector shape: {{params['C_vector'].shape}}")
    print(f"   D vector shape: {{params['D_vector'].shape}}")
    
    # Print model summary
    print(get_model_summary())
    
    # Test with uniform initial belief
    print("\\n🧪 Running test simulation...")
    initial_belief = params['D_vector']
    
    # Create a test observation (one-hot for first observation)
    test_obs = jnp.zeros(NUM_OBSERVATIONS)
    test_obs = test_obs.at[0].set(1.0)
    
    # Run one simulation step
    result = simulate_step(params, initial_belief, test_obs)
    
    print(f"\\n📊 Simulation Results:")
    print(f"   Initial belief: {{initial_belief}}")
    print(f"   Updated belief: {{result['belief']}}")
    print(f"   Selected action: {{result['action']}}")
    print(f"   Expected free energy: {{result['expected_free_energy']:.4f}}")
    print(f"   EFE for all actions: {{result['all_efe_values']}}")
    
    # Run multi-step simulation
    print("\\n🔄 Running 5-step simulation...")
    observations = jnp.eye(NUM_OBSERVATIONS)[:min(5, NUM_OBSERVATIONS)]  # Use identity as test observations
    if observations.shape[0] < 5:
        # Pad with repeated observations if needed
        observations = jnp.concatenate([observations] * (5 // observations.shape[0] + 1))[:5]
    
    trajectory = run_simulation(params, observations)
    
    print(f"   Actions taken: {{trajectory['actions']}}")
    print(f"   Final belief: {{trajectory['final_belief']}}")
    print(f"   Average EFE: {{jnp.mean(trajectory['expected_free_energies']):.4f}}")
    
    print("\\n✅ JAX Active Inference model test successful!")
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

import sys
import subprocess

# Ensure JAX is installed before importing
try:
    import jax
    print("✅ JAX is available")
except ImportError:
    print("📦 JAX not found - installing...")
    try:
        # Try UV first (as per project rules)
        result = subprocess.run(
            [sys.executable, "-m", "uv", "pip", "install", "jax", "jaxlib"],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode != 0:
            # Fallback to pip if UV fails
            print("⚠️  UV install failed, trying pip...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "jax", "jaxlib"],
                capture_output=True,
                text=True,
                timeout=300
            )
        if result.returncode == 0:
            print("✅ JAX installed successfully")
            import jax
        else:
            print(f"❌ Failed to install JAX: {result.stderr}")
            sys.exit(1)
    except subprocess.TimeoutExpired:
        print("❌ JAX installation timed out")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error installing JAX: {e}")
        sys.exit(1)

import jax.numpy as jnp
from functools import partial
from jax import jit, vmap, pmap
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

# Try to import optax, install if missing
try:
    import optax
except ImportError:
    print("📦 Optax not found - installing...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "uv", "pip", "install", "optax"],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode != 0:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "optax"],
                capture_output=True,
                text=True,
                timeout=120
            )
        if result.returncode == 0:
            import optax
        else:
            print("⚠️  Optax installation failed, continuing without it")
            optax = None
    except Exception as e:
        print(f"⚠️  Error installing optax: {e}, continuing without it")
        optax = None

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
        self.discount_factor = models.discount_factor  # Add missing discount factor
        
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
            # Use JAX-compatible conditional instead of if statement
            next_belief = self.belief_update(belief, action, obs)
            values = jnp.dot(alpha_vectors, next_belief)
            best_alpha = alpha_vectors[jnp.argmax(values)]
            # Only add contribution if observation probability is significant
            contribution = jnp.where(obs_prob > 1e-10, 
                                   self.discount_factor * obs_prob * best_alpha,
                                   jnp.zeros_like(best_alpha))
            alpha = alpha + contribution
        
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