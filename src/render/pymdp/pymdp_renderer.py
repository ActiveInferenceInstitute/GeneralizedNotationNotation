#!/usr/bin/env python3
"""
PyMDP Renderer

Renders GNN specifications to PyMDP simulation code that integrates with the 
execute/pymdp module. This renderer creates executable PyMDP simulations
configured from parsed GNN POMDP specifications.

Features:
- GNN-to-PyMDP parameter extraction
- Authentic PyMDP simulation code generation
- Pipeline integration with execute module
- Configurable POMDP matrices from GNN initial parameterization

Author: GNN PyMDP Integration
Date: 2024
"""

import re
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging

try:
    from ...gnn.parsers.markdown_parser import MarkdownGNNParser
    from ...gnn.parsers.json_parser import JSONGNNParser
    from ...gnn.parsers.common import GNNInternalRepresentation, ParseResult
except ImportError:
    # Fallback imports for standalone use
    try:
        from gnn.parsers.markdown_parser import MarkdownGNNParser
        from gnn.parsers.json_parser import JSONGNNParser
        from gnn.parsers.common import GNNInternalRepresentation, ParseResult
    except ImportError:
        # Simple fallback for testing
        logging.warning("GNN parsers not available, using simplified parsing")
        class GNNInternalRepresentation:
            def __init__(self, data): self.data = data
        class ParseResult:
            def __init__(self, success, data): self.success = success; self.data = data
        class MarkdownGNNParser:
            def parse(self, content): return ParseResult(True, {'model_name': 'FallbackModel'})
        class JSONGNNParser:
            def parse(self, content): return ParseResult(True, {'model_name': 'FallbackModel'})


def parse_gnn_markdown(content: str, file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Parse GNN markdown content into a structured dictionary.
    
    Args:
        content: GNN markdown content
        file_path: Path to the source file
        
    Returns:
        Parsed GNN specification dictionary or None if parsing fails
    """
    try:
        parser = MarkdownGNNParser()
        result = parser.parse_string(content)
        
        if result.success:
            # Convert internal representation to dictionary
            gnn_spec = result.model.to_dict()
            return gnn_spec
        else:
            logging.error(f"Failed to parse GNN file {file_path}: {result.errors}")
            return None
            
    except Exception as e:
        logging.error(f"Exception parsing GNN file {file_path}: {e}")
        return None


def parse_state_space_block(content: str) -> Dict[str, Any]:
    """Parse StateSpaceBlock section from GNN content."""
    variables = {}
    
    # Find StateSpaceBlock section
    state_space_pattern = r'## StateSpaceBlock\s*\n(.*?)(?=\n##|\Z)'
    match = re.search(state_space_pattern, content, re.DOTALL)
    
    if match:
        block_content = match.group(1)
        
        # Parse variable definitions
        var_pattern = r'^([A-Za-z_][A-Za-z0-9_]*)\[([0-9,]+)(?:,type=(\w+))?\]'
        
        for line in block_content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                var_match = re.match(var_pattern, line)
                if var_match:
                    var_name = var_match.group(1)
                    dimensions_str = var_match.group(2)
                    var_type = var_match.group(3) or 'float'
                    
                    dimensions = [int(d.strip()) for d in dimensions_str.split(',')]
                    
                    variables[var_name] = {
                        'name': var_name,
                        'dimensions': dimensions,
                        'type': var_type
                    }
    
    return variables


def parse_initial_parameterization(content: str) -> dict:
    """Parse InitialParameterization section from GNN content."""
    params = {}
    
    # Find InitialParameterization section
    params_pattern = r'## InitialParameterization\s*\n(.*?)(?=\n##|\Z)'
    match = re.search(params_pattern, content, re.DOTALL)
    
    if match:
        block_content = match.group(1)
        
        # Parse parameter assignments
        current_param = None
        current_value = []
        
        for line in block_content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                # Check if this is a parameter assignment
                if '=' in line and not line.startswith('{'):
                    # Save previous parameter if exists
                    if current_param and current_value:
                        params[current_param] = ''.join(current_value)
                    
                    # Start new parameter
                    param_name, param_start = line.split('=', 1)
                    current_param = param_name.strip()
                    current_value = [param_start.strip()]
                elif current_param and (line.startswith('{') or line.startswith('(') or current_value):
                    # Continue current parameter value
                    current_value.append(' ' + line)
        
        # Save last parameter
        if current_param and current_value:
            params[current_param] = ''.join(current_value)
    
    return params


def parse_model_parameters(content: str) -> Dict[str, Any]:
    """Parse ModelParameters section from GNN content."""
    params = {}
    
    # Find ModelParameters section
    params_pattern = r'## ModelParameters\s*\n(.*?)(?=\n##|\Z)'
    match = re.search(params_pattern, content, re.DOTALL)
    
    if match:
        block_content = match.group(1)
        
        for line in block_content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Try to parse as number
                try:
                    if '.' in value:
                        params[key] = float(value)
                    else:
                        params[key] = int(value)
                except ValueError:
                    params[key] = value
    
    return params


class PyMDPRenderer:
    """
    PyMDP renderer for generating executable PyMDP simulation code from GNN specifications.
    """
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        """
        Initialize PyMDP renderer.
        
        Args:
            options: Optional configuration options
        """
        self.options = options or {}
        self.logger = logging.getLogger(__name__)
    
    def render_file(self, gnn_file_path: Path, output_path: Path) -> Tuple[bool, str]:
        """
        Render a single GNN file to PyMDP simulation code.
        
        Args:
            gnn_file_path: Path to GNN file
            output_path: Path for output PyMDP script
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Read GNN file
            with open(gnn_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse GNN content
            gnn_spec = parse_gnn_markdown(content, gnn_file_path)
            if not gnn_spec:
                return False, f"Failed to parse GNN file: {gnn_file_path}"
            
            # Generate PyMDP simulation code
            pymdp_code = self._generate_pymdp_simulation_code(gnn_spec, gnn_file_path.stem)
            
            # Write output file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(pymdp_code)
            
            self.logger.info(f"Generated PyMDP simulation: {output_path}")
            return True, f"Successfully generated PyMDP simulation code"
            
        except Exception as e:
            error_msg = f"Error rendering {gnn_file_path}: {e}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def render_directory(self, output_dir: Path, input_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Render all GNN files in a directory to PyMDP simulation code.
        
        Args:
            output_dir: Directory for output files
            input_dir: Input directory with GNN files (optional)
            
        Returns:
            Dictionary with rendering results
        """
        results = {
            'rendered_files': [],
            'failed_files': [],
            'total_files': 0,
            'successful_renders': 0
        }
        
        # Find GNN files
        if input_dir:
            gnn_files = list(input_dir.glob("*.md")) + list(input_dir.glob("*.gnn"))
        else:
            # Use default input directory
            gnn_files = list(Path("input/gnn_files").glob("*.md"))
        
        results['total_files'] = len(gnn_files)
        
        for gnn_file in gnn_files:
            output_file = output_dir / f"{gnn_file.stem}_pymdp_simulation.py"
            success, message = self.render_file(gnn_file, output_file)
            
            if success:
                results['rendered_files'].append({
                    'input_file': str(gnn_file),
                    'output_file': str(output_file),
                    'message': message
                })
                results['successful_renders'] += 1
            else:
                results['failed_files'].append({
                    'input_file': str(gnn_file),
                    'error': message
                })
        
        return results
    
    def _generate_pymdp_simulation_code(self, gnn_spec: Dict[str, Any], model_name: str) -> str:
        """
        Generate executable PyMDP simulation code from GNN specification.
        
        Args:
            gnn_spec: Parsed GNN specification
            model_name: Name of the model
            
        Returns:
            Generated Python code string
        """
        # Extract key information from GNN spec
        model_display_name = gnn_spec.get('model_name', model_name)
        model_annotation = gnn_spec.get('annotation', '')
        
        # Parse state space variables  
        variables = gnn_spec.get('variables', [])
        state_vars = {var['name']: var for var in variables if var.get('name') in ['A', 'B', 'C', 'D', 'E']}
        
        # Extract dimensions
        num_states = 3
        num_observations = 3  
        num_actions = 3
        
        # Try to get dimensions from variables
        if 'A' in state_vars and 'dimensions' in state_vars['A']:
            dims = state_vars['A']['dimensions']
            if len(dims) >= 2:
                num_observations = dims[0]
                num_states = dims[1]
        
        if 'B' in state_vars and 'dimensions' in state_vars['B']:
            dims = state_vars['B']['dimensions']
            if len(dims) >= 3:
                num_actions = dims[2]
        
        # Try model parameters
        model_params = gnn_spec.get('model_parameters', {})
        if model_params:
            num_states = model_params.get('num_hidden_states', num_states)
            num_observations = model_params.get('num_obs', num_observations)
            num_actions = model_params.get('num_actions', num_actions)
        
        # Get initial parameterization (try both key variations)
        initial_params = gnn_spec.get('initialparameterization', {}) or gnn_spec.get('initial_parameterization', {})
        
        # Extract state space matrices/vectors
        A_matrix = initial_params.get('A')
        B_matrix = initial_params.get('B')
        C_vector = initial_params.get('C')
        D_vector = initial_params.get('D')
        E_vector = initial_params.get('E')
        
        # Validate state spaces are present
        if not A_matrix:
            self.logger.warning("A matrix (likelihood) not found in initial parameterization")
        if not B_matrix:
            self.logger.warning("B matrix (transition) not found in initial parameterization")
        if not C_vector:
            self.logger.warning("C vector (preferences) not found in initial parameterization")
        if not D_vector:
            self.logger.warning("D vector (prior) not found in initial parameterization")
        
        # Format matrices for embedding in code
        import json as json_module
        import numpy as np
        
        # Convert matrices to JSON-serializable format for embedding
        # Convert matrices to JSON-serializable format for embedding
        def format_matrix_for_code(matrix):
            """Convert matrix to string representation for code embedding."""
            if matrix is None:
                return "None"
            
            # recursive helper to ensure everything is a list or primitive
            def to_clean_list(obj):
                if isinstance(obj, (np.ndarray, list, tuple)):
                    return [to_clean_list(x) for x in obj]
                if isinstance(obj, (int, float, str, bool, type(None))):
                    return obj
                if hasattr(obj, 'tolist'):
                    return to_clean_list(obj.tolist())
                return str(obj) # Fallback

            try:
                # Ensure it's a clean list structure (handles jagged lists naturally)
                clean_data = to_clean_list(matrix)
                return json_module.dumps(clean_data)
            except Exception as e:
                # Fallback to string representation
                logger.warning(f"Failed to cleanly format matrix: {e}. using raw dumps.")
                return json_module.dumps(matrix)
        
        A_matrix_str = format_matrix_for_code(A_matrix)
        B_matrix_str = format_matrix_for_code(B_matrix)
        C_vector_str = format_matrix_for_code(C_vector)
        D_vector_str = format_matrix_for_code(D_vector)
        E_vector_str = format_matrix_for_code(E_vector)
        
        # Generate the code
        code = f'''#!/usr/bin/env python3
"""
PyMDP Simulation Script for {model_display_name}

This script was automatically generated from a GNN specification.
It uses the GNN pipeline's PyMDP execution module to run an Active Inference simulation.

Model: {model_display_name}
Description: {model_annotation}
Generated: {self._get_timestamp()}

State Space:
- Hidden States: {num_states}
- Observations: {num_observations} 
- Actions: {num_actions}

State Space Matrices (from GNN):
- A (Likelihood): {'Present' if A_matrix else 'Missing'}
- B (Transition): {'Present' if B_matrix else 'Missing'}
- C (Preferences): {'Present' if C_vector else 'Missing'}
- D (Prior): {'Present' if D_vector else 'Missing'}
- E (Habits): {'Present' if E_vector else 'Missing'}
"""

import sys
from pathlib import Path
import os

# Prevent import conflict with local 'pymdp' folder which contains this script
# sys.path[0] is the script directory. If it's named 'pymdp', it masks the installed library.
if sys.path[0] and sys.path[0].endswith("pymdp"):
    print(f"⚠️  Detected namespace conflict with script directory '{{sys.path[0]}}', removing from sys.path")
    sys.path.pop(0)
import logging
import subprocess
import json
import numpy as np

# Ensure PyMDP is installed before importing
# Note: The correct package name is 'inferactively-pymdp', not 'pymdp'
try:
    import pymdp
    # Verify it is the CORRECT pymdp (inferactively-pymdp)
    try:
        from pymdp.agent import Agent
        print("✅ PyMDP (inferactively-pymdp) is available")
    except ImportError:
        # Check if it might be the flat structure (unlikely for modern, but possible)
        if hasattr(pymdp, "Agent"):
             print("✅ PyMDP (flat structure) is available")
        else:
             print("⚠️  PyMDP package found, but it appears to be the wrong version (missing Agent).")
             raise ImportError("Wrong PyMDP package detected")
except ImportError:
    print("📦 PyMDP not found or wrong version - installing inferactively-pymdp...")
    try:
        # Try UV first (as per project rules)
        result = subprocess.run(
            [sys.executable, "-m", "uv", "pip", "install", "inferactively-pymdp"],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode != 0:
            # Fallback to pip if UV fails
            print("⚠️  UV install failed, trying pip...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "inferactively-pymdp"],
                capture_output=True,
                text=True,
                timeout=120
            )
        if result.returncode == 0:
            print("✅ PyMDP (inferactively-pymdp) installed successfully")
            import pymdp
        else:
            print(f"❌ Failed to install PyMDP: {{result.stderr}}")
            print("💡 Install manually with: uv pip install inferactively-pymdp")
            sys.exit(1)
    except subprocess.TimeoutExpired:
        print("❌ PyMDP installation timed out")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error installing PyMDP: {{e}}")
        print("💡 Install manually with: uv pip install inferactively-pymdp")
        sys.exit(1)

# Add project root to path for imports (script is 5 levels deep: output/11_render_output/actinf_pomdp_agent/pymdp/script.py)
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.execute.pymdp import execute_pymdp_simulation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main simulation function."""
    
    # State Space Matrices (extracted from GNN and embedded here)
    A_matrix_data = {A_matrix_str}  # Likelihood matrix P(o|s)
    B_matrix_data = {B_matrix_str}  # Transition matrix P(s'|s,u)
    C_vector_data = {C_vector_str}  # Preferences over observations
    D_vector_data = {D_vector_str}  # Prior beliefs over states
    E_vector_data = {E_vector_str}  # Policy priors (habits)
    
    # Convert to numpy arrays
    if A_matrix_data is not None:
        A_matrix = np.array(A_matrix_data)
        # Normalize A matrix (columns should sum to 1)
        if A_matrix.ndim == 2:
            norm = np.sum(A_matrix, axis=0)
            A_matrix = A_matrix / np.where(norm == 0, 1, norm)
        logger.info(f"A matrix shape: {{A_matrix.shape}}")
    else:
        A_matrix = None
        logger.warning("A matrix not provided")
    
    if B_matrix_data is not None:
        B_matrix = np.array(B_matrix_data)
        # Normalize B matrix (columns should sum to 1)
        # B shape in PyMDP usually (next_state, prev_state, action)
        # But GNN might provide (action, prev_state, next_state) or similar.
        # Here we assume GNN provides B as [action][prev][next] or similar from JSON
        # We will trust the default simple_simulation handling for dimension/transposition,
        # but here we just ensure values are normalized along the last dimension if it sums approx to > 0.
        # Actually, let's just ensure it's normalized in simple_simulation or here?
        # Better to do it in simple_simulation.py where we know the dimensions?
        # The simple_simulation.py loads this gnn_spec.
        # So we should modify simple_simulation.py instead?
        # NO, this script IS the one that passes data to simple_simulation via gnn_spec['initialparameterization'].
        # The simple_simulation.py reads A from gnn_spec['initialparameterization']['A'].
        # So if we update 'A_matrix' variable here, we must ensure it is passed back to gnn_spec correctly.
        # Lines 494 update gnn_spec using 'A_matrix.tolist()'.
        # So normalizing HERE is correct.
        
        # However, for B matrix, dimensions are tricky. 
        # Let's simple_simulation handle B normalization since it does transposition logic.
        logger.info(f"B matrix shape: {{B_matrix.shape}}")
    else:
        B_matrix = None
        logger.warning("B matrix not provided")
    
    if C_vector_data is not None:
        C_vector = np.array(C_vector_data)
        logger.info(f"C vector shape: {{C_vector.shape}}")
    else:
        C_vector = None
        logger.warning("C vector not provided")
    
    if D_vector_data is not None:
        D_vector = np.array(D_vector_data)
        # Normalize D vector
        norm = np.sum(D_vector)
        D_vector = D_vector / np.where(norm == 0, 1, norm)
        logger.info(f"D vector shape: {{D_vector.shape}}")
    else:
        D_vector = None
        logger.warning("D vector not provided")
    
    if E_vector_data is not None:
        E_vector = np.array(E_vector_data)
        logger.info(f"E vector shape: {{E_vector.shape}}")
    else:
        E_vector = None
    
    # GNN Specification (embedded with state spaces)
    gnn_spec = {json_module.dumps(gnn_spec, indent=4, default=str)}
    
    # Ensure state space matrices are in gnn_spec for execution
    if 'initialparameterization' not in gnn_spec:
        gnn_spec['initialparameterization'] = {{}}
    if A_matrix is not None:
        gnn_spec['initialparameterization']['A'] = A_matrix.tolist() if hasattr(A_matrix, 'tolist') else A_matrix
    if B_matrix is not None:
        gnn_spec['initialparameterization']['B'] = B_matrix.tolist() if hasattr(B_matrix, 'tolist') else B_matrix
    if C_vector is not None:
        gnn_spec['initialparameterization']['C'] = C_vector.tolist() if hasattr(C_vector, 'tolist') else C_vector
    if D_vector is not None:
        gnn_spec['initialparameterization']['D'] = D_vector.tolist() if hasattr(D_vector, 'tolist') else D_vector
    if E_vector is not None:
        gnn_spec['initialparameterization']['E'] = E_vector.tolist() if hasattr(E_vector, 'tolist') else E_vector
    
    # Output directory
    output_dir = Path("output") / "pymdp_simulations" / "{model_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting PyMDP simulation for {model_display_name}")
    logger.info(f"Output directory: {{output_dir}}")
    logger.info(f"State space matrices: A={{A_matrix is not None}}, B={{B_matrix is not None}}, C={{C_vector is not None}}, D={{D_vector is not None}}, E={{E_vector is not None}}")
    
    # Run simulation
    try:
        success, results = execute_pymdp_simulation(
            gnn_spec=gnn_spec,
            output_dir=output_dir,
            correlation_id="render_generated_script"
        )
        
        if success:
            logger.info("✓ Simulation completed successfully!")
            logger.info(f"Results summary:")
            logger.info(f"  Correlation ID: {{results.get('correlation_id', 'N/A')}}")
            logger.info(f"  Success: {{results.get('success', False)}}")
            logger.info(f"  Output: {{output_dir}}")
            return 0
        else:
            logger.error("✗ Simulation failed!")
            logger.error(f"Error: {{results.get('error', 'Unknown error')}}")
            return 1
            
    except Exception as e:
        logger.error(f"Unexpected error: {{e}}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''
        
        return code
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def render_gnn_to_pymdp(
    gnn_spec: Dict[str, Any],
    output_script_path: Path,
    options: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, List[str]]:
    """
    Render GNN specification to PyMDP simulation script.
    
    Args:
        gnn_spec: Parsed GNN specification dictionary
        output_script_path: Path for output PyMDP script
        options: Optional rendering options
        
    Returns:
        Tuple of (success, message, warnings)
    """
    try:
        renderer = PyMDPRenderer(options)
        
        # Generate simulation code directly from spec
        model_name = gnn_spec.get('model_name', 'GNN_Model')
        pymdp_code = renderer._generate_pymdp_simulation_code(gnn_spec, model_name)
        
        # Write output file
        output_script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_script_path, 'w', encoding='utf-8') as f:
            f.write(pymdp_code)
        
        message = f"Generated PyMDP simulation script: {output_script_path}"
        warnings = []
        
        # Check for potential issues
        if not gnn_spec.get('initial_parameterization'):
            warnings.append("No initial parameterization found - using defaults")
        
        if not gnn_spec.get('model_parameters'):
            warnings.append("No model parameters found - using inferred dimensions")
        
        return True, message, warnings
        
    except Exception as e:
        return False, f"Error generating PyMDP script: {e}", [] 