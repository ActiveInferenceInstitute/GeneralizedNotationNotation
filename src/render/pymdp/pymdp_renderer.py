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
            def parse(self, content): return ParseResult(True, {'model_name': 'MockModel'})
        class JSONGNNParser:
            def parse(self, content): return ParseResult(True, {'model_name': 'MockModel'})


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
        
        # Get initial parameterization
        initial_params = gnn_spec.get('initial_parameterization', {})
        
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
"""

import sys
from pathlib import Path
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.execute.pymdp import execute_pymdp_simulation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main simulation function."""
    
    # GNN Specification (embedded)
    gnn_spec = {json.dumps(gnn_spec, indent=4)}
    
    # Configuration overrides (can be modified)
    config_overrides = {{
        'num_episodes': 10,
        'max_steps_per_episode': 20,
        'planning_horizon': 5,
        'verbose_output': True,
        'save_visualizations': True,
        'random_seed': 42
    }}
    
    # Output directory
    output_dir = Path("output") / "pymdp_simulations" / "{model_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting PyMDP simulation for {model_display_name}")
    logger.info(f"Output directory: {{output_dir}}")
    
    # Run simulation
    try:
        success, results = execute_pymdp_simulation(
            gnn_spec=gnn_spec,
            output_dir=output_dir,
            config_overrides=config_overrides
        )
        
        if success:
            logger.info("✓ Simulation completed successfully!")
            logger.info(f"Results summary:")
            logger.info(f"  Episodes: {{results.get('total_episodes', 'N/A')}}")
            logger.info(f"  Success Rate: {{results.get('success_rate', 0):.2%}}")
            logger.info(f"  Output: {{results.get('output_directory', output_dir)}}")
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