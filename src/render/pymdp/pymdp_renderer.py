"""
PyMDP Renderer Module for GNN Specifications

This module serves as the main entry point for rendering GNN specifications
to PyMDP-compatible Python scripts. It coordinates the conversion, template
generation, and script assembly processes.
"""

import logging
import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import re

# Import from local modules, assume they exist in the same directory
try:
    # When imported as a module within the package
    from .pymdp_converter import GnnToPyMdpConverter
except ImportError:
    # When run as a standalone script
    print("Warning: Unable to import GnnToPyMdpConverter as a relative import. "
          "This may occur when running the module directly as a script.")
    # Attempt to add the parent directory to the path for direct script execution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.append(project_root)
    try:
        from render.pymdp_converter import GnnToPyMdpConverter
    except ImportError:
        print("Error: Failed to import GnnToPyMdpConverter. "
              "Make sure the pymdp_converter.py file exists in the same directory.")
        sys.exit(1)

logger = logging.getLogger(__name__)

def parse_gnn_markdown(content: str, file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Parse GNN specification from markdown content.
    
    Args:
        content: The markdown content to parse
        file_path: Path to the source file for error reporting
        
    Returns:
        Dictionary containing the parsed GNN specification, or None if parsing fails
    """
    try:
        gnn_spec = {}
        
        # Extract ModelName
        model_name_match = re.search(r'#\s*ModelName\s*\n(.*?)(?=\n#|\n$)', content, re.DOTALL)
        if model_name_match:
            gnn_spec['ModelName'] = model_name_match.group(1).strip()
        
        # Extract ModelAnnotation
        annotation_match = re.search(r'#\s*ModelAnnotation\s*\n(.*?)(?=\n#|\n$)', content, re.DOTALL)
        if annotation_match:
            gnn_spec['ModelAnnotation'] = annotation_match.group(1).strip()
        
        # Extract StateSpaceBlock
        state_block_match = re.search(r'#\s*StateSpaceBlock\s*\n(.*?)(?=\n#|\n$)', content, re.DOTALL)
        if state_block_match:
            state_content = state_block_match.group(1).strip()
            gnn_spec['StateSpaceBlock'] = parse_state_space_block(state_content)
        
        # Extract Connections
        connections_match = re.search(r'#\s*Connections\s*\n(.*?)(?=\n#|\n$)', content, re.DOTALL)
        if connections_match:
            connections_content = connections_match.group(1).strip()
            gnn_spec['Connections'] = parse_connections(connections_content)
        
        # Extract InitialParameterization
        param_match = re.search(r'#\s*InitialParameterization\s*\n(.*?)(?=\n#|\n$)', content, re.DOTALL)
        if param_match:
            param_content = param_match.group(1).strip()
            gnn_spec['InitialParameterization'] = parse_initial_parameterization(param_content)
        
        # Extract Equations
        equations_match = re.search(r'#\s*Equations\s*\n(.*?)(?=\n#|\n$)', content, re.DOTALL)
        if equations_match:
            gnn_spec['Equations'] = equations_match.group(1).strip()
        
        # Extract Time settings
        time_match = re.search(r'#\s*Time\s*\n(.*?)(?=\n#|\n$)', content, re.DOTALL)
        if time_match:
            time_content = time_match.group(1).strip()
            gnn_spec['Time'] = parse_time_settings(time_content)
        
        # Extract ActInfOntologyAnnotation
        ontology_match = re.search(r'#\s*ActInfOntologyAnnotation\s*\n(.*?)(?=\n#|\n$)', content, re.DOTALL)
        if ontology_match:
            gnn_spec['ActInfOntologyAnnotation'] = ontology_match.group(1).strip()
        
        # Extract Footer and Signature
        footer_match = re.search(r'#\s*Footer\s*\n(.*?)(?=\n#|\n$)', content, re.DOTALL)
        if footer_match:
            gnn_spec['Footer'] = footer_match.group(1).strip()
        
        signature_match = re.search(r'#\s*Signature\s*\n(.*?)(?=\n#|\n$)', content, re.DOTALL)
        if signature_match:
            gnn_spec['Signature'] = signature_match.group(1).strip()
        
        # Add metadata
        gnn_spec['_source_file'] = str(file_path)
        gnn_spec['_parsed_at'] = str(Path.cwd())
        
        logger.info(f"Successfully parsed GNN markdown with {len(gnn_spec)} sections")
        return gnn_spec
        
    except Exception as e:
        logger.error(f"Failed to parse GNN markdown from {file_path}: {e}")
        return None

def parse_state_space_block(content: str) -> Dict[str, Any]:
    """Parse the StateSpaceBlock section."""
    variables = {}
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            # Parse variable definitions like "s_fX[2,3,real]"
            var_match = re.match(r'(\w+)\[([^\]]+)\]', line)
            if var_match:
                var_name = var_match.group(1)
                var_spec = var_match.group(2)
                
                # Parse dimensions and type
                parts = var_spec.split(',')
                if len(parts) >= 2:
                    dimensions = [int(d.strip()) for d in parts[:-1]]
                    var_type = parts[-1].strip()
                    
                    variables[var_name] = {
                        'dimensions': dimensions,
                        'type': var_type
                    }
    
    return variables

def parse_connections(content: str) -> List[str]:
    """Parse the Connections section."""
    connections = []
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            connections.append(line)
    
    return connections

def parse_initial_parameterization(content: str) -> Dict[str, Any]:
    """Parse the InitialParameterization section."""
    params = {}
    lines = content.split('\n')
    
    current_matrix = None
    current_matrix_data = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        # Check for matrix definitions like "A ="
        matrix_match = re.match(r'(\w+)\s*=', line)
        if matrix_match:
            # Save previous matrix if exists
            if current_matrix and current_matrix_data:
                params[current_matrix] = parse_matrix_data(current_matrix_data)
            
            current_matrix = matrix_match.group(1)
            current_matrix_data = []
        elif current_matrix:
            current_matrix_data.append(line)
    
    # Save last matrix
    if current_matrix and current_matrix_data:
        params[current_matrix] = parse_matrix_data(current_matrix_data)
    
    return params

def parse_matrix_data(lines: List[str]) -> List[List[float]]:
    """Parse matrix data from lines."""
    matrix = []
    for line in lines:
        # Extract numbers from the line
        numbers = re.findall(r'[-+]?\d*\.?\d+', line)
        if numbers:
            row = [float(n) for n in numbers]
            matrix.append(row)
    return matrix

def parse_time_settings(content: str) -> Dict[str, Any]:
    """Parse the Time section."""
    settings = {}
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            # Parse key-value pairs
            if '=' in line:
                key, value = line.split('=', 1)
                settings[key.strip()] = value.strip()
            else:
                # Single values like "Dynamic" or "Static"
                settings['mode'] = line
    
    return settings

class PyMDPRenderer:
    """
    PyMDP Renderer for converting GNN specifications to PyMDP Python scripts.
    
    This class provides a standardized interface for rendering GNN models
    to PyMDP-compatible simulation code.
    """
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        """
        Initialize the PyMDP renderer.
        
        Args:
            options: Dictionary of rendering options
        """
        self.options = options or {}
        self.logger = logging.getLogger(f"{__name__}.PyMDPRenderer")
        
    def render_file(self, gnn_file_path: Path, output_path: Path) -> Tuple[bool, str]:
        """
        Render a single GNN file to PyMDP format.
        
        Args:
            gnn_file_path: Path to the GNN specification file
            output_path: Path where the rendered output should be saved
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Load GNN specification (assuming JSON format for now)
            import json
            with open(gnn_file_path, 'r', encoding='utf-8') as f:
                if gnn_file_path.suffix.lower() == '.json':
                    gnn_spec = json.load(f)
                else:
                    # Parse markdown GNN files by extracting structured sections
                    content = f.read()
                    gnn_spec = parse_gnn_markdown(content, gnn_file_path)
                    if not gnn_spec:
                        return False, f"Failed to parse GNN markdown file: {gnn_file_path}"
            
            # Use the existing render function
            success, message, artifacts = render_gnn_to_pymdp(gnn_spec, output_path, self.options)
            return success, message
            
        except Exception as e:
            error_msg = f"Failed to render {gnn_file_path}: {e}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def render_directory(self, output_dir: Path, input_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Render all GNN files in a directory to PyMDP format.
        
        Args:
            output_dir: Directory where rendered files should be saved
            input_dir: Directory to search for GNN files (optional)
            
        Returns:
            Dictionary with rendering results
        """
        results = {
            'success': True,
            'files_rendered': 0,
            'files_failed': 0,
            'messages': []
        }
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # If no input directory specified, look for processed files
            if input_dir is None:
                # Look for already processed GNN files in common output locations
                search_paths = [
                    Path("../output/gnn_exports"),
                    Path("output/gnn_exports"),
                ]
                
                gnn_files = []
                for search_path in search_paths:
                    if search_path.exists():
                        gnn_files.extend(search_path.glob("**/*.json"))
                        break
            else:
                # Search input directory for GNN files
                gnn_files = list(input_dir.glob("**/*.json"))
                gnn_files.extend(input_dir.glob("**/*.md"))
            
            if not gnn_files:
                results['messages'].append("No GNN files found to render")
                return results
            
            for gnn_file in gnn_files:
                output_file = output_dir / f"{gnn_file.stem}_pymdp.py"
                success, message = self.render_file(gnn_file, output_file)
                
                if success:
                    results['files_rendered'] += 1
                    results['messages'].append(f"✅ Rendered {gnn_file.name} -> {output_file.name}")
                else:
                    results['files_failed'] += 1
                    results['success'] = False
                    results['messages'].append(f"❌ Failed to render {gnn_file.name}: {message}")
            
            return results
            
        except Exception as e:
            error_msg = f"Failed to render directory: {e}"
            self.logger.error(error_msg)
            results['success'] = False
            results['messages'].append(error_msg)
            return results

def render_gnn_to_pymdp(
    gnn_spec: Dict[str, Any],
    output_script_path: Path,
    options: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, List[str]]:
    """
    Main function to render a GNN specification to a PyMDP Python script.

    Args:
        gnn_spec: The GNN specification as a Python dictionary.
        output_script_path: The path where the generated Python script will be saved.
        options: Dictionary of rendering options. 
                 Currently supports "include_example_usage" (bool, default True).

    Returns:
        A tuple (success: bool, message: str, artifact_uris: List[str]).
        `artifact_uris` will contain a file URI to the generated script on success.
    """
    options = options or {}
    include_example_usage = options.get("include_example_usage", True)

    try:
        logger.info(f"Initializing GNN to PyMDP converter for model: {gnn_spec.get('ModelName', 'UnknownModel')}")
        converter = GnnToPyMdpConverter(gnn_spec)
        
        logger.info("Generating PyMDP Python script content...")
        python_script_content = converter.get_full_python_script(
            include_example_usage=include_example_usage
        )
        
        logger.info(f"Writing PyMDP script to: {output_script_path}")
        with open(output_script_path, "w", encoding='utf-8') as f:
            f.write(python_script_content)
        
        success_msg = f"Successfully wrote PyMDP script: {output_script_path.name}"
        logger.info(success_msg)
        
        # Include conversion log in the final message for clarity, perhaps capped
        log_summary = "\n".join(converter.conversion_log[:20]) # First 20 log messages
        if len(converter.conversion_log) > 20:
            log_summary += "\n... (log truncated)"
            
        return True, f"{success_msg}\nConversion Log Summary:\n{log_summary}", [str(output_script_path.resolve())]

    except Exception as e:
        error_msg = f"Failed to render GNN to PyMDP: {e}"
        logger.exception(error_msg) # Log full traceback
        return False, error_msg, []


# Standalone execution for testing
if __name__ == "__main__":
    # Basic logging setup for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Render a GNN specification to a PyMDP Python script.")
    parser.add_argument("gnn_spec_file", type=Path, help="Path to the GNN specification file (JSON format).")
    parser.add_argument("output_script", type=Path, help="Path to save the output PyMDP script.")
    parser.add_argument("--no-example", action="store_true", help="Exclude example usage code from the output.")
    args = parser.parse_args()
    
    try:
        with open(args.gnn_spec_file, 'r', encoding='utf-8') as f:
            gnn_spec = json.load(f)
        
        options = {
            "include_example_usage": not args.no_example
        }
        
        # Ensure output directory exists
        args.output_script.parent.mkdir(parents=True, exist_ok=True)
        
        success, message, artifacts = render_gnn_to_pymdp(gnn_spec, args.output_script, options)
        
        if success:
            print(f"Success: {message}")
            print(f"Generated: {artifacts}")
            sys.exit(0)
        else:
            print(f"Error: {message}")
            sys.exit(1)
            
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {args.gnn_spec_file}: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1) 