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