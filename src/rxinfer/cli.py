"""
Command-line interface for the RxInfer module.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from . import __version__
from .gnn_parser import parse_gnn_file
from .config_generator import generate_rxinfer_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_gnn_to_config(input_file: Path, output_file: Path, debug: bool = False) -> bool:
    """
    Process a GNN file to generate a RxInfer.jl configuration file.
    
    Args:
        input_file: Path to the input GNN file
        output_file: Path to the output TOML configuration file
        debug: Whether to enable debug logging
        
    Returns:
        True if successful, False otherwise
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Parse the GNN file
        logger.info(f"Parsing GNN file: {input_file}")
        parsed_gnn = parse_gnn_file(input_file)
        
        # Debug: Save parsed content to JSON
        if debug:
            debug_file = output_file.with_suffix('.debug.json')
            with open(debug_file, 'w', encoding='utf-8') as f:
                json.dump(parsed_gnn, f, indent=2, default=str)
            logger.debug(f"Saved debug output to: {debug_file}")
        
        # Generate the configuration file
        logger.info(f"Generating RxInfer.jl configuration file: {output_file}")
        success = generate_rxinfer_config(parsed_gnn, output_file)
        
        if success:
            logger.info(f"Successfully generated configuration file: {output_file}")
            return True
        else:
            logger.error("Failed to generate configuration file")
            return False
    
    except Exception as e:
        logger.error(f"Error processing GNN file: {e}", exc_info=True)
        return False

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Process GNN files to generate RxInfer.jl configurations")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("input", help="Input GNN file path")
    parser.add_argument("output", help="Output TOML configuration file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    input_file = Path(args.input)
    output_file = Path(args.output)
    
    success = process_gnn_to_config(input_file, output_file, args.debug)
    
    # Return appropriate exit code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 