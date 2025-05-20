"""
GNN to DisCoPy Diagram Translator

This module provides functions to parse GNN files (in their string representation)
and convert them into DisCoPy diagrams.
"""
import re
import logging
from pathlib import Path

from discopy import Ty, Box, Diagram, Id, Swap # type: ignore
# DisCoPy might not be in the project's direct dependencies for static analysis,
# so we can ignore type errors for it here.

logger = logging.getLogger(__name__)

def parse_gnn_content(gnn_content: str) -> dict:
    """
    Parses the string content of a GNN file into a dictionary of sections.
    Each section's content is a list of non-empty, non-comment lines.
    """
    parsed_data = {}
    current_section_name = None
    
    # Regex to identify GNN section headers like ## ModelName, ## StateSpaceBlock
    section_header_pattern = re.compile(r"^##\s*([\w\s]+)\s*$", re.MULTILINE)

    lines = gnn_content.splitlines()
    
    for line_number, line in enumerate(lines):
        stripped_line = line.strip()
        
        header_match = section_header_pattern.match(stripped_line)
        if header_match:
            # A section header was found
            current_section_name = header_match.group(1).strip().replace(" ", "")
            if current_section_name not in parsed_data:
                parsed_data[current_section_name] = []
            logger.debug(f"Found section: {current_section_name} at line {line_number + 1}")
            continue # Move to the next line after processing a header

        # If we are inside a section (current_section_name is not None),
        # and the line is actual content (not empty, not a comment).
        if current_section_name is not None and stripped_line and not stripped_line.startswith("#"):
            # current_section_name is guaranteed to be a key in parsed_data if it's not None
            # because it would have been added when the section header was matched.
            parsed_data[current_section_name].append(stripped_line)
            
    # Log a summary of parsed sections and line counts for debugging
    if logger.isEnabledFor(logging.DEBUG):
        for section, content_lines in parsed_data.items():
            logger.debug(f"  Section '{section}' has {len(content_lines)} relevant lines.")
            if content_lines: # Log first line of content for a quick check
                logger.debug(f"    First line of '{section}': '{content_lines[0]}'")
    
    return parsed_data

def gnn_statespace_to_discopy_types(parsed_gnn: dict) -> dict[str, Ty]:
    """
    Converts GNN StateSpaceBlock entries into DisCoPy Ty objects.
    Returns a dictionary mapping variable names to Ty objects.
    """
    types = {}
    statespace_lines = parsed_gnn.get("StateSpaceBlock", [])
    if not statespace_lines:
        logger.warning("StateSpaceBlock not found or empty. No DisCoPy types will be created.")
        return types

    # Variable pattern: VarName or VarName[dim1,dim2,...] or VarName[dim]
    # Allows for simple names (interpreted as Ty('VarName'))
    # and names with dimensions (Ty('VarName[dim1,dim2]'))
    var_pattern = re.compile(r"^([a-zA-Z_][\w_]*)(?:\s*\[([\w\s,\*\:]+)\])?(?:\s*#.*)?$")

    for line in statespace_lines:
        line = line.strip()
        if not line or line.startswith("#"): # Skip empty lines and comments
            continue
        
        match = var_pattern.match(line)
        if match:
            var_name = match.group(1)
            # Dimensions are captured but currently only used to create a more descriptive Ty name.
            # For actual tensor operations, a functor would need to interpret these dimensions.
            dims_str = match.group(2)
            
            type_name = var_name
            if dims_str:
                type_name = f"{var_name}[{dims_str.strip()}]"
            
            types[var_name] = Ty(type_name)
            logger.debug(f"Created DisCoPy type: Ty('{type_name}') for GNN variable '{var_name}'")
        else:
            logger.warning(f"Could not parse StateSpaceBlock line: '{line}'. Skipping.")
            
    return types

def gnn_connections_to_discopy_diagram(parsed_gnn: dict, types: dict[str, Ty]) -> Diagram | None:
    """
    Converts GNN Connections into a DisCoPy Diagram.
    Currently supports simple directed connections (A > B).
    Assumes connections define a sequential flow for simplicity.
    More complex connection patterns (parallel, cycles) would require advanced logic.
    """
    connections_lines = parsed_gnn.get("Connections", [])
    if not connections_lines:
        logger.warning("Connections section not found or empty. Cannot create DisCoPy diagram.")
        return None

    diagram = Id() # Start with an identity diagram
    
    # For this basic version, we assume a linear chain of connections.
    # Track the current domain/codomain to build the diagram sequentially.
    # This is a simplification; real GNNs can have complex graphs.
    
    parsed_connections = [] # Store as (source_var, target_var)

    # Connection pattern: SourceVar > TargetVar
    # Ignores anything after #
    conn_pattern = re.compile(r"^([a-zA-Z_][\w_]*)\s*>\s*([a-zA-Z_][\w_]*)(?:\s*#.*)?$")

    for line in connections_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        
        match = conn_pattern.match(line)
        if match:
            source_var = match.group(1)
            target_var = match.group(2)
            
            if source_var not in types or target_var not in types:
                logger.warning(f"Unknown variable in connection: '{source_var}' -> '{target_var}'. Skipping connection.")
                continue
            
            source_type = types[source_var]
            target_type = types[target_var]
            
            box_name = f"{source_var}_to_{target_var}"
            box = Box(box_name, source_type, target_type)
            logger.debug(f"Created DisCoPy Box: Box('{box_name}', dom={source_type}, cod={target_type})")
            
            # Simple sequential composition for now
            if diagram.dom == Id().dom and diagram.cod == Id().cod : # First box
                 diagram = box
            elif diagram.cod == source_type: # Chainable
                 diagram = diagram >> box
            else:
                # This indicates a more complex structure (e.g. parallel wires or new starting chain)
                # For now, we will log a warning and try to append it as a new parallel component
                # This is a placeholder for more sophisticated diagram construction.
                logger.warning(f"Connection '{source_var} > {target_var}' does not directly chain with previous diagram codomain ({diagram.cod}). Appending in parallel (basic).")
                # Attempting a parallel composition; this assumes variables are distinct flows if not chained.
                # A more robust solution would analyze the full graph structure.
                try:
                    diagram = diagram @ box 
                except Exception as e_parallel:
                    logger.error(f"Failed to compose Box('{box_name}') in parallel: {e_parallel}. Diagram construction may be incorrect.")
                    return diagram # Return what we have so far
        else:
            logger.warning(f"Could not parse Connections line: '{line}'. Supported format: 'Source > Target'.")

    if diagram.dom == Id().dom and diagram.cod == Id().cod: # Check if any boxes were actually added
        logger.warning("No valid connections were parsed to form a diagram.")
        return None
        
    return diagram

def gnn_file_to_discopy_diagram(gnn_file_path: Path, verbose: bool = False) -> Diagram | None:
    """
    Orchestrates the conversion of a GNN file to a DisCoPy diagram.
    Reads the file, parses content, converts state space and connections.
    """
    # Set logger level for this module based on verbose flag from the caller
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    logger.info(f"Attempting to convert GNN file to DisCoPy diagram: {gnn_file_path}")
    if not gnn_file_path.exists():
        logger.error(f"GNN file not found: {gnn_file_path}")
        return None
        
    try:
        with open(gnn_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        parsed_gnn = parse_gnn_content(content)
        if not parsed_gnn:
            logger.error(f"Failed to parse GNN content from {gnn_file_path}. No sections found.")
            return None
            
        discopy_types = gnn_statespace_to_discopy_types(parsed_gnn)
        if not discopy_types:
            logger.warning(f"No DisCoPy types generated from StateSpaceBlock in {gnn_file_path}. Diagram construction might be incomplete.")
            # We might still proceed if connections reference types implicitly, but it's unlikely for a well-formed GNN.
            
        diagram = gnn_connections_to_discopy_diagram(parsed_gnn, discopy_types)
        
        if diagram:
            logger.info(f"Successfully created DisCoPy diagram from GNN file: {gnn_file_path}")
            logger.debug(f"Diagram structure: dom={diagram.dom}, cod={diagram.cod}, #boxes={len(diagram.boxes)}")
        else:
            logger.warning(f"Could not create a DisCoPy diagram from GNN file: {gnn_file_path}. Check Connections section.")
            
        return diagram
        
    except Exception as e:
        logger.error(f"Error converting GNN file {gnn_file_path} to DisCoPy diagram: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    # Example usage for standalone testing of this translator module
    # This requires a dummy GNN file to be present at the specified path.
    
    # Configure basic logging for standalone testing
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create a dummy GNN file for testing
    test_gnn_file_content = """
## ModelName
Test DisCoPy Model

## StateSpaceBlock
# Variable definitions
A[2]
B[3]
C[2]
D # A simple variable

## Connections
# Model connections
A > B
B > C
# D > A # Example of a connection that might cause issues with simple linear assumption if not handled
"""
    dummy_gnn_path = Path("__test_discopy_gnn.md")
    with open(dummy_gnn_path, 'w', encoding='utf-8') as f_dummy:
        f_dummy.write(test_gnn_file_content)
        
    logger.info(f"--- Running Translator Standalone Test with {dummy_gnn_path} ---")
    
    # Test parsing
    parsed_data = parse_gnn_content(test_gnn_file_content)
    logger.info(f"Parsed GNN content: {parsed_data}")
    
    if parsed_data:
        # Test type conversion
        types = gnn_statespace_to_discopy_types(parsed_data)
        logger.info(f"Generated DisCoPy types: {types}")
    
        # Test diagram creation
        diagram = gnn_connections_to_discopy_diagram(parsed_data, types)
        if diagram:
            logger.info(f"Generated DisCoPy diagram: {diagram}")
            logger.info(f"  Diagram DOM: {diagram.dom}, COD: {diagram.cod}")
            logger.info(f"  Diagram Boxes: {diagram.boxes}")
            
            # Try to draw if matplotlib is available
            try:
                from matplotlib import pyplot as plt # type: ignore
                output_image_path = Path("__test_discopy_diagram.png")
                diagram.draw(path=str(output_image_path), show_types=True, figsize=(8,4))
                logger.info(f"Diagram drawn to {output_image_path}")
            except ImportError:
                logger.warning("matplotlib not found, skipping diagram drawing.")
            except Exception as e_draw:
                logger.error(f"Error drawing diagram: {e_draw}")

        else:
            logger.warning("Diagram creation failed in standalone test.")
    else:
        logger.error("Parsing failed in standalone test.")

    # Test the main orchestrator function
    logger.info(f"--- Testing gnn_file_to_discopy_diagram on {dummy_gnn_path} ---")
    overall_diagram = gnn_file_to_discopy_diagram(dummy_gnn_path, verbose=True)
    if overall_diagram:
        logger.info(f"Overall diagram created successfully: {overall_diagram}")
    else:
        logger.error(f"Overall diagram creation failed for {dummy_gnn_path}.")
        
    # Clean up dummy file
    dummy_gnn_path.unlink(missing_ok=True)
    Path("__test_discopy_diagram.png").unlink(missing_ok=True)
    logger.info("--- Standalone Translator Test Finished ---") 