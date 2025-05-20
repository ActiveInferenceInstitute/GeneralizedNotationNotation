"""
GNN to DisCoPy Diagram Translator

This module provides functions to parse GNN files (in their string representation)
and convert them into DisCoPy diagrams.
"""
import re
import logging
from pathlib import Path

from discopy.monoidal import Ty, Box, Diagram, Id # type: ignore
# DisCoPy might not be in the project's direct dependencies for static analysis,
# so we can ignore type errors for it here.

logger = logging.getLogger(__name__)

def parse_gnn_content(gnn_content: str) -> dict:
    """
    Parses the string content of a GNN file into a dictionary of sections.
    Each section's content is a list of non-empty, non-comment lines.
    """
    parsed_data = {}
    current_section_name: str | None = None
    
    # Regex to identify GNN section headers like ## ModelName
    # This regex attempts to be more robust for section titles.
    section_header_pattern = re.compile(r"^##\s*([^#\n]+?)\s*(?:#.*)?$")

    lines = gnn_content.splitlines()

    for line_number, line_content in enumerate(lines):
        stripped_line = line_content.strip()

        # Try to match a section header first
        header_match = section_header_pattern.match(stripped_line) # Match on the stripped line
        if header_match:
            section_title_match = header_match.group(1)
            if section_title_match is not None:
                current_section_name = section_title_match.strip().replace(" ", "")
                if current_section_name not in parsed_data:
                    parsed_data[current_section_name] = []
                logger.debug(f"Found section: {current_section_name} at line {line_number + 1}")
            else:
                # This case should ideally not be reached if the regex is well-defined.
                logger.warning(f"Matched a section header but failed to extract title at line {line_number + 1}: '{stripped_line}'")
                current_section_name = None # Ensure it's reset
            continue # Important: move to the next line after processing a header

        # If it's not a header, and we are inside a section, and it's a content line
        if current_section_name and stripped_line and not stripped_line.startswith("#"):
            # current_section_name should be a valid key if we are here due to header processing.
            if current_section_name in parsed_data:
                parsed_data[current_section_name].append(stripped_line)
            else:
                # This path indicates an issue - current_section_name was set, but not as a key.
                # This implies a logic error if a section name was determined but not added to parsed_data.
                logger.warning(
                    f"Attempting to add line to section '{current_section_name}' which was not "
                    f"initialized in parsed_data. Line: '{stripped_line}'. This may indicate a parsing logic error."
                )
            
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
    # Regex changed to be more permissive for content within brackets for Ty name creation.
    var_pattern = re.compile(r"^([a-zA-Z_][\w_]*)(?:\s*\[([^\]]+)\])?(?:\s*#.*)?$")

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
    
    parsed_connections = [] # Store as (source_vars_list, target_vars_list)

    # Regex for connections.
    
    # Pattern for a single variable name: e.g., "VarName"
    var_id_pattern = r"[a-zA-Z_]\\w*"
    
    # Pattern for a list of one or more comma-separated variable names: 
    # e.g., "Var1", "Var1, Var2", "Var1, Var2, Var3"
    # This pattern itself does not match surrounding parentheses.
    var_list_content_pattern = var_id_pattern + r"(?:\\s*,\\s*" + var_id_pattern + r")*"

    # Pattern for a block of text that forms one side of a connection (source or target).
    # This matches EITHER a parenthesized list OR a non-parenthesized list.
    # Example: matches "( Var1, Var2 )" OR "Var1, Var2".
    # The content matched by this (e.g., "( Var1, Var2 )" or "Var1, Var2") 
    # is then captured by the main connection pattern.
    single_side_block_pattern = (
        r"(?:\\s*\\(\\s*" + var_list_content_pattern + r"\\s*\\)\\s*|"  # Option 1: ( list )
        r"\\s*" + var_list_content_pattern + r"\\s*)"                  # Option 2: list
    )
            
    # Final connection pattern string.
    # It captures the entire block for the source (group 1) and target (group 2).
    conn_pattern_str = (
        r"^\\s*(" + single_side_block_pattern + r")\\s*" +
        r"(?:>|->|-)\\s*" +
        r"(" + single_side_block_pattern + r")\\s*(?:#.*)?$"
    )
    conn_pattern = re.compile(conn_pattern_str)

    def parse_vars_from_group(group_str: str) -> list[str]:
        # Remove parentheses and split by comma, then strip whitespace
        cleaned_str = group_str.strip()
        if cleaned_str.startswith("(") and cleaned_str.endswith(")"):
            cleaned_str = cleaned_str[1:-1]
        return [v.strip() for v in cleaned_str.split(',') if v.strip()]

    for line in connections_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        
        match = conn_pattern.match(line)
        if match:
            source_group_str = match.group(1)
            target_group_str = match.group(2)

            source_vars = parse_vars_from_group(source_group_str)
            target_vars = parse_vars_from_group(target_group_str)

            if not source_vars or not target_vars:
                logger.warning(f"Empty source or target variables after parsing connection: \'{line}\'. Skipping.")
                continue

            # Validate all variables exist in types
            all_vars_valid = True
            for var_list in [source_vars, target_vars]:
                for var_name in var_list:
                    if var_name not in types:
                        logger.warning(f"Unknown variable \'{var_name}\' in connection: \'{line}\'. Skipping connection.")
                        all_vars_valid = False
                        break
                if not all_vars_valid:
                    break
            if not all_vars_valid:
                continue
            
            # Create source and target types (tensor product if multiple)
            dom_type = types[source_vars[0]] if len(source_vars) == 1 else Id()
            if len(source_vars) > 1:
                current_dom = types[source_vars[0]]
                for i in range(1, len(source_vars)):
                    current_dom = current_dom @ types[source_vars[i]]
                dom_type = current_dom
            elif not source_vars: # Should not happen due to check above, but defensive
                logger.warning(f"Source variable list is empty for line \'{line}\'. Skipping box creation.")
                continue


            cod_type = types[target_vars[0]] if len(target_vars) == 1 else Id()
            if len(target_vars) > 1:
                current_cod = types[target_vars[0]]
                for i in range(1, len(target_vars)):
                    current_cod = current_cod @ types[target_vars[i]]
                cod_type = current_cod
            elif not target_vars: # Should not happen
                logger.warning(f"Target variable list is empty for line \'{line}\'. Skipping box creation.")
                continue
            
            source_name_part = "_".join(source_vars)
            target_name_part = "_".join(target_vars)
            box_name = f"{source_name_part}_to_{target_name_part}"
            
            box = Box(box_name, dom_type, cod_type)
            logger.debug(f"Created DisCoPy Box: Box(\'{box_name}\', dom={dom_type}, cod={cod_type})")
            
            # Simple sequential composition for now
            if diagram.dom == Id().dom and diagram.cod == Id().cod : # First box
                 diagram = box
            elif diagram.cod == dom_type: # Chainable
                 diagram = diagram >> box
            else:
                # This indicates a more complex structure (e.g. parallel wires or new starting chain)
                # For now, we will log a warning and try to append it as a new parallel component
                # This is a placeholder for more sophisticated diagram construction.
                logger.warning(f"Connection from \'{source_group_str}\' to \'{target_group_str}\' (Box dom={dom_type}, cod={cod_type}) does not directly chain with previous diagram codomain ({diagram.cod}). Appending in parallel (basic).")
                # Attempting a parallel composition; this assumes variables are distinct flows if not chained.
                # A more robust solution would analyze the full graph structure.
                try:
                    diagram = diagram @ box 
                except Exception as e_parallel:
                    logger.error(f"Failed to compose Box(\'{box_name}\') in parallel: {e_parallel}. Diagram construction may be incorrect.")
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