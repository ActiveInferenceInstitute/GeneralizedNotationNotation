"""
GNN to DisCoPy Diagram Translator

This module provides functions to parse GNN files (in their string representation)
and convert them into DisCoPy diagrams.
"""
import re
import logging
from pathlib import Path
import json # For parsing tensor initializers
import numpy # For loading .npy files
from typing import Union, List, Dict, Any, Callable, Optional # Added typing imports
import sys # Added sys import for sys.path logging
import functools

# Initialize logger early for use in placeholder classes
logger = logging.getLogger(__name__)

# Global variables for DisCoPy components, JAX, and jax.numpy
# These will be replaced by actual imports if available, otherwise they remain placeholders.
Dim, Box, Diagram, Id, Swap, Cup, Cap, Spider, Functor, Matrix = None, None, None, None, None, None, None, None, None, None
Ty, Word = None, None 
jax, jnp = None, None 
discopy_backend = None # For discopy.matrix.backend

TENSOR_COMPONENTS_AVAILABLE = False
TY_AVAILABLE = False
JAX_CORE_AVAILABLE = False # Specific to JAX itself
DISCOPY_MATRIX_MODULE_AVAILABLE = False # Specific to discopy.matrix module
JAX_AVAILABLE = False # Overall flag for JAX-backed DisCoPy readiness

# Placeholder base class (defined once)
class PlaceholderBase:
    """A base placeholder class for DisCoPy JAX-related components when JAX/DisCoPy is not available."""
    def __init__(self, *args, **kwargs):
        self.name = kwargs.get('name', f'UnnamedPlaceholder_{type(self).__name__}')
        # logger.debug(f"{self.name} initialized with args: {args}, kwargs: {kwargs}") # Logger might not be init'd here
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        # logger.debug(f"{self.name} called with args: {args}, kwargs: {kwargs}")
        return self

    def __getattr__(self, name: str):
        if name == 'inside': # Common attribute for Dim
            return () 
        # logger.debug(f"{self.name}.__getattr__('{name}') called, returning self.")
        return self
        
    def __repr__(self):
        return f"{type(self).__name__}Placeholder(args={self.args}, kwargs={self.kwargs})"

# --- Define specific placeholder classes ---
class DimPlaceholder(PlaceholderBase): pass
class BoxPlaceholder(PlaceholderBase): pass
class DiagramPlaceholder(PlaceholderBase): pass
class IdPlaceholder(PlaceholderBase): pass
class SwapPlaceholder(PlaceholderBase): pass
class CupPlaceholder(PlaceholderBase): pass
class CapPlaceholder(PlaceholderBase): pass
class SpiderPlaceholder(PlaceholderBase): pass
class FunctorPlaceholder(PlaceholderBase): pass
class MatrixPlaceholder(BoxPlaceholder): pass # Matrix is a type of Box
class TyPlaceholder(PlaceholderBase): pass
class WordPlaceholder(BoxPlaceholder): pass # Word is a type of Box
class JaxPlaceholder:
    def __getattr__(self, name):
        raise AttributeError(f"JAX not available. Cannot access jax.{name}")
class JnpPlaceholder:
    def __getattr__(self, name):
        raise AttributeError(f"JAX (numpy) not available. Cannot access jnp.{name}")
    def array(self, data, dtype=None): # Mock jnp.array
        if isinstance(data, list) and all(isinstance(x, complex) for x in data):
            return [complex(item.real, item.imag) for item in data] # Keep as list of python complex
        return data # return raw data
class DiscopyBackendPlaceholder:
    def __enter__(self): pass
    def __exit__(self, exc_type, exc_val, exc_tb): pass


# Assign placeholders by default
Dim, Box, Diagram, Id, Swap, Cup, Cap, Spider, Functor, Matrix = (
    DimPlaceholder(), BoxPlaceholder(), DiagramPlaceholder(), IdPlaceholder(), SwapPlaceholder(), 
    CupPlaceholder(), CapPlaceholder(), SpiderPlaceholder(), FunctorPlaceholder(), MatrixPlaceholder()
)
Ty, Word = TyPlaceholder(), WordPlaceholder()
jax, jnp = JaxPlaceholder(), JnpPlaceholder()
discopy_backend = DiscopyBackendPlaceholder()


# --- Attempt Imports ---

# 1. Configure logging (early, so import attempts can log)
# Logging setup is done after this import block by the main script normally.
# For direct execution/testing of this module, basic logging can be enabled here.
# We'll rely on the main script's logger for now.

# 2. Try to import discopy.monoidal and discopy.grammar components (Ty, Word)
try:
    from discopy.monoidal import Ty as Ty_actual
    Ty = Ty_actual
    logger.info("Successfully imported Ty from discopy.monoidal.")
except ImportError as e_monoidal_ty:
    logger.warning(f"Failed to import Ty from discopy.monoidal: {e_monoidal_ty}. Ty remains a placeholder.")
    # Ty remains TyPlaceholder

try:
    from discopy.grammar.pregroup import Word as Word_actual
    Word = Word_actual
    logger.info("Successfully imported Word from discopy.grammar.pregroup.")
except ImportError as e_grammar_word:
    logger.warning(f"Failed to import Word from discopy.grammar.pregroup: {e_grammar_word}. Word remains a placeholder.")
    # Word remains WordPlaceholder

# Check for overall availability of Ty and Word
TY_AVAILABLE = not isinstance(Ty, (TyPlaceholder, PlaceholderBase)) and not isinstance(Word, (WordPlaceholder, PlaceholderBase))

# 3. Try to import discopy.tensor components and discopy.matrix.Matrix
# These are essential for JAX backend.
try:
    from discopy.tensor import (
        Dim as Dim_actual, Box as Box_actual, Diagram as Diagram_actual, Id as Id_actual,
        Swap as Swap_actual, Cup as Cup_actual, Cap as Cap_actual, Spider as Spider_actual,
        Functor as Functor_actual
    )
    from discopy.matrix import Matrix as Matrix_actual

    Dim = Dim_actual
    Box = Box_actual
    Diagram = Diagram_actual
    Id = Id_actual
    Swap = Swap_actual
    Cup = Cup_actual
    Cap = Cap_actual
    Spider = Spider_actual
    Functor = Functor_actual
    Matrix = Matrix_actual

    TENSOR_COMPONENTS_AVAILABLE = True
    logger.info("Successfully imported core components from discopy.tensor and discopy.matrix.Matrix.")

except ImportError as e_tensor_matrix:
    logger.error(f"CRITICAL: Failed to import core components from discopy.tensor or discopy.matrix.Matrix: {e_tensor_matrix}")
    logger.error("DisCoPy JAX translation cannot proceed without these. Ensure DisCoPy is correctly installed.")
    # Dim, Box, etc. remain placeholders as set initially.
    TENSOR_COMPONENTS_AVAILABLE = False

# 4. Attempt to import JAX and jax.numpy only if tensor components are available
if TENSOR_COMPONENTS_AVAILABLE:
    try:
        import jax as jax_actual_import
        import jax.numpy as jnp_actual_import
        
        jax = jax_actual_import
        jnp = jnp_actual_import

        _ = jnp.array([1, 2, 3]) # Test JAX
        JAX_CORE_AVAILABLE = True
        logger.info("JAX and jax.numpy imported successfully and tested.")
    except ImportError as e_jax:
        logger.warning(f"JAX or jax.numpy not found, or basic JAX op failed: {e_jax}. JAX-dependent features will be disabled.")
        # jax, jnp remain placeholders
        JAX_CORE_AVAILABLE = False
    except Exception as e_jax_other:
        logger.warning(f"An error occurred during JAX import or test: {e_jax_other}. JAX-dependent features will be disabled.")
        # jax, jnp remain placeholders
        JAX_CORE_AVAILABLE = False
else:
    logger.warning("Import of JAX and jax.numpy skipped as TENSOR_COMPONENTS_AVAILABLE is False.")
    # jax, jnp remain placeholders
    JAX_CORE_AVAILABLE = False


# 5. Try to import discopy.matrix.backend
# This is needed for `with discopy_backend(\'jax\'):`
if JAX_CORE_AVAILABLE: # Only if JAX itself is loaded
    try:
        from discopy.matrix import backend as backend_actual
        discopy_backend = backend_actual
        DISCOPY_MATRIX_MODULE_AVAILABLE = True # Signifies backend context manager is available
        logger.info("Successfully imported backend from discopy.matrix.")
    except ImportError as e_matrix_backend:
        logger.warning(f"JAX core is available, but failed to import backend from discopy.matrix: {e_matrix_backend}.")
        # discopy_backend remains placeholder
        DISCOPY_MATRIX_MODULE_AVAILABLE = False
else:
    logger.warning("Import of discopy.matrix.backend skipped as JAX_CORE_AVAILABLE is False.")
    # discopy_backend remains placeholder
    DISCOPY_MATRIX_MODULE_AVAILABLE = False

# 6. Determine overall JAX_AVAILABLE readiness for DisCoPy
if TENSOR_COMPONENTS_AVAILABLE and JAX_CORE_AVAILABLE and DISCOPY_MATRIX_MODULE_AVAILABLE and TY_AVAILABLE:
    JAX_AVAILABLE = True
    logger.info("JAX_AVAILABLE is True: JAX and all required DisCoPy components are loaded for JAX backend.")
else:
    JAX_AVAILABLE = False
    logger.warning("JAX_AVAILABLE is False. One or more critical components (DisCoPy tensor/matrix/monoidal, JAX core, DisCoPy backend) failed to load.")


# --- End Import Block ---

def _convert_json_to_complex_array(data: Any) -> Any:
    """
    Recursively converts a list structure (from JSON) potentially containing [real, imag] pairs
    into a structure of Python complex numbers, suitable for jnp.array.
    """
    if isinstance(data, list):
        # Check if it's a [real, imag] pair that should be converted to a complex number
        if len(data) == 2 and isinstance(data[0], (int, float)) and isinstance(data[1], (int, float)):
            # This condition identifies a list of two numbers as a complex pair.
            return complex(data[0], data[1])
        else:
            # It's a list, but not a [real, imag] pair itself.
            # Recursively process its elements.
            return [_convert_json_to_complex_array(item) for item in data]
    # If data is not a list (e.g., it's a number, string, or already a complex number from a previous step), return it as is.
    return data

def _parse_dims_str(dims_str: str | None) -> list[int]:
    """Helper to parse a comma-separated string of dimensions into a list of ints.
    Extracts leading numeric dimensions, ignoring non-numeric parts like 'type=...'.
    """
    if not dims_str:
        return [1] # Default to Dim(1) if no dimensions specified

    parsed_dims = []
    # Split by comma, then attempt to convert each part to int.
    # Only add to parsed_dims if it's a valid integer.
    # This will effectively ignore parts like "type=A" or "foo=bar".
    for part in dims_str.split(','):
        stripped_part = part.strip()
        if not stripped_part: # Skip empty parts
            continue
        try:
            # Try to convert to int. If it works, it's a dimension.
            num = int(stripped_part)
            parsed_dims.append(num)
        except ValueError:
            # This part is not a simple integer (e.g., "type=A"). Log and ignore for dims.
            logger.debug(f"Ignoring non-integer part '{stripped_part}' while parsing dimensions from '{dims_str}'")
            
    if not parsed_dims: # If no numeric dimensions were found (e.g., just "type=A" or empty string after filtering)
        return [1] # Default to Dim(1)
    return parsed_dims

def parse_gnn_content(gnn_content: str) -> dict:
    """
    Parses the string content of a GNN file into a dictionary of sections.
    Each section's content is a list of non-empty, non-comment lines.
    Parses ## TensorDefinitions section specifically.
    """
    parsed_data: Dict[str, Union[List[str], Dict[str, Any]]] = {} # Explicitly typed
    current_section_name: Optional[str] = None
    
    section_header_pattern = re.compile(r"^##\s*([^#\n]+?)\s*(?:#.*)?$")

    lines = gnn_content.splitlines()

    for line_number, line_content in enumerate(lines):
        stripped_line = line_content.strip()

        header_match = section_header_pattern.match(stripped_line)
        if header_match:
            section_title_match = header_match.group(1)
            if section_title_match is not None:
                current_section_name = section_title_match.strip().replace(" ", "")
                if current_section_name not in parsed_data:
                    # Special handling for TensorDefinitions to store structured data
                    if current_section_name == "TensorDefinitions":
                        parsed_data[current_section_name] = {} # Always init as dict
                    else:
                        parsed_data[current_section_name] = []
                logger.debug(f"Found section: {current_section_name} at line {line_number + 1}")
            else:
                logger.warning(f"Matched a section header but failed to extract title at line {line_number + 1}: '{stripped_line}'")
                current_section_name = None
            continue

        if current_section_name and stripped_line and not stripped_line.startswith("#"):
            if current_section_name == "TensorDefinitions":
                # Ensure this section is a dictionary before assigning
                if not isinstance(parsed_data.get(current_section_name), dict):
                    logger.error(f"Section 'TensorDefinitions' was not initialized as a dict. This is a bug.")
                    # Force it to be a dict to prevent further errors, though data might be lost
                    parsed_data[current_section_name] = {}
                
                target_dict_for_tensor_defs = parsed_data[current_section_name]
                # Now we are sure target_dict_for_tensor_defs is a Dict an can perform assignment.
                # However, the type checker might still complain because parsed_data[current_section_name] is a Union.
                # A cast or more refined type structure might be needed if this persists.
                if isinstance(target_dict_for_tensor_defs, dict):
                    parts = [p.strip() for p in stripped_line.split('|')]
                    if len(parts) == 4:
                        box_name, dom_spec, cod_spec, init_str_raw = parts # Renamed
                        
                        init_str_for_json_parse = init_str_raw.strip()
                        potential_json_literal = False

                        # Check if it's a double-quoted string that might contain JSON
                        if init_str_for_json_parse.startswith('"') and init_str_for_json_parse.endswith('"'):
                            inner_str = init_str_for_json_parse[1:-1]
                            # Check if the inner content looks like a JSON array or object
                            if ((inner_str.startswith('[') and inner_str.endswith(']')) or
                                (inner_str.startswith('{') and inner_str.endswith('}'))):
                                init_str_for_json_parse = inner_str # Use the unquoted inner string
                                potential_json_literal = True
                        # Check if it's an unquoted string that already looks like JSON
                        elif ((init_str_for_json_parse.startswith('[') and init_str_for_json_parse.endswith(']')) or
                              (init_str_for_json_parse.startswith('{') and init_str_for_json_parse.endswith('}'))):
                            potential_json_literal = True

                        # Attempt to parse as JSON only if it's a potential literal and not "load:"
                        if potential_json_literal and not init_str_raw.strip().startswith("load:"):
                            try:
                                initializer_from_json = json.loads(init_str_for_json_parse)
                                logger.debug(f"TensorDef: Parsed init_str for '{box_name}' with json.loads. Type: {type(initializer_from_json)}, Value: {str(initializer_from_json)[:200]}")
                                initializer = initializer_from_json
                            except json.JSONDecodeError as e_json:
                                logger.warning(f"TensorDef: json.loads failed for '{box_name}' after attempting to unquote/prepare. Error: {e_json}. init_str_for_json_parse was: {init_str_for_json_parse}")
                                initializer = init_str_raw.strip() # Fallback to stripped raw string
                        else:
                            # Not a JSON literal (e.g. "load:...", "random_normal", or a simple string name for a function)
                            logger.debug(f"TensorDef: Initializer for '{box_name}' not treated as direct JSON: {init_str_raw}")
                            initializer = init_str_raw.strip() # Store it stripped
                        
                        target_dict_for_tensor_defs[box_name] = {
                            "dom_spec": dom_spec, 
                            "cod_spec": cod_spec, 
                            "initializer": initializer
                        }
                    else:
                        logger.warning(
                            f"Could not parse TensorDefinitions line {line_number + 1}: '{stripped_line}'. "
                            "Expected format: BoxName | DomDimsStr | CodDimsStr | InitializerStr"
                        )
                else:
                    # This case should ideally not be reached due to prior checks and initialization
                    logger.error(f"Type error: 'TensorDefinitions' section resolved to non-dict type before assignment attempt at line {line_number + 1}.")

            elif current_section_name in parsed_data:
                 section_content = parsed_data[current_section_name]
                 if isinstance(section_content, list):
                    section_content.append(stripped_line)
                 else: # Should not happen if initialized correctly
                    logger.error(f"Section '{current_section_name}' is not a list. Line: '{stripped_line}'")
            else:
                logger.warning(
                    f"Attempting to add line to section '{current_section_name}' which was not "
                    f"initialized in parsed_data. Line: '{stripped_line}'. This may indicate a parsing logic error."
                )
            
    if logger.isEnabledFor(logging.DEBUG):
        for section, content_lines_or_dict in parsed_data.items():
            if isinstance(content_lines_or_dict, list):
                logger.debug(f"  Section '{section}' has {len(content_lines_or_dict)} relevant lines.")
                if content_lines_or_dict:
                    logger.debug(f"    First line of '{section}': '{content_lines_or_dict[0]}'")
            elif isinstance(content_lines_or_dict, dict):
                 logger.debug(f"  Section '{section}' has {len(content_lines_or_dict)} definitions.")
                 if content_lines_or_dict:
                    first_key = next(iter(content_lines_or_dict))
                    logger.debug(f"    First definition in '{section}': '{first_key}': {content_lines_or_dict[first_key]}")
    
    #logger.debug(f"Final parsed_data before return: {json.dumps(parsed_data, indent=2)[:1000]}")
    return parsed_data

def _get_discopy_dim_from_gnn_spec(dim_spec_str: str, logger: logging.Logger) -> Optional[Dim]:
    """
    Parses a GNN dimension specification string (e.g., "2", "2,3", or even an empty string for Dim())
    and returns a DisCoPy Dim object.
    Returns None if Dim is a placeholder or parsing fails critically.
    """
    if isinstance(Dim, PlaceholderBase):
        logger.error("DisCoPy Dim component is a Placeholder. Cannot create Dim objects from spec.")
        return None

    if not dim_spec_str: # Empty string or None implies Dim() or Dim(1) depending on convention
        logger.debug("Empty dim_spec_str provided, creating Dim().")
        try:
            return Dim() # DisCoPy's Dim() is the unit for tensor product, effectively Dim(1) in some contexts
        except Exception as e:
            logger.error(f"Error creating Dim() for empty dim_spec_str: {e}")
            return None

    # Remove any type annotations like "type=A" or extra spaces before splitting
    # Focus on numeric parts. e.g. "2, type=A, 3" -> "2, 3"
    cleaned_spec = re.sub(r"type=[^,]+", "", dim_spec_str) # Remove type=...
    cleaned_spec = re.sub(r"[^0-9,]", "", cleaned_spec)   # Remove non-numeric, non-comma chars
    
    dim_values_str = [s.strip() for s in cleaned_spec.split(',') if s.strip()]

    if not dim_values_str: # If after cleaning, no numbers remain (e.g. "type=A")
        logger.debug(f"No numeric dimensions found in '{dim_spec_str}' after cleaning. Creating Dim().")
        try:
            return Dim()
        except Exception as e:
            logger.error(f"Error creating Dim() for cleaned non-numeric spec '{dim_spec_str}': {e}")
            return None
            
    try:
        # Convert valid string numbers to integers
        dims_as_ints = [int(val) for val in dim_values_str] # Renamed to avoid conflict
        if not dims_as_ints: # Should be caught by 'if not dim_values_str' earlier, but as a safeguard
             logger.debug(f"No dimensions to create Dim object from '{dim_spec_str}'. Defaulting to Dim().")
             return Dim()
        return Dim(*dims_as_ints)
    except ValueError as ve:
        logger.error(f"Invalid character in dimension specification '{dim_spec_str}' (cleaned: '{cleaned_spec}'). Error: {ve}")
        return None
    except Exception as e:
        logger.error(f"Error creating Dim from spec '{dim_spec_str}' (parsed as {dims_as_ints if 'dims_as_ints' in locals() else 'unknown'}): {e}")
        return None

def gnn_statespace_to_discopy_dims_map(parsed_gnn: dict) -> dict[str, Dim]:
    """
    Converts GNN StateSpaceBlock entries into a dictionary mapping variable names to DisCoPy Dim objects.
    Handles parsing of dimensions like VarName[dim1,dim2,...] or VarName.
    """
    if isinstance(Dim, PlaceholderBase):
        logger.error("DisCoPy Dim component is a Placeholder. Cannot create Dim objects.")
        return {}

    dims_map: dict[str, Dim] = {}
    statespace_lines = parsed_gnn.get("StateSpaceBlock", [])
    if not statespace_lines:
        logger.warning("StateSpaceBlock not found or empty. No DisCoPy Dim objects will be created.")
        return dims_map

    # Pattern for variable names and optional dimensions: VarName[dim1,dim2,...] or VarName[dim]
    # Dimensions are comma-separated integers. Ignores content like type=...
    var_pattern = re.compile(r"^\s*([a-zA-Z_π][\w_π]*)\s*(?:\x5B([^\x5D]*)\x5D)?\s*(?:#.*)?$")

    for line in statespace_lines:
        line_content = line.strip()
        if not line_content or line_content.startswith("#"):
            continue

        match = var_pattern.match(line_content)
        if not match:
            logger.warning(f"Could not parse variable from StateSpaceBlock line: '{line_content}'. Skipping.")
            continue

        var_name = match.group(1)
        dims_str = match.group(2)
        
        parsed_dims_list = _parse_dims_str(dims_str) # _parse_dims_str returns list[int], e.g., [1] or [2,3]

        try:
            # Create Dim object using the parsed integer dimensions
            # Dim(*[1]) creates Dim(1), Dim(*[2,3]) creates Dim(2,3)
            current_dim = Dim(*parsed_dims_list)
            dims_map[var_name] = current_dim
            logger.debug(f"Created DisCoPy Dim: {current_dim} for GNN variable '{var_name}'")
        except Exception as e_dim_creation: # Catch potential errors during Dim creation
            logger.error(f"Error creating DisCoPy Dim for '{var_name}' with dims {parsed_dims_list}: {e_dim_creation}")
            continue
            
    return dims_map

def gnn_connections_to_discopy_diagram(parsed_gnn: dict, dims_map: dict[str, Dim]) -> Optional[Diagram]:
    """
    Converts GNN Connections into a DisCoPy Diagram (from discopy.tensor).
    This version creates a diagram with abstract boxes (no tensor data).
    """
    # Check if Diagram, Box, Id, Dim are placeholders (essential for basic diagram structure)
    # Ty is also fundamental for DisCoPy, even if not directly used for Box dom/cod here, its availability implies DisCoPy loaded.
    if any(isinstance(comp, PlaceholderBase) for comp in [Diagram, Box, Id, Dim, Ty]): 
        logger.error("Core DisCoPy components (Diagram, Box, Id, Dim, or Ty) are Placeholders. Cannot create tensor-based DisCoPy diagram. DisCoPy import likely failed.")
        return None

    connections_lines = parsed_gnn.get("Connections", [])
    if not connections_lines:
        logger.warning("Connections section not found or empty. Cannot create DisCoPy diagram.")
        return None

    diagram: Diagram = Id() # Start with an Id for tensor diagrams, explicitly type hint diagram
    
    # Regex patterns (copied from original, ensure they are correct for this context)
    var_id_pattern = r"[a-zA-Z_π][\w_π]*"
    
    # Pattern for a list of one or more comma-separated variable names: 
    # e.g., "Var1", "Var1, Var2", "Var1, Var2, Var3"
    # This pattern itself does not match surrounding parentheses.
    var_list_content_pattern = var_id_pattern + r"(?:\s*,\s*" + var_id_pattern + r")*"

    # Pattern for a block of text that forms one side of a connection (source or target).
    # This matches EITHER a parenthesized list OR a non-parenthesized list.
    # Example: matches "( Var1, Var2 )" OR "Var1, Var2".
    single_side_block_pattern = (
        # Option 1: ( list of vars ) - captures list content in a group
        # Using \( and \) for literal parentheses.
        r"(?:\s*\(\s*(" + var_list_content_pattern + r")\s*\)\s*|" +
        # Option 2: list of vars 
        r"\s*(" + var_list_content_pattern + r")\s*)"
    )
            
    # Final connection pattern string.
    # It captures the source block (group 1 for parenthesized, group 2 for non-parenthesized)
    # and target block (group 3 for parenthesized, group 4 for non-parenthesized)
    # Supports '>', '->', '-' as connectors.
    # Ignores '=' for assignments for now.
    conn_pattern_str = (
        # Source part: matches either a parenthesized list or a direct list/single var
        # Using \( and \) for literal parentheses.
        r"^\s*(?:\(\s*(" + var_list_content_pattern + r")\s*\)|(" + var_list_content_pattern + r"))\s*" +
        # Connector
        r"(?:>|->|-)\s*" +
        # Target part: matches either a parenthesized list or a direct list/single var
        r"(?:\(\s*(" + var_list_content_pattern + r")\s*\)|(" + var_list_content_pattern + r"))\s*(?:#.*)?$"
    )
    conn_pattern = re.compile(conn_pattern_str)

    # Pattern for assignment-like connections (e.g., G=ExpectedFreeEnergy, t=Time)
    # These are treated as non-diagrammatic for now. Ensuring this uses \w and \s.
    assignment_pattern_str = r"^\s*([a-zA-Z_π][\w_π]*)\s*=\s*([^#]+?)\s*(?:#.*)?$"
    assignment_pattern = re.compile(assignment_pattern_str)

    def parse_vars_from_group(group_str: str | None) -> list[str]:
        if not group_str:
            return []
        # Handles cases where the string might already be clean or needs splitting.
        return [v.strip() for v in group_str.split(',') if v.strip()]

    for line in connections_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        
        assignment_match = assignment_pattern.match(line)
        if assignment_match:
            var_name = assignment_match.group(1)
            value_assigned = assignment_match.group(2).strip()
            logger.info(f"Parsed assignment: '{var_name}' = '{value_assigned}'. Not creating a DisCoPy box for this.")
            # Optionally, these could be stored as annotations on the diagram or nodes if relevant
            continue # Move to next line

        match = conn_pattern.match(line)
        if match:
            # Determine actual source and target strings from the four possible capture groups
            # Groups are: 1 (source parenthesized), 2 (source direct), 3 (target parenthesized), 4 (target direct)
            source_str_paren = match.group(1)
            source_str_direct = match.group(2)
            target_str_paren = match.group(3)
            target_str_direct = match.group(4)

            source_content = source_str_paren if source_str_paren else source_str_direct
            target_content = target_str_paren if target_str_paren else target_str_direct
            
            if not source_content or not target_content:
                logger.warning(f"Could not determine source or target content from connection line: '{line}'. Skipping.")
                continue

            source_vars = parse_vars_from_group(source_content)
            target_vars = parse_vars_from_group(target_content)

            if not source_vars or not target_vars:
                logger.warning(f"Empty source or target variables after parsing connection: \'{line}\'. Skipping.")
                continue

            # Validate all variables exist in types
            all_vars_valid = True
            for var_list in [source_vars, target_vars]:
                for var_name in var_list:
                    if var_name not in dims_map:
                        logger.warning(f"Unknown variable '{var_name}' (not in dims_map) in connection: '{line}'. Skipping connection.")
                        all_vars_valid = False
                        break
                if not all_vars_valid:
                    break
            if not all_vars_valid:
                continue
            
            # Determine domain and codomain types (which are Dim objects)
            if len(source_vars) == 1:
                dom_dim = dims_map[source_vars[0]]
            elif len(source_vars) > 1:
                # Tensor product of Dim objects
                try:
                    dom_dim = functools.reduce(lambda a, b: a @ b, [dims_map[v] for v in source_vars])
                except TypeError: # If @ is not defined for PlaceholderBase or initial reduce object
                    logger.error(f"Cannot compute tensor product for domain with vars {source_vars} due to placeholder Dim objects. Skipping connection.")
                    continue
            else: # No source vars, use empty Dim (identity for tensor product, usually Dim(1))
                dom_dim = Dim() 

            if len(target_vars) == 1:
                cod_dim = dims_map[target_vars[0]]
            elif len(target_vars) > 1:
                try:
                    cod_dim = functools.reduce(lambda a, b: a @ b, [dims_map[v] for v in target_vars])
                except TypeError:
                    logger.error(f"Cannot compute tensor product for codomain with vars {target_vars} due to placeholder Dim objects. Skipping connection.")
                    continue
            else: # No target vars, use empty Dim
                cod_dim = Dim()
            
            source_name_part = "_".join(source_vars)
            target_name_part = "_".join(target_vars)
            box_name = f"{source_name_part}_to_{target_name_part}"
            
            box = Box(box_name, dom_dim, cod_dim) # Box expects Dim for dom/cod in discopy.tensor
            logger.debug(f"Created DisCoPy Box: Box('{box_name}', dom={dom_dim}, cod={cod_dim})")
            
            # Simple sequential composition for now
            if diagram.dom == Dim() and diagram.cod == Dim() and not diagram.boxes: # First box, Dim() is the domain/codomain of Id()
                diagram = box
            elif diagram.cod == dom_dim: # Chainable
                 diagram = diagram >> box
            else:
                # This indicates a more complex structure (e.g. parallel wires or new starting chain)
                # For now, we will log a warning and try to append it as a new parallel component
                # This is a placeholder for more sophisticated diagram construction.
                logger.warning(f"Connection from \'{source_content}\' to \'{target_content}\' (Box dom={dom_dim}, cod={cod_dim}) does not directly chain with previous diagram codomain ({diagram.cod}). Appending in parallel (basic).")
                # Attempting a parallel composition; this assumes variables are distinct flows if not chained.
                # A more robust solution would analyze the full graph structure.
                try:
                    diagram = diagram @ box 
                except Exception as e_parallel:
                    logger.error(f"Failed to compose Box(\'{box_name}\') in parallel: {e_parallel}. Diagram construction may be incorrect.")
                    return diagram # Return what we have so far
        else:
            logger.warning(f"Could not parse Connections line: '{line}'. Supported format: 'Source > Target'.")

    if diagram.dom == Dim() and diagram.cod == Dim() and not diagram.boxes: # Check if any boxes were actually added
        logger.warning("No valid connections were parsed to form a Diagram.")
        return None
        
    return diagram

def gnn_connections_to_discopy_matrix_diagram(
    parsed_gnn: dict, 
    dims_map: dict[str, Dim], 
    tensor_definitions: dict, 
    prng_key_provider: Optional[Callable[[str], Any]] = None, 
    default_dtype_str: str = 'float32'
) -> Optional[Diagram]:
    """
    Converts GNN Connections into a DisCoPy Diagram, where boxes are populated with Matrix objects
    (from discopy.matrix) containing JAX-backed Tensors if JAX is available.
    If JAX is not available or tensor data is missing, boxes may be abstract or use placeholders.
    """
    if not JAX_AVAILABLE: # Check the overall JAX_AVAILABLE flag
        logger.error("JAX or essential DisCoPy components for matrix operations are not available. Cannot create a JAX-backed MatrixDiagram.")
        return None
    
    # Also ensure Diagram, Box, Id, Dim, Matrix, discopy_backend are not placeholders
    if any(isinstance(comp, PlaceholderBase) for comp in [Diagram, Box, Id, Dim, Matrix]) or \
       isinstance(discopy_backend, DiscopyBackendPlaceholder) or \
       jax is None or jnp is None: # jax and jnp should be non-None if JAX_AVAILABLE is True
        logger.error("Critical JAX/DisCoPy components (Diagram, Box, Id, Dim, Matrix, jax, jnp, backend) are placeholders or None. Cannot create MatrixDiagram.")
        return None

    connections_lines = parsed_gnn.get("Connections", [])
    if not connections_lines:
        logger.warning("Connections section not found or empty. Cannot create DisCoPy MatrixDiagram.")
        return None

    diagram: Diagram = Id() # Start with an Id, explicitly type hint
    
    # Regex patterns (ensure they are correct for this context)
    var_id_pattern = r"[a-zA-Z_π][\w_π]*"
    
    # Pattern for a list of one or more comma-separated variable names: 
    # e.g., "Var1", "Var1, Var2", "Var1, Var2, Var3"
    # This pattern itself does not match surrounding parentheses.
    var_list_content_pattern = var_id_pattern + r"(?:\\s*,\\s*" + var_id_pattern + r")*"

    # Pattern for a block of text that forms one side of a connection (source or target).
    # This matches EITHER a parenthesized list OR a non-parenthesized list.
    # Example: matches "( Var1, Var2 )" OR "Var1, Var2".
    single_side_block_pattern = (
        # Option 1: ( list of vars ) - captures list content in a group
        # Using \( and \) for literal parentheses.
        r"(?:\s*\(\s*(" + var_list_content_pattern + r")\s*\)\s*|" +
        # Option 2: list of vars 
        r"\s*(" + var_list_content_pattern + r")\s*)"
    )
            
    # Final connection pattern string.
    # It captures the source block (group 1 for parenthesized, group 2 for non-parenthesized)
    # and target block (group 3 for parenthesized, group 4 for non-parenthesized)
    # Supports '>', '->', '-' as connectors.
    # Ignores '=' for assignments for now.
    conn_pattern_str = (
        # Source part: matches either a parenthesized list or a direct list/single var
        # Using \( and \) for literal parentheses.
        r"^\s*(?:\(\s*(" + var_list_content_pattern + r")\s*\)|(" + var_list_content_pattern + r"))\s*" +
        # Connector
        r"(?:>|->|-)\s*" +
        # Target part: matches either a parenthesized list or a direct list/single var
        r"(?:\(\s*(" + var_list_content_pattern + r")\s*\)|(" + var_list_content_pattern + r"))\s*(?:#.*)?$"
    )
    conn_pattern = re.compile(conn_pattern_str)
    assignment_pattern_str = r"^\\s*([a-zA-Z_π][\\w_π]*)\\s*=\\s*([^#]+?)\\s*(?:#.*)?$"
    assignment_pattern = re.compile(assignment_pattern_str)

    def parse_vars_from_group(group_str: str | None) -> list[str]:
        if not group_str: return []
        return [v.strip() for v in group_str.split(',') if v.strip()]

    # Initialize with an empty diagram or appropriate identity
    diagram: Matrix = Id(Dim(1)) # Start with Identity on Dim(1) for matrix diagrams

    tensor_definitions = parsed_gnn.get("TensorDefinitions", {})
    # Get the raw dtype definition, which might be a string or a type object
    raw_default_dtype = tensor_definitions.get("default_dtype", "float32") 

    # Ensure default_dtype_str is a string name
    if isinstance(raw_default_dtype, str):
        default_dtype_str = raw_default_dtype
    elif hasattr(raw_default_dtype, '__name__'): # Check if it's a type object with a name
        default_dtype_str = raw_default_dtype.__name__
    else:
        logger.warning(f"Unrecognized default_dtype format: {raw_default_dtype}. Defaulting to 'float32'.")
        default_dtype_str = "float32"
    
    logger.debug(f"Processed default_dtype_str: {default_dtype_str}")

    # Attempt to get the actual JAX dtype object using the string name
    if JAX_AVAILABLE and jnp and hasattr(jnp, default_dtype_str):
        jax_dtype = getattr(jnp, default_dtype_str)
    else:
        # Fallback if JAX not available or dtype string not a jnp attribute
        # Using the string name itself as a placeholder if actual jnp dtype can't be resolved.
        # This might be okay if it's only used for numpy array creation later, 
        # or if the actual tensor_def provides a valid jax_array_data directly.
        jax_dtype = default_dtype_str 
        logger.debug(f"JAX/jnp not fully available or '{default_dtype_str}' not in jnp. Using '{jax_dtype}' as jax_dtype placeholder.")

    for line_idx, line in enumerate(connections_lines):
        line = line.strip()
        if not line or line.startswith("#"): continue

        assignment_match = assignment_pattern.match(line)
        if assignment_match:
            # Assignments are not typically part of MatrixDiagram structure, logged and skipped.
            logger.info(f"Skipping assignment in Connections for MatrixDiagram: '{line}'")
            continue

        match = conn_pattern.match(line)
        if match:
            source_str_paren, source_str_direct, target_str_paren, target_str_direct = match.groups()
            source_content = source_str_paren if source_str_paren else source_str_direct
            target_content = target_str_paren if target_str_paren else target_str_direct

            if not source_content or not target_content:
                logger.warning(f"Could not determine source or target content from connection line: '{line}'. Skipping.")
                continue

            source_vars = parse_vars_from_group(source_content)
            target_vars = parse_vars_from_group(target_content)

            if not source_vars or not target_vars:
                logger.warning(f"Empty source or target variables for MatrixDiagram: '{line}'. Skipping.")
                continue

            all_vars_valid = True
            for var_list in [source_vars, target_vars]:
                for var_name in var_list:
                    if var_name not in dims_map:
                        logger.warning(f"Unknown variable '{var_name}' (no Dim found) in connection: '{line}'. Skipping.")
                        all_vars_valid = False; break
                if not all_vars_valid: break
            if not all_vars_valid: continue
            
            dom_dim = dims_map[source_vars[0]] if len(source_vars) == 1 else Dim()
            if len(source_vars) > 1:
                current_dom_dim = dims_map[source_vars[0]]
                for i in range(1, len(source_vars)): current_dom_dim = current_dom_dim @ dims_map[source_vars[i]]
                dom_dim = current_dom_dim
            elif not source_vars: continue

            cod_dim = dims_map[target_vars[0]] if len(target_vars) == 1 else Dim()
            if len(target_vars) > 1:
                current_cod_dim = dims_map[target_vars[0]]
                for i in range(1, len(target_vars)): current_cod_dim = current_cod_dim @ dims_map[target_vars[i]]
                cod_dim = current_cod_dim
            elif not target_vars: continue
            
            box_name_short = f"{'_'.join(source_vars)}_to_{'_'.join(target_vars)}"
            box_name_full_line = line # Or some unique identifier for the box from this line

            # Retrieve tensor data
            tensor_def = tensor_definitions.get(box_name_short) # Try short name first
            if not tensor_def:
                 tensor_def = tensor_definitions.get(box_name_full_line) # Try full line if specific
            
            if not tensor_def:
                logger.warning(f"No tensor definition found for box '{box_name_short}' or full line '{box_name_full_line}'. Skipping box.")
                continue

            # Determine data type for JAX array for this specific box
            raw_box_dtype = tensor_def.get("dtype", default_dtype_str) # Inherit default if not specified

            if isinstance(raw_box_dtype, str):
                box_dtype_str = raw_box_dtype
            elif hasattr(raw_box_dtype, '__name__'):
                box_dtype_str = raw_box_dtype.__name__
            else:
                logger.warning(f"Unrecognized dtype format for box '{box_name_short}': {raw_box_dtype}. Using default: '{default_dtype_str}'.")
                box_dtype_str = default_dtype_str
            
            logger.debug(f"Processed box_dtype_str for '{box_name_short}': {box_dtype_str}")

            if JAX_AVAILABLE and jnp and hasattr(jnp, box_dtype_str):
                current_jax_dtype = getattr(jnp, box_dtype_str)
            else:
                current_jax_dtype = box_dtype_str # Fallback to string name
                logger.debug(f"JAX/jnp not fully available or '{box_dtype_str}' not in jnp for box '{box_name_short}'. Using '{current_jax_dtype}' as current_jax_dtype placeholder.")

            initializer = tensor_def.get("initializer")
            jax_array_data = None
            
            # Ensure dom_dim.inside and cod_dim.inside are tuples for concatenation
            dom_inside_tuple = tuple(dom_dim.inside) if isinstance(dom_dim, DimPlaceholder) else dom_dim.inside
            cod_inside_tuple = tuple(cod_dim.inside) if isinstance(cod_dim, DimPlaceholder) else cod_dim.inside

            if not isinstance(dom_inside_tuple, tuple) or not isinstance(cod_inside_tuple, tuple):
                logger.error(f"Cannot determine box shape for '{box_name_short}'. Expected .inside to be tuples, got {type(dom_inside_tuple)} and {type(cod_inside_tuple)}. Skipping.")
                continue
            
            box_shape = dom_inside_tuple + cod_inside_tuple

            if isinstance(initializer, list): # Direct data from JSON
                logger.debug(f"MatrixDiagram: Initializer for '{box_name_short}' IS a list. Processing with _convert_json_to_complex_array.")
                try:
                    # Convert [real, imag] pairs to complex numbers before creating JAX array
                    processed_initializer = _convert_json_to_complex_array(initializer)
                    
                    if JAX_AVAILABLE and jnp:
                        # Ensure jax_dtype is a JAX dtype object if complex data is detected
                        # This is a bit of a heuristic; ideally, dtype comes from GNN or is more robustly inferred.
                        if any(isinstance(x, complex) for x in processed_initializer) or \
                           (isinstance(processed_initializer, list) and processed_initializer and any(isinstance(x, complex) for row in processed_initializer if isinstance(row,list) for x in row)) : # check nested for complex
                            if isinstance(current_jax_dtype, str) and 'complex' not in current_jax_dtype.lower():
                                logger.debug(f"Initializer for '{box_name_short}' contains complex numbers. Overriding dtype to jnp.complex64 from {current_jax_dtype}.")
                                current_jax_dtype = jnp.complex64 if hasattr(jnp, 'complex64') else 'complex64'
                            elif not hasattr(current_jax_dtype, 'is_complex'): # if it's already a jax dtype, check if complex
                                if jnp.issubdtype(current_jax_dtype, jnp.complexfloating) == False:
                                     logger.debug(f"Initializer for '{box_name_short}' contains complex numbers. Overriding dtype to jnp.complex64 from {current_jax_dtype}.")
                                     current_jax_dtype = jnp.complex64 if hasattr(jnp, 'complex64') else 'complex64'


                        jax_array_data = jnp.array(processed_initializer, dtype=current_jax_dtype).reshape(box_shape)
                    else: 
                        logger.warning("JAX not available, attempting to create NumPy array for MatrixBox data from list. This path might not be fully supported for MatrixDiagrams.")
                        # MatrixBox itself might still expect a JAX tensor if DisCoPy is in JAX mode,
                        # this is more of a graceful degradation attempt.
                        # The proper fix is to not call this function if JAX is not available.
                        # For now, we make a numpy array, but this won't work if a JAX tensor is strictly required by DisCoPy.
                        import numpy # Local import for this fallback
                        jax_array_data = numpy.array(processed_initializer, dtype=default_dtype_str).reshape(box_shape)

                except Exception as e:
                    logger.error(f"Failed to create JAX array from literal for '{box_name_short}': {e}. Shape: {box_shape}, Init: {processed_initializer}")
                    continue
            elif isinstance(initializer, str):
                logger.debug(f"MatrixDiagram: Initializer for '{box_name_short}' IS a string: '{initializer}'. Checking for load/random.")
                if initializer.startswith("load:"):
                    file_path_str = initializer[len("load:"):]
                    try:
                        # Ensure path is absolute or resolve relative to GNN file (if context available)
                        # For now, assume path is resolvable as is or relative to where script runs
                        loaded_np_array = numpy.load(file_path_str)
                        jax_array_data = jnp.array(loaded_np_array, dtype=jax_dtype).reshape(box_shape)
                    except Exception as e:
                        logger.error(f"Failed to load JAX array from '{file_path_str}' for '{box_name_short}': {e}")
                        continue
                elif initializer.startswith("random_normal:") or initializer == "random_normal":
                    if JAX_AVAILABLE and jax and prng_key_provider:
                        key_suffix = initializer.split(":", 1)[1] if ":" in initializer else str(line_idx)
                        current_key = prng_key_provider(f"{box_name_short}_{key_suffix}")
                        jax_array_data = jax.random.normal(current_key, shape=box_shape, dtype=jax_dtype)
                    else:
                        logger.warning(f"JAX or PRNG key provider not available for random_normal initializer of '{box_name_short}'. Skipping.")
                        continue
                elif initializer.startswith("random_uniform:") or initializer == "random_uniform":
                    if JAX_AVAILABLE and jax and prng_key_provider:
                        key_suffix = initializer.split(":", 1)[1] if ":" in initializer else str(line_idx)
                        current_key = prng_key_provider(f"{box_name_short}_{key_suffix}")
                        jax_array_data = jax.random.uniform(current_key, shape=box_shape, dtype=jax_dtype)
                    else:
                        logger.warning(f"JAX or PRNG key provider not available for random_uniform initializer of '{box_name_short}'. Skipping.")
                        continue
                # Add more random initializers (e.g., glorot, he) as needed
                else:
                    logger.warning(f"Unknown string initializer for '{box_name_short}': '{initializer}'. Skipping.")
                    continue
            else: # E.g. dict for future complex initializers, or number for scalar broadcast
                if isinstance(initializer, (int, float)): # Scalar broadcast
                    try:
                        if JAX_AVAILABLE and jnp:
                            jax_array_data = jnp.full(box_shape, initializer, dtype=jax_dtype)
                        else:
                             logger.warning("JAX not available, attempting to create NumPy full array for MatrixBox data from scalar. This path might not be fully supported.")
                             import numpy # Local import
                             jax_array_data = numpy.full(box_shape, initializer, dtype=default_dtype_str)

                    except Exception as e:
                         logger.error(f"Failed to broadcast scalar for '{box_name_short}': {e}. Shape: {box_shape}, Scalar: {initializer}")
                         continue
                else:
                    logger.warning(f"Unsupported initializer type for '{box_name_short}': {type(initializer)}. Skipping.")
                    continue

            if jax_array_data is None:
                logger.warning(f"Could not initialize data for box '{box_name_short}'. Skipping.")
                continue

            # Create a discopy.matrix.Matrix object to hold the JAX array
            # This Matrix IS a Box, so it can be directly used in the diagram.
            try:
                # Ensure dom_dim.inside and cod_dim.inside are tuples for concatenation
                # This was a previous source of error, Dim should handle .inside correctly if not placeholder
                box_shape_tuple = tuple(getattr(dom_dim, 'inside', ())) + tuple(getattr(cod_dim, 'inside', ())) 
                
                if not all(isinstance(d, int) for d in box_shape_tuple):
                    logger.error(f"Box '{box_name_short}' has non-integer dimensions in shape: {box_shape_tuple}. Dom: {dom_dim}, Cod: {cod_dim}. Skipping.")
                    continue
                
                reshaped_jax_array = jax_array_data.reshape(box_shape_tuple) # Reshape based on combined dom and cod dims

                # Create the Matrix (which is a Box)
                logger.debug(f"Preparing to create Matrix for box '{box_name_short}'.")
                logger.debug(f"  dom_dim: {dom_dim} (type: {type(dom_dim)})")
                logger.debug(f"  cod_dim: {cod_dim} (type: {type(cod_dim)})")
                if hasattr(reshaped_jax_array, 'shape') and hasattr(reshaped_jax_array, 'dtype'):
                    logger.debug(f"  reshaped_jax_array: shape={reshaped_jax_array.shape}, dtype={reshaped_jax_array.dtype} (type: {type(reshaped_jax_array)})")
                else:
                    logger.debug(f"  reshaped_jax_array: (type: {type(reshaped_jax_array)}), attributes like shape/dtype might be missing.")

                box = Matrix(dom_dim, cod_dim, reshaped_jax_array) # discopy.matrix.Matrix
                # Manually set the name for the Matrix/Box if not automatically handled by constructor in all versions
                box.name = box_name_short

            except Exception as e_matrix_creation:
                logger.error(f"Error creating discopy.matrix.Matrix for box '{box_name_short}': {e_matrix_creation}. Dom: {dom_dim}, Cod: {cod_dim}, Data shape: {jax_array_data.shape if hasattr(jax_array_data, 'shape') else 'N/A'}")
                continue

            # Use box_name_short for logging as box.name might not be set if box is a placeholder or construction failed early
            box_data_shape_log = getattr(box.data, 'shape', 'unknown') if hasattr(box, 'data') and box.data is not None else 'no data'
            logger.debug(f"Created JAX-backed Matrix: '{box_name_short}', dom={box.dom}, cod={box.cod}, data_shape={box_data_shape_log}")

            if diagram.dom == Dim() and diagram.cod == Dim() and not diagram.boxes: # First box, Dim() is the domain/codomain of Id()
                 diagram = box
            elif diagram.cod == dom_dim: # Chainable
                 diagram = diagram >> box
            else:
                logger.warning(f"Connection for MatrixBox '{box_name_short}' (dom={dom_dim}) doesn't match diagram cod ({diagram.cod}). Appending in parallel.")
                try:
                    diagram = diagram @ box
                except Exception as e_parallel:
                    logger.error(f"Failed to compose MatrixBox '{box_name_short}' in parallel: {e_parallel}. Diagram construction may be incorrect.")
                    return diagram 
        else:
            logger.warning(f"Could not parse Connections line for MatrixDiagram: '{line}'.")

    if diagram.dom == Dim() and diagram.cod == Dim() and not diagram.boxes:
        logger.warning("No valid connections were parsed to form a MatrixDiagram.")
        return None
        
    return diagram

def gnn_file_to_discopy_diagram(gnn_file_path: Path, verbose: bool = False) -> Optional[Diagram]:
    """
    Orchestrates the conversion of a GNN file to a DisCoPy diagram (tensor.Diagram).
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
            
        discopy_dims_map = gnn_statespace_to_discopy_dims_map(parsed_gnn)
        if not discopy_dims_map:
            logger.warning(f"No DisCoPy Dims generated from StateSpaceBlock in {gnn_file_path}.")
            # Proceeding as some diagrams might not need explicit dims (e.g. only names)
            
        diagram = gnn_connections_to_discopy_diagram(parsed_gnn, discopy_dims_map)
        
        if diagram:
            logger.info(f"Successfully created DisCoPy diagram from GNN file: {gnn_file_path}")
            logger.debug(f"Diagram structure: dom={diagram.dom}, cod={diagram.cod}, #boxes={len(diagram.boxes) if hasattr(diagram, 'boxes') else 'N/A'}")
        else:
            logger.warning(f"Could not create a DisCoPy diagram from GNN file: {gnn_file_path}. Check Connections section.")
            
        return diagram
        
    except Exception as e:
        logger.error(f"Error converting GNN file {gnn_file_path} to DisCoPy diagram: {e}", exc_info=True)
        return None

def gnn_file_to_discopy_matrix_diagram(gnn_file_path: Path, verbose: bool = False, jax_seed: int = 0) -> Optional[Diagram]:
    """
    Orchestrates the conversion of a GNN file to a DisCoPy Diagram with JAX-backed matrices.
    Reads the file, parses content (including TensorDefinitions), converts state space to Dims,
    and constructs the MatrixDiagram.
    """
    if not JAX_AVAILABLE: # Check the overall JAX_AVAILABLE flag
        logger.error("JAX or essential DisCoPy components for matrix operations are not available. Cannot create a JAX-backed MatrixDiagram.")
        return None
    
    if verbose: logger.setLevel(logging.DEBUG)
    else: logger.setLevel(logging.INFO)
    
    logger.info(f"Attempting to convert GNN file to DisCoPy MatrixDiagram: {gnn_file_path}")
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
        
        # Set JAX backend for DisCoPy matrix operations
        with discopy_backend('jax'):
            discopy_dims_map = gnn_statespace_to_discopy_dims_map(parsed_gnn)
            if not discopy_dims_map:
                logger.warning(f"No DisCoPy Dims generated from StateSpaceBlock in {gnn_file_path}.")
            
            tensor_definitions = parsed_gnn.get("TensorDefinitions", {})
            if not tensor_definitions:
                logger.warning(f"No 'TensorDefinitions' section found in {gnn_file_path}. Boxes may not be initialized.")

            # PRNG key provider for random initializations - only if JAX is available
            key_provider = None
            if JAX_AVAILABLE and jax:
                jax_random_module_local = getattr(jax, 'random', None)
                if jax_random_module_local and hasattr(jax_random_module_local, 'PRNGKey') and hasattr(jax_random_module_local, 'fold_in'):
                    base_key = jax_random_module_local.PRNGKey(jax_seed)
                    # Capture jax_random_module_local in the closure
                    def _key_provider_impl(name_suffix: str, _jrm=jax_random_module_local) -> Any: 
                        hashed_suffix = hash(name_suffix) & ((1 << 32) -1)
                        return _jrm.fold_in(base_key, hashed_suffix) # Use captured _jrm
                    key_provider = _key_provider_impl
                elif jax_random_module_local:
                    logger.warning("jax.random module is available, but PRNGKey or fold_in attribute is missing. Cannot create PRNG key provider.")
                else:
                    logger.warning("jax.random module is not available within JAX. Cannot create PRNG key provider.")
            else:
                logger.info("JAX is not available, PRNG key provider will not be created.")

            diagram = gnn_connections_to_discopy_matrix_diagram(
                parsed_gnn, 
                discopy_dims_map, 
                tensor_definitions,
                key_provider, # Pass the potentially None key_provider
                default_dtype_str=getattr(jnp, 'float32', 'float32') if JAX_AVAILABLE and jnp else 'float32' # Pass jnp.dtype or fallback
            )
        
        if diagram:
            logger.info(f"Successfully created DisCoPy MatrixDiagram from GNN file: {gnn_file_path}")
            logger.debug(f"MatrixDiagram: dom={diagram.dom}, cod={diagram.cod}, #boxes={len(diagram.boxes) if hasattr(diagram, 'boxes') else 'N/A'}")
            if hasattr(diagram, 'boxes') and diagram.boxes and hasattr(diagram.boxes[0], 'data') and diagram.boxes[0].data is not None:
                first_box_data = diagram.boxes[0].data
                # Check if it's a JAX array (if jnp is not None and it's an instance)
                if JAX_AVAILABLE and jnp and isinstance(first_box_data, jnp.ndarray):
                    logger.info(f"  First box data (JAX array): {first_box_data}")
                elif isinstance(first_box_data, numpy.ndarray): # Check for numpy array if JAX not used or as fallback
                    logger.info(f"  First box data (NumPy array): {first_box_data}")
                elif isinstance(first_box_data, PlaceholderBase):
                    logger.info(f"  First box data is a Placeholder: {type(first_box_data)} (data: {getattr(first_box_data, 'args', '')})")
                else:
                    logger.info(f"  First box data type: {type(first_box_data)}")
                
                # Evaluation test
                if discopy_backend and not isinstance(discopy_backend, DiscopyBackendPlaceholder):
                    backend_context = discopy_backend('jax')
                    if backend_context:
                        with backend_context:
                            eval_result = diagram.eval()
                            logger.info(f"MatrixDiagram evaluation result (JAX backend): {eval_result}")
                            if hasattr(eval_result, 'array'):
                                logger.info(f"  Evaluation result array: {eval_result.array}")
                    else:
                        logger.warning("Could not obtain JAX backend context for evaluation.")
                else:
                    logger.warning("DisCoPy JAX backend is not available or is a placeholder. Skipping evaluation test.")
                # Log matrix_diagram properties safely
                if not isinstance(matrix_diagram, PlaceholderBase):
                    logger.info(f"MatrixDiagram created: dom={matrix_diagram.dom}, cod={matrix_diagram.cod}, boxes: {len(matrix_diagram.boxes) if hasattr(matrix_diagram, 'boxes') else 'N/A'}")
                    if hasattr(matrix_diagram, 'boxes') and matrix_diagram.boxes and hasattr(matrix_diagram.boxes[0], 'data') and matrix_diagram.boxes[0].data is not None:
                        first_box_data = matrix_diagram.boxes[0].data
                        # Check if it's a JAX array (if jnp is not None and it's an instance)
                        if JAX_AVAILABLE and jnp and isinstance(first_box_data, jnp.ndarray):
                            logger.info(f"  First box data (JAX array): {first_box_data}")
                        elif isinstance(first_box_data, numpy.ndarray): # Check for numpy array if JAX not used or as fallback
                            logger.info(f"  First box data (NumPy array): {first_box_data}")
                        elif isinstance(first_box_data, PlaceholderBase):
                            logger.info(f"  First box data is a Placeholder: {type(first_box_data)} (data: {getattr(first_box_data, 'args', '')})")
                        else:
                            logger.info(f"  First box data type: {type(first_box_data)}")
                    else:
                        logger.info("MatrixDiagram has no boxes or first box has no data (or diagram is a placeholder).")
                else:
                    logger.info(f"MatrixDiagram is a placeholder: {type(matrix_diagram)}")
            else:
                logger.info("MatrixDiagram has no boxes or first box has no data (or diagram is a placeholder).")
        else:
            logger.error(f"Overall MatrixDiagram creation failed for {gnn_file_path}.")
            
        return diagram
        
    except Exception as e:
        logger.error(f"Error converting GNN file {gnn_file_path} to DisCoPy MatrixDiagram: {e}", exc_info=True)
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
        # Test Dim map creation (replaces type conversion)
        dims_map_test = gnn_statespace_to_discopy_dims_map(parsed_data)
        logger.info(f"Generated DisCoPy Dims map: {dims_map_test}")
    
        # Test diagram creation
        diagram_test = gnn_connections_to_discopy_diagram(parsed_data, dims_map_test)
        if diagram_test:
            logger.info(f"Generated DisCoPy diagram: {diagram_test}")
            logger.info(f"  Diagram DOM: {diagram_test.dom}, COD: {diagram_test.cod}")
            logger.info(f"  Diagram Boxes: {diagram_test.boxes}")
            
            # Try to draw if matplotlib is available
            try:
                from matplotlib import pyplot as plt # type: ignore
                output_image_path = Path("__test_discopy_diagram.png")
                diagram_test.draw(path=str(output_image_path), show_types=True, figsize=(8,4))
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
        # Log diagram properties safely, checking if it's not a placeholder
        if not isinstance(overall_diagram, PlaceholderBase):
            logger.info(f"Overall diagram created successfully: {overall_diagram}. Dom: {overall_diagram.dom}, Cod: {overall_diagram.cod}, Boxes: {len(overall_diagram.boxes) if hasattr(overall_diagram, 'boxes') else 'N/A'}")
            if TENSOR_COMPONENTS_AVAILABLE and TY_AVAILABLE and hasattr(overall_diagram, 'draw'): # Draw if real components are available
                try:
                    draw_path = Path("__test_discopy_diagram.png")
                    overall_diagram.draw(path=str(draw_path), show_types=True, figsize=(8,4))
                    logger.info(f"Diagram drawn to {draw_path}")
                except Exception as e_draw:
                    logger.error(f"Error drawing diagram: {e_draw}")
        else:
            logger.info(f"Overall diagram is a placeholder: {type(overall_diagram)}")
    else:
        logger.error(f"Overall diagram creation failed for {dummy_gnn_path}.")
        
    # Clean up dummy file
    dummy_gnn_path.unlink(missing_ok=True)
    Path("__test_discopy_diagram.png").unlink(missing_ok=True)

    # Example for MatrixDiagram (if JAX is available and GNN file is adapted)
    if JAX_AVAILABLE:
        test_gnn_matrix_content = """
## ModelName
Test DisCoPy MatrixDiagram Model

## StateSpaceBlock
A[2]
B[2]
C[2]

## TensorDefinitions
# BoxName | DomSpec (ignored) | CodSpec (ignored) | Initializer
A_to_B    | 2                 | 2                 | [[1.0, 0.0], [0.0, 1.0]]
B_to_C    | 2                 | 2                 | "random_normal:bc_key" 
# C_to_A | 2                 | 2                 | "load:./dummy_tensor_data.npy" # Needs dummy_tensor_data.npy

## Connections
A > B
B > C
# C > A # Cycle
"""
        dummy_matrix_gnn_path = Path("__test_discopy_matrix_gnn.md")
        # numpy.save("__dummy_tensor_data.npy", numpy.array([[0.5,0.5],[0.5,0.5]])) # If using load

        with open(dummy_matrix_gnn_path, 'w', encoding='utf-8') as f_dummy_matrix:
            f_dummy_matrix.write(test_gnn_matrix_content)

        logger.info(f"--- Testing gnn_file_to_discopy_matrix_diagram on {dummy_matrix_gnn_path} ---")
        if JAX_AVAILABLE and DISCOPY_MATRIX_MODULE_AVAILABLE and discopy_backend and not isinstance(discopy_backend, DiscopyBackendPlaceholder): # Ensure backend is available for the context manager
            matrix_diagram = gnn_file_to_discopy_matrix_diagram(dummy_matrix_gnn_path, verbose=True, jax_seed=42)
            if matrix_diagram:
                logger.info(f"MatrixDiagram created: dom={matrix_diagram.dom}, cod={matrix_diagram.cod}, boxes: {len(matrix_diagram.boxes) if hasattr(matrix_diagram, 'boxes') else 'N/A'}")
                if hasattr(matrix_diagram, 'boxes') and matrix_diagram.boxes and hasattr(matrix_diagram.boxes[0], 'data') and matrix_diagram.boxes[0].data is not None:
                    first_box_data = matrix_diagram.boxes[0].data
                    # Check if it's a JAX array (if jnp is not None and it's an instance)
                    if JAX_AVAILABLE and jnp and isinstance(first_box_data, jnp.ndarray):
                        logger.info(f"  First box data (JAX array): {first_box_data}")
                    elif isinstance(first_box_data, numpy.ndarray): # Check for numpy array if JAX not used or as fallback
                        logger.info(f"  First box data (NumPy array): {first_box_data}")
                    elif isinstance(first_box_data, PlaceholderBase):
                        logger.info(f"  First box data is a Placeholder: {type(first_box_data)} (data: {getattr(first_box_data, 'args', '')})")
                    else:
                        logger.info(f"  First box data type: {type(first_box_data)}")
                
                # Evaluation test
                if discopy_backend and not isinstance(discopy_backend, DiscopyBackendPlaceholder):
                    backend_context = discopy_backend('jax')
                    if backend_context:
                        with backend_context:
                            eval_result = matrix_diagram.eval()
                            logger.info(f"MatrixDiagram evaluation result (JAX backend): {eval_result}")
                            if hasattr(eval_result, 'array'):
                                logger.info(f"  Evaluation result array: {eval_result.array}")
                    else:
                        logger.warning("Could not obtain JAX backend context for evaluation.")
                else:
                    logger.warning("DisCoPy JAX backend is not available or is a placeholder. Skipping evaluation test.")
                # Log matrix_diagram properties safely
                if not isinstance(matrix_diagram, PlaceholderBase):
                    logger.info(f"MatrixDiagram created: dom={matrix_diagram.dom}, cod={matrix_diagram.cod}, boxes: {len(matrix_diagram.boxes) if hasattr(matrix_diagram, 'boxes') else 'N/A'}")
                    if hasattr(matrix_diagram, 'boxes') and matrix_diagram.boxes and hasattr(matrix_diagram.boxes[0], 'data') and matrix_diagram.boxes[0].data is not None:
                        first_box_data = matrix_diagram.boxes[0].data
                        # Check if it's a JAX array (if jnp is not None and it's an instance)
                        if JAX_AVAILABLE and jnp and isinstance(first_box_data, jnp.ndarray):
                            logger.info(f"  First box data (JAX array): {first_box_data}")
                        elif isinstance(first_box_data, numpy.ndarray): # Check for numpy array if JAX not used or as fallback
                            logger.info(f"  First box data (NumPy array): {first_box_data}")
                        elif isinstance(first_box_data, PlaceholderBase):
                            logger.info(f"  First box data is a Placeholder: {type(first_box_data)} (data: {getattr(first_box_data, 'args', '')})")
                        else:
                            logger.info(f"  First box data type: {type(first_box_data)}")
                    else:
                        logger.info("MatrixDiagram has no boxes or first box has no data (or diagram is a placeholder).")
                else:
                    logger.info(f"MatrixDiagram is a placeholder: {type(matrix_diagram)}")
            else:
                logger.error(f"Overall MatrixDiagram creation failed for {dummy_matrix_gnn_path}.")
        
        dummy_matrix_gnn_path.unlink(missing_ok=True)
        # Path("__dummy_tensor_data.npy").unlink(missing_ok=True) # If using load
        
    logger.info("--- Standalone Translator Test Finished ---") 