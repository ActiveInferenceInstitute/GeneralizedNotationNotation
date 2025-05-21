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

# Initialize logger early for use in placeholder classes
logger = logging.getLogger(__name__)

# --- Define Placeholder Classes ---
class PlaceholderBase:
    """A base placeholder class for DisCoPy JAX-related components when JAX is not available."""
    def __init__(self, *args, **kwargs):
        # Ensure 'name' is captured for better logging if provided.
        self.name_for_debug = kwargs.get('name', f'UnnamedPlaceholder_{type(self).__name__}')
        logger.debug(f"{self.name_for_debug} initialized with args: {args}, kwargs: {kwargs}")
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        logger.debug(f"{self.name_for_debug} called with args: {args}, kwargs: {kwargs}")
        return self

    def __getattr__(self, name: str):
        logger.debug(f"{self.name_for_debug}.__getattr__ called for '{name}'. Returning placeholder for it.")
        if name in ['dom', 'cod', 'boxes', 'shape', 'array', 'inside']:
             if name == 'boxes': return []
             if name == 'inside': return ()
             if name == 'shape': return ()
             if name == 'array': return numpy.array([])
             new_placeholder_name = f"attr_{name}_on_{self.name_for_debug}"
             return PlaceholderBase(name=new_placeholder_name)
        raise AttributeError(f"'{type(self).__name__}' (placeholder: {self.name_for_debug}) object has no attribute '{name}'")

    def eval(self):
        logger.warning(f"eval() called on {self.name_for_debug}. JAX/DisCoPy matrix components likely not available.")
        return PlaceholderBase(name=f"eval_result_on_{self.name_for_debug}")
    
    def __matmul__(self, other):
        logger.debug(f"{self.name_for_debug} __matmul__ called with {other}")
        return PlaceholderBase(name=f"matmul_result_on_{self.name_for_debug}")

    def __rshift__(self, other):
        logger.debug(f"{self.name_for_debug} __rshift__ called with {other}")
        return PlaceholderBase(name=f"rshift_result_on_{self.name_for_debug}")

class DimPlaceholder(PlaceholderBase):
    """A placeholder class for DisCoPy's Dim when JAX is not available."""
    def __init__(self, *values):
        super().__init__(values=values, name=f"DimPlaceholder({values})")
        self.values = values
        # logger.debug(f"DimPlaceholder initialized with values: {values}") # Covered by super

    def __repr__(self):
        return f"DimPlaceholder({', '.join(map(str, self.values))})"
    
    @property
    def inside(self):
        return self.values

class PlaceholderBackend:
    def __init__(self, name="placeholder_backend"):
        self.name = name
        logger.info(f"Using PlaceholderBackend for '{name}'")

    def __enter__(self):
        logger.debug(f"Entering PlaceholderBackend context for {self.name}")
        class PlaceholderNp:
            def __getattr__(self, attr_name):
                logger.debug(f"PlaceholderNp: Attempting to access '{attr_name}'")
                # Simplified for brevity, can be expanded as before
                if attr_name in ['array', 'float32', 'complex64', 'zeros', 'ones', 'eye', 'normal', 'uniform']:
                    def placeholder_np_method(*args, **kwargs):
                        logger.debug(f"PlaceholderNp.{attr_name} called with args: {args}, kwargs: {kwargs}")
                        if attr_name == 'array': return numpy.array(args[0]) if args else numpy.array([])
                        if attr_name == 'float32': return 'float32'
                        if attr_name == 'complex64': return 'complex64'
                        # For functions like zeros, ones, normal, uniform, they often take a shape.
                        # Returning a placeholder that itself might be an empty array or a specific placeholder.
                        return PlaceholderBase(name=f"placeholder_np_{attr_name}_result")
                    return placeholder_np_method
                raise AttributeError(f"PlaceholderNp has no attribute '{attr_name}'")
        return PlaceholderNp()

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.debug(f"Exiting PlaceholderBackend context for {self.name}")

    def __call__(self, backend_name: str):
        logger.debug(f"PlaceholderBackend called with backend_name: {backend_name}. This instance is {self.name}")
        return self
# --- End Placeholder Classes ---

# --- Initialize Global Names and Flags ---
JAX_CORE_AVAILABLE = False
DISCOPY_MATRIX_AVAILABLE = False
JAX_AVAILABLE = False # Overall flag for JAX + DisCoPy matrix readiness

jax = None # Placeholder for jax module
jnp = None # Placeholder for jax.numpy module

# DisCoPy monoidal components
Ty = PlaceholderBase(name="Ty_placeholder")

# DisCoPy tensor components (will be tried first for non-JAX, or as fallback)
TensorDim = DimPlaceholder() # Placeholder for discopy.tensor.Dim
TensorBox = PlaceholderBase(name="TensorBox_placeholder")
TensorDiagram = PlaceholderBase(name="TensorDiagram_placeholder")
TensorId = PlaceholderBase(name="TensorId_placeholder")

# DisCoPy matrix components (tried if JAX is available)
MatrixDim = DimPlaceholder()    # Placeholder for discopy.matrix.Dim
MatrixBox = PlaceholderBase(name="MatrixBox_placeholder")
MatrixDiagram = PlaceholderBase(name="MatrixDiagram_placeholder")
MatrixId = PlaceholderBase(name="MatrixId_placeholder")
Matrix = PlaceholderBase(name="Matrix_data_placeholder") # For discopy.matrix.Matrix (data container)

# Default global Dim to a placeholder. It will be updated based on imports.
Dim = DimPlaceholder()

# Backend
discopy_backend = PlaceholderBackend() # Placeholder for discopy.matrix.backend
# --- End Initialization of Global Names ---

# --- Attempt Imports (This section will be populated by subsequent edits) ---

# 1. Try to import discopy.monoidal.Ty
try:
    from discopy.monoidal import Ty as Ty_actual
    Ty = Ty_actual # Update global Ty if import is successful
    logger.info("Successfully imported Ty from discopy.monoidal.")
except ImportError as e_ty:
    logger.warning(f"Failed to import Ty from discopy.monoidal: {e_ty}. Global Ty remains {type(Ty)}.")
    # Ty remains its placeholder value if import fails

# 2. Try to import JAX core
try:
    import sys # For logging sys.path, though already imported earlier if this is run multiple times
    logger.debug(f"Attempting to import JAX. sys.path: {sys.path}")
    import jax as jax_actual
    import jax.numpy as jnp_actual
    jax = jax_actual # Update global jax
    jnp = jnp_actual # Update global jnp
    JAX_CORE_AVAILABLE = True
    logger.info("JAX and jax.numpy imported successfully.")
except ImportError as e_jax_core:
    logger.warning(f"Failed to import JAX core (jax, jax.numpy): {e_jax_core}. JAX features will be unavailable.")
    # jax, jnp remain None (their initial placeholder value), JAX_CORE_AVAILABLE remains False

# 3. Conditional import of DisCoPy tensor and matrix components
if JAX_CORE_AVAILABLE:
    logger.info("JAX core is available. Attempting to import DisCoPy matrix components and then tensor components.")
    try:
        from discopy.matrix import Dim as MatrixDim_actual, Box as MatrixBox_actual, Diagram as MatrixDiagram_actual, Id as MatrixId_actual, Matrix as Matrix_actual, backend as backend_actual
        
        Dim = MatrixDim_actual                 # Global Dim uses matrix.Dim when JAX is on
        MatrixBox = MatrixBox_actual
        MatrixDiagram = MatrixDiagram_actual
        MatrixId = MatrixId_actual
        Matrix = Matrix_actual
        discopy_backend = backend_actual
        
        DISCOPY_MATRIX_AVAILABLE = True
        JAX_AVAILABLE = True # Overall: JAX core + DisCoPy matrix are ready
        logger.info("Successfully imported DisCoPy matrix components. JAX is fully available for DisCoPy.")

    except ImportError as e_discopy_matrix:
        logger.warning(f"JAX core is available, but DisCoPy matrix components failed to import: {e_discopy_matrix}. Matrix features will use placeholders. JAX_AVAILABLE (overall) remains False.")
        # DISCOPY_MATRIX_AVAILABLE remains False, JAX_AVAILABLE remains False
        # Matrix*, Matrix, discopy_backend, and global Dim (which would have been matrix.Dim) remain placeholders.
        # Global Dim will be attempted by tensor import next.

    # Always attempt to import tensor components if JAX_CORE_AVAILABLE, for separate use or as fallback for Dim
    try:
        from discopy.tensor import Dim as TensorDim_actual, Box as TensorBox_actual, Diagram as TensorDiagram_actual, Id as TensorId_actual
        TensorDim = TensorDim_actual # Store specifically as TensorDim
        TensorBox = TensorBox_actual
        TensorDiagram = TensorDiagram_actual
        TensorId = TensorId_actual
        logger.info("Successfully imported discopy.tensor components (alongside JAX core attempt).")
        if not DISCOPY_MATRIX_AVAILABLE: # If matrix failed, global Dim falls back to tensor.Dim
            Dim = TensorDim_actual
            logger.info("DisCoPy matrix import failed; global Dim is now discopy.tensor.Dim.")
    except ImportError as e_tensor_with_jax:
        logger.warning(f"Failed to import discopy.tensor components (JAX core attempt was made): {e_tensor_with_jax}. Tensor components will use placeholders.")
        # TensorDim, TensorBox, TensorDiagram, TensorId remain placeholders.
        # If DISCOPY_MATRIX_AVAILABLE is also False, Dim remains its original placeholder.

else: # JAX_CORE_AVAILABLE is False
    logger.info("JAX core not available. Attempting to import only discopy.tensor components.")
    # JAX_AVAILABLE remains False, DISCOPY_MATRIX_AVAILABLE remains False
    # Matrix*, Matrix, discopy_backend remain placeholders
    try:
        from discopy.tensor import Dim as TensorDim_actual, Box as TensorBox_actual, Diagram as TensorDiagram_actual, Id as TensorId_actual
        Dim = TensorDim_actual             # Global Dim uses tensor.Dim
        TensorDim = TensorDim_actual       # Store specifically as TensorDim
        TensorBox = TensorBox_actual
        TensorDiagram = TensorDiagram_actual
        TensorId = TensorId_actual
        logger.info("Successfully imported discopy.tensor components (JAX not available).")
    except ImportError as e_discopy_tensor_no_jax:
        logger.warning(f"JAX core not available AND discopy.tensor components also failed to import: {e_discopy_tensor_no_jax}. All DisCoPy components will use placeholders.")
        # Dim, TensorDim, TensorBox, TensorDiagram, TensorId remain placeholders.

# Ensure logger is defined for the rest of the module
if 'logger' not in globals(): # Should be redundant
    logger = logging.getLogger(__name__)

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
                        box_name, dom_spec, cod_spec, init_str = parts
                        try:
                            initializer = json.loads(init_str)
                        except json.JSONDecodeError:
                            initializer = init_str 
                        
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

    # Regex changed to be more permissive for content within brackets for Ty name creation.
    # Also allows for Greek letters like π in variable names.
    # Using (.*?) for bracket content to ensure compilation, will refine later if needed.
    var_pattern = re.compile(r"^\s*([a-zA-Z_π][\w_π]*)\s*(?:\x5B([^\x5D]*)\x5D)?\s*(?:#.*)?$")

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
            if dims_str is not None: # Check if brackets were present, even if empty
                parsed_dims_list = _parse_dims_str(dims_str)
                # For Ty, we just make a descriptive name. Actual dims are for Dim.
                type_name = f"{var_name}[{','.join(map(str, parsed_dims_list))}]" if parsed_dims_list else f"{var_name}[]"
            
            types[var_name] = Ty(type_name)
            logger.debug(f"Created DisCoPy type: Ty('{type_name}') for GNN variable '{var_name}'")
        else:
            logger.warning(f"Could not parse StateSpaceBlock line: '{line}'. Skipping.")
            
    return types

def gnn_statespace_to_discopy_dims(parsed_gnn: dict) -> dict[str, Dim]:
    """
    Converts GNN StateSpaceBlock entries into DisCoPy Dim objects.
    Returns a dictionary mapping variable names to Dim objects.
    Requires JAX_AVAILABLE to be True.
    """
    if not JAX_AVAILABLE:
        logger.error("JAX or DisCoPy's matrix module not available. Cannot create Dim objects.")
        return {}

    dims_map = {}
    statespace_lines = parsed_gnn.get("StateSpaceBlock", [])
    if not statespace_lines:
        logger.warning("StateSpaceBlock not found or empty. No DisCoPy Dim objects will be created.")
        return dims_map

    # Pattern for variable names and optional dimensions: VarName[dim1,dim2,...] or VarName[dim] or VarName
    # Dimensions are comma-separated integers. If no dims, defaults to Dim(1).
    var_pattern = re.compile(r"^\s*([a-zA-Z_π][\w_π]*)\s*(?:\x5B([^\x5D]*)\x5D)?\s*(?:#.*)?$")


    for line in statespace_lines:
        line = line.strip()
        if not line or line.startswith("#"):  # Skip empty lines and comments
            continue

        match = var_pattern.match(line)
        if match:
            var_name = match.group(1)
            dims_str_content = match.group(2) # Content of the brackets e.g. "3,1,type=int" or "type=A" or None
            
            actual_dims = _parse_dims_str(dims_str_content) # Handles None, empty string, and parsing complex content
            
            dims_map[var_name] = Dim(*actual_dims)
            logger.debug(f"Created DisCoPy Dim: Dim(*{actual_dims}) for GNN variable '{var_name}' from dims_str_content '{dims_str_content}'")
        else:
            logger.warning(f"Could not parse StateSpaceBlock line for Dim: '{line}'. Skipping.")
            
    return dims_map

def gnn_connections_to_discopy_diagram(parsed_gnn: dict, types: dict[str, Ty]) -> Optional[TensorDiagram]:
    """
    Converts GNN Connections into a DisCoPy Diagram (specifically, a discopy.tensor.Diagram).
    Currently supports simple directed connections (A > B).
    """
    # Check if TensorDiagram, TensorBox, TensorId are placeholders
    if any(isinstance(comp, PlaceholderBase) for comp in [TensorDiagram, TensorBox, TensorId]): # Simpler check for instance
        logger.error("TensorDiagram, TensorBox, or TensorId is a Placeholder. Cannot create tensor-based DisCoPy diagram. DisCoPy tensor import likely failed.")
        return None

    connections_lines = parsed_gnn.get("Connections", [])
    if not connections_lines:
        logger.warning("Connections section not found or empty. Cannot create DisCoPy diagram.")
        return None

    diagram: TensorDiagram = TensorId() # Start with a TensorId for tensor diagrams, explicitly type hint diagram
    
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
                    if var_name not in types:
                        logger.warning(f"Unknown variable '{var_name}' in connection: '{line}'. Skipping connection.")
                        all_vars_valid = False
                        break
                if not all_vars_valid:
                    break
            if not all_vars_valid:
                continue
            
            # Create source and target types (tensor product if multiple)
            if len(source_vars) == 1:
                dom_type = types[source_vars[0]]
            elif len(source_vars) > 1:
                current_dom = types[source_vars[0]]
                for i in range(1, len(source_vars)):
                    current_dom = current_dom @ types[source_vars[i]]
                dom_type = current_dom
            else: # No source vars, use empty Ty
                dom_type = Ty()

            if len(target_vars) == 1:
                cod_type = types[target_vars[0]]
            elif len(target_vars) > 1:
                current_cod = types[target_vars[0]]
                for i in range(1, len(target_vars)):
                    current_cod = current_cod @ types[target_vars[i]]
                cod_type = current_cod
            else: # No target vars, use empty Ty
                cod_type = Ty()
            
            source_name_part = "_".join(source_vars)
            target_name_part = "_".join(target_vars)
            box_name = f"{source_name_part}_to_{target_name_part}"
            
            box = TensorBox(box_name, dom_type, cod_type)
            logger.debug(f"Created DisCoPy TensorBox: Box('{box_name}', dom={dom_type}, cod={cod_type})")
            
            # Simple sequential composition for now
            if diagram.dom == Ty() and diagram.cod == Ty() and not diagram.boxes: # First box, Ty() is the domain/codomain of TensorId()
                 diagram = box
            elif diagram.cod == dom_type: # Chainable
                 diagram = diagram >> box
            else:
                # This indicates a more complex structure (e.g. parallel wires or new starting chain)
                # For now, we will log a warning and try to append it as a new parallel component
                # This is a placeholder for more sophisticated diagram construction.
                logger.warning(f"Connection from \'{source_content}\' to \'{target_content}\' (Box dom={dom_type}, cod={cod_type}) does not directly chain with previous diagram codomain ({diagram.cod}). Appending in parallel (basic).")
                # Attempting a parallel composition; this assumes variables are distinct flows if not chained.
                # A more robust solution would analyze the full graph structure.
                try:
                    diagram = diagram @ box 
                except Exception as e_parallel:
                    logger.error(f"Failed to compose Box(\'{box_name}\') in parallel: {e_parallel}. Diagram construction may be incorrect.")
                    return diagram # Return what we have so far
        else:
            logger.warning(f"Could not parse Connections line: '{line}'. Supported format: 'Source > Target'.")

    if diagram.dom == Ty() and diagram.cod == Ty() and not diagram.boxes: # Check if any boxes were actually added
        logger.warning("No valid connections were parsed to form a TensorDiagram.")
        return None
        
    return diagram

def gnn_connections_to_discopy_matrix_diagram(
    parsed_gnn: dict, 
    dims_map: dict[str, Dim], 
    tensor_definitions: dict, 
    prng_key_provider: Optional[Callable[[str], Any]] = None, # e.g., lambda name: jax.random.fold_in(base_key, hash(name))
    default_dtype_str: str = 'float32'
) -> Optional[MatrixDiagram]: # Return type updated
    """
    Converts GNN Connections into a DisCoPy MatrixDiagram using JAX-backed Tensors.
    Tensor data is sourced from tensor_definitions.
    """
    if not JAX_AVAILABLE or not discopy_backend or not jax or not jnp: # Added jax and jnp checks
        logger.error("JAX, JAX-dependent DisCoPy modules (backend, Dim, etc.), or jax/jnp itself are not available. Cannot create MatrixDiagram.")
        return None

    # Further check if MatrixId, MatrixBox, jax, jnp, discopy_backend are not placeholders/None
    if any(isinstance(comp, PlaceholderBase) for comp in [MatrixDiagram, MatrixBox, MatrixId]) or \
       isinstance(discopy_backend, PlaceholderBackend) or \
       jax is None or jnp is None:
        logger.error("Critical JAX/DisCoPy matrix components (MatrixDiagram, MatrixBox, MatrixId, jax, jnp, backend) are placeholders or None. Cannot create MatrixDiagram.")
        return None

    connections_lines = parsed_gnn.get("Connections", [])
    if not connections_lines:
        logger.warning("Connections section not found or empty. Cannot create DisCoPy MatrixDiagram.")
        return None

    diagram: MatrixDiagram = MatrixId() # Start with a MatrixId for matrix diagrams, explicitly type hint
    
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

    # Determine JAX dtype
    try:
        jax_dtype = getattr(jnp, default_dtype_str) if JAX_AVAILABLE and jnp else 'float32' # Fallback for non-JAX
        if JAX_AVAILABLE and jnp and jax_dtype != 'float32' and not hasattr(jnp, default_dtype_str): # Check if it's a valid jnp dtype
            logger.warning(f"JAX dtype '{default_dtype_str}' not found in jnp. Defaulting to jnp.float32.")
            jax_dtype = jnp.float32
    except AttributeError: # Should not happen with JAX_AVAILABLE check but as a safeguard
        logger.warning(f"JAX dtype '{default_dtype_str}' caused AttributeError. Defaulting to 'float32' (for NumPy path) or jnp.float32 (if JAX).")
        jax_dtype = jnp.float32 if JAX_AVAILABLE and jnp else 'float32'


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

            initializer = tensor_def.get("initializer")
            jax_array_data = None
            
            # Ensure dom_dim.inside and cod_dim.inside are tuples for concatenation
            dom_inside_tuple = tuple(dom_dim.inside) if isinstance(dom_dim, DimPlaceholder) else dom_dim.inside
            cod_inside_tuple = tuple(cod_dim.inside) if isinstance(cod_dim, DimPlaceholder) else cod_dim.inside

            if not isinstance(dom_inside_tuple, tuple) or not isinstance(cod_inside_tuple, tuple):
                logger.error(f"Cannot determine box shape for '{box_name_short}'. Expected .inside to be tuples, got {type(dom_inside_tuple)} and {type(cod_inside_tuple)}. Skipping.")
                continue
            
            box_shape = dom_inside_tuple + cod_inside_tuple

            if isinstance(initializer, list): # Direct data
                try:
                    if JAX_AVAILABLE and jnp:
                        jax_array_data = jnp.array(initializer, dtype=jax_dtype).reshape(box_shape)
                    else: # Fallback for non-JAX (e.g. if called by something not checking JAX_AVAILABLE)
                        logger.warning("JAX not available, attempting to create NumPy array for MatrixBox data from list. This path might not be fully supported for MatrixDiagrams.")
                        # MatrixBox itself might still expect a JAX tensor if DisCoPy is in JAX mode,
                        # this is more of a graceful degradation attempt.
                        # The proper fix is to not call this function if JAX is not available.
                        # For now, we make a numpy array, but this won't work if a JAX tensor is strictly required by DisCoPy.
                        import numpy # Local import for this fallback
                        jax_array_data = numpy.array(initializer, dtype=default_dtype_str).reshape(box_shape)

                except Exception as e:
                    logger.error(f"Failed to create JAX array from literal for '{box_name_short}': {e}. Shape: {box_shape}, Init: {initializer}")
                    continue
            elif isinstance(initializer, str):
                if initializer.startswith("load:"):
                    file_path_str = initializer.split(":", 1)[1]
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

            box = MatrixBox(box_name_short, dom_dim, cod_dim, data=jax_array_data)
            logger.debug(f"Created JAX-backed MatrixBox: '{box_name_short}', dom={dom_dim}, cod={cod_dim}, shape={jax_array_data.shape}")
            
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

def gnn_file_to_discopy_diagram(gnn_file_path: Path, verbose: bool = False) -> Optional[TensorDiagram]: # Return type updated
    """
    Orchestrates the conversion of a GNN file to a DisCoPy diagram (TensorDiagram).
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

def gnn_file_to_discopy_matrix_diagram(gnn_file_path: Path, verbose: bool = False, jax_seed: int = 0) -> Optional[MatrixDiagram]: # Return type updated
    """
    Orchestrates the conversion of a GNN file to a DisCoPy MatrixDiagram with JAX-backed tensors.
    Reads the file, parses content (including TensorDefinitions), converts state space to Dims,
    and constructs the MatrixDiagram.
    """
    if not JAX_AVAILABLE or not discopy_backend or not jax or not jnp: # Added jax and jnp checks
        logger.error("JAX, JAX-dependent DisCoPy modules (backend, Dim, etc.), or jax/jnp itself are not available. Cannot create MatrixDiagram.")
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
            discopy_dims = gnn_statespace_to_discopy_dims(parsed_gnn)
            if not discopy_dims:
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
                discopy_dims, 
                tensor_definitions,
                key_provider, # Pass the potentially None key_provider
                default_dtype_str=getattr(jnp, 'float32', 'float32') if JAX_AVAILABLE and jnp else 'float32' # Pass jnp.dtype or fallback
            )
        
        if diagram:
            logger.info(f"Successfully created DisCoPy MatrixDiagram from GNN file: {gnn_file_path}")
            logger.debug(f"MatrixDiagram: dom={diagram.dom}, cod={diagram.cod}, #boxes={len(diagram.boxes)}")
            if diagram.boxes and hasattr(diagram.boxes[0], 'data'):
                first_box_data = diagram.boxes[0].data
                if JAX_AVAILABLE and jnp and isinstance(first_box_data, jnp.ndarray):
                    logger.info(f"  First box data (JAX array): {first_box_data}")
                elif isinstance(first_box_data, numpy.ndarray): # Check for numpy array if JAX not used or as fallback
                    logger.info(f"  First box data (NumPy array): {first_box_data}")
                elif isinstance(first_box_data, PlaceholderBase):
                    logger.info(f"  First box data is a Placeholder: {type(first_box_data)} (likely JAX features not fully available)")
                else:
                    logger.info(f"  First box data (unknown type): {first_box_data}")
            
            # Example evaluation (will run JAX computation if JAX is real, or placeholder eval)
            try:
                # Ensure backend context even for placeholder, as script might try to use it.
                backend_context = discopy_backend('jax') if discopy_backend and not isinstance(discopy_backend, PlaceholderBackend) else None
                if backend_context:
                    with backend_context:
                        eval_result = diagram.eval()
                else: # discopy_backend is a placeholder or None
                    eval_result = diagram.eval() # Call eval on placeholder if needed

                logger.info(f"MatrixDiagram evaluation result: {eval_result}")
                
                actual_eval_array = None
                if hasattr(eval_result, 'array'):
                    actual_eval_array = eval_result.array
                
                if JAX_AVAILABLE and jnp and isinstance(actual_eval_array, jnp.ndarray):
                    logger.info(f"Evaluated tensor data (JAX array): {actual_eval_array}, Shape: {actual_eval_array.shape}, Length: {len(actual_eval_array) if actual_eval_array.ndim > 0 else 1}")
                elif isinstance(actual_eval_array, numpy.ndarray):
                    # Ensure ndim check for numpy arrays too before len()
                    length_info = len(actual_eval_array) if hasattr(actual_eval_array, 'ndim') and actual_eval_array.ndim > 0 else 1
                    shape_info = actual_eval_array.shape if hasattr(actual_eval_array, 'shape') else "N/A"
                    logger.info(f"Evaluated tensor data (NumPy array): {actual_eval_array}, Shape: {shape_info}, Length: {length_info}")
                elif isinstance(actual_eval_array, PlaceholderBase):
                    logger.info(f"Evaluated tensor data is a Placeholder: {type(actual_eval_array)}. Cannot determine shape/length.")
                elif actual_eval_array is not None:
                    logger.info(f"Evaluated tensor data (unknown type): {actual_eval_array}. Cannot determine shape/length.")
                else:
                    logger.info("Evaluation result has no .array attribute or it is None.")

            except Exception as e_eval:
                logger.error(f"Error evaluating MatrixDiagram: {e_eval}", exc_info=True)
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
        if JAX_AVAILABLE and DISCOPY_MATRIX_AVAILABLE and discopy_backend and not isinstance(discopy_backend, PlaceholderBackend): # Ensure backend is available for the context manager
            matrix_diagram = gnn_file_to_discopy_matrix_diagram(dummy_matrix_gnn_path, verbose=True, jax_seed=42)
            if matrix_diagram:
                logger.info(f"Overall MatrixDiagram created: dom={matrix_diagram.dom}, cod={matrix_diagram.cod}")
                logger.info(f"  Diagram Boxes: {matrix_diagram.boxes}")
                if matrix_diagram.boxes and hasattr(matrix_diagram.boxes[0], 'data'):
                    first_box_data = matrix_diagram.boxes[0].data
                    if JAX_AVAILABLE and jnp and isinstance(first_box_data, jnp.ndarray):
                        logger.info(f"  First box data (JAX array): {first_box_data}")
                    elif isinstance(first_box_data, numpy.ndarray): # Check for numpy array if JAX not used or as fallback
                        logger.info(f"  First box data (NumPy array): {first_box_data}")
                    elif isinstance(first_box_data, PlaceholderBase):
                        logger.info(f"  First box data is a Placeholder: {type(first_box_data)} (likely JAX features not fully available)")
                    else:
                        logger.info(f"  First box data (unknown type): {first_box_data}")
            
                # Example evaluation (will run JAX computation if JAX is real, or placeholder eval)
                try:
                    # Ensure backend context even for placeholder, as script might try to use it.
                    backend_context = discopy_backend('jax') if discopy_backend and not isinstance(discopy_backend, PlaceholderBackend) else None
                    if backend_context:
                        with backend_context:
                            eval_result = matrix_diagram.eval()
                    else: # discopy_backend is a placeholder or None
                        eval_result = matrix_diagram.eval() # Call eval on placeholder if needed

                    logger.info(f"MatrixDiagram evaluation result: {eval_result}")
                    
                    actual_eval_array = None
                    if hasattr(eval_result, 'array'):
                        actual_eval_array = eval_result.array
                    
                    if JAX_AVAILABLE and jnp and isinstance(actual_eval_array, jnp.ndarray):
                        logger.info(f"Evaluated tensor data (JAX array): {actual_eval_array}, Shape: {actual_eval_array.shape}, Length: {len(actual_eval_array) if actual_eval_array.ndim > 0 else 1}")
                    elif isinstance(actual_eval_array, numpy.ndarray):
                        # Ensure ndim check for numpy arrays too before len()
                        length_info = len(actual_eval_array) if hasattr(actual_eval_array, 'ndim') and actual_eval_array.ndim > 0 else 1
                        shape_info = actual_eval_array.shape if hasattr(actual_eval_array, 'shape') else "N/A"
                        logger.info(f"Evaluated tensor data (NumPy array): {actual_eval_array}, Shape: {shape_info}, Length: {length_info}")
                    elif isinstance(actual_eval_array, PlaceholderBase):
                        logger.info(f"Evaluated tensor data is a Placeholder: {type(actual_eval_array)}. Cannot determine shape/length.")
                    elif actual_eval_array is not None:
                        logger.info(f"Evaluated tensor data (unknown type): {actual_eval_array}. Cannot determine shape/length.")
                    else:
                        logger.info("Evaluation result has no .array attribute or it is None.")

                except Exception as e_eval:
                    logger.error(f"Error evaluating MatrixDiagram: {e_eval}", exc_info=True)
            else:
                logger.error(f"Overall MatrixDiagram creation failed for {dummy_matrix_gnn_path}.")
        else:
            logger.info("Skipping MatrixDiagram test because JAX or discopy_backend is not available.")
        
        dummy_matrix_gnn_path.unlink(missing_ok=True)
        # Path("__dummy_tensor_data.npy").unlink(missing_ok=True) # If using load
        
    logger.info("--- Standalone Translator Test Finished ---") 