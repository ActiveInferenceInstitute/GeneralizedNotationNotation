"""
Module for rendering GNN specifications to RxInfer.jl Julia scripts.

This module translates a parsed GNN (Generalized Notation Notation) specification
into executable Julia code that uses RxInfer.jl for Bayesian inference.
It leverages information from the RxInfer.jl technical report regarding its
architecture, syntax, and inference paradigms.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Removed: logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Global Configuration & Constants ---
# Default directory for rendered RxInfer model outputs, relative to the main output directory

# --- Julia Code Generation Utilities ---

def _parse_active_inference_matrix_str(val_str: str) -> str:
    """
    Parses matrix strings like '{(0.5),(0.5)}' or '{(".9",".1"),(".2",".8")}'
    into Julia-compatible array/matrix strings like '[0.5, 0.5]' or '[0.9 0.1; 0.2 0.8]'.
    Handles basic cases found in GNN examples.
    """
    val_str = val_str.strip()
    if not val_str.startswith("{") or not val_str.endswith("}"):
        logger.debug(f"Matrix string '{val_str}' not in expected {{...}} format. Returning as is.")
        return val_str

    content = val_str[1:-1].strip() # Remove outer {} and trim

    # Case 1: Tuples defining rows e.g. { (v1,v2), (v3,v4) } or { (v1), (v2) }
    if content.startswith("(") and content.endswith(")"):
        # Split rows. Need to be careful with nested structures if they were allowed.
        # Assuming simple (v,v),(v,v) or (v),(v)
        # This simplistic split works if rows are separated by "),(
        raw_rows = content.split("),(")
        parsed_rows_str = []
        for i, r_str in enumerate(raw_rows):
            r_str_cleaned = r_str.replace("(", "").replace(")", "").strip()
            elements = [elem.strip() for elem in r_str_cleaned.split(',')]
            parsed_rows_str.append(" ".join(elements)) # Julia elements in a row are space-separated
        
        if not parsed_rows_str:
            return "[]"
        
        # Heuristic: if each "row" has one element, and there are multiple such rows, format as Julia vector
        is_likely_col_vector = all(len(r.split()) == 1 for r in parsed_rows_str)
        if is_likely_col_vector and len(parsed_rows_str) > 1:
            return "[" + ", ".join(e.strip() for e in parsed_rows_str) + "]"
        # Otherwise, format as Julia matrix (rows separated by ;)
        return "[" + "; ".join(parsed_rows_str) + "]"
        
    # Case 2: Simple comma-separated list e.g. {v1,v2,v3}
    elif ',' in content and not '(' in content:
        elements = [elem.strip() for elem in content.split(',')]
        return "[" + ", ".join(elements) + "]"
    # Case 3: Single value e.g. {v1}
    elif not ',' in content and not '(' in content:
        return content # Return as scalar string, RxInfer might take it as is or it needs type.

    logger.warning(f"Could not parse matrix string '{val_str}' into Julia array/matrix. Original: '{content}'")
    return val_str # Fallback

def _format_params(params: Dict[str, Any]) -> str:
    """Formats a dictionary of parameters into a Julia named tuple string."""
    if not params:
        return ""
    formatted_params = []
    for k, v in params.items():
        if isinstance(v, bool):
            formatted_params.append(f"{k} = {str(v).lower()}")
        elif isinstance(v, str) and not (v.startswith("(") or v.startswith("[") or v.isidentifier()):
            escaped_v = v.replace('"', '\\"') # Escape double quotes for Julia string literals
            formatted_params.append(f'{k} = "{escaped_v}"')
        else:
            # For numbers, expressions, or already formatted parts (like variable names)
            formatted_params.append(f"{k} = {v}")
    return ", ".join(formatted_params)

def generate_julia_variable_declaration(
    var_name: str,
    distribution: str,
    params: Dict[str, Any],
    is_observed: bool,
    is_vectorized: bool = False,
    observed_data_name: Optional[str] = None
) -> str:
    """
    Generates a Julia line for a variable declaration in RxInfer.
    e.g., `β ~ Normal(mean = 0.0, variance = 1.0)`
    or    `y .~ Normal(mean = x * β + intercept, variance = σ²)`
    """
    operator = ". ~" if is_vectorized else "~"
    params_str = _format_params(params)

    if is_observed and observed_data_name:
        return f"    {observed_data_name} {operator} {distribution}({params_str})"
    elif not is_observed:
        return f"    {var_name} {operator} {distribution}({params_str})"
    else:
        logger.warning(f"Attempted to generate observed variable '{var_name}' without proper data handling in declaration.")
        return f"    # Error: Observed variable '{var_name}' needs data source"

def generate_rxinfer_model_definition(model_name: str, model_args: List[str], body_lines: List[str]) -> str:
    """Wraps model body lines with RxInfer's @model function syntax."""
    args_str = ", ".join(model_args)
    body_str = "\n".join(body_lines) # Assuming body_lines are already correctly indented (4 spaces)
    return (
        f"@model function {model_name}({args_str})\n"
        f"{body_str}\n"
        f"end"
    )

def generate_rxinfer_constraints_definition(constraints_name: Optional[str], body_lines: List[str]) -> str:
    """Wraps constraints body lines with RxInfer's @constraints syntax."""
    body_str = "\n".join([f"    {line}" for line in body_lines]) # Ensure 4-space indent for lines
    if constraints_name:
        return (
            f"@constraints function {constraints_name}()\n"
            f"{body_str}\n"
            f"end"
        )
    else:
        return (
            f"@constraints begin\n"
            f"{body_str}\n"
            f"end"
        )

def generate_rxinfer_meta_definition(meta_name: Optional[str], body_lines: List[str]) -> str:
    """Wraps meta configuration lines with RxInfer's @meta syntax."""
    body_str = "\n".join([f"    {line}" for line in body_lines]) # Ensure 4-space indent
    if meta_name:
        return (
            f"@meta function {meta_name}()\n"
            f"{body_str}\n"
            f"end"
        )
    else:
        return (
            f"@meta begin\n"
            f"{body_str}\n"
            f"end"
        )

# --- GNN to RxInfer.jl Mapping Logic ---

class GnnToRxInferConverter:
    """
    Converts a parsed GNN specification into components of an RxInfer.jl script.
    Assumes GNN spec provides nodes, their types (random, observed, constant),
    distributions, parameters, and dependencies.
    """
    def __init__(self, gnn_spec: Dict[str, Any]):
        self.gnn_spec = gnn_spec
        self.model_name = gnn_spec.get("name", "GNNModel")
        # Model arguments are primarily defined by "arguments" in GNN spec.
        # The node processing logic might add to this if not using model_logic.
        self.model_args = list(gnn_spec.get("arguments", []))
        self.julia_model_lines: List[str] = []
        self.julia_constraints_lines: List[str] = []
        self.julia_meta_lines: List[str] = []
        self._dependencies_map: Dict[str, List[str]] = {}
        self._processed_nodes: set[str] = set()
        self.nodes_map: Dict[str, Dict[str, Any]] = {}
        # self.model_return_values is not strictly needed if return is part of model_logic lines
        # self.model_return_values: List[str] = gnn_spec.get("returns", [])

    def _build_dependencies_map(self):
        """Builds a map of node dependencies from the GNN specification."""
        self.nodes_map = {node["id"]: node for node in self.gnn_spec.get("nodes", [])}
        for node_id, node_data in self.nodes_map.items():
            self._dependencies_map[node_id] = node_data.get("dependencies", [])

    def _resolve_processing_order(self) -> List[Dict[str, Any]]:
        """Resolves node processing order based on dependencies (topological sort)."""
        ordered_nodes_ids: List[str] = []
        
        # Ensure nodes_map is built if not already
        if not self.nodes_map:
            self.nodes_map = {node["id"]: node for node in self.gnn_spec.get("nodes", [])}
            for node_id, node_data in self.nodes_map.items():
                self._dependencies_map[node_id] = node_data.get("dependencies", [])

        nodes_to_visit = list(self.nodes_map.keys())
        temp_mark = set()  # For detecting cycles in current DFS path
        perm_mark = set()  # For nodes whose processing is complete

        def visit(node_id):
            if node_id in perm_mark:
                return
            if node_id in temp_mark:
                # Check if the cycle involves only existing nodes before raising error
                is_valid_node = node_id in self.nodes_map
                if is_valid_node:
                    # Attempt to get more info about the cycle for better debugging
                    cycle_path = list(temp_mark)
                    logger.error(f"Cyclic dependency detected involving node '{node_id}'. Path: {cycle_path}")
                    raise ValueError(f"Cyclic dependency detected involving node '{node_id}'. Path: {cycle_path}")
                # If node_id is not in self.nodes_map, it might be an undefined dependency
                logger.warning(f"Node '{node_id}' involved in a potential cycle is not defined. Check dependencies.")
                # Decide if to raise error or try to continue by ignoring this problematic node_id
                # For now, we raise to highlight the issue.
                raise ValueError(f"Undefined node '{node_id}' found in dependency graph, possibly causing a cycle.")

            temp_mark.add(node_id)
            for dep_id in self._dependencies_map.get(node_id, []):
                if dep_id in self.nodes_map:  # Ensure dependency exists as a defined node
                    visit(dep_id)
                elif dep_id in self.model_args: # Dependency might be a model argument
                    pass # Model arguments don't have further dependencies to visit here
                else:
                    # This case means a dependency is listed but isn't a defined node or model arg
                    logger.warning(f"Node '{node_id}' has an undefined dependency: '{dep_id}'. It will be ignored in ordering.")
            
            temp_mark.remove(node_id)
            perm_mark.add(node_id)
            ordered_nodes_ids.append(node_id) # Add after all dependencies are processed

        for node_id_to_visit in nodes_to_visit:
            if node_id_to_visit not in perm_mark:
                visit(node_id_to_visit)
        
        # Return full node dicts in the order they should be declared
        # The `ordered_nodes_ids` is a topological sort (dependencies first).
        return [self.nodes_map[id_] for id_ in ordered_nodes_ids if id_ in self.nodes_map]

    def _parse_param_value(self, value: Any) -> str:
        """Helper to parse parameter values, including matrix strings."""
        if isinstance(value, str):
            # Check if it's a GNN-style matrix string
            if value.startswith("{") and value.endswith("}"):
                return _parse_active_inference_matrix_str(value)
            # Check if it's an identifier (another variable/argument) or needs quotes
            # Allow existing Julia arrays/tuples or expressions that might be complex
            if value.isidentifier() or \
               (value.startswith("[") and value.endswith("]")) or \
               (value.startswith("(") and value.endswith(")")) or \
               any(op in value for op in ["+:", "-:", "*:", "/:", ".*", ".+", ".-", ".:", "[", "]"]) : # crude check for expressions
                return value
            # It's likely a string literal that needs quoting for Julia
            escaped_value = value.replace("\"", "\\\"") # Julia string escape: " -> \"
            return f'"{escaped_value}"'
        elif isinstance(value, bool):
            return str(value).lower()
        # Numbers, etc., can be directly converted to string
        return str(value)

    def _format_params_for_distribution(self, params: Dict[str, Any]) -> str:
        """Formats parameters for a distribution call, parsing values as needed."""
        if not params:
            return ""
        formatted_params = []
        for k, v_raw in params.items():
            v_parsed = self._parse_param_value(v_raw)
            formatted_params.append(f"{k} = {v_parsed}")
        return ", ".join(formatted_params)

    # Update generate_julia_variable_declaration to use _format_params_for_distribution
    def _generate_julia_variable_declaration(
        self,
        var_name: str, # Can be s[t] or observations[t]
        distribution: str,
        params: Dict[str, Any],
        is_observed: bool,
        is_vectorized: bool = False,
        observed_data_name: Optional[str] = None, # If var_name is different from data source name
        base_indent: str = "    "
    ) -> str:
        operator = ". ~" if is_vectorized else "~"
        # Use the new params formatter
        params_str = self._format_params_for_distribution(params)

        # If var_name itself is the observed data (e.g. observations[t]), observed_data_name should be var_name
        target_name_for_observed = observed_data_name if observed_data_name else var_name

        if is_observed:
            return f"{base_indent}{target_name_for_observed} {operator} {distribution}({params_str})"
        else: # RV declaration
            return f"{base_indent}{var_name} {operator} {distribution}({params_str})"

    def _handle_rv_vector_declaration(self, item: Dict[str, Any], base_indent: str) -> str:
        name = item["name"]
        size_var = item["size_var"]
        # Default type of elements in RxInfer RandomVariable vectors
        element_type = item.get("element_type", "RandomVariable") 
        return f"{base_indent}{name} = Vector{{{element_type}}}(undef, {size_var})"

    def _handle_assignment(self, item: Dict[str, Any], base_indent: str) -> str:
        lhs = item["variable"] # e.g., "s[1]", "s[t]", "observations[t]"
        dist = item["distribution"]
        params = item.get("params", {})
        is_observed = item.get("is_observed_data", False)
        is_vectorized = item.get("is_vectorized", False)
        
        # When is_observed_data is true, lhs (e.g. "observations[t]") is the data placeholder
        return self._generate_julia_variable_declaration(
            var_name=lhs,
            distribution=dist,
            params=params,
            is_observed=is_observed,
            is_vectorized=is_vectorized,
            observed_data_name=lhs if is_observed else None, # if observed, var_name is the data name
            base_indent=base_indent
        )

    def _handle_loop(self, item: Dict[str, Any], base_indent: str) -> List[str]:
        loop_var = item["variable"]
        range_start = item["range_start"]
        range_end = item["range_end"] # This could be a variable like 'T'
        body_items = item["body"]
        
        loop_lines = [f"{base_indent}for {loop_var} in {range_start}:{range_end}"]
        # Process body with increased indentation
        loop_lines.extend(self._process_model_logic_block(body_items, base_indent + "    "))
        loop_lines.append(f"{base_indent}end")
        return loop_lines

    def _handle_return_statement(self, item: Dict[str, Any], base_indent: str) -> str:
        values_to_return = item.get("values", [])
        if not values_to_return:
            return f"{base_indent}# No return values specified"
        return f"{base_indent}return {', '.join(values_to_return)}"
        
    def _handle_raw_julia(self, item: Dict[str, Any], base_indent: str) -> str:
        raw_code = item.get("code", "")
        # Ensure the raw code is indented correctly if it's multi-line
        lines = raw_code.splitlines() # Use splitlines() for better handling of newlines
        if not lines:
            return f"{base_indent}# Raw Julia item was empty"
        indented_lines = [f"{base_indent}{lines[0].strip()}"] # Indent first line
        indented_lines.extend([f"{base_indent}{line.strip()}" for line in lines[1:]]) # Indent subsequent lines
        return "\n".join(indented_lines)

    def _process_model_logic_block(self, logic_block: List[Dict[str, Any]], base_indent: str) -> List[str]:
        processed_lines: List[str] = []
        for item in logic_block:
            item_type = item.get("item_type")
            if item_type == "rv_vector_declaration":
                processed_lines.append(self._handle_rv_vector_declaration(item, base_indent))
            elif item_type == "assignment":
                processed_lines.append(self._handle_assignment(item, base_indent))
            elif item_type == "loop":
                processed_lines.extend(self._handle_loop(item, base_indent))
            elif item_type == "return_statement":
                processed_lines.append(self._handle_return_statement(item, base_indent))
            elif item_type == "raw_julia":
                processed_lines.append(self._handle_raw_julia(item, base_indent))
            else:
                logger.warning(f"Unknown model_logic item_type: '{item_type}'. Skipping.")
        return processed_lines

    def convert_node_to_julia(self, node: Dict[str, Any]):
        """Translates a single GNN node into Julia code for the @model block (fallback)."""
        node_id = node["id"]
        node_type = node.get("type", "random_variable")
        act_inf_role = node.get("act_inf_role")
        initial_value_raw = node.get("initial_value") # From GNN InitialParameterization
        julia_value_str = None
        if initial_value_raw and isinstance(initial_value_raw, str):
            julia_value_str = _parse_active_inference_matrix_str(initial_value_raw)

        if node_id in self._processed_nodes:
            return

        # This method should only add to self.model_args if they are not already defined
        # by the main "arguments" field of the GNN spec.
        # And it should primarily add to self.julia_model_lines.

        if act_inf_role == "Prior" and node_type == "constant":
            if node_id not in self.model_args: self.model_args.append(node_id)
            logger.debug(f"Node '{node_id}' (Prior) registered as model argument/data for fallback.")

        elif act_inf_role == "LikelihoodMatrix" and node_type == "constant":
            if node_id not in self.model_args: self.model_args.append(node_id)
            logger.debug(f"Node '{node_id}' (LikelihoodMatrix) registered as model argument/data for fallback.")
        
        elif act_inf_role == "HiddenState" and node_type == "random_variable": # Simplified, no vector handling
            dist = node.get("distribution", "Categorical") 
            params = node.get("params", {})
            # Infer params from dependencies logic for fallback:
            if not params and node.get("dependencies"):
                prior_dep = next((dep for dep in node.get("dependencies", []) if self.nodes_map.get(dep,{}).get("act_inf_role") == "Prior"), None)
                if prior_dep: params = {"p": prior_dep}
            
            self.julia_model_lines.append(
                self._generate_julia_variable_declaration(node_id, dist, params, is_observed=False)
            )
        
        elif act_inf_role == "Observation" and node_type == "observed_data":
            dist = node.get("distribution", "Categorical")
            params = node.get("params", {})
            if node_id not in self.model_args: self.model_args.append(node_id)
            self.julia_model_lines.append(
                self._generate_julia_variable_declaration(
                    var_name=node_id, 
                    distribution=dist, params=params, 
                    is_observed=True, 
                    is_vectorized=node.get("is_vectorized", False),
                    observed_data_name=node_id
                )
            )
        
        elif node_type == "random_variable": # General RV, not specific ActInf role
            dist = node.get("distribution", "Distributions.Normal") # Default if missing
            params = node.get("params", {})
            resolved_params = {k: (self._parse_param_value(v) if isinstance(v, str) else v) for k, v in params.items()}
            self.julia_model_lines.append(
                self._generate_julia_variable_declaration(node_id, dist, resolved_params, is_observed=False)
            )
        elif node_type == "constant": 
            if julia_value_str:
                self.julia_model_lines.append(f"    {node_id} = {julia_value_str}")
            elif node_id not in self.model_args: 
                self.model_args.append(node_id) # Assume passed as argument
                logger.debug(f"General constant '{node_id}' added as model argument for fallback.")

        elif node_type == "submodel_call": # Fallback, less common without model_logic
            submodel_name = node["submodel_name"]
            instance_params = node.get("params", {})
            output_var = node_id
            param_str = self._format_params_for_distribution(instance_params)
            self.julia_model_lines.append(f"    {output_var} ~ {submodel_name}({param_str})")
        else:
            logger.warning(f"Unsupported GNN node type: '{node_type}' for node '{node_id}' in fallback processing.")
        self._processed_nodes.add(node_id)

    def convert_gnn_structure(self):
        """Iterates GNN nodes or uses model_logic, populating Julia code lines."""
        self.julia_model_lines = [] # Reset for each conversion

        # Prioritize model_logic if present
        if "model_logic" in self.gnn_spec and self.gnn_spec["model_logic"]:
            logger.info("Processing GNN specification using 'model_logic' section.")
            self.julia_model_lines = self._process_model_logic_block(self.gnn_spec["model_logic"], base_indent="    ")
            # Ensure model_args from GNN spec are respected, model_logic should use them.
            # No need to auto-add args from nodes if model_logic is used.
        else:
            logger.info("No 'model_logic' found or it's empty. Falling back to node-based processing.")
            self._build_dependencies_map() # Builds nodes_map and _dependencies_map
            if not self.nodes_map:
                logger.warning("No nodes found in GNN specification for node-based processing.")
            else:
                ordered_nodes = self._resolve_processing_order()
                if not ordered_nodes:
                    logger.warning("Node order resolution yielded no nodes. Model body might be empty.")
                for node in ordered_nodes:
                    self.convert_node_to_julia(node)
        
        # Process constraints and meta, these append to their respective lists
        gnn_constraints = self.gnn_spec.get("constraints")
        if gnn_constraints:
            if isinstance(gnn_constraints, list):
                for constr in gnn_constraints:
                    if constr.get("type") == "mean_field" and "factors" in constr:
                        for group in constr["factors"]:
                            self.julia_constraints_lines.append(f"q({', '.join(group)}) = {''.join([f'q({factor})' for factor in group])}")
                    elif constr.get("type") == "form" and "variable" in constr:
                        form_type = constr.get("form_type", "PointMass") + "FormConstraint"
                        self.julia_constraints_lines.append(f"q({constr['variable']}) :: {form_type}()")
            elif isinstance(gnn_constraints, dict) and "raw_lines" in gnn_constraints:
                self.julia_constraints_lines.extend(gnn_constraints["raw_lines"])

        gnn_meta = self.gnn_spec.get("meta")
        if gnn_meta:
            if isinstance(gnn_meta, list):
                for meta_item in gnn_meta:
                    node_ref = meta_item.get("node_id", meta_item.get("factor_ref"))
                    # Use _format_params for consistency in settings string
                    settings_str = _format_params(meta_item.get("settings", {}))
                    if node_ref and settings_str:
                        self.julia_meta_lines.append(f"{node_ref} -> _ where {{ {settings_str} }}")
            elif isinstance(gnn_meta, dict) and "raw_lines" in gnn_meta:
                self.julia_meta_lines.extend(gnn_meta["raw_lines"])

    def generate_inference_script(self, data_bindings: Dict[str, str], iterations: int = 50, free_energy: bool = False) -> str:
        """Generates Julia code for running inference."""
        model_call_args_bindings = [] # For (param = value) in model call
        data_tuple_entries = []       # For data = (obs = my_obs_data, ...)
        
        # self.model_args should be set from gnn_spec["arguments"] primarily
        for arg_name in self.model_args: # Iterate over declared model arguments
            if arg_name in data_bindings:
                val_str = str(data_bindings[arg_name])
                # Assume val_str is a valid Julia expression or variable name for the binding
                model_call_args_bindings.append(f"{arg_name} = {val_str}")
                
                # Check if this arg_name corresponds to an observed_data node or is used for data
                # This heuristic might need refinement: check if arg_name is used as observed data in model_logic or nodes.
                # For now, if it's in data_bindings, assume it *could* be data for the tuple.
                # A more robust way is to identify "observed_data" nodes/vars from GNN spec.
                is_data_var = False
                if "model_logic" in self.gnn_spec and self.gnn_spec["model_logic"]:
                    for item in self.gnn_spec["model_logic"]:
                        if item.get("item_type") == "assignment" and item.get("is_observed_data", False) and item.get("variable","").startswith(arg_name): # e.g. obs[t] for arg obs
                            is_data_var = True
                            break
                        if item.get("item_type") == "loop": # Check inside loops
                            for sub_item in item.get("body",[]):
                                 if sub_item.get("item_type") == "assignment" and sub_item.get("is_observed_data", False) and sub_item.get("variable","").startswith(arg_name):
                                     is_data_var = True; break
                            if is_data_var: break
                else: # Fallback: check nodes
                    node_def = self.nodes_map.get(arg_name)
                    if node_def and node_def.get("type") == "observed_data":
                        is_data_var = True
                
                if is_data_var:
                    data_tuple_entries.append(f"{arg_name} = {val_str}")
            else:
                # If a model argument is not in data_bindings, it's an unbound parameter.
                # RxInfer might require all arguments to be bound or have defaults in the model itself (not handled here).
                logger.warning(f"Model argument '{arg_name}' not found in data_bindings for inference. It will be omitted from the `data` tuple and assumed to be passed directly if needed, or be a model constant.")
                # It will still be part of model_call_args_bindings if it was meant to be a constant value not in data tuple.
                # This part is tricky: should non-data args also be in data_bindings?
                # For RxInfer: model_call_args_bindings are for args like `model = MyModel(N=10, k=0.5)`
                # data_tuple_entries are for `data = (y = y_data, x = x_data)`
                # Let's assume if not in data_bindings, it's not for the `data` tuple for now.
                # If it's a parameter like `transition_matrix` that is NOT data, it should still be in data_bindings.

        model_signature_for_call = self.model_name
        # Parameters passed directly to the model function call
        model_direct_params_str = ", ".join(model_call_args_bindings)
        if model_direct_params_str:
             model_signature_for_call += f"({model_direct_params_str})"
        # else: # If no params, call as MyModel() or just MyModel if it's a submodel ref.
        #    model_signature_for_call += "()" # RxInfer usually needs () if it's a function

        data_arg_str = f"data = ({', '.join(data_tuple_entries)})," if data_tuple_entries else ""
        
        # Determine if constraints/meta functions are named or anonymous
        constraints_name_from_spec = self.gnn_spec.get("constraints",{}).get("name")
        meta_name_from_spec = self.gnn_spec.get("meta",{}).get("name")

        constraints_arg_str = ""
        if self.julia_constraints_lines:
            constraints_func_name = constraints_name_from_spec if constraints_name_from_spec else f"{self.model_name}Constraints"
            if self.gnn_spec.get("constraints",{}).get("is_anonymous"): # Check if anonymous
                 constraints_arg_str = f"constraints = @constraints begin\\n{self.julia_constraints_lines[0]}\\nend," # Simplified for one line
            else:
                 constraints_arg_str = f"constraints = {constraints_func_name}(),"

        meta_arg_str = ""
        if self.julia_meta_lines:
            meta_func_name = meta_name_from_spec if meta_name_from_spec else f"{self.model_name}Meta"
            if self.gnn_spec.get("meta",{}).get("is_anonymous"):
                meta_arg_str = f"meta = @meta begin\\n{self.julia_meta_lines[0]}\\nend," # Simplified for one line
            else:
                meta_arg_str = f"meta = {meta_func_name}(),"

        inference_params_list = [
            f"model = {model_signature_for_call}",
            data_arg_str,
            constraints_arg_str,
            meta_arg_str,
            f"iterations = {iterations}",
        ]
        if free_energy:
            inference_params_list.append("free_energy = true")
        
        # Filter out empty strings and join with newline and indent
        inference_params_str = "\n    ".join(filter(None, [p.strip(",") for p in inference_params_list if p])).strip()

        print_posteriors_lines = []
        for node in self.gnn_spec.get("nodes", []):
            if node.get("type") == "random_variable" and node.get("report_posterior", False):
                node_id = node['id']
                # Julia symbol for dictionary key: :node_id
                julia_println = f'println("Posterior for {node_id}: ", result.posteriors[:{node_id}])'
                print_posteriors_lines.append(julia_println)
        
        if not print_posteriors_lines and any(n.get("type") == "random_variable" for n in self.nodes_map.values()):
            first_rv = next((n['id'] for n_id, n in self.nodes_map.items() if n.get("type") == "random_variable"), None)
            if first_rv:
                julia_println_default = f'println("Posterior for {first_rv} (example): ", result.posteriors[:{first_rv}])'
                print_posteriors_lines.append(julia_println_default)

        print_posteriors_str = "\n".join(print_posteriors_lines)
        if free_energy:
            vfe_line = 'println("Variational Free Energy: ", result.free_energy)'
            if print_posteriors_str:
                print_posteriors_str += f"\n{vfe_line}"
            else:
                print_posteriors_str = vfe_line

        data_vars_comment = ', '.join(data_bindings.values()) if data_bindings else "your_data_variables"
        return (
            f"# --- Inference ---\n"
            f"# Note: Ensure that data variables (e.g., {data_vars_comment})\n"
            f"# are defined and loaded in the Julia environment before this script section.\n"
            f"# Example:\n"
            f"# using CSV, DataFrames\n"
            f"# my_data_table = CSV.read(\"path/to/your/data.csv\", DataFrame)\n"
            f"# y_observed_data = my_data_table.y_column\n"
            f"# X_matrix_data = Matrix(my_data_table[!, [:x1_column, :x2_column]])\n\n"
            f"result = infer(\n    {inference_params_str}\n)\
\n"
            f"{print_posteriors_str}"
        )

    def get_full_julia_script(
        self, 
        include_inference: bool = True, 
        data_bindings: Optional[Dict[str, str]] = None, 
        iterations: int = 50,
        free_energy: bool = False
    ) -> str:
        """Generates the complete RxInfer.jl Julia script content."""
        self.convert_gnn_structure()
        imports = ["using RxInfer"]
        if self.gnn_spec.get("julia_imports"):
            imports.extend(self.gnn_spec["julia_imports"])
        imports_str = "\n".join(imports) + "\n"

        model_definition = generate_rxinfer_model_definition(self.model_name, self.model_args, self.julia_model_lines)
        
        constraints_definition = ""
        if self.julia_constraints_lines:
            constraints_name = self.gnn_spec.get("constraints",{}).get("name") # Get potential custom name
            if not constraints_name and not self.gnn_spec.get("constraints",{}).get("is_anonymous"):
                constraints_name = f"{self.model_name}Constraints" # Default name if not anonymous and no custom name
            constraints_definition = generate_rxinfer_constraints_definition(constraints_name, self.julia_constraints_lines)
            
        meta_definition = ""
        if self.julia_meta_lines:
            meta_name = self.gnn_spec.get("meta",{}).get("name") # Get potential custom name
            if not meta_name and not self.gnn_spec.get("meta",{}).get("is_anonymous"):
                meta_name = f"{self.model_name}Meta" # Default name if not anonymous and no custom name
            meta_definition = generate_rxinfer_meta_definition(meta_name, self.julia_meta_lines)

        script_parts = [imports_str, model_definition]
        if constraints_definition:
            script_parts.append(constraints_definition)
        if meta_definition:
            script_parts.append(meta_definition)
        
        if include_inference:
            inference_code = self.generate_inference_script(data_bindings or {}, iterations, free_energy)
            script_parts.append(f"\n{inference_code}")
            
        return "\n\n".join(filter(None, script_parts))

# --- Main Rendering Function for RxInfer.jl ---

def render_gnn_to_rxinfer_jl(
    gnn_spec: Dict[str, Any],
    output_script_path: Path,
    options: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, List[str]]:
    """Renders GNN to RxInfer.jl script."""
    options = options or {}
    logger.info(f"Rendering GNN spec to RxInfer.jl script '{output_script_path}'.")
    try:
        converter = GnnToRxInferConverter(gnn_spec)
        
        julia_script_content = converter.get_full_julia_script(
            include_inference=options.get("include_inference_script", True),
            data_bindings=options.get("data_bindings", {}),
            iterations=options.get("inference_iterations", 50),
            free_energy=options.get("calculate_free_energy", False)
        )
        output_script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_script_path, "w", encoding="utf-8") as f:
            f.write(julia_script_content)
        logger.info(f"Successfully wrote RxInfer.jl script to {output_script_path}")
        return True, f"Successfully rendered to RxInfer.jl: {output_script_path.name}", [output_script_path.as_uri()]
    except Exception as e:
        logger.error(f"Error rendering GNN to RxInfer.jl: {e}", exc_info=True)
        return False, f"Error rendering to RxInfer.jl: {str(e)}", []

# --- Placeholder GNN Parser (no longer directly used by render_gnn_to_rxinfer_jl but kept for standalone testing) ---
def placeholder_gnn_parser(gnn_file_path: Path) -> Optional[Dict[str, Any]]:
    """Placeholder GNN parser. Loads from JSON."""
    logger.warning(f"Using placeholder GNN parser for '{gnn_file_path.name}'.")
    if not gnn_file_path.exists():
        logger.error(f"GNN file not found: {gnn_file_path}")
        return {
            "name": "DefaultPlaceholderModel",
            "nodes": [{"id": "theta", "type": "random_variable", "distribution": "Normal", 
                      "params": {"mean":0, "variance":1}, "report_posterior": True}],
            "arguments": []}
    try:
        import json
        with open(gnn_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except ImportError:
        logger.error("json module not found for placeholder GNN parser.")
        return None
    except Exception as e:
        logger.error(f"Error parsing GNN file {gnn_file_path} with placeholder: {e}")
        return None

if __name__ == '__main__':
    test_output_dir = Path("temp_rxinfer_render_test")
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Original linear regression test (can be kept or removed)
    dummy_gnn_data_linear_regression = {
        "name": "LinearRegressionGNN",
        "arguments": ["y_obs", "x_matrix", "sigma_sq_val_arg"], # sigma_sq_val_arg to avoid clash with node
        "nodes": [
            {"id": "beta", "type": "random_variable", "distribution": "Normal", "params": {"mean": 0.0, "variance": 1.0}, "report_posterior": True},
            {"id": "intercept", "type": "random_variable", "distribution": "Normal", "params": {"mean": 0.0, "variance": 10.0}, "report_posterior": True},
            {"id": "sigma_sq_val_node", "type": "constant", "initial_value": "0.1"}, # Node for internal constant
            {"id": "y_obs", "type": "observed_data", "distribution": "Normal", 
             "params": {"mean": "x_matrix * beta + intercept", "variance": "sigma_sq_val_arg"}, # use arg here
             "is_vectorized": True, "dependencies": ["x_matrix", "beta", "intercept", "sigma_sq_val_arg"]
            }
        ],
        # No model_logic, so it will use node-based processing.
        "constraints": [{"type": "mean_field", "factors": [["beta", "intercept"]]}],
        "meta": [{"factor_ref": "Normal", "settings": {"damped": "true"}}], # Ensure boolean is string for _format_params
        "julia_imports": ["using Distributions"] # Example of specific imports
    }
    
    output_script_lr = test_output_dir / "generated_linear_regression_script.jl"
    render_options_lr = {
        "data_bindings": {
            "y_obs": "actual_y_data",      # This is data
            "x_matrix": "actual_x_data",    # This is data
            "sigma_sq_val_arg": "0.05"      # This is a parameter passed to model
        },
        "inference_iterations": 75,
        "calculate_free_energy": True
    }

    success_lr, msg_lr, artifacts_lr = render_gnn_to_rxinfer_jl(
        dummy_gnn_data_linear_regression,
        output_script_lr, 
        options=render_options_lr
    )

    if success_lr:
        logger.info(f"RxInfer.jl LR rendering test successful: {msg_lr}")
        if output_script_lr.exists():
            logger.info(f"--- Generated LR Julia Script ({output_script_lr.name}) ---")
            print(f"\n{output_script_lr.read_text(encoding='utf-8')}\n")
            logger.info(f"--- End of LR Script ---")
    else:
        logger.error(f"RxInfer.jl LR rendering test failed: {msg_lr}")

    # --- HMM Test using model_logic ---
    dummy_gnn_data_hmm_ml = {
        "name": "SimpleHMM_from_Logic",
        "arguments": ["observations", "T", "A", "B", "initial_dist_p"], # A=transition, B=emission
        "nodes": [ 
            # Nodes can still define types or be informative, but model structure comes from model_logic
            {"id": "observations", "type": "observed_data", "description": "Vector of observed states"},
            {"id": "T", "type": "constant", "description": "Time horizon / number of observations"},
            {"id": "A", "type": "constant", "description": "Transition matrix"},
            {"id": "B", "type": "constant", "description": "Emission matrix"},
            {"id": "initial_dist_p", "type": "constant", "description": "Initial state distribution parameters (vector p)"},
            {"id": "s", "type": "random_variable_vector", "description": "Latent state sequence"}
        ],
        "model_logic": [
            {"item_type": "raw_julia", "code": "# Hidden Markov Model implementation from GNN model_logic"},
            {"item_type": "rv_vector_declaration", "name": "s", "size_var": "T", "element_type": "RandomVariable"},
            {"item_type": "assignment", "variable": "s[1]", "distribution": "Categorical", "params": {"p": "initial_dist_p"}},
            {
                "item_type": "loop", "variable": "t", "range_start": 2, "range_end": "T",
                "body": [
                    {"item_type": "assignment", "variable": "s[t]", "distribution": "Categorical", 
                     "params": {"p": "A[s[t-1], :]"}} # Assuming A is passed as matrix
                ]
            },
            {
                "item_type": "loop", "variable": "t", "range_start": 1, "range_end": "T",
                "body": [
                    # observations[t] is the data placeholder from model arguments
                    {"item_type": "assignment", "variable": "observations[t]", "is_observed_data": True, 
                     "distribution": "Categorical", "params": {"p": "B[s[t], :]"}} # Assuming B is passed
                ]
            },
            {"item_type": "return_statement", "values": ["s", "observations"]}
        ],
        "julia_imports": ["using Distributions"], # Explicitly list required packages for the model
        "constraints": { # Example of named constraints
            "name": "MyHMMConstraints", # Optional custom name
            "is_anonymous": False, # Explicitly not anonymous
             "raw_lines": ["q(s) :: MeanField()"] # Raw lines for constraints body
        },
        "meta": { # Example of anonymous meta
            "is_anonymous": True,
            "raw_lines": ["Categorical(p) -> ((p = p ./ sum(p)),)"] # Example meta line
        }
    }

    output_script_hmm_ml = test_output_dir / "generated_hmm_model_logic_script.jl"
    render_options_hmm_ml = {
        "data_bindings": {
            "observations": "my_observed_sequence", # Name of Julia variable holding observations
            "T": "length(my_observed_sequence)",   # Julia expression for T
            "A": "transition_matrix_data",         # Name of Julia variable for transition matrix
            "B": "emission_matrix_data",           # Name of Julia variable for emission matrix
            "initial_dist_p": "initial_probabilities_vector" # Julia variable for initial dist
        },
        "inference_iterations": 100,
        "calculate_free_energy": True
    }

    success_hmm_ml, msg_hmm_ml, artifacts_hmm_ml = render_gnn_to_rxinfer_jl(
        dummy_gnn_data_hmm_ml,
        output_script_hmm_ml,
        options=render_options_hmm_ml
    )

    if success_hmm_ml:
        logger.info(f"RxInfer.jl HMM (model_logic) rendering test successful: {msg_hmm_ml}")
        if output_script_hmm_ml.exists():
            logger.info(f"--- Generated HMM (model_logic) Julia Script ({output_script_hmm_ml.name}) ---")
            print(f"\n{output_script_hmm_ml.read_text(encoding='utf-8')}\n")
            logger.info(f"--- End of HMM (model_logic) Script ---")
    else:
        logger.error(f"RxInfer.jl HMM (model_logic) rendering test failed: {msg_hmm_ml}")

    # Cleanup (optional)
    # import shutil
    # if test_output_dir.exists():
    #     shutil.rmtree(test_output_dir)
    #     logger.info(f"Cleaned up test directory: {test_output_dir}") 