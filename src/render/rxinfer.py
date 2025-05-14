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
        self.model_args = list(gnn_spec.get("arguments", []))
        self.julia_model_lines: List[str] = []
        self.julia_constraints_lines: List[str] = []
        self.julia_meta_lines: List[str] = []
        self._dependencies_map: Dict[str, List[str]] = {}
        self._processed_nodes: set[str] = set()
        self.nodes_map: Dict[str, Dict[str, Any]] = {}

    def _build_dependencies_map(self):
        """Builds a map of node dependencies from the GNN specification."""
        self.nodes_map = {node["id"]: node for node in self.gnn_spec.get("nodes", [])}
        for node_id, node_data in self.nodes_map.items():
            self._dependencies_map[node_id] = node_data.get("dependencies", [])

    def _resolve_processing_order(self) -> List[Dict[str, Any]]:
        """Resolves node processing order based on dependencies (topological sort)."""
        ordered_nodes_ids: List[str] = []
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
        # The original `ordered_nodes_ids` is in reverse post-order (parents after children).
        # For declaration, we often want dependencies declared first.
        # However, visit appends after dependencies, so `ordered_nodes_ids` IS the topological sort.
        return [self.nodes_map[id_] for id_ in ordered_nodes_ids]

    def convert_node_to_julia(self, node: Dict[str, Any]):
        """Translates a single GNN node into Julia code for the @model block."""
        node_id = node["id"]
        node_type = node.get("type", "random_variable")
        act_inf_role = node.get("act_inf_role")
        initial_value_raw = node.get("initial_value") # From GNN InitialParameterization
        julia_value_str = None
        if initial_value_raw and isinstance(initial_value_raw, str):
            julia_value_str = _parse_active_inference_matrix_str(initial_value_raw)

        if node_id in self._processed_nodes:
            return

        # --- Handle based on Active Inference Role --- 
        if act_inf_role == "Prior" and node_type == "constant":
            # This constant (e.g., D_param value) will be used by HiddenState
            # It should be defined as a model argument to be passed in during inference call.
            if node_id not in self.model_args:
                self.model_args.append(node_id)
            # The actual assignment of data happens in the `infer` call typically for RxInfer
            # So, no direct `node_id = value` line here in the @model, unless it truly is a fixed model constant.
            # For GNN, D is often a prior that can be fixed. Let's assume it is passed as data.
            logger.debug(f"Node '{node_id}' (Prior) will be a model argument/data.")

        elif act_inf_role == "LikelihoodMatrix" and node_type == "constant":
            if node_id not in self.model_args:
                self.model_args.append(node_id)
            logger.debug(f"Node '{node_id}' (LikelihoodMatrix) will be a model argument/data.")

        elif act_inf_role == "HiddenState":
            dist = node.get("distribution", "Categorical") # Default for discrete hidden state
            # Parameters for this distribution, e.g., {"p": "D_node_id"}
            # These params should be in the GNN spec node definition
            params = node.get("params", {})
            if not params and node.get("dependencies"):
                # Try to infer params from dependencies if not explicit
                # E.g. if depends on D_node (Prior), then p = D_node
                prior_dep = next((dep for dep in node.get("dependencies") if self.nodes_map.get(dep,{}).get("act_inf_role") == "Prior"), None)
                if prior_dep:
                    params = {"p": prior_dep}
                else:
                    logger.warning(f"HiddenState '{node_id}' has no explicit params and suitable Prior dependency not found.")
            
            self.julia_model_lines.append(
                generate_julia_variable_declaration(node_id, dist, params, is_observed=False)
            )
        
        elif act_inf_role == "Observation" and node_type == "observed_data":
            dist = node.get("distribution", "Categorical") # Default for discrete observation
            params = node.get("params", {})
            # Example params: {"p": "A_matrix_node_id * s_node_id"} or {"p": "A_matrix_node_id[:, s_node_id]"}
            # These must be correctly specified in the GNN spec node for RxInfer.
            if not params and node.get("dependencies"):
                logger.warning(f"Observation '{node_id}' has no explicit params in GNN spec.")

            if node_id not in self.model_args: # Observation data variable must be a model argument
                self.model_args.append(node_id)

            self.julia_model_lines.append(
                generate_julia_variable_declaration(
                    var_name=node_id, # This is the data variable name itself
                    distribution=dist,
                    params=params,
                    is_observed=True,
                    is_vectorized=node.get("is_vectorized", False),
                    observed_data_name=node_id
                )
            )
        
        # --- Fallback to original generic type handling if no specific ActInf role logic applies ---
        elif node_type == "random_variable":
            dist = node["distribution"]
            params = node.get("params", {})
            # Resolve param names that might be other nodes
            resolved_params = {k: (v if not isinstance(v, str) or (v not in self.nodes_map and v not in self.model_args) else v)
                               for k, v in params.items()}
            self.julia_model_lines.append(
                generate_julia_variable_declaration(node_id, dist, resolved_params, is_observed=False)
            )
        elif node_type == "constant": # General constant not tied to specific ActInf role handled above
            if julia_value_str:
                 # If it's a model-internal constant not passed as arg
                self.julia_model_lines.append(f"    {node_id} = {julia_value_str}")
            elif node_id not in self.model_args: # Otherwise, assume it might be passed as an argument
                self.model_args.append(node_id)
                logger.debug(f"General constant '{node_id}' added as model argument.")

        elif node_type == "submodel_call":
            submodel_name = node["submodel_name"]
            instance_params = node.get("params", {})
            output_var = node_id
            resolved_instance_params = {k: (v if (v not in self.nodes_map and v not in self.model_args) else v) for k, v in instance_params.items()}
            param_str = _format_params(resolved_instance_params)
            self.julia_model_lines.append(f"    {output_var} ~ {submodel_name}({param_str})")
        else:
            logger.warning(f"Unsupported GNN node type: '{node_type}' for node '{node_id}'.")
        self._processed_nodes.add(node_id)

    def convert_gnn_structure(self):
        """Iterates GNN nodes, populating Julia code lines."""
        self._build_dependencies_map()
        ordered_nodes = self._resolve_processing_order()
        for node in ordered_nodes:
            self.convert_node_to_julia(node)
        
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
                    settings_str = _format_params(meta_item.get("settings", {}))
                    if node_ref and settings_str:
                        self.julia_meta_lines.append(f"{node_ref} -> _ where {{ {settings_str} }}")
            elif isinstance(gnn_meta, dict) and "raw_lines" in gnn_meta:
                self.julia_meta_lines.extend(gnn_meta["raw_lines"])

    def generate_inference_script(self, data_bindings: Dict[str, str], iterations: int = 50, free_energy: bool = False) -> str:
        """Generates Julia code for running inference."""
        model_call_args = []
        data_tuple_entries = []
        for arg_name in self.model_args:
            if arg_name in data_bindings:
                val_str = str(data_bindings[arg_name])
                try:
                    float(val_str) # Check if it's a number
                    model_call_args.append(f"{arg_name} = {val_str}")
                    # Literals don't go into the data tuple typically, only variables
                except ValueError:
                    model_call_args.append(f"{arg_name} = {val_str}")
                    data_tuple_entries.append(f"{arg_name} = {val_str}")
            else:
                logger.warning(f"Model argument '{arg_name}' not found in data_bindings for inference. Will be omitted from call if not a data variable.")

        model_signature_for_call = self.model_name
        if model_call_args: # Only add () if there are actual arguments to pass
            model_signature_for_call += f"({', '.join(model_call_args)})"

        data_arg_str = f"data = ({', '.join(data_tuple_entries)})," if data_tuple_entries else ""
        constraints_arg_str = f"constraints = {self.model_name}Constraints()," if self.julia_constraints_lines else ""
        meta_arg_str = f"meta = {self.model_name}Meta()," if self.julia_meta_lines else ""

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
            constraints_name = f"{self.model_name}Constraints" if not self.gnn_spec.get("constraints",{}).get("is_anonymous") else None
            constraints_definition = generate_rxinfer_constraints_definition(constraints_name, self.julia_constraints_lines)
            
        meta_definition = ""
        if self.julia_meta_lines:
            meta_name = f"{self.model_name}Meta" if not self.gnn_spec.get("meta",{}).get("is_anonymous") else None
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
        "arguments": ["y_obs", "x_matrix", "sigma_sq_val"],
        "nodes": [
            {"id": "beta", "type": "random_variable", "distribution": "Normal", "params": {"mean": 0.0, "variance": 1.0}, "report_posterior": True},
            {"id": "intercept", "type": "random_variable", "distribution": "Normal", "params": {"mean": 0.0, "variance": 10.0}, "report_posterior": True},
            {"id": "sigma_sq_val", "type": "constant", "value": 0.1},
            {"id": "y_obs", "type": "observed_data", "distribution": "Normal", 
             "params": {"mean": "x_matrix * beta + intercept", "variance": "sigma_sq_val"}, 
             "is_vectorized": True, "dependencies": ["x_matrix", "beta", "intercept", "sigma_sq_val"]
            }
        ],
        "constraints": [{"type": "mean_field", "factors": [["beta", "intercept"]]}],
        "meta": [{"factor_ref": "Normal", "settings": {"damped": True}}],
        "julia_imports": ["using Distributions"]
    }
    
    output_script = test_output_dir / "generated_linear_regression_script.jl"
    render_options = {
        "data_bindings": {
            "y_obs": "actual_y_data",
            "x_matrix": "actual_x_data",
        },
        "inference_iterations": 75,
        "calculate_free_energy": True
    }

    success, msg, artifacts = render_gnn_to_rxinfer_jl(
        dummy_gnn_data_linear_regression,
        output_script, 
        options=render_options
    )

    if success:
        logger.info(f"RxInfer.jl rendering test successful: {msg}")
        logger.info(f"Artifacts: {artifacts}")
        if output_script.exists():
            logger.info(f"--- Generated Julia Script ({output_script.name}) ---")
            # Read and print the content for review
            script_content = output_script.read_text(encoding='utf-8')
            print(f"\n{script_content}\n")
            logger.info(f"--- End of Script ---")
    else:
        logger.error(f"RxInfer.jl rendering test failed: {msg}")

    # Cleanup (optional)
    # import shutil
    # if test_output_dir.exists():
    #     shutil.rmtree(test_output_dir)
    #     logger.info(f"Cleaned up test directory: {test_output_dir}") 