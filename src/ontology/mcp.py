"""
Model Context Protocol (MCP) interface for the Ontology module.

This file defines how the Ontology module interacts with the broader GNN system
via MCP. It includes methods for parsing GNN ontology annotations,
validating them against a defined set of terms, and assisting in report generation.
"""
import json
import re
import os
import logging

# Initialize logger for this module if not already present
logger = logging.getLogger(__name__)

def parse_gnn_ontology_section(gnn_file_content: str, verbose: bool = False) -> dict:
    """
    Parses the 'ActInfOntologyAnnotation' section from GNN file content.

    Args:
        gnn_file_content: The string content of a GNN file.
        verbose: If True, prints detailed parsing information.

    Returns:
        A dictionary mapping model variables to ontological terms.
        Returns an empty dictionary if the section is not found or is malformed.
    """
    annotations = {}
    try:
        # Regex to find the ActInfOntologyAnnotation section and capture its content
        match = re.search(r"^## ActInfOntologyAnnotation\s*$\n(.*?)(?:^## \S+|^\Z)", gnn_file_content, re.MULTILINE | re.DOTALL)
        if not match:
            if verbose:
                print("MCP: 'ActInfOntologyAnnotation' section not found.")
            return annotations

        section_content = match.group(1).strip()
        if not section_content:
            if verbose:
                print("MCP: 'ActInfOntologyAnnotation' section is empty.")
            return annotations

        lines = section_content.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty lines or comments within the section
                continue
            if '=' in line:
                parts = line.split('=', 1)
                key = parts[0].strip()
                value_full = parts[1].strip()
                
                # Strip comments from the value
                if '#' in value_full:
                    value = value_full.split('#', 1)[0].strip()
                else:
                    value = value_full
                    
                if key and value:
                    annotations[key] = value
                else:
                    if verbose:
                        print(f"MCP: Malformed line {i+1} in ActInfOntologyAnnotation: '{line}' - skipping.")
            else:
                if verbose:
                    print(f"MCP: Line {i+1} in ActInfOntologyAnnotation does not contain '=': '{line}' - skipping.")
        
        if verbose:
            print(f"MCP: Parsed annotations: {annotations}")
            
    except Exception as e:
        if verbose:
            print(f"MCP: Error parsing ActInfOntologyAnnotation section: {e}")
        # Fallback to empty dict on any parsing error
        return {}
        
    return annotations

def load_defined_ontology_terms(ontology_terms_path: str, verbose: bool = False) -> dict:
    """
    Loads defined ontological terms from a JSON file.
    The JSON file should contain an object where keys are term names
    and values can be descriptions or other metadata.

    Args:
        ontology_terms_path: Path to the JSON file containing ontology terms.
        verbose: If True, prints loading information (now handled by logger levels).

    Returns:
        A dictionary of defined ontology terms (term: description/metadata).
        Returns an empty dictionary if the file cannot be loaded or parsed.
    """
    defined_terms = {}
    # Use logger for messages. Verbose flag can control specific debug logs if needed,
    # but errors/warnings should always use the logger.
    # The level of these messages will be controlled by the pipeline's main logger config.
    
    # Get absolute path for clearer logging, especially if ontology_terms_path is relative
    abs_path_str = "<unknown>"
    try:
        abs_path_str = os.path.abspath(ontology_terms_path)
    except Exception:
        pass # Keep default if abspath fails for some exotic reason

    # Log the attempt at debug level. If verbose is True, this will show.
    logger.debug(f"Attempting to load ontology terms from: {abs_path_str} (Original path argument: {ontology_terms_path})")

    try:
        with open(ontology_terms_path, 'r', encoding='utf-8-sig') as f: # utf-8-sig handles BOM
            data = json.load(f)
        if isinstance(data, dict):
            defined_terms = data
            # Log success at debug level or info if verbose was specifically for this.
            # For consistency, let's use debug for successful operational details.
            logger.debug(f"Successfully loaded {len(defined_terms)} ontology terms from {ontology_terms_path}.")
        else:
            # This is a structural error in the file content.
            logger.warning(f"Ontology terms file {ontology_terms_path} (abs: {abs_path_str}) did not contain a root JSON object as expected. Found type: {type(data)}.")
            return {} # Return empty on structure error
    except FileNotFoundError:
        # This is a common, potentially expected issue if the file is optional or path is wrong.
        logger.warning(f"Ontology terms definition file not found: {ontology_terms_path} (abs: {abs_path_str}). Validation will be skipped or limited.")
        return {}
    except json.JSONDecodeError as e:
        # This indicates a malformed JSON file.
        logger.warning(f"Error decoding JSON from ontology terms file {ontology_terms_path} (abs: {abs_path_str}): {e}. Check JSON syntax.")
        return {}
    except Exception as e:
        # Catch any other unexpected errors during file loading/parsing.
        logger.error(f"An unexpected error occurred while loading or parsing ontology terms from {ontology_terms_path} (abs: {abs_path_str}): {e}", exc_info=True)
        return {}
    return defined_terms

def validate_annotations(parsed_annotations: dict, defined_terms: dict, verbose: bool = False) -> dict:
    """
    Validates parsed GNN annotations against a set of defined ontological terms.

    Args:
        parsed_annotations: Dict of {model_var: ontology_term} from GNN file.
        defined_terms: Dict of {ontology_term: description} loaded from a definition file.
                            Only the keys of defined_terms are used for validation.
        verbose: If True, prints validation details.

    Returns:
        A dictionary with validation results:
        {
            "valid_mappings": {model_var: ontology_term, ...},
            "invalid_terms": {model_var: ontology_term, ...} // Terms not in defined_terms
            "unmapped_model_vars": [] // Model vars in parsed_annotations with no ontology term (should not happen with current parser)
        }
    """
    results = {
        "valid_mappings": {},
        "invalid_terms": {},
        "unmapped_model_vars": [] # Should be empty with current parse_gnn_ontology_section
    }
    defined_term_keys = set(defined_terms.keys())

    for model_var, ontology_term in parsed_annotations.items():
        if not ontology_term: # Should not happen if parser ensures value exists
             results["unmapped_model_vars"].append(model_var)
             if verbose:
                print(f"MCP: Model variable '{model_var}' has no ontology term mapped.")
             continue

        if ontology_term in defined_term_keys:
            results["valid_mappings"][model_var] = ontology_term
        else:
            results["invalid_terms"][model_var] = ontology_term
            if verbose:
                print(f"MCP: Ontology term '{ontology_term}' (for model var '{model_var}') is not in the defined set of terms.")
    
    if verbose:
        print(f"MCP: Validation complete. Valid: {len(results['valid_mappings'])}, Invalid: {len(results['invalid_terms'])}")
    return results

def generate_ontology_report_for_file(gnn_file_path: str, parsed_annotations: dict, validation_results: dict = None) -> str:
    """
    Generates a markdown formatted report string for a single GNN file's ontology annotations.
    """
    report_parts = [f"### Ontological Annotations for `{gnn_file_path}`\n"]
    
    if not parsed_annotations:
        report_parts.append("- No `ActInfOntologyAnnotation` section found or section was empty.\n")
        return "".join(report_parts)

    report_parts.append("#### Mappings:\n")
    for var, term in parsed_annotations.items():
        report_parts.append(f"- `{var}` -> `{term}`")
        if validation_results and validation_results.get("invalid_terms", {}).get(var) == term:
            report_parts.append(" (**INVALID TERM**)")
        report_parts.append("\n")
    
    if validation_results:
        invalid_count = len(validation_results.get("invalid_terms", {}))
        if invalid_count > 0:
            report_parts.append(f"\n**Validation Summary**: {invalid_count} unrecognized ontological term(s) found.\n")
            # for var, term in validation_results["invalid_terms"].items():
            #     report_parts.append(f"  - Model variable `{var}` uses unrecognized term `{term}`.\n")
        else:
            report_parts.append("\n**Validation Summary**: All ontological terms are recognized.\n")
            
    report_parts.append("\n---\n")
    return "".join(report_parts)

# --- MCP Tool Wrappers ---

def parse_gnn_ontology_section_mcp(gnn_file_content: str, verbose: bool = False) -> dict:
    """
    MCP Tool: Parses the 'ActInfOntologyAnnotation' section from GNN file content.
    Returns a dictionary mapping model variables to ontological terms.
    """
    # The original function already returns a dict, but let's ensure a consistent success/error structure
    try:
        annotations = parse_gnn_ontology_section(gnn_file_content, verbose)
        return {
            "success": True,
            "annotations": annotations
        }
    except Exception as e:
        logger.error(f"MCP parse_gnn_ontology_section_mcp failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "annotations": {}
        }

def load_defined_ontology_terms_mcp(ontology_terms_path: str, verbose: bool = False) -> dict:
    """
    MCP Tool: Loads defined ontological terms from a JSON file.
    Returns a dictionary of defined ontology terms.
    """
    try:
        defined_terms = load_defined_ontology_terms(ontology_terms_path, verbose)
        # Check if terms were actually loaded or if an error occurred (e.g., file not found)
        # The original function returns {} on error and logs warnings.
        # For MCP, more explicit success/failure is better.
        if not defined_terms and not Path(ontology_terms_path).exists():
             return {
                "success": False,
                "error": f"Ontology terms file not found: {ontology_terms_path}",
                "defined_terms": {}
            }
        return {
            "success": True,
            "defined_terms": defined_terms,
            "source_path": ontology_terms_path
        }
    except Exception as e:
        logger.error(f"MCP load_defined_ontology_terms_mcp failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "defined_terms": {}
        }

def validate_annotations_mcp(parsed_annotations: dict, defined_terms: dict, verbose: bool = False) -> dict:
    """
    MCP Tool: Validates parsed GNN annotations against a set of defined ontological terms.
    Returns a dictionary with validation results.
    """
    try:
        validation_results = validate_annotations(parsed_annotations, defined_terms, verbose)
        return {
            "success": True,
            "validation_results": validation_results
        }
    except Exception as e:
        logger.error(f"MCP validate_annotations_mcp failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "validation_results": {}
        }

# --- MCP Registration Function ---
def register_tools(mcp_instance):
    """Register Ontology module tools with the MCP instance."""

    mcp_instance.register_tool(
        name="ontology.parse_gnn_ontology_section",
        func=parse_gnn_ontology_section_mcp,
        schema={
            "gnn_file_content": {"type": "string", "description": "The full string content of a GNN file."},
            "verbose": {"type": "boolean", "description": "Enable verbose logging during parsing.", "optional": True}
        },
        description="Parses the 'ActInfOntologyAnnotation' section from GNN file content and extracts mappings."
    )

    mcp_instance.register_tool(
        name="ontology.load_defined_ontology_terms",
        func=load_defined_ontology_terms_mcp,
        schema={
            "ontology_terms_path": {"type": "string", "description": "Path to the JSON file containing defined ontology terms."},
            "verbose": {"type": "boolean", "description": "Enable verbose logging during loading.", "optional": True}
        },
        description="Loads defined ontological terms from a specified JSON file."
    )

    mcp_instance.register_tool(
        name="ontology.validate_annotations",
        func=validate_annotations_mcp,
        schema={
            "parsed_annotations": {"type": "object", "description": "Dictionary of {model_var: ontology_term} from a GNN file."},
            "defined_terms": {"type": "object", "description": "Dictionary of {ontology_term: description} from a definition file."},
            "verbose": {"type": "boolean", "description": "Enable verbose logging during validation.", "optional": True}
        },
        description="Validates parsed GNN annotations against a set of defined ontological terms."
    )
    logger.info("Ontology module MCP tools registered.")


if __name__ == '__main__':
    print("Ontology Module MCP - Self-Test")

    # Test parse_gnn_ontology_section
    sample_gnn_content_ok = """
## GNNSection
SomeModel

## StateSpaceBlock
s_t[2,1,type=float]
A_matrix[2,2,type=float]

## ActInfOntologyAnnotation
s_t=HiddenState
A_matrix=TransitionMatrix
# This is a comment
malformed_line_no_equals
empty_var=
    =empty_term
good_var = GoodTerm

## Footer
End of model
    """
    sample_gnn_content_no_section = """
## GNNSection
NoOntology
    """
    sample_gnn_content_empty_section = """
## ActInfOntologyAnnotation

## Footer
    """

    print("\n--- Testing parse_gnn_ontology_section ---")
    parsed_ok = parse_gnn_ontology_section(sample_gnn_content_ok, verbose=True)
    # Expected: {'s_t': 'HiddenState', 'A_matrix': 'TransitionMatrix', 'good_var': 'GoodTerm'}
    assert parsed_ok == {'s_t': 'HiddenState', 'A_matrix': 'TransitionMatrix', 'good_var': 'GoodTerm'}
    print(f"Parsed OK: {parsed_ok}")
    
    parsed_no_section = parse_gnn_ontology_section(sample_gnn_content_no_section, verbose=True)
    assert parsed_no_section == {}
    print(f"Parsed No Section: {parsed_no_section}")

    parsed_empty_section = parse_gnn_ontology_section(sample_gnn_content_empty_section, verbose=True)
    assert parsed_empty_section == {}
    print(f"Parsed Empty Section: {parsed_empty_section}")

    # Test load_defined_ontology_terms
    print("\n--- Testing load_defined_ontology_terms ---")
    # Use the newly created standard terms file for a more realistic test
    # Ensure this path is relative to where mcp.py might be run from if __name__ == '__main__'
    # Assuming mcp.py is in src/ontology/, so ontology/act_inf_ontology_terms.json is ../ontology/act_inf_ontology_terms.json
    # For robustness in testing, let's try to construct path relative to this script file.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    standard_terms_file = os.path.join(script_dir, "act_inf_ontology_terms.json")

    if not os.path.exists(standard_terms_file):
        print(f"MCP Self-Test WARNING: Standard terms file not found at {standard_terms_file}. Skipping some tests.")
        # Create a dummy for basic tests to pass if main file is missing during isolated test
        temp_terms_file = "temp_ontology_terms.json"
        valid_terms_data = {"HiddenState": {"description": "A state not directly observable."}, "TransitionMatrix": {"description":"Defines state transitions."}}
        with open(temp_terms_file, 'w') as f:
            json.dump(valid_terms_data, f)
        terms_to_load_path = temp_terms_file
        expected_loaded_terms = valid_terms_data
    else:
        terms_to_load_path = standard_terms_file
        # Load expected directly from file to ensure test matches content
        with open(standard_terms_file, 'r') as f_expected:
            expected_loaded_terms = json.load(f_expected)
    
    loaded_terms = load_defined_ontology_terms(terms_to_load_path, verbose=True)
    assert loaded_terms == expected_loaded_terms
    print(f"Loaded terms from {terms_to_load_path}: {len(loaded_terms)} terms")

    load_defined_ontology_terms("non_existent_file.json", verbose=True) # Should not crash

    # Test with a malformed JSON file
    malformed_json_file = "temp_malformed_ontology_terms.json"
    with open(malformed_json_file, 'w') as f: # Malformed JSON
        f.write("not json")
    load_defined_ontology_terms(malformed_json_file, verbose=True)
    os.remove(malformed_json_file) # Clean up malformed file

    # Test validate_annotations using terms from the standard file if available
    print("\n--- Testing validate_annotations ---")
    annotations_to_validate = {
        "s1": "HiddenState", 
        "A": "TransitionMatrix", 
        "obs": "Observation", 
        "param_x": "UnknownTerm", # This term won't be in act_inf_ontology_terms.json
        "model_EFE": "ExpectedFreeEnergy"
    }
    
    # Use the loaded_terms from standard_terms_file (or the fallback dummy)
    defined_terms_for_validation = loaded_terms 
    
    validation_res = validate_annotations(annotations_to_validate, defined_terms_for_validation, verbose=True)
    
    # Dynamically create expected results based on what's in defined_terms_for_validation
    expected_valid = {}
    expected_invalid = {}
    for k, v in annotations_to_validate.items():
        if v in defined_terms_for_validation:
            expected_valid[k] = v
        else:
            expected_invalid[k] = v
            
    assert validation_res["valid_mappings"] == expected_valid
    assert validation_res["invalid_terms"] == expected_invalid
    print(f"Validation results: Valid: {len(validation_res['valid_mappings'])}, Invalid: {len(validation_res['invalid_terms'])}")
    print(f"Validated against: {list(defined_terms_for_validation.keys())[:5]}... and {len(defined_terms_for_validation)-5} more terms")

    # Test generate_ontology_report_for_file
    print("\n--- Testing generate_ontology_report_for_file ---")
    report_str = generate_ontology_report_for_file("example.gnn.md", annotations_to_validate, validation_res)
    print("Generated Report String:\n" + report_str)
    assert "example.gnn.md" in report_str
    assert "`s1` -> `HiddenState`" in report_str
    assert "`obs` -> `Observation` (**INVALID TERM**)" in report_str
    assert "2 unrecognized ontological term(s) found" in report_str
    
    report_no_val = generate_ontology_report_for_file("example_no_val.gnn.md", annotations_to_validate)
    print("Generated Report (No Validation) String:\n" + report_no_val)
    assert "(**INVALID TERM**)" not in report_no_val

    # Clean up temp_terms_file if it was created
    if not os.path.exists(standard_terms_file) and os.path.exists(terms_to_load_path):
        os.remove(terms_to_load_path) 
    print("\nSelf-tests complete.") 