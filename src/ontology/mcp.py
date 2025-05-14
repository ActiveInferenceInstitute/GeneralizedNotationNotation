"""
Model Context Protocol (MCP) interface for the Ontology module.

This file defines how the Ontology module interacts with the broader GNN system
via MCP. It includes methods for parsing GNN ontology annotations,
validating them against a defined set of terms, and assisting in report generation.
"""
import json
import re
import os

def get_mcp_interface():
    """
    Returns the MCP interface for the Ontology module.
    """
    return {
        "status": "Ontology module MCP active",
        "capabilities": [
            "parse_gnn_ontology_section",
            "load_defined_ontology_terms",
            "validate_annotations",
            "generate_ontology_report_for_file"
        ]
    }

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
        verbose: If True, prints loading information.

    Returns:
        A dictionary of defined ontology terms (term: description/metadata).
        Returns an empty dictionary if the file cannot be loaded or parsed.
    """
    defined_terms = {}
    try:
        # Ensure the path is treated as absolute if it's not already.
        # This helps if the CWD is unexpected.
        # However, the path from the log ('src/ontology/act_inf_ontology_terms.json')
        # is likely relative to the project root.
        # For now, let's rely on the path being correctly resolved by the caller (8_ontology.py)
        
        if verbose: # Added this line to see the path being used
            print(f"MCP: Attempting to load ontology terms from: {os.path.abspath(ontology_terms_path)}")

        with open(ontology_terms_path, 'r', encoding='utf-8-sig') as f: # Changed encoding
            data = json.load(f)
        if isinstance(data, dict):
            defined_terms = data
            if verbose:
                print(f"MCP: Loaded {len(defined_terms)} ontology terms from {ontology_terms_path}.")
        else:
            if verbose:
                print(f"MCP: Error: Ontology terms file {ontology_terms_path} does not contain a root JSON object.")
            else: # Added this else block for non-verbose error
                print(f"MCP_ERROR_TRACE: Ontology terms file {ontology_terms_path} does not contain a root JSON object. Data type: {type(data)}")
            return {}
    except FileNotFoundError:
        if verbose:
            print(f"MCP: Ontology terms file not found: {ontology_terms_path}")
        else: # Added this else block for non-verbose error
            print(f"MCP_ERROR_TRACE: Ontology terms file not found: {ontology_terms_path} (Absolute: {os.path.abspath(ontology_terms_path)})")
        return {}
    except json.JSONDecodeError as e:
        if verbose:
            print(f"MCP: Error decoding JSON from {ontology_terms_path}: {e}")
        else: # Added this else block for non-verbose error
            print(f"MCP_ERROR_TRACE: Error decoding JSON from {ontology_terms_path} (Absolute: {os.path.abspath(ontology_terms_path)}): {e}")
        return {}
    except Exception as e:
        if verbose:
            print(f"MCP: An unexpected error occurred while loading ontology terms from {ontology_terms_path}: {e}")
        else: # Added this else block for non-verbose error
            print(f"MCP_ERROR_TRACE: An unexpected error occurred while loading {ontology_terms_path} (Absolute: {os.path.abspath(ontology_terms_path)}): {e}")
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


if __name__ == '__main__':
    print("Ontology Module MCP - Self-Test")
    interface = get_mcp_interface()
    print(f"Interface: {interface}\n")

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