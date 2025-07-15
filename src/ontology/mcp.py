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

logger = logging.getLogger(__name__)

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
        # Updated pattern to be more flexible with section boundaries
        match = re.search(r"^## ActInfOntologyAnnotation\s*$\n(.*?)(?=^## |^\Z)", gnn_file_content, re.MULTILINE | re.DOTALL)
        if not match:
            if verbose:
                logger.debug("'ActInfOntologyAnnotation' section not found.")
            return annotations

        section_content = match.group(1).strip()
        if not section_content:
            if verbose:
                logger.debug("'ActInfOntologyAnnotation' section is empty.")
            return annotations

        if verbose:
            logger.debug(f"Found ActInfOntologyAnnotation section content: {repr(section_content)}")

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
                    if verbose:
                        logger.debug(f"Parsed annotation: {key} = {value}")
                else:
                    if verbose:
                        logger.debug(f"Malformed line {i+1} in ActInfOntologyAnnotation: '{line}' - skipping.")
            else:
                if verbose:
                    logger.debug(f"Line {i+1} in ActInfOntologyAnnotation does not contain '=': '{line}' - skipping.")
        
        if verbose:
            logger.debug(f"Parsed annotations: {annotations}")
            
    except Exception as e:
        logger.error(f"Error parsing ActInfOntologyAnnotation section: {e}", exc_info=True)
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
        
        logger.debug(f"Attempting to load ontology terms from: {os.path.abspath(ontology_terms_path)}")

        with open(ontology_terms_path, 'r', encoding='utf-8-sig') as f: # Changed encoding
            data = json.load(f)
        if isinstance(data, dict):
            defined_terms = data
            logger.debug(f"Loaded {len(defined_terms)} ontology terms from {ontology_terms_path}.")
        else:
            logger.error(f"Ontology terms file {ontology_terms_path} does not contain a root JSON object. Data type: {type(data)}")
            return {}
    except FileNotFoundError:
        logger.error(f"Ontology terms file not found: {ontology_terms_path} (Absolute: {os.path.abspath(ontology_terms_path)})")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {ontology_terms_path} (Absolute: {os.path.abspath(ontology_terms_path)}): {e}", exc_info=True)
        return {}
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading ontology terms from {ontology_terms_path} (Absolute: {os.path.abspath(ontology_terms_path)}): {e}", exc_info=True)
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
             logger.debug(f"Model variable '{model_var}' has no ontology term mapped.")
             continue

        if ontology_term in defined_term_keys:
            results["valid_mappings"][model_var] = ontology_term
        else:
            results["invalid_terms"][model_var] = ontology_term
            logger.debug(f"Ontology term '{ontology_term}' (for model var '{model_var}') is not in the defined set of terms.")
    
    logger.debug(f"Validation complete. Valid: {len(results['valid_mappings'])}, Invalid: {len(results['invalid_terms'])}")
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
    # Setup basic logging for standalone test execution if the utility is available
    # This is a simple setup; for robust standalone, use setup_standalone_logging from utils
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("Ontology Module MCP - Self-Test")
    interface = get_mcp_interface()
    logger.info(f"Interface: {interface}\n")

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

    logger.info("--- Testing parse_gnn_ontology_section ---")
    logger.info("Test Case 1: Valid content")
    parsed_ok = parse_gnn_ontology_section(sample_gnn_content_ok, verbose=True)
    expected_annotations_ok = {'s_t': 'HiddenState', 'A_matrix': 'TransitionMatrix', 'good_var': 'GoodTerm'}
    logger.info(f"  Parsed: {parsed_ok}")
    logger.info(f"  Expected: {expected_annotations_ok}")
    assert parsed_ok == expected_annotations_ok

    logger.info("Test Case 2: No section")
    parsed_no_section = parse_gnn_ontology_section(sample_gnn_content_no_section, verbose=True)
    logger.info(f"  Parsed: {parsed_no_section}")
    assert parsed_no_section == {}

    logger.info("Test Case 3: Empty section")
    parsed_empty_section = parse_gnn_ontology_section(sample_gnn_content_empty_section, verbose=True)
    logger.info(f"  Parsed: {parsed_empty_section}")
    assert parsed_empty_section == {}
    logger.info("parse_gnn_ontology_section tests passed.\n")

    # Test load_defined_ontology_terms
    logger.info("--- Testing load_defined_ontology_terms ---")
    # Create a dummy ontology terms file for testing
    dummy_terms_content = {
        "HiddenState": "Represents the unobservable states of the system.",
        "TransitionMatrix": "Defines probabilities of transitioning between hidden states.",
        "GoodTerm": "A valid term for testing."
    }
    dummy_terms_path = "./dummy_ontology_terms.json"
    with open(dummy_terms_path, 'w') as f_dummy:
        json.dump(dummy_terms_content, f_dummy)

    logger.info(f"Test Case 1: Load valid terms from {dummy_terms_path}")
    loaded_terms = load_defined_ontology_terms(dummy_terms_path, verbose=True)
    logger.info(f"  Loaded: {loaded_terms}")
    assert loaded_terms == dummy_terms_content

    logger.info("Test Case 2: File not found")
    non_existent_path = "./non_existent_terms.json"
    loaded_terms_not_found = load_defined_ontology_terms(non_existent_path, verbose=True)
    logger.info(f"  Loaded (should be empty): {loaded_terms_not_found}")
    assert loaded_terms_not_found == {}
    
    # Clean up dummy file
    os.remove(dummy_terms_path)
    logger.info("load_defined_ontology_terms tests passed.\n")

    # Test validate_annotations
    logger.info("--- Testing validate_annotations ---")
    parsed_ann = {"s1": "HiddenState", "A1": "TransitionMatrix", "B1": "UnknownTerm", "s2": "HiddenState"}
    defined_terms_for_test = {"HiddenState": "desc1", "TransitionMatrix": "desc2"}
    
    logger.info("Test Case 1: Mixed valid and invalid terms")
    validation_res = validate_annotations(parsed_ann, defined_terms_for_test, verbose=True)
    expected_valid = {"s1": "HiddenState", "s2": "HiddenState", "A1": "TransitionMatrix"}
    expected_invalid = {"B1": "UnknownTerm"}
    logger.info(f"  Validation Result: {validation_res}")
    logger.info(f"    Expected Valid: {expected_valid}")
    logger.info(f"    Expected Invalid: {expected_invalid}")
    assert validation_res["valid_mappings"] == expected_valid
    assert validation_res["invalid_terms"] == expected_invalid
    logger.info("validate_annotations tests passed.")

    # Test generate_ontology_report_for_file
    logger.info("\n--- Testing generate_ontology_report_for_file ---")
    report_str = generate_ontology_report_for_file("example.gnn.md", parsed_ann, validation_res)
    logger.info("Generated Report String:\n" + report_str)
    assert "example.gnn.md" in report_str
    assert "`s1` -> `HiddenState`" in report_str
    assert "`obs` -> `Observation` (**INVALID TERM**)" in report_str
    assert "2 unrecognized ontological term(s) found" in report_str
    
    report_no_val = generate_ontology_report_for_file("example_no_val.gnn.md", parsed_ann)
    logger.info("Generated Report (No Validation) String:\n" + report_no_val)
    assert "(**INVALID TERM**)" not in report_no_val

    logger.info("\nSelf-tests complete.") 