"""
Specialized GNN Exporters for Text-Based Formats (Summary, DSL)
"""
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def export_to_plaintext_summary(gnn_model: dict, output_file_path: str):
    """Exports a human-readable plain text summary of the GNN model."""
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(f"GNN Model Summary: {gnn_model.get('name', gnn_model.get('metadata', {}).get('name', 'N/A'))}\n")
            f.write(f"Source File: {gnn_model.get('file_path', 'N/A')}\n\n")

            f.write("Metadata:\n")
            for k, v in gnn_model.get('metadata', {}).items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

            f.write(f"States ({len(gnn_model.get('states', []))}):\n")
            for state in gnn_model.get('states', []):
                f.write(f"  - ID: {state.get('id')}")
                attrs = [f"{k}={v}" for k,v in state.items() if k!='id']
                if attrs: f.write(f" ({', '.join(attrs)})")
                f.write("\n")
            f.write("\n")

            f.write(f"Initial Parameters ({len(gnn_model.get('initial_parameters', {}))}):\n")
            for k, v in gnn_model.get('initial_parameters', {}).items():
                 f.write(f"  {k}: {v}\n") # v here is already parsed by _parse_matrix_string
            f.write("\n")
            
            f.write(f"General Parameters ({len(gnn_model.get('parameters', {}))}):\n")
            for k, v in gnn_model.get('parameters', {}).items():
                 f.write(f"  {k}: {v}\n")
            f.write("\n")
            
            f.write(f"Observations ({len(gnn_model.get('observations', []))}):\n")
            for obs in gnn_model.get('observations', []):
                f.write(f"  - ID: {obs.get('id')}")
                attrs = [f"{k}={v}" for k,v in obs.items() if k!='id']
                if attrs: f.write(f" ({', '.join(attrs)})")
                f.write("\n")
            f.write("\n")

            f.write(f"Transitions ({len(gnn_model.get('transitions', []))}):\n")
            for trans in gnn_model.get('transitions', []):
                attr_str = ", ".join([f"{k}={v}" for k, v in trans.get('attributes', {}).items()])
                f.write(f"  - {trans.get('source')} -> {trans.get('target')}")
                if attr_str: f.write(f" : {attr_str}")
                f.write("\n")
            f.write("\n")

            f.write(f"Ontology Annotations ({len(gnn_model.get('ontology_annotations', {}))}):\n")
            for k, v in gnn_model.get('ontology_annotations', {}).items():
                f.write(f"  {k} = {v}\n")
            f.write("\n")

        logger.debug(f"Successfully exported GNN model to plain text summary: {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to export GNN model to plain text summary {output_file_path}: {e}", exc_info=True)
        raise

def export_to_plaintext_dsl(gnn_model: dict, output_file_path: str):
    """
    Exports the GNN model back to a DSL-like format using the raw sections.
    This aims to reconstruct the original .gnn.md file structure.
    """
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            # A predefined section order might be better for consistency.
            # For now, using a common order, then remaining raw_sections.
            # This order is an example and might need refinement based on GNN standards.
            preferred_order = [
                "GNNSection", "ImageFromPaper", "GNNVersionAndFlags", "ModelName", 
                "ModelAnnotation", "StateSpaceBlock", "ParameterBlock", 
                "InitialParameterization", "ObservationBlock", "Connections", 
                "TransitionBlock", "Equations", "Time", "ActInfOntologyAnnotation",
                "ModelParameters", "Footer", "Signature"
            ]
            
            written_sections = set()
            raw_sections = gnn_model.get('raw_sections', {})

            for section_name in preferred_order:
                if section_name in raw_sections:
                    # Check for the special case of InitialParameterization_raw_content due to previous fix
                    if section_name == "InitialParameterization" and "InitialParameterization_raw_content" in raw_sections:
                        section_content = raw_sections["InitialParameterization_raw_content"]
                    else:
                        section_content = raw_sections[section_name]
                    
                    f.write(f"## {section_name}\n")
                    f.write(f"{section_content}\n\n")
                    written_sections.add(section_name)
                    if section_name == "InitialParameterization": # also mark the _raw_content as written if used
                         written_sections.add("InitialParameterization_raw_content")
                         written_sections.add("InitialParameterization_parsed_kv") # mark helper keys as written

            # Write any remaining sections not in preferred_order
            for section_name, section_content in raw_sections.items():
                if section_name not in written_sections:
                    f.write(f"## {section_name}\n")
                    f.write(f"{section_content}\n\n")
            
        logger.debug(f"Successfully exported GNN model to plain text DSL: {output_file_path}")
    except Exception as e:
        logger.error(f"Failed to export GNN model to plain text DSL {output_file_path}: {e}", exc_info=True)
        raise 