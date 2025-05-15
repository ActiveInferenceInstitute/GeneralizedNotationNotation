#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 11: LLM Operations

This script utilizes Large Language Models (LLMs) to perform tasks such as:
- Summarizing GNN files.
- Explaining aspects of GNN models.
- Generating professional summaries for reports.
- Identifying key components, describing connectivity, inferring purpose, and explaining ontology mappings.

It relies on the llm_operations module.
"""

import argparse
import logging
import sys
from pathlib import Path
import datetime
import os

logger = logging.getLogger(__name__) # GNN_Pipeline.11_llm or __main__

# Attempt to import necessary functions from the llm module
# This structure assumes 11_llm.py is in src/ and other llm files are in src/llm/
try:
    from llm import llm_operations
    from llm import mcp as llm_mcp # To potentially call ensure_llm_tools_registered
except ImportError:
    # Fallback for standalone execution or if src isn't correctly in path
    try:
        # If 11_llm.py is in src/, and llm_operations is in src/llm/
        # We need to ensure src/ is in the path for `from llm import ...`
        current_script_dir = Path(__file__).resolve().parent # src/
        if str(current_script_dir.parent) not in sys.path: # Add project root
            sys.path.insert(0, str(current_script_dir.parent))
        
        # Now try importing assuming structure is src/llm/llm_operations.py
        from llm import llm_operations
        from llm import mcp as llm_mcp
    except ImportError as e_fallback_refined:
        logger.critical(f"Failed to import llm_operations or llm.mcp in 11_llm.py (after path adjustment): {e_fallback_refined}")
        llm_operations = None
        llm_mcp = None


def process_gnn_with_llm(gnn_file_path: Path, output_dir_for_file: Path, verbose: bool = False):
    """Processes a single GNN file with various LLM tasks and saves results."""
    if not llm_operations:
        logger.error(f"LLM operations module not available. Skipping {gnn_file_path.name}")
        # Return False and an empty dict for results to maintain consistent return type
        return False, {"file_name": gnn_file_path.name} 

    results = {
        "file_name": gnn_file_path.name,
        "summary": "",
        "explanation": "",
        "professional_summary": "",
        "key_components": "",
        "connectivity_description": "",
        "inferred_purpose": "",
        "ontology_explanation": ""
    }
    all_tasks_successful = True

    logger.info(f"  🧠 Processing GNN file with LLM: {gnn_file_path.name}")

    try:
        gnn_content = gnn_file_path.read_text(encoding='utf-8')
    except Exception as e:
        logger.error(f"    ❌ Error reading GNN file {gnn_file_path.name}: {e}", exc_info=verbose)
        # For all result fields, indicate reading error
        for key in results:
            if key != "file_name":
                results[key] = f"Error reading file: {e}"
        return False, results

    # Task 1: General Summary
    try:
        logger.debug(f"    Requesting summary for {gnn_file_path.name}")
        task_desc_summary = "Provide a concise summary of this GNN model, highlighting its key components (ModelName, primary states/observations, and main connections)."
        summary = llm_operations.get_llm_response(
            llm_operations.construct_prompt(
                contexts=[f"GNN File Content ({gnn_file_path.name}):\n{gnn_content}"],
                task_description=task_desc_summary
            )
        )
        results["summary"] = summary
        with open(output_dir_for_file / f"{gnn_file_path.stem}_summary.txt", 'w', encoding='utf-8') as f:
            f.write(summary)
        logger.info(f"    ✅ Summary generated for {gnn_file_path.name}")
    except Exception as e:
        logger.error(f"    ❌ Error generating summary for {gnn_file_path.name}: {e}", exc_info=verbose)
        results["summary"] = f"Error: {e}"
        all_tasks_successful = False

    # Task 2: General Explanation
    try:
        logger.debug(f"    Requesting explanation for {gnn_file_path.name}")
        task_desc_explanation = "Provide a general explanation of the GNN model described above. Cover its potential purpose, the nature of its state space, and how its components might interact. Aim for clarity for someone broadly familiar with modeling."
        explanation = llm_operations.get_llm_response(
            llm_operations.construct_prompt(
                contexts=[f"GNN File Content ({gnn_file_path.name}):\n{gnn_content}"],
                task_description=task_desc_explanation
            )
        )
        results["explanation"] = explanation
        with open(output_dir_for_file / f"{gnn_file_path.stem}_explanation.txt", 'w', encoding='utf-8') as f:
            f.write(explanation)
        logger.info(f"    ✅ Explanation generated for {gnn_file_path.name}")
    except Exception as e:
        logger.error(f"    ❌ Error generating explanation for {gnn_file_path.name}: {e}", exc_info=verbose)
        results["explanation"] = f"Error: {e}"
        all_tasks_successful = False

    # Task 3: Professional Summary
    try:
        logger.debug(f"    Requesting professional summary for {gnn_file_path.name}")
        task_desc_prof_summary = "Generate a professional, publication-quality summary of the GNN model. The summary should be targeted at fellow researchers. It should be well-structured, highlight key model characteristics, and be suitable for inclusion in a research paper or technical report."
        prof_summary = llm_operations.get_llm_response(
            llm_operations.construct_prompt(
                contexts=[f"GNN Model Specification ({gnn_file_path.name}):\n{gnn_content}"],
                task_description=task_desc_prof_summary
            ),
            model="gpt-4o-mini" # Ensures gpt-4o-mini is used, max_tokens will be default from llm_operations
        )
        results["professional_summary"] = prof_summary
        with open(output_dir_for_file / f"{gnn_file_path.stem}_professional_summary.md", 'w', encoding='utf-8') as f:
            f.write(f"# Professional Summary for {gnn_file_path.name}\n\n{prof_summary}")
        logger.info(f"    ✅ Professional summary generated for {gnn_file_path.name}")
    except Exception as e:
        logger.error(f"    ❌ Error generating professional summary for {gnn_file_path.name}: {e}", exc_info=verbose)
        results["professional_summary"] = f"Error: {e}"
        all_tasks_successful = False
        
    # Task 4: Identify Key Components
    try:
        logger.debug(f"    Requesting key components identification for {gnn_file_path.name}")
        task_desc_key_comp = "Identify and list the key state variables (including their types, dimensions, and any specified value ranges/labels), observation modalities (with dimensions and labels if provided), and control factors (with dimensions/action labels if provided) defined in this GNN model. Focus on what is explicitly defined in the StateSpaceBlock and related sections."
        components_analysis = llm_operations.get_llm_response(
            llm_operations.construct_prompt(
                contexts=[f"GNN File Content ({gnn_file_path.name}):\n{gnn_content}"],
                task_description=task_desc_key_comp
            )
        )
        results["key_components"] = components_analysis
        with open(output_dir_for_file / f"{gnn_file_path.stem}_key_components.txt", 'w', encoding='utf-8') as f:
            f.write(components_analysis)
        logger.info(f"    ✅ Key components identified for {gnn_file_path.name}")
    except Exception as e:
        logger.error(f"    ❌ Error identifying key components for {gnn_file_path.name}: {e}", exc_info=verbose)
        results["key_components"] = f"Error: {e}"
        all_tasks_successful = False

    # Task 5: Describe Connectivity
    try:
        logger.debug(f"    Requesting connectivity description for {gnn_file_path.name}")
        task_desc_connectivity = "Describe the primary relationships and dependencies between the components (states, observations, controls) as outlined in the ## Connections block. Explain what these connections imply about the model's structure and potential information flow. If parameters like A, B, C, D, E matrices/vectors are mentioned or implied by connections, briefly note their roles."
        connectivity_description = llm_operations.get_llm_response(
            llm_operations.construct_prompt(
                contexts=[f"GNN File Content ({gnn_file_path.name}):\n{gnn_content}"],
                task_description=task_desc_connectivity
            )
        )
        results["connectivity_description"] = connectivity_description
        with open(output_dir_for_file / f"{gnn_file_path.stem}_connectivity_description.txt", 'w', encoding='utf-8') as f:
            f.write(connectivity_description)
        logger.info(f"    ✅ Connectivity described for {gnn_file_path.name}")
    except Exception as e:
        logger.error(f"    ❌ Error describing connectivity for {gnn_file_path.name}: {e}", exc_info=verbose)
        results["connectivity_description"] = f"Error: {e}"
        all_tasks_successful = False

    # Task 6: Infer Purpose/Domain
    try:
        logger.debug(f"    Requesting purpose/domain inference for {gnn_file_path.name}")
        task_desc_purpose = "Based on the entire GNN file content (including ModelName, names of states/observations/actions, structure of connections, any initial parameterizations, and ActInfOntologyAnnotation if present), infer and describe the likely domain, purpose, or type of system being modeled. Provide a justification for your inference, citing specific parts of the GNN file."
        inferred_purpose = llm_operations.get_llm_response(
            llm_operations.construct_prompt(
                contexts=[f"GNN File Content ({gnn_file_path.name}):\n{gnn_content}"],
                task_description=task_desc_purpose
            )
        )
        results["inferred_purpose"] = inferred_purpose
        with open(output_dir_for_file / f"{gnn_file_path.stem}_inferred_purpose.txt", 'w', encoding='utf-8') as f:
            f.write(inferred_purpose)
        logger.info(f"    ✅ Purpose/domain inferred for {gnn_file_path.name}")
    except Exception as e:
        logger.error(f"    ❌ Error inferring purpose/domain for {gnn_file_path.name}: {e}", exc_info=verbose)
        results["inferred_purpose"] = f"Error: {e}"
        all_tasks_successful = False

    # Task 7: Ontology Mapping Explanation (Conditional)
    try:
        if "## ActInfOntologyAnnotation" in gnn_content:
            logger.debug(f"    Requesting ontology mapping explanation for {gnn_file_path.name}")
            task_desc_ontology = "The GNN file includes an ## ActInfOntologyAnnotation block. Explain how the model components are mapped to the Active Inference Ontology terms found in this block. What do these mappings tell us about the intended interpretation of the model within the Active Inference framework? Discuss any specific terms used and their relevance."
            ontology_explanation = llm_operations.get_llm_response(
                llm_operations.construct_prompt(
                    contexts=[f"GNN File Content ({gnn_file_path.name}):\n{gnn_content}"],
                    task_description=task_desc_ontology
                )
            )
            results["ontology_explanation"] = ontology_explanation
            with open(output_dir_for_file / f"{gnn_file_path.stem}_ontology_explanation.txt", 'w', encoding='utf-8') as f:
                f.write(ontology_explanation)
            logger.info(f"    ✅ Ontology mapping explained for {gnn_file_path.name}")
        else:
            logger.info(f"    ℹ️ No ActInfOntologyAnnotation block found in {gnn_file_path.name}. Skipping ontology explanation task.")
            results["ontology_explanation"] = "N/A - No ActInfOntologyAnnotation block found."
    except Exception as e:
        logger.error(f"    ❌ Error explaining ontology mapping for {gnn_file_path.name}: {e}", exc_info=verbose)
        results["ontology_explanation"] = f"Error: {e}"
        all_tasks_successful = False

    return all_tasks_successful, results

def main(args: argparse.Namespace) -> int:
    """Main function for the LLM operations step."""
    logger.info(f"▶️ Starting Step 11: LLM Operations ({Path(__file__).name})")
    if args.verbose:
        logger.debug(f"  Pipeline arguments received: {args}")

    if not llm_operations or not llm_mcp:
        logger.critical("LLM modules (llm_operations, llm.mcp) not loaded. Cannot proceed.")
        return 1

    try:
        llm_operations.load_api_key()
        if hasattr(llm_mcp, 'ensure_llm_tools_registered'):
            llm_mcp.ensure_llm_tools_registered()
    except ValueError as e:
        logger.error(f"❌ OpenAI API Key not configured: {e}. This step cannot run.")
        return 1
    except Exception as e_key_load:
        logger.error(f"❌ Unexpected error loading API key or ensuring tool registration: {e_key_load}. This step cannot run.", exc_info=True)
        return 1
    
    gnn_source_dir = Path(args.target_dir).resolve()
    if not gnn_source_dir.is_dir():
        logger.error(f"Target directory for GNN source files not found: {gnn_source_dir}")
        return 1

    llm_step_output_dir = Path(args.output_dir).resolve() / "llm_processing_step"
    try:
        llm_step_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"LLM processing outputs will be saved in: {llm_step_output_dir}")
    except OSError as e:
        logger.error(f"Failed to create LLM output directory {llm_step_output_dir}: {e}")
        return 1

    search_pattern = "**/*.md" if args.recursive else "*.md"
    gnn_files = list(gnn_source_dir.glob(search_pattern))

    if not gnn_files:
        logger.info(f"No GNN files ('{search_pattern}') found in '{gnn_source_dir}'. Nothing to process with LLM.")
        return 0

    logger.info(f"Found {len(gnn_files)} GNN files to process with LLM from '{gnn_source_dir}'.")

    overall_success_flag = True
    all_file_results = []
    processed_count = 0

    for gnn_file in gnn_files:
        file_specific_llm_output_dir = llm_step_output_dir / gnn_file.stem
        try:
            file_specific_llm_output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Could not create output directory for {gnn_file.name}: {file_specific_llm_output_dir}. Error: {e}. Skipping.")
            overall_success_flag = False
            all_file_results.append({"file_name": gnn_file.name, "error": f"Could not create output dir: {e}"})
            continue
            
        file_success, file_llm_results = process_gnn_with_llm(gnn_file, file_specific_llm_output_dir, args.verbose)
        all_file_results.append(file_llm_results)
        if not file_success:
            overall_success_flag = False
        processed_count += 1

    report_path = llm_step_output_dir / "11_llm_processing_report.md"
    try:
        with open(report_path, 'w', encoding='utf-8') as f_report:
            f_report.write(f"# GNN Processing Pipeline - Step 11: LLM Operations Report\n\n")
            f_report.write(f"🗓️ Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f_report.write(f"🎯 GNN Source Directory: `{gnn_source_dir}` (Pattern: `{search_pattern}`)\n")
            f_report.write(f"Processed {processed_count}/{len(gnn_files)} files.\n\n")
            f_report.write("---\n")

            for res in all_file_results:
                f_report.write(f"\n## Results for: `{res.get('file_name', 'Unknown File')}`\n\n")
                if "error" in res: # Handle cases where directory creation failed
                    f_report.write(f"**ERROR PROCESSING FILE**: {res['error']}\n\n")
                    f_report.write("---\n")
                    continue

                def write_section(title, key, filename_suffix, is_md=False):
                    f_report.write(f"### {title}\n")
                    f_report.write(f"Output file: `{res['file_name'].replace('.md', '')}{filename_suffix}`\n")
                    content = res.get(key, 'N/A - Task may have failed or been skipped.')
                    if is_md:
                         f_report.write(f"*Content written to the markdown file.*\n")
                         # Optionally, include a short preview for .md if desired
                         # f_report.write(f"```markdown\n{content[:200]}{'...' if len(content) > 200 else ''}\n```\n\n")
                    else:
                        f_report.write(f"```text\n{content[:1000]}{'...' if len(content) > 1000 else ''}\n```\n\n")

                write_section("Summary Text Output", "summary", "_summary.txt")
                write_section("Explanation Text Output", "explanation", "_explanation.txt")
                write_section("Professional Summary Markdown Output", "professional_summary", "_professional_summary.md", is_md=True)
                write_section("Key Components Analysis Output", "key_components", "_key_components.txt")
                write_section("Connectivity Description Output", "connectivity_description", "_connectivity_description.txt")
                write_section("Inferred Purpose/Domain Output", "inferred_purpose", "_inferred_purpose.txt")
                write_section("Ontology Mapping Explanation Output", "ontology_explanation", "_ontology_explanation.txt")
                
                f_report.write("---\n")
        logger.info(f"LLM Operations report saved to: {report_path}")
    except Exception as e:
        logger.error(f"Failed to write LLM operations report: {e}", exc_info=True)
        overall_success_flag = False

    if overall_success_flag:
        logger.info(f"✅ Step 11: LLM Operations ({Path(__file__).name}) - COMPLETED successfully.")
        return 0
    else:
        logger.error(f"❌ Step 11: LLM Operations ({Path(__file__).name}) - COMPLETED with errors.")
        return 1

if __name__ == '__main__':
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    parser = argparse.ArgumentParser(description="GNN Pipeline - Step 11: LLM Operations (Standalone)")
    script_dir = Path(__file__).resolve().parent # src/
    # Default target for GNN .md files for this step
    default_target_dir = script_dir / "gnn" / "examples" 
    # Default main output dir for the whole pipeline
    default_pipeline_output_dir = script_dir.parent / "output" 

    parser.add_argument("--target-dir", default=str(default_target_dir),
                        help=f"Target directory for GNN source files (default: {default_target_dir})")
    parser.add_argument("--output-dir", default=str(default_pipeline_output_dir),
                        help=f"Main pipeline output directory (default: {default_pipeline_output_dir})")
    parser.add_argument("--recursive", action="store_true", default=True, # Make default True as per main.py
                        help="Recursively search for GNN files in the target directory.")
    parser.add_argument("--no-recursive", dest='recursive', action='store_false',
                        help="Disable recursive search for GNN files.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output for this script.")
    
    parsed_args = parser.parse_args()

    if parsed_args.verbose:
        logging.getLogger().setLevel(logging.DEBUG) # Set root logger to DEBUG
        logger.setLevel(logging.DEBUG) # Set this script's logger to DEBUG
        if llm_operations: # If module loaded, set its logger to DEBUG too
            logging.getLogger(llm_operations.__name__).setLevel(logging.DEBUG)
        if llm_mcp: # If module loaded, set its logger to DEBUG too
             logging.getLogger(llm_mcp.__name__).setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled for standalone run of 11_llm.py.")

    sys.exit(main(parsed_args)) 