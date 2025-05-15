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

# Import mcp_instance for use with ensure_llm_tools_registered
try:
    from src.mcp.mcp import mcp_instance
except ImportError:
    logger.warning("Could not import mcp_instance from src.mcp.mcp. Tool registration re-check might not be robust.")
    mcp_instance = None


def process_gnn_with_llm(gnn_file_path: Path, output_dir_for_file: Path, verbose: bool = False):
    """Processes a single GNN file with three comprehensive LLM tasks and saves results."""
    if not llm_operations:
        logger.error(f"LLM operations module not available. Skipping {gnn_file_path.name}")
        return False, {"file_name": gnn_file_path.name}

    results = {
        "file_name": gnn_file_path.name,
        "overview_structure": "",
        "purpose_professional_summary": "",
        "ontology_interpretation": ""
    }
    all_tasks_successful = True

    logger.info(f"  🧠 Processing GNN file with LLM (3 consolidated calls): {gnn_file_path.name}")

    try:
        gnn_content = gnn_file_path.read_text(encoding='utf-8')
    except Exception as e:
        logger.error(f"    ❌ Error reading GNN file {gnn_file_path.name}: {e}", exc_info=verbose)
        for key in results:
            if key != "file_name":
                results[key] = f"Error reading file: {e}"
        return False, results

    # Call 1: Comprehensive Model Overview & Structure
    try:
        logger.debug(f"    Requesting comprehensive overview & structure for {gnn_file_path.name}")
        task_desc_overview = (
            "Provide a comprehensive overview of this GNN model. Your response should include:\n"
            "1.  **Summary:** A concise summary highlighting its key components (ModelName, primary states/observations, and main connections).\n"
            "2.  **General Explanation:** A general explanation covering its potential purpose (at a high level for this section), the nature of its state space, and how its components might interact, suitable for someone broadly familiar with modeling.\n"
            "3.  **Key Components Identification:** Identify and list the key state variables (including their types, dimensions, and any specified value ranges/labels), observation modalities (with dimensions and labels if provided), and control factors (with dimensions/action labels if provided) defined in this GNN model, focusing on what is explicitly defined in the StateSpaceBlock and related sections.\n"
            "4.  **Connectivity Description:** Describe the primary relationships and dependencies between the components (states, observations, controls) as outlined in the ## Connections block. Explain what these connections imply about the model's structure and potential information flow. If parameters like A, B, C, D, E matrices/vectors are mentioned or implied by connections, briefly note their roles."
        )
        overview_structure = llm_operations.get_llm_response(
            llm_operations.construct_prompt(
                contexts=[f"GNN File Content ({gnn_file_path.name}):\\n{gnn_content}"],
                task_description=task_desc_overview
            )
        )
        results["overview_structure"] = overview_structure
        output_file_path_overview = output_dir_for_file / f"{gnn_file_path.stem}_overview_structure.txt"
        with open(output_file_path_overview, 'w', encoding='utf-8') as f:
            f.write(overview_structure)
        logger.info(f"    ✅ Saved comprehensive overview & structure to: {output_file_path_overview} (Length: {len(overview_structure)} chars)")
    except Exception as e:
        logger.error(f"    ❌ Error generating comprehensive overview for {gnn_file_path.name}: {e}", exc_info=verbose)
        results["overview_structure"] = f"Error: {e}"
        all_tasks_successful = False

    # Call 2: Model Purpose, Application & Professional Narrative
    try:
        logger.debug(f"    Requesting purpose, application & professional narrative for {gnn_file_path.name}")
        task_desc_purpose_professional = (
            "Analyze the GNN model for its purpose and generate a professional narrative. Your response should cover:\n"
            "1.  **Inferred Purpose/Domain:** Based on the entire GNN file content (including ModelName, names of states/observations/actions, structure of connections, any initial parameterizations, and ActInfOntologyAnnotation if present), infer and describe the likely domain, purpose, or type of system being modeled. Provide a justification for your inference, citing specific parts of the GNN file.\n"
            "2.  **Professional Summary:** Generate a professional, publication-quality summary of the GNN model. If applicable, consider potential experimental contexts or applications. The summary should be targeted at fellow researchers, be well-structured, highlight key model characteristics, and be suitable for inclusion in a research paper or technical report."
        )
        purpose_professional_summary = llm_operations.get_llm_response(
            llm_operations.construct_prompt(
                contexts=[f"GNN Model Specification ({gnn_file_path.name}):\\n{gnn_content}"],
                task_description=task_desc_purpose_professional
            ),
            model="gpt-4o-mini"
        )
        results["purpose_professional_summary"] = purpose_professional_summary
        output_file_path_summary = output_dir_for_file / f"{gnn_file_path.stem}_purpose_professional_summary.md"
        with open(output_file_path_summary, 'w', encoding='utf-8') as f:
            f.write(f"# Model Purpose, Application & Professional Narrative for {gnn_file_path.name}\\n\\n{purpose_professional_summary}")
        logger.info(f"    ✅ Saved purpose, application & professional narrative to: {output_file_path_summary} (Length: {len(purpose_professional_summary)} chars)")
    except Exception as e:
        logger.error(f"    ❌ Error generating purpose & professional summary for {gnn_file_path.name}: {e}", exc_info=verbose)
        results["purpose_professional_summary"] = f"Error: {e}"
        all_tasks_successful = False

    # Call 3: Ontology Interpretation (Conditional)
    try:
        if "## ActInfOntologyAnnotation" in gnn_content:
            logger.debug(f"    Requesting ontology interpretation for {gnn_file_path.name}")
            task_desc_ontology = (
                "The GNN file includes an ## ActInfOntologyAnnotation block. Provide a detailed interpretation of these ontology mappings:\n"
                "1.  **Explain Mappings:** Clearly explain how the model components (states, observations, parameters like A, B, C, D, E if mentioned in ontology) are mapped to the Active Inference Ontology terms found in this block.\n"
                "2.  **Significance:** What do these mappings tell us about the intended interpretation of the model within the Active Inference framework? Discuss any specific terms used (e.g., 'Belief', 'Likelihood', 'TransitionModel', 'PriorPreferences', 'ExpectedFreeEnergy') and their relevance to the model's function and how it might be used or understood as an Active Inference agent/model."
            )
            ontology_interpretation = llm_operations.get_llm_response(
                llm_operations.construct_prompt(
                    contexts=[f"GNN File Content ({gnn_file_path.name}):\\n{gnn_content}"],
                    task_description=task_desc_ontology
                )
            )
            results["ontology_interpretation"] = ontology_interpretation
            output_file_path_ontology = output_dir_for_file / f"{gnn_file_path.stem}_ontology_interpretation.txt"
            with open(output_file_path_ontology, 'w', encoding='utf-8') as f:
                f.write(ontology_interpretation)
            logger.info(f"    ✅ Saved ontology interpretation to: {output_file_path_ontology} (Length: {len(ontology_interpretation)} chars)")
        else:
            logger.info(f"    ℹ️ No ActInfOntologyAnnotation block found in {gnn_file_path.name}. Skipping ontology interpretation task.")
            results["ontology_interpretation"] = "N/A - No ActInfOntologyAnnotation block found."
    except Exception as e:
        logger.error(f"    ❌ Error generating ontology interpretation for {gnn_file_path.name}: {e}", exc_info=verbose)
        results["ontology_interpretation"] = f"Error: {e}"
        all_tasks_successful = False

    return all_tasks_successful, results

def main(args: argparse.Namespace) -> int:
    """Main function for the LLM operations step (Step 11).

    This function orchestrates the LLM processing of GNN files. It handles:
    - Loading the OpenAI API key.
    - Ensuring LLM tools are registered with MCP (if applicable).
    - Identifying GNN files in the target directory.
    - Invoking `process_gnn_with_llm` for each GNN file to perform LLM tasks.
    - Generating a summary report of the LLM operations.

    Args:
        args (argparse.Namespace):
            Parsed command-line arguments from `main.py` or standalone execution.
            Expected attributes include: target_dir (str/Path), output_dir (str/Path),
            recursive (bool), verbose (bool), llm_tasks (str).

    Returns:
        int: 0 if all GNN files are processed successfully by LLM tasks,
             1 if there are any failures or if critical setup (like API key) fails.
    """
    logger.info(f"▶️ Starting Step 11: LLM Operations ({Path(__file__).name})")
    if args.verbose:
        logger.debug(f"  Pipeline arguments received: {args}")

    if not llm_operations or not llm_mcp:
        logger.critical("LLM modules (llm_operations, llm.mcp) not loaded. Cannot proceed.")
        return 1

    try:
        llm_operations.load_api_key()
        if hasattr(llm_mcp, 'ensure_llm_tools_registered'):
            if mcp_instance: # Only call if mcp_instance was successfully imported
                llm_mcp.ensure_llm_tools_registered(mcp_instance)
            else:
                logger.warning("mcp_instance not available, skipping ensure_llm_tools_registered call.")
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

    processed_files_summary = []
    gnn_files = []
    
    # Refactored GNN file discovery logic
    all_potential_files = []
    if args.recursive:
        all_potential_files.extend(sorted(gnn_source_dir.rglob("*.gnn.md")))
        all_potential_files.extend(sorted(gnn_source_dir.rglob("*.md")))
    else:
        all_potential_files.extend(sorted(gnn_source_dir.glob("*.gnn.md")))
        all_potential_files.extend(sorted(gnn_source_dir.glob("*.md")))

    preferred_files = {} # stem_key -> Path
    for f_path in all_potential_files:
        # Create a stem key by removing .gnn if it exists, then .md implicitly by stem property
        stem_candidate = f_path.stem
        if stem_candidate.endswith(".gnn"): # e.g. file.gnn from file.gnn.md
            stem_key = stem_candidate[:-4]
        else: # e.g. file from file.md
            stem_key = stem_candidate
        
        # Prioritize .gnn.md files
        if f_path.name.endswith(".gnn.md"):
            preferred_files[stem_key] = f_path
        elif stem_key not in preferred_files: # Only add .md if a .gnn.md for the same stem isn't already chosen
            preferred_files[stem_key] = f_path
            
    gnn_files = sorted(list(preferred_files.values()), key=lambda p: p.name)

    # Filter out any .md files that are known to be non-GNN, e.g. README.md
    # This is a heuristic; a better way would be GNN validation or specific include patterns.
    gnn_files = [f for f in gnn_files if not f.name.lower() in ["readme.md", "contributing.md", "license.md"]]
    # Also filter out files that might be in '.git' or other hidden/system directories if rglob goes too deep
    gnn_files = [f for f in gnn_files if not any(part.startswith('.') for part in f.parts)]

    if not gnn_files:
        logger.warning(f"No GNN files (.md, .gnn.md) found in {gnn_source_dir} {'recursively' if args.recursive else ''}.")
    else:
        logger.info(f"Found {len(gnn_files)} GNN files to process with LLM from '{gnn_source_dir}'.")

    total_success_count = 0
    for gnn_file in gnn_files:
        # Create a subdirectory for this specific GNN file's LLM outputs
        file_specific_output_dir = llm_step_output_dir / gnn_file.stem
        try:
            file_specific_output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"  ❌ Failed to create output directory {file_specific_output_dir} for {gnn_file.name}: {e}")
            processed_files_summary.append({
                "file_name": gnn_file.name,
                "status": "Error (Output directory creation failed)",
                "outputs": []
            })
            continue # Skip to the next file

        success, llm_results = process_gnn_with_llm(gnn_file, file_specific_output_dir, args.verbose)
        
        output_files_generated = []
        if success:
            total_success_count += 1
            status_msg = "Success"
            if llm_results.get("overview_structure"): output_files_generated.append(f"{gnn_file.stem}_overview_structure.txt")
            if llm_results.get("purpose_professional_summary"): output_files_generated.append(f"{gnn_file.stem}_purpose_professional_summary.md")
            if llm_results.get("ontology_interpretation") and not llm_results["ontology_interpretation"].startswith("N/A") : 
                output_files_generated.append(f"{gnn_file.stem}_ontology_interpretation.txt")
        else:
            status_msg = "Partial or Total Failure (see logs)"
            # Log which specific outputs were generated despite failure
            if Path(file_specific_output_dir / f"{gnn_file.stem}_overview_structure.txt").exists(): output_files_generated.append(f"{gnn_file.stem}_overview_structure.txt")
            if Path(file_specific_output_dir / f"{gnn_file.stem}_purpose_professional_summary.md").exists(): output_files_generated.append(f"{gnn_file.stem}_purpose_professional_summary.md")
            if Path(file_specific_output_dir / f"{gnn_file.stem}_ontology_interpretation.txt").exists(): output_files_generated.append(f"{gnn_file.stem}_ontology_interpretation.txt")

        processed_files_summary.append({
            "file_name": gnn_file.name,
            "status": status_msg,
            "output_dir": str(file_specific_output_dir.relative_to(Path(args.output_dir).resolve())),
            "outputs_generated": output_files_generated,
            "llm_call_results": llm_results # Store detailed results for the report
        })

    # --- Report Generation ---
    report_path = llm_step_output_dir / f"{Path(__file__).stem}_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Step 11: LLM Operations Report\n")
        f.write(f"*Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write(f"Processed **{len(gnn_files)}** GNN files from `{gnn_source_dir}`.\n")
        f.write(f"Successfully processed (all LLM tasks completed): **{total_success_count}** files.\n")
        f.write(f"Failed or partially failed: **{len(gnn_files) - total_success_count}** files.\n\n")
        f.write(f"LLM outputs are stored in subdirectories within: `{llm_step_output_dir}`\n\n")
        f.write("## Detailed Processing Results:\n")
        for summary_item in processed_files_summary:
            f.write(f"\n### 📄 File: `{summary_item['file_name']}`\n")
            f.write(f"- **Status:** {summary_item['status']}\n")
            f.write(f"- **Output Directory:** `{summary_item['output_dir']}`\n")
            f.write(f"- **Generated Outputs:**\n")
            if summary_item['outputs_generated']:
                for out_file in summary_item['outputs_generated']:
                    f.write(f"  - `{out_file}`\n")
            else:
                f.write("  - (No files generated or error before generation)\n")
            
            # Include text content of LLM calls for quick review in report
            # Overview & Structure
            f.write(f"- **Comprehensive Overview & Structure:**\n")
            f.write("  ```text\n")
            f.write(f"  {summary_item['llm_call_results'].get('overview_structure', 'Not generated or error.')[:1000]}...\n")
            f.write("  ```\n")
            # Purpose & Professional Summary
            f.write(f"- **Purpose, Application & Professional Narrative:**\n")
            f.write("  ```markdown\n")
            f.write(f"  {summary_item['llm_call_results'].get('purpose_professional_summary', 'Not generated or error.')[:1000]}...\n")
            f.write("  ```\n")
            # Ontology Interpretation
            ontology_res = summary_item['llm_call_results'].get('ontology_interpretation', 'Not generated or error.')
            if not ontology_res.startswith("N/A"):
                f.write(f"- **Ontology Interpretation:**\n")
                f.write("  ```text\n")
                f.write(f"  {ontology_res[:1000]}...\n")
                f.write("  ```\n")
            else:
                f.write(f"- **Ontology Interpretation:** N/A\n")

    logger.info(f"LLM Operations report saved to: {report_path}")

    if total_success_count < len(gnn_files):
        logger.warning(f"⚠️ Step 11: LLM Operations ({Path(__file__).name}) - COMPLETED, but with errors for some files. Check logs and report.")
        return 1 # Indicate partial failure
    else:
        logger.info(f"✅ Step 11: LLM Operations ({Path(__file__).name}) - COMPLETED successfully.")
        return 0

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
    parser.add_argument("--llm-tasks", type=str, default="all", # Added to match main.py call
                        help="Comma-separated list of LLM tasks to run (e.g., overview,purpose,ontology). Default: 'all'. Currently, all tasks are run by default.")
    
    parsed_args = parser.parse_args()

    if parsed_args.verbose:
        logging.getLogger().setLevel(logging.DEBUG) # Set root logger to DEBUG
        logger.setLevel(logging.DEBUG) # Set this script's logger to DEBUG
        if llm_operations: # If module loaded, set its logger to DEBUG too
            logging.getLogger(llm_operations.__name__).setLevel(logging.DEBUG)
        if llm_mcp: # If module loaded, set its logger to DEBUG too
             logging.getLogger(llm_mcp.__name__).setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled for standalone run of 11_llm.py.")

    # Quiet noisy libraries if run standalone
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    sys.exit(main(parsed_args)) 