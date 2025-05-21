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
import json # For saving structured LLM outputs

# Attempt to import the new logging utility (utils.logging_utils.py)
# This should be tried before other project imports that might depend on logging being set up by it for standalone.
try:
    from utils.logging_utils import setup_standalone_logging
except ImportError:
    # Fallback for standalone execution or if src is not directly in path
    _current_script_path_for_util = Path(__file__).resolve()
    _project_root_for_util = _current_script_path_for_util.parent.parent
    _paths_to_try_util = [str(_project_root_for_util), str(_project_root_for_util / "src")]
    _original_sys_path_util = list(sys.path)
    for _p_try_util in _paths_to_try_util:
        if _p_try_util not in sys.path:
            sys.path.insert(0, _p_try_util)
    try:
        from utils.logging_utils import setup_standalone_logging
    except ImportError:
        setup_standalone_logging = None
        # Minimal logging for this specific failure, as main logger isn't fully set up yet.
        _temp_logger_name_util = __name__ if __name__ != "__main__" else "src.11_llm_util_import_warning"
        _temp_logger_util = logging.getLogger(_temp_logger_name_util)
        if not _temp_logger_util.hasHandlers() and not logging.getLogger().hasHandlers():
            logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format='%(levelname)s: %(message)s')
        _temp_logger_util.warning(
            "Could not import setup_standalone_logging. Standalone logging will be basic."
        )
    finally:
        sys.path = _original_sys_path_util # Restore original sys.path

# --- Standardized Project Imports ---
# The goal is to make imports robust whether running as a script or part of a larger package.
# Assuming this script (11_llm.py) is in src/ and needs to import from src/llm/ and src/mcp/

# Add project root to sys.path to allow for `from module import ...`
# This is a common pattern for scripts within a package structure.
# Path(__file__).resolve() -> /path/to/GeneralizedNotationNotation/src/11_llm.py
# Path(__file__).resolve().parent -> /path/to/GeneralizedNotationNotation/src
# Path(__file__).resolve().parent.parent -> /path/to/GeneralizedNotationNotation (project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# --- End Standardized Project Imports ---

logger = logging.getLogger(__name__) # GNN_Pipeline.11_llm or __main__

# Now, attempt to import necessary functions from the llm module
try:
    # These imports should now work if src/ (parent of 11_llm.py) is effectively the context
    # and PROJECT_ROOT is in sys.path.
    from src.llm import llm_operations
    from src.llm import mcp as llm_mcp # To potentially call ensure_llm_tools_registered
    from src.mcp.mcp import mcp_instance # Import for ensuring tools are registered
    logger.info("Successfully imported llm_operations, llm_mcp, and mcp_instance.")
except ImportError as e_import:
    logger.critical(f"Failed to import LLM modules or mcp_instance in 11_llm.py: {e_import}", exc_info=True)
    llm_operations = None
    llm_mcp = None
    mcp_instance = None # Ensure it's None if import fails

# --- Global constants ---
LLM_OUTPUT_DIR_NAME = "llm_processing_step"
SUMMARY_FILE_SUFFIX = "_summary.txt"
COMPREHENSIVE_ANALYSIS_FILE_SUFFIX = "_comprehensive_analysis.json" # Changed to JSON for structured data
QA_FILE_SUFFIX = "_qa.json"

# Define task constants
TASK_SUMMARY = "summary"
TASK_ANALYSIS = "analysis"
TASK_QA = "qa"
ALL_TASKS = [TASK_SUMMARY, TASK_ANALYSIS, TASK_QA]


def process_gnn_with_llm(gnn_file_path: Path, output_dir_for_file: Path, tasks_to_run: list[str], verbose: bool = False):
    """
    Processes a single GNN file with specified LLM tasks and saves results.
    Tasks:
    1. General Summary of the GNN file.
    2. Comprehensive Analysis (structured into sections like purpose, components, interactions, ontology mapping).
    3. Generate 3-5 relevant questions about the GNN file.
    4. Answer those generated questions based on the GNN file.
    """
    if not llm_operations or not llm_mcp or not mcp_instance:
        logger.error(f"LLM operations module, LLM MCP, or main MCP instance not available. Skipping {gnn_file_path.name}")
        return

    # Ensure LLM tools are registered (this also handles API key loading)
    try:
        logger.info(f"Ensuring LLM tools are registered for {gnn_file_path.name}...")
        llm_mcp.ensure_llm_tools_registered(mcp_instance) # Pass the actual mcp_instance
        # Check if the API key was loaded successfully.
        # Assuming a function `is_api_key_loaded()` exists or can be added to llm_operations.
        # For now, we'll rely on the logging within ensure_llm_tools_registered/initialize_llm_module.
        # If mcp_instance.sdk_status is available and relevant to LLM, could check that too.
        logger.info(f"LLM tools registration check complete for {gnn_file_path.name}.")
    except Exception as e_reg:
        logger.error(f"Failed to ensure LLM tools were registered for {gnn_file_path.name}: {e_reg}", exc_info=True)
        return # Cannot proceed if tools are not registered

    logger.info(f"Processing GNN file with LLM: {gnn_file_path.name}")

    try:
        with open(gnn_file_path, 'r', encoding='utf-8') as f:
            gnn_content = f.read()
        if not gnn_content.strip():
            logger.warning(f"GNN file {gnn_file_path.name} is empty. Skipping LLM processing.")
            return
    except Exception as e:
        logger.error(f"Error reading GNN file {gnn_file_path}: {e}", exc_info=True)
        return

    output_dir_for_file.mkdir(parents=True, exist_ok=True)

    # --- Task 1: General Summary ---
    summary_output_path = output_dir_for_file / (gnn_file_path.stem + SUMMARY_FILE_SUFFIX)
    if TASK_SUMMARY in tasks_to_run:
        logger.info(f"Task 1: Generating summary for {gnn_file_path.name}...")
        try:
            # Using the MCP tool for summarization as an example, though direct llm_operations can also be used.
            # summarize_gnn_file_content is defined in src/llm/mcp.py
            summary_prompt_suffix = "Focus on the model's purpose, main components, and interactions."
            # summary_response = llm_mcp.summarize_gnn_file_content(str(gnn_file_path), user_prompt_suffix=summary_prompt_suffix)
            
            # Direct call for more control and specific prompt construction
            summary_task_desc = f"Provide a concise summary of the following GNN model, highlighting its key components (ModelName, primary states/observations, and main connections). {summary_prompt_suffix}"
            summary_prompt = llm_operations.construct_prompt([f"GNN File Content ({gnn_file_path.name}):\n{gnn_content}"], summary_task_desc)
            
            logger.debug(f"Summary prompt for {gnn_file_path.name} (first 500 chars): {summary_prompt[:500]}...")
            summary_response = llm_operations.get_llm_response(summary_prompt, model="gpt-4o-mini", max_tokens=1000) # Increased max_tokens for summary

            if summary_response.startswith("Error:"):
                logger.error(f"LLM summary generation failed for {gnn_file_path.name}: {summary_response}")
            else:
                with open(summary_output_path, 'w', encoding='utf-8') as f:
                    f.write(summary_response)
                file_size = summary_output_path.stat().st_size
                logger.info(f"Successfully generated summary for {gnn_file_path.name}. Saved to: {summary_output_path} (Size: {file_size} bytes)")
        except Exception as e:
            logger.error(f"Error during summary generation for {gnn_file_path.name}: {e}", exc_info=True)
    else:
        logger.info(f"Skipping Task 1 (Summary) for {gnn_file_path.name} as it's not in the requested tasks.")

    # --- Task 2: Comprehensive Analysis (Structured JSON Output) ---
    analysis_output_path = output_dir_for_file / (gnn_file_path.stem + COMPREHENSIVE_ANALYSIS_FILE_SUFFIX)
    if TASK_ANALYSIS in tasks_to_run:
        logger.info(f"Task 2: Generating comprehensive analysis for {gnn_file_path.name}...")
        try:
            analysis_task_desc = (
                "Analyze the provided GNN file in detail. Structure your response as a JSON object with the following keys: "
                "\'model_purpose\', \'key_components\', \'component_interactions\', \'data_types_and_dimensions\', "
                "\'potential_applications\', \'limitations_or_ambiguities\', and \'ontology_mapping_assessment\'. "
                "For \'key_components\', describe states, observations, actions, etc. "
                "For \'ontology_mapping_assessment\', briefly assess if ActInfOntology terms are present and relevant if any."
            )
            analysis_prompt = llm_operations.construct_prompt([f"GNN File Content ({gnn_file_path.name}):\n{gnn_content}"], analysis_task_desc)
            
            logger.debug(f"Comprehensive analysis prompt for {gnn_file_path.name} (first 500 chars): {analysis_prompt[:500]}...")
            analysis_response_str = llm_operations.get_llm_response(analysis_prompt, model="gpt-4o-mini", max_tokens=3000) # Increased tokens

            if analysis_response_str.startswith("Error:"):
                logger.error(f"LLM comprehensive analysis generation failed for {gnn_file_path.name}: {analysis_response_str}")
            else:
                # Attempt to parse as JSON, with fallback for non-JSON response
                try:
                    # The LLM might wrap the JSON in backticks or "json" language specifier
                    cleaned_response_str = analysis_response_str.strip()
                    if cleaned_response_str.startswith("```json"):
                        cleaned_response_str = cleaned_response_str[len("```json"):]
                    if cleaned_response_str.startswith("```"): # Generic backticks
                        cleaned_response_str = cleaned_response_str[len("```"):]
                    if cleaned_response_str.endswith("```"):
                        cleaned_response_str = cleaned_response_str[:-len("```")]
                    
                    analysis_data = json.loads(cleaned_response_str.strip())
                    with open(analysis_output_path, 'w', encoding='utf-8') as f:
                        json.dump(analysis_data, f, indent=4)
                    file_size = analysis_output_path.stat().st_size
                    logger.info(f"Successfully generated comprehensive analysis for {gnn_file_path.name}. Saved to: {analysis_output_path} (Size: {file_size} bytes)")
                except json.JSONDecodeError:
                    logger.warning(f"LLM comprehensive analysis for {gnn_file_path.name} was not valid JSON. Saving as raw text.")
                    # Fallback: save as text if not valid JSON
                    analysis_output_path_txt = analysis_output_path.with_suffix(".txt")
                    with open(analysis_output_path_txt, 'w', encoding='utf-8') as f:
                        f.write(analysis_response_str)
                    file_size = analysis_output_path_txt.stat().st_size
                    logger.info(f"Saved raw analysis output to: {analysis_output_path_txt} (Size: {file_size} bytes)")

        except Exception as e:
            logger.error(f"Error during comprehensive analysis for {gnn_file_path.name}: {e}", exc_info=True)
    else:
        logger.info(f"Skipping Task 2 (Comprehensive Analysis) for {gnn_file_path.name} as it's not in the requested tasks.")

    # --- Tasks 3 & 4: Generate Questions and Answers (Structured JSON Output) ---
    qa_output_path = output_dir_for_file / (gnn_file_path.stem + QA_FILE_SUFFIX)
    if TASK_QA in tasks_to_run:
        logger.info(f"Tasks 3 & 4: Generating Q&A for {gnn_file_path.name}...")
        try:
            # Task 3: Generate 3-5 relevant questions
            question_gen_task_desc = (
                "Based on the provided GNN file, generate 3 to 5 insightful questions that would help understand the model's nuances, assumptions, or implications. "
                "Return these questions as a JSON list of strings."
            )
            question_gen_prompt = llm_operations.construct_prompt([f"GNN File Content ({gnn_file_path.name}):\n{gnn_content}"], question_gen_task_desc)
            
            logger.debug(f"Question generation prompt for {gnn_file_path.name} (first 500 chars): {question_gen_prompt[:500]}...")
            questions_response_str = llm_operations.get_llm_response(question_gen_prompt, model="gpt-4o-mini", max_tokens=1000)

            generated_questions = []
            if questions_response_str.startswith("Error:"):
                logger.error(f"LLM question generation failed for {gnn_file_path.name}: {questions_response_str}")
            else:
                try:
                    # Clean potential markdown/code block markers
                    cleaned_q_response = questions_response_str.strip()
                    if cleaned_q_response.startswith("```json"):
                        cleaned_q_response = cleaned_q_response[len("```json"):]
                    if cleaned_q_response.startswith("```"):
                         cleaned_q_response = cleaned_q_response[len("```"):]
                    if cleaned_q_response.endswith("```"):
                        cleaned_q_response = cleaned_q_response[:-len("```")]
                    
                    generated_questions = json.loads(cleaned_q_response.strip())
                    if not isinstance(generated_questions, list):
                        logger.warning(f"LLM generated questions for {gnn_file_path.name} was not a list. Found: {type(generated_questions)}")
                        generated_questions = [] # Reset if not a list
                    else:
                        logger.info(f"Successfully generated {len(generated_questions)} questions for {gnn_file_path.name}.")
                except json.JSONDecodeError:
                    logger.warning(f"LLM generated questions for {gnn_file_path.name} was not valid JSON: {questions_response_str[:200]}...")
                    # Fallback: try to parse as a list of strings if it's just text lines
                    if isinstance(questions_response_str, str) and "\n" in questions_response_str: # Check if it's a multi-line string
                        generated_questions = [q.strip() for q in questions_response_str.split("\n") if q.strip()]
                        if generated_questions:
                             logger.info(f"Fallback: Parsed {len(generated_questions)} questions from raw text for {gnn_file_path.name}.")


            # Task 4: Answer the generated questions
            qa_pairs = []
            if generated_questions:
                logger.info(f"Answering {len(generated_questions)} generated questions for {gnn_file_path.name}...")
                for i, question in enumerate(generated_questions):
                    if not isinstance(question, str) or not question.strip(): # Skip empty or non-string questions
                        logger.warning(f"Skipping invalid question at index {i} for {gnn_file_path.name}: {question}")
                        continue

                    logger.debug(f"Answering question {i+1}/{len(generated_questions)}: '{question}'")
                    answer_task_desc = (
                        f"Based *solely* on the provided GNN file content, answer the following question: '{question}'. "
                        "If the GNN file does not contain enough information to answer, state that explicitly. "
                        "Be concise and directly address the question."
                    )
                    answer_prompt = llm_operations.construct_prompt([f"GNN File Content ({gnn_file_path.name}):\n{gnn_content}"], answer_task_desc)
                    
                    answer_response = llm_operations.get_llm_response(answer_prompt, model="gpt-4o-mini", max_tokens=1000)

                    if answer_response.startswith("Error:"):
                        logger.error(f"LLM answer generation failed for question '{question}' on {gnn_file_path.name}: {answer_response}")
                        qa_pairs.append({"question": question, "answer": "Error during answer generation."})
                    else:
                        qa_pairs.append({"question": question, "answer": answer_response})
                        logger.debug(f"Answer for '{question}': {answer_response[:100]}...")
                
                if qa_pairs:
                    with open(qa_output_path, 'w', encoding='utf-8') as f:
                        json.dump(qa_pairs, f, indent=4)
                    file_size = qa_output_path.stat().st_size
                    logger.info(f"Successfully generated and answered questions for {gnn_file_path.name}. Saved to: {qa_output_path} (Size: {file_size} bytes)")
                else:
                    logger.warning(f"No valid Q&A pairs were generated for {gnn_file_path.name}.")
            else:
                logger.warning(f"No questions were generated for {gnn_file_path.name}, so no Q&A will be saved.")

        except Exception as e:
            logger.error(f"Error during Q&A generation for {gnn_file_path.name}: {e}", exc_info=True)
    else:
        logger.info(f"Skipping Tasks 3 & 4 (Q&A) for {gnn_file_path.name} as it's not in the requested tasks.")

    logger.info(f"Finished LLM processing for GNN file: {gnn_file_path.name}")


def discover_gnn_files(target_dir: Path, recursive: bool = False) -> list[Path]:
    """Discovers GNN files (.gnn.md, .md, .json) in the target directory."""
    patterns = ["*.gnn.md", "*.md", "*.json"] # Added .json as per user instruction context. Consider if .json is always a GNN file.
    gnn_files = []
    for pattern in patterns:
        if recursive:
            gnn_files.extend(list(target_dir.rglob(pattern)))
        else:
            gnn_files.extend(list(target_dir.glob(pattern)))
    
    # Filter out files from .venv or other common non-gnn directories if necessary
    # For now, assume all found files are candidates
    unique_files = sorted(list(set(gnn_files))) # Deduplicate and sort
    logger.info(f"Discovered {len(unique_files)} potential GNN input files.")
    if logger.isEnabledFor(logging.DEBUG): # Check logger level for verbose discovery log
        for f_path in unique_files:
            logger.debug(f"  - Found: {f_path}")
    return unique_files

def main(args: argparse.Namespace) -> int:
    """
    Main function to orchestrate LLM processing of GNN files.
    """
    # Set this script's logger level based on args.verbose.
    # If run standalone, setup_standalone_logging in __main__ will have already configured handlers
    # and potentially set this logger's level. This call ensures it respects args.verbose.
    # If run by main.py, main.py's logging config applies, and this sets this script's specific level.
    log_level_for_this_script = logging.DEBUG if args.verbose else logging.INFO
    logger.setLevel(log_level_for_this_script)
    if logger.isEnabledFor(logging.DEBUG): # Log this only if we are actually in debug mode now
        logger.debug(f"Script logger '{logger.name}' level set to {logging.getLevelName(log_level_for_this_script)}.")
    
    # If verbose, also set level for llm_operations module if it exists.
    if args.verbose and llm_operations: # Removed hasattr check, direct string usage
        logging.getLogger("src.llm.llm_operations").setLevel(logging.DEBUG)
        logger.debug("Verbose mode: Set logger for 'src.llm.llm_operations' to DEBUG.")

    logger.info(f"Starting LLM processing step. Target directory: {args.target_dir}, Output directory: {args.output_dir}")
    logger.debug(f"Full arguments: {args}")

    if not llm_operations or not llm_mcp or not mcp_instance:
        logger.critical("LLM support modules (llm_operations, llm_mcp, or mcp_instance) are not available. Terminating LLM processing.")
        return 1 # Indicate failure

    target_dir = Path(args.target_dir)
    pipeline_output_dir = Path(args.output_dir) # This is the main output dir for the whole pipeline
    
    # Specific output directory for this LLM step
    llm_step_output_dir = pipeline_output_dir / LLM_OUTPUT_DIR_NAME
    llm_step_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"LLM outputs will be saved in: {llm_step_output_dir}")

    selected_tasks = []
    if "all" in args.llm_tasks:
        selected_tasks = ALL_TASKS
    else:
        selected_tasks = [task for task in args.llm_tasks if task in ALL_TASKS]
        if not selected_tasks:
            logger.warning(f"No valid LLM tasks specified or recognized: {args.llm_tasks}. Defaulting to all tasks.")
            selected_tasks = ALL_TASKS # Or could choose to run no tasks / error out
    
    logger.info(f"LLM tasks to run: {', '.join(selected_tasks)}")

    gnn_files_to_process = discover_gnn_files(target_dir, args.recursive)

    if not gnn_files_to_process:
        logger.info("No GNN files found to process.")
        return 0

    for gnn_file_path in gnn_files_to_process:
        # Create a subdirectory for each GNN file's outputs within the llm_step_output_dir
        # e.g., output/llm_processing_step/gnn_example_A/
        file_specific_output_dir = llm_step_output_dir / gnn_file_path.stem
        file_specific_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"--- Processing GNN file: {gnn_file_path.relative_to(target_dir) if gnn_file_path.is_relative_to(target_dir) else gnn_file_path} ---")
        process_gnn_with_llm(gnn_file_path, file_specific_output_dir, selected_tasks, args.verbose)

    logger.info("LLM processing step completed.")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
Process GNN files using LLMs for tasks like summarization, analysis, and Q&A generation.
This script is intended to be called as part of the GNN processing pipeline (via main.py)
but can be run standalone for testing with appropriate arguments.
""", formatter_class=argparse.RawTextHelpFormatter) # Added formatter for better help text

    # Define defaults for standalone execution relative to this script's project root
    script_file_path = Path(__file__).resolve()
    project_root_for_defaults = script_file_path.parent.parent # src/ -> project_root
    default_target_dir_standalone = project_root_for_defaults / "src" / "gnn" / "examples"
    default_output_dir_standalone = project_root_for_defaults / "output"

    parser.add_argument("--target-dir", type=Path, default=default_target_dir_standalone, 
                        help=f"Directory containing GNN files to process. Default: {default_target_dir_standalone.relative_to(project_root_for_defaults) if default_target_dir_standalone.is_relative_to(project_root_for_defaults) else default_target_dir_standalone}")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir_standalone, 
                        help=f"Main output directory for the GNN pipeline. LLM outputs will be in a subfolder. Default: {default_output_dir_standalone.relative_to(project_root_for_defaults) if default_output_dir_standalone.is_relative_to(project_root_for_defaults) else default_output_dir_standalone}")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=False, 
                        help="Recursively search for GNN files in target-dir.")
    parser.add_argument("--verbose", "-v", action=argparse.BooleanOptionalAction, default=False, 
                        help="Enable detailed DEBUG logging.")
    parser.add_argument("--llm-tasks", nargs='+', default=["all"], 
                        choices=ALL_TASKS + ["all"], # Allow individual tasks or "all"
                        help=f"Specify which LLM tasks to run. Choices: {', '.join(ALL_TASKS)}, or 'all'. "
                             f"Default is 'all'. Can provide multiple space-separated tasks.")
    
    parsed_args = parser.parse_args()

    # Setup logging for standalone execution using the utility function
    log_level_to_set = logging.DEBUG if parsed_args.verbose else logging.INFO
    if setup_standalone_logging:
        # Pass __name__ so the utility can also set this script's specific logger level
        # The utility now handles splitting streams, so no need for fallback basicConfig.
        setup_standalone_logging(level=log_level_to_set, logger_name=__name__) 
    else:
        # This else block might ideally not be needed if setup_standalone_logging is always available.
        # However, keeping a minimal error message if the import itself failed.
        # The complex sys.path manipulation at the top aims to prevent this.
        _fallback_logger = logging.getLogger(__name__) # Use __name__ for this fallback logger
        if not _fallback_logger.hasHandlers() and not logging.getLogger().hasHandlers():
            # Minimal config to stderr if utility is completely missing, so errors are seen.
            logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format='%(levelname)s: %(message)s')
        _fallback_logger.error(
            "CRITICAL: setup_standalone_logging utility was not imported. Logging may be incomplete or misdirected. Ensure src/utils is in PYTHONPATH."
        )
        # Ensure this script's logger (which is __name__ here) level is set even in this dire fallback
        # to at least INFO to see basic progress, though it might go to stderr.
        logging.getLogger(__name__).setLevel(logging.INFO) 

    # Quieten noisy libraries if run standalone, after main logging is set up
    # Example: logging.getLogger('some_noisy_dependency').setLevel(logging.WARNING)
    
    # For standalone execution, it's important that mcp_instance is available
    # and that LLM tools can be registered.
    if not mcp_instance:
        logger.error("Main MCP instance is not available. Cannot ensure LLM tools registration for standalone run. Exiting.")
        sys.exit(1)
    
    try:
        logger.info("Standalone run: Ensuring LLM tools are registered...")
        if llm_mcp: # if llm_mcp was successfully imported
            llm_mcp.ensure_llm_tools_registered(mcp_instance)
            logger.info("Standalone run: LLM tools registration check complete.")
        else:
            logger.error("Standalone run: llm_mcp module not available. Cannot register LLM tools.")
            sys.exit(1) # Cannot proceed without llm_mcp
    except Exception as e_reg_standalone:
        logger.error(f"Standalone run: Failed to ensure LLM tools were registered: {e_reg_standalone}", exc_info=True)
        sys.exit(1)


    exit_code = main(parsed_args)
    sys.exit(exit_code) 