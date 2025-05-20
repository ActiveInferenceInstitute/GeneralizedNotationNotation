#!/usr/bin/env python3
"""
GNN Processing Pipeline

This script orchestrates the full processing pipeline for GNN files and related artifacts.
It dynamically discovers and runs numbered scripts in the src/ directory, where each script
corresponds to a top-level folder and a specific processing stage.

Pipeline Steps (Dynamically Discovered and Ordered):
- 1_gnn.py (Corresponds to gnn/ folder)
- 2_setup.py (Corresponds to setup/ folder)
- 3_tests.py (Corresponds to tests/ folder)
- 4_gnn_type_checker.py (Corresponds to gnn_type_checker/ folder, uses gnn_type_checker.py)
- 5_export.py (Corresponds to export/ folder)
- 6_visualization.py (Corresponds to visualization/ folder, uses visualize_gnn.py)
- 7_mcp.py (Corresponds to mcp/ folder)
- 8_ontology.py (Corresponds to ontology/ folder)
- 9_render.py (Corresponds to render/ folder)
- 10_execute.py (Corresponds to execute/ folder)
- 11_llm.py (Corresponds to llm/ folder)
- 12_site.py (Corresponds to site/ folder, generates HTML summary site)
- 12_discopy.py (Corresponds to discopy/ folder, generates DisCoPy diagrams from GNN)


Usage:
    python main.py [options]
    
Options:
    --target-dir DIR        Target directory for GNN files (default: src/gnn/examples)
    --output-dir DIR        Directory to save outputs (default: ../output)
    --recursive / --no-recursive    Recursively process directories (default: --recursive)
    --skip-steps LIST       Comma-separated list of steps to skip (e.g., "1_gnn,7_mcp" or "1,7")
    --only-steps LIST       Comma-separated list of steps to run (e.g., "4_gnn_type_checker,6_visualization")
    --verbose               Enable verbose output
    --strict                Enable strict type checking mode (for 4_gnn_type_checker)
    --estimate-resources / --no-estimate-resources 
                            Estimate computational resources (for 4_gnn_type_checker) (default: --estimate-resources)
    --ontology-terms-file   Path to the ontology terms file (default: src/ontology/act_inf_ontology_terms.json)
    --llm-tasks LIST        Comma-separated list of LLM tasks to run for 11_llm.py 
                            (e.g., "summarize,explain_structure")
    --llm-timeout           Timeout in seconds for the LLM processing step (11_llm.py)
    --pipeline-summary-file FILE
                            Path to save the final pipeline summary report (default: output/pipeline_execution_summary.json)
    --site-html-filename NAME
                            Filename for the generated HTML summary site (for 12_site.py, saved in output-dir, default: gnn_pipeline_summary_site.html)
    --discopy-gnn-input-dir DIR
                            Directory containing GNN files for DisCoPy processing (for 12_discopy.py, default: src/gnn/examples or --target-dir if specified)

"""

import os
import sys
import subprocess
import argparse
import glob
from pathlib import Path
import logging
import traceback
import re
import datetime # For pipeline summary timestamp
import json # For pipeline summary structured data

# --- Logger Setup ---
logger = logging.getLogger("GNN_Pipeline")
# --- End Logger Setup ---

def parse_arguments():
    project_root = Path(__file__).resolve().parent.parent
    
    default_output_dir = project_root / "output"
    default_target_dir = project_root / "src" / "gnn" / "examples"
    default_ontology_terms_file = project_root / "src" / "ontology" / "act_inf_ontology_terms.json"
    default_pipeline_summary_file = default_output_dir / "pipeline_execution_summary.json"
    default_discopy_gnn_input_dir = default_target_dir # Default discopy input to general target dir

    parser = argparse.ArgumentParser(
        description="GNN Processing Pipeline: Orchestrates GNN file processing, analysis, and export.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=default_target_dir,
        help=(f"Target directory for GNN source files (e.g., .md GNN specifications).\\n"
              f"Default: {default_target_dir.relative_to(project_root) if default_target_dir.is_relative_to(project_root) else default_target_dir}")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=(f"Main directory to save all pipeline outputs.\\n"
              f"Default: {default_output_dir.relative_to(project_root) if default_output_dir.is_relative_to(project_root) else default_output_dir}")
    )
    parser.add_argument(
        "--recursive", default=True, action=argparse.BooleanOptionalAction,
        help="Recursively process GNN files in the target directory. Enabled by default. Use --no-recursive to disable."
    )
    parser.add_argument(
        "--skip-steps", default="",
        help='Comma-separated list of step numbers or names to skip (e.g., "1,7_mcp" or "1_gnn,7").'
    )
    parser.add_argument(
        "--only-steps", default="",
        help='Comma-separated list of step numbers or names to run exclusively (e.g., "4,6_visualization").'
    )
    parser.add_argument(
        "--verbose", 
        default=True, 
        action=argparse.BooleanOptionalAction, 
        help="Enable verbose output for the pipeline and its steps. On by default (use --no-verbose to disable)."
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Enable strict type checking mode (passed to 4_gnn_type_checker.py)."
    )
    parser.add_argument(
        "--estimate-resources", default=True, action=argparse.BooleanOptionalAction,
        help="Estimate computational resources (passed to 4_gnn_type_checker.py). Enabled by default."
    )
    parser.add_argument(
        "--ontology-terms-file",
        type=Path,
        default=default_ontology_terms_file,
        help=(f"Path to a JSON file defining valid ontological terms (for 8_ontology.py).\\n"
              f"Default: {default_ontology_terms_file.relative_to(project_root) if default_ontology_terms_file.is_relative_to(project_root) else default_ontology_terms_file}")
    )
    parser.add_argument(
        "--llm-tasks", default="all", type=str,
        help='Comma-separated list of LLM tasks for 11_llm.py (e.g., "overview,purpose,ontology"). Default: "all".'
    )
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=360, # Default to 6 minutes for the entire LLM script
        help="Timeout in seconds for the LLM processing step (11_llm.py). Default: 360"
    )
    parser.add_argument(
        "--pipeline-summary-file",
        type=Path,
        default=default_pipeline_summary_file,
        help=(f"Path to save the final pipeline summary report (JSON format).\n"
              f"Default: {default_pipeline_summary_file.relative_to(project_root) if default_pipeline_summary_file.is_relative_to(project_root) else default_pipeline_summary_file}")
    )
    parser.add_argument(
        "--site-html-filename",
        type=str,
        default="gnn_pipeline_summary_site.html",
        help=(f"Filename for the generated HTML summary site (for 12_site.py). It will be saved in the main output directory.\n"
              f"Default: gnn_pipeline_summary_site.html")
    )
    parser.add_argument(
        "--discopy-gnn-input-dir",
        type=Path,
        default=None, # Will be set to default_target_dir or args.target_dir later
        help=(f"Directory containing GNN files for DisCoPy processing (12_discopy.py). \n"
              f"If not set, uses the main --target-dir. Default if --target-dir is also default: {default_discopy_gnn_input_dir.relative_to(project_root) if default_discopy_gnn_input_dir.is_relative_to(project_root) else default_discopy_gnn_input_dir}")
    )
    return parser.parse_args()

def get_pipeline_scripts(current_dir: Path) -> list[str]:
    potential_scripts_pattern = current_dir / "*_*.py"
    logger.debug(f"ℹ️ Discovering potential pipeline scripts using pattern: {potential_scripts_pattern}")
    all_candidate_files = glob.glob(str(potential_scripts_pattern))
    
    pipeline_scripts_info: list[dict[str, int | str | Path]] = [] # Explicitly type hint
    script_name_regex = re.compile(r"^(\d+)_.*\.py$")

    for script_path_str in all_candidate_files:
        script_basename = os.path.basename(script_path_str)
        match = script_name_regex.match(script_basename)
        if match:
            script_num = int(match.group(1))
            pipeline_scripts_info.append({'num': script_num, 'basename': script_basename, 'path': Path(script_path_str)})
            if logger.isEnabledFor(logging.DEBUG):
                 logger.debug(f"ℹ️ Matched script for pipeline: {script_basename} (Number: {script_num})")
    
    pipeline_scripts_info.sort(key=lambda x: (x['num'], x['basename']))
    sorted_script_basenames = [info['basename'] for info in pipeline_scripts_info]

    if logger.isEnabledFor(logging.DEBUG): 
        logger.debug(f"ℹ️ Found and sorted script basenames: {sorted_script_basenames}")
    return sorted_script_basenames

def get_venv_python(script_dir: Path) -> tuple[Path | None, Path | None]:
    venv_path = script_dir / ".venv"
    venv_python_path = None
    site_packages_path = None

    if venv_path.is_dir():
        potential_python_executables = [
            venv_path / "bin" / "python",
            venv_path / "bin" / "python3",
            venv_path / "Scripts" / "python.exe", # Windows
        ]
        for py_exec in potential_python_executables:
            if py_exec.exists() and py_exec.is_file():
                venv_python_path = py_exec
                logger.debug(f"🐍 Found virtual environment Python: {venv_python_path}")
                break
        
        lib_path = venv_path / "lib"
        if lib_path.is_dir():
            for python_version_dir in lib_path.iterdir():
                if python_version_dir.is_dir() and python_version_dir.name.startswith("python"):
                    current_site_packages = python_version_dir / "site-packages"
                    if current_site_packages.is_dir():
                        site_packages_path = current_site_packages
                        logger.debug(f"Found site-packages at: {site_packages_path}")
                        break
    
    if not venv_python_path:
        logger.warning("⚠️ Virtual environment Python not found. Using system Python. This may lead to issues if dependencies are not globally available.")
        venv_python_path = Path(sys.executable) # Fallback to current interpreter
    
    return venv_python_path, site_packages_path

def run_pipeline(args: argparse.Namespace):
    logger.info("""

 ██████╗ ███╗   ██╗███╗   ██╗  
██╔════╝ ████╗  ██║████╗  ██║  
██║  ███╗██╔██╗ ██║██╔██╗ ██║
██║   ██║██║╚██╗██║██║╚██╗██║    
╚██████╔╝██║ ╚████║██║ ╚████║    
                                                                                 
    Generalized Notation Notation Processing Pipeline 
    
    Version 0.1.1 ~ Improvements by AI Assistant
            
    Initializing GNN Processing Pipeline...
    """)

    current_script_dir = Path(__file__).resolve().parent
    venv_python_path, _venv_site_packages_path_for_subproc = get_venv_python(current_script_dir)

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Ensured output directory exists: {args.output_dir}")
    except OSError as e:
        logger.error(f"❌ Failed to create output directory {args.output_dir}: {e}")
        return 1

    all_scripts = get_pipeline_scripts(current_script_dir)
    if not all_scripts:
        logger.error("❌ No pipeline scripts found in src/. Cannot proceed.")
        return 1
    
    logger.info("🚀 Starting GNN Processing Pipeline...")
    if args.verbose:
        logger.debug(f"ℹ️ All discovered script modules (basenames): {all_scripts}")

    skip_steps_input = {s.strip() for s in args.skip_steps.split(",") if s.strip()}
    only_steps_input = {s.strip() for s in args.only_steps.split(",") if s.strip()}
    
    pipeline_run_data = {
        "start_time": datetime.datetime.now().isoformat(),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "steps": []
    }
    
    overall_status = "SUCCESS" # Will change to FAILED or SUCCESS_WITH_WARNINGS

    for i, script_file_basename in enumerate(all_scripts):
        script_name_no_ext = Path(script_file_basename).stem
        step_num_str = script_name_no_ext.split("_")[0]
        
        step_log_data = {
            "step_number": i + 1,
            "script_name": script_name_no_ext,
            "status": "SKIPPED",
            "start_time": None,
            "end_time": None,
            "duration_seconds": None,
            "details": "",
            "stdout": "",
            "stderr": ""
        }
        
        step_header = f"Step {i+1}/{len(all_scripts)}: {script_name_no_ext}"
        is_critical_step = (script_name_no_ext == "2_setup")

        should_skip = False
        if only_steps_input:
            if not (script_name_no_ext in only_steps_input or step_num_str in only_steps_input):
                should_skip = True
                step_log_data["details"] = "Skipped due to --only-steps filter."
        elif skip_steps_input:
            if (script_name_no_ext in skip_steps_input or step_num_str in skip_steps_input):
                should_skip = True
                step_log_data["details"] = "Skipped due to --skip-steps filter."
        
        if should_skip:
            logger.info(f"⏭️ {step_header} - SKIPPED ({step_log_data['details']})")
            pipeline_run_data["steps"].append(step_log_data)
            continue
        
        logger.info("") # Add spacing before step start
        logger.info(f"⚙️ {step_header} (from {script_file_basename}) - STARTING")
        step_log_data["start_time"] = datetime.datetime.now().isoformat()
        
        script_full_path = current_script_dir / script_file_basename
        cmd_list = [str(venv_python_path), str(script_full_path)]

        # Common arguments
        cmd_list.extend(["--output-dir", str(args.output_dir)])
        if args.verbose: 
            cmd_list.append("--verbose")
        else:
            cmd_list.append("--no-verbose")

        # Script-specific arguments
        if script_name_no_ext == "1_gnn":
            cmd_list.extend(["--target-dir", str(args.target_dir)])
            if args.recursive: cmd_list.append("--recursive")
        elif script_name_no_ext == "2_setup":
            cmd_list.extend(["--target-dir", str(args.target_dir)])
        elif script_name_no_ext == "3_tests":
            pass # Uses venv_python_path internally for pytest
        elif script_name_no_ext == "4_gnn_type_checker":
            cmd_list.extend(["--target-dir", str(args.target_dir)])
            if args.recursive:
                cmd_list.append("--recursive")
            else:
                cmd_list.append("--no-recursive")
            if args.strict: cmd_list.append("--strict")
            cmd_list.append("--estimate-resources" if args.estimate_resources else "--no-estimate-resources")
        elif script_name_no_ext == "5_export":
            cmd_list.extend(["--target-dir", str(args.target_dir)])
            if args.recursive: cmd_list.append("--recursive")
        elif script_name_no_ext == "6_visualization":
            cmd_list.extend(["--target-dir", str(args.target_dir)])
            if args.recursive: cmd_list.append("--recursive")
        elif script_name_no_ext == "7_mcp":
            pass # Common args are sufficient
        elif script_name_no_ext == "8_ontology":
            cmd_list.extend(["--target-dir", str(args.target_dir)])
            if args.recursive: cmd_list.append("--recursive") # Script needs to support this
            cmd_list.extend(["--ontology-terms-file", str(args.ontology_terms_file)])
        elif script_name_no_ext == "9_render":
            if args.recursive: cmd_list.append("--recursive") # Searches in output_dir/gnn_exports
        elif script_name_no_ext == "10_execute":
            if args.recursive: cmd_list.append("--recursive") # Searches in output_dir/gnn_rendered_simulators
        elif script_name_no_ext == "11_llm":
            cmd_list.extend(["--target-dir", str(args.target_dir)])
            if args.recursive: cmd_list.append("--recursive")
            if args.llm_tasks:
                tasks = [task.strip() for task in args.llm_tasks.split(',') if task.strip()]
                if tasks:
                    cmd_list.append("--llm-tasks")
                    cmd_list.extend(tasks)
        elif script_name_no_ext == "12_site":
            # 12_site.py uses --output-dir (where it reads from) 
            # and --site-html-file (the name of the file it creates within output-dir)
            cmd_list.extend(["--site-html-file", str(args.site_html_filename)])
        elif script_name_no_ext == "12_discopy":
            # Determine the input directory for 12_discopy.py
            # Priority: --discopy-gnn-input-dir > --target-dir > default (src/gnn/examples)
            discopy_input_dir_to_use = args.target_dir # Default to the general target_dir
            if args.discopy_gnn_input_dir is not None:
                discopy_input_dir_to_use = args.discopy_gnn_input_dir
            
            cmd_list.extend(["--gnn-input-dir", str(discopy_input_dir_to_use)])
            # --output-dir is already added as a common argument
            if args.recursive: # Pass recursive if set for the main pipeline
                cmd_list.append("--recursive")
            # Verbosity is also handled by common args

        step_process_env = os.environ.copy()
        if _venv_site_packages_path_for_subproc:
            python_path_parts = {str(_venv_site_packages_path_for_subproc)}
            if "PYTHONPATH" in step_process_env:
                python_path_parts.update(step_process_env["PYTHONPATH"].split(os.pathsep))
            step_process_env["PYTHONPATH"] = os.pathsep.join(python_path_parts)
        
        logger.debug(f"  Running command: {' '.join(cmd_list)}")
        if args.verbose and "PYTHONPATH" in step_process_env:
            logger.debug(f"    with PYTHONPATH: {step_process_env['PYTHONPATH']}")

        try:
            return_code = -1 # Default return code

            if args.verbose:
                logger.debug(f"  Streaming output for command: {' '.join(cmd_list)}")
                # Use Popen to stream output for verbose mode
                process = subprocess.Popen(
                    cmd_list,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=step_process_env,
                    cwd=current_script_dir,
                    bufsize=1,  # Line buffered
                    universal_newlines=True # Ensure text mode for iter
                )
                
                stdout_lines = []
                stderr_lines = []

                # Log stdout
                if process.stdout:
                    for line in iter(process.stdout.readline, ''):
                        stripped_line = line.strip()
                        if stripped_line: # Avoid logging empty lines from child process
                            # Log stdout from subprocess at DEBUG level when main.py is verbose
                            logger.debug(f"    [{script_name_no_ext}-STDOUT] {stripped_line}")
                        stdout_lines.append(line)
                    process.stdout.close()
                
                # Log stderr
                if process.stderr:
                    for line in iter(process.stderr.readline, ''):
                        stripped_line = line.strip()
                        if stripped_line: # Avoid logging empty lines
                            # Log all stderr from subprocess with a clear prefix and as a warning/error
                            logger.warning(f"    [{script_name_no_ext}-STDERR] {stripped_line}")
                        stderr_lines.append(line)
                    process.stderr.close()
                    
                # Determine timeout for wait() based on the script
                current_step_timeout = None
                if script_name_no_ext == "11_llm":
                    current_step_timeout = args.llm_timeout
                    logger.debug(f"  Applying LLM step-specific timeout of {current_step_timeout}s for wait().")

                return_code = process.wait(timeout=current_step_timeout)
                step_log_data["stdout"] = "".join(stdout_lines)
                step_log_data["stderr"] = "".join(stderr_lines)
            else:
                # Original behavior for non-verbose mode
                # Determine timeout for run() based on the script
                current_step_timeout_run = None
                if script_name_no_ext == "11_llm":
                    current_step_timeout_run = args.llm_timeout
                    logger.debug(f"  Applying LLM step-specific timeout of {current_step_timeout_run}s for run().")

                step_process_result = subprocess.run(
                    cmd_list, 
                    capture_output=True, 
                    text=True, 
                    check=False, 
                    env=step_process_env, 
                    cwd=current_script_dir,
                    timeout=current_step_timeout_run
                )
                step_log_data["stdout"] = step_process_result.stdout
                step_log_data["stderr"] = step_process_result.stderr
                return_code = step_process_result.returncode

            if return_code == 0:
                step_log_data["status"] = "SUCCESS"
                logger.info(f"✅ {step_header} - COMPLETED successfully.")
                logger.info("") # Add spacing after step completion
                # For non-verbose successful runs, if there's any output, log it at DEBUG level.
                # For verbose runs, it's already streamed.
                if not args.verbose and step_log_data["stdout"] and step_log_data["stdout"].strip():
                    logger.debug(f"   Output from {script_name_no_ext}:\n{step_log_data['stdout'].strip()}")
                if not args.verbose and step_log_data["stderr"] and step_log_data["stderr"].strip(): # Also log stderr if any for non-verbose success
                    # If main is not verbose, but a script wrote to stderr and succeeded,
                    # it's good to see this at least at DEBUG level in main log.
                    logger.debug(f"   Stderr from {script_name_no_ext} (non-verbose main, step success):\n{step_log_data['stderr'].strip()}")

            elif return_code == 2: # Special code for success with warnings
                step_log_data["status"] = "SUCCESS_WITH_WARNINGS"
                logger.warning(f"⚠️ {step_header} - COMPLETED with warnings (Code 2). Check script output.")
                logger.info("") # Add spacing after step completion
                if overall_status == "SUCCESS": overall_status = "SUCCESS_WITH_WARNINGS"
                # Log stdout/stderr for warnings as well (if not already streamed in verbose)
                if not args.verbose:
                    if step_log_data["stdout"] and step_log_data["stdout"].strip(): 
                        logger.warning(f"   Stdout for warnings from {script_name_no_ext}:\n{step_log_data['stdout'].strip()}")
                    if step_log_data["stderr"] and step_log_data["stderr"].strip(): 
                        logger.warning(f"   Stderr for warnings from {script_name_no_ext}:\n{step_log_data['stderr'].strip()}")
            else:
                step_log_data["status"] = "FAILED"
                step_log_data["details"] = f"Exited with code {return_code}."
                logger.error(f"❌ {step_header} - FAILED (Code: {return_code}).")
                logger.info("") # Add spacing after step completion
                overall_status = "FAILED"
                # Log stdout/stderr for failures (if not already streamed in verbose)
                if not args.verbose:
                    if step_log_data["stdout"] and step_log_data["stdout"].strip(): 
                        logger.error(f"   Stdout from {script_name_no_ext}:\n{step_log_data['stdout'].strip()}")
                    if step_log_data["stderr"] and step_log_data["stderr"].strip(): 
                        logger.error(f"   Stderr from {script_name_no_ext}:\n{step_log_data['stderr'].strip()}")
                if is_critical_step:
                    logger.critical(f"🔥 Critical step {script_name_no_ext} failed. Halting pipeline.")
                    step_log_data["details"] += " Critical step failure, pipeline halted."
                    pipeline_run_data["steps"].append(step_log_data)
                    break 
        
        except subprocess.TimeoutExpired:
            step_log_data["status"] = "FAILED_TIMEOUT"
            timeout_duration = current_step_timeout if args.verbose else current_step_timeout_run
            step_log_data["details"] = f"Step timed out after {timeout_duration} seconds."
            logger.error(f"❌ {step_header} - FAILED due to TIMEOUT after {timeout_duration}s.")
            logger.info("") # Add spacing after step completion
            overall_status = "FAILED"
            # Ensure process is killed if it timed out during wait()
            if args.verbose and process:
                try:
                    logger.warning(f"  Attempting to terminate timed-out process for {script_name_no_ext} (PID: {process.pid})")
                    process.kill() # or process.terminate()
                    #oudates to captured stdout/stderr might be lost or partial
                    process.wait() # wait for termination to complete
                    logger.info(f"  Process {script_name_no_ext} terminated.")
                except Exception as e_kill:
                    logger.error(f"  Error trying to terminate process {script_name_no_ext}: {e_kill}")
            
            # For non-verbose, subprocess.run() handles termination on timeout.
            # Capture any output that might have occurred before timeout
            if args.verbose and process.stdout:
                 step_log_data["stdout"] = "".join(stdout_lines) + "\n[TIMEOUT OCCURRED - STDOUT MAY BE INCOMPLETE]"
            if args.verbose and process.stderr:
                step_log_data["stderr"] = "".join(stderr_lines) + "\n[TIMEOUT OCCURRED - STDERR MAY BE INCOMPLETE]"
            # For non-verbose, this is already handled by subprocess.run returning the captured output up to timeout.
            
            if is_critical_step:
                logger.critical(f"🔥 Critical step {script_name_no_ext} timed out. Halting pipeline.")
                step_log_data["details"] += " Critical step timeout, pipeline halted."
                pipeline_run_data["steps"].append(step_log_data)
                break

        except Exception as e:
            step_log_data["status"] = "ERROR_UNHANDLED_EXCEPTION"
            step_log_data["details"] = f"Unhandled exception: {str(e)}"
            logger.error(f"❌ Unhandled exception in {step_header}: {e}")
            logger.info("") # Add spacing after step completion
            logger.debug(traceback.format_exc())
            overall_status = "FAILED"
            if is_critical_step:
                logger.critical(f"🔥 Critical step {script_name_no_ext} failed due to unhandled exception. Halting pipeline.")
                step_log_data["details"] += " Critical step failure, pipeline halted."
                pipeline_run_data["steps"].append(step_log_data)
                break
        
        finally:
            step_log_data["end_time"] = datetime.datetime.now().isoformat()
            if step_log_data["start_time"]:
                duration = datetime.datetime.fromisoformat(step_log_data["end_time"]) - datetime.datetime.fromisoformat(step_log_data["start_time"])
                step_log_data["duration_seconds"] = duration.total_seconds()
            pipeline_run_data["steps"].append(step_log_data)

    pipeline_run_data["end_time"] = datetime.datetime.now().isoformat()
    pipeline_run_data["overall_status"] = overall_status
    # Log a brief summary before returning from run_pipeline
    logger.info(f"🏁 Pipeline processing completed. Overall Status: {overall_status}")

    return 0 if overall_status in ["SUCCESS", "SUCCESS_WITH_WARNINGS"] else 1, pipeline_run_data, all_scripts, overall_status

def main():
    # Configure logging early. If GNN_Pipeline or root logger has no handlers,
    # set up a basic one. The user's traceback suggests handlers are present,
    # so this configuration is for robustness, especially if run in different contexts.
    pipeline_logger = logging.getLogger("GNN_Pipeline") # Use the specific logger

    if not pipeline_logger.hasHandlers() and not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO, # Default level
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt=None, # Explicitly use default to get milliseconds
            stream=sys.stdout
        )
        pipeline_logger.info("Initialized basic logging config as no handlers were found for GNN_Pipeline or root.")

    args = parse_arguments()

    # Quieten noisy dependency loggers (PIL, Matplotlib) unconditionally
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    # Add other noisy libraries here if needed, e.g.:
    # logging.getLogger('some_other_library').setLevel(logging.WARNING)

    logger.info("Starting GNN Processing Pipeline...")

    # --- File Handler Setup (after args are parsed) ---
    # Ensure logs directory exists
    log_dir = args.output_dir / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        # Add a file handler to the GNN_Pipeline logger
        log_file_path = log_dir / "pipeline.log"
        file_handler = logging.FileHandler(log_file_path, mode='w') # 'w' to overwrite each run
        # Use the same format as basicConfig, explicitly ensure milliseconds
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt=None)
        file_handler.setFormatter(file_formatter)
        # The level of the file_handler will be determined by the pipeline_logger's effective level,
        # which is set below based on args.verbose.
        pipeline_logger.addHandler(file_handler)
        # Log this message *after* the handler is added so it goes to the file too.
        pipeline_logger.info(f"File logging configured to: {log_file_path}") 
    except Exception as e:
        # Log this error to console, as file handler might have failed.
        temp_error_logger = logging.getLogger("GNN_Pipeline.FileHandlerSetup")
        # Ensure this specific error message can make it to console/stderr if main logger isn't fully working
        if not temp_error_logger.handlers:
            err_handler = logging.StreamHandler(sys.stderr)
            # Use a more specific format for this temp logger to distinguish its origin
            err_formatter = logging.Formatter('%(asctime)s - %(name)s - [TEMP_SETUP_ERROR] - %(levelname)s - %(message)s')
            err_handler.setFormatter(err_formatter)
            temp_error_logger.addHandler(err_handler)
            temp_error_logger.propagate = False # Don't double log this specific error message
        temp_error_logger.error(f"Failed to configure file logging to {log_dir / 'pipeline.log'}: {e}. Continuing with console logging only.")
    # --- End File Handler Setup ---

    # Configure logger level based on verbose flag AFTER parsing args
    if args.verbose:
        pipeline_logger.setLevel(logging.DEBUG)
        # Propagate level to handlers if they exist and basicConfig wasn't just called
        handlers_to_update = list(pipeline_logger.handlers) + list(logging.getLogger().handlers)
        for handler in handlers_to_update:
            # Only set level if handler's current level is not effectively DEBUG or lower
            if handler.level == 0 or handler.level > logging.DEBUG: # 0 means NOTSET, effectively inherits.
                current_level_name = logging.getLevelName(handler.level)
                logger.debug(f"Updating handler {type(handler).__name__} level from {current_level_name} to DEBUG")
                handler.setLevel(logging.DEBUG)
        # For Popen streaming, we use GNN_Pipeline's INFO and ERROR, so DEBUG on GNN_Pipeline is fine.
    else:
        pipeline_logger.setLevel(logging.INFO)
        # Propagate level to handlers
        handlers_to_update_info = list(pipeline_logger.handlers) + list(logging.getLogger().handlers)
        for handler in handlers_to_update_info:
            if handler.level == 0 or handler.level > logging.INFO:
                current_level_name = logging.getLevelName(handler.level)
                logger.debug(f"Updating handler {type(handler).__name__} level from {current_level_name} to INFO")
                handler.setLevel(logging.INFO)

    pipeline_logger.debug("Logger level configured based on verbosity.")

    # --- Defensive Path Conversion ---
    # Ensure critical path arguments are pathlib.Path objects.
    # argparse with type=Path should handle this, but this adds robustness.
    path_args_to_check = [
        'output_dir', 'target_dir', 'ontology_terms_file', 'pipeline_summary_file'
    ]

    for arg_name in path_args_to_check:
        if not hasattr(args, arg_name):
            pipeline_logger.debug(f"Argument --{arg_name.replace('_', '-')} not present in args namespace.")
            continue

        arg_value = getattr(args, arg_name)
        
        # Only proceed if arg_value is not None. If it's None, it might be an optional Path not provided.
        if arg_value is not None and not isinstance(arg_value, Path):
            pipeline_logger.warning(
                f"Argument --{arg_name.replace('_', '-')} was unexpectedly a {type(arg_value).__name__} "
                f"(value: '{arg_value}') instead of pathlib.Path. Converting explicitly. "
                "This might indicate an issue with argument parsing configuration or an external override."
            )
            try:
                setattr(args, arg_name, Path(arg_value))
            except TypeError as e:
                pipeline_logger.error(
                    f"Failed to convert argument --{arg_name.replace('_', '-')} (value: '{arg_value}') to Path: {e}. "
                    "This could be due to an unsuitable value for a path."
                )
                # If a critical path like output_dir fails conversion, it's a fatal error for the script's purpose.
                if arg_name in ['output_dir', 'target_dir']:
                    pipeline_logger.critical(f"Critical path argument --{arg_name.replace('_', '-')} could not be converted to Path. Exiting.")
                    sys.exit(1)
        elif arg_value is None and arg_name in ['output_dir', 'target_dir']: # These should always have a default Path value.
             pipeline_logger.critical(
                f"Critical path argument --{arg_name.replace('_', '-')} is None after parsing. "
                "This indicates a problem with default value setup in argparse. Exiting."
             )
             sys.exit(1)
    # --- End Defensive Path Conversion ---

    pipeline_logger.info(f"🚀 Initializing GNN Pipeline with Target: '{args.target_dir}', Output: '{args.output_dir}'")
    
    # Log the arguments being used, showing their types after potential conversion
    if pipeline_logger.isEnabledFor(logging.DEBUG): # Check level before formatting potentially many lines
        log_msgs = ["🛠️ Effective Arguments (after potential defensive conversion):"]
        for arg, value in sorted(vars(args).items()):
            log_msgs.append(f"  --{arg.replace('_', '-')}: {value} (Type: {type(value).__name__})")
        pipeline_logger.debug('\n'.join(log_msgs))


    # Call the main pipeline execution function
    exit_code, pipeline_run_data, all_scripts, overall_status = run_pipeline(args)

    # --- Pipeline Summary Report ---
    logger.info("\n--- Pipeline Execution Summary ---")
    num_total = len(pipeline_run_data["steps"])
    num_success = len([s for s in pipeline_run_data["steps"] if s["status"] == "SUCCESS"])
    num_warn = len([s for s in pipeline_run_data["steps"] if s["status"] == "SUCCESS_WITH_WARNINGS"])
    num_failed = len([s for s in pipeline_run_data["steps"] if "FAILED" in s["status"] or "ERROR" in s["status"]])
    num_skipped = len([s for s in pipeline_run_data["steps"] if s["status"] == "SKIPPED"])

    logger.info(f"📊 Total Steps Attempted/Processed: {num_total - num_skipped} / {len(all_scripts)}")
    logger.info(f"  ✅ Successful: {num_success}")
    logger.info(f"  ⚠️ Success with Warnings: {num_warn}")
    logger.info(f"  ❌ Failed/Error: {num_failed}")
    logger.info(f"  ⏭️ Skipped: {num_skipped}")

    if overall_status == "SUCCESS":
        logger.info("🎉 PIPELINE FINISHED SUCCESSFULLY.")
    elif overall_status == "SUCCESS_WITH_WARNINGS":
        logger.warning("🎉 PIPELINE FINISHED, but with warnings from some steps.")
    else: # FAILED
        logger.error("🛑 PIPELINE FINISHED WITH ERRORS.")

    # Save detailed summary report
    try:
        args.pipeline_summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.pipeline_summary_file, 'w') as f_summary:
            json.dump(pipeline_run_data, f_summary, indent=4)
        logger.info(f"💾 Detailed pipeline execution summary (JSON) saved to: {args.pipeline_summary_file}")
    except Exception as e:
        logger.error(f"❌ Error saving pipeline summary report: {e}")

    sys.exit(exit_code)

if __name__ == "__main__":
    main() 