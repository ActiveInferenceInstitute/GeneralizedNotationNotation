from pathlib import Path
from typing import Any, Dict, List, Optional
import re
import logging
from utils.path_utils import get_relative_path_if_possible
from pipeline import get_output_dir_for_script
from utils import log_step_start, log_step_success, log_step_warning, log_step_error

def process_gnn_folder(
    target_dir: Path, 
    output_dir: Path, 
    logger: logging.Logger, 
    recursive: bool = False, 
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Process the GNN folder:
    - Discover .md files.
    - Perform basic parsing for key GNN sections.
    - Log findings and simple statistics to a report file.
    
    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Output directory for results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    log_step_start(logger, f"Processing GNN files in directory: '{target_dir}'")
    
    if recursive:
        logger.info("Recursive mode enabled: searching in subdirectories.")
    else:
        logger.info("Recursive mode disabled: searching in top-level directory only.")

    gnn_target_path_abs = target_dir.resolve()

    if not target_dir.is_dir():
        log_step_warning(logger, f"GNN target directory '{gnn_target_path_abs}' not found or not a directory. Skipping GNN processing for this target.")
        return False

    # Use centralized output directory configuration
    step_output_dir = get_output_dir_for_script("1_gnn.py", output_dir)
    
    # Create the step output directory
    try:
        step_output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created output directory: {step_output_dir}")
    except Exception as e:
        log_step_error(logger, f"Failed to create GNN processing output directory '{step_output_dir}': {e}")
        return False

    report_file_path = step_output_dir / "1_gnn_discovery_report.md"
    report_file_path_abs = report_file_path.resolve()

    processed_files_summary = []
    file_pattern = "**/*.md" if recursive else "*.md"
    
    # Counters for summary
    found_model_name_count = 0
    found_statespace_count = 0
    found_connections_count = 0
    files_with_errors_count = 0

    logger.debug(f"Searching for GNN files matching pattern '{file_pattern}' in '{gnn_target_path_abs}'")
    gnn_files = list(target_dir.glob(file_pattern))

    if not gnn_files:
        logger.info(f"No .md files found in '{gnn_target_path_abs}' with pattern '{file_pattern}'.")
        try:
            with open(report_file_path, "w", encoding="utf-8") as f_report:
                f_report.write("# GNN File Discovery Report\n\n")
                f_report.write(f"No .md files found in `{gnn_target_path_abs}` using pattern `{file_pattern}`.\n")
            logger.info(f"Empty report saved to: {report_file_path_abs}")
        except IOError as e:
            log_step_error(logger, f"Failed to write empty report to {report_file_path_abs}: {e}")
        return True  # Return True for empty directory (not an error)

    logger.info(f"Found {len(gnn_files)} .md file(s) to process in '{gnn_target_path_abs}'.")

    for gnn_file_path_obj in gnn_files:
        resolved_gnn_file_path = gnn_file_path_obj.resolve() 
        path_for_report_str = str(resolved_gnn_file_path.relative_to(gnn_target_path_abs)) if resolved_gnn_file_path.is_relative_to(gnn_target_path_abs) else str(resolved_gnn_file_path)
        
        logger.debug(f"Processing file: {path_for_report_str}")
        
        file_summary = {
            "file_name": resolved_gnn_file_path.name,
            "path": path_for_report_str,
            "model_name": "Not found",
            "sections_found": [],
            "model_parameters": {},
            "errors": []
        }
        
        try:
            with open(resolved_gnn_file_path, "r", encoding="utf-8") as f:
                content = f.read()
            logger.debug(f"Successfully read content from {path_for_report_str}.")
            
            # ModelName parsing
            model_name_section_header_text = "ModelName"
            parsed_model_name = "Not found" 

            _model_name_regex_string = rf"^##\s*{re.escape(model_name_section_header_text)}\s*$\r?"
            model_name_header_pattern = re.compile(_model_name_regex_string, re.IGNORECASE | re.MULTILINE)
            model_name_header_match = model_name_header_pattern.search(content)

            if model_name_header_match:
                logger.debug(f"  Found '## {model_name_section_header_text}' header in {path_for_report_str}")
                found_model_name_count += 1
                
                content_after_header = content[model_name_header_match.end():]
                next_section_header_match = re.search(r"^##\s+\w+", content_after_header, re.MULTILINE)
                
                if next_section_header_match:
                    name_region_content = content_after_header[:next_section_header_match.start()]
                else:
                    name_region_content = content_after_header
                
                extracted_name_candidate = ""
                for line in name_region_content.splitlines():
                    stripped_line = line.strip()
                    if stripped_line and not stripped_line.startswith("#"):
                        extracted_name_candidate = stripped_line
                        break
                
                if extracted_name_candidate:
                    parsed_model_name = extracted_name_candidate
                    logger.debug(f"    Extracted {model_name_section_header_text}: '{parsed_model_name}' from {path_for_report_str}")
                else:
                    parsed_model_name = "(Header found, but name line empty or only comments)"
                    logger.debug(f"    '## {model_name_section_header_text}' header found, but no suitable name line in {path_for_report_str}")

            file_summary["model_name"] = parsed_model_name
            file_summary["sections_found"].append(f"ModelName: {'Found: ' + parsed_model_name if parsed_model_name != 'Not found' else 'Not found'}")

            # StateSpaceBlock parsing
            statespace_section_header_text = "StateSpaceBlock"
            statespace_search_pattern = rf"^##\s*{re.escape(statespace_section_header_text)}\s*(?:#.*)?$"
            statespace_match = re.search(statespace_search_pattern, content, re.MULTILINE | re.IGNORECASE)
            if statespace_match:
                file_summary["sections_found"].append("StateSpaceBlock: Found")
                logger.debug(f"  Found {statespace_section_header_text} section in {path_for_report_str}")
                found_statespace_count += 1
            else:
                file_summary["sections_found"].append("StateSpaceBlock: Not found")
                logger.debug(f"  {statespace_section_header_text} section not found in {path_for_report_str}")

            # Connections parsing
            connections_section_header_text = "Connections"
            connections_search_pattern = rf"^##\s*{re.escape(connections_section_header_text)}\s*(?:#.*)?$"
            connections_match = re.search(connections_search_pattern, content, re.MULTILINE | re.IGNORECASE)
            if connections_match:
                file_summary["sections_found"].append("Connections: Found")
                logger.debug(f"  Found {connections_section_header_text} section in {path_for_report_str}")
                found_connections_count += 1
            else:
                file_summary["sections_found"].append("Connections: Not found")
                logger.debug(f"  {connections_section_header_text} section not found in {path_for_report_str}")

            # ModelParameters parsing
            parameters_section_header_text = "ModelParameters"
            parameters_search_pattern = rf"^##\s*{re.escape(parameters_section_header_text)}\s*(?:#.*)?$"
            parameters_match = re.search(parameters_search_pattern, content, re.MULTILINE | re.IGNORECASE)
            if parameters_match:
                logger.debug(f"  Found {parameters_section_header_text} section in {path_for_report_str}")
                
                content_after_header = content[parameters_match.end():]
                next_section_header_match = re.search(r"^##\s+\w+", content_after_header, re.MULTILINE)
                
                if next_section_header_match:
                    parameters_region_content = content_after_header[:next_section_header_match.start()]
                else:
                    parameters_region_content = content_after_header
                
                # Simple parameter extraction - look for key = value patterns
                parameter_pattern = r"^\s*(\w+)\s*=\s*(.+?)(?:\s*###.*)?$"
                for line in parameters_region_content.splitlines():
                    stripped_line = line.strip()
                    if stripped_line and not stripped_line.startswith("#"):
                        param_match = re.match(parameter_pattern, stripped_line)
                        if param_match:
                            param_name = param_match.group(1)
                            param_value_str = param_match.group(2).strip()
                            
                            # Try to parse the value
                            try:
                                import ast
                                param_value = ast.literal_eval(param_value_str)
                            except (ValueError, SyntaxError):
                                param_value = param_value_str
                            
                            file_summary["model_parameters"][param_name] = param_value
                            logger.debug(f"    Parsed ModelParameter: {param_name} = {param_value}")

        except Exception as e:
            error_msg = f"Error processing {path_for_report_str}: {e}"
            file_summary["errors"].append(error_msg)
            log_step_warning(logger, error_msg)
            files_with_errors_count += 1

        processed_files_summary.append(file_summary)

    # Generate report
    try:
        with open(report_file_path, "w", encoding="utf-8") as f_report:
            f_report.write("# GNN File Discovery Report\n\n")
            f_report.write(f"**Target Directory:** `{gnn_target_path_abs}`\n")
            f_report.write(f"**Search Pattern:** `{file_pattern}`\n")
            f_report.write(f"**Files Found:** {len(gnn_files)}\n\n")
            
            f_report.write("## Summary Statistics\n\n")
            f_report.write(f"- **Files with ModelName:** {found_model_name_count}\n")
            f_report.write(f"- **Files with StateSpaceBlock:** {found_statespace_count}\n")
            f_report.write(f"- **Files with Connections:** {found_connections_count}\n")
            f_report.write(f"- **Files with Errors:** {files_with_errors_count}\n\n")
            
            f_report.write("## Detailed File Analysis\n\n")
            for file_summary in processed_files_summary:
                f_report.write(f"### {file_summary['file_name']}\n\n")
                f_report.write(f"**Path:** `{file_summary['path']}`\n")
                f_report.write(f"**Model Name:** {file_summary['model_name']}\n\n")
                
                if file_summary['sections_found']:
                    f_report.write("**Sections Found:**\n")
                    for section in file_summary['sections_found']:
                        f_report.write(f"- {section}\n")
                    f_report.write("\n")
                
                if file_summary['model_parameters']:
                    f_report.write("**Model Parameters:**\n")
                    for param_name, param_value in file_summary['model_parameters'].items():
                        f_report.write(f"- `{param_name}` = `{param_value}`\n")
                    f_report.write("\n")
                
                if file_summary['errors']:
                    f_report.write("**Errors:**\n")
                    for error in file_summary['errors']:
                        f_report.write(f"- {error}\n")
                    f_report.write("\n")
                
                f_report.write("---\n\n")
        
        logger.info(f"Report saved to: {report_file_path_abs}")
        
        # Log summary
        if files_with_errors_count == 0:
            log_step_success(logger, f"Successfully processed {len(gnn_files)} GNN files without errors")
            return True
        elif files_with_errors_count < len(gnn_files):
            log_step_warning(logger, f"Processed {len(gnn_files)} GNN files with {files_with_errors_count} files having errors")
            return True
        else:
            log_step_error(logger, f"All {len(gnn_files)} GNN files had processing errors")
            return False
            
    except IOError as e:
        log_step_error(logger, f"Failed to write report to {report_file_path_abs}: {e}")
        return False 