"""
GNN Type Checker Core

This module provides the core functionality for validating GNN files 
to ensure they adhere to the specification and are correctly typed.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional, Union
import logging
import argparse

from visualization.parser import GNNParser

# Create a logger for this module
logger = logging.getLogger(__name__)


class GNNTypeChecker:
    """
    Type checker for GNN files to validate they adhere to the specification
    and have consistent typing.
    """
    
    # Required sections per GNN specification
    REQUIRED_SECTIONS = {
        'GNNSection',
        'GNNVersionAndFlags',
        'ModelName',
        'StateSpaceBlock',
        'Connections',
        'Footer',
        'Signature'
    }
    
    # Allowed time specifications
    VALID_TIME_SPECS = {
        'Static', 
        'Dynamic'
    }
    
    # Valid types for variables
    VALID_TYPES = {
        'float', 
        'int', 
        'bool', 
        'string', 
        'categorical'
    }
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize the GNN type checker.
        
        Args:
            strict_mode: Whether to enforce strict type checking rules
        """
        self.parser = GNNParser()
        self.strict_mode = strict_mode
        self.errors = []
        self.warnings = []
        logger.info(f"GNNTypeChecker initialized. Strict mode: {strict_mode}")
        
    def check_file(self, file_path: str) -> Tuple[bool, List[str], List[str]]:
        """
        Check a GNN file for type and structure validity.
        
        Args:
            file_path: Path to the GNN file to check
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        logger.info(f"Starting GNN check for file: {file_path}")
        self.errors = []
        self.warnings = []
        
        try:
            # Parse the file
            parsed_content = self.parser.parse_file(file_path)
            logger.debug(f"Successfully parsed file: {file_path}")
            
            # Check required sections
            self._check_required_sections(parsed_content)
            
            # Check state space variables and types
            self._check_state_space(parsed_content)
            
            # Check connections for consistency
            self._check_connections(parsed_content)
            
            # Check time specification
            self._check_time_specification(parsed_content)
            
            # Check equations
            self._check_equations(parsed_content)
            
            # Check version and flags
            self._check_version_and_flags(parsed_content)
            
        except Exception as e:
            logger.error(f"Failed to parse or check file {file_path}: {str(e)}", exc_info=True)
            self.errors.append(f"Failed to parse or check file: {str(e)}")
        
        # Log final errors and warnings for the file
        if self.errors:
            logger.warning(f"File {file_path} has {len(self.errors)} errors: {self.errors}")
        if self.warnings:
            logger.info(f"File {file_path} has {len(self.warnings)} warnings: {self.warnings}")
            
        is_valid = len(self.errors) == 0
        logger.info(f"Finished GNN check for file: {file_path}. Valid: {is_valid}")
        return is_valid, self.errors, self.warnings
    
    def check_directory(self, dir_path: str, recursive: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Check all GNN files in a directory.
        
        Args:
            dir_path: Path to the directory containing GNN files
            recursive: Whether to recursively check subdirectories
            
        Returns:
            Dictionary mapping file paths to check results
        """
        logger.info(f"Starting GNN check for directory: {dir_path}, Recursive: {recursive}")
        results = {}
        path = Path(dir_path)
        
        # Define pattern for GNN files
        pattern = "**/*.md" if recursive else "*.md"
        
        file_count = 0
        for file_path in path.glob(pattern):
            file_count += 1
            file_str = str(file_path)
            logger.debug(f"Processing file in directory: {file_str}")
            is_valid, errors, warnings = self.check_file(file_str)
            results[file_str] = {
                "is_valid": is_valid,
                "errors": errors,
                "warnings": warnings
            }
        logger.info(f"Finished checking directory {dir_path}. Processed {file_count} files.")
        return results
    
    def _check_required_sections(self, content: Dict[str, Any]) -> None:
        """
        Check if all required sections are present in the GNN file.
        
        Args:
            content: Parsed GNN content
        """
        missing_sections = self.REQUIRED_SECTIONS - set(content.keys())
        
        for section in missing_sections:
            error_msg = f"Missing required section: {section}"
            self.errors.append(error_msg)
            logger.debug(f"Validation Error: {error_msg}")
    
    def _check_state_space(self, content: Dict[str, Any]) -> None:
        """
        Check state space variables and their types.
        
        Args:
            content: Parsed GNN content
        """
        if 'Variables' not in content:
            error_msg = "No variables extracted from StateSpaceBlock"
            self.errors.append(error_msg)
            logger.debug(f"Validation Error: {error_msg}")
            return
        
        variables = content['Variables']
        
        for var_name, var_info in variables.items():
            # Check if dimensions are properly specified
            dims = var_info.get('dimensions', [])
            if not dims:
                error_msg = f"Variable '{var_name}' has no dimensions specified"
                self.errors.append(error_msg)
                logger.debug(f"Validation Error: {error_msg}")
            
            # Check variable type if specified
            var_type = var_info.get('type')
            if var_type and var_type not in self.VALID_TYPES:
                error_msg = f"Variable '{var_name}' has invalid type: {var_type}"
                self.errors.append(error_msg)
                logger.debug(f"Validation Error: {error_msg}")
    
    def _check_connections(self, content: Dict[str, Any]) -> None:
        """
        Check connections for consistency with declared variables.
        
        Args:
            content: Parsed GNN content
        """
        if 'Edges' not in content or 'Variables' not in content:
            return
        
        edges = content['Edges']
        variables = content['Variables']
        var_names = set(variables.keys())
        
        # Check for undefined variables in connections
        for edge in edges:
            source = edge.get('source')
            target = edge.get('target')
            
            # Extract base variable names (without time indices)
            source_base = re.sub(r'_t(?:\+\d+)?', '', source)
            target_base = re.sub(r'_t(?:\+\d+)?', '', target)
            
            if source_base not in var_names:
                error_msg = f"Connection references undefined variable: {source}"
                self.errors.append(error_msg)
                logger.debug(f"Validation Error: {error_msg}")
            
            if target_base not in var_names:
                error_msg = f"Connection references undefined variable: {target}"
                self.errors.append(error_msg)
                logger.debug(f"Validation Error: {error_msg}")
    
    def _check_time_specification(self, content: Dict[str, Any]) -> None:
        """
        Check time specification for validity.
        
        Args:
            content: Parsed GNN content
        """
        if 'Time' not in content:
            warning_msg = "Time section not specified"
            self.warnings.append(warning_msg)
            logger.debug(f"Validation Warning: {warning_msg}")
            return
        
        time_spec = content['Time']
        lines = time_spec.split('\n')
        
        primary_spec = lines[0].strip() if lines else ""
        
        if primary_spec not in self.VALID_TIME_SPECS:
            error_msg = f"Invalid time specification: {primary_spec}"
            self.errors.append(error_msg)
            logger.debug(f"Validation Error: {error_msg}")
        
        # If dynamic, check additional time specifications
        if primary_spec == 'Dynamic':
            has_time_var = False
            for line in lines[1:]:
                if line.startswith('DiscreteTime=') or line.startswith('ContinuousTime='):
                    has_time_var = True
                    time_var = line.split('=')[1].strip()
                    
                    # Check if the time variable is defined
                    if 'Variables' in content and time_var not in content['Variables']:
                        error_msg = f"Time variable {time_var} not defined in StateSpaceBlock"
                        self.errors.append(error_msg)
                        logger.debug(f"Validation Error: {error_msg}")
            
            if not has_time_var and self.strict_mode:
                error_msg = "Dynamic model requires DiscreteTime or ContinuousTime specification"
                self.errors.append(error_msg)
                logger.debug(f"Validation Error: {error_msg}")
    
    def _check_equations(self, content: Dict[str, Any]) -> None:
        """
        Check equations for references to undefined variables.
        
        Args:
            content: Parsed GNN content
        """
        if 'Equations' not in content or 'Variables' not in content:
            return
        
        equations = content['Equations']
        variables = content['Variables']
        var_names = set(variables.keys())
        
        # Extract variable names from equations using simple regex
        # This is a simplistic approach and might not catch all variable references
        equation_lines = equations.split('\n')
        referenced_in_equation = set()
        for line in equation_lines:
            if '=' in line:
                left_side = line.split('=')[0].strip()
                
                # Extract the variable name from lhs (handle subscripts and superscripts)
                match = re.match(r'([a-zA-Z0-9_]+)(?:_[a-zA-Z0-9{}\+]+)?(?:\^[a-zA-Z0-9{}]+)?', left_side)
                if match:
                    var_name = match.group(1)
                    if var_name not in var_names and not self._is_common_math_function(var_name):
                        error_msg = f"Equation '{line}' references undefined variable: {var_name}"
                        self.errors.append(error_msg)
                        logger.debug(f"Validation Error: {error_msg}")
                        referenced_in_equation.add(var_name)
    
    def _check_version_and_flags(self, content: Dict[str, Any]) -> None:
        """
        Check GNN version and flags for validity.
        
        Args:
            content: Parsed GNN content
        """
        if 'GNNVersionAndFlags' not in content:
            return
        
        version_flags = content['GNNVersionAndFlags']
        
        # Check if GNN version is specified
        if not re.search(r'GNN v\d+(?:\.\d+)?', version_flags):
            self.errors.append("Invalid GNNVersionAndFlags: Missing GNN version")
    
    def _is_common_math_function(self, name: str) -> bool:
        # Basic check for common math functions to avoid false positives
        return name.lower() in ['ln', 'log', 'exp', 'sin', 'cos', 'tan', 'sqrt', 'softmax', 'sigmoid']

    def generate_report(self, results: Dict[str, Dict[str, Any]], 
                        output_dir_base: Path, 
                        report_md_filename: str = "type_check_report.md",
                        project_root_path: Optional[Union[str, Path]] = None) -> str:
        """
        Generate a markdown report of the type checking results.
        
        Args:
            results: Dictionary mapping file paths to check results
            output_dir_base: The base directory where type checking outputs (like this report) are saved.
            report_md_filename: The specific name for the markdown report file.
            project_root_path: Optional path to the project root for making file paths relative.
            
        Returns:
            String summary of the report.
        """
        logger.info(f"Generating type check report: {report_md_filename} in {output_dir_base}")
        report_parts = ["# GNN Type Checker Report"]
        valid_count = 0
        invalid_count = 0

        # Resolve project_root once
        actual_project_root = None
        if project_root_path:
            actual_project_root = Path(project_root_path).resolve()

        for file_path_str, result in results.items():
            file_path_obj = Path(file_path_str).resolve()
            display_path = file_path_str # Default to original string if not made relative
            if actual_project_root:
                try:
                    display_path = str(file_path_obj.relative_to(actual_project_root))
                except ValueError:
                    display_path = file_path_obj.name # Fallback to filename if not in project root
            
            if result["is_valid"]:
                valid_count += 1
                report_parts.append(f"## {file_path_obj.name}: ✅ VALID")
                report_parts.append(f"Path: {display_path}")
                if result.get("warnings"):
                    report_parts.append("Warnings:")
                    for warning in result["warnings"]:
                        report_parts.append(f"  - {warning}")
            else:
                invalid_count += 1
                report_parts.append(f"## {file_path_obj.name}: ❌ INVALID")
                report_parts.append(f"Path: {display_path}")
                if result.get("errors"):
                    report_parts.append("Errors:")
                    for error in result["errors"]:
                        report_parts.append(f"  - {error}")
                if result.get("warnings"):
                    report_parts.append("Warnings:")
                    for warning in result["warnings"]:
                        report_parts.append(f"  - {warning}")
            report_parts.append("")  # Add a blank line for spacing

        summary = f"Checked {len(results)} files, {valid_count} valid, {invalid_count} invalid"
        report_parts.append(summary)
        report_parts.append("")

        full_report_str = "\n".join(report_parts)
        
        report_path = output_dir_base / report_md_filename
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(full_report_str)
            
        logger.info(f"Successfully wrote GNN type check report to: {report_path}")
        return full_report_str
    
    def _generate_html_report(self, results: Dict[str, Dict[str, Any]], output_file: Path) -> None:
        """
        Generate an HTML report with visualizations.
        
        Args:
            results: Dictionary mapping file paths to check results
            output_file: Path to save the HTML report
        """
        import json
        from pathlib import Path
        
        # Count error and warning types for visualization
        error_types = {}
        warning_types = {}
        
        for result in results.values():
            for error in result["errors"]:
                error_type = error.split(":")[0] if ":" in error else error
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            for warning in result["warnings"]:
                warning_type = warning.split(":")[0] if ":" in warning else warning
                warning_types[warning_type] = warning_types.get(warning_type, 0) + 1
        
        # Create HTML content with embedded charts
        html_content = f"""<!DOCTYPE html>
        <html>
        <head>
            <title>GNN Type Checking Visualization</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .chart-container {{ width: 600px; height: 400px; margin-bottom: 30px; }}
                .summary {{ margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h1>GNN Type Checking Visualization</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total files: {len(results)}</p>
                <p>Valid files: {sum(1 for r in results.values() if r["is_valid"])}</p>
                <p>Invalid files: {sum(1 for r in results.values() if not r["is_valid"])}</p>
                <p>Total errors: {sum(len(r["errors"]) for r in results.values())}</p>
                <p>Total warnings: {sum(len(r["warnings"]) for r in results.values())}</p>
            </div>
            
            <div class="chart-container">
                <h2>File Validity</h2>
                <canvas id="validityChart"></canvas>
            </div>
            
            <div class="chart-container">
                <h2>Error Types</h2>
                <canvas id="errorsChart"></canvas>
            </div>
            
            <div class="chart-container">
                <h2>Warning Types</h2>
                <canvas id="warningsChart"></canvas>
            </div>
            
            <script>
                // Validity pie chart
                const validityCtx = document.getElementById('validityChart').getContext('2d');
                new Chart(validityCtx, {{
                    type: 'pie',
                    data: {{
                        labels: ['Valid', 'Invalid'],
                        datasets: [{{
                            data: [
                                {sum(1 for r in results.values() if r["is_valid"])}, 
                                {sum(1 for r in results.values() if not r["is_valid"])}
                            ],
                            backgroundColor: ['rgba(75, 192, 192, 0.2)', 'rgba(255, 99, 132, 0.2)'],
                            borderColor: ['rgba(75, 192, 192, 1)', 'rgba(255, 99, 132, 1)'],
                            borderWidth: 1
                        }}]
                    }}
                }});
                
                // Error types chart
                const errorsCtx = document.getElementById('errorsChart').getContext('2d');
                new Chart(errorsCtx, {{
                    type: 'bar',
                    data: {{
                        labels: {json.dumps(list(error_types.keys()))},
                        datasets: [{{
                            label: 'Count',
                            data: {json.dumps(list(error_types.values()))},
                            backgroundColor: 'rgba(255, 99, 132, 0.2)',
                            borderColor: 'rgba(255, 99, 132, 1)',
                            borderWidth: 1
                        }}]
                    }},
                    options: {{
                        scales: {{
                            y: {{
                                beginAtZero: true
                            }}
                        }}
                    }}
                }});
                
                // Warning types chart
                const warningsCtx = document.getElementById('warningsChart').getContext('2d');
                new Chart(warningsCtx, {{
                    type: 'bar',
                    data: {{
                        labels: {json.dumps(list(warning_types.keys()))},
                        datasets: [{{
                            label: 'Count',
                            data: {json.dumps(list(warning_types.values()))},
                            backgroundColor: 'rgba(255, 206, 86, 0.2)',
                            borderColor: 'rgba(255, 206, 86, 1)',
                            borderWidth: 1
                        }}]
                    }},
                    options: {{
                        scales: {{
                            y: {{
                                beginAtZero: true
                            }}
                        }}
                    }}
                }});
            </script>
        </body>
        </html>
        """
        
        # Save HTML report
        with open(output_file, 'w') as f:
            f.write(html_content)
        logger.info(f"Successfully wrote HTML report to: {output_file}")
    
    def generate_json_data(self, results: Dict[str, Dict[str, Any]], output_file: Path) -> None:
        """
        Generate JSON data for resource estimator and general use.
        
        Args:
            results: Dictionary mapping file paths to check results
            output_file: Path to save the JSON data
        """
        import json
        
        # Prepare data for resource estimator
        json_data = {
            "type_check_results": results,
            "summary": {
                "total_files": len(results),
                "valid_files": sum(1 for r in results.values() if r["is_valid"]),
                "invalid_files": sum(1 for r in results.values() if not r["is_valid"]),
                "total_errors": sum(len(r["errors"]) for r in results.values()),
                "total_warnings": sum(len(r["warnings"]) for r in results.values())
            }
        }
        
        # Save JSON data
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"Successfully wrote JSON data to: {output_file}") 