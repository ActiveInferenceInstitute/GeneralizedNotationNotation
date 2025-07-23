
"""
GNN Type Checker Core

This module provides the core functionality for validating GNN files 
to ensure they adhere to the specification and are correctly typed.
"""

import re
import logging
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import traceback

# Configure logging to avoid format string issues
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create a logger for this module
logger = logging.getLogger(__name__)

class SimpleGNNParser:
    """Simple GNN parser that handles markdown format correctly."""
    
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """Parse a GNN file and return structured content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self._parse_markdown_format(content)
        except Exception as e:
            logger.error(f"Failed to parse file {file_path}: {e}")
            return {}
    
    def _parse_markdown_format(self, content: str) -> Dict[str, Any]:
        """Parse GNN file in Markdown format."""
        sections: Dict[str, Any] = {}
        
        # Extract header comments (lines before first ## section)
        lines = content.split('\n')
        header_lines = []
        section_start = -1
        
        for i, line in enumerate(lines):
            if line.strip().startswith('## '):
                section_start = i
                break
            header_lines.append(line)
        
        if header_lines:
            header_content = '\n'.join(header_lines).strip()
            sections['_HeaderComments'] = header_content
            
            # Extract ModelName from header
            for line in header_lines:
                if line.strip().startswith('# GNN Example:'):
                    sections['ModelName'] = line.replace('# GNN Example:', '').strip()
                    break
                elif line.strip().startswith('#') and 'ModelName' not in sections:
                    sections['ModelName'] = line.replace('#', '').strip()
        
        # Extract sections
        current_section = None
        current_content = []
        
        for i in range(section_start, len(lines)) if section_start >= 0 else []:
            line = lines[i]
            
            if line.strip().startswith('## '):
                # Save previous section
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = line.replace('##', '').strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        # Process state space and connections
        self._process_state_space(sections)
        self._process_connections(sections)
        
        return sections
    
    def _process_state_space(self, sections: Dict[str, Any]) -> None:
        """Process StateSpaceBlock to extract variables."""
        if 'StateSpaceBlock' not in sections:
            return
        
        content = sections['StateSpaceBlock']
        variables = {}
        
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Match variable definitions like: A[3,3,type=float]
            match = re.match(r'(\w+)\s*\[([^\]]+)\](?:\s*#\s*(.*))?', line)
            if match:
                var_name = match.group(1)
                dims_str = match.group(2)
                comment = match.group(3) if match.group(3) else ""
                
                dimensions = []
                var_type = None
                
                for part in dims_str.split(','):
                    part = part.strip()
                    if part.startswith('type='):
                        var_type = part.split('=')[1]
                    else:
                        try:
                            dimensions.append(int(part))
                        except ValueError:
                            dimensions.append(part)
                
                variables[var_name] = {
                    'dimensions': dimensions,
                    'type': var_type or 'float',  # default to float
                    'comment': comment
                }
        
        if variables:
            sections['Variables'] = variables
    
    def _process_connections(self, sections: Dict[str, Any]) -> None:
        """Process Connections section to extract edges."""
        if 'Connections' not in sections:
            return
        
        content = sections['Connections']
        edges = []
        
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Match connections like: D>s, s-A, etc.
            if '>' in line:
                parts = line.split('>')
                if len(parts) == 2:
                    edges.append({
                        'source': parts[0].strip(),
                        'target': parts[1].strip(),
                        'type': 'directed'
                    })
            elif '-' in line:
                parts = line.split('-')
                if len(parts) == 2:
                    edges.append({
                        'source': parts[0].strip(),
                        'target': parts[1].strip(),
                        'type': 'undirected'
                    })
        
        if edges:
            sections['Edges'] = edges


class GNNTypeChecker:
    """
    Type checker for GNN files to validate they adhere to the specification
    and have consistent typing.
    """
    
    # Required sections per GNN specification (more flexible)
    REQUIRED_SECTIONS = {
        'GNNSection',
        'GNNVersionAndFlags', 
        'StateSpaceBlock',
        'Connections'
    }
    
    # Optional but common sections
    OPTIONAL_SECTIONS = {
        'ModelName',
        'ModelAnnotation',
        'InitialParameterization',
        'Equations',
        'Time',
        'ActInfOntologyAnnotation',
        'ModelParameters',
        'Footer',
        'Signature'
    }
    
    # Allowed time specifications
    VALID_TIME_SPECS = {
        'Static', 
        'Dynamic',
        'Time=t',
        'Discrete',
        'ModelTimeHorizon=Unbounded'
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
        self.parser = SimpleGNNParser()
        self.strict_mode = strict_mode
        self.errors = []
        self.warnings = []
        logger.info(f"GNNTypeChecker initialized. Strict mode: {strict_mode}")
        
    def check_file(self, file_path: str) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
        """
        Check a GNN file for type and structure validity.
        Returns: (is_valid, errors, warnings, details_dict)
        """
        logger.info(f"Starting GNN check for file: {file_path}")
        self.errors = []
        self.warnings = []
        details = {}
        
        try:
            parsed_content = self.parser.parse_file(file_path)
            logger.debug(f"Successfully parsed file: {file_path}")
            logger.debug(f"Parsed sections: {list(parsed_content.keys())}")
            
            self._check_required_sections(parsed_content)
            self._check_state_space(parsed_content)
            self._check_connections(parsed_content)
            self._check_time_specification(parsed_content)
            self._check_equations(parsed_content)
            self._check_version_and_flags(parsed_content)
            
            # Enhanced analysis data collection
            details.update(self._collect_section_analysis(parsed_content))
            details.update(self._collect_variable_analysis(parsed_content))
            details.update(self._collect_connection_analysis(parsed_content))
            details.update(self._collect_model_complexity(parsed_content))
            details.update(self._collect_parameterization_analysis(parsed_content))
            
        except Exception as e:
            logger.error(f"Failed to parse or check file {file_path}: {str(e)}", exc_info=True)
            self.errors.append(f"Failed to parse or check file: {str(e)}")
        
        is_valid = len(self.errors) == 0
        logger.info(f"Finished GNN check for file: {file_path}. Valid: {is_valid}")
        if self.errors:
            logger.info(f"Errors found: {self.errors}")
        if self.warnings:
            logger.info(f"Warnings found: {self.warnings}")
            
        details['is_valid'] = is_valid
        details['errors'] = list(self.errors)
        details['warnings'] = list(self.warnings)
        details['file_path'] = file_path
        details['file_name'] = Path(file_path).name
        return is_valid, self.errors, self.warnings, details
    
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
            is_valid, errors, warnings, details = self.check_file(file_str)
            results[file_str] = {
                "is_valid": is_valid,
                "errors": errors,
                "warnings": warnings,
                "details": details
            }
        logger.info(f"Finished checking directory {dir_path}. Processed {file_count} files.")
        return results
    
    def _check_required_sections(self, content: Dict[str, Any]) -> None:
        """
        Check if all required sections are present in the GNN file.
        
        Args:
            content: Parsed GNN content
        """
        available_sections = set(content.keys())
        missing_sections = self.REQUIRED_SECTIONS - available_sections
        
        for section in missing_sections:
            error_msg = f"Missing required section: {section}"
            self.errors.append(error_msg)
            logger.debug(f"Validation Error: {error_msg}")
        
        # Check for ModelName in header comments if not in sections
        if 'ModelName' not in available_sections and '_HeaderComments' in content:
            if any('GNN Example:' in line for line in content['_HeaderComments'].split('\n')):
                # ModelName can be derived from header
                pass
            else:
                self.warnings.append("ModelName not explicitly defined but may be derivable from header")
    
    def _check_state_space(self, content: Dict[str, Any]) -> None:
        """
        Check state space variables and their types.
        
        Args:
            content: Parsed GNN content
        """
        if 'Variables' not in content:
            if 'StateSpaceBlock' in content:
                # StateSpaceBlock exists but no variables were parsed
                self.warnings.append("StateSpaceBlock found but no variables could be parsed")
            else:
                error_msg = "No StateSpaceBlock section found"
                self.errors.append(error_msg)
                logger.debug(f"Validation Error: {error_msg}")
            return
        
        variables = content['Variables']
        logger.debug(f"Checking {len(variables)} variables")
        
        for var_name, var_info in variables.items():
            # Check if dimensions are properly specified
            dims = var_info.get('dimensions', [])
            if not dims and self.strict_mode:
                error_msg = f"Variable '{var_name}' has no dimensions specified"
                self.errors.append(error_msg)
                logger.debug(f"Validation Error: {error_msg}")
            
            # Check variable type if specified
            var_type = var_info.get('type')
            if var_type and var_type not in self.VALID_TYPES:
                # Only warning for now as the type system might be extensible
                warning_msg = f"Variable '{var_name}' has non-standard type: {var_type}"
                self.warnings.append(warning_msg)
                logger.debug(f"Validation Warning: {warning_msg}")
    
    def _check_connections(self, content: Dict[str, Any]) -> None:
        """
        Check connections for consistency with declared variables.
        
        Args:
            content: Parsed GNN content
        """
        if 'Edges' not in content:
            if 'Connections' in content:
                self.warnings.append("Connections section found but no edges could be parsed")
            else:
                self.warnings.append("No Connections section found")
            return
        
        if 'Variables' not in content:
            self.warnings.append("Cannot validate connections without variables")
            return
        
        edges = content['Edges']
        variables = content['Variables']
        var_names = set(variables.keys())
        
        logger.debug(f"Checking {len(edges)} connections against {len(var_names)} variables")
        
        # Check for undefined variables in connections
        for edge in edges:
            source = edge.get('source', '')
            target = edge.get('target', '')
            
            # Extract base variable names (without time indices or suffixes)
            source_base = re.sub(r'_t(?:\+\d+)?|_prime|\+\d+', '', source)
            target_base = re.sub(r'_t(?:\+\d+)?|_prime|\+\d+', '', target)
            
            if source_base not in var_names:
                # Check if it might be a Greek letter or special symbol
                if source_base not in ['π', 'u', 'G', 't'] and len(source_base) > 0:
                    warning_msg = f"Connection references potentially undefined variable: {source}"
                    self.warnings.append(warning_msg)
                    logger.debug(f"Validation Warning: {warning_msg}")
            
            if target_base not in var_names:
                if target_base not in ['π', 'u', 'G', 't'] and len(target_base) > 0:
                    warning_msg = f"Connection references potentially undefined variable: {target}"
                    self.warnings.append(warning_msg)
                    logger.debug(f"Validation Warning: {warning_msg}")
    
    def _check_time_specification(self, content: Dict[str, Any]) -> None:
        """
        Check time specification for validity.
        
        Args:
            content: Parsed GNN content
        """
        if 'Time' not in content:
            self.warnings.append("Time section not specified")
            return
        
        time_spec = content['Time']
        lines = [line.strip() for line in time_spec.split('\n') if line.strip()]
        
        if not lines:
            self.warnings.append("Empty Time section")
            return
        
        # Check for valid time specifications
        valid_spec_found = False
        for line in lines:
            if any(spec in line for spec in self.VALID_TIME_SPECS):
                valid_spec_found = True
                break
        
        if not valid_spec_found and self.strict_mode:
            error_msg = f"No valid time specification found in Time section"
            self.errors.append(error_msg)
            logger.debug(f"Validation Error: {error_msg}")
        elif not valid_spec_found:
            self.warnings.append("Time section may not contain standard time specifications")
    
    def _check_equations(self, content: Dict[str, Any]) -> None:
        """
        Check equations for references to undefined variables.
        
        Args:
            content: Parsed GNN content
        """
        if 'Equations' not in content:
            # Equations are optional
            return
        
        equations = content['Equations']
        if not equations.strip():
            return
        
        if 'Variables' not in content:
            self.warnings.append("Cannot validate equations without variables")
            return
        
        variables = content['Variables']
        var_names = set(variables.keys())
        
        # Basic validation - just check if equations section is present and non-empty
        equation_lines = [line.strip() for line in equations.split('\n') if line.strip() and not line.startswith('#')]
        
        if equation_lines:
            logger.debug(f"Found {len(equation_lines)} equation lines")
            # For now, just log that equations are present
            # More sophisticated equation parsing could be added later
    
    def _check_version_and_flags(self, content: Dict[str, Any]) -> None:
        """
        Check GNN version and flags for validity.
        
        Args:
            content: Parsed GNN content
        """
        if 'GNNVersionAndFlags' not in content:
            self.warnings.append("GNNVersionAndFlags section not found")
            return
        
        version_flags = content['GNNVersionAndFlags']
        
        # Check if GNN version is specified
        if not re.search(r'GNN\s+v?\d+(?:\.\d+)?', version_flags, re.IGNORECASE):
            self.warnings.append("GNN version not clearly specified in GNNVersionAndFlags")

    def _is_common_math_function(self, name: str) -> bool:
        # Basic check for common math functions to avoid false positives
        return name.lower() in ['ln', 'log', 'exp', 'sin', 'cos', 'tan', 'sqrt', 'softmax', 'sigmoid']

    def _collect_section_analysis(self, parsed_content: Dict[str, Any]) -> Dict[str, Any]:
        """Collect comprehensive section presence and content analysis."""
        all_sections = [
            'GNNSection', 'GNNVersionAndFlags', 'ModelName', 'ModelAnnotation',
            'StateSpaceBlock', 'Connections', 'InitialParameterization',
            'Equations', 'Time', 'ActInfOntologyAnnotation', 'ModelParameters',
            'Footer', 'Signature'
        ]
        
        sections_data = {
            'sections': {s: s in parsed_content for s in all_sections},
            'section_content_lengths': {},
            'required_sections_present': 0,
            'optional_sections_present': 0
        }
        
        required_sections = self.REQUIRED_SECTIONS
        for section in all_sections:
            if section in parsed_content:
                content = parsed_content[section]
                sections_data['section_content_lengths'][section] = len(str(content))
                if section in required_sections:
                    sections_data['required_sections_present'] += 1
                else:
                    sections_data['optional_sections_present'] += 1
        
        return sections_data

    def _collect_variable_analysis(self, parsed_content: Dict[str, Any]) -> Dict[str, Any]:
        """Collect comprehensive variable analysis."""
        variables_data = {
            'variables': [],
            'variable_count': 0,
            'type_distribution': {},
            'dimension_analysis': {
                'scalar_vars': 0,
                'vector_vars': 0,
                'matrix_vars': 0,
                'tensor_vars': 0,
                'max_dimensions': 0
            }
        }
        
        if 'Variables' in parsed_content:
            variables = parsed_content['Variables']
            variables_data['variable_count'] = len(variables)
            
            for vname, vinfo in variables.items():
                var_type = vinfo.get('type', 'unknown')
                dims = vinfo.get('dimensions', [])
                
                # Type distribution
                variables_data['type_distribution'][var_type] = variables_data['type_distribution'].get(var_type, 0) + 1
                
                # Dimension analysis
                dim_count = len(dims)
                variables_data['dimension_analysis']['max_dimensions'] = max(
                    variables_data['dimension_analysis']['max_dimensions'], dim_count
                )
                
                if dim_count == 0:
                    variables_data['dimension_analysis']['scalar_vars'] += 1
                elif dim_count == 1:
                    variables_data['dimension_analysis']['vector_vars'] += 1
                elif dim_count == 2:
                    variables_data['dimension_analysis']['matrix_vars'] += 1
                else:
                    variables_data['dimension_analysis']['tensor_vars'] += 1
                
                variables_data['variables'].append({
                    'name': vname,
                    'type': var_type,
                    'dimensions': dims,
                    'dimension_count': dim_count,
                    'total_elements': self._calculate_total_elements(dims)
                })
        
        return variables_data

    def _collect_connection_analysis(self, parsed_content: Dict[str, Any]) -> Dict[str, Any]:
        """Collect comprehensive connection analysis."""
        connections_data = {
            'connections': [],
            'connection_count': 0,
            'connection_types': {
                'directed': 0,
                'undirected': 0,
                'temporal': 0
            },
            'variable_connectivity': {},
            'connection_patterns': []
        }
        
        if 'Edges' in parsed_content:
            edges = parsed_content['Edges']
            connections_data['connection_count'] = len(edges)
            
            # Track variable connectivity
            for edge in edges:
                source = edge.get('source', '')
                target = edge.get('target', '')
                edge_type = edge.get('type', 'directed')
                
                # Count connection types
                if edge_type == 'directed':
                    connections_data['connection_types']['directed'] += 1
                elif edge_type == 'undirected':
                    connections_data['connection_types']['undirected'] += 1
                
                # Check for temporal connections
                if '+' in source or '+' in target or '_prime' in source or '_prime' in target:
                    connections_data['connection_types']['temporal'] += 1
                
                # Track variable connectivity
                source_base = re.sub(r'_prime|\+\d+', '', source)
                target_base = re.sub(r'_prime|\+\d+', '', target)
                
                connections_data['variable_connectivity'][source_base] = connections_data['variable_connectivity'].get(source_base, 0) + 1
                connections_data['variable_connectivity'][target_base] = connections_data['variable_connectivity'].get(target_base, 0) + 1
                
                connections_data['connections'].append({
                    'source': source,
                    'target': target,
                    'type': edge_type,
                    'is_temporal': '+' in source or '+' in target or '_prime' in source or '_prime' in target
                })
        
        return connections_data

    def _collect_model_complexity(self, parsed_content: Dict[str, Any]) -> Dict[str, Any]:
        """Collect model complexity metrics."""
        complexity_data = {
            'model_complexity': {
                'variable_complexity': 0,
                'connection_complexity': 0,
                'equation_complexity': 0,
                'overall_complexity': 0
            },
            'model_type': 'Static',
            'time_dynamics': {
                'is_dynamic': False,
                'time_variables': [],
                'temporal_connections': 0
            }
        }
        
        # Determine model type
        if 'Time' in parsed_content:
            time_spec = str(parsed_content['Time'])
            logger.debug(f"Time specification content: {repr(time_spec)}")
            # Check for Dynamic in the time specification
            if 'Dynamic' in time_spec:
                complexity_data['model_type'] = 'Dynamic'
                complexity_data['time_dynamics']['is_dynamic'] = True
                logger.debug("Detected Dynamic model")
            elif 'Static' in time_spec:
                complexity_data['model_type'] = 'Static'
                complexity_data['time_dynamics']['is_dynamic'] = False
                logger.debug("Detected Static model")
            else:
                logger.debug(f"Could not determine model type from time spec: {time_spec}")
        
        # Calculate complexity metrics
        var_count = len(parsed_content.get('Variables', {}))
        edge_count = len(parsed_content.get('Edges', []))
        equations = parsed_content.get('Equations', '')
        equation_lines = [line.strip() for line in equations.split('\n') if line.strip()]
        equation_count = len(equation_lines)
        
        complexity_data['model_complexity']['variable_complexity'] = var_count
        complexity_data['model_complexity']['connection_complexity'] = edge_count
        complexity_data['model_complexity']['equation_complexity'] = equation_count
        
        # Overall complexity (weighted combination)
        complexity_data['model_complexity']['overall_complexity'] = (
            var_count * 0.3 + edge_count * 0.4 + equation_count * 0.3
        )
        
        return complexity_data

    def _collect_parameterization_analysis(self, parsed_content: Dict[str, Any]) -> Dict[str, Any]:
        """Collect parameterization analysis."""
        param_data = {
            'parameterization': {
                'has_initial_params': False,
                'parameter_count': 0,
                'parameter_types': {},
                'matrix_parameters': [],
                'vector_parameters': [],
                'scalar_parameters': []
            }
        }
        
        if 'InitialParameterization' in parsed_content:
            param_data['parameterization']['has_initial_params'] = True
            # Basic parameter counting - could be enhanced with actual parsing
            param_content = str(parsed_content['InitialParameterization'])
            param_data['parameterization']['parameter_count'] = param_content.count('=') + param_content.count(':')
        
        return param_data

    def _calculate_total_elements(self, dimensions: List[Any]) -> int:
        """Calculate total number of elements for given dimensions."""
        try:
            total = 1
            for dim in dimensions:
                if isinstance(dim, (int, float)):
                    total *= int(dim)
                elif isinstance(dim, str) and dim.isdigit():
                    total *= int(dim)
                else:
                    # For symbolic dimensions, estimate
                    total *= 2
            return total
        except:
            return 1

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
        report_parts.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_parts.append("")
        
        valid_count = 0
        invalid_count = 0
        total_errors = 0
        total_warnings = 0

        # Resolve project_root once
        actual_project_root = None
        if project_root_path:
            actual_project_root = Path(project_root_path).resolve()

        # Summary section
        report_parts.append("## 📊 Summary")
        report_parts.append("")
        
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
            else:
                invalid_count += 1
                total_errors += len(result.get("errors", []))
            
            total_warnings += len(result.get("warnings", []))
        
        report_parts.append(f"- **Total Files Checked:** {len(results)}")
        report_parts.append(f"- **Valid Files:** {valid_count} ✅")
        report_parts.append(f"- **Invalid Files:** {invalid_count} ❌")
        report_parts.append(f"- **Total Errors:** {total_errors}")
        report_parts.append(f"- **Total Warnings:** {total_warnings}")
        report_parts.append(f"- **Success Rate:** {(valid_count/len(results)*100):.1f}%")
        report_parts.append("")
        
        # Detailed file analysis
        report_parts.append("## 📁 File Analysis")
        report_parts.append("")

        for file_path_str, result in results.items():
            file_path_obj = Path(file_path_str).resolve()
            display_path = file_path_str # Default to original string if not made relative
            if actual_project_root:
                try:
                    display_path = str(file_path_obj.relative_to(actual_project_root))
                except ValueError:
                    display_path = file_path_obj.name # Fallback to filename if not in project root
            
            if result["is_valid"]:
                report_parts.append(f"### {file_path_obj.name}: ✅ VALID")
                report_parts.append(f"**Path:** `{display_path}`")
                
                # Add detailed analysis if available
                if "details" in result and result["details"]:
                    details = result["details"]
                    
                    # Variable analysis
                    if "variable_analysis" in details:
                        var_analysis = details["variable_analysis"]
                        report_parts.append(f"**Variables:** {var_analysis.get('variable_count', 0)}")
                        if var_analysis.get('type_distribution'):
                            report_parts.append("**Type Distribution:**")
                            for var_type, count in var_analysis['type_distribution'].items():
                                report_parts.append(f"  - {var_type}: {count}")
                    
                    # Connection analysis
                    if "connection_analysis" in details:
                        conn_analysis = details["connection_analysis"]
                        report_parts.append(f"**Connections:** {conn_analysis.get('total_connections', 0)}")
                        if conn_analysis.get('connection_types'):
                            report_parts.append("**Connection Types:**")
                            for conn_type, count in conn_analysis['connection_types'].items():
                                report_parts.append(f"  - {conn_type}: {count}")
                    
                    # Model complexity
                    if "model_complexity" in details:
                        complexity = details["model_complexity"]
                        report_parts.append(f"**Model Complexity:**")
                        report_parts.append(f"  - Total Parameters: {complexity.get('total_parameters', 'N/A')}")
                        report_parts.append(f"  - Computational Complexity: {complexity.get('computational_complexity', 'N/A')}")
                
                if result.get("warnings"):
                    report_parts.append("**Warnings:**")
                    for warning in result["warnings"]:
                        report_parts.append(f"  - ⚠️ {warning}")
            else:
                report_parts.append(f"### {file_path_obj.name}: ❌ INVALID")
                report_parts.append(f"**Path:** `{display_path}`")
                if result.get("errors"):
                    report_parts.append("**Errors:**")
                    for error in result["errors"]:
                        report_parts.append(f"  - ❌ {error}")
                if result.get("warnings"):
                    report_parts.append("**Warnings:**")
                    for warning in result["warnings"]:
                        report_parts.append(f"  - ⚠️ {warning}")
            
            report_parts.append("")  # Add a blank line for spacing

        # Overall summary
        report_parts.append("## 🎯 Overall Assessment")
        report_parts.append("")
        
        if valid_count == len(results):
            report_parts.append("🎉 **All files passed validation successfully!**")
        elif valid_count > 0:
            report_parts.append(f"✅ **{valid_count} files are valid**")
            report_parts.append(f"❌ **{invalid_count} files have issues**")
        else:
            report_parts.append("❌ **All files have validation issues**")
        
        if total_warnings > 0:
            report_parts.append(f"⚠️ **{total_warnings} warnings found** - Review recommended")
        
        report_parts.append("")
        report_parts.append("---")
        report_parts.append(f"*Report generated by GNN Type Checker v1.0*")

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


class TypeCheckResult:
    """
    Structured result type for GNN type checking operations.
    """
    
    def __init__(self, is_valid: bool, errors: List[str], warnings: List[str], details: Dict[str, Any]):
        """
        Initialize a TypeCheckResult.
        
        Args:
            is_valid: Whether the GNN file passed all type checks
            errors: List of error messages
            warnings: List of warning messages
            details: Additional details about the check
        """
        self.is_valid = is_valid
        self.errors = errors
        self.warnings = warnings
        self.details = details
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'details': self.details
        }
    
    def __str__(self) -> str:
        """String representation of the result."""
        status = "VALID" if self.is_valid else "INVALID"
        return f"TypeCheckResult({status}, {len(self.errors)} errors, {len(self.warnings)} warnings)"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"TypeCheckResult(is_valid={self.is_valid}, errors={len(self.errors)}, warnings={len(self.warnings)})"


def check_gnn_file(file_path: str, strict_mode: bool = False) -> TypeCheckResult:
    """
    Convenience function to check a single GNN file.
    
    Args:
        file_path: Path to the GNN file to check
        strict_mode: Whether to enforce strict type checking rules
    
    Returns:
        TypeCheckResult with the check results
    """
    checker = GNNTypeChecker(strict_mode=strict_mode)
    is_valid, errors, warnings, details = checker.check_file(file_path)
    return TypeCheckResult(is_valid, errors, warnings, details)


def validate_syntax(gnn_content: str, strict_mode: bool = False) -> TypeCheckResult:
    """
    Validate GNN syntax from content string.
    
    Args:
        gnn_content: GNN content as string
        strict_mode: Whether to enforce strict type checking rules
    
    Returns:
        TypeCheckResult with the validation results
    """
    import tempfile
    
    checker = GNNTypeChecker(strict_mode=strict_mode)
    
    # Create temporary file for parsing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(gnn_content)
        temp_path = f.name
    
    try:
        is_valid, errors, warnings, details = checker.check_file(temp_path)
        return TypeCheckResult(is_valid, errors, warnings, details)
    finally:
        # Clean up temporary file
        os.unlink(temp_path)


def estimate_resources(gnn_content: str) -> Dict[str, Any]:
    """
    Estimate computational resources needed for a GNN model.
    
    Args:
        gnn_content: GNN content as string
    
    Returns:
        Dictionary with resource estimates
    """
    import tempfile
    
    # Create temporary file for parsing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(gnn_content)
        temp_path = f.name
    
    try:
        checker = GNNTypeChecker()
        is_valid, errors, warnings, details = checker.check_file(temp_path)
        
        if not is_valid:
            return {
                "error": "Cannot estimate resources for invalid GNN model",
                "errors": errors
            }
        
        # Extract resource estimation from details
        resource_estimates = {
            "total_variables": details.get("total_variables", 0),
            "total_connections": details.get("total_connections", 0),
            "model_complexity": details.get("model_complexity", "unknown"),
            "estimated_memory_mb": details.get("estimated_memory_mb", 0),
            "estimated_computation_time": details.get("estimated_computation_time", "unknown")
        }
        
        return resource_estimates
    finally:
        # Clean up temporary file
        os.unlink(temp_path) 


def run_type_checking(target_dir: Path, output_dir: Path, logger: logging.Logger, recursive: bool = False, strict: bool = False):
    """Run comprehensive type checking on GNN files."""
    try:
        from utils import log_step_start, log_step_success, log_step_warning, log_step_error, performance_tracker
        from pipeline import get_output_dir_for_script
    except ImportError:
        # Fallback logging functions
        def log_step_start(logger, msg): logger.info(f"🚀 {msg}")
        def log_step_success(logger, msg): logger.info(f"✅ {msg}")
        def log_step_warning(logger, msg): logger.warning(f"⚠️ {msg}")
        def log_step_error(logger, msg): logger.error(f"❌ {msg}")
        def get_output_dir_for_script(script, output_dir): return output_dir / "type_check"
        performance_tracker = None
    
    log_step_start(logger, "Running comprehensive type checking on GNN files")
    
    # Use centralized output directory configuration
    type_check_output_dir = get_output_dir_for_script("4_type_checker.py", output_dir)
    type_check_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize type checker
        type_checker = GNNTypeChecker(strict_mode=strict)
        
        # Find GNN files
        pattern = "**/*.md" if recursive else "*.md"
        gnn_files = list(target_dir.glob(pattern))
        
        if not gnn_files:
            log_step_warning(logger, f"No GNN files found in {target_dir}")
            return 0
        
        logger.info(f"Found {len(gnn_files)} GNN files to type check")
        
        # Process each file and collect results
        results = {}
        successful_checks = 0
        failed_checks = 0
        
        with performance_tracker.track_operation("type_check_all_files") if performance_tracker else nullcontext():
            for gnn_file in gnn_files:
                try:
                    logger.debug(f"Type checking file: {gnn_file}")
                    
                    # Create file-specific output directory
                    file_output_dir = type_check_output_dir / gnn_file.stem
                    file_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Run type checking
                    is_valid, errors, warnings, details = type_checker.check_file(str(gnn_file))
                    
                    # Store results for reporting
                    results[str(gnn_file)] = {
                        'is_valid': is_valid,
                        'errors': errors,
                        'warnings': warnings,
                        'details': details
                    }
                    
                    # Generate individual file report
                    file_report_path = file_output_dir / "type_check_result.json"
                    with open(file_report_path, 'w') as f:
                        json.dump({
                            'file_path': str(gnn_file),
                            'is_valid': is_valid,
                            'errors': errors,
                            'warnings': warnings,
                            'details': details,
                            'timestamp': datetime.now().isoformat()
                        }, f, indent=2)
                    
                    if is_valid:
                        successful_checks += 1
                        logger.debug(f"Type checking completed for {gnn_file.name}")
                    else:
                        failed_checks += 1
                        log_step_warning(logger, f"Type checking failed for {gnn_file.name}")
                        
                except Exception as e:
                    failed_checks += 1
                    log_step_error(logger, f"Failed to type check {gnn_file.name}: {e}")
                    # Add error result to results dict
                    results[str(gnn_file)] = {
                        'is_valid': False,
                        'errors': [f"Exception during type checking: {str(e)}"],
                        'warnings': [],
                        'details': {}
                    }
        
        # Generate comprehensive reports
        if results:
            # Generate markdown report
            report_path = type_check_output_dir / "type_check_report.md"
            type_checker.generate_report(results, type_check_output_dir, "type_check_report.md")
            
            # Generate JSON summary
            summary_path = type_check_output_dir / "type_check_summary.json"
            with open(summary_path, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'total_files': len(results),
                    'successful_checks': successful_checks,
                    'failed_checks': failed_checks,
                    'success_rate': (successful_checks / len(results) * 100) if results else 0,
                    'results': results
                }, f, indent=2)
            
            logger.info(f"Generated type check reports:")
            logger.info(f"  - Markdown report: {report_path}")
            logger.info(f"  - JSON summary: {summary_path}")
        
        # Log results summary
        total_files = len(gnn_files)
        if successful_checks == total_files:
            log_step_success(logger, f"All {total_files} files passed type checking")
            return 0
        elif successful_checks > 0:
            log_step_warning(logger, f"Partial success: {successful_checks}/{total_files} files passed type checking")
            return 0
        else:
            log_step_error(logger, "No files passed type checking")
            return 1
        
    except Exception as e:
        log_step_error(logger, f"Type checking failed: {e}")
        return 1

# Context manager for performance tracking fallback
class nullcontext:
    def __enter__(self): return None
    def __exit__(self, *args): return None 