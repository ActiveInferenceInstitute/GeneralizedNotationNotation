"""
Markdown GNN Parser

This module provides parsing for the standard GNN Markdown format.
This is the reference parser implementation that other formats can use as a model.

Author: @docxology
Date: 2025-01-11
License: MIT
"""

import re
import ast
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import logging

from .common import (
    BaseGNNParser, ParseResult, GNNInternalRepresentation, ParseError,
    Variable, Connection, Parameter, Equation, TimeSpecification, OntologyMapping,
    VariableType, DataType, ConnectionType, 
    normalize_variable_name, parse_dimensions, infer_variable_type, parse_connection_operator
)

logger = logging.getLogger(__name__)

class MarkdownGNNParser(BaseGNNParser):
    """
    Parser for GNN Markdown format.
    
    This parser handles the standard GNN format as defined in the specification,
    including all sections and their contents.
    """
    
    def __init__(self):
        super().__init__()
        self.section_parsers = {
            'GNNSection': self._parse_gnn_section,
            'GNNVersionAndFlags': self._parse_version_section,
            'ModelName': self._parse_model_name,
            'ModelAnnotation': self._parse_annotation,
            'StateSpaceBlock': self._parse_state_space,
            'Connections': self._parse_connections,
            'InitialParameterization': self._parse_parameters,
            'Equations': self._parse_equations,
            'Time': self._parse_time,
            'ActInfOntologyAnnotation': self._parse_ontology,
            'ModelParameters': self._parse_model_parameters,
            'Footer': self._parse_footer,
            'Signature': self._parse_signature
        }
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions."""
        return ['.md', '.markdown']
    
    def parse_file(self, file_path: str) -> ParseResult:
        """Parse a GNN Markdown file."""
        self.current_file = file_path
        file_path = Path(file_path)
        
        try:
            content = file_path.read_text(encoding='utf-8')
            return self.parse_string(content)
        except Exception as e:
            raise ParseError(f"Failed to read file {file_path}: {e}")
    
    def parse_string(self, content: str) -> ParseResult:
        """Parse GNN Markdown content from string."""
        try:
            # Create result container
            result = ParseResult(model=self.create_empty_model())
            
            # Split content into sections
            sections = self._split_into_sections(content)
            
            # Parse each section
            for section_name, section_content in sections.items():
                try:
                    self._parse_section(section_name, section_content, result.model)
                except Exception as e:
                    error_msg = f"Error parsing section '{section_name}': {e}"
                    result.add_warning(error_msg)
                    logger.warning(error_msg)
            
            # Post-processing and validation
            self._post_process_model(result.model)
            
            # Set model name if not found
            if not result.model.model_name or result.model.model_name == "Unnamed Model":
                result.model.model_name = self._extract_model_name_fallback(content)
            
            result.model.raw_sections = {name: content for name, content in sections.items()}
            
            return result
            
        except Exception as e:
            result = ParseResult(
                model=self.create_empty_model("Parse Error"),
                success=False
            )
            result.add_error(str(e))
            return result
    
    def _split_into_sections(self, content: str) -> Dict[str, str]:
        """Split content into named sections based on ## headers."""
        sections = {}
        current_section = None
        current_content = []
        
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            self.current_line = i + 1
            
            # Check for section header
            if line.strip().startswith('## '):
                # Save previous section
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = line.strip()[3:].strip()
                current_content = []
            
            elif current_section:
                current_content.append(line)
        
        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def _parse_section(self, section_name: str, content: str, model: GNNInternalRepresentation):
        """Parse a specific section."""
        if section_name in self.section_parsers:
            self.section_parsers[section_name](content, model)
        else:
            # Handle unknown sections as extensions
            model.extensions[section_name] = content
            logger.debug(f"Unknown section '{section_name}' stored as extension")
    
    def _parse_gnn_section(self, content: str, model: GNNInternalRepresentation):
        """Parse the GNNSection."""
        # Usually contains the model type or identifier
        model.extensions['gnn_section'] = content.strip()
    
    def _parse_version_section(self, content: str, model: GNNInternalRepresentation):
        """Parse the GNNVersionAndFlags section."""
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                if 'GNN v' in line or 'v' in line:
                    # Extract version
                    version_match = re.search(r'v?(\d+\.\d+(?:\.\d+)?)', line)
                    if version_match:
                        model.version = version_match.group(1)
                else:
                    # Handle flags
                    model.extensions.setdefault('flags', []).append(line)
    
    def _parse_model_name(self, content: str, model: GNNInternalRepresentation):
        """Parse the ModelName section."""
        model.model_name = content.strip()
    
    def _parse_annotation(self, content: str, model: GNNInternalRepresentation):
        """Parse the ModelAnnotation section."""
        model.annotation = content.strip()
    
    def _parse_state_space(self, content: str, model: GNNInternalRepresentation):
        """Parse the StateSpaceBlock section."""
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                variable = self._parse_variable_definition(line)
                if variable:
                    model.variables.append(variable)
    
    def _parse_variable_definition(self, line: str) -> Optional[Variable]:
        """Parse a single variable definition line."""
        try:
            # Extract comment first
            comment = None
            if '###' in line:
                line, comment = line.split('###', 1)
                comment = comment.strip()
                line = line.strip()
            
            # Parse variable pattern: name[dimensions],type=datatype
            # Examples: s_f0[2], o_m0[3,type=categorical], X[2,3]
            
            # Extract dimensions and name FIRST
            name = None
            dimensions = [1]  # Default dimension
            type_spec = None
            
            if '[' in line and ']' in line:
                # Find the opening bracket
                bracket_start = line.find('[')
                bracket_end = line.find(']')
                
                if bracket_start != -1 and bracket_end != -1:
                    name = line[:bracket_start].strip()
                    dimensions_str = line[bracket_start+1:bracket_end].strip()
                    
                    # Check if there's a type specification in the dimensions string
                    if ',type=' in dimensions_str:
                        dimensions_str, type_spec = dimensions_str.split(',type=', 1)
                        type_spec = type_spec.strip()
                    
                    dimensions = parse_dimensions('[' + dimensions_str + ']')
                    logger.debug(f"Parsed variable '{name}' with dimensions string '{dimensions_str}' -> {dimensions}")
                else:
                    name = line.strip()
                    logger.debug(f"Parsed variable '{name}' with default dimensions {dimensions}")
            else:
                name = line.strip()
                logger.debug(f"Parsed variable '{name}' with default dimensions {dimensions}")
            
            # Determine variable type
            var_type = infer_variable_type(name)
            
            # Determine data type
            if type_spec:
                data_type = self._parse_data_type(type_spec)
            else:
                data_type = DataType.CATEGORICAL  # Default
            
            # Normalize name
            normalized_name = normalize_variable_name(name)
            
            logger.debug(f"Created variable: name='{normalized_name}', dimensions={dimensions}, type={var_type}")
            
            return Variable(
                name=normalized_name,
                var_type=var_type,
                dimensions=dimensions,
                data_type=data_type,
                description=comment
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse variable definition '{line}': {e}")
            return None
    
    def _parse_data_type(self, type_str: str) -> DataType:
        """Parse data type string."""
        type_str = type_str.lower().strip()
        
        type_map = {
            'categorical': DataType.CATEGORICAL,
            'continuous': DataType.CONTINUOUS,
            'binary': DataType.BINARY,
            'integer': DataType.INTEGER,
            'int': DataType.INTEGER,
            'float': DataType.FLOAT,
            'real': DataType.FLOAT,
            'complex': DataType.COMPLEX
        }
        
        return type_map.get(type_str, DataType.CATEGORICAL)
    
    def _parse_connections(self, content: str, model: GNNInternalRepresentation):
        """Parse the Connections section."""
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                connection = self._parse_connection_definition(line)
                if connection:
                    model.connections.append(connection)
    
    def _parse_connection_definition(self, line: str) -> Optional[Connection]:
        """Parse a single connection definition line."""
        try:
            # Extract comment
            comment = None
            if '###' in line:
                line, comment = line.split('###', 1)
                comment = comment.strip()
                line = line.strip()
            
            # Parse connection pattern: source>target, source-target, (s1,s2)>target
            # Find connection operators
            operators = ['<->', '->', '>', '-', '|']
            found_op = None
            op_pos = -1
            
            for op in operators:
                pos = line.find(op)
                if pos != -1:
                    found_op = op
                    op_pos = pos
                    break
            
            if not found_op:
                logger.warning(f"No connection operator found in: {line}")
                return None
            
            # Split source and target
            source_part = line[:op_pos].strip()
            target_part = line[op_pos + len(found_op):].strip()
            
            # Parse variable groups
            source_vars = self._parse_variable_group(source_part)
            target_vars = self._parse_variable_group(target_part)
            
            # Parse connection type
            conn_type = parse_connection_operator(found_op)
            
            return Connection(
                source_variables=source_vars,
                target_variables=target_vars,
                connection_type=conn_type,
                description=comment
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse connection definition '{line}': {e}")
            return None
    
    def _parse_variable_group(self, group_str: str) -> List[str]:
        """Parse a variable group like 'X' or '(X,Y,Z)'."""
        group_str = group_str.strip()
        
        if group_str.startswith('(') and group_str.endswith(')'):
            # Multiple variables
            inner = group_str[1:-1]
            variables = [var.strip() for var in inner.split(',')]
            return [normalize_variable_name(var) for var in variables if var.strip()]
        else:
            # Single variable
            return [normalize_variable_name(group_str)]
    
    def _parse_parameters(self, content: str, model: GNNInternalRepresentation):
        """Parse the InitialParameterization section."""
        lines = content.strip().split('\n')
        
        current_parameter = None
        current_value_lines = []
        
        for line in lines:
            line = line.strip()
            
            if line and not line.startswith('#'):
                # Check if this is a new parameter definition (contains '=')
                if '=' in line and not line.startswith('#'):
                    # Save previous parameter if exists
                    if current_parameter and current_value_lines:
                        value_str = '\n'.join(current_value_lines)
                        parameter = self._parse_parameter_assignment(f"{current_parameter}={value_str}")
                        if parameter:
                            model.parameters.append(parameter)
                    
                    # Start new parameter
                    if '=' in line:
                        name, value = line.split('=', 1)
                        current_parameter = name.strip()
                        current_value_lines = [value.strip()]
                    else:
                        current_parameter = None
                        current_value_lines = []
                else:
                    # Continue current parameter value
                    if current_parameter:
                        current_value_lines.append(line)
            elif line.startswith('#'):
                # Skip comments
                continue
        
        # Handle last parameter
        if current_parameter and current_value_lines:
            value_str = '\n'.join(current_value_lines)
            parameter = self._parse_parameter_assignment(f"{current_parameter}={value_str}")
            if parameter:
                model.parameters.append(parameter)
    
    def _parse_parameter_assignment(self, line: str) -> Optional[Parameter]:
        """Parse a single parameter assignment line."""
        try:
            # Extract comment
            comment = None
            if '###' in line:
                line, comment = line.split('###', 1)
                comment = comment.strip()
                line = line.strip()
            
            # Split by first '='
            if '=' not in line:
                return None
            
            name, value_str = line.split('=', 1)
            name = name.strip()
            value_str = value_str.strip()
            
            # Parse value
            value = self._parse_parameter_value(value_str)
            
            return Parameter(
                name=name,
                value=value,
                description=comment
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse parameter assignment '{line}': {e}")
            return None
    
    def _parse_parameter_value(self, value_str: str) -> Any:
        """Parse a parameter value string."""
        value_str = value_str.strip()
        
        try:
            # Handle GNN matrix format: A={ (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0) }
            if value_str.startswith('{') and value_str.endswith('}'):
                # Remove comments from the string
                lines = value_str.split('\n')
                cleaned_lines = []
                for line in lines:
                    # Remove comments (everything after #)
                    if '#' in line:
                        line = line.split('#')[0]
                    line = line.strip()
                    if line:
                        cleaned_lines.append(line)
                
                # Reconstruct the value string without comments
                cleaned_value = ' '.join(cleaned_lines)
                
                # Handle matrix format: { (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0) }
                if cleaned_value.startswith('{') and cleaned_value.endswith('}'):
                    inner = cleaned_value[1:-1].strip()
                    
                    # Split by commas, but be careful with nested tuples
                    rows = []
                    current_row = ""
                    paren_count = 0
                    
                    for char in inner:
                        if char == '(':
                            paren_count += 1
                        elif char == ')':
                            paren_count -= 1
                        
                        if char == ',' and paren_count == 0:
                            # End of a row
                            if current_row.strip():
                                rows.append(current_row.strip())
                            current_row = ""
                        else:
                            current_row += char
                    
                    # Add the last row
                    if current_row.strip():
                        rows.append(current_row.strip())
                    
                    # Parse each row
                    matrix = []
                    for row in rows:
                        row = row.strip()
                        if row.startswith('(') and row.endswith(')'):
                            # Parse tuple row
                            inner_row = row[1:-1]
                            row_values = [self._parse_single_value(x.strip()) for x in inner_row.split(',') if x.strip()]
                            matrix.append(row_values)
                        elif row.startswith('((') and row.endswith('))'):
                            # Parse nested tuple row (for B matrix)
                            inner_row = row[2:-2]
                            # Split by '),('
                            nested_tuples = inner_row.split('),(')
                            row_values = []
                            for nested_tuple in nested_tuples:
                                nested_tuple = nested_tuple.strip('()')
                                tuple_values = [self._parse_single_value(x.strip()) for x in nested_tuple.split(',') if x.strip()]
                                row_values.append(tuple_values)
                            matrix.append(row_values)
                    
                    return matrix
            
            # Try to evaluate as Python literal
            return ast.literal_eval(value_str)
            
        except (ValueError, SyntaxError):
            # Handle other formats
            
            # Matrix format: {(1,2,3);(4,5,6)}
            if value_str.startswith('{') and value_str.endswith('}') and ';' in value_str:
                inner = value_str[1:-1]
                rows = inner.split(';')
                matrix = []
                for row in rows:
                    row = row.strip('()')
                    if row:
                        row_values = [float(x.strip()) for x in row.split(',') if x.strip()]
                        matrix.append(row_values)
                return matrix
            
            # Tuple format: (1,2,3)
            elif value_str.startswith('(') and value_str.endswith(')'):
                inner = value_str[1:-1]
                values = [self._parse_single_value(x.strip()) for x in inner.split(',') if x.strip()]
                return tuple(values)
            
            # List format: [1,2,3]
            elif value_str.startswith('[') and value_str.endswith(']'):
                inner = value_str[1:-1]
                values = [self._parse_single_value(x.strip()) for x in inner.split(',') if x.strip()]
                return values
            
            # Single value
            else:
                return self._parse_single_value(value_str)
    
    def _parse_single_value(self, value_str: str) -> Any:
        """Parse a single value."""
        value_str = value_str.strip()
        
        # Boolean
        if value_str.lower() in ['true', 'false']:
            return value_str.lower() == 'true'
        
        # Number
        try:
            if '.' in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            pass
        
        # String (remove quotes if present)
        if value_str.startswith('"') and value_str.endswith('"'):
            return value_str[1:-1]
        elif value_str.startswith("'") and value_str.endswith("'"):
            return value_str[1:-1]
        
        # Return as string
        return value_str
    
    def _parse_equations(self, content: str, model: GNNInternalRepresentation):
        """Parse the Equations section."""
        lines = content.strip().split('\n')
        current_equation = []
        current_label = None
        
        for line in lines:
            original_line = line
            line = line.strip()
            
            if line and not line.startswith('#'):
                # Check for equation label
                if ':' in line and not line.startswith('$') and not '=' in line.split(':')[0]:
                    label_part, eq_part = line.split(':', 1)
                    current_label = label_part.strip()
                    if eq_part.strip():
                        current_equation.append(eq_part.strip())
                else:
                    current_equation.append(line)
            
            # Check if equation is complete (empty line or new label)
            if (not line or line.startswith('#')) and current_equation:
                equation_content = ' '.join(current_equation).strip()
                if equation_content:
                    equation = Equation(
                        label=current_label,
                        content=equation_content,
                        format="latex"
                    )
                    model.equations.append(equation)
                
                current_equation = []
                current_label = None
        
        # Handle last equation
        if current_equation:
            equation_content = ' '.join(current_equation).strip()
            if equation_content:
                equation = Equation(
                    label=current_label,
                    content=equation_content,
                    format="latex"
                )
                model.equations.append(equation)
    
    def _parse_time(self, content: str, model: GNNInternalRepresentation):
        """Parse the Time section."""
        lines = content.strip().split('\n')
        
        time_spec = TimeSpecification(time_type="Static")
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                line_lower = line.lower()
                
                if 'static' in line_lower:
                    time_spec.time_type = "Static"
                elif 'dynamic' in line_lower:
                    time_spec.time_type = "Dynamic"
                elif 'discretetime' in line_lower:
                    time_spec.discretization = "DiscreteTime"
                elif 'continuoustime' in line_lower:
                    time_spec.discretization = "ContinuousTime"
                elif 'modeltimehorizon' in line_lower and '=' in line:
                    _, horizon_str = line.split('=', 1)
                    try:
                        time_spec.horizon = int(horizon_str.strip())
                    except ValueError:
                        time_spec.horizon = horizon_str.strip()
        
        model.time_specification = time_spec
    
    def _parse_ontology(self, content: str, model: GNNInternalRepresentation):
        """Parse the ActInfOntologyAnnotation section."""
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                mapping = self._parse_ontology_mapping(line)
                if mapping:
                    model.ontology_mappings.append(mapping)
    
    def _parse_ontology_mapping(self, line: str) -> Optional[OntologyMapping]:
        """Parse a single ontology mapping line."""
        try:
            # Extract comment
            comment = None
            if '###' in line:
                line, comment = line.split('###', 1)
                comment = comment.strip()
                line = line.strip()
            
            # Split by '='
            if '=' not in line:
                return None
            
            var_name, term = line.split('=', 1)
            var_name = var_name.strip()
            term = term.strip()
            
            return OntologyMapping(
                variable_name=normalize_variable_name(var_name),
                ontology_term=term,
                description=comment
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse ontology mapping '{line}': {e}")
            return None
    
    def _parse_model_parameters(self, content: str, model: GNNInternalRepresentation):
        """Parse the ModelParameters section."""
        # Similar to InitialParameterization but for model-level parameters
        self._parse_parameters(content, model)
    
    def _parse_footer(self, content: str, model: GNNInternalRepresentation):
        """Parse the Footer section."""
        model.extensions['footer'] = content.strip()
    
    def _parse_signature(self, content: str, model: GNNInternalRepresentation):
        """Parse the Signature section."""
        model.extensions['signature'] = content.strip()
    
    def _extract_model_name_fallback(self, content: str) -> str:
        """Extract model name from content if not found in ModelName section."""
        lines = content.split('\n')
        
        # Look for title (first # header)
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        
        # Look for filename pattern
        if self.current_file:
            return Path(self.current_file).stem
        
        return "Unnamed Model"
    
    def _post_process_model(self, model: GNNInternalRepresentation):
        """Post-process the parsed model for consistency."""
        # Ensure variable names in connections exist
        all_var_names = {var.name for var in model.variables}
        
        for connection in model.connections:
            # Check source variables
            missing_sources = [var for var in connection.source_variables if var not in all_var_names]
            if missing_sources:
                logger.warning(f"Connection references unknown source variables: {missing_sources}")
            
            # Check target variables
            missing_targets = [var for var in connection.target_variables if var not in all_var_names]
            if missing_targets:
                logger.warning(f"Connection references unknown target variables: {missing_targets}")
        
        # Ensure ontology mappings reference existing variables
        for mapping in model.ontology_mappings:
            if mapping.variable_name not in all_var_names:
                logger.warning(f"Ontology mapping references unknown variable: {mapping.variable_name}")

# Export the parser class
__all__ = ['MarkdownGNNParser'] 