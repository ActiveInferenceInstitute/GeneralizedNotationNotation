"""
GNN Schema Validation and Parsing Module

This module provides comprehensive validation, parsing, and analysis capabilities
for GNN (Generalized Notation Notation) model files according to the formal schema.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import unicodedata

# Try to import formal parser for enhanced validation
try:
    from .parsers.lark_parser import GNNFormalParser, ParsedGNNFormal, LARK_AVAILABLE
    FORMAL_PARSER_AVAILABLE = LARK_AVAILABLE
except ImportError:
    FORMAL_PARSER_AVAILABLE = False

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    RESEARCH = "research"


class GNNSyntaxError(Exception):
    """Exception raised for GNN syntax errors."""
    def __init__(self, message: str, line: int = None, column: int = None):
        self.message = message
        self.line = line
        self.column = column
        super().__init__(self.format_message())
    
    def format_message(self) -> str:
        if self.line is not None:
            if self.column is not None:
                return f"Line {self.line}, Column {self.column}: {self.message}"
            return f"Line {self.line}: {self.message}"
        return self.message


@dataclass
class ValidationResult:
    """Results from GNN validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class GNNVariable:
    """Represents a GNN variable definition."""
    name: str
    dimensions: List[Union[int, str]]
    data_type: str
    description: Optional[str] = None
    constraints: Optional[Dict[str, Any]] = None
    line_number: Optional[int] = None


@dataclass
class GNNConnection:
    """Represents a connection between GNN variables."""
    source: Union[str, List[str]]
    target: Union[str, List[str]]
    connection_type: str  # 'directed', 'undirected', 'conditional'
    symbol: str  # '>', '-', '->', '|'
    description: Optional[str] = None
    line_number: Optional[int] = None


@dataclass
class ParsedGNN:
    """Complete parsed representation of a GNN file."""
    gnn_section: str
    version: str
    model_name: str
    model_annotation: str
    variables: Dict[str, GNNVariable]
    connections: List[GNNConnection]
    parameters: Dict[str, Any]
    equations: List[Dict[str, str]]
    time_config: Dict[str, Any]
    ontology_mappings: Dict[str, str]
    model_parameters: Dict[str, Any]
    footer: str
    signature: Optional[Dict[str, str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class GNNParser:
    """Parser for GNN file format."""
    
    # Regular expressions for GNN syntax elements
    SECTION_PATTERN = re.compile(r'^## (.+)$')
    VARIABLE_PATTERN = re.compile(r'^([\w_π][\w\d_π]*)(\[([^\]]+)\])?(?:,type=([a-zA-Z]+))?(?:\s*#\s*(.*))?$')
    CONNECTION_PATTERN = re.compile(r'^(.+?)\s*(>|->|-|\|)\s*(.+?)(?:\s*#\s*(.*))?$')
    PARAMETER_PATTERN = re.compile(r'^([\w_π][\w\d_π]*)(\s*=\s*)(.+?)(?:\s*#\s*(.*))?$')
    ONTOLOGY_PATTERN = re.compile(r'^([\w_π][\w\d_π]*)(\s*=\s*)([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*#\s*(.*))?$')
    COMMENT_PATTERN = re.compile(r'^\s*#\s*(.*)$')  # Explicitly handle single hashtag comments
    
    def __init__(self):
        self.current_section = None
        self.line_number = 0
        
    def parse_file(self, file_path: Union[str, Path]) -> ParsedGNN:
        """Parse a GNN file and return structured representation."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"GNN file not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        return self.parse_content(content, str(file_path))
    
    def parse_content(self, content: str, source_name: str = "<string>") -> ParsedGNN:
        """Parse GNN content from string."""
        lines = content.split('\n')
        self.line_number = 0
        self.current_section = None
        
        # Initialize parsed structure
        parsed = ParsedGNN(
            gnn_section="",
            version="",
            model_name="",
            model_annotation="",
            variables={},
            connections=[],
            parameters={},
            equations=[],
            time_config={},
            ontology_mappings={},
            model_parameters={},
            footer=""
        )
        
        current_content = []
        
        for line in lines:
            self.line_number += 1
            line = line.rstrip()
            
            # Check for section headers
            section_match = self.SECTION_PATTERN.match(line)
            if section_match:
                # Process previous section content
                if self.current_section and current_content:
                    self._process_section_content(parsed, self.current_section, current_content)
                
                # Start new section
                self.current_section = section_match.group(1)
                current_content = []
                continue
            
            # Skip empty lines at section boundaries
            if not line.strip() and not current_content:
                continue
                
            current_content.append(line)
        
        # Process final section
        if self.current_section and current_content:
            self._process_section_content(parsed, self.current_section, current_content)
        
        return parsed
    
    def _process_section_content(self, parsed: ParsedGNN, section: str, content: List[str]):
        """Process content for a specific section."""
        content_text = '\n'.join(content).strip()
        
        if section == "GNNSection":
            parsed.gnn_section = content_text
        elif section == "GNNVersionAndFlags":
            parsed.version = content_text
        elif section == "ModelName":
            parsed.model_name = content_text
        elif section == "ModelAnnotation":
            parsed.model_annotation = content_text
        elif section == "StateSpaceBlock":
            self._parse_state_space_block(parsed, content)
        elif section == "Connections":
            self._parse_connections(parsed, content)
        elif section == "InitialParameterization":
            self._parse_parameters(parsed, content, "parameters")
        elif section == "Equations":
            self._parse_equations(parsed, content)
        elif section == "Time":
            self._parse_time_config(parsed, content)
        elif section == "ActInfOntologyAnnotation":
            self._parse_ontology_mappings(parsed, content)
        elif section == "ModelParameters":
            self._parse_parameters(parsed, content, "model_parameters")
        elif section == "Footer":
            parsed.footer = content_text
        elif section == "Signature":
            parsed.signature = self._parse_signature(content_text)
    
    def _parse_state_space_block(self, parsed: ParsedGNN, content: List[str]):
        """Parse StateSpaceBlock section."""
        for i, line in enumerate(content):
            line = line.strip()
            if not line or self.COMMENT_PATTERN.match(line):
                continue
                
            match = self.VARIABLE_PATTERN.match(line)
            if match:
                name = match.group(1)
                dims_str = match.group(3)
                data_type = match.group(4) or "float"
                description = match.group(5)
                
                # Parse dimensions
                dimensions = []
                for dim in dims_str.split(','):
                    dim = dim.strip()
                    if dim.isdigit():
                        dimensions.append(int(dim))
                    else:
                        dimensions.append(dim)
                
                variable = GNNVariable(
                    name=name,
                    dimensions=dimensions,
                    data_type=data_type,
                    description=description,
                    line_number=self.line_number - len(content) + i
                )
                
                parsed.variables[name] = variable
            else:
                logger.warning(f"Could not parse variable definition: {line}")
    
    def _parse_connections(self, parsed: ParsedGNN, content: List[str]):
        """Parse Connections section."""
        for i, line in enumerate(content):
            line = line.strip()
            if not line or self.COMMENT_PATTERN.match(line):
                continue
                
            match = self.CONNECTION_PATTERN.match(line)
            if match:
                source_str = match.group(1).strip()
                symbol = match.group(2)
                target_str = match.group(3).strip()
                description = match.group(4)
                
                # Parse variable groups
                source = self._parse_variable_group(source_str)
                target = self._parse_variable_group(target_str)
                
                # Determine connection type
                connection_type = self._get_connection_type(symbol)
                
                connection = GNNConnection(
                    source=source,
                    target=target,
                    connection_type=connection_type,
                    symbol=symbol,
                    description=description,
                    line_number=self.line_number - len(content) + i
                )
                
                parsed.connections.append(connection)
            else:
                logger.warning(f"Could not parse connection: {line}")
    
    def _parse_variable_group(self, group_str: str) -> Union[str, List[str]]:
        """Parse a variable group (single variable or parenthesized list)."""
        group_str = group_str.strip()
        
        if group_str.startswith('(') and group_str.endswith(')'):
            # Parse comma-separated list
            inner = group_str[1:-1]
            variables = [v.strip() for v in inner.split(',')]
            return variables if len(variables) > 1 else variables[0]
        else:
            return group_str
    
    def _get_connection_type(self, symbol: str) -> str:
        """Determine connection type from symbol."""
        if symbol in ['>', '->']:
            return 'directed'
        elif symbol == '-':
            return 'undirected'
        elif symbol == '|':
            return 'conditional'
        else:
            return 'unknown'
    
    def _parse_parameters(self, parsed: ParsedGNN, content: List[str], param_type: str):
        """Parse parameter sections."""
        target_dict = getattr(parsed, param_type)
        
        for line in content:
            line = line.strip()
            if not line or self.COMMENT_PATTERN.match(line):
                continue
                
            match = self.PARAMETER_PATTERN.match(line)
            if match:
                name = match.group(1)
                value_str = match.group(3)
                description = match.group(4)
                
                # Try to parse the value
                try:
                    value = self._parse_parameter_value(value_str)
                    target_dict[name] = value
                except Exception as e:
                    logger.warning(f"Could not parse parameter value for {name}: {e}")
                    target_dict[name] = value_str
    
    def _parse_parameter_value(self, value_str: str) -> Any:
        """Parse a parameter value string."""
        value_str = value_str.strip()
        
        # Try JSON parsing first
        try:
            return json.loads(value_str)
        except json.JSONDecodeError:
            pass
        
        # Try specific GNN formats
        if value_str.startswith('{') and value_str.endswith('}'):
            # Matrix or tuple format
            return self._parse_matrix_or_tuple(value_str)
        elif value_str.startswith('(') and value_str.endswith(')'):
            # Tuple format
            return self._parse_tuple(value_str)
        elif value_str.replace('.', '').replace('-', '').isdigit():
            # Numeric value
            return float(value_str) if '.' in value_str else int(value_str)
        elif value_str.lower() in ['true', 'false']:
            # Boolean value
            return value_str.lower() == 'true'
        else:
            # String value
            return value_str
    
    def _parse_matrix_or_tuple(self, value_str: str) -> Any:
        """Parse matrix or tuple notation with full Active Inference support."""
        value_str = value_str.strip()
        
        # Handle nested tuple/matrix structures like {((0.8,0.1,0.1),(0.1,0.8,0.1))}
        if value_str.startswith('{') and value_str.endswith('}'):
            inner = value_str[1:-1].strip()
            
            # Check for matrix structure with nested tuples
            if inner.startswith('(') and inner.count('(') > 1:
                return self._parse_nested_matrix(inner)
            else:
                return self._parse_tuple(inner)
        
        # Handle simple tuple like (0.5,0.5)
        elif value_str.startswith('(') and value_str.endswith(')'):
            return self._parse_tuple(value_str)
        
        # Handle list/array notation
        elif value_str.startswith('[') and value_str.endswith(']'):
            return self._parse_array(value_str)
        
        return value_str
    
    def _parse_nested_matrix(self, inner: str) -> List[List[float]]:
        """Parse nested matrix structure like ((0.8,0.1),(0.1,0.8))."""
        matrix = []
        depth = 0
        current_tuple = ""
        
        for char in inner:
            if char == '(':
                depth += 1
                current_tuple += char
            elif char == ')':
                depth -= 1
                current_tuple += char
                if depth == 0:
                    # Parse complete tuple
                    tuple_values = self._parse_tuple(current_tuple)
                    if isinstance(tuple_values, (list, tuple)):
                        matrix.append(list(tuple_values))
                    current_tuple = ""
            elif depth > 0:
                current_tuple += char
            elif char == ',' and depth == 0:
                continue  # Skip commas between tuples
        
        return matrix
    
    def _parse_array(self, value_str: str) -> List[Any]:
        """Parse array notation [1,2,3] or [[1,2],[3,4]]."""
        import ast
        try:
            return ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            # Fallback parsing
            inner = value_str[1:-1].strip()
            if not inner:
                return []
            
            elements = []
            depth = 0
            current = ""
            
            for char in inner:
                if char in '[(':
                    depth += 1
                elif char in '])':
                    depth -= 1
                
                if char == ',' and depth == 0:
                    elements.append(self._parse_scalar_value(current.strip()))
                    current = ""
                else:
                    current += char
            
            if current.strip():
                elements.append(self._parse_scalar_value(current.strip()))
            
            return elements
    
    def _parse_scalar_value(self, value_str: str) -> Union[float, int, bool, str]:
        """Parse a scalar value with proper type conversion."""
        value_str = value_str.strip()
        
        # Boolean values
        if value_str.lower() in ['true', 'false']:
            return value_str.lower() == 'true'
        
        # Numeric values
        try:
            if '.' in value_str or 'e' in value_str.lower():
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            pass
        
        # String values
        if value_str.startswith('"') and value_str.endswith('"'):
            return value_str[1:-1]
        elif value_str.startswith("'") and value_str.endswith("'"):
            return value_str[1:-1]
        
        return value_str
    
    def _parse_tuple(self, value_str: str) -> Tuple[Any, ...]:
        """Parse tuple notation with proper type conversion."""
        if value_str.startswith('(') and value_str.endswith(')'):
            inner = value_str[1:-1].strip()
        else:
            inner = value_str.strip()
        
        if not inner:
            return tuple()
        
        # Split on commas, handling nested structures
        elements = []
        depth = 0
        current = ""
        
        for char in inner:
            if char in '([{':
                depth += 1
                current += char
            elif char in ')]}':
                depth -= 1
                current += char
            elif char == ',' and depth == 0:
                elements.append(self._parse_scalar_value(current.strip()))
                current = ""
            else:
                current += char
        
        if current.strip():
            elements.append(self._parse_scalar_value(current.strip()))
        
        return tuple(elements)
    
    def _parse_equations(self, parsed: ParsedGNN, content: List[str]):
        """Parse Equations section."""
        current_equation = {}
        
        for line in content:
            line = line.strip()
            if not line:
                if current_equation:
                    parsed.equations.append(current_equation)
                    current_equation = {}
                continue
                
            if self.COMMENT_PATTERN.match(line):
                comment_match = self.COMMENT_PATTERN.match(line)
                if current_equation:
                    current_equation['description'] = comment_match.group(1)
            else:
                if not current_equation:
                    current_equation = {'latex': line}
                else:
                    current_equation['latex'] += ' ' + line
        
        if current_equation:
            parsed.equations.append(current_equation)
    
    def _parse_time_config(self, parsed: ParsedGNN, content: List[str]):
        """Parse Time section."""
        for line in content:
            line = line.strip()
            if not line or self.COMMENT_PATTERN.match(line):
                continue
                
            if '=' in line:
                key, value = line.split('=', 1)
                parsed.time_config[key.strip()] = value.strip()
            else:
                # Simple time type specification
                parsed.time_config['type'] = line
    
    def _parse_ontology_mappings(self, parsed: ParsedGNN, content: List[str]):
        """Parse ActInfOntologyAnnotation section."""
        for line in content:
            line = line.strip()
            if not line or self.COMMENT_PATTERN.match(line):
                continue
                
            match = self.ONTOLOGY_PATTERN.match(line)
            if match:
                variable = match.group(1)
                ontology_term = match.group(3)
                parsed.ontology_mappings[variable] = ontology_term
    
    def _parse_signature(self, content: str) -> Dict[str, str]:
        """Parse Signature section."""
        signature = {}
        for line in content.split('\n'):
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                signature[key.strip()] = value.strip()
        return signature


class GNNValidator:
    """Validator for GNN files against the formal schema with enhanced parsing."""
    
    def __init__(self, schema_path: Optional[Path] = None, use_formal_parser: bool = True):
        if schema_path is None:
            schema_path = Path(__file__).parent / "schemas/json.json"
        
        self.schema_path = schema_path
        self.schema = self._load_schema()
        self.use_formal_parser = use_formal_parser and FORMAL_PARSER_AVAILABLE
        
        # Initialize formal parser if available
        if self.use_formal_parser:
            try:
                self.formal_parser = GNNFormalParser()
                logger.info("Formal Lark parser initialized for enhanced validation")
            except Exception as e:
                logger.warning(f"Could not initialize formal parser: {e}")
                self.use_formal_parser = False
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load the JSON schema."""
        try:
            with open(self.schema_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Could not load schema from {self.schema_path}: {e}")
            return {}
    
    def validate_file(self, file_path: Union[str, Path]) -> ValidationResult:
        """Validate a GNN file."""
        result = ValidationResult(is_valid=True)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic structure validation
            self._validate_structure(content, result)
            
            result.is_valid = len(result.errors) == 0
            return result
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"File error: {str(e)}"],
                metadata={"source": str(file_path)}
            )
    
    def _validate_structure(self, content: str, result: ValidationResult):
        """Validate comprehensive GNN file structure and semantics."""
        lines = content.split('\n')
        
        # Check for required sections
        required_sections = [
            'GNNSection', 'GNNVersionAndFlags', 'ModelName', 
            'ModelAnnotation', 'StateSpaceBlock', 'Connections',
            'InitialParameterization', 'Time', 'Footer'
        ]
        
        found_sections = []
        for line in lines:
            if line.startswith('## '):
                section = line[3:].strip()
                found_sections.append(section)
        
        for section in required_sections:
            if section not in found_sections:
                result.errors.append(f"Required section missing: {section}")
        
        # Parse and validate semantic consistency
        try:
            # Use formal parser if available for enhanced validation
            if self.use_formal_parser:
                formal_result = self.formal_parser.parse_content(content)
                if formal_result:
                    # Convert formal parser result to standard format
                    parsed = self._convert_formal_to_standard(formal_result)
                    result.metadata['formal_parser_used'] = True
                    result.metadata['parse_tree_available'] = True
                else:
                    # Fallback to standard parser
                    parser = GNNParser()
                    parsed = parser.parse_content(content)
                    result.warnings.append("Formal parser failed, using fallback parser")
            else:
                parser = GNNParser()
                parsed = parser.parse_content(content)
                result.metadata['formal_parser_used'] = False
            
            self._validate_semantics(parsed, result)
            
            # Additional formal syntax validation if available
            if self.use_formal_parser:
                syntax_valid, syntax_errors = self.formal_parser.validate_syntax(content)
                if not syntax_valid:
                    result.errors.extend([f"Syntax error: {error}" for error in syntax_errors])
                
        except Exception as e:
            result.errors.append(f"Parsing error: {str(e)}")
    
    def _convert_formal_to_standard(self, formal_parsed: 'ParsedGNNFormal') -> ParsedGNN:
        """Convert formal parser result to standard ParsedGNN format."""
        # Convert formal parser variables to standard format
        variables = {}
        for var_name, var_info in formal_parsed.variables.items():
            variables[var_name] = GNNVariable(
                name=var_name,
                dimensions=var_info.get('dimensions', []),
                data_type=var_info.get('data_type', 'float'),
                description=var_info.get('description'),
                line_number=None
            )
        
        # Convert connections
        connections = []
        for conn_info in formal_parsed.connections:
            connections.append(GNNConnection(
                source=conn_info.get('source', ''),
                target=conn_info.get('target', ''),
                connection_type=conn_info.get('connection_type', 'directed'),
                symbol=conn_info.get('operator', '>'),
                description=conn_info.get('description'),
                line_number=None
            ))
        
        return ParsedGNN(
            gnn_section=formal_parsed.gnn_section,
            version=formal_parsed.version,
            model_name=formal_parsed.model_name,
            model_annotation=formal_parsed.model_annotation,
            variables=variables,
            connections=connections,
            parameters=formal_parsed.parameters,
            equations=formal_parsed.equations,
            time_config=formal_parsed.time_config,
            ontology_mappings=formal_parsed.ontology_mappings,
            model_parameters=formal_parsed.model_parameters,
            footer=formal_parsed.footer,
            signature=formal_parsed.signature,
            metadata=formal_parsed.metadata
        )
    
    def _validate_semantics(self, parsed: ParsedGNN, result: ValidationResult):
        """Validate semantic consistency of parsed GNN model."""
        # Validate variable references in connections
        variable_names = set(parsed.variables.keys())
        
        for connection in parsed.connections:
            # Check source variables
            source_vars = connection.source if isinstance(connection.source, list) else [connection.source]
            for var in source_vars:
                if var not in variable_names and not self._is_valid_variable_reference(var):
                    result.errors.append(f"Undefined variable in connection source: {var}")
            
            # Check target variables  
            target_vars = connection.target if isinstance(connection.target, list) else [connection.target]
            for var in target_vars:
                if var not in variable_names and not self._is_valid_variable_reference(var):
                    result.errors.append(f"Undefined variable in connection target: {var}")
        
        # Validate Active Inference conventions
        self._validate_active_inference_conventions(parsed, result)
        
        # Validate mathematical consistency
        self._validate_mathematical_consistency(parsed, result)
    
    def _is_valid_variable_reference(self, var: str) -> bool:
        """Check if variable reference follows GNN conventions."""
        # Handle complex variable references like (A,B) or time-indexed variables
        return (var.startswith('(') and var.endswith(')')) or '=' in var
    
    def _validate_active_inference_conventions(self, parsed: ParsedGNN, result: ValidationResult):
        """Validate Active Inference naming and structure conventions."""
        # Check for proper A, B, C, D matrix naming
        ai_matrices = {'A': [], 'B': [], 'C': [], 'D': []}
        
        for var_name, var in parsed.variables.items():
            if var_name.startswith('A_m'):
                ai_matrices['A'].append(var_name)
            elif var_name.startswith('B_f'):
                ai_matrices['B'].append(var_name)
            elif var_name.startswith('C_m'):
                ai_matrices['C'].append(var_name)
            elif var_name.startswith('D_f'):
                ai_matrices['D'].append(var_name)
        
        # Validate matrix dimension consistency
        if ai_matrices['A'] and ai_matrices['D']:
            result.metadata['active_inference_matrices'] = ai_matrices
        
        # Check for proper state/observation variable naming
        state_vars = [name for name in parsed.variables.keys() if name.startswith('s_f')]
        obs_vars = [name for name in parsed.variables.keys() if name.startswith('o_m')]
        
        if state_vars:
            result.metadata['state_variables'] = state_vars
        if obs_vars:
            result.metadata['observation_variables'] = obs_vars
    
    def _validate_mathematical_consistency(self, parsed: ParsedGNN, result: ValidationResult):
        """Validate mathematical consistency of parameters and dimensions."""
        # Check for matrix dimension consistency with variable definitions
        for param_name, param_value in parsed.parameters.items():
            if param_name in parsed.variables:
                var = parsed.variables[param_name]
                
                # Validate matrix dimensions match variable dimensions
                if isinstance(param_value, list) and isinstance(param_value[0], list):
                    # Matrix parameter
                    matrix_rows = len(param_value)
                    matrix_cols = len(param_value[0]) if param_value else 0
                    
                    expected_dims = var.dimensions
                    if len(expected_dims) >= 2:
                        if matrix_rows != expected_dims[0] or matrix_cols != expected_dims[1]:
                            result.warnings.append(
                                f"Matrix {param_name} dimensions {matrix_rows}x{matrix_cols} "
                                f"don't match variable definition {expected_dims}"
                            )
                
                # Check for stochasticity in probability matrices
                if param_name.startswith(('A_', 'B_', 'D_')) and isinstance(param_value, list):
                    if not self._check_stochasticity(param_value):
                        result.warnings.append(
                            f"Matrix {param_name} may not be properly stochastic (rows should sum to 1)"
                        )
    
    def _check_stochasticity(self, matrix_data: Any, tolerance: float = 1e-6) -> bool:
        """Check if matrix rows sum to 1 (stochastic constraint)."""
        try:
            if isinstance(matrix_data, list) and matrix_data:
                if isinstance(matrix_data[0], (list, tuple)):
                    # 2D matrix
                    for row in matrix_data:
                        if isinstance(row, (list, tuple)):
                            row_sum = sum(float(x) for x in row if isinstance(x, (int, float)))
                            if abs(row_sum - 1.0) > tolerance:
                                return False
                    return True
                else:
                    # 1D vector
                    total_sum = sum(float(x) for x in matrix_data if isinstance(x, (int, float)))
                    return abs(total_sum - 1.0) <= tolerance
        except (TypeError, ValueError):
            pass
        return True  # Conservative: assume valid if can't check


def validate_gnn_file(file_path: Union[str, Path]) -> ValidationResult:
    """Convenience function to validate a GNN file."""
    validator = GNNValidator()
    return validator.validate_file(file_path)


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        result = validate_gnn_file(file_path)
        
        print(f"Validation Result: {'VALID' if result.is_valid else 'INVALID'}")
        
        if result.errors:
            print("\nErrors:")
            for error in result.errors:
                print(f"  - {error}")
        
        if result.warnings:
            print("\nWarnings:")
            for warning in result.warnings:
                print(f"  - {warning}")
        
        if result.suggestions:
            print("\nSuggestions:")
            for suggestion in result.suggestions:
                print(f"  - {suggestion}")
    else:
        print("Usage: python schema_validator.py <gnn_file>") 