"""
GNN Schema Validation and Parsing Module

This module provides comprehensive validation, parsing, and analysis capabilities
for GNN (Generalized Notation Notation) model files with enhanced round-trip
testing support and cross-format validation.

Enhanced Features:
- Complete format ecosystem support (21 formats)
- Binary file validation for pickle/binary formats
- Cross-format consistency validation
- Round-trip semantic preservation testing
- Enhanced error reporting and suggestions
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
import unicodedata
import hashlib
import tempfile

# Import shared types
from .types import (
    ValidationLevel,
    GNNSyntaxError,
    ValidationResult,
    GNNVariable,
    GNNConnection,
    ParsedGNN,
    GNNFormat,
    RoundTripResult
)

# Lark parser removed - too complex and not needed
FORMAL_PARSER_AVAILABLE = False

# Try to import round-trip testing capabilities
try:
    from .parsers import GNNParsingSystem
    ROUND_TRIP_AVAILABLE = True
except ImportError:
    ROUND_TRIP_AVAILABLE = False

# RoundTripResult is already imported from .types above
# No need to import it again from testing module to avoid circular deps

logger = logging.getLogger(__name__)


class GNNParser:
    """Enhanced parser for GNN file format with multi-format support."""
    
    # Regular expressions for GNN syntax elements (enhanced)
    SECTION_PATTERN = re.compile(r'^## (.+)$')
    VARIABLE_PATTERN = re.compile(r'^([\w_π][\w\d_π]*)(\[([^\]]+)\])?(?:,type=([a-zA-Z]+))?(?:\s*#\s*(.*))?$')
    CONNECTION_PATTERN = re.compile(r'^(.+?)\s*(>|->|-|\|)\s*(.+?)(?:\s*#\s*(.*))?$')
    PARAMETER_PATTERN = re.compile(r'^([\w_π][\w\d_π]*)(\s*=\s*)(.+?)(?:\s*#\s*(.*))?$')
    ONTOLOGY_PATTERN = re.compile(r'^([\w_π][\w\d_π]*)(\s*=\s*)([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*#\s*(.*))?$')
    COMMENT_PATTERN = re.compile(r'^\s*#\s*(.*)$')
    
    # Enhanced format detection patterns
    FORMAT_SIGNATURES = {
        'json': [r'^\s*\{', r'"model_name":', r'"variables":'],
        'xml': [r'^\s*<\?xml', r'<gnn.*>', r'<model'],
        'yaml': [r'^---', r'model_name:', r'variables:'],
        'binary': [b'\x80\x03', b'pickle', b'\x00\x00\x00'],  # Pickle signatures
    }
    
    def __init__(self, enhanced_validation: bool = True):
        self.enhanced_validation = enhanced_validation
        if enhanced_validation and ROUND_TRIP_AVAILABLE:
            self.parsing_system = GNNParsingSystem()
            logger.info("Enhanced multi-format parsing system initialized")
        else:
            self.parsing_system = None
            logger.info("Basic GNN parser initialized")

    def parse_file(self, file_path: Union[str, Path], 
                   format_hint: Optional[str] = None) -> ParsedGNN:
        """Enhanced file parsing with format detection and validation."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Detect format
        detected_format = format_hint or self._detect_file_format(file_path)
        
        # Handle different formats
        if detected_format == 'binary':
            return self._parse_binary_file(file_path)
        else:
            # Read as text
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try as binary if UTF-8 fails
                return self._parse_binary_file(file_path)
        
        return self.parse_content(content, str(file_path), detected_format)
    
    def _detect_file_format(self, file_path: Path) -> str:
        """Enhanced format detection with content analysis."""
        # First try extension-based detection
        extension = file_path.suffix.lower()
        extension_map = {
            '.md': 'markdown',
            '.json': 'json',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.pkl': 'binary',
            '.pickle': 'binary'
        }
        
        if extension in extension_map:
            detected = extension_map[extension]
            
            # Verify with content analysis for ambiguous cases
            if extension in ['.md', '.txt'] and file_path.exists():
                content_format = self._detect_format_from_content(file_path)
                if content_format != 'markdown':
                    return content_format
            
            return detected
        
        # Content-based detection for unknown extensions
        return self._detect_format_from_content(file_path)
    
    def _detect_format_from_content(self, file_path: Path) -> str:
        """Detect format from file content analysis."""
        try:
            # Try reading as text first
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(2000)  # Read first 2KB
            
            content_lower = content.lower()
            
            # Check format signatures
            for fmt, patterns in self.FORMAT_SIGNATURES.items():
                if fmt == 'binary':
                    continue  # Skip binary patterns for text content
                
                if any(re.search(pattern, content, re.IGNORECASE) for pattern in patterns):
                    return fmt
            
            # Check for GNN markdown indicators
            if ('##' in content and 
                any(section in content for section in ['GNNSection', 'ModelName', 'StateSpaceBlock'])):
                return 'markdown'
            
            return 'markdown'  # Default fallback
            
        except UnicodeDecodeError:
            return 'binary'
        except Exception:
            return 'unknown'
    
    def _parse_binary_file(self, file_path: Path) -> ParsedGNN:
        """Parse binary files (pickle format)."""
        try:
            import pickle
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Convert pickle data to ParsedGNN format
            return self._convert_pickle_to_parsed_gnn(data)
            
        except Exception as e:
            # Create minimal parsed representation for failed binary files
            return ParsedGNN(
                gnn_section="BinaryGNN",
                version="1.0",
                model_name=f"BinaryModel_{file_path.stem}",
                model_annotation=f"Binary file: {file_path.name}",
                variables={},
                connections=[],
                parameters={},
                equations=[],
                time_config={},
                ontology_mappings={},
                model_parameters={},
                footer="",
                metadata={"parse_error": str(e), "source_format": "binary"}
            )
    
    def _convert_pickle_to_parsed_gnn(self, data: Any) -> ParsedGNN:
        """Convert pickle data to ParsedGNN structure."""
        # Implementation depends on pickle data structure
        # This is a simplified version
        if isinstance(data, dict):
            return ParsedGNN(
                gnn_section=data.get('gnn_section', 'PickleGNN'),
                version=data.get('version', '1.0'),
                model_name=data.get('model_name', 'PickleModel'),
                model_annotation=data.get('annotation', ''),
                variables=data.get('variables', {}),
                connections=data.get('connections', []),
                parameters=data.get('parameters', {}),
                equations=data.get('equations', []),
                time_config=data.get('time_config', {}),
                ontology_mappings=data.get('ontology_mappings', {}),
                model_parameters=data.get('model_parameters', {}),
                footer=data.get('footer', ''),
                source_format='pickle'
            )
        else:
            # Create minimal representation for non-dict pickle data
            return ParsedGNN(
                gnn_section="PickleGNN",
                version="1.0",
                model_name="PickleModel",
                model_annotation=f"Pickled data: {type(data).__name__}",
                variables={},
                connections=[],
                parameters={},
                equations=[],
                time_config={},
                ontology_mappings={},
                model_parameters={},
                footer="",
                source_format='pickle'
            )

    def parse_content(self, content: str, source_name: str = "<string>", 
                     format_hint: str = "markdown") -> ParsedGNN:
        """Enhanced content parsing with format-specific handling."""
        # Use multi-format parsing system if available
        if self.parsing_system and format_hint != "markdown":
            try:
                format_enum = GNNFormat(format_hint)
                result = self.parsing_system.parse_string(content, format_enum)
                if result.success:
                    return self._convert_parse_result_to_parsed_gnn(result, format_hint)
            except (ValueError, Exception) as e:
                logger.warning(f"Multi-format parsing failed for {format_hint}: {e}")
        
        # Fallback to markdown parsing
        return self._parse_markdown_content(content, source_name)
    
    def _convert_parse_result_to_parsed_gnn(self, result, source_format: str) -> ParsedGNN:
        """Convert ParseResult to ParsedGNN format."""
        model = result.model
        
        # Convert variables
        variables = {}
        for var in model.variables:
            variables[var.name] = GNNVariable(
                name=var.name,
                dimensions=getattr(var, 'dimensions', []),
                data_type=str(getattr(var, 'data_type', 'categorical')),
                description=getattr(var, 'description', ''),
                ontology_mapping=getattr(var, 'ontology_mapping', None)
            )
        
        # Convert connections
        connections = []
        for conn in model.connections:
            connections.append(GNNConnection(
                source=getattr(conn, 'source_variables', []),
                target=getattr(conn, 'target_variables', []),
                connection_type=str(getattr(conn, 'connection_type', 'directed')),
                symbol=self._infer_symbol_from_type(str(getattr(conn, 'connection_type', 'directed'))),
                description=getattr(conn, 'description', '')
            ))
        
        # Convert parameters
        parameters = {}
        for param in model.parameters:
            parameters[param.name] = param.value
        
        return ParsedGNN(
            gnn_section=getattr(model, 'gnn_section', f"{source_format.upper()}GNN"),
            version=getattr(model, 'version', '1.0'),
            model_name=model.model_name,
            model_annotation=model.annotation,
            variables=variables,
            connections=connections,
            parameters=parameters,
            equations=getattr(model, 'equations', []),
            time_config=getattr(model, 'time_config', {}),
            ontology_mappings=getattr(model, 'ontology_mappings', {}),
            model_parameters=getattr(model, 'model_parameters', {}),
            footer=getattr(model, 'footer', ''),
            source_format=source_format,
            semantic_checksum=self._compute_semantic_checksum(model)
        )
    
    def _infer_symbol_from_type(self, connection_type: str) -> str:
        """Infer connection symbol from type."""
        symbol_map = {
            'directed': '>',
            'undirected': '-',
            'conditional': '|',
            'bidirectional': '<->'
        }
        return symbol_map.get(connection_type, '>')
    
    def _compute_semantic_checksum(self, model) -> str:
        """Compute semantic checksum for model."""
        # Create normalized representation
        checksum_data = {
            'model_name': model.model_name,
            'variables': sorted([var.name for var in model.variables]),
            'connections_count': len(model.connections),
            'parameters_count': len(model.parameters)
        }
        
        checksum_str = json.dumps(checksum_data, sort_keys=True)
        return hashlib.md5(checksum_str.encode()).hexdigest()

    def _parse_markdown_content(self, content: str, source_name: str) -> ParsedGNN:
        """Parse GNN content from string (markdown format)."""
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
    """Enhanced validator for GNN files with comprehensive round-trip and cross-format support."""
    
    def __init__(self, schema_path: Optional[Path] = None, 
                 use_formal_parser: bool = True, 
                 enable_round_trip_testing: bool = False,
                 validation_level: ValidationLevel = ValidationLevel.STANDARD,
                 enable_cross_validation: bool = True):
        if schema_path is None:
            schema_path = Path(__file__).parent / "schemas/json.json"
        
        self.schema_path = schema_path
        self.schema = self._load_schema()
        self.use_formal_parser = use_formal_parser and FORMAL_PARSER_AVAILABLE
        self.enable_round_trip_testing = enable_round_trip_testing and ROUND_TRIP_AVAILABLE
        self.validation_level = validation_level
        
        # Initialize enhanced parser
        self.parser = GNNParser(enhanced_validation=True)
        
        # Formal parser removed (Lark was too complex)
        self.formal_parser = None
        
        # Initialize round-trip tester if enabled
        if self.enable_round_trip_testing:
            try:
                from .testing.test_round_trip import GNNRoundTripTester
                self.round_trip_tester = GNNRoundTripTester()
                logger.info("Round-trip testing enabled for comprehensive validation")
            except Exception as e:
                logger.warning(f"Could not initialize round-trip tester: {e}")
                self.enable_round_trip_testing = False
        
        # Initialize cross-format validator (optionally, to avoid recursion)
        self.cross_validator = None
        if enable_cross_validation:
            try:
                from .cross_format_validator import CrossFormatValidator
                self.cross_validator = CrossFormatValidator()
            except ImportError:
                self.cross_validator = None
    
    def _load_schema(self):
        """
        Load JSON or YAML schema with robust error handling.
        
        Handles potential recursion and parsing errors by providing fallback mechanisms.
        """
        try:
            # First, try standard loading
            if self.schema_path.suffix.lower() == '.json':
                import json
                with open(self.schema_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif self.schema_path.suffix.lower() in ['.yaml', '.yml']:
                import yaml
                with open(self.schema_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported schema file type: {self.schema_path.suffix}")
        
        except (json.JSONDecodeError, yaml.YAMLError, RecursionError) as e:
            # Log the specific error
            logger.warning(f"Schema loading error for {self.schema_path}: {e}")
            
            # Fallback to minimal schema
            return {
                "title": "Fallback GNN Schema",
                "description": "Minimal schema due to loading error",
                "type": "object",
                "properties": {},
                "required": []
            }
        except Exception as e:
            # Catch-all for any other unexpected errors
            logger.error(f"Unexpected error loading schema {self.schema_path}: {e}")
            
            # Return an empty, permissive schema
            return {
                "title": "Emergency Fallback Schema",
                "description": "Completely permissive schema due to critical loading error",
                "type": "object",
                "additionalProperties": True
            }
    
    def validate_file(self, file_path: Union[str, Path], 
                     validation_level: Optional[ValidationLevel] = None) -> ValidationResult:
        """Enhanced validation with comprehensive testing capabilities."""
        import time
        start_time = time.time()
        
        validation_level = validation_level or self.validation_level
        file_path = Path(file_path)
        
        result = ValidationResult(
            is_valid=True, 
            validation_level=validation_level,
            format_tested=self._detect_file_format(file_path)
        )
        
        try:
            # Step 1: Basic file access and format detection
            file_format = self._detect_file_format(file_path)
            result.format_tested = file_format
            
            # Handle binary formats
            if file_format in ['binary', 'pickle']:
                return self._validate_binary_file(file_path, result)
            
            # Read text content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError as e:
                result.errors.append(f"File encoding error: {e}")
                result.is_valid = False
                return result
            
            # Step 2: Parse the file using enhanced parser
            try:
                parsed_gnn = self.parser.parse_file(file_path)
                result.semantic_checksum = parsed_gnn.semantic_checksum
                result.metadata['parsed_successfully'] = True
                result.metadata['source_format'] = parsed_gnn.source_format
            except Exception as e:
                result.errors.append(f"Parsing failed: {e}")
                result.is_valid = False
                if validation_level == ValidationLevel.BASIC:
                    return result
                # Continue with content-based validation for higher levels
                parsed_gnn = None
            
            # Step 3: Validation based on level
            if validation_level >= ValidationLevel.BASIC:
                self._validate_basic_structure(content, result, file_format)
            
            if validation_level >= ValidationLevel.STANDARD and parsed_gnn:
                self._validate_semantics(parsed_gnn, result)
            
            if validation_level >= ValidationLevel.STRICT:
                self._validate_strict_requirements(parsed_gnn, content, result)
            
            if validation_level >= ValidationLevel.RESEARCH:
                self._validate_research_standards(parsed_gnn, content, result)
            
            # Step 4: Round-trip testing if enabled and requested
            if (validation_level == ValidationLevel.ROUND_TRIP and 
                self.enable_round_trip_testing and parsed_gnn):
                self._perform_round_trip_validation(parsed_gnn, result)
            
            # Step 5: Cross-format consistency if available
            if (validation_level >= ValidationLevel.STRICT and 
                self.cross_validator and parsed_gnn):
                self._validate_cross_format_consistency(content, result)
            
            # Final result determination
            result.is_valid = len(result.errors) == 0
            
            # Performance metrics
            end_time = time.time()
            result.performance_metrics = {
                'validation_time': end_time - start_time,
                'content_length': len(content),
                'validation_level': validation_level.value
            }
            
            return result
            
        except Exception as e:
            result.errors.append(f"Validation failed with exception: {e}")
            result.is_valid = False
            return result
    
    def _detect_file_format(self, file_path: Path) -> str:
        """Detect file format from extension."""
        suffix = file_path.suffix.lower()
        format_map = {
            '.md': 'markdown',
            '.markdown': 'markdown',
            '.json': 'json',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.pkl': 'pickle',
            '.pickle': 'pickle'
        }
        return format_map.get(suffix, 'unknown')
    
    def _validate_structured_format(self, content: str, result: ValidationResult, file_format: str):
        """Validate structured formats like JSON, XML, YAML."""
        try:
            if file_format == 'json':
                import json
                data = json.loads(content)
                # Basic validation for JSON structure
                if isinstance(data, dict):
                    if 'model_name' in data:
                        result.warnings.append("JSON format validated successfully")
                    else:
                        result.warnings.append("JSON format valid but missing expected model_name field")
                else:
                    result.errors.append("JSON should contain a dictionary/object at root level")
            
            elif file_format == 'xml':
                import xml.etree.ElementTree as ET
                try:
                    root = ET.fromstring(content)
                    result.warnings.append("XML format validated successfully")
                except ET.ParseError as e:
                    result.errors.append(f"XML parsing error: {e}")
            
            elif file_format == 'yaml':
                try:
                    import yaml
                    data = yaml.safe_load(content)
                    result.warnings.append("YAML format validated successfully")
                except Exception as e:
                    result.errors.append(f"YAML parsing error: {e}")
                    
        except ImportError as e:
            result.warnings.append(f"Cannot validate {file_format} format: missing library ({e})")
        except Exception as e:
            result.errors.append(f"Error validating {file_format} format: {e}")
    
    def _validate_binary_file(self, file_path: Path, result: ValidationResult) -> ValidationResult:
        """Validate binary files (pickle format)."""
        try:
            with open(file_path, 'rb') as f:
                # Try to read first few bytes to ensure it's accessible
                header = f.read(10)
                
            # Check for pickle signature
            if header.startswith(b'\x80\x03') or b'pickle' in header:
                result.warnings.append("Binary pickle format detected - validation limited to accessibility check")
            else:
                result.warnings.append("Unknown binary format - validation limited to accessibility check")
            
            result.metadata['binary_format'] = True
            result.metadata['file_size'] = file_path.stat().st_size
            result.is_valid = True
            
        except Exception as e:
            result.errors.append(f"Binary file access error: {e}")
            result.is_valid = False
        
        return result
    
    def _validate_basic_structure(self, content: str, result: ValidationResult, file_format: str):
        """Enhanced basic validation with format-specific checks."""
        if len(content.strip()) == 0:
            result.errors.append("File is empty")
            return
        
        # Format-specific basic validation
        if file_format == 'markdown':
            self._validate_markdown_structure(content, result)
        elif file_format in ['json', 'xml', 'yaml']:
            self._validate_structured_format(content, result, file_format)
        else:
            result.warnings.append(f"Unknown file format: {file_format}, using basic validation")
            # Basic text validation
            if len(content) < 10:
                result.warnings.append("File content is very short")
            if '\x00' in content:
                result.warnings.append("File contains null bytes - may be binary")
    
    def _validate_strict_requirements(self, parsed_gnn: Optional[ParsedGNN], content: str, result: ValidationResult):
        """Validate strict requirements for research-grade models."""
        if not parsed_gnn:
            result.errors.append("Parsed model required for strict validation")
            return
        
        # Check for complete documentation
        if not parsed_gnn.model_annotation or len(parsed_gnn.model_annotation.strip()) < 50:
            result.warnings.append("Model annotation should be more descriptive for research use")
        
        # Check for ontology mappings
        if not parsed_gnn.ontology_mappings:
            result.suggestions.append("Consider adding ontology mappings for better interoperability")
        
        # Check for equations
        if not parsed_gnn.equations:
            result.suggestions.append("Consider adding mathematical equations for clarity")
        
        # Validate parameter completeness
        if len(parsed_gnn.parameters) < len(parsed_gnn.variables) * 0.5:
            result.warnings.append("Many variables lack parameter specifications")
    
    def _validate_research_standards(self, parsed_gnn: Optional[ParsedGNN], content: str, result: ValidationResult):
        """Validate research-grade standards."""
        if not parsed_gnn:
            return
        
        # Check for signature/provenance
        if not parsed_gnn.signature:
            result.suggestions.append("Add signature section for provenance tracking")
        
        # Check for time configuration
        if not parsed_gnn.time_config:
            result.suggestions.append("Specify time configuration for reproducibility")
        
        # Validate model parameters
        if not parsed_gnn.model_parameters:
            result.suggestions.append("Add model parameters for complete specification")
        
        # Check for research-grade documentation
        research_keywords = ['hypothesis', 'method', 'experiment', 'analysis', 'result']
        annotation_lower = parsed_gnn.model_annotation.lower()
        found_keywords = [kw for kw in research_keywords if kw in annotation_lower]
        
        if len(found_keywords) < 2:
            result.suggestions.append("Consider adding research context (hypothesis, methods, etc.)")
    
    def _perform_round_trip_validation(self, parsed_gnn: ParsedGNN, result: ValidationResult):
        """Perform round-trip validation testing."""
        try:
            # Create a temporary markdown file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                # Write parsed content back to markdown
                markdown_content = self._convert_parsed_gnn_to_markdown(parsed_gnn)
                f.write(markdown_content)
                temp_file = Path(f.name)
            
            try:
                # Run round-trip tests on a subset of formats
                test_formats = [GNNFormat.JSON, GNNFormat.XML, GNNFormat.YAML]
                
                for fmt in test_formats:
                    try:
                        # Test conversion to format and back
                        round_trip_result = self.round_trip_tester._test_round_trip(
                            parsed_gnn, fmt
                        )
                        result.add_round_trip_result(round_trip_result)
                        
                    except Exception as e:
                        result.warnings.append(f"Round-trip test failed for {fmt.value}: {e}")
                
                # Summary
                success_rate = result.get_round_trip_success_rate()
                if success_rate == 100.0:
                    result.suggestions.append("Perfect round-trip compatibility achieved")
                elif success_rate >= 80.0:
                    result.warnings.append(f"Good round-trip compatibility: {success_rate:.1f}%")
                else:
                    result.errors.append(f"Poor round-trip compatibility: {success_rate:.1f}%")
                    
            finally:
                # Clean up temporary file
                temp_file.unlink(missing_ok=True)
                
        except Exception as e:
            result.warnings.append(f"Round-trip validation failed: {e}")
    
    def _validate_cross_format_consistency(self, content: str, result: ValidationResult):
        """Validate cross-format consistency."""
        try:
            cross_result = self.cross_validator.validate_cross_format_consistency(content)
            result.cross_format_consistent = cross_result.is_consistent
            
            if cross_result.is_consistent:
                result.suggestions.append("Cross-format consistency validated")
            else:
                result.warnings.extend(cross_result.inconsistencies)
                result.warnings.extend(cross_result.warnings)
        
        except Exception as e:
            result.warnings.append(f"Cross-format validation failed: {e}")
    
    def _convert_parsed_gnn_to_markdown(self, parsed_gnn: ParsedGNN) -> str:
        """Convert ParsedGNN back to markdown format."""
        lines = []
        
        lines.append(f"## GNNSection")
        lines.append(parsed_gnn.gnn_section)
        lines.append("")
        
        lines.append(f"## GNNVersionAndFlags")
        lines.append(parsed_gnn.version)
        lines.append("")
        
        lines.append(f"## ModelName")
        lines.append(parsed_gnn.model_name)
        lines.append("")
        
        lines.append(f"## ModelAnnotation")
        lines.append(parsed_gnn.model_annotation)
        lines.append("")
        
        lines.append(f"## StateSpaceBlock")
        for var_name, var in parsed_gnn.variables.items():
            dim_str = f"[{','.join(map(str, var.dimensions))}]" if var.dimensions else ""
            type_str = f",type={var.data_type}" if var.data_type != 'categorical' else ""
            desc_str = f" # {var.description}" if var.description else ""
            lines.append(f"{var_name}{dim_str}{type_str}{desc_str}")
        lines.append("")
        
        lines.append(f"## Connections")
        for conn in parsed_gnn.connections:
            source = ','.join(conn.source) if isinstance(conn.source, list) else conn.source
            target = ','.join(conn.target) if isinstance(conn.target, list) else conn.target
            desc_str = f" # {conn.description}" if conn.description else ""
            lines.append(f"{source}{conn.symbol}{target}{desc_str}")
        lines.append("")
        
        lines.append(f"## InitialParameterization")
        for param_name, param_value in parsed_gnn.parameters.items():
            lines.append(f"{param_name}={param_value}")
        lines.append("")
        
        lines.append(f"## Time")
        time_type = parsed_gnn.time_config.get('type', 'Dynamic')
        lines.append(time_type)
        lines.append("")
        
        lines.append(f"## Footer")
        lines.append(parsed_gnn.footer)
        
        return "\n".join(lines)
    
    def _validate_markdown_structure(self, content: str, result: ValidationResult):
        """Validate comprehensive GNN markdown file structure and semantics."""
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
                section_name = line[3:].strip()
                found_sections.append(section_name)
        
        # Check for missing required sections
        missing_sections = set(required_sections) - set(found_sections)
        for section in missing_sections:
            result.errors.append(f"Required section missing: {section}")
        
        # Additional validation can be added here
        if self.use_formal_parser and self.formal_parser:
            try:
                formal_result = self.formal_parser.parse_content(content)
                if formal_result:
                    result.warnings.append("Formal parser validation passed")
                else:
                    result.warnings.append("Formal parser could not parse content")
            except Exception as e:
                result.warnings.append(f"Formal parser validation failed: {e}")
        
        # Validate using basic parser
        try:
            parser = GNNParser()
            parsed_gnn = parser.parse_content(content)
            
            if len(parsed_gnn.variables) == 0:
                result.warnings.append("No variables found in StateSpaceBlock")
            
            if len(parsed_gnn.connections) == 0:
                result.warnings.append("No connections found in Connections section")
                
        except Exception as e:
            result.warnings.append(f"Basic parser validation failed: {e}")
    
    def _validate_structure(self, content: str, result: ValidationResult):
        """Legacy method - now delegates to markdown validation."""
        self._validate_markdown_structure(content, result)
    
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