"""
Formal Lark Parser for GNN (Generalized Notation Notation)

This module implements a formal parser using the Lark parsing library
based on the EBNF grammar specification for GNN files.
"""

import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field

try:
    from lark import Lark, Transformer, v_args, Tree, Token
    from lark.exceptions import LarkError, ParseError, LexError
    LARK_AVAILABLE = True
except ImportError:
    LARK_AVAILABLE = False
    # Provide stub classes for graceful degradation
    class Lark:
        def __init__(self, *args, **kwargs): pass
        def parse(self, *args, **kwargs): raise NotImplementedError("Lark not available")
    
    class Transformer:
        def __init__(self, *args, **kwargs): pass
    
    class Tree:
        def __init__(self, *args, **kwargs): pass
        def pretty(self): return "Lark not available"
    
    class Token:
        def __init__(self, *args, **kwargs): pass
    
    def v_args(*args, **kwargs):
        def decorator(func): return func
        return decorator

logger = logging.getLogger(__name__)


@dataclass
class ParsedGNNFormal:
    """Formally parsed GNN representation using Lark parser."""
    parse_tree: Any  # Lark Tree object
    gnn_section: str = ""
    version: str = ""
    model_name: str = ""
    model_annotation: str = ""
    variables: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    connections: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    equations: List[Dict[str, str]] = field(default_factory=list)
    time_config: Dict[str, Any] = field(default_factory=dict)
    ontology_mappings: Dict[str, str] = field(default_factory=dict)
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    footer: str = ""
    signature: Optional[Dict[str, str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class GNNTransformer(Transformer):
    """Lark Transformer to convert parse tree to structured data."""
    
    @v_args(inline=True)
    def gnn_file(self, *sections):
        """Transform complete GNN file."""
        result = ParsedGNNFormal(parse_tree=None)
        
        for section in sections:
            if hasattr(section, 'data'):
                section_name = section.data
                if section_name == 'gnn_header':
                    result.gnn_section = self._extract_text(section)
                elif section_name == 'gnn_version_section':
                    result.version = self._extract_text(section)
                elif section_name == 'model_name_section':
                    result.model_name = self._extract_text(section)
                elif section_name == 'model_annotation_section':
                    result.model_annotation = self._extract_text(section)
                elif section_name == 'state_space_section':
                    result.variables = self._extract_variables(section)
                elif section_name == 'connections_section':
                    result.connections = self._extract_connections(section)
                elif section_name == 'parameterization_section':
                    result.parameters = self._extract_parameters(section)
                elif section_name == 'equations_section':
                    result.equations = self._extract_equations(section)
                elif section_name == 'time_section':
                    result.time_config = self._extract_time(section)
                elif section_name == 'ontology_section':
                    result.ontology_mappings = self._extract_ontology(section)
                elif section_name == 'model_parameters_section':
                    result.model_parameters = self._extract_model_parameters(section)
                elif section_name == 'footer_section':
                    result.footer = self._extract_text(section)
                elif section_name == 'signature_section':
                    result.signature = self._extract_signature(section)
        
        return result
    
    def _extract_text(self, section) -> str:
        """Extract text content from a section."""
        if hasattr(section, 'children'):
            return ' '.join(str(child) for child in section.children if isinstance(child, Token))
        return str(section)
    
    def _extract_variables(self, section) -> Dict[str, Dict[str, Any]]:
        """Extract variable definitions from state space section."""
        variables = {}
        
        if hasattr(section, 'children'):
            for child in section.children:
                if hasattr(child, 'data') and child.data == 'variable_definition':
                    var_info = self._parse_variable_definition(child)
                    if var_info:
                        variables[var_info['name']] = var_info
        
        return variables
    
    def _parse_variable_definition(self, var_def) -> Optional[Dict[str, Any]]:
        """Parse a single variable definition."""
        if not hasattr(var_def, 'children') or len(var_def.children) < 3:
            return None
        
        name = str(var_def.children[0])
        dimensions = self._parse_dimensions(var_def.children[1])
        data_type = str(var_def.children[2]) if len(var_def.children) > 2 else "float"
        
        return {
            'name': name,
            'dimensions': dimensions,
            'data_type': data_type,
            'description': None
        }
    
    def _parse_dimensions(self, dim_spec) -> List[Union[int, str]]:
        """Parse dimension specification."""
        dimensions = []
        
        if hasattr(dim_spec, 'children'):
            for child in dim_spec.children:
                dim_str = str(child)
                try:
                    dimensions.append(int(dim_str))
                except ValueError:
                    dimensions.append(dim_str)
        
        return dimensions
    
    def _extract_connections(self, section) -> List[Dict[str, Any]]:
        """Extract connection definitions."""
        connections = []
        
        if hasattr(section, 'children'):
            for child in section.children:
                if hasattr(child, 'data') and child.data == 'connection_definition':
                    conn_info = self._parse_connection_definition(child)
                    if conn_info:
                        connections.append(conn_info)
        
        return connections
    
    def _parse_connection_definition(self, conn_def) -> Optional[Dict[str, Any]]:
        """Parse a single connection definition."""
        if not hasattr(conn_def, 'children') or len(conn_def.children) < 3:
            return None
        
        source = str(conn_def.children[0])
        operator = str(conn_def.children[1])
        target = str(conn_def.children[2])
        
        return {
            'source': source,
            'target': target,
            'operator': operator,
            'connection_type': self._get_connection_type(operator),
            'description': None
        }
    
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
    
    def _extract_parameters(self, section) -> Dict[str, Any]:
        """Extract parameter assignments."""
        parameters = {}
        
        if hasattr(section, 'children'):
            for child in section.children:
                if hasattr(child, 'data') and child.data == 'parameter_assignment':
                    param_info = self._parse_parameter_assignment(child)
                    if param_info:
                        parameters[param_info['name']] = param_info['value']
        
        return parameters
    
    def _parse_parameter_assignment(self, param_assign) -> Optional[Dict[str, Any]]:
        """Parse a single parameter assignment."""
        if not hasattr(param_assign, 'children') or len(param_assign.children) < 2:
            return None
        
        name = str(param_assign.children[0])
        value = self._parse_parameter_value(param_assign.children[1])
        
        return {
            'name': name,
            'value': value
        }
    
    def _parse_parameter_value(self, value_node) -> Any:
        """Parse parameter value from parse tree."""
        if not hasattr(value_node, 'children'):
            return str(value_node)
        
        # Handle different value types based on tree structure
        if hasattr(value_node, 'data'):
            if value_node.data == 'scalar_value':
                return self._parse_scalar_value(value_node)
            elif value_node.data == 'matrix_value':
                return self._parse_matrix_value(value_node)
            elif value_node.data == 'tuple_value':
                return self._parse_tuple_value(value_node)
        
        return str(value_node)
    
    def _parse_scalar_value(self, scalar_node) -> Union[float, int, bool, str]:
        """Parse scalar value."""
        if hasattr(scalar_node, 'children'):
            value_str = str(scalar_node.children[0])
            
            # Try to parse as number
            try:
                if '.' in value_str:
                    return float(value_str)
                else:
                    return int(value_str)
            except ValueError:
                pass
            
            # Try to parse as boolean
            if value_str.lower() in ['true', 'false']:
                return value_str.lower() == 'true'
            
            # Return as string
            return value_str
        
        return str(scalar_node)
    
    def _parse_matrix_value(self, matrix_node) -> List[List[float]]:
        """Parse matrix value."""
        matrix = []
        
        if hasattr(matrix_node, 'children'):
            for child in matrix_node.children:
                if hasattr(child, 'data') and child.data == 'matrix_row':
                    row = self._parse_matrix_row(child)
                    matrix.append(row)
        
        return matrix
    
    def _parse_matrix_row(self, row_node) -> List[float]:
        """Parse matrix row."""
        row = []
        
        if hasattr(row_node, 'children'):
            for child in row_node.children:
                try:
                    row.append(float(str(child)))
                except ValueError:
                    row.append(0.0)  # Default value for unparseable entries
        
        return row
    
    def _parse_tuple_value(self, tuple_node) -> List[Any]:
        """Parse tuple value."""
        tuple_values = []
        
        if hasattr(tuple_node, 'children'):
            for child in tuple_node.children:
                value = self._parse_parameter_value(child)
                tuple_values.append(value)
        
        return tuple_values
    
    def _extract_equations(self, section) -> List[Dict[str, str]]:
        """Extract equation definitions."""
        equations = []
        
        if hasattr(section, 'children'):
            for child in section.children:
                if hasattr(child, 'data') and child.data == 'equation_definition':
                    eq_info = self._parse_equation_definition(child)
                    if eq_info:
                        equations.append(eq_info)
        
        return equations
    
    def _parse_equation_definition(self, eq_def) -> Optional[Dict[str, str]]:
        """Parse equation definition."""
        if not hasattr(eq_def, 'children'):
            return None
        
        latex = ""
        label = None
        
        for child in eq_def.children:
            if hasattr(child, 'data'):
                if child.data == 'equation_label':
                    label = str(child.children[0]) if child.children else None
                elif child.data == 'latex_equation':
                    latex = str(child.children[0]) if child.children else ""
            else:
                latex = str(child)
        
        return {
            'label': label,
            'latex': latex,
            'description': None
        }
    
    def _extract_time(self, section) -> Dict[str, Any]:
        """Extract time configuration."""
        time_config = {}
        
        if hasattr(section, 'children'):
            for child in section.children:
                if hasattr(child, 'data') and child.data == 'time_specification':
                    time_config = self._parse_time_specification(child)
        
        return time_config
    
    def _parse_time_specification(self, time_spec) -> Dict[str, Any]:
        """Parse time specification."""
        config = {'type': 'Static'}
        
        if hasattr(time_spec, 'children'):
            for child in time_spec.children:
                if str(child) in ['Static', 'Dynamic']:
                    config['type'] = str(child)
                elif 'DiscreteTime' in str(child):
                    parts = str(child).split('=')
                    if len(parts) == 2:
                        config['discrete_time'] = parts[1]
                elif 'ContinuousTime' in str(child):
                    parts = str(child).split('=')
                    if len(parts) == 2:
                        config['continuous_time'] = parts[1]
                elif 'ModelTimeHorizon' in str(child):
                    parts = str(child).split('=')
                    if len(parts) == 2:
                        config['time_horizon'] = parts[1]
        
        return config
    
    def _extract_ontology(self, section) -> Dict[str, str]:
        """Extract ontology mappings."""
        mappings = {}
        
        if hasattr(section, 'children'):
            for child in section.children:
                if hasattr(child, 'data') and child.data == 'ontology_mapping':
                    mapping = self._parse_ontology_mapping(child)
                    if mapping:
                        mappings.update(mapping)
        
        return mappings
    
    def _parse_ontology_mapping(self, mapping_node) -> Optional[Dict[str, str]]:
        """Parse ontology mapping."""
        if not hasattr(mapping_node, 'children') or len(mapping_node.children) < 2:
            return None
        
        variable = str(mapping_node.children[0])
        term = str(mapping_node.children[1])
        
        return {variable: term}
    
    def _extract_model_parameters(self, section) -> Dict[str, Any]:
        """Extract model parameters."""
        parameters = {}
        
        if hasattr(section, 'children'):
            for child in section.children:
                if hasattr(child, 'data') and child.data == 'parameter_definition':
                    param_info = self._parse_parameter_assignment(child)
                    if param_info:
                        parameters[param_info['name']] = param_info['value']
        
        return parameters
    
    def _extract_signature(self, section) -> Dict[str, str]:
        """Extract signature information."""
        signature = {}
        
        if hasattr(section, 'children'):
            for child in section.children:
                line = str(child)
                if ':' in line:
                    key, value = line.split(':', 1)
                    signature[key.strip()] = value.strip()
        
        return signature


class GNNFormalParser:
    """Parser for the formal GNN specification."""
    
    def __init__(self, grammar_path: Optional[Path] = None):
        """Initialize the parser with optional explicit grammar path."""
        self.grammar_path = grammar_path
        self.parser = None
        self.transformer = GNNTransformer()
        
        # For more robust Unicode support
        self.unicode_support = True
        
        self._initialize_parser()
    
    def _initialize_parser(self):
        """Initialize the Lark parser with GNN grammar."""
        try:
            if self.grammar_path and self.grammar_path.exists():
                with open(self.grammar_path, 'r') as f:
                    grammar_content = f.read()
                
                # Convert EBNF to Lark format (simplified conversion)
                lark_grammar = self._convert_ebnf_to_lark(grammar_content)
                
                self.parser = Lark(lark_grammar, parser='lalr', transformer=self.transformer)
                logger.info(f"Formal parser initialized with grammar from {self.grammar_path}")
            else:
                logger.error(f"Grammar file not found: {self.grammar_path}")
                self._create_fallback_parser()
        except Exception as e:
            logger.error(f"Failed to initialize parser: {e}")
            self._create_fallback_parser()
    
    def _convert_ebnf_to_lark(self, ebnf_content: str) -> str:
        """Convert EBNF grammar to Lark format."""
        # This is a simplified conversion - full EBNF to Lark conversion
        # would require more sophisticated parsing
        
        lark_grammar = '''
        start: gnn_file
        
        gnn_file: gnn_header gnn_version_section model_name_section model_annotation_section 
                  state_space_section connections_section parameterization_section 
                  equations_section? time_section ontology_section? model_parameters_section?
                  footer_section signature_section?
        
        gnn_header: "## GNNSection" NEWLINE identifier NEWLINE
        gnn_version_section: "## GNNVersionAndFlags" NEWLINE version_spec NEWLINE
        model_name_section: "## ModelName" NEWLINE model_name NEWLINE
        model_annotation_section: "## ModelAnnotation" NEWLINE annotation_text NEWLINE
        state_space_section: "## StateSpaceBlock" NEWLINE (variable_definition | comment)* NEWLINE
        connections_section: "## Connections" NEWLINE (connection_definition | comment)* NEWLINE
        parameterization_section: "## InitialParameterization" NEWLINE (parameter_assignment | comment)* NEWLINE
        equations_section: "## Equations" NEWLINE (equation_definition | comment)* NEWLINE
        time_section: "## Time" NEWLINE time_specification NEWLINE
        ontology_section: "## ActInfOntologyAnnotation" NEWLINE (ontology_mapping | comment)* NEWLINE
        model_parameters_section: "## ModelParameters" NEWLINE (parameter_definition | comment)* NEWLINE
        footer_section: "## Footer" NEWLINE footer_text NEWLINE
        signature_section: "## Signature" NEWLINE signature_text NEWLINE
        
        version_spec: "GNN v" NUMBER ("." NUMBER)? flags?
        flags: " " flag ("," flag)*
        flag: "strict_validation" | "experimental_features" | "extended_syntax"
        
        variable_definition: variable_name dimension_spec type_spec? comment? NEWLINE
        variable_name: identifier
        dimension_spec: "[" dimension ("," dimension)* "]"
        dimension: NUMBER | identifier
        type_spec: ",type=" data_type
        data_type: "float" | "int" | "bool" | "string" | "categorical"
        
        connection_definition: variable_group connection_operator variable_group comment? NEWLINE
        variable_group: variable_ref | "(" variable_ref ("," variable_ref)* ")"
        variable_ref: identifier subscript? time_index?
        connection_operator: ">" | "-" | "->" | "|"
        subscript: "_" (identifier | NUMBER)
        time_index: "[" (identifier | NUMBER) "]"
        
        parameter_assignment: parameter_name "=" parameter_value comment? NEWLINE
        parameter_name: identifier
        parameter_value: scalar_value | matrix_value | tuple_value | reference_value
        scalar_value: NUMBER | BOOLEAN | ESCAPED_STRING
        matrix_value: "{" matrix_row ("," matrix_row)* "}"
        matrix_row: "(" NUMBER ("," NUMBER)* ")"
        tuple_value: "(" parameter_value ("," parameter_value)* ")"
        reference_value: identifier
        
        equation_definition: (equation_label ":")? latex_equation comment? NEWLINE
        equation_label: identifier
        latex_equation: /[^\\n#]+/
        
        time_specification: time_type | time_assignment+
        time_type: "Static" | "Dynamic"
        time_assignment: TIME_KEY "=" identifier
        
        ontology_mapping: identifier "=" identifier comment? NEWLINE
        
        parameter_definition: identifier ":" /[^\\n#]+/ comment? NEWLINE
        
        model_name: /[^\\n]+/
        annotation_text: /[^#]+/
        footer_text: /[^#]+/
        signature_text: /[^#]+/
        
        // Support both single (#) and triple (###) hashtag comments
        comment: /#+ *[^\\n]*/
        
        // Support Unicode characters like π in identifiers
        identifier: /[a-zA-Z_\\u00A0-\\uFFFF][a-zA-Z0-9_\\u00A0-\\uFFFF]*/
        
        TIME_KEY: "DiscreteTime" | "ContinuousTime" | "ModelTimeHorizon"
        
        BOOLEAN: "true" | "false" | "True" | "False"
        
        %import common.SIGNED_NUMBER -> NUMBER
        %import common.ESCAPED_STRING
        %import common.WS
        %import common.NEWLINE
        %ignore WS
        // Explicitly ignore single hashtag comments
        %ignore /[#][^\\n]*/
        '''
        
        return lark_grammar
    
    def _create_fallback_parser(self):
        """Create a minimal fallback parser."""
        fallback_grammar = '''
        start: text*
        text: /[^\\n]*/
        
        %import common.NEWLINE
        %ignore NEWLINE
        '''
        
        if LARK_AVAILABLE:
            self.parser = Lark(fallback_grammar, parser='lalr')
            logger.info("Using fallback parser")
    
    def parse_file(self, file_path: Union[str, Path]) -> Optional[ParsedGNNFormal]:
        """Parse a GNN file using formal parser."""
        if not LARK_AVAILABLE:
            logger.error("Lark parsing library not available")
            return None
        
        if not self.parser:
            logger.error("Parser not initialized")
            return None
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return self.parse_content(content, str(file_path))
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None
    
    def parse_content(self, content: str, source_name: str = "<string>") -> Optional[ParsedGNNFormal]:
        """Parse GNN content using formal parser."""
        if not LARK_AVAILABLE:
            logger.error("Lark parsing library not available")
            return None
        
        if not self.parser:
            logger.error("Parser not initialized")
            return None
        
        try:
            parse_tree = self.parser.parse(content)
            
            # If using transformer, it's already applied
            if isinstance(parse_tree, ParsedGNNFormal):
                parse_tree.parse_tree = parse_tree  # Store reference to self
                parse_tree.metadata['source'] = source_name
                parse_tree.metadata['parser'] = 'lark_formal'
                return parse_tree
            
            # Manual transformation if needed
            result = ParsedGNNFormal(parse_tree=parse_tree)
            result.metadata['source'] = source_name
            result.metadata['parser'] = 'lark_formal'
            
            return result
            
        except (ParseError, LexError) as e:
            logger.error(f"Parse error in {source_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing {source_name}: {e}")
            return None
    
    def validate_syntax(self, content: str) -> Tuple[bool, List[str]]:
        """Validate GNN syntax using formal parser."""
        if not LARK_AVAILABLE:
            return False, ["Lark parsing library not available"]
        
        if not self.parser:
            return False, ["Parser not initialized"]
        
        try:
            self.parser.parse(content)
            return True, []
        except (ParseError, LexError) as e:
            return False, [str(e)]
        except Exception as e:
            return False, [f"Unexpected parsing error: {e}"]
    
    def visualize_parse_tree(self, content: str) -> Optional[str]:
        """Generate a visual representation of the parse tree."""
        if not LARK_AVAILABLE:
            return "Lark parsing library not available"
        
        if not self.parser:
            return "Parser not initialized"
        
        try:
            # Create parser without transformer for tree visualization
            grammar = self.parser.options.transformer
            parser_no_transform = Lark(self.parser.grammar, parser='lalr')
            
            parse_tree = parser_no_transform.parse(content)
            return parse_tree.pretty()
        except Exception as e:
            return f"Error generating parse tree: {e}"


# Convenience functions for external use
def parse_gnn_formal(file_path: Union[str, Path]) -> Optional[ParsedGNNFormal]:
    """Parse a GNN file using formal parser."""
    parser = GNNFormalParser()
    return parser.parse_file(file_path)


def validate_gnn_syntax_formal(content: str) -> Tuple[bool, List[str]]:
    """Validate GNN syntax using formal parser."""
    parser = GNNFormalParser()
    return parser.validate_syntax(content)


def get_parse_tree_visualization(content: str) -> Optional[str]:
    """Get visual representation of parse tree."""
    parser = GNNFormalParser()
    return parser.visualize_parse_tree(content)


# Export key components
__all__ = [
    'GNNFormalParser',
    'ParsedGNNFormal', 
    'GNNTransformer',
    'parse_gnn_formal',
    'validate_gnn_syntax_formal',
    'get_parse_tree_visualization',
    'LARK_AVAILABLE'
] 