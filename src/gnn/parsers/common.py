"""
Common Infrastructure for GNN Parsers

This module provides the shared infrastructure used by all GNN parsers,
including the internal representation, AST nodes, and base classes.

Author: @docxology
Date: 2025-01-11
License: MIT
"""

from typing import Dict, Any, List, Optional, Union, TypeVar, Generic, Protocol
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import uuid
from datetime import datetime

# Type variables for generic types
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

# ================================
# EXCEPTIONS AND ERRORS
# ================================

class ParseError(Exception):
    """Base exception for parsing errors."""
    
    def __init__(self, message: str, line: Optional[int] = None, 
                 column: Optional[int] = None, source: Optional[str] = None):
        self.message = message
        self.line = line
        self.column = column
        self.source = source
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format the error message with location information."""
        msg = self.message
        if self.source:
            msg = f"{self.source}: {msg}"
        if self.line is not None:
            msg = f"Line {self.line}: {msg}"
            if self.column is not None:
                msg = f"Line {self.line}, Column {self.column}: {msg}"
        return msg

class ValidationError(Exception):
    """Exception for validation errors."""
    pass

class ValidationWarning(Warning):
    """Warning for validation issues."""
    pass

class ConversionError(Exception):
    """Exception for format conversion errors."""
    pass

# ================================
# ENUMS AND CONSTANTS
# ================================

class GNNFormat(Enum):
    """Enumeration of all supported GNN formats."""
    MARKDOWN = "markdown"
    SCALA = "scala"
    LEAN = "lean"
    COQ = "coq"
    PYTHON = "python"
    BNF = "bnf"
    EBNF = "ebnf"
    ISABELLE = "isabelle"
    MAXIMA = "maxima"
    XML = "xml"
    PNML = "pnml"
    JSON = "json"
    PROTOBUF = "protobuf"
    YAML = "yaml"
    XSD = "xsd"
    ASN1 = "asn1"
    ALLOY = "alloy"
    Z_NOTATION = "z_notation"
    TLA_PLUS = "tla_plus"
    AGDA = "agda"
    HASKELL = "haskell"
    PICKLE = "pickle"

class VariableType(Enum):
    """Types of variables in Active Inference."""
    HIDDEN_STATE = "hidden_state"
    OBSERVATION = "observation"
    ACTION = "action"
    POLICY = "policy"
    LIKELIHOOD_MATRIX = "likelihood_matrix"
    TRANSITION_MATRIX = "transition_matrix"
    PREFERENCE_VECTOR = "preference_vector"
    PRIOR_VECTOR = "prior_vector"

class DataType(Enum):
    """Data types for GNN variables."""
    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"
    BINARY = "binary"
    INTEGER = "integer"
    FLOAT = "float"
    COMPLEX = "complex"

class ConnectionType(Enum):
    """Types of connections between variables."""
    DIRECTED = "directed"
    UNDIRECTED = "undirected"
    CONDITIONAL = "conditional"
    BIDIRECTIONAL = "bidirectional"

# ================================
# AST NODE HIERARCHY
# ================================

@dataclass
class ASTNode:
    """Base class for all AST nodes."""
    node_type: str = ""
    source_location: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __post_init__(self):
        if not self.node_type:
            self.node_type = self.__class__.__name__
    
    def get_children(self) -> List['ASTNode']:
        """Get all child nodes."""
        children = []
        for value in self.__dict__.values():
            if isinstance(value, ASTNode):
                children.append(value)
            elif isinstance(value, list):
                children.extend([item for item in value if isinstance(item, ASTNode)])
        return children
    
    def accept(self, visitor: 'ASTVisitor') -> Any:
        """Accept a visitor for traversal."""
        return visitor.visit(self)

@dataclass
class Variable(ASTNode):
    """AST node for variable definitions."""
    name: str = ""
    var_type: VariableType = VariableType.HIDDEN_STATE
    dimensions: List[int] = field(default_factory=list)
    data_type: DataType = DataType.CATEGORICAL
    description: Optional[str] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        super().__post_init__()
        self.node_type = "Variable"

@dataclass
class Connection(ASTNode):
    """AST node for connections between variables."""
    source_variables: List[str] = field(default_factory=list)
    target_variables: List[str] = field(default_factory=list)
    connection_type: ConnectionType = ConnectionType.DIRECTED
    weight: Optional[float] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.node_type = "Connection"

@dataclass
class Parameter(ASTNode):
    """AST node for parameter assignments."""
    name: str = ""
    value: Any = None
    type_hint: Optional[str] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.node_type = "Parameter"

@dataclass
class Equation(ASTNode):
    """AST node for mathematical equations."""
    label: Optional[str] = None
    content: str = ""
    format: str = "latex"  # latex, mathml, ascii
    description: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.node_type = "Equation"

@dataclass
class TimeSpecification(ASTNode):
    """AST node for time configuration."""
    time_type: str = "Static"  # "Static", "Dynamic"
    discretization: Optional[str] = None  # "DiscreteTime", "ContinuousTime"
    horizon: Optional[Union[int, str]] = None
    step_size: Optional[float] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.node_type = "TimeSpecification"

@dataclass
class OntologyMapping(ASTNode):
    """AST node for ontology mappings."""
    variable_name: str = ""
    ontology_term: str = ""
    description: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.node_type = "OntologyMapping"

@dataclass
class Section(ASTNode):
    """AST node for GNN sections."""
    section_name: str = ""
    content: List[ASTNode] = field(default_factory=list)
    raw_content: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.node_type = "Section"

# ================================
# INTERNAL REPRESENTATION
# ================================

@dataclass
class GNNInternalRepresentation:
    """
    Unified internal representation for GNN models across all formats.
    
    This is the canonical representation that all parsers convert to
    and all serializers convert from.
    """
    # Core identification
    model_name: str
    version: str = "1.0"
    annotation: str = ""
    
    # Model structure
    variables: List[Variable] = field(default_factory=list)
    connections: List[Connection] = field(default_factory=list)
    parameters: List[Parameter] = field(default_factory=list)
    equations: List[Equation] = field(default_factory=list)
    time_specification: Optional[TimeSpecification] = None
    ontology_mappings: List[OntologyMapping] = field(default_factory=list)
    
    # Metadata
    source_format: Optional[GNNFormat] = None
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    checksum: Optional[str] = None
    
    # Extension points
    extensions: Dict[str, Any] = field(default_factory=dict)
    raw_sections: Dict[str, str] = field(default_factory=dict)
    
    def get_variables_by_type(self, var_type: VariableType) -> List[Variable]:
        """Get all variables of a specific type."""
        return [var for var in self.variables if var.var_type == var_type]
    
    def get_variable_by_name(self, name: str) -> Optional[Variable]:
        """Get a variable by name."""
        for var in self.variables:
            if var.name == name:
                return var
        return None
    
    def get_connections_for_variable(self, variable_name: str) -> List[Connection]:
        """Get all connections involving a specific variable."""
        connections = []
        for conn in self.connections:
            if (variable_name in conn.source_variables or 
                variable_name in conn.target_variables):
                connections.append(conn)
        return connections
    
    def get_parameter_by_name(self, name: str) -> Optional[Parameter]:
        """Get a parameter by name."""
        for param in self.parameters:
            if param.name == name:
                return param
        return None
    
    def validate_structure(self) -> List[str]:
        """Validate the internal structure and return any issues."""
        issues = []
        
        # Check required fields
        if not self.model_name:
            issues.append("Model name is required")
        
        # Check variable name uniqueness
        variable_names = [var.name for var in self.variables]
        if len(variable_names) != len(set(variable_names)):
            issues.append("Variable names must be unique")
        
        # Check connection references
        for conn in self.connections:
            for var_name in conn.source_variables + conn.target_variables:
                if not self.get_variable_by_name(var_name):
                    issues.append(f"Connection references unknown variable: {var_name}")
        
        # Check ontology mapping references
        for mapping in self.ontology_mappings:
            if not self.get_variable_by_name(mapping.variable_name):
                issues.append(f"Ontology mapping references unknown variable: {mapping.variable_name}")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'model_name': self.model_name,
            'version': self.version,
            'annotation': self.annotation,
            'variables': [var.__dict__ for var in self.variables],
            'connections': [conn.__dict__ for conn in self.connections],
            'parameters': [param.__dict__ for param in self.parameters],
            'equations': [eq.__dict__ for eq in self.equations],
            'time_specification': self.time_specification.__dict__ if self.time_specification else None,
            'ontology_mappings': [mapping.__dict__ for mapping in self.ontology_mappings],
            'source_format': self.source_format.value if self.source_format else None,
            'created_at': self.created_at.isoformat(),
            'modified_at': self.modified_at.isoformat(),
            'checksum': self.checksum,
            'extensions': self.extensions,
            'raw_sections': self.raw_sections
        }

# ================================
# VISITOR PATTERN
# ================================

class ASTVisitor(ABC):
    """Abstract base class for AST visitors."""
    
    @abstractmethod
    def visit(self, node: ASTNode) -> Any:
        """Visit an AST node."""
        pass
    
    def visit_children(self, node: ASTNode) -> List[Any]:
        """Visit all children of a node."""
        return [child.accept(self) for child in node.get_children()]

class PrintVisitor(ASTVisitor):
    """Visitor that prints the AST structure."""
    
    def __init__(self, indent: int = 0):
        self.indent = indent
    
    def visit(self, node: ASTNode) -> str:
        """Visit and print a node."""
        indent_str = "  " * self.indent
        result = f"{indent_str}{node.node_type}"
        
        if hasattr(node, 'name'):
            result += f": {node.name}"
        elif hasattr(node, 'label') and node.label:
            result += f": {node.label}"
        
        # Visit children with increased indentation
        child_visitor = PrintVisitor(self.indent + 1)
        for child in node.get_children():
            result += "\n" + child.accept(child_visitor)
        
        return result

# ================================
# PARSER PROTOCOLS AND INTERFACES
# ================================

class GNNParser(Protocol):
    """Protocol defining the interface for GNN parsers."""
    
    def parse_file(self, file_path: str) -> 'ParseResult':
        """Parse a GNN file and return the result."""
        ...
    
    def parse_string(self, content: str) -> 'ParseResult':
        """Parse GNN content from a string."""
        ...
    
    def get_supported_extensions(self) -> List[str]:
        """Get file extensions supported by this parser."""
        ...

class GNNSerializer(Protocol):
    """Protocol defining the interface for GNN serializers."""
    
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Serialize a GNN model to string format."""
        ...
    
    def serialize_to_file(self, model: GNNInternalRepresentation, file_path: str) -> None:
        """Serialize a GNN model to a file."""
        ...

# ================================
# PARSE RESULT
# ================================

@dataclass
class ParseResult:
    """Result of parsing a GNN specification."""
    model: GNNInternalRepresentation
    success: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    parse_time: Optional[float] = None
    source_file: Optional[str] = None
    validation_result: Optional['ValidationResult'] = None
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0
    
    def add_error(self, error: str):
        """Add an error to the result."""
        self.errors.append(error)
        self.success = False
    
    def add_warning(self, warning: str):
        """Add a warning to the result."""
        self.warnings.append(warning)

# ================================
# BASE PARSER IMPLEMENTATION
# ================================

class BaseGNNParser(ABC):
    """Abstract base class for GNN parsers."""
    
    def __init__(self):
        self.current_file: Optional[str] = None
        self.current_line: int = 0
        self.current_column: int = 0
    
    @abstractmethod
    def parse_file(self, file_path: str) -> ParseResult:
        """Parse a GNN file."""
        pass
    
    @abstractmethod
    def parse_string(self, content: str) -> ParseResult:
        """Parse GNN content from string."""
        pass
    
    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions."""
        pass
    
    def create_parse_error(self, message: str, 
                          line: Optional[int] = None,
                          column: Optional[int] = None) -> ParseError:
        """Create a parse error with location information."""
        return ParseError(
            message=message,
            line=line or self.current_line,
            column=column or self.current_column,
            source=self.current_file
        )
    
    def create_empty_model(self, name: str = "Unnamed Model") -> GNNInternalRepresentation:
        """Create an empty GNN model."""
        return GNNInternalRepresentation(
            model_name=name,
            created_at=datetime.now(),
            modified_at=datetime.now()
        )

# ================================
# UTILITY FUNCTIONS
# ================================

def normalize_variable_name(name: str) -> str:
    """
    Normalize variable name for consistent reference.
    
    This function ensures that variable names are treated consistently
    throughout the codebase, regardless of case or minor variations.
    
    Args:
        name: Variable name to normalize
        
    Returns:
        Normalized variable name
    """
    # Handle special Unicode characters like π
    if name == 'π' or name.lower() == 'pi':
        return 'π'  # Standardize on Unicode π
    
    # Remove any whitespace
    name = name.strip()
    
    # Keep case sensitivity for specific Active Inference standard variables
    if name in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        return name
        
    # For other variables, use lowercase for normalization
    return name

def parse_dimensions(dim_str: str) -> List[int]:
    """Parse dimension string like '[2,3,4]' or '[2,3,type=float]' into list of integers."""
    try:
        # Remove brackets and split by comma
        dim_str = dim_str.strip('[]')
        if not dim_str:
            return []
        
        dimensions = []
        for dim in dim_str.split(','):
            dim = dim.strip()
            # Stop parsing if we encounter a type specification
            if dim.startswith('type='):
                break
            if dim.isdigit():
                dimensions.append(int(dim))
            else:
                # Handle symbolic dimensions by defaulting to 1
                dimensions.append(1)
        
        return dimensions
    except Exception:
        return [1]  # Default dimension

def infer_variable_type(name: str) -> VariableType:
    """
    Infer variable type from its name according to Active Inference conventions.
    
    Args:
        name: Variable name
        
    Returns:
        Inferred variable type
    """
    name_lower = name.lower()
    
    # Handle Unicode π (pi) character
    if name == 'π' or name_lower == 'pi':
        return VariableType.POLICY
    
    # Single character variable conventions in Active Inference
    if name == 'A':
        return VariableType.LIKELIHOOD_MATRIX
    elif name == 'B':
        return VariableType.TRANSITION_MATRIX
    elif name == 'C':
        return VariableType.PREFERENCE_VECTOR
    elif name == 'D':
        return VariableType.PRIOR_VECTOR
    elif name == 'E':
        return VariableType.POLICY  # Habit/initial policy prior
    elif name == 'F':
        return VariableType.HIDDEN_STATE  # Variational free energy is often computed on states
    elif name == 'G':
        return VariableType.POLICY  # EFE drives policy
    
    # Other standard Active Inference variables
    if name_lower.startswith('s'):
        return VariableType.HIDDEN_STATE
    elif name_lower.startswith('o'):
        return VariableType.OBSERVATION
    elif name_lower.startswith('u') or name_lower.startswith('a'):
        return VariableType.ACTION
    
    # Check for common prefixes/patterns
    if 'state' in name_lower or 'hidden' in name_lower:
        return VariableType.HIDDEN_STATE
    elif 'obs' in name_lower or 'perception' in name_lower:
        return VariableType.OBSERVATION
    elif 'action' in name_lower or 'control' in name_lower:
        return VariableType.ACTION
    elif 'policy' in name_lower or 'pi' in name_lower or 'habit' in name_lower:
        return VariableType.POLICY
    elif 'prior' in name_lower:
        return VariableType.PRIOR_VECTOR
    elif 'likelihood' in name_lower:
        return VariableType.LIKELIHOOD_MATRIX
    elif 'transition' in name_lower:
        return VariableType.TRANSITION_MATRIX
    elif 'preference' in name_lower or 'utility' in name_lower:
        return VariableType.PREFERENCE_VECTOR
    
    # Default
    return VariableType.HIDDEN_STATE

def parse_connection_operator(op: str) -> ConnectionType:
    """Parse connection operator string to ConnectionType."""
    op = op.strip()
    
    if op in ['>', '->']:
        return ConnectionType.DIRECTED
    elif op == '-':
        return ConnectionType.UNDIRECTED
    elif op == '|':
        return ConnectionType.CONDITIONAL
    elif op == '<->':
        return ConnectionType.BIDIRECTIONAL
    else:
        return ConnectionType.DIRECTED  # Default

# Export all public classes and functions
__all__ = [
    'ParseError', 'ValidationError', 'ValidationWarning', 'ConversionError',
    'GNNFormat', 'VariableType', 'DataType', 'ConnectionType',
    'ASTNode', 'Variable', 'Connection', 'Parameter', 'Equation', 
    'TimeSpecification', 'OntologyMapping', 'Section',
    'GNNInternalRepresentation', 'ParseResult',
    'ASTVisitor', 'PrintVisitor',
    'GNNParser', 'GNNSerializer', 'BaseGNNParser',
    'normalize_variable_name', 'parse_dimensions', 'infer_variable_type', 'parse_connection_operator'
] 