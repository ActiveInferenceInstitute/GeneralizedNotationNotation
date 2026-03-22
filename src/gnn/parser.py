#!/usr/bin/env python3
"""
GNN parser module for GNN pipeline.
"""

from pathlib import Path
from typing import Callable, Dict, Any, List, Tuple, Union, Optional

# Single authoritative definition lives in types.py (includes RESEARCH and ROUND_TRIP).
from .types import ValidationLevel, ParsedGNN

class _GNNParseAccumulator:
    """Internal mutable builder for GNN parse results.

    This is a local implementation detail of GNNParsingSystem._basic_parser.
    The public canonical type is gnn.types.ParsedGNN (a dataclass).
    """

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self.file_name = self.file_path.name
        self.content = ""
        self.sections = []
        self.variables = []
        self.connections = []
        self.parse_errors = []
        self.parse_warnings = []

    def add_section(self, section_name: str, section_content: str = "") -> None:
        """Add a section to the parsed GNN."""
        self.sections.append({
            "name": section_name,
            "content": section_content
        })

    def add_variable(self, variable_name: str, variable_type: str = "", variable_value: str = "") -> None:
        """Add a variable to the parsed GNN."""
        self.variables.append({
            "name": variable_name,
            "type": variable_type,
            "value": variable_value
        })

    def add_connection(self, source: str, target: str, connection_type: str = "") -> None:
        """Add a connection to the parsed GNN."""
        self.connections.append({
            "source": source,
            "target": target,
            "type": connection_type
        })

    def add_error(self, error_message: str) -> None:
        """Add a parse error."""
        self.parse_errors.append(error_message)

    def add_warning(self, warning_message: str) -> None:
        """Add a parse warning."""
        self.parse_warnings.append(warning_message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "file_path": str(self.file_path),
            "file_name": self.file_name,
            "sections": self.sections,
            "variables": self.variables,
            "connections": self.connections,
            "parse_errors": self.parse_errors,
            "parse_warnings": self.parse_warnings
        }

class GNNParsingSystem:
    """System for parsing GNN files."""

    def __init__(self):
        """Initialize the GNN parsing system."""
        self.parsers = {}
        self.validators = {}

    def register_parser(self, format_name: str, parser_func: Callable) -> None:
        """Register a parser for a specific format."""
        self.parsers[format_name] = parser_func

    def register_validator(self, format_name: str, validator_func: Callable) -> None:
        """Register a validator for a specific format."""
        self.validators[format_name] = validator_func

    def parse_file(self, file_path: Union[str, Path], format_name: str = "auto") -> Optional[_GNNParseAccumulator]:
        """Parse a GNN file."""
        file_path = Path(file_path)

        if not file_path.exists():
            return None

        # Auto-detect format if not specified
        if format_name == "auto":
            format_name = self._detect_format(file_path)

        # Get appropriate parser
        parser = self.parsers.get(format_name)
        if parser:
            return parser(file_path)
        else:
            # Recovery to basic parser
            return self._basic_parser(file_path)

    def _detect_format(self, file_path: Path) -> str:
        """Detect the format of a GNN file."""
        extension = file_path.suffix.lower()

        if extension == ".md":
            return "markdown"
        elif extension == ".gnn":
            return "gnn"
        elif extension == ".txt":
            return "text"
        else:
            return "markdown"  # Default to markdown

    def _basic_parser(self, file_path: Path) -> _GNNParseAccumulator:
        """Basic parser for GNN files."""
        parsed = _GNNParseAccumulator(file_path)

        try:
            with open(file_path, 'r') as f:
                parsed.content = f.read()

            # Extract sections
            import re
            section_pattern = r'^#+\s+(.+)$'
            matches = re.finditer(section_pattern, parsed.content, re.MULTILINE)

            for match in matches:
                section_name = match.group(1).strip()
                parsed.add_section(section_name)

            # Extract variables
            var_patterns = [
                r'(\w+)\s*:\s*(\w+)',  # name: type
                r'(\w+)\s*=\s*([^;\n]+)',  # name = value
            ]

            for pattern in var_patterns:
                matches = re.finditer(pattern, parsed.content)
                for match in matches:
                    var_name = match.group(1)
                    var_value = match.group(2)
                    # Normalize dimensions: strip type=... parts and filter out empty
                    if isinstance(var_value, str) and 'type=' in var_value:
                        dims = [d for d in var_value.split(',') if not d.strip().startswith('type=')]
                        var_value = ','.join(dims)
                    parsed.add_variable(var_name, "", var_value)

            # Extract connections
            conn_patterns = [
                r'(\w+)\s*->\s*(\w+)',  # source -> target
                r'(\w+)\s*→\s*(\w+)',   # source → target
            ]

            for pattern in conn_patterns:
                matches = re.finditer(pattern, parsed.content)
                for match in matches:
                    source = match.group(1)
                    target = match.group(2)
                    parsed.add_connection(source, target)

        except Exception as e:
            parsed.add_error(f"Failed to parse file: {e}")

        return parsed

class GNNFormatSpec:
    """Represents a GNN format specification (MIME types, extensions).

    Note: For the GNNFormat Enum used throughout the pipeline, see
    gnn.parsers.common.GNNFormat.
    """

    def __init__(self):
        """Initialize the GNN format spec."""
        self.name = "GNN"
        self.version = "1.0"
        self.extensions = [".gnn", ".md"]
        self.mime_types = ["text/gnn", "text/markdown"]

class GNNFormalParser:
    """Placeholder class for when Lark is not available."""
    def __init__(self): pass
    def parse_file(self, file_path): return None
    def parse_content(self, content, source_name="<string>"): return None
    def validate_syntax(self, content): return False, ["Lark not available"]
    def visualize_parse_tree(self, content): return "Lark not available"

class ParsedGNNFormal:
    """Placeholder class for when Lark is not available."""
    def __init__(self): pass

def parse_gnn_formal(file_path: Union[str, Any]) -> None: return None
def validate_gnn_syntax_formal(content: str) -> Tuple[bool, List[str]]: return False, ["Lark not available"]
def get_parse_tree_visualization(content: str) -> str: return "Lark not available"



def validate_gnn(file_path_or_content: Union[str, Path], validation_level: ValidationLevel = ValidationLevel.STANDARD, **kwargs: Any) -> Tuple[bool, List[str]]:
    """
    Validate a GNN file or content.
    
    Args:
        file_path_or_content: Path to GNN file or content string
        validation_level: Level of validation to perform
        **kwargs: Additional validation options
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    try:
        # Determine if input is file path or content
        if isinstance(file_path_or_content, (str, Path)) and Path(file_path_or_content).exists():
            # It's a file path
            file_path = Path(file_path_or_content)
            with open(file_path, 'r') as f:
                content = f.read()
        else:
            # It's content
            content = str(file_path_or_content)

        errors = []

        # Basic validation
        if not content.strip():
            errors.append("Content is empty")
            return False, errors

        # Structure validation (STANDARD level; also a prerequisite for STRICT)
        needs_structure_check = validation_level in (ValidationLevel.STANDARD, ValidationLevel.STRICT)
        if needs_structure_check:
            import re

            sections = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
            if not sections:
                errors.append("No sections found (use # headers)")

            variables = re.findall(r'(\w+)\s*[:=]', content)
            if not variables:
                errors.append("No variables found")

            connections = re.findall(r'(\w+)\s*[->→]\s*(\w+)', content)
            if not connections:
                errors.append("No connections found")

        # Strict validation (explicitly extends structure checks above)
        if validation_level == ValidationLevel.STRICT:
            # Check for balanced braces
            if content.count('{') != content.count('}'):
                errors.append("Unmatched braces")

            # Check for balanced brackets
            if content.count('[') != content.count(']'):
                errors.append("Unmatched brackets")

            # Check for minimum content length
            if len(content) < 50:
                errors.append("Content too short for valid GNN")

        return len(errors) == 0, errors

    except Exception as e:
        return False, [f"Validation error: {e}"]

def _convert_parse_result_to_parsed_gnn(parse_result, source_format: str = "unknown") -> Optional[ParsedGNN]:
    """
    Convert a ParseResult to a ParsedGNN dataclass object.
    
    Args:
        parse_result: ParseResult object from parser
        source_format: Format hint for the source (e.g., "markdown", "json")
        
    Returns:
        ParsedGNN object or None if parse_result is None
    """
    if parse_result is None:
        return None

    try:
        # Import types needed for conversion
        from .types import ParsedGNN, GNNVariable, GNNConnection
        from .parsers.common import ParseResult as ParseResultType

        # Verify it's a ParseResult
        if not isinstance(parse_result, ParseResultType):
            # If it's already a ParsedGNN, return as-is
            if isinstance(parse_result, ParsedGNN):
                return parse_result
            # Otherwise, create minimal representation
            return ParsedGNN(
                gnn_section=f"{source_format.upper()}GNN",
                version="1.0",
                model_name="Unknown",
                model_annotation="",
                variables={},
                connections=[],
                parameters={},
                equations=[],
                time_config={},
                ontology_mappings={},
                model_parameters={},
                footer="",
                source_format=source_format
            )

        model = parse_result.model

        # Convert variables
        variables = {}
        for var in getattr(model, 'variables', []):
            var_name = getattr(var, 'name', 'unknown')
            variables[var_name] = GNNVariable(
                name=var_name,
                dimensions=getattr(var, 'dimensions', []),
                data_type=str(getattr(var, 'data_type', 'categorical')),
                description=getattr(var, 'description', ''),
                ontology_mapping=getattr(var, 'ontology_mapping', None)
            )

        # Helper function to infer connection symbol
        def _infer_symbol_from_type(connection_type: str) -> str:
            """Infer connection symbol from type."""
            type_lower = connection_type.lower()
            if 'directed' in type_lower or '->' in type_lower:
                return ">"
            elif 'undirected' in type_lower or '-' in type_lower:
                return "-"
            elif 'conditional' in type_lower or '|' in type_lower:
                return "|"
            else:
                return ">"

        # Convert connections
        connections = []
        for conn in getattr(model, 'connections', []):
            connections.append(GNNConnection(
                source=getattr(conn, 'source_variables', getattr(conn, 'source', [])),
                target=getattr(conn, 'target_variables', getattr(conn, 'target', [])),
                connection_type=str(getattr(conn, 'connection_type', 'directed')),
                symbol=_infer_symbol_from_type(str(getattr(conn, 'connection_type', 'directed'))),
                description=getattr(conn, 'description', '')
            ))

        # Convert parameters
        parameters = {}
        for param in getattr(model, 'parameters', []):
            param_name = getattr(param, 'name', 'unknown')
            parameters[param_name] = getattr(param, 'value', None)

        return ParsedGNN(
            gnn_section=getattr(model, 'gnn_section', f"{source_format.upper()}GNN"),
            version=getattr(model, 'version', '1.0'),
            model_name=getattr(model, 'model_name', 'Unknown'),
            model_annotation=getattr(model, 'annotation', ''),
            variables=variables,
            connections=connections,
            parameters=parameters,
            equations=getattr(model, 'equations', []),
            time_config=getattr(model, 'time_config', {}),
            ontology_mappings=getattr(model, 'ontology_mappings', {}),
            model_parameters=getattr(model, 'model_parameters', {}),
            footer=getattr(model, 'footer', ''),
            source_format=source_format
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to convert parse result from format '{source_format}': {e}"
        ) from e
