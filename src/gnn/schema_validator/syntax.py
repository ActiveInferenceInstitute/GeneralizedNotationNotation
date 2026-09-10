"""
GNN syntax parsing for the schema_validator package.

Owns GNNParser: the regex-based parser that turns GNN source text into a
ParsedGNN structure. The multi-level validator lives in
gnn/schema_validator/validator.py and imports GNNParser from here.
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from gnn.types import (
    GNNConnection,
    GNNFormat,
    GNNVariable,
    ParsedGNN,
    ValidationLevel,
    ValidationResult,
)

# Try to import round-trip testing capabilities
try:
    from gnn.parsers import GNNParsingSystem

    ROUND_TRIP_AVAILABLE = True
except ImportError:
    ROUND_TRIP_AVAILABLE = False

logger = logging.getLogger(__name__)


class GNNParser:
    """Enhanced parser for GNN file format with multi-format support."""

    # Regular expressions for GNN syntax elements (enhanced)
    SECTION_PATTERN = re.compile(r"^## (.+)$")
    VARIABLE_PATTERN = re.compile(
        r"^([\w_π][\w\d_π+]*)(\[([^\]]+)\])?(?:,type=([a-zA-Z]+))?(?:\s*#\s*(.*))?$"
    )
    CONNECTION_PATTERN = re.compile(r"^(.+?)\s*(>|->|-|\|)\s*(.+?)(?:\s*#\s*(.*))?$")
    # Accept both '=' and ':' as assignment separators to support previous files
    PARAMETER_PATTERN = re.compile(
        r"^([\w_π][\w\d_π+]*)(\s*[:=]\s*)(.+?)(?:\s*#\s*(.*))?$"
    )
    ONTOLOGY_PATTERN = re.compile(
        r"^([\w_π][\w\d_π+]*)(\s*=\s*)([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*#\s*(.*))?$"
    )
    COMMENT_PATTERN = re.compile(r"^\s*#\s*(.*)$")

    # Enhanced format detection patterns
    FORMAT_SIGNATURES = MappingProxyType(
        {
            "json": (r"^\s*\{", r'"model_name":', r'"variables":'),
            "xml": (r"^\s*<\?xml", r"<gnn.*>", r"<model"),
            "yaml": (r"^---", r"model_name:", r"variables:"),
            "binary": (b"\x80\x03", b"pickle", b"\x00\x00\x00"),  # Pickle signatures
        }
    )

    def __init__(self, enhanced_validation: bool = True) -> None:
        """Initialize the instance."""
        self.enhanced_validation = enhanced_validation
        self.parsing_system: Optional[GNNParsingSystem]
        if enhanced_validation and ROUND_TRIP_AVAILABLE:
            self.parsing_system = GNNParsingSystem()
            logger.info("Enhanced multi-format parsing system initialized")
        else:
            self.parsing_system = None
            logger.info("Basic GNN parser initialized")
        # Expose basic schema metadata expected by tests and previous callers
        try:
            validator = (
                getattr(self.parsing_system, "validator", None)
                if self.parsing_system
                else None
            )
            self.schema = getattr(validator, "schema", {})
        except Exception as e:
            logger.debug(f"Could not load schema from parsing system: {e}")
            self.schema = {}

    def validate_file(
        self,
        file_path: Union[str, Path],
        validation_level: Optional[ValidationLevel] = None,
    ) -> "ValidationResult":
        """Validate a GNN file via the full ``GNNValidator`` implementation."""
        from gnn.schema_validator.validator import GNNValidator

        validator = GNNValidator()
        return validator.validate_file(file_path, validation_level=validation_level)

    def parse_file(
        self, file_path: Union[str, Path], format_hint: Optional[str] = None
    ) -> ParsedGNN:
        """Enhanced file parsing with format detection and validation."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Detect format
        detected_format = format_hint or self._detect_file_format(file_path)

        # Handle different formats
        if detected_format == "binary":
            return self._parse_binary_file(file_path)
        else:
            # Read as text
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try as binary if UTF-8 fails
                return self._parse_binary_file(file_path)

        return self.parse_content(content, str(file_path), detected_format)

    def _detect_file_format(self, file_path: Path) -> str:
        """Enhanced format detection with content analysis."""
        # First try extension-based detection
        extension = file_path.suffix.lower()
        extension_map: dict[str, Any] = {
            ".md": "markdown",
            ".json": "json",
            ".xml": "xml",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".pkl": "pkl",
            ".pickle": "binary",
        }

        if extension in extension_map:
            detected = extension_map[extension]

            # Verify with content analysis for ambiguous cases
            if extension in [".md", ".txt"] and file_path.exists():
                content_format = self._detect_format_from_content(file_path)
                if content_format != "markdown":
                    return content_format

            return cast("str", detected)

        # Content-based detection for unknown extensions
        return self._detect_format_from_content(file_path)

    def _detect_format_from_content(self, file_path: Path) -> str:
        """Detect format from file content analysis."""
        try:
            # Try reading as text first
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(2000)  # Read first 2KB

            content.lower()

            # Check format signatures
            for fmt, patterns in self.FORMAT_SIGNATURES.items():
                if fmt == "binary":
                    continue  # Skip binary patterns for text content

                text_patterns = [
                    pattern for pattern in patterns if isinstance(pattern, str)
                ]
                if any(
                    re.search(pattern, content, re.IGNORECASE)
                    for pattern in text_patterns
                ):
                    return fmt

            # Check for GNN markdown indicators
            if "##" in content and any(
                section in content
                for section in ["GNNSection", "ModelName", "StateSpaceBlock"]
            ):
                return "markdown"

            return "markdown"  # Default recovery

        except UnicodeDecodeError:
            return "binary"
        except Exception as e:
            logger.debug(f"Format detection failed for file: {e}")
            return "unknown"

    def _parse_binary_file(self, file_path: Path) -> ParsedGNN:
        """Parse binary files (pickle format)."""
        try:
            from gnn.parsers.binary_parser import safe_pickle_load

            with open(file_path, "rb") as f:
                data = safe_pickle_load(f)

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
                metadata={"parse_error": str(e), "source_format": "binary"},
            )

    def _convert_pickle_to_parsed_gnn(self, data: Any) -> ParsedGNN:
        """Convert pickle data to ParsedGNN structure."""
        # Implementation depends on pickle data structure
        # This is a simplified version
        if isinstance(data, dict):
            return ParsedGNN(
                gnn_section=data.get("gnn_section", "PickleGNN"),
                version=data.get("version", "1.0"),
                model_name=data.get("model_name", "PickleModel"),
                model_annotation=data.get("annotation", ""),
                variables=data.get("variables", {}),
                connections=data.get("connections", []),
                parameters=data.get("parameters", {}),
                equations=data.get("equations", []),
                time_config=data.get("time_config", {}),
                ontology_mappings=data.get("ontology_mappings", {}),
                model_parameters=data.get("model_parameters", {}),
                footer=data.get("footer", ""),
                source_format="pickle",
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
                source_format="pickle",
            )

    def parse_content(
        self, content: str, source_name: str = "<string>", format_hint: str = "markdown"
    ) -> ParsedGNN:
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

        # Recovery to markdown parsing
        return self._parse_markdown_content(content, source_name)

    def _convert_parse_result_to_parsed_gnn(
        self, result: Any, source_format: str
    ) -> ParsedGNN:
        """Convert ParseResult to ParsedGNN format."""
        model = result.model

        # Convert variables
        variables: dict[Any, Any] = {}
        for var in model.variables:
            variables[var.name] = GNNVariable(
                name=var.name,
                dimensions=getattr(var, "dimensions", []),
                data_type=str(getattr(var, "data_type", "categorical")),
                description=getattr(var, "description", ""),
                ontology_mapping=getattr(var, "ontology_mapping", None),
            )

        # Convert connections
        connections: list[Any] = []
        for conn in model.connections:
            connections.append(
                GNNConnection(
                    source=getattr(conn, "source_variables", []),
                    target=getattr(conn, "target_variables", []),
                    connection_type=str(getattr(conn, "connection_type", "directed")),
                    symbol=self._infer_symbol_from_type(
                        str(getattr(conn, "connection_type", "directed"))
                    ),
                    description=getattr(conn, "description", ""),
                )
            )

        # Convert parameters
        parameters: dict[Any, Any] = {}
        for param in model.parameters:
            parameters[param.name] = param.value

        return ParsedGNN(
            gnn_section=getattr(model, "gnn_section", f"{source_format.upper()}GNN"),
            version=getattr(model, "version", "1.0"),
            model_name=model.model_name,
            model_annotation=model.annotation,
            variables=variables,
            connections=connections,
            parameters=parameters,
            equations=getattr(model, "equations", []),
            time_config=getattr(model, "time_config", {}),
            ontology_mappings=getattr(model, "ontology_mappings", {}),
            model_parameters=getattr(model, "model_parameters", {}),
            footer=getattr(model, "footer", ""),
            source_format=source_format,
            semantic_checksum=self._compute_semantic_checksum(model),
        )

    def _infer_symbol_from_type(self, connection_type: str) -> str:
        """Infer connection symbol from type."""
        symbol_map: dict[str, Any] = {
            "directed": ">",
            "undirected": "-",
            "conditional": "|",
            "bidirectional": "<->",
        }
        return cast("str", symbol_map.get(connection_type, ">"))

    def _compute_semantic_checksum(self, model: Any) -> str:
        """Compute semantic checksum for model."""
        # Create normalized representation
        checksum_data: dict[str, Any] = {
            "model_name": model.model_name,
            "variables": sorted([var.name for var in model.variables]),
            "connections_count": len(model.connections),
            "parameters_count": len(model.parameters),
        }

        checksum_str = json.dumps(checksum_data, sort_keys=True)
        return hashlib.md5(checksum_str.encode(), usedforsecurity=False).hexdigest()

    def _parse_markdown_content(self, content: str, source_name: str) -> ParsedGNN:
        """Parse GNN content from string (markdown format).

        Side effects: resets self.line_number to 0 and self.current_section to None
        at the start of each call, then mutates both as parsing progresses.
        """
        lines = content.split("\n")
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
            footer="",
        )

        current_content: list[Any] = []

        for line in lines:
            self.line_number += 1
            line = line.rstrip()

            # Check for section headers
            section_match = self.SECTION_PATTERN.match(line)
            if section_match:
                # Process previous section content
                if self.current_section and current_content:
                    self._process_section_content(
                        parsed, self.current_section, current_content
                    )

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

    def _process_section_content(
        self, parsed: ParsedGNN, section: str, content: List[str]
    ) -> Any:
        """Process content for a specific section."""
        content_text = "\n".join(content).strip()

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

    def _parse_state_space_block(self, parsed: ParsedGNN, content: List[str]) -> Any:
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
                dimensions: list[Any] = []
                if dims_str:
                    for dim in dims_str.split(","):
                        dim = dim.strip()
                        # Ignore type=... fragments and non-numeric 'type' annotations
                        if dim.startswith("type="):
                            continue
                        # Only count numeric dimensions for tests that expect numeric length
                        if dim.isdigit():
                            dimensions.append(int(dim))
                        else:
                            # Keep string dimensions but do not count them as numeric dims
                            dimensions.append(dim)

                variable = GNNVariable(
                    name=name,
                    dimensions=dimensions,
                    data_type=data_type,
                    description=description,
                    line_number=self.line_number - len(content) + i,
                )

                parsed.variables[name] = variable
            else:
                logger.warning(f"Could not parse variable definition: {line}")

    def _parse_connections(self, parsed: ParsedGNN, content: List[str]) -> Any:
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
                    line_number=self.line_number - len(content) + i,
                )

                parsed.connections.append(connection)
            else:
                logger.warning(f"Could not parse connection: {line}")

    def _parse_variable_group(self, group_str: str) -> Union[str, List[str]]:
        """Parse a variable group (single variable or parenthesized list)."""
        group_str = group_str.strip()

        if group_str.startswith("(") and group_str.endswith(")"):
            # Parse comma-separated list
            inner = group_str[1:-1]
            variables = [v.strip() for v in inner.split(",")]
            return variables if len(variables) > 1 else variables[0]
        else:
            return group_str

    def _get_connection_type(self, symbol: str) -> str:
        """Determine connection type from symbol."""
        if symbol in [">", "->"]:
            return "directed"
        elif symbol == "-":
            return "undirected"
        elif symbol == "|":
            return "conditional"
        else:
            return "unknown"

    def _parse_parameters(
        self, parsed: ParsedGNN, content: List[str], param_type: str
    ) -> Any:
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
                match.group(4)

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
            logger.debug(
                "Value is not valid JSON, trying GNN-specific formats: %s",
                value_str[:80],
            )

        # Try specific GNN formats
        if value_str.startswith("{") and value_str.endswith("}"):
            # Matrix or tuple format
            return self._parse_matrix_or_tuple(value_str)
        elif value_str.startswith("(") and value_str.endswith(")"):
            # Tuple format
            return self._parse_tuple(value_str)
        elif value_str.replace(".", "").replace("-", "").isdigit():
            # Numeric value
            return float(value_str) if "." in value_str else int(value_str)
        elif value_str.lower() in ["true", "false"]:
            # Boolean value
            return value_str.lower() == "true"
        else:
            # String value
            return value_str

    def _parse_matrix_or_tuple(self, value_str: str) -> Any:
        """Parse matrix or tuple notation with full Active Inference support."""
        value_str = value_str.strip()

        # Handle nested tuple/matrix structures like {((0.8,0.1,0.1),(0.1,0.8,0.1))}
        if value_str.startswith("{") and value_str.endswith("}"):
            inner = value_str[1:-1].strip()

            # Check for matrix structure with nested tuples
            if inner.startswith("(") and inner.count("(") > 1:
                return self._parse_nested_matrix(inner)
            else:
                return self._parse_tuple(inner)

        # Handle simple tuple like (0.5,0.5)
        elif value_str.startswith("(") and value_str.endswith(")"):
            return self._parse_tuple(value_str)

        # Handle list/array notation
        elif value_str.startswith("[") and value_str.endswith("]"):
            return self._parse_array(value_str)

        return value_str

    def _parse_nested_matrix(self, inner: str) -> List[List[float]]:
        """Parse nested matrix structure like ((0.8,0.1),(0.1,0.8))."""
        matrix: list[Any] = []
        depth = 0
        current_tuple = ""

        for char in inner:
            if char == "(":
                depth += 1
                current_tuple += char
            elif char == ")":
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
            elif char == "," and depth == 0:
                continue  # Skip commas between tuples

        return matrix

    def _parse_array(self, value_str: str) -> List[Any]:
        """Parse array notation [1,2,3] or [[1,2],[3,4]]."""
        from gnn.utils.safe_eval import safe_literal_eval

        try:
            return cast("list[Any]", safe_literal_eval(value_str))
        except (ValueError, SyntaxError):
            # Recovery parsing
            inner = value_str[1:-1].strip()
            if not inner:
                return []

            elements: list[Any] = []
            depth = 0
            current = ""

            for char in inner:
                if char in "[(":
                    depth += 1
                elif char in "])":
                    depth -= 1

                if char == "," and depth == 0:
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
        if value_str.lower() in ["true", "false"]:
            return value_str.lower() == "true"

        # Numeric values
        try:
            if "." in value_str or "e" in value_str.lower():
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            logger.debug("Value is not numeric, treating as string: %s", value_str[:80])

        # String values
        if value_str.startswith('"') and value_str.endswith('"'):
            return value_str[1:-1]
        elif value_str.startswith("'") and value_str.endswith("'"):
            return value_str[1:-1]

        return value_str

    def _parse_tuple(self, value_str: str) -> Tuple[Any, ...]:
        """Parse tuple notation with proper type conversion."""
        if value_str.startswith("(") and value_str.endswith(")"):
            inner = value_str[1:-1].strip()
        else:
            inner = value_str.strip()

        if not inner:
            return ()

        # Split on commas, handling nested structures
        elements: list[Any] = []
        depth = 0
        current = ""

        for char in inner:
            if char in "([{":
                depth += 1
                current += char
            elif char in ")]}":
                depth -= 1
                current += char
            elif char == "," and depth == 0:
                elements.append(self._parse_scalar_value(current.strip()))
                current = ""
            else:
                current += char

        if current.strip():
            elements.append(self._parse_scalar_value(current.strip()))

        return tuple(elements)

    def _parse_equations(self, parsed: ParsedGNN, content: List[str]) -> Any:
        """Parse Equations section."""
        current_equation: dict[Any, Any] = {}

        for line in content:
            line = line.strip()
            if not line:
                if current_equation:
                    parsed.equations.append(current_equation)
                    current_equation = {}
                continue

            comment_match = self.COMMENT_PATTERN.match(line)
            if comment_match is not None:
                if current_equation:
                    current_equation["description"] = comment_match.group(1)
            else:
                if not current_equation:
                    current_equation = {"latex": line}
                else:
                    current_equation["latex"] += " " + line

        if current_equation:
            parsed.equations.append(current_equation)

    def _parse_time_config(self, parsed: ParsedGNN, content: List[str]) -> Any:
        """Parse Time section."""
        for line in content:
            line = line.strip()
            if not line or self.COMMENT_PATTERN.match(line):
                continue

            if "=" in line:
                key, value = line.split("=", 1)
                parsed.time_config[key.strip()] = value.strip()
            else:
                # Simple time type specification
                parsed.time_config["type"] = line

    def _parse_ontology_mappings(self, parsed: ParsedGNN, content: List[str]) -> Any:
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
        signature: dict[Any, Any] = {}
        for line in content.split("\n"):
            line = line.strip()
            if ":" in line:
                key, value = line.split(":", 1)
                signature[key.strip()] = value.strip()
        return signature
