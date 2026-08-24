"""Consistency and cross-reference checks for GNN models."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

ModelData: TypeAlias = Mapping[str, Any]
ModelInput: TypeAlias = str | Path | ModelData | None


@dataclass(frozen=True)
class _ConnectionReference:
    """Normalized connection used by all supported input formats."""

    index: int
    source: str | None
    target: str | None
    directed: bool = True


@dataclass
class _ModelStructure:
    """Normalized subset required for consistency checks."""

    kind: Literal["raw", "markdown", "structured", "empty"]
    block_names: list[str] = field(default_factory=list)
    connections: list[_ConnectionReference] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    state_blocks: list[str] = field(default_factory=list)
    connection_blocks: list[str] = field(default_factory=list)


def _append_once(items: list[str], message: str) -> None:
    if message not in items:
        items.append(message)


def _extract_field(block: str, field_name: str) -> str | None:
    """Extract one raw block field from inline or multiline syntax."""
    match = re.search(
        rf"\b{re.escape(field_name)}\s*:\s*(.+?)"
        r"(?=\s+[A-Za-z][A-Za-z0-9_]*\s*:|\n|$)",
        block,
    )
    return match.group(1).strip() if match else None


def _section_lines(content: str, section_name: str) -> list[tuple[int, str]]:
    """Return numbered, non-comment lines from one Markdown GNN section."""
    lines: list[tuple[int, str]] = []
    in_section = False
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        stripped = raw_line.strip()
        if stripped.startswith("## "):
            in_section = stripped[3:].strip() == section_name
            continue
        if in_section and stripped and not stripped.startswith("#"):
            lines.append((line_number, stripped))
    return lines


def _connection_group(value: str) -> list[str]:
    group = value.strip()
    if group.startswith("(") and group.endswith(")"):
        group = group[1:-1]
    return [
        "π" if item.strip().lower() == "pi" else item.strip()
        for item in group.split(",")
        if item.strip()
    ]


def _parse_markdown_structure(content: str) -> _ModelStructure:
    declaration_pattern = re.compile(
        r"^(?P<name>[^\s\[\],()><|]+)\s*\[[^\]]+\](?:\s*#.*)?$"
    )
    block_names: list[str] = []
    diagnostics: list[str] = []
    for line_number, line in _section_lines(content, "StateSpaceBlock"):
        match = declaration_pattern.fullmatch(line)
        if match is None:
            _append_once(
                diagnostics,
                f"Unparseable StateSpaceBlock declaration at line {line_number}: '{line}'",
            )
        else:
            raw_name = match.group("name")
            name = "π" if raw_name.lower() == "pi" else raw_name
            if name in block_names:
                diagnostics.append(
                    f"Duplicate variable declaration at line {line_number}: '{name}'"
                )
            block_names.append(name)

    normalized_connections: list[_ConnectionReference] = []
    for line_number, raw_line in _section_lines(content, "Connections"):
        line = raw_line.split("#", 1)[0].strip()
        operator = next(
            (
                candidate
                for candidate in ("<->", "->", ">", "|", "-")
                if candidate in line
            ),
            None,
        )
        if operator is None:
            diagnostics.append(
                f"Unparseable connection at line {line_number}: '{line}'"
            )
            continue
        source_text, target_text = line.split(operator, 1)
        target_text = target_text.split(":", 1)[0]
        sources = _connection_group(source_text)
        targets = _connection_group(target_text)
        if not sources or not targets:
            diagnostics.append(
                f"Connection at line {line_number} has an empty endpoint: '{line}'"
            )
            continue
        directed = operator not in {"-", "<->"}
        for source in sources:
            for target in targets:
                connection_index = len(normalized_connections)
                normalized_connections.append(
                    _ConnectionReference(
                        index=connection_index,
                        source=source,
                        target=target,
                        directed=directed,
                    )
                )

    return _ModelStructure(
        kind="markdown",
        block_names=block_names,
        connections=normalized_connections,
        diagnostics=diagnostics,
    )


def _parse_raw_structure(content: str) -> _ModelStructure:
    state_blocks = re.findall(r"StateSpaceBlock\s*\{([^}]*)\}", content)
    connection_blocks = re.findall(r"Connection\s*\{([^}]*)\}", content)
    block_names = [
        name
        for block in state_blocks
        if (name := _extract_field(block, "Name")) is not None
    ]
    connections = [
        _ConnectionReference(
            index=index,
            source=_extract_field(block, "From"),
            target=_extract_field(block, "To"),
        )
        for index, block in enumerate(connection_blocks)
    ]
    return _ModelStructure(
        kind="raw",
        block_names=block_names,
        connections=connections,
        state_blocks=state_blocks,
        connection_blocks=connection_blocks,
    )


def _parse_content_structure(content: str) -> _ModelStructure:
    if re.search(r"(?m)^##\s+StateSpaceBlock\s*$", content):
        return _parse_markdown_structure(content)
    if re.search(r"\b(?:StateSpaceBlock|Connection)\s*\{", content):
        return _parse_raw_structure(content)
    return _ModelStructure(kind="empty")


def _endpoint_names(value: Any) -> list[str]:
    """Normalize a structured connection endpoint without treating text as a list."""
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            item.strip() for item in value if isinstance(item, str) and item.strip()
        ]
    return []


def _raw_sections_content(model_data: ModelData) -> str | None:
    raw_sections = model_data.get("raw_sections")
    if not isinstance(raw_sections, Mapping) or not raw_sections:
        section_names = (
            "ModelName",
            "StateSpaceBlock",
            "InitialParameterization",
            "Connections",
        )
        raw_sections = {
            name: model_data[name] for name in section_names if name in model_data
        }
    if not raw_sections:
        return None

    parts: list[str] = []
    for section_name, section_content in raw_sections.items():
        if isinstance(section_name, str) and isinstance(section_content, str):
            parts.append(f"## {section_name}\n\n{section_content}")
    return "\n\n".join(parts) if parts else None


def _parse_structured_model(model_data: ModelData) -> _ModelStructure:
    has_structured_keys = "variables" in model_data or "connections" in model_data
    if not has_structured_keys:
        raw_content = _raw_sections_content(model_data)
        if raw_content is not None:
            structure = _parse_content_structure(raw_content)
            structure.kind = "structured"
            return structure

    structure = _ModelStructure(kind="structured")
    variables = model_data.get("variables", [])
    if not isinstance(variables, Sequence) or isinstance(
        variables, (str, bytes, bytearray)
    ):
        structure.diagnostics.append("'variables' must be a sequence of mappings")
        variables = []

    for index, variable in enumerate(variables):
        if not isinstance(variable, Mapping):
            structure.diagnostics.append(f"Variable {index} must be a mapping")
            continue
        name = variable.get("name")
        if not isinstance(name, str) or not name.strip():
            structure.diagnostics.append(
                f"Variable {index} must define a non-empty string name"
            )
            continue
        structure.block_names.append(name.strip())

    connections = model_data.get("connections", [])
    if not isinstance(connections, Sequence) or isinstance(
        connections, (str, bytes, bytearray)
    ):
        structure.diagnostics.append("'connections' must be a sequence of mappings")
        connections = []

    normalized_index = 0
    for index, connection in enumerate(connections):
        if not isinstance(connection, Mapping):
            structure.diagnostics.append(f"Connection {index} must be a mapping")
            continue
        sources = _endpoint_names(
            connection.get("source_variables", connection.get("source"))
        )
        targets = _endpoint_names(
            connection.get("target_variables", connection.get("target"))
        )
        if not sources:
            structure.diagnostics.append(
                f"Connection {index} must define at least one source variable"
            )
        if not targets:
            structure.diagnostics.append(
                f"Connection {index} must define at least one target variable"
            )
        directed = str(connection.get("connection_type", "directed")).lower() not in {
            "undirected",
            "-",
        }
        for source in sources:
            for target in targets:
                structure.connections.append(
                    _ConnectionReference(
                        index=normalized_index,
                        source=source,
                        target=target,
                        directed=directed,
                    )
                )
                normalized_index += 1

    return structure


def _ordered_duplicates(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _cycle_nodes(
    block_names: Sequence[str], connections: Sequence[_ConnectionReference]
) -> list[str]:
    """Return exactly the nodes in directed cycles using Tarjan components."""
    graph: dict[str, list[str]] = {}
    order: list[str] = []
    known = set(block_names)
    for name in block_names:
        if name not in order:
            order.append(name)
        graph.setdefault(name, [])
    for connection in connections:
        if (
            connection.directed
            and connection.source in known
            and connection.target in known
        ):
            source = connection.source
            target = connection.target
            if source is not None and target is not None:
                graph.setdefault(source, []).append(target)

    next_index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    cyclic: set[str] = set()

    def strong_connect(node: str) -> None:
        nonlocal next_index
        indices[node] = next_index
        lowlinks[node] = next_index
        next_index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in graph.get(node, []):
            if neighbor not in indices:
                strong_connect(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbor])

        if lowlinks[node] != indices[node]:
            return
        component: list[str] = []
        while stack:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == node:
                break
        if len(component) > 1 or node in graph.get(node, []):
            cyclic.update(component)

    for node in order:
        if node not in indices:
            strong_connect(node)
    return [node for node in order if node in cyclic]


class ConsistencyChecker:
    """Checker for naming, style, structure, and reference consistency."""

    def check(self, content: str) -> dict[str, Any]:
        """Check raw or canonical Markdown GNN content."""
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        return self._check_structure(content, _parse_content_structure(content))

    def check_model_data(self, model_data: ModelData) -> dict[str, Any]:
        """Check the canonical dictionary emitted by the GNN parser."""
        return self._check_structure("", _parse_structured_model(model_data))

    def _check_structure(
        self, content: str, structure: _ModelStructure
    ) -> dict[str, Any]:
        naming_result = self._check_naming_conventions(structure)
        style_result = self._check_style_consistency(content, structure)
        structure_result = self._check_structural_integrity(content, structure)
        reference_result = self._check_reference_consistency(structure)

        warnings: list[str] = []
        for result in (
            naming_result,
            style_result,
            structure_result,
            reference_result,
        ):
            warnings.extend(str(warning) for warning in result.get("warnings", []))

        checks = {
            "naming_conventions": naming_result,
            "style_consistency": style_result,
            "structural_integrity": structure_result,
            "reference_consistency": reference_result,
        }
        return {
            "is_consistent": all(
                bool(result.get("is_consistent", False)) for result in checks.values()
            ),
            "warnings": warnings,
            "checks": checks,
        }

    def _check_naming_conventions(self, structure: _ModelStructure) -> dict[str, Any]:
        block_names = structure.block_names
        duplicate_names = _ordered_duplicates(block_names)
        warnings: list[str] = []
        if duplicate_names:
            warnings.append(
                f"Duplicate block names found: {', '.join(duplicate_names)}"
            )

        if structure.kind != "raw":
            return {
                "is_consistent": not warnings,
                "warnings": warnings,
                "dominant_style": None,
                "mixed_styles": False,
                "duplicate_names": duplicate_names,
                "non_descriptive_names": [],
            }

        camel_case = sum(
            1 for name in block_names if name and name[0].islower() and "_" not in name
        )
        snake_case = sum(1 for name in block_names if "_" in name)
        pascal_case = sum(
            1 for name in block_names if name and name[0].isupper() and "_" not in name
        )
        naming_styles = {
            "camelCase": camel_case,
            "snake_case": snake_case,
            "PascalCase": pascal_case,
        }
        populated_styles = [
            style for style, count in naming_styles.items() if count > 0
        ]
        mixed_styles = len(populated_styles) > 1
        if mixed_styles:
            warnings.append(
                f"Inconsistent naming conventions: mix of {', '.join(populated_styles)}"
            )

        non_descriptive_names = [name for name in block_names if len(name) < 3]
        if non_descriptive_names:
            warnings.append(
                f"Non-descriptive block names found: {', '.join(non_descriptive_names)}"
            )
        dominant_style = (
            max(naming_styles, key=lambda style: naming_styles[style])
            if block_names
            else None
        )
        return {
            "is_consistent": not warnings,
            "warnings": warnings,
            "dominant_style": dominant_style,
            "mixed_styles": mixed_styles,
            "duplicate_names": duplicate_names,
            "non_descriptive_names": non_descriptive_names,
        }

    def _check_style_consistency(
        self, content: str, structure: _ModelStructure
    ) -> dict[str, Any]:
        if structure.kind != "raw":
            return {
                "is_consistent": True,
                "warnings": [],
                "indentation_patterns": 0,
                "block_format_styles": [],
                "field_order_consistency": True,
            }

        indentation_patterns = {
            len(line) - len(line.lstrip(" "))
            for line in content.splitlines()
            if line.strip() and line.startswith(" ")
        }
        warnings: list[str] = []
        if len(indentation_patterns) > 2:
            warnings.append(
                f"Inconsistent indentation patterns: {len(indentation_patterns)} different patterns detected"
            )

        block_formats: set[str] = set()
        field_orders: list[tuple[str, ...]] = []
        for block in structure.state_blocks:
            fields = re.findall(r"([A-Za-z]+):", block)
            if fields:
                block_formats.add(
                    "inline" if "\n" not in block.strip() else "multiline"
                )
                field_orders.append(tuple(fields))
        if len(block_formats) > 1:
            warnings.append(
                "Inconsistent block formatting: mix of inline and multiline formats"
            )
        unique_orders = set(field_orders)
        if len(unique_orders) > 1:
            warnings.append("Inconsistent field ordering across blocks")
        return {
            "is_consistent": not warnings,
            "warnings": warnings,
            "indentation_patterns": len(indentation_patterns),
            "block_format_styles": sorted(block_formats),
            "field_order_consistency": len(unique_orders) <= 1,
        }

    def _check_structural_integrity(
        self, content: str, structure: _ModelStructure
    ) -> dict[str, Any]:
        if structure.kind != "raw":
            structural_warnings = list(structure.diagnostics)
            if not structure.block_names:
                _append_once(
                    structural_warnings, "No valid StateSpaceBlock declarations found"
                )
            return {
                "is_consistent": not structural_warnings,
                "warnings": structural_warnings,
                "balanced_braces": True,
                "empty_blocks": 0,
                "empty_connections": 0,
                "missing_state_fields": [],
                "missing_connection_fields": [],
            }

        warnings: list[str] = []
        open_braces = content.count("{")
        close_braces = content.count("}")
        if open_braces != close_braces:
            warnings.append(
                f"Unbalanced braces: {open_braces} opening vs {close_braces} closing"
            )
        empty_blocks = sum(1 for block in structure.state_blocks if not block.strip())
        empty_connections = sum(
            1 for connection in structure.connection_blocks if not connection.strip()
        )
        if empty_blocks:
            warnings.append(f"Empty StateSpaceBlock definitions found: {empty_blocks}")
        if empty_connections:
            warnings.append(f"Empty Connection definitions found: {empty_connections}")

        missing_state_fields = [
            (index, field_name)
            for index, block in enumerate(structure.state_blocks)
            for field_name in ("Name", "Dimensions")
            if _extract_field(block, field_name) is None
        ]
        missing_connection_fields = [
            (index, field_name)
            for index, block in enumerate(structure.connection_blocks)
            for field_name in ("From", "To")
            if _extract_field(block, field_name) is None
        ]
        if missing_state_fields:
            details = ", ".join(
                f"Block {index} missing {field_name}"
                for index, field_name in missing_state_fields
            )
            warnings.append(f"Missing required fields in StateSpaceBlocks: {details}")
        if missing_connection_fields:
            details = ", ".join(
                f"Connection {index} missing {field_name}"
                for index, field_name in missing_connection_fields
            )
            warnings.append(f"Missing required fields in Connections: {details}")
        return {
            "is_consistent": not warnings,
            "warnings": warnings,
            "balanced_braces": open_braces == close_braces,
            "empty_blocks": empty_blocks,
            "empty_connections": empty_connections,
            "missing_state_fields": missing_state_fields,
            "missing_connection_fields": missing_connection_fields,
        }

    def _check_reference_consistency(
        self, structure: _ModelStructure
    ) -> dict[str, Any]:
        warnings: list[str] = []
        known_blocks = set(structure.block_names)
        invalid_references: list[tuple[int, str, str]] = []
        connected_blocks: set[str] = set()
        for connection in structure.connections:
            if connection.source is not None:
                connected_blocks.add(connection.source)
                if connection.source not in known_blocks:
                    invalid_references.append(
                        (connection.index, "From", connection.source)
                    )
            if connection.target is not None:
                connected_blocks.add(connection.target)
                if connection.target not in known_blocks:
                    invalid_references.append(
                        (connection.index, "To", connection.target)
                    )

        if invalid_references:
            details = ", ".join(
                f"Connection {index} references non-existent {field_name} block: '{reference}'"
                for index, field_name, reference in invalid_references
            )
            warnings.append(f"Invalid block references: {details}")

        isolated_blocks = [
            name
            for name in dict.fromkeys(structure.block_names)
            if name not in connected_blocks
        ]
        if isolated_blocks:
            warnings.append(
                f"Isolated blocks with no connections: {', '.join(isolated_blocks)}"
            )

        circular_references = _cycle_nodes(structure.block_names, structure.connections)
        if circular_references:
            warnings.append(
                "Circular dependencies detected in blocks: "
                f"{', '.join(circular_references)}"
            )
        return {
            "is_consistent": not warnings,
            "warnings": warnings,
            "invalid_references": invalid_references,
            "isolated_blocks": isolated_blocks,
            "circular_references": circular_references,
        }


def _model_file_path(model_data: ModelData) -> str:
    value = model_data.get("file_path", "unknown")
    return str(value) if isinstance(value, (str, Path)) else "unknown"


def check_consistency(model_data: ModelInput) -> dict[str, Any]:
    """Check consistency for a GNN path or canonical parsed model mapping."""
    try:
        checker = ConsistencyChecker()
        if isinstance(model_data, Mapping):
            file_path = _model_file_path(model_data)
            consistency_result = checker.check_model_data(model_data)
        elif isinstance(model_data, (str, Path)):
            path = Path(model_data)
            file_path = str(path)
            consistency_result = checker.check(path.read_text(encoding="utf-8"))
        else:
            raise TypeError(
                f"model_data must be a path or mapping, got {type(model_data).__name__}"
            )

        return {
            "file_path": file_path,
            "file_name": Path(file_path).name if file_path != "unknown" else "unknown",
            "consistent": consistency_result["is_consistent"],
            "warnings": consistency_result["warnings"],
            "checks": consistency_result["checks"],
            "consistency_score": _calculate_consistency_score(consistency_result),
            "recovery": False,
        }
    except Exception as error:
        file_path = (
            str(model_data) if isinstance(model_data, (str, Path)) else "unknown"
        )
        return {
            "status": "error",
            "file_path": file_path,
            "file_name": Path(file_path).name if file_path != "unknown" else "unknown",
            "error": str(error),
            "consistent": False,
            "warnings": [str(error)],
            "checks": {},
            "consistency_score": 0.0,
            "recovery": True,
        }


def _calculate_consistency_score(consistency_result: Mapping[str, Any]) -> float:
    """Calculate a bounded score from aggregate and structured findings."""
    score = 1.0 - len(consistency_result.get("warnings", [])) * 0.1
    checks = consistency_result.get("checks", {})
    reference_result = (
        checks.get("reference_consistency", {}) if isinstance(checks, Mapping) else {}
    )
    if isinstance(reference_result, Mapping):
        score -= len(reference_result.get("invalid_references", [])) * 0.2
        score -= len(reference_result.get("isolated_blocks", [])) * 0.1
        score -= len(reference_result.get("circular_references", [])) * 0.3
    return max(0.0, min(1.0, score))
