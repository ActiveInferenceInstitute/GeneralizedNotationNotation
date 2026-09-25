#!/usr/bin/env python3
"""
Shape and error-recording support for the POMDP extractor.

Mechanical extraction from ``gnn.extract.pomdp_extractor`` (M-01 band split):
``POMDPExtractorSupportMixin`` holds the verbatim nested-shape and structured
error-recording methods shared by every extraction stage. The topic mixins in
``pomdp_sections.py``, ``pomdp_parameters.py``, and ``pomdp_orientation.py``
inherit from it, and ``POMDPExtractor`` composes them, so every attribute and
method path is unchanged.

This module is stdlib-only at import time.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .pomdp_state import GNNExtractionError, _shape_of

if TYPE_CHECKING:
    import logging


class POMDPExtractorSupportMixin:
    """Verbatim shape and error-recording methods moved from ``POMDPExtractor``."""

    if TYPE_CHECKING:
        # Shared ``POMDPExtractor`` state the moved bodies touch. Declared
        # annotation-only so the mixin type-checks standalone; the real
        # values are created by ``POMDPExtractor.__init__`` in
        # ``pomdp_extractor.py``.
        logger: logging.Logger
        _errors: List[GNNExtractionError]
        _on_error: str
        _parse_failures: Dict[str, Dict[str, Any]]
        _section_line_offset: int

    def _nested_shape(self, value: Any) -> List[int]:
        """Return a best-effort shape for nested Python matrix data."""
        return _shape_of(value)

    def _record_error(
        self,
        code: str,
        message: str,
        line: Optional[int] = None,
        section: Optional[str] = None,
    ) -> GNNExtractionError:
        """Record a structured fault; raise immediately in 'raise' mode."""
        error = GNNExtractionError(
            code=code, message=message, line=line, section=section
        )
        self._errors.append(error)
        if self._on_error == "raise":
            raise error
        if error.severity == "error":
            self.logger.error("structured error: %s", error)
        else:
            self.logger.warning("structured warning: %s", error)
        return error

    def _record_parameter_failure(
        self,
        param_name: str,
        exc: BaseException,
        line_no: Optional[int] = None,
    ) -> None:
        """Record a failed parameter parse (GNN-E006); never silently dropped."""
        if isinstance(exc, (ValueError, SyntaxError)):
            code = "GNN-E006"
        else:
            code = "GNN-E006"
        line = (
            self._section_line_offset + line_no
            if self._section_line_offset and line_no
            else line_no
        )
        self._parse_failures[param_name] = {
            "code": code,
            "message": f"{param_name}: {exc}",
        }
        self.logger.warning("Failed to parse parameter %s: %s", param_name, exc)
        self._record_error(
            code,
            f"failed to parse parameter '{param_name}': {exc}",
            line=line,
            section="InitialParameterization",
        )

    def _section_line_offset_for(self, content: str, section: str) -> int:
        """File-absolute 1-based line number of a section header (0 if absent)."""
        for index, raw_line in enumerate(content.split("\n"), start=1):
            if raw_line.strip().lower() == f"## {section.lower()}":
                return index
        return 0
