"""Validation-level resolution for :mod:`gnn.schema_validator.validator`.

Owns ``_LEVEL_RANKS`` and the string-to-enum level resolution helpers
(``_resolve_level``, ``_level_rank``) mixed into GNNValidator.
"""

import logging
from typing import Union

from gnn.types import ValidationLevel

logger = logging.getLogger(__name__)

_LEVEL_RANKS: dict[ValidationLevel, int] = {
    ValidationLevel.BASIC: 10,
    ValidationLevel.STANDARD: 20,
    ValidationLevel.STRICT: 30,
    ValidationLevel.RESEARCH: 40,
    ValidationLevel.ROUND_TRIP: 50,
}


class LevelResolverMixin:
    """Mixin providing validation-level resolution helpers."""

    def _resolve_level(self, level: str) -> ValidationLevel:
        """Resolve a string validation level to its enum member.

        Accepted forms:

        - an enum value (``"basic"``, ``"standard"``, ``"strict"``,
          ``"research"``, ``"round_trip"``)
        - an enum name (``"BASIC"``, ``"STANDARD"``, ...), matched exactly
          or after ``upper()`` (so ``"Standard"`` also resolves; note no
          member has ``name == value``, so value lookup always wins first)

        Anything else raises ``ValueError`` listing the accepted forms;
        unknown levels never silently skip validation.
        """
        try:
            return ValidationLevel(level)
        except ValueError:
            pass
        try:
            return ValidationLevel[level.upper()]
        except KeyError as e:
            accepted = ", ".join(
                f"{member.name!r} ({member.value!r})" for member in ValidationLevel
            )
            raise ValueError(
                f"Unknown validation level {level!r}; accepted forms: {accepted}"
            ) from e

    def _level_rank(self, level: Union[ValidationLevel, str]) -> int:
        """Map validation level to an integer rank for safe comparisons.

        Accepts a ``ValidationLevel`` member, a string equal to an enum
        value (e.g. ``"strict"``), or a string equal to an enum name
        (e.g. ``"STRICT"``, case-insensitive); see :meth:`_resolve_level`
        for the full accepted forms. Anything else raises ``ValueError``.
        """
        if isinstance(level, ValidationLevel):
            return _LEVEL_RANKS.get(level, 0)
        return _LEVEL_RANKS[self._resolve_level(level)]
