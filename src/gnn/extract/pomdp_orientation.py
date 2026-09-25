#!/usr/bin/env python3
"""
B-orientation metadata detection for the POMDP extractor.

Mechanical extraction from ``gnn.extract.pomdp_extractor`` (M-01 band split):
``POMDPOrientationMixin`` holds the verbatim B-order detection and orientation
metadata methods — declared order parsing, claimed slice-convention parsing,
detected-order detection, and the composite orientation analysis. Detection
only: no method here re-orients data.

This module is stdlib-only at import time.
"""

import re
from typing import Any, Dict, List, Optional

from .pomdp_state import CANONICAL_B_ORDER
from .pomdp_support import POMDPExtractorSupportMixin


class POMDPOrientationMixin(POMDPExtractorSupportMixin):
    """Verbatim B-orientation methods moved from ``POMDPExtractor``."""

    # --- B-orientation metadata (detection only; never re-orients data) ---

    _AXIS_ALIASES = {
        "next_state": "next_state",
        "next": "next_state",
        "s_next": "next_state",
        "states_next": "next_state",
        "s'": "next_state",
        "previous_state": "previous_state",
        "prev_state": "previous_state",
        "previous": "previous_state",
        "prev": "previous_state",
        "s_prev": "previous_state",
        "states_previous": "previous_state",
        "action": "action",
        "actions": "action",
        "u": "action",
    }

    def _parse_declared_b_order(self, state_space_block: str) -> Optional[List[str]]:
        """Parse the declared B axis order from the StateSpaceBlock comment."""
        for line in state_space_block.split("\n"):
            if "B[" not in line:
                continue
            for match in re.finditer(r"B\[([^\]]+)\]", line):
                axes = [part.strip().lower() for part in match.group(1).split(",")]
                order: List[str] = []
                for axis in axes:
                    for alias, canonical in self._AXIS_ALIASES.items():
                        if axis == alias or axis.startswith(alias):
                            if canonical not in order:
                                order.append(canonical)
                            break
                if len(order) == 3:
                    return order
        return None

    def _parse_claimed_slice_convention(self, parameterization: str) -> Optional[str]:
        """Parse the claimed per-slice convention from the InitialParameterization B comment."""
        near_b = re.search(
            r"#\s*B:.*?(?=\n(?:[A-Za-zπ_]\w*\s*=)|$)",
            parameterization,
            re.DOTALL,
        )
        text = near_b.group(0).lower() if near_b else parameterization.lower()
        if ("rows are previous" in text or "rows as previous" in text) and (
            "columns are next" in text or "columns as next" in text
        ):
            return "rows_previous_columns_next"
        if "rows are next" in text or "rows as next" in text:
            return "rows_next_columns_previous"
        return None

    def _detect_b_order(self, b_matrix: Any) -> Optional[List[str]]:
        """Detect the stored tensor's axis order from stochasticity sums.

        Evidence tests over the stored (as-written) tensor T[d0][d1][d2]:
        - Doubly stochastic (dominant): every slice has row sums AND column
          sums = 1 (permutation-style data). Ambiguous — never decisive, and
          never a contradiction by itself.
        - H2 (canonical): sum over the OUTER axis at each (i, j) position = 1
          (law of total probability over next_state) -> stored is
          (next_state, previous_state, action).
        - H1 (action-outer): every slice is row-stochastic (row sums = 1) ->
          stored is (action, previous_state, next_state).
        Neither test decisive -> None.
        """
        shape = self._nested_shape(b_matrix)
        if len(shape) != 3 or 0 in shape:
            return None
        try:
            rows_stochastic = all(
                abs(sum(float(v) for v in row) - 1.0) <= 1e-6
                for slice_ in b_matrix
                for row in slice_
            )
            cols_stochastic = all(
                abs(sum(float(row[j]) for row in slice_) - 1.0) <= 1e-6
                for slice_ in b_matrix
                for j in range(len(slice_[0]))
            )
            doubly_stochastic = rows_stochastic and cols_stochastic
            h2 = all(
                abs(sum(float(s[i][j]) for s in b_matrix) - 1.0) <= 1e-6
                for i in range(shape[1])
                for j in range(shape[2])
            )
        except (TypeError, ValueError, IndexError):
            return None

        if doubly_stochastic:
            return None  # ambiguous — never decisive, never a contradiction
        if h2:
            return list(CANONICAL_B_ORDER)  # (next, prev, action)
        if rows_stochastic:
            return ["action", "previous_state", "next_state"]
        return None

    def _analyze_b_orientation(
        self,
        state_space_block: str,
        parameterization: str,
        b_matrix: Any,
    ) -> Dict[str, Any]:
        """Produce B-orientation metadata for matrix_provenance['B']."""
        declared = self._parse_declared_b_order(state_space_block)
        claimed = self._parse_claimed_slice_convention(parameterization)
        detected = self._detect_b_order(b_matrix)
        contradiction = False
        reason: Optional[str] = None

        # Contradiction requires decisive data (doubly-stochastic/ambiguous
        # data is NEVER a contradiction by itself): the detected orientation
        # must disagree with the declared axis order (or, absent a declaration,
        # with the claimed convention / canonical order).
        if detected is not None:
            reference = declared
            reference_label = "declared"
            if reference is None:
                reference = (
                    list(CANONICAL_B_ORDER)
                    if claimed is None
                    else (
                        ["action", "previous_state", "next_state"]
                        if claimed == "rows_previous_columns_next"
                        else list(CANONICAL_B_ORDER)
                    )
                )
                reference_label = "claimed" if claimed else "canonical"
            if detected != reference:
                contradiction = True
                reason = (
                    f"detected B orientation {detected} contradicts the "
                    f"{reference_label} order {reference}"
                )

        return {
            "declared_order": declared or list(CANONICAL_B_ORDER),
            "declared_order_explicit": declared is not None,
            "claimed_slice_convention": claimed,
            "detected_order": detected,
            "canonical_order": list(CANONICAL_B_ORDER),
            "contradiction": contradiction,
            "reason": reason,
        }
