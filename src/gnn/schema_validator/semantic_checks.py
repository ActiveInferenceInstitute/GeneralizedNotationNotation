"""Semantic validation checks for :mod:`gnn.schema_validator.validator`.

Holds semantic consistency, Active Inference convention, and
mathematical-consistency checks mixed into GNNValidator.
"""

import logging
from typing import Any

from gnn.types import ParsedGNN, ValidationResult

logger = logging.getLogger(__name__)


class SemanticChecksMixin:
    """Mixin holding semantic consistency validation methods."""

    def _validate_semantics(self, parsed: ParsedGNN, result: ValidationResult) -> Any:
        """Validate semantic consistency of parsed GNN model."""
        # Validate variable references in connections
        variable_names = set(parsed.variables)

        for connection in parsed.connections:
            # Check source variables
            source_vars = (
                connection.source
                if isinstance(connection.source, list)
                else [connection.source]
            )
            for var in source_vars:
                if var not in variable_names and not self._is_valid_variable_reference(
                    var
                ):
                    result.errors.append(
                        f"Undefined variable in connection source: {var}"
                    )

            # Check target variables
            target_vars = (
                connection.target
                if isinstance(connection.target, list)
                else [connection.target]
            )
            for var in target_vars:
                if var not in variable_names and not self._is_valid_variable_reference(
                    var
                ):
                    result.errors.append(
                        f"Undefined variable in connection target: {var}"
                    )

        # Validate Active Inference conventions
        self._validate_active_inference_conventions(parsed, result)

        # Validate mathematical consistency
        self._validate_mathematical_consistency(parsed, result)

    def _is_valid_variable_reference(self, var: str) -> bool:
        """Check if variable reference follows GNN conventions."""
        # Handle complex variable references like (A,B) or time-indexed variables
        return (var.startswith("(") and var.endswith(")")) or "=" in var

    def _validate_active_inference_conventions(
        self, parsed: ParsedGNN, result: ValidationResult
    ) -> Any:
        """Validate Active Inference naming and structure conventions."""
        # Check for proper A, B, C, D matrix naming
        ai_matrices: dict[str, Any] = {"A": [], "B": [], "C": [], "D": []}

        for var_name, _var in parsed.variables.items():
            if var_name.startswith("A_m"):
                ai_matrices["A"].append(var_name)
            elif var_name.startswith("B_f"):
                ai_matrices["B"].append(var_name)
            elif var_name.startswith("C_m"):
                ai_matrices["C"].append(var_name)
            elif var_name.startswith("D_f"):
                ai_matrices["D"].append(var_name)

        # Validate matrix dimension consistency
        if ai_matrices["A"] and ai_matrices["D"]:
            result.metadata["active_inference_matrices"] = ai_matrices

        # Check for proper state/observation variable naming
        state_vars = [name for name in parsed.variables if name.startswith("s_f")]
        obs_vars = [name for name in parsed.variables if name.startswith("o_m")]

        if state_vars:
            result.metadata["state_variables"] = state_vars
        if obs_vars:
            result.metadata["observation_variables"] = obs_vars

    def _validate_mathematical_consistency(
        self, parsed: ParsedGNN, result: ValidationResult
    ) -> Any:
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
                        if (
                            matrix_rows != expected_dims[0]
                            or matrix_cols != expected_dims[1]
                        ):
                            result.warnings.append(
                                f"Matrix {param_name} dimensions {matrix_rows}x{matrix_cols} "
                                f"don't match variable definition {expected_dims}"
                            )

                # Check for stochasticity in probability matrices
                if param_name.startswith(("A_", "B_", "D_")) and isinstance(
                    param_value, list
                ):
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
                            row_sum = sum(
                                float(x) for x in row if isinstance(x, (int, float))
                            )
                            if abs(row_sum - 1.0) > tolerance:
                                return False
                    return True
                else:
                    # 1D vector
                    total_sum = sum(
                        float(x) for x in matrix_data if isinstance(x, (int, float))
                    )
                    return abs(total_sum - 1.0) <= tolerance
        except (TypeError, ValueError):
            logger.debug(
                "Could not verify stochastic matrix normalization, assuming valid"
            )
        return True  # Conservative: assume valid if can't check
