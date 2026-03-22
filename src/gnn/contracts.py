#!/usr/bin/env python3
"""
GNN Output Contracts — Framework-specific validation for rendered code.

Validates that rendered output (pymdp, rxinfer, jax) contains required
structural elements: matrix shapes, imports, function calls.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ContractViolation:
    """A violation of a framework output contract."""
    framework: str
    field: str
    expected: str
    actual: str
    file_path: Optional[str] = None
    line: Optional[int] = None

    def __str__(self) -> str:
        loc = f" [{self.file_path}:{self.line}]" if self.file_path else ""
        return f"[{self.framework}] {self.field}: expected {self.expected}, got {self.actual}{loc}"


# ── Framework Contracts ──────────────────────────────────────────────────────────

CONTRACTS: Dict[str, Dict[str, Any]] = {
    "pymdp": {
        "required_imports": ["numpy", "pymdp"],
        "required_variables": ["A", "B"],
        "matrix_patterns": [
            r"\bA\s*=\s*",   # Likelihood matrix assignment
            r"\bB\s*=\s*",   # Transition matrix assignment
        ],
        "optional_variables": ["C", "D", "E"],
    },
    "rxinfer": {
        "required_imports": ["RxInfer"],
        "required_variables": [],
        "matrix_patterns": [
            r"@model",       # Model macro
        ],
        "optional_variables": [],
    },
    "jax": {
        "required_imports": ["jax", "jax.numpy"],
        "required_variables": ["A", "B"],
        "matrix_patterns": [
            r"jnp\.\w+",    # JAX numpy calls
        ],
        "optional_variables": ["C", "D"],
    },
}


def validate_rendered_output(
    code: str,
    framework: str,
    *,
    file_path: Optional[str] = None,
) -> List[ContractViolation]:
    """
    Validate rendered code against framework-specific contract.

    Args:
        code: Generated source code string.
        framework: Target framework name (pymdp, rxinfer, jax).
        file_path: Optional source file for error reporting.

    Returns:
        List of ContractViolation for any issues found.
    """
    contract = CONTRACTS.get(framework.lower())
    if not contract:
        raise ValueError(
            f"No contract defined for framework '{framework}'. "
            f"Known frameworks: {', '.join(sorted(CONTRACTS))}"
        )

    violations = []

    # Check required imports
    for imp in contract.get("required_imports", []):
        if imp not in code:
            violations.append(ContractViolation(
                framework=framework,
                field="import",
                expected=f"import containing '{imp}'",
                actual="not found",
                file_path=file_path,
            ))

    # Check required variable assignments
    for var in contract.get("required_variables", []):
        pattern = rf"\b{re.escape(var)}\s*="
        if not re.search(pattern, code):
            violations.append(ContractViolation(
                framework=framework,
                field=f"variable_{var}",
                expected=f"assignment to '{var}'",
                actual="not found",
                file_path=file_path,
            ))

    # Check matrix patterns
    for pattern in contract.get("matrix_patterns", []):
        if not re.search(pattern, code):
            violations.append(ContractViolation(
                framework=framework,
                field="pattern",
                expected=f"pattern matching '{pattern}'",
                actual="not found",
                file_path=file_path,
            ))

    if violations:
        logger.warning(f"⚠️ {len(violations)} contract violation(s) for {framework}")
    else:
        logger.debug(f"✅ {framework} contract satisfied")

    return violations
