#!/usr/bin/env python3
"""
Intelligent Remediation — Auto-suggest fixes for contract violations.

Provides:
  - suggest_fix(): generates a targeted diff for a ContractViolation
  - RemediationPlan: structured fix proposal
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RemediationPlan:
    """A proposed fix for a contract violation."""
    violation_summary: str
    suggested_code: str
    insertion_point: str   # "top", "after_imports", "before_return"
    confidence: float      # 0.0–1.0
    auto_apply: bool = False

    def to_diff(self) -> str:
        """Render as a unified diff snippet."""
        return f"+ {self.suggested_code}  # Auto-remediation: {self.violation_summary}"


# Known fix templates per violation type
_FIX_TEMPLATES = {
    ("pymdp", "import"): {
        "code": "import numpy as np\nfrom pymdp import utils\nfrom pymdp.agent import Agent",
        "point": "top",
        "confidence": 0.9,
    },
    ("pymdp", "variable_A"): {
        "code": "A = utils.random_A_matrix(num_obs, num_states)  # Likelihood matrix",
        "point": "after_imports",
        "confidence": 0.7,
    },
    ("pymdp", "variable_B"): {
        "code": "B = utils.random_B_matrix(num_states, num_actions)  # Transition matrix",
        "point": "after_imports",
        "confidence": 0.7,
    },
    ("jax", "import"): {
        "code": "import jax\nimport jax.numpy as jnp",
        "point": "top",
        "confidence": 0.9,
    },
    ("jax", "variable_A"): {
        "code": "A = jnp.eye(num_states)  # Likelihood matrix placeholder",
        "point": "after_imports",
        "confidence": 0.6,
    },
    ("jax", "variable_B"): {
        "code": "B = jnp.zeros((num_states, num_states, num_actions))  # Transition matrix placeholder",
        "point": "after_imports",
        "confidence": 0.6,
    },
    ("rxinfer", "import"): {
        "code": "using RxInfer",
        "point": "top",
        "confidence": 0.9,
    },
}


def suggest_fix(violation) -> Optional[RemediationPlan]:
    """
    Propose a fix for a given ContractViolation.

    Args:
        violation: A ContractViolation from contracts.py.

    Returns:
        RemediationPlan if a fix is known, None otherwise.
    """
    key = (violation.framework.lower(), violation.field)
    template = _FIX_TEMPLATES.get(key)

    if template is None:
        logger.debug(f"No remediation template for {key}")
        return None

    plan = RemediationPlan(
        violation_summary=str(violation),
        suggested_code=template["code"],
        insertion_point=template["point"],
        confidence=template["confidence"],
        auto_apply=template["confidence"] >= 0.85,
    )

    logger.info(
        f"💡 Remediation suggested for {violation.field} "
        f"(confidence: {plan.confidence:.0%}, auto: {plan.auto_apply})"
    )
    return plan
