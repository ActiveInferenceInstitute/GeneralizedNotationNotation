"""Deterministic tests for the llm.prompts registry.

Regression: every PromptType member must have a prompt config —
COMPARE_MODELS and VALIDATE_SYNTAX previously raised ValueError from
get_prompt despite being advertised by get_all_prompt_types.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm.prompts import (
    GNN_ANALYSIS_PROMPTS,
    PromptType,
    get_all_prompt_types,
    get_default_prompt_sequence,
    get_prompt,
    get_prompt_title,
)

pytestmark = pytest.mark.unit

SAMPLE = "## ModelName\nMyModel\n## StateSpaceBlock\ns[3,1]"


def test_registry_covers_every_prompt_type() -> None:
    assert set(GNN_ANALYSIS_PROMPTS) == set(PromptType)


@pytest.mark.parametrize("prompt_type", list(PromptType), ids=lambda pt: pt.value)
def test_every_type_produces_valid_prompt(prompt_type: PromptType) -> None:
    config = get_prompt(prompt_type, SAMPLE)
    assert config["system_message"]
    assert SAMPLE in config["user_prompt"]
    assert config["expected_output"] == "markdown"
    assert config["max_tokens"] > 0


def test_get_prompt_returns_shallow_copy() -> None:
    first = get_prompt(PromptType.EXPLAIN_MODEL, SAMPLE)
    first["user_prompt"] = "mutated"
    second = get_prompt(PromptType.EXPLAIN_MODEL, SAMPLE)
    assert SAMPLE in second["user_prompt"]


def test_get_prompt_title_nonempty_for_all() -> None:
    for prompt_type in PromptType:
        assert get_prompt_title(prompt_type)


def test_get_all_prompt_types_matches_enum() -> None:
    assert get_all_prompt_types() == list(PromptType)


def test_default_prompt_sequence_is_subset() -> None:
    sequence = get_default_prompt_sequence()
    assert sequence
    assert set(sequence) <= set(PromptType)
