"""Named answer bundles preserve explicit task coverage and reject ambiguity."""

from __future__ import annotations

import pytest

from gnn.llm.prompt_batch import AnswerLimitReached, parse_answers, response_schema


@pytest.mark.parametrize(
    "content",
    [
        "{}",
        '{"one":"answer"}',
        '{"one":"a","two":"b","extra":"c"}',
        '{"one":"a","two":null}',
        '{"one":"a","two":" "}',
        '{"one":"a","one":"b","two":"c"}',
        "not JSON",
    ],
)
def test_missing_extra_empty_or_ambiguous_answers_refuse(content: str) -> None:
    with pytest.raises(ValueError):
        parse_answers(content, ["one", "two"])


def test_exact_named_answers_and_schema() -> None:
    labels = ["one", "two"]
    assert parse_answers('{"two":"second","one":"first"}', labels) == {
        "one": "first",
        "two": "second",
    }
    schema = response_schema(labels)
    assert schema["required"] == labels
    assert set(schema["properties"]) == set(labels)
    assert schema["additionalProperties"] is False


def test_answer_bound_counts_unicode_characters_and_refuses_overflow() -> None:
    schema = response_schema(["answer"], max_chars=4)
    assert schema["properties"]["answer"]["maxLength"] == 4
    assert parse_answers('{"answer":"αβγ"}', ["answer"], max_chars=4) == {
        "answer": "αβγ"
    }
    with pytest.raises(AnswerLimitReached):
        parse_answers('{"answer":"αβγδ"}', ["answer"], max_chars=4)
    with pytest.raises(AnswerLimitReached):
        parse_answers('{"answer":"αβγδε"}', ["answer"], max_chars=4)


@pytest.mark.parametrize("limit", [True, 0, -1, 2.5, "600"])
def test_invalid_answer_character_bounds_refuse(limit: object) -> None:
    with pytest.raises(ValueError):
        response_schema(["answer"], max_chars=limit)  # type: ignore[arg-type]
