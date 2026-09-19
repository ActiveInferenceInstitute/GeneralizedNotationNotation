"""GNN completion vocabulary — the shared source both LSP servers serve from.

Pygls-free by design: pure data + plain functions, so the pygls server
(:mod:`gnn.lsp`) and the hand-rolled CLI server (:mod:`gnn.cli.lsp`) answer
``textDocument/completion`` from one vocabulary with one context model.

Section vocabulary is imported from :mod:`gnn.schemas.section_contract` —
never duplicated here. Two vocabularies have no code-level canonical list
(renderers read them through scattered ``model_params.get(...)`` calls and
structural kind-derivation), so they are pinned here with doc citations:

- ``GNN_SECTION_VALUES`` — docs/gnn/gnn_syntax.md:18 (``## GNNSection`` value,
  e.g. ``ActInfPOMDP``), :169-170 (``ActInfContinuous`` example), :232-233
  (model kind derives structurally from the ``## GNNSection`` value).
- ``MODEL_PARAMETER_KEYS`` — docs/gnn/gnn_syntax.md:46-62, "``ModelParameters``
  keys read by renderers".

Dtype values come from the code-level canonical source: the ``dtype`` enum in
``GNN_MODEL_SCHEMA`` (gnn/schema/parser.py, re-exported by :mod:`gnn.schema`).

Completion items are plain dicts with ``label``/``kind``/``detail``/
``insert_text`` keys; ``kind`` is the numeric LSP CompletionItemKind so both
the pygls adapter (``CompletionItemKind(int)``) and the raw JSON-RPC path can
use it directly.
"""

from typing import Any

from gnn.schema import GNN_MODEL_SCHEMA
from gnn.schemas.section_contract import (
    CANONICAL_GNN_SECTIONS,
    OPTIONAL_SECTIONS,
    REQUIRED_SECTIONS,
)

__all__ = [
    "MODEL_PARAMETER_KEYS",
    "all_section_completions",
    "completion_context",
    "context_completions",
    "dtype_completions",
    "gnn_section_value_completions",
    "model_parameter_key_completions",
]

# Model kind values declared under `## GNNSection`. No code-level canonical
# list exists; pinned per docs/gnn/gnn_syntax.md:18,169-170,232-233.
GNN_SECTION_VALUES: tuple[str, ...] = (
    "ActInfPOMDP",
    "ActInfPOMDP_MultiAgent",
    "ActInfContinuous",
)

# `key: value` parameter names read by renderers. No code-level canonical
# list exists; pinned per docs/gnn/gnn_syntax.md:46-62.
MODEL_PARAMETER_KEYS: tuple[str, ...] = (
    "num_hidden_states",
    "num_obs",
    "num_actions",
    "num_timesteps",
    "num_modalities",
    "learning_rate",
    "num_factors",
    "nr_agents",
    "inference_mode",
    "inference_iterations",
)

# Code-level canonical source for dtype values: the `dtype` enum in
# GNN_MODEL_SCHEMA (gnn/schema/parser.py).
_DTYPE_VALUES: tuple[str, ...] = tuple(
    GNN_MODEL_SCHEMA["properties"]["state_space"]["items"]["properties"]["dtype"][
        "enum"
    ]
)

# Numeric LSP CompletionItemKind values (lsprotocol CompletionItemKind).
_KIND_SECTION = 7  # Class
_KIND_PARAMETER_KEY = 5  # Field
_KIND_VALUE = 12  # Value
_KIND_DTYPE = 25  # TypeParameter


def all_section_completions() -> list[dict[str, Any]]:
    """Completion items for all canonical `## ` section headers, declared order."""
    items: list[dict[str, Any]] = []
    for name in CANONICAL_GNN_SECTIONS:
        if name in REQUIRED_SECTIONS:
            detail = "Required GNN section"
        elif name in OPTIONAL_SECTIONS:
            detail = "Optional GNN section"
        else:
            detail = "GNN section"
        items.append(
            {
                "label": name,
                "kind": _KIND_SECTION,
                "detail": detail,
                "insert_text": name,
            }
        )
    return items


def gnn_section_value_completions() -> list[dict[str, Any]]:
    """Completion items for the model-kind value under `## GNNSection`."""
    return [
        {
            "label": value,
            "kind": _KIND_VALUE,
            "detail": "Model kind declared by ## GNNSection",
            "insert_text": value,
        }
        for value in GNN_SECTION_VALUES
    ]


def model_parameter_key_completions() -> list[dict[str, Any]]:
    """Completion items for `key: value` lines under `## ModelParameters`."""
    return [
        {
            "label": key,
            "kind": _KIND_PARAMETER_KEY,
            "detail": "ModelParameters key (key: value)",
            "insert_text": f"{key}: ",
        }
        for key in MODEL_PARAMETER_KEYS
    ]


def dtype_completions() -> list[dict[str, Any]]:
    """Completion items for StateSpaceBlock `type=` declarations."""
    return [
        {
            "label": value,
            "kind": _KIND_DTYPE,
            "detail": "State-space variable dtype",
            "insert_text": value,
        }
        for value in _DTYPE_VALUES
    ]


def completion_context(
    text: str, line_no: int, character: int
) -> tuple[str, bool, bool]:
    """Extract ``(line_prefix, in_model_parameters, in_gnn_section)``.

    ``line_prefix`` is the cursor line truncated at ``character``. The section
    flags reflect the closest `## ` header at or before the cursor line, using
    the exact header normalization of ``MarkdownGNNParser._split_into_sections``
    (``line.strip().startswith("## ")`` then ``line.strip()[3:].strip()``).
    """
    lines = text.splitlines()
    if 0 <= line_no < len(lines):
        line = lines[line_no]
        prefix = line[: max(0, min(character, len(line)))]
    else:
        prefix = ""

    in_model_parameters = False
    in_gnn_section = False
    for i in range(line_no, -1, -1):
        if i >= len(lines):
            continue
        stripped = lines[i].strip()
        if stripped.startswith("## "):
            header = stripped[3:].strip()
            in_model_parameters = header == "ModelParameters"
            in_gnn_section = header == "GNNSection"
            break

    return prefix, in_model_parameters, in_gnn_section


def context_completions(
    line_prefix: str,
    *,
    in_model_parameters: bool = False,
    in_gnn_section: bool = False,
) -> list[dict[str, Any]]:
    """Select completion items for the cursor context.

    Simple deterministic rules, in order:
    1. line starts with `## `             -> all canonical section headers
    2. inside `## GNNSection`             -> model-kind values
    3. line contains ``type=``/``dtype=`` -> dtype values
    4. inside `## ModelParameters`        -> parameter keys
    5. otherwise                          -> all canonical section headers

    Leading whitespace is ignored for rule 1 (mirroring the markdown
    parser's header normalization); trailing whitespace is significant
    because the prefix is cursor-truncated mid-typing (e.g. exactly
    ``"## "`` while starting a header).
    """
    if line_prefix.lstrip().startswith("## "):
        return all_section_completions()
    if in_gnn_section:
        return gnn_section_value_completions()
    if "type=" in line_prefix or "dtype=" in line_prefix:
        return dtype_completions()
    if in_model_parameters:
        return model_parameter_key_completions()
    return all_section_completions()
