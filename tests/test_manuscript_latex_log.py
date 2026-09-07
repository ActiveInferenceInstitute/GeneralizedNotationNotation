"""The LaTeX log's ``Infinite glue shrinkage`` warnings, counted and bounded.

Four of these warnings ship in every render of this manuscript, and they have
been re-reported as an open defect twice. This module ends that: it pins the
*count* (against a probe that does not miss wrapped messages) and the *bound*
that explains them (at most one per paragraph-column ``longtable``).

The finding itself, with the experiments that established it, is written up in
``manuscript/AGENTS.md`` under "Known benign LaTeX diagnostics".

Both artifacts these tests read are tracked, so they are asserted present rather
than skipped over; ``.gitignore`` carries a named exception for the log.

Two facts these tests exist to protect:

* TeX breaks its own log lines and will split the message mid-word, so
  ``grep -c 'Infinite glue'`` under-counts. On the log at the time of writing it
  returned 3 against a true count of 4. Every probe here joins lines first.
* ``\\LT@start`` (``longtable.sty``) runs once per ``longtable``, and its
  ``\\vsplit`` is the only ``\\vsplit`` reachable from ``\\endlongtable``, so one
  table can contribute at most one message. A count above the number of
  paragraph-column tables means something new is emitting them and the write-up
  no longer explains the log.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = REPO_ROOT / "output" / "pdf" / "_combined_manuscript.log"
TEX_PATH = REPO_ROOT / "output" / "pdf" / "_combined_manuscript.tex"

MESSAGE = "Infinite glue shrinkage found in box being split"

# A `longtable` preamble runs from \begin{longtable} to whichever of \caption or
# \toprule opens the table body; a `p{...}` there is a paragraph column.
_LONGTABLE_PREAMBLE_RE = re.compile(
    r"\\begin\{longtable\}(.*?)(?:\\caption|\\toprule)", re.DOTALL
)


def count_message(log_text: str) -> int:
    """Occurrences of *MESSAGE*, counting ones TeX wrapped across a line break."""
    return log_text.replace("\n", "").count(MESSAGE)


def count_paragraph_column_longtables(tex_text: str) -> int:
    """``longtable`` environments whose column preamble declares a ``p{...}``."""
    return sum(
        1 for preamble in _LONGTABLE_PREAMBLE_RE.findall(tex_text) if "p{" in preamble
    )


# --- the counting probe -----------------------------------------------------


def test_counting_survives_a_message_tex_wrapped_mid_word() -> None:
    """Verbatim from ``_combined_manuscript.log``: TeX split this one in two."""
    wrapped = "ignored: In\nfinite glue shrinkage found in box being split [25]\n"
    assert MESSAGE not in wrapped, "the naive scan must miss this line"
    assert count_message(wrapped) == 1


def test_counting_finds_two_messages_on_one_line() -> None:
    line = f"ignored: {MESSAGE} [7] [8]\nignored: {MESSAGE} [14]\n"
    assert count_message(line) == 2


def test_counting_reports_zero_on_a_clean_log() -> None:
    assert count_message("[1] [2] [3]\nOutput written on x.pdf (3 pages).\n") == 0


# --- the column-spec probe --------------------------------------------------


def test_paragraph_column_longtables_are_told_apart_from_plain_ones() -> None:
    tex = (
        "\\begin{longtable}[]{@{}lll@{}}\n\\caption{plain}\n\\end{longtable}\n"
        "\\begin{longtable}[]{@{}\n"
        "  >{\\raggedright\\arraybackslash}p{(\\linewidth) * \\real{0.5}}@{}}\n"
        "\\caption{wide}\n\\end{longtable}\n"
    )
    assert count_paragraph_column_longtables(tex) == 1


# --- the shipped artifact ---------------------------------------------------


def _shipped() -> tuple[str, str]:
    """The committed render artifacts, or an explicit failure.

    No skip guard: both files are tracked (``.gitignore`` carries an explicit
    exception for the log), so their absence means the committed render is
    incomplete, and the repo's zero-skip contract
    (``src/tests/test_zero_skip_contracts.py``) forbids hiding that behind a
    skip. A skip here also silently disarmed the only check on the shipped
    LaTeX diagnostics.
    """
    missing = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in (LOG_PATH, TEX_PATH)
        if not path.exists()
    ]
    assert not missing, (
        f"committed render artifacts missing: {missing}. These are tracked "
        "files; regenerate them with the template's stage_03_render and commit "
        "the result."
    )
    return (
        LOG_PATH.read_text(encoding="utf-8", errors="replace"),
        TEX_PATH.read_text(encoding="utf-8", errors="replace"),
    )


def test_every_infinite_glue_warning_is_accounted_for_by_a_wide_table() -> None:
    """At most one message per paragraph-column ``longtable``.

    ``\\LT@start`` is reached once per table, so this bound is structural, not a
    tally. If it ever fails, a *new* emitter has appeared and the AGENTS.md
    write-up has stopped explaining the log.
    """
    log_text, tex_text = _shipped()
    seen = count_message(log_text)
    wide = count_paragraph_column_longtables(tex_text)
    assert seen <= wide, (
        f"{seen} '{MESSAGE}' messages against {wide} paragraph-column "
        "longtables; something other than longtable's \\LT@start is emitting them"
    )


def test_a_manuscript_with_no_wide_table_would_carry_no_such_warning() -> None:
    """The bound has teeth: zero wide tables must mean zero messages."""
    log_text, tex_text = _shipped()
    if count_paragraph_column_longtables(tex_text) == 0:
        assert count_message(log_text) == 0


def test_the_log_carries_no_overfull_boxes() -> None:
    """Corroborates that nothing is clipped or run into the margin."""
    log_text, _ = _shipped()
    assert log_text.replace("\n", "").count("Overfull \\hbox") == 0
    assert log_text.replace("\n", "").count("Overfull \\vbox") == 0


if __name__ == "__main__":  # pragma: no cover - convenience
    raise SystemExit(pytest.main([__file__, "-q"]))
