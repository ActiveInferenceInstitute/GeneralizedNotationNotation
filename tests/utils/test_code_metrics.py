"""Pins for ``utils/config_io/code_metrics.count_code_metrics`` (previously untested)."""

from __future__ import annotations

from pathlib import Path

from gnn.utils.config_io.code_metrics import count_code_metrics


def test_counts_python_structures(tmp_path: Path) -> None:
    source = tmp_path / "model.py"
    source.write_text(
        "\n".join(
            [
                "import math",
                "",
                "class Model:",
                "    def fit(self):",
                "        return 1",
                "",
                "def helper():",
                "    # a comment line is not code",
                "    return 2",
                "",
            ]
        ),
        encoding="utf-8",
    )

    metrics = count_code_metrics(source)

    assert metrics["classes"] == 1
    assert metrics["functions"] == 2  # class method + module function
    assert metrics["total_lines"] == 10
    # Non-empty, non-comment lines: import, class, def, return, def, return.
    assert metrics["lines_of_code"] == 6


def test_counts_julia_functions_and_jax_decorators(tmp_path: Path) -> None:
    source = tmp_path / "rendered.py"
    source.write_text(
        "\n".join(
            [
                "@jit",
                "def kernel(x):",
                "    return x",
                "function julia_like()",
                "    return nothing",
                "end",
            ]
        ),
        encoding="utf-8",
    )

    metrics = count_code_metrics(source)

    # @jit line, "def kernel", and "function julia_like" all count.
    assert metrics["functions"] == 3
    assert metrics["lines_of_code"] == 6


def test_missing_file_returns_zero_metrics(tmp_path: Path) -> None:
    metrics = count_code_metrics(tmp_path / "missing.py")

    assert metrics == {
        "lines_of_code": 0,
        "total_lines": 0,
        "functions": 0,
        "classes": 0,
    }
