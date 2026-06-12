from __future__ import annotations

import ast
from pathlib import Path

from setup.package_names import package_name_for_import


def test_pipeline_execution_does_not_import_posix_resource_module() -> None:
    source = Path("src/pipeline/execution.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)
    assert "resource" not in imported_modules


def test_import_to_package_name_mapping_handles_common_mismatches() -> None:
    assert package_name_for_import("yaml") == "PyYAML"
    assert package_name_for_import("sklearn") == "scikit-learn"
    assert package_name_for_import("numpy") == "numpy"
