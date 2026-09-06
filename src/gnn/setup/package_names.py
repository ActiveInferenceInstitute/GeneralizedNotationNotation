"""Package/import-name contracts used by setup diagnostics."""

from __future__ import annotations

from typing import Dict

IMPORT_TO_PACKAGE: Dict[str, str] = {
    "yaml": "PyYAML",
    "sklearn": "scikit-learn",
}


def package_name_for_import(import_name: str) -> str:
    """Return the installable package name for a Python import name."""
    return IMPORT_TO_PACKAGE.get(import_name, import_name)
