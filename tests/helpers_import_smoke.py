"""Shared import-smoke registry and assertions for GNN modules.

Replaces the copy-pasted ``test_module_imports`` / ``test_module_importable`` /
``test_get_module_info`` blocks that were duplicated across the per-module
overall test files. The registry below is the union of every module path those
clones covered (verified before deletion); the parametrized test in
``tests/test_module_import_smoke.py`` consumes it.
"""

from __future__ import annotations

import importlib

# Union of the module sets exercised by the deleted clone blocks. Top-level
# gnn.* packages that expose version/FEATURES/get_module_info metadata, plus
# the sub-modules the clones imported explicitly. ``extra_info_keys`` pins
# module-specific keys that the deleted per-module get_module_info tests
# asserted on top of the universal version/description/features trio.
MODULE_REGISTRY: dict[str, dict] = {
    "gnn": {"extra_info_keys": ("available_validators", "available_parsers", "schema_formats")},
    "gnn.type_checker": {},
    "gnn.export": {
        "extra_info_keys": (
            "available_formats",
            "graph_formats",
            "text_formats",
            "data_formats",
        )
    },
    "gnn.visualization": {},
    "gnn.visualization.visualizer": {},
    "gnn.render": {"extra_info_keys": ("available_targets", "supported_formats")},
    "gnn.execute": {},
    "gnn.execute.executor": {},
    "gnn.llm": {},
    "gnn.llm.llm_processor": {},
    "gnn.audio": {"extra_info_keys": ("audio_capabilities", "supported_formats")},
    "gnn.analysis": {},
    "gnn.integration": {},
    "gnn.security": {},
    "gnn.research": {},
    "gnn.website": {
        "extra_info_keys": ("supported_file_types", "embedding_capabilities")
    },
    "gnn.report": {},
    "gnn.ontology": {
        "extra_info_keys": ("processing_capabilities", "supported_formats")
    },
    "gnn.mcp": {},
    "gnn.setup": {"extra_info_keys": ("environment_types",)},
    "gnn.utils": {},
    "gnn.pipeline": {},
    "gnn.advanced_visualization": {},
    "gnn.intelligent_analysis": {"extra_info_keys": ("report_formats",)},
    "gnn.intelligent_analysis.mcp": {},
    "gnn.intelligent_analysis.analyzer": {},
    "gnn.template": {},
    "gnn.template.utils": {},
}


def assert_module_import_smoke(module_name: str) -> None:
    """Import one registered module and assert its public metadata contract.

    Fails hard on any import error (the module shows up as the failing
    param, so a deliberately-broken module can never pass silently).
    Sub-modules (>= 2 dots below ``gnn``) only must import; top-level
    modules additionally expose the standard metadata trio.
    """
    module = importlib.import_module(module_name)

    if module_name.count(".") <= 1:
        assert isinstance(getattr(module, "__version__", None), str) and getattr(
            module, "__version__", ""
        ), f"{module_name}.__version__ must be a non-empty string"
        features = getattr(module, "FEATURES", None)
        assert isinstance(features, dict) and len(features) > 0, (
            f"{module_name}.FEATURES must be a non-empty dict"
        )
        get_module_info = getattr(module, "get_module_info", None)
        assert callable(get_module_info), (
            f"{module_name} must expose callable get_module_info"
        )
        info = get_module_info()
        assert isinstance(info, dict), f"{module_name}.get_module_info() must be a dict"
        for key in ("version", "description", "features"):
            assert key in info, f"{module_name}.get_module_info() missing {key!r}"

    for key in MODULE_REGISTRY[module_name].get("extra_info_keys", ()):
        info = module.get_module_info()
        assert key in info, f"{module_name}.get_module_info() missing {key!r}"
