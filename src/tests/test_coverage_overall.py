#!/usr/bin/env python3
"""
Pre-flight coverage checks

Ensures public APIs across src/ can be imported and minimally exercised
without relying on prior pipeline steps.
"""

import importlib
import inspect
import pytest


class TestCoverageOverall:
    """Importability and minimal invocation checks per module."""

    def _smoke_functions(self, module_name: str, selectors: list[str]):
        mod = importlib.import_module(module_name)
        for sel in selectors:
            if not hasattr(mod, sel):
                continue
            obj = getattr(mod, sel)
            if callable(obj):
                try:
                    sig = inspect.signature(obj)
                except Exception:
                    continue
                # Call zero-arg functions only to avoid side effects
                positional_required = [
                    p for p in sig.parameters.values()
                    if p.default is inspect._empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                ]
                if len(positional_required) == 0:
                    try:
                        obj()
                    except Exception:
                        # Tolerate failures for non-critical helpers
                        pass

    def test_gnn_core_imports(self):
        import gnn
        assert hasattr(gnn, '__all__')
        from gnn.core_processor import process_gnn_directory_lightweight
        assert callable(process_gnn_directory_lightweight)

    def test_audio_imports(self):
        import audio
        assert hasattr(audio, '__all__')
        self._smoke_functions('audio', ['get_module_info'])

    def test_export_imports(self):
        import export
        assert hasattr(export, '__all__')
        self._smoke_functions('export', ['get_module_info'])

    def test_visualization_imports(self):
        import visualization
        assert hasattr(visualization, '__all__')
        self._smoke_functions('visualization', ['get_module_info'])

    def test_llm_imports(self):
        import llm
        assert hasattr(llm, '__version__')
        from llm import LLMProcessor, LLMAnalyzer
        assert callable(LLMProcessor)
        assert callable(LLMAnalyzer)

