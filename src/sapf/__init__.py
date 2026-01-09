"""
Top-level sapf package shim that re-exports real SAPF functionality from
`audio.sapf`. This keeps tests and external callers working with
`import sapf` without duplicating code.
"""
try:
    # Prefer explicit absolute import of the package to support both `import sapf`
    # and `import src.sapf` invocation contexts during tests.
    from audio.sapf import (
        convert_gnn_to_sapf,
        generate_sapf_audio,
        generate_audio_from_sapf,
        validate_sapf_code,
        process_gnn_to_audio,
        create_sapf_visualization,
        generate_sapf_report,
    )
except Exception:
    # Fall back to relative import if running as package under `src.` namespace
    # We need to go up one level to 'src', then down to 'audio.sapf'
    from ..audio.sapf import (
        convert_gnn_to_sapf,
        generate_sapf_audio,
        generate_audio_from_sapf,
        validate_sapf_code,
        process_gnn_to_audio,
        create_sapf_visualization,
        generate_sapf_report,
    )


# --- Module metadata and compatibility shims ---
try:
    # Attempt to reference the underlying implementation for richer metadata
    from ..audio import sapf as _audio_sapf
except Exception:
    _audio_sapf = None

__version__ = getattr(_audio_sapf, '__version__', '1.1.1')
FEATURES = getattr(_audio_sapf, 'FEATURES', {
    'convert_gnn_to_sapf': True,
    'generate_audio_from_sapf': True,
    'validate_sapf_code': True,
    'process_gnn_to_audio': True,
    'mcp_integration': True,
})


def get_module_info() -> dict:
    """Return module metadata in the shape tests expect.

    Prefer delegating to the underlying `audio.sapf.get_module_info()` when
    available, otherwise return a minimal dictionary.
    """
    if _audio_sapf and hasattr(_audio_sapf, 'get_module_info'):
        try:
            return _audio_sapf.get_module_info()
        except Exception:
            pass

    return {
        'version': __version__,
        'description': 'SAPF audio bridge for GNN pipeline',
        'features': FEATURES,
        'supported_formats': ['sapf', 'wav']
    }

__all__ = [
    "convert_gnn_to_sapf",
    "generate_sapf_audio",
    "generate_audio_from_sapf",
    "validate_sapf_code",
    "process_gnn_to_audio",
    "create_sapf_visualization",
    "generate_sapf_report",
]


