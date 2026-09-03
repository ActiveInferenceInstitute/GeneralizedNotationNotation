# SAPF Audio Submodule - Agent Scaffolding

## Module Overview

**Purpose**: SAPF (Sound As Pure Form) code generation from GNN models and Python synthesis of that code to WAV audio

**Parent Module**: Audio Module (Step 15: Audio processing)

**Category**: Audio Framework / Sonification

---

## Core Functionality

### Primary Responsibilities
1. Parse GNN sections (state space, connections, parameters, time configuration)
2. Convert parsed sections into SAPF code (oscillators, routing, matrix processing, temporal structure)
3. Validate SAPF code for basic structural issues
4. Synthesize SAPF code to a WAV file with `SyntheticAudioGenerator`
5. Emit optional JSON visualization data and reports

### Key Capabilities
- Deterministic GNN -> SAPF code generation keyed by model complexity and name
- Oscillator / LFO / envelope synthesis in NumPy with a stdlib WAV writer
- Waveform and spectrum analysis PNG alongside each WAV
- Re-exported unchanged by the top-level `src/sapf/` package

---

## API Reference

### Public Functions

#### `process_gnn_to_audio(gnn_content: str, model_name: str, output_dir: str, duration: float = 10.0, validate_only: bool = False) -> Dict[str, Any]`
**Description**: Convert GNN content to SAPF code and synthesize `{model_name}_sapf_audio.wav` into `output_dir`. `model_name` is a required positional argument. With `validate_only=True` the function returns the validation result and SAPF code without writing audio. Failures are returned as `{"success": False, "error": ...}` rather than raised.

**Returns**: `success`, `audio_file`, `model_name`, `sapf_code`, `duration`, `audio_result` (or `validation_result` when validating only)

#### `convert_gnn_to_sapf(gnn_content: str, model_name: str) -> str`
**Description**: Parse GNN content and return the generated SAPF code as a string (`sapf_gnn_processor.py`).

#### `validate_sapf_code(sapf_code: str) -> Tuple[bool, List[str]]`
**Description**: Check for empty code, unbalanced brackets, a `play` command, and variable assignments. Returns `(is_valid, issues)`.

#### `generate_audio_from_sapf(sapf_code: str, output_file: Path, duration: float = 10.0) -> bool`
**Description**: Synthesize SAPF code to `output_file` via `SyntheticAudioGenerator.generate_from_sapf`. Returns `True` on success.

#### `generate_sapf_audio(sapf_code: str, output_path: str, **kwargs) -> Dict[str, Any]`
**Description**: Dict-returning wrapper over `generate_audio_from_sapf` (`processor.py`); honours `duration` in `kwargs`. Returns `{"success": bool, "output_path": str}` or an `error` entry.

#### `create_sapf_visualization(sapf_code: str, output_path: Optional[str] = None) -> Dict[str, Any]`
**Description**: Parse `oscillator` / `envelope` lines into `visualization_data`; writes JSON when `output_path` is given.

#### `generate_sapf_report(sapf_results: Dict[str, Any], output_path: Optional[str] = None) -> Dict[str, Any]`
**Description**: Summarize a results dict (`success`, `components_count`, `duration`); writes JSON when `output_path` is given.

#### `SAPFGNNProcessor` (`sapf_gnn_processor.py`)
**Description**: Class with `parse_gnn_sections(gnn_content) -> Dict` and `convert_to_sapf(gnn_sections, model_name) -> str`, plus internal state-space, connection, parameter, and time-config parsers.

#### Audio generators (`audio_generators.py`)
- `SyntheticAudioGenerator(sample_rate=44100)` — `generate_from_sapf(sapf_code, output_file, duration, create_visualization=True) -> bool`
- `generate_oscillator_audio`, `apply_envelope`, `mix_audio_channels`

#### Module info (`module_info.py`)
- `get_module_info()`, `get_audio_generation_options()`, `register_tools()`

---

## Dependencies

### Required Dependencies
- `numpy` - Synthesis and array handling
- `matplotlib` - Imported unconditionally by `audio_generators.py` for the analysis PNG

### Internal Dependencies
- `pipeline` / `utils` - Used by `generator.py` (`generate_sapf_audio(target_dir, output_dir, logger, ...)`) for directory-level runs

---

## Usage Examples

### GNN to SAPF Audio
```python
from audio.sapf import process_gnn_to_audio

result = process_gnn_to_audio(gnn_content, "my_model", "output/15_audio_output", duration=10.0)
print(result["success"], result["audio_file"])
```

### GNN to SAPF Code
```python
from audio.sapf import convert_gnn_to_sapf, validate_sapf_code

sapf_code = convert_gnn_to_sapf(gnn_content, "my_model")
is_valid, issues = validate_sapf_code(sapf_code)
```

### Audio from SAPF Code
```python
from pathlib import Path
from audio.sapf import generate_audio_from_sapf

ok = generate_audio_from_sapf(sapf_code, Path("output/15_audio_output/my_model_sapf_audio.wav"), duration=10.0)
```

### Oscillator Utilities
```python
from audio.sapf import SyntheticAudioGenerator, generate_oscillator_audio, apply_envelope, mix_audio_channels
```

---

## Output Specification

### Output Products
- `{model}_sapf_audio.wav` - SAPF-sonified audio per model (44.1 kHz, 16-bit mono)
- `{model}_sapf_audio_waveform_analysis.png` - Waveform / spectrum / spectrogram panels (when `create_visualization` is enabled)
- Optional JSON from `create_sapf_visualization` / `generate_sapf_report`

### Output Directory Structure
```
output/15_audio_output/
├── {model}_sapf_audio.wav
└── {model}_sapf_audio_waveform_analysis.png
```

---

## Error Handling

- `process_gnn_to_audio`, `generate_sapf_audio`, `create_sapf_visualization`, and `generate_sapf_report` catch exceptions and return `{"success": False, "error": str(e)}`
- `generate_audio_from_sapf` and `SyntheticAudioGenerator.generate_from_sapf` return `False` on failure
- `validate_sapf_code` never raises; issues are returned in the list

```python
result = process_gnn_to_audio(gnn_content, "my_model", "output/15_audio_output")
if not result["success"]:
    logger.error(f"SAPF audio generation failed: {result.get('error')}")
```

---

## Integration Points

### Orchestrated By
- **Parent Module**: `src/audio/` (Step 15)
- **Main Script**: `15_audio.py`

### Imported By
- `sapf/__init__.py` (top-level package) - re-exports these functions verbatim
- `src/tests/sapf/` and `src/tests/audio/test_audio_sapf.py`

### Data Flow
```
GNN Model → SAPFGNNProcessor (sections) → SAPF code → SyntheticAudioGenerator → WAV + PNG
```

---

## Testing

### Test Files
- `src/tests/sapf/` - processor, edge-case, and MCP tool tests
- `src/tests/audio/test_audio_sapf.py` - SAPF tests within the audio suite

### Test Commands
```bash
uv run --extra dev python -m pytest src/tests/sapf/ src/tests/audio/test_audio_sapf.py -v

uv run --extra dev python -m pytest src/tests/sapf/ src/tests/audio/test_audio_sapf.py \
    --cov=src/audio/sapf --cov-report=term-missing
```

---

## MCP Integration

### Tools Registered
See `src/audio/sapf/module_info.py` (`register_tools`) for the live tool inventory;
the top-level `src/sapf/mcp.py` registers the pipeline-facing tools.

### Tool Endpoints
```python
from audio.sapf.module_info import register_tools
```

---

## Development Guidelines

### Extending SAPF Generation
1. Add new section handling in `SAPFGNNProcessor` (`sapf_gnn_processor.py`)
2. Teach `SyntheticAudioGenerator._analyze_sapf_code` to recognise any new SAPF constructs
3. Update `validate_sapf_code` if new required elements are introduced
4. Add tests under `src/tests/sapf/`

---

## Troubleshooting

#### Issue 1: `TypeError` calling `process_gnn_to_audio`
**Cause**: `model_name` omitted; it is a required positional argument
**Solution**: `process_gnn_to_audio(gnn_content, model_name, output_dir)`

#### Issue 2: `validate_sapf_code` reports "No 'play' command found"
**Cause**: SAPF code was hand-edited or truncated
**Solution**: Regenerate with `convert_gnn_to_sapf`

#### Issue 3: Silent or very short audio
**Cause**: Few or no state variables parsed from the GNN content
**Solution**: Check that the GNN file has `StateSpaceBlock` and `Connections` sections

---

## Version History

### Current Version: 3.2.0

**Features**:
- GNN -> SAPF code generation
- NumPy synthesis with stdlib WAV output
- Waveform / spectrum analysis PNG
- JSON visualization data and reports

---

## References

### Related Documentation
- [Audio Module](../AGENTS.md) - Parent audio module
- [SAPF Specification](../../../doc/sapf/sapf.md) - SAPF framework details
- [Pipeline Overview](../../../README.md) - Main pipeline documentation

---

**Last Updated**: 2026-09-02
**Maintainer**: Audio Processing Team
**Status**: Production Ready
