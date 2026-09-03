# Audio Module - Agent Scaffolding

## Module Overview

**Purpose**: Generate tonal, rhythmic, ambient, and sonification WAV renderings of GNN models with NumPy synthesis, plus SAPF (Sound As Pure Form) code generation and optional streaming chunk metadata from Step 12 telemetry

**Pipeline Step**: Step 15: Audio processing (15_audio.py)

**Category**: Audio Generation / Sonification

**Status**: Production Ready

**Version**: 3.2.0 (package version; the module-level `__version__` in `__init__.py` is tracked independently)

**Last Updated**: 2026-09-02

---

## Core Functionality

### Primary Responsibilities
1. Convert GNN specifications to audio representations (tonal, rhythmic, ambient)
2. Generate SAPF (Sound As Pure Form) code via the `sapf/` sub-package
3. Create sonifications of model dynamics
4. Emit streaming chunk metadata when execution telemetry is available
5. Probe optional audio libraries (`soundfile`, `librosa`, `pedalboard`)

### Key Capabilities
- SAPF code generation from GNN models
- NumPy audio synthesis (oscillators, envelopes, channel mixing)
- Model sonification (state transitions, time configuration)
- Backend probing (`check_audio_backends`)
- WAV file generation with a stdlib fallback writer

---

## API Reference

### Public Functions

#### `process_audio(target_dir, output_dir, verbose=False, **kwargs) -> bool`
**Description**: Main audio processing function called by orchestrator (15_audio.py)

**Parameters**:
- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for audio files
- `verbose` (bool): Enable verbose logging
- `**kwargs`: Streaming options (see Configuration)

**Returns**: `True` if audio generation succeeded

#### `generate_audio_from_gnn(file_path_or_content, output_dir=None, verbose=False) -> Dict[str, Any]`
**Description**: Generate tonal, rhythmic, and ambient WAV files from a GNN file path or raw GNN content

**Returns**: Dictionary with `file_path`, `file_name`, `audio_files` (type → path), `variables_count`, `connections_count`, `generation_timestamp`. Raises `ValueError` when `output_dir` is `None` and `RuntimeError` on generation failure.

#### `create_sonification(file_path, output_dir, verbose=False) -> Dict[str, Any]`
**Description**: Create a dynamics-driven sonification WAV of the model

**Returns**: Dictionary with `file_path`, `sonification_file`, `dynamics_analyzed`, `sonification_type`, `generation_timestamp`

#### `analyze_audio_characteristics(audio_result, verbose=False) -> Dict[str, Any]`
**Description**: Read each generated WAV (requires `soundfile`) and compute duration, amplitude, and spectral metrics

#### `check_audio_backends() -> Dict[str, Any]`
**Description**: Report availability and version of `librosa`, `soundfile`, `pedalboard`, and `numpy`

#### `generate_audio_summary(results) -> str`
**Description**: Render the Markdown written to `audio_summary.md`

---

## Configuration

### Configuration Options

#### Streaming Options (`process_audio` kwargs consumed by `_process_audio_streaming`)
- `telemetry` (dict): Inline execution trace
- `telemetry_file` / `telemetry_files` (path or list of paths): Telemetry JSON files to load
- `execution_output_dir` / `execution_results_dir` (path): Directory of Step 12 outputs; when omitted a sibling `12_execute_output/` next to `output_dir` is used if it exists
- `audio_chunk_size` (int): Frames per streaming chunk (default: `32`)

#### Fixed Generation Parameters
- Sample rate is 44100 Hz for every generator; the stdlib fallback writes 16-bit mono PCM
- Duration is derived from the model content (variable/connection counts and time configuration), not from a kwarg

---

## Dependencies

### Required Dependencies
- `numpy` - Audio sample generation

### Optional Dependencies (`audio` extra)
- `soundfile` - WAV file I/O and reading for `analyze_audio_characteristics` (recovery: stdlib WAV writer; analysis records per-type errors)
- `librosa` - Reported by `check_audio_backends()`; not used by the generation path
- `pedalboard` - Reported by `check_audio_backends()`; planned effects processing (see `pedalboard/`)

---

## Usage Examples

### Basic Usage
```python
from audio import process_audio

success = process_audio(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/15_audio_output"),
    verbose=True,
)
```

### Generate Specific Audio
```python
from audio import generate_audio_from_gnn

# Accepts a file path or raw GNN content
results = generate_audio_from_gnn(
    Path("input/gnn_files/actinf_pomdp_agent.md"),
    output_dir=Path("output/15_audio_output/actinf_pomdp_agent"),
)
print(results["audio_files"])  # {"tonal": ..., "rhythmic": ..., "ambient": ...}
```

---

## Output Specification

### Output Products
- `{model}_tonal.wav`, `{model}_rhythmic.wav`, `{model}_ambient.wav` - Generated audio renderings
- `{model}_sonification.wav` - Dynamics-driven sonification
- `audio_results.json` - Processing results
- `audio_summary.md` - Processing summary
- `audio_stream_manifest.json`, `audio_stream_chunks.json` - Streaming metadata, written when `_process_audio_streaming` finds telemetry (inline kwargs, telemetry files, or `12_execute_output/execution_summary.json` and per-run telemetry JSON under the sibling Step 12 output directory)

### Output Directory Structure
```
output/15_audio_output/
├── {model}_tonal.wav
├── {model}_rhythmic.wav
├── {model}_ambient.wav
├── {model}_sonification.wav
├── audio_results.json
├── audio_summary.md
├── audio_stream_manifest.json   # when telemetry is available
└── audio_stream_chunks.json     # when telemetry is available
```

---

## Performance Characteristics

### Latest Execution
See `output/15_audio_output/audio_results.json` and the pipeline summary for the
current run's duration, memory, and file counts; this document does not track them.

---

## Sonification Strategies

### Model-to-Sound Mapping
1. **States → Pitch**: State values map to musical pitches
2. **Observations → Timbre**: Observation probabilities affect tone
3. **Actions → Rhythm**: Action selection creates rhythmic patterns
4. **Free Energy → Volume**: Lower FE = louder (more confident)
5. **Connections → Harmonies**: Connected variables create harmonies

---

## Error Handling

### Graceful Degradation
- **No soundfile**: WAV files are still written by `write_basic_wav`; `analyze_audio_characteristics` records an `error` per audio type
- **No telemetry**: Streaming artifacts are skipped; the main renderings are unaffected
- **Invalid GNN model**: `generate_audio_from_gnn` raises `RuntimeError`; `process_audio` records the failure for that file and continues

### Error Categories
1. **Missing optional library**: Reported by `check_audio_backends()`; generation continues
2. **Audio Generation Failure**: `RuntimeError` from `generate_audio_from_gnn` / `create_sonification`
3. **File I/O Errors**: `OSError` from `save_audio_file` (non-WAV targets re-raise when `soundfile` is unavailable)
4. **Model Parsing Errors**: Empty variable/connection lists produce short, quiet renderings rather than an exception

### Error Recovery
- **Partial Generation**: Generate what's possible, report failures in `audio_results.json`
- **Per-file isolation**: One failing GNN file does not abort the step

---

## Integration Points

### Pipeline Integration
- **Input**: Receives GNN models from Step 3 (gnn processing)
- **Output**: Generates audio files for Step 20 (website generation) and Step 23 (report generation)
- **Dependencies**: Requires GNN parsing results from `3_gnn.py` output

### Module Dependencies
- **utils/**: Pipeline logging and step helpers
- **audio/sapf/**: SAPF code generation and synthesis (also re-exported by the top-level `sapf` package)
- **audio/streaming.py**: Converts Step 12 execution traces into chunk metadata

### External Integration
- **soundfile**: Optional WAV I/O
- **Pedalboard**: Planned effects processing; currently probe-only

### Data Flow
```
3_gnn.py (GNN parsing)
  ↓
15_audio.py (Audio generation)
  ↓
  ├→ 20_website.py (Audio embedding)
  ├→ 23_report.py (Audio analysis)
  └→ output/15_audio_output/ (Standalone audio files)
```

---

## Testing

### Test Files
- `src/tests/audio/` (generation, edge cases, integration, MCP tools, overall, SAPF, streaming)

### Test Coverage
Measure on demand:

```bash
uv run --extra dev python -m pytest src/tests/audio/ \
    --cov=src/audio --cov-report=term-missing
```
### Key Test Scenarios
1. Audio generation from GNN models
2. SAPF code generation
3. Audio backend validation
4. Sonification strategies

---

## MCP Integration

### Tools Registered
- `process_audio` - Run the Step 15 audio processing over a directory
- `check_audio_backends` - Report optional library availability
- `get_audio_generation_options` - List generation options
- `analyze_audio_characteristics` - Analyze a generated audio file
- `validate_audio_content` - Validate audio content
- `get_audio_module_info` - Module metadata and features

### Tool Endpoints
```python
def register_tools(mcp_instance):
    mcp_instance.register_tool(
        "process_audio",
        process_audio_mcp,
        {"target_directory": {...}, "output_directory": {...}, "verbose": {...}},
        "Process GNN files with audio generation and sonification",
    )
```

### MCP File Location
- `src/audio/mcp.py` - MCP tool registrations

---

## Troubleshooting

### Common Issues

#### Issue 1: Audio backend not available
**Symptom**: Audio generation fails with backend errors  
**Cause**: Required audio libraries not installed  
**Solution**: 
- Install audio dependencies: `uv sync --extra audio`
- Check backend availability: `python -c "from audio import check_audio_backends; print(check_audio_backends())"`
- Generation itself needs only `numpy`; missing optional libraries only reduce analysis

#### Issue 2: WAV file generation fails
**Symptom**: Audio processing completes but no WAV files created  
**Cause**: File permissions or disk space issues  
**Solution**:
- Check output directory permissions
- Verify sufficient disk space
- Check file system format supports WAV files

#### Issue 3: Sonification produces silence
**Symptom**: Generated audio files are silent  
**Cause**: Model dynamics not extracted or sonification strategy mismatch  
**Solution**:
- Verify GNN model has a `Time` section and state transitions
- Inspect `dynamics_analyzed` in the `create_sonification` result
- Check the `audio_characteristics` block in `audio_results.json`

---

## Version History

### Current Version: 3.2.0

**Features**:
- SAPF code generation
- NumPy audio synthesis
- Model sonification
- Streaming chunk metadata from Step 12 telemetry

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Enhanced sonification strategies
- **Future**: Real-time audio streaming

---

## References

### Related Documentation
- [Pipeline Overview](../../README.md)
- [Architecture Guide](../../ARCHITECTURE.md)
- [SAPF Documentation](../../doc/sapf/)
- [Pedalboard Documentation](../../doc/pedalboard/)

### External Resources
- [SAPF Specification](../../doc/sapf/README.md)
- [Pedalboard Documentation](https://github.com/spotify/pedalboard)
- [Librosa Documentation](https://librosa.org/)

---

**Last Updated**: 2026-09-02
**Maintainer**: GNN Pipeline Team
**Status**: Production Ready
**Version**: 3.2.0
**Architecture Compliance**: Thin Orchestrator Pattern


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
