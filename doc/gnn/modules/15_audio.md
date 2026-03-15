# Step 15: Audio — Sonification and Analysis

## Overview

Generates audio sonifications from GNN model structures and performs audio analysis. Supports multiple backends (SAPF, Pedalboard) with configurable duration and analysis options.

## Usage

```bash
# Default sonification
python src/15_audio.py --target-dir input/gnn_files --output-dir output --verbose

# Custom duration and backend
python src/15_audio.py --duration 60.0 --audio-backend sapf --verbose

# Full analysis mode
python src/15_audio.py --full-analysis --verbose
```

## Architecture

| Component | Path |
|-----------|------|
| Orchestrator | `src/15_audio.py` (82 lines) |
| Module | `src/audio/` |
| Processor | `src/audio/processor.py` |
| Module function | `process_audio()` |
| SAPF framework | `src/sapf/` |

## CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--duration` | `float` | `30.0` | Audio duration in seconds |
| `--audio-backend` | `str` | `auto` | Backend: `auto`, `sapf`, `pedalboard` |
| `--sonification` | `bool` | `True` | Generate sonification |
| `--full-analysis` | `bool` | `False` | Run full audio analysis |

## Optional Dependencies

Install with: `uv pip install -e .[audio]`

## Output

- **Directory**: `output/15_audio_output/`
- Audio files (`.wav`, `.mp3`), spectrograms, and analysis reports

## MCP Tools (audio module — 6 real tools)

Registered by `src/audio/mcp.py` via `register_tools()`:

| Tool | Description |
|------|-------------|
| `process_audio` | Run GNN audio processing pipeline: convert GNN models to audio files |
| `check_audio_backends` | Check which audio backends are available (SAPF, Pedalboard, soundfile) |
| `get_audio_generation_options` | List all configurable audio generation options |
| `analyze_audio_characteristics` | Analyse characteristics of a GNN model for sonification |
| `validate_audio_content` | Validate audio content derived from GNN specifications |
| `get_audio_module_info` | Return audio module version and capabilities |

## Source

- **Script**: [src/15_audio.py](../../src/15_audio.py)
