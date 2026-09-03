---
name: gnn-audio-generation
description: GNN audio generation and sonification. Use when creating audio representations of GNN models, generating sonification of state spaces, or working with SAPF and Pedalboard audio backends.
---

# GNN Audio Generation (Step 15)

## Purpose

Generates audio representations of GNN models, mapping model structures, state transitions, and matrix values to sound parameters. Generation uses NumPy synthesis plus the SAPF sub-package; `librosa`/`soundfile`/`pedalboard` are optional backends reported by `check_audio_backends()`.

## Key Commands

```bash
# Run audio generation
python src/15_audio.py --target-dir input/gnn_files --output-dir output --verbose

# As part of pipeline
python src/main.py --only-steps 15 --verbose
```

## API

```python
from audio import (
    AudioGenerator,
    process_audio,
    generate_audio_from_gnn,
    create_sonification,
    analyze_audio_characteristics,
    SAPFGNNProcessor,
    SyntheticAudioGenerator,
    check_audio_backends,
    generate_oscillator_audio,
    apply_envelope,
    mix_audio_channels,
    convert_gnn_to_sapf,
    generate_audio_from_sapf,
)

# Process audio step (used by pipeline)
process_audio(target_dir, output_dir, verbose=True)

# Generate audio from GNN model
result = generate_audio_from_gnn(gnn_content, output_dir="output/")

# Create sonification
sonification = create_sonification(file_path, output_dir)

# Use AudioGenerator class
gen = AudioGenerator()

# Convert GNN to SAPF format and generate audio (sapf subpackage signatures)
from audio.sapf import convert_gnn_to_sapf as sapf_convert, generate_audio_from_sapf as sapf_gen
sapf_code = sapf_convert(gnn_content, "my_model")
sapf_gen(sapf_code, Path("output/my_model_sapf_audio.wav"))

# Check available backends
backends = check_audio_backends()
# Returns: {'librosa': {'available': True}, 'soundfile': {...}, 'pedalboard': {...}}
```

## Key Exports

- `AudioGenerator` / `SAPFGNNProcessor` — audio generator classes
- `process_audio` / `generate_audio_from_gnn` — main generation functions
- `create_sonification` — sonification from model data
- `SyntheticAudioGenerator` — procedural audio generation
- `generate_oscillator_audio`, `apply_envelope`, `mix_audio_channels` — low-level audio
- `check_audio_backends` — check librosa/soundfile/pedalboard availability

## Dependencies

```bash
# Audio generation deps
uv sync --extra audio

# Includes: librosa, soundfile, pedalboard
```

## Output

- Audio files (WAV) in `output/15_audio_output/`
- Sonification parameter maps
- Audio generation reports


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `analyze_audio_characteristics`
- `check_audio_backends`
- `get_audio_generation_options`
- `get_audio_module_info`
- `process_audio`
- `validate_audio_content`

## References

- [AGENTS.md](AGENTS.md) — Module documentation
- [README.md](README.md) — Usage guide
- [SPEC.md](SPEC.md) — Module specification
- [sapf/](sapf/) — SAPF framework


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
