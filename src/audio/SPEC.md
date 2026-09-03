# Audio Module Specification

## Overview
Audio generation for GNN models: NumPy synthesis of tonal, rhythmic, ambient, and
sonification WAV renderings, SAPF (Sound As Pure Form) code generation, and optional
streaming chunk metadata from Step 12 execution telemetry.

## Components

### SAPF
- `sapf/` - SAPF code generation and Python synthesis of that code

### Audio Generation
- `generator.py` - Tonal / rhythmic / ambient / sonification synthesis
- `processor.py` - `process_audio` entry point, WAV writing (soundfile or stdlib fallback), analysis
- `streaming.py` - Execution telemetry -> streaming chunk metadata
- `pedalboard/` - Documentation-only scaffold for planned effects processing

## Features
- SAPF generation
- NumPy audio synthesis
- WAV file generation (the only output format)
- Optional library probing via `check_audio_backends()`

## Key Exports
```python
from audio import process_audio, generate_audio_from_gnn, create_sonification, check_audio_backends
```


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
