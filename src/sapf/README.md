# SAPF Module

The SAPF (Synthetic Audio Processing Framework) module provides audio generation capabilities for GNN models, including sonification, audio synthesis, and multi-backend audio processing.

## Overview

This module enables the conversion of GNN models to audio representations, supporting various audio backends and providing comprehensive audio analysis capabilities.

## Features

- **GNN to Audio Conversion**: Convert GNN model specifications to audio representations
- **Multi-backend Support**: Support for SAPF, Pedalboard, and basic audio generation
- **Audio Analysis**: Comprehensive analysis of generated audio characteristics
- **Sonification**: Convert model dynamics and structures to sound
- **Waveform Generation**: Generate audio waveforms from model parameters

## Module Structure

```
src/sapf/
├── __init__.py                    # Module initialization and exports
├── README.md                      # This documentation
├── AGENTS.md                      # Module agent scaffolding
└── [Implementation files]
```

## Usage

```python
from sapf import process_gnn_to_audio

# Generate audio from GNN model
result = process_gnn_to_audio(
    gnn_content=gnn_model_content,
    output_dir=Path("output/audio")
)
```

## Dependencies

- `numpy` - Audio sample generation
- `scipy` - Advanced audio processing
- `soundfile` - Audio file I/O
- `librosa` - Audio analysis (optional)

## Integration

This module integrates with the GNN pipeline as Step 15: Audio processing.

## Status

✅ Production Ready
