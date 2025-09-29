# Audio Module - Agent Scaffolding

## Module Overview

**Purpose**: Generate audio representations and sonifications of GNN models using SAPF, Pedalboard, and other audio backends

**Pipeline Step**: Step 15: Audio processing (15_audio.py)

**Category**: Audio Generation / Sonification

---

## Core Functionality

### Primary Responsibilities
1. Convert GNN specifications to audio representations
2. Generate SAPF (Structured Audio Processing Format) code
3. Apply audio effects with Pedalboard
4. Create sonifications of model dynamics
5. Support multiple audio backends

### Key Capabilities
- SAPF code generation from GNN models
- Audio synthesis and processing
- Model sonification (state transitions, observations)
- Multi-backend support (SAPF, Pedalboard, Pure Data)
- WAV file generation

---

## API Reference

### Public Functions

#### `process_audio(target_dir, output_dir, **kwargs) -> bool`
**Description**: Main audio processing function

**Parameters**:
- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for audio files
- `audio_backend` (str): Audio backend ("auto", "sapf", "pedalboard")
- `duration` (float): Audio duration in seconds
- `**kwargs**: Additional options

**Returns**: `True` if audio generation succeeded

#### `generate_audio_from_gnn(gnn_model, duration=30.0) -> Path`
**Description**: Generate audio file from GNN model

**Returns**: Path to generated WAV file

#### `create_sonification(gnn_model, **kwargs) -> bytes`
**Description**: Create sonification of model dynamics

**Returns**: Audio data as bytes

---

## Dependencies

### Required Dependencies
- `numpy` - Audio sample generation
- `soundfile` - WAV file I/O

### Optional Dependencies
- `librosa` - Audio analysis (fallback: basic generation)
- `pedalboard` - Audio effects (fallback: skip effects)
- `sapf` - SAPF backend (fallback: skip SAPF)

---

## Usage Examples

### Basic Usage
```python
from audio import process_audio

success = process_audio(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/15_audio_output"),
    audio_backend="auto",
    duration=30.0
)
```

### Generate Specific Audio
```python
from audio import generate_audio_from_gnn
from gnn import load_parsed_model

model = load_parsed_model("actinf_pomdp_agent.md")
audio_file = generate_audio_from_gnn(model, duration=60.0)
```

---

## Output Specification

### Output Products
- `*.wav` - Generated audio files
- `*_sapf.py` - SAPF code files
- `audio_processing_summary.json` - Processing summary

### Output Directory Structure
```
output/15_audio_output/
├── model_name_sonification.wav
├── model_name_sapf.py
└── audio_processing_summary.json
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: 114ms
- **Memory**: 19.09 MB
- **Status**: SUCCESS
- **Files Processed**: 1
- **Audio Generated**: 0 (module in development)

### Audio Generation Times
- **SAPF Generation**: ~50ms per model
- **WAV Rendering**: ~1-5s for 30s audio
- **Effects Processing**: +500ms-2s with Pedalboard

---

## Sonification Strategies

### Model-to-Sound Mapping
1. **States → Pitch**: State values map to musical pitches
2. **Observations → Timbre**: Observation probabilities affect tone
3. **Actions → Rhythm**: Action selection creates rhythmic patterns
4. **Free Energy → Volume**: Lower FE = louder (more confident)
5. **Connections → Harmonies**: Connected variables create harmonies

---

## Testing

### Test Files
- `src/tests/test_audio_integration.py`
- `src/tests/test_audio_sapf.py`

### Test Coverage
- **Current**: 74%
- **Target**: 80%+

---

**Last Updated**: September 29, 2025  
**Status**: 🔄 In Development


