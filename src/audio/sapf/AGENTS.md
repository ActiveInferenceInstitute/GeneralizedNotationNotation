# SAPF Audio Submodule - Agent Scaffolding

## Module Overview

**Purpose**: Spectral Audio Processing Framework (SAPF) for advanced audio generation and spectral processing of GNN models

**Parent Module**: Audio Module (Step 15: Audio processing)

**Category**: Audio Framework / Spectral Processing

---

## Core Functionality

### Primary Responsibilities
1. Spectral domain audio processing and analysis
2. Advanced audio synthesis using frequency domain techniques
3. GNN model sonification through spectral mapping
4. Real-time spectral effects and processing
5. Harmonic analysis and synthesis

### Key Capabilities
- FFT-based spectral analysis and synthesis
- Phase and magnitude manipulation
- Harmonic enhancement and processing
- Spectral filtering and envelope shaping
- Real-time spectral effects processing
- Advanced model sonification using spectral techniques

---

## API Reference

### Public Functions

#### `process_gnn_to_audio(gnn_content: str, output_dir: str | Path, **kwargs) -> Dict[str, Any]`
**Description**: Parse GNN content and write `{model}_sapf_audio.wav` into `output_dir`.

#### `convert_gnn_to_sapf(gnn_content: str, output_dir: str | Path, **kwargs) -> Dict[str, Any]`
**Description**: Convert GNN sections (state space, connections, parameters) into SAPF code.

#### `generate_audio_from_sapf(sapf_code: str, output_dir: str | Path, **kwargs) -> Dict[str, Any]`
**Description**: Generate audio from SAPF code via `SyntheticAudioGenerator`.

#### `validate_sapf_code(sapf_code: str) -> Dict[str, Any]`
**Description**: Validate SAPF code structure.

#### `generate_sapf_audio(sapf_code: str, output_dir: str | Path, **kwargs) -> Path`
**Description**: Render SAPF code to a WAV file.

#### `create_sapf_visualization(...)` / `generate_sapf_report(...)`
**Description**: Write visualization data and processing reports as JSON.

#### `SAPFGNNProcessor` (`sapf_gnn_processor.py`)
**Description**: Class with `parse_gnn_sections`, plus internal state-space and
connection parsers that feed the SAPF conversion pipeline.

#### Audio generators (`audio_generators.py`)
- `SyntheticAudioGenerator` — oscillator/LFO synthesis from SAPF code analysis
- `generate_oscillator_audio`, `apply_envelope`, `mix_audio_channels`

---

## Dependencies

### Required Dependencies
- `numpy` - Numerical computing for audio processing
- `scipy` - Scientific computing and FFT operations
- `librosa` - Audio analysis and spectral processing

### Optional Dependencies
- `soundfile` - Audio file I/O (recovery: basic WAV support)
- `matplotlib` - Spectral visualization (recovery: no visualization)
- `pyaudio` - Real-time audio I/O (recovery: file-based processing)

### Internal Dependencies
- `audio.classes` - Base audio classes and utilities
- `utils.pipeline_template` - Pipeline utilities

---

## Configuration

### Spectral Processing Configuration
```python
SPECTRAL_CONFIG = {
    "window_size": 2048,  # FFT window size
    "hop_size": 512,  # Hop size for STFT
    "window_type": "hann",  # Window function
    "fft_size": 4096,  # FFT size (zero-padded)
    "sample_rate": 44100,  # Sample rate in Hz
    "quality": "high",  # Processing quality
}
```

### Sonification Configuration
```python
SONIFICATION_CONFIG = {
    "mapping": {
        "variables": "frequency_components",  # Variables → frequencies
        "connections": "phase_relationships",  # Connections → phases
        "weights": "magnitude_envelope",  # Weights → magnitudes
        "structure": "harmonic_structure",  # Structure → harmonics
    },
    "spectral_effects": [
        {"type": "harmonic_synthesis", "harmonics": [1, 2, 3]},
        {"type": "spectral_filter", "frequency_range": [50, 5000]},
        {"type": "phase_modulation", "modulation_depth": 0.3},
    ],
    "duration": 10.0,  # Duration in seconds
    "sample_rate": 44100,  # Sample rate in Hz
}
```

---

## Usage Examples

### GNN to SAPF Audio
```python
from audio.sapf import process_gnn_to_audio

result = process_gnn_to_audio(gnn_content, output_dir="output/audio_sapf")
```

### GNN to SAPF Code
```python
from audio.sapf import convert_gnn_to_sapf, validate_sapf_code

conversion = convert_gnn_to_sapf(gnn_content, output_dir="output/audio_sapf")
validation = validate_sapf_code(conversion["sapf_code"])
```

### Audio from SAPF Code
```python
from audio.sapf import generate_audio_from_sapf

audio = generate_audio_from_sapf(sapf_code, output_dir="output/audio_sapf")
```

### Oscillator Utilities
```python
from audio.sapf import SyntheticAudioGenerator, generate_oscillator_audio, apply_envelope, mix_audio_channels
```

---

## Output Specification

### Output Products
- `{model}_sapf_audio.wav` - SAPF-sonified audio per model

### Output Directory Structure
```
output/audio_sapf/
└── {model}_sapf_audio.wav
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: 50-500ms per audio file
- **Memory**: 10-200MB depending on window size
- **Status**: ✅ Production Ready

### Performance Breakdown
- **FFT Processing**: 1-10ms per window
- **Spectral Analysis**: 5-50ms per analysis
- **Spectral Synthesis**: 5-50ms per synthesis
- **Sonification Generation**: 2-60 seconds for complex models

### Optimization Notes
- Larger window sizes improve frequency resolution but increase computation
- Smaller hop sizes improve time resolution but increase overlap processing
- Real-time processing requires optimized window/hop size combinations

---

## Error Handling

### Spectral Processing Errors
1. **Invalid Window Size**: Must be power of 2
2. **Insufficient Audio Data**: Minimum samples required for analysis
3. **FFT Computation Errors**: Numerical issues in spectral domain

### Recovery Strategies
- **Window Size Adjustment**: Automatically adjust to nearest power of 2
- **Recovery Processing**: Use time-domain processing as recovery
- **Error Logging**: Comprehensive error reporting with suggestions

### Error Examples
```python
try:
    result = process_gnn_to_audio(gnn_content, output_dir="output/audio_sapf")
except Exception as e:
    logger.error(f"SAPF audio generation failed: {e}")
```

---

## Integration Points

### Orchestrated By
- **Parent Module**: `src/audio/` (Step 15)
- **Main Script**: `15_audio.py`

### Imports From
- `audio.classes` - Base audio processing classes
- `utils.pipeline_template` - Pipeline utilities

### Imported By
- `audio.processor` - Main audio processing integration
- `tests.test_audio_sapf*` - SAPF-specific tests

### Data Flow
```
GNN Model → SAPFGNNProcessor (sections) → SAPF code → SyntheticAudioGenerator → WAV Output
```

---

## Testing

### Test Files
- `src/tests/audio/test_audio_sapf.py` - SAPF backend tests
- `src/tests/audio/test_audio_integration.py` - Pipeline integration tests for audio outputs

### Test Coverage
Measure on demand:

```bash
uv run --extra dev python -m pytest src/tests/test_sapf*.py \
    --cov=src/audio/sapf --cov-report=term-missing
```
### Key Test Scenarios
1. Spectral analysis accuracy validation
2. Round-trip synthesis quality testing
3. Model sonification mapping verification
4. Real-time processing performance testing
5. Error handling and recovery testing

### Test Commands
```bash
# Run SAPF-specific tests
uv run --extra dev python -m pytest src/tests/test_audio_sapf*.py -v

# Run with coverage
uv run --extra dev python -m pytest src/tests/test_audio_sapf*.py --cov=src/audio/sapf --cov-report=term-missing
```

---

## MCP Integration

### Tools Registered
See `src/audio/sapf/module_info.py` (`register_tools`) for the live tool inventory;
tools delegate to the conversion/generation functions above.

### Tool Endpoints
```python
from audio.sapf.module_info import register_tools
```

---

## Spectral Effects Reference

### Filtering Effects
- **Low-pass Filter**: High-frequency attenuation
- **High-pass Filter**: Low-frequency attenuation
- **Band-pass Filter**: Frequency band selection
- **Notch Filter**: Specific frequency rejection

### Phase Effects
- **Phase Shift**: Phase angle modification
- **Phase Modulation**: Dynamic phase changes
- **Phase Synchronization**: Phase alignment across frequencies

### Harmonic Effects
- **Harmonic Enhancement**: Harmonic amplification
- **Harmonic Suppression**: Harmonic reduction
- **Harmonic Synthesis**: Harmonic generation from fundamentals

### Envelope Effects
- **Spectral Envelope Shaping**: Modify spectral envelope
- **Magnitude Compression**: Dynamic range compression
- **Spectral Gating**: Noise gating in frequency domain

---

## Development Guidelines

### Adding New Spectral Effects
1. Implement effect function in `src/audio/sapf/spectral.py`
2. Add effect configuration validation
3. Update documentation and examples
4. Add comprehensive tests

### Performance Optimization
- Use appropriate window sizes for frequency/time resolution trade-offs
- Implement efficient FFT algorithms
- Cache spectral analysis results when possible
- Use vectorized operations for batch processing

---

## Troubleshooting

### Common Issues

#### Issue 1: "FFT window size must be power of 2"
**Symptom**: Spectral processing fails with window size error
**Cause**: Invalid window size specification
**Solution**: Use power of 2 (512, 1024, 2048, 4096, etc.)

#### Issue 2: "Insufficient audio data for spectral analysis"
**Symptom**: Analysis fails with data length error
**Cause**: Audio too short for specified window size
**Solution**: Ensure audio length > window_size or reduce window_size

#### Issue 3: "Spectral reconstruction quality poor"
**Symptom**: Synthesized audio quality degraded
**Cause**: Phase information lost or hop size too large
**Solution**: Use smaller hop sizes or implement phase reconstruction

### Debug Mode
```python
# Enable verbose logging during GNN-to-audio conversion
result = process_gnn_to_audio(gnn_content, output_dir="output/audio_sapf", verbose=True)
```

---

## Version History

### Current Version: 3.0.0

**Features**:
- Complete spectral analysis and synthesis pipeline
- Advanced spectral effects processing
- GNN model sonification capabilities
- Real-time spectral processing support
- Comprehensive error handling and recovery

**Known Limitations**:
- Real-time processing limited by FFT computation
- Memory usage scales with window size
- Phase reconstruction may introduce artifacts

### Roadmap
- **Next Version**: GPU acceleration for spectral processing
- **Future**: Machine learning-based spectral effects
- **Advanced**: Neural network-based sonification

---

## References

### Related Documentation
- [Audio Module](../../audio/AGENTS.md) - Parent audio module
- [SAPF Specification](../../../doc/sapf/sapf.md) - SAPF framework details
- [Pipeline Overview](../../../README.md) - Main pipeline documentation

### External Resources
- [FFT Algorithms](https://en.wikipedia.org/wiki/Fast_Fourier_transform)
- [Spectral Processing](https://en.wikipedia.org/wiki/Spectral_music)
- [Audio Signal Processing](https://en.wikipedia.org/wiki/Digital_signal_processing)

---

**Last Updated**: 2026-04-16
**Maintainer**: Audio Processing Team
**Status**: ✅ Production Ready




