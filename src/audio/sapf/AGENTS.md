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

#### `process_sapf_audio(audio_data: np.ndarray, spectral_config: Dict, **kwargs) -> np.ndarray`
**Description**: Main spectral audio processing function

**Parameters**:
- `audio_data` (np.ndarray): Input audio data
- `spectral_config` (Dict): Spectral processing configuration
- `**kwargs`: Additional processing options (debug, verbose)

**Returns**: Processed audio data as numpy array

**Example**:
```python
from audio.sapf import process_sapf_audio

spectral_config = {
    "window_size": 2048,
    "hop_size": 512,
    "effects": [
        {"type": "spectral_filter", "frequency_range": [100, 1000]},
        {"type": "harmonic_enhancement", "harmonics": [2, 3, 4]}
    ]
}

processed_audio = process_sapf_audio(audio_data, spectral_config)
```

#### `analyze_spectrum(audio_data: np.ndarray, window_size: int = 2048) -> Dict[str, np.ndarray]`
**Description**: Analyze audio data in the spectral domain

**Parameters**:
- `audio_data` (np.ndarray): Input audio data
- `window_size` (int): FFT window size (default: 2048)

**Returns**: Dictionary with spectral data (magnitude, phase, centroid, etc.)

#### `synthesize_spectrum(spectral_data: Dict[str, np.ndarray], **kwargs) -> np.ndarray`
**Description**: Synthesize audio from spectral data

**Parameters**:
- `spectral_data` (Dict): Spectral data dictionary
- `**kwargs`: Synthesis parameters (window_size, hop_size, etc.)

**Returns**: Synthesized audio data

#### `sonify_gnn_model_spectral(model_data: Dict[str, Any], sonification_config: Dict) -> np.ndarray`
**Description**: Convert GNN model data to audio using spectral processing

**Parameters**:
- `model_data` (Dict): GNN model data dictionary
- `sonification_config` (Dict): Sonification configuration

**Returns**: Audio representation of the model

#### `create_spectral_mapping(model_structure: Dict[str, Any]) -> Dict[str, Any]`
**Description**: Create spectral mapping for model sonification

**Parameters**:
- `model_structure` (Dict): Model structure data

**Returns**: Spectral mapping configuration

---

## Dependencies

### Required Dependencies
- `numpy` - Numerical computing for audio processing
- `scipy` - Scientific computing and FFT operations
- `librosa` - Audio analysis and spectral processing

### Optional Dependencies
- `soundfile` - Audio file I/O (fallback: basic WAV support)
- `matplotlib` - Spectral visualization (fallback: no visualization)
- `pyaudio` - Real-time audio I/O (fallback: file-based processing)

### Internal Dependencies
- `audio.classes` - Base audio classes and utilities
- `utils.pipeline_template` - Pipeline utilities

---

## Configuration

### Spectral Processing Configuration
```python
SPECTRAL_CONFIG = {
    'window_size': 2048,        # FFT window size
    'hop_size': 512,           # Hop size for STFT
    'window_type': 'hann',     # Window function
    'fft_size': 4096,          # FFT size (zero-padded)
    'sample_rate': 44100,      # Sample rate in Hz
    'quality': 'high'          # Processing quality
}
```

### Sonification Configuration
```python
SONIFICATION_CONFIG = {
    'mapping': {
        'variables': 'frequency_components',     # Variables → frequencies
        'connections': 'phase_relationships',    # Connections → phases
        'weights': 'magnitude_envelope',         # Weights → magnitudes
        'structure': 'harmonic_structure'        # Structure → harmonics
    },
    'spectral_effects': [
        {'type': 'harmonic_synthesis', 'harmonics': [1, 2, 3]},
        {'type': 'spectral_filter', 'frequency_range': [50, 5000]},
        {'type': 'phase_modulation', 'modulation_depth': 0.3}
    ],
    'duration': 10.0,          # Duration in seconds
    'sample_rate': 44100       # Sample rate in Hz
}
```

---

## Usage Examples

### Basic Spectral Processing
```python
from audio.sapf import process_sapf_audio
import numpy as np

# Generate test audio
audio_data = np.random.randn(44100)  # 1 second of noise

# Configure spectral effects
spectral_config = {
    "window_size": 2048,
    "hop_size": 512,
    "effects": [
        {"type": "spectral_filter", "frequency_range": [200, 2000]},
        {"type": "harmonic_enhancement", "harmonics": [2, 3]},
        {"type": "phase_shift", "shift_amount": 0.5}
    ]
}

# Process audio
processed_audio = process_sapf_audio(audio_data, spectral_config)
```

### Spectral Analysis and Synthesis
```python
from audio.sapf import analyze_spectrum, synthesize_spectrum

# Analyze audio spectrum
spectral_data = analyze_spectrum(audio_data, window_size=2048)

print(f"Magnitude shape: {spectral_data['magnitude'].shape}")
print(f"Phase shape: {spectral_data['phase'].shape}")
print(f"Spectral centroid: {spectral_data['centroid']}")

# Synthesize back to audio
reconstructed_audio = synthesize_spectrum(
    spectral_data,
    window_size=2048,
    hop_size=512
)
```

### Model Sonification
```python
from audio.sapf import sonify_gnn_model_spectral

# Example GNN model data
model_data = {
    "variables": {
        "A": {"value": [0.1, 0.2, 0.3], "type": "matrix"},
        "B": {"value": [0.4, 0.5, 0.6], "type": "vector"}
    },
    "connections": [
        {"from": "A", "to": "B", "weight": 0.7}
    ]
}

# Configure sonification
sonification_config = {
    "mapping": {
        "variables": "frequency_components",
        "connections": "phase_relationships",
        "weights": "magnitude_envelope"
    },
    "spectral_effects": [
        {"type": "harmonic_synthesis", "harmonics": [1, 2, 3]},
        {"type": "spectral_filter", "frequency_range": [50, 5000]}
    ]
}

# Generate sonification
audio_output = sonify_gnn_model_spectral(model_data, sonification_config)
```

### Real-time Spectral Processing
```python
from audio.sapf import create_spectral_processor

# Create real-time processor
processor_config = {
    "window_size": 1024,
    "hop_size": 256,
    "effects": [
        {"type": "spectral_filter", "frequency_range": [200, 2000]},
        {"type": "harmonic_enhancement", "harmonics": [2, 3]}
    ]
}

spectral_processor = create_spectral_processor(processor_config)

# Process audio chunks in real-time
def process_realtime_audio(audio_chunk):
    return spectral_processor.process(audio_chunk)
```

---

## Output Specification

### Output Products
- `processed_audio.wav` - Processed audio files
- `spectral_analysis.json` - Spectral analysis results
- `sonification_audio.wav` - Model sonification files
- `spectral_data.pkl` - Pickled spectral data

### Output Directory Structure
```
output/audio_sapf/
├── processed_audio.wav
├── spectral_analysis.json
├── sonification_audio.wav
└── spectral_data.pkl
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
- **Fallback Processing**: Use time-domain processing as fallback
- **Error Logging**: Comprehensive error reporting with suggestions

### Error Examples
```python
try:
    processed_audio = process_sapf_audio(audio_data, spectral_config)
except SpectralProcessingError as e:
    logger.error(f"Spectral processing failed: {e}")
    # Fallback to time-domain processing
    processed_audio = process_time_domain(audio_data, config)
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
GNN Model → Spectral Mapping → FFT Analysis → Effects Processing → IFFT Synthesis → Audio Output
```

---

## Testing

### Test Files
- `src/tests/test_audio_sapf_integration.py` - Integration tests
- `src/tests/test_audio_sapf_spectral.py` - Spectral processing tests
- `src/tests/test_audio_sapf_sonification.py` - Sonification tests

### Test Coverage
- **Current**: 75%
- **Target**: 85%+

### Key Test Scenarios
1. Spectral analysis accuracy validation
2. Round-trip synthesis quality testing
3. Model sonification mapping verification
4. Real-time processing performance testing
5. Error handling and recovery testing

### Test Commands
```bash
# Run SAPF-specific tests
pytest src/tests/test_audio_sapf*.py -v

# Run with coverage
pytest src/tests/test_audio_sapf*.py --cov=src/audio/sapf --cov-report=term-missing
```

---

## MCP Integration

### Tools Registered
- `sapf.analyze_spectrum` - Analyze audio spectrum
- `sapf.process_audio` - Process audio with spectral effects
- `sapf.sonify_model` - Create spectral sonification of GNN models
- `sapf.create_mapping` - Create spectral mapping configuration

### Tool Endpoints
```python
@mcp_tool("sapf.analyze_spectrum")
def analyze_spectrum_tool(audio_file_path: str) -> Dict[str, Any]:
    """Analyze the spectrum of an audio file"""
    audio_data, sr = librosa.load(audio_file_path)
    return analyze_spectrum(audio_data)
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
# Enable debug output for spectral processing
result = process_sapf_audio(audio_data, spectral_config, debug=True, verbose=True)
```

---

## Version History

### Current Version: 1.0.0

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

**Last Updated**: 2025-12-30
**Maintainer**: Audio Processing Team
**Status**: ✅ Production Ready




