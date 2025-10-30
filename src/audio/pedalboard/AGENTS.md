# Pedalboard Audio Submodule - Agent Scaffolding

## Module Overview

**Purpose**: Real-time audio processing and effects for GNN model sonification using the Pedalboard library

**Parent Module**: Audio Module (Step 15: Audio processing)

**Category**: Audio Framework / Real-time Effects

---

## Core Functionality

### Primary Responsibilities
1. Real-time audio effects processing using Pedalboard
2. Effects chain management and application
3. Parameter automation and modulation
4. GNN model sonification through audio effects
5. High-performance audio processing pipeline

### Key Capabilities
- Real-time audio effects processing
- Effects chain creation and management
- Parameter automation and modulation
- GNN model sonification via audio mapping
- High-quality audio processing with low latency
- Support for various audio effects (reverb, delay, distortion, etc.)

---

## API Reference

### Public Functions

#### `process_pedalboard_audio(audio_data: np.ndarray, effects_chain: List[Dict], **kwargs) -> np.ndarray`
**Description**: Process audio data through a Pedalboard effects chain

**Parameters**:
- `audio_data` (np.ndarray): Input audio data
- `effects_chain` (List[Dict]): List of effect configurations
- `**kwargs`: Additional processing options (sample_rate, buffer_size, etc.)

**Returns**: Processed audio data as numpy array

**Example**:
```python
from audio.pedalboard import process_pedalboard_audio

effects_chain = [
    {"type": "reverb", "room_size": 0.8, "wet_level": 0.3},
    {"type": "delay", "delay_seconds": 0.5, "feedback": 0.3},
    {"type": "chorus", "rate_hz": 1.5, "depth": 0.5}
]

processed_audio = process_pedalboard_audio(audio_data, effects_chain)
```

#### `create_effects_chain(effects_config: List[Dict]) -> List[pedalboard.Plugin]`
**Description**: Create a Pedalboard effects chain from configuration

**Parameters**:
- `effects_config` (List[Dict]): List of effect configuration dictionaries

**Returns**: List of Pedalboard plugin instances

#### `apply_effects_chain(audio: np.ndarray, effects_chain: List[pedalboard.Plugin]) -> np.ndarray`
**Description**: Apply an effects chain to audio data

**Parameters**:
- `audio` (np.ndarray): Input audio data
- `effects_chain` (List): List of Pedalboard plugins

**Returns**: Processed audio data

#### `sonify_gnn_model(model_data: Dict[str, Any], sonification_config: Dict) -> np.ndarray`
**Description**: Convert GNN model data to audio using Pedalboard effects

**Parameters**:
- `model_data` (Dict): GNN model data dictionary
- `sonification_config` (Dict): Sonification configuration

**Returns**: Audio representation of the model

#### `create_sonification_chain(model_structure: Dict[str, Any]) -> List[pedalboard.Plugin]`
**Description**: Create a sonification effects chain based on model structure

**Parameters**:
- `model_structure` (Dict): Model structure data

**Returns**: Configured effects chain for sonification

---

## Dependencies

### Required Dependencies
- `pedalboard` - Audio effects processing library
- `numpy` - Numerical computing for audio
- `soundfile` - Audio file I/O

### Optional Dependencies
- `librosa` - Audio analysis (fallback: basic processing)
- `scipy` - Advanced signal processing (fallback: numpy-only)
- `pyaudio` - Real-time audio I/O (fallback: file-based)

### Internal Dependencies
- `audio.classes` - Base audio processing classes
- `utils.pipeline_template` - Pipeline utilities

---

## Configuration

### Audio Processing Configuration
```python
AUDIO_CONFIG = {
    'sample_rate': 44100,      # Sample rate in Hz
    'bit_depth': 24,           # Bit depth
    'channels': 2,             # Number of channels
    'buffer_size': 1024,       # Processing buffer size
    'real_time': True,         # Enable real-time processing
    'quality': 'high'          # Processing quality
}
```

### Effects Configuration
```python
EFFECTS_CONFIG = {
    'reverb': {
        'room_size': 0.8,       # Room size (0-1)
        'damping': 0.5,         # High frequency damping
        'wet_level': 0.3,       # Wet signal level
        'dry_level': 0.7        # Dry signal level
    },
    'delay': {
        'delay_seconds': 0.5,   # Delay time in seconds
        'feedback': 0.3,        # Feedback amount
        'mix': 0.5              # Wet/dry mix
    },
    'chorus': {
        'rate_hz': 1.5,         # Modulation rate
        'depth': 0.5,           # Modulation depth
        'mix': 0.3              # Effect mix
    }
}
```

### Sonification Configuration
```python
SONIFICATION_CONFIG = {
    'mapping': {
        'variables': 'frequency',      # Variables → audio frequencies
        'connections': 'amplitude',    # Connections → amplitude modulation
        'weights': 'modulation',       # Weights → parameter modulation
        'structure': 'spatial'         # Structure → spatial effects
    },
    'effects': [
        {'type': 'reverb', 'room_size': 0.6},
        {'type': 'delay', 'delay_seconds': 0.3}
    ],
    'duration': 10.0,          # Duration in seconds
    'sample_rate': 44100       # Sample rate in Hz
}
```

---

## Usage Examples

### Basic Effects Processing
```python
from audio.pedalboard import process_pedalboard_audio

# Configure effects chain
effects_chain = [
    {
        "type": "reverb",
        "room_size": 0.8,
        "damping": 0.5,
        "wet_level": 0.3,
        "dry_level": 0.7
    },
    {
        "type": "delay",
        "delay_seconds": 0.5,
        "feedback": 0.3,
        "mix": 0.5
    }
]

# Process audio
processed_audio = process_pedalboard_audio(audio_data, effects_chain)
```

### Effects Chain Creation
```python
from audio.pedalboard import create_effects_chain, apply_effects_chain

# Create effects chain
effects_config = [
    {"type": "compression", "threshold_db": -20, "ratio": 4},
    {"type": "eq", "low_shelf_frequency": 100, "low_shelf_gain_db": 3},
    {"type": "limiter", "threshold_db": -1}
]

effects_chain = create_effects_chain(effects_config)

# Apply to audio
processed_audio = apply_effects_chain(audio_data, effects_chain)
```

### Model Sonification
```python
from audio.pedalboard import sonify_gnn_model

# Example model data
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
        "variables": "frequency",
        "connections": "amplitude",
        "weights": "modulation"
    },
    "effects": [
        {"type": "reverb", "room_size": 0.6},
        {"type": "delay", "delay_seconds": 0.3}
    ]
}

# Generate sonification
audio_output = sonify_gnn_model(model_data, sonification_config)
```

### Real-time Processing
```python
from audio.pedalboard import create_effects_chain, apply_effects_chain

# Create effects chain for real-time use
effects_config = [
    {"type": "compression", "threshold_db": -20, "ratio": 4},
    {"type": "chorus", "rate_hz": 1.0, "depth": 0.3}
]

effects_chain = create_effects_chain(effects_config)

# Process audio chunks in real-time
def process_realtime_chunk(audio_chunk):
    return apply_effects_chain(audio_chunk, effects_chain)
```

### Parameter Automation
```python
from audio.pedalboard import create_automated_effects_chain

# Create effects with parameter automation
automation_config = {
    "effects": [
        {
            "type": "reverb",
            "room_size": {"automation": "sine", "frequency": 0.1, "amplitude": 0.3}
        },
        {
            "type": "delay",
            "delay_seconds": {"automation": "random", "range": [0.1, 0.5]}
        }
    ],
    "duration": 10.0,
    "sample_rate": 44100
}

automated_chain = create_automated_effects_chain(automation_config)
```

---

## Supported Effects

### Reverb Effects
- **Room Reverb**: Small room acoustic simulation
- **Hall Reverb**: Large hall acoustic simulation
- **Plate Reverb**: Metal plate reverb simulation
- **Spring Reverb**: Spring reverb simulation

### Delay Effects
- **Digital Delay**: Clean digital delay
- **Tape Delay**: Analog tape delay simulation
- **Ping Pong Delay**: Stereo ping pong delay

### Modulation Effects
- **Chorus**: Chorus effect with adjustable rate and depth
- **Flanger**: Flanging effect
- **Phaser**: Phase shifting effect
- **Tremolo**: Amplitude modulation

### Distortion Effects
- **Overdrive**: Tube overdrive simulation
- **Distortion**: Hard distortion
- **Fuzz**: Classic fuzz effect
- **Bit Crusher**: Digital bit reduction

### Dynamics Effects
- **Compressor**: Dynamic range compression
- **Limiter**: Peak limiting and protection
- **Noise Gate**: Noise gating
- **Expander**: Dynamic range expansion

### Equalization Effects
- **Parametric EQ**: Multi-band parametric equalization
- **Graphic EQ**: Graphic equalization
- **High/Low Shelf**: Shelf equalization filters

---

## Output Specification

### Output Products
- `processed_audio.wav` - Processed audio files
- `effects_chain.json` - Effects chain configuration
- `sonification_audio.wav` - Model sonification files
- `automation_data.json` - Parameter automation data

### Output Directory Structure
```
output/audio_pedalboard/
├── processed_audio.wav
├── effects_chain.json
├── sonification_audio.wav
└── automation_data.json
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: 10-500ms per audio file
- **Memory**: 10-100MB depending on effects chain
- **Status**: ✅ Production Ready

### Performance Breakdown
- **Real-time Latency**: < 10ms for typical effects chains
- **Effects Processing**: 1-50ms per effect
- **Chain Processing**: 5-200ms for complete chains
- **Memory Usage**: Scales with effect complexity

### Optimization Notes
- Effects chain order affects performance
- Some effects are more CPU-intensive than others
- Real-time processing requires careful buffer management

---

## Error Handling

### Effects Processing Errors
1. **Invalid Effect Parameters**: Parameter values out of range
2. **Unsupported Sample Rates**: Incompatible audio formats
3. **Memory Allocation Failures**: Insufficient memory for processing

### Recovery Strategies
- **Parameter Validation**: Automatic parameter clamping and validation
- **Fallback Effects**: Substitute unsupported effects with alternatives
- **Format Conversion**: Automatic sample rate conversion when possible

### Error Examples
```python
try:
    processed_audio = process_pedalboard_audio(audio_data, effects_chain)
except PedalboardError as e:
    logger.error(f"Effects processing failed: {e}")
    # Fallback to basic processing
    processed_audio = process_basic_audio(audio_data)
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
- `tests.test_audio_pedalboard*` - Pedalboard-specific tests

### Data Flow
```
GNN Model → Effects Mapping → Chain Creation → Audio Processing → Output Files
```

---

## Testing

### Test Files
- `src/tests/test_audio_pedalboard_integration.py` - Integration tests
- `src/tests/test_audio_pedalboard_effects.py` - Effects processing tests
- `src/tests/test_audio_pedalboard_sonification.py` - Sonification tests

### Test Coverage
- **Current**: 80%
- **Target**: 90%+

### Key Test Scenarios
1. Effects chain creation and validation
2. Audio processing accuracy and quality
3. Parameter automation functionality
4. Real-time processing performance
5. Model sonification mapping accuracy

### Test Commands
```bash
# Run Pedalboard-specific tests
pytest src/tests/test_audio_pedalboard*.py -v

# Run with coverage
pytest src/tests/test_audio_pedalboard*.py --cov=src/audio/pedalboard --cov-report=term-missing
```

---

## MCP Integration

### Tools Registered
- `pedalboard.process_audio` - Process audio with effects
- `pedalboard.create_chain` - Create effects chain
- `pedalboard.sonify_model` - Create audio sonification of GNN models
- `pedalboard.automate_parameters` - Set up parameter automation

### Tool Endpoints
```python
@mcp_tool("pedalboard.process_audio")
def process_audio_tool(audio_file_path: str, effects_config: List[Dict]) -> Dict[str, Any]:
    """Process an audio file with Pedalboard effects"""
    # Implementation
```

---

## Sonification Mapping Strategies

### Model-to-Audio Mapping
1. **Variables → Frequencies**: State variables control oscillator frequencies
2. **Connections → Amplitude**: Connection strengths modulate amplitude
3. **Weights → Parameters**: Edge weights control effect parameters
4. **Dynamics → Automation**: Model dynamics drive parameter changes
5. **Structure → Effects**: Model topology determines effects routing

### Real-time Considerations
- **Latency**: Effects must maintain low latency for real-time sonification
- **CPU Usage**: Complex effects chains may impact real-time performance
- **Memory**: Large models require efficient memory management
- **Quality**: Balance between audio quality and real-time constraints

---

## Development Guidelines

### Adding New Effects
1. Implement effect configuration in `effects.py`
2. Add parameter validation and defaults
3. Update documentation with examples
4. Add comprehensive tests

### Performance Optimization
- Profile effects for CPU usage
- Optimize parameter update rates
- Use efficient data structures for real-time processing
- Implement effect-specific optimizations

---

## Troubleshooting

### Common Issues

#### Issue 1: "Effect parameter out of range"
**Symptom**: Effects processing fails with parameter errors
**Cause**: Invalid parameter values in configuration
**Solution**: Validate parameters before creating effects chain

#### Issue 2: "Sample rate not supported"
**Symptom**: Audio processing fails with sample rate errors
**Cause**: Incompatible sample rates between audio and effects
**Solution**: Resample audio to supported rate or use fallback effects

#### Issue 3: "Real-time processing latency too high"
**Symptom**: Audio glitches or delays in real-time processing
**Cause**: Effects chain too complex for real-time requirements
**Solution**: Simplify effects chain or increase buffer size

### Debug Mode
```python
# Enable debug output for effects processing
result = process_pedalboard_audio(audio_data, effects_chain, debug=True, verbose=True)
```

---

## Version History

### Current Version: 1.0.0

**Features**:
- Complete Pedalboard effects integration
- Real-time audio processing pipeline
- GNN model sonification capabilities
- Parameter automation system
- Comprehensive effects library support

**Known Limitations**:
- Real-time performance depends on effects complexity
- Memory usage scales with effect count
- Some advanced effects require additional dependencies

### Roadmap
- **Next Version**: GPU acceleration for effects processing
- **Future**: AI-powered parameter optimization
- **Advanced**: Machine learning-based effects generation

---

## References

### Related Documentation
- [Audio Module](../../audio/AGENTS.md) - Parent audio module
- [Pedalboard Documentation](https://spotify.github.io/pedalboard/) - Official Pedalboard docs
- [Pipeline Overview](../../../../README.md) - Main pipeline documentation

### External Resources
- [Digital Signal Processing](https://en.wikipedia.org/wiki/Digital_signal_processing)
- [Audio Effects](https://en.wikipedia.org/wiki/Audio_signal_processing)
- [Real-time Audio Processing](https://en.wikipedia.org/wiki/Real-time_computing)

---

**Last Updated**: October 28, 2025
**Maintainer**: Audio Processing Team
**Status**: ✅ Production Ready




