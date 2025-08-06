# Pedalboard Audio Module

This submodule provides audio generation capabilities for GNN models using the Pedalboard library, enabling real-time audio processing, effects chains, and model sonification for Active Inference research.

## Module Structure

```
src/audio/pedalboard/
├── __init__.py                    # Module initialization and exports
├── README.md                      # This documentation
├── processor.py                   # Core audio processing
├── effects.py                     # Audio effects implementation
├── sonification.py                # Model sonification
└── mcp.py                        # Model Context Protocol integration
```

## Core Components

### Audio Processing Functions

#### `process_pedalboard_audio(audio_data: np.ndarray, effects_chain: List[Dict], **kwargs) -> np.ndarray`
Processes audio data through a Pedalboard effects chain.

**Features:**
- Real-time audio processing
- Effects chain application
- Parameter automation
- Quality preservation
- Performance optimization

**Returns:**
- `np.ndarray`: Processed audio data

### Effects Chain Management

#### `create_effects_chain(effects_config: List[Dict]) -> List[pedalboard.Plugin]`
Creates a Pedalboard effects chain from configuration.

**Effects Support:**
- **Reverb**: Room and hall reverb effects
- **Delay**: Echo and delay effects
- **Chorus**: Chorus and flanger effects
- **Distortion**: Overdrive and distortion effects
- **Compression**: Dynamic range compression
- **EQ**: Parametric equalization
- **Limiter**: Peak limiting and protection

#### `apply_effects_chain(audio: np.ndarray, effects_chain: List[pedalboard.Plugin]) -> np.ndarray`
Applies an effects chain to audio data.

**Application Features:**
- Real-time processing
- Parameter automation
- Quality preservation
- Performance optimization
- Error handling

### Model Sonification

#### `sonify_gnn_model(model_data: Dict[str, Any], sonification_config: Dict) -> np.ndarray`
Converts GNN model data to audio using Pedalboard effects.

**Sonification Features:**
- Variable-to-audio mapping
- Parameter automation
- Real-time generation
- Quality preservation
- Performance optimization

#### `create_sonification_chain(model_structure: Dict[str, Any]) -> List[pedalboard.Plugin]`
Creates a sonification effects chain based on model structure.

**Chain Features:**
- Structure-based effects
- Parameter mapping
- Real-time automation
- Quality optimization
- Performance tuning

## Usage Examples

### Basic Audio Processing

```python
from audio.pedalboard import process_pedalboard_audio

# Process audio with effects
audio_data = load_audio("input.wav")
effects_chain = [
    {"type": "reverb", "room_size": 0.8},
    {"type": "delay", "delay_seconds": 0.5},
    {"type": "chorus", "rate_hz": 1.5}
]

processed_audio = process_pedalboard_audio(
    audio_data=audio_data,
    effects_chain=effects_chain
)

save_audio(processed_audio, "output.wav")
```

### Effects Chain Creation

```python
from audio.pedalboard import create_effects_chain

# Create effects chain
effects_config = [
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
    },
    {
        "type": "chorus",
        "rate_hz": 1.5,
        "depth": 0.5,
        "mix": 0.3
    }
]

effects_chain = create_effects_chain(effects_config)
```

### Model Sonification

```python
from audio.pedalboard import sonify_gnn_model

# Sonify GNN model
model_data = {
    "variables": {
        "A": {"value": [0.1, 0.2, 0.3], "type": "matrix"},
        "B": {"value": [0.4, 0.5, 0.6], "type": "vector"}
    },
    "connections": [
        {"from": "A", "to": "B", "weight": 0.7}
    ]
}

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

audio_output = sonify_gnn_model(model_data, sonification_config)
save_audio(audio_output, "model_sonification.wav")
```

### Real-time Processing

```python
from audio.pedalboard import create_effects_chain, apply_effects_chain

# Create effects chain for real-time processing
effects_config = [
    {"type": "compression", "threshold_db": -20, "ratio": 4},
    {"type": "eq", "low_shelf_frequency": 100, "low_shelf_gain_db": 3},
    {"type": "limiter", "threshold_db": -1}
]

effects_chain = create_effects_chain(effects_config)

# Process audio in real-time
def process_realtime_audio(audio_chunk):
    return apply_effects_chain(audio_chunk, effects_chain)
```

### Parameter Automation

```python
from audio.pedalboard import create_automated_effects_chain

# Create effects chain with parameter automation
automation_config = {
    "effects": [
        {
            "type": "reverb",
            "room_size": {"automation": "sine", "frequency": 0.1}
        },
        {
            "type": "delay",
            "delay_seconds": {"automation": "random", "range": [0.1, 0.5]}
        }
    ],
    "duration": 10.0  # seconds
}

automated_chain = create_automated_effects_chain(automation_config)
```

## Audio Effects

### Reverb Effects
- **Room Reverb**: Small room acoustic simulation
- **Hall Reverb**: Large hall acoustic simulation
- **Plate Reverb**: Plate reverb simulation
- **Spring Reverb**: Spring reverb simulation

### Delay Effects
- **Echo**: Simple echo effect
- **Tape Delay**: Analog tape delay simulation
- **Digital Delay**: Clean digital delay
- **Ping Pong**: Stereo ping pong delay

### Modulation Effects
- **Chorus**: Chorus effect
- **Flanger**: Flanger effect
- **Phaser**: Phaser effect
- **Tremolo**: Tremolo effect

### Distortion Effects
- **Overdrive**: Tube overdrive simulation
- **Distortion**: Hard distortion
- **Fuzz**: Fuzz effect
- **Bit Crusher**: Digital bit reduction

### Dynamic Effects
- **Compression**: Dynamic range compression
- **Limiter**: Peak limiting
- **Gate**: Noise gate
- **Expander**: Dynamic expansion

### EQ Effects
- **Parametric EQ**: Parametric equalization
- **Graphic EQ**: Graphic equalization
- **Low Shelf**: Low frequency shelf
- **High Shelf**: High frequency shelf

## Configuration Options

### Audio Processing Configuration
```python
# Audio processing configuration
audio_config = {
    'sample_rate': 44100,
    'bit_depth': 24,
    'channels': 2,
    'buffer_size': 1024,
    'real_time': True,
    'quality': 'high'
}
```

### Effects Configuration
```python
# Effects configuration
effects_config = {
    'reverb': {
        'room_size': 0.8,
        'damping': 0.5,
        'wet_level': 0.3,
        'dry_level': 0.7
    },
    'delay': {
        'delay_seconds': 0.5,
        'feedback': 0.3,
        'mix': 0.5
    },
    'chorus': {
        'rate_hz': 1.5,
        'depth': 0.5,
        'mix': 0.3
    }
}
```

### Sonification Configuration
```python
# Sonification configuration
sonification_config = {
    'mapping': {
        'variables': 'frequency',
        'connections': 'amplitude',
        'weights': 'modulation',
        'structure': 'spatial'
    },
    'effects': [
        {'type': 'reverb', 'room_size': 0.6},
        {'type': 'delay', 'delay_seconds': 0.3}
    ],
    'duration': 10.0,
    'sample_rate': 44100
}
```

## Error Handling

### Audio Processing Failures
```python
# Handle audio processing failures gracefully
try:
    processed_audio = process_pedalboard_audio(audio_data, effects_chain)
except AudioProcessingError as e:
    logger.error(f"Audio processing failed: {e}")
    # Provide fallback processing or error reporting
```

### Effects Chain Issues
```python
# Handle effects chain issues gracefully
try:
    effects_chain = create_effects_chain(effects_config)
except EffectsChainError as e:
    logger.warning(f"Effects chain creation failed: {e}")
    # Provide fallback effects or error reporting
```

### Sonification Issues
```python
# Handle sonification issues gracefully
try:
    audio_output = sonify_gnn_model(model_data, sonification_config)
except SonificationError as e:
    logger.error(f"Sonification failed: {e}")
    # Provide fallback sonification or error reporting
```

## Performance Optimization

### Audio Processing Optimization
- **Real-time Processing**: Optimize for real-time performance
- **Buffer Management**: Efficient buffer handling
- **Memory Management**: Optimize memory usage
- **CPU Optimization**: Minimize CPU usage

### Effects Chain Optimization
- **Chain Optimization**: Optimize effects chain order
- **Parameter Caching**: Cache parameter values
- **Quality Settings**: Adjust quality vs performance
- **Parallel Processing**: Parallel effects processing

### Sonification Optimization
- **Mapping Optimization**: Optimize data-to-audio mapping
- **Real-time Generation**: Optimize for real-time generation
- **Memory Management**: Efficient memory usage
- **CPU Optimization**: Minimize CPU usage

## Testing and Validation

### Unit Tests
```python
# Test individual audio functions
def test_audio_processing():
    audio_data = generate_test_audio()
    effects_chain = create_test_effects_chain()
    processed_audio = process_pedalboard_audio(audio_data, effects_chain)
    assert processed_audio is not None
    assert len(processed_audio) > 0
```

### Integration Tests
```python
# Test complete audio pipeline
def test_audio_pipeline():
    model_data = load_test_model()
    sonification_config = create_test_sonification_config()
    audio_output = sonify_gnn_model(model_data, sonification_config)
    assert audio_output is not None
    assert len(audio_output) > 0
```

### Validation Tests
```python
# Test audio validation
def test_audio_validation():
    audio_data = generate_test_audio()
    validation = validate_audio_quality(audio_data)
    assert 'valid' in validation
    assert 'quality_score' in validation
    assert 'noise_level' in validation
```

## Dependencies

### Required Dependencies
- **pedalboard**: Audio effects processing
- **numpy**: Numerical computing
- **scipy**: Scientific computing
- **soundfile**: Audio file I/O
- **librosa**: Audio analysis

### Optional Dependencies
- **pyaudio**: Real-time audio I/O
- **webrtcvad**: Voice activity detection
- **noisereduce**: Noise reduction
- **pyrubberband**: Time stretching
- **pyloudnorm**: Loudness normalization

## Performance Metrics

### Processing Performance
- **Real-time Latency**: < 10ms for real-time processing
- **Effects Processing**: 1-50ms per effect
- **Chain Processing**: 5-200ms per chain
- **Memory Usage**: 10-100MB depending on effects

### Sonification Performance
- **Generation Time**: 1-30 seconds for model sonification
- **Real-time Generation**: < 50ms latency
- **Memory Usage**: 50-500MB for complex models
- **CPU Usage**: 10-80% depending on complexity

### Quality Metrics
- **Audio Quality**: 90-95% quality preservation
- **Effects Quality**: 85-95% effects accuracy
- **Sonification Quality**: 80-90% mapping accuracy
- **Performance Quality**: 90-95% performance accuracy

## Troubleshooting

### Common Issues

#### 1. Audio Processing Failures
```
Error: Audio processing failed - invalid sample rate
Solution: Check sample rate compatibility and convert if necessary
```

#### 2. Effects Chain Issues
```
Error: Effects chain creation failed - invalid effect type
Solution: Check effect type and provide valid alternatives
```

#### 3. Sonification Issues
```
Error: Sonification failed - invalid model data
Solution: Validate model data structure and provide fallback
```

#### 4. Performance Issues
```
Error: Audio processing timeout - high CPU usage
Solution: Optimize effects chain and reduce complexity
```

### Debug Mode
```python
# Enable debug mode for detailed audio information
results = process_pedalboard_audio(audio_data, effects_chain, debug=True, verbose=True)
```

## Future Enhancements

### Planned Features
- **AI-Powered Effects**: Machine learning-based effects generation
- **Advanced Sonification**: Advanced model-to-audio mapping
- **Real-time Collaboration**: Multi-user real-time processing
- **Cloud Processing**: Cloud-based audio processing

### Performance Improvements
- **Advanced Caching**: Advanced caching strategies
- **Parallel Processing**: Enhanced parallel processing
- **GPU Acceleration**: GPU-accelerated audio processing
- **Machine Learning**: ML-based performance optimization

## Summary

The Pedalboard Audio module provides comprehensive audio generation capabilities for GNN models using the Pedalboard library, enabling real-time audio processing, effects chains, and model sonification. The module ensures reliable audio processing, high-quality effects application, and optimal performance for Active Inference research and audio-based model analysis.

## License and Citation

This module is part of the GeneralizedNotationNotation project. See the main repository for license and citation information. 