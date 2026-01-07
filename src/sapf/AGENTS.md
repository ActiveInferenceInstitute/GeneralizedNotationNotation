# SAPF Module - Agent Scaffolding

## Module Overview

**Purpose**: Synthetic Audio Processing Framework (SAPF) for audio generation and sonification of GNN models

**Pipeline Step**: Infrastructure module (not a numbered step)

**Category**: Audio Framework / Sonification

---

## Core Functionality

### Primary Responsibilities
1. Audio synthesis and processing framework
2. GNN model sonification and audio representation
3. Multi-backend audio generation
4. Audio analysis and processing
5. Real-time audio processing capabilities

### Key Capabilities
- Synthetic audio generation from mathematical models
- Real-time audio processing and effects
- GNN model sonification and audio mapping
- Multi-format audio output (WAV, MP3, etc.)
- Audio analysis and feature extraction

---

## API Reference

### Public Functions

#### `get_module_info() -> Dict[str, Any]`
**Description**: Get SAPF module information

**Returns**: Dictionary with module metadata

#### `process_gnn_to_audio(gnn_content, output_dir) -> Dict[str, Any]`
**Description**: Process GNN content to generate audio

**Parameters**:
- `gnn_content`: GNN model content
- `output_dir`: Output directory for audio files

**Returns**: Dictionary with processing results

#### `convert_gnn_to_sapf(gnn_content, output_dir) -> Dict[str, Any]`
**Description**: Convert GNN content to SAPF format

**Parameters**:
- `gnn_content`: GNN model content
- `output_dir`: Output directory for SAPF files

**Returns**: Dictionary with conversion results

#### `generate_audio_from_sapf(sapf_config, output_dir) -> Dict[str, Any]`
**Description**: Generate audio from SAPF configuration

**Parameters**:
- `sapf_config`: SAPF configuration data
- `output_dir`: Output directory for audio files

**Returns**: Dictionary with generation results

#### `validate_sapf_code(sapf_code) -> Dict[str, Any]`
**Description**: Validate SAPF code syntax and structure

**Parameters**:
- `sapf_code`: SAPF code to validate

**Returns**: Dictionary with validation results

---

## Dependencies

### Required Dependencies
- `numpy` - Numerical computations for audio
- `scipy` - Scientific computing for audio processing
- `soundfile` - Audio file I/O

### Optional Dependencies
- `librosa` - Audio analysis
- `pedalboard` - Audio effects
- `pyaudio` - Real-time audio processing

### Internal Dependencies
- `utils.pipeline_template` - Pipeline utilities

---

## Configuration

### Audio Generation Settings
```python
SAPF_CONFIG = {
    'sample_rate': 44100,
    'bit_depth': 16,
    'channels': 2,
    'duration': 30.0,
    'output_format': 'wav'
}
```

### Sonification Parameters
```python
SONIFICATION_CONFIG = {
    'mapping_strategy': 'frequency',
    'frequency_range': (100, 2000),
    'amplitude_mapping': 'linear',
    'temporal_resolution': 0.1
}
```

---

## Usage Examples

### Basic Audio Generation
```python
from sapf import process_gnn_to_audio

result = process_gnn_to_audio(
    gnn_content=model_content,
    output_dir="output/audio"
)
```

### SAPF Conversion
```python
from sapf import convert_gnn_to_sapf

conversion = convert_gnn_to_sapf(
    gnn_content=model_content,
    output_dir="output/sapf"
)
```

### Audio Generation from SAPF
```python
from sapf import generate_audio_from_sapf

audio = generate_audio_from_sapf(
    sapf_config=sapf_data,
    output_dir="output/audio"
)
```

---

## Output Specification

### Output Products
- `*.wav` - Generated audio files
- `*.sapf` - SAPF configuration files
- `audio_analysis.json` - Audio analysis results
- `sonification_report.md` - Sonification report

### Output Directory Structure
```
output/sapf/
├── model_audio.wav
├── model_sapf_config.json
├── audio_analysis.json
└── sonification_report.md
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: ~2-10 seconds for audio generation
- **Memory**: ~50-200MB for complex audio
- **Status**: ✅ Production Ready

### Expected Performance
- **Audio Generation**: 1-5 seconds per 30s audio
- **SAPF Conversion**: < 1 second
- **Audio Analysis**: 1-3 seconds
- **Real-time Processing**: < 10ms latency

---

## Error Handling

### Audio Errors
1. **Generation Failures**: Audio synthesis errors
2. **File I/O Errors**: Audio file writing failures
3. **Format Errors**: Invalid audio format specifications
4. **Resource Errors**: Insufficient resources for audio generation

### Recovery Strategies
- **Format Fallback**: Try alternative audio formats
- **Quality Reduction**: Reduce audio quality for compatibility
- **Backend Fallback**: Use alternative audio backends
- **Error Documentation**: Provide detailed error reports

---

## Integration Points

### Orchestrated By
- **Script**: `15_audio.py` (Step 15)
- **Function**: Audio generation integration

### Imports From
- `utils.pipeline_template` - Pipeline utilities

### Imported By
- Audio processing components
- `tests.test_audio_*` - Audio tests

### Data Flow
```
GNN Content → SAPF Conversion → Audio Generation → Audio Analysis → Output Files
```

---

## Testing

### Test Files
- `src/tests/test_sapf_integration.py` - Integration tests
- `src/tests/test_sapf_audio.py` - Audio tests

### Test Coverage
- **Current**: 75%
- **Target**: 85%+

### Key Test Scenarios
1. Audio generation with various GNN models
2. SAPF conversion and validation
3. Audio format compatibility
4. Error handling and recovery

---

## MCP Integration

### Tools Registered
- `sapf.convert_gnn` - Convert GNN to SAPF
- `sapf.generate_audio` - Generate audio from SAPF
- `sapf.validate_code` - Validate SAPF code
- `sapf.analyze_audio` - Analyze generated audio

### Tool Endpoints
```python
@mcp_tool("sapf.convert_gnn")
def convert_gnn_to_sapf_tool(gnn_content, output_dir):
    """Convert GNN content to SAPF format"""
    # Implementation
```

---

**Last Updated**: 2026-01-07
**Status**: ✅ Production Ready