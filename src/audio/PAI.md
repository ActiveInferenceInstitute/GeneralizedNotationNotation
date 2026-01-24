# Audio Module - PAI Context

## Quick Reference

**Purpose:** Audio processing and sonification of model data.

**When to use this module:**
- Sonify belief trajectories
- Generate audio representations
- Create audio feedback from simulations

## Common Operations

```python
# Process audio
from audio.processor import AudioProcessor
processor = AudioProcessor(input_dir, output_dir)
results = processor.process(verbose=True)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | execute | Simulation data |
| **Output** | report | Audio files |

## Key Files

- `processor.py` - Main processor class
- `sapf/` - SAPF audio framework
- `__init__.py` - Public API exports

## Tips for AI Assistants

1. **Step 15:** Audio processing is Step 15
2. **Sonification:** Converts data to sound
3. **Output Location:** `output/15_audio_output/`
4. **SAPF:** Uses SAPF for audio generation

---

**Version:** 1.1.3 | **Step:** 15 (Audio Processing)
