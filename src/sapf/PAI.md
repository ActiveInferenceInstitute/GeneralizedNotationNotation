# SAPF Module - PAI Context

## Quick Reference

**Purpose:** Top-level re-export shim for SAPF (Structured Audio Processing Framework) audio generation from GNN models.

**When to use this module:**

- Convert GNN specifications to SAPF audio code
- Generate audio from SAPF representations
- Validate SAPF code structure

## Common Operations

```python
# Convert GNN to SAPF and generate audio
from sapf import convert_gnn_to_sapf, generate_audio_from_sapf

sapf_code = convert_gnn_to_sapf(gnn_model)
audio_data = generate_audio_from_sapf(sapf_code)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | gnn | GNN model specifications |
| **Delegates to** | audio.sapf | All implementation lives in `audio/sapf/` |
| **Output** | report | Generated audio files and SAPF code |

## Key Files

- `__init__.py` - Re-export shim (delegates to `audio.sapf`)
- `AGENTS.md` - Agent scaffolding documentation
- `README.md` - Module overview
- `SPEC.md` - Architectural specification

## Architecture Note

This is a **thin re-export module**. All real implementation lives in `src/audio/sapf/` (processor.py, audio_generators.py, sapf_gnn_processor.py, generator.py, utils.py). This shim enables `import sapf` / `import src.sapf` without duplicating code.

## Tips for AI Assistants

1. **Step 15:** SAPF is part of Step 15 (Audio Processing)
2. **Real code:** Edit `src/audio/sapf/` — not this shim
3. **Exports:** `convert_gnn_to_sapf`, `generate_sapf_audio`, `generate_audio_from_sapf`, `validate_sapf_code`, `process_gnn_to_audio`, `create_sapf_visualization`, `generate_sapf_report`
4. **Output Location:** `output/15_audio_output/`

---

**Version**: 1.1.3 | **Step**: 15 (Audio Processing)
