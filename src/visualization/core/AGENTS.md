# visualization.core

| Symbol | Location |
|--------|----------|
| `process_visualization`, `process_single_gnn_file` | `process.py` |
| `load_visualization_model`, `resolve_gnn_step3_output_dir`, `write_stale_json_note_if_needed` | `parsed_model.py` |

Orchestrates step-8 file loop, sampling, calls `graph`, `matrix`, `analysis`. Writes `{model}_viz_manifest.json` per model (artifact list, `_viz_meta`, counts).

**Pipeline:** `8_visualization.py` → `visualization.process_visualization` (re-export from `processor.py`).
