# Output Directory Specification

**Version**: 1.0.0  
**Status**: Generated Directory

---

## Purpose

The `output/` directory stores all generated artifacts from the GNN pipeline. This is NOT a processing module - it contains only generated content.

## Directory Standards

### Naming Convention
- Step outputs: `{N}_{step_name}_output/`
- Pipeline summary: `00_pipeline_summary/`

### File Formats
| Type | Format | Location |
|------|--------|----------|
| Execution summary | JSON | `00_pipeline_summary/` |
| GNN parsed models | JSON | `3_gnn_output/` |
| Rendered code | .py, .jl | `11_render_output/` |
| Simulation results | JSON, pickle | `12_execute_output/` |
| Visualizations | PNG, SVG, HTML | `8_visualization_output/` |
| Reports | Markdown, PDF | `23_report_output/` |

### Lifecycle
- Created automatically on pipeline run
- Cleared with `--clean` flag
- Not version controlled (in `.gitignore`)

---

**Generated Directory**: Contains no executable code.
