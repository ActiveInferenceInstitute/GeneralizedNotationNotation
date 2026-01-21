# Output Directory

## Overview

The `output/` directory contains generated artifacts from pipeline execution. This directory is automatically created and managed by the GNN pipeline.

## Directory Structure

When the pipeline runs, it creates step-specific subdirectories:

```
output/
├── 00_pipeline_summary/          # Pipeline execution metadata
│   └── pipeline_execution_summary.json
├── 2_tests_output/               # Test suite results
├── 3_gnn_output/                 # GNN parsing results
├── 4_model_registry_output/      # Model registry exports
├── 5_type_checker_output/        # Type checking reports
├── 6_validation_output/          # Validation results
├── 7_export_output/              # Multi-format exports
├── 8_visualization_output/       # Generated visualizations
├── 9_advanced_viz_output/        # Advanced/interactive plots
├── 10_ontology_output/           # Ontology processing results
├── 11_render_output/             # Generated code (PyMDP, RxInfer, etc.)
├── 12_execute_output/            # Simulation execution results
├── 13_llm_output/                # LLM analysis results
├── 14_ml_integration_output/     # ML integration outputs
├── 15_audio_output/              # Generated audio files
├── 16_analysis_output/           # Statistical analysis results
├── 17_integration_output/        # Integration reports
├── 18_security_output/           # Security scan results
├── 19_research_output/           # Research tool outputs
├── 20_website_output/            # Generated HTML website
├── 21_mcp_output/                # MCP tool exports
├── 22_gui_output/                # GUI state/exports
└── 23_report_output/             # Final analysis reports
```

## Usage

```bash
# Run full pipeline (creates output/)
python src/main.py --output-dir output/

# Check pipeline results
cat output/00_pipeline_summary/pipeline_execution_summary.json

# View specific step outputs
ls -la output/3_gnn_output/
```

## Important Notes

- This directory is **generated** - do not add files manually
- Contents are overwritten on each pipeline run unless `--no-clobber` is used
- Not tracked in git (see `.gitignore`)
