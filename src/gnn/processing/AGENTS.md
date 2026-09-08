# GNN Processing Module - Agent Scaffolding

## Module Overview

**Purpose**: GNN file processing — corpus discovery, lightweight
parse/validate/report surface, five-phase orchestration engine, and step-3
multi-format serialization

**Pipeline Step**: Not a numbered step itself. Step 3 (`src/gnn/3_gnn.py`)
delegates its body to `multi_format_processor.process_gnn_multi_format`;
`core_processor.GNNProcessor` drives the standalone five-phase engine.

**Category**: Processing / Orchestration

**Status**: Production Ready

**Version**: 3.2.0

**Last Updated**: 2026-09-06

---

## Module Structure

Four concerns, one package:

1. `processor.py` — lightweight discovery, parsing, validation, and
   reporting surface with no heavy dependencies: `discover_gnn_files`,
   `parse_gnn_file`, `check_gnn_file_structure`,
   `process_gnn_directory`, `process_gnn_directory_lightweight`,
   `generate_gnn_report`, `get_module_info`.
2. `core_processor.py` — `GNNProcessor` five-phase orchestration engine:
   `ProcessingPhase.DISCOVERY → VALIDATION → ROUND_TRIP → CROSS_FORMAT →
   REPORTING` over a `ProcessingContext`; `create_processor` is the factory.
   Round-trip uses `gnn.schema_validator.CrossFormatValidator`; reporting
   uses `gnn.report.processing_report.ReportGenerator`.
3. `multi_format_processor.py` — `process_gnn_multi_format`: discovers,
   parses, and serializes GNN models to the supported formats with
   `full` / `minimal` presets (minimal keeps markdown, json, python). This
   is the step-3 body consumed by `src/gnn/3_gnn.py` through
   `create_standardized_pipeline_script`.
4. `discovery.py` — `is_model_source_path` corpus filtering (excludes
   `README.md` and friends plus `*.example.md` / `*.template.md` suffixes),
   `FileDiscoveryStrategy` content-aware discovery, and `DiscoveryResult`.

## Agent Guidance

- Import the public surface from `gnn.processing` (see `__all__`, 17 names),
  not from the submodules, except for genuinely internal use within the
  package.
- The root package facade re-exports the `processor.py` surface and
  `process_gnn_multi_format` via `_EXPORT_MAP` (`processing.processor`,
  `processing.multi_format_processor`); `GNNProcessor` and the discovery
  predicates are package-level only.
- Discovery filtering lives in `discovery.py`; `processor.py` and
  `multi_format_processor.py` both consume `is_model_source_path` — extend
  the predicate there, never re-implement per module.
- `process_gnn_multi_format` resolves the step output directory through
  `gnn.pipeline.config.get_output_dir_for_script("3_gnn.py", ...)`; keep
  the numbered script thin, put step behavior here.

## Test Coverage

```
uv run pytest tests/gnn/test_gnn_overall.py tests/pipeline/test_pipeline_functionality.py tests/pipeline/test_pipeline_recovery.py tests/infrastructure/test_coverage_overall.py tests/test_gnn_multi_format_preset.py -q
```

- `tests/gnn/test_gnn_overall.py` — engine imports and report generation
- `tests/pipeline/test_pipeline_functionality.py` — `process_gnn_directory`
  through the engine
- `tests/pipeline/test_pipeline_recovery.py` — lightweight-mode recovery
- `tests/infrastructure/test_coverage_overall.py` — package facade surface
- `tests/test_gnn_multi_format_preset.py` — `full` / `minimal` preset
  filtering in `multi_format_processor`
