# Reliability validation implementation receipt

2026-09-04. Implemented after parent GO; no commits or pushes. Existing fleet edits were extended and preserved. Changed-file inventory below compares bytes against the supplied baseline snapshot, not Git HEAD. `src/gnn/schema_validator.py` and existing Markdown annotation edits were inspected and retained without additional changes.

## Results

- Current Step 6 outcome requires a nonempty current pass with no failed files. Semantic invalidity, operational recovery, missing parsed artifacts, and parser failures fail the pass; prior success cannot mask them.
- Receipts replace the previous entry for each source path within a run/configuration. Keys include input hashes, parser outcome, path, run identity, and validation level. Counts/averages are rebuilt from distinct receipts. `current_summary` is additive; existing aggregate summary fields remain compatible.
- Run identity resolves explicit `run_id`, manifest `run_id`, then the actual Step 3 manifest timestamp. Timestamp-less legacy manifests remain output-directory scoped; explicit `run_id` is required for intentional cross-manifest accumulation or independent legacy runs sharing an output directory. Receipt files are individually atomic, not a paired transaction.
- File, reconstructed dictionary, CLI, and validation MCP callers share semantic evidence. Grouped endpoints and annotations survive reconstruction; numeric non-positive dimensions fail, symbolic dimensions and comments remain accepted. CLI warning/strict exits and MCP transport-success policy remain unchanged.
- Annotation round-trips cover all 22 serializer/parser pairs, absent labels and Unicode labels (44 cases), including JSON/YAML and embedded reconstruction.
- CLI composition tests now patch the actual imported run/server/LSP/watcher boundaries and assert calls. Environment probes are bounded in signature tests. Corrected the YAML builtins lookup and captured LSP diagnostics at its actual output sink. No production changes were made to accommodate these tests.

## Verification

Commands ran from the repository root. The environment prefix on all uv commands was `UV_CACHE_DIR=/tmp/gnn-reliability-uv`; `--offline --no-sync` used the existing environment without installing dependencies.

```bash
uv run --offline --no-sync --extra dev python -m pytest src/tests/validation src/tests/gnn/test_connection_annotation_roundtrip.py src/tests/cli/test_validation_semantic_parity.py src/tests/gnn/test_gnn_parsers_common.py src/tests/gnn/test_gnn_parsers_base_serializer.py src/tests/gnn/test_gnn_parsers_json.py src/tests/gnn/test_gnn_xml_parser.py src/tests/gnn/test_gnn_parsing.py src/tests/gnn/test_gnn_schema.py src/tests/gnn/test_gnn_validation.py src/tests/cli/test_templates_cli.py::test_packaged_templates_pass_strict_cli_validation src/tests/cli/test_cli_public_api.py::TestCmdHandlers::test_cmd_validate_warning_uses_exit_code_two src/tests/cli/test_cli_composition.py::TestEnvelopeMeta::test_validate_json_envelope_includes_command -q --tb=short
# 335 passed in 16.21s; /tmp/gnn-validation-final-tests.log

uv run --offline --no-sync --extra dev python -m pytest src/tests/cli/test_cli_composition.py src/tests/cli/test_cli.py::test_run_combines_and_serializes_skip_steps src/tests/cli/test_cli.py::test_run_serializes_only_steps_for_pipeline_parser src/tests/cli/test_cli.py::test_run_rejects_overlapping_only_and_skip_steps -q --tb=short
# 60 passed in 4.14s; /tmp/gnn-cli-isolation.log

uv run --offline --no-sync --extra dev ruff check --no-cache src/validation src/gnn/parsers src/gnn/schema_validator.py src/cli/__init__.py src/tests/validation/test_reliability_validation.py src/tests/gnn/test_connection_annotation_roundtrip.py src/tests/cli/test_validation_semantic_parity.py src/tests/cli/test_cli_composition.py
# All checks passed; /tmp/gnn-validation-ruff-final.log

uv run --offline --no-sync --extra dev mypy --no-incremental src/validation src/gnn/parsers src/gnn/schema_validator.py src/cli/__init__.py --config-file pyproject.toml
# Success: no issues found in 54 source files; /tmp/gnn-validation-mypy-final.log

uv run --offline --no-sync --extra dev python doc/development/docs_audit.py --strict --check-anchors --no-write
# Exit 1: zero broken links/anchors, one unrelated missing AGENTS.md under src/tests/tests.
# /tmp/gnn-validation-docs.log

git -c core.fsmonitor=false diff --check -- src/validation src/gnn/parsers src/cli/__init__.py src/tests/validation/test_reliability_validation.py src/tests/gnn/test_connection_annotation_roundtrip.py src/tests/cli/test_validation_semantic_parity.py src/tests/cli/test_cli_composition.py src/cli/README.md
# Passed.
```

Failing-before evidence: initial 16 regressions all failed (`/tmp/gnn-validation-red.log`); review probes exposed five further failures (`/tmp/gnn-validation-review-red.log`); expanded format probe exposed YAML label loss (1 failed/43 passed, `/tmp/gnn-annotation-all-red.log`); two comment regressions failed before correction (`/tmp/gnn-validation-comments-red.log`). All are covered by the passing commands above. A direct read-only semantic probe over every discovered Markdown exemplar reported no invalid results.

A fresh independent read-only reviewer found grouped-endpoint, negative-dimension, default-run-identity, and comment issues; all were corrected and re-reviewed with no blocking findings. Review used direct source/caller searches; GitNexus was unavailable for this nested repository, so graph confidence is reduced.

## Baseline limits and parent handoff

Parent retained `/tmp/gnn-fep-implementation-20260904/GeneralizedNotationNotation-baseline.json` (HEAD `64d49355acf197a0570b06ab334d97570774be64`). Full baseline attempts were interrupted amid existing native test fleets. Parent focused baseline `/tmp/gnn-fep-gnn-focused-baseline.log` and this worker's nearby test selection both exposed `TestHandlerSignatures` entering `_cmd_run`, the real pipeline, and Step 2 recursively. Worker selection was interrupted after 134 passes/3 failures; that launch boundary is now fixed and the complete CLI composition file passes. An initial CLI verification invocation selected a misspelled node ID and collected no tests; the corrected command above passed. Neither interrupted nor zero-test runs are reported as passing baselines.

Full repository validation remains parent-owned. No native build was intentionally launched. The documentation audit's `src/tests/tests/AGENTS.md` gap is outside this ownership. No shared receipt-schema files, API/MCP core/render/execute files, bridge files, or ISA were edited. Validation MCP changes are confined to `src/validation/mcp.py`.

## Changed files

- `src/cli/README.md`
- `src/cli/__init__.py`
- `src/gnn/parsers/README.md`
- `src/gnn/parsers/base_serializer.py`
- `src/gnn/parsers/binary_parser.py`
- `src/gnn/parsers/binary_serializer.py`
- `src/gnn/parsers/common.py`
- `src/gnn/parsers/coq_serializer.py`
- `src/gnn/parsers/functional_serializer.py`
- `src/gnn/parsers/grammar_serializer.py`
- `src/gnn/parsers/isabelle_serializer.py`
- `src/gnn/parsers/json_parser.py`
- `src/gnn/parsers/json_serializer.py`
- `src/gnn/parsers/lean_serializer.py`
- `src/gnn/parsers/maxima_serializer.py`
- `src/gnn/parsers/pkl_serializer.py`
- `src/gnn/parsers/protobuf_parser.py`
- `src/gnn/parsers/protobuf_serializer.py`
- `src/gnn/parsers/python_serializer.py`
- `src/gnn/parsers/scala_serializer.py`
- `src/gnn/parsers/schema_parser.py`
- `src/gnn/parsers/temporal_serializer.py`
- `src/gnn/parsers/xml_parser.py`
- `src/gnn/parsers/xml_serializer.py`
- `src/gnn/parsers/xsd_serializer.py`
- `src/gnn/parsers/yaml_parser.py`
- `src/gnn/parsers/yaml_serializer.py`
- `src/gnn/parsers/znotation_serializer.py`
- `src/tests/cli/test_cli_composition.py`
- `src/tests/cli/test_validation_semantic_parity.py`
- `src/tests/gnn/test_connection_annotation_roundtrip.py`
- `src/tests/validation/test_reliability_validation.py`
- `src/validation/AGENTS.md`
- `src/validation/README.md`
- `src/validation/SPEC.md`
- `src/validation/__init__.py`
- `src/validation/mcp.py`
- `src/validation/semantic_validator.py`
- `src/validation/structure.py`
- `src/validation/workflow.py`
- `docs/development/fleet-logs/reliability-validation-REPORT.md`
