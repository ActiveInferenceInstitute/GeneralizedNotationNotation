# Security-Ontology-Research Report

## Outcome

Completed the charter within the owned security, ontology, research, and mirrored-test paths. All changes remain uncommitted and unstaged; no commit or push was performed. Concurrent changes outside this scope were left untouched.

## Changes

### Security

- `src/security/processor.py`
  - Made `basic`, `standard`, and `strict` apply distinct scan/enforcement profiles. Explicit thresholds enable scanning, while strict or enforced policies fail closed if scanning is disabled or configuration is invalid.
  - Added structured allow/deny decisions to pre-execution receipts and made unreadable or syntactically unscannable Python produce consistent blocking findings.
  - Hashes the exact inspected bytes, reports the corresponding byte size, redacts sensitive assignment context, and returns findings in deterministic order.
  - Expanded subprocess detection to `call`, `Popen`, `run`, `check_call`, `check_output`, `getoutput`, and `getstatusoutput`, including imported/assigned aliases. Dynamic or truthy `shell` arguments are treated conservatively.
- `src/tests/security/test_security_functional.py`
  - Added policy-profile, strict fail-closed, byte-integrity, deterministic-inspection, redaction, and subprocess-call regressions.
- `src/tests/security/test_pre_exec_gate.py`
  - Added subprocess-alias/shell bypass and malformed-Python regressions; strengthened unreadable-script receipt assertions.

### Ontology

- `src/ontology/processor.py`
  - Wired the existing `ontology_terms_file` option through the real processing path. Explicit missing/invalid vocabularies fail closed with aggregate receipts.
  - Normalized canonical maps, `terms` wrappers, lists, and category-list vocabularies into a deterministic term map; ambiguous case-folded names are rejected.
  - Honored the documented recursive option and emitted nested reports under matching relative directories so equal stems cannot overwrite each other while preserving the report-path API.
  - Made ontology headings case-insensitive, ignored commented mappings, accepted Markdown list mappings, rejected conflicting variable-to-term mappings, and stabilized term/suggestion traversal.
- `src/ontology/__init__.py`
  - Made convenience validation fail closed on invalid inputs and aligned `OntologyProcessor.validate_terms()` with canonical case-insensitive validation.
- `src/tests/ontology/test_ontology_annotations.py`
  - Added case, list-marker, comment, and section-boundary parsing coverage.
- `src/tests/ontology/test_ontology_public_api.py`
  - Added strict custom-vocabulary, missing-vocabulary receipt, recursive collision, conflicting-mapping, invalid-input, and committed-model determinism regressions.

### Research

- `src/research/processor.py`
  - Centralized exact, case-insensitive GNN section parsing for model-family, state-space, connection, ontology, and parameterization handling.
  - Fixed hierarchical POMDP precedence so committed `ActInfPOMDP_Hierarchical` models take the hierarchical reasoning path; also recognizes full Hidden Markov section names.
  - Rejects non-positive dimensions, validates the recursive option with a structured error receipt, and uses relative paths as recursive model identities to prevent same-name evidence collisions.
- `src/tests/research/test_research_functional.py`
  - Added committed hierarchical-exemplar reasoning, case-insensitive parsing, invalid-dimension/configuration, same-name recursive determinism, and per-hypothesis source/claim-scope coverage.

The scoped collection increased by 17 executable regression cases: 6 security, 7 ontology, and 4 research.

## Verification

- `uv run ruff check src/security src/ontology src/research`
  - Passed: `All checks passed!`
- `uv run pytest src/tests/security src/tests/ontology src/tests/research -q --tb=no -x`
  - Passed: `135 passed in 2.12s`
- `uv run mypy src/security src/ontology src/research --config-file pyproject.toml`
  - Passed: `Success: no issues found in 10 source files`
- Scoped `git diff --check`
  - Passed with no whitespace errors.

## Changed Files

- `src/security/processor.py`
- `src/tests/security/test_security_functional.py`
- `src/tests/security/test_pre_exec_gate.py`
- `src/ontology/processor.py`
- `src/ontology/__init__.py`
- `src/tests/ontology/test_ontology_annotations.py`
- `src/tests/ontology/test_ontology_public_api.py`
- `src/research/processor.py`
- `src/tests/research/test_research_functional.py`
- `.agents/dispatch/gnn-swarm2/REPORT-security-ontology-research.md`
