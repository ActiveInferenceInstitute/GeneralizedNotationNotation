# Docs Tests

This directory contains tests that exercise documentation and roadmap contract
audits. These tests are intentionally small and should not write generated
documentation artifacts.

## Test Files

- `test_add_module_docstrings.py` — regression tests for `scripts/experiments/add_module_docstrings.py` (dry-run and write contracts).
- `test_capability_contracts.py` — capability-count contracts across repository docs.
- `test_check_external_links.py` — regression tests for `scripts/check_external_links.py` URL-capture helpers.
- `test_doc_accuracy_contracts.py` — doc-accuracy contracts (orchestrator line counts, stale-citation detection).
- `test_doc_contracts.py` — regression tests for source-backed documentation contracts.
- `test_docs_audit.py` — tests for the `docs/development/docs_audit.py` helpers.
- `test_skill_contracts.py` — skill-documentation contract checks.
