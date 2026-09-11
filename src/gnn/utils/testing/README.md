# `gnn/utils/testing/` — README

Concern-package home for the test-harness family (`TestRunner`, fixtures,
reports, environment, perf, assertions), extracted from `testing_utils.py`
(S2-33 Step 1; design in `docs/development/utils_split_design.md` §6).
The old path `gnn/utils/testing_utils.py` is a `DeprecationWarning` facade
for the deprecation window — new code imports from this package.

Layout, invariants (lazy facade, frozen `_EXPORT_MAP` values, private
`_PerformanceTracker`), and the gating tests are listed in
[`AGENTS.md`](AGENTS.md).
