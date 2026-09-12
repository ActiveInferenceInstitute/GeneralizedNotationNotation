# `gnn/utils/config_io/` — README

Concern-package home (S2-33 Step 7, family 2/3). The old
`gnn/utils/config_loader.py`, `io_utils.py`, `code_metrics.py`, and
`path_utils.py` paths are DeprecationWarning facades — import from this
package. Layout, invariants (frozen `_EXPORT_MAP` values, guarded-PyYAML eager
loading), and gating tests are listed in [AGENTS.md](AGENTS.md); the package
map and migration plan live in
[`docs/development/utils_split_design.md`](../../../../docs/development/utils_split_design.md).
