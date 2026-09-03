# GNN Version Map

One-stop map of what changed at each release and where the authoritative record
lives. Current version: **3.2.0** (see `pyproject.toml`, `CHANGELOG.md`).

| Version | Date | Theme | Primary record |
| --- | --- | --- | --- |
| 3.2.0 | 2026-09-02 | Exemplar Gold Standard: pure continuous (linear-Gaussian) exemplars with native JAX/NumPyro/PyTorch/Stan/RxInfer.jl backends, `unsupported` render status for categorical backends, runnable Stan HMM/LGSSM programs + cmdstanpy executor, Step 12 per-folder summary merge, Julia pre-exec gate fix | [CHANGELOG §3.2.0](../CHANGELOG.md) |
| 3.1.0 | 2026-08-30 | Release hardening: `GNN_STEP_TIMEOUT_SCALE` for slow-storage checkouts, meta-analysis correctness fix, justfile repair | [CHANGELOG §3.1.0](../CHANGELOG.md) |
| 3.0.0 | 2026-06-20 | Long-Running Orchestration: durable streams, run sessions, resumable manifests, safe-by-design contracts in `src/pipeline/` | [src/pipeline/AGENTS.md](../src/pipeline/AGENTS.md) |
| 2.0.0 | 2026-06-12 | Major architecture revision | [CHANGELOG §2.0.0](../CHANGELOG.md) |
| 1.x (1.0.0–1.9.0) | 2025-12 → 2026-06 | Initial pipeline through iterative hardening | [CHANGELOG](../CHANGELOG.md) |

## Where version facts live

- **Package version**: `pyproject.toml` (`version` field) — single source of truth.
- **Release history**: [CHANGELOG.md](../CHANGELOG.md) (Keep-a-Changelog format).
- **Release process docs**: [releases/README.md](releases/README.md).
- **Roadmap / next target**: [TO-DO.md](../TO-DO.md) (currently targeting v4.0.0).

## Version-sensitive surfaces (check these after a bump)

- `pyproject.toml` `version`
- `CHANGELOG.md` release heading + compare links
- `CITATION.cff`
- `TO-DO.md` "Current Version" line
- Release test-evidence claims in `doc/releases/`
- `src/__init__.py` `__version__` (documented in [gnn/modules/init.md](gnn/modules/init.md); at the 3.2.0 release this field still reads `1.6.0` and needs a maintainer bump)
- Package-version lines in [gnn/README.md](gnn/README.md), [gnn/AGENTS.md](gnn/AGENTS.md) and [gnn/reference/SPEC.md](gnn/reference/SPEC.md)
