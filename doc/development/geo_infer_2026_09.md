# GEO-INFER contract expansion verification

The paired change adds explicit Gaussian and factored interchange alongside the
strict categorical v1 export. Gaussian models declare discrete F/G/H/Q/R, units
and initial beliefs; factored JSON declares dependency axes and enumerated
policies. The repositories remain independently installable. See the
[Gaussian/categorical contract](../../src/export/geo_infer_contract.md) and
[factored contract](../../src/export/geo_infer_factored_contract.md).

Step 7 now consumes original source bytes and per-model physical metadata from
the API or CLI, retains nested identities, rejects output escapes and duplicate
metadata, and reports partial failures. Ordinary five-format defaults are kept.

## Ancestry and integration

The topic starts from local fleet commits `3f2694d3a` and `64d49355a`, based on
remote main `aa20514c77bb2cc4757be645755b3e3755ff8530`. Its PR therefore includes
the earlier module-quality sweep (deduplication, hardening, contract tests and
documentation), in addition to the GEO work. The original concurrently edited
checkout was preserved; integration uses an isolated topic worktree.

## Verification

- Full initial run: 4087 passed, 2 failed, 3 skipped, 2 warnings. The registry
  alias and optional-extra environment failures were reproduced and corrected.
- Targeted baseline corrections: 31 passed, including the numbered CLI, step
  registry, environment check and sync/async provider bridge. The bridge now
  detects a running event loop before allocating a coroutine and never retries
  provider RuntimeError as a loop error.
- Export/utils integration: 330 passed before final additional edge cases.
- Gaussian filtering: analytic unequal dimensions and overflow/axis rejection
  verified independently in GEO Python 3.11 and 3.12.
- Strict mypy and repository-wide Ruff checks run on the integrated tree.
- Paired CI in GEO pins a reviewed GNN commit and retains both revisions and
  categorical/H3/Gaussian/factored digests. Hosted results belong to the PR
  checks; local success does not establish hosted success.

Fresh independent reviews reproduced and then verified fixes for double
conditioning, covariance-overflow validation, contradictory source axes, output
symlinks, duplicate metadata and factored numerical overflow. All completed
capabilities have explicit bounded semantics; Gaussian control selection and
large unbounded exact policy trees are not claimed.
