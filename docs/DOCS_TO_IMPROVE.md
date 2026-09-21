# Documentation to improve

This file explains how to find **maintained** documentation that still needs substance. It replaces an older auto-list that mixed thousands of paths (including **generated run output** under `docs/activeinference_jl/actinf_jl_src/`, `docs/rxinfer/.../results/`, etc.) with real doc packages.

## Ground truth

1. Run from the repository root:

   `uv run --extra dev python docs/development/docs_audit.py --strict`

2. Read the generated report:

   [development/docs_audit_report.md](development/docs_audit_report.md)

That report lists broken relative links, AGENTS↔README pairing gaps, `docs/**/AGENTS.md` orientation issues, and (when requested) suspicious `#anchors`.

## Optional anchor check

Heading fragments in Markdown links are not validated in the default audit (only target **files** are). To approximate GitHub-style heading IDs:

`uv run --extra dev python docs/development/docs_audit.py --check-anchors`

Use `--strict --check-anchors` only when you intend to fix or tolerate anchor mismatches; many older links use short `#fragments` that do not match computed heading slugs.

## Exclusions (same as `docs_audit.py`)

Paths under captured outputs, gridworld runs, multi-agent result trees, and similar **artifact** directories are treated as non-maintained documentation packages. See `_doc_path_is_generated_dump()` in [development/docs_audit.py](development/docs_audit.py). Do not hand-author AGENTS/README trees there; prefer moving artifacts to `output/` or `.gitignore`.

## Top-level inventory

The canonical list of first-level folders under `docs/` is [expected_dirs.txt](expected_dirs.txt) (see [SPEC.md](SPEC.md)).

## Change history

- **2026-09-17 truth pass** (ops/pipeline slices): verified commands, paths, step
  counts, and framework/model-kind coverage against the live tree across
  `docs/{pipeline,execution,api,mcp,security,testing,troubleshooting,performance,
  configuration,dependencies,deployment,releases,development,dev,templates}/` and
  the top-level ops docs. Fixed stale snippets (`from execute import …` → `gnn.execute`,
  `from gnn.utils.config_io.config_loader` → YAML direct-load, bare `python`/`gnn` invocations →
  `uv run`), added continuous-linear-Gaussian execution-path framing, and corrected
  counts (render 9 / execute 10 families incl. Lean, RxInfer 5.5.0 committed env).
  No entries were removed; the audit workflow above remains the source of truth.
