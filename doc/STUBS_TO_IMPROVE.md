# Documentation stubs to improve

This file explains how to find **maintained** documentation that still needs substance. It replaces an older auto-list that mixed thousands of paths (including **generated run output** under `doc/activeinference_jl/actinf_jl_src/`, `doc/rxinfer/.../results/`, etc.) with real doc packages.

## Ground truth

1. Run from the repository root:

   `uv run python doc/development/docs_audit.py --strict`

2. Read the generated report:

   [development/docs_audit_report.md](development/docs_audit_report.md)

That report lists broken relative links, AGENTS↔README pairing gaps, `doc/**/AGENTS.md` orientation issues, and (when requested) suspicious `#anchors`.

## Optional anchor check

Heading fragments in Markdown links are not validated in the default audit (only target **files** are). To approximate GitHub-style heading IDs:

`uv run python doc/development/docs_audit.py --check-anchors`

Use `--strict --check-anchors` only when you intend to fix or tolerate anchor mismatches; many older links use short `#fragments` that do not match computed heading slugs.

## Exclusions (same as `docs_audit.py`)

Paths under captured outputs, gridworld runs, multi-agent result trees, and similar **artifact** directories are treated as non-maintained documentation packages. See `_doc_path_is_generated_dump()` in [development/docs_audit.py](development/docs_audit.py). Do not hand-author AGENTS/README trees there; prefer moving artifacts to `output/` or `.gitignore`.

## Top-level inventory

The canonical list of first-level folders under `doc/` is [expected_dirs.txt](expected_dirs.txt) (see [SPEC.md](SPEC.md)).
