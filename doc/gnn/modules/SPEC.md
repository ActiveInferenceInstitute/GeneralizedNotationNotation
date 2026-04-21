# Specification: Pipeline Module Docs

## Scope
One document per pipeline step (0-24), mirroring the layout of `src/*/`.
Each document describes the step's orchestrator, its module-level API,
CLI flags, output artifacts, testing strategy, and dependencies.

## Contents
25 step documents (`00_template.md` … `24_intelligent_analysis.md`) plus
an AGENTS.md enumerating them. See [`AGENTS.md`](AGENTS.md) for the full
table mapping step → document → source module.

## Naming
Files match the pipeline-script naming: `N_module.md` where N is the
zero-indexed step number and `module` matches the source directory name
(`src/<module>/`).

## Document Structure (canonical)
Reference: [`13_llm.md`](13_llm.md) (most complete, ~720 lines).
Sections:
1. Architectural Mapping — orchestrator + implementation paths
2. Module Description — what the step does and why
3. Public API — exported functions/classes
4. CLI — invocation examples
5. Configuration — env vars, flags, optional deps
6. Output Artifacts — files produced and their schemas
7. Testing — test file locations + coverage expectations
8. Troubleshooting — common symptoms + remediation
9. Source References — links back to `src/<module>/`

## Versioning
Each step doc inherits the bundle version from [`../SPEC.md`](../SPEC.md).
Document version must stay in sync with the module's `__version__` in
`src/<module>/__init__.py`.

## Status
Maintained. Hard-import steps (20, 21, 24) documented with that property
explicitly flagged.
