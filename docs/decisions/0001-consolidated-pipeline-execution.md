# ADR 0001: Consolidated pipeline execution

- **Status:** Accepted (first slice landed)
- **Date:** 2026-09-10
- **Scope item:** SCOPE-2026-09-10 / S2-11 (V4-STAGE)

## Context

The GNN pipeline is a 25-step numbered-script contract: `main.py` selects and
sequences `0_template.py` … `24_intelligent_analysis.py`, and
`execute_pipeline_step` (`src/gnn/main.py`) launches **one subprocess per
step**. Every numbered script is a thin orchestrator that parses CLI arguments
(`gnn.utils.pipeline_template.create_standardized_pipeline_script`) and calls a
single module function. `src/gnn/pipeline/step_registry.py` is the single
source of truth for that contract: `STEPS` (stem, description,
`module_function`), the derived lookup maps, and
`CONSOLIDATED_STEP_ALIASES` (the renumbering seam).

The subprocess-per-step shape has real costs and real benefits:

- **Costs:** one Python interpreter startup and full `gnn` import per step;
  argument passing across a CLI boundary that each step must re-parse;
  results handed back only as files plus a captured exit code, so parsed
  models are re-derived downstream instead of passed forward in memory.
- **Benefits:** a hard isolation boundary per step (process exit codes, memory
  and timeout accounting, crash containment, stdout/stderr capture), which
  matters for LLM calls, rendering arbitrary generated code, and the
  testing-matrix folder dispatch.

Consolidation must therefore not destroy the numbered-script surface that
tests, docs, MCP metadata, and users rely on.

## Decision

1. **The 25-step numbered-script contract is preserved as the CLI surface.**
   `step_registry.py` remains the single source for step metadata and wiring;
   numbered scripts keep existing and remain directly runnable.
2. **Execution consolidates through a shared in-process step executor.**
   Numbered scripts delegate to
   `gnn.pipeline.step_executor.execute_step_in_process` instead of one
   subprocess per step. The executor is registry-driven: it imports
   `gnn.<stem>` and calls `StepInfo.module_function`, mirroring the argument
   contract of `create_standardized_pipeline_script`
   (`StepConfiguration`-declared args forwarded as keyword values, standard
   `<stem>_output` directory, shared `coerce_step_exit_code` semantics). It
   adds no second step mapping.
3. **Adoption is additive and opt-in.** A new `--consolidated-steps` flag
   (default off) routes supported steps in the serial path through the
   executor. The subprocess path is untouched and remains the default; it
   remains mandatory for isolation-sensitive steps. `--parallel` runs the
   subprocess path in this slice. Steps that decline consolidated execution
   (not whitelisted, or testing-matrix folder dispatch active) transparently
   fall back to subprocess.
4. **Receipts record which mode ran.** Every recorded step carries an
   `execution_mode` field (`"consolidated"` or `"subprocess"`,
   defaulted by `_record_step_result`), and `arguments.consolidated_steps`
   appears in the pipeline execution summary receipt.

## First slice (V4-STAGE slice 1)

`CONSOLIDATED_IN_PROCESS_STEMS = {"0_template", "3_gnn", "5_type_checker"}` —
the discovery/schema steps — for `--target-dir` inputs. The slice proves the
equivalence contract with `tests/pipeline/test_step_executor.py`: in-process
step 3 produces the same `3_gnn_output` artifact set and model counts as
subprocess mode on `input/gnn_files/basics`, the receipt records the mode, and
unknown steps are refused (`UnsupportedStepError`).

## Consequences

**Positive**

- Fewer interpreter startups per consolidated run; parsed models already live
  in the caller's memory, opening the path to passing them forward between
  consolidated steps without re-reading disk.
- One argument surface (`StepConfiguration`) and one exit-code contract drive
  both modes.
- The whitelist (`CONSOLIDATED_IN_PROCESS_STEMS`) is the single, reviewable
  seam for expanding consolidation.

**Negative / accepted limitations of slice 1**

- No per-step timeout or stdout/stderr capture in consolidated mode; a hung or
  crashing module function runs in-process (exceptions become `exit_code` 1
  receipts).
- Isolation-sensitive steps (LLM analysis, code rendering/execution,
  arbitrary-dependency steps) stay on subprocess mode by design.
- Consolidation and `--parallel` are mutually deferred: the parallel tier
  executor keeps spawning subprocesses.

## Alternatives considered

- **Rewrite `main.py` to always run steps in-process** — rejected: it breaks
  isolation guarantees for steps that execute generated code and LLM calls,
  and couples pipeline survival to single-step memory/timeouts.
- **Ship consolidated execution as a permanent replacement behind a version
  bump** — rejected: no opt-out path, and the subprocess mode is still the
  reference semantics the tests compare against.
- **Extend the numbered scripts to import a shared runner themselves**
  (scripts call in-process when imported) — rejected: changes all 25 scripts'
  behavior and entangles CLI parsing with in-process invocation; the executor
  calling `module_function` directly is smaller and keeps scripts as the CLI
  surface only.

## References

- `src/gnn/pipeline/step_registry.py` (`STEPS`,
  `CONSOLIDATED_STEP_ALIASES`, `CONSOLIDATED_IN_PROCESS_STEMS`)
- `src/gnn/pipeline/step_executor.py`
- `src/gnn/main.py` (`_consolidated_step_selected`,
  `execute_pipeline_step`, `_record_step_result`)
- `tests/pipeline/test_step_executor.py`