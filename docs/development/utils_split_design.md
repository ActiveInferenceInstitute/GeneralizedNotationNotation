# utils/ Concern-Package Split — Design (S2-33 / SC-38)

Status: design-only for SCOPE-2026-09-10 wave C. **No code moves in this wave.**
Scope item: S2-33 (design doc + first-extraction plan). Full mechanical split (SC-38) is
deferred to a later wave by owner decision.

Related: `SCOPE-2026-09-09.md` SC-38 (original framing), SC-7 (the proven
DeprecationWarning-alias migration pattern), SCOPE-2026-09-10 waves A+B (already landed:
`test_utils.py → testing_utils.py` rename S2-18, `utils/__init__.py` docstring drift fix
S2-28, lazy PEP 562 export map).

---

## 1. Evidence base (current composition)

Measured 2026-09-10 on `improvements/2026-09-10-scope2` @ 94612fff8.
`src/gnn/utils/` is 43 `.py` files / **14,662 lines** (41 modules + `__init__.py` +
`logging/__init__.py`).

### 1.1 The lazy facade (`utils/__init__.py`, 499 lines)

- `_EXPORT_MAP`: **113 name → submodule entries across 17 source modules**
  (`__init__.py:191-324`). Resolved lazily via PEP 562 `__getattr__`
  (`__init__.py:327-344`); importing `utils` executes no submodule, so heavy
  module-scope deps (psutil via `structured_logging`/`resource_manager`, matplotlib via
  `simulation_utils`) are paid only on first attribute access.
- `TYPE_CHECKING` block mirrors the map for mypy (`__init__.py:37-172`).
- `__all__` has 118 entries: 113 unique names + `UTILS_AVAILABLE` + 4 names
  (`log_step_start/success/warning/error`) deliberately listed under both the
  logging and structured-logging sections.
- Known drift signal: the docstring claims "111 exported names" (actual: 113) — the
  same class of drift S2-28 fixed in wave A. Counts like this must be asserted by a
  test, not restated in prose (see §4 invariants).

### 1.2 Grab-bag cores (largest unsplit modules)

| Module | Lines | Concern per its own docstring |
|---|---|---|
| `testing_utils.py` | 1,246 | test runner + stages + categories + coverage + fixtures + reports + perf tracking |
| `pipeline_monitor.py` | 806 | pipeline health monitoring |
| `dependency_validator.py` | 705 | dependency validation |
| `timeout_manager.py` | 548 | timeout handling for long-running operations |
| `structured_logging.py` | 513 | structured log emission with correlation context; module-level `LogAggregator` at `:405`, `import psutil` at `:24` |
| `visualization_optimizer.py` | 493 | visualization performance optimization |
| `error_handling.py` | 481 | error handling / recovery / reporting framework |
| `step_config.py` | 477 | declarative `StepConfiguration` |
| `visual_logging.py` | 472 | visual/accessibility logging presentation |
| `config_loader.py` | 426 | YAML config loading/validation |

### 1.3 The de facto split that already exists (~40%)

Prior work has already carved real seams inside the grab-bag; the split formalizes
these seams rather than inventing new ones:

- **`logging/`** subpackage exists: `logging/logging_utils.py` (1,338 lines, internal
  implementation) behind `logging_utils.py` (106 lines, "single public entry point"
  public facade) + `logging/__init__.py` (25 lines).
- **Argument family** exists as a seam: `arg_parsing.py` (1,458), `arg_definitions.py`
  (69), `pipeline_arguments.py` (220), `pipeline_config_merge.py` (64),
  `step_config.py` (477), `path_conversion.py` (106), behind the
  `argument_utils.py` facade (59 lines).
- **`pipeline_*` family** exists as a seam: `pipeline.py` (163, self-described "compat
  entry… thin delegates to canonical homes"), `pipeline_monitor.py` (806),
  `pipeline_template.py` (290), `pipeline_validator.py` (311),
  `pipeline_dependencies.py` (382), `pipeline_step_dependencies.py` (129).
- **MCP dispatch** is already split: `mcp_dispatch.py` (136, generic dispatcher for
  `process_<module>_mcp` tools) beside `mcp.py` (392, per-tool MCP server).

### 1.4 Import-path census (the migration-critical fact)

`grep`-census over `src/`, `scripts/`, `tests/` (`*.py`):

- **191 files** reference `gnn.utils` in some form.
- **174 files** import submodules directly (`from gnn.utils.X import …` /
  `import gnn.utils.X`). This is the dominant traffic.
- **14 files** use the top-level facade (`from gnn.utils import X`).

Per-submodule import frequency (top): `pipeline_template` 44, `logging_utils` 26,
`mcp_dispatch` 21, `error_recovery` 16, `argument_utils` 16, `safe_eval` 15,
`resource_manager` 15, `testing_utils` 6 (all under `tests/` — no src consumers).

Consequence: the expensive surface is the *submodule paths*, not the facade. Every
migration step must therefore keep old submodule paths importable (shims), and the
facade `_EXPORT_MAP` can be repointed in the same PR cheaply.

---

## 2. Explicit non-goals

- **No renaming of any public name.** The 113 `_EXPORT_MAP` keys are frozen; the split
  changes *where implementations live*, not what any consumer can import.
- **No breaking `_EXPORT_MAP` consumers.** `from gnn.utils import X` must keep working
  through the entire migration.
- **No breaking old submodule paths.** `from gnn.utils.testing_utils import X` keeps
  working (facade + DeprecationWarning) until a separate, owner-decided deprecation
  window ends.
- **No MCP tool registry changes.** Tool names and discovery paths are untouched
  (SC-7 precedent: "MCP tool registry unchanged").
- **No `validate_gnn*` alias retirement** — explicitly deferred per
  `SCOPE-2026-09-10.md` (needs a deprecation-window decision).
- **No behavior changes.** Every step is a mechanical move behind a facade.

---

## 3. Concern-package map

Seven packages, each matching an existing seam. The existing separated families get
*names*, the grab-bag cores get *homes*. Every entry cites real files.

### 3.1 `gnn/utils/logging/` — EXISTS (formalize, no move needed)

| File | Lines |
|---|---|
| `logging/logging_utils.py` (internal implementation) | 1,338 |
| `logging_utils.py` (public entry facade) | 106 |
| `logging/__init__.py` | 25 |

Already the "single public entry point" pattern (its docstring says exactly that).
Proposed end state: this subpackage is the canonical home; facade entries
`PipelineLogger`, `setup_main_logging`, `setup_step_logging`, `log_section_header`,
`setup_correlation_context`, `get_performance_summary` resolve here
(`__init__.py:246-252`).

### 3.2 `gnn/utils/arguments/` — argument parsing & step configuration

| File | Lines |
|---|---|
| `arg_parsing.py` | 1,458 |
| `step_config.py` | 477 |
| `pipeline_arguments.py` | 220 |
| `path_conversion.py` | 106 |
| `arg_definitions.py` | 69 |
| `pipeline_config_merge.py` | 64 |
| `argument_utils.py` (facade) | 59 |

Seam already proven by the `argument_utils` facade re-exporting `ArgumentParser`,
`PipelineArguments`, `StepConfiguration`, `parse_arguments`,
`build_step_command_args`, etc. (`__init__.py:40-49,193-200`). All eight exported
argument names repoint into this package.

### 3.3 `gnn/utils/pipeline/` — pipeline orchestration (the `pipeline_*` family)

| File | Lines |
|---|---|
| `pipeline_monitor.py` | 806 |
| `pipeline_dependencies.py` | 382 |
| `base_processor.py` | 314 |
| `pipeline_validator.py` | 311 |
| `pipeline_template.py` | 290 |
| `pipeline.py` (compat facade) | 163 |
| `pipeline_step_dependencies.py` | 129 |
| `execution_utils.py` | 171 |

Seam already proven: `pipeline.py` is a self-declared "compat entry… thin delegates to
canonical homes" (`pipeline.py:1-14`). Note `pipeline_template` is the single most
imported submodule in the repo (44 import sites) — its old path facade must be perfect.
Careful: `gnn.pipeline` (a *different*, pre-existing package elsewhere in `gnn`) already
exists; the utils subpackage must not collide at the `gnn.pipeline` name. Proposal: name
the package `gnn/utils/pipeline_orchestration/` — see risk R1.

### 3.4 `gnn/utils/testing/` — test harness (first extraction, see §5)

| File | Lines |
|---|---|
| `testing_utils.py` | 1,246 |

Biggest grab-bag core. Wave A already renamed `test_utils.py → testing_utils.py`
(S2-18), so every consumer already references a fresh, unambiguous name — the rename
cycle for this family is already paid. Consumers: `tests/__init__.py:47` (imports
`PROJECT_ROOT`, `SRC_DIR`, `TEST_DIR`, constants and helpers — note `PROJECT_ROOT`,
`SRC_DIR`, `TEST_DIR` are **not** facade exports, they are submodule-path-only),
`tests/utils/test_shared_helpers.py` (pins the delegating facade), and
`tests/tests/test_infrastructure_exports.py` ("guards the surface
utils/testing_utils.py uses"). Zero `src/` consumers besides the facade.

### 3.5 `gnn/utils/runtime_safety/` — timeouts, dependencies, resource bounds

| File | Lines |
|---|---|
| `dependency_validator.py` | 705 |
| `timeout_manager.py` | 548 |
| `resource_manager.py` | 275 |
| `validation_schemas.py` | 156 |
| `jax_stack_validation.py` | 151 |
| `framework_availability.py` | 122 |
| `safe_eval.py` | 80 |

These are the "is it safe to run / did it finish / how much did it cost" concerns:
dependency availability + installation, operation timeouts, memory probes
(`resource_manager.py:37` explicitly owns the canonical MB-scale process-memory
probe), bounded `ast.literal_eval` for untrusted parameter strings, and framework
availability probing (single source of truth used by Steps 11/12).

### 3.6 `gnn/utils/observability/` — structured logs, timing, visual output

| File | Lines |
|---|---|
| `structured_logging.py` | 513 |
| `visualization_optimizer.py` | 493 |
| `visual_logging.py` | 472 |
| `performance_tracking.py` | 236 |

Distinct from `logging/`: that package owns the *canonical logger and correlation
context*; these modules emit structured records, track operation timing, and format
visual output. `performance_tracking` carries a load-bearing naming invariant:
the exported object must not share its module's name (`__init__.py:253-255` — the
wave-A rename `performance_tracker.py → performance_tracking.py` exists precisely to
stop `import utils.performance_tracker` from shadowing the re-export). Preserve that
invariant for any module moved near a same-named export.

### 3.7 `gnn/utils/mcp/` — MCP dispatch (rename of `utils/mcp.py`)

| File | Lines |
|---|---|
| `mcp.py` | 392 |
| `mcp_dispatch.py` | 136 |

`mcp.py` exposes utility/system-info tools over MCP; `mcp_dispatch.py` is the generic
dispatcher for `process_<module>_mcp` pipeline-step tools (21 direct import sites —
highest churn of any non-template module). Proposed final path:
`gnn/utils/mcp/server.py` + `gnn/utils/mcp/dispatch.py`. `mcp.py` imports `psutil` at
module scope (`mcp.py:24`), so this package's `__init__.py` must be empty — see
invariant I1 and risk R2. This is the last family to move (risk register R2/R3).

### 3.8 Residual grab-bag — decision table (not assigned this wave)

These do not fit the seven seams above; each gets a one-line recommendation, final
assignment is an owner decision during the mechanical wave:

| File | Lines | Recommendation |
|---|---|---|
| `error_handling.py` | 481 | new `errors/` family with `error_recovery.py` (339) — both are re-exported, tightly coupled (correlation IDs, recovery strategies) |
| `error_recovery.py` | 339 | see above |
| `config_loader.py` | 426 | `config_io/` family with `io_utils.py` (310) — both are file/config I/O |
| `io_utils.py` | 310 | see above |
| `simulation_utils.py` | 355 | leave until the matplotlib import-time question resolves (R4); it is the only matplotlib-heavy module |
| `code_metrics.py` | 56 | `config_io/` (counts generated files) |
| `path_utils.py` | 19 | `config_io/` |
| `system_utils.py` | 58 | `system_env/` with `venv_utils.py` (86), `matplotlib_setup.py` (20) |
| `venv_utils.py` | 86 | see above |
| `matplotlib_setup.py` | 20 | see above |

Everything above this line is ~3,000 lines; the seven seams absorb ~11,600.

---

## 4. Migration mechanics

### 4.1 The move pattern (SC-7 style, proven)

SC-7 established the pattern in this repo: "rename to unambiguous names with
`DeprecationWarning` aliases, one module per PR". Applied to file moves:

1. **Move** the implementation with `git mv` into its new package
   (e.g. `src/gnn/utils/testing/`), splitting internally if the design table says so.
2. **Replace** the old module path with a facade:

   ```python
   """Earlier name; implementation moved to gnn/utils/testing/runners.py."""
   import warnings

   warnings.warn(
       "gnn.utils.testing_utils is the earlier name; import gnn.utils.testing instead",
       DeprecationWarning,
       stacklevel=2,
   )
   from gnn.utils.testing.runners import *  # noqa: F401,F403 — intentional re-export
   from gnn.utils.testing.runners import __all__ as _new_all
   __all__ = list(_new_all)
   ```

   Precedents in-tree: `utils/logging_utils.py` (public-entry facade over
   `utils/logging/logging_utils.py`) and `utils/pipeline.py` (thin-delegate compat
   entry with lazy imports).
3. **Repoint the facade** in the same PR: `_EXPORT_MAP` values change (e.g.
   `"TestRunner": "testing_utils"` → `"TestRunner": "testing.runners"`); the
   `TYPE_CHECKING` block mirrors the same imports. Keys never change.
4. **One module per PR.** Gate per PR (§5), never batch.

### 4.2 Facade contract (unchanged invariants, now asserted)

- **I1 — no import-time side effects.** New package `__init__.py` files execute
  nothing beyond `__path__` bookkeeping; module-scope heavy imports (psutil,
  matplotlib) stay deferred exactly as the current lazy map achieves
  (`__init__.py:11-14`). Concretely: `mcp/server.py`'s `import psutil` stays inside
  the leaf module; `gnn/utils/mcp/__init__.py` must not import it.
- **I2 — lazy map preserved.** `_EXPORT_MAP` stays a plain `dict[str, str]` resolved
  through `__getattr__` with `import_module` inside the function; the
  ImportError-propagates-unchanged contract (`__init__.py:327-344`) is not weakened
  (no silent fallback — repo rule).
- **I3 — public surface frozen.** `_EXPORT_MAP` keys + `__all__` unique names (113)
  are asserted by an extended `tests/tests/test_infrastructure_exports.py` (it
  already resolves every `__all__` entry); add assertions for
  `len(_EXPORT_MAP) == 113` and "no `_EXPORT_MAP` key removals" against a checked-in
  golden list, so prose counts (the recurring 111-vs-113 drift) are impossible.
- **I4 — old paths are shims until the owner ends the window.** Every old
  `gnn.utils.<module>` path keeps importing with a `DeprecationWarning`.
- **I5 — import-linter boundaries.** The repo currently has **no import-linter
  config** (`pyproject.toml` has no `[importlinter]`; no `.importlinter`). Step 0
  introduces one:
  - `gnn.utils` facade must not import any of its submodules eagerly (enforce I1/I2);
  - new concern packages must not import the facade (`gnn.utils.__init__`) — kills
    accidental cycles;
  - `gnn.utils.logging` may import `gnn.utils.runtime_safety` for probes but nothing
    inside `gnn.utils` may import `gnn.utils.logging` except through the public
    `logging_utils` entry (single entry point, per its docstring).

### 4.3 Facade FR (requirements per moved family)

For each concern package P with modules M1..Mn:

1. `gnn/utils/<pkg>/__init__.py` re-exports the family's public names (real objects,
   not lazy), so intra-family imports read `from gnn.utils.testing import TestRunner`.
2. Every `_EXPORT_MAP` entry whose implementation moved is repointed to the new leaf
   module in the same PR as the move.
3. Every old path `gnn/utils/<Mi>.py` becomes a warning facade (§4.1) — no exception,
   including "internal" modules: the census (§1.4) shows submodule paths are the
   traffic, and `tests/__init__.py` proves test code depends on a submodule path
   (`testing_utils`) that the facade never exported.
4. The facade `__all__` covers exactly the moved module's public names; private names
   (`_`-prefixed, e.g. `_PerformanceTracker` in `testing_utils.py:1145`) do not get
   facade re-exports and are free to move without a facade alias.

---

## 5. Ordered migration plan (acceptance gate per step)

Ordering principle: lowest-risk, highest-payoff first; MCP last (import-order and
registry coupling). "Gate" = commands that must be green before the next step.

- **Step 0 — guardrails (no moves).**
  Add import-linter config (§4.2 I5); extend
  `tests/tests/test_infrastructure_exports.py` with the golden `_EXPORT_MAP`
  assertions (§4.2 I3); add a light-import regression test: `python -c "import
  gnn.utils; import sys; assert 'psutil' not in sys.modules and 'matplotlib' not in
  sys.modules"`.
  *Gate:* `uv run --extra dev pytest tests/tests/test_infrastructure_exports.py -q`
  green; import-linter green; light-import check green.

- **Step 1 — testing/ (first extraction, §6).** Zero `src/` consumers; consumer set
  is entirely `tests/`, which we control. Biggest offender (1,246 lines) shrinks
  first, proving the pattern on the least-coupled family.
  *Gate:* `uv run --extra dev pytest tests/ -q` green (SC-38's stated acceptance
  probe per moved module) with emphasis on `tests/utils/test_shared_helpers.py` and
  `tests/tests/test_infrastructure_exports.py`; grep census shows no new direct
  imports of `gnn.utils.testing_utils` outside `tests/` and the facade itself.

- **Step 2 — arguments/.** Move `arg_parsing.py`, `arg_definitions.py`,
  `pipeline_arguments.py`, `pipeline_config_merge.py`, `step_config.py`,
  `path_conversion.py` into `gnn/utils/arguments/`; `argument_utils.py` becomes the
  package facade (it already is one). 16 direct import sites ride the facade.
  *Gate:* Step-0 tests green; mypy + ruff clean; facade resolution spot-check
  (`from gnn.utils import StepConfiguration, parse_arguments`).

- **Step 3 — pipeline_orchestration/.** Move the `pipeline_*` family + `base_processor`
  + `execution_utils`. Highest import traffic (`pipeline_template`: 44 sites) rides
  shims. **Do not** take the name `pipeline/` (R1).
  *Gate:* Step-0 tests green; `uv run --extra dev python scripts/check_mcp_skills_health.py --strict`
  green (registry untouched but traffic-heavy); Step 12 + MCP suites green.

- **Step 4 — runtime_safety/.** Move the seven safety modules (§3.5). `resource_manager`
  is facade-exported (`get_current_memory_usage`) and imported 15×; `dependency_validator`
  (705) and `timeout_manager` (548) are the two remaining giant cores.
  *Gate:* Step-0 tests green; light-import check green (psutil must remain lazy).

- **Step 5 — observability/.** Move `structured_logging`, `performance_tracking`,
  `visual_logging`, `visualization_optimizer`. Riskiest non-MCP step (R4: two
  module-level logging singletons meet for the first time in one package).
  *Gate:* Step-0 tests green; correlation-context behavior tests green; light-import
  check green (psutil still deferred).

- **Step 6 — mcp/.** Move `mcp.py → mcp/server.py`, `mcp_dispatch.py → mcp/dispatch.py`
  **last** (R2/R3).
  *Gate:* `uv run --extra dev python scripts/check_mcp_skills_health.py --strict`
  green; mcp-audit CI job green; all 21 `mcp_dispatch` import sites resolve through
  the facade.

- **Step 7 — residual decision (owner).** `errors/`, `config_io/`, `system_env/`,
  `simulation_utils` disposition per §3.8. Out of this design's committed scope.

Deferred to the same later wave as SC-38 (owner decision, per the spec): ending the
facade deprecation window (deleting old paths), and `validate_gnn*` alias retirement.

---

## 6. First extraction in detail: `testing_utils.py` → `gnn/utils/testing/`

Why first: no `src/` consumers (only `tests/__init__.py:47`,
`tests/utils/test_shared_helpers.py`, `tests/pipeline/test_pipeline_performance.py`,
`tests/tests/test_infrastructure_exports.py`); wave A already paid the rename cycle
(`test_utils.py → testing_utils.py`, S2-18) so downstream paths are fresh; 1,246 lines
is the largest grab-bag core.

Internal split by the module's own structure (line numbers from current file):

| New module | Contents (current lines) | Names |
|---|---|---|
| `testing/constants.py` | `TEST_DIR` :23, `TEST_CATEGORIES` :28, `TEST_STAGES` :39, `COVERAGE_TARGETS` :57, `TEST_CONFIG` :65, plus `PROJECT_ROOT`/`SRC_DIR` | `TEST_CATEGORIES`, `TEST_STAGES`, `COVERAGE_TARGETS`, `TEST_CONFIG` |
| `testing/runner.py` | `TestRunner` :102, `TestResult` :122, `TestCategory` :158, `TestStage` :176, `CoverageTarget` :204, `run_tests` :222, `run_test_category` :254, `run_test_stage` :261, `run_all_tests` :762, `run_fast/standard/slow/performance/coverage_tests` :814-858 | `TestRunner`, `TestResult`, `TestCategory`, `TestStage`, `CoverageTarget`, `run_tests`, … |
| `testing/fixtures.py` | `get_test_args` :391, `get_sample_pipeline_arguments` :419, `get_step_metadata_dict` :445, `is_safe_mode` :471, `create_missing_test_files` :476, `create_sample_config` :504, `create_sample_ontology` :538, `create_test_gnn_files` :555, `create_test_files` :571, `create_sample_gnn_content` :589, `get_test_filesystem_structure` :732 | the `create_*` + `get_sample_*` families |
| `testing/reports.py` | `get_test_results` :268 through `get_test_progress` :386, `validate_report_data` :908, `run_all_tests_mcp` :954, `generate_html/markdown/json_report_file` :991/:1049/:1089, `generate_comprehensive_report` :1102 | the `get_test_*` accessor family |
| `testing/environment.py` | `validate_test_environment` :288, `setup_test_environment` :293, `cleanup_test_environment` :297, `get_test_coverage` :301, `validate_coverage_targets` :306, `get_test_dependencies` :326, `validate/install_test_dependencies` :331/:336, `get/validate_test_configuration` :341/:346, `get_test_environment` :351 | env + coverage + dependency helpers |
| `testing/perf.py` | `_PerformanceTracker` :1145, `performance_tracker` :1171, `track_peak_memory` :1183, `with_resource_limits` :1216 | perf harness (private class stays private) |
| `testing/assertions.py` | `assert_file_exists` :869, `assert_valid_json` :875, `assert_directory_structure` :886 | assertion helpers consumed by `tests/__init__.py` |

Mechanics:

1. Create `gnn/utils/testing/` with the six/seven leaf modules above; family
   `__init__.py` re-exports all public names.
2. `gnn/utils/testing_utils.py` becomes a facade re-exporting everything public
   (including `PROJECT_ROOT`, `SRC_DIR`, `TEST_DIR` — submodule-path-only names that
   `tests/__init__.py:47` depends on), emitting one `DeprecationWarning`.
3. Repoint `_EXPORT_MAP`'s 34 `testing_utils` entries to the new leaf modules
   (`__init__.py:286-321`); update the `TYPE_CHECKING` block to mirror.
4. Intra-package imports (`fixtures.py` needs
   `pipeline_arguments.DEFAULT_ONTOLOGY_TERMS_FILE` :17 and
   `resource_manager.get_memory_usage` :18) are cross-family: import through the leaf
   modules, never the facade, per I5.

Acceptance: full `uv run --extra dev pytest tests/ -q` green;
`tests/utils/test_shared_helpers.py::test_testing_utils_delegates` green (it asserts
the facade delegates); `tests/tests/test_infrastructure_exports.py` green (asserts every
`__all__` entry still resolves — the guard named in that test's docstring).

---

## 7. Risk register

| # | Risk | Evidence | Mitigation |
|---|---|---|---|
| R1 | Name collision: a `gnn/utils/pipeline/` subpackage vs the pre-existing top-level `gnn.pipeline` package | `pipeline.py:11` already delegates to `gnn.pipeline.config` | Name the family `pipeline_orchestration/`; keep `pipeline.py` facade delegating to it |
| R2 | MCP dispatch import order: `mcp_dispatch.py` has 21 import sites and `mcp.py` imports `psutil` at module scope (`mcp.py:24`); if a new package `__init__` eagerly imports the server, the light-import invariant (I1/I2) breaks and every `import gnn.utils` pays for psutil | `__init__.py:11-14` documents this exact invariant; `structured_logging.py:24` same | MCP family moves last; `mcp/__init__.py` empty; server/dispatch are leaves; light-import regression test (Step 0) is the gate |
| R3 | MCP tool registry/discovery breaks on path change | SC-8's dispatcher contract; `scripts/check_mcp_skills_health.py --strict` is the repo's probe for exactly this | No registry edits; old `utils/mcp.py` path stays as a warning facade; skills-health `--strict` in every MCP-adjacent gate |
| R4 | Logging config at import: moving `structured_logging` (module-level `LogAggregator` `:405`, `_correlation_context` threadlocal `:406`) and `visual_logging` into one package creates two logging singletons' init order dependency; `logging/logging_utils.py` (1,338 lines) also configures the pipeline logger | module-level state at `structured_logging.py:405-406` | Move observability modules one at a time, logging last within the family; keep correlation-context tests green per move; never let the new package `__init__` touch logging config |
| R5 | Facade `import *` misses names or adds them twice (the `__all__` already double-lists 4 `log_step_*` names across logging/structured-logging sections) | `__init__.py:353-489` structure; 118 entries vs 113 unique | Facades enumerate explicit re-exports (§4.1 pattern imports `__all__` from the new module); golden-list assertions (I3) catch both drift and duplication |
| R6 | Same-name export/module shadowing (the `performance_tracker` incident) | `__init__.py:253-255` documents it; wave-A rename fixed one instance | I6 (new): any module name must differ from every facade-exported name it defines; import-linter + the golden list enforce |
| R7 | `tests/__init__.py` depends on submodule-path-only names (`PROJECT_ROOT`, `SRC_DIR`, `TEST_DIR`) that no facade export covers | `tests/__init__.py:47-60` | First extraction (Step 1) explicitly re-exports them from the facade; test helpers are in our control so the window can be short |

---

## 8. What remains after this design wave (named honestly)

- The mechanical split itself (Steps 1-7): 7 PRs, each one family, each gated.
- Residual-family assignment (§3.8) and the two owner decisions (facade window length,
  `validate_gnn*` retirement).
- `mypy`/`ruff` cleanliness is maintained per-step; this wave's only code-file touch is
  the `utils/__init__.py` docstring pointer below, which is comment-only.