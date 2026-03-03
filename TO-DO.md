# TO-DO — GNN Pipeline Roadmap

**Last Updated**: 2026-03-02  
**Current Version**: 1.3.0

---

## Completed Releases

<details>
<summary><strong>v1.2.0 — Data Accuracy & ActiveInference.jl</strong> ✅ Released 2026-02-23</summary>

Fixed ActiveInference.jl renderer bugs (`POLICY_LEN` typo, `validation_status` reorder), unified test counts (1,319), version strings, execution-time claims, and coverage percentages across 35+ documentation files.
</details>

<details>
<summary><strong>v1.3.0 — MCP Deadlock Fix & Pipeline Hardening</strong> ✅ Released 2026-03-02</summary>

- **MCP deadlock resolved** — multithreading lock in `src/mcp/mcp.py` prevented module discovery; 131 tools now register across 30 modules in <5s
- **LLM glob fix** — `glob("*.md")` → `rglob("*.md")` in `src/llm/processor.py` for recursive file discovery
- **ML class imbalance fix** — `n_folds` capped by `min_class_count` in `src/ml_integration/processor.py`
- **Full documentation audit** — all files synchronized to v1.3.0, 30 modules, 131 tools, dates updated repo-wide

</details>

---

## v1.3.1 — LLM Step Performance & Reliability

> The LLM step (Step 13) is currently disabled due to slow `ollama pull` operations. Fix & optimize.

### Core Performance

- [ ] **Pre-pull guard** — check if model is already available before calling `ollama pull`; skip pull entirely if hash matches
- [ ] **Configurable model** — move model name from hardcoded to `config.yaml` (`llm.model: "llama3.2"`)
- [ ] **Async prompt batching** — merge prompts that analyze the same file into fewer round-trips (currently 9 sequential calls)
- [ ] **Content-hash caching** — skip re-analysis for unchanged GNN inputs; cache keyed on `sha256(input + model + prompt)`

### Developer Ergonomics

- [ ] **`--skip-llm` convenience flag** — alias for `--skip-steps 13` for quick dev cycles
- [ ] **LLM timeout configuration** — expose `llm.timeout_seconds` in `config.yaml` (currently hardcoded 600s)
- [ ] **Graceful degradation** — if LLM step times out, log warning and continue pipeline instead of hard-failing

### Re-enablement Criteria

- [ ] Confirm `ollama pull` completes in <30s for pre-cached models
- [ ] Pipeline run with LLM step enabled completes in <5 min total
- [ ] Re-enable Step 13 in `config.yaml` (`enabled: true`)

---

## v1.3.2 — Test Infrastructure Modernization

> Increase test robustness and enable CI enforcement.

### Test Organization

- [ ] **Add `@pytest.mark.pipeline` marker** — separate full-pipeline integration tests from fast unit tests
- [ ] **Add `@pytest.mark.mcp` marker** — isolate MCP audit tests that require module discovery
- [ ] **Reproducible test count** — generate badge number from `pytest --co -q | tail -1` so docs never go stale
- [ ] **Test timing report** — add `--durations=20` to default pytest config to catch slow tests early

### CI/CD Pipeline

- [ ] **GitHub Actions CI gate** — block merges on pipeline success + test-suite pass
- [ ] **Matrix testing** — Python 3.10, 3.11, 3.12 in CI
- [ ] **Per-module coverage enforcement** — fail CI if any module drops below 80%
- [ ] **Automated pipeline regression test** — single `pytest` integration test running `main.py` on `input/gnn_files/`, asserting 25/25 SUCCESS

### MCP Audit Automation

- [ ] **MCP tool count assertion** — CI test that asserts `tool_count >= 131` to catch regressions in module registration
- [ ] **Stale documentation detector** — CI check that greps for outdated version strings or tool counts in `doc/`

---

## v1.4.0 — GNN Language & Parser Improvements

> Formalize the GNN specification and improve developer experience with the notation.

### Syntax Specification

- [ ] **GNN v1.1 syntax spec** — formalize `## Connections` operator semantics (`>`, `-`, `<`) in `doc/gnn/gnn_syntax.md`
- [ ] **Connection type annotations** — support optional labels: `D >weight=0.5 s`
- [ ] **Matrix dimension validation** — parse state counts from `## StateSpaceBlock` and verify matrix dimensions match
- [ ] **Default value syntax** — allow `A = uniform(3,2)` shorthand for common initializations

### Parser Enhancements

- [ ] **Multi-model file support** — allow multiple `## ModelName` blocks in a single `.md` with independent state-space sections
- [ ] **Better error messages** — report line numbers and expected structure when a section is missing/malformed
- [ ] **GNN schema validation** — JSON Schema for parsed GNN objects; validate at Step 3 and reject malformed specs early
- [ ] **Incremental parsing** — parse only changed sections when re-running pipeline on modified files

### Editor Support

- [ ] **GNN TextMate grammar** — basic `.gnn.md` syntax highlighting for VSCode/Cursor
- [ ] **GNN snippets package** — common templates (POMDP, MDP, HMM) as editor snippets
- [ ] **Markdown front-matter** — support YAML front-matter for GNN metadata (author, version, framework targets)

---

## v1.5.0 — Website & Reporting

> Replace static output with interactive, shareable reports.

### Interactive Dashboard

- [ ] **Single-page app** — replace static `index.html` with a dashboard showing per-step status, artifacts, and drill-down analysis
- [ ] **Pipeline timeline visualization** — Gantt-style chart showing step durations and dependencies
- [ ] **Output artifact browser** — navigate rendered code, analysis results, and visualizations from the browser
- [ ] **Diff-aware reporting** — detect when new pipeline results differ from previous run and highlight deltas

### Export & Sharing

- [ ] **PDF report export** — polished PDF from Step 23 output with embedded visualizations
- [ ] **Self-contained HTML snapshots** — shareable single-file HTML reports (no server required)
- [ ] **Markdown summary export** — auto-generate a `PIPELINE_REPORT.md` after each run

### MCP Client Integration

- [ ] **Web-based MCP playground** — browser UI to invoke individual MCP tools and inspect results
- [ ] **Tool chain builder** — visual tool for composing multi-step MCP workflows
- [ ] **Streaming output** — real-time pipeline progress via Server-Sent Events

---

## v2.0.0 — Deep Roadmap

> Major feature milestones. Breaking changes acceptable.

### Multi-Model Composition

- [ ] Compose multiple GNN models into a single multi-agent simulation
- [ ] Shared state-space definitions with inter-model message passing
- [ ] Visual composition editor in GUI (Step 22)

### GNN Language Server Protocol (LSP)

- [ ] VSCode/Cursor extension with full syntax highlighting, linting, and autocompletion
- [ ] Live preview of parsed state-space and connections as you type
- [ ] Jump-to-definition for matrix references across sections

### Cloud Execution Backend

- [ ] Remote simulation execution for Julia frameworks without local install
- [ ] Pipeline-as-a-Service API — submit a GNN file, receive rendered code and results
- [ ] GPU-accelerated JAX execution on cloud instances

### Provenance & Reproducibility

- [ ] Content-addressable model registry — every GNN spec gets a unique hash
- [ ] Immutable pipeline run records with full input/output/config snapshot
- [ ] `gnn reproduce <run-hash>` CLI command

### Framework Ecosystem Expansion

- [ ] **SPM** (Statistical Parametric Mapping) renderer and executor
- [ ] **Stan** probabilistic programming renderer
- [ ] **Turing.jl** renderer for Julia-native probabilistic programming
- [ ] **NumPyro** (JAX-based PPL) renderer enhancement

---

## Conventions

- Versions follow [SemVer](https://semver.org/) — `MAJOR.MINOR.PATCH`
- Patch releases (`x.y.z`) should be completable in 1 focused session
- Minor releases (`x.y.0`) should be completable in 1–3 focused sessions
- Deep roadmap items are tracked for visibility but not scheduled until prior milestones ship
