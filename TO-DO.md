# TO-DO — GNN Pipeline Roadmap

**Last Updated**: 2026-02-23  
**Current Version**: 1.2.0

---

## ~~v1.2.0 — Data Accuracy & ActiveInference.jl~~ ✅ Released 2026-02-23

> Polish what exists. No new features — only correctness and data integrity.

- [x] **ActiveInference.jl renderer bugs fixed** — `POLICY_LEN` typo and `validation_status` reorder applied
- [x] **ActiveInference.jl output path** — executor already correctly wires `simulation_data/` discovery; blocker was renderer bugs (now fixed)
- [x] **Update `doc/gnn/implementations/README.md`** — correlation entries marked *pending* until Julia runtime re-verified
- [x] **Sync test counts across docs** — unified to 1,319 across 35+ files
- [x] **Sync version strings across docs** — all files updated to v1.2.0 (was mix of v1.1.0 and v1.1.4)
- [x] **Sync execution-time claims** — standardized to "~5 minutes (with LLM step)" across all docs
- [x] **Sync coverage percentages** — unified `src/AGENTS.md` and root `AGENTS.md` module status matrices
- [x] **Sync dates** — all docs updated to 2026-02-23

---

## v1.2.1 — LLM Step Performance

> The LLM step (step 13) is the largest single bottleneck at ~93 s (~36 % of total pipeline time).

- [ ] **Profile prompt calls** — identify which of the 9 Ollama prompts are slowest
- [ ] **Batch compatible prompts** — merge prompts that analyze the same file into fewer round-trips
- [ ] **Add `--skip-llm` convenience flag** — alias for `--skip-steps 13` for quick dev cycles
- [ ] **Cache LLM results** — skip re-analysis for unchanged GNN inputs (content-hash comparison)

---

## v1.3.0 — Testing & CI Improvements

- [ ] **Automated pipeline regression test** — a single `pytest` integration test that runs `main.py` on `input/gnn_files/`, asserts 25/25 SUCCESS, and spot-checks key output artefacts (connections > 0, analysis count > 0, etc.)
- [ ] **GitHub Actions CI gate** — block merges on pipeline success + test-suite pass
- [ ] **Reproducible test count** — generate the badge number programmatically from `pytest --co -q | tail -1` so docs never go stale
- [ ] **Add `@pytest.mark.pipeline` marker** — separate full-pipeline integration tests from fast unit tests
- [ ] **Per-module coverage enforcement** — fail CI if any module drops below 80 %

---

## v1.4.0 — GNN Language & Parser Improvements

- [ ] **GNN v1.1 syntax specification** — formalize the `## Connections` operator semantics (`>`, `-`, `<`) in `doc/gnn/gnn_syntax.md` (currently underdocumented, 2.7 KB)
- [ ] **Multi-model file support** — allow multiple `## ModelName` blocks in a single `.md` with independent `## StateSpaceBlock` / `## Connections` sections
- [ ] **GNN schema validation** — JSON Schema for parsed GNN objects, validate at Step 3 and reject malformed specs early
- [ ] **Better parser error messages** — report line numbers and expected structure when a section is missing or malformed
- [ ] **Connection type annotations** — support optional labels on connections (`D >weight=0.5 s`)

---

## v1.5.0 — Website & Reporting

- [ ] **Interactive website dashboard** — replace static `index.html` with a single-page app showing per-step status, output artifacts, and drill-down analysis
- [ ] **Diff-aware reporting** — detect when a new pipeline run's results differ from the previous run and highlight deltas in the report
- [ ] **PDF/HTML report export** — generate a polished PDF or self-contained HTML report from Step 23 output
- [ ] **Shareable pipeline summary URLs** — generate self-contained HTML snapshots of pipeline runs that can be shared without a server

---

## v2.0.0 — Deep Roadmap

> Major feature milestones. Breaking changes acceptable.

### Multi-Model Composition

- [ ] Compose multiple GNN models into a single multi-agent simulation
- [ ] Shared state-space definitions with inter-model message passing
- [ ] Visual composition editor in GUI (Step 22)

### GNN Language Server Protocol (LSP)

- [ ] VSCode/Cursor extension with syntax highlighting, linting, and autocompletion for `.gnn.md` files
- [ ] Live preview of parsed state-space and connections as you type
- [ ] Jump-to-definition for matrix references across sections

### Cloud Execution Backend

- [ ] Remote simulation execution for Julia frameworks (RxInfer, ActiveInference.jl) without local Julia install
- [ ] Pipeline-as-a-Service API — submit a GNN file, receive rendered code and simulation results
- [ ] GPU-accelerated JAX execution on cloud instances

### Provenance & Reproducibility

- [ ] Content-addressable model registry — every GNN spec version gets a unique hash
- [ ] Immutable pipeline run records with full input/output/config snapshot
- [ ] `gnn reproduce <run-hash>` CLI command to exactly replay any prior run

### Framework Ecosystem Expansion

- [ ] **SPM (Statistical Parametric Mapping)** renderer and executor
- [ ] **Stan** probabilistic programming renderer
- [ ] **Turing.jl** renderer for Julia-native probabilistic programming
- [ ] **NumPyro** (JAX-based PPL) renderer

---

## Conventions

- Versions follow [SemVer](https://semver.org/) — `MAJOR.MINOR.PATCH`
- Each minor release should be completable in 1–3 focused sessions
- Deep roadmap items are tracked here for visibility but not scheduled until prior milestones ship
