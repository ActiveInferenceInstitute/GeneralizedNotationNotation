# TO-DO - GNN Pipeline Roadmap

**Last Updated**: 2026-08-02
**Current Version**: 3.0.0
**Next Target**: v4.0.0 (bounded autonomy and reviewed self-editing workflows)

**Last reviewed**: 2026-08-02 — mega-deep documentation review pass; completed
items were removed from this file per the conventions below. The full audit
trail of that pass (findings, fixes, verification) is in
[REVIEW_LOG_2026-08-02.md](REVIEW_LOG_2026-08-02.md).

## Current 3.0.0 Status

GNN v3.0.0 is released. The long-running orchestration contracts are in place:
durable observation streams, resumable run sessions, and auditable container
plans. Current local catch-up checks pass for the strict v3.0.0 acceptance gate,
run-manifest emission from `output/`, and container-plan generation from
`input/config.yaml`. The session-wrapped all-family acceptance path has also
been exercised with 9 of 9 families marked `DONE` and no failed units.

This roadmap is forward-only. Shipped-version history belongs in `CHANGELOG.md`,
release notes, and verification artifacts, not in this open-work queue.

## Open Work — Documentation Review Follow-ups (scoped 2026-08-02)

Two open workstreams remain from the 2026-08-02 mega-deep docs review. Each is
scoped with concrete tasks, file paths, verification commands, and acceptance
criteria. No other open roadmap items are currently tracked.

---

### Workstream 1 — Dead external citations & transient links (Major)

**Goal**: eliminate the remaining dead/transient external URLs in maintained
docs, without fabricating citations. Inventory was produced by
`scripts/check_external_links.py` (58 findings: 34×404 + 24 network errors, of
which several are bot-block false positives and regex artifacts).

**Task 1.1 — P0: Remove transient personal-upload links (hygiene, do first)**

Perplexity chat-export paste links (`ppl-ai-file-upload.s3.amazonaws.com`) are
transient, personal, and must not appear in a public repo:

- `doc/nock/jock/jock.md:146` and `:484` — footnote `[1]` →
  `https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/343060/6c2825a0-7e1f-40b8-bfdb-985b88a60757/paste.txt`
- `doc/nock/nockchain/nockchain.md:110` and `:556` — footnote `[1]` →
  `https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/343060/d6948f28-a712-4d0f-86aa-fb34c4bf0ff0/paste.txt`

Action: for each file, identify what the pasted content cited (read the
surrounding prose — both docs attribute repository/source claims to `[1]`),
replace the footnote with the actual stable source (GitHub repo or paper URL,
verified 200), or remove the footnote together with its inline `[1]` markers.
Verify no dangling `[N]` references remain (`grep -n "\[1\]" <file>`).

**Task 1.2 — P1: Malformed arXiv IDs (resolve via citation context)**

- `doc/arc-agi/arc-agi-gnn.md:1172` — `http://arxiv.org/pdf/21.3.00848.pdf`
  (invalid ID "21.3.00848"). Find the intended paper from the inline citation
  context and replace with `https://arxiv.org/abs/<real-id>`.
- `doc/vec2text/vec2text_gnn.md:284` and `:291` — `arxiv.org/html/2406.1.3.0v1`
  and `arxiv.org/abs/2406.1.3.0` (invalid ID "2406.1.3.0"). Same treatment.
- `doc/vec2text/vec2text_gnn.md:216` — `arxiv.org/html/2411.05034` (404);
  verify `https://arxiv.org/abs/2411.05034` and the `.../html/2411.05034v1`
  variant before swapping.

**Task 1.3 — P1: Moved/deleted pages with known live targets**

- `doc/dspy/gnn_dspy.md:493` and `:496` — dead `github.com/stanfordnlp/dspy/
  blob/main/docs/docs/deep-dive/modules/...` paths → `https://dspy.ai/learn/`
  or the current repo path.
- `doc/timep/timep.md:241` — `github.com/jkool702/timep/blob/main/TESTS/OUTPUT/
  out.profile` (removed upstream) → repo root or drop the link, keeping the text.
- `doc/activeinference_jl/activeinference-jl_source_code.md` — Turing docs link
  (`turing.ml/dev/docs/using-turing/samplers/`) → verify the current home
  (`turinglang.github.io/Turing.jl/...`) before swapping.
- `doc/d2/d2.md:59` and `src/mcp/model_context_protocol.md:298` — **DONE** in
  this pass (glued footnote marker `releases.[1]` → `releases[1]`; cline
  quickstart → `docs.cline.bot/`).

**Task 1.4 — P2: Retired services & dead hosts (Wayback or canonical replacement)**

- CiteSeerX (service retired) ×3:
  `doc/arc-agi/arc-agi-gnn.md:1211`, `doc/onefilellm/onefilellm_gnn.md:297`,
  `doc/quadray/quadray.md:359`. Each cited paper has a title in the surrounding
  text → locate its DOI/arXiv version (Crossref/arXiv API) and swap.
- `doc/vec2text/vec2text_gnn.md:253` — dead `core.ac.uk/download/pdf/
  288349048.pdf` → core.ac.uk article landing page or DOI.
- `doc/activeinference_jl/activeinference-jl.md:171` — `nms.kcl.ac.uk/
  osvaldo.simeone/freeenergymin.pdf` → Simeone's paper (arXiv version).
- `doc/quadray/quadray.md:367` — archive.org details page (item removed) →
  Wayback capture or alternative Synergetics source.
- Dead university hosts (Wayback capture or canonical replacement):
  `doc/spm/spm.md:115` (brainresearch.de), `:123` and `:143`
  (brainimaging.waisman.wisc.edu ×2 → `fil.ion.ucl.ac.uk/spm/doc/`),
  `doc/timep/timep.md:221` (lacasa.uah.edu), `doc/arc-agi/arc-agi-gnn.md:1214`
  (pcl.sitehost.iu.edu) and the `web.eecs.umich.edu` + `zora.uzh.ch` + 
  `mathaware.org` citations in the same file, `doc/kit/gnn_kit.md:352`
  (pfw.edu), `doc/type-inference-zoo/type-inference-zoo.md:161`
  (pls-lab.org) and the `papl.cs.brown.edu` citation.
- Dead domains (drop hyperlink, keep text, or Wayback):
  `doc/nock/nockchain/nockchain.md:120` (forum.nockchain.org), `:130`
  (relidator.com), `:137` (icodrops.com/nockchain), `:112` (zorp-corp/nockapp),
  `doc/onefilellm/onefilellm_gnn.md:251` (hub.athina.ai), `:252`
  (docfork.com llms.txt), `:262` (juejin.cn post), `doc/arc-agi/arc-agi-gnn.md`
  (arcprize.kongjiang.org mirror), `doc/quadray/quadray.md:75`
  (cosmometry.net), `doc/timep/timep.md:223` (dev.to post),
  `doc/dspy/gnn_dspy.md` (aidoczh.com mirror), `doc/onefilellm/onefilellm_gnn.md`
  (kdjingpai.com mirror).

**Task 1.5 — P3: Browser-verify bot-blocked hosts (do not delete blindly)**

These return 403/404/406 to automated clients but are usually fine in a
browser: paperswithcode ×6 (`doc/arc-agi/arc-agi-gnn.md`,
`doc/vec2text/vec2text_gnn.md`, `doc/activeinference_jl/activeinference-jl.md`),
crates.io ×8 (`doc/iroh/iroh.md`), medium.com ×5 (`doc/bnlearn/README.md`),
direct.mit.edu ×4, dl.acm.org ×3, stackoverflow ×9, sourceforge ×2, lib.rs ×2,
royalsocietypublishing, academic.oup.com, npmjs.com/search,
scholar.google/scholar.harvard, ssrn, sciprofiles, community.cisco,
darreljarmusch.com (406), academia.edu ×2, cs.cmu.edu ×2. Verify in a real
browser session; only fix confirmed-dead ones; leave the rest with a comment
in the review log.

**Task 1.6 — P2: Checker accuracy (already partially done)**

`scripts/check_external_links.py` now strips trailing backticks from captured
URLs (kills the `storage.googleapis.com/jax-releases/*.html` and
`learn.microsoft.com` false positives — those URLs are live). Optional next
step: bucket 403/429/999 as "bot-blocked (verify in browser)" in the report so
they do not surface as findings.

**Verification (Workstream 1)**

```bash
uv run --extra dev python scripts/check_external_links.py --strict
# Expect: only bot-blocked hosts (403/429/999) + documented exceptions
uv run --extra dev python doc/development/docs_audit.py --strict --check-anchors --no-write
uv run --extra dev python scripts/check_repo_terminology.py --strict
grep -rn "ppl-ai-file-upload" doc/   # expect: no output
```

**Acceptance**: zero 404/ERR findings that are not bot-blocked or
Wayback-annotated; no dangling footnote `[N]` references; no
`ppl-ai-file-upload` URLs in the tree.

**Effort**: 1.1 (S), 1.2 (S), 1.3 (S), 1.4 (M), 1.5 (M), 1.6 (S).

---

### Workstream 2 — Full CI parity: Julia backends + Ollama (Major)

**Goal**: make the full test suite pass locally (0 failed, 0 skipped) so
`README.md`/`AGENTS.md`/`SETUP_GUIDE.md` can report a green run. Currently
2,622 passed / 1 failed, where the failure is environment-dependent:
`src/tests/pipeline/test_pomdp_gridworld_cross_framework.py::test_gridworld_render_execute_analyze_visualize_strict`
requires the Julia packages RxInfer and ActiveInference.jl. Julia 1.12.6 is
already installed (`/opt/homebrew/bin/julia`).

**Task 2.1 — Julia packages in a clean project env**

```bash
export PATH="$HOME/.juliaup/bin:$PATH"
julia --project=/tmp/julia_test_env --startup-file=no -e 'using Pkg; Pkg.add(["RxInfer", "ReactiveMP", "GraphPPL", "ActiveInference", "Distributions", "StatsBase", "JSON"])'
# 10-20 min with precompilation; the DistributionsAD @check_args patch in
# src/execute/activeinference_jl/setup_environment.jl handles Julia 1.12 compat
```

**Task 2.2 — Re-run the failing test**

```bash
uv run --extra dev python -m pytest src/tests/pipeline/test_pomdp_gridworld_cross_framework.py::test_gridworld_render_execute_analyze_visualize_strict -q --tb=short
```

If the test needs the packages on the default project path instead of
`/tmp/julia_test_env`, add the packages to the default env or set `JULIA_PROJECT`
as the test expects (check `_assert_julia_packages()` in the test file).

**Task 2.3 — Ollama LLM tests**

```bash
brew install ollama   # or: curl -fsSL https://ollama.com/install.sh | sh
ollama serve &        # background daemon
ollama pull smollm2:135m-instruct-q4_K_S
uv run --extra dev python -m pytest src/tests/llm/test_llm_ollama.py src/tests/llm/test_llm_ollama_integration.py -q --tb=short
```

**Task 2.4 — Full parity run and doc evidence refresh**

```bash
# command of record, WITHOUT the two Ollama ignores once the daemon is up
uv run --extra dev python -m pytest src/tests/ -q --tb=no -rsx --timeout=300
# optional CI-parity extras:
uv run --extra dev bandit -r src -c pyproject.toml --severity-level medium --confidence-level medium -q
PYTHONPATH=src uv run --frozen python scripts/run_v3_orchestration_acceptance.py --strict
```

Update the evidence numbers in `README.md` (Test Suite Evidence),
`AGENTS.md` (Current Validation), `SETUP_GUIDE.md` (Latest Validation), and
`doc/HANDOFF.md` (state table) with the measured results.

**Acceptance**: full suite 0 failed / 0 skipped; the two Ollama test files
re-enabled in the documented command of record; all evidence docs updated with
measured numbers; `git diff --check` clean.

**Effort**: 2.1 (L), 2.2 (S), 2.3 (M), 2.4 (S).

---

## v4.0.0 - Bounded Autonomy & Reviewed Self-Editing

The local bounded-autonomy surface now emits proposal-only artifacts via
`--autonomous`: candidate scores, review gates, rollback descriptors, audit
events, and non-mutating security policy. No source edit, commit, container run,
or cluster mutation is automatic.

No additional v4.0.0 implementation item is open in this roadmap at this time.

## Verification Commands

Use `uv run --frozen` for roadmap catch-up checks until `uv.lock` is deliberately
refreshed.

```bash
PYTHONPATH=src uv run --frozen python scripts/run_v3_orchestration_acceptance.py --strict
PYTHONPATH=src uv run --frozen python scripts/emit_run_manifest.py output --out /tmp/gnn-v3-run-manifest
PYTHONPATH=src uv run --frozen python scripts/generate_pipeline_container_plan.py --config input/config.yaml --out /tmp/gnn-v3-container-plan.json
PYTHONPATH=src uv run --frozen python scripts/run_session_acceptance.py --manifest input/model_family_manifest.json --output-dir /tmp/gnn-v3-session-acceptance --session /tmp/gnn-v3-session.json --strict
PYTHONPATH=src uv run --frozen python src/main.py --autonomous --target-dir input/gnn_files --output-dir /tmp/gnn-autonomous-smoke

uv run --frozen --extra dev python doc/development/docs_audit.py --strict --check-anchors --no-write
uv run --frozen --extra dev python scripts/check_gnn_doc_patterns.py --strict
uv run --frozen --extra dev python scripts/check_maintained_doc_terms.py --strict
uv run --frozen --extra dev python scripts/check_repo_terminology.py --strict
uv run --frozen --extra dev python scripts/check_external_links.py   # informational
git diff --check
```

## Conventions

- Keep this file limited to unchecked, forward-looking work.
- Move shipped-version details to release notes, changelog entries, or durable
  verification artifacts.
- Keep closed work out of this file: completed items are removed when they
  land; the audit trail lives in `REVIEW_LOG_2026-08-02.md` and git history.
- Scope open items with concrete tasks, file paths, verification commands, and
  acceptance criteria so the next session can execute without re-deriving them.
