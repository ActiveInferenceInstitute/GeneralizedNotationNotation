# ADR 049: Dual async-job surfaces

- **Status:** Accepted
- **Date:** 2026-09-22
- **Scope item:** integration-surface census, 2026-09-21 (F3)
- **Links backward:** [ADR 0001 — Consolidated pipeline execution](0001-consolidated-pipeline-execution.md)

## Context

**Provenance.** The `src/gnn/api/` module ships two FastAPI factories that
both execute the same GNN pipeline asynchronously, and the integration-surface
census (F3, 2026-09-21) flagged the pair for a canonical-model decision: is
one of them vestigial, or are both canonical for different consumers?

**The two surfaces, re-verified in this tree:**

- **Runs surface — `gnn.api.app`** ([src/gnn/api/app.py](../../src/gnn/api/app.py)),
  the app behind `gnn serve`
  (`src/gnn/cli/__init__.py:1124` dispatches `_cmd_serve`, which imports and
  starts `gnn.api.app.start_server` at `src/gnn/cli/__init__.py:1157-1159`).
  A run is content-hash addressed: `POST /api/v1/run`
  (`src/gnn/api/app.py:122`) computes `compute_run_hash`
  (`src/gnn/api/app.py:127`) over the request and deduplicates against
  existing records before creating one. Supporting routes: `GET
  /api/v1/runs/{run_hash}` (`src/gnn/api/app.py:201`), the Markdown report
  (`src/gnn/api/app.py:222`), `DELETE /api/v1/runs/{run_hash}` (cancel-then-
  delete, `src/gnn/api/app.py:235`), the SSE progress stream
  (`src/gnn/api/app.py:260`), and `GET /api/v1/runs`
  (`src/gnn/api/app.py:291`). The store is `processor.RUNS_STORE`
  (`src/gnn/api/processor.py:39`, re-exported at `src/gnn/api/app.py:55`);
  the executor is app-local `_execute_pipeline`
  (`src/gnn/api/app.py:381`) driving a `RunTracker`
  (`src/gnn/api/app.py:309`) with a `CancelToken`
  (`src/gnn/api/app.py:184`).
- **Jobs surface — `gnn.api.server`**
  ([src/gnn/api/server.py](../../src/gnn/api/server.py)), a module-scope app
  for `python -m gnn.api.server` / `uvicorn gnn.api.server:app`
  (`src/gnn/api/server.py:282`). A job is an opaque ID with explicit step
  management: `POST /api/v1/process` queueing execution via BackgroundTasks
  (`src/gnn/api/server.py:111`, `:141`), `GET /api/v1/jobs/{job_id}`
  (`src/gnn/api/server.py:159`), `DELETE /api/v1/jobs/{job_id}`
  (`src/gnn/api/server.py:187`), `GET /api/v1/jobs`
  (`src/gnn/api/server.py:205`), and `GET`/`POST /api/v1/tools[/{step}]`
  for tool listing and single-step invocation
  (`src/gnn/api/server.py:215`, `:222`). The store is `processor._JOBS`
  (`src/gnn/api/processor.py:36`) via `create_job`/`get_job`/`cancel_job`/
  `list_jobs` (`src/gnn/api/processor.py:64`, `:129`, `:173`, `:296`); the
  executor is `execute_job_async` (`src/gnn/api/processor.py:304`).

**Why both exist.** The run surface is the webui-style run UX: runs are
deduplicated by content hash, streamed live, cancelable, and produce a
downloadable report — the vocabulary of "run this pipeline and watch it".
The job surface is the explicit job/step management API: opaque job IDs,
polling, cancellation, and single-step invocation (`POST
/api/v1/tools/{step}`) — the vocabulary of "queue this work and manage it".
The MCP tool surface consumes the job vocabulary for job lifecycle
(`gnn_get_job_status_mcp`, `gnn_cancel_job_mcp`, `gnn_list_jobs_mcp` at
`src/gnn/api/mcp.py:81`, `:93`, `:130`) and the run vocabulary for run
cleanup (`gnn_delete_run_mcp` at `src/gnn/api/mcp.py:111`, which calls
`delete_run` (`src/gnn/api/processor.py:228`, defaulting to `RUNS_STORE`
at `src/gnn/api/processor.py:252`)).

**Shared infrastructure — both factories:**

- register the identical 12-route parity set via `register_parity_routes`
  (`src/gnn/api/parity.py:859`, wired at `src/gnn/api/app.py:106` and
  `src/gnn/api/server.py:95`), including the verify-only
  `POST /api/v1/reproduce` (resolves and verifies an indexed run and
  returns the reconstructed configuration; execution is dispatched by the
  caller — the CLI re-executes locally, the runs surface starts it via
  `POST /api/v1/run`);
- use the canonical `{status, data, error, meta}` envelope
  (`src/gnn/api/responses.py:32`, `:51`, `:61`) with the same installed
  exception handlers (`src/gnn/api/responses.py:115`);
- share the auth policy (`api_key_middleware` + `require_secure_bind` at
  `src/gnn/api/app.py:41` and `src/gnn/api/server.py:32`) and the rate-limit
  policy (`rate_limit_middleware` at `src/gnn/api/app.py:103` and
  `src/gnn/api/server.py:92`);
- both boot green: `tests/api/test_api_entrypoint.py:62` boots
  `python -m gnn.api.server` as a subprocess and answers `/api/v1/health`.

Collapsing either surface is a behavior change — route sets, status codes,
and the documented entry points would move or vanish — with no census
receipt that any consumer wants that collapse.

## Decision

**Documented dual.** Both async-job surfaces are canonical, each for its
consumer class; neither is vestigial:

- `gnn.api.app` is the canonical **run UX surface**: content-hashed,
  deduplicated, streamable runs behind `gnn serve`.
- `gnn.api.server` is the canonical **job/step management API surface**:
  queued jobs plus single-step tool invocation behind
  `python -m gnn.api.server` / `uvicorn gnn.api.server:app`.

Coherence comes from shared infrastructure plus pinned route tables, not
from merging the models:

1. the same `processor` module owns both stores (`_JOBS` and `RUNS_STORE`);
2. both factories register the identical parity route set, and the
   per-factory route tables are pinned in
   [tests/api/test_api_parity.py](../../tests/api/test_api_parity.py)
   (`PARITY_ROUTES`, `RUN_FACTORY_ROUTES`, `JOB_FACTORY_ROUTES`,
   `META_ROUTES` — `tests/api/test_api_parity.py:53`, `:70`, `:80`);
3. the envelope and the auth/rate-limit policy are identical on both
   factories;
4. the MCP job tools bind the Jobs store; `gnn_delete_run_mcp` binds the
   Runs store;
5. the MCP submit tool is documented as **record-creation only**: it
   creates a pending job record and returns immediately; no built-in
   executor consumes externally created records, and `gnn serve` never
   consumes them — execution starts only where the caller also starts it
   (`POST /api/v1/process`, `POST /api/v1/tools/{step}`, or a direct
   `execute_job_async` call) (`src/gnn/api/mcp.py:38-72`).

## Consequences

**Positive**

- No behavior change: both surfaces, their route sets, and their entry
  points remain exactly as they are.
- Honest MCP contract: the submit tool no longer implies an execution
  guarantee it never had; its message states plainly that the job is
  created but not started.
- Drift between the surfaces is caught by tests rather than by review:
  the pinned route tables assert which routes are shared and which are
  exclusive to each factory.

**Negative / accepted limitations**

- Two job vocabularies (`run`/`run_hash` vs `job`/`job_id`) must be kept
  coherent. Accepted: the split is meaningful (content-hash identity vs
  opaque queue IDs), and coherence is enforced by the pinned parity surface
  and the shared `processor` module rather than by documentation alone.
- A contributor must learn which surface serves which consumer. Mitigated
  by [src/gnn/api/AGENTS.md](../../src/gnn/api/AGENTS.md) and this record.

## References

- [src/gnn/api/app.py](../../src/gnn/api/app.py) — Runs factory (route
  registrations 110-291, `_execute_pipeline:381`, `RunTracker:309`,
  `CancelToken:184`)
- [src/gnn/api/server.py](../../src/gnn/api/server.py) — Jobs factory
  (route registrations 97-260, module-scope `app:282`)
- [src/gnn/api/processor.py](../../src/gnn/api/processor.py) — both stores
  and executors (`_JOBS:36`, `RUNS_STORE:39`, `create_job:64`,
  `execute_job_async:304`, `delete_run:228`)
- [src/gnn/api/parity.py](../../src/gnn/api/parity.py) `:859`;
  [src/gnn/api/responses.py](../../src/gnn/api/responses.py) `:32`;
  [src/gnn/api/mcp.py](../../src/gnn/api/mcp.py) `:38`
- `src/gnn/cli/__init__.py:1124` — `gnn serve` dispatches to
  `gnn.api.app.start_server`
- [tests/api/test_api_parity.py](../../tests/api/test_api_parity.py) `:53`
  — pinned route tables
- [tests/api/test_api_entrypoint.py](../../tests/api/test_api_entrypoint.py)
  `:62` — `python -m gnn.api.server` boots and serves health
- Integration-surface census, 2026-09-21 (F3)

## Status log

| Date | Slice | Status | Notes |
|------|-------|--------|-------|
| 2026-09-22 | Initial decision | Accepted | Dual documented with the coherence contract (shared processor module, pinned route tables, record-creation-only MCP submit) |