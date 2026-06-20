# v3.0.0 Long-Running Orchestration

This page documents the three **safe-by-design** orchestration contracts introduced for the
v3.0.0 milestone. They are the foundation the TODO requires *before any live infrastructure
mutation*: each module generates, validates, replays, or plans **data only** — none executes a
container, contacts a cluster, or opens a device/sensor stream.

All three live in `src/pipeline/` and follow the repository conventions (Pydantic v2 models,
`from pipeline.X import Y` imports, atomic file writes, deterministic hashing). They are exercised by
no-mocks unit tests in `src/tests/pipeline/` and an end-to-end gate
(`scripts/run_v3_orchestration_acceptance.py`).

## 1. Durable Observation Streams — `pipeline.durable_streams`

Standardizes file- and array-backed stream manifests plus replayable execution traces, so an
extended run can be observed and re-derived deterministically before any live sensor is introduced.

| API | Purpose |
|---|---|
| `StreamManifest.from_array(id, np_array, source=...)` | Manifest with shape/dtype/`n_elements` and a content checksum computed from the array bytes. |
| `StreamManifest.from_file(id, path)` | Manifest checksumming file bytes. |
| `validate_stream_manifest(manifest, base_dir)` | Returns a list of problems (empty == valid); for FILE streams recomputes and compares the checksum — **tampering is detected**. |
| `ExecutionTrace.append_event(step, action, payload_bytes)` | Returns a new trace with the next monotonic `seq` and the payload checksum. |
| `trace_integrity(trace)` | Lists integrity problems: `seq` must start at 0, be contiguous and strictly increasing, with no duplicates and a checksum per event. |
| `replay_trace(trace)` / `verify_replay(trace, digest)` | Deterministic fold of ordered events into a digest, and verification against an expected digest. |
| `write_stream_manifest` / `read_stream_manifest` / `write_trace` / `read_trace` | Atomic JSON IO (temp file + `os.replace`). |

## 2. Long-Running Pipeline Sessions — `pipeline.run_session`

Resumable run manifests, status inspection, and cancellation-safe cleanup for extended
model-family acceptance runs.

| API | Purpose |
|---|---|
| `start_session(session_id, units, created_by=...)` | Create a `RunSession` over `WorkUnit`s with a deterministic `run_hash`. |
| `mark(session, unit_id, status, artifact_refs=..., error=...)` | Return an updated session (immutable-style copy). |
| `checkpoint(session, path)` / `load_session(path)` | Atomic checkpoint write and reload — an interrupted write never corrupts the prior checkpoint. |
| `remaining_units(session)` / `resume_plan(session)` | The `PENDING`/`FAILED` units to resume. |
| `status_report(session)` | `{total, by_status, completed, percent_complete, done}`. |
| `cancel_safe_cleanup(session, workdir)` | Remove **only non-`DONE`** artifact files under `workdir`; path-escape-safe (never deletes outside `workdir`) and idempotent. |

## 3. Auditable Container Plans — `pipeline.container_plan`

Generates validated container plans with a static security review and rollback semantics. **No real
cluster is mutated and no container is executed** — the module imports no `subprocess`, `docker`,
`kubernetes`, or `socket`, and a source-level test enforces this.

| API | Purpose |
|---|---|
| `generate_container_plan(plan_id, specs_config, previous=None)` | Build a **hardened** plan (non-root user, read-only rootfs, `cap_drop=["ALL"]`, pinned `@sha256:` image, resource limits). With `previous`, bumps `version` and attaches a `RollbackDescriptor`. |
| `security_review(plan)` | Returns `Finding`s: privileged → CRITICAL; root/empty user → HIGH; unpinned image → HIGH; plaintext secret in env → HIGH; missing resource limits → MEDIUM; writable rootfs / no cap-drop → LOW. Empty == clean. |
| `compute_plan_hash(plan)` | Deterministic hash over the specs (excludes the hash field). |
| `serialize_plan(plan)` / `plan_to_compose(plan)` | Deterministic JSON / compose-shaped dict (pure data; nothing is written or executed). |

## MCP tools

`src/pipeline/mcp.py` exposes three read-only tools (no live mutation):
`get_v3_orchestration_capabilities`, `run_v3_container_security_review` (demonstrates the review has
teeth against an insecure example), and `run_v3_orchestration_self_check` (in-process checks of all
three contracts).

## Reproduce

```bash
# Unit contracts (no mocks; includes negative controls):
PYTHONPATH=src uv run python -m pytest \
  src/tests/pipeline/test_durable_streams.py \
  src/tests/pipeline/test_run_session.py \
  src/tests/pipeline/test_container_plan.py -q

# End-to-end acceptance gate (fails closed):
PYTHONPATH=src uv run python scripts/run_v3_orchestration_acceptance.py --strict
# Demonstrate the gate is fail-closed:
PYTHONPATH=src uv run python scripts/run_v3_orchestration_acceptance.py --inject-defect  # exits non-zero
```

## Safety model

These modules are deliberately inert with respect to infrastructure: they produce and check
**plans, manifests, and traces**. Promoting them to live behavior (real container execution, live
streams, distributed runs) is gated behind the v3.0.0 → v4.0.0 reliability work and is intentionally
**not** implemented here. See [`../../TO-DO.md`](../../TO-DO.md) for the release path.
