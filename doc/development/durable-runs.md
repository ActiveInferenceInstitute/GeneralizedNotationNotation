# Durable runs and reproducible inputs

A run ID identifies one invocation. Its `gnn-run-v2` hash identifies the
canonical relative model paths and file contents, effective arguments, selected
steps, pipeline/input configuration and live child runtime configuration.
Output locations, logging choices and invocation IDs do not change that hash.
Renaming or adding a supported model does.

`gnn reproduce` validates the complete saved input inventory, configuration
and resolved step selection before dispatch. Changed or deleted sources,
changed runtime configuration, malformed history and unbound records without
verifiable identity are rejected. Create a new indexed run for unbound history;
a digest refresh is not reproduction of the old run. Sources and configuration
must remain stable during execution; final summary publication rechecks them.

A session reuses DONE only when source, family/framework/profile, acceptance
policy and complete output artifact bytes still match. Unbound DONE entries (records without a verifiable identity)
rerun. Interrupted RUNNING units resume, and cleanup protects artifacts
referenced by DONE units, including resolved aliases and descendants. One
writer must own a session/output directory at a time: atomic checkpoint
replacement does not provide a cross-process execution lock.

Manifest index schema 3.1 verifies the complete discovered artifact inventory,
stream identities and top-level summary provenance. Missing entries, duplicate
identities, escapes and inconsistent traces fail validation. Indexes below schema 3.1
require explicit re-emission. Individual files are published atomically; a
crash across several manifest writes can require re-emission. Directory-only
verification remains an explicitly narrower fallback.

Container review normalizes numeric root identities and sensitive mount paths.
Compose export preserves reviewed capabilities, network/PID/IPC settings and
named volumes; output arguments match their writable mount. These are static
plan checks. They do not validate an image or deployment.

Implementation: [session acceptance](../../src/pipeline/session_acceptance.py),
[run identity](../../src/pipeline/hasher.py),
[manifest verification](../../src/pipeline/run_manifest.py), and
[container plans](../../src/pipeline/container_plan.py).
