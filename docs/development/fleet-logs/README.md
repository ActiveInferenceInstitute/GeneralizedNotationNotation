# Fleet logs

Per-worker progress checkpoints and end-of-turn REPORTs from herdr-dispatched
module fleets. Convention (2026-09-04, fleet 3, after two boot-time fleet
losses): every fleet worker appends one line per completed phase here and
writes its final REPORT.md HERE FIRST (repo = durable), then mirrors it to
~/.omp/agent/fleet-manifests/gnn-module-fleet3-2026-09-04/reports/.
Nothing task-critical lives in /tmp any more.
Orchestrators: see ~/.omp/agent/fleet-manifests/.
