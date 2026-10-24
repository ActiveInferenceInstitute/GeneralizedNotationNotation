# Python 3.15 backend profiling and headless processing

The lightweight processing surface uses only the Python standard library.
The numerical inference stack keeps its declared frozen runtime and dependency
profiles. These commands use an isolated Python 3.15 interpreter directly from
source; they do not install the complete scientific package on Python 3.15.

## Headless agent interface

```bash
just backend-json --target-dir input/gnn_files
just backend-json --target-dir input/gnn_files/discrete/actinf_pomdp_agent.md --output-dir output/headless
just backend-json --target-dir input/gnn_files --parallel
```

Equivalent direct source command:

```bash
PYTHONPATH=src python3.15 -m gnn.processing --target-dir input/gnn_files
```

Stdout contains one JSON result. Successful processing exits 0. Invalid inputs,
failed files, invalid structure and failed report writes exit 1 and retain
explicit diagnostics. Invalid CLI syntax uses argparse's exit 2 and stderr.
`files` contains the complete discovered selection; `processed_files` contains
only successful, structurally valid siblings. Empty directory selection succeeds
without model work. An explicitly supplied excluded/unsupported file fails.
These checks are lightweight syntax/structure checks, not scientific validation.

Library callers use `from gnn.processing import process_gnn_directory`.
The lightweight API additionally exposes `files_found`, `files_processed`
(successful parses), `files_failed`, per-file errors, parsed metadata and
validation outcomes. Directory success requires every selected file to parse
and validate. An output-write failure is distinct from a model failure.
Report publication uses unique temporary files and atomic replacement; an
interrupted write does not publish a partial JSON document.

## Repeatable measurements

```bash
just profile-backend --target-dir input/gnn_files --output-dir output/profile_315_run1 --repeats 5
just profile-backend --target-dir input/gnn_files --output-dir output/profile_315_process4 --repeats 5 --workers 4 --executor process
```

The output directory must be new and outside the selected input. The profile
admits 1–256 model files, at most 64 MiB total input, and 1–25 repeats. It never
reduces an excessive selection. One warmup is followed by the requested number
of unprofiled wall/CPU timing samples and one separate deterministic trace.
The original trace is excluded from timing statistics.

The directory contains `profile_report.json`, `trace.txt` and `trace.pstats`.
Agents should read JSON or text; pstats is a trusted local binary artifact.
The JSON records every selected path/hash/size, Python/platform identity,
processing-package source hashes, repeat controls, raw timings, median/range,
a normalized result digest and input/source bookend checks. Only volatile
timestamp fields are excluded from result comparisons. Result changes or
input/source drift fail and retain a failed report with available samples.
Bookends do not prove absence of hostile concurrent replacement or ABA edits.
The input/repeat caps do not establish a hard wall-clock deadline.

Variable extraction admits only a complete Unicode word before each declaration
delimiter. A word-boundary guard avoids retrying every suffix of a long word;
the possessive name quantifier avoids backtracking within that complete word.
The following whitespace or delimiter cannot consume a word character, so that
backtracking cannot add a valid match. [Possessive quantifiers are supported
since Python 3.11](https://docs.python.org/3.11/library/re.html#regular-expression-syntax),
the producer's declared compatibility floor. Corpus output digests and explicit
Unicode, multiline, whitespace, digit/underscore and nested-bracket controls
compare this optimization with the original extraction behavior.

Omitting `--workers` preserves serial processing. An explicit count from 1 to
256 selects processes unless `--executor thread` is supplied; one effective
worker runs in the caller. Effective concurrency is bounded by the selected
file count. Windows process pools further cap effective workers at 61;
thread pools retain their selected-file capacity. The requested limit is
preserved separately. The receipt retains requested/effective workers, executor and actual
worker PIDs for the warmup and every timing sample. Repeated-result comparison
removes volatile timestamps and worker PIDs; the separate model-result digest
also excludes the execution-control report, allowing comparison across worker
counts without removing parsed model data.

`cpu_seconds` measures the calling process. POSIX hosts additionally report
`children_cpu_seconds` for terminated worker processes, so parent dispatch/wait
time is not presented as worker process CPU time. Thread workers share the
calling process's CPU measurement. Wall samples include worker startup
and shutdown. The deterministic trace covers the calling thread only; worker
threads and processes are explicitly excluded from its call counts and timing.
Use the unprofiled wall/CPU samples to compare complete parallel workloads.

Python 3.15 uses `profiling.tracing`. The library's compatibility path uses
`cProfile` on earlier supported interpreters. A new profile run never overwrites
an existing destination. Interrupting the in-process workload retains a
cancelled report when Python can handle the interrupt.

## Optional statistical sampling

On a host where process-memory inspection is permitted:

```bash
PYTHONPATH=src python3.15 -m profiling.sampling run --mode wall -o wall_samples.pstats -d 10 scripts/run_backend_profile.py --target-dir input/gnn_files --output-dir output/profile_wall --repeats 5
PYTHONPATH=src python3.15 -m profiling.sampling run --mode cpu -o cpu_samples.pstats -d 10 scripts/run_backend_profile.py --target-dir input/gnn_files --output-dir output/profile_cpu --repeats 5
```

The current Mac denied process-memory access in both wall and CPU sampling
controls. Their original commands and permission errors are retained; they are
unavailable sampling controls, not successful sampling evidence. Deterministic tracing and unprofiled timing ran successfully without
that capability. The implementation does not change OS permissions.

The Python documentation describes [deterministic profiling](https://docs.python.org/3.15/library/profiling.tracing.html)
and [statistical sampling](https://docs.python.org/3.15/library/profiling.sampling.html).
The [Python 3.15.1 release](https://www.python.org/downloads/release/python-3151/)
and profiling tools do not by themselves qualify GNN's full scientific profile;
the [named runtime exception](../../README.md#runtime-defaults-and-required-profiles)
still governs that dependency stack.

## Measured bounded result

At the pre-change source `fc1101a8bb5fa5acc5b420c8d32164623d784d52`, tracing
identified 76 variable-extraction calls for 38 inputs, consuming approximately
94% of the lightweight processing trace. Parsing and validation independently
rescanned each source. Validation now reuses the actual parse metadata.

Five unprofiled warm samples on Python 3.15.0 / macOS arm64 gave a median of
1.304 seconds before and 0.688 seconds after (about 1.90×). Every original
parse/validation field for all 38 authored models matched after removing only
volatile timestamps. The new selection/failure fields are additive. These are
local lightweight processing measurements, not whole-pipeline, framework,
GPU or scientific-inference speed claims. Raw baseline/after samples and exact
working-source hashes are retained with the task's performance evidence.

### 2026-10-23 source-bound follow-up

The 38-file, 3,219,617-byte corpus was measured with one warmup, seven unprofiled
samples and a separate deterministic trace per selection. Complete normalized
model outputs and ordered variable lists matched the original extraction on
every file. The final processing owner hashes and selected input hashes are
retained in each profile receipt; the project metadata, frozen lock and default
runtime configuration were unchanged.

| Interpreter | Original median wall time | Optimized median wall time | Local speedup |
| --- | ---: | ---: | ---: |
| CPython 3.14.9 | 0.31361 s | 0.13443 s | 2.33× |
| CPython 3.15.1 | 0.29545 s | 0.13329 s | 2.22× |

These final-v3 measurements include the corrected worker-future polling and
crash cleanup policy (helper SHA-256 prefix `85acd2dbecc2`). Earlier baseline,
boundary-only, final and final-v2 profiles retain their own source hashes and
were not overwritten by this rerun.

Both interpreters were arm64 builds with the GIL enabled and the experimental
JIT disabled on the same 18-CPU Mac. These figures compare the complete
lightweight workload before and after the regex optimization; they do not
establish a general Python-version speed advantage.

On the final Python 3.15.1 source, two/four thread workers took median
0.13380/0.13267 seconds, and two/four process workers took 0.16006/0.16049
seconds. Every process sample used the requested two/four distinct worker PIDs.
Their parent CPU medians were only 0.00940/0.01271 seconds, while terminated
children consumed 0.21677/0.31689 CPU seconds. Wall timing includes pool startup
and shutdown. Thread medians were close to serial, within observed run variation;
process workers were slower for this small corpus. Serial remains the default;
select a worker count using measurements of the actual workload.

A fresh Python 3.15.1 wall-sampling control again failed with the host's
process-memory permission error. No sampled performance result or permission
change is claimed. The successful unprofiled samples and deterministic traces
remain separate evidence.

### Joint 5.0 source follow-up

A separate final-v4 timing epoch includes the reviewed Windows process-capacity
and typed-configuration projection fixes (worker helper SHA-256 prefix
`1e666e01de42`). On the same 38-file corpus, seven unprofiled samples gave
serial medians of 0.16512 seconds on CPython 3.14.9 and 0.16645 seconds on
CPython 3.15.1. Thread-worker medians on 3.15 were 0.16558/0.16248 seconds for
two/four workers; process medians were 0.22111/0.22626 seconds. Every process
sample returned results from exactly the requested two/four distinct PIDs.
The complete normalized model digest still matches the original reference.

The earlier final-v3 source and measurements remain historical evidence.
These separate timing epochs do not isolate causes of their timing variation;
the fresh worker sweep again supplies no process wall-time speedup for this
small workload. Both current interpreter medians are close. Each receipt
retains raw samples, exact source/input hashes and parent/child CPU statistics;
the scientific environment and its native acceptance remain separate.
