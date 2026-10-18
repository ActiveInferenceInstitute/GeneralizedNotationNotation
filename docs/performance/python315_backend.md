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
