# Specification: Input

## Design Requirements

`input/` is the data tree consumed by the pipeline; it contains no Python source. Its contract is defined by the files it holds and the modules that read them:

- **GNN spec files** under `gnn_files/<task folder>/*.md` follow the normative syntax in [`../doc/gnn/reference/gnn_syntax.md`](../doc/gnn/reference/gnn_syntax.md) and are discovered by `gnn.discovery.is_model_source_path`, which excludes the `INDEX.md`, `AGENTS.md` and `README.md` scaffolds (plus `*.example.md` / `*.template.md`).
- **`config.yaml`** routes the run: pipeline steps and skip lists, Step 2 test mode (`fast_only`), timeouts, and per-step options. It is merged with CLI arguments by `src/utils/pipeline_config_merge.py`.
- **`model_family_manifest.json`** declares acceptance profiles per model family (`schema`, `acceptance_profile_defaults`, `families`) for `scripts/run_session_acceptance.py`.
- **`multi_agent_models/`** and **`recursive_models/`** are acceptance fixtures for the roadmap verification commands in `TO-DO.md`.

## Model-kind contract

Each spec is one of two kinds, classified by `render.pomdp_contract.detect_model_kind`:

- **Discrete** — categorical `A/B/C/D[/E]` matrices (flat, factored, hierarchical, multi-agent or learning variants). Renders and executes on all nine frameworks declared in `src/render/framework_registry.py`.
- **Continuous** — a linear-Gaussian state-space block only: `F` (dynamics), `H` (observation), `Q`/`R` (noise covariances), `prior_mean`/`prior_cov`, and optionally `goal_mean`/`control_gain` for closed-loop control. Renders and executes on the frameworks whose registry entry has `supports_continuous: True` (JAX, NumPyro, PyTorch, Stan, RxInfer.jl); the others report the `unsupported` render status, which is excluded from success rates and never executed by Step 12.

## Components

No classes or functions are exported from this directory. Consumers: `src/gnn/` (parsing and discovery), `src/render/` (code generation), `src/execute/` (execution), `src/utils/pipeline_config_merge.py` (configuration).
