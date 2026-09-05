# Supplemental Source Surface {#sec:source_surface}

This supplement records the top-level source surfaces a reader or reviewer should inspect before turning any prose in the main manuscript into a verifiable claim. Each surface below is authored material under version control; the generated artifacts it produces are described separately so that the boundary between hand-written source and reproducible output stays explicit.

The `src/` tree is the executable core of GeneralizedNotationNotation: A Text Language for Active Inference Models. It is organized into 31 Python packages spanning 565 source files and roughly 190691 lines of code. Among these packages sit the 25 step modules that implement the numbered pipeline (0–24), each module owning one stage of the progression from a parsed GNN text model through visualization, type checking, code export, and executable cognitive simulation. Every claim the manuscript makes about pipeline behavior should be traceable to one of these step modules rather than to descriptive prose alone.

The `input/` tree holds the model corpora that exercise the pipeline. It contains 10 GNN corpora organized by model family, together totaling 29 example files, plus a `model_family_manifest.json` that enumerates the families and their representative models. This manifest is the authoritative registry that downstream steps and the manuscript variable producer read when they report family counts and cross-framework coverage; it should be consulted directly rather than inferred from directory listings.

The `scripts/` tree contains the thin orchestrators that gate and reproduce the project. These include the acceptance scripts that confirm the pipeline runs end to end, the reliability gates that enforce determinism and coverage expectations, and the manuscript variable producer (`src/manuscript_variables.py` driven from this layer) that emits the double-brace `{{...}}` token values consumed throughout the manuscript. Treating these scripts as the source of reproduction commands keeps reported numbers bound to what the code actually computes.

The `doc/` tree is the prose and reference surface, comprising 614 files of specification, tutorial, and design documentation for the GNN language and its Active Inference grounding [@gnn2023]. It is the place to verify that a manuscript statement about GNN syntax or semantics matches the documented language rather than a convenient paraphrase.

The `output/` tree collects per-step pipeline artifacts: the data dumps, intermediate representations, validation reports, and the 105 figures regenerated on each run. Everything here is disposable and reproducible from the surfaces above, so it should be read as evidence of a run rather than as authored source. Notably, `output/data/manuscript_variables.json` is where the manuscript's substituted token values are materialized.

## Expansion Checklist

- Confirm which files are authored source and which are generated.
- Confirm which commands reproduce the current outputs.
- Confirm which values should become manuscript variables.
- Confirm which external references need verified BibTeX entries.
- Confirm whether any private material must be summarized rather than quoted or copied.
