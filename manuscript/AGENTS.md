# Manuscript Agent Notes - GeneralizedNotationNotation

This directory follows the docxology/template manuscript contract:

- `00_` through `09_` files are main sections.
- `S01_` files are supplemental material.
- `98_` and `99_` files are back matter.
- Every manuscript section file starts with one H1 and a stable `{#sec:...}` label.
- Citations must use Pandoc syntax and resolve in `references.bib`.
- Generated numbers belong behind `{{TOKEN}}` variables, not hard-coded prose.

## Editing Rules

- Treat this scaffold as an outline until project-specific evidence is bound.
- Do not fabricate results, benchmark numbers, citations, DOIs, or release claims.
- Keep project-specific computation in source modules and scripts; keep manuscript files as prose and evidence maps.
- Prefer explicit paths to source surfaces when describing evidence.
- If adding figures, write them under `../output/figures/` and reference them with Pandoc-crossref labels.

## Current Scope

A standardized text-based language and processing pipeline for Active Inference generative models, transforming specifications into validation, visualization, simulation, and analysis outputs.

Evidence boundary: Do not treat root output churn as manuscript evidence until the specific run target and generated artifacts are recorded.

## Figures

- `output/figures/figure_registry.json` is **generated** by
  `python -m scripts.manuscript_build_figures`, never hand-edited. Its label,
  filename and generator columns come from the same table the build loop runs, so
  it cannot drift from the figures that were actually produced. Add a figure by
  adding a row to `_FIGURES` in that script; the build fails if the manuscript
  declares a `{#fig:...}` label the table does not carry, or vice versa.
- `alt_text` is the one authored field. It must say what the figure *shows* for a
  reader who cannot see it — different words from the caption, which is a title.
- `scripts/check_manuscript_tokens.py` re-checks registry coverage independently of
  the template. The template's own `validate_figure_registry` runs at
  `stage_04_validate`, not `stage_03_render`, so relying on it alone let a 29-page
  PDF ship with no registry at all.

## Paths are claims, and are checked like counts

Do not type a model family's target directory into prose. The manifest owns it,
and the producer emits the whole sentence:
`{{GNN_UNSCANNED_CORPUS_NOTE}}` for the coverage paragraph and
`{{GNN_TARGET_DIR_COVERAGE_NOTE}}` for the `--target-dir` note. Both flip with
the manifest, including between the "all inside `input/gnn_files`" and "some
outside" wordings.

This exists because a typed path is the one claim every other check was blind
to. A commit repointed the `multiagent` family's `target_dir` from
`input/multi_agent_models` into `input/gnn_files/multiagent`; three prose sites
went on describing the old layout, every count beside them stayed correct, and
`check_manuscript_tokens.py --strict` reported "clean" over the contradiction —
so the false text shipped in the PDF.

Rule 8 of that gate now cross-checks the manifest: an `input/...` literal must
exist on disk, and a literal typed next to a family name must be that family's
declared `target_dir` (or an ancestor of it, so `input/gnn_files` still reads as
covering every family). `src/tests/test_manuscript_path_claims.py` pins both the
generated sentences and the gate rule, and its two live-repository tests fail on
the pre-fix prose.


## Known benign LaTeX diagnostics

`output/pdf/_combined_manuscript.log` carries one
`Infinite glue shrinkage found in box being split` warning at each longtable page
break — count them with
`grep -c 'Infinite glue' output/pdf/_combined_manuscript.log`, and expect the
number to move when pagination does. They are TeX *informational* messages
("ignored: ..."); no content is lost — every data row of `tbl:gnn_constructs` and
`tbl:actinf_symbols` is present in the rendered PDF, and
`grep -c Overfull output/pdf/_combined_manuscript.log` is 0.

Two candidate remediations were tested against a full render on 2026-09-05 and
both are recorded here as **not** working, so they are not retried:

1. `\setlength{\LTpre}{0pt}\setlength{\LTpost}{0pt}` — no change, still 4
   occurrences at the same folios. (This was the remediation the audit
   prescribed; it is wrong: `\LTpre`/`\LTpost` default to `\bigskipamount`,
   which is finite and therefore not the infinite-shrink glue TeX is reporting.)
2. `\setlength{\@flushglue}{0pt plus 2em}` (finite ragged glue, to test whether
   the `\raggedright` minipage column headers Pandoc emits are the source) —
   made it strictly worse: 4 occurrences *and* 144 Underfull/Overfull boxes.

The glue is emitted by Pandoc's own longtable header construction, which this
project does not control from `preamble.md`; a real fix belongs in the template's
LaTeX post-processing. Do not suppress the message by dropping a table.
