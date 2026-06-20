# Introduction {#sec:introduction}

## Motivation

Active Inference has matured into a broad research program spanning the free energy principle [@friston2010], discrete state-space formulations of perception and action [@dacosta2020], and a growing body of pedagogy and reference implementations [@parr2022;@smith2022]. As the field has grown, so has a quieter problem: the models themselves are difficult to share, reproduce, and compare. A generative model that lives only as a tangle of matrix definitions inside one author's script, a diagram in a slide deck, and a paragraph of prose in a paper has no single authoritative form. The same model is described three times, in three incompatible media, with no guarantee that they agree. When a reader wants to re-run that model in a different toolkit, they must reverse-engineer it from whichever fragment they happen to have.

This is a reproducibility and interoperability crisis specific to the structure of Active Inference work. The discipline depends on precisely specified state spaces, observation and transition tensors, prior preferences, and policy structures; small notational ambiguities propagate into materially different behaviour. Yet the community has lacked a notation that is simultaneously human-readable, machine-parseable, and faithful to the underlying mathematics. Generalized Notation Notation (GNN) was introduced to fill exactly that gap [@gnn2023]: a standard, text-based way to write down an Active Inference generative model once, such that every downstream use derives from the same source of truth.

## The GNN Approach

GNN treats the model specification as a first-class artifact. A model is written in a plain-text language whose syntax captures the components an Active Inference modeller actually reasons about — state factors, observation modalities, the matrices that link them, control structure, and the temporal organization of the model. Because the specification is text, it lives comfortably in version control, diffs cleanly across revisions, and can be authored and reviewed by humans without specialized tooling.

The defining commitment of GNN is what the project calls the Triple Play: a single text specification is the common origin for three coordinated renderings of the same model. The text form is the authoritative, editable description. From it, GNN produces graphical visualizations that expose the model's factor and dependency structure for inspection and communication. And from the same source it produces executable cognitive models that can actually be run, so that the diagram, the equations, and the running code are provably the same model rather than three artifacts that merely claim to be. The Triple Play turns a model from a description scattered across media into one specification with multiple faithful projections (see @fig:triple_play).

## Contributions

This work contributes a standard notation together with the infrastructure that makes it usable in practice:

- **A parseable, human-readable notation** for Active Inference generative models, whose text form is authoritative and from which every other representation is derived.
- **A {{GNN_STEP_COUNT}}-step processing pipeline**, implemented across {{GNN_SRC_PACKAGE_COUNT}} source packages ({{GNN_STEP_RANGE}}), that carries a specification from parsing and validation through visualization, rendering, and execution.
- **{{GNN_BACKEND_COUNT}} rendering backends** ({{GNN_BACKEND_LIST}}) that materialize a single GNN specification as executable cognitive models across multiple simulation frameworks, realizing the executable arm of the Triple Play.
- **{{GNN_MCP_TOOL_COUNT}} Model Context Protocol tools** that expose the pipeline's capabilities to agentic and programmatic clients, so the notation and its tooling are directly accessible to automated workflows.
- **{{GNN_FAMILY_COUNT}}-family reliability gates** that exercise the pipeline against a curated set of model families ({{GNN_FAMILY_LIST}}), turning interoperability claims into checks that must pass rather than assertions that are merely made.

## Reader Orientation

The remainder of this manuscript is organized to move from language to mechanism to evidence. The notation, its semantics, and the realized pipeline — including its model families and rendering backends — are developed in the Methods section (@sec:methods), which describes how a GNN specification is structured and how the pipeline transforms it. There the pipeline structure (@fig:pipeline), the family-by-framework coverage (@fig:family_matrix), the backend capabilities (@fig:backend_matrix), and repository-scale metrics (@fig:repo_metrics) are presented. Claims of independent re-execution are addressed in the Reproducibility section (@sec:reproducibility), and full source details are collected in the References (@sec:references). Read in this order, the manuscript moves from why a standard notation is needed, through how GNN realizes it, to the evidence that the standard holds.
