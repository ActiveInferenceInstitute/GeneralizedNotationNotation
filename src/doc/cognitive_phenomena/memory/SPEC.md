# Memory Model — Specification

## Model Structure

The memory model implements working memory maintenance and retrieval as Active Inference.

## State Space

- **Memory slots**: Discrete item representations
- **Attention states**: Allocation of processing resources
- **Retrieval cues**: External prompts triggering recall

## Matrices

- `A` (observation model): Maps internal memory states to observable recall accuracy
- `B` (transition model): Memory decay, rehearsal, and interference dynamics
- `C` (preference): Accuracy-maximizing preferences
