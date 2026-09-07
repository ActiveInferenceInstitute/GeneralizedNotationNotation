# Recursive Model Fixtures

This directory is a reserved target for bounded `--autonomous` proposal-loop
runs. It holds no committed models, and no committed command points at it:
`TO-DO.md`'s `--autonomous` verification command runs against
`--target-dir input/gnn_files`.

`pipeline.autonomous.run_autonomous_proposal_loop` writes every candidate,
patch and report under `<output-dir>/autonomous/` and makes no write to the
target directory at all, so pointing `--target-dir` here leaves it unchanged.
