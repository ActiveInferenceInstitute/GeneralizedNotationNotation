# GNN Grammars

This directory contains formal grammar specifications for the GNN language:

- `bnf.bnf` - Backus-Naur Form grammar
- `ebnf.ebnf` - Extended Backus-Naur Form grammar

These grammars define the syntax rules for parsing GNN model files.

## Unicode Character Support

Both grammar specifications have been updated to handle Unicode characters, especially the Greek letter π (pi) used in Active Inference policy variables. This ensures proper parsing of the reference actinf_pomdp_agent.md model.

## Comment Handling

Single hashtag comments (e.g., `# comment`) are now properly recognized in both grammars, matching the format used in the reference model.

## Active Inference Support

Special handling for Active Inference specific variables (A, B, C, D, E, G, π) has been added to ensure compatibility with standard Active Inference POMDP models. The grammars now explicitly recognize these variables and their roles in the model.
