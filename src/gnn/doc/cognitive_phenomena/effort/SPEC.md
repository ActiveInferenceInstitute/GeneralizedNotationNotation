# Effort Model — Specification

## Model Structure

The effort model implements expected free energy minimization for resource allocation decisions.

## State Space

- **Effort levels**: Discrete set of metabolic expenditure states
- **Task demands**: Observable task difficulty signals
- **Reward signals**: Outcome-contingent feedback

## Matrices

- `A` (observation model): Maps effort→performance observations
- `B` (transition model): Effort dynamics with fatigue and recovery
- `C` (preference): Reward-maximizing preferences over outcomes
