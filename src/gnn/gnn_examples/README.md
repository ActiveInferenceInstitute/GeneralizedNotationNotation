# GNN Examples

Reference GNN model files demonstrating proper notation, structure, and Active Inference parameterization.

## Files

### `actinf_pomdp_agent.md`

A complete Active Inference POMDP (Partially Observable Markov Decision Process) agent specification:

- **Model**: `ActInf_POMDP_Agent`
- **Hidden States**: 4 states (`s[0]` through `s[3]`)
- **Observations**: 5 observations (`o[0]` through `o[4]`)
- **Control States**: 3 actions (`u[0]` through `u[2]`)
- **Matrices**: Full A (likelihood), B (transition), C (preference), D (prior) parameterization
- **Equations**: Variational free energy minimization, posterior updates, expected free energy computation
- **Time**: Dynamic discrete with configurable horizon

## Usage

Use these files as:

1. **Reference** for writing new GNN specifications
2. **Test fixtures** for parser and serializer validation
3. **Documentation** of Active Inference model structure in GNN format

## Integration

These examples are the reference inputs for the round-trip testing system (`gnn/testing/test_round_trip.py`), which validates semantic preservation across all 23 supported formats.
