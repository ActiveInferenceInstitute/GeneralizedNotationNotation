# Fixture: structurally valid GNN file with an undeclared connection target.

# Used by tests/api/test_api_parity.py to exercise the CLI-parity exit-code
# mapping: all required sections are present, but `A>Z` references an
# undeclared variable, so both `gnn parse` and `gnn validate` exit 2
# (EXIT_WARNING) and the API returns 200 with data.exit_code == 2.

## GNNSection

Test

## GNNVersionAndFlags

GNN v1

## ModelName

Fixture Unknown Connection Target

## StateSpaceBlock

A[2,2,type=float]

## Connections

A>Z
