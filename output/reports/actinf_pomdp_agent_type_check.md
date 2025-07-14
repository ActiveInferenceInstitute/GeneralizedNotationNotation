# Type Check Report: actinf_pomdp_agent
Status: ✅ VALID
File: ../input/gnn_files/actinf_pomdp_agent.md

## Model Overview
- **Model Type**: Dynamic
- **Variables**: 12
- **Connections**: 8
- **Overall Complexity**: 8.00

## Section Presence:
### Required Sections:
- GNNSection: ✅
- GNNVersionAndFlags: ✅
- ModelName: ✅
- StateSpaceBlock: ✅
- Connections: ✅
- Footer: ✅
- Signature: ✅

### Optional Sections:
- ModelAnnotation: ✅
- InitialParameterization: ✅
- Equations: ✅
- Time: ✅
- ActInfOntologyAnnotation: ✅
- ModelParameters: ✅

## Variable Analysis
- **Total Variables**: 12
- **Type Distribution**:
  - float: 10
  - int: 2
- **Dimension Analysis**:
  - Scalars: 0
  - Vectors: 7
  - Matrices: 4
  - Tensors: 1
  - Max Dimensions: 3

### Variables Table:
| Name | Type | Dimensions | Elements |
|---|---|---|---|
| A | float | [3, 3] | 9 |
| B | float | [3, 3, 3] | 27 |
| C | float | [3] | 3 |
| D | float | [3] | 3 |
| E | float | [3] | 3 |
| s | float | [3, 1] | 3 |
| s_prime | float | [3, 1] | 3 |
| o | float | [3, 1] | 3 |
| π | float | [3] | 3 |
| u | int | [1] | 1 |
| G | float | [1] | 1 |
| t | int | [1] | 1 |

## Connection Analysis
- **Total Connections**: 8
- **Connection Types**:
  - Directed: 8
  - Undirected: 0
  - Temporal: 0

### Connections:
| Source | Target | Type | Temporal |
|---|---|---|---|
| D | s | directed | No |
| s | A | directed | No |
| A | o | directed | No |
| B | s_prime | directed | No |
| C | G | directed | No |
| E | π | directed | No |
| G | π | directed | No |
| π | u | directed | No |

## Complexity Analysis
- **Variable Complexity**: 12
- **Connection Complexity**: 8
- **Equation Complexity**: 4
- **Overall Complexity Score**: 8.00
