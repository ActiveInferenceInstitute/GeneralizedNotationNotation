# EXTRACT_PARAMETERS

Here is the structured specification of GNN models:
```json
{
  "model": {
    "type": "gnn",
    "layers": [
      {
        "name": "layer1",
        "shape": [[], [], []]
      },
      {
        "name": "layer2",
        "shape": [[], [], []]
      }
    ],
    "initial_state": [
      {
        "type": "float",
        "value": 0.85,
        "dtype": "float64"
      },
      {
        "type": "int",
        "value": 1.0,
        "dtype": "int32"
      }
    ],
    "action_states": [
      {
        "shape": [[], []]
      },
      {
        "shape": [[], []]
      }
    ]
  },
  "initial_state_values": [
    {
      "type": "float",
      "value": 0.85,
      "dtype": "float64"
    },
    {
      "type": "int",
      "value": 1.0,
      "dtype": "int32"
    }
  ]
}
```
The structure of the model is as follows:

1. **Layer 1**:
   - `layer1`: A list containing a single layer with two dimensions (shape) and type (float). Each dimension has three types (`[]`), which are used to represent the input, action state, and observation space dimensions for each modality. The shape of each dimension is determined by the number of elements in that dimension.
   - `layer1`: A list containing a single layer with two dimensions (shape) and type (float). Each dimension has three types (`[]`), which are used to represent the input, action state, and observation space dimensions for each modality. The shape of each dimension is determined by the number of elements in that dimension.

2. **Layer 2**:
   - `layer2`: A list containing a single layer with two dimensions (shape) and type (int). Each dimension has three types (`[]`), which are used to represent the input, action state, and observation space dimensions for each modality. The shape of each dimension is determined by the number of elements in that dimension.
   - `layer2`: A list containing a single layer with two dimensions (shape) and type (int). Each dimension has three types