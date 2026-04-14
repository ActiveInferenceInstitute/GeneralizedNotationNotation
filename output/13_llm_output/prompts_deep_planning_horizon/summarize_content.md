# SUMMARIZE_CONTENT

Here's a concise summary of the GNN specification:

```json
{
  "summary": [
    {
      "name": "GNN",
      "type": "Active Inference",
      "model_annotation": "Deep Planning Horizon POMDP",
      "description": "A model that uses rolling out Expected Free Energy to evaluate a sequence of actions and policies.",
      "inputs": [
        {
          "name": "actions",
          "value": 0.9,
          "type": "float"
        },
        {
          "name": "policies",
          "value": 0.1,
          "type": "float"
        }
      ],
      "outputs": [
        {
          "name": "G_tau1",
          "value": -E_Q[G_tau1],
          "type": "float"
        },
        {
          "name": "G_tau2",
          "value": E_Q[G_tau2] + G_tau3,
          "type": "float"
        }
      ],
      "outputs": [
        {
          "name": "B",
          "value": 0.95 * G_tau1 + 0.86 * G_tau2,
          "type": "float"
        },
        {
          "name": "C",
          "value": E_Q[G_tau1] - G_tau3,
          "type": "float"
        }
      ],
      "outputs": [
        {
          "name": "D",
          "value": 0.95 * G_tau2 + 0.86 * G_tau1,
          "type": "float"
        },
        {
          "name": "E",
          "value": E_Q[G_tau1] - G_tau3,
          "type": "float"
        }
      ],
      "outputs": [
        {
          "name": "π",
          "value": 0.95 * G_tau2 + 0.86 * G_tau1,
          "type": "float"
        },
        {
          "name": "G_tau1",
          "value": E_Q[G_tau1] - G_tau3,
          "type": "float"
        }
      ],
      "outputs": [
        {
          "name": "π",
          "value": 0.95