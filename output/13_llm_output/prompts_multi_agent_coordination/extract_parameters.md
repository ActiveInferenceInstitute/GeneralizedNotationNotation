# EXTRACT_PARAMETERS

Here is the structured specification of GNN:
```python
# GNNVersionAndFlags
GNN v1

## ModelName
Multi-Agent Cooperative Active Inference

## ModelAnnotation

A1={
  (0.85, 0.05, 0.05, 0.05),
  (0.05, 0.85, 0.05, 0.05),
  (0.05, 0.05, 0.85, 0.05)
}
A2={
  (0.85, 0.05, 0.05, 0.05),
  (0.05, 0.85, 0.05, 0.05),
  (0.05, 0.05, 0.85, 0.05)
}
B1={
  (-1.0, -1.0, -1.0, 2.0)}
B2={
  ((-1.0, -1.0, -1.0, 2.0), ((-1.0, -1.0, -1.0, 2.0)), (-1.0, -1.0, -1.0, 2.0))
}
C1={
  (0.95, 0.85, 0.05, 0.05),
  (0.05, 0.95, 0.05, 0.05)
}
D1={(0.25, 0.25, 0.25, 0.25)}
D2={
  (()=()){} {} 

B2={
  (()=()){} {} 

A3={
  (()=()){} {} 

C4={
  (()=()){} {} 

G1={
  (()=()){} {} 

# Shared environment state: agent 1's last action and agent 2's next actions

s_joint = [[0.95, 0.85], [0.05]]
o_joint = [[0.95, 0.85], [0.05]]
