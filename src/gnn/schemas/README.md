# GNN Schemas

This directory contains schema definitions for GNN models in various formats:

- `json.json` - JSON Schema definition with Unicode character support
- `yaml.yaml` - YAML Schema definition with single hashtag comments
- `proto.proto` - Protocol Buffers schema
- `xsd.xsd` - XML Schema definition
- `pkl.pkl` - Apple Pkl schema
- `asn1.asn1` - ASN.1 schema definition

These schemas provide validation and structure for GNN model specifications.

## Unicode Character Support

All schemas have been updated to handle Unicode characters, especially the Greek letter π (pi) used in Active Inference policy variables. This ensures proper support for the reference actinf_pomdp_agent.md model.

## Comment Handling

Single hashtag comments (e.g., `# comment`) are properly supported in all schemas, matching the format used in the reference model.

## Active Inference Support

Special handling for Active Inference specific variables (A, B, C, D, E, G, π) has been added to all schemas to ensure compatibility with standard Active Inference POMDP models.
