# GNN Schema Definitions

Schema files defining the structural contracts for valid GNN models across multiple serialization formats.

## Included Schemas

| File | Format | Purpose |
|------|--------|---------|
| `json.json` | JSON Schema | Primary validation schema |
| `yaml.yaml` | YAML | YAML format validation |
| `xsd.xsd` | XML Schema | XML format validation |
| `proto.proto` | Protocol Buffers | Binary format definition |
| `asn1.asn1` | ASN.1 | Formal notation schema |
| `pkl.pkl` | Pickle | Serialization reference |

## Usage

Schemas are loaded by `gnn/schema_validator.py` during Steps 5–6 (type checking and validation).

## See Also

- [Parent: gnn/README.md](../README.md)
- [AGENTS.md](AGENTS.md) — Architecture documentation
