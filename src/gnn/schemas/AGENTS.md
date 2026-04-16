# GNN Schema Definitions

## Overview

Contains schema definitions for GNN model validation in multiple serialization formats. These schemas define the structural contracts that valid GNN files must conform to.

## Architecture

```
schemas/
├── __init__.py     # Package marker
├── json.json       # JSON Schema for GNN model validation
├── yaml.yaml       # YAML schema definition
├── xsd.xsd         # XML Schema Definition for GNN XML format
├── proto.proto      # Protocol Buffers schema
├── asn1.asn1        # ASN.1 schema definition
└── pkl.pkl          # Pickle schema reference
```

## Purpose

- **Validation anchors** — Used by `gnn/schema_validator.py` to validate parsed models.
- **Format contracts** — Define the expected structure for each serialization format.
- **Cross-format consistency** — Ensure semantic equivalence across JSON, YAML, XML, and binary representations.

## Usage

Schemas are loaded by the validation pipeline (Steps 5–6) and referenced during export (Step 7) to ensure output conformity.

## Parent Module

See [gnn/AGENTS.md](../AGENTS.md) for the overall GNN processing architecture.

**Version**: 1.6.0
