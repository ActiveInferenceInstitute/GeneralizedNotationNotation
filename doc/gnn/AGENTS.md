# GNN Documentation Hub - Agent Scaffolding

## Overview

**Purpose**: This directory serves as the centralized documentation hub and specification authority for the Generalized Notation Notation (GNN) framework.

**Pipeline Step**: N/A (Static Reference Material)

**Category**: Documentation / Specification

**Status**: ✅ Production Ready

**Documentation hub version**: v2.0.0 (aligns with [README.md](README.md) front matter; **GNN language** syntax is v1.1 per [gnn_syntax.md](gnn_syntax.md); **Python package** version is `1.5.0` in [pyproject.toml](../../pyproject.toml)).

---

## Core Functionality

### Primary Responsibilities

1. **Syntax Authority**: Maintains the definitive `.md` parser specifications required to author Active Inference matrices correctly.
2. **Architecture Mapping**: Defines the structural rules connecting `language/`, `reference/`, `implementations/`, and `integration/`.
3. **End-User Guidance**: Houses tutorials, external tool integrations, and operational guides.

### Subsystem Indices Available

- `doc/gnn/advanced/`: Ontology, multi-agent, LLM/neurosymbolic topics, and advanced modeling patterns.
- `doc/gnn/implementations/`: Specific platform references (PyMDP, RxInfer, etc.).
- `doc/gnn/language/`: DSL specific syntax rules.
- `doc/gnn/modules/`: Component behaviors (including structural documentation for all 25 modules).
- `doc/gnn/operations/`: Internal pipeline processing guides, troubleshooting, and coherence checks.
- `doc/gnn/reference/`: Raw variable mappings and type systems.
- `doc/gnn/testing/`: Testing standards for the notation itself.
- `doc/gnn/tutorials/`: End-user construction guides.

## Implementation Details

These documents are consumed by human end-users and parsed globally by the pipeline's LLM context protocol (via Model Context Protocol) to ensure generative models created via text prompts are hallucination-free and syntactically correct against GNN v1.1.
