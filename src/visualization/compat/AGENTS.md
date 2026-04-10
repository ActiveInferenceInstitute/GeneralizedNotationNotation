# Compat - Agent Scaffolding

## Module Overview

**Purpose**: Responsible for `Compat` operations within the GNN pipeline architecture.
**Category**: Generated Pipeline Component
**Status**: Development

---

## Core Functionality

### Primary Responsibilities
Shared matplotlib/numpy/seaborn imports for visualization and analysis.

Both visualization (step 8) and analysis (step 16) import from the package-root
`visualization._viz_compat` shim, which re-exports this module.

### Extracted Code Entities

- **Classes**: No specific classes exported.
- **Functions**: No specific public functions exported.

## Implementation Details

This module follows the Thin Orchestrator Pattern. It is governed by the Zero-Mock testing policy.
