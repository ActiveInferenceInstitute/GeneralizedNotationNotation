# Step 21: MCP — Model Context Protocol Server

## Overview

Orchestrates Model Context Protocol processing, enabling GNN pipeline integration with MCP-compatible AI tools and IDE extensions.

## Usage

```bash
python src/21_mcp.py --target-dir input/gnn_files --output-dir output --verbose
```

## Architecture

| Component | Path |
|-----------|------|
| Orchestrator | `src/21_mcp.py` (63 lines) |
| Module | `src/mcp/` |
| Processor | `src/mcp/processor.py` |
| Module function | `process_mcp()` |

## Output

- **Directory**: `output/21_mcp_output/`
- MCP server configuration, resource definitions, and tool registrations

## Source

- **Script**: [src/21_mcp.py](file:///Users/4d/Documents/GitHub/generalizednotationnotation/src/21_mcp.py)
