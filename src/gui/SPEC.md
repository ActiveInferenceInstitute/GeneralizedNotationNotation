# GUI Module Specification

## Overview
Graphical user interface for GNN pipeline.

## Components

### Core
- `processor.py` - GUI processor and navigation generation
- `backend.py` - GUI backend availability probing
- `websocket_bridge.py` - WebSocket message contract for interactive sessions
- `mcp.py` - MCP tool registration

## Features
- Interactive model editing
- Visualization preview
- Headless artifact generation when no GUI backend is available

## Key Exports
```python
from gui import process_gui, get_available_guis
```


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
