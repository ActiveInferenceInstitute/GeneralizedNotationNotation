# GNN Pipeline Dependency Setup Guide

## Quick Start (Recommended)
```bash
# Install with all optional dependencies for full functionality
uv pip install -e .[all]
```

## Selective Installation
Choose specific feature sets based on your needs:

### Core Pipeline (Always Required)
```bash
uv pip install -e .
```

### GUI Functionality
```bash
uv pip install -e .[gui]
# Enables: Interactive model construction, visual matrix editor
```

### Audio Processing  
```bash
uv pip install -e .[audio]
# Enables: Audio sonification, SAPF backend
```

### Machine Learning & LLM
```bash
uv pip install -e .[ml-ai,llm]
# Enables: PyTorch integration, OpenAI/Anthropic APIs
```

### Advanced Visualization
```bash
uv pip install -e .[visualization,graphs]
# Enables: Plotly, Bokeh, Graphviz visualizations
```

## Troubleshooting Common Issues

### PyMDP Installation
```bash
uv pip install pymdp inferactively-pymdp
```

### GUI Backend Issues
If step 22 (GUI) fails:
```bash
uv pip install gradio>=4.0.0 plotly>=6.2.0
```

### Audio Processing Issues
```bash
uv pip install librosa>=0.10.1 soundfile>=0.12.1 pedalboard>=0.6.0
```

## Environment Variables
```bash
export GNN_DEBUG=1          # Enable debug logging
export GNN_HEADLESS=1       # Force headless mode for GUIs
export GNN_AUDIO_BACKEND=sapf  # Specify audio backend
```

