# LLM Processing Module - Agent Scaffolding

## Module Overview

**Purpose**: LLM-enhanced analysis, model interpretation, and AI assistance for GNN models

**Pipeline Step**: Step 13: LLM processing (13_llm.py)

**Category**: AI Enhancement / Analysis

---

## Core Functionality

### Primary Responsibilities
1. LLM-based model analysis and interpretation
2. Natural language explanations of GNN structures
3. Active Inference concept clarification
4. Model optimization suggestions
5. Automated documentation generation

### Key Capabilities
- Multi-provider LLM support (OpenAI, Anthropic, Ollama)
- Automated fallback to local Ollama if no API keys
- Context-aware prompt generation
- Structured output parsing
- Rate limiting and error handling

---

## API Reference

### Public Functions

#### `process_llm(target_dir, output_dir, logger, **kwargs) -> bool`
**Description**: Main LLM processing function with Ollama fallback

**Parameters**:
- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for LLM analyses
- `logger` (Logger): Logger instance
- `analysis_type` (str): Type of analysis ("comprehensive", "summary", "explain")
- `provider` (str): LLM provider ("auto", "openai", "anthropic", "ollama")
- `llm_tasks` (str): Specific tasks ("all", "summarize", "explain", "optimize")
- `llm_timeout` (int): Timeout for LLM API calls in seconds
- `**kwargs`: Additional options

**Returns**: `True` if processing succeeded

---

## LLM Providers

### Supported Providers
1. **OpenAI** - GPT-4, GPT-3.5-turbo
2. **Anthropic** - Claude-3, Claude-2
3. **Ollama** - Local models (llama2, mistral, etc.)

### Fallback Mechanism
1. Check for API keys in environment
2. If no keys → Check Ollama availability (`ollama list`)
3. If Ollama available → Use local model
4. If no LLM available → Skip with informative message

---

## Dependencies

### Required Dependencies
- `json` - Configuration and output
- `pathlib` - File operations

### Optional Dependencies
- `openai` - OpenAI API (fallback: skip cloud LLMs)
- `anthropic` - Anthropic API (fallback: skip cloud LLMs)
- `ollama` (subprocess) - Local LLM (fallback: skip LLM analysis)

### Internal Dependencies
- `utils.pipeline_template` - Logging utilities
- `pipeline.config` - Configuration management
- `llm.processor` - Core LLM logic

---

## Usage Examples

### Basic Usage (Auto-detect Provider)
```python
from llm import process_llm

success = process_llm(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/13_llm_output"),
    logger=logger,
    analysis_type="comprehensive"
)
```

### Specific Provider
```python
success = process_llm(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/13_llm_output"),
    logger=logger,
    provider="ollama",  # Force Ollama
    llm_tasks="all"
)
```

---

## Output Specification

### Output Products
- `{model}_llm_analysis.md` - Full analysis report
- `{model}_llm_summary.json` - Structured summary
- `{model}_llm_explanations.md` - Concept explanations
- `llm_processing_summary.json` - Processing summary

### Output Directory Structure
```
output/13_llm_output/
├── model_name_llm_analysis.md
├── model_name_llm_summary.json
├── model_name_llm_explanations.md
└── llm_processing_summary.json
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: 3.73 seconds
- **Memory**: 28.8 MB
- **Status**: SUCCESS
- **Provider Used**: Ollama (fallback)

---

## Recent Improvements

### Ollama Fallback Enhancement ✅
**Added**: Automatic Ollama availability check
```python
# Check if Ollama is available
result = subprocess.run(['ollama', 'list'], 
                       capture_output=True, timeout=5)
if result.returncode == 0:
    # Use Ollama as fallback
```

---

## Testing

### Test Files
- `src/tests/test_llm_integration.py`

### Test Coverage
- **Current**: 76%
- **Target**: 85%+

---

**Last Updated**: September 29, 2025  
**Status**: ✅ Production Ready - Ollama Fallback Working
