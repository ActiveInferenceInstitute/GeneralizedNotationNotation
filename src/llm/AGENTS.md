# LLM Processing Module - Agent Scaffolding

## Module Overview

**Purpose**: LLM-enhanced analysis, model interpretation, and AI assistance for GNN models

**Pipeline Step**: Step 13: LLM processing (13_llm.py)

**Category**: AI Enhancement / Analysis

**Status**: Production Ready

**Last Updated**: 2026-09-02


---

## Core Functionality

### Primary Responsibilities
1. LLM-based model analysis and interpretation
2. Natural language explanations of GNN structures
3. Active Inference concept clarification
4. Model optimization suggestions
5. Automated documentation generation

### Key Capabilities
- Multi-provider LLM support (Ollama, OpenAI, OpenRouter, Perplexity; Anthropic keys appear in the provider matrix when present)
- Automated preference for local Ollama when no cloud API keys are set (`LLMProcessor`)
- Context-aware prompt generation
- Structured output parsing
- Rate limiting and error handling

---

## API Reference

### Public Functions

#### `process_llm(target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs) -> bool`
**Description**: Main LLM processing function with automatic Ollama recovery. Processes GNN files using LLM analysis with multi-provider support.

**Parameters**:
- `target_dir` (Path): Directory containing GNN files to analyze
- `output_dir` (Path): Output directory for LLM analyses
- `verbose` (bool): Enable verbose logging (default: False)
- `llm_timeout` (int, optional): Timeout budget for the step in seconds (default: 600; also settable via `input/config.yaml` `llm.timeout_seconds`)
- `max_files` (int, optional): Cap on the number of GNN files processed per run (also settable via `llm.max_files`)
- `custom_prompts` (list, optional): Custom prompt payloads; each becomes a per-file output
- `max_prompt_timeout` (int, optional): Per-prompt timeout in seconds (default: 45; also `llm.prompt_timeout`)
- `**kwargs`: Additional LLM processing options

Provider and model selection are driven by environment variables and `input/config.yaml` (see Configuration), not by `process_llm` kwargs.

**Returns**: `bool` - True if processing succeeded, False otherwise

**Example**:
```python
from llm import process_llm
from pathlib import Path
import logging

success = process_llm(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/13_llm_output"),
    verbose=True,
)
```

#### `analyze_gnn_file_with_llm(file_path: Path, verbose: bool = False, ollama_model: Optional[str] = None) -> Dict[str, Any] | Coroutine`
**Description**: Analyze a GNN file (heuristic extractors + optional LLM summary). When `ollama_model` is set (e.g. from `process_llm` after `_select_best_ollama_model`), summarization uses that tag on Ollama via `LLMOperations.summarize_gnn`.

**Parameters**:
- `file_path` (Path): Path to the GNN `.md` file
- `verbose` (bool): Verbose logging
- `ollama_model` (str, optional): Resolved Ollama tag for the per-file summary; if omitted, summarization follows `LLMProcessor` defaults

**Returns**: Result dict, or a coroutine if called while an event loop is already running

#### `extract_variables(content: str) -> List[Dict[str, Any]]`
**Description**: Extract variable definitions from GNN content.

**Parameters**:
- `content` (str): GNN content string

**Returns**: `List[Dict[str, Any]]` - List of variable dictionaries with name, type, dimensions

#### `extract_connections(content: str) -> List[Dict[str, Any]]`
**Description**: Extract connection definitions from GNN content.

**Parameters**:
- `content` (str): GNN content string

**Returns**: `List[Dict[str, Any]]` - List of connection dictionaries with source, target, type

#### `generate_model_insights(gnn_content: str, analysis_results: Dict[str, Any] = None) -> Dict[str, Any]`
**Description**: Generate insights from GNN model analysis.

**Parameters**:
- `gnn_content` (str): GNN content string
- `analysis_results` (Dict[str, Any], optional): Previous analysis results

**Returns**: `Dict[str, Any]` - Insights dictionary with complexity, patterns, recommendations

#### `generate_documentation(gnn_content: str, model_name: str = None) -> str`
**Description**: Generate comprehensive documentation for GNN model using LLM.

**Parameters**:
- `gnn_content` (str): GNN content string
- `model_name` (str, optional): Name of the model

**Returns**: `str` - Generated documentation as markdown string

---

## LLM Providers

### Supported Providers (`LLMProcessor` / `ProviderType`)
1. **Ollama** — local inference via the `ollama` Python client when functional, else CLI recovery (`ollama chat` JSON mode when supported, else `ollama run`). Default model tag `smollm2:135m-instruct-q4_K_S` (`llm.defaults.DEFAULT_OLLAMA_MODEL`; overridable via `OLLAMA_MODEL`, `OLLAMA_TEST_MODEL`, or `input/config.yaml` `llm.model`).
2. **OpenAI** — cloud API when `OPENAI_API_KEY` is set.
3. **OpenRouter** — when `OPENROUTER_API_KEY` is set.
4. **Perplexity** — when `PERPLEXITY_API_KEY` is set.

### Ollama (local) — implementation surface

| Location | Role |
|----------|------|
| `llm/providers/ollama_provider.py` | `OllamaProvider`: `initialize`, `validate_config`, `generate_response`, `generate_stream`, `analyze`, `close` |
| `llm/llm_processor.py` | `LLMProcessor` merges `get_default_provider_configs()` into `provider_configs` so env vars apply even when the caller passes `None`/`{}` |
| `llm/processor.py` | `_start_ollama_if_needed`, `_select_best_ollama_model`, `_model_is_cached`; step-13 orchestration and `provider_matrix` |

**Model selection** (`_select_best_ollama_model`): `OLLAMA_MODEL` or `OLLAMA_TEST_MODEL` → optional `input/config.yaml` `llm.model` if that name matches an installed tag → built-in preference list (smaller models first: `smollm2`, `tinyllama`, `gemma3:4b`, `gemma2:2b`, …) → first `ollama list` entry → `llm.defaults.DEFAULT_OLLAMA_MODEL`.

**Request wiring**: The tag chosen above is passed to `LLMProcessor.get_response` as `model_name` for every structured PromptType prompt and for custom prompts (same value as cache keys). `AnalysisType.SUMMARY` tasks prefer **Ollama first** when registered, then OpenAI / OpenRouter / Perplexity, so local runs are not blocked by exhausted cloud quota when a key is still present. For per-file summaries, `process_llm` passes the resolved tag into `analyze_gnn_file_with_llm` so it matches the prompt loop. Override defaults with `OLLAMA_MODEL` or `input/config.yaml` `llm.model`. To avoid OpenAI retries when quota is zero, unset `OPENAI_API_KEY` for local-only runs.

### Recovery Mechanism
1. `LLMProcessor` loads API keys from the environment; Ollama is enabled unless `OLLAMA_DISABLED` is truthy (`1`, `true`).
2. Step 13 (`process_llm`) probes the Ollama CLI (`ollama list`) and records status in `provider_matrix.ollama`.
3. If the unified processor initializes, prompts use the selected local model; on failure, structured fallbacks and cache still write outputs.
4. If no provider works, processing continues with recovery text and logged warnings.

---

## Configuration

### Configuration Options

Runtime knobs come from `process_llm` kwargs and `input/config.yaml` (`llm:` section); provider/model selection comes from environment variables.

#### `process_llm` kwargs
- `llm_timeout` (int): Step timeout budget in seconds (default: `600`; `llm.timeout_seconds`)
- `max_files` (int): Cap on GNN files processed per run (`llm.max_files`)
- `custom_prompts` (list): Extra prompts run per file (in addition to the structured prompt set)
- `max_prompt_timeout` (int): Per-prompt timeout in seconds (default: `45`; `llm.prompt_timeout`)

#### Environment Variables (Ollama)
- `OLLAMA_MODEL`: Model tag for requests (default `llm.defaults.DEFAULT_OLLAMA_MODEL`)
- `OLLAMA_TEST_MODEL`: Overrides model name in tests when set
- `OLLAMA_MAX_TOKENS`: Default `num_predict` cap (default `256` in processor config)
- `OLLAMA_TIMEOUT`: Client/CLI subprocess timeout in seconds (default `60` in env wiring; provider default 30s unless configured)
- `OLLAMA_HOST`: Optional base URL for the Python `ollama` client (empty = client default)
- `OLLAMA_DISABLED`: Set to `1` or `true` to skip registering Ollama as a provider
- `OLLAMA_AUTO_START`: Set to `1` to allow starting the Ollama daemon when installed but not running
- `OLLAMA_AUTO_PULL`: Set to `1` to allow pulling the default model when no model is installed

#### Environment Variables (cloud)
- `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `PERPLEXITY_API_KEY`, `ANTHROPIC_API_KEY` (Anthropic appears in summaries/matrix when present)
- `DEFAULT_PROVIDER`: e.g. `ollama` (see `llm_processor.get_preferred_providers_from_env`)

---

## Dependencies

### Required Dependencies
- `json` - Configuration and output
- `pathlib` - File operations

### Core Dependencies (installed by `uv sync`; no dedicated pip extra for this module)
- `openai` — OpenAI/OpenRouter API access
- `ollama` (PyPI) — Python client; if import fails or `chat` is missing, `OllamaProvider` uses the `ollama` CLI when on `PATH`

Local inference additionally requires the Ollama runtime (CLI/daemon from https://ollama.com), which is not a pip package.

There is no Anthropic provider module; `ANTHROPIC_API_KEY` only influences the provider matrix and error attribution.

### Internal Dependencies
- `utils.pipeline_template` - Logging utilities
- `pipeline.config` - Configuration management
- `llm.processor` - Core LLM logic

---

## Usage Examples

### Basic Usage
```python
from llm import process_llm

success = process_llm(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/13_llm_output"),
    verbose=True,
)
```

### Local-Only Run (force Ollama)
```bash
# Provider preference is environment-driven; unset cloud keys and point at Ollama
unset OPENAI_API_KEY OPENROUTER_API_KEY PERPLEXITY_API_KEY
export OLLAMA_MODEL=smollm2:135m-instruct-q4_K_S
python src/13_llm.py --target-dir input/gnn_files --output-dir output --verbose
```

---

## Output Specification

### Output Products
- `llm_results.json` - Processing results
- `llm_summary.md` - Human-readable processing summary

### Output Directory Structure
```
output/13_llm_output/
├── llm_results.json
└── llm_summary.md
```

---

## Performance Characteristics

### Latest Execution
See `output/13_llm_output/` and the pipeline summary for the current run's duration,
memory, and provider status; this document does not track them.

---

## Testing

### Test Files
- `src/tests/llm/test_llm_overall.py` - Module-level tests
- `src/tests/llm/test_llm_functional.py` - Functional tests
- `src/tests/llm/test_llm_ollama.py` - Ollama-specific tests
- `src/tests/llm/test_llm_ollama_integration.py` - Ollama integration tests

### Test Coverage
Measure on demand:

```bash
uv run --extra dev python -m pytest src/tests/test_llm*.py \
    --cov=src/llm --cov-report=term-missing
```
### Key Test Scenarios
1. Ollama detection and availability check
2. Model selection and prioritization
3. LLM processing with Ollama integration
4. Recovery mode when Ollama unavailable
5. Error handling and recovery
6. Timeout management for LLM calls

---

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Ollama Not Found
**Symptom**: "Ollama not found in PATH" message

**Solution**:
```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Or download from https://ollama.com
```

**Verification**:
```bash
ollama --version
which ollama
```

#### 2. Ollama Service Not Running
**Symptom**: "Ollama is installed but may not be running"

**Solution**:
```bash
# Start Ollama service
ollama serve

# In a separate terminal, verify it's running
ollama list
```

**Alternative**: Run Ollama in background
```bash
# macOS/Linux
nohup ollama serve > /dev/null 2>&1 &

# Or use system service (if configured)
systemctl start ollama
```

#### 3. No Models Installed
**Symptom**: "Ollama is running but no models are installed"

**Solution**:
```bash
# Default pipeline tag (see llm.defaults.DEFAULT_OLLAMA_MODEL)
ollama pull smollm2:135m-instruct-q4_K_S

# Alternates
ollama pull gemma3:4b
ollama pull tinyllama
ollama pull llama2:7b
```

**Suggested models** (latency and RAM depend on CPU/GPU — measure locally):
- **Default / small instruct**: `smollm2:135m-instruct-q4_K_S`
- **Balanced**: `tinyllama`, `gemma3:4b`
- **Larger**: `llama2:7b` and up

**View Installed Models**:
```bash
ollama list
```

#### 4. LLM Timeout Issues
**Symptom**: "Prompt execution timed out" or slow responses

**Solution**:
- **Automatic**: Module uses adaptive timeouts based on prompt complexity
- Environment variable override:
  ```bash
  export OLLAMA_TIMEOUT=120  # Increase timeout to 120 seconds
  ```

- Use a smaller/faster tag:
  ```bash
  export OLLAMA_MODEL=smollm2:135m-instruct-q4_K_S
  ```

**Performance Tips**:
- Use GPU acceleration if available (Ollama detects automatically)
- Close other applications to free memory
- Use smaller models for routine analysis

#### 5. Model Selection Issues
**Symptom**: Wrong model being used or "model not found" errors

**Solution**:
```bash
# Override model selection via environment variable
export OLLAMA_MODEL=tinyllama

# Or specify in command
OLLAMA_MODEL=tinyllama python src/13_llm.py --target-dir input/gnn_files
```

**Automatic Selection**:
`_select_best_ollama_model` picks from installed tags using the ordered preference list in `llm/processor.py` (starts with `smollm2`, `tinyllama`, `gemma3:4b`, `gemma2:2b`, …)

**Check Which Model Was Used**:
```bash
# View LLM results
cat output/13_llm_output/llm_results/llm_results.json | grep "selected_model"
```

#### 6. Recovery Mode Warnings
**Symptom**: "Proceeding with recovery LLM analysis" messages

**Explanation**: This is expected when Ollama is not available. The module provides basic analysis without live LLM interaction.

**Solution** (if you want LLM features):
1. Install and start Ollama (see issues #1 and #2)
2. Install at least one model (see issue #3)
3. Re-run the LLM step

**Recovery Capabilities**:
- Basic pattern extraction
- Variable and connection identification
- Structure analysis
- No natural language generation
- No model interpretation

#### 7. Slow LLM Processing
**Symptom**: Step 13 takes several minutes (3m+ per model)

**Causes**:
- Large models (llama2:70b, etc.)
- CPU-only inference (no GPU)
- Complex/long prompts
- Multiple GNN files being processed

**Solutions**:
1. **Use a smaller tag** (see `llm.defaults.DEFAULT_OLLAMA_MODEL`):
   ```bash
   export OLLAMA_MODEL=smollm2:135m-instruct-q4_K_S
   ```

2. **Reduce prompt complexity**:
   ```bash
   export OLLAMA_MAX_TOKENS=256  # Shorter responses
   ```

3. **GPU acceleration** (if available):
   ```bash
   # Ollama uses the GPU automatically when detected
   ollama run llama2
   ```

4. **Limit files per run**:
   ```bash
   # Cap the number of files processed
   python src/13_llm.py --target-dir input/gnn_files --max-files 5
   ```

**Performance**: Measure with your hardware; smaller instruct models are usually faster on CPU.

#### 8. Memory Issues
**Symptom**: System slowdown or "out of memory" errors

**Solution**:
1. **Use smaller models** (e.g. default `smollm2:135m-instruct-q4_K_S`).

2. **Limit concurrent processing**:
   - Process files one at a time
   - Close other applications

3. **Monitor resource usage**:
   ```bash
   # Monitor Ollama memory usage
   ps aux | grep ollama
   htop  # or top
   ```

**Memory**: Check `ollama show <tag>` and system monitor while loading models.

### Ollama Integration Features

#### Enhanced Detection (October 2025)
- Automatic Ollama availability check
- Model listing and validation
- Service health monitoring (port 11434)
- Helpful installation instructions when not found

#### Intelligent Model Selection
- Prioritizes small, fast models for quick execution
- Automatic recovery chain
- Environment variable override support
- Logs selected model for transparency

#### Progress Tracking
- File-by-file progress indicators
- Prompt-by-prompt completion tracking
- Detailed progress logging
- Clear success/failure indicators in logs

#### Error Recovery
- Graceful recovery when Ollama unavailable
- Per-prompt error handling
- Timeout protection with retry logic
- Comprehensive error messages

### Best Practices

1. **Install and Start Ollama Before Running**:
   ```bash
   # Terminal 1: Start Ollama
   ollama serve
   
   # Terminal 2: Run pipeline
   python src/main.py --only-steps "13" --verbose
   ```

2. **Use Appropriate Model for Task**:
   - **Quick / default**: `smollm2:135m-instruct-q4_K_S`
   - **Balanced**: `tinyllama`, `gemma3:4b`
   - **Deep Analysis**: `llama2:7b` and larger

3. **Monitor Performance**:
   ```bash
   # Run with verbose logging
   python src/13_llm.py --verbose --target-dir input/gnn_files
   
   # Check timing in results
   cat output/13_llm_output/llm_results/llm_results.json
   ```

4. **Optimize for Speed**:
   ```bash
   export OLLAMA_MODEL=smollm2:135m-instruct-q4_K_S
   export OLLAMA_MAX_TOKENS=256
   export OLLAMA_TIMEOUT=30
   ```

5. **Check Results Quality**:
   ```bash
   # View generated analyses
   cat output/13_llm_output/llm_results/prompts_*/technical_description.md
   cat output/13_llm_output/llm_results/llm_summary.md
   ```

### Advanced Configuration

#### Environment Variables
```bash
# Model selection
export OLLAMA_MODEL=tinyllama           # Override automatic selection
export OLLAMA_TEST_MODEL=smollm2:135m-instruct-q4_K_S   # Test/CI model

# Performance tuning
export OLLAMA_MAX_TOKENS=512            # Maximum response length
export OLLAMA_TIMEOUT=60                # Request timeout (seconds)
export OLLAMA_HOST=http://localhost:11434  # Ollama server URL

# Behavior
export OLLAMA_DISABLED=1                # Skip Ollama provider registration
export DEFAULT_PROVIDER=ollama          # Prefer Ollama when keys allow
```

#### Custom Model Configuration
```python
# In your code or config
from llm.llm_processor import get_default_provider_configs

configs = get_default_provider_configs()
configs["ollama"]["default_model"] = "my-custom-model"
configs["ollama"]["default_max_tokens"] = 1024
```

---

## Error Handling

### Graceful Degradation
- **No API Keys**: Automatic recovery to Ollama if available
- **Ollama Unavailable**: Skip LLM analysis, log informative message, continue pipeline
- **LLM Timeout**: Retry with shorter timeout, then skip if still fails
- **Invalid Response**: Parse what's possible, log warning

### Error Categories
1. **Provider Unavailable**: No API keys and Ollama not available (recovery: skip analysis)
2. **API Errors**: Rate limits, network errors (recovery: retry with backoff)
3. **Timeout Errors**: LLM response too slow (recovery: use faster model or skip)
4. **Parsing Errors**: Invalid LLM response format (recovery: use raw response)

### Error Recovery
- **Automatic Recovery**: Try next available provider automatically
- **Partial Analysis**: Generate what's possible, report failures
- **Resource Cleanup**: Proper cleanup of LLM connections on errors
- **Informative Messages**: Clear error messages with recovery suggestions

---

## Integration Points

### Pipeline Integration
- **Input**: Receives GNN models from Step 3 (gnn processing) and execution results from Step 12 (execute)
- **Output**: Generates LLM analyses for Step 16 (analysis), Step 20 (website generation), and Step 23 (report generation)
- **Dependencies**: Requires GNN parsing results from `3_gnn.py` output, optionally uses execution results from `12_execute.py`

### Module Dependencies
- **gnn/**: Reads parsed GNN model data for analysis
- **execute/**: Optionally uses execution results for enhanced analysis
- **analysis/**: Provides LLM insights for statistical analysis
- **report/**: Provides LLM-generated summaries for reports

### External Integration
- **OpenAI / OpenRouter / Perplexity APIs**: Cloud-based LLM analysis (OpenRouter and Perplexity via their own provider modules)
- **Ollama**: Local LLM execution for privacy and offline use

### Data Flow
```
3_gnn.py (GNN parsing)
  ↓
12_execute.py (Execution results) [optional]
  ↓
13_llm.py (LLM analysis)
  ↓
  ├→ 16_analysis.py (Enhanced analysis)
  ├→ 20_website.py (LLM summaries)
  ├→ 23_report.py (Report generation)
  └→ output/13_llm_output/ (Standalone analyses)
```

---

## Version History

### Current Version: 3.2.0

**Features**:
- Multi-provider LLM support (Ollama, OpenAI, OpenRouter, Perplexity)
- Automatic Ollama recovery
- Context-aware prompt generation
- Structured output parsing
- Rate limiting and error handling

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Enhanced prompt optimization
- **Future**: Multi-modal LLM support

---

## References

### Related Documentation
- [Pipeline Overview](../../README.md)
- [Architecture Guide](../../ARCHITECTURE.md)
- [Ollama Integration Guide](../../doc/llm/)
- [LLM Configuration](../../.agent_rules#ollama-llm-integration-standards)

### External Resources
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic API Documentation](https://docs.anthropic.com)
- [Ollama Documentation](https://ollama.ai/docs)

---

**Last Updated**: 2026-09-02
**Maintainer**: GNN Pipeline Team
**Status**: Production Ready
**Version**: 3.2.0
**Architecture Compliance**: Thin Orchestrator Pattern


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
