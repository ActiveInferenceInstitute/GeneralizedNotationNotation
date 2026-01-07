# LLM Processing Module - Agent Scaffolding

## Module Overview

**Purpose**: LLM-enhanced analysis, model interpretation, and AI assistance for GNN models

**Pipeline Step**: Step 13: LLM processing (13_llm.py)

**Category**: AI Enhancement / Analysis

**Status**: ✅ Production Ready

**Version**: 1.0.0

**Last Updated**: 2026-01-07

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

#### `process_llm(target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs) -> bool`
**Description**: Main LLM processing function with automatic Ollama fallback. Processes GNN files using LLM analysis with multi-provider support.

**Parameters**:
- `target_dir` (Path): Directory containing GNN files to analyze
- `output_dir` (Path): Output directory for LLM analyses
- `verbose` (bool): Enable verbose logging (default: False)
- `analysis_type` (str, optional): Type of analysis ("comprehensive", "summary", "explain", "optimize") (default: "comprehensive")
- `provider` (str, optional): LLM provider ("auto", "openai", "anthropic", "ollama") (default: "auto")
  - `"auto"`: Automatically select best available provider (checks API keys, then Ollama)
  - `"openai"`: Use OpenAI API (requires OPENAI_API_KEY)
  - `"anthropic"`: Use Anthropic API (requires ANTHROPIC_API_KEY)
  - `"ollama"`: Use local Ollama (requires Ollama installation)
- `llm_tasks` (str, optional): Specific tasks ("all", "summarize", "explain", "optimize") (default: "all")
- `llm_timeout` (int, optional): Timeout for LLM API calls in seconds (default: 60)
- `max_tokens` (int, optional): Maximum tokens in response (default: 2000)
- `model` (str, optional): Specific model to use (provider-specific)
- `**kwargs`: Additional LLM processing options

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
    analysis_type="comprehensive",
    provider="auto",
    llm_tasks="all"
)
```

#### `analyze_gnn_file_with_llm(content: str, model_name: str = None, analysis_type: str = "comprehensive", **kwargs) -> Dict[str, Any]`
**Description**: Analyze a GNN file using LLM with automatic provider selection.

**Parameters**:
- `content` (str): GNN file content as string
- `model_name` (str, optional): Name of the model for context
- `analysis_type` (str): Type of analysis to perform (default: "comprehensive")
- `**kwargs`: Additional analysis options

**Returns**: `Dict[str, Any]` - Analysis results dictionary

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

## Configuration

### Configuration Options

#### LLM Provider Selection
- `provider` (str): LLM provider to use (default: `"auto"`)
  - `"auto"`: Automatically select best available provider
  - `"openai"`: Use OpenAI API (requires OPENAI_API_KEY)
  - `"anthropic"`: Use Anthropic API (requires ANTHROPIC_API_KEY)
  - `"ollama"`: Use local Ollama (requires Ollama installation)

#### Analysis Type
- `analysis_type` (str): Type of analysis to perform (default: `"comprehensive"`)
  - `"comprehensive"`: Full model analysis
  - `"summary"`: Brief summary only
  - `"explain"`: Concept explanations
  - `"optimize"`: Optimization suggestions

#### LLM Tasks
- `llm_tasks` (str): Specific tasks to perform (default: `"all"`)
  - `"all"`: All available tasks
  - `"summarize"`: Generate model summary
  - `"explain"`: Explain Active Inference concepts
  - `"optimize"`: Suggest optimizations

#### Performance Settings
- `llm_timeout` (int): Timeout for LLM API calls in seconds (default: `60`)
- `max_tokens` (int): Maximum tokens in response (default: `2000`)
- `temperature` (float): LLM temperature (default: `0.7`)

#### Environment Variables
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `OLLAMA_HOST`: Ollama server host (default: `localhost:11434`)

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
- `src/tests/test_llm_integration.py` - Integration tests
- `src/tests/test_llm_ollama_integration.py` - Comprehensive Ollama tests ✨ NEW

### Test Coverage
- **Current**: 76%
- **Target**: 85%+

### Key Test Scenarios
1. Ollama detection and availability check
2. Model selection and prioritization
3. LLM processing with Ollama integration
4. Fallback mode when Ollama unavailable
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
# Install a lightweight model for testing
ollama pull smollm2:135m

# Or install a more capable model
ollama pull tinyllama
ollama pull llama2:7b
```

**Recommended Models for GNN Analysis**:
- **Fast & Small**: `smollm2:135m` (~135MB, 1-2s per prompt)
- **Balanced**: `tinyllama` (~637MB, 3-5s per prompt)
- **High Quality**: `llama2:7b` (~3.8GB, 10-30s per prompt)

**View Installed Models**:
```bash
ollama list
```

#### 4. LLM Timeout Issues
**Symptom**: "Prompt execution timed out" or slow responses

**Solution**:
- ✅ **Automatic**: Module now uses adaptive timeouts based on prompt complexity
- Environment variable override:
  ```bash
  export OLLAMA_TIMEOUT=120  # Increase timeout to 120 seconds
  ```

- Use faster model:
  ```bash
  export OLLAMA_MODEL=smollm2:135m
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
- ✅ Module automatically selects best available model
- Priority: smallest/fastest models first
- Preference order: smollm2 → tinyllama → llama2 → mistral

**Check Which Model Was Used**:
```bash
# View LLM results
cat output/13_llm_output/llm_results/llm_results.json | grep "selected_model"
```

#### 6. Fallback Mode Warnings
**Symptom**: "Proceeding with fallback LLM analysis" messages

**Explanation**: This is expected when Ollama is not available. The module provides basic analysis without live LLM interaction.

**Solution** (if you want LLM features):
1. Install and start Ollama (see issues #1 and #2)
2. Install at least one model (see issue #3)
3. Re-run the LLM step

**Fallback Capabilities**:
- ✅ Basic pattern extraction
- ✅ Variable and connection identification
- ✅ Structure analysis
- ❌ No natural language generation
- ❌ No model interpretation

#### 7. Slow LLM Processing
**Symptom**: Step 13 takes several minutes (3m+ per model)

**Causes**:
- Large models (llama2:70b, etc.)
- CPU-only inference (no GPU)
- Complex/long prompts
- Multiple GNN files being processed

**Solutions**:
1. **Use faster model**:
   ```bash
   export OLLAMA_MODEL=smollm2:135m  # ~1-2s per prompt
   ```

2. **Reduce prompt complexity**:
   ```bash
   export OLLAMA_MAX_TOKENS=256  # Shorter responses
   ```

3. **Enable GPU acceleration** (if available):
   ```bash
   # Ollama uses GPU automatically if detected
   ollama run llama2 --gpu
   ```

4. **Process files individually**:
   ```bash
   # Process one file at a time
   python src/13_llm.py --target-dir input/gnn_files --gnn-file specific_model.md
   ```

**Performance Benchmarks**:
- `smollm2:135m`: ~1-2s per prompt
- `tinyllama`: ~3-5s per prompt
- `llama2:7b` (CPU): ~10-30s per prompt
- `llama2:7b` (GPU): ~2-5s per prompt

#### 8. Memory Issues
**Symptom**: System slowdown or "out of memory" errors

**Solution**:
1. **Use smaller models**:
   ```bash
   ollama pull smollm2:135m  # Requires ~200MB RAM
   ```

2. **Limit concurrent processing**:
   - Process files one at a time
   - Close other applications

3. **Monitor resource usage**:
   ```bash
   # Monitor Ollama memory usage
   ps aux | grep ollama
   htop  # or top
   ```

**Memory Requirements**:
- `smollm2:135m`: ~200MB RAM
- `tinyllama`: ~700MB RAM
- `llama2:7b`: ~4-6GB RAM
- `llama2:13b`: ~8-12GB RAM

### Ollama Integration Features

#### ✅ Enhanced Detection (October 2025)
- Automatic Ollama availability check
- Model listing and validation
- Service health monitoring (port 11434)
- Helpful installation instructions when not found

#### ✅ Intelligent Model Selection
- Prioritizes small, fast models for quick execution
- Automatic fallback chain
- Environment variable override support
- Logs selected model for transparency

#### ✅ Progress Tracking
- File-by-file progress indicators
- Prompt-by-prompt completion tracking
- Detailed logging with emoji indicators 📝
- Clear success/failure indicators ✅/❌

#### ✅ Error Recovery
- Graceful fallback when Ollama unavailable
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
   - **Quick Testing**: `smollm2:135m`
   - **Balanced Analysis**: `tinyllama`
   - **Deep Analysis**: `llama2:7b`

3. **Monitor Performance**:
   ```bash
   # Run with verbose logging
   python src/13_llm.py --verbose --target-dir input/gnn_files
   
   # Check timing in results
   cat output/13_llm_output/llm_results/llm_results.json
   ```

4. **Optimize for Speed**:
   ```bash
   # Use fastest model and limit tokens
   export OLLAMA_MODEL=smollm2:135m
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
export OLLAMA_TEST_MODEL=smollm2:135m   # Test/CI model

# Performance tuning
export OLLAMA_MAX_TOKENS=512            # Maximum response length
export OLLAMA_TIMEOUT=60                # Request timeout (seconds)
export OLLAMA_HOST=http://localhost:11434  # Ollama server URL

# Behavior
export OLLAMA_DISABLED=0                # Disable Ollama (use fallback)
export DEFAULT_PROVIDER=ollama          # Default LLM provider
```

#### Custom Model Configuration
```python
# In your code or config
from llm.llm_processor import get_default_provider_configs

configs = get_default_provider_configs()
configs['ollama']['default_model'] = 'my-custom-model'
configs['ollama']['default_max_tokens'] = 1024
```

---

## Error Handling

### Graceful Degradation
- **No API Keys**: Automatically fallback to Ollama if available
- **Ollama Unavailable**: Skip LLM analysis, log informative message, continue pipeline
- **LLM Timeout**: Retry with shorter timeout, then skip if still fails
- **Invalid Response**: Parse what's possible, log warning

### Error Categories
1. **Provider Unavailable**: No API keys and Ollama not available (fallback: skip analysis)
2. **API Errors**: Rate limits, network errors (fallback: retry with backoff)
3. **Timeout Errors**: LLM response too slow (fallback: use faster model or skip)
4. **Parsing Errors**: Invalid LLM response format (fallback: use raw response)

### Error Recovery
- **Automatic Fallback**: Try next available provider automatically
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
- **OpenAI API**: Cloud-based LLM analysis
- **Anthropic API**: Cloud-based LLM analysis
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

### Current Version: 1.0.0

**Features**:
- Multi-provider LLM support (OpenAI, Anthropic, Ollama)
- Automatic Ollama fallback
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
- [LLM Configuration](../../.cursorrules#ollama-llm-integration-standards)

### External Resources
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic API Documentation](https://docs.anthropic.com)
- [Ollama Documentation](https://ollama.ai/docs)

---

**Last Updated**: 2026-01-07
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Version**: 1.0.0
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern
