# LLM Multi-Provider System

The GNN pipeline includes a sophisticated multi-provider LLM system that supports OpenAI, OpenRouter, and Perplexity APIs for analyzing GNN models. This system provides unified access to different AI capabilities while maintaining provider-specific optimizations.

## Architecture Overview

```
LLMProcessor
├── OpenAIProvider (GPT-4, GPT-3.5, embeddings)
├── OpenRouterProvider (unified access to 100+ models)
└── PerplexityProvider (search-enhanced AI analysis)
```

## Quick Start

### 1. Environment Setup

Create a `.env` file based on the provided template:

```bash
# Copy the example environment file
cp src/llm/.env.example .env

# Edit with your API keys
nano .env
```

Required environment variables:
```bash
# At least one provider key is required
OPENAI_API_KEY=your_openai_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here  
PERPLEXITY_API_KEY=your_perplexity_api_key_here

# Optional configurations
DEFAULT_PROVIDER=openai
OPENROUTER_SITE_NAME="GNN Analysis Pipeline"
```

### 2. Basic Usage

```python
import asyncio
from src.llm import initialize_global_processor, AnalysisType

async def analyze_gnn_model():
    # Initialize processor with environment settings
    processor = await initialize_global_processor()
    
    # Read your GNN file
    with open("path/to/your/model.md", "r") as f:
        gnn_content = f.read()
    
    # Analyze the model
    response = await processor.analyze_gnn(
        gnn_content=gnn_content,
        analysis_type=AnalysisType.SUMMARY
    )
    
    print(f"Analysis from {response.provider}:")
    print(response.content)
    
    # Clean up
    await processor.close()

# Run the analysis
asyncio.run(analyze_gnn_model())
```

## Provider Capabilities

### OpenAI Provider
- **Models**: GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo
- **Strengths**: General analysis, structured reasoning, embeddings
- **Best for**: Summary, structure analysis, validation

```python
# Use OpenAI specifically
response = await processor.analyze_gnn(
    gnn_content=content,
    provider_type=ProviderType.OPENAI,
    analysis_type=AnalysisType.STRUCTURE
)
```

### OpenRouter Provider
- **Models**: 100+ models from multiple providers (OpenAI, Anthropic, Google, Meta, etc.)
- **Strengths**: Model diversity, cost optimization, provider routing
- **Best for**: Enhancement suggestions, comparisons, specialized models

```python
# Use specific model through OpenRouter
config = LLMConfig(
    model="anthropic/claude-3.5-sonnet",
    max_tokens=2000,
    temperature=0.1
)

response = await processor.analyze_gnn(
    gnn_content=content,
    provider_type=ProviderType.OPENROUTER,
    config=config
)
```

### Perplexity Provider
- **Models**: Sonar models with online search capabilities
- **Strengths**: Real-time information, research augmentation, citations
- **Best for**: Search-enhanced analysis, current research integration

```python
# Search-enhanced analysis with Perplexity
response = await processor.analyze_gnn(
    gnn_content=content,
    analysis_type=AnalysisType.SEARCH_ENHANCED
)

# Access citations if available
if response.metadata and 'citations' in response.metadata:
    print("Sources:", response.metadata['citations'])
```

## Analysis Types

The system supports different analysis types optimized for GNN models:

```python
class AnalysisType(Enum):
    SUMMARY = "summary"           # Concise model overview
    STRUCTURE = "structure"       # Detailed structural analysis  
    QUESTIONS = "questions"       # Generate insight questions
    ENHANCEMENT = "enhancement"   # Improvement suggestions
    VALIDATION = "validation"     # Model validation checks
    COMPARISON = "comparison"     # Compare with other models
    SEARCH_ENHANCED = "search_enhanced"  # Research-augmented analysis
```

## Advanced Features

### Provider Comparison

Compare responses from multiple providers:

```python
# Get analysis from all available providers
results = await processor.compare_providers(
    gnn_content=content,
    analysis_type=AnalysisType.SUMMARY
)

for provider_name, response in results.items():
    if response:
        print(f"\n--- {provider_name.upper()} ---")
        print(response.content)
```

### Streaming Responses

Get real-time streaming responses:

```python
messages = [
    LLMMessage(role="system", content="You are a GNN expert."),
    LLMMessage(role="user", content=f"Analyze this model: {gnn_content}")
]

async for chunk in processor.generate_stream(messages):
    print(chunk, end="", flush=True)
```

### Custom Configuration

Fine-tune provider behavior:

```python
config = LLMConfig(
    model="gpt-4o",
    max_tokens=3000,
    temperature=0.2,
    top_p=0.9,
    frequency_penalty=0.1
)

response = await processor.analyze_gnn(
    gnn_content=content,
    config=config
)
```

## Provider Selection Logic

The system automatically selects the best provider for each task:

- **SEARCH_ENHANCED**: Perplexity → OpenRouter → OpenAI
- **STRUCTURE**: OpenAI → OpenRouter → Perplexity  
- **ENHANCEMENT**: OpenRouter → OpenAI → Perplexity
- **Default**: Your configured preferred order

Override with specific provider:
```python
response = await processor.analyze_gnn(
    gnn_content=content,
    provider_type=ProviderType.PERPLEXITY
)
```

## Error Handling

The system includes comprehensive error handling with automatic fallbacks:

```python
try:
    response = await processor.analyze_gnn(gnn_content)
except RuntimeError as e:
    print(f"All providers failed: {e}")
except ValueError as e:
    print(f"Configuration error: {e}")
```

Monitor provider availability:
```python
available = processor.get_available_providers()
print(f"Available providers: {[p.value for p in available]}")

info = processor.get_provider_info()
for provider, details in info.items():
    print(f"{provider}: {details}")
```

## Integration with GNN Pipeline

The LLM processor integrates seamlessly with the GNN pipeline:

```python
# From pipeline step 11 (11_llm.py)
from src.llm import get_processor

async def analyze_discovered_models(gnn_files):
    processor = get_processor()
    
    if not processor:
        processor = await initialize_global_processor()
    
    results = []
    for gnn_file in gnn_files:
        try:
            response = await processor.analyze_gnn(
                gnn_content=gnn_file.read_text(),
                analysis_type=AnalysisType.STRUCTURE
            )
            results.append({
                'file': gnn_file.name,
                'analysis': response.content,
                'provider': response.provider,
                'tokens': response.usage
            })
        except Exception as e:
            logger.error(f"Failed to analyze {gnn_file}: {e}")
    
    return results
```

## Configuration Options

### Environment Variables

```bash
# Provider API Keys
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...
PERPLEXITY_API_KEY=pplx-...

# Provider Preferences
DEFAULT_PROVIDER=openai|openrouter|perplexity
ENABLE_FALLBACK=true|false
ENABLE_STREAMING=true|false

# Default Parameters
DEFAULT_TEMPERATURE=0.3
DEFAULT_MAX_TOKENS=2000

# OpenAI Specific
OPENAI_ORG_ID=org-...
OPENAI_BASE_URL=https://api.openai.com/v1

# OpenRouter Specific  
OPENROUTER_SITE_URL=https://your-site.com
OPENROUTER_SITE_NAME="Your Application Name"
```

### Programmatic Configuration

```python
# Custom provider configuration
provider_configs = {
    'openai': {
        'organization': 'org-your-org',
        'base_url': 'https://api.openai.com/v1'
    },
    'openrouter': {
        'site_url': 'https://myapp.com',
        'site_name': 'My GNN Analysis App'
    }
}

processor = LLMProcessor(
    preferred_providers=[ProviderType.PERPLEXITY, ProviderType.OPENAI],
    api_keys={'perplexity': 'pplx-...', 'openai': 'sk-...'},
    provider_configs=provider_configs
)
```

## Best Practices

1. **API Key Security**:
   - Use environment variables, never hardcode keys
   - Rotate keys regularly
   - Monitor usage and costs

2. **Provider Selection**:
   - Use Perplexity for research-heavy analysis
   - Use OpenAI for general analysis and structured tasks
   - Use OpenRouter for cost optimization and model diversity

3. **Error Handling**:
   - Always handle provider failures gracefully
   - Implement retry logic for transient errors
   - Monitor provider availability

4. **Performance**:
   - Use streaming for long responses
   - Implement caching for repeated analyses
   - Consider token limits and costs

5. **GNN-Specific**:
   - Provide context about Active Inference concepts
   - Use structured prompts for consistent analysis
   - Validate responses for scientific accuracy

## Troubleshooting

### Common Issues

**No providers initialized**:
```bash
# Check API keys are set
env | grep -E "(OPENAI|OPENROUTER|PERPLEXITY)_API_KEY"

# Verify key format and permissions
python -c "import os; print('✓' if os.getenv('OPENAI_API_KEY', '').startswith('sk-') else '✗ Invalid OpenAI key format')"
```

**Provider connection errors**:
```python
# Test individual providers
provider = OpenAIProvider(api_key="sk-...")
success = provider.initialize()
print(f"OpenAI initialization: {'✓' if success else '✗'}")
```

**Rate limiting**:
```python
# Implement backoff
import asyncio

async def analyze_with_backoff(processor, content, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await processor.analyze_gnn(content)
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
                continue
            raise
```

## Contributing

To add a new provider:

1. Create `src/llm/providers/new_provider.py`
2. Inherit from `BaseLLMProvider`  
3. Implement required abstract methods
4. Add to `ProviderType` enum
5. Register in `LLMProcessor._create_provider`
6. Add configuration support
7. Write tests and documentation

## API Reference

See the individual provider modules for detailed API documentation:
- `src/llm/providers/base_provider.py` - Base interface
- `src/llm/providers/openai_provider.py` - OpenAI implementation
- `src/llm/providers/openrouter_provider.py` - OpenRouter implementation  
- `src/llm/providers/perplexity_provider.py` - Perplexity implementation
- `src/llm/llm_processor.py` - Main processor interface 