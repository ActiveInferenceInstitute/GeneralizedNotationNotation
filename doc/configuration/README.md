# GNN Configuration Guide

> **📋 Document Metadata**  
> **Type**: Configuration Guide | **Audience**: Developers & System Administrators | **Complexity**: Intermediate  
> **Cross-References**: [Pipeline Architecture](../gnn/gnn_tools.md) | [Deployment Guide](../deployment/README.md)

## Overview
This guide covers all configuration options for the GeneralizedNotationNotation (GNN) pipeline, from basic settings to advanced customization.

## Configuration Hierarchy

GNN uses a layered configuration system with the following precedence (highest to lowest):

1. **Command Line Arguments** - Override all other settings
2. **Environment Variables** - System-level configuration  
3. **Project Config File** - `config.yaml` in project root
4. **User Config File** - `~/.gnn/config.yaml`
5. **Default Settings** - Built into the code

## Main Configuration File

### Location
Primary configuration file: `config.yaml` (create in project root)

### Complete Configuration Template

```yaml
# config.yaml - Complete GNN Configuration
# Copy and modify this template for your needs

# Global Pipeline Settings
pipeline:
  # Which steps to run (0-24, or "all")
steps: "all"  # or a subset, e.g., [1, 4, 5, 6]
  
  # Steps to skip
  skip_steps: []  # e.g., [11, 12, 13] to skip LLM and DisCoPy steps
  
  # Execution mode
  parallel: true        # Run compatible steps in parallel
  sequential: false     # Force sequential execution
  fail_fast: true      # Stop on first error
  
  # Directories
  target_dir: "input/gnn_files/"     # Input GNN files
  output_dir: "output/"             # All output files
  temp_dir: "temp/"                # Temporary files
  
  # Resource management
  max_memory_gb: 8.0    # Maximum memory usage
  max_processes: 4      # Parallel process limit
  cleanup: true         # Clean temp files after completion
  
  # Logging
  log_level: "INFO"     # DEBUG, INFO, WARNING, ERROR
  log_file: "output/logs/pipeline.log"
  verbose: false

# Step 1: GNN File Discovery and Parsing
gnn:
  # File discovery
  file_patterns: ["*.md", "*.gnn"]
  recursive_search: true
  exclude_patterns: [".*", "_*", "temp*"]
  
  # Parsing options
  strict_syntax: true
  allow_incomplete: false  # Allow missing optional sections
  validate_on_load: true
  
  # Encoding
  file_encoding: "utf-8"
  normalize_whitespace: true

# Step 2: Environment Setup
setup:
  # Virtual environment
  create_venv: true
  venv_dir: "src/.venv"
  python_version: "3.8+"
  
  # Dependencies
  install_deps: true
  upgrade_deps: false
  requirements_file: "requirements.txt"
  extra_packages: []
  
  # External tools
  check_julia: true      # For RxInfer backend
  check_graphviz: true   # For visualization
  
  # System checks
  min_memory_gb: 2.0
  min_disk_gb: 1.0

# Step 3: Testing Configuration
testing:
  # Test discovery
  test_dir: "src/tests/"
  test_patterns: ["test_*.py", "*_test.py"]
  
  # Test execution
  parallel_tests: true
  coverage: true
  coverage_threshold: 80.0
  
  # Test data
  use_example_data: true
  generate_test_cases: false

# Step 4: Type Checking and Validation
validation:
  # Syntax checking
  strict_mode: true
  check_dimensions: true
  check_stochasticity: true
  check_matrix_compatibility: true
  
  # Semantic validation
  validate_active_inference: true
  check_causal_consistency: true
  warn_unused_variables: true
  
  # Resource estimation
  estimate_memory: true
  estimate_compute: true
  warn_large_models: true
  max_model_size: 1000000  # Max total parameters

# Step 5: Export Configuration
export:
  # Output formats
  formats: ["json", "xml", "graphml", "dot", "yaml"]
  
  # JSON export
  json:
    pretty_print: true
    indent: 2
    ensure_ascii: false
  
  # XML export  
  xml:
    pretty_print: true
    encoding: "utf-8"
    include_schema: true
  
  # GraphML export
  graphml:
    include_attributes: true
    node_labels: true
    edge_labels: true
  
  # General export options
  overwrite_existing: true
  create_subdirs: true

# Step 6: Visualization Configuration
visualization:
  # Graph layout
  layout: "spring"  # spring, hierarchical, circular, random
  
  # Output formats
  formats: ["png", "svg", "pdf"]
  dpi: 300
  
  # Graph appearance
  node_size: 1000
  node_color: "lightblue"
  edge_width: 2.0
  font_size: 12
  
  # Graph filtering
  max_nodes: 100
  max_edges: 200
  hide_isolated_nodes: false
  
  # Advanced options
  use_graphviz: true
  hierarchical_layout: false
  save_source_dot: true

# Step 22: Model Context Protocol (MCP)
mcp:
  # Server configuration
  enabled: true
  host: "localhost"
  port: 8000
  protocol: "http"  # http, stdio
  
  # Tools to register
  tools: ["gnn_parse", "gnn_validate", "gnn_export", "gnn_visualize"]
  
  # Security
  require_auth: false
  api_key: null
  allowed_origins: ["*"]
  
  # Rate limiting
  rate_limit: 100  # requests per minute
  max_request_size: "10MB"

# Step 10: Ontology Processing
ontology:
  # Active Inference Ontology integration
  enabled: true
  ontology_url: "https://github.com/ActiveInferenceInstitute/ActiveInferenceOntology"
  local_ontology_path: "ontologies/"
  
  # Validation
  validate_terms: true
  suggest_terms: true
  strict_matching: false
  
  # Annotation
  auto_annotate: true
  confidence_threshold: 0.8

# Step 11: Code Rendering
rendering:
  # Target backends
  backends: ["pymdp", "rxinfer"]  # Available: pymdp, rxinfer, jax, custom
  
  # PyMDP backend
  pymdp:
    version: "latest"
    template_dir: "src/render/pymdp/templates/"
    include_visualization: true
    optimization_level: 1
  
  # RxInfer backend
  rxinfer:
    julia_version: "1.9+"
    template_dir: "src/render/rxinfer/templates/"
    package_env: "default"
    compile_static: false
  
  # JAX backend
  jax:
    use_jit: true
    use_gpu: false  # auto-detect
    precision: "float32"
  
  # General rendering
  output_structure: "backend_separated"  # backend_separated, mixed
  include_docs: true
  include_tests: true

# Step 12: Simulation Execution
execution:
  # Execution modes
  dry_run: false
  timeout_seconds: 300
  capture_output: true
  
  # Resource limits
  max_memory_per_sim: "2GB"
  max_cpu_cores: 2
  
  # Output handling
  save_results: true
  compress_output: false
  
  # Error handling
  continue_on_error: false
  retry_failed: 1

# Step 13: LLM Integration
llm:
  # LLM providers and models
  default_provider: "openai"
  
  # OpenAI configuration
  openai:
    model: "gpt-4"
    api_key: "${OPENAI_API_KEY}"  # Environment variable
    temperature: 0.1
    max_tokens: 2000
    timeout: 30
  
  # Anthropic configuration
  anthropic:
    model: "claude-3-sonnet-20240229"
    api_key: "${ANTHROPIC_API_KEY}"
    temperature: 0.1
    max_tokens: 2000
  
  # Local models (via Ollama/similar)
  local:
    enabled: false
    endpoint: "http://localhost:11434"
    model: "llama2"
  
  # Analysis options
  analysis_types: ["validation", "explanation", "optimization"]
  include_suggestions: true
  critique_models: true
  
  # Safety and filtering
  content_filter: true
  max_retries: 3
  fallback_to_local: false

## Rendering Option: DisCoPy Categorical Diagrams (within Step 11)
discopy:
  # Category theory settings
  category_type: "pregroup"  # pregroup, monoidal, hypergraph
  
  # Diagram generation
  layout: "tree"  # tree, circuit, spiral
  save_diagrams: true
  formats: ["svg", "png", "tikz"]
  
  # Mathematical rigor
  check_composition: true
  validate_functors: true
  
  # Optimization
  simplify_diagrams: true
  remove_identities: true

## Execution Option: JAX Evaluation (within Step 12)
jax_eval:
  # JAX configuration
  platform: "auto"  # auto, cpu, gpu, tpu
  jit_compile: true
  
  # Numerical settings
  precision: "float32"  # float16, float32, float64
  backend: "xla"
  
  # Performance
  parallel_evaluation: true
  batch_size: 32
  
  # Memory management
  preallocate_memory: true
  memory_fraction: 0.8

# Step 20: Website Generation
site:
  # Static site configuration
  enabled: true
  theme: "default"  # default, academic, minimal
  
  # Content generation
  include_models: true
  include_visualizations: true
  include_analysis: true
  
  # Output
  output_dir: "output/site/"
  base_url: "/"
  
  # Features
  search_enabled: true
  comments_enabled: false
  analytics_enabled: false

# Environment-specific overrides
development:
  pipeline:
    log_level: "DEBUG"
    verbose: true
  validation:
    strict_mode: false
  testing:
    coverage_threshold: 70.0

production:
  pipeline:
    log_level: "WARNING"
    fail_fast: true
  validation:
    strict_mode: true
  export:
    overwrite_existing: false

# External integrations
integrations:
  # GitHub integration
  github:
    enabled: false
    token: "${GITHUB_TOKEN}"
    repo: "user/repo"
    create_issues: false
  
  # Weights & Biases
  wandb:
    enabled: false
    project: "gnn-models"
    api_key: "${WANDB_API_KEY}"
  
  # Custom webhooks
  webhooks:
    enabled: false
    urls: []
    events: ["pipeline_complete", "error"]
```

## Environment Variables

### Required Variables
```bash
# LLM Integration (choose one or more)
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Optional integrations
export GITHUB_TOKEN="your-github-token"
export WANDB_API_KEY="your-wandb-key"
```

### System Configuration
```bash
# Java/Julia paths (if not in PATH)
export JAVA_HOME="/usr/lib/jvm/java-11"
export JULIA_PATH="/opt/julia/bin/julia"

# Custom Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/gnn/src"

# Resource limits
export GNN_MAX_MEMORY="8GB"
export GNN_MAX_PROCESSES="4"
```

## Command Line Configuration

### Override Any Config Setting
```bash
# Override pipeline settings
python src/main.py --config pipeline.parallel=false --config validation.strict_mode=true

# Override nested settings
python src/main.py --config llm.openai.model=gpt-3.5-turbo

# Multiple overrides
python src/main.py \
  --config pipeline.steps=[1,2,4] \
  --config export.formats=['json'] \
  --config visualization.layout=hierarchical
```

### Common Command Patterns
```bash
# Development mode (more verbose, less strict)
python src/main.py --profile development

# Production mode (strict, minimal output)
python src/main.py --profile production

# Custom target and output
python src/main.py --target-dir my_models/ --output-dir results/

# Skip expensive steps  
python src/main.py --skip 11,12,13

# Run only specific steps
python src/main.py --only-steps 1,4,6

# Debug mode
python src/main.py --debug --verbose
```

## Profile-Based Configuration

### Creating Profiles
Create profile-specific config files:

```yaml
# config.development.yaml
pipeline:
  log_level: "DEBUG"
  verbose: true
validation:
  strict_mode: false
testing:
  coverage_threshold: 60.0
```

```yaml
# config.production.yaml  
pipeline:
  log_level: "ERROR"
  fail_fast: true
validation:
  strict_mode: true
export:
  overwrite_existing: false
```

### Using Profiles
```bash
python src/main.py --profile development
python src/main.py --profile production
```

## Step-Specific Configuration

### Individual Step Configuration Files
```yaml
# config.step4.yaml - Type checker specific
validation:
  strict_mode: true
  custom_rules: "rules/active_inference.yaml"
  
# config.step6.yaml - Visualization specific  
visualization:
  custom_themes: "themes/"
  export_interactive: true
```

### Loading Step Configuration
```bash
python src/5_type_checker.py --config config.step4.yaml
python src/6_visualization.py --config config.step6.yaml
```

## Advanced Configuration Patterns

### Dynamic Configuration
```python
# Python code can modify config at runtime
import src.utils.config as config

# Load base config
cfg = config.load_config("config.yaml")

# Modify based on runtime conditions
if model_size > 10000:
    cfg['validation']['strict_mode'] = False
    cfg['visualization']['max_nodes'] = 50

# Apply modified config
config.apply_config(cfg)
```

### Conditional Configuration
```yaml
# Conditional based on environment
pipeline:
  steps: "all"
  parallel: true
  # Override for CI environment
  ${CI:pipeline.parallel}: false
  ${CI:pipeline.log_level}: "DEBUG"
```

### Template Variables
```yaml
# Using template variables
directories:
  output: "${PROJECT_ROOT}/output"
  temp: "${TEMP_DIR}/gnn"
  
model_settings:
  max_size: ${MAX_MODEL_SIZE:1000000}
```

## Validation and Testing

### Validate Configuration
```bash
# Check config syntax and completeness
python src/main.py --validate-config

# Test configuration with dry run
python src/main.py --dry-run --config config.yaml

# Show effective configuration (after all overrides)
python src/main.py --show-config
```

### Configuration Schema
The configuration follows a JSON Schema for validation:
```bash
# Validate against schema
python -m src.utils.validate_config config.yaml
```

## Best Practices

### 1. Environment Separation
- Use different config files for dev/test/prod
- Keep sensitive data in environment variables
- Use profiles for common configuration sets

### 2. Version Control
- Track config files in version control
- Use `.env.example` for environment variables
- Document configuration changes

### 3. Security
- Never commit API keys to version control
- Use environment variables for secrets
- Restrict file permissions on config files

### 4. Performance
- Adjust resource limits based on your hardware
- Use parallel processing when available
- Skip unnecessary steps for faster iteration

### 5. Debugging
- Enable debug logging for troubleshooting
- Use dry-run mode to test configurations
- Validate configuration before running pipeline

This comprehensive configuration system allows fine-tuned control over every aspect of the GNN pipeline while maintaining sensible defaults for common use cases. 