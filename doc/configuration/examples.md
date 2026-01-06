# GNN Configuration Examples

This document provides comprehensive examples of GNN configuration for different use cases, deployment scenarios, and advanced workflows.

## Table of Contents

1. [Basic Configuration Examples](#basic-configuration)
2. [Development Configurations](#development-configurations)
3. [Production Configurations](#production-configurations)
4. [Multi-Target Configurations](#multi-target-configurations)
5. [Performance Optimization](#performance-optimization)
6. [Cloud and Distributed Configurations](#cloud-configurations)
7. [Specialized Workflows](#specialized-workflows)
8. [Troubleshooting Configurations](#troubleshooting-configurations)

## Basic Configuration Examples

### Minimal Configuration
```yaml
# config/minimal.yaml
version: "1.0"
name: "minimal_gnn_pipeline"

# Basic pipeline settings
pipeline:
  steps: [1, 2, 3, 4, 5]  # Just essential steps
  continue_on_error: false
  
# Basic I/O
paths:
  target_dir: "./examples"
  output_dir: "./output"
  
# Simple logging
logging:
  level: INFO
  console: true
  file: false
```

### Standard Development Configuration
```yaml
# config/development.yaml
version: "1.0"
name: "gnn_development"
description: "Standard development configuration with full pipeline"

# Complete pipeline for development
pipeline:
  steps: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
  continue_on_error: true
  parallel_execution: false
  timeout_seconds: 300

# Development paths
paths:
  target_dir: "./src/gnn/gnn_examples"
  output_dir: "./dev_output"
  cache_dir: "./cache"
  logs_dir: "./logs"
  
# Detailed logging for development
logging:
  level: DEBUG
  console: true
  file: true
  rotation: daily
  max_files: 30
  
# Development-friendly settings
development:
  auto_reload: true
  debug_mode: true
  verbose_output: true
  show_progress: true
  
# Basic validation
validation:
  strict_mode: false
  check_dependencies: true
  validate_gnn_syntax: true
```

### Research Configuration
```yaml
# config/research.yaml
version: "1.0"
name: "gnn_research_pipeline"
description: "Configuration optimized for research workflows"

# Research-focused pipeline
pipeline:
  steps: [1, 2, 3, 4, 5, 6, 9, 11, 12, 13]  # Skip visualization, MCP for faster iteration
  continue_on_error: true
  parallel_execution: true
  max_workers: 4

# Research paths
paths:
  target_dir: "./research/models"
  output_dir: "./research/results"
  archive_dir: "./research/archive"
  
# Performance settings for research
performance:
  cache_enabled: true
  cache_size: "2GB"
  memory_limit: "8GB"
  
# Research-specific outputs
outputs:
  save_intermediates: true
  generate_reports: true
  export_formats: ["json", "xml", "graphml"]
  
# Reproducibility settings
reproducibility:
  seed: 42
  version_tracking: true
  parameter_logging: true
```

## Development Configurations

### Local Development with Hot Reload
```yaml
# config/dev_hotreload.yaml
version: "1.0"
name: "dev_hotreload"

# File watching for development
development:
  hot_reload: true
  watch_directories: ["./src/gnn/gnn_examples", "./src/templates"]
  watch_extensions: [".md", ".gnn", ".yaml"]
  reload_delay: 2.0
  
# Fast iteration settings
pipeline:
  steps: [1, 2, 4, 5]  # Skip heavy steps during development
  timeout_seconds: 60
  
# Development debugging
debugging:
  enable_pdb: true
  save_stack_traces: true
  profile_performance: true
  memory_profiling: false
  
# Live output
output:
  live_console: true
  progress_bars: true
  color_output: true
```

### Testing and Validation Focus
```yaml
# config/testing.yaml
version: "1.0"
name: "testing_pipeline"

# Testing-focused pipeline
pipeline:
  steps: [1, 2, 3, 4]  # Focus on parsing, setup, tests, validation
  
# Comprehensive validation
validation:
  strict_mode: true
  check_syntax: true
  check_semantics: true
  check_dependencies: true
  validate_matrices: true
  check_dimensions: true
  validate_connections: true
  
# Testing settings
testing:
  run_unit_tests: true
  run_integration_tests: true
  run_performance_tests: false
  test_timeout: 120
  
# Coverage and reporting
coverage:
  enable: true
  min_coverage: 80
  report_format: "html"
  
# Error handling for testing
error_handling:
  fail_fast: true
  detailed_errors: true
  error_context: 5  # Lines of context around errors
```

### Benchmark Configuration
```yaml
# config/benchmark.yaml
version: "1.0"
name: "gnn_benchmarks"

# Performance benchmarking
pipeline:
  steps: [1, 2, 3, 4, 5, 9, 10]  # Core execution pipeline
  iterations: 10  # Run multiple times for benchmarking
  
# Benchmarking settings
benchmarking:
  enable: true
  measure_memory: true
  measure_cpu: true
  measure_disk_io: true
  profile_detailed: true
  
# Performance monitoring
monitoring:
  sample_interval: 0.1
  memory_threshold: "4GB"
  cpu_threshold: 90
  
# Resource limits for consistent benchmarking
resources:
  max_memory: "6GB"
  max_cpu_cores: 4
  timeout: 600
  
# Benchmark reporting
reporting:
  generate_charts: true
  compare_baselines: true
  export_metrics: true
  format: ["json", "csv", "html"]
```

## Production Configurations

### Production Deployment
```yaml
# config/production.yaml
version: "1.0"
name: "gnn_production"
description: "Production-ready configuration with reliability features"

# Reliable pipeline execution
pipeline:
  steps: [1, 2, 3, 4, 5, 6, 8, 9, 10, 12]  # Skip development tools
  continue_on_error: false
  retry_attempts: 3
  retry_delay: 5.0
  timeout_seconds: 1800
  
# Production paths with proper isolation
paths:
  target_dir: "/data/gnn/models"
  output_dir: "/data/gnn/output"
  logs_dir: "/var/log/gnn"
  cache_dir: "/var/cache/gnn"
  temp_dir: "/tmp/gnn"
  
# Production logging
logging:
  level: INFO
  console: false
  file: true
  syslog: true
  structured: true
  format: "json"
  
# Security settings
security:
  sandbox_execution: true
  restrict_file_access: true
  allowed_paths: ["/data/gnn", "/tmp/gnn"]
  disable_shell_access: true
  
# Monitoring and health checks
monitoring:
  enable: true
  health_check_interval: 60
  metrics_endpoint: "http://localhost:8080/metrics"
  alerting:
    email: "admin@example.com"
    slack_webhook: "https://hooks.slack.com/..."
    
# Resource management
resources:
  max_memory: "16GB"
  max_cpu_cores: 8
  disk_space_warning: "1GB"
  cleanup_temp_files: true
```

### High-Performance Production
```yaml
# config/production_hpc.yaml
version: "1.0"
name: "gnn_hpc_production"

# High-performance pipeline
pipeline:
  steps: [1, 2, 3, 4, 5, 9, 10, 12, 13]  # Focus on computation-heavy steps
  parallel_execution: true
  max_workers: 16
  worker_type: "process"  # Use multiprocessing for CPU-bound tasks
  
# HPC-specific settings
hpc:
  scheduler: "slurm"  # or "pbs", "sge"
  partition: "gpu"
  nodes: 4
  tasks_per_node: 8
  memory_per_node: "64GB"
  walltime: "04:00:00"
  
# GPU acceleration
gpu:
  enable: true
  devices: [0, 1, 2, 3]
  memory_fraction: 0.8
  
# Optimized I/O
io:
  parallel_io: true
  buffer_size: "64MB"
  compression: "lz4"
  async_writes: true
  
# Advanced caching
caching:
  strategy: "distributed"
  cache_size: "32GB"
  cache_levels: 3
  compression: true
```

### Container Production Configuration
```yaml
# config/container_production.yaml
version: "1.0"
name: "gnn_container"

# Container-optimized settings
container:
  base_image: "python:3.11-slim"
  working_dir: "/app"
  user: "gnn"
  
# Containerized paths
paths:
  target_dir: "/data/input"
  output_dir: "/data/output"
  logs_dir: "/logs"
  
# Container resource limits
resources:
  memory_limit: "4GB"
  cpu_limit: 2.0
  shared_memory: "512MB"
  
# Health and monitoring for containers
health:
  check_command: "python -c 'import src.main; print(\"OK\")'"
  check_interval: 30
  check_timeout: 10
  
# Environment variables
environment:
  PYTHONPATH: "/app/src"
  PYTHONUNBUFFERED: "1"
  TZ: "UTC"
```

## Multi-Target Configurations

### PyMDP and RxInfer Configuration
```yaml
# config/multi_target.yaml
version: "1.0"
name: "multi_target_rendering"

# Pipeline with multiple renderers
pipeline:
  steps: [1, 2, 3, 4, 5, 9, 10]
  
# Multiple target configurations
targets:
  pymdp:
    enable: true
    template_dir: "./src/render/pymdp/templates"
    output_subdir: "pymdp"
    matrix_validation: true
    placeholder_matrices: false
    
  rxinfer:
    enable: true
    template_dir: "./src/render/rxinfer/templates"
    output_subdir: "rxinfer"
    factor_graph_validation: true
    message_passing_algorithm: "belief_propagation"
    
  discopy:
    enable: true
    output_subdir: "discopy"
    jax_backend: true
    category_type: "compact_closed"
    
# Target-specific parameters
rendering:
  pymdp:
    agent_class: "Agent"
    planning_horizon: 5
    use_utility: true
    
  rxinfer:
    inference_engine: "variational_bayes"
    iterations: 100
    tolerance: 1e-6
    
  discopy:
    backend: "jax"
    precision: "float32"
```

### Research Multi-Framework
```yaml
# config/research_multi.yaml
version: "1.0"
name: "research_comparison"

# Comparative analysis pipeline
pipeline:
  steps: [1, 2, 3, 4, 5, 9, 10, 11]  # Include LLM analysis
  
# Multiple frameworks for comparison
frameworks:
  active_inference:
    pymdp:
      enable: true
      variants: ["standard", "planning", "learning"]
    rxinfer:
      enable: true
      algorithms: ["belief_propagation", "variational_bayes"]
      
  category_theory:
    discopy:
      enable: true
      categories: ["monoidal", "compact_closed", "traced"]
      
  machine_learning:
    pytorch:
      enable: true
      model_types: ["neural_ode", "graph_neural_network"]
    jax:
      enable: true
      compilation: true
      
# Comparison analysis
analysis:
  comparative_metrics: true
  performance_comparison: true
  accuracy_comparison: true
  generate_comparison_report: true
  
# LLM-enhanced analysis
llm:
  provider: "openai"  # or "anthropic", "local"
  model: "gpt-4"
  tasks:
    - "framework_comparison"
    - "performance_analysis"
    - "recommendation_generation"
```

## Performance Optimization

### High-Throughput Configuration
```yaml
# config/high_throughput.yaml
version: "1.0"
name: "high_throughput_processing"

# Optimized for processing many models
pipeline:
  steps: [1, 2, 4, 5, 9]  # Minimal essential steps
  parallel_execution: true
  max_workers: 12
  batch_size: 20
  
# Performance optimizations
performance:
  precompile_templates: true
  cache_parsed_models: true
  lazy_loading: true
  memory_mapping: true
  
# Efficient I/O
io:
  async_io: true
  buffer_size: "128MB"
  batch_writes: true
  compression: "zstd"
  
# Memory management
memory:
  pool_size: "8GB"
  garbage_collection: "aggressive"
  memory_monitoring: true
  
# Caching strategy
caching:
  levels: ["parsing", "validation", "rendering"]
  eviction_policy: "lru"
  cache_size: "4GB"
```

### Memory-Optimized Configuration
```yaml
# config/memory_optimized.yaml
version: "1.0"
name: "memory_efficient"

# Memory-conscious pipeline
pipeline:
  steps: [1, 2, 4, 5, 9]
  sequential_execution: true  # Avoid memory overhead of parallelism
  
# Memory optimization
memory:
  limit: "2GB"
  streaming_processing: true
  lazy_evaluation: true
  memory_profiling: true
  
# Reduced memory usage
processing:
  small_batch_size: 1
  immediate_cleanup: true
  minimal_caching: true
  compress_intermediates: true
  
# Garbage collection
gc:
  frequency: "aggressive"
  threshold: "100MB"
  force_collection: true
```

### CPU-Optimized Configuration  
```yaml
# config/cpu_optimized.yaml
version: "1.0"
name: "cpu_optimized"

# CPU-intensive optimization
pipeline:
  steps: [1, 2, 3, 4, 5, 9, 10, 12, 13]  # Include computation-heavy steps
  parallel_execution: true
  
# CPU optimization
cpu:
  max_cores: 16
  affinity: "auto"
  numa_aware: true
  thread_pool_size: 32
  
# Computation settings
computation:
  numerical_precision: "float32"  # Faster than float64
  vectorization: true
  jit_compilation: true
  
# Algorithm choices
algorithms:
  matrix_backend: "numpy"  # or "jax", "torch"
  linear_algebra: "openblas"
  fft_backend: "fftw"
```

## Cloud and Distributed Configurations

### AWS Configuration
```yaml
# config/aws.yaml
version: "1.0"
name: "gnn_aws_deployment"

# AWS-specific settings
aws:
  region: "us-west-2"
  instance_type: "m5.4xlarge"
  storage_type: "gp3"
  
# S3 integration
storage:
  input_bucket: "gnn-models-input"
  output_bucket: "gnn-models-output"
  cache_bucket: "gnn-models-cache"
  
# CloudWatch monitoring
monitoring:
  cloudwatch: true
  custom_metrics: true
  log_groups: "/aws/gnn/pipeline"
  
# Auto-scaling
scaling:
  min_instances: 1
  max_instances: 10
  scale_metric: "cpu_utilization"
  scale_threshold: 70
  
# Security
security:
  iam_role: "GNNPipelineRole"
  vpc_id: "vpc-12345678"
  security_groups: ["sg-12345678"]
  encryption: true
```

### Kubernetes Configuration
```yaml
# config/kubernetes.yaml
version: "1.0"
name: "gnn_k8s"

# Kubernetes deployment
kubernetes:
  namespace: "gnn-pipeline"
  deployment_name: "gnn-processor"
  replicas: 3
  
# Resource requests and limits
resources:
  requests:
    cpu: "500m"
    memory: "2Gi"
  limits:
    cpu: "2000m"
    memory: "8Gi"
    
# Persistent storage
storage:
  volume_size: "100Gi"
  storage_class: "fast-ssd"
  access_mode: "ReadWriteMany"
  
# Service configuration
service:
  type: "ClusterIP"
  port: 8080
  
# Health checks
health:
  liveness_probe:
    path: "/health"
    initial_delay: 30
    period: 10
  readiness_probe:
    path: "/ready"
    initial_delay: 5
    period: 5
```

### Distributed Processing
```yaml
# config/distributed.yaml
version: "1.0"
name: "distributed_gnn"

# Distributed execution
distributed:
  backend: "ray"  # or "dask", "celery"
  cluster_address: "ray://head-node:10001"
  
# Worker configuration
workers:
  num_workers: 8
  worker_resources:
    cpu: 4
    memory: "8GB"
    
# Task distribution
tasks:
  distribution_strategy: "round_robin"
  load_balancing: true
  fault_tolerance: true
  retry_failed_tasks: 3
  
# Communication
communication:
  serialization: "cloudpickle"
  compression: "lz4"
  timeout: 300
```

## Specialized Workflows

### Research Publication Pipeline
```yaml
# config/publication.yaml
version: "1.0"
name: "research_publication"

# Complete publication pipeline
pipeline:
  steps: [1, 2, 3, 4, 5, 6, 9, 10, 11, 12]
  
# Publication outputs
outputs:
  figures: true
  tables: true
  supplementary_data: true
  reproducibility_package: true
  
# Documentation generation
documentation:
  api_docs: true
  model_descriptions: true
  usage_examples: true
  performance_benchmarks: true
  
# Reproducibility
reproducibility:
  version_control: true
  environment_capture: true
  data_provenance: true
  parameter_tracking: true
  
# Site generation for publication
site:
  generate: true
  template: "academic"
  include_code: true
  include_data: true
```

### Educational Configuration
```yaml
# config/educational.yaml
version: "1.0"
name: "gnn_education"

# Educational-focused pipeline
pipeline:
  steps: [1, 2, 3, 4, 5, 6, 11, 12]  # Include visualization and explanations
  
# Educational outputs
education:
  interactive_notebooks: true
  step_by_step_explanations: true
  visualization_emphasis: true
  simplified_outputs: true
  
# LLM for explanations
llm:
  provider: "openai"
  model: "gpt-4"
  tasks:
    - "explain_concepts"
    - "generate_examples" 
    - "create_tutorials"
    
# Beginner-friendly settings
beginner:
  verbose_logging: true
  error_explanations: true
  helpful_hints: true
  progress_indicators: true
```

### Continuous Integration Configuration
```yaml
# config/ci.yaml
version: "1.0"
name: "continuous_integration"

# CI/CD pipeline
ci:
  trigger: "on_commit"
  parallel_jobs: 4
  timeout: 1800
  
# Testing in CI
testing:
  unit_tests: true
  integration_tests: true
  performance_regression: true
  security_scanning: true
  
# Quality gates
quality:
  code_coverage: 85
  lint_checks: true
  type_checking: true
  documentation_coverage: 80
  
# Artifacts
artifacts:
  test_reports: true
  coverage_reports: true
  build_artifacts: true
  documentation: true
  
# Notifications
notifications:
  slack: true
  email: true
  github_status: true
```

## Troubleshooting Configurations

### Debug Configuration
```yaml
# config/debug.yaml
version: "1.0"
name: "debug_pipeline"

# Debug-focused settings
debugging:
  enable_all: true
  verbose_logging: true
  save_intermediates: true
  profile_memory: true
  profile_cpu: true
  
# Detailed error reporting
error_handling:
  full_stack_traces: true
  error_context: 10
  save_error_state: true
  interactive_debugging: true
  
# Step-by-step execution
execution:
  step_by_step: true
  pause_between_steps: true
  confirm_continuation: true
  
# Validation and checking
validation:
  extra_checks: true
  paranoid_mode: true
  double_check_results: true
```

### Performance Troubleshooting
```yaml
# config/performance_debug.yaml
version: "1.0"
name: "performance_troubleshooting"

# Performance monitoring
monitoring:
  detailed_profiling: true
  memory_tracking: true
  cpu_profiling: true
  io_monitoring: true
  
# Bottleneck identification
profiling:
  line_profiler: true
  memory_profiler: true
  cpu_profiler: true
  flame_graphs: true
  
# Resource usage
resources:
  track_peak_usage: true
  resource_warnings: true
  usage_alerts: true
  
# Optimization suggestions
optimization:
  suggest_improvements: true
  benchmark_alternatives: true
  identify_bottlenecks: true
```

### Error Analysis Configuration
```yaml
# config/error_analysis.yaml
version: "1.0"
name: "error_analysis"

# Comprehensive error tracking
error_tracking:
  collect_all_errors: true
  error_categorization: true
  error_frequency: true
  error_patterns: true
  
# Error context
context:
  input_data: true
  system_state: true
  environment_info: true
  stack_traces: true
  
# Recovery strategies
recovery:
  auto_retry: false  # Manual investigation preferred
  save_failure_state: true
  suggest_fixes: true
  
# Reporting
reporting:
  error_summary: true
  failure_analysis: true
  improvement_suggestions: true
```

## Configuration Best Practices

### Environment-Specific Configurations
```yaml
# Use environment variables for sensitive data
database:
  host: "${DB_HOST:-localhost}"
  password: "${DB_PASSWORD}"
  
# Environment-specific overrides
development:
  debug: "${DEBUG:-true}"
  
production:
  debug: "${DEBUG:-false}"
  monitoring: "${MONITORING:-true}"
```

### Modular Configuration
```yaml
# config/base.yaml
version: "1.0"
name: "base_config"

# Base configuration that others extend
pipeline:
  timeout_seconds: 300
  
logging:
  level: INFO

---
# config/development.yaml extends base.yaml
extends: "base.yaml"

# Override specific settings
logging:
  level: DEBUG
  
development:
  hot_reload: true
```

### Validation Schema
```yaml
# Configuration validation
$schema: "gnn-config-schema-v1.json"

# Required fields
required: ["version", "name", "pipeline"]

# Type validation
pipeline:
  type: "object"
  required: ["steps"]
  properties:
    steps:
      type: "array"
      items:
        type: "integer"
        minimum: 1
        maximum: 13
```

This comprehensive collection of configuration examples provides templates for virtually any GNN deployment scenario, from simple development setups to complex distributed production environments. 