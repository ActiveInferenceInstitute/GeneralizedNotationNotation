# Pipeline Module Specification

## Overview
Pipeline orchestration, configuration, and execution utilities.

## Components

### Configuration
- `config.py` - Pipeline configuration management
- `config_schema.py` - Configuration schema definitions

### Execution
- `executor.py` - Step execution engine
- `pipeline_step_template.py` - Template for new steps

### Utilities
- `output_utils.py` - Output directory management
- `validation.py` - Pipeline validation

## Key Exports
```python
from pipeline import execute_pipeline_step, get_output_dir_for_script
from pipeline.config import PipelineConfig, get_step_config
```

## Step Naming Convention
Steps follow `N_name.py` pattern (0-23)
