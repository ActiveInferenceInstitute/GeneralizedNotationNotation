# Pipeline Module - Agent Scaffolding

## Module Overview

**Purpose**: Pipeline configuration management, step coordination, and execution orchestration

**Category**: Core Infrastructure / Pipeline Management

---

## Core Functionality

### Primary Responsibilities
1. Centralized pipeline configuration
2. Step-specific output directory resolution
3. Pipeline-wide settings management
4. Step dependency tracking
5. Execution flow coordination

### Key Capabilities
- Dynamic output directory mapping
- Configuration loading and validation
- Step metadata management
- Pipeline state tracking
- Cross-step data flow management

---

## Module Components

### Configuration Management
- `config.py` - Core configuration functions
- `pipeline_config.yaml` - Configuration file

### Pipeline Orchestration
- `pipeline_validation.py` - Validation utilities
- `pipeline_step_template.py` - Template for new steps

### Step Management
- Output directory mapping
- Step status tracking
- Dependency resolution

---

## API Reference

### Core Functions

#### `get_output_dir_for_script(script_name: str, base_output_dir: Path = None) -> Path`
**Description**: Get standardized output directory for a pipeline script

**Parameters**:
- `script_name` (str): Script name (e.g., "3_gnn.py", "gnn")
- `base_output_dir` (Path): Base output directory (default: project root "output/")

**Returns**: Path to step-specific output directory

**Example**:
```python
from pipeline.config import get_output_dir_for_script

output_dir = get_output_dir_for_script("3_gnn.py")
# Returns: Path("output/3_gnn_output")
```

#### `get_pipeline_config(config_path: Path = None) -> Dict`
**Description**: Load pipeline configuration from YAML

**Parameters**:
- `config_path` (Path): Path to config file (default: "input/config.yaml")

**Returns**: Configuration dictionary

---

## Output Directory Mapping

### Script to Directory Mapping
```python
SCRIPT_OUTPUT_MAPPING = {
    "0_template.py": "0_template_output",
    "1_setup.py": "1_setup_output",
    "2_tests.py": "2_tests_output",
    "3_gnn.py": "3_gnn_output",
    "4_model_registry.py": "4_model_registry_output",
    "5_type_checker.py": "5_type_checker_output",
    "6_validation.py": "6_validation_output",
    "7_export.py": "7_export_output",
    "8_visualization.py": "8_visualization_output",
    "9_advanced_viz.py": "9_advanced_viz_output",
    "10_ontology.py": "10_ontology_output",
    "11_render.py": "11_render_output",
    "12_execute.py": "12_execute_output",
    "13_llm.py": "13_llm_output",
    "14_ml_integration.py": "14_ml_integration_output",
    "15_audio.py": "15_audio_output",
    "16_analysis.py": "16_analysis_output",
    "17_integration.py": "17_integration_output",
    "18_security.py": "18_security_output",
    "19_research.py": "19_research_output",
    "20_website.py": "20_website_output",
    "21_mcp.py": "21_mcp_output",
    "22_gui.py": "22_gui_output",
    "23_report.py": "23_report_output"
}
```

---

## Pipeline Configuration

### Configuration Structure
```yaml
pipeline:
  target_dir: "input/gnn_files"
  output_dir: "output"
  log_level: "INFO"
  
steps:
  3_gnn:
    recursive: true
    enable_round_trip: true
    enable_cross_format: true
    
  5_type_checker:
    strict: false
    estimate_resources: true
```

---

## Usage Examples

### Get Output Directory
```python
from pipeline.config import get_output_dir_for_script

# Using script name
gnn_output = get_output_dir_for_script("3_gnn.py")

# Using module name
gnn_output = get_output_dir_for_script("gnn")

# Custom base directory
output = get_output_dir_for_script("3_gnn.py", Path("/custom/output"))
```

### Load Pipeline Configuration
```python
from pipeline.config import get_pipeline_config

config = get_pipeline_config()
target_dir = config.get("pipeline", {}).get("target_dir")
```

### Access Previous Step Output
```python
from pipeline.config import get_output_dir_for_script

# Step 5 accessing Step 3 output
gnn_output_dir = get_output_dir_for_script("3_gnn.py")
results_file = gnn_output_dir / "gnn_processing_results.json"
```

---

## Pipeline Validation

### Validation Checks
1. All step output directories exist or can be created
2. Configuration file is valid YAML
3. Required dependencies are available
4. Step dependencies are satisfied
5. Input directories exist

---

## Dependencies

### Required Dependencies
- `pathlib` - Path operations
- `yaml` - Configuration parsing
- `json` - Result serialization

### Internal Dependencies
- None (core infrastructure)

---

## Recent Improvements

### Output Directory Normalization ✅
**Issue**: Inconsistent handling of nested output directories (e.g., `3_gnn_output/3_gnn_output/`)

**Fix**: Enhanced `get_output_dir_for_script` with:
- Intelligent path resolution
- Fallback checking for nested directories
- Consistent return values across all steps

**Impact**: Steps 5, 6, 7, 8 now correctly locate GNN processing results

---

## Testing

### Test Files
- `src/tests/test_pipeline_integration.py`
- `src/tests/test_pipeline_validation.py`

### Test Coverage
- **Current**: 85%
- **Target**: 90%+

---

**Last Updated**: September 29, 2025  
**Status**: ✅ Production Ready - Core Infrastructure

