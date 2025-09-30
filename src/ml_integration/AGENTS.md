# ML Integration Module - Agent Scaffolding

## Module Overview

**Purpose**: Machine learning integration and model training capabilities for GNN models with multiple framework support

**Pipeline Step**: Step 14: ML integration (14_ml_integration.py)

**Category**: Machine Learning / Model Training

---

## Core Functionality

### Primary Responsibilities
1. Integrate machine learning frameworks with GNN models
2. Provide model training and validation capabilities
3. Support multiple ML frameworks (PyTorch, TensorFlow, JAX)
4. Enable model optimization and hyperparameter tuning
5. Generate ML-ready datasets from GNN specifications

### Key Capabilities
- Multi-framework ML model generation
- Training pipeline automation
- Model validation and performance evaluation
- Dataset preparation and preprocessing
- Hyperparameter optimization
- Model serialization and deployment

---

## API Reference

### Public Functions

#### `process_ml_integration(target_dir, output_dir, logger, **kwargs) -> bool`
**Description**: Main ML integration processing function for GNN models

**Parameters**:
- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for ML integration results
- `logger` (Logger): Logger instance for progress reporting
- `model_type` (str): ML model type ("auto", "supervised", "unsupervised")
- `training_mode` (str): Training mode ("train", "evaluate", "predict")
- `**kwargs`: Additional ML-specific options

**Returns**: `True` if ML integration processing succeeded

---

## ML Framework Support

### PyTorch Integration
**Status**: ✅ Supported
**Features**:
- Neural network model generation from GNN specifications
- Custom loss functions for Active Inference models
- Training loop automation with early stopping
- Model checkpointing and serialization

### TensorFlow/Keras Integration
**Status**: ✅ Supported
**Features**:
- TensorFlow model generation and training
- Custom layers for cognitive modeling
- TensorBoard integration for visualization
- Model export for deployment

### JAX/Flax Integration
**Status**: ✅ Supported
**Features**:
- JAX-based model implementation
- Functional programming approach for cognitive models
- JIT compilation for performance
- Flax neural network library integration

### Scikit-learn Integration
**Status**: ✅ Supported
**Features**:
- Traditional ML model generation
- Feature engineering and preprocessing
- Model evaluation and comparison
- Integration with cognitive modeling workflows

---

## Dependencies

### Required Dependencies
- `numpy` - Numerical computations for ML models
- `pandas` - Data manipulation for datasets
- `scikit-learn` - Traditional ML algorithms

### Optional Dependencies
- `torch` - PyTorch deep learning (fallback: simplified models)
- `tensorflow` - TensorFlow/Keras (fallback: scikit-learn)
- `jax` - JAX high-performance computing (fallback: numpy)
- `flax` - JAX neural networks (fallback: basic implementations)
- `optax` - JAX optimization (fallback: basic optimizers)

### Internal Dependencies
- `utils.pipeline_template` - Standardized pipeline processing
- `pipeline.config` - Configuration management

---

## Configuration

### Environment Variables
- `ML_FRAMEWORK` - Preferred ML framework ("torch", "tensorflow", "jax", "sklearn")
- `ML_MODEL_TYPE` - Model type ("classification", "regression", "autoencoder")
- `ML_TRAINING_EPOCHS` - Default training epochs (default: 100)
- `ML_BATCH_SIZE` - Default batch size (default: 32)

### Configuration Files
- `ml_config.yaml` - ML framework settings and hyperparameters

### Default Settings
```python
DEFAULT_ML_SETTINGS = {
    'framework': 'auto',
    'model_type': 'auto',
    'training': {
        'epochs': 100,
        'batch_size': 32,
        'validation_split': 0.2,
        'early_stopping': True,
        'patience': 10
    },
    'optimization': {
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'loss_function': 'auto'
    }
}
```

---

## Usage Examples

### Basic ML Integration
```python
from ml_integration.processor import process_ml_integration

success = process_ml_integration(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/14_ml_integration_output"),
    logger=logger,
    model_type="auto"
)
```

### Framework-Specific Training
```python
from ml_integration.processor import process_ml_integration

success = process_ml_integration(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/14_ml_integration_output"),
    logger=logger,
    model_type="supervised",
    training_mode="train",
    ml_framework="torch"
)
```

---

## Output Specification

### Output Products
- `{model}_ml_model.pkl` - Trained ML model
- `{model}_training_history.json` - Training metrics and history
- `{model}_model_evaluation.json` - Model performance evaluation
- `{model}_dataset_preparation.json` - Dataset preprocessing information
- `ml_integration_summary.json` - Processing summary

### Output Directory Structure
```
output/14_ml_integration_output/
├── model_name_ml_model.pkl
├── model_name_training_history.json
├── model_name_model_evaluation.json
├── model_name_dataset_preparation.json
└── ml_integration_summary.json
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: ~30-120 seconds (depending on model complexity)
- **Memory**: ~100-500MB for training
- **Status**: ✅ Production Ready

### Expected Performance
- **Fast Path**: ~10-30s for simple models
- **Slow Path**: ~2-5min for complex neural networks
- **Memory**: ~50-200MB for typical models, ~500MB+ for large models

---

## Error Handling

### Graceful Degradation
- **No ML frameworks**: Fallback to statistical analysis only
- **Training failures**: Generate model evaluation report
- **Memory issues**: Reduce batch size and complexity

### Error Categories
1. **Framework Errors**: Missing or incompatible ML frameworks
2. **Training Errors**: Model training failures or convergence issues
3. **Data Errors**: Invalid or insufficient training data
4. **Resource Errors**: Memory or computational resource exhaustion

---

## Integration Points

### Orchestrated By
- **Script**: `14_ml_integration.py` (Step 14)
- **Function**: `process_ml_integration()`

### Imports From
- `utils.pipeline_template` - Standardized processing patterns
- `pipeline.config` - Configuration management

### Imported By
- `tests.test_ml_integration_unit.py` - ML integration tests
- `main.py` - Pipeline orchestration

### Data Flow
```
GNN Models → ML Framework Selection → Dataset Preparation → Model Training → Evaluation → Deployment
```

---

## Testing

### Test Files
- `src/tests/test_ml_integration_unit.py` - Unit tests
- `src/tests/test_ml_integration_frameworks.py` - Framework tests

### Test Coverage
- **Current**: 72%
- **Target**: 85%+

### Key Test Scenarios
1. Framework detection and selection
2. Model generation and training across frameworks
3. Dataset preparation and validation
4. Performance evaluation and comparison
5. Error handling with missing frameworks

---

## MCP Integration

### Tools Registered
- `ml_train` - Train ML models from GNN specifications
- `ml_evaluate` - Evaluate trained ML models
- `ml_framework_detect` - Detect available ML frameworks

### Tool Endpoints
```python
@mcp_tool("ml_train")
def train_ml_model(gnn_file, framework="auto", model_type="auto"):
    """Train ML model from GNN specification"""
    # Implementation
```

---

**Last Updated**: September 30, 2025
**Status**: ✅ Production Ready
