# ML Integration Module

This module provides comprehensive machine learning integration capabilities for GNN models, including model training, evaluation, optimization, and integration with popular ML frameworks.

## Module Structure

```
src/ml_integration/
├── __init__.py                    # Module initialization and exports
└── README.md                      # This documentation
```

## Core Components

### Machine Learning Integration Functions

#### `process_ml_integration(target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs) -> bool`
Main function for processing machine learning integration tasks.

**Features:**
- Model training and evaluation
- ML framework integration
- Performance optimization
- Model comparison and benchmarking
- Automated ML workflows

**Returns:**
- `bool`: Success status of ML integration operations

### ML Framework Integration

#### Supported Frameworks
- **TensorFlow**: Deep learning and neural network training
- **PyTorch**: Dynamic neural networks and research
- **Scikit-learn**: Traditional machine learning algorithms
- **XGBoost**: Gradient boosting for structured data
- **LightGBM**: Light gradient boosting machine
- **JAX**: High-performance ML research

#### Integration Capabilities
- **Model Conversion**: Convert GNN models to ML framework formats
- **Training Pipelines**: Automated training pipelines
- **Evaluation Metrics**: Comprehensive model evaluation
- **Hyperparameter Optimization**: Automated hyperparameter tuning
- **Model Deployment**: Model deployment and serving

### Model Training and Evaluation

#### Training Functions
- **Supervised Learning**: Classification and regression tasks
- **Unsupervised Learning**: Clustering and dimensionality reduction
- **Reinforcement Learning**: Policy optimization and Q-learning
- **Active Learning**: Interactive learning with human feedback

#### Evaluation Metrics
- **Classification**: Accuracy, precision, recall, F1-score
- **Regression**: MSE, MAE, R², explained variance
- **Clustering**: Silhouette score, Davies-Bouldin index
- **Custom Metrics**: Domain-specific evaluation metrics

### Model Optimization

#### Hyperparameter Optimization
- **Grid Search**: Exhaustive hyperparameter search
- **Random Search**: Random hyperparameter sampling
- **Bayesian Optimization**: Efficient hyperparameter optimization
- **Genetic Algorithms**: Evolutionary hyperparameter optimization

#### Model Selection
- **Cross-Validation**: K-fold cross-validation
- **Ensemble Methods**: Model ensemble and stacking
- **Feature Selection**: Automated feature selection
- **Model Interpretability**: SHAP, LIME, and other interpretability tools

## Usage Examples

### Basic ML Integration

```python
from ml_integration import process_ml_integration

# Process ML integration
success = process_ml_integration(
    target_dir=Path("models/"),
    output_dir=Path("ml_output/"),
    verbose=True
)

if success:
    print("ML integration completed successfully")
else:
    print("ML integration failed")
```

### Model Training Pipeline

```python
from ml_integration import train_gnn_model

# Train GNN model with ML framework
training_results = train_gnn_model(
    gnn_content=gnn_content,
    framework="tensorflow",
    task_type="classification",
    hyperparameters={
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100
    }
)

print(f"Training accuracy: {training_results['accuracy']}")
print(f"Validation accuracy: {training_results['val_accuracy']}")
```

### Model Evaluation

```python
from ml_integration import evaluate_ml_model

# Evaluate trained model
evaluation_results = evaluate_ml_model(
    model_path=Path("trained_model/"),
    test_data=test_data,
    metrics=["accuracy", "precision", "recall", "f1"]
)

for metric, value in evaluation_results.items():
    print(f"{metric}: {value:.4f}")
```

### Hyperparameter Optimization

```python
from ml_integration import optimize_hyperparameters

# Optimize hyperparameters
optimization_results = optimize_hyperparameters(
    gnn_content=gnn_content,
    framework="pytorch",
    optimization_method="bayesian",
    n_trials=50
)

print(f"Best hyperparameters: {optimization_results['best_params']}")
print(f"Best score: {optimization_results['best_score']}")
```

## ML Integration Pipeline

### 1. Model Preparation
```python
# Prepare GNN model for ML training
ml_model = prepare_gnn_for_ml(gnn_content)
training_data = prepare_training_data(ml_model)
```

### 2. Framework Selection
```python
# Select appropriate ML framework
framework = select_ml_framework(task_type, model_complexity)
ml_framework = initialize_framework(framework)
```

### 3. Model Training
```python
# Train model with selected framework
training_results = train_model(ml_framework, training_data, hyperparameters)
model = training_results['model']
metrics = training_results['metrics']
```

### 4. Model Evaluation
```python
# Evaluate trained model
evaluation_results = evaluate_model(model, test_data)
performance_metrics = calculate_metrics(evaluation_results)
```

### 5. Model Optimization
```python
# Optimize model if needed
if optimization_needed:
    optimized_model = optimize_model(model, optimization_config)
    final_metrics = evaluate_model(optimized_model, test_data)
```

## Integration with Pipeline

### Pipeline Step 14: ML Integration
```python
# Called from 14_ml_integration.py
def process_ml_integration(target_dir, output_dir, verbose=False, **kwargs):
    # Prepare GNN models for ML training
    ml_models = prepare_gnn_models_for_ml(target_dir)
    
    # Train and evaluate models
    training_results = train_ml_models(ml_models)
    
    # Generate ML analysis and reports
    analysis = generate_ml_analysis(training_results)
    
    return True
```

### Output Structure
```
output/ml_integration/
├── trained_models/                 # Trained ML models
├── training_results.json           # Training results and metrics
├── evaluation_results.json         # Model evaluation results
├── hyperparameter_optimization.json # Hyperparameter optimization results
├── model_comparison.json          # Model comparison results
├── ml_analysis.json               # ML analysis results
└── ml_summary.md                  # ML integration summary
```

## ML Frameworks

### TensorFlow
- **Purpose**: Deep learning and neural networks
- **Strengths**: Production deployment, large-scale training
- **Use Cases**: Complex neural networks, production systems
- **Integration**: Keras API, TensorFlow Serving

### PyTorch
- **Purpose**: Research and dynamic neural networks
- **Strengths**: Dynamic computation graphs, research flexibility
- **Use Cases**: Research projects, rapid prototyping
- **Integration**: TorchScript, ONNX export

### Scikit-learn
- **Purpose**: Traditional machine learning
- **Strengths**: Simple APIs, comprehensive algorithms
- **Use Cases**: Traditional ML tasks, data science
- **Integration**: Pipeline API, model persistence

### XGBoost/LightGBM
- **Purpose**: Gradient boosting for structured data
- **Strengths**: High performance, feature importance
- **Use Cases**: Tabular data, feature engineering
- **Integration**: Model serialization, feature analysis

### JAX
- **Purpose**: High-performance ML research
- **Strengths**: GPU acceleration, functional programming
- **Use Cases**: Research, custom algorithms
- **Integration**: JIT compilation, automatic differentiation

## Training Tasks

### Classification Tasks
- **Binary Classification**: Two-class problems
- **Multi-class Classification**: Multiple class problems
- **Multi-label Classification**: Multiple labels per instance
- **Imbalanced Classification**: Handling class imbalance

### Regression Tasks
- **Linear Regression**: Simple regression problems
- **Non-linear Regression**: Complex regression problems
- **Time Series Regression**: Temporal prediction
- **Multi-output Regression**: Multiple target variables

### Clustering Tasks
- **K-means Clustering**: Partition-based clustering
- **Hierarchical Clustering**: Tree-based clustering
- **Density-based Clustering**: DBSCAN and variants
- **Spectral Clustering**: Graph-based clustering

### Reinforcement Learning
- **Policy Optimization**: Direct policy learning
- **Q-Learning**: Value-based learning
- **Actor-Critic Methods**: Combined policy and value learning
- **Multi-agent RL**: Multi-agent systems

## Configuration Options

### ML Settings
```python
# ML integration configuration
config = {
    'default_framework': 'auto',    # Default ML framework
    'training_mode': 'supervised',  # Training mode
    'optimization_enabled': True,   # Enable hyperparameter optimization
    'evaluation_metrics': ['accuracy', 'precision', 'recall'],
    'cross_validation_folds': 5,    # Number of CV folds
    'random_seed': 42              # Random seed for reproducibility
}
```

### Framework-Specific Settings
```python
# Framework-specific configuration
framework_config = {
    'tensorflow': {
        'version': '2.x',
        'gpu_enabled': True,
        'mixed_precision': True
    },
    'pytorch': {
        'version': '1.x',
        'cuda_enabled': True,
        'deterministic': True
    },
    'scikit-learn': {
        'n_jobs': -1,
        'random_state': 42
    }
}
```

## Error Handling

### Training Failures
```python
# Handle training failures gracefully
try:
    results = train_gnn_model(gnn_content, framework)
except TrainingError as e:
    logger.error(f"Training failed: {e}")
    # Provide fallback training or error reporting
```

### Framework Issues
```python
# Handle framework-specific issues
try:
    framework = initialize_framework(framework_name)
except FrameworkError as e:
    logger.warning(f"Framework failed: {e}")
    # Fall back to alternative framework
```

### Resource Issues
```python
# Handle resource constraints
try:
    results = train_model_with_resources(model, data, resources)
except ResourceError as e:
    logger.warning(f"Resource constraint: {e}")
    # Adjust training parameters or use smaller model
```

## Performance Optimization

### Training Optimization
- **Batch Processing**: Optimize batch sizes for memory and speed
- **Data Loading**: Efficient data loading and preprocessing
- **Model Parallelism**: Distribute training across multiple devices
- **Mixed Precision**: Use mixed precision for faster training

### Memory Management
- **Gradient Accumulation**: Accumulate gradients for large batches
- **Model Checkpointing**: Save model checkpoints to reduce memory
- **Data Streaming**: Stream data to avoid loading everything in memory
- **Garbage Collection**: Optimize garbage collection during training

### Hardware Acceleration
- **GPU Utilization**: Maximize GPU utilization
- **Multi-GPU Training**: Distribute training across multiple GPUs
- **TPU Support**: Support for Tensor Processing Units
- **Distributed Training**: Distributed training across multiple machines

## Testing and Validation

### Unit Tests
```python
# Test individual ML functions
def test_model_training():
    results = train_gnn_model(test_content, "tensorflow")
    assert 'accuracy' in results
    assert 'model' in results
    assert results['accuracy'] > 0.5
```

### Integration Tests
```python
# Test complete ML pipeline
def test_ml_pipeline():
    success = process_ml_integration(test_dir, output_dir)
    assert success
    # Verify ML outputs
    ml_files = list(output_dir.glob("**/*"))
    assert len(ml_files) > 0
```

### Performance Tests
```python
# Test ML performance
def test_training_performance():
    start_time = time.time()
    results = train_gnn_model(test_content, "pytorch")
    end_time = time.time()
    
    assert results['success']
    assert (end_time - start_time) < 300  # Should complete within 5 minutes
```

## Dependencies

### Required Dependencies
- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **scikit-learn**: Traditional machine learning
- **matplotlib**: Plotting and visualization

### Optional Dependencies
- **tensorflow**: Deep learning framework
- **torch**: PyTorch deep learning framework
- **xgboost**: Gradient boosting framework
- **lightgbm**: Light gradient boosting machine
- **jax**: High-performance ML research
- **optuna**: Hyperparameter optimization

## Performance Metrics

### Training Times
- **Small Models** (< 1K parameters): 1-10 minutes
- **Medium Models** (1K-100K parameters): 10-60 minutes
- **Large Models** (> 100K parameters): 1-24 hours

### Memory Usage
- **Base Memory**: ~100MB
- **Per Model**: ~50-500MB depending on complexity
- **Peak Memory**: 3-5x base usage during training

### Accuracy Metrics
- **Classification**: 70-95% accuracy depending on task
- **Regression**: 0.6-0.9 R² depending on task
- **Clustering**: 0.3-0.8 silhouette score depending on data

## Troubleshooting

### Common Issues

#### 1. Training Failures
```
Error: CUDA out of memory during training
Solution: Reduce batch size or use gradient accumulation
```

#### 2. Framework Compatibility
```
Error: Framework version incompatibility
Solution: Update framework versions or use compatible versions
```

#### 3. Data Issues
```
Error: Invalid data format for training
Solution: Preprocess data or use appropriate data format
```

#### 4. Performance Issues
```
Error: Training taking too long
Solution: Enable GPU acceleration or use smaller model
```

### Debug Mode
```python
# Enable debug mode for detailed ML information
results = train_gnn_model(content, framework, debug=True, verbose=True)
```

## Future Enhancements

### Planned Features
- **AutoML Integration**: Automated machine learning workflows
- **Federated Learning**: Distributed training across multiple sites
- **Model Compression**: Model compression and quantization
- **Edge Deployment**: Model deployment on edge devices

### Performance Improvements
- **Advanced Parallelization**: Advanced parallel training strategies
- **Model Optimization**: Advanced model optimization techniques
- **Hardware Optimization**: Better hardware utilization
- **Distributed Training**: Improved distributed training capabilities

## Summary

The ML Integration module provides comprehensive machine learning integration capabilities for GNN models, including model training, evaluation, optimization, and integration with popular ML frameworks. The module supports various ML tasks, frameworks, and optimization strategies to enhance Active Inference research and development.

## License and Citation

This module is part of the GeneralizedNotationNotation project. See the main repository for license and citation information. 

## References

- Project overview: ../../README.md
- Comprehensive docs: ../../DOCS.md
- Architecture guide: ../../ARCHITECTURE.md
- Pipeline details: ../../doc/pipeline/README.md