# ActiveInference.jl Analysis Documentation

## Analysis Framework

This documentation describes the analysis modules available in the ActiveInference.jl implementation.

## Overview

Your ActiveInference.jl implementation includes analysis capabilities for:

1. **Meta-cognitive processes** - How agents think about their own thinking
2. **Adaptive precision mechanisms** - Dynamic attention and resource allocation
3. **Counterfactual reasoning** - What-if analysis and alternative scenarios
4. **Multi-scale temporal understanding** - Hierarchical time reasoning
5. **Advanced uncertainty quantification** - Epistemic vs aleatoric decomposition

## Analysis Modules

### 🧠 Meta-Cognitive Analysis Module

**File:** `meta_cognitive_analysis.jl`

**Purpose:** Provides analysis of higher-order cognitive processes including hierarchical reasoning, meta-awareness, and theory of mind capabilities.

**Key Features:**
- **Meta-cognitive awareness scoring** based on confidence-accuracy correlation
- **Higher-order belief analysis** including belief volatility and coherence
- **Hierarchical temporal abstraction** across multiple time scales
- **Theory of mind modeling** for multi-agent scenarios
- **Meta-learning analysis** examining learning about learning processes

**Key Functions:**
```julia
# Calculate meta-cognitive awareness metrics
calculate_metacognitive_awareness(beliefs_trace, observations, confidence_trace)

# Analyze higher-order beliefs
analyze_higher_order_beliefs(beliefs_trace, belief_change_trace)

# Multi-scale temporal analysis
multi_scale_temporal_analysis(actions, beliefs_trace, time_scales)

# Meta-learning analysis
analyze_meta_learning(parameter_traces, performance_trace)

# Theory of mind analysis
analyze_theory_of_mind(agent_beliefs, other_agent_actions, predicted_other_actions)
```

**Outputs:**
- Metacognitive sensitivity and calibration analysis
- Belief volatility and coherence metrics
- Learning phase characterization
- Social learning indicators

### 🎯 Adaptive Precision and Attention Module

**File:** `adaptive_precision_attention.jl`

**Purpose:** Analyzes dynamic precision modulation and attention allocation mechanisms for optimal resource distribution.

**Key Features:**
- **Dynamic precision calculation** based on uncertainty and context
- **Attention weight computation** using salience and relevance
- **Multi-modal attention coordination** across different sensory modalities
- **Cognitive load monitoring** and adaptive capacity management
- **Precision parameter learning** through gradient-based optimization

**Key Functions:**
```julia
# Calculate adaptive precision parameters
calculate_adaptive_precision(beliefs_trace, prediction_errors, context_factors)

# Learn precision parameters over time
learn_precision_parameters(observations, beliefs_trace, actions)

# Calculate attention weights
calculate_attention_weights(stimuli_features, current_goals, surprise_values)

# Dynamic attention allocation
dynamic_attention_allocation(beliefs_trace, observations, actions)

# Multi-modal coordination
multimodal_attention_coordination(modality_beliefs, modality_precisions, cross_modal_weights)
```

**Outputs:**
- Adaptive precision traces and statistics
- Attention allocation patterns
- Multi-modal coordination metrics
- Cognitive load assessments

### 🔀 Counterfactual Reasoning Module

**File:** `counterfactual_reasoning.jl`

**Purpose:** Enables what-if analysis and alternative scenario exploration for understanding decision consequences.

**Key Features:**
- **Alternative scenario generation** for different action sequences
- **Counterfactual outcome simulation** using modified belief evolution
- **Regret and relief quantification** based on outcome comparisons
- **Causal impact analysis** measuring intervention effects
- **Belief counterfactuals** examining alternative evidence scenarios

**Key Functions:**
```julia
# Generate counterfactual action sequences
generate_counterfactual_actions(original_actions, n_possible_actions, intervention_points)

# Simulate counterfactual outcomes
simulate_counterfactual_outcomes(counterfactual_scenarios, A_matrix, B_matrix, initial_beliefs, observations)

# Calculate regret and relief
calculate_regret_relief(original_outcome, counterfactual_outcomes)

# Analyze causal impact
analyze_causal_impact(original_beliefs, intervention_outcomes)

# Belief counterfactuals
analyze_belief_counterfactuals(original_beliefs, original_observations, A_matrix)
```

**Outputs:**
- Alternative scenario outcomes
- Regret/relief quantification
- Causal impact measurements
- Critical evidence point identification

### ⏰ Multi-Scale Temporal Analysis Module

**File:** `multi_scale_temporal_analysis.jl`

**Purpose:** Provides hierarchical temporal reasoning analysis across different time horizons and planning depths.

**Key Features:**
- **Hierarchical temporal chunking** at multiple scales
- **Planning depth optimization** for different horizons
- **Temporal coherence analysis** across time windows
- **Cross-scale relationship analysis** examining scale dependencies
- **Predictability assessment** at different temporal resolutions

**Key Functions:**
```julia
# Analyze temporal hierarchies
analyze_temporal_hierarchies(beliefs_trace, actions, observations, temporal_scales)

# Planning depth analysis
analyze_planning_depth(beliefs_trace, actions, planning_horizons)

# Temporal coherence analysis
analyze_temporal_coherence(beliefs_trace, actions, observations, coherence_windows)

# Calculate belief smoothness
calculate_belief_smoothness(beliefs_window)

# Action consistency analysis
calculate_action_consistency(actions_window)
```

**Outputs:**
- Temporal hierarchy metrics
- Optimal planning horizons
- Coherence measurements
- Cross-scale dependencies

### 📊 Advanced Uncertainty Quantification Module

**File:** `uncertainty_quantification.jl`

**Purpose:** Provides comprehensive uncertainty analysis including epistemic vs aleatoric decomposition.

**Key Features:**
- **Uncertainty decomposition** separating knowledge from environmental uncertainty
- **Model uncertainty quantification** through ensemble methods
- **Information-theoretic measures** including entropy and mutual information
- **Uncertainty-aware decision analysis** examining risk preferences
- **Bayesian confidence intervals** with calibration assessment

**Key Functions:**
```julia
# Decompose uncertainty types
decompose_uncertainty(beliefs_trace, observations, n_bootstrap_samples)

# Quantify model uncertainty
quantify_model_uncertainty(beliefs_trace, actions, observations, n_ensemble_models)

# Information-theoretic uncertainty
calculate_information_uncertainty(beliefs_trace, observations, actions)

# Uncertainty-aware decisions
analyze_uncertainty_aware_decisions(beliefs_trace, actions, uncertainty_estimates)

# Mutual information calculation
mutual_information(joint_dist)
```

**Outputs:**
- Epistemic vs aleatoric uncertainty ratios
- Model disagreement metrics
- Information gain measurements
- Risk preference characterization

## Integration Suite

**File:** `integration_suite.jl`

**Purpose:** Orchestrates all analysis modules in a pipeline.

**Usage:**
```bash
julia integration_suite.jl /path/to/output/directory
```

**Integration Features:**
- **Phased execution** with analysis modules
- **Error handling** and progress tracking
- **Reporting** with HTML and Markdown summaries
- **Cross-module integration**
- **Multi-format export** for research integration

## Usage Examples

### Running Individual Modules

```julia
# Meta-cognitive analysis
include("meta_cognitive_analysis.jl")
results = comprehensive_metacognitive_analysis("/path/to/output")

# Adaptive precision analysis
include("adaptive_precision_attention.jl")
results = comprehensive_precision_attention_analysis("/path/to/output")

# Counterfactual analysis
include("counterfactual_reasoning.jl")
results = comprehensive_counterfactual_analysis("/path/to/output")

# Multi-scale temporal analysis
include("multi_scale_temporal_analysis.jl")
results = comprehensive_temporal_analysis("/path/to/output")

# Uncertainty quantification
include("uncertainty_quantification.jl")
results = comprehensive_uncertainty_analysis("/path/to/output")
```

### Running Complete Suite

```julia
include("integration_suite.jl")
analysis_log = run_integration_suite("/path/to/output")
```

### Working with Analysis Results

```julia
# Load results from JSON
using JSON

# Meta-cognitive results
metacog_results = JSON.parsefile("output/metacognitive_analysis/metacognitive_analysis_results.json")
metacog_awareness = metacog_results["metacognitive_awareness"]
sensitivity = metacog_awareness["metacognitive_sensitivity"]

# Precision results
precision_results = JSON.parsefile("output/adaptive_precision_attention/precision_attention_results.json")
adaptive_precision = precision_results["adaptive_precision"]

# Counterfactual results
cf_results = JSON.parsefile("output/counterfactual_reasoning/counterfactual_analysis_results.json")
regret_relief = cf_results["regret_relief"]

# Temporal results
temporal_results = JSON.parsefile("output/multi_scale_temporal/temporal_analysis_results.json")
hierarchies = temporal_results["temporal_hierarchies"]

# Uncertainty results
uncertainty_results = JSON.parsefile("output/uncertainty_quantification/uncertainty_analysis_results.json")
decomposition = uncertainty_results["uncertainty_decomposition"]
```

## Key Insights and Metrics

### Meta-Cognitive Insights
- **Metacognitive Sensitivity:** Correlation between confidence and accuracy
- **Calibration Error:** How well confidence matches actual accuracy
- **Meta-Uncertainty:** Uncertainty about uncertainty estimates
- **Higher-Order Belief Coherence:** Consistency of beliefs about beliefs

### Precision and Attention Insights
- **Adaptive Precision Traces:** Dynamic precision modulation over time
- **Attention Allocation Patterns:** Where cognitive resources are focused
- **Multi-Modal Coordination:** How different sensory modalities are integrated
- **Cognitive Load Evolution:** Mental effort requirements over time

### Counterfactual Insights
- **Regret Quantification:** How much better alternative actions could have been
- **Relief Assessment:** How much worse things could have gone
- **Causal Impact Measurement:** Effect size of specific interventions
- **Critical Evidence Points:** Observations that most changed beliefs

### Temporal Insights
- **Optimal Planning Horizons:** Best time scales for prediction and planning
- **Temporal Coherence:** Consistency of reasoning across time windows
- **Cross-Scale Dependencies:** How different time scales relate to each other
- **Hierarchical Chunking:** Natural temporal segmentation patterns

### Uncertainty Insights
- **Epistemic vs Aleatoric Ratios:** Knowledge vs environmental uncertainty
- **Model Uncertainty:** Disagreement between different model parameterizations
- **Information Gain Efficiency:** How effectively observations reduce uncertainty
- **Risk Preferences:** Whether agent is risk-seeking or risk-averse

## Research Applications

### Cognitive Science Research
- **Meta-cognition studies:** Understanding self-awareness and reflection
- **Attention research:** Dynamic resource allocation mechanisms
- **Decision making:** Choice under uncertainty and temporal reasoning
- **Learning processes:** How agents learn about their own learning

### AI and Machine Learning
- **Explainable AI:** Understanding agent reasoning processes
- **Robustness analysis:** How agents handle uncertainty and change
- **Multi-agent systems:** Theory of mind and social reasoning
- **Active learning:** Optimal information gathering strategies

### Neuroscience Applications
- **Cognitive modeling:** Computational models of brain function
- **Clinical applications:** Understanding metacognitive deficits
- **Brain-computer interfaces:** Uncertainty-aware control systems
- **Predictive coding:** Hierarchical temporal prediction

### Computational Psychiatry
- **Metacognitive disorders:** Confidence and insight problems
- **Anxiety and uncertainty:** Maladaptive uncertainty processing
- **Decision-making deficits:** Suboptimal choice patterns
- **Social cognition:** Theory of mind impairments

## Performance Optimization

### Computational Considerations
- **Bootstrap sampling:** Adjustable sample sizes for epistemic uncertainty
- **Ensemble models:** Configurable ensemble sizes for model uncertainty
- **Temporal windows:** Scalable analysis windows for multi-scale analysis
- **Parallel processing:** Independent module execution for speed

### Memory Management
- **Incremental analysis:** Process data in chunks for large datasets
- **Result caching:** Store intermediate results to avoid recomputation
- **Selective analysis:** Run only needed modules for specific research questions
- **Data compression:** Efficient storage of analysis results

## Future Extensions

### Planned Enhancements
1. **Real-time analysis:** Online processing of streaming data
2. **Interactive visualization:** Web-based exploration tools
3. **Comparative analysis:** Multi-model and multi-agent comparisons
4. **Causal discovery:** Automated causal structure learning
5. **Predictive modeling:** Future behavior forecasting

### Research Directions
1. **Hierarchical active inference:** Multi-level agent architectures
2. **Social active inference:** Multi-agent theory of mind
3. **Continual learning:** Lifelong adaptation mechanisms
4. **Embodied cognition:** Sensorimotor integration
5. **Consciousness modeling:** Integrated information theory applications

## Troubleshooting

### Common Issues

**Module Loading Errors:**
- Ensure all required packages are installed
- Check Julia version compatibility (Julia 1.6+)
- Verify file paths and module dependencies

**Memory Issues:**
- Reduce bootstrap sample sizes
- Use smaller temporal windows
- Process data in batches
- Close unused analysis results

**Performance Issues:**
- Run modules in parallel when possible
- Use subset of data for initial exploration
- Optimize temporal window sizes
- Cache intermediate results

**Missing Data:**
- Check input data format and structure
- Verify required columns are present
- Handle missing values appropriately
- Use data validation functions

### Error Recovery
- Each module includes comprehensive error handling
- Failed modules don't prevent others from running
- Detailed error logs for debugging
- Graceful degradation when data is insufficient

## Support and Development

### Getting Help
- Check error logs for specific issues
- Review input data requirements
- Consult function documentation
- Use built-in validation functions

### Contributing
- Follow existing code style and documentation
- Add comprehensive error handling
- Include validation and testing
- Provide clear examples and use cases

### Citation
When using these enhanced analysis capabilities, please cite:
- The original ActiveInference.jl package
- Relevant theoretical papers on active inference
- This enhanced analysis framework
- Specific modules used in your research

## Conclusion

This ActiveInference.jl implementation provides insight into POMDP reasoning processes through analysis of meta-cognitive, temporal, uncertainty, and counterfactual aspects of agent behavior. The modular design allows for flexible research applications while the integrated suite provides understanding of complex cognitive processes.

The framework bridges theoretical active inference with practical computational analysis, enabling researchers to explore how intelligent agents reason, plan, and adapt in uncertain environments. 