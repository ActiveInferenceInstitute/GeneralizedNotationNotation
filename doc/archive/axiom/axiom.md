# AXIOM: Active eXpanding Inference with Object-centric Models

## Overview

AXIOM (Active eXpanding Inference with Object-centric Models) represents a breakthrough in artificial intelligence that combines Active Inference principles with object-centric modeling to achieve human-like learning efficiency. Published on arXiv on May 30, 2025, this research demonstrates how AI systems can learn complex tasks from minimal data by understanding objects and their interactions, rather than memorizing statistical patterns.

## Key Innovation

AXIOM addresses fundamental limitations of current deep reinforcement learning (DRL) approaches:

- **Sample Efficiency**: Learns game proficiency in just 10,000 interaction steps vs. tens of thousands for traditional DRL
- **Interpretability**: Builds structured, human-readable world models with object-centric representations  
- **Adaptability**: Automatically expands and prunes its model structure to match environmental complexity
- **Robustness**: Demonstrates resilience to visual perturbations and domain shifts

## Architecture

### Core Components

AXIOM employs four interconnected mixture models that work together to parse, understand, and predict object-centric dynamics:

#### 1. Slot Mixture Model (sMM)
- **Purpose**: Segments visual scenes into object-centric representations
- **Function**: Processes RGB images pixel-by-pixel, assigning each pixel to one of K object slots
- **Innovation**: Automatically expands to accommodate new objects in the scene
- **Technical Details**: Uses Gaussian mixture model with truncated stick-breaking priors

#### 2. Identity Mixture Model (iMM)  
- **Purpose**: Assigns discrete identity codes to objects based on continuous features
- **Function**: Groups objects by type (color, shape) to enable shared dynamics learning
- **Benefit**: Allows same dynamics to apply across object instances of the same type
- **Robustness**: Enables identity remapping when visual features change

#### 3. Transition Mixture Model (tMM)
- **Purpose**: Models object dynamics as piecewise linear trajectories
- **Function**: Captures distinct motion patterns (falling, sliding, bouncing, etc.)
- **Efficiency**: Shared across all objects, learning universal motion primitives
- **Technical**: Switching Linear Dynamical System (SLDS) with expanding components

#### 4. Recurrent Mixture Model (rMM)
- **Purpose**: Models sparse object-object interactions and context dependencies
- **Function**: Predicts switch states for tMM based on multi-object configurations
- **Innovation**: Generative (vs. discriminative) approach to recurrent switching
- **Scope**: Handles actions, rewards, and complex multi-object relationships

### Learning Algorithm

AXIOM uses **variational Bayesian inference** with several key innovations:

- **Sequential Learning**: Updates one frame at a time without replay buffers or gradients
- **Online Structure Learning**: Dynamically adds new mixture components when needed
- **Bayesian Model Reduction (BMR)**: Periodically merges redundant components for generalization
- **Fast Updates**: Exponential family likelihoods enable closed-form coordinate ascent

### Planning with Active Inference

AXIOM employs **expected free energy minimization** for policy selection:

```
π* = argmin[π] Σ[-E[log p(reward|O,π)] + DKL(q(parameters|O,π) || q(parameters))]
                 ↑ Utility (reward seeking)    ↑ Information Gain (exploration)
```

This balances:
- **Utility**: Maximizing expected rewards
- **Information Gain**: Seeking informative experiences for model improvement

## Experimental Validation

### Gameworld 10k Benchmark

The researchers introduced a new benchmark specifically designed to test sample-efficient learning:

- **10 Games**: Diverse physics-based environments with object interactions
- **10k Step Limit**: Forces focus on rapid learning rather than extensive exploration  
- **Visual Simplicity**: Clean sprites allow focus on dynamics rather than perception complexity
- **Controlled Perturbations**: Test robustness to color/shape changes

### Performance Results

AXIOM demonstrates superior performance across multiple metrics:

| **Metric** | **DREAMER V3** | **AXIOM** | **Improvement** |
|------------|----------------|-----------|-----------------|
| Normalized Score | 0.48 | 0.77 | **+60% better** |
| Steps to Competence | 24,207 | 3,175 | **7x faster** |
| GPU Time | 6.23 hrs | 0.16 hrs | **39x more efficient** |
| Model Size | 420M params | 0.95M params | **440x smaller** |

### Key Findings

1. **Sample Efficiency**: AXIOM reaches competence in ~3k steps vs. 24k+ for baselines
2. **Parameter Efficiency**: Uses 99.8% fewer parameters than DreamerV3
3. **Computational Efficiency**: 97% reduction in compute time requirements  
4. **Consistent Performance**: Outperforms baselines across all 10 game environments
5. **Robustness**: Maintains performance under visual perturbations

## Technical Advantages

### Interpretable World Models

Unlike black-box neural networks, AXIOM's representations are directly interpretable:

- **Object Properties**: Position, velocity, color, shape explicitly represented
- **Motion Patterns**: Linear dynamics components correspond to recognizable behaviors
- **Interaction Clusters**: rMM clusters show spatial reward/punishment associations
- **Causal Structure**: Clear object-to-object interaction dependencies

### Online Adaptation

AXIOM continuously adapts its structure:

- **Growth**: Adds new components when existing ones can't explain observations
- **Pruning**: Merges redundant components to prevent overfitting
- **Efficiency**: Maintains minimal sufficient complexity for each environment
- **Generalization**: BMR enables transfer beyond training experiences

### Gradient-Free Learning

AXIOM achieves its results without:
- Gradient-based optimization
- Replay buffers  
- Backpropagation
- Large neural networks

Instead using principled Bayesian updates with exponential family conjugacy.

## Broader Implications

### For AI Research

AXIOM demonstrates that:

1. **Object-centric priors** can dramatically improve sample efficiency
2. **Bayesian methods** remain competitive with deep learning when properly structured
3. **Interpretable models** don't require sacrificing performance
4. **Active inference** provides a unified framework for learning and planning

### For Practical Applications

The approach suggests potential for:

- **Edge Computing**: Tiny models that run on low-power devices
- **Robotics**: Sample-efficient learning for real-world interactions
- **Scientific Modeling**: Interpretable models for understanding complex systems
- **Safety-Critical AI**: Transparent decision-making for high-stakes applications

### Comparison to Human Learning

AXIOM's approach mirrors human cognition:

- **Core Priors**: Built-in understanding of objects, physics, causality
- **Few-Shot Learning**: Rapid adaptation to new scenarios
- **Transfer Learning**: Applying learned principles to novel situations  
- **Interpretable Reasoning**: Clear cause-and-effect understanding

## Limitations and Future Work

### Current Limitations

1. **Engineered Priors**: Core object assumptions are hand-crafted rather than learned
2. **Simple Environments**: Tested on relatively simple visual scenes
3. **Scale**: Not yet demonstrated on complex domains like Atari or Minecraft
4. **Object Discovery**: Requires relatively clean object segmentation

### Research Directions

The authors identify several important extensions:

1. **Automatic Prior Discovery**: Learning core priors from data rather than engineering them
2. **Complex Domains**: Scaling to more visually complex environments  
3. **Hierarchical Structure**: Multi-level object composition and abstraction
4. **Real-World Transfer**: Application to robotics and physical systems

## Connection to Active Inference

AXIOM represents a significant advancement in Active Inference research:

- **Scalability**: Shows Active Inference can work on pixel-based control tasks
- **Structure Learning**: Demonstrates automatic model complexity adaptation
- **Object-Centric Priors**: Integrates computer vision advances with principled inference
- **Practical Performance**: Achieves competitive results against state-of-the-art DRL

## Implementation Details

### Key Hyperparameters

- **Planning Horizon**: 32 steps
- **Rollout Policies**: 512 per planning step  
- **Expansion Thresholds**: Tuned per mixture model type
- **BMR Frequency**: Every 500 frames
- **Information Gain Weight**: 0.1

### Computational Requirements

- **Training Time**: ~30 minutes per 10k steps on A100 GPU
- **Memory**: Minimal due to small model size
- **Inference**: 18ms per step (vs. 221ms for DreamerV3)
- **Planning**: 252-534ms depending on rollout count

## Relationship to GNN Project

AXIOM's principles align closely with GeneralizedNotationNotation (GNN) objectives:

- **Structured Representations**: Both emphasize interpretable, structured models
- **Active Inference**: Both build on Active Inference theoretical foundations  
- **Object-Centric Modeling**: Both recognize objects as fundamental modeling units
- **Scientific Rigor**: Both prioritize mathematical precision and reproducibility

The AXIOM architecture could potentially be:
1. **Translated to GNN notation** for formal specification
2. **Integrated into GNN pipelines** for dynamic model learning
3. **Extended with GNN ontologies** for richer semantic understanding
4. **Used as a target** for GNN-to-executable code generation

## Conclusion

AXIOM represents a paradigm shift toward more human-like artificial intelligence that learns efficiently through structured understanding rather than brute-force statistical learning. By combining object-centric representations with Active Inference principles, it achieves dramatic improvements in sample efficiency, interpretability, and computational requirements while maintaining competitive performance.

The work demonstrates that well-structured Bayesian approaches can compete with and often exceed the performance of large-scale deep learning systems, particularly in scenarios where data efficiency and interpretability are crucial. This has profound implications for the future of AI, suggesting paths toward more sustainable, interpretable, and robust artificial intelligence systems.

For the GNN project specifically, AXIOM provides concrete evidence that Active Inference can scale to practical applications while maintaining the theoretical rigor and interpretability that make it attractive for scientific modeling and reproducible research.

---

**Paper Reference**: Heins, C., Van de Maele, T., Tschantz, A., et al. (2025). AXIOM: Learning to Play Games in Minutes with Expanding Object-Centric Models. arXiv:2505.24784 [cs.AI]

**Code Availability**: https://github.com/VersesTech/axiom  
**Benchmark**: https://github.com/VersesTech/gameworld
