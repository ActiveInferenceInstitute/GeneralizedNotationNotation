# DSPy Integration for GNN

> **📋 Document Metadata**  
> **Type**: Integration Guide | **Audience**: Developers, Researchers | **Complexity**: Beginner to Advanced  
> **Cross-References**: [AGENTS.md](AGENTS.md) | [DSPy GNN Guide](gnn_dspy.md) | [LLM Integration](../gnn/gnn_llm_neurosymbolic_active_inference.md) | [Main Documentation](../README.md)

## Overview

This directory contains comprehensive documentation, resources, and implementation guides for integrating **DSPy** (Declarative Structured Prompting for Language Models) with GNN (Generalized Notation Notation). DSPy provides a systematic approach to LLM programming, moving from prompt engineering to programmatic LLM workflows.

**Status**: ✅ Production Ready  
**Version**: 2.0  
**Last Updated**: January 2026

---

## Quick Start

```python
import dspy

# Configure LM
lm = dspy.LM('openai/gpt-4o-mini', api_key='YOUR_KEY')
dspy.configure(lm=lm)

# Create a simple module
classifier = dspy.ChainOfThought('sentence -> sentiment: bool')
result = classifier(sentence="Active Inference is fascinating!")
print(result.sentiment)  # True
```

---

## Documentation Index

### Core Documentation

| Document | Description | Complexity |
|----------|-------------|------------|
| **[gnn_dspy.md](gnn_dspy.md)** | Complete DSPy-GNN integration theory | Advanced |
| **[dspy_modules_reference.md](dspy_modules_reference.md)** | Comprehensive module catalog | Intermediate |
| **[dspy_agents_guide.md](dspy_agents_guide.md)** | Building ReAct agents with tools | Advanced |
| **[dspy_optimizers_guide.md](dspy_optimizers_guide.md)** | MIPROv2, BootstrapFinetune, optimization | Advanced |
| **[dspy_assertions_guide.md](dspy_assertions_guide.md)** | Output validation and constraints | Intermediate |
| **[dspy_retrieval_guide.md](dspy_retrieval_guide.md)** | RAG, ColBERT, vector databases | Intermediate |
| **[dspy_typed_predictors.md](dspy_typed_predictors.md)** | Pydantic and structured output | Intermediate |
| **[dspy_gnn_integration_patterns.md](dspy_gnn_integration_patterns.md)** | Practical GNN integration patterns | Advanced |

### Navigation

- **[README.md](README.md)**: This overview (start here)
- **[AGENTS.md](AGENTS.md)**: Technical scaffolding and navigation

---

## DSPy Core Concepts

### Signatures
Define input-output behavior without implementation details:
```python
"question -> answer"
"context, question -> reasoning, answer"
"document -> summary: str, topics: list[str]"
```

### Modules
Building blocks for LLM programs:
- **`dspy.Predict`**: Basic prediction
- **`dspy.ChainOfThought`**: Step-by-step reasoning
- **`dspy.ProgramOfThought`**: Code generation and execution
- **`dspy.ReAct`**: Agent with tool use
- **`dspy.Refine`**: Output refinement

### Optimizers
Automatic prompt and weight tuning:
- **`MIPROv2`**: Bayesian instruction optimization
- **`BootstrapFewShot`**: Example synthesis
- **`BootstrapFinetune`**: LM weight fine-tuning

### Key Benefits
- **Programming over Prompting**: Systematic, maintainable approach
- **Automatic Optimization**: Data-driven prompt tuning
- **Modular Design**: Composable, reusable components
- **Model Portability**: Works across different LLMs

---

## Integration with GNN Pipeline

This documentation is integrated with the 24-step GNN processing pipeline:

### Core Processing (Steps 0-9)
- DSPy enhances GNN parsing with LLM assistance
- [dspy_gnn_integration_patterns.md](dspy_gnn_integration_patterns.md): Observation processing patterns

### Simulation (Steps 10-16)
- DSPy-optimized LLM analysis (Step 13: LLM)
- [dspy_agents_guide.md](dspy_agents_guide.md): Active Inference agent patterns

### Integration (Steps 17-23)
- DSPy results integrated into comprehensive outputs
- [dspy_typed_predictors.md](dspy_typed_predictors.md): Structured output generation

See [src/AGENTS.md](../../src/AGENTS.md) for complete pipeline documentation.

---

## File Structure

```
doc/dspy/
├── README.md                          # This overview
├── AGENTS.md                          # Technical scaffolding
├── gnn_dspy.md                        # Core DSPy-GNN theory
├── dspy_modules_reference.md          # Module reference
├── dspy_agents_guide.md               # Agent development
├── dspy_optimizers_guide.md           # Optimization guide
├── dspy_assertions_guide.md           # Output validation
├── dspy_retrieval_guide.md            # RAG and retrieval
├── dspy_typed_predictors.md           # Typed outputs
└── dspy_gnn_integration_patterns.md   # Integration patterns
```

**Files**: 10 | **Total Documentation**: ~3000 lines

---

## Learning Paths

### Beginner Path
1. Start with this README
2. Read [dspy_modules_reference.md](dspy_modules_reference.md)
3. Try examples in [dspy_typed_predictors.md](dspy_typed_predictors.md)

### Intermediate Path
1. Complete beginner path
2. Study [dspy_assertions_guide.md](dspy_assertions_guide.md)
3. Explore [dspy_retrieval_guide.md](dspy_retrieval_guide.md)

### Advanced Path
1. Complete intermediate path
2. Master [dspy_optimizers_guide.md](dspy_optimizers_guide.md)
3. Build agents with [dspy_agents_guide.md](dspy_agents_guide.md)
4. Integrate with GNN via [dspy_gnn_integration_patterns.md](dspy_gnn_integration_patterns.md)

---

## Quick Reference

### Common Patterns

```python
# Classification
classifier = dspy.Predict('text -> category: str')

# Question Answering with reasoning
qa = dspy.ChainOfThought('context, question -> answer')

# Agent with tools
agent = dspy.ReAct('question -> answer', tools=[search, calculate])

# Structured output
from pydantic import BaseModel
class Output(BaseModel):
    summary: str
    keywords: list[str]
    
predictor = dspy.ChainOfThought('document -> result: Output')
```

### Configuration

```python
import dspy

# OpenAI
lm = dspy.LM('openai/gpt-4o', api_key='...')

# Anthropic
lm = dspy.LM('anthropic/claude-3-opus', api_key='...')

# Ollama (local)
lm = dspy.LM('ollama/llama3.2:3b')

# Configure globally
dspy.configure(lm=lm)
```

---

## Related Resources

### Main GNN Documentation
- **[GNN Overview](../gnn/gnn_overview.md)**: Core GNN concepts
- **[GNN Quickstart](../gnn/quickstart_tutorial.md)**: Getting started guide
- **[LLM Integration](../gnn/gnn_llm_neurosymbolic_active_inference.md)**: Neurosymbolic architecture

### Development Resources
- **[AutoGenLib Integration](../autogenlib/gnn_autogenlib.md)**: Dynamic code generation
- **[PoE-World Integration](../poe-world/poe-world_gnn.md)**: Compositional world modeling
- **[Development Guide](../development/README.md)**: Development workflows

### Pipeline Architecture
- **[Pipeline Documentation](../gnn/gnn_tools.md)**: Complete pipeline guide
- **[Pipeline AGENTS](../../src/AGENTS.md)**: Implementation details
- **[Pipeline README](../../src/README.md)**: Pipeline overview

### External Resources
- **[DSPy Official Documentation](https://dspy.ai)**: Official DSPy docs
- **[DSPy GitHub](https://github.com/stanfordnlp/dspy)**: Source code
- **[MLflow DSPy Integration](https://mlflow.org/docs/latest/llms/dspy/)**: Observability

---

## Standards and Guidelines

All documentation in this module adheres to professional standards:

- **Clarity**: Concrete, technical writing with LLM programming foundations
- **Functionality**: Describes actual DSPy integration capabilities
- **Completeness**: Comprehensive coverage of structured prompting integration
- **Consistency**: Uniform structure and style with GNN documentation ecosystem
- **Code Examples**: All examples are syntactically correct and tested

---

## See Also

- **[DSPy Cross-Reference](../CROSS_REFERENCE_INDEX.md#dspy)**: Cross-reference index entry
- **[LLM Integration](../gnn/gnn_llm_neurosymbolic_active_inference.md)**: LLM-enhanced analysis
- **[AutoGenLib Integration](../autogenlib/gnn_autogenlib.md)**: Dynamic code generation
- **[Main Index](../README.md)**: Return to main documentation

---

**Status**: ✅ Production Ready  
**Compliance**: Professional documentation standards  
**Maintenance**: Regular updates with new DSPy features and integration capabilities
