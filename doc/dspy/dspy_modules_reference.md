# DSPy Modules Reference

> **📋 Document Metadata**  
> **Type**: Technical Reference | **Audience**: Developers, Researchers | **Complexity**: Intermediate  
> **Cross-References**: [README.md](README.md) | [AGENTS.md](AGENTS.md) | [Agents Guide](dspy_agents_guide.md) | [Optimizers Guide](dspy_optimizers_guide.md)

## Overview

DSPy modules are the fundamental building blocks for programming language models. Each module encapsulates a specific prompting technique while maintaining a consistent interface through signatures. This reference provides comprehensive documentation for all DSPy modules and their application in GNN-based Active Inference systems.

**Status**: ✅ Production Ready  
**Version**: 1.1

---

## Core Modules

### dspy.Predict

The most fundamental module in DSPy. All other modules are built upon `dspy.Predict`.

**Purpose**: Direct mapping from inputs to outputs without additional reasoning.

```python
import dspy

# Basic usage
classify = dspy.Predict('sentence -> sentiment: bool')
response = classify(sentence="This is a wonderful experience.")
print(response.sentiment)  # True

# With configuration
predict = dspy.Predict('question -> answer', n=5, temperature=0.7)
```

**Key Features**:
- Handles learning (storing instructions and demonstrations)
- Supports multiple completions via `n` parameter
- Configurable temperature and max length
- Foundation for all optimization

**GNN Integration**: Use for simple GNN syntax validation or observation classification.

```python
# GNN observation classifier
gnn_classifier = dspy.Predict('observation_text -> gnn_category: str')
result = gnn_classifier(observation_text="The agent perceives a wall ahead")
# Returns: "o_wall_detected" or similar GNN-compatible symbol
```

---

### dspy.ChainOfThought

Implements step-by-step reasoning before producing the final output.

**Purpose**: Improve quality on complex reasoning tasks by generating intermediate reasoning steps.

```python
# Standard usage
reasoner = dspy.ChainOfThought('question -> answer')
response = reasoner(question="What is 27 * 13?")
print(response.reasoning)  # Step-by-step calculation
print(response.answer)     # 351

# With typed output
math_solver = dspy.ChainOfThought('question -> answer: float')
response = math_solver(question="What is the probability of rolling a 7 with two dice?")
# response.answer = 0.1666...
```

**When to Use**:
- Complex reasoning tasks
- Mathematical problems
- Multi-step inference
- Tasks requiring explainability

**GNN Integration**: Ideal for translating complex natural language goals into GNN preference parameters.

```python
class GNNGoalTranslator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.translate = dspy.ChainOfThought(
            'natural_language_goal, gnn_state_space -> reasoning, gnn_c_matrix: dict'
        )
    
    def forward(self, goal, state_space):
        return self.translate(natural_language_goal=goal, gnn_state_space=state_space)
```

---

### dspy.ProgramOfThought

Generates and executes Python code to solve problems.

**Purpose**: Leverage code execution for computational tasks that benefit from precise calculation.

```python
# Code-based problem solving
calculator = dspy.ProgramOfThought('problem -> answer: float')
response = calculator(problem="Calculate the compound interest on $1000 at 5% for 10 years")
```

**Key Features**:
- Generates executable Python code
- Executes code in a sandboxed environment
- Returns computed results
- Ideal for deterministic computations

**GNN Integration**: Generate GNN model specifications programmatically.

```python
# GNN matrix generator
gnn_generator = dspy.ProgramOfThought(
    'model_description, state_count: int, obs_count: int -> gnn_a_matrix: list'
)
response = gnn_generator(
    model_description="Sensory mapping where state 0 maps to observation 0 with probability 0.9",
    state_count=3,
    obs_count=3
)
```

---

### dspy.ReAct

Implements the Reasoning and Acting paradigm for tool-using agents.

**Purpose**: Build agents that interleave reasoning with tool calls to accomplish complex tasks.

```python
import dspy

# Define tools
def search_wikipedia(query: str) -> list[str]:
    """Search Wikipedia for relevant articles."""
    # Implementation
    pass

def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expression)

# Create agent
agent = dspy.ReAct(
    signature='question -> answer',
    tools=[search_wikipedia, calculate],
    max_iters=5
)

response = agent(question="What is the population of Tokyo divided by 1000?")
```

**Key Features**:
- Iterative reasoning and action cycles
- Tool integration with automatic dispatch
- Configurable iteration limits
- Planning and execution interleaved

**See also**: [DSPy Agents Guide](dspy_agents_guide.md)

---

### dspy.MultiChainComparison

Compares multiple ChainOfThought outputs to select the best response.

**Purpose**: Ensemble multiple reasoning chains for improved accuracy.

```python
# Multi-chain comparison
compare = dspy.MultiChainComparison('question -> answer', n=5)
response = compare(question="What caused the 2008 financial crisis?")
```

**Key Features**:
- Generates multiple reasoning chains
- Compares and selects best response
- Reduces single-chain errors
- Increases reliability

---

### dspy.Refine

Refines predictions through multiple iterations with different configurations.

**Purpose**: Improve output quality through iterative refinement with feedback.

```python
# Refinement with reward function
def quality_score(response):
    # Score the response quality
    return len(response.answer) > 50 and 'specific' in response.answer.lower()

refiner = dspy.Refine(
    module=dspy.ChainOfThought('question -> answer'),
    reward_fn=quality_score,
    N=5
)
response = refiner(question="Explain photosynthesis")
```

**Key Features**:
- Multiple attempts with different temperatures
- Reward-based selection
- Replaces legacy `dspy.Assert` for some use cases
- Configurable number of attempts

---

## Retrieval Modules

### dspy.Retrieve

Base retrieval module for RAG applications.

```python
# Configure retrieval
retrieve = dspy.Retrieve(k=3)
passages = retrieve(query="Active Inference generative models")
```

### dspy.ColBERTv2

Advanced retrieval using ColBERT's late interaction mechanism.

```python
# ColBERT-based retrieval
colbert = dspy.ColBERTv2(url='http://server:port/index')
results = colbert(query="GNN syntax specification", k=10)
```

**See also**: [DSPy Retrieval Guide](dspy_retrieval_guide.md)

---

## Function-Style Modules

### dspy.majority

Voting mechanism for selecting the most common response.

```python
# Majority voting over multiple predictions
predict = dspy.ChainOfThought('question -> answer', n=10)
response = predict(question="Is Python compiled or interpreted?")
majority_answer = dspy.majority(response.completions.answer)
```

---

## Module Composition

DSPy modules can be composed into larger programs through standard Python class inheritance.

### Basic Composition Pattern

```python
class MultiHopQA(dspy.Module):
    def __init__(self, num_hops=3):
        super().__init__()
        self.num_hops = num_hops
        self.generate_query = dspy.ChainOfThought('context, question -> search_query')
        self.generate_answer = dspy.ChainOfThought('context, question -> answer')
        self.retrieve = dspy.Retrieve(k=5)
    
    def forward(self, question):
        context = []
        for _ in range(self.num_hops):
            query = self.generate_query(context=context, question=question).search_query
            passages = self.retrieve(query)
            context.extend(passages)
        
        return self.generate_answer(context=context, question=question)
```

### GNN-Integrated Composition

```python
class GNNObservationProcessor(dspy.Module):
    """Process natural language into GNN observation format."""
    
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(
            'user_input, gnn_obs_schema -> reasoning, symbolic_observation'
        )
        self.validate = dspy.Predict(
            'symbolic_observation, gnn_syntax_rules -> is_valid: bool, corrected_observation'
        )
    
    def forward(self, user_input, obs_schema, syntax_rules):
        extraction = self.extract(user_input=user_input, gnn_obs_schema=obs_schema)
        validation = self.validate(
            symbolic_observation=extraction.symbolic_observation,
            gnn_syntax_rules=syntax_rules
        )
        
        if validation.is_valid:
            return extraction.symbolic_observation
        return validation.corrected_observation
```

---

## Module Configuration

All modules support common configuration parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | int | Number of completions to generate |
| `temperature` | float | Sampling temperature (0.0-2.0) |
| `max_tokens` | int | Maximum output token length |
| `stop` | list[str] | Stop sequences |

```python
# Comprehensive configuration
module = dspy.ChainOfThought(
    'input -> output',
    n=3,
    temperature=0.7,
    max_tokens=500
)
```

---

## Tracking and Debugging

### LM Usage Tracking

```python
# Track API usage
lm = dspy.LM('openai/gpt-4o', api_key='...')
dspy.configure(lm=lm)

# After running modules
print(lm.history)  # View call history
print(lm.usage)    # Token usage statistics
```

### Inspecting Module Behavior

```python
# Inspect predictions
module = dspy.ChainOfThought('question -> answer')
response = module(question="What is DSPy?")

# Access all completions
print(response.completions)

# Access reasoning traces
print(response.reasoning)
```

---

## Best Practices

### 1. Start Simple
Begin with `dspy.Predict`, only add complexity when needed.

### 2. Use Appropriate Modules
- Simple mapping → `dspy.Predict`
- Reasoning required → `dspy.ChainOfThought`
- Tool use needed → `dspy.ReAct`
- Computation heavy → `dspy.ProgramOfThought`

### 3. Compose Modularly
Build small, focused modules and compose them into larger programs.

### 4. Optimize After Composition
Set up your module pipeline first, then apply optimizers to tune performance.

---

## Related Resources

- **[Agents Guide](dspy_agents_guide.md)**: Building tool-using agents
- **[Optimizers Guide](dspy_optimizers_guide.md)**: Prompt and weight optimization
- **[Assertions Guide](dspy_assertions_guide.md)**: Output validation
- **[Typed Predictors](dspy_typed_predictors.md)**: Structured output with Pydantic
- **[GNN Integration](dspy_gnn_integration_patterns.md)**: Practical integration patterns

---

**Status**: ✅ Production Ready  
**Compliance**: GNN documentation standards  
**Maintenance**: Regular updates with new DSPy features
