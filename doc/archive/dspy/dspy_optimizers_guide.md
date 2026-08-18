# DSPy Optimizers Guide

> **📋 Document Metadata**  
> **Type**: Technical Guide | **Audience**: Developers, ML Engineers | **Complexity**: Advanced  
> **Cross-References**: [Modules Reference](dspy_modules_reference.md) | [Assertions Guide](dspy_assertions_guide.md) | [GNN Integration](dspy_gnn_integration_patterns.md)

## Overview

DSPy optimizers automatically tune the prompts and weights of language model programs to maximize specified metrics. This guide provides comprehensive coverage of all DSPy optimizers, their configurations, and best practices for GNN-integrated systems.

**Status**: ✅ Production Ready  
**Version**: 1.0

---

## Optimization Philosophy

DSPy's optimization approach differs fundamentally from traditional prompt engineering:

```
Traditional Prompt Engineering:
┌─────────────────┐
│ Manual Prompt   │ → Try → Evaluate → Tweak → Repeat
│ Engineering     │   (tedious, error-prone, non-systematic)
└─────────────────┘

DSPy Optimization:
┌─────────────────┐
│ Define Metric   │ → Compile → Optimize → Deploy
│ + Training Data │   (systematic, automatic, reproducible)
└─────────────────┘
```

---

## Core Concepts

### What Gets Optimized

DSPy optimizers can tune:
1. **Instructions**: Natural language directives in prompts
2. **Demonstrations**: Few-shot examples included in prompts
3. **LM Weights**: Fine-tuning the underlying model weights

### Optimizer Inputs

All optimizers require:
- **Program**: The DSPy module(s) to optimize
- **Metric**: A function that scores program outputs
- **Training Set**: Examples for optimization (often just 5-20 needed)

```python
import dspy

# Basic optimizer setup pattern
def my_metric(example, prediction, trace=None):
    """Score the prediction quality."""
    return prediction.answer == example.expected_answer

optimizer = dspy.MIPROv2(
    metric=my_metric,
    num_candidates=10
)

optimized_program = optimizer.compile(
    student=my_program,
    trainset=training_examples
)
```

---

## Available Optimizers

### MIPROv2 (Multiprompt Instruction PRoposal Optimizer v2)

The most powerful general-purpose optimizer. Uses Bayesian optimization to jointly tune instructions and demonstrations.

**How It Works**:

```
Phase 1: Bootstrapping
┌────────────────────────────────────────┐
│ Run program on training examples       │
│ Collect execution traces               │
│ Filter traces by metric scores         │
└────────────────────────────────────────┘
           │
           ▼
Phase 2: Grounded Proposal
┌────────────────────────────────────────┐
│ Analyze program code and traces        │
│ Generate candidate instructions        │
│ Create instruction variations          │
└────────────────────────────────────────┘
           │
           ▼
Phase 3: Discrete Search
┌────────────────────────────────────────┐
│ Sample instruction/demo combinations   │
│ Evaluate on mini-batches               │
│ Update surrogate model (Bayesian)      │
│ Identify optimal configuration         │
└────────────────────────────────────────┘
```

**Usage**:

```python
from dspy.teleprompt import MIPROv2

optimizer = MIPROv2(
    metric=my_metric,
    num_candidates=10,          # Number of instruction candidates
    init_temperature=1.0,       # Initial sampling temperature
    prompt_model=None,          # LM for generating instructions (default: same as student)
    task_model=None,            # LM for task execution (default: same as student)
    num_threads=6,              # Parallel evaluation threads
    max_bootstrapped_demos=4,   # Max demos per prompt
    max_labeled_demos=16,       # Max labeled examples to use
    verbose=True
)

optimized = optimizer.compile(
    student=my_program,
    trainset=trainset,
    num_trials=50,              # Number of Bayesian trials
    minibatch_size=25,          # Evaluation batch size
    minibatch_full_eval_steps=10  # Full eval frequency
)
```

**Cost Considerations**:
- Typical optimization: ~$2-10 USD with GPT-4
- Time: 15-60 minutes depending on configuration
- Cost scales with `num_trials` and `num_candidates`

**Best For**:
- Production prompt optimization
- Complex multi-step programs
- When quality matters more than cost

---

### BootstrapFewShot

Synthesizes effective few-shot examples by running the program and collecting successful traces.

**How It Works**:
1. Run the program on training examples
2. Score outputs using the metric
3. Collect high-scoring input-output pairs as demonstrations
4. Add these demonstrations to future prompts

```python
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(
    metric=my_metric,
    max_bootstrapped_demos=4,   # Max synthetic demos
    max_labeled_demos=4,        # Max labeled demos from trainset
    max_rounds=1,               # Bootstrap iterations
    max_errors=5                # Max errors before stopping
)

optimized = optimizer.compile(
    student=my_program,
    trainset=trainset
)
```

**Best For**:
- Quick optimization baseline
- When you have limited compute budget
- Programs that benefit from examples

---

### BootstrapFewShotWithRandomSearch

Extends BootstrapFewShot with random search over demo combinations.

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

optimizer = BootstrapFewShotWithRandomSearch(
    metric=my_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    num_candidate_programs=16,  # Number of random configurations
    num_threads=6
)

optimized = optimizer.compile(
    student=my_program,
    trainset=trainset
)
```

---

### BootstrapFinetune

Creates fine-tuning datasets and tunes the underlying LM weights.

**How It Works**:
1. Run program to collect traces
2. Filter high-quality traces
3. Create fine-tuning dataset
4. Fine-tune LM on the dataset

```python
from dspy.teleprompt import BootstrapFinetune

optimizer = BootstrapFinetune(
    metric=my_metric,
    multitask=False,            # Whether to fine-tune on multiple tasks
    target='gpt-3.5-turbo'      # Target model for fine-tuning
)

# Note: This requires API access to fine-tuning endpoints
optimized = optimizer.compile(
    student=my_program,
    trainset=trainset,
    teacher=teacher_program     # Optional: use a stronger model as teacher
)
```

**Best For**:
- Maximum performance
- When you can afford fine-tuning
- Production deployments with high traffic

---

### BetterTogether

Combines multiple optimization strategies sequentially.

```python
from dspy.teleprompt import BetterTogether

# First, prompt optimization
prompt_optimizer = MIPROv2(metric=my_metric)

# Then, fine-tuning
finetune_optimizer = BootstrapFinetune(metric=my_metric)

# Combine them
optimizer = BetterTogether(
    prompt_optimizer,
    finetune_optimizer
)

optimized = optimizer.compile(
    student=my_program,
    trainset=trainset
)
```

---

### LabeledFewShot

Uses labeled examples directly as demonstrations (no bootstrapping).

```python
from dspy.teleprompt import LabeledFewShot

optimizer = LabeledFewShot(k=4)  # Use 4 labeled examples

optimized = optimizer.compile(
    student=my_program,
    trainset=trainset  # Must have labels
)
```

**Best For**:
- When you have high-quality labeled data
- Quick baseline without bootstrapping

---

### COPRO (Cooperative Prompt Optimization)

Cooperative optimization between instruction generation and evaluation.

```python
from dspy.teleprompt import COPRO

optimizer = COPRO(
    metric=my_metric,
    breadth=10,     # Number of instructions to sample
    depth=3,        # Refinement iterations
    init_temperature=1.4
)

optimized = optimizer.compile(
    student=my_program,
    trainset=trainset,
    eval_kwargs={'num_threads': 4}
)
```

---

## Writing Effective Metrics

### Basic Metric Structure

```python
def my_metric(example, prediction, trace=None):
    """
    Args:
        example: The input example with expected outputs
        prediction: The program's prediction
        trace: Optional execution trace for debugging
        
    Returns:
        float or bool: Score (higher is better)
    """
    # Exact match
    if prediction.answer == example.expected_answer:
        return 1.0
    
    # Partial credit
    if example.expected_answer.lower() in prediction.answer.lower():
        return 0.5
    
    return 0.0
```

### Advanced Metrics

```python
def composite_metric(example, prediction, trace=None):
    """Multi-criteria metric with weighted components."""
    scores = {}
    
    # Correctness (50%)
    scores['correctness'] = 1.0 if prediction.answer == example.expected else 0.0
    
    # Conciseness (20%) - prefer shorter answers
    max_len = 200
    scores['conciseness'] = max(0, 1 - len(prediction.answer) / max_len)
    
    # Reasoning quality (30%) - check for reasoning steps
    if hasattr(prediction, 'reasoning'):
        scores['reasoning'] = min(1.0, len(prediction.reasoning.split('.')) / 5)
    else:
        scores['reasoning'] = 0.0
    
    # Weighted combination
    return (
        0.5 * scores['correctness'] +
        0.2 * scores['conciseness'] +
        0.3 * scores['reasoning']
    )
```

### LLM-as-Judge Metrics

```python
def llm_judge_metric(example, prediction, trace=None):
    """Use an LLM to evaluate quality."""
    judge = dspy.ChainOfThought(
        'question, expected_answer, actual_answer -> score: float, reasoning'
    )
    
    result = judge(
        question=example.question,
        expected_answer=example.expected_answer,
        actual_answer=prediction.answer
    )
    
    return result.score
```

---

## GNN-Specific Optimization

### Optimizing GNN Observation Parsers

```python
class GNNObservationParser(dspy.Module):
    def __init__(self):
        super().__init__()
        self.parse = dspy.ChainOfThought(
            'user_input, gnn_schema -> symbolic_observation'
        )
    
    def forward(self, user_input, gnn_schema):
        return self.parse(user_input=user_input, gnn_schema=gnn_schema)

# Metric: Check if output is valid GNN syntax
def gnn_syntax_metric(example, prediction, trace=None):
    """Check if the generated observation is valid GNN syntax."""
    try:
        # Try to parse as GNN
        is_valid = validate_gnn_syntax(prediction.symbolic_observation)
        
        # Check semantic correctness
        matches_expected = (
            prediction.symbolic_observation == example.expected_observation
        )
        
        return 0.5 * is_valid + 0.5 * matches_expected
    except Exception:
        return 0.0

# Training data
trainset = [
    dspy.Example(
        user_input="I see a wall ahead",
        gnn_schema="o[wall]: bool, o[distance]: float",
        expected_observation="o[wall]=true, o[distance]=1.0"
    ).with_inputs('user_input', 'gnn_schema')
    for _ in examples
]

# Optimize
optimizer = MIPROv2(metric=gnn_syntax_metric)
optimized_parser = optimizer.compile(
    student=GNNObservationParser(),
    trainset=trainset
)
```

### Optimizing Policy Evaluation

```python
class PolicyEvaluator(dspy.Module):
    """Evaluate policy semantic alignment with goals."""
    
    def __init__(self):
        super().__init__()
        self.evaluate = dspy.ChainOfThought(
            'policy_description, predicted_outcomes, user_goal -> alignment_score: float, reasoning'
        )
    
    def forward(self, policy, outcomes, goal):
        return self.evaluate(
            policy_description=policy,
            predicted_outcomes=outcomes,
            user_goal=goal
        )

def policy_metric(example, prediction, trace=None):
    """Measure how well the evaluator matches human judgments."""
    score_diff = abs(prediction.alignment_score - example.human_score)
    return 1.0 - min(1.0, score_diff)  # Penalize deviation

optimizer = MIPROv2(metric=policy_metric, num_candidates=15)
optimized_evaluator = optimizer.compile(
    student=PolicyEvaluator(),
    trainset=policy_trainset
)
```

---

## Composable Optimization Strategies

### Sequential Optimization

```python
# Step 1: Optimize instructions
mipro = MIPROv2(metric=my_metric)
instruction_optimized = mipro.compile(my_program, trainset)

# Step 2: Further optimize demos
bootstrap = BootstrapFewShotWithRandomSearch(metric=my_metric)
fully_optimized = bootstrap.compile(instruction_optimized, trainset)
```

### Cascade Optimization

```python
# First optimize with a large model
with dspy.context(lm=gpt4):
    optimizer = MIPROv2(metric=my_metric)
    optimized_big = optimizer.compile(my_program, trainset)

# Extract the optimized prompts
best_instructions = optimized_big.signature.instructions

# Apply to smaller model
with dspy.context(lm=llama_7b):
    small_program = my_program.reset_copy()
    small_program.signature = small_program.signature.with_instructions(best_instructions)
```

### Ensemble Optimization

```python
from dspy.teleprompt import Ensemble

# Run optimization multiple times
candidates = []
for seed in range(5):
    optimizer = MIPROv2(metric=my_metric, seed=seed)
    candidate = optimizer.compile(my_program.reset_copy(), trainset)
    candidates.append(candidate)

# Create ensemble
ensemble = Ensemble(reduce_fn=dspy.majority)
ensemble_program = ensemble.compile(candidates)
```

---

## Cost Management

### Estimating Optimization Cost

```python
def estimate_optimization_cost(
    optimizer_type: str,
    trainset_size: int,
    num_trials: int = 50,
    model: str = 'gpt-4o'
) -> dict:
    """Estimate optimization cost."""
    
    # Approximate costs per 1K tokens (adjust as needed)
    costs = {
        'gpt-4o': {'input': 0.0025, 'output': 0.01},
        'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015}
    }
    
    model_cost = costs.get(model, costs['gpt-4o'])
    
    if optimizer_type == 'MIPROv2':
        # Rough estimation
        total_calls = trainset_size * 3 + num_trials * trainset_size * 0.5
        avg_tokens = 1000
        estimated_cost = total_calls * avg_tokens * (model_cost['input'] + model_cost['output']) / 1000
        
    elif optimizer_type == 'BootstrapFewShot':
        total_calls = trainset_size * 2
        estimated_cost = total_calls * 500 * (model_cost['input'] + model_cost['output']) / 1000
    
    return {
        'estimated_cost_usd': estimated_cost,
        'estimated_calls': total_calls
    }
```

### Budget-Constrained Optimization

```python
# Start with quick optimization
optimizer = BootstrapFewShot(metric=my_metric, max_bootstrapped_demos=2)
baseline = optimizer.compile(my_program, trainset[:10])

# Evaluate baseline
baseline_score = evaluate(baseline, devset)

# Only do expensive optimization if baseline is insufficient
if baseline_score < 0.8:
    expensive_optimizer = MIPROv2(
        metric=my_metric,
        num_candidates=5,  # Reduced from default
        num_trials=20      # Reduced budget
    )
    optimized = expensive_optimizer.compile(my_program, trainset)
else:
    optimized = baseline
```

---

## Debugging Optimization

### Inspecting Optimized Programs

```python
# View optimized instructions
print(optimized_program.signature.instructions)

# View selected demonstrations
for demo in optimized_program.demos:
    print(f"Input: {demo.input}")
    print(f"Output: {demo.output}")
    print("---")
```

### Tracking Optimization Progress

```python
import mlflow

mlflow.set_experiment("DSPy-Optimization")

with mlflow.start_run():
    optimizer = MIPROv2(metric=my_metric, verbose=True)
    
    # Log parameters
    mlflow.log_params({
        'optimizer': 'MIPROv2',
        'num_candidates': 10,
        'trainset_size': len(trainset)
    })
    
    optimized = optimizer.compile(my_program, trainset)
    
    # Evaluate and log
    score = evaluate(optimized, devset)
    mlflow.log_metric('dev_score', score)
```

---

## Best Practices

### 1. Start Simple
Begin with `BootstrapFewShot` before trying `MIPROv2`.

### 2. Use Appropriate Training Set Size
- Minimum: 5-10 examples
- Recommended: 20-50 examples
- More data helps but has diminishing returns

### 3. Design Good Metrics
- Should correlate with actual quality
- Should be fast to compute
- Consider using multiple metrics

### 4. Validate on Held-Out Data
Always evaluate on separate dev/test sets.

### 5. Save Optimized Programs
```python
optimized.save('optimized_program.json')
loaded = MyProgram()
loaded.load('optimized_program.json')
```

---

## Related Resources

- **[Modules Reference](dspy_modules_reference.md)**: Module documentation
- **[Agents Guide](dspy_agents_guide.md)**: Agent optimization
- **[Assertions Guide](dspy_assertions_guide.md)**: Output constraints
- **[GNN Integration](dspy_gnn_integration_patterns.md)**: GNN-specific optimization

---

**Status**: ✅ Production Ready  
**Compliance**: GNN documentation standards  
**Maintenance**: Regular updates with new DSPy optimizer features
