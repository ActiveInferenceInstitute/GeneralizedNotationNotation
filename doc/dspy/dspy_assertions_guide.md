# DSPy Assertions Guide

> **📋 Document Metadata**  
> **Type**: Technical Guide | **Audience**: Developers, ML Engineers | **Complexity**: Intermediate  
> **Cross-References**: [Modules Reference](dspy_modules_reference.md) | [Typed Predictors](dspy_typed_predictors.md) | [GNN Integration](dspy_gnn_integration_patterns.md)

## Overview

DSPy Assertions provide a mechanism for enforcing computational constraints on LLM outputs. They enable developers to define strict rules and soft suggestions that guide model behavior, improving reliability and consistency in production systems.

**Status**: ✅ Production Ready  
**Version**: 1.0

---

## Core Concepts

### Why Assertions?

LLMs are probabilistic and may not consistently follow instructions. DSPy Assertions provide:

1. **Guaranteed Constraints**: Enforce output requirements programmatically
2. **Self-Refinement**: Automatic retry with feedback when constraints fail
3. **Debugging Insight**: Clear visibility into constraint violations
4. **Production Reliability**: Prevent invalid outputs from reaching users

```
Without Assertions:
┌──────────────┐
│ LLM Output   │ → May be invalid, wrong format, or violate rules
└──────────────┘

With Assertions:
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ LLM Output   │ → │ Assertion    │ → │ Valid Output │
│              │     │ Check        │     │ (guaranteed) │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                            ▼ (if failed)
                     ┌──────────────┐
                     │ Retry with   │
                     │ Feedback     │
                     └──────────────┘
```

---

## Assertion Types

### dspy.Assert (Hard Constraints)

Hard constraints that **must** be satisfied. Triggers backtracking and retry on failure.

```python
import dspy

class ConstrainedGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought('topic -> summary')
    
    def forward(self, topic):
        result = self.generate(topic=topic)
        
        # Hard constraint: summary must be under 100 words
        dspy.Assert(
            len(result.summary.split()) <= 100,
            "Summary must be 100 words or fewer"
        )
        
        return result
```

**Behavior on Failure**:
1. Logs the assertion error
2. Modifies the signature to include failure feedback
3. Retries generation with the modified prompt
4. After max retries, raises `dspy.AssertionError`

**Parameters**:
- `constraint`: Boolean condition that must be `True`
- `message`: Feedback message provided to the model on failure
- `max_retries`: Maximum retry attempts (default: 3)

---

### dspy.Suggest (Soft Constraints)

Soft constraints that **should** be satisfied. Provides guidance without hard failures.

```python
class GuidedGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought('topic -> explanation')
    
    def forward(self, topic):
        result = self.generate(topic=topic)
        
        # Soft constraint: prefer explanations with examples
        dspy.Suggest(
            'example' in result.explanation.lower() or 'for instance' in result.explanation.lower(),
            "Consider including a concrete example in your explanation"
        )
        
        return result
```

**Behavior on Failure**:
1. Logs the suggestion
2. Optionally triggers retry (configurable)
3. **Never** raises an error—program continues

**Use Cases**:
- Style preferences
- Optional improvements
- Non-critical quality guidelines

---

## Common Validation Patterns

### Length Constraints

```python
def validate_length(text, min_len=10, max_len=500):
    """Validate text length."""
    word_count = len(text.split())
    
    dspy.Assert(
        word_count >= min_len,
        f"Response too short. Minimum {min_len} words required, got {word_count}."
    )
    
    dspy.Assert(
        word_count <= max_len,
        f"Response too long. Maximum {max_len} words allowed, got {word_count}."
    )

class LengthConstrainedWriter(dspy.Module):
    def forward(self, prompt):
        result = self.write(prompt=prompt)
        validate_length(result.text, min_len=50, max_len=200)
        return result
```

### Format Constraints

```python
import json
import re

class JSONOutputGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict('request -> json_output')
    
    def forward(self, request):
        result = self.generate(request=request)
        
        # Validate JSON format
        try:
            parsed = json.loads(result.json_output)
            dspy.Assert(True, "")  # Assertion passes
        except json.JSONDecodeError as e:
            dspy.Assert(
                False,
                f"Output must be valid JSON. Error: {str(e)}. "
                f"Received: {result.json_output[:100]}..."
            )
        
        return parsed


class EmailGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought('request -> email_address')
    
    def forward(self, request):
        result = self.generate(request=request)
        
        # Validate email format
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        dspy.Assert(
            re.match(email_pattern, result.email_address),
            f"Invalid email format: {result.email_address}. "
            "Email must be in format: user@domain.tld"
        )
        
        return result
```

### Content Constraints

```python
class SafeContentGenerator(dspy.Module):
    FORBIDDEN_WORDS = ['confidential', 'secret', 'password', 'private']
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought('topic -> content')
    
    def forward(self, topic):
        result = self.generate(topic=topic)
        
        # Check for forbidden words
        found_forbidden = [
            word for word in self.FORBIDDEN_WORDS 
            if word in result.content.lower()
        ]
        
        dspy.Assert(
            len(found_forbidden) == 0,
            f"Content contains forbidden words: {found_forbidden}. "
            "Please rephrase without using these terms."
        )
        
        return result
```

### Uniqueness Constraints

```python
class UniqueListGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought('topic -> items: list[str]')
    
    def forward(self, topic):
        result = self.generate(topic=topic)
        
        # Check for duplicates
        items = result.items
        unique_items = set(items)
        
        dspy.Assert(
            len(items) == len(unique_items),
            f"List contains duplicates. Found {len(items)} items but only "
            f"{len(unique_items)} unique. Please provide all unique items."
        )
        
        return result
```

---

## GNN-Specific Assertions

### GNN Syntax Validation

```python
class GNNSyntaxValidator:
    """Validate GNN syntax patterns."""
    
    VARIABLE_PATTERN = r'^[a-zA-Z_][a-zA-Z0-9_]*\[\d+(?:,\d+)*(?:,type=[a-zA-Z]+)?\]'
    OBSERVATION_PATTERN = r'^o\[[a-zA-Z_][a-zA-Z0-9_]*\]\s*=\s*.+'
    
    @classmethod
    def validate_variable(cls, var_def):
        return bool(re.match(cls.VARIABLE_PATTERN, var_def))
    
    @classmethod
    def validate_observation(cls, obs_def):
        return bool(re.match(cls.OBSERVATION_PATTERN, obs_def))


class GNNObservationParser(dspy.Module):
    """Parse natural language into GNN observations with validation."""
    
    def __init__(self):
        super().__init__()
        self.parse = dspy.ChainOfThought(
            'user_input, observation_schema -> symbolic_observation'
        )
    
    def forward(self, user_input, schema):
        result = self.parse(user_input=user_input, observation_schema=schema)
        
        # Validate GNN syntax
        dspy.Assert(
            GNNSyntaxValidator.validate_observation(result.symbolic_observation),
            f"Invalid GNN observation syntax: '{result.symbolic_observation}'. "
            f"Expected format: o[name] = value. "
            f"Schema: {schema}"
        )
        
        return result


class GNNVariableGenerator(dspy.Module):
    """Generate GNN variable definitions."""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(
            'component_desc, existing_vars -> variable_definition'
        )
    
    def forward(self, description, existing_variables):
        result = self.generate(
            component_desc=description,
            existing_vars=existing_variables
        )
        
        # Validate syntax
        dspy.Assert(
            GNNSyntaxValidator.validate_variable(result.variable_definition),
            f"Invalid GNN variable syntax: '{result.variable_definition}'. "
            "Expected: name[dim1,dim2,...,type=T] format"
        )
        
        # Check for name collisions
        var_name = result.variable_definition.split('[')[0]
        existing_names = [v.split('[')[0] for v in existing_variables]
        
        dspy.Assert(
            var_name not in existing_names,
            f"Variable name '{var_name}' already exists. "
            f"Existing variables: {existing_names}. Choose a unique name."
        )
        
        return result
```

### Matrix Dimension Validation

```python
class GNNMatrixGenerator(dspy.Module):
    """Generate GNN matrices with dimension validation."""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(
            'matrix_type, state_dim: int, obs_dim: int, description -> matrix_values: list'
        )
    
    def forward(self, matrix_type, state_dim, obs_dim, description):
        result = self.generate(
            matrix_type=matrix_type,
            state_dim=state_dim,
            obs_dim=obs_dim,
            description=description
        )
        
        matrix = result.matrix_values
        
        # Validate dimensions based on matrix type
        if matrix_type == 'A':  # Observation model
            expected_shape = (obs_dim, state_dim)
            actual_shape = (len(matrix), len(matrix[0]) if matrix else 0)
            
            dspy.Assert(
                actual_shape == expected_shape,
                f"A matrix dimensions incorrect. Expected {expected_shape}, "
                f"got {actual_shape}. A matrix maps states to observations."
            )
            
        elif matrix_type == 'B':  # Transition model
            expected_shape = (state_dim, state_dim)
            actual_shape = (len(matrix), len(matrix[0]) if matrix else 0)
            
            dspy.Assert(
                actual_shape == expected_shape,
                f"B matrix dimensions incorrect. Expected {expected_shape}, "
                f"got {actual_shape}. B matrix maps states to next states."
            )
        
        # Validate probability normalization for A and B matrices
        for i, row in enumerate(matrix):
            row_sum = sum(row)
            dspy.Assert(
                abs(row_sum - 1.0) < 0.001,
                f"Row {i} does not sum to 1.0 (got {row_sum}). "
                "Each row must be a valid probability distribution."
            )
        
        return result
```

---

## dspy.Refine: Alternative to Assertions

For some use cases, `dspy.Refine` provides an alternative approach that samples multiple outputs and selects the best one.

```python
def quality_score(response):
    """Score response quality for Refine selection."""
    score = 0
    
    # Length check
    if 50 <= len(response.summary.split()) <= 100:
        score += 1
    
    # Contains key elements
    if 'conclusion' in response.summary.lower():
        score += 1
    
    # No forbidden words
    forbidden = ['maybe', 'perhaps', 'unclear']
    if not any(word in response.summary.lower() for word in forbidden):
        score += 1
    
    return score

# Use Refine instead of Assert
summarizer = dspy.Refine(
    module=dspy.ChainOfThought('document -> summary'),
    reward_fn=quality_score,
    N=5  # Generate 5 candidates, select best
)

result = summarizer(document="...")
```

**When to Use Refine vs Assert**:

| Scenario | Use Assert | Use Refine |
|----------|------------|------------|
| Must satisfy constraint | ✅ | ❌ |
| Prefer best among valid | ❌ | ✅ |
| Hard format requirements | ✅ | ❌ |
| Quality optimization | ❌ | ✅ |
| Cost-sensitive | ✅ | ❌ |

---

## Combining Assertions with Optimization

Assertions work with DSPy optimizers:

```python
class ConstrainedSummarizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.summarize = dspy.ChainOfThought('article -> summary')
    
    def forward(self, article):
        result = self.summarize(article=article)
        
        # Assertions that the optimizer will learn to satisfy
        dspy.Assert(
            len(result.summary.split()) <= 100,
            "Summary must be 100 words or fewer"
        )
        
        dspy.Suggest(
            result.summary.endswith('.'),
            "Summary should end with a period"
        )
        
        return result

# Metric that accounts for assertion satisfaction
def metric(example, prediction, trace=None):
    # Check if constraints were satisfied without retries
    assertion_cost = 0
    if trace:
        assertion_cost = trace.get('assertion_retries', 0) * 0.1
    
    # Base correctness score
    correctness = 1.0 if example.key_point in prediction.summary else 0.0
    
    return correctness - assertion_cost

# Optimization will learn to satisfy constraints naturally
optimizer = dspy.MIPROv2(metric=metric)
optimized = optimizer.compile(ConstrainedSummarizer(), trainset)
```

---

## Best Practices

### 1. Write Clear Error Messages

```python
# Good: Specific and actionable
dspy.Assert(
    len(items) >= 5,
    f"Need at least 5 items, got {len(items)}. "
    "Please expand your list with more relevant examples."
)

# Bad: Vague
dspy.Assert(len(items) >= 5, "Not enough items")
```

### 2. Order Assertions by Importance

```python
def forward(self, request):
    result = self.generate(request=request)
    
    # Check critical constraints first
    dspy.Assert(result.output is not None, "Output cannot be empty")
    
    # Then format constraints
    dspy.Assert(is_valid_json(result.output), "Must be valid JSON")
    
    # Finally, soft suggestions
    dspy.Suggest(len(result.output) > 50, "Consider providing more detail")
    
    return result
```

### 3. Use Appropriate Max Retries

```python
# For critical constraints, allow more retries
dspy.Assert(
    constraint,
    message,
    max_retries=5  # More chances for important constraints
)

# For simpler constraints, fewer retries
dspy.Assert(
    simple_constraint,
    message,
    max_retries=2
)
```

### 4. Combine with Type Hints

See [Typed Predictors Guide](dspy_typed_predictors.md) for using Pydantic with assertions.

---

## Debugging Assertions

### Viewing Assertion Traces

```python
import logging
logging.getLogger('dspy').setLevel(logging.DEBUG)

# Now assertion failures and retries will be logged
result = constrained_module(input="test")
```

### MLflow Integration

```python
import mlflow

mlflow.dspy.autolog()

with mlflow.start_run():
    result = constrained_module(input="test")
    # Assertion traces visible in MLflow UI
```

---

## Related Resources

- **[Modules Reference](dspy_modules_reference.md)**: Core module documentation
- **[Typed Predictors](dspy_typed_predictors.md)**: Structured output validation
- **[Optimizers Guide](dspy_optimizers_guide.md)**: Optimization with constraints
- **[GNN Integration](dspy_gnn_integration_patterns.md)**: GNN validation patterns

---

**Status**: ✅ Production Ready  
**Compliance**: GNN documentation standards  
**Maintenance**: Regular updates with new DSPy assertion features
