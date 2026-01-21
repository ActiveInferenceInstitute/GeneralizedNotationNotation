# DSPy Typed Predictors Guide

> **📋 Document Metadata**  
> **Type**: Technical Guide | **Audience**: Developers, ML Engineers | **Complexity**: Intermediate  
> **Cross-References**: [Modules Reference](dspy_modules_reference.md) | [Assertions Guide](dspy_assertions_guide.md) | [GNN Integration](dspy_gnn_integration_patterns.md)

## Overview

DSPy Typed Predictors enable structured output generation with strong type guarantees through integration with Pydantic. This guide covers schema definition, JSON output handling, validation strategies, and GNN-specific typed output patterns.

**Status**: ✅ Production Ready  
**Version**: 1.0

---

## Why Typed Outputs?

LLMs naturally produce unstructured text. For integration with structured systems like GNN, we need:

1. **Predictable Structure**: Consistent output formats
2. **Type Safety**: Guaranteed field types
3. **Validation**: Automatic checking of constraints
4. **Composability**: Outputs that integrate with downstream systems

```
Unstructured Output:
"The temperature is around 72 degrees and it's sunny outside"

Typed Output:
{
  "temperature": 72.0,
  "unit": "fahrenheit",
  "conditions": ["sunny"],
  "humidity": null
}
```

---

## Pydantic Integration

### Basic Schema Definition

```python
from pydantic import BaseModel, Field
from typing import List, Optional
import dspy

class WeatherReport(BaseModel):
    """Structured weather information."""
    temperature: float = Field(description="Temperature value")
    unit: str = Field(description="Temperature unit (celsius/fahrenheit)")
    conditions: List[str] = Field(description="Weather conditions")
    humidity: Optional[float] = Field(default=None, description="Humidity percentage")

# Use in signature
predictor = dspy.Predict('location, date -> report: WeatherReport')
result = predictor(location="San Francisco", date="2024-01-15")

# Access typed fields
print(result.report.temperature)  # 65.0
print(result.report.conditions)   # ['partly cloudy']
```

### Complex Nested Schemas

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Subtask(BaseModel):
    """A subtask within a larger task."""
    name: str
    description: str
    estimated_hours: float
    dependencies: List[str] = Field(default_factory=list)

class ProjectPlan(BaseModel):
    """Complete project plan structure."""
    title: str
    objective: str = Field(description="Main goal of the project")
    priority: Priority
    subtasks: List[Subtask]
    milestones: Dict[str, str] = Field(description="Milestone name to date mapping")
    risks: Optional[List[str]] = None
    
    class Config:
        use_enum_values = True

# Usage
planner = dspy.ChainOfThought('project_description -> plan: ProjectPlan')
result = planner(project_description="Build a recommendation system")
```

---

## Signatures with Typed Fields

### Type Annotations in Signatures

DSPy signatures support inline type annotations:

```python
# Primitive types
predictor = dspy.Predict('text -> sentiment: bool')
predictor = dspy.Predict('numbers: list[int] -> sum: int')
predictor = dspy.Predict('data -> analysis: dict')

# With descriptions
predictor = dspy.Predict(
    'article -> '
    'summary: str "A concise summary", '
    'topics: list[str] "Main topics covered", '
    'word_count: int "Approximate word count"'
)
```

### Class-Based Signatures

For more control, use class-based signature definitions:

```python
class SentimentSignature(dspy.Signature):
    """Analyze the sentiment of text."""
    
    text: str = dspy.InputField(desc="The text to analyze")
    sentiment: str = dspy.OutputField(desc="Sentiment: positive, negative, or neutral")
    confidence: float = dspy.OutputField(desc="Confidence score from 0 to 1")
    key_phrases: List[str] = dspy.OutputField(desc="Phrases influencing the sentiment")

predictor = dspy.ChainOfThought(SentimentSignature)
result = predictor(text="I absolutely love this product!")
```

---

## JSON Adapters

### JSONAdapter for Reliable Parsing

DSPy's JSONAdapter instructs the LM to return structured JSON:

```python
# Configure JSONAdapter globally
dspy.configure(adapter=dspy.JSONAdapter())

# Now all outputs will be JSON formatted
predictor = dspy.Predict('query -> data: dict')
result = predictor(query="List the planets")
# result.data is a Python dict, not a string
```

### ChatAdapter (Default)

The default adapter for conversational interaction:

```python
dspy.configure(adapter=dspy.ChatAdapter())

# Outputs are natural language
predictor = dspy.Predict('topic -> explanation')
result = predictor(topic="quantum computing")
```

### Custom Adapters

```python
class CustomAdapter(dspy.Adapter):
    def format_request(self, signature, inputs):
        # Custom request formatting
        pass
    
    def parse_response(self, signature, response):
        # Custom response parsing
        pass

dspy.configure(adapter=CustomAdapter())
```

---

## GNN-Specific Typed Outputs

### GNN Observation Schema

```python
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Union, Optional

class GNNObservation(BaseModel):
    """A single GNN observation."""
    name: str = Field(description="Observation variable name")
    value: Union[bool, int, float, str] = Field(description="Observed value")
    modality: str = Field(description="Observation modality (visual, auditory, etc.)")
    
    @validator('name')
    def validate_name(cls, v):
        if not v.startswith('o_'):
            return f'o_{v}'
        return v

class GNNObservationSet(BaseModel):
    """Complete set of observations at a timestep."""
    timestep: int
    observations: List[GNNObservation]
    raw_input: Optional[str] = Field(description="Original input that was parsed")

# Usage
class ObservationParser(dspy.Module):
    def __init__(self):
        super().__init__()
        self.parse = dspy.ChainOfThought(
            'natural_language_input, observation_schema -> observations: GNNObservationSet'
        )
    
    def forward(self, text, schema):
        return self.parse(
            natural_language_input=text,
            observation_schema=schema
        )
```

### GNN State Space Definition

```python
class GNNStateVariable(BaseModel):
    """Definition of a GNN state variable."""
    name: str
    dimensions: List[int]
    dtype: str = Field(default="float", description="Data type: float, int, bool")
    description: str
    
    def to_gnn_syntax(self) -> str:
        dims = ",".join(map(str, self.dimensions))
        return f"{self.name}[{dims},type={self.dtype}]"

class GNNStateSpace(BaseModel):
    """Complete GNN state space definition."""
    variables: List[GNNStateVariable]
    
    def to_gnn_syntax(self) -> str:
        return "\n".join(v.to_gnn_syntax() for v in self.variables)

# Generator
state_generator = dspy.ChainOfThought(
    'model_description -> state_space: GNNStateSpace'
)
result = state_generator(
    model_description="A navigation agent with position and orientation states"
)
print(result.state_space.to_gnn_syntax())
```

### GNN Matrix Specifications

```python
from pydantic import BaseModel, Field, validator
from typing import List

class GNNMatrix(BaseModel):
    """A GNN probability matrix."""
    name: str = Field(description="Matrix name (A, B, C, D, E)")
    rows: int
    cols: int
    values: List[List[float]]
    description: str
    
    @validator('values')
    def validate_dimensions(cls, v, values):
        if 'rows' in values and 'cols' in values:
            if len(v) != values['rows']:
                raise ValueError(f"Expected {values['rows']} rows, got {len(v)}")
            for row in v:
                if len(row) != values['cols']:
                    raise ValueError(f"Expected {values['cols']} cols, got {len(row)}")
        return v
    
    @validator('values')
    def validate_probabilities(cls, v):
        for row in v:
            row_sum = sum(row)
            if abs(row_sum - 1.0) > 0.001:
                raise ValueError(f"Row must sum to 1.0, got {row_sum}")
        return v

class GNNAMatrix(dspy.Module):
    """Generate A matrix (observation likelihood)."""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(
            'state_space, observation_space, description -> a_matrix: GNNMatrix'
        )
    
    def forward(self, states, observations, description):
        return self.generate(
            state_space=str(states),
            observation_space=str(observations),
            description=description
        )
```

---

## Validation and Error Handling

### Automatic Pydantic Validation

Pydantic automatically validates outputs:

```python
class StrictOutput(BaseModel):
    count: int = Field(ge=0, le=100)  # 0 <= count <= 100
    ratio: float = Field(ge=0, le=1)  # 0 <= ratio <= 1
    category: str = Field(regex=r'^[A-Z]{2,4}$')  # 2-4 uppercase letters

    class Config:
        validate_assignment = True

predictor = dspy.Predict('data -> result: StrictOutput')

try:
    result = predictor(data="some input")
except pydantic.ValidationError as e:
    print(f"Validation failed: {e}")
```

### Combining with DSPy Assertions

```python
class ValidatedGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(
            'request -> output: GNNMatrix'
        )
    
    def forward(self, request):
        result = self.generate(request=request)
        
        # Additional validation beyond Pydantic
        matrix = result.output
        
        dspy.Assert(
            matrix.name in ['A', 'B', 'C', 'D', 'E'],
            f"Invalid matrix name: {matrix.name}. Must be A, B, C, D, or E."
        )
        
        dspy.Assert(
            all(all(0 <= v <= 1 for v in row) for row in matrix.values),
            "All values must be probabilities between 0 and 1."
        )
        
        return result
```

### Retry on Validation Failure

```python
class RobustTypedPredictor(dspy.Module):
    def __init__(self, signature, output_type, max_retries=3):
        super().__init__()
        self.predictor = dspy.ChainOfThought(signature)
        self.output_type = output_type
        self.max_retries = max_retries
    
    def forward(self, **kwargs):
        for attempt in range(self.max_retries):
            try:
                result = self.predictor(**kwargs)
                # Validate against Pydantic model
                validated = self.output_type.model_validate(result.output)
                return dspy.Prediction(output=validated)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                # Add error context for retry
                kwargs['_previous_error'] = str(e)
```

---

## Advanced Patterns

### Discriminated Unions

Handle multiple possible output types:

```python
from typing import Union, Literal

class SuccessResult(BaseModel):
    status: Literal["success"] = "success"
    data: Dict
    message: str

class ErrorResult(BaseModel):
    status: Literal["error"] = "error"
    error_code: str
    error_message: str

class Result(BaseModel):
    result: Union[SuccessResult, ErrorResult] = Field(discriminator='status')

processor = dspy.ChainOfThought('input -> output: Result')
```

### Self-Describing Schemas

```python
class DynamicSchema(BaseModel):
    """Schema that describes its own structure."""
    schema_name: str
    fields: List[Dict[str, str]]  # name, type, description
    
    def to_pydantic_model(self):
        """Dynamically create a Pydantic model."""
        from pydantic import create_model
        
        field_definitions = {}
        for field in self.fields:
            field_type = eval(field['type'])  # Careful with eval!
            field_definitions[field['name']] = (field_type, Field(description=field['description']))
        
        return create_model(self.schema_name, **field_definitions)
```

### Streaming Typed Outputs

```python
class StreamingTypedGenerator(dspy.Module):
    """Generate typed output with streaming."""
    
    def __init__(self, output_type):
        super().__init__()
        self.output_type = output_type
    
    def forward(self, prompt):
        # Use streaming LM
        accumulated = ""
        for chunk in dspy.settings.lm.stream(prompt):
            accumulated += chunk
            
            # Try to parse partial JSON
            try:
                partial = self.output_type.model_validate_json(accumulated)
                yield partial
            except:
                continue
```

---

## Best Practices

### 1. Start with Simple Schemas

```python
# Good: Start simple
class BasicOutput(BaseModel):
    answer: str
    confidence: float

# Add complexity only when needed
class DetailedOutput(BaseModel):
    answer: str
    confidence: float
    sources: List[str]
    reasoning_steps: List[str]
```

### 2. Use Descriptive Field Descriptions

```python
class WellDocumentedOutput(BaseModel):
    """Clear documentation helps the LM understand what to generate."""
    
    summary: str = Field(
        description="A 2-3 sentence summary of the main points"
    )
    key_facts: List[str] = Field(
        description="List of 3-5 key facts, each as a single sentence"
    )
    sentiment: str = Field(
        description="Overall sentiment: 'positive', 'negative', or 'neutral'"
    )
```

### 3. Validate Early and Often

```python
class EarlyValidationModule(dspy.Module):
    def forward(self, input_data):
        # Validate input
        assert isinstance(input_data, dict), "Input must be a dictionary"
        
        result = self.process(input_data)
        
        # Validate intermediate results
        dspy.Assert(result.intermediate is not None, "Intermediate step failed")
        
        final = self.finalize(result)
        
        # Validate output
        assert isinstance(final.output, ExpectedType), "Type mismatch"
        
        return final
```

### 4. Handle Null/Optional Fields Gracefully

```python
class SafeOutput(BaseModel):
    required_field: str
    optional_field: Optional[str] = None
    
    @validator('optional_field', pre=True)
    def empty_string_to_none(cls, v):
        if v == "" or v == "null" or v == "None":
            return None
        return v
```

---

## Related Resources

- **[Modules Reference](dspy_modules_reference.md)**: Core module documentation
- **[Assertions Guide](dspy_assertions_guide.md)**: Constraint validation
- **[Optimizers Guide](dspy_optimizers_guide.md)**: Optimizing typed predictors
- **[GNN Integration](dspy_gnn_integration_patterns.md)**: GNN schema patterns

---

**Status**: ✅ Production Ready  
**Compliance**: GNN documentation standards  
**Maintenance**: Regular updates with new Pydantic integration features
