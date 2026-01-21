# DSPy-GNN Integration Patterns

> **📋 Document Metadata**  
> **Type**: Integration Guide | **Audience**: Developers, Researchers | **Complexity**: Advanced  
> **Cross-References**: [gnn_dspy.md](gnn_dspy.md) | [Modules Reference](dspy_modules_reference.md) | [LLM Integration](../gnn/gnn_llm_neurosymbolic_active_inference.md)

## Overview

This guide provides practical integration patterns for combining DSPy's programmatic LLM capabilities with GNN (Generalized Notation Notation) for neurosymbolic Active Inference systems. It complements the theoretical discussion in [gnn_dspy.md](gnn_dspy.md) with implementation-focused examples.

**Status**: ✅ Production Ready  
**Version**: 1.0

---

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      GNN-DSPy Integration                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Natural    │    │    DSPy      │    │     GNN      │      │
│  │   Language   │ → │   Modules    │ → │    Model     │      │
│  │   Input      │    │              │    │              │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         │                   │                   ▼               │
│         │                   │          ┌──────────────┐        │
│         │                   │          │   Active     │        │
│         │                   │          │  Inference   │        │
│         │                   │          │   Engine     │        │
│         │                   │          └──────────────┘        │
│         │                   │                   │               │
│         │                   │                   ▼               │
│         │                   │          ┌──────────────┐        │
│         │                   └──────── │   Actions    │        │
│         └────────────────────────── │  & Outputs   │        │
│                                        └──────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Pattern 1: Observation Processing

Convert natural language observations into GNN-compatible symbolic format.

### Implementation

```python
import dspy
from pydantic import BaseModel, Field
from typing import List, Dict, Union

class GNNObservation(BaseModel):
    """Structured GNN observation."""
    variable: str = Field(description="Observation variable name (e.g., o_position)")
    value: Union[bool, int, float, str]
    confidence: float = Field(default=1.0, ge=0, le=1)

class ObservationSet(BaseModel):
    """Complete observation set for a timestep."""
    observations: List[GNNObservation]
    raw_input: str

class GNNObservationProcessor(dspy.Module):
    """Process natural language into GNN observations."""
    
    def __init__(self, observation_schema: Dict[str, str]):
        super().__init__()
        self.schema = observation_schema
        
        self.extract = dspy.ChainOfThought(
            'user_input, observation_schema -> observations: ObservationSet'
        )
        
        self.validate = dspy.Predict(
            'observation, valid_variables -> is_valid: bool, error_message'
        )
    
    def forward(self, user_input: str) -> ObservationSet:
        # Extract observations
        result = self.extract(
            user_input=user_input,
            observation_schema=str(self.schema)
        )
        
        obs_set = result.observations
        
        # Validate each observation
        for obs in obs_set.observations:
            validation = self.validate(
                observation=str(obs),
                valid_variables=list(self.schema.keys())
            )
            
            dspy.Assert(
                validation.is_valid,
                f"Invalid observation: {obs.variable}. {validation.error_message}"
            )
        
        return obs_set

# Usage
schema = {
    'o_position': 'The current position (near, far)',
    'o_obstacle': 'Whether an obstacle is detected (true/false)',
    'o_target_visible': 'Whether the target is visible (true/false)'
}

processor = GNNObservationProcessor(schema)
obs = processor("I can see the target in the distance, but there's a wall in the way")
print(obs.observations)
# [GNNObservation(variable='o_target_visible', value=True, confidence=0.9),
#  GNNObservation(variable='o_obstacle', value=True, confidence=1.0)]
```

---

## Pattern 2: Goal Translation

Translate high-level natural language goals into GNN preference parameters.

### Implementation

```python
from pydantic import BaseModel, Field
from typing import List, Dict

class GNNPreference(BaseModel):
    """GNN preference specification (C matrix row)."""
    state_name: str
    preference_value: float = Field(ge=-10, le=10, description="Log preference")
    
class GoalTranslation(BaseModel):
    """Complete goal translation result."""
    preferences: List[GNNPreference]
    reasoning: str
    confidence: float

class GNNGoalTranslator(dspy.Module):
    """Translate natural language goals to GNN preferences."""
    
    def __init__(self, state_space: List[str]):
        super().__init__()
        self.state_space = state_space
        
        self.translate = dspy.ChainOfThought(
            'goal_description, available_states -> translation: GoalTranslation'
        )
    
    def forward(self, goal: str) -> GoalTranslation:
        result = self.translate(
            goal_description=goal,
            available_states=self.state_space
        )
        
        translation = result.translation
        
        # Validate all referenced states exist
        for pref in translation.preferences:
            dspy.Assert(
                pref.state_name in self.state_space,
                f"Unknown state: {pref.state_name}. Valid states: {self.state_space}"
            )
        
        return translation

# Usage
state_space = ['at_home', 'at_work', 'at_gym', 'at_store']
translator = GNNGoalTranslator(state_space)

translation = translator("I want to be at the gym by evening")
print(translation.preferences)
# [GNNPreference(state_name='at_gym', preference_value=5.0)]
```

---

## Pattern 3: Policy Semantic Evaluation

Evaluate GNN policies using LLM semantic understanding.

### Implementation

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class PolicyAnalysis(BaseModel):
    """Semantic analysis of a policy."""
    semantic_score: float = Field(ge=0, le=1)
    goal_alignment: str
    potential_issues: List[str]
    suggested_improvements: Optional[List[str]]

class GNNPolicyEvaluator(dspy.Module):
    """Evaluate policies using semantic understanding."""
    
    def __init__(self):
        super().__init__()
        
        self.analyze_policy = dspy.ChainOfThought(
            'policy_description, predicted_trajectory, user_goal -> analysis: PolicyAnalysis'
        )
        
        self.compare_policies = dspy.ChainOfThought(
            'policies: list[str], goal -> best_policy_index: int, reasoning'
        )
    
    def evaluate(self, policy: Dict, trajectory: List[str], goal: str) -> PolicyAnalysis:
        return self.analyze_policy(
            policy_description=str(policy),
            predicted_trajectory=str(trajectory),
            user_goal=goal
        ).analysis
    
    def rank_policies(self, policies: List[Dict], goal: str) -> int:
        """Return index of best policy."""
        policy_descriptions = [str(p) for p in policies]
        result = self.compare_policies(
            policies=policy_descriptions,
            goal=goal
        )
        return result.best_policy_index

# Usage
evaluator = GNNPolicyEvaluator()

policy = {
    'name': 'explore_then_exploit',
    'actions': ['move_random', 'check_environment', 'move_to_target']
}
trajectory = ['at_start', 'explored_area_A', 'found_target', 'at_target']

analysis = evaluator.evaluate(
    policy, 
    trajectory, 
    "Find and reach the target efficiently"
)
print(f"Semantic score: {analysis.semantic_score}")
print(f"Issues: {analysis.potential_issues}")
```

---

## Pattern 4: GNN Model Authoring Assistant

Interactive assistant for creating GNN model specifications.

### Implementation

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class GNNModelComponent(BaseModel):
    """A component of a GNN model."""
    component_type: str  # 'variable', 'matrix', 'connection'
    gnn_syntax: str
    description: str

class GNNModelDraft(BaseModel):
    """Draft GNN model specification."""
    model_name: str
    state_variables: List[GNNModelComponent]
    observation_variables: List[GNNModelComponent]
    matrices: List[GNNModelComponent]
    comments: List[str]
    
    def to_gnn_code(self) -> str:
        lines = [f"### GNN Model: {self.model_name}"]
        lines.append("\n## States")
        for s in self.state_variables:
            lines.append(f"{s.gnn_syntax}  # {s.description}")
        lines.append("\n## Observations")
        for o in self.observation_variables:
            lines.append(f"{o.gnn_syntax}  # {o.description}")
        lines.append("\n## Matrices")
        for m in self.matrices:
            lines.append(f"{m.gnn_syntax}")
        return "\n".join(lines)

class GNNAuthoringAssistant(dspy.Module):
    """Assist in authoring GNN models from natural language."""
    
    def __init__(self):
        super().__init__()
        
        self.understand_requirements = dspy.ChainOfThought(
            'user_description -> '
            'model_purpose, '
            'required_states: list[str], '
            'required_observations: list[str], '
            'clarifying_questions: list[str]'
        )
        
        self.generate_model = dspy.ChainOfThought(
            'requirements, constraints -> draft: GNNModelDraft'
        )
        
        self.refine_model = dspy.ChainOfThought(
            'current_model, feedback -> refined_model: GNNModelDraft'
        )
    
    def analyze(self, description: str) -> Dict:
        """Analyze user requirements."""
        return self.understand_requirements(user_description=description)
    
    def generate(self, requirements: str, constraints: str = "") -> GNNModelDraft:
        """Generate initial model draft."""
        result = self.generate_model(
            requirements=requirements,
            constraints=constraints or "Follow GNN syntax specification"
        )
        return result.draft
    
    def refine(self, model: GNNModelDraft, feedback: str) -> GNNModelDraft:
        """Refine model based on feedback."""
        result = self.refine_model(
            current_model=model.to_gnn_code(),
            feedback=feedback
        )
        return result.refined_model

# Usage
assistant = GNNAuthoringAssistant()

# Step 1: Analyze requirements
analysis = assistant.analyze(
    "I need a model for a simple navigation task where an agent "
    "moves between locations and observes walls"
)

if analysis.clarifying_questions:
    print("Questions:", analysis.clarifying_questions)

# Step 2: Generate model
model = assistant.generate(
    requirements=f"States: {analysis.required_states}, "
                 f"Observations: {analysis.required_observations}",
    constraints="Use discrete state space, 4 possible positions"
)

# Step 3: View and refine
print(model.to_gnn_code())

# Step 4: Iterate with feedback
refined = assistant.refine(
    model,
    "Add a state for whether the agent has found the goal"
)
```

---

## Pattern 5: Explanation Generation

Generate natural language explanations of GNN model behavior.

### Implementation

```python
from pydantic import BaseModel, Field
from typing import List, Dict

class InferenceExplanation(BaseModel):
    """Explanation of inference results."""
    summary: str
    state_interpretation: str
    belief_description: str
    action_rationale: str
    confidence_level: str

class GNNExplainer(dspy.Module):
    """Generate explanations for GNN inference results."""
    
    def __init__(self):
        super().__init__()
        
        self.explain_state = dspy.ChainOfThought(
            'state_distribution, state_meanings -> interpretation'
        )
        
        self.explain_action = dspy.ChainOfThought(
            'chosen_action, efe_values, goal -> rationale'
        )
        
        self.generate_summary = dspy.ChainOfThought(
            'state_info, action_info, context -> explanation: InferenceExplanation'
        )
    
    def explain(
        self,
        beliefs: Dict[str, float],
        action: str,
        efe_values: Dict[str, float],
        state_meanings: Dict[str, str],
        goal: str
    ) -> InferenceExplanation:
        
        # Explain current beliefs
        state_explanation = self.explain_state(
            state_distribution=str(beliefs),
            state_meanings=str(state_meanings)
        ).interpretation
        
        # Explain action choice
        action_explanation = self.explain_action(
            chosen_action=action,
            efe_values=str(efe_values),
            goal=goal
        ).rationale
        
        # Generate complete explanation
        result = self.generate_summary(
            state_info=state_explanation,
            action_info=action_explanation,
            context=f"Goal: {goal}"
        )
        
        return result.explanation

# Usage
explainer = GNNExplainer()

explanation = explainer.explain(
    beliefs={'at_start': 0.1, 'in_corridor': 0.7, 'at_goal': 0.2},
    action='move_forward',
    efe_values={'move_forward': -2.5, 'turn_left': -1.0, 'stay': 0.0},
    state_meanings={
        'at_start': 'Agent is at the starting position',
        'in_corridor': 'Agent is navigating through corridor',
        'at_goal': 'Agent has reached the goal'
    },
    goal='Reach the goal location'
)

print(explanation.summary)
print(explanation.action_rationale)
```

---

## Pattern 6: Multi-Step Active Inference Loop

Integrate DSPy with a complete Active Inference loop.

### Implementation

```python
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np

class ActiveInferenceStep(BaseModel):
    """State of one Active Inference step."""
    timestep: int
    observations: List[str]
    beliefs: Dict[str, float]
    selected_action: str
    predicted_next_state: str

class GNNActiveInferenceAgent(dspy.Module):
    """DSPy-enhanced Active Inference agent."""
    
    def __init__(self, gnn_engine, state_space: List[str]):
        super().__init__()
        self.gnn_engine = gnn_engine
        self.state_space = state_space
        
        # DSPy modules for semantic enhancement
        self.observation_processor = GNNObservationProcessor(
            observation_schema=self._build_obs_schema()
        )
        
        self.policy_evaluator = GNNPolicyEvaluator()
        
        self.explainer = GNNExplainer()
        
        # For goal understanding
        self.goal_translator = dspy.ChainOfThought(
            'natural_goal -> formal_preferences'
        )
    
    def _build_obs_schema(self) -> Dict[str, str]:
        # Build from GNN model
        return self.gnn_engine.get_observation_schema()
    
    def perceive(self, observation_text: str) -> np.ndarray:
        """Process natural language observation into GNN format."""
        obs_set = self.observation_processor(observation_text)
        return self.gnn_engine.encode_observations(obs_set.observations)
    
    def update_beliefs(self, observation: np.ndarray) -> Dict[str, float]:
        """Run GNN belief update."""
        beliefs = self.gnn_engine.infer_states(observation)
        return {s: float(p) for s, p in zip(self.state_space, beliefs)}
    
    def select_action(
        self, 
        beliefs: Dict[str, float], 
        goal: str
    ) -> tuple[str, Dict[str, float]]:
        """Select action using GNN + semantic evaluation."""
        
        # Get candidate policies from GNN
        candidates = self.gnn_engine.get_candidate_policies()
        
        # Calculate EFE for each
        efe_values = {}
        for policy in candidates:
            efe = self.gnn_engine.calculate_efe(policy, beliefs)
            efe_values[policy['name']] = efe
        
        # Use semantic evaluation to refine selection
        best_idx = self.policy_evaluator.rank_policies(candidates, goal)
        best_policy = candidates[best_idx]
        
        return best_policy['name'], efe_values
    
    def step(
        self, 
        observation_text: str, 
        goal: str,
        explain: bool = False
    ) -> ActiveInferenceStep:
        """Execute one step of Active Inference."""
        
        # 1. Process observation
        obs = self.perceive(observation_text)
        
        # 2. Update beliefs
        beliefs = self.update_beliefs(obs)
        
        # 3. Select action
        action, efe_values = self.select_action(beliefs, goal)
        
        # 4. Predict next state
        predicted = self.gnn_engine.predict_state(beliefs, action)
        predicted_state = max(predicted.items(), key=lambda x: x[1])[0]
        
        step_result = ActiveInferenceStep(
            timestep=self.gnn_engine.current_timestep,
            observations=[observation_text],
            beliefs=beliefs,
            selected_action=action,
            predicted_next_state=predicted_state
        )
        
        # 5. Optionally explain
        if explain:
            explanation = self.explainer.explain(
                beliefs=beliefs,
                action=action,
                efe_values=efe_values,
                state_meanings=self.gnn_engine.get_state_meanings(),
                goal=goal
            )
            print(f"Explanation: {explanation.summary}")
        
        return step_result

# Usage
# agent = GNNActiveInferenceAgent(gnn_engine, state_space)
# step = agent.step("I can see a door ahead", "Navigate to the exit", explain=True)
```

---

## Pattern 7: Optimization for GNN Tasks

Optimize DSPy modules for GNN-specific tasks.

### Implementation

```python
# Define GNN-specific metrics
def gnn_syntax_accuracy(example, prediction, trace=None):
    """Measure GNN syntax correctness."""
    try:
        # Try to parse as GNN
        is_valid = validate_gnn_syntax(prediction.gnn_code)
        return 1.0 if is_valid else 0.0
    except:
        return 0.0

def semantic_alignment_metric(example, prediction, trace=None):
    """Measure semantic alignment with expected behavior."""
    # Use LLM to judge alignment
    judge = dspy.Predict('expected, actual -> alignment: float')
    result = judge(
        expected=example.expected_behavior,
        actual=prediction.behavior_description
    )
    return result.alignment

def combined_gnn_metric(example, prediction, trace=None):
    """Combined metric for GNN tasks."""
    syntax_score = gnn_syntax_accuracy(example, prediction, trace)
    semantic_score = semantic_alignment_metric(example, prediction, trace)
    
    # Syntax is required, semantic is bonus
    if syntax_score == 0:
        return 0.0
    return 0.6 * syntax_score + 0.4 * semantic_score

# Optimize observation processor
from dspy.teleprompt import MIPROv2

optimizer = MIPROv2(
    metric=combined_gnn_metric,
    num_candidates=10
)

trainset = [
    dspy.Example(
        user_input="I see a wall on my left",
        expected_observation="o[wall_left]=true",
        expected_behavior="Detect obstacle to the left"
    ).with_inputs('user_input')
    for _ in observation_examples
]

optimized_processor = optimizer.compile(
    student=GNNObservationProcessor(schema),
    trainset=trainset
)
```

---

## Best Practices

### 1. Validate GNN Syntax Early

```python
def validate_before_execution(gnn_code: str) -> bool:
    """Always validate before sending to GNN engine."""
    dspy.Assert(
        is_valid_gnn_syntax(gnn_code),
        f"Invalid GNN syntax. Check specification."
    )
    return True
```

### 2. Use Type-Safe Outputs for GNN Integration

```python
# Always use Pydantic models for GNN data
class GNNOutput(BaseModel):
    model_config = {'strict': True}  # Enforce strict types
```

### 3. Log All LLM-to-GNN Translations

```python
import logging

logger = logging.getLogger('gnn_dspy')

def log_translation(input_text: str, gnn_output: str):
    logger.info(f"Translation: '{input_text}' -> '{gnn_output}'")
```

### 4. Handle Uncertainty Explicitly

```python
class UncertainTranslation(BaseModel):
    translation: str
    confidence: float
    alternatives: List[str]
    
# Use confidence thresholds
if translation.confidence < 0.7:
    # Fall back to safer option or ask for clarification
    pass
```

---

## Related Resources

- **[gnn_dspy.md](gnn_dspy.md)**: Theoretical foundation
- **[Modules Reference](dspy_modules_reference.md)**: DSPy modules
- **[Typed Predictors](dspy_typed_predictors.md)**: Structured outputs
- **[GNN Overview](../gnn/gnn_overview.md)**: GNN specification
- **[LLM Integration](../gnn/gnn_llm_neurosymbolic_active_inference.md)**: Neurosymbolic architecture

---

**Status**: ✅ Production Ready  
**Compliance**: GNN documentation standards  
**Maintenance**: Regular updates with new integration patterns
