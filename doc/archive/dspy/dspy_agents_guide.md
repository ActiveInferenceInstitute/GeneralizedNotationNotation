# DSPy Agents Guide

> **📋 Document Metadata**  
> **Type**: Development Guide | **Audience**: Developers, Researchers | **Complexity**: Advanced  
> **Cross-References**: [Modules Reference](dspy_modules_reference.md) | [Optimizers Guide](dspy_optimizers_guide.md) | [GNN Integration](dspy_gnn_integration_patterns.md)

## Overview

DSPy enables the construction of sophisticated AI agents that can reason, plan, and interact with external tools. This guide covers the `dspy.ReAct` module and advanced agent architectures for GNN-based Active Inference systems.

**Status**: ✅ Production Ready  
**Version**: 1.0

---

## The ReAct Paradigm

ReAct (Reasoning and Acting) is a framework that allows language models to interleave:
- **Thought**: Internal reasoning about the current state
- **Action**: Calling external tools or APIs
- **Observation**: Processing tool outputs

This creates a powerful loop for solving complex, multi-step problems.

```
┌─────────────┐
│   Thought   │ ← Analyze current state and plan next step
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Action    │ ← Execute a tool or finish
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Observation │ ← Process tool output
└──────┬──────┘
       │
       └──────→ (loop back to Thought)
```

---

## Building a Basic Agent

### Step 1: Define Tools

Tools are Python functions with docstrings that describe their behavior:

```python
def search_documents(query: str) -> list[str]:
    """Search the document database for relevant passages.
    
    Args:
        query: The search query string
        
    Returns:
        List of relevant document passages
    """
    # Implementation
    results = vector_db.search(query, k=5)
    return [r.text for r in results]

def calculate(expression: str) -> float:
    """Evaluate a mathematical expression safely.
    
    Args:
        expression: Mathematical expression to evaluate
        
    Returns:
        The computed result
    """
    import ast
    return eval(compile(ast.parse(expression, mode='eval'), '', 'eval'))

def get_current_date() -> str:
    """Get the current date in YYYY-MM-DD format."""
    from datetime import date
    return date.today().isoformat()
```

### Step 2: Create the Agent

```python
import dspy

# Configure LM
lm = dspy.LM('openai/gpt-4o', temperature=0.7)
dspy.configure(lm=lm)

# Create ReAct agent
agent = dspy.ReAct(
    signature='question -> answer',
    tools=[search_documents, calculate, get_current_date],
    max_iters=5
)

# Run the agent
response = agent(question="What were the key findings in the 2024 AI safety report?")
print(response.answer)
```

---

## Advanced Agent Patterns

### Multi-Tool Agent with Memory

```python
class ResearchAgent(dspy.Module):
    """Agent that maintains conversation memory and uses multiple tools."""
    
    def __init__(self, tools, max_iters=10):
        super().__init__()
        self.tools = tools
        self.max_iters = max_iters
        
        # ReAct for tool use
        self.react = dspy.ReAct(
            signature='context, question -> answer',
            tools=tools,
            max_iters=max_iters
        )
        
        # Summarization for memory compression
        self.summarize = dspy.ChainOfThought('conversation_history -> summary')
        
        self.memory = []
    
    def forward(self, question):
        # Build context from memory
        context = self.summarize(conversation_history=str(self.memory[-5:])).summary if self.memory else ""
        
        # Run agent
        response = self.react(context=context, question=question)
        
        # Update memory
        self.memory.append({
            'question': question,
            'answer': response.answer
        })
        
        return response
```

### Hierarchical Agent Architecture

For complex tasks, agents can orchestrate other agents:

```python
class OrchestratorAgent(dspy.Module):
    """High-level agent that delegates to specialized sub-agents."""
    
    def __init__(self):
        super().__init__()
        
        # Planning agent
        self.planner = dspy.ChainOfThought(
            'task_description -> step_by_step_plan: list[str], required_capabilities: list[str]'
        )
        
        # Specialized agents
        self.research_agent = dspy.ReAct(
            signature='research_question -> findings',
            tools=[search_web, search_papers]
        )
        
        self.analysis_agent = dspy.ReAct(
            signature='data, analysis_request -> analysis_result',
            tools=[calculate, plot_data, statistical_test]
        )
        
        self.synthesis_agent = dspy.ChainOfThought(
            'all_findings: list[str] -> final_report'
        )
    
    def forward(self, task):
        # Create plan
        plan = self.planner(task_description=task)
        
        findings = []
        for step in plan.step_by_step_plan:
            if 'research' in step.lower():
                result = self.research_agent(research_question=step)
                findings.append(result.findings)
            elif 'analyz' in step.lower():
                result = self.analysis_agent(data=findings[-1], analysis_request=step)
                findings.append(result.analysis_result)
        
        # Synthesize results
        return self.synthesis_agent(all_findings=findings)
```

---

## GNN-Integrated Agents

### Active Inference Policy Agent

An agent that assists with GNN-based Active Inference policy selection:

```python
class GNNPolicyAgent(dspy.Module):
    """Agent for GNN-based policy evaluation and selection."""
    
    def __init__(self, gnn_tools):
        super().__init__()
        
        # Core agent for policy reasoning
        self.policy_evaluator = dspy.ReAct(
            signature='current_belief, candidate_policies, goal -> recommended_policy, reasoning',
            tools=gnn_tools,
            max_iters=5
        )
    
    def forward(self, belief_state, policies, goal):
        return self.policy_evaluator(
            current_belief=str(belief_state),
            candidate_policies=str(policies),
            goal=goal
        )

# Define GNN-specific tools
def evaluate_expected_free_energy(policy_id: str, gnn_model: str) -> dict:
    """Calculate the Expected Free Energy for a policy in the GNN model.
    
    Args:
        policy_id: Identifier of the policy to evaluate
        gnn_model: The GNN model specification
        
    Returns:
        Dict with pragmatic_value, epistemic_value, and total_efe
    """
    # Integration with GNN execution engine
    pass

def predict_future_state(policy_id: str, steps: int) -> str:
    """Predict future states under a given policy.
    
    Args:
        policy_id: The policy to simulate
        steps: Number of timesteps to predict
        
    Returns:
        Predicted state trajectory
    """
    pass

def check_gnn_syntax(gnn_snippet: str) -> dict:
    """Validate GNN syntax.
    
    Args:
        gnn_snippet: GNN code to validate
        
    Returns:
        Dict with is_valid and error_messages
    """
    pass

# Create the agent
gnn_agent = GNNPolicyAgent(
    gnn_tools=[evaluate_expected_free_energy, predict_future_state, check_gnn_syntax]
)
```

### GNN Model Authoring Assistant

```python
class GNNAuthoringAssistant(dspy.Module):
    """Interactive agent for GNN model creation."""
    
    def __init__(self):
        super().__init__()
        
        self.clarify = dspy.ChainOfThought(
            'user_description, current_model -> clarifying_questions: list[str]'
        )
        
        self.generate = dspy.ReAct(
            signature='model_requirements, constraints -> gnn_model_code',
            tools=[
                self.validate_gnn_syntax,
                self.suggest_state_space,
                self.suggest_observation_model,
                self.lookup_gnn_examples
            ]
        )
        
        self.refine = dspy.ChainOfThought(
            'current_model, validation_errors, user_feedback -> refined_model'
        )
    
    @staticmethod
    def validate_gnn_syntax(gnn_code: str) -> dict:
        """Validate GNN syntax and return errors if any."""
        # Integration with GNN type checker
        pass
    
    @staticmethod
    def suggest_state_space(description: str) -> str:
        """Suggest a state space structure based on description."""
        pass
    
    @staticmethod  
    def suggest_observation_model(states: str, observations: str) -> str:
        """Suggest an A-matrix structure for state-observation mapping."""
        pass
    
    @staticmethod
    def lookup_gnn_examples(model_type: str) -> list[str]:
        """Look up similar GNN model examples."""
        pass
    
    def forward(self, user_request, context=""):
        # First, ask clarifying questions if needed
        questions = self.clarify(
            user_description=user_request,
            current_model=context
        ).clarifying_questions
        
        if questions:
            return {'needs_clarification': True, 'questions': questions}
        
        # Generate model
        model = self.generate(
            model_requirements=user_request,
            constraints="Must follow GNN syntax specification"
        )
        
        return {'gnn_model': model.gnn_model_code}
```

---

## Tool Design Best Practices

### 1. Clear Docstrings

The LM uses docstrings to understand tool capabilities:

```python
def search_knowledge_base(
    query: str,
    category: str = "all",
    max_results: int = 5
) -> list[dict]:
    """Search the knowledge base for relevant information.
    
    Use this tool when you need to find specific facts, definitions,
    or background information about a topic.
    
    Args:
        query: Natural language search query
        category: Filter by category ('all', 'science', 'history', 'technology')
        max_results: Maximum number of results to return (1-20)
        
    Returns:
        List of dicts with 'title', 'content', and 'relevance_score' keys
        
    Example:
        search_knowledge_base("quantum computing principles", category="science")
    """
    pass
```

### 2. Informative Return Values

Return structured data that helps the agent reason:

```python
def check_system_status() -> dict:
    """Check the status of all system components."""
    return {
        'database': {'status': 'healthy', 'latency_ms': 45},
        'cache': {'status': 'healthy', 'hit_rate': 0.87},
        'api': {'status': 'degraded', 'error_rate': 0.02, 'message': 'High latency detected'}
    }
```

### 3. Error Handling

Handle errors gracefully with informative messages:

```python
def query_external_api(endpoint: str, params: dict) -> dict:
    """Query an external API endpoint."""
    try:
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        return {'success': True, 'data': response.json()}
    except requests.Timeout:
        return {'success': False, 'error': 'Request timed out after 10 seconds'}
    except requests.HTTPError as e:
        return {'success': False, 'error': f'HTTP error: {e.response.status_code}'}
```

---

## Observability with MLflow

MLflow provides tracing and experiment tracking for DSPy agents:

```python
import mlflow

# Setup MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("GNN-Agent-Development")

# Enable automatic tracing
mlflow.dspy.autolog()

# Now all agent runs are automatically logged
with mlflow.start_run():
    agent = dspy.ReAct(
        signature='question -> answer',
        tools=[search_documents, calculate]
    )
    
    response = agent(question="What is the capital of France?")
    
    # Log custom metrics
    mlflow.log_metric("response_length", len(response.answer))
```

### Viewing Traces

Access the MLflow UI at `http://localhost:5000` to view:
- Complete execution traces
- Tool call sequences
- Reasoning steps
- Token usage
- Latency breakdown

---

## Optimization for Agents

Agents can be optimized using DSPy optimizers:

```python
# Define evaluation metric
def agent_accuracy(example, prediction, trace=None):
    """Check if agent found correct information."""
    return example.expected_answer.lower() in prediction.answer.lower()

# Prepare training data
trainset = [
    dspy.Example(
        question="What is the population of Tokyo?",
        expected_answer="13.96 million"
    ).with_inputs('question')
    for _ in range(10)
]

# Optimize the agent
from dspy.teleprompt import MIPROv2

optimizer = MIPROv2(
    metric=agent_accuracy,
    num_candidates=10,
    init_temperature=1.0
)

optimized_agent = optimizer.compile(
    agent,
    trainset=trainset
)
```

**See also**: [DSPy Optimizers Guide](dspy_optimizers_guide.md)

---

## Agent Limitations and Mitigations

### Common Issues

| Issue | Mitigation |
|-------|------------|
| Infinite loops | Set `max_iters` appropriately |
| Wrong tool selection | Improve tool docstrings |
| Hallucinated tool calls | Use `dspy.Assert` for validation |
| High latency | Use caching, smaller models for routing |
| High cost | Limit iterations, use cheaper models for planning |

### Early Stopping

```python
class SafeAgent(dspy.Module):
    def __init__(self, base_agent, max_cost=0.10):
        super().__init__()
        self.agent = base_agent
        self.max_cost = max_cost
    
    def forward(self, question):
        # Track costs during execution
        initial_usage = dspy.settings.lm.usage.copy()
        
        try:
            result = self.agent(question=question)
        finally:
            # Check cost
            current_usage = dspy.settings.lm.usage
            cost = self._calculate_cost(initial_usage, current_usage)
            if cost > self.max_cost:
                raise RuntimeError(f"Agent exceeded cost limit: ${cost:.4f}")
        
        return result
```

---

## Related Resources

- **[Modules Reference](dspy_modules_reference.md)**: Complete module documentation
- **[Optimizers Guide](dspy_optimizers_guide.md)**: Agent optimization
- **[Assertions Guide](dspy_assertions_guide.md)**: Output validation for agents
- **[GNN Integration Patterns](dspy_gnn_integration_patterns.md)**: Active Inference agents
- **[MLflow Documentation](https://mlflow.org/docs/latest/llms/dspy/)**: Observability setup

---

**Status**: ✅ Production Ready  
**Compliance**: GNN documentation standards  
**Maintenance**: Regular updates with new DSPy agent features
