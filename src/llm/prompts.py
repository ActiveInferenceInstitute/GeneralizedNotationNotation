#!/usr/bin/env python3
"""
GNN Analysis Prompts for LLM Processing

This module contains structured prompts for analyzing GNN (Generalized Notation Notation) 
files using Large Language Models. Each prompt is designed for specific analysis tasks 
that help understand Active Inference generative models.
"""

from typing import Dict, List, Any
from enum import Enum

class PromptType(Enum):
    """Types of analysis prompts available for GNN processing."""
    EXPLAIN_MODEL = "explain_model"
    ANALYZE_STRUCTURE = "analyze_structure"
    SUMMARIZE_CONTENT = "summarize_content"
    IDENTIFY_COMPONENTS = "identify_components"
    MATHEMATICAL_ANALYSIS = "mathematical_analysis"
    PRACTICAL_APPLICATIONS = "practical_applications"
    COMPARE_MODELS = "compare_models"
    VALIDATE_SYNTAX = "validate_syntax"
    SUGGEST_IMPROVEMENTS = "suggest_improvements"
    EXTRACT_PARAMETERS = "extract_parameters"

# Base system message for all GNN analysis tasks
GNN_SYSTEM_MESSAGE = """You are an expert in Active Inference, Bayesian inference, and GNN (Generalized Notation Notation) specifications. You have deep knowledge of:

- Active Inference theory and mathematical foundations
- Generative models and probabilistic graphical models
- GNN syntax and semantic meaning
- Hidden states, observations, actions, and control variables
- A, B, C, D matrices in Active Inference contexts
- Expected Free Energy and belief updating
- Markov Decision Processes and POMDPs
- Scientific modeling and analysis

When analyzing GNN files, provide accurate, detailed, and scientifically rigorous explanations. Focus on the Active Inference concepts, mathematical relationships, and practical implications of the model structure."""

# Structured prompts for different analysis types
GNN_ANALYSIS_PROMPTS: Dict[PromptType, Dict[str, Any]] = {
    
    PromptType.EXPLAIN_MODEL: {
        "title": "Model Explanation and Overview",
        "system_message": GNN_SYSTEM_MESSAGE,
        "user_prompt": """Please analyze the following GNN specification and explain what this generative model does:

{gnn_content}

Provide a comprehensive explanation that covers:

1. **Model Purpose**: What real-world phenomenon or problem does this model represent?

2. **Core Components**: 
   - What are the hidden states (s_f0, s_f1, etc.) and what do they represent?
   - What are the observations (o_m0, o_m1, etc.) and what do they capture?
   - What actions/controls (u_c0, π_c0, etc.) are available and what do they do?

3. **Model Dynamics**: How does the model evolve over time? What are the key relationships?

4. **Active Inference Context**: How does this model implement Active Inference principles? What beliefs are being updated and how?

5. **Practical Implications**: What can you learn or predict using this model? What decisions can it inform?

Please write in clear, accessible language while maintaining scientific accuracy.""",
        "expected_output": "markdown",
        "max_tokens": 2000
    },
    
    PromptType.ANALYZE_STRUCTURE: {
        "title": "Structural Analysis and Graph Properties",
        "system_message": GNN_SYSTEM_MESSAGE,
        "user_prompt": """Analyze the structure and graph properties of this GNN specification:

{gnn_content}

Provide a detailed structural analysis covering:

1. **Graph Structure**:
   - Number of variables and their types
   - Connection patterns (directed/undirected edges)
   - Graph topology (hierarchical, network, etc.)

2. **Variable Analysis**:
   - State space dimensionality for each variable
   - Dependencies and conditional relationships
   - Temporal vs. static variables

3. **Mathematical Structure**:
   - Matrix dimensions and compatibility
   - Parameter structure and organization  
   - Symmetries or special properties

4. **Complexity Assessment**:
   - Computational complexity indicators
   - Model scalability considerations
   - Potential bottlenecks or challenges

5. **Design Patterns**:
   - What modeling patterns or templates does this follow?
   - How does the structure reflect the domain being modeled?""",
        "expected_output": "markdown",
        "max_tokens": 1800
    },
    
    PromptType.SUMMARIZE_CONTENT: {
        "title": "Content Summary and Key Points",
        "system_message": GNN_SYSTEM_MESSAGE,
        "user_prompt": """Provide a concise but comprehensive summary of this GNN specification:

{gnn_content}

Create a structured summary including:

1. **Model Overview** (2-3 sentences): What is this model and what does it do?

2. **Key Variables**:
   - Hidden states: [list with brief descriptions]
   - Observations: [list with brief descriptions]  
   - Actions/Controls: [list with brief descriptions]

3. **Critical Parameters**:
   - Most important matrices (A, B, C, D) and their roles
   - Key hyperparameters and their settings

4. **Notable Features**:
   - Special properties or constraints
   - Unique aspects of this model design

5. **Use Cases**: What scenarios would this model be applied to?

Keep the summary focused and informative, suitable for someone familiar with Active Inference but new to this specific model.""",
        "expected_output": "markdown",
        "max_tokens": 1200
    },
    
    PromptType.IDENTIFY_COMPONENTS: {
        "title": "Component Identification and Classification",
        "system_message": GNN_SYSTEM_MESSAGE,
        "user_prompt": """Identify and classify all components in this GNN specification:

{gnn_content}

Provide a systematic breakdown:

1. **State Variables (Hidden States)**:
   - Variable names and dimensions
   - What each state represents conceptually
   - State space structure (discrete/continuous, finite/infinite)

2. **Observation Variables**:
   - Observation modalities and their meanings
   - Sensor/measurement interpretations
   - Noise models or uncertainty characterization

3. **Action/Control Variables**:
   - Available actions and their effects
   - Control policies and decision variables
   - Action space properties

4. **Model Matrices**:
   - A matrices: Observation models P(o|s)
   - B matrices: Transition dynamics P(s'|s,u)
   - C matrices: Preferences/goals
   - D matrices: Prior beliefs over initial states

5. **Parameters and Hyperparameters**:
   - Precision parameters (γ, α, etc.)
   - Learning rates and adaptation parameters
   - Fixed vs. learnable parameters

6. **Temporal Structure**:
   - Time horizons and temporal dependencies
   - Dynamic vs. static components""",
        "expected_output": "markdown",
        "max_tokens": 1600
    },
    
    PromptType.MATHEMATICAL_ANALYSIS: {
        "title": "Mathematical Analysis and Formalism",
        "system_message": GNN_SYSTEM_MESSAGE,
        "user_prompt": """Provide a mathematical analysis of this GNN specification:

{gnn_content}

Focus on the mathematical formalism and relationships:

1. **Probabilistic Structure**:
   - Joint probability distribution P(s,o,u)
   - Conditional dependencies and factorization
   - Marginal and posterior distributions

2. **Free Energy Formulation**:
   - Variational free energy F = E_q[ln q(s) - ln p(s,o)]
   - Expected free energy G for policy evaluation
   - Information-theoretic quantities (entropy, divergence)

3. **Belief Updating**:
   - Posterior beliefs over hidden states
   - Belief propagation or message passing
   - Precision-weighted prediction errors

4. **Action Selection**:
   - Policy evaluation via expected free energy
   - Softmax action selection with precision
   - Exploration vs. exploitation trade-offs

5. **Learning Mechanisms**:
   - Parameter updates and learning rules
   - Bayesian model averaging or selection
   - Adaptation timescales

6. **Mathematical Properties**:
   - Convergence guarantees
   - Stability analysis
   - Theoretical foundations""",
        "expected_output": "markdown", 
        "max_tokens": 2000
    },
    
    PromptType.PRACTICAL_APPLICATIONS: {
        "title": "Practical Applications and Use Cases",
        "system_message": GNN_SYSTEM_MESSAGE,
        "user_prompt": """Analyze the practical applications and use cases for this GNN model:

{gnn_content}

Discuss practical considerations:

1. **Real-World Applications**:
   - What domains could this model be applied to?
   - Specific use cases and scenarios
   - Industry or research applications

2. **Implementation Considerations**:
   - Computational requirements and scalability
   - Data requirements and collection strategies
   - Integration with existing systems

3. **Performance Expectations**:
   - What kinds of performance can be expected?
   - Metrics for evaluation and validation
   - Limitations and failure modes

4. **Deployment Scenarios**:
   - Online vs. offline processing
   - Real-time constraints and requirements
   - Hardware and software dependencies

5. **Benefits and Advantages**:
   - What problems does this model solve well?
   - Unique capabilities or features
   - Comparison to alternative approaches

6. **Challenges and Considerations**:
   - Potential difficulties in implementation
   - Tuning and optimization requirements
   - Maintenance and monitoring needs""",
        "expected_output": "markdown",
        "max_tokens": 1600
    },
    
    PromptType.EXTRACT_PARAMETERS: {
        "title": "Parameter Extraction and Configuration",
        "system_message": GNN_SYSTEM_MESSAGE,
        "user_prompt": """Extract and organize all parameters from this GNN specification:

{gnn_content}

Provide a systematic parameter breakdown:

1. **Model Matrices**:
   - A matrices: dimensions, structure, interpretation
   - B matrices: dimensions, structure, interpretation  
   - C matrices: dimensions, structure, interpretation
   - D matrices: dimensions, structure, interpretation

2. **Precision Parameters**:
   - γ (gamma): precision parameters and their roles
   - α (alpha): learning rates and adaptation parameters
   - Other precision/confidence parameters

3. **Dimensional Parameters**:
   - State space dimensions for each factor
   - Observation space dimensions for each modality
   - Action space dimensions for each control factor

4. **Temporal Parameters**:
   - Time horizons (T)
   - Temporal dependencies and windows
   - Update frequencies and timescales

5. **Initial Conditions**:
   - Prior beliefs over initial states
   - Initial parameter values
   - Initialization strategies

6. **Configuration Summary**:
   - Parameter file format recommendations
   - Tunable vs. fixed parameters
   - Sensitivity analysis priorities""",
        "expected_output": "markdown",
        "max_tokens": 1400
    },
    
    PromptType.SUGGEST_IMPROVEMENTS: {
        "title": "Model Improvements and Recommendations", 
        "system_message": GNN_SYSTEM_MESSAGE,
        "user_prompt": """Analyze this GNN specification and suggest improvements:

{gnn_content}

Provide constructive recommendations:

1. **Model Structure Improvements**:
   - Are there missing variables or relationships?
   - Could the graph structure be optimized?
   - Suggestions for better factorization

2. **Parameter Optimization**:
   - Are parameter values reasonable?
   - Suggestions for better initialization
   - Hyperparameter tuning recommendations

3. **Computational Efficiency**:
   - Ways to reduce computational complexity
   - Approximation strategies
   - Scalability improvements

4. **Robustness Enhancements**:
   - Ways to improve model robustness
   - Error handling and edge cases
   - Regularization techniques

5. **Extension Possibilities**:
   - How could this model be extended?
   - Additional features or capabilities
   - Integration with other models

6. **Best Practices**:
   - Alignment with GNN best practices
   - Active Inference implementation standards
   - Documentation and maintainability""",
        "expected_output": "markdown",
        "max_tokens": 1600
    }
}

def get_prompt(prompt_type: PromptType, gnn_content: str, **kwargs) -> Dict[str, Any]:
    """
    Get a formatted prompt for GNN analysis.
    
    Args:
        prompt_type: Type of analysis prompt to retrieve
        gnn_content: The GNN file content to analyze
        **kwargs: Additional parameters for prompt formatting
        
    Returns:
        Dictionary containing formatted prompt and metadata
    """
    if prompt_type not in GNN_ANALYSIS_PROMPTS:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    prompt_config = GNN_ANALYSIS_PROMPTS[prompt_type].copy()
    
    # Format the user prompt with the GNN content
    prompt_config["user_prompt"] = prompt_config["user_prompt"].format(
        gnn_content=gnn_content,
        **kwargs
    )
    
    return prompt_config

def get_all_prompt_types() -> List[PromptType]:
    """Get all available prompt types."""
    return list(PromptType)

def get_prompt_title(prompt_type: PromptType) -> str:
    """Get the human-readable title for a prompt type."""
    return GNN_ANALYSIS_PROMPTS[prompt_type]["title"]

def get_default_prompt_sequence() -> List[PromptType]:
    """
    Get the default sequence of prompts for comprehensive GNN analysis.
    
    Returns:
        List of prompt types in recommended analysis order
    """
    return [
        PromptType.SUMMARIZE_CONTENT,
        PromptType.EXPLAIN_MODEL,
        PromptType.IDENTIFY_COMPONENTS,
        PromptType.ANALYZE_STRUCTURE,
        PromptType.EXTRACT_PARAMETERS,
        PromptType.PRACTICAL_APPLICATIONS
    ] 