#!/usr/bin/env python3
"""
Enhanced LLM System Demonstration

This script demonstrates the enhanced LLM capabilities that integrate
the new multi-provider system with the existing LLM operations interface.
"""

import sys
from pathlib import Path
import logging
from contextlib import contextmanager

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the enhanced LLM operations
from src.llm.llm_operations import (
    LLMOperations,
    summarize_gnn,
    analyze_gnn_structure,
    generate_questions,
    enhance_gnn,
    validate_gnn
)

logger = logging.getLogger(__name__)


@contextmanager
def _demo_section(title: str) -> Any:
    """Context manager for demo sections with consistent error handling."""
    print(f"\n--- {title} ---")
    try:
        yield
    except Exception as e:
        print(f"Error in {title}: {e}")


# Sample GNN content for demonstration
SAMPLE_GNN_CONTENT = """
## GNNVersionAndFlags
GNNVersion: 1.0
Flags: None

## ModelName
HierarchicalAttentionModel

## ModelAnnotation
A hierarchical Active Inference model that implements selective attention
mechanisms across multiple levels of abstraction. This model demonstrates
how attention can be modeled as precision-weighted message passing in
a hierarchical generative model.

## StateSpaceBlock
# Level 1: Basic features
s_f0: [BasicFeatures] dimension(8) # Low-level perceptual features

# Level 2: Object representations  
s_f1: [Objects] dimension(4) # Mid-level object representations

# Level 3: Scene understanding
s_f2: [Scene] dimension(2) # High-level scene context

# Attention states
s_att: [AttentionState] dimension(4) # Attention allocation states

# Observations: Visual input
o_m0: [VisualInput] dimension(16) # Raw visual observations

# Actions: Eye movements and attention control
u_c0: [EyeMovement] dimension(4) # {Up, Down, Left, Right}
u_c1: [AttentionControl] dimension(2) # {Focus, Broaden}

## Connections
# Hierarchical state connections
s_f0 > s_f1  # Features influence objects
s_f1 > s_f2  # Objects influence scene understanding
s_f2 > s_f1  # Top-down scene context
s_f1 > s_f0  # Top-down object predictions

# Attention modulation
s_att > s_f0  # Attention modulates feature processing
s_att > s_f1  # Attention modulates object processing
s_f2 > s_att  # Scene context influences attention

# Observations and actions
s_f0 > o_m0  # Features generate observations
u_c0 > s_f0  # Eye movements affect feature sampling
u_c1 > s_att # Attention control affects attention state

# Temporal connections
s_f0 > s_f0  # Feature dynamics
s_f1 > s_f1  # Object dynamics
s_f2 > s_f2  # Scene dynamics
s_att > s_att # Attention dynamics

## InitialParameterization
# A matrices: Observation models
A_m0 = [[0.8, 0.1, 0.05, 0.05], [0.2, 0.7, 0.05, 0.05], [0.1, 0.2, 0.6, 0.1], [0.05, 0.05, 0.1, 0.8]]

# B matrices: Transition models with attention modulation
B_f0 = [[[0.9, 0.1], [0.1, 0.9]], [[0.8, 0.2], [0.2, 0.8]]]
B_f1 = [[[0.85, 0.15], [0.15, 0.85]], [[0.7, 0.3], [0.3, 0.7]]]
B_f2 = [[[0.95, 0.05], [0.05, 0.95]], [[0.9, 0.1], [0.1, 0.9]]]
B_att = [[[0.8, 0.2], [0.2, 0.8]], [[0.6, 0.4], [0.4, 0.6]]]

# C vectors: Preferences
C_m0 = [2.0, 1.0, 0.0, -1.0]  # Prefer clear, attended features

# D vectors: Priors
D_f0 = [0.25, 0.25, 0.25, 0.25]  # Uniform over features
D_f1 = [0.5, 0.3, 0.2]  # Biased toward simple objects
D_f2 = [0.7, 0.3]  # Prefer familiar scenes
D_att = [0.4, 0.3, 0.2, 0.1]  # Biased attention allocation

## Equations
# Attention-modulated precision
τ_att = γ_att * s_att

# Hierarchical prediction error
ε_1 = s_f1 - f(s_f0, τ_att)
ε_2 = s_f2 - g(s_f1)

# Free energy with attention costs
F = D_kl[q(s)|p(s)] + E_q[ln p(o|s)] + λ_att * ||u_c1||²

## Time
Dynamic: True
DiscreteTime: True
ModelTimeHorizon: 50
"""

def demo_backward_compatibility() -> None:
    """Demonstrate that existing code still works with enhanced system."""
    print("Testing Backward Compatibility...")

    with _demo_section("Testing convenience functions"):
        summary = summarize_gnn(SAMPLE_GNN_CONTENT, max_length=200)
        print(f"Summary generated ({len(summary)} chars): {summary[:100]}...")

        structure = analyze_gnn_structure(SAMPLE_GNN_CONTENT)
        print(f"Structure analysis completed ({len(structure)} chars)")

        questions = generate_questions(SAMPLE_GNN_CONTENT, num_questions=3)
        print(f"Generated {len(questions)} questions")
        for i, q in enumerate(questions, 1):
            print(f"   {i}. {q[:80]}...")

def demo_enhanced_capabilities() -> None:
    """Demonstrate new capabilities enabled by multi-provider system."""
    print("\nTesting Enhanced Capabilities...")

    with _demo_section("Enhancement suggestions"):
        enhancements = enhance_gnn(SAMPLE_GNN_CONTENT)
        print(f"Enhancement suggestions generated ({len(enhancements)} chars): {enhancements[:150]}...")

    with _demo_section("Validation"):
        validation = validate_gnn(SAMPLE_GNN_CONTENT)
        print(f"Validation completed ({len(validation)} chars): {validation[:150]}...")

def demo_processor_modes() -> None:
    """Demonstrate processor configuration.

    LLMOperations uses a unified constructor — provider selection happens
    at call time based on available API keys, not at construction time.
    """
    print("\nTesting Processor Configuration...")

    with _demo_section("Processor Info"):
        ops = LLMOperations()
        print(f"Processor: {ops.get_processor_info()}")
        print(f"Available providers: {ops.get_available_providers()}")

def demo_real_world_usage() -> None:
    """Demonstrate real-world usage scenarios."""
    print("\n🌍 Real-World Usage Scenarios...")

    with _demo_section("Scenario 1: Quick Model Analysis"):
        # This is how someone would use it in practice
        ops = LLMOperations()

        # Get basic summary
        summary = ops.summarize_gnn(SAMPLE_GNN_CONTENT, max_length=100)
        print(f"Model Summary: {summary[:200]}...")

        # Check what providers are available
        providers = ops.get_available_providers()
        print(f"Available LLM providers: {providers}")

        # Get processor information
        proc_info = ops.get_processor_info()
        print(f"Processor mode: {proc_info.get('mode', 'unknown')}")

    with _demo_section("Scenario 2: Comprehensive Analysis"):
        ops = LLMOperations()

        print("Running comprehensive analysis...")

        # Structure analysis
        structure = ops.analyze_gnn_structure(SAMPLE_GNN_CONTENT)
        print(f"✅ Structure analysis: {len(structure)} chars")

        # Generate questions for deeper understanding
        questions = ops.generate_questions(SAMPLE_GNN_CONTENT, 5)
        print(f"✅ Generated {len(questions)} research questions")

        # Get enhancement suggestions
        enhancements = ops.enhance_gnn(SAMPLE_GNN_CONTENT)
        print(f"✅ Enhancement suggestions: {len(enhancements)} chars")

        # Validate the model
        validation = ops.validate_gnn(SAMPLE_GNN_CONTENT)
        print(f"✅ Validation report: {len(validation)} chars")

        print("Comprehensive analysis complete!")


def demo_error_handling() -> None:
    """Demonstrate error handling and recovery mechanisms."""
    print("\n🛡️ Testing Error Handling and Recoveries...")

    with _demo_section("Testing with no API keys"):
        ops = LLMOperations()
        providers = ops.get_available_providers()
        print(f"Available providers with no keys: {providers}")

        # This should still work with fallbacks
        result = ops.summarize_gnn(SAMPLE_GNN_CONTENT, max_length=50)
        print(f"Result with no keys: {result[:100]}...")

    with _demo_section("Testing recovery"):
        legacy_ops = LLMOperations()
        result = legacy_ops.summarize_gnn(SAMPLE_GNN_CONTENT, max_length=50)
        print(f"Result: {result[:100]}...")

def main() -> None:
    """Run all demonstration scenarios."""
    print("🎭 Enhanced LLM System Demonstration")
    print("=" * 50)

    # Run all demonstration scenarios
    demo_backward_compatibility()
    demo_enhanced_capabilities()
    demo_processor_modes()
    demo_real_world_usage()
    demo_error_handling()

    print("\n" + "=" * 50)
    print("✅ Demonstration completed!")
    print("\nKey improvements:")
    print("• Multi-provider support (OpenAI, OpenRouter, Perplexity)")
    print("• Backward compatibility with existing code")
    print("• Enhanced analysis capabilities (enhancement, validation)")
    print("• Robust error handling and recovery mechanisms")
    print("• Async support with sync wrappers")
    print("• Provider selection based on task requirements")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()
