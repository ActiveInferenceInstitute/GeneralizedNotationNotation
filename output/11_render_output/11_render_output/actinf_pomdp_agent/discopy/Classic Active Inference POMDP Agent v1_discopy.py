#!/usr/bin/env python3
"""
DisCoPy Categorical Diagram Generation
Generated from GNN Model: Classic Active Inference POMDP Agent v1
Generated: 2025-09-15 09:41:48

This script creates categorical diagrams representing the Active Inference model
structure using DisCoPy's compositional framework.
"""

from discopy import *
from discopy.quantum import *
from discopy.drawing import Equation
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Model parameters extracted from GNN specification
NUM_STATES = 3
NUM_OBSERVATIONS = 3
NUM_ACTIONS = 3

print("🔬 DisCoPy Categorical Diagram Generation")
print(f"📊 State Space: {NUM_STATES} states, {NUM_OBSERVATIONS} observations, {NUM_ACTIONS} actions")

# Define basic types for Active Inference
def define_types():
    """Define the basic types for our Active Inference model."""
    
    # Basic types
    S = Ty('S')  # Hidden states
    O = Ty('O')  # Observations  
    A = Ty('A')  # Actions
    P = Ty('P')  # Probabilities
    
    print("✓ Defined basic types: S (states), O (observations), A (actions), P (probabilities)")
    return S, O, A, P

# Define morphisms (boxes) for model components
def define_model_components(S, O, A, P):
    """Define the morphisms representing Active Inference model components."""
    
    components = {}
    
    # A matrix: Observation model P(o|s)
    components['A_matrix'] = Box('A', S, O @ P, draw_as_box=True)
    
    # B matrix: Transition model P(s'|s,a)  
    components['B_matrix'] = Box('B', S @ A, S @ P, draw_as_box=True)
    
    # C vector: Preference vector over observations
    components['C_vector'] = Box('C', Ty(), O @ P, draw_as_box=True)
    
    # D vector: Prior beliefs over states
    components['D_vector'] = Box('D', Ty(), S @ P, draw_as_box=True)
    
    # E vector: Policy priors (if applicable)
    components['E_vector'] = Box('E', Ty(), A @ P, draw_as_box=True)
    
    # Inference processes
    components['state_inference'] = Box('StateInf', O, S @ P, draw_as_box=True)
    components['policy_inference'] = Box('PolicyInf', S @ P, A @ P, draw_as_box=True)
    components['action_selection'] = Box('ActionSel', A @ P, A, draw_as_box=True)
    
    print(f"✓ Defined {len(components)} model components as morphisms")
    return components

# Create Active Inference circuit
def create_active_inference_circuit(S, O, A, P, components):
    """Create the full Active Inference circuit as a categorical diagram."""
    
    print("\n🏗️  Building Active Inference circuit...")
    
    # Extract components
    A_matrix = components['A_matrix']
    B_matrix = components['B_matrix'] 
    C_vector = components['C_vector']
    D_vector = components['D_vector']
    state_inf = components['state_inference']
    policy_inf = components['policy_inference']
    action_sel = components['action_selection']
    
    # Basic perception-action loop
    # Observation -> State Inference -> Policy Inference -> Action Selection
    perception_action_loop = (
        state_inf 
        >> policy_inf 
        >> action_sel
    )
    
    # Full generative model with priors
    # D (prior) and C (preferences) influence the inference
    generative_model = (
        (D_vector @ Id(O)) >>  # Prior beliefs
        (Id(S @ P) @ state_inf) >>  # State inference with observations
        (policy_inf @ Id(Ty())) >>  # Policy inference
        (action_sel @ Id(Ty()))  # Action selection
    )
    
    print("✓ Created perception-action loop")
    print("✓ Created generative model with priors")
    
    return {
        'perception_action_loop': perception_action_loop,
        'generative_model': generative_model,
        'components': components
    }

# Visualization functions
def visualize_diagrams(circuit_dict, output_dir="discopy_diagrams"):
    """Visualize the categorical diagrams."""
    
    print(f"\n📊 Creating visualizations in {output_dir}/...")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    try:
        # Visualize perception-action loop
        loop_diagram = circuit_dict['perception_action_loop']
        
        # Draw the diagram
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        loop_diagram.draw(ax=ax, figsize=(12, 8))
        ax.set_title("Active Inference: Perception-Action Loop", fontsize=16, fontweight='bold')
        
        # Save perception-action loop
        loop_file = Path(output_dir) / "perception_action_loop.png"
        plt.savefig(loop_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved perception-action loop: {loop_file}")
        
        # Visualize generative model
        gen_diagram = circuit_dict['generative_model']
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        gen_diagram.draw(ax=ax, figsize=(14, 10))
        ax.set_title("Active Inference: Generative Model", fontsize=16, fontweight='bold')
        
        # Save generative model
        gen_file = Path(output_dir) / "generative_model.png"
        plt.savefig(gen_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved generative model: {gen_file}")
        
        # Create component diagram
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        components = circuit_dict['components']
        component_names = list(components.keys())
        
        for i, (name, component) in enumerate(components.items()):
            if i < len(axes):
                component.draw(ax=axes[i], figsize=(6, 4))
                axes[i].set_title(f"Component: {name}", fontsize=12, fontweight='bold')
        
        # Hide unused subplots
        for i in range(len(components), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle("Active Inference Model Components", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save components diagram
        comp_file = Path(output_dir) / "model_components.png"
        plt.savefig(comp_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved model components: {comp_file}")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Visualization failed: {e}")
        print("This might be due to missing graphical backend or DisCoPy drawing dependencies")
        return False

# Analysis functions
def analyze_circuit_structure(circuit_dict):
    """Analyze the structure of the categorical diagrams."""
    
    print("\n🔍 Analyzing circuit structure...")
    
    components = circuit_dict['components']
    loop = circuit_dict['perception_action_loop']
    model = circuit_dict['generative_model']
    
    # Basic structural analysis
    print(f"📊 Circuit Analysis:")
    print(f"  - Number of components: {len(components)}")
    print(f"  - Perception-action loop domain: {loop.dom}")
    print(f"  - Perception-action loop codomain: {loop.cod}")
    print(f"  - Generative model domain: {model.dom}")
    print(f"  - Generative model codomain: {model.cod}")
    
    # Component analysis
    print(f"\n🧩 Component Details:")
    for name, component in components.items():
        print(f"  - {name}: {component.dom} → {component.cod}")
    
    # Compositional structure
    print(f"\n🔗 Compositional Structure:")
    print(f"  - Loop composition depth: {len(str(loop).split('>>'))}")
    print(f"  - Model composition depth: {len(str(model).split('>>'))}")
    
    return {
        'num_components': len(components),
        'loop_domain': str(loop.dom),
        'loop_codomain': str(loop.cod),
        'model_domain': str(model.dom),
        'model_codomain': str(model.cod)
    }

# Export functions
def export_circuit_data(circuit_dict, analysis_results, output_dir="discopy_diagrams"):
    """Export circuit data and analysis results."""
    
    print(f"\n💾 Exporting circuit data to {output_dir}/...")
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Export analysis results
    analysis_file = Path(output_dir) / "circuit_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"✓ Exported analysis: {analysis_file}")
    
    # Export circuit information
    circuit_info = {
        'model_name': 'Classic Active Inference POMDP Agent v1',
        'timestamp': '2025-09-15 09:41:48',
        'parameters': {
            'num_states': NUM_STATES,
            'num_observations': NUM_OBSERVATIONS, 
            'num_actions': NUM_ACTIONS
        },
        'components': list(circuit_dict['components'].keys()),
        'analysis': analysis_results
    }
    
    circuit_file = Path(output_dir) / "circuit_info.json"
    with open(circuit_file, 'w') as f:
        json.dump(circuit_info, f, indent=2)
    
    print(f"✓ Exported circuit info: {circuit_file}")

# Main execution
def main():
    """Main execution function."""
    
    print("="*60)
    print("DisCoPy Categorical Diagrams - GNN Generated")
    print(f"Model: Classic Active Inference POMDP Agent v1")
    print("="*60)
    
    try:
        # Define types
        S, O, A, P = define_types()
        
        # Define model components
        components = define_model_components(S, O, A, P)
        
        # Create Active Inference circuit
        circuit_dict = create_active_inference_circuit(S, O, A, P, components)
        
        # Analyze circuit structure
        analysis_results = analyze_circuit_structure(circuit_dict)
        
        # Create visualizations
        viz_success = visualize_diagrams(circuit_dict)
        
        # Export data
        export_circuit_data(circuit_dict, analysis_results)
        
        if viz_success:
            print("\n✅ DisCoPy categorical diagram generation completed successfully!")
            print("🎨 Visualizations created and saved")
        else:
            print("\n⚠️  DisCoPy diagram generation completed with visualization warnings")
            print("📊 Circuit analysis and data export successful")
        
        print("\n🎉 Categorical representation of Active Inference model complete!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Diagram generation failed: {e}")
        print("🔍 Stack trace:")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
