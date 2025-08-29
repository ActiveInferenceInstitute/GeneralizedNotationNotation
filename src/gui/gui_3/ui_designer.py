#!/usr/bin/env python3
"""
GUI 3: State Space Design Studio UI
Low-dependency visual design experience for Active Inference models
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False


def build_design_studio(markdown_text: str, export_path: Path, logger: logging.Logger) -> "gr.Blocks":
    """
    Build the State Space Design Studio GUI interface
    
    Features:
    - Visual state space designer
    - Ontology term editor
    - Connection graph interface
    - Parameter tuning controls
    - Low-dependency HTML/CSS design
    """
    
    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio is required for Design Studio functionality")

    logger.info("🎨 Building State Space Design Studio...")

    # Parse GNN content for design studio
    design_data = _parse_gnn_for_design(markdown_text)
    
    with gr.Blocks(
        title="State Space Design Studio", 
        theme=gr.themes.Base()
    ) as demo:
        
        gr.Markdown("# 🎨 State Space Design Studio")
        gr.Markdown("**Low-dependency visual design experience for Active Inference models**")
        
        with gr.Tabs():
            
            # Tab 1: State Space Designer
            with gr.TabItem("🏗️ State Space"):
                gr.Markdown("### Visual State Space Architecture")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Current State Spaces:**")
                        state_spaces = gr.Dataframe(
                            value=design_data.get("state_spaces", [["A", "3,3", "Likelihood Matrix"]]),
                            headers=["Variable", "Dimensions", "Description"],
                            label="State Space Variables"
                        )
                        
                        with gr.Row():
                            add_variable = gr.Button("➕ Add Variable", variant="primary")
                            remove_variable = gr.Button("➖ Remove Selected", variant="secondary")
                    
                    with gr.Column():
                        gr.Markdown("**Visual Designer:**")
                        designer_output = gr.HTML(value=_generate_visual_designer_html())
                        
            # Tab 2: Ontology Editor  
            with gr.TabItem("📚 Ontology"):
                gr.Markdown("### Active Inference Ontology Terms")
                
                ontology_editor = gr.Dataframe(
                    value=_format_ontology_data(design_data.get("ontology", {})),
                    headers=["Variable", "Ontology Term", "Description"],
                    label="Ontology Mappings"
                )
                
                with gr.Row():
                    ontology_var = gr.Textbox(label="Variable", placeholder="A")
                    ontology_term = gr.Textbox(label="Ontology Term", placeholder="LikelihoodMatrix")
                    add_ontology = gr.Button("Add Mapping", variant="primary")
            
            # Tab 3: Connections
            with gr.TabItem("🔗 Connections"):
                gr.Markdown("### Model Connections Graph")
                
                connections_text = gr.Textbox(
                    value=design_data.get("connections_text", "D>s\ns-A\nA-o"),
                    label="Connections (one per line)",
                    lines=8
                )
                
                connections_visual = gr.HTML(value=_generate_connections_html())
                
                with gr.Row():
                    validate_connections = gr.Button("✓ Validate", variant="primary")
                    auto_layout = gr.Button("🔄 Auto Layout", variant="secondary")
            
            # Tab 4: Parameters
            with gr.TabItem("⚙️ Parameters"):
                gr.Markdown("### Model Parameters")
                
                with gr.Row():
                    with gr.Column():
                        num_states = gr.Slider(1, 10, value=3, label="Hidden States")
                        num_obs = gr.Slider(1, 10, value=3, label="Observations") 
                        num_actions = gr.Slider(1, 10, value=3, label="Actions")
                        
                    with gr.Column():
                        planning_horizon = gr.Slider(1, 5, value=1, label="Planning Horizon")
                        time_horizon = gr.Dropdown(
                            ["Bounded", "Unbounded"], 
                            value="Unbounded",
                            label="Time Horizon"
                        )
        
        # Export Section
        gr.Markdown("---")
        with gr.Row():
            with gr.Column():
                export_btn = gr.Button("💾 Export GNN Model", variant="primary", size="lg")
                export_status = gr.Textbox(label="Export Status", lines=2)
                
            with gr.Column():  
                preview_btn = gr.Button("👁️ Preview Model", variant="secondary")
                model_preview = gr.Code(language="markdown", label="GNN Preview")
        
        # Event Handlers
        def export_design(spaces, ontology, connections, states, obs, actions, horizon, time_h):
            """Export current design to GNN format"""
            try:
                gnn_content = _generate_gnn_from_design(
                    spaces, ontology, connections, states, obs, actions, horizon, time_h
                )
                
                export_path.write_text(gnn_content)
                return f"✅ Model exported to {export_path.name}"
                
            except Exception as e:
                logger.error(f"Export failed: {e}")
                return f"❌ Export failed: {str(e)}"
        
        def preview_design(spaces, ontology, connections, states, obs, actions, horizon, time_h):
            """Preview the current design as GNN"""
            try:
                return _generate_gnn_from_design(
                    spaces, ontology, connections, states, obs, actions, horizon, time_h
                )
            except Exception as e:
                return f"Error generating preview: {e}"
        
        # Wire up event handlers
        export_btn.click(
            export_design,
            inputs=[state_spaces, ontology_editor, connections_text, 
                   num_states, num_obs, num_actions, planning_horizon, time_horizon],
            outputs=[export_status]
        )
        
        preview_btn.click(
            preview_design,
            inputs=[state_spaces, ontology_editor, connections_text,
                   num_states, num_obs, num_actions, planning_horizon, time_horizon],
            outputs=[model_preview]
        )

    logger.info("✅ State Space Design Studio built successfully")
    return demo


def _parse_gnn_for_design(gnn_content: str) -> Dict[str, Any]:
    """Parse GNN content for design studio"""
    
    design_data = {
        "state_spaces": [],
        "ontology": {},
        "connections_text": "",
        "parameters": {}
    }
    
    lines = gnn_content.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('## '):
            current_section = line[3:]
            continue
            
        if current_section == "StateSpaceBlock":
            if '[' in line and ']' in line and not line.startswith('#'):
                var_name = line.split('[')[0].strip()
                dimensions = line.split('[')[1].split(']')[0].strip()
                desc = line.split('#')[1].strip() if '#' in line else ""
                design_data["state_spaces"].append([var_name, dimensions, desc])
                
        elif current_section == "ActInfOntologyAnnotation":
            if '=' in line and not line.startswith('#'):
                var, concept = line.split('=', 1)
                design_data["ontology"][var.strip()] = concept.strip()
                
        elif current_section == "Connections":
            if line and not line.startswith('#'):
                if design_data["connections_text"]:
                    design_data["connections_text"] += "\n"
                design_data["connections_text"] += line
    
    return design_data


def _format_ontology_data(ontology_dict: Dict[str, str]) -> List[List[str]]:
    """Format ontology data for dataframe"""
    if not ontology_dict:
        return [["A", "LikelihoodMatrix", "Maps states to observations"]]
    
    formatted = []
    descriptions = {
        "LikelihoodMatrix": "Maps states to observations",
        "TransitionMatrix": "State transitions given actions",
        "LogPreferenceVector": "Preferences over observations",
        "PriorOverHiddenStates": "Initial state beliefs"
    }
    
    for var, term in ontology_dict.items():
        desc = descriptions.get(term, "Active Inference component")
        formatted.append([var, term, desc])
    
    return formatted


def _generate_visual_designer_html() -> str:
    """Generate HTML for visual state space designer"""
    return '''
    <div style="border: 1px solid #ddd; padding: 20px; border-radius: 8px; background: #f9f9f9;">
        <h4>🎯 Visual State Space</h4>
        <div style="display: flex; align-items: center; gap: 20px; margin: 20px 0;">
            <div style="padding: 15px; background: #e3f2fd; border-radius: 6px; text-align: center;">
                <strong>States (s)</strong><br/>3 dimensions
            </div>
            <div style="font-size: 20px;">→</div>
            <div style="padding: 15px; background: #f3e5f5; border-radius: 6px; text-align: center;">
                <strong>Obs (o)</strong><br/>3 dimensions  
            </div>
        </div>
        <div style="text-align: center; margin-top: 15px;">
            <em>Interactive designer coming soon...</em>
        </div>
    </div>
    '''


def _generate_connections_html() -> str:
    """Generate HTML for connections visualization"""
    return '''
    <div style="border: 1px solid #ddd; padding: 20px; border-radius: 8px; background: #f9f9f9;">
        <h4>🔗 Connection Graph</h4>
        <svg width="300" height="200" style="border: 1px solid #ccc; background: white;">
            <circle cx="50" cy="50" r="20" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
            <text x="50" y="55" text-anchor="middle" font-size="12">D</text>
            
            <circle cx="150" cy="50" r="20" fill="#e8f5e8" stroke="#388e3c" stroke-width="2"/>
            <text x="150" y="55" text-anchor="middle" font-size="12">s</text>
            
            <circle cx="250" cy="50" r="20" fill="#fff3e0" stroke="#f57c00" stroke-width="2"/>
            <text x="250" y="55" text-anchor="middle" font-size="12">o</text>
            
            <line x1="70" y1="50" x2="130" y2="50" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
            <line x1="170" y1="50" x2="230" y2="50" stroke="#666" stroke-width="2" marker-end="url(#arrowhead)"/>
            
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#666"/>
                </marker>
            </defs>
        </svg>
        <div style="margin-top: 10px; font-size: 12px;">
            <span style="color: #1976d2;">●</span> Priors &nbsp;&nbsp;
            <span style="color: #388e3c;">●</span> States &nbsp;&nbsp;
            <span style="color: #f57c00;">●</span> Observations
        </div>
    </div>
    '''


def _generate_gnn_from_design(spaces, ontology, connections, states, obs, actions, horizon, time_h):
    """Generate GNN content from design studio inputs"""
    
    gnn_lines = [
        "# GNN: State Space Design Studio Export",
        "# Generated by Design Studio GUI",
        "",
        "## ModelName",
        "Active Inference Model - Design Studio Export",
        "",
        "## StateSpaceBlock"
    ]
    
    # Add state spaces
    if hasattr(spaces, 'values'):
        for row in spaces.values:
            if len(row) >= 2:
                var, dims = row[0], row[1]
                desc = row[2] if len(row) > 2 else ""
                comment = f"   # {desc}" if desc else ""
                gnn_lines.append(f"{var}[{dims},type=float]{comment}")
    
    gnn_lines.extend([
        "",
        "## Connections"
    ])
    
    # Add connections  
    if connections:
        for line in str(connections).split('\n'):
            if line.strip():
                gnn_lines.append(line.strip())
    
    gnn_lines.extend([
        "",
        "## ActInfOntologyAnnotation"
    ])
    
    # Add ontology
    if hasattr(ontology, 'values'):
        for row in ontology.values:
            if len(row) >= 2:
                var, term = row[0], row[1]
                gnn_lines.append(f"{var}={term}")
    
    gnn_lines.extend([
        "",
        "## ModelParameters",
        f"num_hidden_states: {states}",
        f"num_obs: {obs}", 
        f"num_actions: {actions}",
        f"planning_horizon: {horizon}",
        f"time_horizon: {time_h}",
        "",
        "## Footer",
        "Generated by State Space Design Studio - Low Dependency GUI"
    ])
    
    return '\n'.join(gnn_lines)
