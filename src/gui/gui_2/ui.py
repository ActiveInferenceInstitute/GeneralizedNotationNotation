"""
UI for GUI 2: Visual Matrix Editor

Provides a visual drag-and-drop interface for editing matrix structures
and real-time GNN markdown generation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, List

try:
    import gradio as gr
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    gr = None  # type: ignore
    np = None  # type: ignore
    go = None  # type: ignore
    px = None  # type: ignore

from .matrix_editor import (
    create_matrix_from_gnn,
    update_gnn_from_matrix,
    validate_matrix_dimensions,
)


def build_visual_gui(markdown_text: str, export_path: Path, logger: logging.Logger) -> "gr.Blocks":  # type: ignore[name-defined]
    """Build the visual matrix editor interface"""
    if gr is None:
        raise RuntimeError("Gradio not available")

    # Initialize visual data from markdown
    visual_data = create_matrix_from_gnn(markdown_text)
    
    with gr.Blocks(title="GNN Visual Matrix Editor", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎯 GNN Visual Matrix Editor")
        gr.Markdown("Interactive matrix editing with real-time GNN markdown generation")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Visual Matrix Editor")
                
                with gr.Tab("Matrix A (Likelihood)"):
                    matrix_a_plot = gr.Plot(label="A Matrix Visualization")
                    with gr.Row():
                        a_rows = gr.Slider(1, 10, value=3, step=1, label="Rows (Observations)")
                        a_cols = gr.Slider(1, 10, value=3, step=1, label="Cols (States)")
                    a_values = gr.Dataframe(
                        headers=["State 0", "State 1", "State 2"],
                        value=[[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]],
                        interactive=True,
                        label="A Matrix Values"
                    )
                
                with gr.Tab("Matrix B (Transitions)"):
                    b_slice_selector = gr.Slider(0, 2, value=0, step=1, label="Action Slice")
                    matrix_b_plot = gr.Plot(label="B Matrix Visualization")
                    with gr.Row():
                        b_states = gr.Slider(1, 10, value=3, step=1, label="States")
                        b_actions = gr.Slider(1, 10, value=3, step=1, label="Actions")
                    b_values = gr.Dataframe(
                        headers=["State 0", "State 1", "State 2"],
                        value=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        interactive=True,
                        label="B Matrix Values (Current Slice)"
                    )
                
                with gr.Tab("Vectors C & D"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### C (Preferences)")
                            c_plot = gr.Plot(label="C Vector")
                            c_values = gr.Dataframe(
                                headers=["Preference"],
                                value=[[0.1], [0.1], [1.0]],
                                interactive=True,
                                label="C Values"
                            )
                        with gr.Column():
                            gr.Markdown("#### D (Prior)")
                            d_plot = gr.Plot(label="D Vector")
                            d_values = gr.Dataframe(
                                headers=["Prior"],
                                value=[[0.33], [0.33], [0.33]],
                                interactive=True,
                                label="D Values"
                            )
                
                with gr.Row():
                    update_visual_btn = gr.Button("🔄 Update Visualizations", variant="primary")
                    reset_btn = gr.Button("🔄 Reset to Template", variant="secondary")
            
            with gr.Column(scale=1):
                gr.Markdown("### 📝 GNN Markdown Output")
                gnn_output = gr.Code(
                    value=markdown_text,
                    language="markdown",
                    label="Generated GNN",
                    lines=25
                )
                
                with gr.Row():
                    save_btn = gr.Button("💾 Save GNN", variant="primary")
                    validate_btn = gr.Button("✅ Validate", variant="secondary")
                
                validation_output = gr.Markdown("")
                save_status = gr.Markdown("")

        # Event handlers
        def update_matrix_plot(matrix_data: List[List[float]], title: str, colorscale: str = "Viridis"):
            """Create a matrix heatmap visualization"""
            if go is None:
                return gr.Plot(visible=False)
            
            try:
                fig = go.Figure(data=go.Heatmap(
                    z=matrix_data,
                    colorscale=colorscale,
                    showscale=True,
                    text=[[f"{val:.3f}" for val in row] for row in matrix_data],
                    texttemplate="%{text}",
                    textfont={"size": 10},
                ))
                fig.update_layout(
                    title=title,
                    width=400,
                    height=300,
                    xaxis_title="Columns",
                    yaxis_title="Rows"
                )
                return fig
            except Exception:
                return gr.Plot(visible=False)

        def update_vector_plot(vector_data: List[List[float]], title: str, color: str = "blue"):
            """Create a vector bar plot"""
            if go is None:
                return gr.Plot(visible=False)
            
            try:
                values = [row[0] for row in vector_data]
                fig = go.Figure(data=go.Bar(
                    x=list(range(len(values))),
                    y=values,
                    marker_color=color,
                    text=[f"{val:.3f}" for val in values],
                    textposition='auto',
                ))
                fig.update_layout(
                    title=title,
                    width=300,
                    height=200,
                    xaxis_title="Index",
                    yaxis_title="Value"
                )
                return fig
            except Exception:
                return gr.Plot(visible=False)

        def update_all_visualizations(a_data, b_data, c_data, d_data):
            """Update all matrix and vector visualizations"""
            try:
                a_plot = update_matrix_plot(a_data, "A Matrix (Likelihood)", "Blues")
                c_plot = update_vector_plot(c_data, "C Vector (Preferences)", "red")
                d_plot = update_vector_plot(d_data, "D Vector (Prior)", "green")
                b_plot = update_matrix_plot(b_data, "B Matrix (Transitions)", "Oranges")
                return a_plot, b_plot, c_plot, d_plot
            except Exception:
                return gr.Plot(visible=False), gr.Plot(visible=False), gr.Plot(visible=False), gr.Plot(visible=False)

        def generate_gnn_from_matrices(a_data, b_data, c_data, d_data, b_slice):
            """Generate GNN markdown from matrix data"""
            try:
                # Convert dataframe data to visual format
                visual_matrices = {
                    "A": {
                        "type": "matrix",
                        "rows": len(a_data),
                        "cols": len(a_data[0]) if a_data else 0,
                        "values": a_data,
                        "description": "Likelihood matrix"
                    },
                    "C": {
                        "type": "vector", 
                        "size": len(c_data),
                        "values": [row[0] for row in c_data],
                        "description": "Preference vector"
                    },
                    "D": {
                        "type": "vector",
                        "size": len(d_data), 
                        "values": [row[0] for row in d_data],
                        "description": "Prior vector"
                    },
                    "B": {
                        "type": "tensor",
                        "depth": 3,  # Simplified for now
                        "rows": len(b_data),
                        "cols": len(b_data[0]) if b_data else 0,
                        "values": [b_data, b_data, b_data],  # Simplified - all slices same
                        "description": "Transition matrices"
                    }
                }
                
                visual_data = {"visual_matrices": visual_matrices}
                updated_gnn = update_gnn_from_matrix(visual_data, markdown_text)
                return updated_gnn
            except Exception as e:
                logger.error(f"Failed to generate GNN: {e}")
                return markdown_text

        def save_gnn(gnn_text):
            """Save GNN markdown to file"""
            try:
                export_path.write_text(gnn_text)
                return f"✅ Saved to {export_path.name}"
            except Exception as e:
                return f"❌ Save failed: {e}"

        def validate_gnn(a_data, b_data, c_data, d_data):
            """Validate matrix dimensions and consistency"""
            try:
                errors = []
                
                # Basic dimension checks
                if not a_data or not a_data[0]:
                    errors.append("A matrix cannot be empty")
                if not b_data or not b_data[0]:
                    errors.append("B matrix cannot be empty") 
                if not c_data:
                    errors.append("C vector cannot be empty")
                if not d_data:
                    errors.append("D vector cannot be empty")
                
                # Consistency checks
                if a_data and d_data:
                    if len(a_data[0]) != len(d_data):
                        errors.append(f"A matrix columns ({len(a_data[0])}) must match D vector size ({len(d_data)})")
                
                if a_data and c_data:
                    if len(a_data) != len(c_data):
                        errors.append(f"A matrix rows ({len(a_data)}) must match C vector size ({len(c_data)})")
                
                if errors:
                    return "❌ **Validation Errors:**\n" + "\n".join(f"- {e}" for e in errors)
                else:
                    return "✅ **Validation Passed:** All matrix dimensions are consistent"
                    
            except Exception as e:
                return f"❌ Validation failed: {e}"

        # Wire up the events
        update_visual_btn.click(
            update_all_visualizations,
            inputs=[a_values, b_values, c_values, d_values],
            outputs=[matrix_a_plot, matrix_b_plot, c_plot, d_plot]
        )
        
        # Auto-update GNN when matrices change
        for matrix_input in [a_values, b_values, c_values, d_values]:
            matrix_input.change(
                generate_gnn_from_matrices,
                inputs=[a_values, b_values, c_values, d_values, b_slice_selector],
                outputs=[gnn_output]
            )
        
        save_btn.click(save_gnn, inputs=[gnn_output], outputs=[save_status])
        validate_btn.click(
            validate_gnn,
            inputs=[a_values, b_values, c_values, d_values],
            outputs=[validation_output]
        )
        
        # Initialize plots on load
        demo.load(
            update_all_visualizations,
            inputs=[a_values, b_values, c_values, d_values],
            outputs=[matrix_a_plot, matrix_b_plot, c_plot, d_plot]
        )

    return demo
