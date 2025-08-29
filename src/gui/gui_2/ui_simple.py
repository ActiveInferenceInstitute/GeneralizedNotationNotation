#!/usr/bin/env python3
"""
Simplified Visual Matrix Editor UI for GNN Models
This version focuses on core functionality without complex event handling
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import numpy as np
    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    np = None
    PLOTLY_AVAILABLE = False


def build_simple_visual_gui(markdown_text: str, export_path: Path, logger: logging.Logger) -> "gr.Blocks":
    """Build a simplified visual matrix editor that actually works"""
    
    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio is required for GUI functionality")

    # Initialize with simple default matrices
    default_a = [[1.0, 0.5, 0.2], [0.8, 1.0, 0.3], [0.1, 0.4, 1.0]]
    default_b = [[0.7, 0.3], [0.6, 0.4]]
    default_c = [[0.5], [0.3], [0.2]]
    default_d = [[0.8], [0.2]]

    def create_simple_heatmap(data: List[List[float]], title: str) -> Optional[go.Figure]:
        """Create a simple heatmap"""
        if not PLOTLY_AVAILABLE or not data:
            return None
        
        try:
            fig = go.Figure(data=go.Heatmap(
                z=data,
                colorscale='Viridis',
                showscale=True
            ))
            fig.update_layout(
                title=f"<b>{title}</b>",
                width=400,
                height=300,
                font=dict(size=12)
            )
            return fig
        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
            return None

    def create_bar_plot(data: List[List[float]], title: str) -> Optional[go.Figure]:
        """Create a simple bar plot for vectors"""
        if not PLOTLY_AVAILABLE or not data:
            return None
            
        try:
            # Extract values from vector format
            values = [row[0] if isinstance(row, list) and len(row) > 0 else 0.0 for row in data]
            
            fig = go.Figure(data=go.Bar(
                y=values,
                x=list(range(len(values))),
                marker_color='blue'
            ))
            fig.update_layout(
                title=f"<b>{title}</b>",
                width=400,
                height=300,
                xaxis_title="Index",
                yaxis_title="Value",
                font=dict(size=12)
            )
            return fig
        except Exception as e:
            logger.error(f"Error creating bar plot: {e}")
            return None

    def update_visualizations(matrix_a, matrix_b, vector_c, vector_d):
        """Update all visualizations - simplified version"""
        try:
            # Convert gradio dataframes to lists safely
            def safe_convert(data):
                if hasattr(data, 'values'):
                    return data.values.tolist()
                elif isinstance(data, list):
                    return data
                else:
                    return []

            a_data = safe_convert(matrix_a) or default_a
            b_data = safe_convert(matrix_b) or default_b
            c_data = safe_convert(vector_c) or default_c
            d_data = safe_convert(vector_d) or default_d

            results = []
            
            if PLOTLY_AVAILABLE:
                results.append(create_simple_heatmap(a_data, "Matrix A"))
                results.append(create_simple_heatmap(b_data, "Matrix B"))  
                results.append(create_bar_plot(c_data, "Vector C"))
                results.append(create_bar_plot(d_data, "Vector D"))
            else:
                results.extend([None, None, None, None])
            
            # Simple statistics
            stats = f"""
**Matrix Statistics:**
- Matrix A: {len(a_data)}×{len(a_data[0]) if a_data else 0}
- Matrix B: {len(b_data)}×{len(b_data[0]) if b_data else 0}  
- Vector C: {len(c_data)} elements
- Vector D: {len(d_data)} elements
"""
            results.append(stats)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in update_visualizations: {e}")
            error_fig = None
            if PLOTLY_AVAILABLE:
                error_fig = go.Figure().add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5)
            return [error_fig, error_fig, error_fig, error_fig, f"Error: {str(e)}"]

    # Build the simplified interface
    with gr.Blocks(
        title="Visual Matrix Editor - Simplified", 
        theme=gr.themes.Default()
    ) as demo:
        
        gr.Markdown("# 🎯 Visual Matrix Editor (Simplified)")
        gr.Markdown("Interactive matrix editor for Active Inference models")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Matrix A (Likelihood)")
                matrix_a_input = gr.Dataframe(
                    value=default_a,
                    label="Matrix A",
                    interactive=True
                )
                
                gr.Markdown("### Matrix B (Transition)")
                matrix_b_input = gr.Dataframe(
                    value=default_b,
                    label="Matrix B", 
                    interactive=True
                )
            
            with gr.Column():
                gr.Markdown("### Vector C (Preference)")
                vector_c_input = gr.Dataframe(
                    value=default_c,
                    label="Vector C",
                    interactive=True
                )
                
                gr.Markdown("### Vector D (Prior)")
                vector_d_input = gr.Dataframe(
                    value=default_d,
                    label="Vector D",
                    interactive=True
                )
        
        # Update button
        update_btn = gr.Button("🔄 Update Visualizations", variant="primary")
        
        # Visualizations row
        if PLOTLY_AVAILABLE:
            with gr.Row():
                plot_a = gr.Plot(label="Matrix A Heatmap")
                plot_b = gr.Plot(label="Matrix B Heatmap") 
            
            with gr.Row():
                plot_c = gr.Plot(label="Vector C Plot")
                plot_d = gr.Plot(label="Vector D Plot")
        
        # Statistics
        stats_output = gr.Markdown("Click 'Update Visualizations' to see statistics")
        
        # Event handling - only on button click to avoid loading issues
        if PLOTLY_AVAILABLE:
            update_btn.click(
                update_visualizations,
                inputs=[matrix_a_input, matrix_b_input, vector_c_input, vector_d_input],
                outputs=[plot_a, plot_b, plot_c, plot_d, stats_output]
            )
        else:
            def update_stats_only(matrix_a, matrix_b, vector_c, vector_d):
                return update_visualizations(matrix_a, matrix_b, vector_c, vector_d)[-1]
            
            update_btn.click(
                update_stats_only,
                inputs=[matrix_a_input, matrix_b_input, vector_c_input, vector_d_input], 
                outputs=[stats_output]
            )
        
        # Export functionality
        with gr.Row():
            export_btn = gr.Button("💾 Export GNN", variant="secondary")
            export_output = gr.Textbox(label="Export Status", lines=2)
        
        def export_matrices(matrix_a, matrix_b, vector_c, vector_d):
            """Export current matrices to GNN format"""
            try:
                # Simple export logic
                gnn_content = f"""# Active Inference POMDP Agent

## Matrices
- Matrix A: {len(matrix_a) if matrix_a else 0}×{len(matrix_a[0]) if matrix_a and matrix_a[0] else 0}
- Matrix B: {len(matrix_b) if matrix_b else 0}×{len(matrix_b[0]) if matrix_b and matrix_b[0] else 0}  
- Vector C: {len(vector_c) if vector_c else 0} elements
- Vector D: {len(vector_d) if vector_d else 0} elements

Generated by Visual Matrix Editor (Simplified)
"""
                
                export_path.write_text(gnn_content)
                return f"✅ Exported to {export_path.name}"
                
            except Exception as e:
                logger.error(f"Export failed: {e}")
                return f"❌ Export failed: {str(e)}"
        
        export_btn.click(
            export_matrices,
            inputs=[matrix_a_input, matrix_b_input, vector_c_input, vector_d_input],
            outputs=[export_output]
        )

    return demo
