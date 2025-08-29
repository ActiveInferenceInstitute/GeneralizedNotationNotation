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
    """Build the enhanced visual matrix editor interface with real-time heatmaps and dimension controls"""
    if gr is None:
        raise RuntimeError("Gradio not available")

    # Initialize visual data from markdown
    visual_data = create_matrix_from_gnn(markdown_text)
    
    with gr.Blocks(title="GNN Visual Matrix Editor", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎯 GNN Visual Matrix Editor")
        gr.Markdown("🚀 **Interactive matrix editing with real-time heatmap visualization and dimension controls**")
        
        # State variables for matrix dimensions (using Gradio state)
        matrix_state = gr.State({
            "A": {"rows": 3, "cols": 3, "values": [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]},
            "B": {"depth": 3, "rows": 3, "cols": 3, "current_slice": 0, 
                  "values": [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] for _ in range(3)]},
            "C": {"size": 3, "values": [0.1, 0.1, 1.0]},
            "D": {"size": 3, "values": [0.33, 0.33, 0.33]}
        })
        
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### 📊 **Interactive Visual Matrix Editor**")
                
                with gr.Tab("🔵 Matrix A (Likelihood)") as tab_a:
                    with gr.Row():
                        gr.Markdown("#### **Current Size Display**")
                    
                    # Matrix size display and controls
                    with gr.Row():
                        with gr.Column(scale=1):
                            a_size_display = gr.Markdown("**Matrix A**: `3×3` (Observations × States)")
                            
                            # Dimension controls
                            with gr.Row():
                                gr.Markdown("**Rows (Obs):**")
                                a_rows_minus = gr.Button("➖", size="sm")
                                a_rows_display = gr.Markdown("**3**", elem_classes=["dimension-display"])
                                a_rows_plus = gr.Button("➕", size="sm")
                            
                            with gr.Row():
                                gr.Markdown("**Cols (States):**")
                                a_cols_minus = gr.Button("➖", size="sm") 
                                a_cols_display = gr.Markdown("**3**", elem_classes=["dimension-display"])
                                a_cols_plus = gr.Button("➕", size="sm")
                    
                    # Large heatmap visualization
                    matrix_a_plot = gr.Plot(label="A Matrix Heatmap Visualization", scale=2)
                    
                    # Matrix values editor
                    a_values = gr.Dataframe(
                        headers=["State_0", "State_1", "State_2"],
                        value=[[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]],
                        interactive=True,
                        label="A Matrix Values - Edit cells directly",
                        row_count=3,
                        col_count=3
                    )
                
                with gr.Tab("🟠 Matrix B (Transitions)") as tab_b:
                    with gr.Row():
                        gr.Markdown("#### **Current Size Display**")
                    
                    # Matrix size display and controls
                    with gr.Row():
                        with gr.Column(scale=1):
                            b_size_display = gr.Markdown("**Matrix B**: `3×3×3` (States × States × Actions)")
                            
                            # Dimension controls
                            with gr.Row():
                                gr.Markdown("**States:**")
                                b_states_minus = gr.Button("➖", size="sm")
                                b_states_display = gr.Markdown("**3**", elem_classes=["dimension-display"])
                                b_states_plus = gr.Button("➕", size="sm")
                            
                            with gr.Row():
                                gr.Markdown("**Actions (Depth):**")
                                b_actions_minus = gr.Button("➖", size="sm")
                                b_actions_display = gr.Markdown("**3**", elem_classes=["dimension-display"])
                                b_actions_plus = gr.Button("➕", size="sm")
                    
                    # Action slice selector
                    with gr.Row():
                        b_slice_selector = gr.Slider(0, 2, value=0, step=1, label="Action Slice - Select which action transition matrix to view")
                    
                    # Large heatmap visualization
                    matrix_b_plot = gr.Plot(label="B Matrix Heatmap Visualization (Current Slice)", scale=2)
                    
                    # Matrix values editor
                    b_values = gr.Dataframe(
                        headers=["State_0", "State_1", "State_2"],
                        value=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        interactive=True,
                        label="B Matrix Values - Current Action Slice",
                        row_count=3,
                        col_count=3
                    )
                
                with gr.Tab("🔴 Vectors C & D") as tab_cd:
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### **C Vector (Preferences)**")
                            
                            # Size display and controls for C
                            with gr.Row():
                                c_size_display = gr.Markdown("**Vector C**: `3` (Observation Preferences)")
                                c_size_minus = gr.Button("➖", size="sm")
                                c_size_display_num = gr.Markdown("**3**", elem_classes=["dimension-display"])
                                c_size_plus = gr.Button("➕", size="sm")
                            
                            c_plot = gr.Plot(label="C Vector Visualization")
                            c_values = gr.Dataframe(
                                headers=["Preference"],
                                value=[[0.1], [0.1], [1.0]],
                                interactive=True,
                                label="C Values",
                                row_count=3,
                                col_count=1
                            )
                        
                        with gr.Column():
                            gr.Markdown("#### **D Vector (Prior)**")
                            
                            # Size display and controls for D  
                            with gr.Row():
                                d_size_display = gr.Markdown("**Vector D**: `3` (State Prior)")
                                d_size_minus = gr.Button("➖", size="sm")
                                d_size_display_num = gr.Markdown("**3**", elem_classes=["dimension-display"])
                                d_size_plus = gr.Button("➕", size="sm")
                            
                            d_plot = gr.Plot(label="D Vector Visualization")
                            d_values = gr.Dataframe(
                                headers=["Prior"],
                                value=[[0.33], [0.33], [0.33]],
                                interactive=True,
                                label="D Values",
                                row_count=3,
                                col_count=1
                            )
                
                # Control buttons
                with gr.Row():
                    auto_update_checkbox = gr.Checkbox(value=True, label="🔄 Auto-update visualizations")
                    manual_update_btn = gr.Button("🔄 Manual Update", variant="primary")
                    reset_btn = gr.Button("🔄 Reset to POMDP Template", variant="secondary")
                    randomize_btn = gr.Button("🎲 Randomize Values", variant="secondary")
            
            with gr.Column(scale=1):
                gr.Markdown("### 📝 **GNN Markdown Output**")
                gnn_output = gr.Code(
                    value=markdown_text,
                    language="markdown",
                    label="Generated GNN",
                    lines=30
                )
                
                with gr.Row():
                    save_btn = gr.Button("💾 Save GNN", variant="primary")
                    validate_btn = gr.Button("✅ Validate", variant="secondary")
                
                validation_output = gr.Markdown("")
                save_status = gr.Markdown("")
                
                # Matrix statistics
                gr.Markdown("### 📈 **Matrix Statistics**")
                stats_output = gr.Markdown("Click 'Update' to see matrix statistics")

        # Enhanced Event Handlers for Interactive Matrix Editing
        
        def create_enhanced_heatmap(matrix_data, title: str, colorscale: str = "Viridis", 
                                   show_values: bool = True) -> "go.Figure":  # type: ignore[name-defined]
            """Create an enhanced matrix heatmap with better visualization"""
            if go is None:
                return None
            
            try:
                # Convert Gradio DataFrame to Python list
                if hasattr(matrix_data, 'values'):
                    # It's a pandas/Gradio DataFrame
                    matrix_list = matrix_data.values.tolist()
                elif isinstance(matrix_data, list):
                    matrix_list = matrix_data
                else:
                    # Try to convert to list
                    matrix_list = list(matrix_data) if matrix_data is not None else []
                
                # Ensure we have valid data
                if not matrix_list or len(matrix_list) == 0:
                    return go.Figure().add_annotation(text="No data", x=0.5, y=0.5)
                
                # Convert strings to floats if needed
                cleaned_data = []
                for row in matrix_list:
                    if isinstance(row, list):
                        cleaned_row = []
                        for val in row:
                            try:
                                cleaned_row.append(float(val) if val is not None else 0.0)
                            except (ValueError, TypeError):
                                cleaned_row.append(0.0)
                        cleaned_data.append(cleaned_row)
                    else:
                        # Single value row
                        try:
                            cleaned_data.append([float(row) if row is not None else 0.0])
                        except (ValueError, TypeError):
                            cleaned_data.append([0.0])
                
                if not cleaned_data or not cleaned_data[0]:
                    return go.Figure().add_annotation(text="No valid data", x=0.5, y=0.5)
                
                z_data = np.array(cleaned_data) if np is not None else cleaned_data
                
                fig = go.Figure(data=go.Heatmap(
                    z=z_data,
                    colorscale=colorscale,
                    showscale=True,
                    text=[[f"{val:.3f}" for val in row] for row in cleaned_data] if show_values else None,
                    texttemplate="%{text}" if show_values else None,
                    textfont={"size": 12, "color": "white"},
                    hoverongaps=False,
                    colorbar=dict(title="Value"),
                ))
                
                rows, cols = len(cleaned_data), len(cleaned_data[0]) if cleaned_data else (0, 0)
                
                fig.update_layout(
                    title=dict(text=f"<b>{title}</b><br>Size: {rows}×{cols}", x=0.5),
                    width=500,
                    height=400,
                    xaxis_title="<b>Columns</b>",
                    yaxis_title="<b>Rows</b>",
                    font=dict(size=12),
                    margin=dict(l=80, r=80, t=100, b=80)
                )
                
                # Add grid lines for better readability
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                
                return fig
            except Exception as e:
                logger.error(f"Error creating heatmap: {e}")
                return go.Figure().add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5)

        def create_enhanced_vector_plot(vector_data, title: str, color: str = "blue") -> "go.Figure":  # type: ignore[name-defined]
            """Create an enhanced vector visualization"""
            if go is None:
                return None
            
            try:
                # Convert Gradio DataFrame to Python list
                if hasattr(vector_data, 'values'):
                    # It's a pandas/Gradio DataFrame
                    vector_list = vector_data.values.tolist()
                elif isinstance(vector_data, list):
                    vector_list = vector_data
                else:
                    # Try to convert to list
                    vector_list = list(vector_data) if vector_data is not None else []
                
                # Extract values and convert to floats
                values = []
                for item in vector_list:
                    if isinstance(item, list):
                        # Take first element if it's a list
                        try:
                            val = float(item[0]) if len(item) > 0 and item[0] is not None else 0.0
                        except (ValueError, TypeError):
                            val = 0.0
                        values.append(val)
                    else:
                        # Single value
                        try:
                            val = float(item) if item is not None else 0.0
                        except (ValueError, TypeError):
                            val = 0.0
                        values.append(val)
                
                if not values:
                    return go.Figure().add_annotation(text="No vector data", x=0.5, y=0.5)
                
                indices = list(range(len(values)))
                
                fig = go.Figure()
                
                # Add bar plot
                fig.add_trace(go.Bar(
                    x=indices,
                    y=values,
                    marker_color=color,
                    text=[f"{val:.3f}" for val in values],
                    textposition='outside',
                    textfont=dict(size=11),
                    name=title
                ))
                
                # Add line plot for trend
                fig.add_trace(go.Scatter(
                    x=indices,
                    y=values,
                    mode='lines+markers',
                    line=dict(color='red', width=2),
                    marker=dict(size=6),
                    name='Trend',
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title=dict(text=f"<b>{title}</b><br>Size: {len(values)}", x=0.5),
                    width=400,
                    height=300,
                    xaxis_title="<b>Index</b>",
                    yaxis_title="<b>Value</b>",
                    font=dict(size=11),
                    showlegend=False,
                    margin=dict(l=60, r=60, t=80, b=60)
                )
                
                # Add secondary y-axis for trend line
                fig.update_layout(yaxis2=dict(overlaying='y', side='right', showticklabels=False))
                
                return fig
            except Exception as e:
                logger.error(f"Error creating vector plot: {e}")
                return go.Figure().add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5)
        
        def resize_matrix(matrix_data: List[List[float]], new_rows: int, new_cols: int, 
                         fill_value: float = 0.0) -> List[List[float]]:
            """Resize matrix to new dimensions, preserving existing values"""
            if not matrix_data:
                return [[fill_value for _ in range(new_cols)] for _ in range(new_rows)]
            
            current_rows, current_cols = len(matrix_data), len(matrix_data[0]) if matrix_data else 0
            new_matrix = []
            
            for i in range(new_rows):
                row = []
                for j in range(new_cols):
                    if i < current_rows and j < current_cols:
                        # Preserve existing value
                        row.append(matrix_data[i][j])
                    elif i < current_rows and j >= current_cols:
                        # New column, use normalized probability
                        row.append(0.1)
                    elif i >= current_rows and j < current_cols:
                        # New row, use identity-like pattern
                        row.append(1.0 if i == j else 0.1)
                    else:
                        # New row and column
                        row.append(fill_value)
                new_matrix.append(row)
            
            return new_matrix
        
        def resize_vector(vector_data: List, new_size: int, fill_value: float = 0.33) -> List[List[float]]:
            """Resize vector to new size, preserving existing values"""
            current_values = [row[0] if isinstance(row, list) else row for row in vector_data]
            current_size = len(current_values)
            
            if new_size == current_size:
                return [[val] for val in current_values]
            elif new_size > current_size:
                # Add new elements
                new_values = current_values + [fill_value] * (new_size - current_size)
            else:
                # Truncate
                new_values = current_values[:new_size]
            
            return [[val] for val in new_values]
        
        def calculate_matrix_stats(matrix_data, name: str) -> str:
            """Calculate and format matrix statistics"""
            try:
                # Convert Gradio DataFrame to Python list
                if hasattr(matrix_data, 'values'):
                    # It's a pandas/Gradio DataFrame
                    matrix_list = matrix_data.values.tolist()
                elif isinstance(matrix_data, list):
                    matrix_list = matrix_data
                else:
                    matrix_list = []
                
                if not matrix_list:
                    return f"**{name}**: No data"
                
                # Convert to floats and flatten
                flat_values = []
                for row in matrix_list:
                    if isinstance(row, list):
                        for val in row:
                            try:
                                flat_values.append(float(val) if val is not None else 0.0)
                            except (ValueError, TypeError):
                                flat_values.append(0.0)
                    else:
                        try:
                            flat_values.append(float(row) if row is not None else 0.0)
                        except (ValueError, TypeError):
                            flat_values.append(0.0)
                
                if not flat_values:
                    return f"**{name}**: No valid data"
                
                # Calculate shape
                if matrix_list and isinstance(matrix_list[0], list):
                    shape = f"{len(matrix_list)}×{len(matrix_list[0])}"
                else:
                    shape = f"{len(matrix_list)}"
                
                stats = {
                    "Shape": shape,
                    "Min": f"{min(flat_values):.3f}",
                    "Max": f"{max(flat_values):.3f}",
                    "Mean": f"{sum(flat_values)/len(flat_values):.3f}",
                    "Sum": f"{sum(flat_values):.3f}"
                }
                
                stats_text = f"**{name}**:\n"
                for key, value in stats.items():
                    stats_text += f"- {key}: `{value}`\n"
                
                return stats_text
            except Exception as e:
                return f"**{name}**: Error calculating stats - {e}"

        def generate_gnn_from_matrices(a_data, b_data, c_data, d_data, b_slice):
            """Generate GNN markdown from matrix data"""
            try:
                # Convert Gradio DataFrames to Python lists
                def convert_df_to_list(data):
                    if hasattr(data, 'values'):
                        return data.values.tolist()
                    elif isinstance(data, list):
                        return data
                    else:
                        return []
                
                a_list = convert_df_to_list(a_data)
                b_list = convert_df_to_list(b_data)
                c_list = convert_df_to_list(c_data)
                d_list = convert_df_to_list(d_data)
                
                # Extract vector values properly
                c_values = []
                for item in c_list:
                    if isinstance(item, list) and len(item) > 0:
                        try:
                            c_values.append(float(item[0]))
                        except (ValueError, TypeError):
                            c_values.append(0.0)
                    else:
                        c_values.append(0.0)
                
                d_values = []
                for item in d_list:
                    if isinstance(item, list) and len(item) > 0:
                        try:
                            d_values.append(float(item[0]))
                        except (ValueError, TypeError):
                            d_values.append(0.0)
                    else:
                        d_values.append(0.0)
                
                # Convert dataframe data to visual format
                visual_matrices = {
                    "A": {
                        "type": "matrix",
                        "rows": len(a_list),
                        "cols": len(a_list[0]) if a_list and len(a_list) > 0 else 0,
                        "values": a_list,
                        "description": "Likelihood matrix"
                    },
                    "C": {
                        "type": "vector", 
                        "size": len(c_values),
                        "values": c_values,
                        "description": "Preference vector"
                    },
                    "D": {
                        "type": "vector",
                        "size": len(d_values), 
                        "values": d_values,
                        "description": "Prior vector"
                    },
                    "B": {
                        "type": "tensor",
                        "depth": 3,  # Simplified for now
                        "rows": len(b_list),
                        "cols": len(b_list[0]) if b_list and len(b_list) > 0 else 0,
                        "values": [b_list, b_list, b_list],  # Simplified - all slices same
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
                
                # Convert DataFrames to lists
                def convert_df_to_list(data):
                    if hasattr(data, 'values'):
                        return data.values.tolist()
                    elif isinstance(data, list):
                        return data
                    else:
                        return []
                
                a_list = convert_df_to_list(a_data)
                b_list = convert_df_to_list(b_data)
                c_list = convert_df_to_list(c_data)
                d_list = convert_df_to_list(d_data)
                
                # Basic dimension checks
                if not a_list or (isinstance(a_list[0], list) and not a_list[0]):
                    errors.append("A matrix cannot be empty")
                if not b_list or (isinstance(b_list[0], list) and not b_list[0]):
                    errors.append("B matrix cannot be empty") 
                if not c_list:
                    errors.append("C vector cannot be empty")
                if not d_list:
                    errors.append("D vector cannot be empty")
                
                # Consistency checks
                if a_list and d_list:
                    a_cols = len(a_list[0]) if a_list and isinstance(a_list[0], list) else 0
                    d_size = len(d_list)
                    if a_cols != d_size:
                        errors.append(f"A matrix columns ({a_cols}) must match D vector size ({d_size})")
                
                if a_list and c_list:
                    a_rows = len(a_list)
                    c_size = len(c_list)
                    if a_rows != c_size:
                        errors.append(f"A matrix rows ({a_rows}) must match C vector size ({c_size})")
                
                if errors:
                    return "❌ **Validation Errors:**\n" + "\n".join(f"- {e}" for e in errors)
                else:
                    return "✅ **Validation Passed:** All matrix dimensions are consistent"
                    
            except Exception as e:
                return f"❌ Validation failed: {e}"

        # === ENHANCED INTERACTIVE EVENT HANDLERS ===
        
        # Matrix A dimension control handlers
        def update_a_dimensions(current_state, delta_rows=0, delta_cols=0):
            """Update Matrix A dimensions and resize data"""
            state = current_state.copy()
            a_info = state["A"]
            
            new_rows = max(1, min(10, a_info["rows"] + delta_rows))
            new_cols = max(1, min(10, a_info["cols"] + delta_cols))
            
            if new_rows != a_info["rows"] or new_cols != a_info["cols"]:
                # Resize matrix
                new_values = resize_matrix(a_info["values"], new_rows, new_cols, 0.1)
                state["A"] = {"rows": new_rows, "cols": new_cols, "values": new_values}
                
                # Create new headers
                headers = [f"State_{i}" for i in range(new_cols)]
                
                return (state, 
                       f"**Matrix A**: `{new_rows}×{new_cols}` (Observations × States)",
                       f"**{new_rows}**", f"**{new_cols}**",
                       gr.Dataframe(value=new_values, headers=headers, interactive=True, 
                                  row_count=new_rows, col_count=new_cols))
            
            return (state, gr.update(), gr.update(), gr.update(), gr.update())
        
        # Matrix B dimension control handlers  
        def update_b_dimensions(current_state, delta_states=0, delta_actions=0):
            """Update Matrix B dimensions and resize data"""
            state = current_state.copy()
            b_info = state["B"]
            
            new_states = max(1, min(10, b_info["rows"] + delta_states))
            new_actions = max(1, min(10, b_info["depth"] + delta_actions))
            
            if new_states != b_info["rows"] or new_actions != b_info["depth"]:
                # Resize tensor (simplified - just current slice for now)
                current_slice = b_info.get("current_slice", 0)
                current_slice = min(current_slice, new_actions - 1)
                
                new_slice_values = resize_matrix(b_info["values"][min(current_slice, len(b_info["values"])-1)] 
                                               if b_info["values"] else [], new_states, new_states, 0.0)
                
                state["B"] = {
                    "depth": new_actions, "rows": new_states, "cols": new_states, 
                    "current_slice": current_slice,
                    "values": [new_slice_values for _ in range(new_actions)]
                }
                
                headers = [f"State_{i}" for i in range(new_states)]
                
                return (state,
                       f"**Matrix B**: `{new_states}×{new_states}×{new_actions}` (States × States × Actions)",
                       f"**{new_states}**", f"**{new_actions}**",
                       gr.Slider(maximum=new_actions-1, value=current_slice),
                       gr.Dataframe(value=new_slice_values, headers=headers, interactive=True,
                                  row_count=new_states, col_count=new_states))
            
            return (state, gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
        
        # Vector size control handlers
        def update_c_size(current_state, delta_size=0):
            """Update C vector size"""
            state = current_state.copy()
            c_info = state["C"]
            new_size = max(1, min(10, c_info["size"] + delta_size))
            
            if new_size != c_info["size"]:
                new_values = resize_vector([[v] for v in c_info["values"]], new_size, 0.1)
                state["C"] = {"size": new_size, "values": [row[0] for row in new_values]}
                
                return (state,
                       f"**Vector C**: `{new_size}` (Observation Preferences)", 
                       f"**{new_size}**",
                       gr.Dataframe(value=new_values, headers=["Preference"], interactive=True,
                                  row_count=new_size, col_count=1))
            
            return (state, gr.update(), gr.update(), gr.update())
        
        def update_d_size(current_state, delta_size=0):
            """Update D vector size"""
            state = current_state.copy()
            d_info = state["D"]
            new_size = max(1, min(10, d_info["size"] + delta_size))
            
            if new_size != d_info["size"]:
                new_values = resize_vector([[v] for v in d_info["values"]], new_size, 0.33)
                state["D"] = {"size": new_size, "values": [row[0] for row in new_values]}
                
                return (state,
                       f"**Vector D**: `{new_size}` (State Prior)",
                       f"**{new_size}**", 
                       gr.Dataframe(value=new_values, headers=["Prior"], interactive=True,
                                  row_count=new_size, col_count=1))
            
            return (state, gr.update(), gr.update(), gr.update())
        
        # Comprehensive update function
        def update_all_with_state(current_state, a_data, b_data, c_data, d_data):
            """Update all visualizations and generate statistics"""
            try:
                # Update plots
                a_plot = create_enhanced_heatmap(a_data, "A Matrix (Likelihood)", "Blues")
                b_plot = create_enhanced_heatmap(b_data, "B Matrix (Transitions)", "Oranges") 
                c_plot = create_enhanced_vector_plot(c_data, "C Vector (Preferences)", "red")
                d_plot = create_enhanced_vector_plot(d_data, "D Vector (Prior)", "green")
                
                # Generate statistics
                stats_text = "### 📊 **Real-time Matrix Statistics**\n\n"
                stats_text += calculate_matrix_stats(a_data, "Matrix A") + "\n"
                stats_text += calculate_matrix_stats(b_data, "Matrix B") + "\n"
                stats_text += calculate_matrix_stats(c_data, "Vector C") + "\n" 
                stats_text += calculate_matrix_stats(d_data, "Vector D") + "\n"
                
                # Generate GNN
                gnn_text = generate_gnn_from_matrices(a_data, b_data, c_data, d_data, 0)
                
                return a_plot, b_plot, c_plot, d_plot, stats_text, gnn_text
                
            except Exception as e:
                logger.error(f"Error in update_all_with_state: {e}")
                return (gr.update(), gr.update(), gr.update(), gr.update(), 
                       f"Error updating: {e}", gr.update())
        
        def reset_to_pomdp():
            """Reset all matrices to POMDP template values"""
            default_state = {
                "A": {"rows": 3, "cols": 3, "values": [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]},
                "B": {"depth": 3, "rows": 3, "cols": 3, "current_slice": 0,
                      "values": [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] for _ in range(3)]},
                "C": {"size": 3, "values": [0.1, 0.1, 1.0]},
                "D": {"size": 3, "values": [0.33, 0.33, 0.33]}
            }
            
            return (default_state,
                   "**Matrix A**: `3×3` (Observations × States)", "**3**", "**3**",
                   gr.Dataframe(value=default_state["A"]["values"], interactive=True),
                   "**Matrix B**: `3×3×3` (States × States × Actions)", "**3**", "**3**",
                   gr.Dataframe(value=default_state["B"]["values"][0], interactive=True),
                   "**Vector C**: `3` (Observation Preferences)", "**3**",
                   gr.Dataframe(value=[[v] for v in default_state["C"]["values"]], interactive=True),
                   "**Vector D**: `3` (State Prior)", "**3**",
                   gr.Dataframe(value=[[v] for v in default_state["D"]["values"]], interactive=True))
        
        # === WIRE UP ALL INTERACTIVE EVENTS ===
        
        # Matrix A dimension controls
        a_rows_plus.click(lambda s: update_a_dimensions(s, delta_rows=1), 
                         inputs=[matrix_state], 
                         outputs=[matrix_state, a_size_display, a_rows_display, a_cols_display, a_values])
        
        a_rows_minus.click(lambda s: update_a_dimensions(s, delta_rows=-1), 
                          inputs=[matrix_state],
                          outputs=[matrix_state, a_size_display, a_rows_display, a_cols_display, a_values])
        
        a_cols_plus.click(lambda s: update_a_dimensions(s, delta_cols=1),
                         inputs=[matrix_state],
                         outputs=[matrix_state, a_size_display, a_rows_display, a_cols_display, a_values])
        
        a_cols_minus.click(lambda s: update_a_dimensions(s, delta_cols=-1),
                          inputs=[matrix_state], 
                          outputs=[matrix_state, a_size_display, a_rows_display, a_cols_display, a_values])
        
        # Matrix B dimension controls
        b_states_plus.click(lambda s: update_b_dimensions(s, delta_states=1),
                           inputs=[matrix_state],
                           outputs=[matrix_state, b_size_display, b_states_display, b_actions_display, 
                                  b_slice_selector, b_values])
        
        b_states_minus.click(lambda s: update_b_dimensions(s, delta_states=-1),
                            inputs=[matrix_state],
                            outputs=[matrix_state, b_size_display, b_states_display, b_actions_display,
                                   b_slice_selector, b_values])
        
        b_actions_plus.click(lambda s: update_b_dimensions(s, delta_actions=1),
                            inputs=[matrix_state],
                            outputs=[matrix_state, b_size_display, b_states_display, b_actions_display,
                                   b_slice_selector, b_values])
        
        b_actions_minus.click(lambda s: update_b_dimensions(s, delta_actions=-1),
                             inputs=[matrix_state],
                             outputs=[matrix_state, b_size_display, b_states_display, b_actions_display,
                                    b_slice_selector, b_values])
        
        # Vector size controls
        c_size_plus.click(lambda s: update_c_size(s, delta_size=1),
                         inputs=[matrix_state],
                         outputs=[matrix_state, c_size_display, c_size_display_num, c_values])
        
        c_size_minus.click(lambda s: update_c_size(s, delta_size=-1),
                          inputs=[matrix_state],
                          outputs=[matrix_state, c_size_display, c_size_display_num, c_values])
        
        d_size_plus.click(lambda s: update_d_size(s, delta_size=1),
                         inputs=[matrix_state],
                         outputs=[matrix_state, d_size_display, d_size_display_num, d_values])
        
        d_size_minus.click(lambda s: update_d_size(s, delta_size=-1),
                          inputs=[matrix_state],
                          outputs=[matrix_state, d_size_display, d_size_display_num, d_values])
        
        # Auto-update functionality
        def maybe_auto_update(auto_enabled, state, a_data, b_data, c_data, d_data):
            """Auto-update visualizations if enabled"""
            if auto_enabled:
                return update_all_with_state(state, a_data, b_data, c_data, d_data)
            return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
        
        # Connect auto-update to matrix changes
        for matrix_input in [a_values, b_values, c_values, d_values]:
            matrix_input.change(
                maybe_auto_update,
                inputs=[auto_update_checkbox, matrix_state, a_values, b_values, c_values, d_values],
                outputs=[matrix_a_plot, matrix_b_plot, c_plot, d_plot, stats_output, gnn_output]
            )
        
        # Manual update button
        manual_update_btn.click(
            update_all_with_state,
            inputs=[matrix_state, a_values, b_values, c_values, d_values],
            outputs=[matrix_a_plot, matrix_b_plot, c_plot, d_plot, stats_output, gnn_output]
        )
        
        # Reset button
        reset_btn.click(
            reset_to_pomdp,
            outputs=[matrix_state, a_size_display, a_rows_display, a_cols_display, a_values,
                    b_size_display, b_states_display, b_actions_display, b_values,
                    c_size_display, c_size_display_num, c_values,
                    d_size_display, d_size_display_num, d_values]
        )
        
        # File operations
        save_btn.click(save_gnn, inputs=[gnn_output], outputs=[save_status])
        validate_btn.click(validate_gnn, 
                          inputs=[a_values, b_values, c_values, d_values],
                          outputs=[validation_output])
        
        # Initialize on load
        demo.load(
            update_all_with_state,
            inputs=[matrix_state, a_values, b_values, c_values, d_values],
            outputs=[matrix_a_plot, matrix_b_plot, c_plot, d_plot, stats_output, gnn_output]
        )

    return demo
