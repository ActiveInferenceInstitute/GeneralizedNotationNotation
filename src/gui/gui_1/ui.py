from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

try:
    import gradio as gr
except Exception:
    gr = None  # type: ignore

from .markdown import (
    add_component_to_markdown,
    update_component_states,
    remove_component_from_markdown,
    parse_state_space_from_markdown,
    add_state_space_entry,
    update_state_space_entry,
    remove_state_space_entry,
)


def build_gui(markdown_text: str, export_path: Path, logger: logging.Logger = None) -> "gr.Blocks":  # type: ignore[name-defined]
    if gr is None:
        raise RuntimeError("Gradio not available")

    # Compute initial state-space options and selected entry
    initial_items = parse_state_space_from_markdown(markdown_text)
    initial_names = [str(i.get("name", "")) for i in initial_items if i.get("name")]
    initial_selected = initial_names[0] if initial_names else ""
    def _vals_for(name: str):
        for e in initial_items:
            if str(e.get("name")) == name:
                dims_csv = ", ".join(str(d) for d in e.get("dims", []))
                typ = str(e.get("type", ""))
                cmt = str(e.get("comment", ""))
                return str(e.get("name", "")), dims_csv, typ, cmt
        return "", "", "", ""
    init_name_val, init_dims_val, init_type_val, init_comment_val = _vals_for(initial_selected)

    with gr.Blocks(title="🔧 GNN Form-based Constructor", theme=gr.themes.Base()) as demo:
        gr.Markdown("""# 🔧 GNN Form-based Constructor
        
        **Interactive two-pane editor for systematic GNN model construction**
        
        - **Left Panel**: Component and state space management controls
        - **Right Panel**: Live-updating GNN markdown editor
        - **Real-time Sync**: Changes are immediately reflected in both panels
        """)
        
        # Status indicator
        status_display = gr.Markdown("🟢 **Status**: Ready for component editing")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🛠️ Construction Controls")
                with gr.Tab("📋 Components"):
                    gr.Markdown("**Add or modify GNN model components**")
                    component_name = gr.Textbox(
                        label="Component name", 
                        value="example_component",
                        placeholder="Enter unique component name..."
                    )
                    component_type = gr.Dropdown(
                        ["observation", "hidden", "action", "policy"], 
                        value="observation", 
                        label="Component Type (Select Active Inference component type)"
                    )
                    state_list = gr.Textbox(
                        label="States (comma-separated)",
                        placeholder="s1, s2, s3..."
                    )
                    add_button = gr.Button("➕ Add Component", variant="primary")
                    replace_states_button = gr.Button("🔄 Replace States")
                    append_states_button = gr.Button("📄 Append States")
                    remove_button = gr.Button("🗑️ Remove Component", variant="stop")
                with gr.Tab("🏗️ State Space"):
                    gr.Markdown("**Manage state space entries and dimensions**")
                    state_entries = gr.Dropdown(
                        choices=initial_names, 
                        value=initial_selected, 
                        label="Select State Entry (Choose entry to edit)", 
                        interactive=True
                    )
                    st_name = gr.Textbox(
                        label="Name", 
                        value=init_name_val,
                        placeholder="Enter state name..."
                    )
                    st_dims = gr.Textbox(
                        label="Dimensions (comma-separated)", 
                        value=init_dims_val,
                        placeholder="e.g., 3, 4, 2"
                    )
                    st_type = gr.Textbox(
                        label="Type (optional)", 
                        value=init_type_val,
                        placeholder="e.g., continuous, discrete"
                    )
                    st_comment = gr.Textbox(
                        label="Comment (optional)", 
                        value=init_comment_val,
                        placeholder="Description or notes..."
                    )
                    with gr.Row():
                        st_refresh = gr.Button("🔄 Refresh State List")
                        st_add = gr.Button("➕ Add State Entry", variant="primary")
                    with gr.Row():
                        st_update = gr.Button("✏️ Update Selected")
                        st_remove = gr.Button("🗑️ Remove Selected", variant="stop")
                
                # Action buttons
                with gr.Row():
                    save_button = gr.Button("💾 Save Markdown", variant="primary", scale=2)
                    export_button = gr.Button("📄 Export Model", variant="secondary")
            with gr.Column(scale=2):
                gr.Markdown("### 📝 Live GNN Markdown Editor")
                markdown_editor = gr.Code(
                    value=markdown_text, 
                    language="markdown", 
                    label="GNN Markdown - Real-time synchronized model specification",
                    lines=25
                )
                
                # Model statistics and validation
                model_stats = gr.JSON(
                    label="📊 Model Statistics",
                    value={"components": 1, "state_entries": len(initial_items), "total_states": sum(len(item.get("dims", [])) for item in initial_items) if initial_items else 0}
                )

        def _split_states(s: str) -> list[str]:
            return [x.strip() for x in s.split(",") if x.strip()]

        def add_component(md: str, name: str, ctype: str, states_csv: str) -> str:
            return add_component_to_markdown(md, name, ctype, _split_states(states_csv))

        def replace_states(md: str, name: str, states_csv: str) -> str:
            return update_component_states(md, name, _split_states(states_csv), mode="replace")

        def append_states(md: str, name: str, states_csv: str) -> str:
            return update_component_states(md, name, _split_states(states_csv), mode="append")

        def remove_component(md: str, name: str) -> str:
            return remove_component_from_markdown(md, name)

        def save_md(md: str):
            export_path.write_text(md)
            if logger:
                logger.info(f"📄 Model saved to {export_path}")
            save_message = f"✅ **Saved Successfully**\n\nFile: `{export_path}`\nSize: {len(md)} characters"
            
            # Calculate updated statistics
            items = parse_state_space_from_markdown(md)
            stats = {
                "components": len([line for line in md.split('\n') if 'name:' in line and 'type:' in line]),
                "state_entries": len(items),
                "total_states": sum(len(item.get("dims", [])) for item in items) if items else 0,
                "file_size_chars": len(md),
                "last_saved": datetime.now().strftime("%H:%M:%S")
            }
            
            return save_message, "🟢 **Status**: Model saved successfully", stats

        # Status messages and save functionality
        save_status = gr.Markdown("💾 **Ready to save**: Use the save button to export your model")
        
        # Component management event handlers
        add_button.click(
            add_component, 
            inputs=[markdown_editor, component_name, component_type, state_list], 
            outputs=[markdown_editor]
        ).then(
            lambda: "🟡 **Status**: Component added - remember to save your changes",
            outputs=[status_display]
        )
        
        replace_states_button.click(
            replace_states, 
            inputs=[markdown_editor, component_name, state_list], 
            outputs=[markdown_editor]
        ).then(
            lambda: "🟡 **Status**: States replaced - remember to save your changes",
            outputs=[status_display]
        )
        
        append_states_button.click(
            append_states, 
            inputs=[markdown_editor, component_name, state_list], 
            outputs=[markdown_editor]
        ).then(
            lambda: "🟡 **Status**: States appended - remember to save your changes", 
            outputs=[status_display]
        )
        
        remove_button.click(
            remove_component, 
            inputs=[markdown_editor, component_name], 
            outputs=[markdown_editor]
        ).then(
            lambda: "🟡 **Status**: Component removed - remember to save your changes",
            outputs=[status_display]
        )
        
        # Save functionality with enhanced feedback
        save_button.click(
            save_md, 
            inputs=[markdown_editor], 
            outputs=[save_status, status_display, model_stats]
        )

        # State Space actions
        def _compute_state_choices(md: str) -> list[str]:
            items = parse_state_space_from_markdown(md)
            return [str(i.get("name", "")) for i in items if i.get("name")]

        def refresh_states(md: str):
            return gr.update(choices=_compute_state_choices(md))

        def add_state(md: str, name: str, dims_csv: str, typ: str, comment: str):
            dims = [int(x.strip()) for x in dims_csv.split(",") if x.strip().isdigit()]
            return add_state_space_entry(md, name, dims, typ or None, comment or None)

        def update_state(md: str, selected: str, name: str, dims_csv: str, typ: str, comment: str):
            dims = [int(x.strip()) for x in dims_csv.split(",") if x.strip().isdigit()]
            return update_state_space_entry(md, selected or name, name or (selected or ""), dims, typ or None, comment or None)

        def remove_state(md: str, selected: str):
            if not selected:
                return md
            return remove_state_space_entry(md, selected)

        st_refresh.click(refresh_states, inputs=[markdown_editor], outputs=[state_entries])
        markdown_editor.change(refresh_states, inputs=[markdown_editor], outputs=[state_entries])
        st_add.click(add_state, inputs=[markdown_editor, st_name, st_dims, st_type, st_comment], outputs=[markdown_editor])
        st_update.click(update_state, inputs=[markdown_editor, state_entries, st_name, st_dims, st_type, st_comment], outputs=[markdown_editor])
        st_remove.click(remove_state, inputs=[markdown_editor, state_entries], outputs=[markdown_editor])

        def populate_fields(md: str, selected: str):
            entries = parse_state_space_from_markdown(md)
            for e in entries:
                if str(e.get("name")) == selected:
                    dims = ", ".join(str(d) for d in e.get("dims", []))
                    typ = str(e.get("type", ""))
                    cmt = str(e.get("comment", ""))
                    return [e.get("name", ""), dims, typ, cmt]
            return ["", "", "", ""]

        state_entries.change(populate_fields, inputs=[markdown_editor, state_entries], outputs=[st_name, st_dims, st_type, st_comment])

        # Real-time editing: typing in any field applies update and keeps dropdown synced
        def update_state_live(md: str, selected: str, name: str, dims_csv: str, typ: str, comment: str):
            dims = [int(x.strip()) for x in dims_csv.split(",") if x.strip().isdigit()]
            new_md = update_state_space_entry(md, selected or name, name or (selected or ""), dims, typ or None, comment or None)
            choices = _compute_state_choices(new_md)
            new_selected = name or (selected if selected in choices else (choices[0] if choices else ""))
            return new_md, gr.update(choices=choices, value=new_selected)

        st_name.change(update_state_live, inputs=[markdown_editor, state_entries, st_name, st_dims, st_type, st_comment], outputs=[markdown_editor, state_entries])
        st_dims.change(update_state_live, inputs=[markdown_editor, state_entries, st_name, st_dims, st_type, st_comment], outputs=[markdown_editor, state_entries])
        st_type.change(update_state_live, inputs=[markdown_editor, state_entries, st_name, st_dims, st_type, st_comment], outputs=[markdown_editor, state_entries])
        st_comment.change(update_state_live, inputs=[markdown_editor, state_entries, st_name, st_dims, st_type, st_comment], outputs=[markdown_editor, state_entries])

    return demo


