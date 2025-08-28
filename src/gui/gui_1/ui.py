from __future__ import annotations

import logging
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


def build_gui(markdown_text: str, export_path: Path) -> "gr.Blocks":  # type: ignore[name-defined]
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

    with gr.Blocks(title="GNN Constructor") as demo:
        gr.Markdown("## GNN Constructor")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Controls")
                with gr.Tab("Components"):
                    component_name = gr.Textbox(label="Component name", value="example_component")
                    component_type = gr.Dropdown(["observation", "hidden", "action", "policy"], value="observation", label="Type")
                    state_list = gr.Textbox(label="States (comma-separated)")
                    add_button = gr.Button("Add Component")
                    replace_states_button = gr.Button("Replace States")
                    append_states_button = gr.Button("Append States")
                    remove_button = gr.Button("Remove Component")
                with gr.Tab("State Space"):
                    state_entries = gr.Dropdown(choices=initial_names, value=initial_selected, label="Select State Entry", interactive=True)
                    st_name = gr.Textbox(label="Name", value=init_name_val)
                    st_dims = gr.Textbox(label="Dimensions (comma-separated)", value=init_dims_val)
                    st_type = gr.Textbox(label="Type (optional)", value=init_type_val)
                    st_comment = gr.Textbox(label="Comment (optional)", value=init_comment_val)
                    st_refresh = gr.Button("Refresh State List")
                    st_add = gr.Button("Add State Entry")
                    st_update = gr.Button("Update Selected")
                    st_remove = gr.Button("Remove Selected")
                save_button = gr.Button("Save Markdown")
            with gr.Column(scale=2):
                markdown_editor = gr.Code(value=markdown_text, language="markdown", label="GNN Markdown")

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

        def save_md(md: str) -> str:
            export_path.write_text(md)
            return f"Saved to {export_path}"

        add_button.click(add_component, inputs=[markdown_editor, component_name, component_type, state_list], outputs=[markdown_editor])
        replace_states_button.click(replace_states, inputs=[markdown_editor, component_name, state_list], outputs=[markdown_editor])
        append_states_button.click(append_states, inputs=[markdown_editor, component_name, state_list], outputs=[markdown_editor])
        remove_button.click(remove_component, inputs=[markdown_editor, component_name], outputs=[markdown_editor])
        save_status = gr.Markdown()
        save_button.click(save_md, inputs=[markdown_editor], outputs=[save_status])

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


