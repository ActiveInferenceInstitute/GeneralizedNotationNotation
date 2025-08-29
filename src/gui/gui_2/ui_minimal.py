#!/usr/bin/env python3
"""
Ultra-minimal Visual Matrix Editor UI for debugging gray screen issue
"""

import logging
from pathlib import Path

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False


def build_minimal_visual_gui(markdown_text: str, export_path: Path, logger: logging.Logger) -> "gr.Blocks":
    """Build ultra-minimal GUI to debug gray screen issue"""
    
    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio is required for GUI functionality")

    logger.info("🔧 Building ultra-minimal GUI...")

    with gr.Blocks(title="Matrix Editor - Debug") as demo:
        gr.Markdown("# 🔧 Matrix Editor Debug Version")
        gr.Markdown("Testing basic functionality...")
        
        # Ultra simple components
        text_input = gr.Textbox(label="Test Input", value="Hello")
        text_output = gr.Textbox(label="Test Output", value="Ready")
        test_button = gr.Button("Test Button")
        
        def simple_test(input_text):
            return f"Processed: {input_text}"
        
        test_button.click(simple_test, inputs=[text_input], outputs=[text_output])
        
        gr.Markdown("---")
        gr.Markdown("If you can see this text and the button works, the basic UI is functional.")

    logger.info("✅ Ultra-minimal GUI built successfully")
    return demo
