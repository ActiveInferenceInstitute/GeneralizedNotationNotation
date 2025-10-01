#!/usr/bin/env python3
"""
GUI 3: State Space Design Studio Processor
Low-dependency visual design interface for Active Inference models
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

# Check for Gradio availability
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

def run_gui(target_dir: Path, output_dir: Path, logger: logging.Logger, **kwargs) -> bool:
    """
    Launch the State Space Design Studio GUI
    
    Args:
        target_dir: Directory containing GNN files
        output_dir: Output directory for GUI results  
        logger: Logger instance
        **kwargs: Additional arguments
        
    Returns:
        True if GUI launched successfully, False otherwise
    """
    
    try:
        output_root = output_dir.parent if output_dir.name.endswith('_output') else output_dir
        output_root.mkdir(parents=True, exist_ok=True)
        
        # Setup export path
        export_filename = kwargs.get('export_filename', 'designed_model_gui_3.md') 
        starter_path = output_root / export_filename
        
        # Load starter GNN content
        starter_md = _load_starter_content(target_dir, logger)
        
        # Check if running in headless mode
        headless = kwargs.get('headless', False)
        open_browser = kwargs.get('open_browser', False)
        
        if headless or not GRADIO_AVAILABLE:
            # Generate design artifacts without launching GUI
            if not GRADIO_AVAILABLE:
                logger.warning("⚠️ Gradio not available - generating fallback artifacts only")
                logger.info("💡 Install GUI support with: uv pip install -e .[gui]")
            else:
                logger.info("📦 Running GUI 3 in HEADLESS mode - generating artifacts only")
            
            design_analysis = _analyze_gnn_design(starter_md)
            
            # Write starter model to file
            starter_path.write_text(starter_md)
            
            # Save design analysis
            analysis_file = output_root / "design_analysis.json"
            analysis_file.write_text(json.dumps({
                "gui_type": "design_studio",
                "backend": "gradio" if GRADIO_AVAILABLE else "none",
                "status": "headless_mode" if GRADIO_AVAILABLE else "fallback_mode",
                "analysis": design_analysis,
                "export_path": str(starter_path),
                "headless_mode": True,
                "recommendations": [
                    "Run with --interactive to launch GUI server on port 7862"
                ] if GRADIO_AVAILABLE else [
                    "Install gradio with: uv pip install gradio>=4.0.0",
                    "Run with --interactive for full GUI experience"
                ]
            }, indent=2))
            
            logger.info(f"🎨 Design analysis saved to: {analysis_file}")
            return True
            
        # Interactive mode - build the Design Studio GUI
        logger.info("🔧 Building State Space Design Studio...")
        from .ui_designer import build_design_studio
        demo = build_design_studio(
            markdown_text=starter_md,
            export_path=starter_path, 
            logger=logger
        )
        logger.info("✅ State Space Design Studio UI built successfully")
        
        # Launch GUI
        logger.info(f"🌐 Launching GUI 3 on http://localhost:7862 (open_browser={open_browser})")
        
        import threading
        import time
        
        def launch_gui():
            logger.info("🎨 Design Studio starting...")
            demo.launch(
                share=False,
                prevent_thread_lock=False,  # Let the thread properly block on the server
                server_name="0.0.0.0",
                server_port=7862,
                inbrowser=open_browser,
                show_error=True,
                quiet=False,  # Show server startup messages
            )

        gui_thread = threading.Thread(target=launch_gui, daemon=False)
        gui_thread.start()
        time.sleep(3)
        logger.info("🎨 Design Studio is running on http://localhost:7862")
        logger.info("🔍 Features: Visual state space design, ontology editing, connection graphs, low-dependency approach")
        
        # Save launch status
        status_file = output_root / "design_studio_status.json"
        status_file.write_text(json.dumps({
            "gui_type": "design_studio",
            "backend": "gradio",
            "launched": True,
            "export_file": str(starter_path),
            "port": 7862,
            "url": "http://localhost:7862",
            "features": [
                "State space visual designer",
                "Ontology term editor",
                "Connection graph interface", 
                "Parameter tuning controls"
            ]
        }, indent=2))
        
        return True
        
    except Exception as e:
        logger.error(f"❌ GUI 3 launch failed: {e}")
        return False


def _load_starter_content(target_dir: Path, logger: logging.Logger) -> str:
    """Load starter GNN content from target directory"""
    
    # Look for the POMDP agent file specifically
    pomdp_file = target_dir / "actinf_pomdp_agent.md"
    if pomdp_file.exists():
        logger.info(f"📖 Loading POMDP agent model: {pomdp_file}")
        return pomdp_file.read_text()
        
    # Fallback to any GNN file
    gnn_files = list(target_dir.glob("*.md"))
    if gnn_files:
        logger.info(f"📖 Loading GNN file: {gnn_files[0]}")
        return gnn_files[0].read_text()
        
    # Default template if no files found
    logger.warning("⚠️ No GNN files found, using default POMDP template")
    return _get_default_pomdp_template()


def _get_default_pomdp_template() -> str:
    """Get default POMDP template for design studio"""
    return '''# GNN Example: Active Inference POMDP Agent (Design Studio)
# GNN Version: 1.0

## ModelName
Active Inference POMDP Agent - Design Studio Template

## StateSpaceBlock
A[3,3,type=float]   # Likelihood matrix
B[3,3,3,type=float] # Transition matrix
C[3,type=float]     # Preference vector
D[3,type=float]     # Prior vector

## Connections
D>s
s-A
A-o
s-B
B>u

## ActInfOntologyAnnotation
A=LikelihoodMatrix
B=TransitionMatrix
C=LogPreferenceVector
D=PriorOverHiddenStates

## ModelParameters
num_hidden_states: 3
num_obs: 3
num_actions: 3
'''


def _analyze_gnn_design(gnn_content: str) -> Dict[str, Any]:
    """Analyze GNN content for design studio insights"""
    
    analysis = {
        "state_spaces": [],
        "ontology_terms": {},
        "connections": [],
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
            if '[' in line and ']' in line:
                # Extract state space definitions
                var_name = line.split('[')[0]
                dimensions = line.split('[')[1].split(']')[0]
                analysis["state_spaces"].append({
                    "variable": var_name,
                    "dimensions": dimensions
                })
                
        elif current_section == "ActInfOntologyAnnotation":
            if '=' in line:
                var, concept = line.split('=', 1)
                analysis["ontology_terms"][var.strip()] = concept.strip()
                
        elif current_section == "Connections":
            if line and not line.startswith('#'):
                analysis["connections"].append(line)
                
        elif current_section == "ModelParameters":
            if ':' in line:
                param, value = line.split(':', 1)
                analysis["parameters"][param.strip()] = value.strip()
    
    return analysis
