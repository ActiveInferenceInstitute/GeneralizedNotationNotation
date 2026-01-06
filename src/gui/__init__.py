"""
GUI module for Interactive GNN Constructors.

This module provides multiple GUI implementations:
- GUI 1: Form-based Interactive GNN Constructor  
- GUI 2: Visual Matrix Editor with drag-and-drop
- GUI 3: State Space Design Studio
- oxdraw: Visual diagram-as-code interface

Public API:
- process_gui: main processing function (runs all available GUIs)
- gui_1: form-based GUI with component management
- gui_2: visual matrix editor with drag-and-drop
- gui_3: state space design studio
- oxdraw: visual diagram-as-code with Mermaid
- get_available_guis: list all available GUI implementations
"""

# Import GUI runners
from .gui_1 import gui_1, get_gui_1_info
from .gui_2 import gui_2, get_gui_2_info
from .gui_3 import gui_3, get_gui_3_info
from .oxdraw import oxdraw_gui, get_oxdraw_info

# Import GUI 1 utilities
from .gui_1 import (
    add_component_to_markdown,
    update_component_states,
    remove_component_from_markdown,
    parse_components_from_markdown,
    parse_state_space_from_markdown,
    add_state_space_entry,
    update_state_space_entry,
    remove_state_space_entry,
)

def get_available_guis():
    """Get list of available GUI implementations with their info"""
    return {
        "gui_1": get_gui_1_info(),
        "gui_2": get_gui_2_info(),
        "gui_3": get_gui_3_info(),
        "oxdraw": get_oxdraw_info(),
    }

def process_gui(target_dir, output_dir, verbose=False, **kwargs):
    """
    Main processing function for GUI module.
    
    By default, runs all available GUI implementations in headless mode.
    Can be restricted using gui_types parameter.
    
    Args:
        target_dir: Directory containing files to process
        output_dir: Output directory for results
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options
            - gui_types: List of GUI types to run (default: gui_1, gui_2)
            - headless: Run in headless mode (default: True for pipeline)
            - interactive: Launch interactive GUI servers (overrides headless)
            - open_browser: Whether to open browser for interactive GUIs
        
    Returns:
        Boolean indicating success of all GUI runs
    """
    import logging
    from pathlib import Path
    import json
    
    logger = logging.getLogger(__name__)
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    # Handle interactive vs headless mode
    # Interactive mode overrides headless
    interactive = kwargs.get('interactive', False)
    if interactive:
        kwargs['headless'] = False
        logger.info("🎮 Running in INTERACTIVE mode - will launch GUI servers")
    else:
        # Default to headless mode for pipeline integration
        kwargs['headless'] = kwargs.get('headless', True)
        if kwargs['headless']:
            logger.info("📦 Running in HEADLESS mode - generating artifacts only (fast)")
    
    # Determine which GUIs to run
    gui_types = kwargs.get('gui_types', 'gui_1,gui_2')
    if isinstance(gui_types, str):
        gui_types = [g.strip() for g in gui_types.split(',')]
    
    # Prepare kwargs for GUI functions - remove keys that are passed explicitly
    gui_kwargs = {k: v for k, v in kwargs.items() 
                  if k not in ['logger', 'target_dir', 'output_dir', 'verbose']}
    
    results = {}
    overall_success = True
    
    try:
        logger.info(f"Processing GUI module for files in {target_dir}")
        logger.info(f"Running GUI types: {gui_types}")
        logger.info(f"Mode: {'INTERACTIVE' if not kwargs['headless'] else 'HEADLESS'}")
        
        # Run each requested GUI
        for gui_type in gui_types:
            try:
                if gui_type == 'gui_1':
                    result = gui_1(
                        target_dir=Path(target_dir), 
                        output_dir=Path(output_dir),
                        logger=logger,
                        verbose=verbose,
                        **gui_kwargs
                    )
                elif gui_type == 'gui_2':
                    result = gui_2(
                        target_dir=Path(target_dir),
                        output_dir=Path(output_dir), 
                        logger=logger,
                        verbose=verbose,
                        **gui_kwargs
                    )
                elif gui_type == 'gui_3':
                    result = gui_3(
                        target_dir=Path(target_dir),
                        output_dir=Path(output_dir),
                        logger=logger,
                        verbose=verbose,
                        **gui_kwargs
                    )
                elif gui_type == 'oxdraw':
                    result = oxdraw_gui(
                        target_dir=Path(target_dir),
                        output_dir=Path(output_dir),
                        logger=logger,
                        verbose=verbose,
                        **gui_kwargs
                    )
                else:
                    logger.warning(f"Unknown GUI type: {gui_type}")
                    result = {
                        "gui_type": gui_type,
                        "success": False,
                        "error": f"Unknown GUI type: {gui_type}"
                    }
                
                results[gui_type] = result
                if not result.get('success', False):
                    overall_success = False
                    
            except Exception as e:
                logger.error(f"GUI {gui_type} failed: {e}")
                results[gui_type] = {
                    "gui_type": gui_type,
                    "success": False,
                    "error": str(e)
                }
                overall_success = False
        
        # Save processing summary
        try:
            output_path = Path(output_dir)
            summary_file = output_path / "gui_processing_summary.json"
            summary_file.write_text(json.dumps({
                "mode": "interactive" if not kwargs['headless'] else "headless",
                "gui_types": gui_types,
                "results": results,
                "overall_success": overall_success
            }, indent=2))
            logger.info(f"📊 GUI processing summary saved to: {summary_file}")
        except Exception as e:
            logger.warning(f"Failed to save GUI processing summary: {e}")
        
        # Generate HTML navigation page for all outputs
        try:
            pipeline_output_dir = output_path.parent  # output/22_gui_output -> output/
            nav_success = generate_html_navigation(pipeline_output_dir, output_path, logger)
            if nav_success:
                logger.info("✅ HTML navigation page generated")
        except Exception as e:
            logger.warning(f"Failed to generate HTML navigation: {e}")
        
        return overall_success
        
    except Exception as e:
        logger.error(f"GUI processing failed: {e}")
        return False

def generate_html_navigation(pipeline_output_dir: Path, output_dir: Path, logger: logging.Logger) -> bool:
    """
    Generate HTML navigation page that links to all pipeline output types.
    
    Args:
        pipeline_output_dir: Directory containing all pipeline outputs (typically output/)
        output_dir: GUI output directory where navigation.html will be created
        logger: Logger instance
        
    Returns:
        True if navigation page generated successfully, False otherwise
    """
    try:
        logger.info("Generating HTML navigation page for pipeline outputs")
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define pipeline steps and their output directories
        pipeline_steps = [
            ("Template", "0_template_output", ["*.json", "*.md"]),
            ("Setup", "1_setup_output", ["*.json"]),
            ("Tests", "2_tests_output", ["*.txt", "*.json"]),
            ("GNN Processing", "3_gnn_output", ["*.json", "*.md", "*.pkl"]),
            ("Model Registry", "4_model_registry_output", ["*.json"]),
            ("Type Checker", "5_type_checker_output", ["*.json", "*.md"]),
            ("Validation", "6_validation_output", ["*.json"]),
            ("Export", "7_export_output", ["*.json", "*.xml", "*.pkl"]),
            ("Visualization", "8_visualization_output", ["*.png", "*.svg", "*.csv", "*.json"]),
            ("Advanced Visualization", "9_advanced_viz_output", ["*.png", "*.json"]),
            ("Ontology", "10_ontology_output", ["*.json"]),
            ("Render", "11_render_output", ["*.py", "*.jl", "*.md", "*.json", "*.png"]),
            ("Execute", "12_execute_output", ["*.txt", "*.json", "*.md", "*.png"]),
            ("LLM", "13_llm_output", ["*.md", "*.json"]),
            ("ML Integration", "14_ml_integration_output", ["*.json"]),
            ("Audio", "15_audio_output", ["*.json", "*.wav"]),
            ("Analysis", "16_analysis_output", ["*.json"]),
            ("Integration", "17_integration_output", ["*.json"]),
            ("Security", "18_security_output", ["*.json"]),
            ("Research", "19_research_output", ["*.json"]),
            ("Website", "20_website_output", ["*.html", "*.json"]),
            ("MCP", "21_mcp_output", ["*.json"]),
            ("GUI", "22_gui_output", ["*.md", "*.json"]),
            ("Report", "23_report_output", ["*.html", "*.md", "*.json"]),
        ]
        
        # Collect output information
        output_sections = []
        total_files = 0
        
        for step_name, step_dir, patterns in pipeline_steps:
            step_path = pipeline_output_dir / step_dir
            if not step_path.exists():
                continue
            
            step_files = []
            for pattern in patterns:
                for file_path in step_path.rglob(pattern):
                    if file_path.is_file():
                        try:
                            rel_path = str(file_path.relative_to(pipeline_output_dir))
                            file_size = file_path.stat().st_size
                            file_size_mb = file_size / (1024 * 1024)
                            
                            step_files.append({
                                "name": file_path.name,
                                "path": rel_path,
                                "size_mb": round(file_size_mb, 3),
                                "type": file_path.suffix.lower()
                            })
                            total_files += 1
                        except Exception:
                            pass
            
            if step_files:
                # Sort files by type and name
                step_files.sort(key=lambda x: (x["type"], x["name"]))
                output_sections.append({
                    "step_name": step_name,
                    "step_dir": step_dir,
                    "file_count": len(step_files),
                    "files": step_files[:20]  # Limit to first 20 files per section
                })
        
        # Generate HTML
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GNN Pipeline Output Navigation</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #007bff;
        }}
        h1 {{
            color: #2c3e50;
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-left: 4px solid #007bff;
            padding-left: 15px;
        }}
        .summary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            margin: 30px 0;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .summary-card {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }}
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
            margin: 10px 0;
        }}
        .step-section {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }}
        .step-header {{
            font-weight: bold;
            color: #495057;
            margin-bottom: 15px;
            font-size: 1.2em;
        }}
        .file-list {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }}
        .file-item {{
            background-color: white;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 10px;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .file-item:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .file-item a {{
            color: #007bff;
            text-decoration: none;
            font-weight: 500;
        }}
        .file-item a:hover {{
            text-decoration: underline;
        }}
        .file-meta {{
            color: #6c757d;
            font-size: 12px;
            margin-top: 5px;
        }}
        .link {{
            color: #007bff;
            text-decoration: none;
        }}
        .link:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 GNN Pipeline Output Navigation</h1>
            <p>Comprehensive navigation to all pipeline outputs and artifacts</p>
        </div>
        
        <div class="summary">
            <h2>Pipeline Overview</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <div>Pipeline Steps</div>
                    <div class="value">{len(output_sections)}</div>
                </div>
                <div class="summary-card">
                    <div>Total Files</div>
                    <div class="value">{total_files}</div>
                </div>
                <div class="summary-card">
                    <div>Output Directory</div>
                    <div class="value" style="font-size: 0.8em;">{pipeline_output_dir.name}</div>
                </div>
            </div>
        </div>
        
        <h2>📁 Output Sections</h2>
"""
        
        # Add each step section
        for section in output_sections:
            html_content += f"""
        <div class="step-section">
            <div class="step-header">📂 {section['step_name']} ({section['step_dir']})</div>
            <p><strong>Files:</strong> {section['file_count']}</p>
            <div class="file-list">
"""
            for file_info in section['files']:
                html_content += f"""
                <div class="file-item">
                    <a href="../{file_info['path']}" target="_blank">{file_info['name']}</a>
                    <div class="file-meta">
                        {file_info['type']} • {file_info['size_mb']} MB
                    </div>
                </div>
"""
            if section['file_count'] > len(section['files']):
                html_content += f"""
                <div class="file-item" style="opacity: 0.7; font-style: italic;">
                    ... and {section['file_count'] - len(section['files'])} more files
                </div>
"""
            html_content += """
            </div>
        </div>
"""
        
        html_content += """
        <div style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #dee2e6; color: #6c757d;">
            <p>Generated by GNN Pipeline GUI Module</p>
            <p><a href="../23_report_output/comprehensive_analysis_report.html" class="link">View Comprehensive Report</a></p>
        </div>
    </div>
</body>
</html>
"""
        
        # Write HTML file
        nav_file = output_dir / "navigation.html"
        with open(nav_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✅ HTML navigation page generated: {nav_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate HTML navigation: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

__all__ = [
    "process_gui",
    "gui_1", 
    "gui_2",
    "gui_3",
    "oxdraw_gui",
    "get_available_guis",
    "get_gui_1_info",
    "get_gui_2_info",
    "get_gui_3_info",
    "get_oxdraw_info",
    "generate_html_navigation",
    # GUI 1 utilities
    "add_component_to_markdown",
    "update_component_states", 
    "remove_component_from_markdown",
    "parse_components_from_markdown",
    "parse_state_space_from_markdown",
    "add_state_space_entry",
    "update_state_space_entry",
    "remove_state_space_entry",
]


