"""
oxdraw Processor - Main orchestration logic for GNN-oxdraw integration

Handles:
- GNN file discovery and conversion to Mermaid
- oxdraw editor launching (interactive mode)
- Mermaid to GNN conversion (headless mode)
- Validation and error handling
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import subprocess
import json
import shutil
import logging

from gnn.processor import discover_gnn_files, parse_gnn_file
from .mermaid_converter import convert_gnn_file_to_mermaid
from .mermaid_parser import convert_mermaid_file_to_gnn


def process_oxdraw(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    mode: str = "headless",
    auto_convert: bool = True,
    validate_on_save: bool = True,
    launch_editor: bool = False,
    port: int = 5151,
    host: str = "127.0.0.1",
    **kwargs
) -> bool:
    """
    Process GNN files through oxdraw visual interface.
    
    Args:
        target_dir: Directory containing GNN files
        output_dir: Output directory for oxdraw results
        logger: Logger instance for progress reporting
        mode: "interactive" or "headless" (default: headless for pipeline)
        auto_convert: Automatically convert GNN files to Mermaid
        validate_on_save: Validate models when converting back from Mermaid
        launch_editor: Launch oxdraw editor (only if mode=interactive)
        port: Port for oxdraw server
        host: Host for oxdraw server
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded
    """
    logger.info("🎨 oxdraw Integration - Visual GNN Model Interface")
    logger.info(f"Mode: {mode}")
    logger.info(f"Target directory: {target_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check oxdraw availability for interactive mode
    if mode == "interactive" and launch_editor:
        if not check_oxdraw_installed():
            logger.warning("⚠️  oxdraw CLI not found. Install with: cargo install oxdraw")
            logger.info("Falling back to headless conversion mode...")
            mode = "headless"
    
    # Discover GNN files
    gnn_files = discover_gnn_files(target_dir, recursive=True)
    logger.info(f"📁 Found {len(gnn_files)} GNN file(s)")
    
    if not gnn_files:
        logger.warning("No GNN files found to process")
        return False
    
    results = {
        "mode": mode,
        "timestamp": _get_timestamp(),
        "files_processed": [],
        "gnn_to_mermaid_conversions": [],
        "mermaid_to_gnn_conversions": [],
        "errors": []
    }
    
    # Phase 1: Convert GNN files to Mermaid
    if auto_convert:
        logger.info("📝 Phase 1: Converting GNN files to Mermaid format...")
        
        for i, gnn_file in enumerate(gnn_files, 1):
            logger.info(f"  [{i}/{len(gnn_files)}] Processing: {gnn_file.name}")
            
            try:
                mermaid_file = output_dir / f"{gnn_file.stem}.mmd"
                mermaid_content = convert_gnn_file_to_mermaid(gnn_file, mermaid_file)
                
                results["gnn_to_mermaid_conversions"].append({
                    "gnn_file": str(gnn_file),
                    "mermaid_file": str(mermaid_file),
                    "success": True,
                    "lines": len(mermaid_content.split('\n'))
                })
                
                logger.info(f"  ✅ Converted: {gnn_file.name} → {mermaid_file.name}")
                
            except Exception as e:
                logger.error(f"  ❌ Conversion failed for {gnn_file.name}: {e}")
                results["errors"].append({
                    "file": str(gnn_file),
                    "phase": "gnn_to_mermaid",
                    "error": str(e)
                })
                results["gnn_to_mermaid_conversions"].append({
                    "gnn_file": str(gnn_file),
                    "success": False,
                    "error": str(e)
                })
    
    # Phase 2: Launch interactive editor or process in headless mode
    if mode == "interactive" and launch_editor and results["gnn_to_mermaid_conversions"]:
        logger.info("🚀 Phase 2: Launching oxdraw interactive editor...")
        
        # Launch editor for first successfully converted file
        first_success = next(
            (c for c in results["gnn_to_mermaid_conversions"] if c["success"]),
            None
        )
        
        if first_success:
            first_mermaid = Path(first_success["mermaid_file"])
            
            try:
                success = launch_oxdraw_editor(
                    mermaid_file=first_mermaid,
                    port=port,
                    host=host,
                    logger=logger
                )
                
                if success:
                    logger.info(f"✅ oxdraw editor launched at http://{host}:{port}")
                    logger.info("Edit your model visually, then save and close the editor")
                    
                    results["editor_launched"] = True
                    results["editor_url"] = f"http://{host}:{port}"
                else:
                    logger.warning("⚠️  Failed to launch oxdraw editor")
                    results["editor_launched"] = False
                    
            except Exception as e:
                logger.error(f"❌ Editor launch error: {e}")
                results["errors"].append({
                    "phase": "editor_launch",
                    "error": str(e)
                })
                results["editor_launched"] = False
    
    # Phase 3: Convert Mermaid files back to GNN (if any .mmd files exist)
    mermaid_files = list(output_dir.glob("*.mmd"))
    
    if mermaid_files and validate_on_save:
        logger.info("🔄 Phase 3: Converting Mermaid files back to GNN...")
        
        for i, mermaid_file in enumerate(mermaid_files, 1):
            logger.info(f"  [{i}/{len(mermaid_files)}] Processing: {mermaid_file.name}")
            
            try:
                output_gnn_file = output_dir / f"{mermaid_file.stem}_from_mermaid.md"
                parsed_model = convert_mermaid_file_to_gnn(
                    mermaid_file,
                    output_gnn_file
                )
                
                results["mermaid_to_gnn_conversions"].append({
                    "mermaid_file": str(mermaid_file),
                    "gnn_file": str(output_gnn_file),
                    "success": True,
                    "variables": len(parsed_model.get('variables', {})),
                    "connections": len(parsed_model.get('connections', []))
                })
                
                logger.info(f"  ✅ Converted: {mermaid_file.name} → {output_gnn_file.name}")
                
            except Exception as e:
                logger.error(f"  ❌ Conversion failed for {mermaid_file.name}: {e}")
                results["errors"].append({
                    "file": str(mermaid_file),
                    "phase": "mermaid_to_gnn",
                    "error": str(e)
                })
                results["mermaid_to_gnn_conversions"].append({
                    "mermaid_file": str(mermaid_file),
                    "success": False,
                    "error": str(e)
                })
    
    # Save processing results
    results_file = output_dir / "oxdraw_processing_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"📊 Processing results saved to: {results_file}")
    
    # Summary
    success_count = sum(1 for c in results["gnn_to_mermaid_conversions"] if c["success"])
    total_count = len(results["gnn_to_mermaid_conversions"])
    
    logger.info(f"✨ Summary:")
    logger.info(f"   GNN → Mermaid: {success_count}/{total_count} successful")
    
    if results["mermaid_to_gnn_conversions"]:
        back_success = sum(1 for c in results["mermaid_to_gnn_conversions"] if c["success"])
        back_total = len(results["mermaid_to_gnn_conversions"])
        logger.info(f"   Mermaid → GNN: {back_success}/{back_total} successful")
    
    if results["errors"]:
        logger.warning(f"   Errors: {len(results['errors'])}")
    
    return success_count > 0


def check_oxdraw_installed() -> bool:
    """
    Check if oxdraw CLI is installed and available.
    
    Returns:
        True if oxdraw is available, False otherwise
    """
    return shutil.which("oxdraw") is not None


def launch_oxdraw_editor(
    mermaid_file: Path,
    port: int = 5151,
    host: str = "127.0.0.1",
    logger: Optional[logging.Logger] = None,
    background: bool = False
) -> bool:
    """
    Launch oxdraw interactive editor for a Mermaid file.
    
    Args:
        mermaid_file: Path to Mermaid file to edit
        port: Port for oxdraw server
        host: Host address for oxdraw server
        logger: Optional logger instance
        background: Run in background (non-blocking)
        
    Returns:
        True if launch succeeded, False otherwise
    """
    if not check_oxdraw_installed():
        if logger:
            logger.error("oxdraw CLI not found. Install with: cargo install oxdraw")
        return False
    
    if not mermaid_file.exists():
        if logger:
            logger.error(f"Mermaid file not found: {mermaid_file}")
        return False
    
    try:
        cmd = [
            "oxdraw",
            "--input", str(mermaid_file),
            "--edit",
            "--serve-host", host,
            "--serve-port", str(port)
        ]
        
        if logger:
            logger.info(f"Launching oxdraw: {' '.join(cmd)}")
        
        if background:
            # Non-blocking launch
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            # Blocking launch
            subprocess.run(cmd, check=True)
        
        return True
        
    except subprocess.CalledProcessError as e:
        if logger:
            logger.error(f"oxdraw launch failed: {e}")
        return False
    except Exception as e:
        if logger:
            logger.error(f"Unexpected error launching oxdraw: {e}")
        return False


def get_module_info() -> Dict[str, Any]:
    """
    Get module information and capabilities.
    
    Returns:
        Dictionary with module metadata
    """
    return {
        "name": "oxdraw",
        "version": "1.0.0",
        "description": "Visual diagram-as-code interface for GNN Active Inference models",
        "capabilities": [
            "GNN to Mermaid conversion",
            "Mermaid to GNN parsing",
            "Interactive visual editing",
            "Headless batch conversion",
            "Ontology preservation",
            "Connection validation"
        ],
        "modes": ["interactive", "headless"],
        "supported_formats": ["mermaid", "gnn_markdown"],
        "oxdraw_cli_available": check_oxdraw_installed()
    }


def _get_timestamp() -> str:
    """Get ISO 8601 timestamp."""
    from datetime import datetime
    return datetime.now().isoformat()

