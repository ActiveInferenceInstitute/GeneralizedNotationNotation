#!/usr/bin/env python3
"""
Step 11: Render Processing (Thin Orchestrator)

This step generates simulation code for PyMDP, RxInfer, and ActiveInference.jl from GNN models.
This is a thin orchestrator that delegates rendering to modular framework implementations.

How to run:
  python src/11_render.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - Rendered simulation code in the specified output directory
  - PyMDP, RxInfer.jl, ActiveInference.jl, and DisCoPy code files
  - Comprehensive rendering reports and summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that render dependencies are installed
  - Check that src/render/ contains render modules
  - Check that the output directory is writable
  - Verify GNN specifications are valid and complete
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import (
    setup_step_logging,
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)
from utils.argument_utils import EnhancedArgumentParser
from pipeline.config import get_output_dir_for_script, get_pipeline_config

# Import core rendering functions from render module
try:
    from render import (
        generate_pymdp_code,
        generate_rxinfer_code,
        generate_rxinfer_fallback_code,
        generate_activeinference_jl_code,
        generate_activeinference_jl_fallback_code,
        generate_discopy_code,
        generate_discopy_fallback_code
    )
    RENDER_AVAILABLE = True
except ImportError:
    RENDER_AVAILABLE = False
    # Fallback function definitions if render module is not available
    def generate_pymdp_code(model_data: Dict) -> str:
        return f"""#!/usr/bin/env python3
# PyMDP code generation failed: Render module not available
print("Error: PyMDP code generation failed - render module not available")
"""
    
    def generate_rxinfer_code(model_data: Dict) -> str:
        return f"""#!/usr/bin/env python3
# RxInfer code generation failed: Render module not available
print("Error: RxInfer code generation failed - render module not available")
"""
    
    def generate_rxinfer_fallback_code(model_data: Dict) -> str:
        return f"""#!/usr/bin/env python3
# RxInfer fallback code generation failed: Render module not available
print("Error: RxInfer fallback code generation failed - render module not available")
"""
    
    def generate_activeinference_jl_code(model_data: Dict) -> str:
        return f"""#!/usr/bin/env python3
# ActiveInference.jl code generation failed: Render module not available
print("Error: ActiveInference.jl code generation failed - render module not available")
"""
    
    def generate_activeinference_jl_fallback_code(model_data: Dict) -> str:
        return f"""#!/usr/bin/env python3
# ActiveInference.jl fallback code generation failed: Render module not available
print("Error: ActiveInference.jl fallback code generation failed - render module not available")
"""
    
    def generate_discopy_code(model_data: Dict) -> str:
        return f"""#!/usr/bin/env python3
# DisCoPy code generation failed: Render module not available
print("Error: DisCoPy code generation failed - render module not available")
"""
    
    def generate_discopy_fallback_code(model_data: Dict) -> str:
        return f"""#!/usr/bin/env python3
# DisCoPy fallback code generation failed: Render module not available
print("Error: DisCoPy fallback code generation failed - render module not available")
"""

def process_render_standardized(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Standardized render processing function.
    
    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Output directory for render results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        # Get pipeline configuration
        config = get_pipeline_config()
        step_output_dir = get_output_dir_for_script("11_render.py", output_dir)
        step_output_dir.mkdir(parents=True, exist_ok=True)
        
        log_step_start(logger, "Processing render")
        
        # Load parsed GNN data from previous step
        gnn_output_dir = get_output_dir_for_script("3_gnn.py", output_dir)
        gnn_results_file = gnn_output_dir / "gnn_processing_results.json"
        
        if not gnn_results_file.exists():
            log_step_error(logger, "GNN processing results not found. Run step 3 first.")
            return False
        
        with open(gnn_results_file, 'r') as f:
            gnn_results = json.load(f)
        
        logger.info(f"Loaded {len(gnn_results['processed_files'])} parsed GNN files")
        
        # Render results
        render_results = {
            "timestamp": datetime.now().isoformat(),
            "source_directory": str(target_dir),
            "output_directory": str(step_output_dir),
            "files_rendered": [],
            "summary": {
                "total_files": 0,
                "successful_renders": 0,
                "failed_renders": 0,
                "code_files_generated": {
                    "pymdp": 0,
                    "rxinfer": 0,
                    "activeinference_jl": 0,
                    "discopy": 0
                }
            }
        }
        
        # Render targets - delegate to modular framework renderers
        render_targets = [
            ("pymdp", generate_pymdp_code, ".py"),
            ("rxinfer", generate_rxinfer_code, ".jl"),
            ("activeinference_jl", generate_activeinference_jl_code, ".jl"),
            ("discopy", generate_discopy_code, ".py")
        ]
        
        for file_result in gnn_results["processed_files"]:
            if not file_result["parse_success"]:
                continue
            
            file_name = file_result["file_name"]
            logger.info(f"Rendering code for: {file_name}")
            
            # Load the actual parsed GNN specification
            parsed_model_file = file_result.get("parsed_model_file")
            if parsed_model_file and Path(parsed_model_file).exists():
                try:
                    with open(parsed_model_file, 'r') as f:
                        actual_gnn_spec = json.load(f)
                    logger.info(f"Loaded parsed GNN specification from {parsed_model_file}")
                    # Use the actual GNN specification instead of the summary
                    model_data = actual_gnn_spec
                except Exception as e:
                    logger.error(f"Failed to load parsed GNN spec from {parsed_model_file}: {e}")
                    # Fall back to using the summary data
                    model_data = file_result
            else:
                logger.warning(f"Parsed model file not found for {file_name}, using summary data")
                model_data = file_result
            
            # Create file-specific output directory
            file_output_dir = step_output_dir / file_name.replace('.md', '')
            file_output_dir.mkdir(exist_ok=True)
            
            file_render_result = {
                "file_name": file_name,
                "file_path": file_result["file_path"],
                "renders": {},
                "success": True
            }
            
            # Generate code for each target using modular renderers
            for target_name, code_generator, extension in render_targets:
                try:
                    # Generate code using the modular renderer
                    code = code_generator(model_data)
                    
                    # Save code file
                    code_file = file_output_dir / f"{file_name.replace('.md', '')}_{target_name}{extension}"
                    with open(code_file, 'w') as f:
                        f.write(code)
                    
                    file_render_result["renders"][target_name] = {
                        "success": True,
                        "code_file": str(code_file),
                        "code_length": len(code)
                    }
                    
                    render_results["summary"]["code_files_generated"][target_name] += 1
                    logger.info(f"Generated {target_name} code for {file_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to generate {target_name} code for {file_name}: {e}")
                    file_render_result["renders"][target_name] = {
                        "success": False,
                        "error": str(e)
                    }
                    file_render_result["success"] = False
            
            render_results["files_rendered"].append(file_render_result)
            render_results["summary"]["total_files"] += 1
            
            if file_render_result["success"]:
                render_results["summary"]["successful_renders"] += 1
            else:
                render_results["summary"]["failed_renders"] += 1
        
        # Save render results
        render_results_file = step_output_dir / "render_results.json"
        with open(render_results_file, 'w') as f:
            json.dump(render_results, f, indent=2)
        
        # Save render summary
        render_summary_file = step_output_dir / "render_summary.json"
        with open(render_summary_file, 'w') as f:
            json.dump(render_results["summary"], f, indent=2)
        
        logger.info(f"Render processing completed:")
        logger.info(f"  Total files: {render_results['summary']['total_files']}")
        logger.info(f"  Successful renders: {render_results['summary']['successful_renders']}")
        logger.info(f"  Failed renders: {render_results['summary']['failed_renders']}")
        logger.info(f"  Code files generated: {render_results['summary']['code_files_generated']}")
        
        log_step_success(logger, "Render processing completed successfully")
        return True
        
    except Exception as e:
        log_step_error(logger, f"Render processing failed: {e}")
        return False

def main():
    """Main render processing function."""
    args = EnhancedArgumentParser.parse_step_arguments("11_render.py")
    
    # Setup logging
    logger = setup_step_logging("render", args)
    
    # Check if render module is available
    if not RENDER_AVAILABLE:
        log_step_warning(logger, "Render module not available, using fallback functions")
    
    # Process render
    success = process_render_standardized(
        target_dir=args.target_dir,
        output_dir=args.output_dir,
        logger=logger,
        recursive=args.recursive,
        verbose=args.verbose
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
