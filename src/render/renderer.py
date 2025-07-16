from pathlib import Path
import logging
import re
from pipeline import get_output_dir_for_script
from utils import log_step_start, log_step_success, log_step_warning, log_step_error, performance_tracker
from gnn.parsers.markdown_parser import MarkdownGNNParser

# Import rendering functionality
try:
    from .render import render_gnn_spec
    RENDER_AVAILABLE = True
except ImportError as e:
    render_gnn_spec = None
    RENDER_AVAILABLE = False

def _parse_initial_parameterization_robust(section_content: str) -> dict:
    """
    Robustly parse the InitialParameterization section, properly handling multi-line matrix blocks.
    Handles cases where the opening brace is on a line by itself, and accumulates lines until braces are balanced.
    """
    data = {}
    lines = section_content.split('\n')
    current_key = None
    current_block = []
    in_matrix = False
    open_brace = None
    close_brace = None
    brace_count = 0

    for line in lines:
        line_strip = line.strip()
        if not line_strip or line_strip.startswith('#'):
            continue

        # Detect start of a matrix block (A=, B=, C=, D=, E=)
        matrix_start = re.match(r'^(A|B|C|D|E)\s*=\s*([{(])', line_strip)
        if not in_matrix and matrix_start:
            # Save previous block if exists
            if current_key and current_block:
                data[current_key] = '\n'.join(current_block)
            # Start new matrix block
            current_key = matrix_start.group(1)
            open_brace = matrix_start.group(2)
            close_brace = '}' if open_brace == '{' else ')'
            # Find the position of the opening brace
            brace_pos = line_strip.find(open_brace)
            # Start collecting from the opening brace
            block_line = line_strip[brace_pos:]
            current_block = [block_line]
            # Initialize brace count for this block
            brace_count = block_line.count(open_brace) - block_line.count(close_brace)
            in_matrix = True
            # If the block is closed on the same line, finish immediately
            if brace_count == 0:
                data[current_key] = '\n'.join(current_block)
                in_matrix = False
                current_key = None
                current_block = []
                open_brace = None
                close_brace = None
            continue

        # Handle lines inside a matrix block
        if in_matrix:
            if not matrix_start:  # Don't double-add the start line
                current_block.append(line_strip)
                brace_count += line_strip.count(open_brace)
                brace_count -= line_strip.count(close_brace)
            # If braces are balanced, end the matrix block
            if brace_count == 0:
                data[current_key] = '\n'.join(current_block)
                in_matrix = False
                current_key = None
                current_block = []
                open_brace = None
                close_brace = None
            continue

        # Handle single-line assignments (e.g., C = (0.0, 0.0, 1.0))
        single_assign = re.match(r'^(C|D|E)\s*=\s*(.+)$', line_strip)
        if single_assign:
            data[single_assign.group(1)] = single_assign.group(2).strip()

    # Save any trailing block
    if in_matrix and current_key and current_block:
        data[current_key] = '\n'.join(current_block)

    return data

def render_gnn_files(
    target_dir: Path, 
    output_dir: Path, 
    logger: logging.Logger, 
    recursive: bool = False, 
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Render GNN files to simulation environments.
    
    Args:
        target_dir: Directory containing GNN files to render
        output_dir: Output directory for results
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        **kwargs: Additional rendering options
        
    Returns:
        True if rendering succeeded, False otherwise
    """
    log_step_start(logger, "Rendering GNN files to simulation environments")
    
    # Use centralized output directory configuration
    render_output_dir = get_output_dir_for_script("9_render.py", output_dir)
    render_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find GNN files
    pattern = "**/*.md" if recursive else "*.md"
    gnn_files = list(target_dir.glob(pattern))
    
    if not gnn_files:
        log_step_warning(logger, f"No GNN files found in {target_dir} using pattern '{pattern}'")
        return False

    logger.info(f"Found {len(gnn_files)} GNN files to render")
    
    successful_renders = 0
    failed_renders = 0
    
    # Define rendering targets with appropriate options
    render_targets = [
        ("pymdp", "pymdp"),
        ("rxinfer_toml", "rxinfer"),
        ("discopy_combined", "discopy"),
        ("activeinference_combined", "activeinference_jl"),
        ("jax_pomdp", "jax"),
        ("jax", "jax")
    ]
    
    parser = MarkdownGNNParser()
    
    try:
        with performance_tracker.track_operation("render_all_gnn_files"):
            for gnn_file in gnn_files:
                try:
                    # Parse the GNN file
                    parse_result = parser.parse_file(str(gnn_file))
                    if not parse_result.success:
                        log_step_warning(logger, f"Parsing failed for {gnn_file.name}: {parse_result.errors}")
                        failed_renders += len(render_targets)
                        continue
                    
                    model = parse_result.model
                    
                    # Convert model to dictionary for renderers
                    gnn_spec = {
                        "name": model.model_name,
                        "annotation": model.annotation,
                        "variables": [vars(v) for v in model.variables],
                        "connections": [vars(c) for c in model.connections],
                        "parameters": [vars(p) for p in model.parameters],
                        "equations": model.equations,
                        "time": vars(model.time_specification),
                        "ontology": [vars(m) for m in model.ontology_mappings],
                        "model_parameters": model.extensions.get('model_parameters', {}),
                        "source_file": str(gnn_file)
                    }
                    
                    # Extract InitialParameterization as a dictionary for matrix access
                    initial_params = {}
                    
                    # Method 1: Extract from parsed parameters (most reliable)
                    param_lines = [p for p in model.parameters if hasattr(p, 'name') and hasattr(p, 'value')]
                    matrix_keys = ["A", "B", "C", "D", "E"]
                    
                    for param in param_lines:
                        name = param.name.strip()
                        value = param.value
                        if name in matrix_keys:
                            initial_params[name] = value
                    
                    # Method 2: Extract from raw InitialParameterization section if available
                    if hasattr(model, 'extensions') and 'initial_parameterization' in model.extensions:
                        raw_init_params = model.extensions['initial_parameterization']
                        if isinstance(raw_init_params, str):
                            # Use the robust parsing method from export module
                            parsed_raw_params = _parse_initial_parameterization_robust(raw_init_params)
                            for key, value in parsed_raw_params.items():
                                if key in matrix_keys:
                                    initial_params[key] = value
                    
                    # Method 3: Fallback to direct file parsing if matrices are still missing
                    missing_matrices = [key for key in matrix_keys if key not in initial_params or not initial_params[key]]
                    if missing_matrices:
                        # Read the original file content to extract matrices
                        with open(gnn_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Find the InitialParameterization section
                        if '## InitialParameterization' in content:
                            start_idx = content.find('## InitialParameterization')
                            end_idx = content.find('##', start_idx + 1)
                            if end_idx == -1:
                                end_idx = len(content)
                            
                            init_section = content[start_idx:end_idx]
                            
                            # Use the robust parsing method from export module
                            parsed_section_params = _parse_initial_parameterization_robust(init_section)
                            for key, value in parsed_section_params.items():
                                if key in matrix_keys and key not in initial_params:
                                    initial_params[key] = value
                    
                    # Debug logging
                    for key in matrix_keys:
                        if key in initial_params:
                            logger.debug(f"Extracted {key} matrix: {initial_params[key]}")
                        else:
                            logger.warning(f"Missing {key} matrix in InitialParameterization")
                    
                    gnn_spec["InitialParameterization"] = initial_params
                    
                    for target_format, output_subdir in render_targets:
                        try:
                            # Check if rendering is available
                            if not RENDER_AVAILABLE or render_gnn_spec is None:
                                failed_renders += 1
                                log_step_warning(logger, f"Render functionality not available for {target_format}")
                                continue
                            
                            sub_output_dir = render_output_dir / output_subdir
                            sub_output_dir.mkdir(exist_ok=True)
                            
                            # Use model name for filename
                            base_name = model.model_name.lower().replace(" ", "_")
                            if target_format.endswith("_jl") or target_format == "activeinference_combined":
                                output_file = sub_output_dir / f"{base_name}.jl"
                            elif target_format == "jax" or target_format == "jax_pomdp":
                                output_file = sub_output_dir / f"{base_name}.py"
                            else:
                                output_file = sub_output_dir / f"{base_name}.py"
                            
                            # Render the model
                            success, message, artifacts = render_gnn_spec(
                                gnn_spec=gnn_spec,
                                target=target_format,
                                output_directory=sub_output_dir,
                                options={"output_file": str(output_file)}
                            )
                            
                            if success:
                                successful_renders += 1
                                logger.debug(f"Successfully rendered {gnn_file.name} to {target_format}")
                            else:
                                failed_renders += 1
                                log_step_warning(logger, f"Failed to render {gnn_file.name} to {target_format}: {message}")
                                
                        except Exception as e:
                            failed_renders += 1
                            log_step_error(logger, f"Exception rendering {gnn_file.name} to {target_format}: {e}")
                            
                except Exception as e:
                    failed_renders += len(render_targets)
                    log_step_error(logger, f"Failed to process {gnn_file.name}: {e}")
        
        # Log results summary
        total_renders = successful_renders + failed_renders
        if total_renders > 0:
            success_rate = successful_renders / total_renders * 100
            log_step_success(logger, f"Rendering completed. Success rate: {success_rate:.1f}% ({successful_renders}/{total_renders})")
            return failed_renders == 0
        else:
            log_step_warning(logger, "No files were rendered")
            return False
            
    except Exception as e:
        log_step_error(logger, f"Rendering failed: {e}")
        return False 