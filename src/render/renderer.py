from pathlib import Path
import logging
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
                    param_lines = [p for p in model.parameters if hasattr(p, 'name') and hasattr(p, 'value')]
                    matrix_keys = ["A", "B", "C", "D", "E"]
                    i = 0
                    while i < len(param_lines):
                        param = param_lines[i]
                        name = param.name.strip()
                        value = param.value
                        if name in matrix_keys and isinstance(value, str) and (value.strip().startswith("{") or value.strip().startswith("(")):
                            block = value.strip()
                            open_brace = block[0]
                            close_brace = '}' if open_brace == '{' else ')'
                            block_lines = []
                            # Always include the first line if it starts with '('
                            if block.startswith('('):
                                block_lines.append(block)
                            i += 1
                            while i < len(param_lines):
                                next_value = param_lines[i].value
                                if isinstance(next_value, str):
                                    next_value_str = next_value.strip()
                                    if next_value_str.startswith('('):
                                        block_lines.append(next_value_str)
                                    if next_value_str.endswith(close_brace):
                                        break
                                i += 1
                            # Wrap with braces/parens
                            full_block = open_brace + '\n' + '\n'.join(block_lines) + '\n' + close_brace
                            print(f"DEBUG: InitialParameterization {name} value =\n{full_block}")
                            initial_params[name] = full_block
                            i += 1
                        elif name in matrix_keys and name not in initial_params:
                            initial_params[name] = value
                            i += 1
                        else:
                            i += 1
                    
                    # Also extract from the raw InitialParameterization section if available
                    if hasattr(model, 'extensions') and 'initial_parameterization' in model.extensions:
                        raw_init_params = model.extensions['initial_parameterization']
                        if isinstance(raw_init_params, str):
                            # Parse the raw InitialParameterization section as multiline strings
                            lines = raw_init_params.split('\n')
                            current_matrix = None
                            current_content = []
                            
                            for line in lines:
                                line = line.strip()
                                if not line or line.startswith('#'):
                                    continue
                                
                                # Check if this line starts a new matrix definition
                                for matrix_key in matrix_keys:
                                    if line.startswith(f"{matrix_key}="):
                                        # Save previous matrix if exists
                                        if current_matrix and current_content:
                                            initial_params[current_matrix] = '\n'.join(current_content)
                                        
                                        # Start new matrix
                                        current_matrix = matrix_key
                                        current_content = [line]
                                        break
                                else:
                                    # Continue current matrix
                                    if current_matrix:
                                        current_content.append(line)
                            
                            # Save last matrix
                            if current_matrix and current_content:
                                initial_params[current_matrix] = '\n'.join(current_content)
                    
                    # If matrices are still empty, try to parse them from the raw content
                    if not any(initial_params.get(key) for key in matrix_keys):
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
                            
                            def extract_matrix_block(section, key):
                                # Find the start of the matrix definition
                                import re
                                pattern = key + r'\s*=\s*\{'
                                match = re.search(pattern, section)
                                if not match:
                                    return None
                                start = match.end()  # position after the opening brace
                                # Now extract until the matching closing brace
                                brace_count = 1
                                i = start
                                block = ['{']
                                while i < len(section):
                                    c = section[i]
                                    if c == '{':
                                        brace_count += 1
                                    elif c == '}':
                                        brace_count -= 1
                                    block.append(c)
                                    if brace_count == 0:
                                        break
                                    i += 1
                                return key + '=' + ''.join(block)
                            
                            for matrix_key in matrix_keys:
                                matrix_block = extract_matrix_block(init_section, matrix_key)
                                if matrix_block:
                                    initial_params[matrix_key] = matrix_block
                    
                    gnn_spec["InitialParameterization"] = initial_params
                    
                    for target_format, output_subdir in render_targets:
                        try:
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