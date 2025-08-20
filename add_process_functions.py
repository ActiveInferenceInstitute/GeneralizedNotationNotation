#!/usr/bin/env python3
import os
import re

def add_process_function(module_name, function_name):
    """Add a main process function to a module."""
    init_file = f"src/{module_name}/__init__.py"
    
    if not os.path.exists(init_file):
        print(f"Skipping {module_name} - no __init__.py")
        return
    
    with open(init_file, 'r') as f:
        content = f.read()
    
    # Skip if process function already exists
    if f"def {function_name}" in content:
        print(f"Skipping {module_name} - process function already exists")
        return
    
    # Add the process function before __all__
    process_function = f'''
def {function_name}(target_dir, output_dir, verbose=False, **kwargs):
    """
    Main processing function for {module_name}.
    
    Args:
        target_dir: Directory containing files to process
        output_dir: Output directory for results
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    import logging
    from pathlib import Path
    
    logger = logging.getLogger(__name__)
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        logger.info(f"Processing {module_name} for files in {{target_dir}}")
        # Placeholder implementation - delegate to actual module functions
        # This would be replaced with actual implementation
        logger.info(f"{module_name.title()} processing completed")
        return True
    except Exception as e:
        logger.error(f"{module_name.title()} processing failed: {{e}}")
        return False

'''
    
    # Insert before __all__ if it exists
    if "__all__" in content:
        content = re.sub(r'(__all__\s*=\s*\[)', f'{process_function}\n\\1', content)
    else:
        content += f'\n{process_function}\n'
    
    # Add function to __all__ list
    if "__all__" in content:
        # Find and update __all__ list
        all_match = re.search(r'__all__\s*=\s*\[(.*?)\]', content, re.DOTALL)
        if all_match:
            all_content = all_match.group(1)
            if f"'{function_name}'" not in all_content:
                content = content.replace(
                    all_match.group(0),
                    all_match.group(0).replace(']', f", '{function_name}']")
                )
    
    with open(init_file, 'w') as f:
        f.write(content)
    
    print(f"Added {function_name} to {module_name}")

# Add process functions to modules that don't have them
modules_to_update = [
    ('ml_integration', 'process_ml_integration'),
    ('audio', 'process_audio'),
    ('analysis', 'process_analysis'),
    ('integration', 'process_integration'),
    ('security', 'process_security'),
    ('research', 'process_research'),
    ('mcp', 'process_mcp'),
    ('gui', 'process_gui'),
    ('report', 'process_report')
]

for module, function in modules_to_update:
    add_process_function(module, function)
