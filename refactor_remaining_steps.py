#!/usr/bin/env python3
import os

# Map step numbers to their module and function names
step_configs = {
    11: ("render", "process_render"),
    12: ("execute", "process_execute"),  # execute module already has process_execute
    13: ("llm", "process_llm"),
    14: ("ml_integration", "process_ml_integration"),
    15: ("audio", "process_audio"),
    16: ("analysis", "process_analysis"),
    17: ("integration", "process_integration"),
    18: ("security", "process_security"),
    19: ("research", "process_research"),
    20: ("website", "process_website"),
    21: ("mcp", "process_mcp"),
    22: ("gui", "process_gui"),
    23: ("report", "process_report")
}

def refactor_step(step_num, module_name, function_name):
    """Refactor a step to use thin orchestrator pattern."""
    step_file = f"src/{step_num}_*.py"
    
    # Find the actual step file
    import glob
    files = glob.glob(step_file)
    if not files:
        print(f"No file found for step {step_num}")
        return False
    
    step_file = files[0]
    step_name = os.path.basename(step_file).replace('.py', '')
    
    # Create the thin orchestrator content
    content = f'''#!/usr/bin/env python3
"""
Step {step_num}: {step_name.replace('_', ' ').replace(f'{step_num}', '').strip().title()} Processing (Thin Orchestrator)

This step orchestrates {module_name} processing for GNN models.

Architectural Role:
    This is a "thin orchestrator" - a minimal script that delegates core functionality
    to the corresponding module (src/{module_name}/). It handles argument parsing, logging
    setup, and calls the actual processing functions from the {module_name} module.

Pipeline Flow:
    main.py → {step_num}_{step_name}.py (this script) → {module_name}/ (modular implementation)

How to run:
  python src/{step_num}_{step_name}.py --target-dir input/gnn_files --output-dir output --verbose
  python src/main.py  # (runs as part of the pipeline)

Expected outputs:
  - {module_name.title()} processing results in the specified output directory
  - Comprehensive {module_name} reports and summaries
  - Actionable error messages if dependencies or paths are missing
  - Clear logging of all resolved arguments and paths

If you encounter errors:
  - Check that {module_name} dependencies are installed
  - Check that src/{module_name}/ contains {module_name} modules
  - Check that the output directory is writable
  - Verify {module_name} configuration and requirements
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import create_standardized_pipeline_script
from {module_name} import {function_name}

# Create the standardized pipeline script
run_script = create_standardized_pipeline_script(
    "{step_num}_{step_name}.py",
    {function_name},
    "{module_name.title()} processing"
)

def main() -> int:
    """Main entry point for the {step_name} step."""
    return run_script()

if __name__ == "__main__":
    sys.exit(main())
'''
    
    with open(step_file, 'w') as f:
        f.write(content)
    
    print(f"Refactored {step_file} to use thin orchestrator pattern")
    return True

# Refactor all remaining steps
for step_num, (module_name, function_name) in step_configs.items():
    refactor_step(step_num, module_name, function_name)

print("All remaining steps have been refactored to use the thin orchestrator pattern!")
