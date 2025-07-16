import glob
import re
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)

def get_pipeline_scripts(current_dir: Path) -> list[dict[str, int | str | Path]]:
    potential_scripts_pattern = current_dir / "*_*.py"
    logger.debug(f"ℹ️ Discovering potential pipeline scripts using pattern: {potential_scripts_pattern}")
    all_candidate_files = glob.glob(str(potential_scripts_pattern))
    
    pipeline_scripts_info: list[dict[str, int | str | Path]] = [] # Explicitly type hint
    script_name_regex = re.compile(r"^(\d+)_.*\.py$")

    for script_path_str in all_candidate_files:
        script_basename = os.path.basename(script_path_str)
        match = script_name_regex.match(script_basename)
        if match:
            script_num = int(match.group(1))
            pipeline_scripts_info.append({'num': script_num, 'basename': script_basename, 'path': Path(script_path_str)})
            if logger.isEnabledFor(logging.DEBUG):
                 logger.debug(f"ℹ️ Matched script for pipeline: {script_basename} (Number: {script_num})")
    
    pipeline_scripts_info.sort(key=lambda x: (x['num'], x['basename']))
    sorted_script_basenames: list[str] = [str(info['basename']) for info in pipeline_scripts_info] # Explicitly cast to str and type hint

    if logger.isEnabledFor(logging.DEBUG): 
        logger.debug(f"ℹ️ Found and sorted script basenames: {sorted_script_basenames}")
    return pipeline_scripts_info 