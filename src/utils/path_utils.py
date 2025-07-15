from pathlib import Path
from typing import Optional

def get_relative_path_if_possible(absolute_path_obj: Path, project_root: Optional[Path] = None) -> str:
    """Returns a path string relative to project_root if provided and applicable, otherwise absolute."""
    if project_root:
        try:
            return str(absolute_path_obj.relative_to(project_root))
        except ValueError:
            return str(absolute_path_obj)
    return str(absolute_path_obj) 