from pathlib import Path
import logging
import sys

logger = logging.getLogger(__name__)

def get_venv_python(script_dir: Path) -> tuple[Path | None, Path | None]:
    """
    Find the virtual environment Python executable and site-packages path.
    
    Args:
        script_dir: The directory where the script is located (typically src/)
        
    Returns:
        Tuple of (venv_python_path, site_packages_path)
    """
    venv_python_path = None
    site_packages_path = None
    
    # Try multiple common virtual environment locations
    # Prioritize project root .venv for consistency
    venv_candidates = [
        script_dir.parent / ".venv",  # .venv in project root (preferred)
        script_dir / ".venv",  # Standard .venv in script directory (fallback)
        script_dir.parent / "venv",  # venv in parent directory (legacy)
        script_dir.parent.parent / ".venv",  # .venv in grandparent (project root from src/)
    ]
    
    for venv_path in venv_candidates:
        logger.debug(f"🔍 Checking for virtual environment at: {venv_path}")
        
        if venv_path.is_dir():
            logger.debug(f"✓ Found virtual environment directory: {venv_path}")
            
            potential_python_executables = [
                venv_path / "bin" / "python",
                venv_path / "bin" / "python3",
                venv_path / "Scripts" / "python.exe", # Windows
            ]
            
            for py_exec in potential_python_executables:
                if py_exec.exists() and py_exec.is_file():
                    venv_python_path = py_exec
                    logger.debug(f"🐍 Found virtual environment Python: {venv_python_path}")
                    break
            
            # Find site-packages path
            lib_path = venv_path / "lib"
            if lib_path.is_dir():
                for python_version_dir in lib_path.iterdir():
                    if python_version_dir.is_dir() and python_version_dir.name.startswith("python"):
                        current_site_packages = python_version_dir / "site-packages"
                        if current_site_packages.is_dir():
                            site_packages_path = current_site_packages
                            logger.debug(f"📦 Found site-packages at: {site_packages_path}")
                            break
            
            # If we found a Python executable, break out of the loop
            if venv_python_path:
                logger.info(f"✅ Using virtual environment: {venv_path}")
                break
    
    if not venv_python_path:
        logger.warning("⚠️ Virtual environment Python not found. Using system Python. This may lead to issues if dependencies are not globally available.")
        venv_python_path = Path(sys.executable) # Fallback to current interpreter
    
    return venv_python_path, site_packages_path 