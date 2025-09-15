#!/usr/bin/env python3
"""
UV Setup Script for GNN Pipeline

This script sets up the complete UV environment for the GNN pipeline,
including installation, environment creation, and dependency management.

Usage:
    python setup_uv.py [options]

Examples:
    # Basic setup
    python setup_uv.py
    
    # Setup with all optional dependencies
    python setup_uv.py --all-extras --dev
    
    # Recreate environment
    python setup_uv.py --recreate --verbose
"""

import sys
import argparse
import logging
import subprocess
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.uv_utils import (
    check_uv_available,
    ensure_uv_environment,
    get_uv_environment_info
)
from utils.uv_validator import comprehensive_uv_validation

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

def install_uv():
    """Install UV if not available."""
    logger = logging.getLogger(__name__)
    
    if check_uv_available():
        logger.info("UV is already available")
        return True
    
    logger.info("Installing UV...")
    
    # Try multiple installation methods
    installation_methods = [
        {
            "name": "curl installer",
            "command": ["curl", "-LsSf", "https://astral.sh/uv/install.sh", "|", "sh"],
            "shell": True
        },
        {
            "name": "pip with --user",
            "command": [sys.executable, "-m", "pip", "install", "--user", "uv"],
            "shell": False
        },
        {
            "name": "pip with --break-system-packages",
            "command": [sys.executable, "-m", "pip", "install", "--break-system-packages", "uv"],
            "shell": False
        }
    ]
    
    for method in installation_methods:
        try:
            logger.info(f"Trying {method['name']}...")
            result = subprocess.run(
                method["command"],
                capture_output=True,
                text=True,
                timeout=120,
                shell=method["shell"]
            )
            
            if result.returncode == 0:
                logger.info(f"UV installed successfully using {method['name']}")
                
                # Verify installation
                if check_uv_available():
                    logger.info("UV installation verified")
                    return True
                else:
                    logger.warning("UV installed but not working, trying next method...")
            else:
                logger.warning(f"{method['name']} failed: {result.stderr.strip()}")
                
        except Exception as e:
            logger.warning(f"{method['name']} failed with exception: {e}")
            continue
    
    logger.error("All UV installation methods failed")
    logger.error("Please install UV manually:")
    logger.error("  curl -LsSf https://astral.sh/uv/install.sh | sh")
    logger.error("  or visit: https://docs.astral.sh/uv/getting-started/installation/")
    return False

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="UV Setup for GNN Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate UV environment even if it exists"
    )
    
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Install development dependencies"
    )
    
    parser.add_argument(
        "--all-extras",
        action="store_true",
        help="Install all optional dependencies"
    )
    
    parser.add_argument(
        "--extras",
        nargs="+",
        help="Install specific optional dependency groups (e.g., llm visualization audio)"
    )
    
    parser.add_argument(
        "--python-version",
        default="3.12",
        help="Python version to use (default: 3.12)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip environment validation after setup"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting UV setup for GNN Pipeline...")
    
    # Get project root
    project_root = Path(__file__).parent
    
    try:
        # Install UV if needed
        if not install_uv():
            logger.error("Failed to install UV")
            return 1
        
        # Determine extras to install
        extras = []
        if args.all_extras:
            extras = ["llm", "visualization", "audio", "gui", "ml-ai", "graphs", "research"]
        elif args.extras:
            extras = args.extras
        
        # Ensure UV environment
        logger.info("Setting up UV environment...")
        if not ensure_uv_environment(
            python_version=args.python_version,
            dev=args.dev,
            extras=extras if extras else None,
            cwd=project_root,
            verbose=args.verbose
        ):
            logger.error("Failed to set up UV environment")
            return 1
        
        # Validate environment if requested
        if not args.skip_validation:
            logger.info("Validating UV environment...")
            validation = comprehensive_uv_validation(project_root)
            
            if validation["overall_status"] == "HEALTHY":
                logger.info("UV environment validation passed")
            elif validation["overall_status"] == "PARTIAL":
                logger.warning("UV environment validation had warnings")
            else:
                logger.error("UV environment validation failed")
                return 1
        
        # Get environment info
        env_info = get_uv_environment_info(project_root)
        logger.info(f"UV environment setup completed successfully")
        logger.info(f"UV version: {env_info.get('uv_version', 'Unknown')}")
        logger.info(f"Packages installed: {env_info.get('validation', {}).get('packages_installed', 0)}")
        
        logger.info("Setup completed successfully!")
        logger.info("You can now run the pipeline with: python src/main.py")
        
        return 0
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
