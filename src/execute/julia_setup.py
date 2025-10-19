#!/usr/bin/env python3
"""
Julia Environment Setup for GNN Pipeline

This module provides automated Julia environment setup and package installation
for RxInfer.jl and ActiveInference.jl frameworks used in the execution step.
"""

import subprocess
import sys
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def check_julia_availability() -> tuple[bool, Optional[str]]:
    """
    Check if Julia is available and return its path.

    Returns:
        Tuple of (is_available, julia_path)
    """
    julia_path = shutil.which("julia")
    if julia_path:
        logger.info(f"✅ Julia found at: {julia_path}")
        return True, julia_path
    else:
        logger.warning("❌ Julia not found in PATH")
        return False, None


def check_julia_version(julia_path: str) -> Optional[str]:
    """
    Get Julia version.

    Args:
        julia_path: Path to Julia executable

    Returns:
        Version string or None if check fails
    """
    try:
        result = subprocess.run(
            [julia_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            version_line = result.stdout.strip().split('\n')[0]
            logger.info(f"Julia version: {version_line}")
            return version_line
        else:
            logger.warning(f"Failed to get Julia version: {result.stderr}")
            return None
    except Exception as e:
        logger.error(f"Error checking Julia version: {e}")
        return None


def run_julia_setup_script(
    julia_path: str,
    setup_script: Path,
    verbose: bool = False,
    force_reinstall: bool = False,
    validate_only: bool = False
) -> bool:
    """
    Run the Julia setup script.

    Args:
        julia_path: Path to Julia executable
        setup_script: Path to setup_environment.jl script
        verbose: Enable verbose output
        force_reinstall: Force reinstall of packages
        validate_only: Only validate, don't install

    Returns:
        True if successful
    """
    if not setup_script.exists():
        logger.error(f"Julia setup script not found: {setup_script}")
        return False

    logger.info(f"Running Julia setup script: {setup_script}")

    cmd = [julia_path, str(setup_script)]

    if verbose:
        cmd.append("--verbose")
    if force_reinstall:
        cmd.append("--force-reinstall")
    if validate_only:
        cmd.append("--validate-only")

    try:
        result = subprocess.run(
            cmd,
            cwd=setup_script.parent,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for package installation
        )

        if result.stdout:
            logger.info(f"Julia setup stdout:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"Julia setup stderr:\n{result.stderr}")

        if result.returncode == 0:
            logger.info("✅ Julia setup completed successfully")
            return True
        else:
            logger.error(f"❌ Julia setup failed with return code: {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("⏰ Julia setup timed out after 5 minutes")
        return False
    except Exception as e:
        logger.error(f"❌ Julia setup failed: {e}")
        return False


def setup_julia_environment(
    verbose: bool = False,
    force_reinstall: bool = False,
    validate_only: bool = False,
    frameworks: List[str] = None
) -> Dict[str, Any]:
    """
    Setup Julia environment for specified frameworks.

    Args:
        verbose: Enable verbose output
        force_reinstall: Force reinstall of packages
        validate_only: Only validate, don't install
        frameworks: List of frameworks to setup (None = all)

    Returns:
        Dictionary with setup results
    """
    logger.info("🔧 Setting up Julia environment for GNN execution")

    # Check Julia availability
    julia_available, julia_path = check_julia_availability()
    if not julia_available:
        return {
            "success": False,
            "julia_available": False,
            "error": "Julia not found in PATH",
            "suggestion": "Install Julia from https://julialang.org/downloads/"
        }

    # Check Julia version
    version = check_julia_version(julia_path)
    if not version:
        return {
            "success": False,
            "julia_available": True,
            "error": "Failed to determine Julia version",
            "julia_path": julia_path
        }

    # Define framework setup scripts
    framework_scripts = {
        "rxinfer": "src/execute/rxinfer/setup_environment.jl",
        "activeinference_jl": "src/execute/activeinference_jl/setup_environment.jl"
    }

    # Filter frameworks if specified
    if frameworks:
        framework_scripts = {k: v for k, v in framework_scripts.items() if k in frameworks}

    setup_results = {}
    overall_success = True

    for framework, script_path in framework_scripts.items():
        script_file = Path(script_path)
        if not script_file.exists():
            logger.warning(f"Setup script not found for {framework}: {script_file}")
            setup_results[framework] = {
                "success": False,
                "error": f"Setup script not found: {script_file}",
                "skipped": True
            }
            continue

        logger.info(f"Setting up {framework}...")

        success = run_julia_setup_script(
            julia_path,
            script_file,
            verbose=verbose,
            force_reinstall=force_reinstall,
            validate_only=validate_only
        )

        setup_results[framework] = {
            "success": success,
            "script_path": str(script_file),
            "julia_path": julia_path,
            "version": version
        }

        if not success:
            overall_success = False

    return {
        "success": overall_success,
        "julia_available": True,
        "julia_path": julia_path,
        "julia_version": version,
        "frameworks_setup": setup_results
    }


def main():
    """Main entry point for Julia setup."""
    import argparse

    parser = argparse.ArgumentParser(description="Setup Julia environment for GNN execution")
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--force-reinstall', action='store_true', help='Force reinstall of packages')
    parser.add_argument('--validate-only', action='store_true', help='Only validate, don\'t install')
    parser.add_argument('--frameworks', type=str, help='Comma-separated list of frameworks to setup (rxinfer,activeinference_jl)')

    args = parser.parse_args()

    # Parse frameworks
    frameworks = None
    if args.frameworks:
        frameworks = [f.strip() for f in args.frameworks.split(',')]

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Run setup
    results = setup_julia_environment(
        verbose=args.verbose,
        force_reinstall=args.force_reinstall,
        validate_only=args.validate_only,
        frameworks=frameworks
    )

    # Print results
    print(f"\n{'='*60}")
    print("JULIA SETUP RESULTS")
    print(f"{'='*60}")

    if results["julia_available"]:
        print(f"✅ Julia found: {results['julia_path']}")
        print(f"📦 Version: {results['julia_version']}")
    else:
        print(f"❌ Julia not found")
        print(f"💡 Install Julia from: https://julialang.org/downloads/")

    print(f"\nFramework Setup Results:")
    for framework, result in results["frameworks_setup"].items():
        if result.get("skipped"):
            print(f"⏭️  {framework}: Skipped ({result['error']})")
        elif result["success"]:
            print(f"✅ {framework}: Setup completed")
        else:
            print(f"❌ {framework}: Setup failed")

    print(f"\nOverall Success: {'✅ Yes' if results['success'] else '❌ No'}")

    # Exit with appropriate code
    return 0 if results["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
