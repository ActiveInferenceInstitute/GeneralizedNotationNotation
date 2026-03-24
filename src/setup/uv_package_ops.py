"""
UV Package Operations

Add/remove/update/lock dependency operations extracted from uv_management.py.
"""

import logging
import subprocess  # nosec B404 -- subprocess calls with controlled/trusted input
import sys

from .constants import (
    PROJECT_ROOT,
)

logger = logging.getLogger(__name__)




def add_uv_dependency(package: str, dev: bool = False, verbose: bool = False) -> bool:
    """
    Add a dependency to the project using UV add command.

    Args:
        package: Package name (optionally with version specifier)
        dev: If True, add as development dependency
        verbose: Enable verbose logging

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"📦 Adding package: {package}" + (" (dev)" if dev else ""))
    sys.stdout.flush()

    try:
        cmd = ["uv", "add", package]
        if dev:
            cmd.append("--dev")

        if verbose:
            logger.debug(f"Running: {' '.join(cmd)}")

        result = subprocess.run(  # nosec B603 -- subprocess calls with controlled/trusted input
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )

        if result.returncode != 0:
            logger.error(f"❌ Failed to add package {package}")
            logger.error(result.stderr)
            return False

        logger.info(f"✅ Successfully added {package}")
        return True

    except Exception as e:
        logger.error(f"❌ Error adding package {package}: {e}")
        return False


def remove_uv_dependency(package: str, verbose: bool = False) -> bool:
    """
    Remove a dependency from the project using UV remove command.

    Args:
        package: Package name to remove
        verbose: Enable verbose logging

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"🗑️ Removing package: {package}")
    sys.stdout.flush()

    try:
        cmd = ["uv", "remove", package]

        if verbose:
            logger.debug(f"Running: {' '.join(cmd)}")

        result = subprocess.run(  # nosec B603 -- subprocess calls with controlled/trusted input
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )

        if result.returncode != 0:
            logger.error(f"❌ Failed to remove package {package}")
            logger.error(result.stderr)
            return False

        logger.info(f"✅ Successfully removed {package}")
        return True

    except Exception as e:
        logger.error(f"❌ Error removing package {package}: {e}")
        return False


def update_uv_dependencies(verbose: bool = False, upgrade: bool = False) -> bool:
    """
    Update dependencies using UV sync command.

    Args:
        verbose: Enable verbose logging
        upgrade: If True, upgrade dependencies to latest compatible versions

    Returns:
        True if successful, False otherwise
    """
    logger.info("🔄 Updating dependencies...")
    sys.stdout.flush()

    try:
        cmd = ["uv", "sync"]
        if upgrade:
            cmd.append("--upgrade")

        if verbose:
            cmd.append("--verbose")
            logger.debug(f"Running: {' '.join(cmd)}")

        result = subprocess.run(  # nosec B603 -- subprocess calls with controlled/trusted input
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )

        if result.returncode != 0:
            logger.error("❌ Failed to update dependencies")
            logger.error(result.stderr)
            return False

        logger.info("✅ Successfully updated dependencies")
        if verbose:
            logger.info(result.stdout)
        return True

    except Exception as e:
        logger.error(f"❌ Error updating dependencies: {e}")
        return False


def lock_uv_dependencies(verbose: bool = False) -> bool:
    """
    Update the lock file using UV lock command without installing.

    Args:
        verbose: Enable verbose logging

    Returns:
        True if successful, False otherwise
    """
    logger.info("🔒 Updating dependency lock file...")
    sys.stdout.flush()

    try:
        cmd = ["uv", "lock"]

        if verbose:
            logger.debug(f"Running: {' '.join(cmd)}")

        result = subprocess.run(  # nosec B603 -- subprocess calls with controlled/trusted input
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )

        if result.returncode != 0:
            logger.error("❌ Failed to update lock file")
            logger.error(result.stderr)
            return False

        logger.info("✅ Successfully updated uv.lock")
        return True

    except Exception as e:
        logger.error(f"❌ Error updating lock file: {e}")
        return False
