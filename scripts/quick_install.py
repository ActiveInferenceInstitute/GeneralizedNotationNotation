#!/usr/bin/env python3
"""
Quick Dependency Installation Script

This script quickly installs the essential missing dependencies that are causing
the pipeline execution failures.
"""

import subprocess
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_package(package):
    """Install a single package with pip."""
    logger.info(f"📦 Installing {package}...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0:
            logger.info(f"✅ {package} installed successfully")
            return True
        else:
            logger.error(f"❌ Failed to install {package}: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ {package} installation timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Error installing {package}: {e}")
        return False

def main():
    """Install essential missing dependencies."""
    logger.info("🚀 Quick installation of essential dependencies...")
    
    # Essential packages that are causing failures
    essential_packages = [
        "aiohttp>=3.9.0",
        "ollama",
        "discopy[matrix]>=1.0.0"
    ]
    
    success_count = 0
    total_count = len(essential_packages)
    
    for package in essential_packages:
        if install_package(package):
            success_count += 1
    
    logger.info("=" * 50)
    logger.info("INSTALLATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Successfully installed: {success_count}/{total_count} packages")
    
    if success_count == total_count:
        logger.info("🎉 All essential packages installed!")
        logger.info("💡 You can now run the pipeline: python3 src/main.py --target-dir input/gnn_files")
        return 0
    else:
        logger.warning("⚠️ Some packages failed to install. Check the logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

