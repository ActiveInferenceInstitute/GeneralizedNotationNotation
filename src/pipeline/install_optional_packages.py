#!/usr/bin/env python3
"""
Standalone script to install optional GNN package groups.

This script provides an easy way to install optional dependencies for the GNN pipeline
using UV. It can be run from a cold start and will install the specified package groups.

Usage:
    # Install all optional packages
    python3 src/pipeline/install_optional_packages.py --all
    
    # Install specific groups
    python3 src/pipeline/install_optional_packages.py --groups jax,pymdp,visualization
    
    # Install with verbose output
    python3 src/pipeline/install_optional_packages.py --all --verbose
    
Available package groups:
    - jax: JAX, jaxlib, optax, flax (high-performance computing)
    - pymdp: inferactively-pymdp (Active Inference framework)
    - visualization: plotly, altair, seaborn (data visualization)
    - audio: librosa, soundfile, pedalboard (audio processing)
    - llm: openai, anthropic (LLM integration)
    - ml: torch, torchvision, transformers (machine learning)
"""

import argparse
import sys
from pathlib import Path

# Add src to path (now in src/pipeline/, so go up one level to src/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from setup.setup import (
    install_optional_package_group,
    install_all_optional_packages,
    get_installed_package_versions,
    logger
)


def main():
    """Main entry point for optional package installation."""
    parser = argparse.ArgumentParser(
        description="Install optional GNN package groups using UV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                           # Install all optional packages
  %(prog)s --groups jax,pymdp             # Install specific groups
  %(prog)s --groups visualization --verbose  # Install with verbose output
  
Available groups: jax, pymdp, visualization, audio, llm, ml
        """
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Install all optional package groups"
    )
    
    parser.add_argument(
        "--groups",
        type=str,
        help="Comma-separated list of package groups to install"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available package groups and exit"
    )
    
    args = parser.parse_args()
    
    # List available groups
    if args.list:
        print("Available package groups:")
        print("  - jax: JAX, jaxlib, optax, flax (high-performance computing)")
        print("  - pymdp: inferactively-pymdp (Active Inference framework)")
        print("  - visualization: plotly, altair, seaborn (data visualization)")
        print("  - audio: librosa, soundfile, pedalboard (audio processing)")
        print("  - llm: openai, anthropic (LLM integration)")
        print("  - ml: torch, torchvision, transformers (machine learning)")
        return 0
    
    # Check that at least one option is provided
    if not args.all and not args.groups:
        parser.error("Please specify --all or --groups")
    
    try:
        logger.info("🚀 Starting optional package installation...")
        
        # Install all or specific groups
        if args.all:
            logger.info("📦 Installing ALL optional package groups...")
            results = install_all_optional_packages(verbose=args.verbose)
            
            # Print summary
            successful = sum(1 for v in results.values() if v)
            total = len(results)
            
            print(f"\n{'='*70}")
            print(f"📊 Installation Summary: {successful}/{total} groups installed successfully")
            print(f"{'='*70}")
            
            for group, success in results.items():
                status = "✅" if success else "❌"
                print(f"  {status} {group}")
            
            if successful < total:
                print(f"\n⚠️  Some packages failed to install (non-critical)")
                return 2  # Success with warnings
            else:
                print(f"\n✅ All optional packages installed successfully!")
                return 0
                
        else:
            # Install specific groups
            groups = [g.strip() for g in args.groups.split(',')]
            logger.info(f"📦 Installing package groups: {', '.join(groups)}")
            
            results = {}
            for group in groups:
                results[group] = install_optional_package_group(group, verbose=args.verbose)
            
            # Print summary
            successful = sum(1 for v in results.values() if v)
            total = len(results)
            
            print(f"\n{'='*70}")
            print(f"📊 Installation Summary: {successful}/{total} groups installed successfully")
            print(f"{'='*70}")
            
            for group, success in results.items():
                status = "✅" if success else "❌"
                print(f"  {status} {group}")
            
            if successful < total:
                print(f"\n⚠️  Some packages failed to install")
                return 2  # Success with warnings
            else:
                print(f"\n✅ Selected packages installed successfully!")
                return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Installation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Installation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())


