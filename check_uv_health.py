#!/usr/bin/env python3
"""
UV Health Check Script for GNN Pipeline

This script performs comprehensive validation of the UV environment
and provides recommendations for fixing any issues.

Usage:
    python check_uv_health.py [--output-dir OUTPUT_DIR] [--verbose]

Examples:
    # Basic health check
    python check_uv_health.py
    
    # Verbose health check with custom output directory
    python check_uv_health.py --output-dir output/uv_health --verbose
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.uv_validator import (
    comprehensive_uv_validation,
    generate_uv_health_report,
    save_uv_validation_report
)

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

def main():
    """Main health check function."""
    parser = argparse.ArgumentParser(
        description="UV Health Check for GNN Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/uv_health"),
        help="Output directory for health check reports (default: output/uv_health)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting UV health check...")
    
    # Get project root
    project_root = Path(__file__).parent
    
    try:
        # Perform comprehensive validation
        validation = comprehensive_uv_validation(project_root)
        
        # Generate and display health report
        health_report = generate_uv_health_report(validation)
        print(health_report)
        
        # Save reports
        save_uv_validation_report(validation, args.output_dir)
        
        # Return appropriate exit code
        if validation["overall_status"] == "HEALTHY":
            logger.info("UV environment is healthy")
            return 0
        elif validation["overall_status"] == "PARTIAL":
            logger.warning("UV environment has some issues")
            return 1
        else:
            logger.error("UV environment is unhealthy")
            return 2
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return 3

if __name__ == "__main__":
    sys.exit(main())
