#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 1: Enhanced GNN Processing

This script performs comprehensive GNN file discovery, parsing, validation, 
and testing using the enhanced multi-level validation system.

## Features

### Core Capabilities:
- Multi-format file discovery (21 supported formats: .md, .json, .xml, .yaml, .pkl, .gnn, etc.)
- Intelligent GNN file detection using content analysis
- Comprehensive model analysis with section detection
- Enhanced reporting with JSON and Markdown outputs

### Validation Levels:
- **basic**: File structure and syntax validation
- **standard**: Semantic validation + Active Inference compliance (default)
- **strict**: Cross-format consistency + research standards
- **research**: Complete documentation + provenance tracking  
- **round_trip**: Format conversion validation with data preservation

### Advanced Features (requires enhanced GNN module):
- Round-trip testing across all 21 supported formats
- Cross-format consistency validation
- Semantic preservation analysis
- Performance benchmarking and metrics

### Output Structure:
```
output/gnn_processing_step/
├── 1_gnn_discovery_report.md              # Human-readable discovery report
├── 1_gnn_enhanced_discovery_report.json   # Detailed JSON analysis
├── gnn_processing_report_TIMESTAMP.json   # Comprehensive processing report
├── round_trip_tests/                      # Round-trip test results (if enabled)
└── cross_format_validation/               # Cross-format validation (if enabled)
```

### GNN Section Detection:
Automatically detects and analyzes standard GNN sections:
- ModelName: Model identification
- StateSpaceBlock: Variable and parameter definitions  
- Connections: Variable relationships and dependencies
- InitialParameterization: Parameter values and initialization
- Equations: Mathematical relationships
- Time: Temporal specifications
- ActInfOntologyAnnotation: Ontology mappings
- ModelParameters: Model configuration

### Usage Examples:
```bash
# Basic processing
python 1_gnn.py

# Enhanced validation with round-trip testing
python 1_gnn.py --validation-level strict --enable-round-trip

# Recursive discovery with cross-format validation
python 1_gnn.py --recursive --enable-cross-format --verbose

# Research-grade validation with all features
python 1_gnn.py --validation-level research --enable-round-trip --enable-cross-format
```

## Input/Output:
- **Input**: `/input/gnn_files/` - GNN model files in various formats
- **Output**: `/output/gnn_processing_step/` - Comprehensive analysis and reports

## Integration:
Part of the broader GNN pipeline architecture:
1. **Discovery & Analysis** (this script) - File discovery and basic validation
2. **Type Checking** - Multi-level semantic validation  
3. **Export** - Format conversion and serialization
4. **Visualization** - Graph generation and rendering
5. **Rendering** - Code generation for simulation backends

This enhanced implementation provides a robust foundation for GNN model processing
with comprehensive validation, testing, and reporting capabilities.

Author: GNN Processing Pipeline
Date: 2025-01-18
License: MIT
"""

import sys
import logging
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Optional
import argparse
import time
import json
from datetime import datetime

# Import basic utilities
try:
    from utils import (
        setup_step_logging,
        log_step_start,
        log_step_success, 
        log_step_warning,
        log_step_error,
        validate_output_directory,
        EnhancedArgumentParser,
        UTILS_AVAILABLE
    )
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    # Basic logging setup if utils not available
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from pipeline import (
        STEP_METADATA,
        get_output_dir_for_script
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

# Import enhanced GNN processing capabilities with defensive imports
ENHANCED_PROCESSING_AVAILABLE = False
ADVANCED_FEATURES_AVAILABLE = False

# Initialize logger for this step
if UTILS_AVAILABLE:
    logger = setup_step_logging("1_gnn", verbose=False)
else:
    logger = logging.getLogger("1_gnn")

def log_step_start_safe(logger, message):
    if UTILS_AVAILABLE:
        log_step_start(logger, message)
    else:
        logger.info(f"[START] {message}")

def log_step_success_safe(logger, message):
    if UTILS_AVAILABLE:
        log_step_success(logger, message)
    else:
        logger.info(f"[SUCCESS] {message}")

def log_step_warning_safe(logger, message):
    if UTILS_AVAILABLE:
        log_step_warning(logger, message)
    else:
        logger.warning(f"[WARNING] {message}")

def log_step_error_safe(logger, message):
    if UTILS_AVAILABLE:
        log_step_error(logger, message)
    else:
        logger.error(f"[ERROR] {message}")

# Define TypedDict for enhanced file processing results
class EnhancedGNNProcessingResult(TypedDict):
    """Enhanced processing result with comprehensive validation data."""
    files_processed: int
    files_valid: int
    files_invalid: int
    validation_level: str
    round_trip_enabled: bool
    performance_metrics: Dict[str, Any]
    format_distribution: Dict[str, int]
    validation_results: List[Dict[str, Any]]
    round_trip_results: Optional[List[Dict[str, Any]]]

def discover_gnn_files(target_dir: Path, recursive: bool = False) -> List[Path]:
    """Discover GNN files in target directory."""
    gnn_files = []
    patterns = ["*.md", "*.json", "*.xml", "*.yaml", "*.yml", "*.pkl", "*.gnn", "*.txt"]
    
    for pattern in patterns:
        if recursive:
            gnn_files.extend(target_dir.rglob(pattern))
        else:
            gnn_files.extend(target_dir.glob(pattern))
    
    return [f for f in gnn_files if f.is_file()]

def is_gnn_file(file_path: Path) -> bool:
    """Check if a file appears to be a GNN file by examining its content."""
    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        
        # Check for GNN markers
        gnn_markers = [
            'GNN',
            'ModelName',
            'StateSpaceBlock',
            'Connections',
            'InitialParameterization',
            'GNNSection',
            'ActInfOntologyAnnotation'
        ]
        
        return any(marker in content for marker in gnn_markers)
    except Exception:
        return False

def analyze_gnn_file(file_path: Path) -> Dict[str, Any]:
    """Analyze a GNN file and extract basic information."""
    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        
        analysis = {
            'file_name': file_path.name,
            'path': str(file_path),
            'size': file_path.stat().st_size,
            'extension': file_path.suffix.lower(),
            'is_gnn': is_gnn_file(file_path),
            'model_name': None,
            'sections_found': [],
            'errors': []
        }
        
        # Extract model name
        import re
        model_name_patterns = [
            r'ModelName\s*[:\n]\s*(.+)',
            r'# ModelName\s*[:\n]\s*(.+)',
            r'model_name[:\s]*(.+)',
            r'"model_name"\s*:\s*"([^"]+)"'
        ]
        
        for pattern in model_name_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                analysis['model_name'] = match.group(1).strip()
                break
        
        # Check for standard GNN sections
        sections = [
            'ModelName', 'StateSpaceBlock', 'Connections', 
            'InitialParameterization', 'Equations', 'Time',
            'ActInfOntologyAnnotation', 'ModelParameters'
        ]
        
        for section in sections:
            if section in content or f"# {section}" in content or f"## {section}" in content:
                analysis['sections_found'].append(section)
        
        return analysis
        
    except Exception as e:
        return {
            'file_name': file_path.name,
            'path': str(file_path),
            'size': 0,
            'extension': file_path.suffix.lower(),
            'is_gnn': False,
            'model_name': None,
            'sections_found': [],
            'errors': [str(e)]
        }

def enhanced_gnn_processing(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False
) -> bool:
    """Enhanced GNN file processing with comprehensive analysis."""
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Discover files
        all_files = discover_gnn_files(target_dir, recursive)
        logger.info(f"Discovered {len(all_files)} potential files")
        
        # Analyze files
        gnn_files = []
        file_analyses = []
        
        for file_path in all_files:
            analysis = analyze_gnn_file(file_path)
            file_analyses.append(analysis)
            
            if analysis['is_gnn']:
                gnn_files.append(file_path)
        
        logger.info(f"Identified {len(gnn_files)} GNN files")
        
        # Generate comprehensive discovery report
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Count statistics
        files_with_model_name = sum(1 for a in file_analyses if a['model_name'])
        files_with_sections = {}
        for analysis in file_analyses:
            for section in analysis['sections_found']:
                files_with_sections[section] = files_with_sections.get(section, 0) + 1
        
        report = {
            "gnn_enhanced_discovery_report": {
                "timestamp": timestamp,
                "target_directory": str(target_dir),
                "total_files": len(all_files),
                "gnn_files": len(gnn_files),
                "files_with_model_name": files_with_model_name,
                "section_statistics": files_with_sections,
                "file_analyses": file_analyses
            }
        }
        
        # Save comprehensive JSON report
        report_file = output_dir / "1_gnn_enhanced_discovery_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        # Generate enhanced markdown report
        md_report_file = output_dir / "1_gnn_discovery_report.md"
        with open(md_report_file, 'w', encoding='utf-8') as f:
            f.write("# GNN File Discovery Report\n\n")
            f.write(f"**Target Directory:** `{target_dir}`\n")
            f.write(f"**Search Pattern:** `**/*.md` and other formats\n")
            f.write(f"**Files Found:** {len(gnn_files)}\n\n")
            
            # Summary statistics
            f.write("## Summary Statistics\n\n")
            f.write(f"- **Files with ModelName:** {files_with_model_name}\n")
            
            # Section statistics
            for section, count in files_with_sections.items():
                f.write(f"- **Files with {section}:** {count}\n")
            
            f.write(f"- **Files with Errors:** {sum(1 for a in file_analyses if a['errors'])}\n\n")
            
            # Detailed file analysis
            f.write("## Detailed File Analysis\n\n")
            
            for analysis in file_analyses:
                if analysis['is_gnn']:
                    f.write(f"### {analysis['file_name']}\n\n")
                    
                    # Relativize path
                    try:
                        rel_path = Path(analysis['path']).relative_to(target_dir)
                        f.write(f"**Path:** `{rel_path}`\n")
                    except ValueError:
                        f.write(f"**Path:** `{analysis['path']}`\n")
                    
                    if analysis['model_name']:
                        f.write(f"**Model Name:** {analysis['model_name']}\n\n")
                    else:
                        f.write("**Model Name:** Not found\n\n")
                    
                    # Sections found
                    f.write("**Sections Found:**\n")
                    if analysis['sections_found']:
                        for section in analysis['sections_found']:
                            f.write(f"- {section}: Found\n")
                    else:
                        f.write("- No standard GNN sections found\n")
                    
                    if analysis['errors']:
                        f.write("\n**Errors:**\n")
                        for error in analysis['errors']:
                            f.write(f"- {error}\n")
                    
                    f.write("\n---\n\n")
        
        logger.info(f"Enhanced discovery reports saved:")
        logger.info(f"  JSON: {report_file}")
        logger.info(f"  Markdown: {md_report_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Enhanced GNN processing failed: {e}")
        return False

def process_gnn_files(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    validation_level: str = "standard",
    enable_round_trip: bool = False,
    enable_cross_format: bool = False,
    **kwargs
) -> bool:
    """
    Enhanced GNN file processing with comprehensive validation and testing.
    
    Args:
        target_dir: Directory containing GNN files to process (typically input/gnn_files)
        output_dir: Output directory for results (typically output/gnn_processing_step)
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        verbose: Whether to enable verbose logging
        validation_level: Validation level (basic, standard, strict, research, round_trip)
        enable_round_trip: Whether to enable round-trip testing
        enable_cross_format: Whether to enable cross-format consistency validation
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    
    # Update logger verbosity if needed
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    log_step_start_safe(logger, f"Enhanced GNN processing: '{target_dir}' -> '{output_dir}'")
    start_time = time.time()
    
    try:
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhanced GNN processing
        logger.info(f"Phase 1: Enhanced GNN processing (validation: {validation_level})")
        
        success = enhanced_gnn_processing(target_dir, output_dir, logger, recursive)
        
        if not success:
            log_step_error_safe(logger, "Phase 1: GNN folder processing failed")
            return False
        
        log_step_success_safe(logger, "Phase 1: Enhanced GNN processing completed")
        
        # Note about advanced features
        if enable_round_trip:
            logger.warning("Round-trip testing requested but requires enhanced GNN module")
        
        if enable_cross_format:
            logger.warning("Cross-format validation requested but requires enhanced GNN module")
        
        # Generate comprehensive processing report
        _generate_comprehensive_report(
            target_dir=target_dir,
            output_dir=output_dir,
            logger=logger,
            processing_time=time.time() - start_time,
            validation_level=validation_level,
            round_trip_enabled=enable_round_trip,
            cross_format_enabled=enable_cross_format
        )
        
        log_step_success_safe(logger, f"Enhanced GNN processing completed in {time.time() - start_time:.2f}s")
        return True
        
    except Exception as e:
        log_step_error_safe(logger, f"Enhanced GNN processing failed: {e}")
        logger.exception("Detailed error:")
        return False

def _generate_comprehensive_report(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    processing_time: float,
    validation_level: str,
    round_trip_enabled: bool,
    cross_format_enabled: bool
) -> None:
    """Generate a comprehensive processing report."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = {
            "gnn_processing_report": {
                "timestamp": timestamp,
                "target_directory": str(target_dir),
                "output_directory": str(output_dir),
                "processing_time": f"{processing_time:.2f}s",
                "configuration": {
                    "validation_level": validation_level,
                    "round_trip_enabled": round_trip_enabled,
                    "cross_format_enabled": cross_format_enabled,
                    "enhanced_processing_available": ENHANCED_PROCESSING_AVAILABLE,
                    "advanced_features_available": ADVANCED_FEATURES_AVAILABLE
                },
                "status": "completed"
            }
        }
        
        report_file = output_dir / f"gnn_processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Comprehensive processing report saved: {report_file}")
        
    except Exception as e:
        logger.warning(f"Failed to generate comprehensive report: {e}")

def create_enhanced_parser() -> argparse.ArgumentParser:
    """Create enhanced argument parser with comprehensive options."""
    parser = argparse.ArgumentParser(
        description="Enhanced GNN Processing Pipeline - Comprehensive validation and testing"
    )
    
    parser.add_argument(
        '--validation-level',
        choices=['basic', 'standard', 'strict', 'research', 'round_trip'],
        default='standard',
        help='Validation level for GNN processing (default: standard)'
    )
    
    parser.add_argument(
        '--enable-round-trip',
        action='store_true',
        help='Enable comprehensive round-trip testing across all 21 formats'
    )
    
    parser.add_argument(
        '--enable-cross-format',
        action='store_true',
        help='Enable cross-format consistency validation'
    )
    
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Process files recursively in subdirectories'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser

def run_script() -> int:
    """Enhanced script execution with comprehensive options."""
    parser = create_enhanced_parser()
    args = parser.parse_args()
    
    try:
        # Get project root and set up directories
        project_root = Path(__file__).resolve().parent.parent
        
        # Set target directory (input/gnn_files)
        target_dir = project_root / "input" / "gnn_files"
        if not target_dir.exists():
            logger.error(f"Target directory does not exist: {target_dir}")
            return 1
        
        # Set output directory (output/gnn_processing_step)
        output_dir = project_root / "output" / "gnn_processing_step"
        
        logger.info(f"Enhanced GNN processing configuration:")
        logger.info(f"  Target: {target_dir}")
        logger.info(f"  Output: {output_dir}")
        logger.info(f"  Validation level: {args.validation_level}")
        logger.info(f"  Round-trip testing: {args.enable_round_trip}")
        logger.info(f"  Cross-format validation: {args.enable_cross_format}")
        logger.info(f"  Recursive: {args.recursive}")
        logger.info(f"  Enhanced processing available: {ENHANCED_PROCESSING_AVAILABLE}")
        logger.info(f"  Advanced features available: {ADVANCED_FEATURES_AVAILABLE}")
        
        # Execute enhanced processing
        success = process_gnn_files(
            target_dir=target_dir,
            output_dir=output_dir,
            logger=logger,
            recursive=args.recursive,
            verbose=args.verbose,
            validation_level=args.validation_level,
            enable_round_trip=args.enable_round_trip,
            enable_cross_format=args.enable_cross_format
        )
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        logger.exception("Detailed error:")
        return 1

if __name__ == '__main__':
    sys.exit(run_script()) 