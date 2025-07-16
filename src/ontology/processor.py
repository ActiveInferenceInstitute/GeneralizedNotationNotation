#!/usr/bin/env python3
"""
GNN Ontology Processing Module

Enhanced ontology processing with comprehensive validation, error handling,
and reporting capabilities for Active Inference models.
"""

from pathlib import Path
import logging
from pipeline import get_output_dir_for_script
from utils import log_step_start, log_step_success, log_step_warning, log_step_error, performance_tracker
import time
import json
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

# Import ontology functionality from mcp.py
try:
    from .mcp import (
        parse_gnn_ontology_section,
        load_defined_ontology_terms,
        validate_annotations,
        generate_ontology_report_for_file
    )
    ONTOLOGY_AVAILABLE = True
except ImportError as e:
    # Fallback implementation
    ONTOLOGY_AVAILABLE = False
    
    def parse_gnn_ontology_section(content, verbose=False):
        return {}
    
    def load_defined_ontology_terms(path):
        return {}
    
    def validate_annotations(annotations, terms):
        return {"valid_mappings": {}, "invalid_terms": {}, "unmapped_model_vars": []}
    
    def generate_ontology_report_for_file(file_path, annotations, validation_results=None):
        return f"# Ontology Report for {file_path}\n\nOntology functionality not available."

class ValidationMode(Enum):
    """Ontology validation modes."""
    STRICT = "strict"        # All terms must be valid
    LENIENT = "lenient"      # Warnings for invalid terms
    PERMISSIVE = "permissive" # No validation, just parsing

@dataclass
class OntologyProcessingOptions:
    """Configuration options for ontology processing."""
    validation_mode: ValidationMode = ValidationMode.LENIENT
    case_sensitive: bool = True
    fuzzy_matching: bool = False
    fuzzy_threshold: float = 0.8
    generate_detailed_reports: bool = True
    include_statistics: bool = True
    export_mappings: bool = True

@dataclass 
class FileProcessingResult:
    """Result of processing a single GNN file."""
    file_path: Path
    success: bool
    annotations: Dict[str, str]
    validation_results: Optional[Dict]
    error_message: Optional[str] = None
    processing_time: float = 0.0
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

@dataclass
class OntologyProcessingSummary:
    """Summary of ontology processing across all files."""
    total_files: int
    successful_files: int
    failed_files: int
    total_annotations: int
    valid_annotations: int
    invalid_annotations: int
    unique_terms_used: Set[str]
    unique_invalid_terms: Set[str]
    processing_time: float
    
    @property
    def success_rate(self) -> float:
        return (self.successful_files / self.total_files * 100) if self.total_files > 0 else 0.0
    
    @property
    def validation_rate(self) -> float:
        return (self.valid_annotations / self.total_annotations * 100) if self.total_annotations > 0 else 0.0

def fuzzy_match_term(term: str, defined_terms: Dict[str, any], threshold: float = 0.8) -> Optional[str]:
    """Find the best fuzzy match for a term in defined terms."""
    try:
        from difflib import SequenceMatcher
        best_match = None
        best_ratio = 0.0
        
        for defined_term in defined_terms.keys():
            ratio = SequenceMatcher(None, term.lower(), defined_term.lower()).ratio()
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                best_match = defined_term
        
        return best_match
    except ImportError:
        return None

def enhanced_validate_annotations(
    annotations: Dict[str, str], 
    defined_terms: Dict[str, any], 
    options: OntologyProcessingOptions
) -> Dict:
    """Enhanced validation with fuzzy matching and case sensitivity options."""
    results = {
        "valid_mappings": {},
        "invalid_terms": {},
        "unmapped_model_vars": [],
        "fuzzy_suggestions": {},
        "case_mismatches": {}
    }
    
    # Create comparison sets based on case sensitivity
    if options.case_sensitive:
        defined_term_keys = set(defined_terms.keys())
        def term_match(term, defined_set):
            return term in defined_set
    else:
        defined_term_keys = set(term.lower() for term in defined_terms.keys())
        def term_match(term, defined_set):
            return term.lower() in defined_set
    
    for model_var, ontology_term in annotations.items():
        if not ontology_term:
            results["unmapped_model_vars"].append(model_var)
            continue
        
        # Direct match
        if term_match(ontology_term, defined_term_keys):
            results["valid_mappings"][model_var] = ontology_term
        else:
            # Check for case mismatch if case sensitive
            if options.case_sensitive:
                for defined_term in defined_terms.keys():
                    if ontology_term.lower() == defined_term.lower():
                        results["case_mismatches"][model_var] = {
                            "used": ontology_term,
                            "correct": defined_term
                        }
                        break
            
            # Try fuzzy matching if enabled
            if options.fuzzy_matching:
                fuzzy_match = fuzzy_match_term(ontology_term, defined_terms, options.fuzzy_threshold)
                if fuzzy_match:
                    results["fuzzy_suggestions"][model_var] = {
                        "used": ontology_term,
                        "suggested": fuzzy_match
                    }
            
            results["invalid_terms"][model_var] = ontology_term
    
    return results

def process_single_gnn_file(
    gnn_file: Path,
    defined_terms: Dict[str, any],
    options: OntologyProcessingOptions,
    logger: logging.Logger
) -> FileProcessingResult:
    """Process a single GNN file with enhanced error handling."""
    start_time = time.time()
    
    try:
        # Read GNN file content
        with open(gnn_file, 'r', encoding='utf-8') as f:
            gnn_content = f.read()
        
        # Parse ontology annotations
        annotations = parse_gnn_ontology_section(
            gnn_content, 
            verbose=logger.isEnabledFor(logging.DEBUG)
        )
        
        # Validate annotations
        validation_results = None
        warnings = []
        
        if defined_terms:
            if options.validation_mode != ValidationMode.PERMISSIVE:
                validation_results = enhanced_validate_annotations(
                    annotations, defined_terms, options
                )
                
                # Generate warnings based on validation mode
                if options.validation_mode == ValidationMode.STRICT:
                    if validation_results.get("invalid_terms"):
                        raise ValueError(f"Invalid ontology terms found in strict mode: {list(validation_results['invalid_terms'].values())}")
                elif options.validation_mode == ValidationMode.LENIENT:
                    if validation_results.get("invalid_terms"):
                        warnings.append(f"Found {len(validation_results['invalid_terms'])} invalid ontology terms")
                    if validation_results.get("case_mismatches"):
                        warnings.append(f"Found {len(validation_results['case_mismatches'])} case mismatches")
        
        processing_time = time.time() - start_time
        
        return FileProcessingResult(
            file_path=gnn_file,
            success=True,
            annotations=annotations,
            validation_results=validation_results,
            processing_time=processing_time,
            warnings=warnings
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        return FileProcessingResult(
            file_path=gnn_file,
            success=False,
            annotations={},
            validation_results=None,
            error_message=str(e),
            processing_time=processing_time
        )

def generate_enhanced_report(
    file_result: FileProcessingResult,
    options: OntologyProcessingOptions
) -> str:
    """Generate enhanced ontology report with detailed analysis."""
    report_lines = [
        f"# Ontology Analysis Report for `{file_result.file_path.name}`\n",
        f"**File:** `{file_result.file_path}`\n",
        f"**Processing Time:** {file_result.processing_time:.3f}s\n",
        f"**Status:** {'✅ Success' if file_result.success else '❌ Failed'}\n",
    ]
    
    if file_result.error_message:
        report_lines.extend([
            f"**Error:** {file_result.error_message}\n",
            "\n---\n"
        ])
        return "".join(report_lines)
    
    # Annotations summary
    report_lines.extend([
        f"**Annotations Found:** {len(file_result.annotations)}\n",
        "\n## Annotations\n"
    ])
    
    if not file_result.annotations:
        report_lines.append("- No `ActInfOntologyAnnotation` section found.\n")
        return "".join(report_lines)
    
    # List all annotations
    for var, term in file_result.annotations.items():
        status_icon = "✅" if file_result.validation_results and var in file_result.validation_results.get("valid_mappings", {}) else "❓"
        report_lines.append(f"- {status_icon} `{var}` → `{term}`\n")
    
    # Validation results
    if file_result.validation_results and options.generate_detailed_reports:
        validation = file_result.validation_results
        
        report_lines.append("\n## Validation Results\n")
        
        valid_count = len(validation.get("valid_mappings", {}))
        invalid_count = len(validation.get("invalid_terms", {}))
        
        report_lines.extend([
            f"- **Valid Terms:** {valid_count}\n",
            f"- **Invalid Terms:** {invalid_count}\n",
            f"- **Validation Rate:** {(valid_count / (valid_count + invalid_count) * 100):.1f}%\n"
        ])
        
        # Case mismatches
        if validation.get("case_mismatches"):
            report_lines.append("\n### Case Mismatches\n")
            for var, mismatch in validation["case_mismatches"].items():
                report_lines.append(f"- `{var}`: Used `{mismatch['used']}`, should be `{mismatch['correct']}`\n")
        
        # Fuzzy suggestions
        if validation.get("fuzzy_suggestions"):
            report_lines.append("\n### Suggested Corrections\n")
            for var, suggestion in validation["fuzzy_suggestions"].items():
                report_lines.append(f"- `{var}`: `{suggestion['used']}` → `{suggestion['suggested']}` (suggested)\n")
        
        # Invalid terms
        if validation.get("invalid_terms"):
            report_lines.append("\n### Invalid Terms\n")
            for var, term in validation["invalid_terms"].items():
                report_lines.append(f"- `{var}`: `{term}` (not found in ontology)\n")
    
    # Warnings
    if file_result.warnings:
        report_lines.append("\n## Warnings\n")
        for warning in file_result.warnings:
            report_lines.append(f"- ⚠️ {warning}\n")
    
    report_lines.append("\n---\n")
    return "".join(report_lines)

def process_ontology_operations(
    target_dir: Path, 
    output_dir: Path, 
    logger: logging.Logger, 
    recursive: bool = False, 
    ontology_terms_file: Optional[Path] = None,
    validation_mode: str = "lenient",
    case_sensitive: bool = True,
    fuzzy_matching: bool = True,
    fuzzy_threshold: float = 0.8
) -> bool:
    """
    Enhanced ontology processing with comprehensive validation and reporting.
    
    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Output directory for ontology results  
        logger: Logger instance for this step
        recursive: Whether to process files recursively
        ontology_terms_file: Path to ontology terms JSON file
        validation_mode: Validation mode (strict, lenient, permissive)
        case_sensitive: Whether term matching is case sensitive
        fuzzy_matching: Whether to enable fuzzy term matching
        fuzzy_threshold: Threshold for fuzzy matching (0.0-1.0)
        
    Returns:
        True if processing succeeded, False otherwise
    """
    log_step_start(logger, "Processing Active Inference ontology operations with enhanced validation")
    
    # Use centralized output directory configuration
    ontology_output_dir = get_output_dir_for_script("8_ontology.py", output_dir)
    ontology_output_dir.mkdir(parents=True, exist_ok=True)
    
    if not ONTOLOGY_AVAILABLE:
        log_step_error(logger, "Ontology functionality not available")
        return False
    
    # Configure processing options
    try:
        validation_mode_enum = ValidationMode(validation_mode)
    except ValueError:
        validation_mode_enum = ValidationMode.LENIENT
        logger.warning(f"Invalid validation mode '{validation_mode}', using 'lenient'")
    
    options = OntologyProcessingOptions(
        validation_mode=validation_mode_enum,
        case_sensitive=case_sensitive,
        fuzzy_matching=fuzzy_matching,
        fuzzy_threshold=fuzzy_threshold,
        generate_detailed_reports=True,
        include_statistics=True,
        export_mappings=True
    )
    
    logger.info(f"Ontology processing configuration: {options}")
    
    try:
        # Load ontology terms if provided
        defined_terms = {}
        if ontology_terms_file and ontology_terms_file.exists():
            logger.info(f"Loading ontology terms from: {ontology_terms_file}")
            defined_terms = load_defined_ontology_terms(str(ontology_terms_file))
            logger.info(f"Loaded {len(defined_terms)} ontology terms")
        else:
            logger.warning("No ontology terms file provided or file not found")
            if options.validation_mode != ValidationMode.PERMISSIVE:
                logger.warning("Validation disabled due to missing ontology terms file")
        
        # Find GNN files
        pattern = "**/*.md" if recursive else "*.md"
        gnn_files = list(target_dir.glob(pattern))
        
        if not gnn_files:
            log_step_warning(logger, f"No GNN files found in {target_dir}")
            return True
        
        logger.info(f"Found {len(gnn_files)} GNN files for ontology processing")
        
        # Process files with enhanced validation
        file_results = []
        summary_stats = {
            "total_annotations": 0,
            "valid_annotations": 0,
            "invalid_annotations": 0,
            "unique_terms": set(),
            "unique_invalid_terms": set()
        }
        
        with performance_tracker.track_operation("enhanced_ontology_processing"):
            for gnn_file in gnn_files:
                logger.debug(f"Processing {gnn_file.name} with enhanced ontology validation")
                
                result = process_single_gnn_file(gnn_file, defined_terms, options, logger)
                file_results.append(result)
                
                # Update summary statistics
                if result.success:
                    summary_stats["total_annotations"] += len(result.annotations)
                    summary_stats["unique_terms"].update(result.annotations.values())
                    
                    if result.validation_results:
                        valid_count = len(result.validation_results.get("valid_mappings", {}))
                        invalid_count = len(result.validation_results.get("invalid_terms", {}))
                        summary_stats["valid_annotations"] += valid_count
                        summary_stats["invalid_annotations"] += invalid_count
                        summary_stats["unique_invalid_terms"].update(
                            result.validation_results.get("invalid_terms", {}).values()
                        )
                
                # Create file-specific output directory and save results
                file_output_dir = ontology_output_dir / gnn_file.stem
                file_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate and save enhanced report
                report_content = generate_enhanced_report(result, options)
                report_file = file_output_dir / "ontology_report.md"
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                
                # Save detailed results as JSON
                result_data = {
                    "file": str(result.file_path),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "success": result.success,
                    "processing_time": result.processing_time,
                    "annotations_found": len(result.annotations),
                    "annotations": result.annotations,
                    "validation_results": result.validation_results,
                    "error_message": result.error_message,
                    "warnings": result.warnings
                }
                
                results_file = file_output_dir / "ontology_results.json"
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, indent=2, ensure_ascii=False)
                    
        # Generate comprehensive summary
        successful_files = sum(1 for r in file_results if r.success)
        failed_files = len(file_results) - successful_files
        
        summary = OntologyProcessingSummary(
            total_files=len(file_results),
            successful_files=successful_files,
            failed_files=failed_files,
            total_annotations=summary_stats["total_annotations"],
            valid_annotations=summary_stats["valid_annotations"],
            invalid_annotations=summary_stats["invalid_annotations"],
            unique_terms_used=summary_stats["unique_terms"],
            unique_invalid_terms=summary_stats["unique_invalid_terms"],
            processing_time=sum(r.processing_time for r in file_results)
        )
        
        # Save summary report
        summary_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_options": {
                "validation_mode": options.validation_mode.value,
                "case_sensitive": options.case_sensitive,
                "fuzzy_matching": options.fuzzy_matching,
                "fuzzy_threshold": options.fuzzy_threshold
            },
            "summary_statistics": {
                "total_files": summary.total_files,
                "successful_files": summary.successful_files,
                "failed_files": summary.failed_files,
                "success_rate": summary.success_rate,
                "total_annotations": summary.total_annotations,
                "valid_annotations": summary.valid_annotations,
                "invalid_annotations": summary.invalid_annotations,
                "validation_rate": summary.validation_rate,
                "unique_terms_count": len(summary.unique_terms_used),
                "unique_invalid_terms_count": len(summary.unique_invalid_terms),
                "processing_time": summary.processing_time
            },
            "ontology_terms_loaded": len(defined_terms),
            "files_processed": [
                {
                    "file": str(r.file_path),
                    "success": r.success,
                    "annotations_count": len(r.annotations),
                    "processing_time": r.processing_time,
                    "error": r.error_message,
                    "warnings_count": len(r.warnings) if r.warnings else 0
                }
                for r in file_results
            ]
        }
        
        summary_file = ontology_output_dir / "ontology_processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        # Generate enhanced markdown summary
        markdown_summary = generate_enhanced_markdown_summary(summary, file_results, options, ontology_output_dir)
        summary_md_file = ontology_output_dir / "ontology_summary_report.md"
        with open(summary_md_file, 'w', encoding='utf-8') as f:
            f.write(markdown_summary)
        
        # Export all mappings if requested
        if options.export_mappings:
            all_mappings = {}
            for result in file_results:
                if result.success and result.annotations:
                    all_mappings[str(result.file_path)] = result.annotations
            
            mappings_file = ontology_output_dir / "all_ontology_mappings.json"
            with open(mappings_file, 'w', encoding='utf-8') as f:
                json.dump(all_mappings, f, indent=2, ensure_ascii=False)
        
        # Log final results
        if summary.success_rate == 100.0:
            log_step_success(logger, 
                f"All {summary.total_files} files processed successfully. "
                f"Found {summary.total_annotations} annotations with "
                f"{summary.validation_rate:.1f}% validation rate")
            return True
        elif summary.successful_files > 0:
            log_step_warning(logger, 
                f"Partial success: {summary.successful_files}/{summary.total_files} files processed. "
                f"Validation rate: {summary.validation_rate:.1f}%")
            return True
        else:
            log_step_error(logger, "No files were processed successfully with ontology")
            return False
        
    except Exception as e:
        log_step_error(logger, f"Enhanced ontology processing failed: {e}")
        return False

def generate_enhanced_markdown_summary(
    summary: OntologyProcessingSummary,
    file_results: List[FileProcessingResult],
    options: OntologyProcessingOptions,
    output_dir: Path
) -> str:
    """Generate a comprehensive enhanced markdown summary report."""
    lines = [
        "# Enhanced Active Inference Ontology Processing Summary\n\n",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        "## Processing Configuration\n\n",
        f"- **Validation Mode:** {options.validation_mode.value}\n",
        f"- **Case Sensitive:** {options.case_sensitive}\n", 
        f"- **Fuzzy Matching:** {options.fuzzy_matching}\n",
        f"- **Fuzzy Threshold:** {options.fuzzy_threshold}\n\n",
        "## Summary Statistics\n\n",
        f"| Metric | Value |\n",
        f"|--------|-------|\n",
        f"| Total Files | {summary.total_files} |\n",
        f"| Successful Files | {summary.successful_files} |\n",
        f"| Failed Files | {summary.failed_files} |\n",
        f"| Success Rate | {summary.success_rate:.1f}% |\n",
        f"| Total Annotations | {summary.total_annotations} |\n",
        f"| Valid Annotations | {summary.valid_annotations} |\n",
        f"| Invalid Annotations | {summary.invalid_annotations} |\n",
        f"| Validation Rate | {summary.validation_rate:.1f}% |\n",
        f"| Unique Terms Used | {len(summary.unique_terms_used)} |\n",
        f"| Unique Invalid Terms | {len(summary.unique_invalid_terms)} |\n",
        f"| Total Processing Time | {summary.processing_time:.3f}s |\n\n"
    ]
    
    # Terms analysis
    if summary.unique_terms_used:
        lines.extend([
            "## Terms Analysis\n\n",
            "### All Terms Used\n",
            "```\n"
        ])
        for term in sorted(summary.unique_terms_used):
            lines.append(f"{term}\n")
        lines.append("```\n\n")
    
    if summary.unique_invalid_terms:
        lines.extend([
            "### Invalid Terms Found\n",
            "```\n"
        ])
        for term in sorted(summary.unique_invalid_terms):
            lines.append(f"{term}\n")
        lines.append("```\n\n")
    
    # File processing results
    lines.append("## File Processing Results\n\n")
    
    for result in file_results:
        status_icon = "✅" if result.success else "❌"
        lines.append(f"### {status_icon} {result.file_path.name}\n\n")
        lines.append(f"- **File:** `{result.file_path}`\n")
        lines.append(f"- **Processing Time:** {result.processing_time:.3f}s\n")
        lines.append(f"- **Annotations Found:** {len(result.annotations)}\n")
        
        if result.error_message:
            lines.append(f"- **Error:** {result.error_message}\n")
        
        if result.warnings:
            lines.append(f"- **Warnings:** {len(result.warnings)}\n")
            for warning in result.warnings:
                lines.append(f"  - ⚠️ {warning}\n")
        
        if result.validation_results:
            valid_count = len(result.validation_results.get("valid_mappings", {}))
            invalid_count = len(result.validation_results.get("invalid_terms", {}))
            if valid_count + invalid_count > 0:
                validation_rate = valid_count / (valid_count + invalid_count) * 100
                lines.append(f"- **Validation Rate:** {validation_rate:.1f}% ({valid_count}/{valid_count + invalid_count})\n")
        
        lines.append("\n")
        
        # Add annotation details for successful processing
        if result.success and result.annotations:
            lines.append("#### Annotations:\n")
            for var, term in result.annotations.items():
                valid = (result.validation_results and 
                        var in result.validation_results.get("valid_mappings", {}))
                status = "✅" if valid else "❓"
                lines.append(f"- {status} `{var}` → `{term}`\n")
            lines.append("\n")
    
    lines.extend([
        "---\n\n",
        "*Report generated by Enhanced GNN Ontology Processing Module*\n"
    ])
    
    return "".join(lines) 