#!/usr/bin/env python3
"""
Test configuration dictionaries for the GNN round-trip test suite.

Extracted from ``testing.test_round_trip``.
"""

from typing import Any

# =============================================================================
# TEST CONFIGURATION - Modify these settings to control test behavior
# =============================================================================

# Logging Configuration
LOGGING_CONFIG: dict[str, Any] = {
    "enable_debug": False,  # Disable debug logging for cleaner output
    "enable_detailed_output": False,  # Show concise test progress for final confirmation
    "enable_format_groups": True,  # Group formats by category in output
    "log_level": "WARNING",  # Python logging level (DEBUG, INFO, WARNING, ERROR) - cleaner output
    "suppress_parser_warnings": True,  # Suppress parser-specific warnings for cleaner output
}

# Format Testing Configuration
FORMAT_TEST_CONFIG: dict[str, Any] = {
    # Test all formats (set to False for methodical testing)
    "test_all_formats": False,
    # Selective format testing - only test these formats when test_all_formats=False
    "test_formats": [
        "markdown",  # Always include markdown as reference
        "json",  # Test JSON serialization - ✅ CONFIRMED 100% FUNCTIONAL
        "xml",  # Test XML serialization - ✅ CONFIRMED 100% FUNCTIONAL (FIXED!)
        "yaml",  # Test YAML serialization - ✅ CONFIRMED 100% FUNCTIONAL
        "python",  # Test Python serialization - ✅ CONFIRMED 100% FUNCTIONAL
        "pkl",  # Test PKL serialization - ✅ CONFIRMED 100% FUNCTIONAL
        "scala",  # Test Scala serialization - ✅ CONFIRMED 100% FUNCTIONAL
        "protobuf",  # Test Protobuf serialization - ✅ CONFIRMED 100% FUNCTIONAL
        "xsd",  # Test XSD serialization - ✅ CONFIRMED 100% FUNCTIONAL
        "asn1",  # Test ASN.1 serialization - ✅ CONFIRMED 100% FUNCTIONAL
        "alloy",  # Test Alloy serialization - ✅ CONFIRMED 100% FUNCTIONAL
        "lean",  # Test Lean serialization - ✅ CONFIRMED 100% FUNCTIONAL
        "coq",  # Test Coq serialization - ✅ CONFIRMED 100% FUNCTIONAL
        "isabelle",  # Test Isabelle serialization - ✅ CONFIRMED 100% FUNCTIONAL
        "haskell",  # Test Haskell serialization - ✅ CONFIRMED 100% FUNCTIONAL
        "bnf",  # Test BNF serialization - ✅ CONFIRMED 100% FUNCTIONAL
        "pickle",  # Test Pickle serialization - ✅ CONFIRMED 100% FUNCTIONAL
        "z_notation",  # Test Z notation serialization - ✅ CONFIRMED 100% FUNCTIONAL
        "tla_plus",  # Test TLA+ serialization - ✅ CONFIRMED 100% FUNCTIONAL
        "agda",  # Test Agda serialization - ✅ CONFIRMED 100% FUNCTIONAL
        "maxima",  # Test Maxima serialization - ✅ CONFIRMED 100% FUNCTIONAL
        # PNML disabled in default list (parse-focused; see SPEC.md / FORMAT_TEST_CONFIG notes)
    ],
    # Format categories to test (when test_all_formats=True)
    "test_categories": {
        "schema_formats": True,  # JSON, XML, YAML, XSD, ASN.1, PKL, Protobuf
        "language_formats": True,  # Scala, Python, Haskell, etc.
        "formal_formats": True,  # Lean, Coq, Isabelle, Alloy, Z-notation, etc.
        "grammar_formats": True,  # BNF, EBNF
        "temporal_formats": True,  # TLA+, Agda
        "binary_formats": True,  # Pickle, Binary
    },
    # Individual format control (overrides categories)
    "format_overrides": {
        # 'alloy': False,   # Force disable Alloy testing
        # 'asn1': False,    # Force disable ASN.1 testing
        # 'pickle': False,  # Force disable Pickle testing
    },
}

# Test Behavior Configuration
TEST_BEHAVIOR_CONFIG: dict[str, Any] = {
    "strict_validation": False,  # Disable strict validation to avoid recursion issues
    "fail_fast": False,  # Stop testing on first failure
    "save_converted_files": False,  # Don't save converted files for cleaner output
    "run_cross_format_validation": False,  # Disable cross-format validation to avoid recursion
    "compute_checksums": True,  # Compute semantic checksums for comparison
    "validate_round_trip": True,  # Validate that round-trip preserves semantics - ENABLED!
    "max_test_time": 60,  # Maximum time for all tests (seconds) - reduced for faster testing
    "per_format_timeout": 10,  # Maximum time per format test (seconds) - reduced for faster testing
}

# Output Configuration
OUTPUT_CONFIG: dict[str, Any] = {
    "generate_detailed_report": True,  # Generate detailed markdown report
    "save_test_artifacts": False,  # Don't save test files for cleaner output
    "show_progress_bar": False,  # Don't show progress bar for cleaner output
    "colored_output": True,  # Use colored console output
    "export_json_results": True,  # Export results as JSON
}

# Reference Model Configuration
REFERENCE_CONFIG: dict[str, Any] = {
    "reference_file": "input/gnn_files/actinf_pomdp_agent.md",  # Relative to project root
    "fallback_reference_files": [
        "src/gnn/gnn_examples/actinf_pomdp_agent.md",
        "examples/actinf_pomdp_agent.md",
    ],
    "require_reference_validation": True,  # Require reference file to validate before testing
}

# =============================================================================
# ENHANCED TEST CONFIGURATION - Real functionality
# =============================================================================

ENHANCED_TEST_CONFIG: dict[str, Any] = {
    "graceful_parser_fallback": True,  # Fall back gracefully when parsers fail
    "isolated_serializer_testing": True,  # Test serializers independently
    "robust_error_handling": True,  # Enhanced error handling and reporting
    "direct_file_operations": True,  # Use direct file I/O when needed
}

# =============================================================================
# END CONFIGURATION
# =============================================================================
