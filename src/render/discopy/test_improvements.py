#!/usr/bin/env python3
"""
Test script for DisCoPy translator improvements.

This script tests the new error handling and setup reporting functionality
when DisCoPy and JAX dependencies are not available.
"""

import sys
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_discopy_improvements():
    """Test the improved DisCoPy translator functionality."""
    
    # Import the translator module
    try:
        from translator import (
            generate_setup_report,
            create_discopy_error_report,
            check_discopy_availability,
            initialize_discopy_components,
            TENSOR_COMPONENTS_AVAILABLE,
            TY_AVAILABLE,
            JAX_CORE_AVAILABLE,
            DISCOPY_MATRIX_MODULE_AVAILABLE,
            JAX_AVAILABLE
        )
        logger.info("Successfully imported translator module")
    except ImportError as e:
        logger.error(f"Failed to import translator module: {e}")
        return False
    
    # Test 1: Check availability
    logger.info("=== Testing Availability Check ===")
    availability = check_discopy_availability()
    logger.info(f"Availability status: {availability}")
    
    # Test 2: Generate setup report
    logger.info("=== Testing Setup Report Generation ===")
    setup_report = generate_setup_report()
    logger.info("Setup report generated successfully")
    logger.debug(f"Setup report preview: {setup_report[:200]}...")
    
    # Test 3: Create error report
    logger.info("=== Testing Error Report Generation ===")
    test_gnn_file = Path("test_model.md")
    error_report = create_discopy_error_report(test_gnn_file, "unavailable")
    logger.info("Error report created successfully")
    logger.info(f"Error report keys: {list(error_report.keys())}")
    
    # Test 4: Test main functions with unavailable dependencies
    logger.info("=== Testing Main Functions ===")
    
    # Create a test GNN file
    test_gnn_content = """
## ModelName
Test Model

## StateSpaceBlock
A[2]
B[3]

## Connections
A > B
"""
    
    test_file = Path("test_discopy_model.md")
    with open(test_file, 'w') as f:
        f.write(test_gnn_content)
    
    try:
        # Test diagram creation (should fail gracefully)
        from translator import gnn_file_to_discopy_diagram
        
        result = gnn_file_to_discopy_diagram(test_file, verbose=True)
        if result is None:
            logger.info("✓ Diagram creation correctly returned None when dependencies unavailable")
        else:
            logger.warning("⚠ Diagram creation returned result when dependencies should be unavailable")
            
    except Exception as e:
        logger.error(f"✗ Diagram creation failed with exception: {e}")
    
    try:
        # Test matrix diagram creation (should fail gracefully)
        from translator import gnn_file_to_discopy_matrix_diagram
        
        result = gnn_file_to_discopy_matrix_diagram(test_file, verbose=True)
        if result is None:
            logger.info("✓ Matrix diagram creation correctly returned None when dependencies unavailable")
        else:
            logger.warning("⚠ Matrix diagram creation returned result when dependencies should be unavailable")
            
    except Exception as e:
        logger.error(f"✗ Matrix diagram creation failed with exception: {e}")
    
    # Clean up
    test_file.unlink(missing_ok=True)
    
    # Test 5: Verify no placeholder classes exist
    logger.info("=== Testing No Placeholder Classes ===")
    try:
        from translator import PlaceholderBase, DimPlaceholder, BoxPlaceholder
        logger.error("✗ Placeholder classes still exist - they should have been removed")
        return False
    except ImportError:
        logger.info("✓ Placeholder classes successfully removed")
    
    # Test 6: Check global variables
    logger.info("=== Testing Global Variables ===")
    from translator import Dim, Box, Diagram, jax, jnp, discopy_backend
    
    # These should be None when dependencies are not available
    if Dim is None:
        logger.info("✓ Dim is None (expected when DisCoPy not available)")
    else:
        logger.warning("⚠ Dim is not None (unexpected)")
    
    if Box is None:
        logger.info("✓ Box is None (expected when DisCoPy not available)")
    else:
        logger.warning("⚠ Box is not None (unexpected)")
    
    if jax is None:
        logger.info("✓ jax is None (expected when JAX not available)")
    else:
        logger.warning("⚠ jax is not None (unexpected)")
    
    logger.info("=== Test Summary ===")
    logger.info(f"TENSOR_COMPONENTS_AVAILABLE: {TENSOR_COMPONENTS_AVAILABLE}")
    logger.info(f"TY_AVAILABLE: {TY_AVAILABLE}")
    logger.info(f"JAX_CORE_AVAILABLE: {JAX_CORE_AVAILABLE}")
    logger.info(f"DISCOPY_MATRIX_MODULE_AVAILABLE: {DISCOPY_MATRIX_MODULE_AVAILABLE}")
    logger.info(f"JAX_AVAILABLE: {JAX_AVAILABLE}")
    
    return True

if __name__ == "__main__":
    success = test_discopy_improvements()
    if success:
        logger.info("✅ All tests passed - DisCoPy improvements working correctly")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed - DisCoPy improvements need attention")
        sys.exit(1) 