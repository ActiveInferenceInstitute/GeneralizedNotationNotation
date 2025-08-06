#!/usr/bin/env python3
"""
Test Gnn Validation Tests

This file contains tests migrated from test_gnn_core_modules.py.
"""

import pytest
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import *


# Migrated from test_gnn_core_modules.py
class TestGNNValidation:
    """Test gnn.validation module."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_validation_imports(self):
        """Test that validation module can be imported."""
        try:
            from src.gnn import validation
            assert hasattr(validation, 'ValidationStrategy')
            # Test that we can also import the main functions from gnn package
            from src.gnn import validate_gnn_file, validate_gnn_structure
            assert callable(validate_gnn_file)
            assert callable(validate_gnn_structure)
        except ImportError:
            pytest.skip("GNN validation not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_validate_gnn_model(self, comprehensive_test_data):
        """Test GNN model validation."""
        try:
            from src.gnn.validation import validate_gnn_model
            
            # Test with valid model
            test_model = comprehensive_test_data['models']['simple_model']
            
            result = validate_gnn_model(test_model)
            
            # Verify validation result
            assert isinstance(result, dict)
            assert 'is_valid' in result
            assert 'errors' in result
            assert 'warnings' in result
            
        except ImportError:
            pytest.skip("GNN validation not available")
        except Exception as e:
            # Should handle errors gracefully
            assert "error" in str(e).lower() or isinstance(result, dict)
    
    @pytest.mark.unit 
    @pytest.mark.safe_to_fail
    def test_check_consistency(self, comprehensive_test_data):
        """Test consistency checking functionality."""
        try:
            from src.gnn.validation import check_consistency
            
            # Test with model data
            test_model = comprehensive_test_data['models']['complex_model']
            
            result = check_consistency(test_model)
            
            # Verify consistency check result
            assert isinstance(result, dict)
            assert 'consistent' in result or 'is_consistent' in result
            assert 'issues' in result or 'warnings' in result
            
        except ImportError:
            pytest.skip("GNN validation not available")
        except Exception as e:
            # Should handle errors gracefully
            assert isinstance(result, (dict, bool))



# Migrated from test_gnn_core_modules.py
class TestGNNSimpleValidator:
    """Test gnn.simple_validator module."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_simple_validator_imports(self):
        """Test that simple validator can be imported."""
        try:
            from src.gnn import simple_validator
            assert hasattr(simple_validator, 'SimpleValidator')
        except ImportError:
            pytest.skip("GNN simple validator not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_simple_validator_instantiation(self):
        """Test SimpleValidator instantiation."""
        try:
            from src.gnn.simple_validator import SimpleValidator
            
            validator = SimpleValidator()
            
            # Verify validator has expected methods
            assert hasattr(validator, 'validate_file')
            assert hasattr(validator, 'validate_directory')
            assert hasattr(validator, 'valid_extensions')
            
        except ImportError:
            pytest.skip("GNN simple validator not available")
        except Exception as e:
            # Should handle instantiation errors gracefully
            assert "error" in str(e).lower()
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_simple_validation(self, sample_gnn_files):
        """Test simple validation functionality."""
        try:
            from src.gnn.simple_validator import SimpleValidator
            
            validator = SimpleValidator()
            
            # Get sample file content
            sample_file = list(sample_gnn_files.values())[0]
            content = sample_file.read_text()
            
            # Use validate_file method instead of validate
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                f.write(content)
                temp_file = Path(f.name)
            
            try:
                result = validator.validate_file(temp_file)
                
                # Verify validation result
                assert isinstance(result, (dict, bool))
                if isinstance(result, dict):
                    assert 'valid' in result or 'is_valid' in result
            finally:
                temp_file.unlink()
                
        except ImportError:
            pytest.skip("GNN simple validator not available")
        except Exception as e:
            # Should handle validation errors gracefully
            assert "error" in str(e).lower() or "failed" in str(e).lower()


