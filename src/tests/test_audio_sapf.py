#!/usr/bin/env python3
"""
Test Audio Sapf Tests

This file contains tests migrated from test_sapf.py.
"""

import pytest
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import *


# Migrated from test_sapf.py
class TestSAPFCodeValidation:
    """Test SAPF code validation functionality."""
    
    @pytest.mark.unit
    def test_validate_sapf_code_valid(self):
        """Test validation of valid SAPF code."""
        try:
            from sapf.sapf_gnn_processor import validate_sapf_code
        except ImportError:
            pytest.skip("SAPF validation not available")
        
        valid_code = """
; Valid SAPF code
261.63 = base_freq
base_freq 0 sinosc 0.3 * = osc1
10 sec 0 1 0.8 0.2 env = env1
osc1 env1 * = final_audio
final_audio play
"""
        
        is_valid, issues = validate_sapf_code(valid_code)
        assert is_valid is True
        assert len(issues) == 0
    
    @pytest.mark.unit
    def test_validate_sapf_code_invalid(self):
        """Test validation of invalid SAPF code."""
        try:
            from sapf.sapf_gnn_processor import validate_sapf_code
        except ImportError:
            pytest.skip("SAPF validation not available")
        
        # Test empty code
        is_valid, issues = validate_sapf_code("")
        assert is_valid is False
        assert len(issues) > 0
        assert "Empty SAPF code" in issues[0]
        
        # Test unbalanced brackets
        invalid_code = """
261.63 = base_freq
[ base_freq 0 sinosc 0.3 *
final_audio play
"""
        is_valid, issues = validate_sapf_code(invalid_code)
        assert is_valid is False
        assert any("Unbalanced brackets" in issue for issue in issues)
        
        # Test missing play command
        no_play_code = """
261.63 = base_freq
base_freq 0 sinosc 0.3 * = final_audio
"""
        is_valid, issues = validate_sapf_code(no_play_code)
        assert is_valid is False
        assert any("No 'play' command found" in issue for issue in issues)



# Migrated from test_sapf.py
class TestSAPFStandaloneFunctions:
    """Test standalone SAPF functions."""
    
    @pytest.mark.unit
    def test_convert_gnn_to_sapf_function(self):
        """Test standalone convert_gnn_to_sapf function."""
        try:
            from sapf.sapf_gnn_processor import convert_gnn_to_sapf
        except ImportError:
            pytest.skip("SAPF conversion function not available")
        
        gnn_content = """
## ModelName
TestModel

## StateSpaceBlock
s_f0[2,type=continuous]

## Connections
s_f0 > s_f0

## Time
Static
"""
        
        sapf_code = convert_gnn_to_sapf(gnn_content, "TestModel")
        
        assert isinstance(sapf_code, str)
        assert len(sapf_code) > 0
        assert "TestModel" in sapf_code
        assert "base_freq" in sapf_code
    
    @pytest.mark.unit
    def test_generate_oscillator_audio_function(self):
        """Test standalone generate_oscillator_audio function."""
        try:
            from sapf.audio_generators import generate_oscillator_audio
        except ImportError:
            pytest.skip("SAPF oscillator function not available")
        
        audio = generate_oscillator_audio(440.0, 0.5, 1.0, 44100, 'sine')
        
        assert isinstance(audio, np.ndarray)
        assert len(audio) == 44100  # 1 second at 44.1kHz
        assert audio.max() <= 0.5
        assert audio.min() >= -0.5
    
    @pytest.mark.unit
    def test_apply_envelope_function(self):
        """Test standalone apply_envelope function."""
        try:
            from sapf.audio_generators import apply_envelope
        except ImportError:
            pytest.skip("SAPF envelope function not available")
        
        # Create test audio
        test_audio = np.ones(44100, dtype=np.float32)  # 1 second of constant signal
        
        # Apply envelope
        enveloped = apply_envelope(test_audio, 0.1, 0.1, 0.7, 0.2, 44100)
        
        assert isinstance(enveloped, np.ndarray)
        assert len(enveloped) == len(test_audio)
        assert enveloped[0] == 0.0  # Should start at 0
        assert enveloped[-1] <= 0.1  # Should end near 0
    
    @pytest.mark.unit
    def test_mix_audio_channels_function(self):
        """Test standalone mix_audio_channels function."""
        try:
            from sapf.audio_generators import mix_audio_channels
        except ImportError:
            pytest.skip("SAPF mixing function not available")
        
        # Create test audio channels
        channel1 = np.ones(1000) * 0.5
        channel2 = np.ones(1000) * 0.3
        channel3 = np.ones(1000) * 0.2
        
        # Test mixing without weights
        mixed = mix_audio_channels([channel1, channel2, channel3])
        assert isinstance(mixed, np.ndarray)
        assert len(mixed) == 1000
        
        # Test mixing with weights
        weights = [0.5, 0.3, 0.2]
        mixed_weighted = mix_audio_channels([channel1, channel2, channel3], weights)
        assert isinstance(mixed_weighted, np.ndarray)
        assert len(mixed_weighted) == 1000
        
        # Test empty input
        empty_mixed = mix_audio_channels([])
        assert len(empty_mixed) == 0


