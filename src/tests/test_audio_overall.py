#!/usr/bin/env python3
"""
Test Audio Overall Tests

This file contains comprehensive tests for the audio module functionality.
"""

import pytest
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import *


class TestAudioModuleComprehensive:
    """Comprehensive tests for the audio module."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_audio_module_imports(self):
        """Test that audio module can be imported."""
        try:
            import audio
            assert hasattr(audio, '__version__')
            assert hasattr(audio, 'AudioGenerator')
            assert hasattr(audio, 'SAPFProcessor')
            assert hasattr(audio, 'PedalboardProcessor')
        except ImportError:
            pytest.skip("Audio module not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_audio_generator_instantiation(self):
        """Test AudioGenerator class instantiation."""
        try:
            from audio import AudioGenerator
            generator = AudioGenerator()
            assert generator is not None
            assert hasattr(generator, 'generate_audio')
            assert hasattr(generator, 'process_gnn_to_audio')
        except ImportError:
            pytest.skip("AudioGenerator not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_sapf_processor_instantiation(self):
        """Test SAPFProcessor class instantiation."""
        try:
            from audio import SAPFProcessor
            processor = SAPFProcessor()
            assert processor is not None
            assert hasattr(processor, 'convert_gnn_to_sapf')
            assert hasattr(processor, 'generate_audio')
        except ImportError:
            pytest.skip("SAPFProcessor not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_pedalboard_processor_instantiation(self):
        """Test PedalboardProcessor class instantiation."""
        try:
            from audio import PedalboardProcessor
            processor = PedalboardProcessor()
            assert processor is not None
            assert hasattr(processor, 'process_audio')
            assert hasattr(processor, 'apply_effects')
        except ImportError:
            pytest.skip("PedalboardProcessor not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_audio_module_info(self):
        """Test audio module information retrieval."""
        try:
            from audio import get_module_info
            info = get_module_info()
            assert isinstance(info, dict)
            assert 'version' in info
            assert 'description' in info
            assert 'features' in info
        except ImportError:
            pytest.skip("Audio module info not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_audio_generation_options(self):
        """Test audio generation options retrieval."""
        try:
            from audio import get_audio_generation_options
            options = get_audio_generation_options()
            assert isinstance(options, dict)
            assert 'formats' in options
            assert 'effects' in options
            assert 'backends' in options
        except ImportError:
            pytest.skip("Audio generation options not available")


class TestAudioProcessing:
    """Tests for audio processing functionality."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_gnn_to_audio_conversion(self, sample_gnn_files):
        """Test GNN to audio conversion."""
        try:
            from audio import AudioGenerator
            generator = AudioGenerator()
            
            # Test with sample GNN content
            gnn_content = "test GNN content"
            result = generator.process_gnn_to_audio(gnn_content)
            assert result is not None
        except ImportError:
            pytest.skip("AudioGenerator not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_audio_validation(self):
        """Test audio validation functionality."""
        try:
            from audio import validate_audio_content
            result = validate_audio_content("test audio content")
            assert isinstance(result, bool)
        except ImportError:
            pytest.skip("Audio validation not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_audio_generation(self, safe_filesystem):
        """Test audio generation functionality."""
        try:
            from audio import generate_audio_from_gnn
            
            # Create test GNN content and output directory
            output_dir = safe_filesystem.create_dir("audio_output")
            
            result = generate_audio_from_gnn("test GNN content\nwith variables", output_dir=output_dir)
            assert result is not None
            assert "audio_files" in result
        except ImportError:
            pytest.skip("Audio generation not available")


class TestAudioIntegration:
    """Integration tests for audio module."""
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_audio_pipeline_integration(self, sample_gnn_files, isolated_temp_dir):
        """Test audio module integration with pipeline."""
        try:
            from audio import AudioGenerator
            generator = AudioGenerator()
            
            # Test end-to-end audio generation
            gnn_file = list(sample_gnn_files.values())[0]
            with open(gnn_file, 'r') as f:
                gnn_content = f.read()
            
            result = generator.process_gnn_to_audio(gnn_content)
            assert result is not None
            
        except ImportError:
            pytest.skip("Audio module not available")
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_audio_mcp_integration(self):
        """Test audio MCP integration."""
        try:
            from audio.mcp import register_tools
            # Test that MCP tools can be registered
            assert callable(register_tools)
        except ImportError:
            pytest.skip("Audio MCP not available")


def test_audio_module_completeness():
    """Test that audio module has all required components."""
    required_components = [
        'AudioGenerator',
        'SAPFProcessor', 
        'PedalboardProcessor',
        'get_module_info',
        'get_audio_generation_options'
    ]
    
    try:
        import audio
        for component in required_components:
            assert hasattr(audio, component), f"Missing component: {component}"
    except ImportError:
        pytest.skip("Audio module not available")


@pytest.mark.slow
def test_audio_module_performance():
    """Test audio module performance characteristics."""
    try:
        from audio import AudioGenerator
        import time
        
        generator = AudioGenerator()
        start_time = time.time()
        
        # Test processing time
        result = generator.process_gnn_to_audio("test content")
        
        processing_time = time.time() - start_time
        assert processing_time < 10.0  # Should complete within 10 seconds
        
    except ImportError:
        pytest.skip("Audio module not available")

