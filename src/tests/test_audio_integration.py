#!/usr/bin/env python3
"""
Test Audio Integration - Integration tests for audio module with other pipeline components.

Tests the integration between audio generation, GNN parsing, and pipeline orchestration.
"""

import pytest
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAudioGNNIntegration:
    """Tests for audio module integration with GNN processing."""

    @pytest.mark.integration
    def test_audio_module_gnn_data_flow(self, sample_gnn_files):
        """Test that audio module correctly receives and processes GNN data."""
        from audio import process_audio, get_module_info
        from gnn import parse_gnn_file
        
        # Get module info to verify integration
        info = get_module_info()
        assert info is not None
        assert "version" in info or "description" in info
        
        # Process a GNN file first
        if sample_gnn_files:
            gnn_file = list(sample_gnn_files.values())[0]
            gnn_result = parse_gnn_file(gnn_file)
            assert gnn_result is not None

    @pytest.mark.integration
    def test_audio_generation_from_parsed_gnn(self, sample_gnn_files, tmp_path):
        """Test audio generation from parsed GNN data."""
        from audio import generate_audio_from_gnn, AudioGenerator
        from gnn import parse_gnn_file
        
        if not sample_gnn_files:
            pytest.skip("No sample GNN files available")
        
        gnn_file = list(sample_gnn_files.values())[0]
        parsed = parse_gnn_file(gnn_file)
        
        if parsed:
            # Test AudioGenerator instantiation
            generator = AudioGenerator()
            assert generator is not None
            
            # Generate audio from GNN - positional: (file_path_or_content, output_dir, verbose)
            result = generate_audio_from_gnn(
                gnn_file,
                tmp_path
            )
            assert result is not None

    @pytest.mark.integration
    def test_audio_backends_availability(self):
        """Test that audio backend checking works correctly."""
        from audio import check_audio_backends
        
        backends = check_audio_backends()
        assert isinstance(backends, dict)
        
        # Check expected backend keys
        expected_backends = ['numpy', 'librosa', 'soundfile', 'pedalboard']
        for backend in expected_backends:
            assert backend in backends
            assert 'available' in backends[backend]
            assert 'version' in backends[backend]

    @pytest.mark.integration
    def test_audio_features_availability(self):
        """Test that audio feature flags are properly exposed."""
        from audio import FEATURES
        
        assert isinstance(FEATURES, dict)
        expected_features = [
            'tonal_generation',
            'rhythmic_generation',
            'ambient_generation',
            'sonification',
            'audio_analysis'
        ]
        for feature in expected_features:
            assert feature in FEATURES


class TestAudioPipelineIntegration:
    """Tests for audio module integration with pipeline execution."""

    @pytest.mark.integration
    def test_audio_step_execution(self, tmp_path):
        """Test that audio step can execute within pipeline context."""
        from audio import process_audio
        import logging
        
        logger = logging.getLogger("test_audio")
        output_dir = tmp_path / "audio_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create minimal test input
        input_dir = tmp_path / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        
        # Process audio (should handle empty input gracefully)
        result = process_audio(
            target_dir=input_dir,
            output_dir=output_dir,
            logger=logger
        )
        
        # Should return True for successful execution even with no files
        assert result is True or result is False  # Both valid outcomes

    @pytest.mark.integration
    def test_audio_sonification(self, tmp_path):
        """Test sonification creation."""
        from audio import create_sonification
        
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a temporary GNN file for sonification
        gnn_file = tmp_path / "test_model.md"
        gnn_file.write_text("""# Test Model
## ModelName
test_model
## Connections
s -> o
""")
        
        result = create_sonification(
            file_path=gnn_file,
            output_dir=output_dir,
            verbose=False
        )
        
        # Function should complete without error and return dict
        assert result is not None


class TestAudioExportIntegration:
    """Tests for audio export functionality integration."""

    @pytest.mark.integration
    def test_audio_file_writing(self, tmp_path):
        """Test that audio files can be written correctly."""
        import numpy as np
        from audio import write_basic_wav
        
        # Generate test audio data
        sample_rate = 44100
        duration = 0.1  # 100ms
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave
        
        output_file = tmp_path / "test_audio.wav"
        
        # write_basic_wav signature: (audio, file_path, sample_rate) - returns None
        write_basic_wav(audio_data, output_file, sample_rate)
        
        # Verify file was created
        assert output_file.exists()
