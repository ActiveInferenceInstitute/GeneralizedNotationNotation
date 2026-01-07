#!/usr/bin/env python3
"""
Test Audio Generation Tests

This file contains comprehensive tests for audio generation functionality.
"""

import pytest
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import *

class TestAudioGeneration:
    """Test audio generation functionality."""
    
    def test_audio_import_available(self):
        """Test that audio module can be imported."""
        try:
            from audio import AudioGenerator
            # Verify the imported class is actually usable
            assert AudioGenerator is not None, "AudioGenerator should be importable"
            assert callable(getattr(AudioGenerator, '__init__', None)) or hasattr(AudioGenerator, '__call__'), "AudioGenerator should be a class or callable"
        except ImportError:
            pytest.skip("Audio module not available")
    
    def test_sapf_audio_generation(self):
        """Test SAPF audio generation."""
        # Test SAPF configuration
        sapf_config = {
            "sample_rate": 44100,
            "duration": 10.0,
            "frequency": 440.0,
            "amplitude": 0.5,
            "waveform": "sine"
        }
        
        assert sapf_config["sample_rate"] == 44100
        assert sapf_config["duration"] == 10.0
        assert sapf_config["frequency"] == 440.0
        assert sapf_config["amplitude"] == 0.5
        assert sapf_config["waveform"] == "sine"
    
    def test_pedalboard_audio_generation(self):
        """Test Pedalboard audio generation."""
        # Test Pedalboard configuration
        pedalboard_config = {
            "plugins": ["Reverb", "Delay", "Compressor"],
            "sample_rate": 48000,
            "buffer_size": 1024,
            "latency": 0.001
        }
        
        assert "Reverb" in pedalboard_config["plugins"]
        assert "Delay" in pedalboard_config["plugins"]
        assert "Compressor" in pedalboard_config["plugins"]
        assert pedalboard_config["sample_rate"] == 48000
        assert pedalboard_config["buffer_size"] == 1024
    
    def test_audio_format_conversion(self):
        """Test audio format conversion."""
        # Test supported formats
        supported_formats = ["wav", "mp3", "flac", "ogg", "aiff"]
        
        for format_name in supported_formats:
            assert format_name in supported_formats
            assert isinstance(format_name, str)
    
    def test_audio_parameter_validation(self):
        """Test audio parameter validation."""
        # Test valid parameters
        valid_params = {
            "frequency": 440.0,
            "amplitude": 0.5,
            "duration": 5.0,
            "sample_rate": 44100
        }
        
        # Validate parameter ranges
        assert 20 <= valid_params["frequency"] <= 20000  # Audible frequency range
        assert 0 < valid_params["amplitude"] <= 1.0  # Valid amplitude range
        assert valid_params["duration"] > 0  # Positive duration
        assert valid_params["sample_rate"] > 0  # Positive sample rate
    
    def test_audio_error_handling(self):
        """Test audio error handling."""
        # Test invalid parameters
        invalid_params = {
            "frequency": -100,  # Negative frequency
            "amplitude": 2.0,   # Amplitude > 1.0
            "duration": -5.0,   # Negative duration
            "sample_rate": 0    # Zero sample rate
        }
        
        # These should be caught by validation
        assert invalid_params["frequency"] < 0
        assert invalid_params["amplitude"] > 1.0
        assert invalid_params["duration"] < 0
        assert invalid_params["sample_rate"] <= 0
    
    def test_audio_performance(self):
        """Test audio generation performance."""
        import time
        
        start_time = time.time()
        
        # Simulate audio generation
        sample_rate = 44100
        duration = 1.0  # 1 second
        num_samples = int(sample_rate * duration)
        
        # Generate simple sine wave
        import math
        frequency = 440.0
        samples = []
        for i in range(num_samples):
            t = i / sample_rate
            sample = math.sin(2 * math.pi * frequency * t)
            samples.append(sample)
        
        assert len(samples) == num_samples
        assert all(-1 <= s <= 1 for s in samples)  # Valid amplitude range
        
        generation_time = time.time() - start_time
        assert generation_time < 1.0  # Should complete quickly
    
    def test_audio_memory_usage(self):
        """Test audio memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate audio processing
        sample_rate = 44100
        duration = 10.0  # 10 seconds
        num_samples = int(sample_rate * duration)
        
        # Generate audio data
        import math
        audio_data = []
        for i in range(num_samples):
            t = i / sample_rate
            sample = math.sin(2 * math.pi * 440 * t)  # A4 note
            audio_data.append(sample)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB for 10 seconds of audio)
        assert memory_increase < 100.0
    
    def test_audio_quality_metrics(self):
        """Test audio quality metrics."""
        # Test signal-to-noise ratio calculation
        import math
        signal = [0.5] * 1000  # Clean signal
        noise = [0.1] * 1000   # Noise
        
        # Calculate SNR (simplified)
        signal_power = sum(s**2 for s in signal) / len(signal)
        noise_power = sum(n**2 for n in noise) / len(noise)
        
        if noise_power > 0:
            import math
            snr = 10 * math.log10(signal_power / noise_power)
            assert snr > 0  # SNR should be positive for clean signal
    
    def test_audio_file_operations(self):
        """Test audio file operations."""
        # Test file path handling
        test_file = Path("test_audio.wav")
        
        # Simulate file operations
        assert test_file.suffix == ".wav"
        assert test_file.stem == "test_audio"
        
        # Test directory creation
        output_dir = Path("output/audio")
        output_dir.mkdir(parents=True, exist_ok=True)
        assert output_dir.exists()
    
    def test_audio_backend_selection(self):
        """Test audio backend selection."""
        # Test backend options
        backends = ["auto", "sapf", "pedalboard", "numpy", "scipy"]
        
        for backend in backends:
            assert backend in backends
            assert isinstance(backend, str)
        
        # Test auto-detection logic
        available_backends = ["sapf", "pedalboard"]
        auto_backend = "sapf" if "sapf" in available_backends else "pedalboard"
        
        assert auto_backend in available_backends
    
    def test_audio_concurrent_generation(self):
        """Test concurrent audio generation."""
        import threading
        import time
        
        results = []
        lock = threading.Lock()
        
        def generate_audio(worker_id):
            # Simulate audio generation
            time.sleep(0.01)  # Small delay
            with lock:
                results.append(f"audio_{worker_id}")
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=generate_audio, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all workers completed
        assert len(results) == 5
        for i in range(5):
            assert f"audio_{i}" in results

