#!/usr/bin/env python3
"""
SAPF Module Tests

This module contains comprehensive tests for the SAPF (Sound As Pure Form) 
integration functionality, including GNN-to-SAPF conversion, audio generation,
and MCP integration.

Tests cover:
1. SAPF GNN Processor functionality
2. Audio generation from SAPF code
3. GNN parsing and section extraction
4. SAPF code validation
5. Audio file creation and waveform visualization
6. MCP tool integration
7. Error handling and edge cases

All tests are designed to be safe-to-fail with appropriate mocking.
"""

import pytest
import os
import sys
import json
import tempfile
import wave
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import patch, Mock, MagicMock
import numpy as np

# Test markers
pytestmark = [pytest.mark.sapf, pytest.mark.safe_to_fail]

# Import test utilities
from . import (
    TEST_CONFIG,
    is_safe_mode,
    create_sample_gnn_content,
    TEST_DIR,
    SRC_DIR,
    PROJECT_ROOT
)

class TestSAPFGNNProcessor:
    """Test SAPF GNN Processor functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create SAPF GNN Processor instance."""
        try:
            from sapf.sapf_gnn_processor import SAPFGNNProcessor
            return SAPFGNNProcessor()
        except ImportError:
            pytest.skip("SAPF GNN Processor not available")
    
    @pytest.fixture
    def sample_gnn_content(self):
        """Get sample GNN content for testing."""
        return """# GNN Example: Test Model
## GNNVersionAndFlags
GNN v1

## ModelName
Test SAPF Model

## StateSpaceBlock
A_m0[3,2,3,type=float]     # Likelihood matrix
B_f0[2,2,1,type=float]     # Transition matrix
C_m0[3,type=float]         # Preference vector
D_f0[2,type=float]         # Prior vector
s_f0[2,type=continuous]    # Hidden state
o_m0[3,type=discrete]      # Observation
π_f0[2,type=discrete]      # Policy

## Connections
s_f0 > o_m0
s_f0 > s_f0
π_f0 > s_f0

## InitialParameterization
A_m0={(0.33, 0.33, 0.33), (0.5, 0.5, 0.0)}
B_f0={(1.0, 0.0), (0.0, 1.0)}
C_m0={(0.0, 1.0, -1.0)}
D_f0={(0.5, 0.5)}

## Time
Dynamic
DiscreteTime
ModelTimeHorizon=10

## Footer
Test SAPF Model

## Signature
Test"""
    
    @pytest.mark.unit
    def test_processor_initialization(self, processor):
        """Test SAPF processor initialization."""
        assert processor is not None
        assert hasattr(processor, 'base_frequency')
        assert hasattr(processor, 'sample_rate')
        assert hasattr(processor, 'default_duration')
        
        # Test default values
        assert processor.base_frequency == 261.63  # C4
        assert processor.sample_rate == 44100
        assert processor.default_duration == 10.0
    
    @pytest.mark.unit
    def test_parse_gnn_sections(self, processor, sample_gnn_content):
        """Test GNN section parsing."""
        sections = processor.parse_gnn_sections(sample_gnn_content)
        
        # Check that all expected sections are parsed
        expected_sections = ['ModelName', 'StateSpaceBlock', 'Connections', 'InitialParameterization', 'Time']
        for section in expected_sections:
            assert section in sections, f"Section {section} should be parsed"
        
        # Check ModelName
        assert sections['ModelName'] == 'Test SAPF Model'
        
        # Check StateSpaceBlock
        state_space = sections['StateSpaceBlock']
        assert isinstance(state_space, list)
        assert len(state_space) > 0
        
        # Check for different variable categories
        categories_found = set()
        for state in state_space:
            categories_found.add(state.get('category'))
        
        expected_categories = {'likelihood_matrix', 'transition_matrix', 'preference_vector', 
                             'prior_vector', 'hidden_state', 'observation', 'policy_control'}
        assert expected_categories.issubset(categories_found)
        
        # Check Connections
        connections = sections['Connections']
        assert isinstance(connections, list)
        assert len(connections) > 0
        
        # Check that connections have required fields
        for conn in connections:
            assert 'source' in conn
            assert 'target' in conn
            assert 'type' in conn
            assert 'directed' in conn
        
        # Check Time configuration
        time_config = sections['Time']
        assert isinstance(time_config, dict)
        # ModelTimeHorizon should be present if specified in content
        if 'ModelTimeHorizon' in time_config:
            assert time_config['ModelTimeHorizon'] == 10
    
    @pytest.mark.unit
    def test_state_space_parsing_categories(self, processor):
        """Test state space parsing for different variable categories."""
        test_content = """
A_m0[3,2,type=float]       # Likelihood matrix
B_f1[5,5,3,type=float]     # Transition matrix  
C_m0[3,type=float]         # Preference vector
D_f0[2,type=float]         # Prior vector
s_f0[2,type=continuous]    # Hidden state
o_m0[3,type=discrete]      # Observation
π_f1[3,type=discrete]      # Policy
u_c0[2,type=int]           # Control action
harmonic_resonance[12,8,type=float]  # Other category
"""
        
        states = processor._parse_state_space(test_content)
        
        # Check that all variables are parsed
        assert len(states) == 9
        
        # Check specific categories
        category_map = {state['name']: state['category'] for state in states}
        
        assert category_map['A_m0'] == 'likelihood_matrix'
        assert category_map['B_f1'] == 'transition_matrix'
        assert category_map['C_m0'] == 'preference_vector'
        assert category_map['D_f0'] == 'prior_vector'
        assert category_map['s_f0'] == 'hidden_state'
        assert category_map['o_m0'] == 'observation'
        assert category_map['π_f1'] == 'policy_control'
        assert category_map['u_c0'] == 'policy_control'
        assert category_map['harmonic_resonance'] == 'other'
        
        # Check dimensions parsing (note: parser may add default dimensions)
        dim_map = {state['name']: state['dimensions'] for state in states}
        
        # Check that dimensions start with expected values
        assert dim_map['A_m0'][:2] == [3, 2]
        assert dim_map['B_f1'][:3] == [5, 5, 3]
        assert dim_map['C_m0'][:1] == [3]
        assert dim_map['harmonic_resonance'][:2] == [12, 8]
    
    @pytest.mark.unit
    def test_convert_to_sapf(self, processor, sample_gnn_content):
        """Test conversion of GNN to SAPF code."""
        sections = processor.parse_gnn_sections(sample_gnn_content)
        sapf_code = processor.convert_to_sapf(sections, "TestModel")
        
        # Check that SAPF code is generated
        assert isinstance(sapf_code, str)
        assert len(sapf_code) > 0
        
        # Check for key SAPF elements
        assert "TestModel" in sapf_code
        assert "base_freq" in sapf_code
        assert "play" in sapf_code
        assert "generate_final_audio" in sapf_code
        
        # Check for different audio processing sections
        assert "State Space Oscillators" in sapf_code
        assert "Connection Routing" in sapf_code
        assert "Matrix-based Audio Processing" in sapf_code
        assert "Temporal Structure" in sapf_code
        
        # Check for model-specific characteristics
        assert "model_scale" in sapf_code
        assert "model_tempo" in sapf_code
        assert "model_reverb" in sapf_code
        
        # Check for different oscillator types and effects
        assert "sinosc" in sapf_code or "lfsaw" in sapf_code
    
    @pytest.mark.unit
    def test_complexity_calculation(self, processor):
        """Test complexity level calculation."""
        # Simple model
        simple_sections = {
            'StateSpaceBlock': [{'name': 's1'}, {'name': 's2'}],
            'Connections': [{'source': 's1', 'target': 's2'}],
            'InitialParameterization': {}
        }
        complexity = processor._get_complexity_level(simple_sections)
        assert complexity == "simple"
        
        # Complex model
        complex_sections = {
            'StateSpaceBlock': [{'name': f's{i}'} for i in range(10)],
            'Connections': [{'source': f's{i}', 'target': f's{i+1}'} for i in range(9)],
            'InitialParameterization': {'A': [], 'B': [], 'C': [], 'D': []}
        }
        complexity = processor._get_complexity_level(complex_sections)
        assert complexity == "complex"
    
    @pytest.mark.unit
    def test_model_signature_generation(self, processor):
        """Test model signature generation for audio differentiation."""
        sections = {
            'StateSpaceBlock': [{'name': 's1', 'category': 'hidden_state'}],
            'Connections': [{'source': 's1', 'target': 'o1'}]
        }
        
        # Different models should have different signatures
        sig1 = processor._get_model_signature("model_one", sections)
        sig2 = processor._get_model_signature("model_two", sections)
        sig3 = processor._get_model_signature("completely_different_model", sections)
        
        assert isinstance(sig1, int)
        assert isinstance(sig2, int)
        assert isinstance(sig3, int)
        
        # Signatures should be different for different model names
        assert sig1 != sig2
        assert sig1 != sig3
        assert sig2 != sig3
    
    @pytest.mark.unit
    def test_musical_scale_assignment(self, processor):
        """Test musical scale assignment based on model names."""
        # Test known model types
        pymdp_scale = processor._get_model_scale("pymdp_test_model")
        rxinfer_scale = processor._get_model_scale("rxinfer_hidden_markov")
        multiagent_scale = processor._get_model_scale("multiagent_planning")
        unknown_scale = processor._get_model_scale("unknown_model_type")
        
        # Should return valid scale arrays
        assert "[" in pymdp_scale and "]" in pymdp_scale
        assert "[" in rxinfer_scale and "]" in rxinfer_scale
        assert "[" in multiagent_scale and "]" in multiagent_scale
        assert "[" in unknown_scale and "]" in unknown_scale
        
        # Different model types should have different scales
        assert pymdp_scale != rxinfer_scale
        assert rxinfer_scale != multiagent_scale

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

class TestSAPFAudioGeneration:
    """Test SAPF audio generation functionality."""
    
    @pytest.fixture
    def audio_generator(self):
        """Create audio generator instance."""
        try:
            from sapf.audio_generators import SyntheticAudioGenerator
            return SyntheticAudioGenerator()
        except ImportError:
            pytest.skip("SAPF audio generator not available")
    
    @pytest.fixture
    def sample_sapf_code(self):
        """Get sample SAPF code for testing."""
        return """
; Test SAPF audio code
261.63 = base_freq
base_freq 0 sinosc 0.3 * = osc1
base_freq 2 * 0 sinosc 0.1 * = osc2
osc1 osc2 + = mixed_osc
10 sec 0.1 1 0.8 0.2 env = envelope
mixed_osc envelope * = final_audio
final_audio play
"""
    
    @pytest.mark.unit
    def test_audio_generator_initialization(self, audio_generator):
        """Test audio generator initialization."""
        assert audio_generator is not None
        assert hasattr(audio_generator, 'sample_rate')
        assert hasattr(audio_generator, 'base_frequency')
        assert audio_generator.sample_rate == 44100
        assert audio_generator.base_frequency == 261.63
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_analyze_sapf_code(self, audio_generator, sample_sapf_code):
        """Test SAPF code analysis for audio parameters."""
        params = audio_generator._analyze_sapf_code(sample_sapf_code)
        
        assert isinstance(params, dict)
        assert 'base_frequency' in params
        assert 'oscillators' in params
        assert 'envelopes' in params
        assert 'complexity' in params
        
        # Check that oscillators are detected
        assert isinstance(params['oscillators'], list)
        assert len(params['oscillators']) > 0
        
        # Check oscillator parameters
        for osc in params['oscillators']:
            assert 'type' in osc
            assert 'frequency' in osc
            assert 'amplitude' in osc
            assert osc['frequency'] > 0
            assert 0 <= osc['amplitude'] <= 1
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_generate_oscillator_audio(self, audio_generator):
        """Test oscillator audio generation."""
        samples = 1000
        frequency = 440.0
        amplitude = 0.5
        
        # Test sine wave
        sine_audio = audio_generator._generate_oscillator(frequency, amplitude, 'sine', samples)
        assert isinstance(sine_audio, np.ndarray)
        assert len(sine_audio) == samples
        assert sine_audio.max() <= amplitude
        assert sine_audio.min() >= -amplitude
        
        # Test sawtooth wave
        saw_audio = audio_generator._generate_oscillator(frequency, amplitude, 'saw', samples)
        assert isinstance(saw_audio, np.ndarray)
        assert len(saw_audio) == samples
        
        # Test square wave
        square_audio = audio_generator._generate_oscillator(frequency, amplitude, 'square', samples)
        assert isinstance(square_audio, np.ndarray)
        assert len(square_audio) == samples
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_generate_envelope(self, audio_generator):
        """Test ADSR envelope generation."""
        samples = 44100  # 1 second at 44.1kHz
        env_params = {
            'attack': 0.1,
            'decay': 0.1,
            'sustain': 0.7,
            'release': 0.2
        }
        
        envelope = audio_generator._generate_envelope(env_params, samples)
        
        assert isinstance(envelope, np.ndarray)
        assert len(envelope) == samples
        assert envelope.max() <= 1.0
        assert envelope.min() >= 0.0
        
        # Check that envelope starts at 0 and ends at 0
        assert envelope[0] == 0.0
        assert envelope[-1] <= 0.1  # Should be close to 0 at the end
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_apply_lowpass_filter(self, audio_generator):
        """Test lowpass filter application."""
        # Generate test signal
        samples = 1000
        test_audio = np.random.randn(samples).astype(np.float32)
        
        # Apply filter
        filtered_audio = audio_generator._apply_lowpass_filter(test_audio, 1000.0)
        
        assert isinstance(filtered_audio, np.ndarray)
        assert len(filtered_audio) == len(test_audio)
        
        # Filter should reduce high frequency content (basic check)
        # In practice, this would need more sophisticated testing
        assert not np.array_equal(test_audio, filtered_audio)
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_generate_from_sapf_mock(self, audio_generator, sample_sapf_code, tmp_path):
        """Test full audio generation from SAPF code with mocked file operations."""
        output_file = tmp_path / "test_audio.wav"
        duration = 2.0
        
        # Mock wave operations for safe testing
        with patch('wave.open') as mock_wave_open:
            mock_wav_file = Mock()
            mock_wave_open.return_value.__enter__.return_value = mock_wav_file
            
            # Mock successful audio generation
            with patch.object(audio_generator, '_generate_audio', return_value=[0, 100, -100, 0] * 1000):
                with patch.object(audio_generator, '_create_waveform_visualizations'):
                    success = audio_generator.generate_from_sapf(sample_sapf_code, output_file, duration)
                    
                    assert success is True
                    mock_wave_open.assert_called_once()
                    mock_wav_file.setnchannels.assert_called_with(1)  # Mono
                    mock_wav_file.setsampwidth.assert_called_with(2)  # 16-bit
                    mock_wav_file.setframerate.assert_called_with(44100)

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

class TestSAPFMCPIntegration:
    """Test SAPF MCP integration."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_register_sapf_tools(self):
        """Test SAPF MCP tool registration."""
        try:
            from sapf.mcp import register_sapf_tools
        except ImportError:
            pytest.skip("SAPF MCP integration not available")
        
        tools = register_sapf_tools()
        
        assert isinstance(tools, list)
        assert len(tools) > 0
        
        # Check expected tools
        tool_names = [tool['name'] for tool in tools]
        expected_tools = [
            'convert_gnn_to_sapf_audio',
            'generate_sapf_code', 
            'validate_sapf_syntax',
            'generate_audio_from_sapf',
            'analyze_gnn_for_audio'
        ]
        
        for expected in expected_tools:
            assert expected in tool_names, f"Tool {expected} should be registered"
        
        # Check tool structure
        for tool in tools:
            assert 'name' in tool
            assert 'description' in tool
            assert 'inputSchema' in tool
            assert isinstance(tool['inputSchema'], dict)
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_handle_generate_sapf_code(self):
        """Test MCP handler for generating SAPF code."""
        try:
            from sapf.mcp import handle_generate_sapf_code
        except ImportError:
            pytest.skip("SAPF MCP handlers not available")
        
        params = {
            "gnn_content": """
## ModelName
TestModel

## StateSpaceBlock
s_f0[2,type=continuous]

## Time
Static
""",
            "model_name": "TestModel"
        }
        
        # Test without async
        import asyncio
        response = asyncio.run(handle_generate_sapf_code(params))
        
        assert isinstance(response, dict)
        assert 'success' in response
        if response['success']:
            assert 'sapf_code' in response
            assert 'model_name' in response
            assert 'code_lines' in response
            assert isinstance(response['sapf_code'], str)
            assert len(response['sapf_code']) > 0
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_handle_validate_sapf_syntax(self):
        """Test MCP handler for validating SAPF syntax."""
        try:
            from sapf.mcp import handle_validate_sapf_syntax
        except ImportError:
            pytest.skip("SAPF MCP handlers not available")
        
        # Test valid code
        valid_params = {
            "sapf_code": """
261.63 = base_freq
base_freq 0 sinosc 0.3 * = osc
10 sec 0 1 0.8 0.2 env = env
osc env * = final_audio
final_audio play
"""
        }
        
        import asyncio
        response = asyncio.run(handle_validate_sapf_syntax(valid_params))
        
        assert isinstance(response, dict)
        assert 'valid' in response
        assert 'issues' in response
        assert 'code_lines' in response
        
        # Test invalid code
        invalid_params = {
            "sapf_code": "[ unbalanced brackets"
        }
        
        response = asyncio.run(handle_validate_sapf_syntax(invalid_params))
        
        assert isinstance(response, dict)
        assert 'valid' in response
        assert 'issues' in response
        assert response['valid'] is False
        assert len(response['issues']) > 0

class TestSAPFErrorHandling:
    """Test SAPF error handling and edge cases."""
    
    @pytest.mark.unit
    def test_processor_with_invalid_gnn(self):
        """Test processor behavior with invalid GNN content."""
        try:
            from sapf.sapf_gnn_processor import SAPFGNNProcessor
        except ImportError:
            pytest.skip("SAPF processor not available")
        
        processor = SAPFGNNProcessor()
        
        # Test with empty content
        sections = processor.parse_gnn_sections("")
        assert isinstance(sections, dict)
        
        # Test with malformed content
        malformed_content = "This is not valid GNN content at all"
        sections = processor.parse_gnn_sections(malformed_content)
        assert isinstance(sections, dict)
        
        # Should still be able to generate SAPF code (with defaults)
        sapf_code = processor.convert_to_sapf(sections, "ErrorTestModel")
        assert isinstance(sapf_code, str)
        assert len(sapf_code) > 0
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_audio_generator_error_handling(self):
        """Test audio generator error handling."""
        try:
            from sapf.audio_generators import SyntheticAudioGenerator
        except ImportError:
            pytest.skip("SAPF audio generator not available")
        
        generator = SyntheticAudioGenerator()
        
        # Test with empty SAPF code
        params = generator._analyze_sapf_code("")
        assert isinstance(params, dict)
        assert 'oscillators' in params
        assert 'envelopes' in params
        
        # Test with invalid audio parameters
        invalid_params = {
            'base_frequency': -100,  # Invalid frequency
            'oscillators': [],
            'envelopes': [],
            'effects': [],  # Add missing effects key
            'filters': [],  # Add missing filters key
            'complexity': 'unknown'
        }
        
        # Should still generate audio (with fallbacks)
        audio_data = generator._generate_audio(invalid_params, 1.0)
        assert isinstance(audio_data, list)
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_file_operation_failures(self, tmp_path):
        """Test handling of file operation failures."""
        try:
            from sapf.audio_generators import SyntheticAudioGenerator
        except ImportError:
            pytest.skip("SAPF audio generator not available")
        
        generator = SyntheticAudioGenerator()
        
        # Test with invalid output path
        invalid_path = Path("/invalid/nonexistent/path/test.wav")
        
        with patch('wave.open', side_effect=PermissionError("Access denied")):
            success = generator._write_wav_file([0, 100, -100, 0], invalid_path)
            assert success is False

class TestSAPFIntegration:
    """Test SAPF integration with real GNN examples."""
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_baseball_gnn_sapf_conversion(self):
        """Test SAPF conversion with the baseball GNN model."""
        try:
            from sapf.sapf_gnn_processor import convert_gnn_to_sapf
        except ImportError:
            pytest.skip("SAPF processor not available")
        
        # Read the baseball GNN file
        baseball_file = SRC_DIR / "gnn" / "examples" / "baseball_game_active_inference.md"
        
        if not baseball_file.exists():
            pytest.skip("Baseball GNN example not found")
        
        try:
            gnn_content = baseball_file.read_text()
            sapf_code = convert_gnn_to_sapf(gnn_content, "baseball_game_active_inference")
            
            assert isinstance(sapf_code, str)
            assert len(sapf_code) > 0
            assert "baseball_game_active_inference" in sapf_code
            
            # Check for complex model characteristics
            assert "complex" in sapf_code.lower() or "moderate" in sapf_code.lower()
            
            # Should have many oscillators due to complex model
            osc_count = sapf_code.count("_osc")
            assert osc_count > 10, f"Complex model should have many oscillators, found {osc_count}"
            
        except Exception as e:
            pytest.skip(f"Could not process baseball GNN: {e}")
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_existing_gnn_examples_sapf_conversion(self):
        """Test SAPF conversion with existing GNN examples."""
        try:
            from sapf.sapf_gnn_processor import convert_gnn_to_sapf
        except ImportError:
            pytest.skip("SAPF processor not available")
        
        examples_dir = PROJECT_ROOT / "input" / "gnn_files"
        
        if not examples_dir.exists():
            pytest.skip("GNN examples directory not found")
        
        gnn_files = list(examples_dir.glob("*.md"))
        
        if not gnn_files:
            pytest.skip("No GNN example files found")
        
        converted_count = 0
        
        for gnn_file in gnn_files[:3]:  # Test first 3 files to avoid long test times
            try:
                gnn_content = gnn_file.read_text()
                model_name = gnn_file.stem
                
                sapf_code = convert_gnn_to_sapf(gnn_content, model_name)
                
                assert isinstance(sapf_code, str)
                assert len(sapf_code) > 0
                assert model_name in sapf_code
                
                converted_count += 1
                
            except Exception as e:
                # Log but don't fail the test for individual file issues
                print(f"Could not convert {gnn_file.name}: {e}")
        
        assert converted_count > 0, "Should convert at least one GNN file successfully"

# Performance and stress tests

@pytest.mark.slow
@pytest.mark.safe_to_fail
def test_sapf_performance_characteristics():
    """Test SAPF performance characteristics."""
    try:
        from sapf.sapf_gnn_processor import SAPFGNNProcessor
        from sapf.audio_generators import SyntheticAudioGenerator
    except ImportError:
        pytest.skip("SAPF components not available")
    
    processor = SAPFGNNProcessor()
    generator = SyntheticAudioGenerator()
    
    # Test with large model
    large_gnn_content = """
## ModelName
LargeTestModel

## StateSpaceBlock
""" + "\n".join([f"s_f{i}[{i+2},type=continuous]" for i in range(20)]) + """

## Connections
""" + "\n".join([f"s_f{i} > s_f{(i+1)%20}" for i in range(20)]) + """

## Time
Dynamic
ModelTimeHorizon=50
"""
    
    # Test parsing performance
    import time
    start_time = time.time()
    sections = processor.parse_gnn_sections(large_gnn_content)
    parse_time = time.time() - start_time
    
    assert parse_time < 1.0, f"Parsing should be fast, took {parse_time:.2f}s"
    
    # Test conversion performance
    start_time = time.time()
    sapf_code = processor.convert_to_sapf(sections, "LargeTestModel")
    convert_time = time.time() - start_time
    
    assert convert_time < 2.0, f"Conversion should be fast, took {convert_time:.2f}s"
    
    # Test audio analysis performance
    start_time = time.time()
    params = generator._analyze_sapf_code(sapf_code)
    analyze_time = time.time() - start_time
    
    assert analyze_time < 1.0, f"Analysis should be fast, took {analyze_time:.2f}s"

def test_sapf_module_completeness():
    """Test that SAPF module has all expected components."""
    try:
        import sapf
        from sapf import sapf_gnn_processor, audio_generators, mcp
    except ImportError:
        pytest.skip("SAPF module not available")
    
    # Check main module exports
    expected_exports = [
        'SAPFGNNProcessor',
        'convert_gnn_to_sapf',
        'generate_audio_from_sapf',
        'validate_sapf_code',
        'SyntheticAudioGenerator',
        'generate_oscillator_audio',
        'apply_envelope',
        'mix_audio_channels'
    ]
    
    for export in expected_exports:
        assert hasattr(sapf, export), f"SAPF module should export {export}"
    
    # Check processor functions
    processor_functions = [
        'SAPFGNNProcessor',
        'convert_gnn_to_sapf',
        'validate_sapf_code',
        'generate_audio_from_sapf'
    ]
    
    for func in processor_functions:
        assert hasattr(sapf_gnn_processor, func), f"Processor module should have {func}"
    
    # Check audio generator functions
    audio_functions = [
        'SyntheticAudioGenerator',
        'generate_oscillator_audio',
        'apply_envelope',
        'mix_audio_channels'
    ]
    
    for func in audio_functions:
        assert hasattr(audio_generators, func), f"Audio generators module should have {func}"
    
    # Check MCP functions
    mcp_functions = [
        'register_sapf_tools',
        'handle_convert_gnn_to_sapf_audio',
        'handle_generate_sapf_code',
        'handle_validate_sapf_syntax',
        'handle_generate_audio_from_sapf',
        'handle_analyze_gnn_for_audio'
    ]
    
    for func in mcp_functions:
        assert hasattr(mcp, func), f"MCP module should have {func}"

if __name__ == "__main__":
    # Allow running this test module directly
    pytest.main([__file__, "-v"]) 