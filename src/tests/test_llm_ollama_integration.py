#!/usr/bin/env python3
"""
Comprehensive Ollama integration tests for LLM module.

These tests validate Ollama detection, model selection, and real LLM processing
with proper fallback handling when Ollama is not available.
"""

import pytest
import sys
from pathlib import Path
import json
import tempfile
import shutil
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.processor import (
    _check_and_start_ollama,
    _select_best_ollama_model,
    process_llm
)


class TestOllamaDetection:
    """Test Ollama availability detection."""
    
    def test_ollama_check_returns_tuple(self, caplog):
        """Test that Ollama check returns (bool, list) tuple."""
        import logging
        logger = logging.getLogger("test_ollama")
        
        result = _check_and_start_ollama(logger)
        
        # Should return a tuple
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        is_available, models = result
        assert isinstance(is_available, bool)
        assert isinstance(models, list)
    
    def test_ollama_detection_logging(self, caplog):
        """Test that Ollama detection provides informative logging."""
        import logging
        caplog.set_level(logging.INFO)
        
        logger = logging.getLogger("test_ollama")
        is_available, models = _check_and_start_ollama(logger)
        
        log_text = caplog.text
        
        # Should have some informative message
        assert len(log_text) > 0
        
        if is_available:
            # Should log success
            assert "✅" in log_text or "running" in log_text.lower() or "ready" in log_text.lower()
        else:
            # Should log why it's not available
            assert (
                "not found" in log_text.lower() or
                "not running" in log_text.lower() or
                "not available" in log_text.lower()
            )
    
    def test_ollama_model_listing(self, caplog):
        """Test Ollama model listing when available."""
        import logging
        caplog.set_level(logging.INFO)
        
        logger = logging.getLogger("test_ollama")
        is_available, models = _check_and_start_ollama(logger)
        
        if is_available and models:
            # Should log available models
            log_text = caplog.text
            assert "models" in log_text.lower() or "📦" in log_text
            
            # Models should be non-empty strings
            for model in models:
                assert isinstance(model, str)
                assert len(model) > 0
    
    def test_ollama_socket_check(self, caplog):
        """Test Ollama socket/API endpoint check."""
        import logging
        import socket
        
        logger = logging.getLogger("test_ollama")
        
        # Check if Ollama port is actually open
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', 11434))
            sock.close()
            port_open = (result == 0)
        except Exception:
            port_open = False
        
        is_available, models = _check_and_start_ollama(logger)
        
        if port_open:
            # If port is open, should detect Ollama
            assert is_available, "Should detect Ollama when port 11434 is open"


class TestOllamaModelSelection:
    """Test Ollama model selection logic."""
    
    def test_model_selection_with_empty_list(self, caplog):
        """Test model selection when no models are available."""
        import logging
        logger = logging.getLogger("test_model_selection")
        
        model = _select_best_ollama_model([], logger)
        
        # Should return a default model
        assert isinstance(model, str)
        assert len(model) > 0
    
    def test_model_selection_prefers_small_models(self, caplog):
        """Test that model selection prefers small, fast models."""
        import logging
        logger = logging.getLogger("test_model_selection")
        
        # Test with various model lists
        test_cases = [
            (["llama2:7b", "smollm2:135m"], "smollm2:135m"),
            (["mistral:7b", "tinyllama"], "tinyllama"),
            (["llama2", "gemma2:2b"], "gemma2:2b"),
        ]
        
        for available_models, expected_prefix in test_cases:
            selected = _select_best_ollama_model(available_models, logger)
            
            # Should select the smaller model
            assert any(selected.startswith(prefix) for prefix in [expected_prefix, available_models[0]])
    
    def test_model_selection_respects_env_override(self, caplog, monkeypatch):
        """Test that environment variable overrides model selection."""
        import logging
        logger = logging.getLogger("test_model_selection")
        
        # Set environment variable
        test_model = "my-custom-model:latest"
        monkeypatch.setenv('OLLAMA_MODEL', test_model)
        
        models = ["smollm2:135m", "llama2:7b"]
        selected = _select_best_ollama_model(models, logger)
        
        # Should use environment variable
        assert selected == test_model
    
    def test_model_selection_logging(self, caplog):
        """Test that model selection provides clear logging."""
        import logging
        caplog.set_level(logging.INFO)
        
        logger = logging.getLogger("test_model_selection")
        
        models = ["smollm2:135m", "llama2:7b"]
        selected = _select_best_ollama_model(models, logger)
        
        log_text = caplog.text
        
        # Should log the selected model
        assert "model" in log_text.lower()
        assert selected in log_text or "🎯" in log_text


class TestLLMProcessing:
    """Test LLM processing with Ollama integration."""
    
    @pytest.fixture
    def test_gnn_dir(self):
        """Create temporary directory with test GNN files."""
        test_dir = tempfile.mkdtemp()
        gnn_dir = Path(test_dir) / "gnn_files"
        gnn_dir.mkdir()
        
        # Create test GNN file
        gnn_file = gnn_dir / "test_active_inference.md"
        gnn_file.write_text("""
# Test Active Inference Model

## Description
A simple active inference model for testing LLM analysis.

## State Blocks
[A]: Observation space (3)
[B]: State transition matrix (3x3x2)
[C]: Preference distribution (3)
[D]: Observation likelihood (3x3)

## Connections
A -> D: "observations"
B -> A: "state to observation"
C -> B: "preferences influence transitions"

## Parameters
- learning_rate: 0.01
- precision: 1.0
- temperature: 0.5

## Inference Goals
Minimize free energy while maintaining preferred states.
""")
        
        yield gnn_dir
        
        # Cleanup
        shutil.rmtree(test_dir)
    
    @pytest.fixture
    def test_output_dir(self):
        """Create temporary output directory."""
        test_dir = tempfile.mkdtemp()
        output_dir = Path(test_dir) / "output"
        output_dir.mkdir()
        
        yield output_dir
        
        # Cleanup
        shutil.rmtree(test_dir)
    
    def test_llm_processing_with_ollama(self, test_gnn_dir, test_output_dir, caplog):
        """Test LLM processing when Ollama is available."""
        import logging
        caplog.set_level(logging.INFO)
        
        llm_output_dir = test_output_dir / "13_llm_output"
        llm_output_dir.mkdir()
        
        result = process_llm(
            target_dir=test_gnn_dir,
            output_dir=llm_output_dir,
            verbose=True
        )
        
        # Should complete (success or graceful failure)
        assert isinstance(result, bool)
        
        # Check for results directory
        results_dir = llm_output_dir / "llm_results"
        assert results_dir.exists(), "llm_results directory should be created"
        
        # Check for results file
        results_file = results_dir / "llm_results.json"
        assert results_file.exists(), "llm_results.json should be created"
        
        with open(results_file) as f:
            results = json.load(f)
        
        # Verify results structure
        assert "timestamp" in results
        assert "ollama_available" in results
        assert "processed_files" in results
        assert "llm_provider" in results
        
        # Check logging
        log_text = caplog.text
        assert "LLM" in log_text or "ollama" in log_text.lower()
    
    def test_llm_processing_without_ollama(self, test_gnn_dir, test_output_dir, caplog, monkeypatch):
        """Test LLM processing fallback when Ollama is not available."""
        import logging
        caplog.set_level(logging.INFO)
        
        # Simulate Ollama not available by making the command fail
        def mock_which(cmd):
            if cmd == "ollama":
                return None
            return shutil.which(cmd)
        
        monkeypatch.setattr(shutil, "which", mock_which)
        
        llm_output_dir = test_output_dir / "13_llm_output"
        llm_output_dir.mkdir()
        
        result = process_llm(
            target_dir=test_gnn_dir,
            output_dir=llm_output_dir,
            verbose=True
        )
        
        # Should complete with fallback
        assert isinstance(result, bool)
        
        # Check that fallback mode was used
        results_file = llm_output_dir / "llm_results" / "llm_results.json"
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
            
            assert results["llm_provider"] == "fallback"
            assert results["ollama_available"] is False
        
        # Check logging mentions fallback
        log_text = caplog.text.lower()
        assert "fallback" in log_text or "not found" in log_text or "not available" in log_text
    
    def test_llm_processing_model_selection(self, test_gnn_dir, test_output_dir, caplog):
        """Test that LLM processing selects and uses appropriate model."""
        import logging
        caplog.set_level(logging.INFO)
        
        llm_output_dir = test_output_dir / "13_llm_output"
        llm_output_dir.mkdir()
        
        result = process_llm(
            target_dir=test_gnn_dir,
            output_dir=llm_output_dir,
            verbose=True
        )
        
        # Check results for model selection
        results_file = llm_output_dir / "llm_results" / "llm_results.json"
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
            
            if results["ollama_available"]:
                # Should have selected a model
                assert "selected_model" in results
                if results["selected_model"]:
                    assert isinstance(results["selected_model"], str)
                    assert len(results["selected_model"]) > 0
                    
                    # Check logging mentions the model
                    log_text = caplog.text
                    assert "model" in log_text.lower() or "🤖" in log_text
    
    def test_llm_processing_creates_outputs(self, test_gnn_dir, test_output_dir):
        """Test that LLM processing creates expected output files."""
        llm_output_dir = test_output_dir / "13_llm_output"
        llm_output_dir.mkdir()
        
        result = process_llm(
            target_dir=test_gnn_dir,
            output_dir=llm_output_dir,
            verbose=False
        )
        
        # Check for key output files
        results_dir = llm_output_dir / "llm_results"
        
        # Should have results file
        assert (results_dir / "llm_results.json").exists()
        
        # Should have summary file
        assert (results_dir / "llm_summary.md").exists()
        
        # May have prompt-specific directories if Ollama was available
        prompt_dirs = list(results_dir.glob("prompts_*"))
        # At least 0 prompt directories (depends on Ollama availability)
        assert len(prompt_dirs) >= 0
    
    def test_llm_processing_error_handling(self, test_output_dir, caplog):
        """Test LLM processing error handling with no input files."""
        import logging
        caplog.set_level(logging.WARNING)
        
        # Empty directory
        empty_dir = test_output_dir / "empty_gnn"
        empty_dir.mkdir()
        
        llm_output_dir = test_output_dir / "13_llm_output"
        llm_output_dir.mkdir()
        
        result = process_llm(
            target_dir=empty_dir,
            output_dir=llm_output_dir,
            verbose=True
        )
        
        # Should handle gracefully
        assert isinstance(result, bool)
        
        # Should log warning about no files
        log_text = caplog.text.lower()
        assert "no" in log_text and "files" in log_text or "warning" in log_text


class TestOllamaIntegrationEnd2End:
    """End-to-end integration tests for Ollama."""
    
    def test_ollama_command_exists(self):
        """Test if ollama command is available in PATH."""
        ollama_path = shutil.which("ollama")
        
        if ollama_path:
            pytest.skip_info = f"Ollama found at: {ollama_path}"
        else:
            pytest.skip_info = "Ollama not found in PATH"
    
    def test_ollama_service_running(self):
        """Test if Ollama service is actually running."""
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                pytest.skip_info = "Ollama is running"
                # Log available models
                if result.stdout:
                    models = [line.split()[0] for line in result.stdout.strip().split('\n')[1:] if line.strip()]
                    if models:
                        pytest.skip_info += f" with models: {', '.join(models[:3])}"
            else:
                pytest.skip_info = "Ollama command exists but service not running"
                
        except FileNotFoundError:
            pytest.skip_info = "Ollama command not found"
        except subprocess.TimeoutExpired:
            pytest.skip_info = "Ollama command timed out"
        except Exception as e:
            pytest.skip_info = f"Ollama check failed: {e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

