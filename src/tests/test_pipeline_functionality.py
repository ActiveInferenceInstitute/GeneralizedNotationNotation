#!/usr/bin/env python3
"""
Simple functionality tests for GNN pipeline core capabilities.

Tests that the pipeline generates expected outputs and visualizations.
"""

import pytest
import json
import os
from pathlib import Path


class TestPipelineFunctionality:
    """Test core pipeline functionality."""
    
    def test_visualization_generates_images(self):
        """Test that visualization step generates actual image files."""
        viz_dir = Path("output/visualization/actinf_pomdp_agent")
        
        # Check directory exists
        assert viz_dir.exists(), f"Visualization output directory should exist: {viz_dir}"
        
        # Check for PNG files
        png_files = list(viz_dir.glob("*.png"))
        assert len(png_files) > 0, f"Should generate PNG files. Found: {list(viz_dir.iterdir())}"
        
        # Check specific visualization types
        expected_files = ["matrix_analysis.png", "matrix_statistics.png", "pomdp_transition_analysis.png"]
        found_files = [f.name for f in png_files]
        
        for expected in expected_files:
            assert expected in found_files, f"Missing expected visualization: {expected}"
        
        # Check file sizes are reasonable (not empty)
        for png in png_files:
            size = png.stat().st_size
            assert size > 1000, f"Image file {png.name} too small ({size} bytes)"
    
    def test_gnn_processing_generates_parsed_data(self):
        """Test that GNN processing generates parsed model data."""
        parsed_file = Path("output/gnn_processing_step/actinf_pomdp_agent/actinf_pomdp_agent_parsed.json")
        
        assert parsed_file.exists(), "Parsed GNN data file should exist"
        
        with open(parsed_file) as f:
            data = json.load(f)
        
        # Check for essential data structure
        assert "model_name" in data, "Parsed data should contain model name"
        assert "variables" in data, "Parsed data should contain variables"
        assert len(data["variables"]) > 0, "Should parse variables from GNN file"
        
        # Check for Active Inference variables
        var_names = [var["name"] for var in data["variables"]]
        expected_vars = ["A", "B", "C", "D", "E"]  # Core AI variables
        
        for var in expected_vars:
            assert var in var_names, f"Missing expected Active Inference variable: {var}"
    
    def test_multi_format_export_generates_files(self):
        """Test that export step generates files in multiple formats."""
        export_dir = Path("output/gnn_exports/actinf_pomdp_agent")
        
        assert export_dir.exists(), "Export directory should exist"
        
        export_files = list(export_dir.iterdir())
        assert len(export_files) > 3, f"Should export multiple formats. Found: {len(export_files)} files"
        
        # Check for specific formats
        expected_extensions = [".json", ".xml", ".yaml", ".pkl"]
        found_extensions = [f.suffix for f in export_files if f.is_file()]
        
        for ext in expected_extensions:
            assert any(e == ext for e in found_extensions), f"Missing export format: {ext}"
    
    def test_audio_generation_produces_output(self):
        """Test that audio processing generates audio files."""
        audio_dir = Path("output/audio_processing_step/audio_results")
        
        assert audio_dir.exists(), "Audio output directory should exist"
        
        audio_files = list(audio_dir.glob("*.wav"))
        assert len(audio_files) > 0, "Should generate audio files"
        
        # Check file sizes
        for audio_file in audio_files:
            size = audio_file.stat().st_size
            assert size > 1000, f"Audio file {audio_file.name} too small ({size} bytes)"
    
    def test_pipeline_results_are_valid_json(self):
        """Test that pipeline step results are valid JSON."""
        result_files = [
            "output/gnn_processing_step/gnn_processing_results.json",
            "output/visualization/visualization_results.json", 
            "output/type_check/type_check_results.json",
            "output/execution_results/execution_results.json"
        ]
        
        for result_file in result_files:
            path = Path(result_file)
            assert path.exists(), f"Result file should exist: {result_file}"
            
            with open(path) as f:
                data = json.load(f)
            
            assert isinstance(data, dict), f"Result should be valid JSON object: {result_file}"
            assert len(data) > 0, f"Result should not be empty: {result_file}"
    
    def test_visualization_results_show_success(self):
        """Test that visualization results indicate successful generation."""
        results_file = Path("output/visualization/visualization_results.json")
        
        with open(results_file) as f:
            results = json.load(f)
        
        summary = results["summary"]
        assert summary["successful_visualizations"] > 0, "Should have successful visualizations"
        assert summary["total_images_generated"] > 0, "Should generate image files"
        assert summary["failed_visualizations"] == 0, "Should not have failed visualizations"
        
        # Check dependency status
        deps = results["dependency_status"]
        assert deps["matplotlib"] is True, "Matplotlib should be available"
        assert deps["matrix_visualizer"] is True, "Matrix visualizer should be available"
    
    def test_core_pipeline_functionality_metrics(self):
        """Test overall pipeline functionality metrics."""
        # Count successful outputs
        success_indicators = [
            Path("output/visualization/actinf_pomdp_agent").glob("*.png"),  # Image files
            Path("output/gnn_exports/actinf_pomdp_agent").iterdir(),       # Export files  
            Path("output/audio_processing_step/audio_results").glob("*.wav"), # Audio files
            [Path("output/gnn_processing_step/actinf_pomdp_agent/actinf_pomdp_agent_parsed.json")] # Parsed data
        ]
        
        total_outputs = sum(len(list(indicator)) for indicator in success_indicators)
        assert total_outputs > 10, f"Pipeline should generate substantial output. Found: {total_outputs} files"
        
        # Check that we have real functionality, not just text files
        png_count = len(list(Path("output/visualization/actinf_pomdp_agent").glob("*.png")))
        wav_count = len(list(Path("output/audio_processing_step/audio_results").glob("*.wav")))
        
        assert png_count >= 3, f"Should generate multiple visualizations. Found: {png_count}"
        assert wav_count >= 1, f"Should generate audio files. Found: {wav_count}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 