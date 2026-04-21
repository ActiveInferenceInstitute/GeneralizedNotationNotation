"""
Tests for the 11_render.py thin orchestrator.
"""

import os
import runpy
import sys

import pytest

import render


def test_11_render_success(monkeypatch, tmp_path):
    target_dir = tmp_path / "test_input"
    output_dir = tmp_path / "test_output"
    
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "11_render.py"))
    
    original_process = getattr(render, "process_render", None)
    render.process_render = lambda *args, **kwargs: True
    
    try:
        monkeypatch.setattr("sys.argv", ["11_render.py", "--target-dir", str(target_dir), "--output-dir", str(output_dir)])
        
        with pytest.raises(SystemExit) as e:
            runpy.run_path(script_path, run_name="__main__")
            
        assert e.value.code == 0
    finally:
        if original_process:
            render.process_render = original_process

def test_11_render_failure(monkeypatch, tmp_path):
    target_dir = tmp_path / "test_input"
    
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "11_render.py"))
    
    original_process = getattr(render, "process_render", None)
    render.process_render = lambda *args, **kwargs: False
    
    try:
        monkeypatch.setattr("sys.argv", ["11_render.py", "--target-dir", str(target_dir)])
        
        with pytest.raises(SystemExit) as e:
            runpy.run_path(script_path, run_name="__main__")
            
        assert e.value.code == 1
    finally:
        if original_process:
            render.process_render = original_process
