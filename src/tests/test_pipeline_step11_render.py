"""
Tests for the 11_render.py thin orchestrator.
"""

import sys
import os
import pytest
from unittest.mock import patch
import runpy

def test_11_render_success(monkeypatch):
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "11_render.py"))
    
    with patch("render.process_render", return_value=True):
        monkeypatch.setattr("sys.argv", ["11_render.py", "--target-dir", "dummy", "--output-dir", "dummy_out"])
        
        with pytest.raises(SystemExit) as e:
            runpy.run_path(script_path, run_name="__main__")
            
        assert e.value.code == 0
        
def test_11_render_failure(monkeypatch):
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "11_render.py"))
    
    with patch("render.process_render", return_value=False):
        monkeypatch.setattr("sys.argv", ["11_render.py", "--target-dir", "dummy"])
        
        with pytest.raises(SystemExit) as e:
            runpy.run_path(script_path, run_name="__main__")
            
        assert e.value.code == 1
