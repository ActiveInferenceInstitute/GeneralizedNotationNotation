#!/usr/bin/env python3
"""
Test Environment Python - Tests for Python environment configuration.

Tests Python version, path configuration, and interpreter settings.
"""

import pytest
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPythonVersion:
    """Tests for Python version requirements."""

    @pytest.mark.fast
    def test_python_version_minimum(self):
        """Test Python version meets minimum requirements."""
        # GNN requires Python 3.9+
        assert sys.version_info >= (3, 9), \
            f"Python 3.9+ required, got {sys.version}"

    @pytest.mark.fast
    def test_python_version_compatible(self):
        """Test Python version is in compatible range."""
        # Should work with Python 3.9 - 3.13
        major, minor = sys.version_info[:2]
        
        assert major == 3
        assert 9 <= minor <= 13

    @pytest.mark.fast
    def test_python_implementation(self):
        """Test Python implementation is CPython."""
        import platform
        
        impl = platform.python_implementation()
        # CPython is the expected implementation
        assert impl in ('CPython', 'PyPy')


class TestPythonPath:
    """Tests for Python path configuration."""

    @pytest.mark.fast
    def test_src_in_path(self):
        """Test src directory is in Python path."""
        src_dir = Path(__file__).parent.parent
        
        # src should be importable
        assert str(src_dir) in sys.path or any(
            src_dir.samefile(Path(p)) for p in sys.path if Path(p).exists()
        )

    @pytest.mark.fast
    def test_project_root_accessible(self):
        """Test project root is accessible."""
        project_root = Path(__file__).parent.parent.parent
        
        assert project_root.exists()
        assert (project_root / "pyproject.toml").exists() or \
               (project_root / "setup.py").exists() or \
               (project_root / "src").exists()

    @pytest.mark.fast
    def test_modules_importable(self):
        """Test core modules are importable."""
        # These should all import successfully
        import gnn
        import render
        import report
        
        assert gnn is not None
        assert render is not None
        assert report is not None


class TestPythonInterpreter:
    """Tests for Python interpreter settings."""

    @pytest.mark.fast
    def test_encoding_utf8(self):
        """Test default encoding is UTF-8."""
        assert sys.getdefaultencoding() == 'utf-8'

    @pytest.mark.fast
    def test_recursion_limit_adequate(self):
        """Test recursion limit is adequate."""
        limit = sys.getrecursionlimit()
        
        # Should be at least 1000 (default is usually 1000 or more)
        assert limit >= 1000

    @pytest.mark.fast
    def test_platform_supported(self):
        """Test platform is supported."""
        import platform
        
        system = platform.system()
        
        # Should be one of the supported platforms
        assert system in ('Darwin', 'Linux', 'Windows')


class TestPythonFeatures:
    """Tests for Python language features."""

    @pytest.mark.fast
    def test_async_await_available(self):
        """Test async/await is available."""
        async def async_func():
            return 42
        
        import asyncio
        result = asyncio.run(async_func())
        assert result == 42

    @pytest.mark.fast
    def test_type_hints_work(self):
        """Test type hints work correctly."""
        from typing import List, Dict, Optional
        
        def typed_func(items: List[str]) -> Dict[str, int]:
            return {item: len(item) for item in items}
        
        result = typed_func(["a", "bb", "ccc"])
        assert result == {"a": 1, "bb": 2, "ccc": 3}

    @pytest.mark.fast
    def test_dataclasses_available(self):
        """Test dataclasses are available."""
        from dataclasses import dataclass
        
        @dataclass
        class TestData:
            name: str
            value: int
        
        data = TestData(name="test", value=42)
        assert data.name == "test"
        assert data.value == 42

    @pytest.mark.fast
    def test_pathlib_fully_functional(self):
        """Test pathlib is fully functional."""
        from pathlib import Path
        
        p = Path(".")
        assert p.exists()
        assert p.is_dir()
        assert p.resolve().is_absolute()


class TestPythonModules:
    """Tests for Python standard library modules."""

    @pytest.mark.fast
    def test_json_module(self):
        """Test json module works."""
        import json
        
        data = {"key": "value", "number": 42}
        encoded = json.dumps(data)
        decoded = json.loads(encoded)
        
        assert decoded == data

    @pytest.mark.fast
    def test_logging_module(self):
        """Test logging module works."""
        import logging
        
        logger = logging.getLogger("test")
        logger.setLevel(logging.DEBUG)
        
        # Should not raise
        logger.debug("test")

    @pytest.mark.fast
    def test_subprocess_module(self):
        """Test subprocess module works."""
        import subprocess
        
        result = subprocess.run(
            [sys.executable, "--version"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "Python" in result.stdout or "python" in result.stdout.lower()

    @pytest.mark.fast
    def test_threading_module(self):
        """Test threading module works."""
        import threading
        
        results = []
        
        def worker():
            results.append(42)
        
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        
        assert results == [42]


class TestPythonEnvironment:
    """Tests for Python environment variables."""

    @pytest.mark.fast
    def test_environment_variables_accessible(self):
        """Test environment variables are accessible."""
        import os
        
        # Should be able to access environment
        env = os.environ
        assert isinstance(env, dict) or hasattr(env, '__getitem__')

    @pytest.mark.fast
    def test_cwd_accessible(self):
        """Test current working directory is accessible."""
        import os
        
        cwd = os.getcwd()
        assert cwd is not None
        assert len(cwd) > 0
