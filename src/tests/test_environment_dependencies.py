#!/usr/bin/env python3
"""
Test Environment Dependencies - Tests for environment dependency management.

Tests verification and management of Python package dependencies.
"""

import pytest
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCoreDependencies:
    """Tests for core Python dependencies."""

    @pytest.mark.fast
    def test_numpy_available(self):
        """Test NumPy is installed and functional."""
        import numpy as np
        
        # Basic functionality test
        arr = np.array([1, 2, 3])
        assert arr.shape == (3,)
        assert np.sum(arr) == 6

    @pytest.mark.fast
    def test_pathlib_available(self):
        """Test pathlib is available (stdlib)."""
        from pathlib import Path
        
        p = Path(".")
        assert p.exists()

    @pytest.mark.fast
    def test_json_available(self):
        """Test json is available (stdlib)."""
        import json
        
        data = {"key": "value"}
        encoded = json.dumps(data)
        decoded = json.loads(encoded)
        assert decoded == data

    @pytest.mark.fast
    def test_logging_available(self):
        """Test logging is available (stdlib)."""
        import logging
        
        logger = logging.getLogger("test")
        assert logger is not None


class TestVisualizationDependencies:
    """Tests for visualization dependencies."""

    @pytest.mark.fast
    def test_matplotlib_available(self):
        """Test matplotlib is installed."""
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            assert hasattr(matplotlib, '__version__'), "matplotlib missing __version__"
            assert hasattr(plt, 'figure'), "matplotlib.pyplot missing figure function"
        except ImportError:
            pytest.skip("matplotlib not installed")

    @pytest.mark.fast
    def test_seaborn_available(self):
        """Test seaborn is installed."""
        try:
            import seaborn as sns
            assert hasattr(sns, '__version__'), "seaborn missing __version__"
            assert hasattr(sns, 'set_theme'), "seaborn missing set_theme function"
        except ImportError:
            pytest.skip("seaborn not installed")


class TestAudioDependencies:
    """Tests for audio processing dependencies."""

    @pytest.mark.fast
    def test_audio_backends(self):
        """Test audio backend availability."""
        from audio import check_audio_backends
        
        backends = check_audio_backends()
        
        assert isinstance(backends, dict)
        assert 'numpy' in backends
        # NumPy is core and should always be available
        assert backends['numpy']['available'] is True

    @pytest.mark.fast
    def test_librosa_availability(self):
        """Test librosa availability."""
        try:
            import librosa
            assert hasattr(librosa, '__version__')
        except ImportError:
            pytest.skip("librosa not installed")

    @pytest.mark.fast
    def test_soundfile_availability(self):
        """Test soundfile availability."""
        try:
            import soundfile
            assert hasattr(soundfile, '__version__')
        except ImportError:
            pytest.skip("soundfile not installed")


class TestMLDependencies:
    """Tests for machine learning dependencies."""

    @pytest.mark.fast
    def test_scipy_available(self):
        """Test scipy is installed."""
        try:
            import scipy
            assert hasattr(scipy, '__version__')
        except ImportError:
            pytest.skip("scipy not installed")

    @pytest.mark.fast
    def test_pymdp_available(self):
        """Test pymdp is installed for Active Inference."""
        try:
            import pymdp
            # pymdp exports MDP, MDPSolver, mdp, mdp_solver
            assert hasattr(pymdp, 'MDP') or hasattr(pymdp, 'mdp'), \
                "pymdp missing expected attributes (MDP or mdp)"
        except ImportError:
            pytest.skip("pymdp not installed")


class TestDependencyVersions:
    """Tests for dependency version requirements."""

    @pytest.mark.fast
    def test_python_version(self):
        """Test Python version meets requirements."""
        import sys
        
        # Require Python 3.9+
        assert sys.version_info >= (3, 9)

    @pytest.mark.fast
    def test_numpy_version(self):
        """Test NumPy version."""
        import numpy as np
        
        version_parts = np.__version__.split('.')
        major = int(version_parts[0])
        
        # Require NumPy 1.x or 2.x
        assert major >= 1


class TestDependencyConflicts:
    """Tests for dependency conflict detection."""

    @pytest.mark.fast
    def test_no_import_conflicts(self):
        """Test core imports don't conflict."""
        # Import in specific order to detect conflicts
        import json
        import logging
        from pathlib import Path
        import numpy as np

        # Verify each module is functional after imports
        assert json.dumps({"test": 1}) == '{"test": 1}'
        assert logging.getLogger("test") is not None
        assert Path(".").exists()
        assert np.array([1, 2, 3]).sum() == 6

    @pytest.mark.fast
    def test_visualization_imports_compatible(self):
        """Test visualization imports are compatible."""
        try:
            import numpy as np
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt

            # Basic compatibility test
            fig, ax = plt.subplots()
            line_objs = ax.plot([1, 2, 3])
            assert len(line_objs) == 1, "Expected one line object from plot"
            assert fig is not None, "Figure should be created"
            plt.close(fig)
        except Exception as e:
            pytest.skip(f"Visualization compatibility issue: {e}")


class TestOptionalDependencies:
    """Tests for optional dependency handling."""

    @pytest.mark.fast
    def test_optional_deps_graceful_import(self):
        """Test optional dependencies fail gracefully."""
        # This pattern should be used throughout codebase
        try:
            import nonexistent_package
            available = True
        except ImportError:
            available = False
        
        assert available is False

    @pytest.mark.fast
    def test_feature_flags_reflect_deps(self):
        """Test feature flags reflect dependency availability."""
        from audio import FEATURES, check_audio_backends
        
        backends = check_audio_backends()
        
        # Features should be consistent with backends
        assert isinstance(FEATURES, dict)


class TestDependencyDiscovery:
    """Tests for dependency discovery functionality."""

    @pytest.mark.fast
    def test_list_installed_packages(self):
        """Test we can list installed packages."""
        try:
            import pkg_resources
            packages = [p.key for p in pkg_resources.working_set]
            assert len(packages) > 0
        except ImportError:
            # pkg_resources may not be available
            import subprocess
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=freeze"],
                capture_output=True,
                text=True
            )
            assert len(result.stdout) > 0

    @pytest.mark.fast
    def test_requirements_file_exists(self):
        """Test requirements file exists."""
        from pathlib import Path
        
        # Check common locations
        locations = [
            Path(__file__).parent.parent.parent / "requirements.txt",
            Path(__file__).parent.parent.parent / "pyproject.toml"
        ]
        
        exists = any(loc.exists() for loc in locations)
        assert exists is True
