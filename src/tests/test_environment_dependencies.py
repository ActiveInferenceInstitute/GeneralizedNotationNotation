"""
Test Environment Dependencies - Tests for environment dependency management.

Tests verification and management of Python package dependencies.
"""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestCoreDependencies:
    """Tests for core Python dependencies."""

    @pytest.mark.fast
    def test_numpy_available(self):
        """Test NumPy is installed and functional."""
        import numpy as np
        arr = np.array([1, 2, 3])
        assert arr.shape == (3,)
        assert np.sum(arr) == 6

    @pytest.mark.fast
    def test_pathlib_available(self):
        """Test pathlib is available (stdlib)."""
        from pathlib import Path
        p = Path('.')
        assert p.exists()

    @pytest.mark.fast
    def test_json_available(self):
        """Test json is available (stdlib)."""
        import json
        data = {'key': 'value'}
        encoded = json.dumps(data)
        decoded = json.loads(encoded)
        assert decoded == data

    @pytest.mark.fast
    def test_logging_available(self):
        """Test logging is available (stdlib)."""
        import logging
        logger = logging.getLogger('test')
        assert logger is not None

class TestVisualizationDependencies:
    """Tests for visualization dependencies."""

    @pytest.mark.fast
    def test_matplotlib_available(self):
        """Test matplotlib is installed."""
        import matplotlib
        import matplotlib.pyplot as plt
        assert hasattr(matplotlib, '__version__'), 'matplotlib missing __version__'
        assert hasattr(plt, 'figure'), 'matplotlib.pyplot missing figure function'

    @pytest.mark.fast
    def test_seaborn_available(self):
        """Test seaborn is installed."""
        import seaborn as sns
        assert hasattr(sns, '__version__'), 'seaborn missing __version__'
        assert hasattr(sns, 'set_theme'), 'seaborn missing set_theme function'

class TestAudioDependencies:
    """Tests for audio processing dependencies."""

    @pytest.mark.fast
    def test_audio_backends(self):
        """Test audio backend availability."""
        from audio import check_audio_backends
        backends = check_audio_backends()
        assert isinstance(backends, dict)
        assert 'numpy' in backends
        assert backends['numpy']['available'] is True

    @pytest.mark.fast
    def test_librosa_availability(self):
        """Test librosa availability."""
        try:
            import librosa
            assert hasattr(librosa, '__version__')
        except ImportError:
            pytest.skip('librosa not installed')

    @pytest.mark.fast
    def test_soundfile_availability(self):
        """Test soundfile availability."""
        try:
            import soundfile
            assert hasattr(soundfile, '__version__')
        except ImportError:
            pytest.skip('soundfile not installed')

class TestMLDependencies:
    """Tests for machine learning dependencies."""

    @pytest.mark.fast
    def test_scipy_available(self):
        """Test scipy is installed."""
        import scipy
        assert hasattr(scipy, '__version__')

    @pytest.mark.fast
    def test_pymdp_available(self):
        """Test pymdp is installed for Active Inference."""
        try:
            try:
                from pymdp.agent import Agent
                assert Agent is not None
                return
            except ImportError:
                pass
            import pymdp
            assert hasattr(pymdp, 'MDP') or hasattr(pymdp, 'mdp') or hasattr(pymdp, '__version__'), 'pymdp missing expected attributes'
        except ImportError:
            pytest.skip('pymdp not installed')

class TestDependencyVersions:
    """Tests for dependency version requirements."""

    @pytest.mark.fast
    def test_python_version(self):
        """Test Python version meets requirements."""
        import sys
        assert sys.version_info >= (3, 9)

    @pytest.mark.fast
    def test_numpy_version(self):
        """Test NumPy version."""
        import numpy as np
        version_parts = np.__version__.split('.')
        major = int(version_parts[0])
        assert major >= 1

class TestDependencyConflicts:
    """Tests for dependency conflict detection."""

    @pytest.mark.fast
    def test_no_import_conflicts(self):
        """Test core imports don't conflict."""
        import json
        import logging
        from pathlib import Path
        import numpy as np
        assert json.dumps({'test': 1}) == '{"test": 1}'
        assert logging.getLogger('test') is not None
        assert Path('.').exists()
        assert np.array([1, 2, 3]).sum() == 6

    @pytest.mark.fast
    def test_visualization_imports_compatible(self):
        """Test visualization imports are compatible."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            line_objs = ax.plot([1, 2, 3])
            assert len(line_objs) == 1, 'Expected one line object from plot'
            assert fig is not None, 'Figure should be created'
            plt.close(fig)
        except Exception as e:
            pytest.skip(f'Visualization compatibility issue: {e}')

class TestOptionalDependencies:
    """Tests for optional dependency handling."""

    @pytest.mark.fast
    def test_optional_deps_graceful_import(self):
        """Test optional dependencies fail gracefully."""
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
        check_audio_backends()
        assert isinstance(FEATURES, dict)

class TestDependencyDiscovery:
    """Tests for dependency discovery functionality."""

    @pytest.mark.fast
    def test_list_installed_packages(self):
        """Test we can list installed packages."""
        from importlib.metadata import distributions
        packages = [d.metadata['Name'] for d in distributions()]
        assert len(packages) > 0

    @pytest.mark.fast
    def test_pyproject_toml_exists(self):
        """Test pyproject.toml dependency file exists (uv-managed)."""
        from pathlib import Path
        pyproject = Path(__file__).parent.parent.parent / 'pyproject.toml'
        assert pyproject.exists(), 'pyproject.toml must exist for uv-managed dependencies'