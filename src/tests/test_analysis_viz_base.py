"""Tests for analysis/viz_base.py — safe_savefig."""
import os
import stat
from pathlib import Path

import pytest

import analysis.viz_base as viz_base


class TestMatplotlibAvailable:
    def test_is_bool(self):
        assert isinstance(viz_base.MATPLOTLIB_AVAILABLE, bool)

class TestSavefigReal:
    def test_returns_none_when_unavailable(self, tmp_path):
        """Test fallback behavior if matplotlib is physically disabled."""
        if not viz_base.MATPLOTLIB_AVAILABLE:
            result = viz_base.safe_savefig(tmp_path / "out.png")
            assert result is None

    def test_returns_string_path_on_success(self, tmp_path):
        """Test saving a real figure."""
        if not viz_base.MATPLOTLIB_AVAILABLE:
            pytest.skip("Matplotlib not available")
            
        fig, ax = viz_base.plt.subplots()
        ax.plot([0, 1], [0, 1])
        
        out = tmp_path / "fig.png"
        result = viz_base.safe_savefig(out)
        assert result == str(out)
        assert out.exists()

    def test_parent_directory_created(self, tmp_path):
        """Test parent directory creation."""
        if not viz_base.MATPLOTLIB_AVAILABLE:
            pytest.skip("Matplotlib not available")
            
        fig, ax = viz_base.plt.subplots()
        ax.plot([0, 1])
        
        nested = tmp_path / "a" / "b" / "fig.png"
        viz_base.safe_savefig(nested)
        assert nested.parent.exists()
        assert nested.exists()

    def test_returns_none_on_savefig_failure(self, tmp_path):
        """Test failure handling using a readonly directory to trigger OSError."""
        if not viz_base.MATPLOTLIB_AVAILABLE:
            pytest.skip("Matplotlib not available")
            
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        # Make the directory readonly so creating a file inside it fails
        os.chmod(readonly_dir, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
        
        try:
            fig, ax = viz_base.plt.subplots()
            ax.plot([0, 1])
            
            out = readonly_dir / "fig.png"
            result = viz_base.safe_savefig(out)
            assert result is None
            # Some platforms leave a zero-byte or partial file after failed savefig; contract is None return.
        finally:
            # Restore permissions so pytest can properly clean up tmp_path afterward
            os.chmod(readonly_dir, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
