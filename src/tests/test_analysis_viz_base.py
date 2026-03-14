"""Tests for analysis/viz_base.py — safe_savefig and check_matplotlib."""
from unittest.mock import patch, MagicMock
import analysis.viz_base as viz_base


# ── check_matplotlib ───────────────────────────────────────────────────────

class TestCheckMatplotlib:
    def test_returns_bool(self):
        result = viz_base.check_matplotlib()
        assert isinstance(result, bool)

    def test_matches_module_flag(self):
        assert viz_base.check_matplotlib() == viz_base.MATPLOTLIB_AVAILABLE


# ── safe_savefig — matplotlib unavailable ─────────────────────────────────

class TestSavefigMatplotlibUnavailable:
    def test_returns_none_when_unavailable(self, tmp_path):
        with patch.object(viz_base, 'MATPLOTLIB_AVAILABLE', False), \
             patch.object(viz_base, 'plt', None):
            result = viz_base.safe_savefig(tmp_path / "out.png")
        assert result is None

    def test_no_file_created_when_unavailable(self, tmp_path):
        out = tmp_path / "out.png"
        with patch.object(viz_base, 'MATPLOTLIB_AVAILABLE', False), \
             patch.object(viz_base, 'plt', None):
            viz_base.safe_savefig(out)
        assert not out.exists()


# ── safe_savefig — matplotlib available ───────────────────────────────────

class TestSavefigMatplotlibAvailable:
    def _mock_plt(self):
        mock = MagicMock()
        mock.savefig = MagicMock()
        mock.close = MagicMock()
        return mock

    def test_returns_string_path_on_success(self, tmp_path):
        mock_plt = self._mock_plt()
        out = tmp_path / "fig.png"
        with patch.object(viz_base, 'MATPLOTLIB_AVAILABLE', True), \
             patch.object(viz_base, 'plt', mock_plt):
            result = viz_base.safe_savefig(out)
        assert result == str(out)

    def test_savefig_called_with_correct_path(self, tmp_path):
        mock_plt = self._mock_plt()
        out = tmp_path / "fig.png"
        with patch.object(viz_base, 'MATPLOTLIB_AVAILABLE', True), \
             patch.object(viz_base, 'plt', mock_plt):
            viz_base.safe_savefig(out, dpi=150)
        mock_plt.savefig.assert_called_once_with(str(out), dpi=150, bbox_inches='tight')

    def test_close_called_on_success(self, tmp_path):
        mock_plt = self._mock_plt()
        with patch.object(viz_base, 'MATPLOTLIB_AVAILABLE', True), \
             patch.object(viz_base, 'plt', mock_plt):
            viz_base.safe_savefig(tmp_path / "fig.png")
        mock_plt.close.assert_called()

    def test_close_called_even_on_savefig_failure(self, tmp_path):
        """plt.close must be called even when plt.savefig raises."""
        mock_plt = self._mock_plt()
        mock_plt.savefig.side_effect = RuntimeError("disk full")
        out = tmp_path / "fig.png"
        with patch.object(viz_base, 'MATPLOTLIB_AVAILABLE', True), \
             patch.object(viz_base, 'plt', mock_plt):
            result = viz_base.safe_savefig(out)
        assert result is None
        mock_plt.close.assert_called()

    def test_returns_none_on_savefig_failure(self, tmp_path):
        mock_plt = self._mock_plt()
        mock_plt.savefig.side_effect = OSError("permission denied")
        with patch.object(viz_base, 'MATPLOTLIB_AVAILABLE', True), \
             patch.object(viz_base, 'plt', mock_plt):
            result = viz_base.safe_savefig(tmp_path / "fig.png")
        assert result is None

    def test_parent_directory_created(self, tmp_path):
        mock_plt = self._mock_plt()
        nested = tmp_path / "a" / "b" / "fig.png"
        with patch.object(viz_base, 'MATPLOTLIB_AVAILABLE', True), \
             patch.object(viz_base, 'plt', mock_plt):
            viz_base.safe_savefig(nested)
        assert nested.parent.exists()

    def test_custom_logger_used(self, tmp_path):
        mock_plt = self._mock_plt()
        custom_log = MagicMock()
        out = tmp_path / "fig.png"
        with patch.object(viz_base, 'MATPLOTLIB_AVAILABLE', True), \
             patch.object(viz_base, 'plt', mock_plt):
            viz_base.safe_savefig(out, log=custom_log)
        custom_log.info.assert_called()
