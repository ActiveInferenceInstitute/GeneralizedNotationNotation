"""Pins for ``utils/system_utils.get_system_info`` (previously untested)."""

from __future__ import annotations

from gnn.utils.system_utils import PSUTIL_AVAILABLE, get_system_info


def test_get_system_info_returns_base_contract() -> None:
    info = get_system_info()

    assert "error" not in info
    assert isinstance(info["python_version"], str)
    assert info["platform"] in {"posix", "nt", "java"}
    assert isinstance(info["cpu_count"], int) and info["cpu_count"] >= 1
    assert isinstance(info["working_directory"], str)
    assert info["user"]


def test_get_system_info_memory_fields_match_psutil_availability() -> None:
    info = get_system_info()

    if PSUTIL_AVAILABLE:
        assert isinstance(info["memory_total_gb"], float)
        assert info["memory_total_gb"] > 0
        assert isinstance(info["disk_free_gb"], float)
    else:
        assert "unavailable" in str(info["memory_total_gb"])
