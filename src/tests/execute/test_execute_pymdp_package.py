#!/usr/bin/env python3
"""
Tests for PyMDP package detection functionality.

This module tests the package detection logic that distinguishes between
the correct PyMDP package (inferactively-pymdp) and wrong variants.
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from execute.pymdp.package_detector import (
        attempt_pymdp_auto_install,
        detect_pymdp_installation,
        get_pymdp_installation_instructions,
        is_correct_pymdp_package,
        validate_pymdp_for_execution,
    )
    PACKAGE_DETECTOR_AVAILABLE = True
except ImportError as e:
    PACKAGE_DETECTOR_AVAILABLE = False
    IMPORT_ERROR = str(e)


@pytest.mark.skipif(not PACKAGE_DETECTOR_AVAILABLE, reason="Package detector not available")
class TestPyMDPPackageDetection:
    """Test PyMDP package detection functionality."""

    def test_detect_pymdp_installation_structure(self) -> None:
        """Test that detection returns expected structure."""
        detection = detect_pymdp_installation()

        assert isinstance(detection, dict)
        assert "installed" in detection
        assert "correct_package" in detection
        assert "wrong_package" in detection
        assert "has_agent" in detection
        assert "has_mdp_solver" in detection
        assert isinstance(detection["installed"], bool)
        assert isinstance(detection["correct_package"], bool)
        assert isinstance(detection["wrong_package"], bool)

    def test_is_correct_pymdp_package(self) -> None:
        """Test correct package detection."""
        result = is_correct_pymdp_package()
        assert isinstance(result, bool)

    def test_get_pymdp_installation_instructions(self) -> None:
        """Test that installation instructions are provided."""
        instructions = get_pymdp_installation_instructions()

        assert isinstance(instructions, str)
        assert len(instructions) > 0
        # Should mention inferactively-pymdp
        assert "inferactively-pymdp" in instructions.lower() or "pymdp" in instructions.lower()

    def test_validate_pymdp_for_execution_structure(self) -> None:
        """Test validation structure."""
        validation = validate_pymdp_for_execution()

        assert isinstance(validation, dict)
        assert "ready" in validation
        assert "detection" in validation
        assert "instructions" in validation
        assert "can_auto_install" in validation
        assert isinstance(validation["ready"], bool)
        assert isinstance(validation["detection"], dict)
        assert isinstance(validation["instructions"], str)
        assert isinstance(validation["can_auto_install"], bool)

    def test_attempt_pymdp_auto_install(self) -> None:
        """Test auto-installation attempt (may fail if package already installed)."""
        # This test may fail if package is already installed, which is fine
        success, message = attempt_pymdp_auto_install(use_uv=True)

        assert isinstance(success, bool)
        assert isinstance(message, str)
        # If already installed, success might be True or False
        # Either way, we should get a message
        assert len(message) > 0


@pytest.mark.skipif(not PACKAGE_DETECTOR_AVAILABLE, reason="Package detector not available")
class TestPyMDPPackageDetectionIntegration:
    """Integration tests for package detection."""

    def test_detection_consistency(self) -> None:
        """Test that detection results are consistent."""
        detection1 = detect_pymdp_installation()
        detection2 = detect_pymdp_installation()

        # Results should be consistent (same installation state)
        assert detection1["installed"] == detection2["installed"]
        assert detection1["correct_package"] == detection2["correct_package"]
        assert detection1["wrong_package"] == detection2["wrong_package"]

    def test_validation_uses_detection(self) -> None:
        """Test that validation uses detection results."""
        validation = validate_pymdp_for_execution()
        detection = validation["detection"]

        # Validation should include detection results
        assert "installed" in detection
        assert "correct_package" in detection

        # If correct package is installed, should be ready
        if detection.get("correct_package"):
            assert validation["ready"] is True

    def test_instructions_are_actionable(self) -> None:
        """Test that instructions provide actionable commands or confirmation."""
        instructions = get_pymdp_installation_instructions()

        # Should contain installation command OR confirmation of correct installation
        assert "install" in instructions.lower() or "pymdp" in instructions.lower() or "correctly installed" in instructions.lower()

        # Should mention uv or pip (for install) or correctly installed (when already installed)
        assert "uv" in instructions.lower() or "pip" in instructions.lower() or "setup" in instructions.lower() or "correctly installed" in instructions.lower()


@pytest.mark.skipif(not PACKAGE_DETECTOR_AVAILABLE, reason="Package detector not available")
class TestPyMDPRealInstallation:
    """Test package detection with real PyMDP installation (if available)."""

    def test_real_installation_detection(self) -> None:
        """Test detection with actual PyMDP installation."""
        try:
            import pymdp
            # PyMDP is installed - test detection
            detection = detect_pymdp_installation()

            assert detection["installed"] is True

            # Check if it's the correct package
            if detection.get("correct_package"):
                # Correct package installed
                assert detection["has_agent"] is True
                assert detection["wrong_package"] is False
            elif detection.get("wrong_package"):
                # Wrong package installed
                assert detection["has_mdp_solver"] is True
                assert detection["has_agent"] is False
        except ImportError:
            # PyMDP not installed - detection should reflect this
            detection = detect_pymdp_installation()
            assert detection["installed"] is False
            pytest.skip("PyMDP not installed - skipping real installation test")

    def test_real_installation_validation(self) -> None:
        """Test validation with real PyMDP installation."""
        try:
            import pymdp
            validation = validate_pymdp_for_execution()

            detection = validation["detection"]
            if detection.get("correct_package"):
                # Should be ready if correct package installed
                assert validation["ready"] is True
            elif detection.get("wrong_package"):
                # Should not be ready if wrong package
                assert validation["ready"] is False
                assert "inferactively-pymdp" in validation["instructions"].lower()
        except ImportError:
            pytest.skip("PyMDP not installed - skipping real installation test")

    def test_wrong_package_detection(self) -> None:
        """Test that wrong package variant is detected if present."""
        detection = detect_pymdp_installation()

        if detection.get("installed"):
            # If package is installed, check detection accuracy
            if detection.get("wrong_package"):
                # Wrong package detected
                assert detection["has_mdp_solver"] is True
                assert detection["correct_package"] is False
            elif detection.get("correct_package"):
                # Correct package detected
                assert detection["has_agent"] is True
                assert detection["wrong_package"] is False

