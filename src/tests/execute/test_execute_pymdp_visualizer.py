#!/usr/bin/env python3
"""
Test script for PyMDP visualizer

This script tests the PyMDP visualization utilities to ensure
they work correctly for discrete POMDP simulations.
"""

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from analysis.pymdp.visualizer import PyMDPVisualizer, create_visualizer


class TestPyMDPVisualizer:
    """Test suite for PyMDP visualizer functionality."""

    @pytest.fixture
    def temp_output_dir(self) -> Any:
        """Create a temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def visualizer(self, temp_output_dir: Any) -> Any:
        """Create a PyMDPVisualizer instance for testing."""
        return PyMDPVisualizer(output_dir=temp_output_dir, show_plots=False)

    def test_visualizer_creation(self, temp_output_dir: Any) -> Any:
        """Test PyMDP visualizer creation."""
        visualizer = PyMDPVisualizer(output_dir=temp_output_dir, show_plots=False)
        assert visualizer is not None
        assert visualizer.save_dir == temp_output_dir

    def test_visualizer_factory_creation(self, temp_output_dir: Any) -> Any:
        """Test visualizer creation via factory function."""
        visualizer = create_visualizer({"save_dir": temp_output_dir})
        assert visualizer is not None

    def test_discrete_state_visualization(
        self, visualizer: Any, temp_output_dir: Any
    ) -> Any:
        """Test discrete state visualization."""
        state_sequence: list[Any] = [0, 1, 2, 1, 0, 2]
        num_states = 3
        save_path = temp_output_dir / "test_states.png"

        fig = visualizer.plot_discrete_states(
            state_sequence=state_sequence,
            num_states=num_states,
            title="Test State Sequence",
            save_path=save_path,
        )

        import matplotlib.pyplot as plt

        plt.close(fig)

        assert save_path.exists(), "State plot file should be created"

    def test_belief_visualization(self, visualizer: Any, temp_output_dir: Any) -> Any:
        """Test belief evolution visualization."""
        belief_traces: list[Any] = [
            np.array([0.8, 0.2, 0.0]),
            np.array([0.6, 0.3, 0.1]),
            np.array([0.2, 0.3, 0.5]),
            np.array([0.1, 0.1, 0.8]),
        ]
        save_path = temp_output_dir / "test_beliefs.png"

        fig = visualizer.plot_belief_evolution(
            belief_traces=belief_traces,
            title="Test Belief Evolution",
            save_path=save_path,
        )

        if fig:
            import matplotlib.pyplot as plt

            plt.close(fig)

        # Either file exists or figure was returned (empty traces case)
        assert fig is not None or save_path.exists()

    def test_performance_metrics(self, visualizer: Any, temp_output_dir: Any) -> Any:
        """Test performance metrics visualization."""
        test_metrics: dict[str, Any] = {
            "episode_rewards": [0.9, 1.2, 0.8, 1.5, 1.0],
            "episode_lengths": [15, 12, 18, 10, 14],
            "belief_entropies": [1.8, 1.2, 0.9, 1.1, 1.0],
            "success_rates": [0.2, 0.4, 0.6, 0.75, 0.8],
        }
        save_path = temp_output_dir / "test_performance.png"

        fig = visualizer.plot_performance_metrics(
            metrics=test_metrics, save_path=save_path
        )

        import matplotlib.pyplot as plt

        plt.close(fig)

        assert save_path.exists(), "Performance metrics plot file should be created"

    def test_action_visualization(self, visualizer: Any, temp_output_dir: Any) -> Any:
        """Test action sequence visualization."""
        action_sequence: list[Any] = [0, 1, 2, 0, 1, 2, 1, 0]
        num_actions = 3
        save_path = temp_output_dir / "test_actions.png"

        fig = visualizer.plot_action_sequence(
            action_sequence=action_sequence,
            num_actions=num_actions,
            title="Test Actions",
            save_path=save_path,
        )

        import matplotlib.pyplot as plt

        plt.close(fig)

        assert save_path.exists(), "Action sequence plot file should be created"

    def test_observation_visualization(
        self, visualizer: Any, temp_output_dir: Any
    ) -> Any:
        """Test observation sequence visualization."""
        observation_sequence: list[Any] = [0, 0, 1, 2, 1, 0, 2]
        num_observations = 3
        save_path = temp_output_dir / "test_observations.png"

        fig = visualizer.plot_observation_sequence(
            observation_sequence=observation_sequence,
            num_observations=num_observations,
            title="Test Observations",
            save_path=save_path,
        )

        import matplotlib.pyplot as plt

        plt.close(fig)

        assert save_path.exists(), "Observation sequence plot file should be created"

    def test_episode_summary(self, visualizer: Any, temp_output_dir: Any) -> Any:
        """Test episode summary visualization."""
        episode_trace: dict[str, Any] = {
            "true_states": [0, 1, 2, 1, 0],
            "observations": [0, 1, 2, 1, 0],
            "actions": [1, 2, 0, 1],
            "rewards": [-0.1, -0.1, 1.0, -0.1],
            "beliefs": [
                np.array([0.8, 0.2, 0.0]),
                np.array([0.3, 0.6, 0.1]),
                np.array([0.1, 0.2, 0.7]),
                np.array([0.2, 0.7, 0.1]),
                np.array([0.8, 0.1, 0.1]),
            ],
        }
        save_path = temp_output_dir / "test_episode_summary.png"

        fig = visualizer.plot_episode_summary(
            episode_trace=episode_trace, episode_num=1, save_path=save_path
        )

        import matplotlib.pyplot as plt

        plt.close(fig)

        assert save_path.exists(), "Episode summary file should be created"

    def test_comprehensive_visualization(
        self, visualizer: Any, temp_output_dir: Any
    ) -> Any:
        """Test comprehensive visualization generation using save_all_visualizations."""
        from analysis.pymdp.visualizer import save_all_visualizations

        sim_results: dict[str, Any] = {
            "states": [0, 1, 2, 1, 0],
            "beliefs": [
                np.array([0.8, 0.2, 0.0]),
                np.array([0.3, 0.6, 0.1]),
                np.array([0.1, 0.2, 0.7]),
                np.array([0.2, 0.7, 0.1]),
                np.array([0.8, 0.1, 0.1]),
            ],
            "metrics": {
                "expected_free_energy": [0.5, 0.4, 0.3, 0.2],
                "actions": [1, 2, 0, 1],
                "belief_confidence": [0.8, 0.6, 0.7, 0.7, 0.8],
                "cumulative_preference": [0.1, 0.2, 0.3, 0.4],
            },
            "num_states": 3,
        }

        saved_files = save_all_visualizations(
            simulation_results=sim_results,
            output_dir=temp_output_dir,
            config={"save_dir": temp_output_dir},
        )

        # Check if files were created
        assert len(saved_files) > 0, "save_all_visualizations should create files"


# Standalone test functions
def test_visualizer_creation() -> Any:
    """Test PyMDP visualizer creation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        visualizer = PyMDPVisualizer(output_dir=temp_path, show_plots=False)
        assert visualizer is not None

        visualizer2 = create_visualizer({"save_dir": temp_path})
        assert visualizer2 is not None


def test_discrete_state_visualization() -> Any:
    """Test discrete state visualization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        visualizer = PyMDPVisualizer(temp_path, show_plots=False)

        state_sequence: list[Any] = [0, 1, 2, 1, 0, 2]
        num_states = 3

        fig = visualizer.plot_discrete_states(
            state_sequence=state_sequence,
            num_states=num_states,
            title="Test State Sequence",
            save_path=temp_path / "test_states.png",
        )

        import matplotlib.pyplot as plt

        plt.close(fig)

        assert (temp_path / "test_states.png").exists()


def test_belief_visualization() -> Any:
    """Test belief evolution visualization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        visualizer = PyMDPVisualizer(temp_path, show_plots=False)

        belief_traces: list[Any] = [
            np.array([0.8, 0.2, 0.0]),
            np.array([0.6, 0.3, 0.1]),
            np.array([0.2, 0.3, 0.5]),
            np.array([0.1, 0.1, 0.8]),
        ]

        fig = visualizer.plot_belief_evolution(
            belief_traces=belief_traces,
            title="Test Belief Evolution",
            save_path=temp_path / "test_beliefs.png",
        )

        if fig:
            import matplotlib.pyplot as plt

            plt.close(fig)

        assert fig is not None or (temp_path / "test_beliefs.png").exists()


def test_performance_metrics() -> Any:
    """Test performance metrics visualization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        visualizer = PyMDPVisualizer(temp_path, show_plots=False)

        test_metrics: dict[str, Any] = {
            "episode_rewards": [0.9, 1.2, 0.8, 1.5, 1.0],
            "episode_lengths": [15, 12, 18, 10, 14],
            "belief_entropies": [1.8, 1.2, 0.9, 1.1, 1.0],
            "success_rates": [0.2, 0.4, 0.6, 0.75, 0.8],
        }

        fig = visualizer.plot_performance_metrics(
            metrics=test_metrics, save_path=temp_path / "test_performance.png"
        )

        import matplotlib.pyplot as plt

        plt.close(fig)

        assert (temp_path / "test_performance.png").exists()


def test_action_visualization() -> Any:
    """Test action sequence visualization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        visualizer = PyMDPVisualizer(temp_path, show_plots=False)

        action_sequence: list[Any] = [0, 1, 2, 0, 1, 2, 1, 0]
        num_actions = 3

        fig = visualizer.plot_action_sequence(
            action_sequence=action_sequence,
            num_actions=num_actions,
            title="Test Actions",
            save_path=temp_path / "test_actions.png",
        )

        import matplotlib.pyplot as plt

        plt.close(fig)

        assert (temp_path / "test_actions.png").exists()


def test_observation_visualization() -> Any:
    """Test observation sequence visualization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        visualizer = PyMDPVisualizer(temp_path, show_plots=False)

        observation_sequence: list[Any] = [0, 0, 1, 2, 1, 0, 2]
        num_observations = 3

        fig = visualizer.plot_observation_sequence(
            observation_sequence=observation_sequence,
            num_observations=num_observations,
            title="Test Observations",
            save_path=temp_path / "test_observations.png",
        )

        import matplotlib.pyplot as plt

        plt.close(fig)

        assert (temp_path / "test_observations.png").exists()


def test_episode_summary() -> Any:
    """Test episode summary visualization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        visualizer = PyMDPVisualizer(temp_path, show_plots=False)

        episode_trace: dict[str, Any] = {
            "true_states": [0, 1, 2, 1, 0],
            "observations": [0, 1, 2, 1, 0],
            "actions": [1, 2, 0, 1],
            "rewards": [-0.1, -0.1, 1.0, -0.1],
            "beliefs": [
                np.array([0.8, 0.2, 0.0]),
                np.array([0.3, 0.6, 0.1]),
                np.array([0.1, 0.2, 0.7]),
                np.array([0.2, 0.7, 0.1]),
                np.array([0.8, 0.1, 0.1]),
            ],
        }

        fig = visualizer.plot_episode_summary(
            episode_trace=episode_trace,
            episode_num=1,
            save_path=temp_path / "test_episode_summary.png",
        )

        import matplotlib.pyplot as plt

        plt.close(fig)

        assert (temp_path / "test_episode_summary.png").exists()


def test_comprehensive_visualization() -> Any:
    """Test comprehensive visualization generation using save_all_visualizations."""
    from analysis.pymdp.visualizer import save_all_visualizations

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        sim_results: dict[str, Any] = {
            "states": [0, 1, 2, 1, 0],
            "beliefs": [
                np.array([0.8, 0.2, 0.0]),
                np.array([0.3, 0.6, 0.1]),
                np.array([0.1, 0.2, 0.7]),
                np.array([0.2, 0.7, 0.1]),
                np.array([0.8, 0.1, 0.1]),
            ],
            "metrics": {
                "expected_free_energy": [0.5, 0.4, 0.3, 0.2],
                "actions": [1, 2, 0, 1],
                "belief_confidence": [0.8, 0.6, 0.7, 0.7, 0.8],
                "cumulative_preference": [0.1, 0.2, 0.3, 0.4],
            },
            "num_states": 3,
        }

        saved_files = save_all_visualizations(
            simulation_results=sim_results,
            output_dir=temp_path,
            config={"save_dir": temp_path},
        )

        assert len(saved_files) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
