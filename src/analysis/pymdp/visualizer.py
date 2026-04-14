#!/usr/bin/env python3
"""
PyMDP Visualization Module

Comprehensive visualization utilities for PyMDP simulations.
This module provides plotting capabilities for discrete POMDP environments,
agent trajectories, belief distributions, and performance metrics.

Features:
- Discrete state visualization 
- Belief distribution plots
- Performance metrics visualization
- Active Inference analysis plots
- Pipeline integration support

Author: GNN PyMDP Integration
Date: 2024
"""
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Import shared visualization utilities (centralized matplotlib setup)
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..viz_base import MATPLOTLIB_AVAILABLE, np, plt


class PyMDPVisualizer:
    """Visualization utilities for PyMDP simulation

    Backwards-compatible initializer: tests and callers may pass either
    positional `grid_size` or an `output_dir`/`save_dir` along with
    `show_plots` flag. This constructor accepts both styles and normalizes them.
    """
    def __init__(self, *args, grid_size: Optional[int] = None,
                 figsize: Tuple[int, int] = (10, 8),
                 style: str = 'default',
                 save_dir: Optional[Union[Path, str]] = None,
                 output_dir: Optional[Union[Path, str]] = None,
                 show_plots: bool = True,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize PyMDP visualizer with configuration from GNN specifications.
        
        Args:
            grid_size: Size of discrete state space (default: auto-detect from config)
            figsize: Default figure size for plots
            style: Matplotlib style to use
            save_dir: Directory to save visualizations
        """
        # Normalize previous calling conventions
        # If first positional arg provided and is a Path -> treat as output_dir
        if args:
            first = args[0]
            if isinstance(first, (Path, str)):
                output_dir = first
            elif isinstance(first, int):
                grid_size = first

        self.grid_size = grid_size
        self.figsize = figsize
        self.style = style
        # Allow passing config dict as alias for previous callers
        if config:
            save_dir = save_dir or config.get('save_dir') or config.get('output_dir')
            grid_size = grid_size or config.get('grid_size')

        # Prefer explicit save_dir, recovery to output_dir
        save_target = save_dir or output_dir
        self.save_dir = Path(save_target) if save_target is not None else None
        self.show_plots = bool(show_plots)

        # Set matplotlib style if available
        if MATPLOTLIB_AVAILABLE and plt is not None:
            plt.style.use(self.style)

        # Color schemes for different visualizations
        self.colors = {
            'agent': 'red',
            'goal': 'green',
            'obstacle': 'black',
            'empty': 'white',
            'visited': 'lightblue',
            'path': 'blue'
        }

        # Track visualization metadata
        self.figures = {}
        self.plot_count = 0

    # Backwards-compatible wrappers expected by tests
    def plot_discrete_states(self, state_sequence: List[int], num_states: int, title: str = "Discrete State Sequence", save_path: Optional[Union[Path, str]] = None) -> Any:
        fig = self.visualize_discrete_states(state_sequence, num_states, title)
        if fig and save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            return fig
        return fig

    def plot_belief_evolution(self, belief_traces: List[np.ndarray], title: str = "Belief Evolution", save_path: Optional[Union[Path, str]] = None) -> Any:
        fig = self.visualize_belief_evolution(belief_traces, title)
        if fig and save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            return fig
        return fig

    def plot_performance_metrics(self, metrics: Dict[str, Any], title: str = "Performance Metrics", save_path: Optional[Union[Path, str]] = None) -> Any:
        fig = self.visualize_performance_metrics(metrics, title)
        if fig and save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            return fig
        return fig

    def plot_action_sequence(self, action_sequence: List[int], num_actions: Optional[int] = None, title: str = "Action Sequence", save_path: Optional[Union[Path, str]] = None) -> Any:
        # Infer number of actions if not provided
        if num_actions is None:
            try:
                num_actions = int(max(action_sequence) + 1) if len(action_sequence) > 0 else 1
            except Exception:
                num_actions = 1
        # Use discrete states plot as proxy for actions
        return self.plot_discrete_states(action_sequence, num_actions, title, save_path)

    def plot_observation_sequence(self, observation_sequence: List[int], num_observations: int, title: str = "Observation Sequence", save_path: Optional[Union[Path, str]] = None) -> Any:
        return self.plot_discrete_states(observation_sequence, num_observations, title, save_path)

    def plot_episode_summary(self, episode_trace: Dict[str, Any], episode_num: int = 0, save_path: Optional[Union[Path, str]] = None) -> Any:
        # Create a combined visualization: states + belief + performance
        metrics = {
            'actions': episode_trace.get('actions', []),
            'belief_confidence': [max(b) if hasattr(b, 'tolist') or isinstance(b, (list, tuple, np.ndarray)) else 0 for b in episode_trace.get('beliefs', [])],
            'cumulative_preference': episode_trace.get('rewards', [])
        }
        fig = self.visualize_performance_metrics(metrics, title=f"Episode {episode_num} Summary")
        if fig and save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            return fig
        return fig

    def visualize_discrete_states(
        self,
        states: List[int],
        num_states: int,
        title: str = "Discrete State Sequence"
    ) -> Optional[Any]:
        """
        Visualize sequence of discrete states over time.
        
        Args:
            states: List of discrete state indices
            num_states: Total number of possible states
            title: Plot title
            
        Returns:
            matplotlib Figure object or None if matplotlib not available
        """
        if not MATPLOTLIB_AVAILABLE or plt is None:
            return None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)

        # Plot state sequence over time
        ax1.plot(states, 'o-', color=self.colors['agent'], linewidth=2, markersize=6)
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('State Index')
        ax1.set_title(f'{title} - State Trajectory')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.5, num_states - 0.5)

        # Plot state histogram
        state_counts = np.bincount(states, minlength=num_states)
        ax2.bar(range(num_states), state_counts, color=self.colors['visited'], alpha=0.7)
        ax2.set_xlabel('State Index')
        ax2.set_ylabel('Visit Count')
        ax2.set_title(f'{title} - State Visitation')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        self.figures[f'discrete_states_{self.plot_count}'] = fig
        self.plot_count += 1

        return fig

    def visualize_belief_evolution(
        self,
        beliefs: List[np.ndarray],
        title: str = "Belief Evolution"
    ) -> Optional[Any]:
        """
        Visualize evolution of belief distributions over time.
        
        Args:
            beliefs: List of belief distributions (probability vectors)
            title: Plot title
            
        Returns:
            matplotlib Figure object or None if matplotlib not available
        """
        if not MATPLOTLIB_AVAILABLE or plt is None:
            return None

        # Convert beliefs to numpy array and handle various input formats
        beliefs_array = np.array(beliefs)

        # Squeeze out singleton dimensions (e.g., shape (10, 4, 1) -> (10, 4))
        beliefs_array = np.squeeze(beliefs_array)

        # Ensure 2D (timesteps, states) - handle edge cases
        if beliefs_array.ndim == 1:
            beliefs_array = beliefs_array.reshape(1, -1)
        elif beliefs_array.ndim == 3:
            # If still 3D after squeeze, take first factor modality
            beliefs_array = beliefs_array[:, :, 0] if beliefs_array.shape[2] <= beliefs_array.shape[0] else beliefs_array[0, :, :]

        # Transpose if needed to get (timesteps, states) format
        if beliefs_array.shape[0] < beliefs_array.shape[1]:
            # Likely (states, timesteps), transpose it
            beliefs_array = beliefs_array.T

        beliefs_array.shape[1]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)

        # Heatmap of beliefs over time
        im = ax1.imshow(beliefs_array.T, aspect='auto', cmap='Blues', origin='lower')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('State Index')
        ax1.set_title(f'{title} - Belief Heatmap')
        plt.colorbar(im, ax=ax1, label='Belief Probability')

        # Plot belief entropy over time
        entropies = []
        for b in beliefs:
            b_arr = np.asarray(b)
            entropies.append(-np.sum(b_arr * np.log(b_arr + 1e-10)))
        ax2.plot(entropies, 'o-', color=self.colors['path'], linewidth=2)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Entropy (nats)')
        ax2.set_title(f'{title} - Belief Entropy')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        self.figures[f'belief_evolution_{self.plot_count}'] = fig
        self.plot_count += 1

        return fig

    def visualize_performance_metrics(
        self,
        metrics: Dict[str, Any],
        title: str = "Performance Metrics"
    ) -> Optional[Any]:
        """
        Visualize various performance metrics from simulation.
        
        Args:
            metrics: Dictionary containing performance metrics
            title: Plot title
            
        Returns:
            matplotlib Figure object or None if matplotlib not available
        """
        if not MATPLOTLIB_AVAILABLE or plt is None:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Expected free energy over time
        if 'expected_free_energy' in metrics:
            axes[0, 0].plot(metrics['expected_free_energy'], 'o-',
                          color=self.colors['agent'], linewidth=2)
            axes[0, 0].set_xlabel('Time Step')
            axes[0, 0].set_ylabel('Expected Free Energy')
            axes[0, 0].set_title('Expected Free Energy')
            axes[0, 0].grid(True, alpha=0.3)

        # Action distribution
        if 'actions' in metrics and len(metrics['actions']) > 0:
            actions_arr = np.asarray(metrics['actions'])
            action_counts = np.bincount(actions_arr.astype(int))
            axes[0, 1].bar(range(len(action_counts)), action_counts,
                          color=self.colors['visited'], alpha=0.7)
            axes[0, 1].set_xlabel('Action Index')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Action Distribution')
            axes[0, 1].grid(True, alpha=0.3)

        # Belief confidence over time
        if 'belief_confidence' in metrics:
            axes[1, 0].plot(metrics['belief_confidence'], 'o-',
                          color=self.colors['goal'], linewidth=2)
            axes[1, 0].set_xlabel('Time Step')
            axes[1, 0].set_ylabel('Max Belief Probability')
            axes[1, 0].set_title('Belief Confidence')
            axes[1, 0].grid(True, alpha=0.3)

        # Cumulative reward/preference
        if 'cumulative_preference' in metrics:
            axes[1, 1].plot(np.cumsum(metrics['cumulative_preference']), 'o-',
                          color=self.colors['path'], linewidth=2)
            axes[1, 1].set_xlabel('Time Step')
            axes[1, 1].set_ylabel('Cumulative Preference')
            axes[1, 1].set_title('Cumulative Preference')
            axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        self.figures[f'performance_{self.plot_count}'] = fig
        self.plot_count += 1

        return fig

    def visualize_vfe_vs_efe(
        self,
        vfe: List[float],
        efe: List[Any],
        title: str = "Variational vs Expected Free Energy"
    ) -> Optional[Any]:
        """
        Visualize Variational Free Energy against Expected Free Energy.
        
        Args:
            vfe: List of variational free energy values (scalars)
            efe: List of expected free energy values (lists of scalars, one per policy)
            title: Plot title
            
        Returns:
            matplotlib Figure object or None if matplotlib not available
        """
        if not MATPLOTLIB_AVAILABLE or plt is None:
            return None

        # Process EFE (take min across policies if it's a list)
        efe_summary = []
        for efe_t in efe:
            if hasattr(efe_t, '__iter__') and not isinstance(efe_t, str):
                efe_summary.append(min(efe_t) if len(efe_t) > 0 else 0)
            else:
                efe_summary.append(efe_t)

        fig, ax1 = plt.subplots(figsize=(10, 6))

        color1 = 'tab:blue'
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Variational Free Energy (VFE)', color=color1)
        ax1.plot(vfe, 'o-', color=color1, linewidth=2, label='VFE (Belief Update Cost)')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        color2 = 'tab:orange'
        ax2.set_ylabel('Expected Free Energy (EFE)', color=color2)
        ax2.plot(efe_summary, 's--', color=color2, linewidth=2, label='Min EFE (Policy Cost)')
        ax2.tick_params(axis='y', labelcolor=color2)

        fig.suptitle(title, fontsize=14)
        fig.tight_layout()

        # Combine legends from both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        plt.subplots_adjust(bottom=0.2)

        self.figures[f'energy_dynamics_{self.plot_count}'] = fig
        self.plot_count += 1

        return fig

    def save_all_plots(self, output_dir: Optional[Path] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
        """
        Save all generated plots to files.
        
        Args:
            output_dir: Directory to save plots (overrides default)
            metadata: Optional dictionary with metadata (e.g. timestamp) to embed in images
            
        Returns:
            Dictionary mapping plot names to saved file paths
        """
        save_dir = output_dir or self.save_dir
        if not save_dir:
            raise ValueError("No save directory specified")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        for plot_name, fig in self.figures.items():
            if metadata:
                meta_text = " | ".join(f"{k}: {v}" for k, v in metadata.items())
                fig.text(0.99, 0.01, meta_text, ha='right', va='bottom', fontsize=8, color='gray', alpha=0.7)
                
            filepath = save_dir / f"{plot_name}.png"
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files[plot_name] = filepath

        return saved_files

    def close_all_plots(self) -> None:
        """Close all matplotlib figures to free memory."""
        if MATPLOTLIB_AVAILABLE and plt is not None:
            for fig in self.figures.values():
                plt.close(fig)
        self.figures.clear()
        self.plot_count = 0


def create_visualizer(config: Dict[str, Any]) -> PyMDPVisualizer:
    """
    Factory function to create PyMDPVisualizer from configuration.
    
    Args:
        config: Configuration dictionary with visualization parameters
        
    Returns:
        Configured PyMDPVisualizer instance
    """
    return PyMDPVisualizer(
        grid_size=config.get('grid_size', None),
        figsize=config.get('figsize', (10, 8)),
        style=config.get('style', 'default'),
        save_dir=config.get('save_dir', None)
    )


def save_all_visualizations(
    simulation_results: Dict[str, Any],
    output_dir: Path,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Path]:
    """
    Generate and save all visualizations for simulation results.
    
    Args:
        simulation_results: Results from PyMDP simulation
        output_dir: Directory to save visualizations
        config: Optional visualization configuration
        
    Returns:
        Dictionary mapping visualization names to saved file paths
    """
    config = config or {}
    visualizer = create_visualizer({**config, 'save_dir': output_dir})

    saved_files = {}

    try:
        # Visualize state sequence
        if 'states' in simulation_results:
            visualizer.visualize_discrete_states(
                simulation_results['states'],
                simulation_results.get('num_states', max(simulation_results['states']) + 1),
                "Agent State Trajectory"
            )

        # Visualize belief evolution
        if 'beliefs' in simulation_results:
            visualizer.visualize_belief_evolution(
                simulation_results['beliefs'],
                "Belief State Evolution"
            )

        # Visualize performance metrics
        if 'metrics' in simulation_results:
            visualizer.visualize_performance_metrics(
                simulation_results['metrics'],
                "Simulation Performance"
            )
            
            # Surface dual-energy tracking inherently if both exist natively
            metrics = simulation_results['metrics']
            vfe = metrics.get('variational_free_energy') or metrics.get('vfe_history')
            efe = metrics.get('expected_free_energy') or metrics.get('efe_history')
            if vfe and efe:
                visualizer.visualize_vfe_vs_efe(vfe, efe, "Active Inference Energy Dynamics")

        # Build metadata for plots
        metadata = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "PyMDP Framework Mode": "Discrete"
        }
        if 'num_states' in simulation_results:
            metadata["States"] = simulation_results['num_states']
        
        # Save all plots
        saved_files = visualizer.save_all_plots(metadata=metadata)

    finally:
        visualizer.close_all_plots()

    return saved_files


if __name__ == "__main__":
    print("PyMDP Visualizer - Visualization utilities for PyMDP simulations")
    print("This module should be imported and used with the PyMDP simulation class.")
