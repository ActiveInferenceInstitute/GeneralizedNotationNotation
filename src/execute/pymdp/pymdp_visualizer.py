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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class PyMDPVisualizer:
    """Visualization utilities for PyMDP simulation"""
    
    def __init__(self, output_dir: Path, 
                 plot_style: str = 'seaborn-v0_8', 
                 figure_size: Tuple[int, int] = (12, 8),
                 show_plots: bool = False):
        """
        Initialize the PyMDPVisualizer.
        
        Args:
            output_dir: Directory to save visualizations
            plot_style: Matplotlib style to use
            figure_size: Figure size for plots
            show_plots: Whether to display plots during execution
        """
        self.output_dir = output_dir
        self.show_plots = show_plots
        self.figure_size = figure_size
        
        # Set up plotting style
        try:
            plt.style.use(plot_style)
        except:
            plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_discrete_states(self, state_sequence: List[int], 
                           num_states: int,
                           title: str = "State Sequence",
                           save_path: Optional[Path] = None) -> plt.Figure:
        """Plot discrete state sequence over time"""
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        time_steps = range(len(state_sequence))
        
        # Plot state trajectory
        ax.step(time_steps, state_sequence, 'b-', linewidth=2, where='post', alpha=0.7)
        ax.scatter(time_steps, state_sequence, c=state_sequence, cmap='viridis', 
                  s=100, edgecolors='black', linewidth=2)
        
        # Customize plot
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('State')
        ax.set_yticks(range(num_states))
        ax.set_yticklabels([f'State {i}' for i in range(num_states)])
        ax.grid(True, alpha=0.3)
        
        # Add state labels on points
        for t, state in enumerate(state_sequence):
            ax.annotate(f'S{state}', (t, state), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if self.show_plots:
            plt.show()
        
        return fig
    
    def plot_belief_evolution(self, belief_traces: List[np.ndarray],
                             title: str = "Belief Evolution",
                             save_path: Optional[Path] = None) -> plt.Figure:
        """Plot evolution of beliefs over time steps"""
        if not belief_traces:
            return None
            
        num_states = len(belief_traces[0])
        num_steps = len(belief_traces)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figure_size[0], self.figure_size[1] * 1.2))
        
        # Plot 1: Belief evolution heatmap
        belief_matrix = np.array(belief_traces).T  # states x time
        
        im1 = ax1.imshow(belief_matrix, aspect='auto', cmap='viridis', origin='lower')
        ax1.set_title('Belief Evolution Heatmap', fontweight='bold')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('State')
        ax1.set_yticks(range(num_states))
        ax1.set_yticklabels([f'State {i}' for i in range(num_states)])
        plt.colorbar(im1, ax=ax1, label='Belief Probability')
        
        # Plot 2: Belief entropy over time
        entropies = [-np.sum(belief * np.log(belief + 1e-16)) for belief in belief_traces]
        time_steps = range(len(entropies))
        
        ax2.plot(time_steps, entropies, 'r-o', linewidth=2, markersize=4)
        ax2.set_title('Belief Entropy Over Time', fontweight='bold')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Entropy (nats)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if self.show_plots:
            plt.show()
        
        return fig
    
    def plot_performance_metrics(self, metrics: Dict[str, List], 
                                save_path: Optional[Path] = None) -> plt.Figure:
        """Plot performance metrics over episodes"""
        fig, axes = plt.subplots(2, 2, figsize=self.figure_size)
        
        # Episode rewards
        if 'episode_rewards' in metrics and metrics['episode_rewards']:
            axes[0, 0].plot(metrics['episode_rewards'], 'b-o', linewidth=2, markersize=4)
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Total Reward')
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'No reward data', ha='center', va='center')
            axes[0, 0].set_title('Episode Rewards')
        
        # Episode lengths
        if 'episode_lengths' in metrics and metrics['episode_lengths']:
            axes[0, 1].plot(metrics['episode_lengths'], 'g-o', linewidth=2, markersize=4)
            axes[0, 1].set_title('Episode Lengths')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Steps')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No length data', ha='center', va='center')
            axes[0, 1].set_title('Episode Lengths')
        
        # Average belief entropy
        if 'belief_entropies' in metrics and metrics['belief_entropies']:
            axes[1, 0].plot(metrics['belief_entropies'], 'm-o', linewidth=2, markersize=4)
            axes[1, 0].set_title('Average Belief Entropy per Episode')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Entropy')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No entropy data', ha='center', va='center')
            axes[1, 0].set_title('Average Belief Entropy per Episode')
        
        # Success rate (cumulative)
        if 'success_rates' in metrics and metrics['success_rates']:
            axes[1, 1].plot(metrics['success_rates'], 'r-o', linewidth=2, markersize=4)
            axes[1, 1].set_title('Cumulative Success Rate')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Success Rate')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No success rate data', ha='center', va='center')
            axes[1, 1].set_title('Cumulative Success Rate')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if self.show_plots:
            plt.show()
        
        return fig
    
    def plot_action_sequence(self, action_sequence: List[int],
                           num_actions: int,
                           title: str = "Action Sequence",
                           save_path: Optional[Path] = None) -> plt.Figure:
        """Plot action selection over time"""
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        time_steps = range(len(action_sequence))
        action_names = [f'Action {i}' for i in range(num_actions)]
        
        # Create action timeline
        ax.step(time_steps, action_sequence, 'g-', linewidth=3, where='post', alpha=0.7)
        ax.scatter(time_steps, action_sequence, c=action_sequence, cmap='tab10', 
                  s=100, edgecolors='black', linewidth=2)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Action')
        ax.set_yticks(range(num_actions))
        ax.set_yticklabels(action_names)
        ax.grid(True, alpha=0.3)
        
        # Add action labels on points
        for t, action in enumerate(action_sequence):
            ax.annotate(f'A{action}', (t, action), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if self.show_plots:
            plt.show()
        
        return fig
    
    def plot_observation_sequence(self, observation_sequence: List[int],
                                num_observations: int,
                                title: str = "Observation Sequence", 
                                save_path: Optional[Path] = None) -> plt.Figure:
        """Plot observation sequence over time"""
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        time_steps = range(len(observation_sequence))
        obs_names = [f'Obs {i}' for i in range(num_observations)]
        
        # Create observation timeline
        ax.step(time_steps, observation_sequence, 'orange', linewidth=3, where='post', alpha=0.7)
        ax.scatter(time_steps, observation_sequence, c=observation_sequence, cmap='plasma', 
                  s=100, edgecolors='black', linewidth=2)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Observation')
        ax.set_yticks(range(num_observations))
        ax.set_yticklabels(obs_names)
        ax.grid(True, alpha=0.3)
        
        # Add observation labels on points
        for t, obs in enumerate(observation_sequence):
            ax.annotate(f'O{obs}', (t, obs), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if self.show_plots:
            plt.show()
        
        return fig
    
    def plot_episode_summary(self, episode_trace: Dict[str, Any], 
                           episode_num: int,
                           save_path: Optional[Path] = None) -> plt.Figure:
        """Create a comprehensive summary plot for an episode"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
        
        num_states = max(episode_trace.get('true_states', [0])) + 1
        num_observations = max(episode_trace.get('observations', [0])) + 1 
        num_actions = max(episode_trace.get('actions', [0])) + 1
        
        # Plot 1: State sequence
        ax1 = fig.add_subplot(gs[0, 0])
        if episode_trace.get('true_states'):
            time_steps = range(len(episode_trace['true_states']))
            ax1.step(time_steps, episode_trace['true_states'], 'b-', linewidth=2, where='post')
            ax1.scatter(time_steps, episode_trace['true_states'], c='blue', s=50)
            ax1.set_title(f'Episode {episode_num}: True States')
            ax1.set_ylabel('State')
            ax1.set_yticks(range(num_states))
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Observations
        ax2 = fig.add_subplot(gs[0, 1])
        if episode_trace.get('observations'):
            time_steps = range(len(episode_trace['observations']))
            ax2.step(time_steps, episode_trace['observations'], 'orange', linewidth=2, where='post')
            ax2.scatter(time_steps, episode_trace['observations'], c='orange', s=50)
            ax2.set_title(f'Episode {episode_num}: Observations')
            ax2.set_ylabel('Observation')
            ax2.set_yticks(range(num_observations))
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Actions
        ax3 = fig.add_subplot(gs[1, 0])
        if episode_trace.get('actions'):
            time_steps = range(len(episode_trace['actions']))
            ax3.step(time_steps, episode_trace['actions'], 'green', linewidth=2, where='post')
            ax3.scatter(time_steps, episode_trace['actions'], c='green', s=50)
            ax3.set_title(f'Episode {episode_num}: Actions')
            ax3.set_ylabel('Action')
            ax3.set_yticks(range(num_actions))
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Rewards
        ax4 = fig.add_subplot(gs[1, 1])
        if episode_trace.get('rewards'):
            time_steps = range(len(episode_trace['rewards']))
            ax4.plot(time_steps, episode_trace['rewards'], 'r-o', linewidth=2, markersize=4)
            ax4.set_title(f'Episode {episode_num}: Rewards')
            ax4.set_ylabel('Reward')
            ax4.grid(True, alpha=0.3)
            
            # Add cumulative reward line
            cumulative_rewards = np.cumsum(episode_trace['rewards'])
            ax4_twin = ax4.twinx()
            ax4_twin.plot(time_steps, cumulative_rewards, 'darkred', linewidth=1, alpha=0.7, linestyle='--')
            ax4_twin.set_ylabel('Cumulative Reward', color='darkred')
        
        # Plot 5: Belief evolution (spanning bottom)
        ax5 = fig.add_subplot(gs[2, :])
        if episode_trace.get('beliefs'):
            belief_matrix = np.array(episode_trace['beliefs']).T
            im = ax5.imshow(belief_matrix, aspect='auto', cmap='viridis', origin='lower')
            ax5.set_title(f'Episode {episode_num}: Belief Evolution')
            ax5.set_xlabel('Time Step')
            ax5.set_ylabel('State')
            ax5.set_yticks(range(num_states))
            ax5.set_yticklabels([f'S{i}' for i in range(num_states)])
            plt.colorbar(im, ax=ax5, label='Belief Probability')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if self.show_plots:
            plt.show()
        
        return fig
    
    def create_comprehensive_visualizations(self, all_traces: List[Dict], 
                                          performance_metrics: Dict[str, List]):
        """Create all visualizations for the simulation"""
        
        # Overall performance metrics
        fig = self.plot_performance_metrics(
            performance_metrics,
            save_path=self.output_dir / 'performance_metrics.png'
        )
        plt.close(fig)
        
        # Individual episode summaries
        for i, trace in enumerate(all_traces[:5]):  # Limit to first 5 episodes
            episode_num = i + 1
            episode_dir = self.output_dir / f"episode_{episode_num}"
            episode_dir.mkdir(exist_ok=True)
            
            # Episode summary
            fig = self.plot_episode_summary(
                trace, episode_num,
                save_path=episode_dir / "episode_summary.png"
            )
            plt.close(fig)
            
            # Individual plots
            if trace.get('true_states'):
                num_states = max(trace['true_states']) + 1
                fig = self.plot_discrete_states(
                    trace['true_states'], num_states,
                    title=f"Episode {episode_num}: State Sequence",
                    save_path=episode_dir / "state_sequence.png"
                )
                plt.close(fig)
            
            if trace.get('beliefs'):
                fig = self.plot_belief_evolution(
                    trace['beliefs'],
                    title=f"Episode {episode_num}: Belief Evolution",
                    save_path=episode_dir / "belief_evolution.png"
                )
                if fig:
                    plt.close(fig)
            
            if trace.get('actions'):
                num_actions = max(trace['actions']) + 1
                fig = self.plot_action_sequence(
                    trace['actions'], num_actions,
                    title=f"Episode {episode_num}: Actions",
                    save_path=episode_dir / "action_sequence.png"
                )
                plt.close(fig)
            
            if trace.get('observations'):
                num_observations = max(trace['observations']) + 1
                fig = self.plot_observation_sequence(
                    trace['observations'], num_observations,
                    title=f"Episode {episode_num}: Observations",
                    save_path=episode_dir / "observation_sequence.png"
                )
                plt.close(fig)


def create_visualizer(output_dir: Path, **kwargs) -> PyMDPVisualizer:
    """Factory function to create a PyMDPVisualizer instance"""
    return PyMDPVisualizer(output_dir, **kwargs)


def save_all_visualizations(visualizer: PyMDPVisualizer, 
                           all_traces: List[Dict],
                           performance_metrics: Dict[str, List]) -> None:
    """Save all visualization outputs for a simulation"""
    visualizer.create_comprehensive_visualizations(all_traces, performance_metrics)


if __name__ == "__main__":
    print("PyMDP Visualizer - Visualization utilities for PyMDP simulations")
    print("This module should be imported and used with the PyMDP simulation class.") 