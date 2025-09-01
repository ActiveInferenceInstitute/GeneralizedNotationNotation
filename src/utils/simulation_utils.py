#!/usr/bin/env python3
"""
Generic simulation utilities for GNN-generated implementations.
Reusable across different GNN inputs and frameworks.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging

class SimulationTracker:
    """Generic tracker for simulation data across different frameworks."""
    
    def __init__(self, model_name: str, framework: str, output_dir: Path):
        self.model_name = model_name
        self.framework = framework
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now().isoformat()
        
        # Create output directories
        (self.output_dir / "simulation_data").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "visualizations").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "traces").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "execution_logs").mkdir(parents=True, exist_ok=True)
        
        # Initialize data storage
        self.data = {
            "metadata": {
                "model_name": model_name,
                "framework": framework,
                "timestamp": self.timestamp,
                "simulation_steps": 0
            },
            "traces": {
                "belief_states": [],
                "actions": [],
                "observations": [],
                "rewards": [],
                "step_timestamps": []
            },
            "matrices": {},
            "summary_stats": {}
        }
        
        # Set up logging
        log_file = self.output_dir / "execution_logs" / f"{model_name}_{framework}_simulation.log"
        self.logger = logging.getLogger(f"{framework}_simulation")
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def log_step(self, step: int, state: Any, action: Any, observation: Any, reward: float):
        """Log a simulation step with all relevant data."""
        step_time = datetime.now().isoformat()
        
        # Convert numpy arrays to lists for JSON serialization
        if hasattr(state, 'tolist'):
            state = state.tolist()
        if hasattr(action, 'tolist'):
            action = action.tolist()
        if hasattr(observation, 'tolist'):
            observation = observation.tolist()
            
        self.data["traces"]["belief_states"].append(state)
        self.data["traces"]["actions"].append(action)
        self.data["traces"]["observations"].append(observation)
        self.data["traces"]["rewards"].append(reward)
        self.data["traces"]["step_timestamps"].append(step_time)
        self.data["metadata"]["simulation_steps"] = step + 1
        
        self.logger.info(f"Step {step}: State={state}, Action={action}, Obs={observation}, Reward={reward}")
        
    def log_matrices(self, matrices: Dict[str, Any]):
        """Log model matrices (A, B, C, D, etc.)."""
        for name, matrix in matrices.items():
            if hasattr(matrix, 'tolist'):
                self.data["matrices"][name] = {
                    "data": matrix.tolist(),
                    "shape": matrix.shape,
                    "dtype": str(matrix.dtype)
                }
            else:
                self.data["matrices"][name] = matrix
        self.logger.info(f"Logged matrices: {list(matrices.keys())}")
        
    def calculate_summary_stats(self):
        """Calculate summary statistics for the simulation."""
        rewards = self.data["traces"]["rewards"]
        if rewards:
            self.data["summary_stats"] = {
                "total_reward": sum(rewards),
                "average_reward": np.mean(rewards),
                "max_reward": max(rewards),
                "min_reward": min(rewards),
                "reward_std": np.std(rewards),
                "total_steps": len(rewards)
            }
        self.logger.info(f"Summary stats: {self.data['summary_stats']}")
        
    def generate_visualizations(self):
        """Generate standard visualizations for any Active Inference simulation."""
        if not self.data["traces"]["rewards"]:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.model_name} - {self.framework} Simulation Results', fontsize=16)
        
        # Plot 1: Reward over time
        axes[0, 0].plot(self.data["traces"]["rewards"], 'b-', linewidth=2)
        axes[0, 0].set_title('Rewards Over Time')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Cumulative reward
        cumulative_rewards = np.cumsum(self.data["traces"]["rewards"])
        axes[0, 1].plot(cumulative_rewards, 'g-', linewidth=2)
        axes[0, 1].set_title('Cumulative Rewards')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Cumulative Reward')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Action distribution
        if self.data["traces"]["actions"]:
            actions = self.data["traces"]["actions"]
            action_counts = {}
            for action in actions:
                action_str = str(action)
                action_counts[action_str] = action_counts.get(action_str, 0) + 1
            
            axes[1, 0].bar(action_counts.keys(), action_counts.values())
            axes[1, 0].set_title('Action Distribution')
            axes[1, 0].set_xlabel('Action')
            axes[1, 0].set_ylabel('Count')
            
        # Plot 4: Observation distribution
        if self.data["traces"]["observations"]:
            observations = self.data["traces"]["observations"]
            obs_counts = {}
            for obs in observations:
                obs_str = str(obs)
                obs_counts[obs_str] = obs_counts.get(obs_str, 0) + 1
                
            axes[1, 1].bar(obs_counts.keys(), obs_counts.values())
            axes[1, 1].set_title('Observation Distribution')
            axes[1, 1].set_xlabel('Observation')
            axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        viz_file = self.output_dir / "visualizations" / f"{self.model_name}_{self.framework}_analysis.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Generated visualization: {viz_file}")
        
        # Generate belief state evolution if available
        self._generate_belief_evolution()
        
    def _generate_belief_evolution(self):
        """Generate belief state evolution visualization if data is available."""
        belief_states = self.data["traces"]["belief_states"]
        if not belief_states or not hasattr(belief_states[0], '__len__'):
            return
            
        try:
            # Convert to numpy array
            belief_array = np.array(belief_states)
            if belief_array.ndim == 2:  # Multiple states over time
                plt.figure(figsize=(12, 8))
                for i in range(belief_array.shape[1]):
                    plt.plot(belief_array[:, i], label=f'State {i}', linewidth=2)
                plt.title(f'{self.model_name} - Belief State Evolution')
                plt.xlabel('Step')
                plt.ylabel('Belief Probability')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                belief_file = self.output_dir / "visualizations" / f"{self.model_name}_belief_evolution.png"
                plt.savefig(belief_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"Generated belief evolution: {belief_file}")
        except Exception as e:
            self.logger.warning(f"Could not generate belief evolution plot: {e}")
            
    def save_data(self):
        """Save all collected data to JSON files."""
        self.calculate_summary_stats()
        
        # Save main data file
        data_file = self.output_dir / "simulation_data" / f"{self.model_name}_{self.framework}_data.json"
        with open(data_file, 'w') as f:
            json.dump(self.data, f, indent=2)
            
        # Save separate trace file for easy analysis
        trace_file = self.output_dir / "traces" / f"{self.model_name}_{self.framework}_traces.json"
        with open(trace_file, 'w') as f:
            json.dump(self.data["traces"], f, indent=2)
            
        self.logger.info(f"Saved data to {data_file} and {trace_file}")
        
    def finalize(self):
        """Finalize the simulation by generating all outputs."""
        self.generate_visualizations()
        self.save_data()
        
        # Create summary report
        summary_file = self.output_dir / f"{self.model_name}_{self.framework}_summary.md"
        with open(summary_file, 'w') as f:
            f.write(f"# {self.model_name} - {self.framework} Simulation Summary\n\n")
            f.write(f"**Generated:** {self.timestamp}\n\n")
            
            if self.data["summary_stats"]:
                f.write("## Summary Statistics\n\n")
                for key, value in self.data["summary_stats"].items():
                    f.write(f"- **{key.replace('_', ' ').title()}:** {value:.4f}\n")
                f.write("\n")
                
            f.write("## Files Generated\n\n")
            f.write("- Simulation data (JSON)\n")
            f.write("- Trace data (JSON) \n")
            f.write("- Analysis visualizations (PNG)\n")
            f.write("- Execution logs\n")
            f.write("- This summary report\n")
            
        self.logger.info(f"Generated summary report: {summary_file}")

class DiagramAnalyzer:
    """Generic analyzer for categorical diagrams and mathematical structures."""
    
    def __init__(self, model_name: str, output_dir: Path):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        
        (self.output_dir / "diagram_outputs").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "analysis").mkdir(parents=True, exist_ok=True)
        
        self.analysis_data = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "diagrams": [],
            "morphisms": [],
            "properties": {}
        }
        
    def log_diagram(self, diagram_name: str, domain: str, codomain: str, properties: Dict[str, Any]):
        """Log a categorical diagram with its properties."""
        diagram_info = {
            "name": diagram_name,
            "domain": str(domain),
            "codomain": str(codomain),
            "properties": properties,
            "timestamp": datetime.now().isoformat()
        }
        self.analysis_data["diagrams"].append(diagram_info)
        
    def log_morphism(self, morphism_name: str, source: str, target: str, composition_info: Dict[str, Any]):
        """Log a morphism with composition information."""
        morphism_info = {
            "name": morphism_name,
            "source": str(source),
            "target": str(target),
            "composition": composition_info,
            "timestamp": datetime.now().isoformat()
        }
        self.analysis_data["morphisms"].append(morphism_info)
        
    def generate_diagram_report(self):
        """Generate a comprehensive report on the categorical structure."""
        report_file = self.output_dir / "analysis" / f"{self.model_name}_diagram_analysis.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# {self.model_name} - Categorical Diagram Analysis\n\n")
            f.write(f"**Generated:** {self.analysis_data['timestamp']}\n\n")
            
            f.write("## Diagrams Created\n\n")
            for diagram in self.analysis_data["diagrams"]:
                f.write(f"### {diagram['name']}\n\n")
                f.write(f"- **Domain:** `{diagram['domain']}`\n")
                f.write(f"- **Codomain:** `{diagram['codomain']}`\n")
                if diagram['properties']:
                    f.write("- **Properties:**\n")
                    for key, value in diagram['properties'].items():
                        f.write(f"  - {key}: {value}\n")
                f.write("\n")
                
            f.write("## Morphisms\n\n")
            for morphism in self.analysis_data["morphisms"]:
                f.write(f"### {morphism['name']}\n\n")
                f.write(f"- **Source:** `{morphism['source']}`\n")
                f.write(f"- **Target:** `{morphism['target']}`\n")
                if morphism['composition']:
                    f.write("- **Composition Details:**\n")
                    for key, value in morphism['composition'].items():
                        f.write(f"  - {key}: {value}\n")
                f.write("\n")
                
        # Save data as JSON too
        data_file = self.output_dir / "diagram_outputs" / f"{self.model_name}_diagrams.json"
        with open(data_file, 'w') as f:
            json.dump(self.analysis_data, f, indent=2)
            
        return report_file
