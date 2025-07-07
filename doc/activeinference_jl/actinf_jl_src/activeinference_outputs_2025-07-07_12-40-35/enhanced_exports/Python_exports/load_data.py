#!/usr/bin/env python3
'''
ActiveInference.jl Python Data Import Module
Generated: 2025-07-07T12:44:11.691
'''

import numpy as np
import pandas as pd
import json
from pathlib import Path

def load_activeinference_data(data_dir="/Users/4d/Documents/GitHub/GeneralizedNotationNotation/doc/activeinference_jl/actinf_jl_src/activeinference_outputs_2025-07-07_12-40-35/enhanced_exports/Python_exports"):
    """Load ActiveInference.jl exported data."""
    data_dict = {}
    data_dir = Path(data_dir)
    
    # Load learning_curve
    try:
        learning_curve = pd.read_csv(data_dir / "learning_curve.csv")
        data_dict["learning_curve"] = learning_curve
    except Exception as e:
        print(f"Failed to load learning_curve: {e}")
    
    # Load actions_trace
    try:
        actions_trace = pd.read_csv(data_dir / "actions_trace.csv")
        data_dict["actions_trace"] = actions_trace
    except Exception as e:
        print(f"Failed to load actions_trace: {e}")
    
    # Load rewards_trace
    try:
        rewards_trace = pd.read_csv(data_dir / "rewards_trace.csv")
        data_dict["rewards_trace"] = rewards_trace
    except Exception as e:
        print(f"Failed to load rewards_trace: {e}")
    
    # Load planning_rewards
    try:
        planning_rewards = pd.read_csv(data_dir / "planning_rewards.csv")
        data_dict["planning_rewards"] = planning_rewards
    except Exception as e:
        print(f"Failed to load planning_rewards: {e}")
    
    # Load beliefs_trace
    try:
        beliefs_trace = pd.read_csv(data_dir / "beliefs_trace.csv")
        data_dict["beliefs_trace"] = beliefs_trace
    except Exception as e:
        print(f"Failed to load beliefs_trace: {e}")
    
    # Load beliefs_over_time
    try:
        beliefs_over_time = pd.read_csv(data_dir / "beliefs_over_time.csv")
        data_dict["beliefs_over_time"] = beliefs_over_time
    except Exception as e:
        print(f"Failed to load beliefs_over_time: {e}")
    
    # Load learning_comparison
    try:
        learning_comparison = pd.read_csv(data_dir / "learning_comparison.csv")
        data_dict["learning_comparison"] = learning_comparison
    except Exception as e:
        print(f"Failed to load learning_comparison: {e}")
    
    # Load planning_actions
    try:
        planning_actions = pd.read_csv(data_dir / "planning_actions.csv")
        data_dict["planning_actions"] = planning_actions
    except Exception as e:
        print(f"Failed to load planning_actions: {e}")
    
    # Load actions_per_trial_trace
    try:
        actions_per_trial_trace = pd.read_csv(data_dir / "actions_per_trial_trace.csv")
        data_dict["actions_per_trial_trace"] = actions_per_trial_trace
    except Exception as e:
        print(f"Failed to load actions_per_trial_trace: {e}")
    
    # Load basic_simulation_trace
    try:
        basic_simulation_trace = pd.read_csv(data_dir / "basic_simulation_trace.csv")
        data_dict["basic_simulation_trace"] = basic_simulation_trace
    except Exception as e:
        print(f"Failed to load basic_simulation_trace: {e}")
    
    # Load learning_comparison_trace
    try:
        learning_comparison_trace = pd.read_csv(data_dir / "learning_comparison_trace.csv")
        data_dict["learning_comparison_trace"] = learning_comparison_trace
    except Exception as e:
        print(f"Failed to load learning_comparison_trace: {e}")
    
    # Load actions_over_time
    try:
        actions_over_time = pd.read_csv(data_dir / "actions_over_time.csv")
        data_dict["actions_over_time"] = actions_over_time
    except Exception as e:
        print(f"Failed to load actions_over_time: {e}")
    
    # Load observations_over_time
    try:
        observations_over_time = pd.read_csv(data_dir / "observations_over_time.csv")
        data_dict["observations_over_time"] = observations_over_time
    except Exception as e:
        print(f"Failed to load observations_over_time: {e}")
    
    # Load learning_trace
    try:
        learning_trace = pd.read_csv(data_dir / "learning_trace.csv")
        data_dict["learning_trace"] = learning_trace
    except Exception as e:
        print(f"Failed to load learning_trace: {e}")
    
    # Load observations_trace
    try:
        observations_trace = pd.read_csv(data_dir / "observations_trace.csv")
        data_dict["observations_trace"] = observations_trace
    except Exception as e:
        print(f"Failed to load observations_trace: {e}")
    
    # Load planning_trace
    try:
        planning_trace = pd.read_csv(data_dir / "planning_trace.csv")
        data_dict["planning_trace"] = planning_trace
    except Exception as e:
        print(f"Failed to load planning_trace: {e}")
    
    return data_dict

def summary_statistics(data_dict):
    """Compute summary statistics for all datasets."""
    for name, df in data_dict.items():
        print(f"\n{name}:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        if df.select_dtypes(include=[np.number]).shape[1] > 0:
            print(f"  Numeric summary:")
            print(df.describe().to_string(max_cols=5))

if __name__ == "__main__":
    # Load data
    data = load_activeinference_data()
    print(f"Loaded {len(data)} datasets")
    
    # Show summary
    summary_statistics(data)
