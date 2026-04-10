#!/usr/bin/env python3
"""
Enhanced bnlearn causal analysis for Hidden Markov Model Baseline
Generated from GNN specification: unknown.md
"""

import bnlearn as bn
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime

class EnhancedHiddenMarkovModelBaselineBnlearnAnalyzer:
    def __init__(self):
        self.model_name = "Hidden Markov Model Baseline"
        self.gnn_source = "unknown.md"
        self.performance_metrics = {}

    def create_and_analyze(self):
        print(f"✅ Starting bnlearn analysis for {self.model_name}")
        
        # Defining the structure based on Active Inference boundaries
        # Hidden State (S) causes Observation (O)
        # Action (A) and previous State (S_prev) cause State (S)
        edges = [('S_prev', 'S'), ('A', 'S'), ('S', 'O')]
        
        # Create DAG
        DAG = bn.make_DAG(edges)
        print("✅ DAG Created Successfully.")
        
        # Generate synthetic trace data based on the structure layout to fit CPDs
        np.random.seed(42)
        n_samples = max(1000, 15 * 20)
        
        # Simulate tabular data for the edges
        print(f"📊 Simulating {n_samples} traces for structure parameter learning...")
        s_prev_data = np.random.randint(0, 2, n_samples)
        a_data = np.random.randint(0, 2, n_samples)
        
        # S depends on S_prev and A
        s_data = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            if a_data[i] == 1:
                s_data[i] = 1 if np.random.rand() > 0.1 else 0
            else:
                s_data[i] = s_prev_data[i]
                
        # O depends on S (Observation model)
        o_data = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            o_data[i] = s_data[i] if np.random.rand() > 0.1 else 1 - s_data[i]
            
        df = pd.DataFrame({
            'S_prev': s_prev_data,
            'A': a_data,
            'S': s_data,
            'O': o_data
        })
        
        # Parameter learning
        model_mle = bn.parameter_learning.fit(DAG, df, methodtype='maximumlikelihood')
        print("✅ Parameter Learning (MLE) successful.")
        
        # Perform inference given an observation
        print("📊 Testing Exact Inference (Junction Tree): Querying P(S=1 | O=1)")
        query = bn.inference.fit(model_mle, variables=['S'], evidence={'O': 1}, verbose=0)
        
        self.performance_metrics = {
            "edges_learned": len(edges),
            "samples_processed": n_samples,
            "inference_success": True
        }
        
        return {
            "metadata": {
                "model_name": self.model_name,
                "framework": "bnlearn",
                "gnn_source": self.gnn_source
            },
            "summary": self.performance_metrics,
            "query_results": str(query.df)
        }

if __name__ == "__main__":
    analyzer = EnhancedHiddenMarkovModelBaselineBnlearnAnalyzer()
    results = analyzer.create_and_analyze()
    print("=" * 50)
    print("✅ bnlearn execution complete.")
    for k, v in results['summary'].items():
        print(f"  {k}: {v}")
    print("=" * 50)
