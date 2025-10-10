#!/usr/bin/env python3
"""
PyMDP Simulation Script for Active Inference Chronic Pain Multi-Theory Model v1

This script was automatically generated from a GNN specification.
It uses the GNN pipeline's PyMDP execution module to run an Active Inference simulation.

Model: Active Inference Chronic Pain Multi-Theory Model v1
Description: 
Generated: 2025-10-10 10:34:21

State Space:
- Hidden States: 3
- Observations: 3 
- Actions: 3
"""

import sys
from pathlib import Path
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.execute.pymdp import execute_pymdp_simulation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main simulation function."""
    
    # GNN Specification (embedded)
    gnn_spec = {
    "name": "Active Inference Chronic Pain Multi-Theory Model v1",
    "model_name": "Active Inference Chronic Pain Multi-Theory Model v1",
    "description": "This model integrates multiple coherent theories of chronic pain mechanisms across THREE NESTED CONTINUOUS TIMESCALES:\n**Multi-Theory Integration:**\n- Peripheral Sensitization: Enhanced nociceptor responsiveness and reduced thresholds (slow timescale)\n- Central Sensitization: Amplified CNS processing and reduced inhibition (slow timescale, one-way process)\n- Gate Control Theory: Spinal modulation of ascending pain signals (fast timescale)\n- Neuromatrix Theory: Distributed network generating pain experience (fast-medium coupling)\n- Predictive Coding: Pain as precision-weighted prediction error (all timescales)\n- Biopsychosocial Integration: Cognitive, emotional, and behavioral factors (medium timescale)\n**Three Nested Timescales:**\n1. Fast (ms-s): Neural signaling, gate control, descending modulation, acute pain perception\n2. Medium (min-hrs): Cognitive-affective processes, behavioral strategies, functional capacity\n3. Slow (hrs-days): Tissue healing, peripheral/central sensitization, chronic adaptations\n**State Space Structure:**\n- Six hidden state factors (378 combinations): tissue state (slow), peripheral sensitivity (slow), spinal gate (fast), central sensitization (slow), descending modulation (fast), cognitive-affective state (medium)\n- Four observation modalities (72 outcomes): pain intensity (fast), pain quality (fast), functional capacity (medium), autonomic response (fast)\n- Four control factors (81 actions): attention allocation (medium), behavioral strategy (medium), cognitive reappraisal (medium), descending control (fast)\n**Key Features:**\n- Timescale separation: \u03b5 (fast/medium) \u2248 10^-3, \u03b4 (medium/slow) \u2248 10^-2\n- Cross-timescale coupling: slow states modulate fast dynamics, fast observations (averaged) drive medium cognition, medium behaviors (averaged) influence slow healing\n- Testable predictions about pain chronification pathways across multiple timescales\n- Intervention targets at each timescale: fast (descending control), medium (CBT/behavioral), slow (prevent sensitization)",
    "model_parameters": {
        "num_hidden_states": 3,
        "num_obs": 3,
        "num_actions": 3
    },
    "initialparameterization": {
        "A": [
            [
                0.7,
                0.25,
                0.05,
                0.0,
                0.8,
                0.15,
                0.05,
                0.9,
                0.08,
                0.02,
                0.85,
                0.15
            ],
            [
                0.6,
                0.3,
                0.1,
                0.0,
                0.75,
                0.2,
                0.05,
                0.85,
                0.1,
                0.05,
                0.8,
                0.2
            ],
            [
                0.45,
                0.35,
                0.15,
                0.05,
                0.7,
                0.2,
                0.1,
                0.75,
                0.15,
                0.1,
                0.7,
                0.3
            ],
            [
                0.2,
                0.3,
                0.35,
                0.15,
                0.5,
                0.3,
                0.2,
                0.5,
                0.3,
                0.2,
                0.4,
                0.6
            ],
            [
                "amplified pain"
            ],
            [
                0.3,
                0.4,
                0.25,
                0.05,
                0.6,
                0.25,
                0.15,
                0.6,
                0.25,
                0.15,
                0.5,
                0.5
            ],
            [
                0.4,
                0.35,
                0.2,
                0.05,
                0.65,
                0.25,
                0.1,
                0.7,
                0.2,
                0.1,
                0.6,
                0.4
            ],
            [
                0.55,
                0.3,
                0.12,
                0.03,
                0.7,
                0.22,
                0.08,
                0.8,
                0.15,
                0.05,
                0.75,
                0.25
            ],
            [
                0.8,
                0.15,
                0.05,
                0.0,
                0.85,
                0.12,
                0.03,
                0.95,
                0.04,
                0.01,
                0.9,
                0.1
            ],
            [
                0.75,
                0.2,
                0.05,
                0.0,
                0.8,
                0.15,
                0.05,
                0.9,
                0.08,
                0.02,
                0.85,
                0.15
            ],
            [
                0.65,
                0.25,
                0.08,
                0.02,
                0.75,
                0.18,
                0.07,
                0.85,
                0.12,
                0.03,
                0.8,
                0.2
            ],
            [
                0.9,
                0.08,
                0.02,
                0.0,
                0.9,
                0.08,
                0.02,
                0.98,
                0.02,
                0.0,
                0.95,
                0.05
            ],
            [
                "gate closed"
            ],
            [
                0.85,
                0.12,
                0.03,
                0.0,
                0.88,
                0.1,
                0.02,
                0.95,
                0.04,
                0.01,
                0.92,
                0.08
            ],
            [
                0.05,
                0.15,
                0.45,
                0.35,
                0.3,
                0.4,
                0.3,
                0.2,
                0.4,
                0.4,
                0.25,
                0.75
            ],
            [
                0.02,
                0.1,
                0.4,
                0.48,
                0.2,
                0.35,
                0.45,
                0.1,
                0.35,
                0.55,
                0.15,
                0.85
            ],
            [
                0.35,
                0.4,
                0.2,
                0.05,
                0.7,
                0.25,
                0.05,
                0.7,
                0.2,
                0.1,
                0.75,
                0.25
            ],
            [
                0.2,
                0.35,
                0.3,
                0.15,
                0.55,
                0.3,
                0.15,
                0.55,
                0.3,
                0.15,
                0.6,
                0.4
            ],
            [
                0.15,
                0.3,
                0.4,
                0.15,
                0.4,
                0.35,
                0.25,
                0.4,
                0.35,
                0.25,
                0.45,
                0.55
            ],
            [
                0.05,
                0.2,
                0.45,
                0.3,
                0.3,
                0.4,
                0.3,
                0.3,
                0.4,
                0.3,
                0.35,
                0.65
            ],
            [
                0.4,
                0.4,
                0.15,
                0.05,
                0.75,
                0.2,
                0.05,
                0.8,
                0.15,
                0.05,
                0.8,
                0.2
            ],
            [
                0.25,
                0.4,
                0.25,
                0.1,
                0.65,
                0.25,
                0.1,
                0.65,
                0.25,
                0.1,
                0.7,
                0.3
            ],
            [
                0.2,
                0.35,
                0.3,
                0.15,
                0.55,
                0.3,
                0.15,
                0.5,
                0.3,
                0.2,
                0.6,
                0.4
            ],
            [
                0.08,
                0.25,
                0.4,
                0.27,
                0.4,
                0.35,
                0.25,
                0.35,
                0.4,
                0.25,
                0.45,
                0.55
            ],
            [
                0.05,
                0.2,
                0.4,
                0.35,
                0.3,
                0.4,
                0.3,
                0.25,
                0.45,
                0.3,
                0.35,
                0.65
            ],
            [
                0.02,
                0.1,
                0.35,
                0.53,
                0.2,
                0.35,
                0.45,
                0.15,
                0.45,
                0.4,
                0.2,
                0.8
            ],
            [
                0.02,
                0.08,
                0.3,
                0.6,
                0.15,
                0.3,
                0.55,
                0.1,
                0.4,
                0.5,
                0.15,
                0.85
            ],
            [
                "all risk factors"
            ],
            [
                0.1,
                0.3,
                0.4,
                0.2,
                0.5,
                0.35,
                0.15,
                0.45,
                0.35,
                0.2,
                0.5,
                0.5
            ],
            [
                0.05,
                0.25,
                0.45,
                0.25,
                0.4,
                0.4,
                0.2,
                0.4,
                0.4,
                0.2,
                0.45,
                0.55
            ],
            [
                0.03,
                0.15,
                0.42,
                0.4,
                0.3,
                0.4,
                0.3,
                0.3,
                0.45,
                0.25,
                0.35,
                0.65
            ],
            [
                0.01,
                0.08,
                0.35,
                0.56,
                0.2,
                0.35,
                0.45,
                0.2,
                0.45,
                0.35,
                0.25,
                0.75
            ],
            [
                0.01,
                0.05,
                0.3,
                0.64,
                0.15,
                0.3,
                0.55,
                0.15,
                0.5,
                0.35,
                0.2,
                0.8
            ],
            [
                0.0,
                0.02,
                0.2,
                0.78,
                0.1,
                0.25,
                0.65,
                0.08,
                0.45,
                0.47,
                0.1,
                0.9
            ],
            [
                0.7,
                0.25,
                0.05,
                0.0,
                0.8,
                0.15,
                0.05,
                0.8,
                0.15,
                0.05,
                0.85,
                0.15
            ],
            [
                0.55,
                0.35,
                0.08,
                0.02,
                0.75,
                0.2,
                0.05,
                0.75,
                0.2,
                0.05,
                0.8,
                0.2
            ],
            [
                0.15,
                0.3,
                0.4,
                0.15,
                0.45,
                0.35,
                0.2,
                0.5,
                0.35,
                0.15,
                0.55,
                0.45
            ],
            [
                0.05,
                0.2,
                0.45,
                0.3,
                0.3,
                0.4,
                0.3,
                0.35,
                0.4,
                0.25,
                0.4,
                0.6
            ]
        ],
        "B": [
            ""
        ],
        "C": [
            2.2,
            1.4,
            0.5,
            0.7,
            -0.1,
            -0.8,
            -0.5,
            -1.3,
            -2.2,
            -1.7,
            -2.5,
            -3.4,
            2.0,
            1.2,
            0.3,
            0.5,
            -0.3,
            -1.0,
            -0.7,
            -1.5,
            -2.4,
            -1.9,
            -2.7,
            -3.6,
            1.6,
            0.8,
            -0.1,
            0.1,
            -0.7,
            -1.4,
            -1.1,
            -1.9,
            -2.8,
            -2.3,
            -3.1,
            -4.0,
            1.0,
            0.2,
            -0.7,
            -0.5,
            -1.3,
            -2.0,
            -1.7,
            -2.5,
            -3.4,
            -2.9,
            -3.7,
            -4.6,
            0.5,
            -0.3,
            -1.2,
            -1.0,
            -1.8,
            -2.5,
            -2.2,
            -3.0,
            -3.9,
            -3.4,
            -4.2,
            -5.1,
            -0.2,
            -1.0,
            -1.9,
            -1.7,
            -2.5,
            -3.2,
            -2.9,
            -3.7,
            -4.6,
            -4.1,
            -4.9,
            -5.8
        ],
        "D": [
            [
                0.1,
                0.5,
                0.4,
                "# T distribution (healed",
                "inflamed",
                "damaged) 0.60",
                0.3,
                0.1,
                "# P_sens distribution (normal",
                "moderate",
                "severe) 0.40",
                0.4,
                0.2,
                "# G distribution (open",
                "modulated",
                "closed) 0.90",
                0.1,
                "# C_sens distribution (absent",
                "present) 0.20",
                0.6,
                0.2,
                "# D_mod distribution (facilitation",
                "neutral",
                "inhibition) 0.30",
                0.2,
                0.15,
                0.1,
                0.1,
                0.1,
                0.05
            ],
            [
                "adaptive \u2192 alexithymic"
            ]
        ],
        "E": [
            0.015,
            0.023,
            0.015,
            0.025,
            0.038,
            0.025,
            0.015,
            0.023,
            0.015,
            0.03,
            0.045,
            0.03,
            0.05,
            0.075,
            0.05,
            0.03,
            0.045,
            0.03,
            0.015,
            0.023,
            0.015,
            0.025,
            0.038,
            0.025,
            0.015,
            0.023,
            0.015,
            0.01,
            0.015,
            0.01,
            0.017,
            0.025,
            0.017,
            0.01,
            0.015,
            0.01,
            0.02,
            0.03,
            0.02,
            0.033,
            0.05,
            0.033,
            0.02,
            0.03,
            0.02,
            0.01,
            0.015,
            0.01,
            0.017,
            0.025,
            0.017,
            0.01,
            0.015,
            0.01,
            0.006,
            0.009,
            0.006,
            0.01,
            0.015,
            0.01,
            0.006,
            0.009,
            0.006,
            0.012,
            0.018,
            0.012,
            0.02,
            0.03,
            0.02,
            0.012,
            0.018,
            0.012,
            0.006,
            0.009,
            0.006,
            0.01,
            0.015,
            0.01,
            0.006,
            0.009,
            0.006
        ]
    },
    "variables": [
        {
            "name": "A",
            "dimensions": [
                72,
                378
            ],
            "type": "float",
            "comment": "72 observations x 378 hidden state combinations"
        },
        {
            "name": "B",
            "dimensions": [
                378,
                378,
                81
            ],
            "type": "float",
            "comment": "State transitions given previous state and action"
        },
        {
            "name": "D",
            "dimensions": [
                378
            ],
            "type": "float",
            "comment": "Prior over initial hidden states (acute vs chronic pain)"
        },
        {
            "name": "T",
            "dimensions": [
                3,
                1
            ],
            "type": "float",
            "comment": "Tissue state (3 levels: healed, inflamed, damaged)"
        },
        {
            "name": "P_sens",
            "dimensions": [
                3,
                1
            ],
            "type": "float",
            "comment": "Peripheral sensitization (3 levels: normal, moderate, severe)"
        },
        {
            "name": "G",
            "dimensions": [
                3,
                1
            ],
            "type": "float",
            "comment": "Spinal gate state (3 levels: open, modulated, closed)"
        },
        {
            "name": "C_sens",
            "dimensions": [
                2,
                1
            ],
            "type": "float",
            "comment": "Central sensitization (2 levels: absent, present)"
        },
        {
            "name": "D_mod",
            "dimensions": [
                3,
                1
            ],
            "type": "float",
            "comment": "Descending modulation (3 levels: facilitation, neutral, inhibition)"
        },
        {
            "name": "Cog",
            "dimensions": [
                7,
                1
            ],
            "type": "float",
            "comment": "Cognitive-affective state (7 levels: adaptive, vigilant, fearful, catastrophizing, depressed, anxious, alexithymic)"
        },
        {
            "name": "Pain_I",
            "dimensions": [
                4,
                1
            ],
            "type": "float",
            "comment": "Pain intensity (4 levels: none, mild, moderate, severe)"
        },
        {
            "name": "Pain_Q",
            "dimensions": [
                3,
                1
            ],
            "type": "float",
            "comment": "Pain quality (3 levels: nociceptive, neuropathic, nociplastic)"
        },
        {
            "name": "Func",
            "dimensions": [
                3,
                1
            ],
            "type": "float",
            "comment": "Functional capacity (3 levels: full, limited, disabled)"
        },
        {
            "name": "Auto",
            "dimensions": [
                2,
                1
            ],
            "type": "float",
            "comment": "Autonomic response (2 levels: normal, hyperarousal)"
        },
        {
            "name": "F",
            "dimensions": [
                "T"
            ],
            "type": "float",
            "comment": "Variational Free Energy for state inference"
        },
        {
            "name": "t_fast",
            "dimensions": [
                1
            ],
            "type": "float",
            "comment": "Fast timescale (milliseconds to seconds: neural responses, acute signaling)"
        },
        {
            "name": "t_medium",
            "dimensions": [
                1
            ],
            "type": "float",
            "comment": "Medium timescale (minutes to hours: cognitive-affective, behavioral adaptation)"
        },
        {
            "name": "t_slow",
            "dimensions": [
                1
            ],
            "type": "float",
            "comment": "Slow timescale (hours to days: tissue healing, sensitization, chronic changes)"
        },
        {
            "name": "C",
            "dimensions": [
                72
            ],
            "type": "float",
            "comment": "Log-preferences over pain-related observations"
        },
        {
            "name": "E",
            "dimensions": [
                81
            ],
            "type": "float",
            "comment": "Initial policy prior over pain coping strategies"
        },
        {
            "name": "Attn",
            "dimensions": [
                3,
                1
            ],
            "type": "float",
            "comment": "Attention allocation (3 actions: distraction, monitoring, catastrophizing)"
        },
        {
            "name": "Behav",
            "dimensions": [
                3,
                1
            ],
            "type": "float",
            "comment": "Behavioral strategy (3 actions: avoidance, pacing, engagement)"
        },
        {
            "name": "Reapp",
            "dimensions": [
                3,
                1
            ],
            "type": "float",
            "comment": "Cognitive reappraisal (3 actions: negative, neutral, positive)"
        },
        {
            "name": "Desc_C",
            "dimensions": [
                3,
                1
            ],
            "type": "float",
            "comment": "Descending control (3 actions: low, moderate, high endogenous analgesia)"
        },
        {
            "name": "G",
            "dimensions": [
                "Attn"
            ],
            "type": "float",
            "comment": "Expected Free Energy (per policy)"
        }
    ],
    "connections": [
        {
            "source": "D",
            "relation": ">",
            "target": "T"
        },
        {
            "source": "D",
            "relation": ">",
            "target": "P_sens"
        },
        {
            "source": "D",
            "relation": ">",
            "target": "C_sens"
        },
        {
            "source": "D",
            "relation": ">",
            "target": "Cog"
        },
        {
            "source": "T",
            "relation": ">",
            "target": "P_sens"
        },
        {
            "source": "P_sens",
            "relation": ">",
            "target": "B"
        },
        {
            "source": "P_sens",
            "relation": ">",
            "target": "G"
        },
        {
            "source": "D_mod",
            "relation": ">",
            "target": "G"
        },
        {
            "source": "G",
            "relation": ">",
            "target": "B"
        },
        {
            "source": "G",
            "relation": ">",
            "target": "C_sens"
        },
        {
            "source": "C_sens",
            "relation": ">",
            "target": "B"
        },
        {
            "source": "T",
            "relation": ">",
            "target": "C_sens"
        },
        {
            "source": "Cog",
            "relation": ">",
            "target": "D_mod"
        },
        {
            "source": "D_mod",
            "relation": ">",
            "target": "B"
        },
        {
            "source": "Pain_I",
            "relation": ">",
            "target": "Cog"
        },
        {
            "source": "Func",
            "relation": ">",
            "target": "Cog"
        },
        {
            "source": "Cog",
            "relation": ">",
            "target": "B"
        },
        {
            "source": "T",
            "relation": ">",
            "target": "A"
        },
        {
            "source": "P_sens",
            "relation": ">",
            "target": "A"
        },
        {
            "source": "G",
            "relation": ">",
            "target": "A"
        },
        {
            "source": "C_sens",
            "relation": ">",
            "target": "A"
        },
        {
            "source": "D_mod",
            "relation": ">",
            "target": "A"
        },
        {
            "source": "Cog",
            "relation": ">",
            "target": "A"
        },
        {
            "source": "Attn",
            "relation": ">",
            "target": "Cog"
        },
        {
            "source": "Behav",
            "relation": ">",
            "target": "T"
        },
        {
            "source": "Behav",
            "relation": ">",
            "target": "Func"
        },
        {
            "source": "Reapp",
            "relation": ">",
            "target": "Cog"
        },
        {
            "source": "Desc_C",
            "relation": ">",
            "target": "D_mod"
        },
        {
            "source": "C",
            "relation": ">",
            "target": "G"
        },
        {
            "source": "E",
            "relation": ">",
            "target": "Attn"
        },
        {
            "source": "E",
            "relation": ">",
            "target": "Behav"
        },
        {
            "source": "G",
            "relation": ">",
            "target": "Attn"
        },
        {
            "source": "G",
            "relation": ">",
            "target": "Behav"
        },
        {
            "source": "G",
            "relation": ">",
            "target": "Reapp"
        },
        {
            "source": "G",
            "relation": ">",
            "target": "Desc_C"
        },
        {
            "source": "Attn",
            "relation": ">",
            "target": "Behav"
        },
        {
            "source": "Behav",
            "relation": ">",
            "target": "Reapp"
        },
        {
            "source": "Reapp",
            "relation": ">",
            "target": "Desc_C"
        }
    ],
    "ontology_mapping": {
        "A": "LikelihoodMatrix",
        "B": "TransitionMatrices",
        "C": "LogPreferenceVector",
        "D": "PriorOverHiddenStates",
        "E": "HabitVector",
        "F": "VariationalFreeEnergy",
        "G": "SpinalGateState",
        "T": "TissueState",
        "P_sens": "PeripheralSensitization",
        "C_sens": "CentralSensitization",
        "D_mod": "DescendingModulation",
        "Cog": "CognitiveAffectiveState",
        "Pain_I": "PainIntensityObservation",
        "Pain_Q": "PainQualityObservation",
        "Func": "FunctionalCapacityObservation",
        "Auto": "AutonomicResponseObservation",
        "Attn": "AttentionAllocationControl",
        "Behav": "BehavioralStrategyControl",
        "Reapp": "CognitiveReappraisalControl",
        "Desc_C": "DescendingControlAction",
        "t_fast": "FastTimescale",
        "t_medium": "MediumTimescale",
        "t_slow": "SlowTimescale"
    }
}
    
    # Configuration parameters (can be modified)
    num_episodes = 10
    verbose_output = True
    
    # Output directory
    output_dir = Path("output") / "pymdp_simulations" / "Active Inference Chronic Pain Multi-Theory Model v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting PyMDP simulation for Active Inference Chronic Pain Multi-Theory Model v1")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Episodes: {num_episodes}")
    
    # Run simulation
    try:
        success, results = execute_pymdp_simulation(
            gnn_spec=gnn_spec,
            output_dir=output_dir,
            num_episodes=num_episodes,
            verbose=verbose_output
        )
        
        if success:
            logger.info("✓ Simulation completed successfully!")
            logger.info(f"Results summary:")
            logger.info(f"  Episodes: {results.get('total_episodes', 'N/A')}")
            logger.info(f"  Success Rate: {results.get('success_rate', 0):.2%}")
            logger.info(f"  Output: {results.get('output_directory', output_dir)}")
            return 0
        else:
            logger.error("✗ Simulation failed!")
            logger.error(f"Error: {results.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
