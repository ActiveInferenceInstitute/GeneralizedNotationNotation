# GNN Rendering Results

Generated: 2026-04-09T14:51:18.032758
Processing Type: **POMDP-aware rendering**

## Summary

- **Total Files**: 9
- **Successfully Processed**: 9
- **Failed**: 0
- **Framework Renderings**: 66/72 (91.7% success rate)

## Configuration

- **Frameworks**: all
- **Strict Validation**: True
- **Verbose**: True
- **POMDP Processing**: ✅ Available

## File Results

- ✅ **simple_mdp.md** - Successfully processed
  - ✅ pymdp: Generated pymdp 1.0.0 runner: output_test/11_render_output/simple_mdp/pymdp/Simple MDP Agent_pymdp.py
  - ✅ rxinfer: Generated RxInfer.jl simulation script: output_test/11_render_output/simple_mdp/rxinfer/Simple MDP Agent_rxinfer.jl
  - ✅ activeinference_jl: Successfully rendered ActiveInference.jl script to Simple MDP Agent_activeinference.jl
  - ✅ jax: JAX model generated successfully.
  - ✅ discopy: Generated DisCoPy categorical diagram script: output_test/11_render_output/simple_mdp/discopy/Simple MDP Agent_discopy.py
  - ✅ pytorch: PyTorch script generated: output_test/11_render_output/simple_mdp/pytorch/Simple MDP Agent_pytorch.py
  - ✅ numpyro: NumPyro script generated: output_test/11_render_output/simple_mdp/numpyro/Simple MDP Agent_numpyro.py
  - ✅ bnlearn: bnlearn code generated
- ✅ **multi_armed_bandit.md** - Successfully processed
  - ✅ pymdp: Generated pymdp 1.0.0 runner: output_test/11_render_output/multi_armed_bandit/pymdp/Multi Armed Bandit Agent_pymdp.py
  - ✅ rxinfer: Generated RxInfer.jl simulation script: output_test/11_render_output/multi_armed_bandit/rxinfer/Multi Armed Bandit Agent_rxinfer.jl
  - ✅ activeinference_jl: Successfully rendered ActiveInference.jl script to Multi Armed Bandit Agent_activeinference.jl
  - ✅ jax: JAX model generated successfully.
  - ✅ discopy: Generated DisCoPy categorical diagram script: output_test/11_render_output/multi_armed_bandit/discopy/Multi Armed Bandit Agent_discopy.py
  - ✅ pytorch: PyTorch script generated: output_test/11_render_output/multi_armed_bandit/pytorch/Multi Armed Bandit Agent_pytorch.py
  - ✅ numpyro: NumPyro script generated: output_test/11_render_output/multi_armed_bandit/numpyro/Multi Armed Bandit Agent_numpyro.py
  - ✅ bnlearn: bnlearn code generated
- ✅ **deep_planning_horizon.md** - Successfully processed
  - ✅ pymdp: Generated pymdp 1.0.0 runner: output_test/11_render_output/deep_planning_horizon/pymdp/Deep Planning Horizon POMDP_pymdp.py
  - ✅ rxinfer: Generated RxInfer.jl simulation script: output_test/11_render_output/deep_planning_horizon/rxinfer/Deep Planning Horizon POMDP_rxinfer.jl
  - ✅ activeinference_jl: Successfully rendered ActiveInference.jl script to Deep Planning Horizon POMDP_activeinference.jl
  - ✅ jax: JAX model generated successfully.
  - ✅ discopy: Generated DisCoPy categorical diagram script: output_test/11_render_output/deep_planning_horizon/discopy/Deep Planning Horizon POMDP_discopy.py
  - ✅ pytorch: PyTorch script generated: output_test/11_render_output/deep_planning_horizon/pytorch/Deep Planning Horizon POMDP_pytorch.py
  - ✅ numpyro: NumPyro script generated: output_test/11_render_output/deep_planning_horizon/numpyro/Deep Planning Horizon POMDP_numpyro.py
  - ✅ bnlearn: bnlearn code generated
- ✅ **actinf_pomdp_agent.md** - Successfully processed
  - ✅ pymdp: Generated pymdp 1.0.0 runner: output_test/11_render_output/actinf_pomdp_agent/pymdp/Active Inference POMDP Agent_pymdp.py
  - ✅ rxinfer: Generated RxInfer.jl simulation script: output_test/11_render_output/actinf_pomdp_agent/rxinfer/Active Inference POMDP Agent_rxinfer.jl
  - ✅ activeinference_jl: Successfully rendered ActiveInference.jl script to Active Inference POMDP Agent_activeinference.jl
  - ✅ jax: JAX model generated successfully.
  - ✅ discopy: Generated DisCoPy categorical diagram script: output_test/11_render_output/actinf_pomdp_agent/discopy/Active Inference POMDP Agent_discopy.py
  - ✅ pytorch: PyTorch script generated: output_test/11_render_output/actinf_pomdp_agent/pytorch/Active Inference POMDP Agent_pytorch.py
  - ✅ numpyro: NumPyro script generated: output_test/11_render_output/actinf_pomdp_agent/numpyro/Active Inference POMDP Agent_numpyro.py
  - ✅ bnlearn: bnlearn code generated
- ✅ **hmm_baseline.md** - Successfully processed
  - ✅ pymdp: Generated pymdp 1.0.0 runner: output_test/11_render_output/hmm_baseline/pymdp/Hidden Markov Model Baseline_pymdp.py
  - ✅ rxinfer: Generated RxInfer.jl simulation script: output_test/11_render_output/hmm_baseline/rxinfer/Hidden Markov Model Baseline_rxinfer.jl
  - ✅ activeinference_jl: Successfully rendered ActiveInference.jl script to Hidden Markov Model Baseline_activeinference.jl
  - ✅ jax: JAX model generated successfully.
  - ✅ discopy: Generated DisCoPy categorical diagram script: output_test/11_render_output/hmm_baseline/discopy/Hidden Markov Model Baseline_discopy.py
  - ✅ pytorch: PyTorch script generated: output_test/11_render_output/hmm_baseline/pytorch/Hidden Markov Model Baseline_pytorch.py
  - ✅ numpyro: NumPyro script generated: output_test/11_render_output/hmm_baseline/numpyro/Hidden Markov Model Baseline_numpyro.py
  - ✅ bnlearn: bnlearn code generated
- ✅ **bnlearn_causal_model.md** - Successfully processed
  - ❌ pymdp: POMDP not compatible with pymdp: Missing required matrices: ['B']
  - ❌ rxinfer: POMDP not compatible with rxinfer: Missing required matrices: ['B']
  - ❌ activeinference_jl: POMDP not compatible with activeinference_jl: Missing required matrices: ['B']
  - ❌ jax: POMDP not compatible with jax: Missing required matrices: ['B']
  - ✅ discopy: Generated DisCoPy categorical diagram script: output_test/11_render_output/bnlearn_causal_model/discopy/Bnlearn Causal Model_discopy.py
  - ❌ pytorch: POMDP not compatible with pytorch: Missing required matrices: ['B']
  - ❌ numpyro: POMDP not compatible with numpyro: Missing required matrices: ['B']
  - ✅ bnlearn: bnlearn code generated
- ✅ **two_state_bistable.md** - Successfully processed
  - ✅ pymdp: Generated pymdp 1.0.0 runner: output_test/11_render_output/two_state_bistable/pymdp/Two State Bistable POMDP_pymdp.py
  - ✅ rxinfer: Generated RxInfer.jl simulation script: output_test/11_render_output/two_state_bistable/rxinfer/Two State Bistable POMDP_rxinfer.jl
  - ✅ activeinference_jl: Successfully rendered ActiveInference.jl script to Two State Bistable POMDP_activeinference.jl
  - ✅ jax: JAX model generated successfully.
  - ✅ discopy: Generated DisCoPy categorical diagram script: output_test/11_render_output/two_state_bistable/discopy/Two State Bistable POMDP_discopy.py
  - ✅ pytorch: PyTorch script generated: output_test/11_render_output/two_state_bistable/pytorch/Two State Bistable POMDP_pytorch.py
  - ✅ numpyro: NumPyro script generated: output_test/11_render_output/two_state_bistable/numpyro/Two State Bistable POMDP_numpyro.py
  - ✅ bnlearn: bnlearn code generated
- ✅ **markov_chain.md** - Successfully processed
  - ✅ pymdp: Generated pymdp 1.0.0 runner: output_test/11_render_output/markov_chain/pymdp/Simple Markov Chain_pymdp.py
  - ✅ rxinfer: Generated RxInfer.jl simulation script: output_test/11_render_output/markov_chain/rxinfer/Simple Markov Chain_rxinfer.jl
  - ✅ activeinference_jl: Successfully rendered ActiveInference.jl script to Simple Markov Chain_activeinference.jl
  - ✅ jax: JAX model generated successfully.
  - ✅ discopy: Generated DisCoPy categorical diagram script: output_test/11_render_output/markov_chain/discopy/Simple Markov Chain_discopy.py
  - ✅ pytorch: PyTorch script generated: output_test/11_render_output/markov_chain/pytorch/Simple Markov Chain_pytorch.py
  - ✅ numpyro: NumPyro script generated: output_test/11_render_output/markov_chain/numpyro/Simple Markov Chain_numpyro.py
  - ✅ bnlearn: bnlearn code generated
- ✅ **tmaze_epistemic.md** - Successfully processed
  - ✅ pymdp: Generated pymdp 1.0.0 runner: output_test/11_render_output/tmaze_epistemic/pymdp/T-Maze Epistemic Foraging Agent_pymdp.py
  - ✅ rxinfer: Generated RxInfer.jl simulation script: output_test/11_render_output/tmaze_epistemic/rxinfer/T-Maze Epistemic Foraging Agent_rxinfer.jl
  - ✅ activeinference_jl: Successfully rendered ActiveInference.jl script to T-Maze Epistemic Foraging Agent_activeinference.jl
  - ✅ jax: JAX model generated successfully.
  - ✅ discopy: Generated DisCoPy categorical diagram script: output_test/11_render_output/tmaze_epistemic/discopy/T-Maze Epistemic Foraging Agent_discopy.py
  - ✅ pytorch: PyTorch script generated: output_test/11_render_output/tmaze_epistemic/pytorch/T-Maze Epistemic Foraging Agent_pytorch.py
  - ✅ numpyro: NumPyro script generated: output_test/11_render_output/tmaze_epistemic/numpyro/T-Maze Epistemic Foraging Agent_numpyro.py
  - ✅ bnlearn: bnlearn code generated


## Output Structure

The rendered files are organized in implementation-specific subfolders:

```
output_test/11_render_output/
├── [model_name]/
│   ├── pymdp/              # PyMDP Python simulations
│   ├── rxinfer/            # RxInfer.jl Julia simulations
│   ├── activeinference_jl/ # ActiveInference.jl Julia simulations
│   ├── jax/                # JAX Python simulations
│   └── discopy/            # DisCoPy categorical diagrams
└── render_processing_summary.json  # Detailed results
```

## Generated Files

Each framework subdirectory contains:
- Main simulation/diagram script
- Framework-specific README.md with model details
- Configuration files (if applicable)

## Next Steps

1. Navigate to specific framework directories to find generated code
2. Follow framework-specific READMEs for execution instructions  
3. Check the processing summary JSON for detailed results and any warnings

---

*Generated by GNN Render Processor v1.0*
