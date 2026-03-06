# Research Hypotheses Report
**Analysis mode**: rule_based

## simple_mdp.md (pomdp model)
### High Priority
- **precision_modulation**: Add precision parameters to modulate sensory and policy uncertainty
  - *Rationale*: The model lacks precision parameters. Active Inference with precision weighting better captures attentional modulation and epistemic confidence.
### Medium Priority
- **connectivity_enrichment**: Investigate potential missing causal links between model components
  - *Rationale*: Graph density is 0.14 (10 connections, 9 variables). Sparse graphs may miss important dependencies.
- **planning_horizon**: Define explicit planning horizon T for tractable policy inference
  - *Rationale*: Unbounded time horizon requires truncation for policy selection. Setting T=3-5 enables efficient Expected Free Energy computation.
- **parameter_learning**: Add Dirichlet concentration parameters for online model learning
  - *Rationale*: Static A and B matrices cannot adapt. Adding concentration parameters (a, b) enables Bayesian learning from experience.

## multi_armed_bandit.md (pomdp model)
### High Priority
- **precision_modulation**: Add precision parameters to modulate sensory and policy uncertainty
  - *Rationale*: The model lacks precision parameters. Active Inference with precision weighting better captures attentional modulation and epistemic confidence.
### Medium Priority
- **connectivity_enrichment**: Investigate potential missing causal links between model components
  - *Rationale*: Graph density is 0.14 (10 connections, 9 variables). Sparse graphs may miss important dependencies.
- **planning_horizon**: Define explicit planning horizon T for tractable policy inference
  - *Rationale*: Unbounded time horizon requires truncation for policy selection. Setting T=3-5 enables efficient Expected Free Energy computation.
- **parameter_learning**: Add Dirichlet concentration parameters for online model learning
  - *Rationale*: Static A and B matrices cannot adapt. Adding concentration parameters (a, b) enables Bayesian learning from experience.

## deep_planning_horizon.md (pomdp model)
### High Priority
- **precision_modulation**: Add precision parameters to modulate sensory and policy uncertainty
  - *Rationale*: The model lacks precision parameters. Active Inference with precision weighting better captures attentional modulation and epistemic confidence.
### Medium Priority
- **connectivity_enrichment**: Investigate potential missing causal links between model components
  - *Rationale*: Graph density is 0.09 (33 connections, 20 variables). Sparse graphs may miss important dependencies.
- **planning_horizon**: Define explicit planning horizon T for tractable policy inference
  - *Rationale*: Unbounded time horizon requires truncation for policy selection. Setting T=3-5 enables efficient Expected Free Energy computation.
- **parameter_learning**: Add Dirichlet concentration parameters for online model learning
  - *Rationale*: Static A and B matrices cannot adapt. Adding concentration parameters (a, b) enables Bayesian learning from experience.

## actinf_pomdp_agent.md (pomdp model)
### Medium Priority
- **connectivity_enrichment**: Investigate potential missing causal links between model components
  - *Rationale*: Graph density is 0.12 (11 connections, 10 variables). Sparse graphs may miss important dependencies.
- **planning_horizon**: Define explicit planning horizon T for tractable policy inference
  - *Rationale*: Unbounded time horizon requires truncation for policy selection. Setting T=3-5 enables efficient Expected Free Energy computation.
- **parameter_learning**: Add Dirichlet concentration parameters for online model learning
  - *Rationale*: Static A and B matrices cannot adapt. Adding concentration parameters (a, b) enables Bayesian learning from experience.

## hmm_baseline.md (hmm model)
### Medium Priority
- **connectivity_enrichment**: Investigate potential missing causal links between model components
  - *Rationale*: Graph density is 0.13 (12 connections, 10 variables). Sparse graphs may miss important dependencies.
### Low Priority
- **upgrade_to_pomdp**: Consider extending HMM to full POMDP with action-dependent transitions
  - *Rationale*: HMMs have no action selection. Adding a B[states,states,actions] tensor and preference vector C enables Active Inference policy optimization.

## two_state_bistable.md (pomdp model)
### High Priority
- **precision_modulation**: Add precision parameters to modulate sensory and policy uncertainty
  - *Rationale*: The model lacks precision parameters. Active Inference with precision weighting better captures attentional modulation and epistemic confidence.
### Medium Priority
- **connectivity_enrichment**: Investigate potential missing causal links between model components
  - *Rationale*: Graph density is 0.12 (11 connections, 10 variables). Sparse graphs may miss important dependencies.
- **planning_horizon**: Define explicit planning horizon T for tractable policy inference
  - *Rationale*: Unbounded time horizon requires truncation for policy selection. Setting T=3-5 enables efficient Expected Free Energy computation.
- **parameter_learning**: Add Dirichlet concentration parameters for online model learning
  - *Rationale*: Static A and B matrices cannot adapt. Adding concentration parameters (a, b) enables Bayesian learning from experience.

## markov_chain.md (hmm model)
### Medium Priority
- **connectivity_enrichment**: Investigate potential missing causal links between model components
  - *Rationale*: Graph density is 0.14 (6 connections, 7 variables). Sparse graphs may miss important dependencies.
### Low Priority
- **upgrade_to_pomdp**: Consider extending HMM to full POMDP with action-dependent transitions
  - *Rationale*: HMMs have no action selection. Adding a B[states,states,actions] tensor and preference vector C enables Active Inference policy optimization.

## tmaze_epistemic.md (pomdp model)
### High Priority
- **precision_modulation**: Add precision parameters to modulate sensory and policy uncertainty
  - *Rationale*: The model lacks precision parameters. Active Inference with precision weighting better captures attentional modulation and epistemic confidence.
### Medium Priority
- **connectivity_enrichment**: Investigate potential missing causal links between model components
  - *Rationale*: Graph density is 0.08 (19 connections, 16 variables). Sparse graphs may miss important dependencies.
- **parameter_learning**: Add Dirichlet concentration parameters for online model learning
  - *Rationale*: Static A and B matrices cannot adapt. Adding concentration parameters (a, b) enables Bayesian learning from experience.

