# GNN Specification: AXIOM Structure Learning
GNNVersionAndFlags: 1.4

ModelName: AXIOM_Structure_Learning

ModelAnnotation: |
  Online Bayesian structure learning implementing fast component addition
  and Bayesian Model Reduction (BMR) for mixture model optimization.
  Dynamically expands and prunes model complexity to match data.

StateSpaceBlock:
  # Component counts (dynamic)
  K_slots[1,discrete]              ### Number of active slots
  V_identities[1,discrete]         ### Number of identity types
  L_dynamics[1,discrete]           ### Number of dynamics modes  
  M_contexts[1,discrete]           ### Number of context modes
  
  # Component maximum limits
  K_max[1,discrete]                ### Maximum number of slots
  V_max[1,discrete]                ### Maximum number of identities
  L_max[1,discrete]                ### Maximum number of dynamics modes
  M_max[1,discrete]                ### Maximum number of contexts
  
  # Expansion thresholds
  tau_smm[1,continuous]            ### sMM expansion threshold
  tau_imm[1,continuous]            ### iMM expansion threshold
  tau_tmm[1,continuous]            ### tMM expansion threshold
  tau_rmm[1,continuous]            ### rMM expansion threshold
  
  # BMR schedule and parameters
  T_bmr[1,discrete]                ### BMR application interval
  n_bmr_pairs[1,discrete]          ### Number of merge candidates per BMR
  bmr_threshold[1,continuous]      ### Free energy threshold for merging
  
  # Component utilization tracking
  usage_smm[K_max,continuous]      ### Slot usage counters
  usage_imm[V_max,continuous]      ### Identity usage counters
  usage_tmm[L_max,continuous]      ### Dynamics mode usage counters
  usage_rmm[M_max,continuous]      ### Context mode usage counters
  
  # Component quality metrics
  quality_smm[K_max,continuous]    ### Slot posterior predictive likelihood
  quality_imm[V_max,continuous]    ### Identity posterior predictive likelihood
  quality_tmm[L_max,continuous]    ### Dynamics posterior predictive likelihood
  quality_rmm[M_max,continuous]    ### Context posterior predictive likelihood
  
  # BMR candidate pairs
  merge_candidates_smm[K_max,K_max,continuous] ### Slot merge scores
  merge_candidates_imm[V_max,V_max,continuous] ### Identity merge scores
  merge_candidates_tmm[L_max,L_max,continuous] ### Dynamics merge scores
  merge_candidates_rmm[M_max,M_max,continuous] ### Context merge scores

Connections:
  K_slots -> usage_smm             ### Active slots tracked for usage
  V_identities -> usage_imm        ### Active identities tracked for usage
  L_dynamics -> usage_tmm          ### Active dynamics tracked for usage
  M_contexts -> usage_rmm          ### Active contexts tracked for usage
  
  quality_smm -> merge_candidates_smm ### Quality informs merge decisions
  quality_imm -> merge_candidates_imm ### Quality informs merge decisions
  quality_tmm -> merge_candidates_tmm ### Quality informs merge decisions
  quality_rmm -> merge_candidates_rmm ### Quality informs merge decisions

InitialParameterization:
  # Initial component counts (start small)
  K_slots = 4                      ### Start with 4 slots
  V_identities = 2                 ### Start with 2 identity types
  L_dynamics = 3                   ### Start with 3 dynamics modes
  M_contexts = 5                   ### Start with 5 context modes
  
  # Maximum component limits
  K_max = 16                       ### Max 16 slots
  V_max = 10                       ### Max 10 identity types
  L_max = 20                       ### Max 20 dynamics modes
  M_max = 100                      ### Max 100 context modes
  
  # Expansion thresholds (tuned per module)
  tau_smm = -2.0                   ### Aggressive slot expansion
  tau_imm = -1.5                   ### Moderate identity expansion
  tau_tmm = -1.0                   ### Conservative dynamics expansion
  tau_rmm = -0.5                   ### Very conservative context expansion
  
  # BMR parameters
  T_bmr = 500                      ### Apply BMR every 500 timesteps
  n_bmr_pairs = 10                 ### Consider top 10 merge candidates
  bmr_threshold = 0.0              ### Merge if free energy decreases
  
  # Initialize usage and quality tracking
  usage_smm = zeros(K_max)
  usage_imm = zeros(V_max)
  usage_tmm = zeros(L_max)
  usage_rmm = zeros(M_max)
  
  quality_smm = -inf * ones(K_max)
  quality_imm = -inf * ones(V_max)
  quality_tmm = -inf * ones(L_max)
  quality_rmm = -inf * ones(M_max)

Equations:
  # Component addition criterion (based on posterior predictive likelihood)
  AddComponent(module) = max_c ℓ_{t,c} < τ_module + log α_module
  
  # Where ℓ_{t,c} is log posterior predictive likelihood for component c
  # and α_module is the stick-breaking concentration parameter
  
  # Slot expansion criterion
  AddNewSlot() = (max_k quality_smm[k] < tau_smm + log(alpha_smm)) AND (K_slots < K_max)
  
  # Identity expansion criterion  
  AddNewIdentity() = (max_v quality_imm[v] < tau_imm + log(alpha_imm)) AND (V_identities < V_max)
  
  # Dynamics expansion criterion
  AddNewDynamics() = (max_l quality_tmm[l] < tau_tmm + log(alpha_tmm)) AND (L_dynamics < L_max)
  
  # Context expansion criterion
  AddNewContext() = (max_m quality_rmm[m] < tau_rmm + log(alpha_rmm)) AND (M_contexts < M_max)
  
  # Posterior predictive likelihood calculation
  # For sMM (slot quality)
  quality_smm[k] = E_{q(Theta_smm_k)}[log p(o_pixels_new | s_slot[k], Theta_smm_k)]
  
  # For iMM (identity quality)
  quality_imm[v] = E_{q(Theta_imm_v)}[log p(s_appearance_new | Theta_imm_mu[v], Theta_imm_Sigma[v])]
  
  # For tMM (dynamics quality)
  quality_tmm[l] = E_{q(Theta_tmm_l)}[log p(s_slot_t1_new | s_slot_t_new, Theta_tmm_D[l], Theta_tmm_b[l])]
  
  # For rMM (context quality)
  quality_rmm[m] = E_{q(Theta_rmm_m)}[log p(f_continuous_new, d_discrete_new | Theta_rmm_mu[m], Theta_rmm_Sigma[m], Theta_rmm_alpha[m])]
  
  # BMR merge criterion (variational free energy comparison)
  MergeComponents(c1, c2) = F_merged < F_separate
  
  # Where F is variational free energy
  F_merged = -∑_data log p(data | merged_params) + D_KL[q(merged_params) || p(merged_params)]
  F_separate = -∑_data log p(data | separate_params) + D_KL[q(separate_params) || p(separate_params)]
  
  # Component merge score calculation
  merge_score(c1, c2) = F_separate(c1, c2) - F_merged(c1, c2)
  
  # If merge_score > bmr_threshold, then merge is beneficial
  
  # Usage tracking (exponential moving average)
  UpdateUsage(module, component):
    α_usage = 0.99  # Decay factor
    usage[component] = α_usage * usage[component] + (1 - α_usage) * assignment_probability[component]
  
  # Component pruning (remove unused components)
  PruneComponent(module, component):
    if usage[component] < usage_threshold:
      RemoveComponent(module, component)
      CompactIndices(module)
  
  # BMR algorithm (applied every T_bmr timesteps)
  BayesianModelReduction():
    For each module in [sMM, iMM, tMM, rMM]:
      # Find best merge candidates
      merge_scores = ComputeAllMergeScores(module)
      candidates = TopK(merge_scores, n_bmr_pairs)
      
      # Execute beneficial merges
      For (c1, c2) in candidates:
        if merge_scores[c1, c2] > bmr_threshold:
          MergeComponents(module, c1, c2)
          UpdateComponentCount(module, -1)
      
      # Compact component indices
      CompactIndices(module)
  
  # Component merging procedure
  MergeComponents(module, c1, c2):
    if module == sMM:
      # Merge slot parameters (weighted average by usage)
      w1 = usage_smm[c1]
      w2 = usage_smm[c2]
      Theta_smm_merged = (w1 * Theta_smm[c1] + w2 * Theta_smm[c2]) / (w1 + w2)
      
    elif module == iMM:
      # Merge identity parameters (NIW sufficient statistics)
      MergeNIWParameters(Theta_imm[c1], Theta_imm[c2])
      
    elif module == tMM:
      # Merge dynamics parameters (weighted linear regression statistics)
      MergeDynamicsParameters(Theta_tmm[c1], Theta_tmm[c2])
      
    elif module == rMM:
      # Merge context parameters (NIW + Dirichlet sufficient statistics)
      MergeContextParameters(Theta_rmm[c1], Theta_rmm[c2])
  
  # Sufficient statistics merging for NIW distributions
  MergeNIWParameters(params1, params2):
    # Combine sufficient statistics
    n1, x_bar1, S1 = ExtractNIWSuffStats(params1)
    n2, x_bar2, S2 = ExtractNIWSuffStats(params2)
    
    n_merged = n1 + n2
    x_bar_merged = (n1 * x_bar1 + n2 * x_bar2) / n_merged
    S_merged = S1 + S2 + (n1 * n2 / n_merged) * (x_bar1 - x_bar2) * (x_bar1 - x_bar2)^T
    
    return ConstructNIWParams(n_merged, x_bar_merged, S_merged)
  
  # Structure learning triggers
  OnlineStructureLearning():
    # Check for expansion every timestep
    if AddNewSlot():
      ExpandSlots()
    if AddNewIdentity():
      ExpandIdentities()
    if AddNewDynamics():
      ExpandDynamics()
    if AddNewContext():
      ExpandContexts()
    
    # Apply BMR periodically
    if (timestep % T_bmr) == 0:
      BayesianModelReduction()
  
  # Component expansion procedures
  ExpandSlots():
    K_slots += 1
    InitializeNewSlot(K_slots)
    
  ExpandIdentities():
    V_identities += 1
    InitializeNewIdentity(V_identities)
    
  ExpandDynamics():
    L_dynamics += 1
    InitializeNewDynamics(L_dynamics)
    
  ExpandContexts():
    M_contexts += 1
    InitializeNewContext(M_contexts)

Time:
  ModelTimeHorizon: ∞
  DiscreteTime: true
  Dynamic: true

ActInfOntologyAnnotation:
  K_slots: "model_complexity_slots"
  V_identities: "model_complexity_identities"
  L_dynamics: "model_complexity_dynamics"
  M_contexts: "model_complexity_contexts"
  tau_smm: "expansion_threshold_perception"
  tau_imm: "expansion_threshold_categorization"
  tau_tmm: "expansion_threshold_dynamics"
  tau_rmm: "expansion_threshold_interaction"
  T_bmr: "model_reduction_schedule"
  usage_smm: "component_utilization_perception"
  usage_imm: "component_utilization_categorization"
  usage_tmm: "component_utilization_dynamics"
  usage_rmm: "component_utilization_interaction"
  quality_smm: "component_quality_perception"
  quality_imm: "component_quality_categorization"
  quality_tmm: "component_quality_dynamics"
  quality_rmm: "component_quality_interaction"

Footer:
  GeneratedBy: "AXIOM-GNN Specification Generator v1.4"
  CreatedDate: "2025-01-23"
  LastModified: "2025-01-23"
  Version: "1.0.0"
  
Signature:
  Author: "AXIOM Research Team - Structure Learning Module"
  Institution: "VERSES AI / Active Inference Institute"
  ContactEmail: "axiom-structure@verses.ai"
  DOI: "10.5281/zenodo.axiom.gnn.structure" 