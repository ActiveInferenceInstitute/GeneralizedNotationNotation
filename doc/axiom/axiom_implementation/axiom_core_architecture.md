# GNN Specification: Complete AXIOM Architecture
GNNVersionAndFlags: 1.4

ModelName: AXIOM_Complete_System

ModelAnnotation: |
  Integrated AXIOM agent combining all four mixture models for object-centric
  world modeling, learning, and planning. Implements online Bayesian structure
  learning with model expansion and reduction.

StateSpaceBlock:
  # === INPUTS ===
  o_pixels[N,5,continuous]         ### Pixel observations (RGB + XY)
  u_action[1,discrete]             ### Control actions  
  r_reward[1,continuous]           ### Reward signals
  
  # === OBJECT SLOTS ===
  s_slot[K,7,continuous]           ### Object slot features
  z_slot_present[K,binary]         ### Slot presence indicators
  z_slot_moving[K,binary]          ### Slot movement indicators
  
  # === MIXTURE ASSIGNMENTS ===
  z_smm[N,K,binary]                ### Pixel-to-slot assignments
  z_imm[K,V,binary]                ### Slot-to-identity assignments  
  s_tmm[K,L,binary]                ### Slot-to-dynamics assignments
  s_rmm[K,M,binary]                ### Slot-to-context assignments
  
  # === MODEL PARAMETERS ===
  Theta_smm_pi[K,continuous]       ### sMM mixing weights
  Theta_imm_mu[V,5,continuous]     ### iMM identity means
  Theta_imm_Sigma[V,5,5,continuous] ### iMM identity covariances
  Theta_imm_pi[V,continuous]       ### iMM mixing weights
  Theta_tmm_D[L,7,7,continuous]    ### tMM dynamics matrices
  Theta_tmm_b[L,7,continuous]      ### tMM dynamics biases
  Theta_tmm_pi[L,continuous]       ### tMM mixing weights
  Theta_rmm_mu[M,F,continuous]     ### rMM context means
  Theta_rmm_Sigma[M,F,F,continuous] ### rMM context covariances
  Theta_rmm_alpha[M,D,continuous]  ### rMM categorical parameters
  Theta_rmm_pi[M,continuous]       ### rMM mixing weights
  
  # === PLANNING VARIABLES ===
  pi_policy[H,A,continuous]        ### Policy distributions over horizon H
  G_expected_free_energy[H,continuous] ### Expected free energy per timestep
  s_predicted[H,K,7,continuous]    ### Predicted slot trajectories
  r_predicted[H,continuous]        ### Predicted rewards

Connections:
  # Perception pathway
  o_pixels > s_slot               ### Pixels to slots via sMM
  s_slot > z_imm                  ### Slots to identities via iMM
  
  # Dynamics pathway  
  s_slot > s_tmm                  ### Slots to dynamics via context
  s_rmm > s_tmm                   ### Context determines dynamics
  s_tmm > s_slot                  ### Dynamics evolve slots
  
  # Interaction pathway
  s_slot > s_rmm                  ### Slots to context via features
  u_action > s_rmm                ### Actions influence context
  r_reward > s_rmm                ### Rewards influence context
  s_rmm > r_reward                ### Context predicts rewards
  
  # Planning pathway
  s_slot > pi_policy              ### Current state influences policy
  G_expected_free_energy > pi_policy ### Free energy determines policy
  pi_policy > u_action            ### Policy determines actions

InitialParameterization:
  # Hierarchical Bayesian priors for all mixture models
  Theta_smm_pi ~ StickBreaking(alpha_smm=1.0)
  Theta_imm_pi ~ StickBreaking(alpha_imm=1.0)
  Theta_tmm_pi ~ StickBreaking(alpha_tmm=1.0)
  Theta_rmm_pi ~ StickBreaking(alpha_rmm=1.0)
  
  # Identity type parameters with conjugate NIW priors
  (Theta_imm_mu[v], Theta_imm_Sigma[v]) ~ NIW(m_v, κ_v, U_v, n_v)
  
  # Dynamics parameters with uniform priors
  Theta_tmm_D[l] ~ Uniform(-1, 1)
  Theta_tmm_b[l] ~ Uniform(-1, 1)
  
  # Context parameters with conjugate priors
  (Theta_rmm_mu[m], Theta_rmm_Sigma[m]) ~ NIW(m_rmm, κ_rmm, U_rmm, n_rmm)
  Theta_rmm_alpha[m,d] ~ Dirichlet(α_rmm)
  
  # Initial slot configurations
  s_slot ~ N(μ_slot_init, Σ_slot_init)
  
  # Initial policy prior
  pi_policy ~ Uniform(action_space)

Equations:
  # === GENERATIVE MODEL ===
  p(o_pixels, s_slot, assignments, rewards | Theta, actions) = 
    ∏_t [p(o_pixels_t | s_slot_t, z_smm_t, Theta_smm) ·
          p(z_imm_t | s_slot_t, Theta_imm) ·
          p(s_slot_{t+1} | s_slot_t, s_tmm_t, Theta_tmm) ·
          p(s_tmm_t, r_reward_t | s_slot_t, u_action_t, s_rmm_t, Theta_rmm)]
  
  # === VARIATIONAL INFERENCE ===
  q(all_latents, all_parameters) = ∏_modules q(latents_module, params_module)
  
  Free_Energy = D_KL[q || p] ≥ -log p(observations)
  
  # === ACTIVE INFERENCE PLANNING ===
  G_expected_free_energy[τ] = -E_q[log p(r_τ | o_τ, π)] + 
                               D_KL[q(Theta_rmm | o_τ, π) || q(Theta_rmm)]
  
  π* = argmin_π ∑_{τ=0}^H G_expected_free_energy[τ]
  
  # === STRUCTURE LEARNING ===
  AddComponent(module) = max_c ℓ_{t,c} < τ_module + log α_module
  
  MergeComponents(c1, c2) = F_merged < F_separate

Time:
  ModelTimeHorizon: H_planning
  DiscreteTime: true
  Dynamic: true

ActInfOntologyAnnotation:
  o_pixels: "sensory_observation"
  s_slot: "hidden_state_object_centric"
  u_action: "control_action"
  r_reward: "utility_signal"
  pi_policy: "policy_distribution"
  G_expected_free_energy: "expected_free_energy"
  z_smm: "perceptual_binding"
  z_imm: "object_categorization"
  s_tmm: "dynamical_regime"
  s_rmm: "interaction_context"
  Theta_smm_pi: "perceptual_prior_beliefs"
  Theta_imm_mu: "categorical_prototypes"
  Theta_tmm_D: "transition_model_parameters"
  Theta_rmm_mu: "interaction_context_means"

Footer:
  GeneratedBy: "AXIOM-GNN Specification Generator v1.4"
  CreatedDate: "2025-01-23"
  LastModified: "2025-01-23"
  Version: "1.0.0"
  
Signature:
  Author: "AXIOM Research Team"
  Institution: "VERSES AI / Active Inference Institute"
  ContactEmail: "axiom-gnn@verses.ai"
  DOI: "10.5281/zenodo.axiom.gnn.core" 