# GNN Specification: AXIOM Transition Mixture Model  
GNNVersionAndFlags: 1.4

ModelName: AXIOM_TransitionMixtureModel

ModelAnnotation: |
  Object dynamics module modeling each slot's evolution as piecewise linear
  trajectories. Represents a switching linear dynamical system (SLDS) where
  different linear modes capture distinct motion patterns (falling, bouncing, etc).

StateSpaceBlock:
  # Slot states across time
  s_slot_t[K,7,continuous]        ### Current slot states
  s_slot_t1[K,7,continuous]       ### Next slot states
  
  # Dynamics mode assignments
  s_tmm_mode[K,L,binary]          ### K slots assigned to L dynamics modes
  c_tmm_unused[L,continuous]      ### Unused dynamics mode counter
  
  # Linear dynamics parameters
  Theta_tmm_D[L,7,7,continuous]   ### Linear transition matrices
  Theta_tmm_b[L,7,continuous]     ### Linear bias terms
  Theta_tmm_Sigma[L,7,7,continuous] ### Process noise covariances
  Theta_tmm_pi[L,continuous]      ### Mode mixing weights
  
  # Mode-specific properties
  z_tmm_sticky[L,binary]          ### Mode stickiness indicators
  gamma_tmm_stick[L,continuous]   ### Stickiness parameters
  
  # Hyperparameters
  alpha_tmm[1,continuous]         ### Stick-breaking concentration
  sigma_tmm_noise[1,continuous]   ### Base process noise level

Connections:
  s_slot_t -> s_slot_t1           ### Current state influences next state
  s_tmm_mode -> s_slot_t1         ### Mode selection determines dynamics
  Theta_tmm_D -> s_slot_t1        ### Linear dynamics transform states
  Theta_tmm_b -> s_slot_t1        ### Bias terms shift dynamics
  z_tmm_sticky -> s_tmm_mode      ### Stickiness affects mode transitions
  c_tmm_unused -> s_tmm_mode      ### Track unused modes

InitialParameterization:
  # Mode assignments with sticky transitions
  s_tmm_mode[k,l,t] ~ StickyTransition(Theta_tmm_pi, gamma_tmm_stick[l])
  
  # Linear dynamics with structured priors
  Theta_tmm_D[l] ~ MatrixNormal(I_7, 0.1*I_7, 0.1*I_7)  # Near-identity prior
  Theta_tmm_b[l] ~ N(0, 0.1*I_7)                        # Small bias prior
  Theta_tmm_Sigma[l] ~ InverseWishart(nu=9, Psi=sigma_tmm_noise*I_7)
  
  # Mode mixing weights
  Theta_tmm_pi ~ StickBreaking(alpha_tmm=1.0)
  
  # Stickiness parameters
  gamma_tmm_stick[l] ~ Gamma(2.0, 1.0)  # Encourages moderate stickiness
  z_tmm_sticky[l] ~ Bernoulli(0.8)      # Most modes are sticky
  
  # Hyperparameters
  alpha_tmm = 1.0
  sigma_tmm_noise = 0.01
  c_tmm_unused[l] = 0.0

Equations:
  # Linear dynamics likelihood
  p(s_slot_t1[k] | s_slot_t[k], s_tmm_mode[k]) = 
    ∏_{l=1}^L N(Theta_tmm_D[l]·s_slot_t[k] + Theta_tmm_b[l], 
                 Theta_tmm_Sigma[l])^{s_tmm_mode[k,l]}
  
  # Sticky mode transition probabilities
  p(s_tmm_mode[k,l,t] = 1 | s_tmm_mode[k,:,t-1]) = 
    (1-γ_stick[l]) * Theta_tmm_pi[l] + γ_stick[l] * s_tmm_mode[k,l,t-1]
  
  # Where γ_stick[l] = gamma_tmm_stick[l] if z_tmm_sticky[l] = 1, else 0
  
  # Mode probability (initial and non-sticky)
  p(s_tmm_mode[k,l,0] = 1) = Theta_tmm_pi[l]
  
  # Structure learning for dynamics modes
  AddNewDynamicsMode() = max_l ℓ_{t,l} < τ_tmm + log(alpha_tmm)
  
  # Where ℓ_{t,l} is posterior predictive log-likelihood for dynamics mode l
  ℓ_{t,l} = E_{q(Theta_tmm_D[l], Theta_tmm_b[l], Theta_tmm_Sigma[l])}[
    log p(s_slot_t1_new | s_slot_t_new, Theta_tmm_D[l], Theta_tmm_b[l], Theta_tmm_Sigma[l])]
  
  # Variational updates (coordinate ascent)
  # E-step: Update posterior over mode assignments
  q(s_tmm_mode[k,l,t] = 1) ∝ 
    p(s_tmm_mode[k,l,t] | s_tmm_mode[k,:,t-1]) · 
    N(s_slot_t1[k]; Theta_tmm_D[l]·s_slot_t[k] + Theta_tmm_b[l], Theta_tmm_Sigma[l])
  
  # M-step: Update dynamics parameters
  # For each mode l, collect assigned transitions
  N_l = ∑_{k,t} q(s_tmm_mode[k,l,t] = 1)
  X_l = {(s_slot_t[k], s_slot_t1[k]) : q(s_tmm_mode[k,l,t] = 1) > threshold}
  
  # Linear regression for dynamics matrix and bias
  # s_slot_t1 = D_l * s_slot_t + b_l + noise
  [Theta_tmm_D[l], Theta_tmm_b[l]] = WeightedLinearRegression(X_l, weights=q(assignments))
  
  # Update noise covariance
  residuals_l = {s_slot_t1[k] - Theta_tmm_D[l]·s_slot_t[k] - Theta_tmm_b[l] 
                 for assigned transitions}
  Theta_tmm_Sigma[l] = SampleCovariance(residuals_l)

Time:
  ModelTimeHorizon: T_max
  DiscreteTime: true
  Dynamic: true

ActInfOntologyAnnotation:
  s_slot_t: "hidden_state_dynamics"
  s_tmm_mode: "dynamical_regime_selection"  
  Theta_tmm_D: "transition_model_parameters"
  Theta_tmm_b: "dynamical_bias_terms"
  Theta_tmm_Sigma: "process_noise_covariance"
  Theta_tmm_pi: "dynamics_regime_priors"
  z_tmm_sticky: "regime_persistence_indicators"
  gamma_tmm_stick: "regime_stickiness_parameters"
  c_tmm_unused: "dynamics_regime_utilization"

Footer:
  GeneratedBy: "AXIOM-GNN Specification Generator v1.4"
  CreatedDate: "2025-01-23"
  LastModified: "2025-01-23"
  Version: "1.0.0"
  
Signature:
  Author: "AXIOM Research Team - tMM Module"
  Institution: "VERSES AI / Active Inference Institute"
  ContactEmail: "axiom-tmm@verses.ai"
  DOI: "10.5281/zenodo.axiom.gnn.tmm" 