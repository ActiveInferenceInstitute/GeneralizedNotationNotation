# GNN Specification: AXIOM Recurrent Mixture Model
GNNVersionAndFlags: 1.4

ModelName: AXIOM_RecurrentMixtureModel

ModelAnnotation: |
  Interaction and control module that models dependencies between objects,
  actions, and rewards. Predicts next transition modes and rewards based on
  multi-object features, enabling sparse interaction modeling and planning.

StateSpaceBlock:
  # Multi-object context features
  f_continuous[K,F_c,continuous]   ### Continuous context (positions, distances)
  d_discrete[K,F_d,discrete]       ### Discrete context (identities, actions, rewards)
  
  # Context assignments to mixture components
  s_rmm_context[K,M,binary]        ### K slots assigned to M context modes
  c_rmm_unused[M,continuous]       ### Unused context mode counter
  
  # Output predictions
  s_tmm_next[K,L,binary]           ### Next transition mode predictions
  r_reward_next[1,continuous]      ### Next reward prediction
  
  # Model parameters
  Theta_rmm_mu[M,F_c,continuous]   ### Context means for continuous features
  Theta_rmm_Sigma[M,F_c,F_c,continuous] ### Context covariances
  Theta_rmm_alpha[M,F_d,continuous] ### Categorical parameters for discrete features
  Theta_rmm_pi[M,continuous]       ### Context mode mixing weights
  
  # Prediction parameters
  Theta_rmm_tmm[M,L,continuous]    ### Context-to-dynamics prediction weights
  Theta_rmm_reward[M,continuous]   ### Context-to-reward prediction weights
  
  # Hyperparameters
  alpha_rmm[1,continuous]          ### Stick-breaking concentration
  F_c[1,discrete]                  ### Number of continuous features
  F_d[1,discrete]                  ### Number of discrete features

Connections:
  f_continuous -> s_rmm_context    ### Continuous features determine context
  d_discrete -> s_rmm_context      ### Discrete features determine context  
  s_rmm_context -> s_tmm_next      ### Context predicts dynamics mode
  s_rmm_context -> r_reward_next   ### Context predicts reward
  c_rmm_unused -> s_rmm_context    ### Track unused context modes
  
InitialParameterization:
  # Context assignments
  s_rmm_context[k,m] ~ Categorical(Theta_rmm_pi)
  
  # Continuous feature parameters with NIW priors
  (Theta_rmm_mu[m], Theta_rmm_Sigma[m]) ~ NIW(m_rmm, κ_rmm, U_rmm, n_rmm)
  
  # Discrete feature parameters with Dirichlet priors
  Theta_rmm_alpha[m,d] ~ Dirichlet(α_rmm_discrete)
  
  # Context mixing weights
  Theta_rmm_pi ~ StickBreaking(alpha_rmm=1.0)
  
  # Prediction parameters
  Theta_rmm_tmm[m] ~ Dirichlet(α_tmm_prediction)  # Predicts dynamics mode probabilities
  Theta_rmm_reward[m] ~ N(μ_reward_prior, σ_reward_prior)
  
  # Hyperparameter initialization
  alpha_rmm = 1.0
  F_c = 10  # Position differences, distances, velocities, etc.
  F_d = 5   # Identity types, previous actions, reward history, etc.
  
  # NIW hyperparameters for continuous features
  m_rmm = zeros(F_c)               # Neutral prior mean
  κ_rmm = 1.0                      # Weak prior on mean
  U_rmm = 0.1 * I_F_c              # Small prior scale
  n_rmm = F_c + 2                  # Degrees of freedom
  
  # Dirichlet hyperparameters
  α_rmm_discrete = ones(max_discrete_values)  # Uniform prior
  α_tmm_prediction = ones(L_max)   # Uniform prior over dynamics modes
  
  # Reward prediction priors
  μ_reward_prior = 0.0             # Neutral reward expectation
  σ_reward_prior = 1.0             # Moderate uncertainty
  
  # Unused counters
  c_rmm_unused[m] = 0.0

Equations:
  # Joint context likelihood
  p(f_continuous[k], d_discrete[k] | s_rmm_context[k]) = 
    ∏_{m=1}^M [N(f_continuous[k]; Theta_rmm_mu[m], Theta_rmm_Sigma[m]) · 
               ∏_i Cat(d_discrete[k,i]; Theta_rmm_alpha[m,i])]^{s_rmm_context[k,m]}
  
  # Predictive distributions
  p(s_tmm_next[k,l] = 1 | s_rmm_context[k]) = 
    ∑_{m=1}^M s_rmm_context[k,m] · Theta_rmm_tmm[m,l]
  
  p(r_reward_next | s_rmm_context) = 
    ∑_{k=1}^K ∑_{m=1}^M s_rmm_context[k,m] · N(Theta_rmm_reward[m], σ_reward_noise)
  
  # Context feature construction
  f_continuous[k] = [
    s_slot[k, 1:2],                 # Object k position
    ∑_{j≠k} ||s_slot[k] - s_slot[j]||,  # Distances to other objects
    s_slot[k, 1:2] - s_slot[k, 1:2]_{t-1},  # Velocity
    min_j≠k ||s_slot[k] - s_slot[j]||,      # Distance to nearest object
    center_of_mass(all_slots) - s_slot[k, 1:2],  # Relative to center
    reward_history[t-1:t-3]         # Recent reward history
  ]
  
  d_discrete[k] = [
    z_identity[k],                  # Object identity
    u_action[t-1],                  # Previous action
    sign(r_reward[t-1]),            # Previous reward sign
    collision_indicator[k],         # Collision flag
    boundary_contact[k]             # Boundary contact flag
  ]
  
  # Structure learning for context modes
  AddNewContextMode() = max_m ℓ_{t,m} < τ_rmm + log(alpha_rmm)
  
  # Where ℓ_{t,m} is posterior predictive log-likelihood for context mode m
  ℓ_{t,m} = E_{q(Theta_rmm_mu[m], Theta_rmm_Sigma[m], Theta_rmm_alpha[m])}[
    log p(f_continuous_new, d_discrete_new | Theta_rmm_mu[m], Theta_rmm_Sigma[m], Theta_rmm_alpha[m])]
  
  # Variational updates (coordinate ascent)
  # E-step: Update posterior over context assignments
  q(s_rmm_context[k,m] = 1) ∝ 
    Theta_rmm_pi[m] · 
    N(f_continuous[k]; Theta_rmm_mu[m], Theta_rmm_Sigma[m]) ·
    ∏_i Cat(d_discrete[k,i]; Theta_rmm_alpha[m,i])
  
  # M-step: Update continuous feature parameters (NIW)
  N_m = ∑_k q(s_rmm_context[k,m] = 1)
  f_bar_m = (1/N_m) ∑_k q(s_rmm_context[k,m] = 1) · f_continuous[k]
  S_m = ∑_k q(s_rmm_context[k,m] = 1) · (f_continuous[k] - f_bar_m)(f_continuous[k] - f_bar_m)^T
  
  # Updated NIW hyperparameters
  κ_m_new = κ_rmm + N_m
  m_m_new = (κ_rmm * m_rmm + N_m * f_bar_m) / κ_m_new
  n_m_new = n_rmm + N_m
  U_m_new = U_rmm + S_m + (κ_rmm * N_m / κ_m_new) * (f_bar_m - m_rmm)(f_bar_m - m_rmm)^T
  
  # M-step: Update discrete feature parameters (Dirichlet)
  For each discrete feature i:
    N_m_i_v = ∑_k q(s_rmm_context[k,m] = 1) · I(d_discrete[k,i] = v)
    α_m_i_v_new = α_rmm_discrete[v] + N_m_i_v
  
  # M-step: Update prediction parameters
  # For dynamics mode prediction
  target_tmm_counts = ∑_k q(s_rmm_context[k,m] = 1) · s_tmm_actual[k]
  Theta_rmm_tmm[m] = Dirichlet_MLE(target_tmm_counts + α_tmm_prediction)
  
  # For reward prediction
  assigned_rewards = {r_reward[t] for k where q(s_rmm_context[k,m] = 1) > threshold}
  Theta_rmm_reward[m] = mean(assigned_rewards)

Time:
  ModelTimeHorizon: T_max
  DiscreteTime: true
  Dynamic: true

ActInfOntologyAnnotation:
  f_continuous: "spatial_interaction_features"
  d_discrete: "symbolic_interaction_features"
  s_rmm_context: "interaction_context_classification"
  s_tmm_next: "predicted_dynamics_regime"
  r_reward_next: "expected_utility"
  Theta_rmm_mu: "interaction_context_means"
  Theta_rmm_Sigma: "interaction_context_covariances"
  Theta_rmm_alpha: "symbolic_interaction_parameters"
  Theta_rmm_pi: "interaction_context_priors"
  Theta_rmm_tmm: "context_to_dynamics_mapping"
  Theta_rmm_reward: "context_to_reward_mapping"
  c_rmm_unused: "interaction_context_utilization"

Footer:
  GeneratedBy: "AXIOM-GNN Specification Generator v1.4"
  CreatedDate: "2025-01-23"
  LastModified: "2025-01-23"
  Version: "1.0.0"
  
Signature:
  Author: "AXIOM Research Team - rMM Module"
  Institution: "VERSES AI / Active Inference Institute"
  ContactEmail: "axiom-rmm@verses.ai"
  DOI: "TBD - Not yet published" 