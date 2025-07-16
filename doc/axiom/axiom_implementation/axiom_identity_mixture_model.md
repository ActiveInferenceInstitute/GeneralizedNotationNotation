# GNN Specification: AXIOM Identity Mixture Model
GNNVersionAndFlags: 1.4

ModelName: AXIOM_IdentityMixtureModel

ModelAnnotation: |
  Object identity classification module that assigns discrete type labels
  to object slots based on their color and shape features. Enables
  type-specific rather than instance-specific learning of dynamics.

StateSpaceBlock:
  # Input from slots (color + shape features only)
  s_appearance[K,5,continuous]     ### Color(3) + shape(2) features from slots
  
  # Identity assignments
  z_identity[K,V,binary]           ### K slots assigned to V identity types
  c_identity_unused[V,continuous]  ### Unused identity type counter
  
  # Identity type parameters
  Theta_imm_mu[V,5,continuous]     ### Type means for appearance features
  Theta_imm_Sigma[V,5,5,continuous] ### Type covariance matrices
  Theta_imm_pi[V,continuous]       ### Identity type mixing weights
  
  # Hyperparameters for NIW priors
  m_identity[V,5,continuous]       ### Prior means for identity types
  kappa_identity[V,continuous]     ### Prior concentration for means
  U_identity[V,5,5,continuous]     ### Prior scale matrices
  nu_identity[V,continuous]        ### Prior degrees of freedom
  alpha_imm[1,continuous]          ### Stick-breaking concentration

Connections:
  s_appearance -> z_identity       ### Appearance determines identity
  z_identity -> Theta_imm_mu       ### Identity types have characteristic appearances
  Theta_imm_pi -> z_identity       ### Prior over identity types
  c_identity_unused -> z_identity  ### Tracks unused identity types

InitialParameterization:
  # Identity assignments  
  z_identity[k,v] ~ Categorical(Theta_imm_pi)
  
  # Type parameters with conjugate NIW priors
  (Theta_imm_mu[v], Theta_imm_Sigma[v]) ~ NIW(m_identity[v], kappa_identity[v], 
                                               U_identity[v], nu_identity[v])
  
  # Mixing weights with stick-breaking
  Theta_imm_pi ~ StickBreaking(alpha_imm=1.0)
  
  # Hyperparameter initialization
  alpha_imm = 1.0
  m_identity[v] = [0.5, 0.5, 0.5, 0.1, 0.1]  # Default gray, small object
  kappa_identity[v] = 1.0                      # Weak prior on mean
  U_identity[v] = 0.1 * I_5                   # Small prior scale
  nu_identity[v] = 6.0                        # Degrees of freedom > dimensionality
  
  # Unused counters
  c_identity_unused[v] = 0.0

Equations:
  # Likelihood of appearance given identity
  p(s_appearance[k] | z_identity[k], Theta_imm) = 
    ∏_{v=1}^V N(Theta_imm_mu[v], Theta_imm_Sigma[v])^{z_identity[k,v]}
  
  # Prior over type parameters (Normal-Inverse-Wishart)
  p(Theta_imm_mu[v], Theta_imm_Sigma[v]^{-1}) = 
    NIW(m_identity[v], kappa_identity[v], U_identity[v], nu_identity[v])
  
  # Expanded form of NIW prior
  p(Theta_imm_Sigma[v]^{-1}) = Wishart(U_identity[v]^{-1}, nu_identity[v])
  p(Theta_imm_mu[v] | Theta_imm_Sigma[v]) = 
    N(m_identity[v], (kappa_identity[v])^{-1} * Theta_imm_Sigma[v])
  
  # Identity assignment probability
  p(z_identity[k,v] = 1) = Theta_imm_pi[v]
  
  # Structure learning for identity types
  AddNewIdentityType() = max_v ℓ_{t,v} < τ_imm + log(alpha_imm)
  
  # Where ℓ_{t,v} is posterior predictive log-likelihood for identity type v
  ℓ_{t,v} = E_{q(Theta_imm_mu[v], Theta_imm_Sigma[v])}[log p(s_appearance_new | Theta_imm_mu[v], Theta_imm_Sigma[v])]
  
  # Variational updates (coordinate ascent)
  # E-step: Update posterior over assignments
  q(z_identity[k,v] = 1) ∝ Theta_imm_pi[v] · N(s_appearance[k]; Theta_imm_mu[v], Theta_imm_Sigma[v])
  
  # M-step: Update NIW parameters
  N_v = ∑_k q(z_identity[k,v] = 1)
  x_bar_v = (1/N_v) ∑_k q(z_identity[k,v] = 1) · s_appearance[k]
  S_v = ∑_k q(z_identity[k,v] = 1) · (s_appearance[k] - x_bar_v)(s_appearance[k] - x_bar_v)^T
  
  # Updated NIW hyperparameters
  kappa_v_new = kappa_identity[v] + N_v
  m_v_new = (kappa_identity[v] * m_identity[v] + N_v * x_bar_v) / kappa_v_new
  nu_v_new = nu_identity[v] + N_v
  U_v_new = U_identity[v] + S_v + 
            (kappa_identity[v] * N_v / kappa_v_new) * (x_bar_v - m_identity[v])(x_bar_v - m_identity[v])^T

Time:
  ModelTimeHorizon: T_max
  DiscreteTime: true  
  Dynamic: true

ActInfOntologyAnnotation:
  s_appearance: "object_feature_representation"
  z_identity: "object_categorization"
  Theta_imm_mu: "categorical_prototypes"
  Theta_imm_Sigma: "categorical_uncertainty"
  Theta_imm_pi: "category_prior_beliefs"
  c_identity_unused: "category_utilization_tracking"
  m_identity: "prior_category_means"
  kappa_identity: "prior_mean_confidence"
  U_identity: "prior_covariance_scale"
  nu_identity: "prior_covariance_confidence"

Footer:
  GeneratedBy: "AXIOM-GNN Specification Generator v1.4"
  CreatedDate: "2025-01-23"
  LastModified: "2025-01-23"
  Version: "1.0.0"
  
Signature:
  Author: "AXIOM Research Team - iMM Module"
  Institution: "VERSES AI / Active Inference Institute"
  ContactEmail: "axiom-imm@verses.ai"
  DOI: "TBD - Not yet published" 