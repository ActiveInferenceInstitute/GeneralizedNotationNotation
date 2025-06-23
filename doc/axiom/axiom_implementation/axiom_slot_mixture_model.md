# GNN Specification: AXIOM Slot Mixture Model
GNNVersionAndFlags: 1.4

ModelName: AXIOM_SlotMixtureModel

ModelAnnotation: |
  Object-centric visual perception module that decomposes pixel observations
  into K competing object slots using Gaussian mixture modeling. Each slot
  represents continuous object features (position, color, shape) that generate
  pixel likelihoods through linear projections.

StateSpaceBlock:
  # Input observations
  o_pixels[N,5,continuous]     ### N pixels with RGB+XY coordinates
  
  # Slot representations  
  s_slot[K,7,continuous]       ### K slots with position(2) + color(3) + shape(2)
  z_slot_assign[N,K,binary]    ### Pixel-to-slot assignment variables
  z_slot_presence[K,binary]    ### Slot presence indicators
  z_slot_moving[K,binary]      ### Slot movement indicators
  c_slot_unused[K,continuous]  ### Unused slot counter
  
  # Model parameters
  Theta_smm_pi[K,continuous]   ### Slot mixing weights with stick-breaking prior
  Theta_smm_A[5,7,continuous]  ### Projection matrix for slot to pixel features
  Theta_smm_B[2,7,continuous]  ### Shape projection matrix
  Theta_smm_sigma[K,3,continuous] ### Color variance parameters per slot
  
  # Hyperparameters
  alpha_smm[1,continuous]      ### Stick-breaking concentration parameter
  mu_slot_prior[7,continuous]  ### Prior mean for slot features
  Sigma_slot_prior[7,7,continuous] ### Prior covariance for slot features

Connections:
  s_slot -> o_pixels           ### Slots generate pixel observations
  z_slot_assign -> o_pixels    ### Assignment determines which slot explains pixel
  Theta_smm_pi -> z_slot_assign ### Mixing weights determine assignment probabilities
  z_slot_presence -> s_slot    ### Presence gates slot activity
  z_slot_moving -> s_slot      ### Movement state affects dynamics

InitialParameterization:
  # Slot features (position, color, shape)
  s_slot ~ N(mu_slot_prior, Sigma_slot_prior)
  
  # Pixel assignment probabilities  
  z_slot_assign[n,k] ~ Categorical(Theta_smm_pi)
  
  # Mixing weights with stick-breaking prior
  Theta_smm_pi ~ StickBreaking(alpha_smm=1.0)
  
  # Projection matrices (fixed)
  Theta_smm_A = [[1,0,0,0,0,0,0],    # X position
                 [0,1,0,0,0,0,0],    # Y position  
                 [0,0,1,0,0,0,0],    # R color
                 [0,0,0,1,0,0,0],    # G color
                 [0,0,0,0,1,0,0]]    # B color
  
  Theta_smm_B = [[0,0,0,0,0,1,0],    # X shape extent
                 [0,0,0,0,0,0,1]]    # Y shape extent
  
  # Color variances with Gamma priors
  Theta_smm_sigma[k,c] ~ Gamma(α_sigma, β_sigma)
  
  # Hyperparameters
  alpha_smm = 1.0
  mu_slot_prior = [0.0, 0.0, 0.5, 0.5, 0.5, 0.1, 0.1]  # Center, gray, small
  Sigma_slot_prior = 0.1 * I_7

Equations:
  # Generative model for pixels
  p(o_pixels[n] | s_slot, z_slot_assign[n]) = 
    ∏_{k=1}^K N(Θ_smm_A · s_slot[k], 
                 diag([Θ_smm_B · s_slot[k], σ_c[k]]))^{z_slot_assign[n,k]}
  
  # Where:
  # - Θ_smm_A selects position(2) + color(3) features
  # - Θ_smm_B selects shape(2) features for spatial covariance
  # - σ_c[k] provides fixed color variance per slot
  
  # Slot assignment probability
  p(z_slot_assign[n,k] = 1) = Theta_smm_pi[k]
  
  # Presence and movement dynamics
  p(z_slot_presence[k,t] | z_slot_presence[k,t-1], data) = 
    σ(α_presence + β_presence · evidence[k,t])
  
  p(z_slot_moving[k,t] | s_slot[k,t], s_slot[k,t-1]) = 
    σ(α_moving + β_moving · ||s_slot[k,t] - s_slot[k,t-1]||)
  
  # Structure learning criterion
  AddNewSlot() = max_k ℓ_{t,k} < τ_smm + log(alpha_smm)
  
  # Where ℓ_{t,k} is posterior predictive log-likelihood for slot k

Time:
  ModelTimeHorizon: T_max
  DiscreteTime: true
  Dynamic: true

ActInfOntologyAnnotation:
  s_slot: "object_state_representation"
  o_pixels: "sensory_observation" 
  z_slot_assign: "perceptual_binding"
  z_slot_presence: "object_existence_belief"
  z_slot_moving: "object_motion_state"
  Theta_smm_pi: "prior_beliefs_about_objects"
  Theta_smm_A: "sensory_mapping_parameters"
  c_slot_unused: "slot_utilization_tracking"

Footer:
  GeneratedBy: "AXIOM-GNN Specification Generator v1.4"
  CreatedDate: "2025-01-23"
  LastModified: "2025-01-23"
  Version: "1.0.0"
  
Signature:
  Author: "AXIOM Research Team - sMM Module"
  Institution: "VERSES AI / Active Inference Institute"
  ContactEmail: "axiom-smm@verses.ai"
  DOI: "10.5281/zenodo.axiom.gnn.smm" 