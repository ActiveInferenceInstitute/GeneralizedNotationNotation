# GNN Specification: AXIOM Active Inference Planning
GNNVersionAndFlags: 1.4

ModelName: AXIOM_Planning

ModelAnnotation: |
  Active inference planning module implementing expected free energy
  minimization with utility maximization and information gain. Plans
  optimal action sequences by balancing reward-seeking and exploration.

StateSpaceBlock:
  # Planning horizon
  H_planning[1,discrete]           ### Planning horizon length
  
  # Policy space
  pi_actions[H,A,continuous]       ### Action probabilities over horizon
  u_planned[H,discrete]            ### Planned action sequence
  
  # Predicted trajectories  
  s_predicted[H,K,7,continuous]    ### Predicted slot trajectories
  r_predicted[H,continuous]        ### Predicted rewards
  o_predicted[H,N,5,continuous]    ### Predicted observations
  
  # Expected free energy components
  G_expected_free_energy[H,continuous] ### Expected free energy per timestep
  U_pragmatic[H,continuous]        ### Pragmatic utility (reward expectation)
  IG_epistemic[H,continuous]       ### Epistemic information gain
  
  # Policy evaluation
  Q_policy[A,continuous]           ### Action values (Q-function)
  gamma_precision[1,continuous]    ### Inverse temperature parameter
  
  # Rollout parameters
  N_rollouts[1,discrete]           ### Number of rollout samples
  rollout_policies[N_rollouts,H,A,continuous] ### Sampled rollout policies

Connections:
  s_slot -> s_predicted             ### Current slots to predicted trajectories
  pi_actions -> s_predicted         ### Actions influence predictions
  s_predicted -> r_predicted        ### Trajectories determine rewards
  s_predicted -> o_predicted        ### Trajectories determine observations
  s_predicted -> IG_epistemic       ### Trajectories provide information
  r_predicted -> U_pragmatic        ### Rewards provide utility
  U_pragmatic -> G_expected_free_energy ### Utility contributes to free energy
  IG_epistemic -> G_expected_free_energy ### Information gain contributes to free energy
  G_expected_free_energy -> pi_actions  ### Free energy determines policy

InitialParameterization:
  # Planning parameters
  H_planning = 16                   ### 16-step planning horizon
  N_rollouts = 512                 ### 512 rollout samples per planning step
  gamma_precision = 16.0           ### High precision (low temperature)
  
  # Initial policy (uniform)
  pi_actions[h,a] ~ Uniform(0, 1) for all h,a
  Normalize(pi_actions[h,:]) = 1 for all h
  
  # Initial predictions (copy current state)
  s_predicted[0] = s_slot_current
  r_predicted[0] = 0.0
  o_predicted[0] = o_current

Equations:
  # Expected free energy decomposition
  G_expected_free_energy[τ] = -U_pragmatic[τ] - IG_epistemic[τ]
  
  # Pragmatic value (expected utility)
  U_pragmatic[τ] = E_{q(s_τ|π)}[log p(r_τ | s_τ, π)]
  
  # Where expectation is over predicted trajectories
  U_pragmatic[τ] = ∑_rollouts (1/N_rollouts) · log p(r_predicted[τ,rollout] | s_predicted[τ,rollout], π)
  
  # Epistemic value (expected information gain about parameters)
  IG_epistemic[τ] = D_KL[q(Theta_rmm | s_τ, π) || q(Theta_rmm)]
  
  # Approximated as expected model uncertainty reduction
  IG_epistemic[τ] = E_{q(s_τ|π)}[H[q(Theta_rmm | s_{1:τ-1})] - H[q(Theta_rmm | s_{1:τ})]]
  
  # Where H[·] is entropy and expectation is over predicted trajectories
  
  # Optimal policy via softmax
  π*[τ,a] = σ(γ · Q_policy[a]) = exp(γ · Q_policy[a]) / ∑_a' exp(γ · Q_policy[a'])
  
  # Where Q-values sum expected free energy over future
  Q_policy[a] = -∑_{τ=0}^{H-1} G_expected_free_energy[τ] | u[0] = a
  
  # Forward simulation for trajectory prediction
  For τ = 0 to H-1:
    # Sample action from current policy
    u_sampled[τ] ~ π_actions[τ]
    
    # Predict next slot states via tMM
    s_predicted[τ+1] = ForwardDynamics(s_predicted[τ], u_sampled[τ])
    
    # Predict observations via sMM  
    o_predicted[τ+1] = ForwardObservation(s_predicted[τ+1])
    
    # Predict rewards via rMM
    r_predicted[τ+1] = ForwardReward(s_predicted[τ+1], u_sampled[τ])
  
  # Forward dynamics (tMM prediction)
  ForwardDynamics(s_slot, u_action):
    # Predict context from rMM
    s_rmm_predicted = PredictContext(s_slot, u_action)
    
    # Predict dynamics mode from rMM
    s_tmm_predicted = PredictDynamicsMode(s_rmm_predicted)
    
    # Apply linear dynamics from tMM
    s_slot_next = ∑_l s_tmm_predicted[l] · (Theta_tmm_D[l] · s_slot + Theta_tmm_b[l])
    
    return s_slot_next
  
  # Forward observation (sMM prediction)
  ForwardObservation(s_slot):
    # Generate pixel observations from slots
    o_pixels = ∑_k ∑_n z_slot_assign[n,k] · N(Θ_smm_A · s_slot[k], Θ_smm_Sigma[k])
    
    return o_pixels
  
  # Forward reward (rMM prediction)  
  ForwardReward(s_slot, u_action):
    # Compute context features
    f_continuous = ComputeContinuousFeatures(s_slot)
    d_discrete = ComputeDiscreteFeatures(u_action, previous_state)
    
    # Predict context assignment
    s_rmm_predicted = PredictContextAssignment(f_continuous, d_discrete)
    
    # Predict reward from context
    r_reward = ∑_m s_rmm_predicted[m] · Theta_rmm_reward[m]
    
    return r_reward
  
  # Policy optimization via coordinate ascent
  PolicyUpdate():
    For τ = 0 to H-1:
      For a = 1 to A:
        # Compute Q-value for action a at timestep τ
        Q_policy[a] = -∑_{τ'=τ}^{H-1} E[G_expected_free_energy[τ'] | u[τ] = a]
        
      # Update policy via softmax
      π_actions[τ,a] = exp(γ · Q_policy[a]) / ∑_a' exp(γ · Q_policy[a'])
  
  # Monte Carlo estimation of expected values
  MonteCarloEstimation(N_samples=N_rollouts):
    expectations = []
    For i = 1 to N_samples:
      # Sample rollout trajectory
      trajectory = SampleTrajectory(π_actions, H_planning)
      
      # Compute trajectory values
      utility_trajectory = ∑_τ log p(r_predicted[τ] | trajectory)
      information_trajectory = ComputeInformationGain(trajectory)
      
      expectations.append([utility_trajectory, information_trajectory])
    
    # Return sample averages
    return mean(expectations, axis=0)

Time:
  ModelTimeHorizon: H_planning
  DiscreteTime: true
  Dynamic: true

ActInfOntologyAnnotation:
  H_planning: "planning_horizon"
  pi_actions: "policy_distribution"
  u_planned: "planned_action_sequence"
  s_predicted: "predicted_hidden_states"
  r_predicted: "predicted_rewards"
  o_predicted: "predicted_observations"
  G_expected_free_energy: "expected_free_energy"
  U_pragmatic: "expected_utility"
  IG_epistemic: "expected_information_gain"
  Q_policy: "action_value_function"
  gamma_precision: "policy_precision_parameter"
  N_rollouts: "monte_carlo_sample_size"
  rollout_policies: "sampled_policy_trajectories"

Footer:
  GeneratedBy: "AXIOM-GNN Specification Generator v1.4"
  CreatedDate: "2025-01-23"
  LastModified: "2025-01-23"
  Version: "1.0.0"
  
Signature:
  Author: "AXIOM Research Team - Planning Module"
  Institution: "VERSES AI / Active Inference Institute"
  ContactEmail: "axiom-planning@verses.ai"
  DOI: "TBD - Not yet published" 