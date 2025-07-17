# RxInfer.jl POMDP Model for Classic Active Inference POMDP Agent v1
# Generated from GNN specification
# This model describes a classic Active Inference agent for a discrete POMDP:
- One observation modality ("state_observation") with 3 possible outcomes.
- One hidden state factor ("location") with 3 possible states.
- The hidden state is fully controllable via 3 discrete actions.
- The agent's preferences are encoded as log-probabilities over observations.
- The agent has an initial policy prior (habit) encoded as log-probabilities over actions.
- All parameterizations are explicit and suitable for translation to code or simulation in any Active Inference framework.

using RxInfer
using Distributions
using Plots
using Random
using ProgressMeter

# Model parameters from GNN specification
const A_matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
const B_matrix = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
const C_preferences = [0.0, 0.0, 1.0]
const D_prior = [0.33333, 0.33333, 0.33333]
const E_habit = [0.33333, 0.33333, 0.33333]

const num_states = 3
const num_obs = 3
const num_actions = 3

# Create Dirichlet priors for parameter learning
const p_A = DirichletCollection([[1.1, 0.1, 0.1], [0.1, 1.1, 0.1], [0.1, 0.1, 1.1]])
const p_B = DirichletCollection([[[1.1, 0.1, 0.1], [0.1, 1.1, 0.1], [0.1, 0.1, 1.1]], [[1.1, 0.1, 0.1], [0.1, 1.1, 0.1], [0.1, 0.1, 1.1]], [[1.1, 0.1, 0.1], [0.1, 1.1, 0.1], [0.1, 0.1, 1.1]]])

# Goal state (preference for state 2 based on C vector)
const goal_state = Categorical(C_preferences)

# POMDP Model with Active Inference structure
@model function classic_active_inference_pomdp_agent_v1_pomdp_model(
    p_A, p_B, p_goal, p_control, previous_control, 
    p_previous_state, current_y, future_y, T, m_A, m_B
)
    # Model parameters with priors
    A ~ p_A
    B ~ p_B
    previous_state ~ p_previous_state
    
    # Parameter inference (learning from observations)
    current_state ~ DiscreteTransition(previous_state, B, previous_control)
    current_y ~ DiscreteTransition(current_state, A)

    prev_state = current_state
    
    # Inference-as-planning (future prediction)
    for t in 1:T
        controls[t] ~ p_control
        s[t] ~ DiscreteTransition(prev_state, m_B, controls[t])
        future_y[t] ~ DiscreteTransition(s[t], m_A)
        prev_state = s[t]
    end
    
    # Goal prior on final state
    s[end] ~ p_goal
end

# Initialize inference procedure
init = @initialization begin
    q(A) = DirichletCollection(diageye(3) .+ 0.1)
    q(B) = DirichletCollection(ones(3, 3, 3))
end

# Variational constraints for parameter learning
constraints = @constraints begin
    q(previous_state, previous_control, current_state, B) = q(previous_state, previous_control, current_state)q(B)
    q(current_state, current_y, A) = q(current_state, current_y)q(A)
    q(current_state, s, controls, B) = q(current_state, s, controls)q(B)
    q(s, future_y, A) = q(s, future_y)q(A)
end

# Utility functions for state/observation conversion
function state_to_index(state::Int)
    return state
end

function index_to_state(index::Int)
    return index
end

function observation_to_one_hot(obs::Int)
    return [i == obs ? 1.0 : 0.0 for i in 1:3]
end

function action_to_one_hot(action::Int)
    return [i == action ? 1.0 : 0.0 for i in 1:3]
end

# Main control loop function
function run_pomdp_control(T_steps = 10, n_experiments = 10)
    println("Running POMDP control for Classic Active Inference POMDP Agent v1")
    println("Parameters: 3 states, 3 observations, 3 actions")
    
    successes = []
    
    @showprogress for i in 1:n_experiments
        # Initialize state belief to uniform prior
        p_s = Categorical(D_prior)
        
        # Initialize previous action as neutral
        policy = [Categorical(E_habit)]
        prev_u = E_habit
        
        # Run control loop
        for t in 1:T_steps
            # Convert policy to action
            current_action = mode(first(policy))
            prev_u = action_to_one_hot(current_action)
            
            # Generate synthetic observation (in real scenario, this comes from environment)
            # For demonstration, we'll use the current state to generate observation
            current_state = mode(p_s)
            observation = argmax(A_matrix[:, current_state])
            last_observation = observation_to_one_hot(observation)
            
            # Perform inference using the POMDP model
            inference_result = infer(
                model = classic_active_inference_pomdp_agent_v1_pomdp_model(
                    p_A = p_A,
                    p_B = p_B,
                    T = max(T_steps - t, 1),
                    p_previous_state = p_s,
                    p_goal = goal_state,
                    p_control = vague(Categorical, 3),
                    m_A = mean(p_A),
                    m_B = mean(p_B)
                ),
                data = (
                    previous_control = prev_u,
                    current_y = last_observation,
                    future_y = UnfactorizedData(fill(missing, max(T_steps - t, 1)))
                ),
                constraints = constraints,
                initialization = init,
                iterations = 10
            )
            
            # Update beliefs based on inference results
            p_s = last(inference_result.posteriors[:current_state])
            policy = last(inference_result.posteriors[:controls])
            
            # Update model parameters globally
            global p_A = last(inference_result.posteriors[:A])
            global p_B = last(inference_result.posteriors[:B])
            
            # Check if goal reached (preference for state 2)
            if current_state == argmax(C_preferences)
                break
            end
        end
        
        # Record success if goal was reached
        final_state = mode(p_s)
        success = final_state == argmax(C_preferences)
        push!(successes, success)
    end
    
    success_rate = mean(successes)
    println("Control experiment completed. Success rate: $(round(success_rate * 100, digits=1))%")
    
    return successes, success_rate
end

# Visualization function
function plot_results(successes)
    p = bar(successes, 
            label="Success/Failure", 
            color=successes .? :green : :red,
            title="POMDP Control Results",
            xlabel="Experiment",
            ylabel="Success (1) / Failure (0)")
    display(p)
    return p
end

# Run the control experiment
println("Starting POMDP control experiment...")
successes, success_rate = run_pomdp_control()

# Plot results
plot_results(successes)

println("\nModel Summary:")
println("- States: 3")
println("- Observations: 3")
println("- Actions: 3")
println("- A Matrix (Likelihood):")
for (i, row) in enumerate(A_matrix)
    println("  [$(join(row, ", "))]")
end
println("- B Matrix (Transition):")
for (i, action_matrix) in enumerate(B_matrix)
    println("  Action $(i-1):")
    for row in action_matrix
        println("    [$(join(row, ", "))]")
    end
end
println("- C Preferences: [$(join(C_vector, ", "))]")
println("- D Prior: [$(join(D_vector, ", "))]")
println("- E Habit: [$(join(E_vector, ", "))]")
