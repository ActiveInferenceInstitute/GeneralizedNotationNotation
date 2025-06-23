# PyMDP Advanced Tutorials and Research Applications

## Overview

This document provides advanced tutorials and cutting-edge research applications using PyMDP in conjunction with GNN models. These examples showcase the latest developments in Active Inference and demonstrate sophisticated modeling techniques.

## Table of Contents

1. [Real-Time Learning and Adaptation](#real-time-learning-and-adaptation)
2. [Multi-Modal Sensory Integration](#multi-modal-sensory-integration)  
3. [Emotion and Interoception Modeling](#emotion-and-interoception-modeling)
4. [Collective Intelligence and Swarm Behavior](#collective-intelligence-and-swarm-behavior)
5. [Neurodevelopmental Modeling](#neurodevelopmental-modeling)
6. [Advanced Temporal Dynamics](#advanced-temporal-dynamics)
7. [Research Integration Patterns](#research-integration-patterns)

## Real-Time Learning and Adaptation

### Adaptive Learning Rates

```python
"""
Real-time adaptation of learning rates based on prediction error
"""
import numpy as np
from pymdp.agent import Agent
from pymdp import utils

class AdaptiveLearningAgent:
    def __init__(self, A, B, C, D, base_lr=0.1, adaptation_rate=0.01):
        self.base_lr = base_lr
        self.adaptation_rate = adaptation_rate
        self.prediction_errors = []
        self.learning_rates = {'A': base_lr, 'B': base_lr}
        
        # Initialize agent with dynamic learning rates
        self.agent = Agent(A=A, B=B, C=C, D=D,
                          lr_pA=self.learning_rates['A'],
                          lr_pB=self.learning_rates['B'],
                          use_param_info_gain=True)
        
    def adapt_learning_rates(self, prediction_error):
        """Adapt learning rates based on recent prediction errors"""
        self.prediction_errors.append(prediction_error)
        
        # Keep only recent errors
        if len(self.prediction_errors) > 50:
            self.prediction_errors.pop(0)
        
        # Calculate error trend
        if len(self.prediction_errors) >= 10:
            recent_errors = self.prediction_errors[-10:]
            error_trend = np.mean(recent_errors) - np.mean(self.prediction_errors[:-10])
            
            # Increase learning rate if errors are increasing
            if error_trend > 0:
                self.learning_rates['A'] = min(0.5, self.learning_rates['A'] + self.adaptation_rate)
                self.learning_rates['B'] = min(0.5, self.learning_rates['B'] + self.adaptation_rate)
            else:
                # Decrease learning rate if errors are decreasing
                self.learning_rates['A'] = max(0.01, self.learning_rates['A'] - self.adaptation_rate/2)
                self.learning_rates['B'] = max(0.01, self.learning_rates['B'] - self.adaptation_rate/2)
            
            # Update agent learning rates
            self.agent.lr_pA = self.learning_rates['A']
            self.agent.lr_pB = self.learning_rates['B']
    
    def step(self, observation, true_state=None):
        """Execute one step with adaptive learning"""
        # Predict observation
        qs = self.agent.infer_states(observation)
        
        # Calculate prediction error if true state is known
        if true_state is not None:
            predicted_state = np.argmax(qs[0])
            prediction_error = 1.0 if predicted_state != true_state else 0.0
            self.adapt_learning_rates(prediction_error)
        
        # Select action
        action = self.agent.sample_action()
        
        return action, qs, self.learning_rates

# Example usage with changing environment
def simulate_non_stationary_environment(agent, num_steps=1000):
    """Simulate an environment that changes its dynamics"""
    results = {'learning_rates': [], 'prediction_errors': [], 'accuracy': []}
    
    # Environment switches dynamics halfway through
    switch_point = num_steps // 2
    
    for step in range(num_steps):
        # Generate observation based on current environment regime
        if step < switch_point:
            # Environment 1: predictable pattern
            true_state = step % 4
        else:
            # Environment 2: different pattern
            true_state = (step * 2) % 4
        
        obs = [true_state + np.random.randint(-1, 2)]  # Noisy observation
        obs[0] = max(0, min(3, obs[0]))  # Clip to valid range
        
        action, qs, lr_dict = agent.step([obs[0]], true_state)
        
        # Track performance
        predicted_state = np.argmax(qs[0])
        accuracy = 1.0 if predicted_state == true_state else 0.0
        
        results['learning_rates'].append(lr_dict.copy())
        results['prediction_errors'].append(agent.prediction_errors[-1] if agent.prediction_errors else 0)
        results['accuracy'].append(accuracy)
        
        if step % 100 == 0:
            recent_accuracy = np.mean(results['accuracy'][-20:])
            print(f"Step {step}: LR_A={lr_dict['A']:.4f}, Accuracy={recent_accuracy:.3f}")
    
    return results
```

## Multi-Modal Sensory Integration

### Cross-Modal Binding and Sensory Fusion

```python
"""
Multi-modal sensory integration using PyMDP
"""

class MultiModalAgent:
    def __init__(self, visual_dim=8, auditory_dim=6, tactile_dim=4):
        self.visual_dim = visual_dim
        self.auditory_dim = auditory_dim
        self.tactile_dim = tactile_dim
        
        # Multi-modal state space
        # States represent bound multi-modal objects
        num_objects = 5
        
        # Observation models for each modality
        A = utils.obj_array(3)  # Visual, auditory, tactile
        A[0] = utils.random_A_matrix([visual_dim], [num_objects])    # Visual features
        A[1] = utils.random_A_matrix([auditory_dim], [num_objects])  # Auditory features  
        A[2] = utils.random_A_matrix([tactile_dim], [num_objects])   # Tactile features
        
        # Dynamics (object persistence)
        B = utils.obj_array(1)
        B[0] = utils.random_B_matrix([num_objects], [3])  # Attend, ignore, explore
        
        # Preferences (attend to coherent multi-modal objects)
        C = utils.obj_array(3)
        C[0] = np.zeros(visual_dim)   # No visual preference
        C[1] = np.zeros(auditory_dim) # No auditory preference  
        C[2] = np.array([2.0, 1.0, 0.0, -1.0])  # Prefer certain tactile sensations
        
        # Uniform priors
        D = utils.obj_array(1)
        D[0] = utils.uniform_categorical([num_objects])
        
        self.agent = Agent(A=A, B=B, C=C, D=D,
                          use_utility=True,
                          use_states_info_gain=True,
                          policy_len=3)
        
    def cross_modal_binding(self, visual_obs, auditory_obs, tactile_obs):
        """Bind cross-modal observations to unified object representation"""
        # Combine multi-modal observations
        multi_obs = [visual_obs, auditory_obs, tactile_obs]
        
        # Infer unified object state
        qs = self.agent.infer_states(multi_obs)
        
        # Calculate binding strength (certainty about object identity)
        binding_strength = 1.0 - (-np.sum(qs[0] * np.log(qs[0] + 1e-16)))
        
        # Select attention action
        action = self.agent.sample_action()
        
        return qs, binding_strength, action
    
    def simulate_cross_modal_illusion(self, num_trials=100):
        """Simulate cross-modal illusions (e.g., McGurk effect)"""
        results = {'binding_strength': [], 'illusion_strength': [], 'object_identity': []}
        
        for trial in range(num_trials):
            # Create congruent or incongruent multi-modal stimuli
            if trial % 3 == 0:  # Incongruent trial
                visual_obs = 1  # Object A visual features
                auditory_obs = 3  # Object B auditory features  
                tactile_obs = 2   # Object C tactile features
                congruent = False
            else:  # Congruent trial
                object_id = np.random.choice(5)
                visual_obs = object_id if np.random.rand() > 0.1 else np.random.choice(8)
                auditory_obs = object_id if np.random.rand() > 0.1 else np.random.choice(6)
                tactile_obs = object_id if np.random.rand() > 0.1 else np.random.choice(4)
                congruent = True
            
            qs, binding_strength, action = self.cross_modal_binding(
                visual_obs, auditory_obs, tactile_obs)
            
            # Measure illusion strength (fusion of incongruent inputs)
            if not congruent:
                # Strong binding despite incongruence indicates illusion
                illusion_strength = binding_strength
            else:
                illusion_strength = 0.0
            
            predicted_object = np.argmax(qs[0])
            
            results['binding_strength'].append(binding_strength)
            results['illusion_strength'].append(illusion_strength)
            results['object_identity'].append(predicted_object)
        
        return results

# Example usage
multi_modal_agent = MultiModalAgent()
illusion_results = multi_modal_agent.simulate_cross_modal_illusion(200)

print(f"Average binding strength: {np.mean(illusion_results['binding_strength']):.3f}")
print(f"Average illusion strength: {np.mean(illusion_results['illusion_strength']):.3f}")
```

## Emotion and Interoception Modeling

### Affective Active Inference

```python
"""
Modeling emotion and interoception using Active Inference
"""

class AffectiveAgent:
    def __init__(self):
        # Interoceptive state space (physiological states)
        self.physio_states = 6  # Arousal levels
        self.external_states = 4  # Environmental contexts
        
        # Observation modalities
        num_intero_obs = 5    # Interoceptive sensations (heart rate, etc.)
        num_extero_obs = 6    # External sensory observations
        num_affective_obs = 4 # Affective feelings (valence categories)
        
        # Multi-factor state space: [physiological, environmental]
        A = utils.obj_array(3)
        A[0] = utils.random_A_matrix([num_intero_obs], [self.physio_states, self.external_states])
        A[1] = utils.random_A_matrix([num_extero_obs], [self.physio_states, self.external_states])
        A[2] = utils.random_A_matrix([num_affective_obs], [self.physio_states, self.external_states])
        
        # Transition dynamics
        B = utils.obj_array(2)
        B[0] = utils.random_B_matrix([self.physio_states], [4])  # Physiological regulation
        B[1] = utils.random_B_matrix([self.external_states], [3])  # Environmental navigation
        
        # Preferences (homeostatic regulation)
        C = utils.obj_array(3)
        C[0] = np.array([2.0, 1.0, 0.0, -1.0, -2.0])  # Prefer moderate interoception
        C[1] = np.array([1.0, 1.0, 0.0, 0.0, -1.0, -2.0])  # Environmental preferences
        C[2] = np.array([3.0, 1.0, -1.0, -3.0])  # Strong valence preferences
        
        # Priors (resting state)
        D = utils.obj_array(2)
        D[0] = utils.uniform_categorical([self.physio_states])
        D[1] = utils.uniform_categorical([self.external_states])
        
        self.agent = Agent(A=A, B=B, C=C, D=D,
                          use_utility=True,
                          use_states_info_gain=True,
                          policy_len=4)
        
        # Emotion tracking
        self.emotion_history = []
        
    def compute_emotion(self, qs, observations):
        """Compute emotional state from beliefs and observations"""
        # Physiological arousal
        arousal = np.dot(qs[0], np.arange(len(qs[0]))) / len(qs[0])
        
        # Valence from affective observations
        valence_obs = observations[2]
        valence = (valence_obs - 1.5) / 1.5  # Normalize to [-1, 1]
        
        # Uncertainty (anxiety/stress)
        uncertainty = -np.sum(qs[0] * np.log(qs[0] + 1e-16))
        
        emotion = {
            'arousal': arousal,
            'valence': valence,
            'uncertainty': uncertainty,
            'dominance': 1.0 - uncertainty  # Feeling of control
        }
        
        return emotion
    
    def emotional_regulation_step(self, intero_obs, extero_obs, affect_obs):
        """Execute one step of emotional regulation"""
        observations = [intero_obs, extero_obs, affect_obs]
        
        # Inference
        qs = self.agent.infer_states(observations)
        
        # Compute emotional state
        emotion = self.compute_emotion(qs, observations)
        self.emotion_history.append(emotion)
        
        # Action selection (regulation strategy)
        action = self.agent.sample_action()
        
        # Interpret actions as regulation strategies
        physio_action = action[0]  # 0: relax, 1: activate, 2: suppress, 3: amplify
        environ_action = action[1]  # 0: approach, 1: avoid, 2: explore
        
        regulation_strategy = {
            'physiological': ['relax', 'activate', 'suppress', 'amplify'][physio_action],
            'environmental': ['approach', 'avoid', 'explore'][environ_action]
        }
        
        return emotion, regulation_strategy, qs
    
    def simulate_emotional_episode(self, stressor_onset=20, episode_length=100):
        """Simulate emotional episode with stressor"""
        results = {
            'emotions': [],
            'regulation_strategies': [],
            'beliefs': []
        }
        
        for step in range(episode_length):
            # Simulate observations
            if stressor_onset <= step < stressor_onset + 30:
                # Stressor present: high arousal, negative valence
                intero_obs = min(4, np.random.poisson(3))  # High interoception
                extero_obs = 5  # Threatening environment
                affect_obs = 0  # Negative valence
            else:
                # Normal conditions
                intero_obs = np.random.poisson(1)  # Normal interoception
                extero_obs = np.random.choice([0, 1, 2])  # Neutral environment
                affect_obs = np.random.choice([1, 2])  # Neutral to positive valence
            
            emotion, strategy, qs = self.emotional_regulation_step(
                intero_obs, extero_obs, affect_obs)
            
            results['emotions'].append(emotion)
            results['regulation_strategies'].append(strategy)
            results['beliefs'].append([q.copy() for q in qs])
            
            if step % 20 == 0:
                print(f"Step {step}: Arousal={emotion['arousal']:.2f}, "
                      f"Valence={emotion['valence']:.2f}, Strategy={strategy['physiological']}")
        
        return results

# Example usage
affective_agent = AffectiveAgent()
episode_results = affective_agent.simulate_emotional_episode(stressor_onset=30, episode_length=120)

# Analyze emotional dynamics
emotions = episode_results['emotions']
arousal_trajectory = [e['arousal'] for e in emotions]
valence_trajectory = [e['valence'] for e in emotions]

print(f"Peak arousal: {max(arousal_trajectory):.3f}")
print(f"Emotional recovery time: {len([a for a in arousal_trajectory[60:] if a > 0.5])} steps")
```

## Collective Intelligence and Swarm Behavior

### Distributed Active Inference

```python
"""
Collective intelligence using distributed active inference
"""

class SwarmAgent:
    def __init__(self, agent_id, swarm_size, communication_range=2):
        self.agent_id = agent_id
        self.swarm_size = swarm_size
        self.communication_range = communication_range
        
        # Individual state space
        self.spatial_states = 16  # Grid positions
        self.resource_states = 4  # Resource levels
        
        # Observations: local environment + neighbor communications
        local_obs_dim = 8  # Local environmental features
        comm_obs_dim = swarm_size  # Communication from other agents
        
        A = utils.obj_array(2)
        A[0] = utils.random_A_matrix([local_obs_dim], [self.spatial_states, self.resource_states])
        A[1] = utils.random_A_matrix([comm_obs_dim], [self.spatial_states, self.resource_states])
        
        # Movement and communication actions
        B = utils.obj_array(2)
        B[0] = utils.random_B_matrix([self.spatial_states], [5])  # 4 directions + stay
        B[1] = utils.random_B_matrix([self.resource_states], [3])  # Collect, share, conserve
        
        # Preferences: find resources and coordinate with others
        C = utils.obj_array(2)
        C[0] = np.random.rand(local_obs_dim)  # Environmental preferences
        C[1] = np.ones(comm_obs_dim) * 0.5   # Moderate communication preference
        
        # Priors
        D = utils.obj_array(2)
        D[0] = utils.uniform_categorical([self.spatial_states])
        D[1] = utils.uniform_categorical([self.resource_states])
        
        self.agent = Agent(A=A, B=B, C=C, D=D,
                          use_utility=True,
                          use_states_info_gain=True,
                          policy_len=3)
        
        # Swarm coordination
        self.neighbors = []
        self.shared_beliefs = {}
        self.collective_memory = []
        
    def update_neighbors(self, all_agents, positions):
        """Update neighbor list based on spatial proximity"""
        my_pos = positions[self.agent_id]
        self.neighbors = []
        
        for other_id, other_pos in enumerate(positions):
            if other_id != self.agent_id:
                distance = np.linalg.norm(np.array(my_pos) - np.array(other_pos))
                if distance <= self.communication_range:
                    self.neighbors.append(other_id)
    
    def communicate_beliefs(self, other_agents):
        """Share beliefs with neighboring agents"""
        if not hasattr(self, 'current_beliefs'):
            return
        
        # Send beliefs to neighbors
        for neighbor_id in self.neighbors:
            if neighbor_id < len(other_agents):
                other_agents[neighbor_id].receive_communication(
                    self.agent_id, self.current_beliefs)
    
    def receive_communication(self, sender_id, beliefs):
        """Receive and integrate beliefs from other agents"""
        self.shared_beliefs[sender_id] = beliefs
    
    def collective_inference(self, local_obs):
        """Perform inference incorporating neighbor information"""
        # Create communication observation from neighbor beliefs
        comm_obs = np.zeros(self.swarm_size)
        for neighbor_id in self.neighbors:
            if neighbor_id in self.shared_beliefs:
                # Simplified: use most likely state of neighbor
                neighbor_belief = self.shared_beliefs[neighbor_id][0]
                comm_obs[neighbor_id] = np.argmax(neighbor_belief)
        
        # Combined observation
        observations = [local_obs, np.argmax(comm_obs) if np.any(comm_obs) else 0]
        
        # Inference
        qs = self.agent.infer_states(observations)
        self.current_beliefs = qs
        
        # Action selection
        action = self.agent.sample_action()
        
        return qs, action
    
    def update_collective_memory(self, observation, action, reward):
        """Update collective memory with experience"""
        experience = {
            'agent_id': self.agent_id,
            'observation': observation,
            'action': action,
            'reward': reward,
            'neighbors': self.neighbors.copy()
        }
        self.collective_memory.append(experience)
        
        # Keep memory bounded
        if len(self.collective_memory) > 100:
            self.collective_memory.pop(0)

class SwarmEnvironment:
    def __init__(self, grid_size=8, num_resources=5, num_agents=10):
        self.grid_size = grid_size
        self.num_resources = num_resources
        self.num_agents = num_agents
        
        # Initialize resource locations
        self.resources = np.random.randint(0, grid_size, size=(num_resources, 2))
        self.resource_values = np.random.rand(num_resources) * 10
        
        # Agent positions
        self.agent_positions = np.random.randint(0, grid_size, size=(num_agents, 2))
        
    def get_local_observation(self, agent_pos):
        """Get local environmental observation for agent"""
        obs = np.zeros(8)
        
        # Distance to nearest resources
        distances = [np.linalg.norm(agent_pos - resource) for resource in self.resources]
        nearest_resource = np.argmin(distances)
        
        obs[0] = distances[nearest_resource] / (self.grid_size * np.sqrt(2))  # Normalized distance
        obs[1] = self.resource_values[nearest_resource] / 10  # Normalized value
        
        # Local density of other agents
        agent_distances = [np.linalg.norm(agent_pos - other_pos) 
                          for other_pos in self.agent_positions]
        local_density = sum(1 for d in agent_distances if d < 3 and d > 0)
        obs[2] = local_density / self.num_agents
        
        # Fill remaining observations with environmental features
        obs[3:] = np.random.rand(5) * 0.1  # Environmental noise
        
        return obs

def simulate_swarm_foraging(num_agents=8, num_steps=200):
    """Simulate collective foraging behavior"""
    # Initialize environment and agents
    env = SwarmEnvironment(grid_size=10, num_resources=6, num_agents=num_agents)
    agents = [SwarmAgent(i, num_agents) for i in range(num_agents)]
    
    results = {
        'collective_performance': [],
        'individual_rewards': [[] for _ in range(num_agents)],
        'coordination_level': [],
        'resource_discovery': []
    }
    
    for step in range(num_steps):
        # Update neighbor relationships
        for agent in agents:
            agent.update_neighbors(agents, env.agent_positions)
        
        # Communication phase
        for agent in agents:
            agent.communicate_beliefs(agents)
        
        # Action phase
        step_rewards = []
        for i, agent in enumerate(agents):
            local_obs = env.get_local_observation(env.agent_positions[i])
            qs, action = agent.collective_inference(local_obs)
            
            # Execute movement action
            move_action = action[0]
            if move_action < 4:  # Valid movement
                direction_map = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # N, E, S, W
                dx, dy = direction_map[move_action]
                new_pos = env.agent_positions[i] + np.array([dx, dy])
                
                # Boundary checking
                new_pos = np.clip(new_pos, 0, env.grid_size - 1)
                env.agent_positions[i] = new_pos
            
            # Calculate reward (resource collection)
            reward = 0
            for j, resource_pos in enumerate(env.resources):
                if np.linalg.norm(env.agent_positions[i] - resource_pos) < 1.5:
                    reward += env.resource_values[j] * 0.1
            
            step_rewards.append(reward)
            results['individual_rewards'][i].append(reward)
            
            # Update agent memory
            agent.update_collective_memory(local_obs, action, reward)
        
        # Calculate collective metrics
        total_reward = sum(step_rewards)
        results['collective_performance'].append(total_reward)
        
        # Coordination level (similarity of agent behaviors)
        agent_actions = [len(agent.neighbors) for agent in agents]
        coordination = 1.0 - (np.std(agent_actions) / (np.mean(agent_actions) + 1e-6))
        results['coordination_level'].append(coordination)
        
        # Resource discovery (agents near resources)
        agents_near_resources = 0
        for agent_pos in env.agent_positions:
            for resource_pos in env.resources:
                if np.linalg.norm(agent_pos - resource_pos) < 2.0:
                    agents_near_resources += 1
                    break
        results['resource_discovery'].append(agents_near_resources / num_agents)
        
        if step % 50 == 0:
            print(f"Step {step}: Collective reward={total_reward:.2f}, "
                  f"Coordination={coordination:.3f}, "
                  f"Resource discovery={results['resource_discovery'][-1]:.3f}")
    
    return results, agents, env

# Example usage
swarm_results, swarm_agents, swarm_env = simulate_swarm_foraging(num_agents=6, num_steps=300)

print("\nSwarm Intelligence Analysis:")
print(f"Average collective performance: {np.mean(swarm_results['collective_performance']):.3f}")
print(f"Peak coordination level: {max(swarm_results['coordination_level']):.3f}")
print(f"Final resource discovery rate: {swarm_results['resource_discovery'][-1]:.3f}") 