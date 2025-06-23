# PyMDP Performance Optimization Guide

## Overview

This guide provides comprehensive performance optimization strategies, benchmarking tools, and scaling techniques for PyMDP applications with GNN models.

## Table of Contents

1. [Performance Benchmarking](#performance-benchmarking)
2. [Memory Optimization](#memory-optimization)
3. [Computational Optimization](#computational-optimization)
4. [Scaling Strategies](#scaling-strategies)
5. [Parallel Processing](#parallel-processing)
6. [Hardware Acceleration](#hardware-acceleration)

## Performance Benchmarking

### Comprehensive Benchmarking Suite

```python
"""
Comprehensive PyMDP performance benchmarking tools
"""
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
from pymdp.agent import Agent
from pymdp import utils
import cProfile
import pstats
from memory_profiler import profile
from typing import Dict, List, Tuple

class PyMDPBenchmark:
    def __init__(self):
        self.results = {}
        self.system_info = self._get_system_info()
        
    def _get_system_info(self):
        """Collect system information for benchmarking context"""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'python_version': psutil.__version__,
            'numpy_version': np.__version__
        }
    
    def benchmark_inference_scaling(self, max_states=32, max_obs=16, max_actions=8):
        """Benchmark inference performance across different model sizes"""
        results = {
            'model_sizes': [],
            'inference_times': [],
            'memory_usage': [],
            'setup_times': []
        }
        
        state_sizes = [2, 4, 8, 16, max_states]
        obs_sizes = [2, 4, 8, max_obs]
        action_sizes = [2, 4, max_actions]
        
        for num_states in state_sizes:
            for num_obs in obs_sizes:
                for num_actions in action_sizes:
                    if num_states * num_obs * num_actions > 4096:  # Skip very large models
                        continue
                    
                    # Setup timing
                    setup_start = time.time()
                    
                    # Create model
                    A = utils.random_A_matrix([num_obs], [num_states])
                    B = utils.random_B_matrix([num_states], [num_actions])
                    C = utils.obj_array([np.random.rand(num_obs)])
                    D = utils.uniform_categorical([num_states])
                    
                    agent = Agent(A=A, B=B, C=C, D=D)
                    setup_time = time.time() - setup_start
                    
                    # Memory before inference
                    process = psutil.Process()
                    memory_before = process.memory_info().rss / (1024**2)  # MB
                    
                    # Inference timing
                    obs = [0]
                    inference_times = []
                    
                    for _ in range(50):  # Multiple inference runs
                        start_time = time.time()
                        qs = agent.infer_states(obs)
                        action = agent.sample_action()
                        inference_times.append(time.time() - start_time)
                    
                    # Memory after inference
                    memory_after = process.memory_info().rss / (1024**2)  # MB
                    memory_used = memory_after - memory_before
                    
                    # Store results
                    model_size = (num_states, num_obs, num_actions)
                    results['model_sizes'].append(model_size)
                    results['inference_times'].append(np.mean(inference_times))
                    results['memory_usage'].append(memory_used)
                    results['setup_times'].append(setup_time)
                    
                    print(f"Model {model_size}: "
                          f"Inference={np.mean(inference_times)*1000:.2f}ms, "
                          f"Memory={memory_used:.1f}MB")
        
        return results
    
    def benchmark_planning_horizon(self, max_horizon=10):
        """Benchmark planning performance vs horizon length"""
        # Standard model size
        A = utils.random_A_matrix([6], [8])
        B = utils.random_B_matrix([8], [4])
        C = utils.obj_array([np.random.rand(6)])
        D = utils.uniform_categorical([8])
        
        results = {
            'horizons': [],
            'planning_times': [],
            'memory_usage': [],
            'policy_quality': []
        }
        
        for horizon in range(1, max_horizon + 1):
            agent = Agent(A=A, B=B, C=C, D=D, 
                         policy_len=horizon,
                         use_utility=True)
            
            obs = [2]
            planning_times = []
            policy_qualities = []
            
            process = psutil.Process()
            memory_before = process.memory_info().rss / (1024**2)
            
            for _ in range(20):  # Multiple planning runs
                start_time = time.time()
                qs = agent.infer_states(obs)
                
                if hasattr(agent, 'infer_policies'):
                    q_pi, neg_efe = agent.infer_policies()
                    planning_time = time.time() - start_time
                    
                    # Policy quality (negative entropy = more decisive)
                    policy_quality = -(-np.sum(q_pi * np.log(q_pi + 1e-16)))
                    policy_qualities.append(policy_quality)
                else:
                    planning_time = time.time() - start_time
                    policy_qualities.append(0)
                
                planning_times.append(planning_time)
            
            memory_after = process.memory_info().rss / (1024**2)
            memory_used = memory_after - memory_before
            
            results['horizons'].append(horizon)
            results['planning_times'].append(np.mean(planning_times))
            results['memory_usage'].append(memory_used)
            results['policy_quality'].append(np.mean(policy_qualities))
            
            print(f"Horizon {horizon}: "
                  f"Planning={np.mean(planning_times)*1000:.2f}ms, "
                  f"Quality={np.mean(policy_qualities):.3f}")
        
        return results
    
    def benchmark_learning_performance(self, num_episodes=10):
        """Benchmark learning algorithm performance"""
        A = utils.random_A_matrix([4], [6])
        B = utils.random_B_matrix([6], [3])
        C = utils.obj_array([np.array([1.0, 0.5, 0.0, -0.5])])
        D = utils.uniform_categorical([6])
        
        learning_configs = [
            {'lr_pA': 0.01, 'lr_pB': 0.01, 'use_param_info_gain': False},
            {'lr_pA': 0.05, 'lr_pB': 0.05, 'use_param_info_gain': False},
            {'lr_pA': 0.1, 'lr_pB': 0.1, 'use_param_info_gain': False},
            {'lr_pA': 0.1, 'lr_pB': 0.1, 'use_param_info_gain': True}
        ]
        
        results = {
            'configs': [],
            'learning_curves': [],
            'convergence_times': [],
            'final_performance': []
        }
        
        for config in learning_configs:
            agent = Agent(A=A.copy(), B=B.copy(), C=C, D=D, **config)
            
            episode_rewards = []
            convergence_time = None
            
            episode_start = time.time()
            
            for episode in range(num_episodes):
                episode_reward = 0
                
                for step in range(100):  # Steps per episode
                    obs = [np.random.choice(4)]
                    qs = agent.infer_states(obs)
                    action = agent.sample_action()
                    
                    # Simplified reward
                    reward = 1.0 if action[0] == obs[0] else 0.0
                    episode_reward += reward
                
                episode_rewards.append(episode_reward)
                
                # Check convergence (performance plateau)
                if (len(episode_rewards) >= 5 and 
                    convergence_time is None and
                    np.std(episode_rewards[-5:]) < 2.0):
                    convergence_time = time.time() - episode_start
            
            total_time = time.time() - episode_start
            if convergence_time is None:
                convergence_time = total_time
            
            results['configs'].append(config)
            results['learning_curves'].append(episode_rewards)
            results['convergence_times'].append(convergence_time)
            results['final_performance'].append(np.mean(episode_rewards[-5:]))
            
            print(f"Config {config}: "
                  f"Convergence={convergence_time:.1f}s, "
                  f"Final={np.mean(episode_rewards[-5:]):.1f}")
        
        return results
    
    def profile_detailed_inference(self, num_states=8, num_obs=6):
        """Detailed profiling of inference components"""
        A = utils.random_A_matrix([num_obs], [num_states])
        B = utils.random_B_matrix([num_states], [4])
        C = utils.obj_array([np.random.rand(num_obs)])
        D = utils.uniform_categorical([num_states])
        
        agent = Agent(A=A, B=B, C=C, D=D)
        
        def profile_target():
            for _ in range(100):
                obs = [np.random.choice(num_obs)]
                qs = agent.infer_states(obs)
                action = agent.sample_action()
        
        # Profile execution
        profiler = cProfile.Profile()
        profiler.enable()
        profile_target()
        profiler.disable()
        
        # Analyze results
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        # Extract key metrics
        total_calls = stats.total_calls
        total_time = stats.total_tt
        
        print(f"Total calls: {total_calls}")
        print(f"Total time: {total_time:.4f}s")
        print("\nTop 10 functions by cumulative time:")
        stats.print_stats(10)
        
        return stats

# Example usage
benchmark = PyMDPBenchmark()

# Run benchmarks
print("=== PyMDP Performance Benchmark ===")
print(f"System: {benchmark.system_info}")
print()

print("1. Inference Scaling Benchmark:")
inference_results = benchmark.benchmark_inference_scaling(max_states=16, max_obs=8, max_actions=4)

print("\n2. Planning Horizon Benchmark:")
planning_results = benchmark.benchmark_planning_horizon(max_horizon=8)

print("\n3. Learning Performance Benchmark:")
learning_results = benchmark.benchmark_learning_performance(num_episodes=20)

print("\n4. Detailed Profiling:")
profile_stats = benchmark.profile_detailed_inference(num_states=8, num_obs=6)
```

## Memory Optimization

### Efficient Matrix Storage

```python
"""
Memory optimization techniques for large PyMDP models
"""
import numpy as np
from scipy.sparse import csr_matrix
from pymdp import utils
from pymdp.agent import Agent

def optimize_sparse_matrices(A, B, sparsity_threshold=0.1):
    """Convert dense matrices to sparse format when beneficial"""
    A_opt = utils.obj_array(len(A))
    B_opt = utils.obj_array(len(B))
    
    for m, A_m in enumerate(A):
        density = np.count_nonzero(A_m) / A_m.size
        if density < sparsity_threshold:
            A_opt[m] = csr_matrix(A_m)
            print(f"A[{m}] converted to sparse (density: {density:.3f})")
        else:
            A_opt[m] = A_m
    
    for f, B_f in enumerate(B):
        if B_f.ndim == 3:
            # Check each action matrix separately
            sparse_actions = {}
            for a in range(B_f.shape[2]):
                action_matrix = B_f[:, :, a]
                density = np.count_nonzero(action_matrix) / action_matrix.size
                if density < sparsity_threshold:
                    sparse_actions[a] = csr_matrix(action_matrix)
                else:
                    sparse_actions[a] = action_matrix
            
            if len(sparse_actions) > 0:
                B_opt[f] = sparse_actions
                print(f"B[{f}] partially converted to sparse")
            else:
                B_opt[f] = B_f
        else:
            B_opt[f] = B_f
    
    return A_opt, B_opt

def reduce_precision(matrices, target_dtype=np.float32):
    """Reduce numerical precision to save memory"""
    optimized = []
    for matrix in matrices:
        if hasattr(matrix, 'astype'):
            optimized.append(matrix.astype(target_dtype))
        else:
            optimized.append(matrix)
    return optimized
```

## Computational Optimization

### Vectorized Operations

```python
"""
Optimized inference algorithms
"""
from numba import jit
import time

@jit(nopython=True)
def fast_categorical_sample(probs):
    """Fast categorical sampling"""
    cumsum = np.cumsum(probs)
    r = np.random.random()
    return np.searchsorted(cumsum, r)

@jit(nopython=True)
def fast_softmax(logits):
    """Numerically stable softmax"""
    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)
    return exp_logits / np.sum(exp_logits)

class OptimizedInference:
    def __init__(self, A, B, C, D):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        
        # Precompute log matrices for numerical stability
        self.log_A = [np.log(A_m + 1e-16) for A_m in A]
        self.log_D = [np.log(D_f + 1e-16) for D_f in D]
    
    def fast_infer_states(self, obs):
        """Optimized state inference"""
        qs = utils.obj_array(len(self.D))
        
        for f in range(len(self.A)):
            log_likelihood = self.log_A[f][obs[f], :]
            log_prior = self.log_D[f]
            log_posterior = log_likelihood + log_prior
            qs[f] = fast_softmax(log_posterior)
        
        return qs

def benchmark_inference_speed():
    """Benchmark standard vs optimized inference"""
    # Create test model
    A = utils.random_A_matrix([6], [8])
    B = utils.random_B_matrix([8], [4])
    C = utils.obj_array([np.random.rand(6)])
    D = utils.uniform_categorical([8])
    
    # Standard agent
    standard_agent = Agent(A=A, B=B, C=C, D=D)
    
    # Optimized inference
    optimized = OptimizedInference(A, B, C, D)
    
    obs = [2]
    num_trials = 1000
    
    # Benchmark standard inference
    start_time = time.time()
    for _ in range(num_trials):
        qs = standard_agent.infer_states(obs)
    standard_time = time.time() - start_time
    
    # Benchmark optimized inference
    start_time = time.time()
    for _ in range(num_trials):
        qs = optimized.fast_infer_states(obs)
    optimized_time = time.time() - start_time
    
    speedup = standard_time / optimized_time
    print(f"Standard: {standard_time:.4f}s")
    print(f"Optimized: {optimized_time:.4f}s")
    print(f"Speedup: {speedup:.2f}x")
    
    return speedup

# Run benchmark
print("=== Inference Speed Benchmark ===")
speedup = benchmark_inference_speed()
```

## Scaling Strategies

### Batch Processing

```python
"""
Batch processing for improved throughput
"""
import concurrent.futures
from multiprocessing import Pool

class BatchProcessor:
    def __init__(self, agent, batch_size=32, use_multiprocessing=False):
        self.agent = agent
        self.batch_size = batch_size
        self.use_multiprocessing = use_multiprocessing
    
    def process_batch(self, observations):
        """Process a batch of observations"""
        if self.use_multiprocessing:
            return self._multiprocess_batch(observations)
        else:
            return self._sequential_batch(observations)
    
    def _sequential_batch(self, observations):
        """Sequential batch processing"""
        results = []
        for obs in observations:
            qs = self.agent.infer_states(obs)
            action = self.agent.sample_action()
            results.append((qs, action))
        return results
    
    def _multiprocess_batch(self, observations):
        """Multiprocess batch processing"""
        def process_single(obs):
            qs = self.agent.infer_states(obs)
            action = self.agent.sample_action()
            return (qs, action)
        
        with Pool() as pool:
            results = pool.map(process_single, observations)
        
        return results
    
    def stream_process(self, observation_stream):
        """Process streaming observations in batches"""
        batch = []
        for obs in observation_stream:
            batch.append(obs)
            
            if len(batch) >= self.batch_size:
                yield self.process_batch(batch)
                batch = []
        
        # Process remaining observations
        if batch:
            yield self.process_batch(batch)

# Example usage
def generate_observations(num_obs=1000):
    """Generate test observations"""
    for _ in range(num_obs):
        yield [np.random.choice(4)]

# Create test agent
A = utils.random_A_matrix([4], [6])
B = utils.random_B_matrix([6], [3])
C = utils.obj_array([np.random.rand(4)])
D = utils.uniform_categorical([6])
agent = Agent(A=A, B=B, C=C, D=D)

# Batch processing
processor = BatchProcessor(agent, batch_size=50)
obs_stream = generate_observations(500)

print("Processing observation stream...")
batch_count = 0
for batch_results in processor.stream_process(obs_stream):
    batch_count += 1
    if batch_count % 5 == 0:
        print(f"Processed {batch_count * 50} observations")

print(f"Total batches processed: {batch_count}")
```

## Performance Monitoring

### Real-time Performance Tracking

```python
"""
Performance monitoring and profiling tools
"""
import psutil
import time
from collections import deque

class PerformanceMonitor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.inference_times = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.action_times = deque(maxlen=window_size)
        
    def start_timing(self):
        """Start timing an operation"""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    def end_timing(self, operation_type='inference'):
        """End timing and record metrics"""
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        execution_time = end_time - self.start_time
        memory_delta = end_memory - self.start_memory
        
        if operation_type == 'inference':
            self.inference_times.append(execution_time)
        elif operation_type == 'action':
            self.action_times.append(execution_time)
        
        self.memory_usage.append(end_memory)
        
        return execution_time, memory_delta
    
    def get_statistics(self):
        """Get performance statistics"""
        stats = {}
        
        if self.inference_times:
            stats['inference'] = {
                'mean': np.mean(self.inference_times),
                'std': np.std(self.inference_times),
                'min': min(self.inference_times),
                'max': max(self.inference_times)
            }
        
        if self.action_times:
            stats['action'] = {
                'mean': np.mean(self.action_times),
                'std': np.std(self.action_times),
                'min': min(self.action_times),
                'max': max(self.action_times)
            }
        
        if self.memory_usage:
            stats['memory'] = {
                'current': self.memory_usage[-1],
                'peak': max(self.memory_usage),
                'average': np.mean(self.memory_usage)
            }
        
        return stats
    
    def report(self):
        """Generate performance report"""
        stats = self.get_statistics()
        
        print("=== Performance Report ===")
        
        if 'inference' in stats:
            inf_stats = stats['inference']
            print(f"Inference: {inf_stats['mean']*1000:.2f} ± {inf_stats['std']*1000:.2f} ms")
            print(f"  Range: {inf_stats['min']*1000:.2f} - {inf_stats['max']*1000:.2f} ms")
        
        if 'action' in stats:
            act_stats = stats['action']
            print(f"Action: {act_stats['mean']*1000:.2f} ± {act_stats['std']*1000:.2f} ms")
        
        if 'memory' in stats:
            mem_stats = stats['memory']
            print(f"Memory: {mem_stats['current']:.1f} MB (peak: {mem_stats['peak']:.1f} MB)")

# Example usage with monitoring
monitor = PerformanceMonitor()
agent = Agent(A=A, B=B, C=C, D=D)

print("Running monitored simulation...")
for step in range(200):
    obs = [np.random.choice(4)]
    
    # Monitor inference
    monitor.start_timing()
    qs = agent.infer_states(obs)
    monitor.end_timing('inference')
    
    # Monitor action selection
    monitor.start_timing()
    action = agent.sample_action()
    monitor.end_timing('action')
    
    if step % 50 == 0:
        print(f"Step {step}")
        monitor.report()
        print()

print("Final performance summary:")
monitor.report()
```

## References

- [PyMDP Documentation](https://pymdp.readthedocs.io/)
- [NumPy Performance Tips](https://numpy.org/doc/stable/user/basics.performance.html)
- [SciPy Sparse Matrices](https://docs.scipy.org/doc/scipy/reference/sparse.html)
- [Numba Documentation](https://numba.pydata.org/)
- [Performance Profiling in Python](https://docs.python.org/3/library/profile.html) 