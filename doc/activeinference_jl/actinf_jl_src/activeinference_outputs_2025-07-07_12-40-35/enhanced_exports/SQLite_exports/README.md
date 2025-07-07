# ActiveInference.jl SQLite Export

To create the database, run:
```bash
sqlite3 activeinference_data.db < create_database.sql
```

## Tables
- `learning_curve`: learning_curve data
- `actions_trace`: actions_trace data
- `rewards_trace`: rewards_trace data
- `planning_rewards`: planning_rewards data
- `beliefs_trace`: beliefs_trace data
- `beliefs_over_time`: beliefs_over_time data
- `learning_comparison`: learning_comparison data
- `planning_actions`: planning_actions data
- `actions_per_trial_trace`: actions_per_trial_trace data
- `basic_simulation_trace`: basic_simulation_trace data
- `learning_comparison_trace`: learning_comparison_trace data
- `actions_over_time`: actions_over_time data
- `observations_over_time`: observations_over_time data
- `learning_trace`: learning_trace data
- `observations_trace`: observations_trace data
- `planning_trace`: planning_trace data
