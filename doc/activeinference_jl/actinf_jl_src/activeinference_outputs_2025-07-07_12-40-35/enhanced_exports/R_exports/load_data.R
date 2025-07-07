# ActiveInference.jl R Data Import Script
# Generated: 2025-07-07T12:44:11.520

library(data.table)

# Function to load ActiveInference.jl data
load_activeinference_data <- function(data_dir = "/Users/4d/Documents/GitHub/GeneralizedNotationNotation/doc/activeinference_jl/actinf_jl_src/activeinference_outputs_2025-07-07_12-40-35/enhanced_exports/R_exports") {
  data_list <- list()
  
  # Load learning_curve
  learning_curve <- fread(file.path(data_dir, "learning_curve.csv"), skip = 3)
  data_list[["learning_curve"]] <- learning_curve
  
  # Load actions_trace
  actions_trace <- fread(file.path(data_dir, "actions_trace.csv"), skip = 3)
  data_list[["actions_trace"]] <- actions_trace
  
  # Load rewards_trace
  rewards_trace <- fread(file.path(data_dir, "rewards_trace.csv"), skip = 3)
  data_list[["rewards_trace"]] <- rewards_trace
  
  # Load planning_rewards
  planning_rewards <- fread(file.path(data_dir, "planning_rewards.csv"), skip = 3)
  data_list[["planning_rewards"]] <- planning_rewards
  
  # Load beliefs_trace
  beliefs_trace <- fread(file.path(data_dir, "beliefs_trace.csv"), skip = 3)
  data_list[["beliefs_trace"]] <- beliefs_trace
  
  # Load beliefs_over_time
  beliefs_over_time <- fread(file.path(data_dir, "beliefs_over_time.csv"), skip = 3)
  data_list[["beliefs_over_time"]] <- beliefs_over_time
  
  # Load learning_comparison
  learning_comparison <- fread(file.path(data_dir, "learning_comparison.csv"), skip = 3)
  data_list[["learning_comparison"]] <- learning_comparison
  
  # Load planning_actions
  planning_actions <- fread(file.path(data_dir, "planning_actions.csv"), skip = 3)
  data_list[["planning_actions"]] <- planning_actions
  
  # Load actions_per_trial_trace
  actions_per_trial_trace <- fread(file.path(data_dir, "actions_per_trial_trace.csv"), skip = 3)
  data_list[["actions_per_trial_trace"]] <- actions_per_trial_trace
  
  # Load basic_simulation_trace
  basic_simulation_trace <- fread(file.path(data_dir, "basic_simulation_trace.csv"), skip = 3)
  data_list[["basic_simulation_trace"]] <- basic_simulation_trace
  
  # Load learning_comparison_trace
  learning_comparison_trace <- fread(file.path(data_dir, "learning_comparison_trace.csv"), skip = 3)
  data_list[["learning_comparison_trace"]] <- learning_comparison_trace
  
  # Load actions_over_time
  actions_over_time <- fread(file.path(data_dir, "actions_over_time.csv"), skip = 3)
  data_list[["actions_over_time"]] <- actions_over_time
  
  # Load observations_over_time
  observations_over_time <- fread(file.path(data_dir, "observations_over_time.csv"), skip = 3)
  data_list[["observations_over_time"]] <- observations_over_time
  
  # Load learning_trace
  learning_trace <- fread(file.path(data_dir, "learning_trace.csv"), skip = 3)
  data_list[["learning_trace"]] <- learning_trace
  
  # Load observations_trace
  observations_trace <- fread(file.path(data_dir, "observations_trace.csv"), skip = 3)
  data_list[["observations_trace"]] <- observations_trace
  
  # Load planning_trace
  planning_trace <- fread(file.path(data_dir, "planning_trace.csv"), skip = 3)
  data_list[["planning_trace"]] <- planning_trace
  
  return(data_list)
}

# Load all data
activeinference_data <- load_activeinference_data()

# Print summary
cat("Loaded", length(activeinference_data), "datasets\n")
for(name in names(activeinference_data)) {
  cat("- ", name, ": ", nrow(activeinference_data[[name]]), " rows\n")
}
