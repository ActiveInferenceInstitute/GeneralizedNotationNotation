Repository: computationalpsychiatry/activeinference.jl
Files analyzed: 65

Estimated tokens: 64.4k

Directory structure:
└── computationalpsychiatry-activeinference.jl/
    ├── README.md
    ├── LICENSE
    ├── Project.toml
    ├── docs/
    │   ├── make.jl
    │   ├── Project.toml
    │   ├── julia_files/
    │   │   ├── AgentCreation.jl
    │   │   ├── Fitting.jl
    │   │   ├── GenerativeModelCreation.jl
    │   │   ├── GenerativeModelTheory.jl
    │   │   ├── Introduction.jl
    │   │   ├── Simulation.jl
    │   │   ├── SimulationActionModels.jl
    │   │   ├── TMazeSimulationExample.jl
    │   │   ├── WhyActiveInference.jl
    │   │   └── WorkflowsUnfinished.jl
    │   └── src/
    │       ├── AgentCreation.md
    │       ├── Fitting.md
    │       ├── GenerativeModelCreation.md
    │       ├── GenerativeModelTheory.md
    │       ├── index.md
    │       ├── Introduction.md
    │       ├── Simulation.md
    │       ├── SimulationActionModels.md
    │       ├── TMazeSimulationExample.md
    │       ├── WhyActiveInference.md
    │       ├── WorkflowsUnfinished.md
    │       └── assets/
    ├── src/
    │   ├── ActiveInference.jl
    │   ├── ActionModelsExtensions/
    │   │   ├── get_history.jl
    │   │   ├── get_parameters.jl
    │   │   ├── get_states.jl
    │   │   ├── give_inputs.jl
    │   │   ├── reset.jl
    │   │   ├── set_parameters.jl
    │   │   └── set_save_history.jl
    │   ├── Environments/
    │   │   ├── EpistChainEnv.jl
    │   │   └── TMazeEnv.jl
    │   ├── pomdp/
    │   │   ├── inference.jl
    │   │   ├── learning.jl
    │   │   ├── POMDP.jl
    │   │   └── struct.jl
    │   └── utils/
    │       ├── create_matrix_templates.jl
    │       ├── helper_functions.jl
    │       ├── maths.jl
    │       └── utils.jl
    ├── test/
    │   ├── Project.toml
    │   ├── quicktests.jl
    │   ├── runtests.jl
    │   ├── pymdp_cross_val/
    │   │   ├── cross_val_complete_run/
    │   │   │   ├── julia_complete_script/
    │   │   │   │   └── complete_run_julia.jl
    │   │   │   └── python_complete_script/
    │   │   │       └── complete_run_python.py
    │   │   ├── cross_val_results/
    │   │   │   ├── complete_run_data.h5
    │   │   │   ├── results_comparison.csv
    │   │   │   └── results_comparison.jl
    │   │   └── generative_model_creation/
    │   │       ├── rand_generative_model.jl
    │   │       └── gm_data/
    │   │           └── gm_matrices.h5
    │   └── testsuite/
    │       ├── aif_tests.jl
    │       ├── aqua.jl
    │       └── utils_tests.jl
    └── .github/
        ├── agent_output.PNG
        ├── dependabot.yml
        └── workflows/
            ├── CI_full.yml
            ├── CI_small.yml
            ├── CompatHelper.yml
            ├── Documenter.yml
            ├── register.yml
            └── TagBot.yml


================================================
FILE: README.md
================================================
# ActiveInference.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://computationalpsychiatry.github.io/ActiveInference.jl/stable/Introduction/)
[![Build Status](https://github.com/samuelnehrer02/ActiveInference.jl/actions/workflows/CI_full.yml/badge.svg?branch=master)](https://github.com/samuelnehrer02/ActiveInference.jl/actions/workflows/CI_full.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/samuelnehrer02/ActiveInference.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/samuelnehrer02/ActiveInference.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

ActiveInference.jl is a new Julia package for the computational modeling of active inference. We provide the necessary infrastructure for defining active inference models, currently implemented as partially observable Markov decision processes. After defining the generative model, you can simulate actions using agent-based simulations. We also provide the functionality to fit experimental data to active inference models for parameter recovery. 

![Maze Animation](.github/animation_maze.gif)
* Example visualization of an agent navigating a maze, inspired by the one described in [Bruineberg et al., 2018](https://www.sciencedirect.com/science/article/pii/S0022519318303151?via%3Dihub).
Left: A synthetic agent wants to reach the end of the maze environment while avoiding dark-colored locations.
Right: The agent's noisy prior expectations about the state of the environment parameterized by Dirichlet distributions are updated dynamically as it moves through the maze.

## News
#### Version 0.1.1 - December 2024
- Initial release of the package [documentation](https://computationalpsychiatry.github.io/ActiveInference.jl/stable/Introduction/).


## Installation
Install ActiveInference.jl using the Julia package manager:
````@example Introduction
using Pkg
Pkg.add("ActiveInference")

using ActiveInference
````


## Getting Started 

### Understanding Vector Data Types in ActiveInference.jl
The generative model is defined using vectors of arrays, where each element can itself be a multi-dimensional array or matrix. For example: 

* If there is only one modality
````@example Introduction

# Initialize States, Observations, and Controls
states = [25]
observations = [25]
controls = [2] # Two controls (e.g. left and right)
policy_length = 2

# Generate random Generative Model 
A, B = create_matrix_templates(states, observations, controls, policy_length);

# Here, the A_matrix is a one element Vector{Matrix{Float64}} where the element is a 25x25 Matrix
size(A[1]) 

````

* If there are more modalities
````@example Introduction

# Initialize States, Observations, and Controls
states = [25,2] 
observations = [25,2]
controls = [2,1] # Only the first factor is controllable (e.g. left and right)
policy_length = 2

# Generate random Generative Model 
A, B = create_matrix_templates(states, observations, controls, policy_length);

# Each modality is stored as a separate element.
size(A[1]) # Array{Float64, 3} with these dimensions: (25, 25, 2)
size(A[2]) # Array{Float64, 3} with these dimensions: (2, 25, 2)

````
More detailed description of Julia arrays can be found in the official [Julia Documentation](https://docs.julialang.org/en/v1/base/arrays/)

### Basic Usage 

````@example Introduction
# Define some settings as a dictionary.
settings = Dict( "policy_len" => 3)

# Define some parameters as a dictionary.
parameters = Dict("alpha" => 16.0 )

# Initialize the AIF-type agent.
aif = init_aif(A,
               B;
               settings = settings,
               parameters = parameters);
````
![Agent Output](.github/agent_output.PNG)
````@example Introduction
# Give observation to the agent and run state inference.
observation = [3,1]
infer_states!(aif, observation)

# Infer policies 
infer_policies!(aif)

# Sample action
sample_action!(aif)

````



================================================
FILE: LICENSE
================================================
MIT License

Copyright (c) 2023 Jonathan Ehrenreich Laursen, Samuel William Nehrer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



================================================
FILE: Project.toml
================================================
name = "ActiveInference"
uuid = "688b0e7a-0122-4325-8669-5ff08899a59e"
authors = ["Jonathan Ehrenreich Laursen", "Samuel William Nehrer"]
version = "0.1.2"

[deps]
ActionModels = "320cf53b-cc3b-4b34-9a10-0ecb113566a3"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
IterTools = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"

[compat]
ActionModels = "0.6"
Distributions = "0.25"
IterTools = "1.10"
LinearAlgebra = "1"
LogExpFunctions = "0.3"
Random = "1"
ReverseDiff = "1.15"
julia = "1.10"



================================================
FILE: docs/make.jl
================================================
using ActiveInference
using Documenter
using Literate

# Set project directory
if haskey(ENV, "GITHUB_WORKSPACE")
    project_dir = ENV["GITHUB_WORKSPACE"]
    input_folder = joinpath(project_dir, "docs", "julia_files")
else
    project_dir = pwd()
    input_folder = raw"..\julia_files"
end

cd(joinpath(project_dir, "docs", "src"))

DocMeta.setdocmeta!(ActiveInference, :DocTestSetup, :(using ActiveInference); recursive=true)

# Automating the creating of the markdown files
julia_files = filter(file -> endswith(file, ".jl"), readdir(input_folder))

for file in julia_files
    input_path = joinpath(input_folder, file)
    Literate.markdown(input_path, outputdir="", execute=true, documenter=true, codefence =  "```julia" => "```")
end

# Creating the documentation
makedocs(;
    modules=[ActiveInference, ActiveInference.Environments],
    authors="Jonathan Ehrenreich Laursen, Samuel William Nehrer",
    repo="https://github.com/ilabcode/ActiveInference.jl/blob/{commit}{path}#{line}",
    sitename="ActiveInference.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ilabcode.github.io/ActiveInference.jl",
        edit_link="master",
        assets=[],
    ),
    pages=[

        "General Introduction" => [

            "Introduction" => "Introduction.md",
            "Creation of the Generative Model" => "GenerativeModelCreation.md",
            "Creating the Agent" => "AgentCreation.md",
            "Simulation" => "Simulation.md",
            "Model Fitting" => "Fitting.md",
            "Simulation with ActionModels.jl" => "SimulationActionModels.md",

        ],

        "Usage Examples" => [

            "T-Maze Simulation" => "TMazeSimulationExample.md",
            # "T-Maze Model Fitting" => [],

        ],

        "Theory" => [

            # "Active Inference Theory" => [
            #     "Perception" => [],
            #     "Action" => [],
            #     "Learning" => [],
            # ],

            "POMDP Theory" => "GenerativeModelTheory.md",



        ],

        # "Why Active Inference?" => "WhyActiveInference.md",

        "Index" => "index.md",
    ],
    doctest=true,
)

deploydocs(;
    repo="github.com/ilabcode/ActiveInference.jl",
    devbranch="master",
)



================================================
FILE: docs/Project.toml
================================================
[deps]
ActionModels = "320cf53b-cc3b-4b34-9a10-0ecb113566a3"
ActiveInference = "688b0e7a-0122-4325-8669-5ff08899a59e"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Documenter = "e30172f5-a6a5-5a46-863b-614d45cd2de4"
DocumenterTools = "35a29f4d-8980-5a13-9543-d66fff28ecb8"
Literate = "98b081ad-f1c9-55d3-8b20-4c87d4299306"
MarkdownTables = "1862ce21-31c7-451e-824c-f20fa3f90fa2"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"



================================================
FILE: docs/julia_files/AgentCreation.jl
================================================
# # Creating the Agent

# Having created the generative model parameters in the precious section, we're not ready to intialise an active inference agent.
# Firstly, we'll have to specify some settings and hyperparameters that go into the agent struct. We'll begin with the setting:

# ### Settings
# The settings are a dictionary that contains the following keys:

# ```julia
# settings = Dict(
#     "policy_len" => 1, 
#     "use_utility" => true, 
#     "use_states_info_gain" => true, 
#     "use_param_info_gain" => false,
#     "action_selection" => "stochastic", 
#     "modalities_to_learn" => "all",
#     "factors_to_learn" => "all",
#     "FPI_num_iter" => 10,
#     "FPI_dF_tol" => 0.001
# )
# ```

# The above shown values are the default and will work in most cases. If you're unsure about what to specify in the settings, you can just use the default values by not specifying them in the settings Dict for the agent.
# Here, we'll briefly describe the keys in the settings dictionary:

# - **`policy_len`** - Is the policy length, and as described previously is the number of actions the agent should plan in the future. This is provided as an integer.
# - **`use_utility`** - Is a boolean that specifies whether the agent should use **C** in the expected free energy calculation, that guides the action selection in active inference. If set to `false`, the agent will not use the parameters specified in **C**.
# - **`use_states_info_gain`** - Is a boolean that specifies whether the agent should use the information gain over states in the expected free energy calculation. If set to `false`, the agent will not use the information gain over states.
# - **`use_param_info_gain`** - Is a boolean that specifies whether the agent should use the information gain over parameters in the expected free energy calculation. If set to `false`, the agent will not use the information gain over parameters. Only relevant when learning is included.
# - **`action_selection`** - Is a string that specifies the action selection method. The options are `"stochastic"` and `"deterministic"`. If set to `"stochastic"`, the agent will sample from the posterior over policies, and if set to `"deterministic"`, the agent will choose the most probable action.
# - **`modalities_to_learn`** - Is a vector of integers that specifies which modalities the agent should learn. If set to string `"all"`, the agent will learn all modalities. If set to `[1,2]`, the agent will only learn the first and second modality. Only relevant when learning of A is included.
# - **`factors_to_learn`** - Is a vector of integers that specifies which factors the agent should learn. If set to string `"all"`, the agent will learn all factors. If set to `[1,2]`, the agent will only learn the first and second factor. Only relevant when learning of B and D is included.
# - **`FPI_num_iter`** - Is an integer that specifies the number of fixed point iterations (FPI) to perform in the free energy minimization. It can be described as a stop function of the FPI algorithm.
# - **`FPI_dF_tol`** - Is a float that specifies the tolerance of the free energy change in the FPI algorithm over each iteration. If the change in free energy is below this value, the FPI algorithm will also stop.

# For more information on the specifics of the impact of these settings, look under the `Active Inference Theory` section in the documentation.

# ### Parameters
# The parameters are a dictionary that contains the following keys:

# ```julia
# parameters = Dict(
# "gamma" => 16.0,
# "alpha" => 16.0,
# "lr_pA" => 1.0,
# "fr_pA" => 1.0,
# "lr_pB" => 1.0,
# "fr_pB" => 1.0,
# "lr_pD" => 1.0,
# "fr_pD" => 1.0
# )
# ```

# The above shown values are the default. If you're unsure about what to specify in the parameters, you can just use the default values by not specifying them in the parameter Dict for the agent.
# Here, we'll briefly describe the keys in the parameters dictionary containing the hyperparameters:
# - **`alpha`** - Is the inverse temperature of the action selection process, and usually takes a value between 1 and 32. This is only relevant when action_selection is set to `"stochastic"`.
# - **`gamma`** - Is the inverse temperature precision of the expected free energy, and usually takes a value between 1 and 32. If the value is high, the agent will be more certain in its beliefs regarding the posterior probability over policies.
# - **`lr_pA`** - Is the learning rate of **A**, and usually takes a value between 0 and 1. Only relevant when learning is included, and this goes for all learning and forgetting rates. 
# - **`fr_pA`** - Is the forgetting rate of **A**, and usually takes a value between 0 and 1. If forgetting rate is 1 it means no forgetting.
# - **`lr_pB`** - Is the learning rate of **B**, and usually takes a value between 0 and 1.
# - **`fr_pB`** - Is the forgetting rate of **B**, and usually takes a value between 0 and 1. If forgetting rate is 1 it means no forgetting.
# - **`lr_pD`** - Is the learning rate of **D**, and usually takes a value between 0 and 1.
# - **`fr_pD`** - Is the forgetting rate of **D**, and usually takes a value between 0 and 1. If forgetting rate is 1 it means no forgetting.

# 
# Having now specified the setting and parameters, we can now initialise the active inference agent. This is done by calling the `init_aif` function, which takes the following arguments:

# ## Initilising the Agent

include("../julia_files/GenerativeModelCreation.jl") #hide
A, B, C, D, E = create_matrix_templates([4,2], [4,3,2], [4,1], 1, "uniform"); #hide
parameters = Dict( #hide
"gamma" => 16.0, #hide
"alpha" => 16.0, #hide
"lr_pA" => 1.0, #hide
"fr_pA" => 1.0, #hide
"lr_pB" => 1.0, #hide
"fr_pB" => 1.0, #hide
"lr_pD" => 1.0, #hide
"fr_pD" => 1.0 #hide
); #hide
settings = Dict( #hide
    "policy_len" => 1, #hide 
    "use_utility" => true, #hide
    "use_states_info_gain" => true,  #hide
    "use_param_info_gain" => false, #hide
    "action_selection" => "stochastic", #hide
    "modalities_to_learn" => "all", #hide
    "factors_to_learn" => "all", #hide
    "FPI_num_iter" => 10, #hide
    "FPI_dF_tol" => 0.001 #hide
); #hide
aif_agent = init_aif(
    A, B, C = C, D = D, E = E, settings = settings, parameters = parameters, verbose = false
);


# You can access the settings and parameters of the agent by calling the agent struct on the agent:
aif_agent.parameters
#-
aif_agent.settings

# Having now initialised the agent, we are ready to implement it either in a simulation with a perception-action loop, or for use in model fitting with observed data.

# ## Initialising the Agent with Learning
# If you want to include learning in the agent, you can do so by specifying the prior parameters `init_aif` function. Here is an example of how to initialise the agent with learning:

# ```julia
# aif_agent = init_aif(
#     A, B, C = C, D = D, E = E, pA = pA, pB = pB, pD = pD, settings = settings, parameters = parameters, verbose = false
# );
# ```

# Here, only the prior of the parameters that are to be learned should be specified.



================================================
FILE: docs/julia_files/Fitting.jl
================================================
# # Model Fitting 

# In many cases, we want to be able to draw conclusions about specific observed phenomena, such as behavioural differences between distinct populations. A conventional approach in this context is model fitting, which involves estimating the parameter values of a model (e.g., prior beliefs) that are most likely given the observed behavior of a participant. This approach is often used in fields such as computational psychiatry or mathematical psychology  to develop more precise models and theories of mental processes, to find mechanistic differences between clinical populations, or to investigate the relationship between computational constructs such as Bayesian beliefs and neuronal dynamics.
# ## Quick Start
# #### Model Fitting with ActionModels.jl

# Model fitting in '**ActiveInference**' is mediated through '**ActionModels**', which is our sister package for implementing and fitting various behavioural models to data. The core of '**ActionModels**' is the action model function, which takes a single observation, runs the inference scheme (updating the agent's beliefs), and calculates the probability distribution over actions from which the agent samples its actions.
# *(Check out the [ActionModels documentation](https://ilabcode.github.io/ActionModels.jl/dev/markdowns/Introduction/) for more details)*

using Pkg#hide
using ActiveInference#hide
n_states=[4]#hide
n_observations=[4]#hide
n_controls=[2]#hide
policy_length=1#hide
A,B=create_matrix_templates(n_states, n_observations, n_controls, policy_length);#hide
aif = init_aif(A, B, verbose=false);#hide
using Distributions#hide
priors = Dict("alpha" => Gamma(1, 1));#hide
using DataFrames#hide 
using ActionModels#hide
# To demonstrate this, let's define a very simple generative model with a single state factor and two possible actions, and then initialize our active inference object:
# ```julia
# # Define the number of states, observations, and controls
# n_states = [4]
# n_observations = [4]
# n_controls = [2]

# # Define the policy length
# policy_length = 1

# # Use the create_matrix_templates function to create uniform A and B matrices.
# A, B = create_matrix_templates(n_states, n_observations, n_controls, policy_length)

# # Initialize an active inference object with the created matrices
# aif = init_aif(A, B)
# ```

# We can now use the `action_pomdp!` function (which serves as our active inference "action model") to calculate the probability distribution over actions for a single observation:
# ```julia
# # Define observation
# observation = [1]

# # Calculate action probabilities
# action_distribution = action_pomdp!(aif, observation)
# ```

# #### Agent in ActionModels.jl
# Another key component of '**ActionModels**' is an `Agent`, which wraps the action model and active inference object in a more abstract structure. The `Agent` is initialized using a `substruct` to include our active inference object, and the action model is our `action_pomdp!` function.

# Let's first install '**ActionModels**' from the official Julia registry and import it:
# ```julia
# Pkg.add("ActionModels")
# using ActionModels
# ```

# We can now create an `Agent` with the `action_pomdp!` function and the active inference object:

# ```julia
# # Initialize agent with active inference object as substruct
# agent = init_agent(
#     action_pomdp!,  # The active inference action model
#     substruct = aif # The active inference object
# )
# ```
# We use an initialized `Agent` primarily for fitting; however, it can also be used with a set of convenience functions to run simulations, which are described in [Simulation with ActionModels](./SimulationActionModels.md).

# #### Fitting a Single Subject Model
# We have our `Agent` object defined as above. Next, we need to specify priors for the parameters we want to estimate. 

# For example, let's estimate the action precision parameter `α` and use a Gamma distribution as its prior.

# ```julia
# # Import the Distributions package
# using Distributions

# # Define the prior distribution for the alpha parameters inside a dictionary
# priors = Dict("alpha" => Gamma(1, 1))
# ```
# We can now use the `create_model` function to instantiate a probabilistic model object with data. This function takes the `Agent` object, the priors, and a set of observations and actions as arguments.
#
# First, let's define some observations and actions as vectors:
# ```julia
# # Define observations and actions
# observations = [1, 1, 2, 3, 1, 4, 2, 1]
# actions = [2, 1, 2, 2, 2, 1, 2, 2]
# ```

# Now we can instantiate the probabilistic model object:
# ```julia
# # Create the model object
# single_subject_model = create_model(agent, priors, observations, actions)
# ```
# The `single_subject_model` can be used as a standard Turing object. Performing inference on this model is as simple as: 
# ```julia
# results = fit_model(single_subject_model)
# ```
# #### Fitting a Model with Multiple Subjects
# Often, we have data from multiple subjects that we would like to fit simultaneously. The good news is that this can be done by instantiating our probabilisitc model on an entire dataset containing data from multiple subjects.
#
# Let's define some dataset with observations and actions for three subjects:

# ```julia
# # Import the DataFrames package
# using DataFrames
#
# # Create a DataFrame 
# data = DataFrame(
#    subjectID = [1, 1, 1, 2, 2, 2, 3, 3, 3], # Subject IDs
#    observations = [1, 1, 2, 3, 1, 4, 2, 1, 3], # Observations
#    actions = [2, 1, 2, 2, 2, 1, 2, 2, 1] # Actions
# )
# ```
data = DataFrame(subjectID = [1, 1, 1, 2, 2, 2, 3, 3, 3], observations = [1, 1, 2, 3, 1, 4, 2, 1, 3], actions = [2, 1, 2, 2, 2, 1, 2, 2, 1] )#hide
#
# To instantiate the probabilistic model on our dataset, we pass the `data` DataFrame to the `create_model` function along with the names of the columns that contain the subject identifiers, observations, and actions:
# ```julia
# # Create the model object
# multi_subject_model = create_model(
#     agent, 
#     priors, 
#     data; # Dataframe
#     grouping_cols = [:subjectID], # Column with subject IDs
#     input_cols = ["observations"], # Column with observations
#     action_cols = ["actions"] # Column with actions
# )
# ```
agent = init_agent(action_pomdp!, substruct = aif);#hide
multi_subject_model = create_model(agent, priors, data; grouping_cols = [:subjectID], input_cols = ["observations"], action_cols = ["actions"]);#hide

# To fit the model, we use the `fit_model` function as before:
# ```julia
# results = fit_model(multi_subject_model)
# ```
results=fit_model(multi_subject_model, show_progress=false);#hide
# #### Customizing the Fitting Procedure
# The `fit_model` function has several optional arguments that allow us to customize the fitting procedure. For example, you can specify the number of iterations, the number of chains, the sampling algorithm, or to parallelize over chains:

# ```julia
# results = fit_model(
#     multi_subject_model, # The model object
#     parallelization = MCMCDistributed(), # Run chains in parallel
#     sampler = NUTS(;adtype=AutoReverseDiff(compile=true)), # Specify the type of sampler
#     n_itererations = 1000, # Number of iterations, 
#     n_chains = 4, # Number of chains
# )
# ```
# '**Turing**' allows us to run distributed `MCMCDistributed()` or threaded `MCMCThreads()` parallel sampling. The default is to run chains serially `MCMCSerial()`. For information on the available samplers see the [Turing documentation](https://turing.ml/dev/docs/using-turing/samplers/). 
# 
# #### Results
# 
# The output of the `fit_model` function is an object that contains the standard '**Turing**' chains which we can use to extract the summary statistics of the posterior distribution.
#
# Let's extract the chains from the results object:
chains = results.chains
#
# Note that the parameter names in the chains are somewhat cryptic. We can use the `rename_chains` function to rename them to something more understandable:
renamed_chains = rename_chains(chains, multi_subject_model)
#
# That looks better! We can now use the '**StatsPlots**' package to plot the chain traces and density plots of the posterior distributions for all subjects:
# ```julia
# using StatsPlots # Load the StatsPlots package
#
# plot(renamed_chains)
# ```
#
# > [!WARNING] Image missing: assets/chain_traces.png
> *(image2)*
#
# We can also visualize the posterior distributions against the priors. This can be done by first taking samples from the prior:
# ```julia
# # Sample from the prior
# prior_chains = sample(multi_subject_model, Prior(), 1000)
# # Rename parameters in the prior chains
# renamed_prior_chains = rename_chains(prior_chains, multi_subject_model)
# ```
# To plot the posterior distributions against the priors, we use the `plot_parameters` function:
# ```julia
# plot_parameters(renamed_prior_chains, renamed_chains)
# ```

# > [!WARNING] Image missing: assets/posteriors.png
> *(image3)*



================================================
FILE: docs/julia_files/GenerativeModelCreation.jl
================================================
# # Creating the POMDP Generative Model

# In this section we will go through the process of creating a generative model and how it should be structured. In this part, we will show the code necessary for correct typing of the generative model.
# For a theoretical explanation of POMDPs look under the "Theory" section further down in the documentation.

# ## Typing of the POMDP parameters

# In ActiveInference.jl, it is important that the parameters describing the generative model is typed correctly.
# The correct typing of the generative model parameters, which often take the shapes of matrices, tensors and vectors.
# The collections of generative model parameters are colloquially referred to as **A**, **B**, **C**, **D**, and **E**. We will denote these parameters by their letter in bold. For a quick refresher this is the vernacular used to describe these parameter collections:

# - **A** - Observation Likelihood Model
# - **B** - Transition Likelihood Model
# - **C** - Prior over Observations
# - **D** - Prior over States
# - **E** - Prior over Policies

# These should be typed the following way in ActiveInference.jl:

# ```julia
# A = Vector{Array{Float64, 3}}(undef, n_modalities)
# B = Vector{Array{Float64, 3}}(undef, n_factors)
# C = Vector{Vector{Float64, 3}}(undef, n_modalities)
# D = Vector{Vector{Float64, 3}}(undef, n_factors)
# D = Vector{Float64, 3}(undef, n_policies)
# ```

# Each of the parameter collections are vectors, where each index in the vector contains the parameters associated with a specific modality or factor.
# However, creating these from scratch is not necessary, as we have created a helper function that can create a template for these parameters.

# ## Helper Function for GM Templates
# Luckily, there is a helper function that helps create templates for the generative model parameters. This function is called `create_matrix_templates`.

# ```julia
# A, B, C, D, E = create_matrix_templates(n_states::Vector{Int64}, n_observations::Vector{Int64}, n_controls::Vector{Int64}, policy_length::Int64, template_type::String)
# ```

# This function takes the five arguments `n_states`, `n_observations`, `n_controls`, `policy_length`, and `template_type`, which have all the necessary information to create the 
# right structure of the generative model parameters. We will go through these arguments one by one:

# \

# - **n_states** - This is the number of states in the environment. The environment can have different kinds of states, which are often referred to as factors. Could be a location factor and a reward condition factor. It takes a vector of integers, where each integer represents a factor, and the value of the integer is the number of states in that factor. E.g. if we had an environment with two factors, one location factor with 4 states and one reward condition factor with 2 states, the argument would look like this: `[4,2]`
# \

# - **n_observations** - This is the number of observations the agent can make in the environment. The observations are often referred to as modalities. Could be a location modality, a reward modality and a cue modality. Similarly to the first argument, it takes a vector of integers, where each integer represents a modality, and the value of the integer is the number of observations in that modality. E.g. if we had an environment with three modalities, one location modality with 4 observations, one reward modality with 3 observations and one cue modality with 2 observations, the argument would look like this: `[4,3,2]`
# \

# - **n_controls** - This is the number of controls the agent have in the environment. The controls are the actions the agent can take in the different factors. Could be moving left or right, or choosing between two different rewards. It has one control integer for each factor, where the integer represents the number of actions in that factor. If the agent cannot control a factor, the integer should be 1. E.g. if we had an environment with two factors, one location factor with 4 actions and one reward condition factor with 1 action, the argument would look like this: `[4,1]`
# \
  
# - **policy_length** - This is the length of the policies of the agent, and is taken as an integer. The policy is a sequence of actions the agent can take in the environment. The length of the policy describes how many actions into the future the agent is planning. For example, if the agent is planning two steps into the future, the policy length would be 2, and each policy would consist of 2 actions. In that case the argument would look like this: `2`
# \

# - **template_type** - This is a string that describes the type of template you want to create, or in other words, the initial filling of the generative model structure. There are three options; `"uniform"`, which is default, `"random"`, and `"zeros"`.


# If we were to use the arguments from the examples above, the function call would look like this:
using ActiveInference #hide
n_states = [4,2]
n_observations = [4,3,2]
n_controls = [4,1]
policy_length = 2
template_type = "zeros"

A, B, C, D, E = create_matrix_templates(n_states, n_observations, n_controls, policy_length, template_type);

# When these parameter collections have been made, each factor/modality can be accessed by indexing the collection with the factor/modality index like:

# ```julia
# A[1] # Accesses the first modality in the observation likelihood model
# B[2] # Accesses the second factor in the transition likelihood model
# C[3] # Accesses the third modality in the prior over observations
# D[1] # Accesses the first factor in the prior over states
# ```

# The E-parameters are not a divided into modalities or factors, as they are the prior over policies.

# ## Populating the Parameters
# Now that the generative model parameter templates ahave been created, they can now be filled with the desired values, ie. populating the parameters.
# Let's take the example of filling **A** with some valus. To start, let's print out the first modality of the A so we get a sense of the dimensions:
A[1]
# For a quick recap on the POMDP generative model parameteres look up the [`POMDP Theory`](@ref "The Generative Model Conceptually") section further down in the documentation.

# For now, we'll suffice to say that the first modality of **A** is a 3D tensor, where the first dimension are observations in the first modality, the second dimension the first factor, and the third dimension is the second factor.
# Remember **A** maps the agents beliefs on how states generate observations. In this case, we have two 4x4 matrices, one matrix for each state int the second factor. This could be how location observations (1st dimenstion) map onto location states (2nd dimension) and reward condition (3rd dimension).
# For the sake of simplicity, let's assume that the agent can infer location states with certainty based on location observations. In this case we could populate the first modality of **A** like this:

# ```julia
# # For reward condition right
# A[1][:,:,1] = [ 1.0  0.0  0.0  0.0
#                 0.0  1.0  0.0  0.0
#                 0.0  0.0  1.0  0.0
#                 0.0  0.0  0.0  1.0 ]

# # For reward condition left
# A[1][:,:,2] = [ 1.0  0.0  0.0  0.0
#                 0.0  1.0  0.0  0.0
#                 0.0  0.0  1.0  0.0
#                 0.0  0.0  0.0  1.0 ]
# ```

# In this case the agent would infer the location state with certainty based on the location observations. One could also make the **A** more noisy in this modality, which could look like:

# ```julia
# # For reward condition right
# A[1][:,:,1] = [ 0.7  0.1  0.1  0.1
#                 0.1  0.7  0.1  0.1
#                 0.1  0.1  0.7  0.1
#                 0.1  0.1  0.1  0.7 ]

# # For reward condition left
# A[1][:,:,2] = [ 0.7  0.1  0.1  0.1
#                 0.1  0.7  0.1  0.1
#                 0.1  0.1  0.7  0.1
#                 0.1  0.1  0.1  0.7 ]
# ```

# Importantly the columns should always add up to 1, as we are here dealing with categorical probability distributions.
# For the other parameters, the process is similar, but the dimensions of the matrices will differ. For **B** the dimensions are states to states, and for **C** and **D** the dimensions are states to observations and states to factors respectively.
# Look up the `T-Maze Simulation` (insert reference here) example for a full example of how to populate the generative model parameters.

# ## Creating Dirichlet Priors over Parameters
# When learning is included, we create Dirichlet priors over the parameters **A**, **B**, and **D**. We usually do this by taking the created **A**, **B**, and **D** parameters and multiplying them with a scalar, which is the concentration parameter of the Dirichlet distribution.
# For more information on the specifics of learning and Dirichlet priors, look under the `Active Inference Theory` section in the documentation. Note here, that when we implement learning of a parameter, the parameter is going to be defined by its prior and no longer the initial 
# parameter that we specified. This is because the agent will update the parameter based on the prior and the data it receives. An example of how we would create a Dirichlet prior over **A** could look:

# ```julia
# pA = deepcopy(A)
# scale_concentration_parameter = 2.0
# pA .*= scale_concentration_parameter
# ```

# This is not relevant if learning is not included. If learning is not included, the parameters are fixed and the agent will not update them. The value of the scaling parameter determines how much each data observation impacts the update of the parameter.
# If the scaling is high, e.g. 50, then adding one data point will have a small impact on the parameter. If the scaling is low, e.g. 0.1, then adding one data point will have a large impact on the parameter. The update function updates the parameters by normalising the concentration parameters of the Dirichlet distribution.




================================================
FILE: docs/julia_files/GenerativeModelTheory.jl
================================================
# # The Generative Model Conceptually

# The generative model is the parameters that constitute the agent's beliefs on how the hidden states of the environment generates observations based on states, and how hidden underlying states changes over time.
# In the generative model is also the beliefs of how the agent through actions can influence the states of the environment. Together this holds the buidling blocks that allows for the perception-action loop.

# There are five main buidling blocks of the generative model which are; **A**, **B**, **C**, **D**, and **E**.
# Each of these contain parameters that describe the agent's beliefs about the environment.
# We will now go through these conecptually one at a time.

# ## A
# **A** is the observation likelihood model, and describes the agent's beliefs about how the hidden states of the environment generates observations.
# Practically in this package, and other POMDP implemantations as well, this is described through a series of categorical distributions, meaning that for each observation, there is a categorical probability distribution over how likely each hidden state is to generate that observation.
# Let us for example imagine a simple case, where the agent is in a four location state environment, could be a 2x2 gridworld. In this case, there would be one obseration linked to each hidden state, and **A** then maps the agent's belief of how likely each hidden location state is to generate each observation.
# The agent can then use this belief to infer what state it is in based on the observation it receives. Let's look at an example **A**, which in this case would be a 4x4 matrix:


# ```math
# A =
# \overset{\text{\normalsize States}\vphantom{\begin{array}{c} 0 \\ 0 \end{array}}}{
#     \begin{array}{cccc}
#         1 & 0 & 0 & 0 \\
#         0 & 1 & 0 & 0 \\
#         0 & 0 & 1 & 0 \\
#         0 & 0 & 0 & 1
#     \end{array}
# }
# \quad
# \text{\normalsize Observations}
# ```

# In this case, the agent is quite certain about which states produces which observations. This matrix could be made more uncertain to the point of complete uniformity and it could be made certain in the sense of each column being a one-hot vector.
# In the case of a certain **A**, the generative model stops being a "partially observable" Markov decision process, and becomes a fully observable one, making it a Markov decision process (MDP). For a more technical and mathematical definition of the observation likelihood model.

# ## B
# **B** is the transition likelihood model that encodes the agent's beliefs about how the hidden states of the environment changes over time.
# This is also made up of categorical distributions, though instead of observations to states, it maps states to states. 
# If we take the same case again, a 2x2 gridworld, we would have a 4x4 matrix that describes how the agent believes the states evolve over time.
# An extra addition to **B**, is that it can depend on actions, meaning that it can believe that the hidden states of the environment change differently depending on the action taken by the agent.
# Due to this fact, we would the have a matrix for each action, making **B** a 3 dimensional tensor, with 2 dimensions for the "from" state and the "to" state, and then an action dimension.
# Let's look at an example of a slice of **B** for the action "down" in the grid world, which in this case would be a 4x4 matrix:

# ```math
# B("down") =
# \overset{\text{\normalsize Previous State}\vphantom{\begin{array}{c} 0 \\ 0 \end{array}}}{
#     \begin{array}{cccc}
#         0 & 0 & 0 & 0 \\
#         1 & 1 & 0 & 0 \\
#         0 & 0 & 0 & 0 \\
#         0 & 0 & 1 & 1
#     \end{array}
# }
# \quad
# \text{\normalsize Current State}
# ```

# We could make 3 more similar matrices for the actions "up", "left", and "right", and then we would have the full **B** tensor for the gridworld. But here, the main point is that
# **B** decsribes the agent's belief of how hidden states change over time, and this can be dependent on actions, but might also be independent of actions, and thus the agent believes that the changes are out of its control.

# ## C
# **C** is the prior over observations, also called preferences over observations. This is an integral part of the utility of certain observations, i.e. it encodes how much the agent prefers or dislikes certain observations.
# **C** is a simple vector over observations, where each entry is a value that describes the utility or preference of that specific observation.
# If we continue with the simple 2x2 gridworld example, we would have 4 observations, one for each location state (same amount of observations as in **A**).
# Let's say that we would like for the agent to dislike observing the top left location (indexed as 1), and prefer the bottom right location (indexed as 4). We would then create **C** in the following way:

# ```math
# C =
# \begin{array}{cccc}
#     -2 & 0 & 0 & 2 \\
# \end{array}
# ```

# The magnitude of the values in **C** is arbitrary, and denotes a ratio and amount of dislike/preference. Here, we have chosen the value of -2 and 2 
# to encode that the agent dislikes the top left location just as much as it likes the bottom right location. The zeros in between just means that the agent has not preference or dislike for these locatin observations.
# Note that since **C** is not a categorical distribution, it does not need to sum to 1, and the values can be any real number.

# ## D
# **D** is the prior over states, and is the agent's beliefs about the initial state of the environment. This is also a simple vector that is a categorical distribution.
# Note that if **A** is certain, then **D** does not matter a lot for the inference process, as the agent can infer the state from the observation. However, if **A** is uncertain,
# then **D** becomes very important, as it serves as the agent's anchor point of where it is initially in the environment. In the case of out
# 2x2 gridworld, we would have a vector with 4 entries, one for each location state. If we assume that the agent correctly infers it's initial location as upper left corner, **D** would look like:

# ```math
# D =
# \begin{array}{cccc}
#     1 & 0 & 0 & 0 \\
# \end{array}
# ```

# ## E
# **E** is the prior over policies, and can be described as the agent's habits. Policies in Active Inference vernacular are sets of actions, with an action for each step in the future, specified by a policy length.
# It is a categorical distribution over policies, with a probability for each policy. This will have an effect on the agent posterior over policies,
# which is the probability of taking a certain action at a time step. This will often be set to a uniform distribution, if we are not interested in giving the agent habits.
# Let us assume that we will give our agent a uniform **E** for a policy length of 2, this mean that we will have a uniform categorical distribution over 16 possible policies ``(4 (actions) ^ {2 (policy length)})``:

# ```math
# E =
# \begin{array}{cccc}
# 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 \\
# \end{array}
# ```






================================================
FILE: docs/julia_files/Introduction.jl
================================================
# # Introduction to the ActiveInference.jl package

# This package is a Julia implementation of the Active Inference framework, with a specific focus on cognitive modelling.
# In its current implementation, the package is designed to handle scenarios that can be modelled as discrete state spaces, with 'partially observable Markov decision process' (POMDP).
# In this documentation we will go through the basic concepts of how to use the package for different purposes; simulation and model inversion with Active Inference, also known as parameter estimation.

# ## Installing Package
# Installing the package is done by adding the package from the julia official package registry in the following way:

# ```julia
# using Pkg
# Pkg.add("ActiveInference")
# ```

# Now, having added the package, we simply import the package to start using it:
using ActiveInference

# In the next section we will go over the basic concepts of how to start using the package. We do this by providing instructions on how to create and design a generative model, that can be used for both simulation and parameter estimation.



================================================
FILE: docs/julia_files/Simulation.jl
================================================
# # Simulation with ActiveInference.jl
# When simulating with active inference we need a perception-action loop. This loop will perform the following steps:
# 1. The agent will infer the states of the environment based on its generative model and an observation. The inference here is optimized through the minimization of the variational free energy (see `Active Inference Theory Perception`).
# 2. The agent will infer the best action based on the minimization of the expected free energy (see `Active Inference Theory Action`).
# 3. The agent will perform the action in the environment and receive an observation for use in the next iteration.

# *Note: for learning included, look at the section below.*

# #### The Perception-Action loop:
# ```julia
# T = n_iterations

# for t = 1:T

#     infer_states!(aif_agent, observation)

#     infer_policies!(aif_agent)

#     chosen_action = sample_action!(aif_agent)

#     observation = environment!(env, chosen_action)

# end
# ```

# #### The Perception-Action-Learning loop:
# When learning is included, the loop is very similar except for the addition of the update functions, which should be implemented at different points in the loop.
# Below we will show how to include learning of the parameters. It is important that only the parameters which have been provided to the agent as a prior are being updated.
# ```julia
# T = n_iterations

# for t = 1:T

#    infer_states!(aif_agent, observation)

#    update_parameters!(aif_agent)

#    infer_policies!(aif_agent)

#    chosen_action = sample_action!(aif_agent)

#    observation = environment!(env, chosen_action)

# end
# ```

# The only addition here is the `update_parameters!(aif_agent)` function, which updates the parameters of the agent, based on which priors it has been given. 



================================================
FILE: docs/julia_files/SimulationActionModels.jl
================================================
# # Simulation with ActionModels.jl



================================================
FILE: docs/julia_files/TMazeSimulationExample.jl
================================================
# # Simulation Example T-Maze

# We will start from the importing of the necessary modules.

using ActiveInference
using ActiveInference.Environments

# We will create a T-Maze environment with a probability of 0.9 for reward in the the reward condition arm.
# This is a premade environment in the ActiveInference.jl package.

# ```julia
# env = TMazeEnv(0.9)
# initialize_gp(env)
# ```

# ### Creating the Generative Model
# #### The Helper Function

# When creating the generative model we can make use of the helper function, making it convenient to create the correct structure for the generative model parameters.

# To use the helper function we need to know the following:

# - Number of states in each factor of the environment
# - Number of observations in each modality
# - Number of controls or actions in each factor
# - Policy length of the agent
# - Initial fill for the parameters

# Let's start with the factors of the environment. Let's take a look at the T-Maze environment:

# > [!WARNING] Image missing: assets/TMazeIllustrationSmaller.png
> *(image1)*

# We here have two factors with the following number of states:

# |       | Location Factor   |       | Reward Condition Factor   |
# |:------|:------------------|:------|:------------------------- |
# | 1.    | Centre            | 1.    | Reward Condition Left     |
# | 2.    | Left Arm          | 2.    | Reward Condition Right    |
# | 3.    | Right Arm         |       |                           |
# | 4.    | Cue               |       |                           |

# We will define this as a vector the following way:

# ```julia
# n_states = [4, 2]
# ```

# We will now define the modalities:

# |       | Location Modality |       | Reward Modality           |       | Cue Modality    |
# |:------|:------------------|:------|:------------------------- |:------|:--------------- |
# | 1.    | Centre            | 1.    | No Reward                 | 1.    | Cue Left        |
# | 2.    | Left Arm          | 2.    | Reward                    | 2.    | Cue Right       |
# | 3.    | Right Arm         | 3.    | Loss                      |       |                 |
# | 4.    | Cue               |       |                           |       |                 |

# Here we have 3 modalities, with 4, 3, and 2 observations in each. We will define this as a vector the following way:

# ```julia
# n_observations = [4, 3, 2]
# ```

# Now, let's take a look at the actions, or controls:

# |       | Controls Location Factor       |    | Controls Reward Condition Factor       |
# |:------|:-------------------------------|:-- |:---------------------------------------|
# | 1.    | Go to Centre                   | 1. | No Control                                      |
# | 2.    | Go to Left Arm                 |    |                                       |
# | 3.    | Go to Right Arm                |    |                                       |
# | 4.    | Go to Cue                      |    |                                       |

# As we see here, the agent cannot control the reward condition factor, and it therefore believes that there is only one way states can transition in this factor, which is independent of the agent's actions.
# We will define this as a vector the following way:

# ```julia
# n_controls = [4, 1]
# ```

# Now we can define the policy length of the agent. In this case we will just set it to 2, meaning that the agent plans two timesteps ahead in the future.
# We will just specify this as an integer:

# ```julia
# policy_length = 2
# ```

# The last thing we need to define is the initial fill for the parameters. We will just set this to zeros for now.

# ```julia
# template_type = "zeros"
# ```

# Having defined all the arguments that go into the helper function, we can now create the templates for the generative model parameters.

# ```julia
# A, B, C, D, E = create_matrix_templates(n_states, n_observations, n_controls, policy_length, template_type);
# ```

# #### Populating the Generative Model
# ##### Populating **A**

# Let's take a look at the shape of the first modality in the A parameters:
A, B, C, D, E = create_matrix_templates([4, 2], [4, 3, 2], [4, 1], 2, "zeros");#hide
A[1]

# For this first modality we provide the agent with certain knowledge on how location observations map onto location states.
# We do this the following way:

# ```julia
# # For reward condition right
# A[1][:,:,1] = [ 1.0  0.0  0.0  0.0
#                 0.0  1.0  0.0  0.0
#                 0.0  0.0  1.0  0.0
#                 0.0  0.0  0.0  1.0 ]

# # For reward condition left
# A[1][:,:,2] = [ 1.0  0.0  0.0  0.0
#                 0.0  1.0  0.0  0.0
#                 0.0  0.0  1.0  0.0
#                 0.0  0.0  0.0  1.0 ]
# ```

# For the second modality, the reward modality, we want the agent to be able to infer "no reward" with certainty when in the centre and cue locations.
# In the left and right arm though, the agent should be agnostic as to which arm produces reward and loss. This is the modality that will be learned in this example.

# ```julia
# # For reward condition right
# A[2][:,:,1] = [ 1.0  0.0  0.0  1.0
#                 0.0  0.5  0.5  0.0
#                 0.0  0.5  0.5  0.0 ]

# # For reward condition left
# A[2][:,:,2] = [ 1.0  0.0  0.0  1.0
#                 0.0  0.5  0.5  0.0
#                 0.0  0.5  0.5  0.0 ]
# ```

# In the third modality, we want the agent to infer the reward condition state when in the cue location.
# To do this, we give it an uniform probability for all locations except the cue location, where it veridically will observe the reward condition state. 

# ```julia
# # For reward condition right
# A[3][:,:,1] = [ 0.5  0.5  0.5  1.0
#                 0.5  0.5  0.5  0.0 ]

# # For reward condition left
# A[3][:,:,2] = [ 0.5  0.5  0.5  0.0
#                 0.5  0.5  0.5  1.0 ]
# ```

# ##### Populating **B**

# For the first factor we populate the **B** with determined beliefs about how the location states change depended on its actions.
# For each action, it determines where to go, without having to go through any of the other location states.
# We encode this as:

# ```julia
# # For action "Go to Center Location"
# B[1][:,:,1] = [ 1.0  1.0  1.0  1.0 
#                 0.0  0.0  0.0  0.0
#                 0.0  0.0  0.0  0.0
#                 0.0  0.0  0.0  0.0 ]

# # For action "Go to Right Arm"
# B[1][:,:,2] = [ 0.0  0.0  0.0  0.0 
#                 1.0  1.0  1.0  1.0
#                 0.0  0.0  0.0  0.0
#                 0.0  0.0  0.0  0.0 ]

# # For action "Go to Left Arm"
# B[1][:,:,3] = [ 0.0  0.0  0.0  0.0 
#                 0.0  0.0  0.0  0.0
#                 1.0  1.0  1.0  1.0
#                 0.0  0.0  0.0  0.0 ]

# # For action "Go to Cue Location"
# B[1][:,:,4] = [ 0.0  0.0  0.0  0.0 
#                 0.0  0.0  0.0  0.0
#                 0.0  0.0  0.0  0.0
#                 1.0  1.0  1.0  1.0 ]
# ```

# For the last factor there is no control, so we will just set the **B** to be the identity matrix.

# ```julia
# # For second factor, which is not controlable by the agent
# B[2][:,:,1] = [ 1.0  0.0 
#                 0.0  1.0 ] 
# ```

# ##### Populating **C**
# For the preference parameters **C** we are not interested in the first and third modality, which we will just set to a vector of zeros for each observation in that modality.
# However, for the second modality, we want the agent to prefer the "reward observation" indexed as 2, and the dislike the "loss observation" indexed as 3.

# ```julia
# # Preference over locations modality
# C[1] = [0.0, 0.0, 0.0, 0.0]

# # Preference over reward modality
# C[2] = [0.0, 3.0, -3.0]

# # Preference over cue modality
# C[3] = [0.0, 0.0]
# ```   

# ##### Populating **D**
# For the prior over states **D** we will set the agent's belief to be correct in the location state factor and uniform, or agnostic, in the reward condition factor. 

# ```julia	
# # For the location state factor
# D[1] = [1.0, 0.0, 0.0, 0.0]

# # For the reward condition state factor
# D[2] = [0.5, 0.5]
# ```


# ##### Populating **E**
# For the prior over policies **E** we will set it to be uniform, meaning that the agent has no prior preference for any policy.

# ```julia	
# # Creating a vector of a uniform distribution over the policies. This means no preferences over policies.
# E .= 1.0/length(E)
# ```

# ##### Creating the prior over **A**
# When creating the prior over **A**, we use **A** as a template, by using 'deepcopy()'.
# Then we multiply this with a scaling parameter, setting the initial concentration parameters for the Dirichlet prior over **A**, **pA**.

# ```julia	
# pA = deepcopy(A)
# scale_concentration_parameter = 2.0
# pA .*= scale_concentration_parameter
# ```

# #### Creating Settings and Parameters Dictionary

# For the settings we set the 'use_param_info_gain' and 'use_states_info_gain' to true, meaning that the agent will take exploration and parameter learning into account when calculating the prior over policies.
# We set the policy length to 2, and specify modalities to learn, which in our case is the reward modality, indexed as 2.

# ```julia	
# settings = Dict(
#     "use_param_info_gain" => true,
#     "use_states_info_gain" => true,
#     "policy_len" => 2,
#     "modalities_to_learn" => [2]
# )
# ```

# For the parameters, we just use the default values, but specify the learning rate here, just to point it out.
# ```julia	
# parameters = Dict{String, Real}(
#     "lr_pA" => 1.0,
# )
# ```

# ### Initilising the Agent
# We can now initialise the agent with the parameters and settings we have just specified. 

# ```julia
# aif_agent = init_aif(
#     A, B, C = C, D = D, E = E, pA = pA, settings = settings, parameters = parameters
# );
# ```

# ### Simulation
# We are now ready for the perception-action-learning loop:

# ```julia
# # Settting the number of trials
# T = 100

# # Creating an initial observation and resetting environment (reward condition might change)
# obs = reset_TMaze!(Env)

# # Creating a for-loop that loops over the perception-action-learning loop T amount of times
# for t = 1:T

#     # Infer states based on the current observation
#     infer_states!(aif_agent, obs)

#     # Updates the A parameters
#     update_parameters!(aif_agent)

#     # Infer policies and calculate expected free energy
#     infer_policies!(aif_agent)

#     # Sample an action based on the inferred policies
#     chosen_action = sample_action!(aif_agent)

#     # Feed the action into the environment and get new observation.
#     obs = step_TMaze!(Env, chosen_action)
# end
# ```






================================================
FILE: docs/julia_files/WhyActiveInference.jl
================================================
# # Why Work with Active Inference?

# | Pros             | Cons             |
# |------------------|------------------|
# | Easy to use      | Limited features |
# | Widely supported | Not fully customizable |
# | Lightweight      | Lacks some advanced formatting |


================================================
FILE: docs/julia_files/WorkflowsUnfinished.jl
================================================

# ## Workflows
# This package has two main functions that can be used in a variety of workflows; `simulation` and `model fitting`.
# We will here outline two different kind of workflows that can be implemented using the ActiveInference.jl package.
# The first one will be a simulation workflow, where we are interested in simulating the agent's behaviour in a given environment.
# Here, we might be interested in the behevaiour of a simulated active inference agent in an environment, given some specified parameters.
# The second is a model fitting workflow, which is interesting for people in computational psychiatry/mathematical psychology. Here, we use observed data to fit an active inference mode and we will use a classical bayesian workflow in this regard.
# See [Bayesian Workflow for Generative Modeling in Computational Psychiatry](https://www.biorxiv.org/content/10.1101/2024.02.19.581001v1)

# ### Simulation
# In the simulation workflow, we are interested in simulating the agent's behaviour in a given environment. We might have some question wrt. behaviour expected under active inference,
# or we want to figure out whether our experimental task is suitable for active inference modelling. For these purposes, we will use a simple simulation workflow:

# - Decide on an environment the agent will interact with
# - Create a generative model based on that environment
# - Simulate the agent's behaviour in that environment
# - Analyse and visualize the agent's behaviour and inferences
# - Potential parameter recovery by model fitting on observed data

# First, deciding on the environment entails that we have some dynamic that we are interested in from an active inference perspective - a specific research question.
# Classical examples of environments are T-Mazes and Multi-Armed Bandits, that often involves some decision-making, explore-exploit and information seeking dynamics. These environments are easy to encode as POMDPs and are therefore suitable for active inference modelling.
# Importantly though this can be any kind of environment that provides the active inference agent with observations, and most often will also take actions so that the agent can interact with the environment.

# Based on an environment, you then create the generative model of the agent. Look under the [`Creating the POMDP Generative Model`](@ref "Creating the POMDP Generative Model") section for more information on how to do this.

# You then simulate the agent's behaviour in that environment through a perception-action-learning loop, as described under the 'Simulation' section.
# After this, you can analyse and visualize the agent's behaviour and inferences, and investigate what was important to the research question you had in mind.

# Parameter recovery is also a possibility here, if you are interested in seeing whether the parameters you are interested in are in fact recoverable, or there is a dynamic in the agent-environment interaction, where a parameter cannot be specified but only inferred.
# For an example of the latter, look up the 'As One and Many: Relating Individual and Emergent Group-Level Generative Models in Active Inference' paper, where parameters are inferred from group-level behaviour.

# ### Model Fitting with observed data
# For


================================================
FILE: docs/src/AgentCreation.md
================================================
```@meta
EditURL = "../julia_files/AgentCreation.jl"
```

# Creating the Agent

Having created the generative model parameters in the precious section, we're not ready to intialise an active inference agent.
Firstly, we'll have to specify some settings and hyperparameters that go into the agent struct. We'll begin with the setting:

### Settings
The settings are a dictionary that contains the following keys:

```julia
settings = Dict(
    "policy_len" => 1,
    "use_utility" => true,
    "use_states_info_gain" => true,
    "use_param_info_gain" => false,
    "action_selection" => "stochastic",
    "modalities_to_learn" => "all",
    "factors_to_learn" => "all",
    "FPI_num_iter" => 10,
    "FPI_dF_tol" => 0.001
)
```

The above shown values are the default and will work in most cases. If you're unsure about what to specify in the settings, you can just use the default values by not specifying them in the settings Dict for the agent.
Here, we'll briefly describe the keys in the settings dictionary:

- **`policy_len`** - Is the policy length, and as described previously is the number of actions the agent should plan in the future. This is provided as an integer.
- **`use_utility`** - Is a boolean that specifies whether the agent should use **C** in the expected free energy calculation, that guides the action selection in active inference. If set to `false`, the agent will not use the parameters specified in **C**.
- **`use_states_info_gain`** - Is a boolean that specifies whether the agent should use the information gain over states in the expected free energy calculation. If set to `false`, the agent will not use the information gain over states.
- **`use_param_info_gain`** - Is a boolean that specifies whether the agent should use the information gain over parameters in the expected free energy calculation. If set to `false`, the agent will not use the information gain over parameters. Only relevant when learning is included.
- **`action_selection`** - Is a string that specifies the action selection method. The options are `"stochastic"` and `"deterministic"`. If set to `"stochastic"`, the agent will sample from the posterior over policies, and if set to `"deterministic"`, the agent will choose the most probable action.
- **`modalities_to_learn`** - Is a vector of integers that specifies which modalities the agent should learn. If set to string `"all"`, the agent will learn all modalities. If set to `[1,2]`, the agent will only learn the first and second modality. Only relevant when learning of A is included.
- **`factors_to_learn`** - Is a vector of integers that specifies which factors the agent should learn. If set to string `"all"`, the agent will learn all factors. If set to `[1,2]`, the agent will only learn the first and second factor. Only relevant when learning of B and D is included.
- **`FPI_num_iter`** - Is an integer that specifies the number of fixed point iterations (FPI) to perform in the free energy minimization. It can be described as a stop function of the FPI algorithm.
- **`FPI_dF_tol`** - Is a float that specifies the tolerance of the free energy change in the FPI algorithm over each iteration. If the change in free energy is below this value, the FPI algorithm will also stop.

For more information on the specifics of the impact of these settings, look under the `Active Inference Theory` section in the documentation.

### Parameters
The parameters are a dictionary that contains the following keys:

```julia
parameters = Dict(
"gamma" => 16.0,
"alpha" => 16.0,
"lr_pA" => 1.0,
"fr_pA" => 1.0,
"lr_pB" => 1.0,
"fr_pB" => 1.0,
"lr_pD" => 1.0,
"fr_pD" => 1.0
)
```

The above shown values are the default. If you're unsure about what to specify in the parameters, you can just use the default values by not specifying them in the parameter Dict for the agent.
Here, we'll briefly describe the keys in the parameters dictionary containing the hyperparameters:
- **`alpha`** - Is the inverse temperature of the action selection process, and usually takes a value between 1 and 32. This is only relevant when action_selection is set to `"stochastic"`.
- **`gamma`** - Is the inverse temperature precision of the expected free energy, and usually takes a value between 1 and 32. If the value is high, the agent will be more certain in its beliefs regarding the posterior probability over policies.
- **`lr_pA`** - Is the learning rate of **A**, and usually takes a value between 0 and 1. Only relevant when learning is included, and this goes for all learning and forgetting rates.
- **`fr_pA`** - Is the forgetting rate of **A**, and usually takes a value between 0 and 1. If forgetting rate is 1 it means no forgetting.
- **`lr_pB`** - Is the learning rate of **B**, and usually takes a value between 0 and 1.
- **`fr_pB`** - Is the forgetting rate of **B**, and usually takes a value between 0 and 1. If forgetting rate is 1 it means no forgetting.
- **`lr_pD`** - Is the learning rate of **D**, and usually takes a value between 0 and 1.
- **`fr_pD`** - Is the forgetting rate of **D**, and usually takes a value between 0 and 1. If forgetting rate is 1 it means no forgetting.

Having now specified the setting and parameters, we can now initialise the active inference agent. This is done by calling the `init_aif` function, which takes the following arguments:

## Initilising the Agent

```julia
aif_agent = init_aif(
    A, B, C = C, D = D, E = E, settings = settings, parameters = parameters, verbose = false
);
```

You can access the settings and parameters of the agent by calling the agent struct on the agent:

```julia
aif_agent.parameters
```

````
Dict{String, Real} with 8 entries:
  "lr_pA" => 1.0
  "fr_pA" => 1.0
  "lr_pB" => 1.0
  "lr_pD" => 1.0
  "alpha" => 16.0
  "gamma" => 16.0
  "fr_pB" => 1.0
  "fr_pD" => 1.0
````

```julia
aif_agent.settings
```

````
Dict{String, Any} with 11 entries:
  "policy_len" => 1
  "FPI_dF_tol" => 0.001
  "control_fac_idx" => [1]
  "action_selection" => "stochastic"
  "num_controls" => [4, 1]
  "FPI_num_iter" => 10
  "modalities_to_learn" => "all"
  "use_utility" => true
  "factors_to_learn" => "all"
  "use_param_info_gain" => false
  "use_states_info_gain" => true
````

Having now initialised the agent, we are ready to implement it either in a simulation with a perception-action loop, or for use in model fitting with observed data.

## Initialising the Agent with Learning
If you want to include learning in the agent, you can do so by specifying the prior parameters `init_aif` function. Here is an example of how to initialise the agent with learning:

```julia
aif_agent = init_aif(
    A, B, C = C, D = D, E = E, pA = pA, pB = pB, pD = pD, settings = settings, parameters = parameters, verbose = false
);
```

Here, only the prior of the parameters that are to be learned should be specified.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*




================================================
FILE: docs/src/Fitting.md
================================================
```@meta
EditURL = "../julia_files/Fitting.jl"
```

# Model Fitting

In many cases, we want to be able to draw conclusions about specific observed phenomena, such as behavioural differences between distinct populations. A conventional approach in this context is model fitting, which involves estimating the parameter values of a model (e.g., prior beliefs) that are most likely given the observed behavior of a participant. This approach is often used in fields such as computational psychiatry or mathematical psychology  to develop more precise models and theories of mental processes, to find mechanistic differences between clinical populations, or to investigate the relationship between computational constructs such as Bayesian beliefs and neuronal dynamics.
## Quick Start
#### Model Fitting with ActionModels.jl

Model fitting in '**ActiveInference**' is mediated through '**ActionModels**', which is our sister package for implementing and fitting various behavioural models to data. The core of '**ActionModels**' is the action model function, which takes a single observation, runs the inference scheme (updating the agent's beliefs), and calculates the probability distribution over actions from which the agent samples its actions.
*(Check out the [ActionModels documentation](https://ilabcode.github.io/ActionModels.jl/dev/markdowns/Introduction/) for more details)*


To demonstrate this, let's define a very simple generative model with a single state factor and two possible actions, and then initialize our active inference object:
```julia
# Define the number of states, observations, and controls
n_states = [4]
n_observations = [4]
n_controls = [2]

# Define the policy length
policy_length = 1

# Use the create_matrix_templates function to create uniform A and B matrices.
A, B = create_matrix_templates(n_states, n_observations, n_controls, policy_length)

# Initialize an active inference object with the created matrices
aif = init_aif(A, B)
```

We can now use the `action_pomdp!` function (which serves as our active inference "action model") to calculate the probability distribution over actions for a single observation:
```julia
# Define observation
observation = [1]

# Calculate action probabilities
action_distribution = action_pomdp!(aif, observation)
```

#### Agent in ActionModels.jl
Another key component of '**ActionModels**' is an `Agent`, which wraps the action model and active inference object in a more abstract structure. The `Agent` is initialized using a `substruct` to include our active inference object, and the action model is our `action_pomdp!` function.

Let's first install '**ActionModels**' from the official Julia registry and import it:
```julia
Pkg.add("ActionModels")
using ActionModels
```

We can now create an `Agent` with the `action_pomdp!` function and the active inference object:

```julia
# Initialize agent with active inference object as substruct
agent = init_agent(
    action_pomdp!,  # The active inference action model
    substruct = aif # The active inference object
)
```
We use an initialized `Agent` primarily for fitting; however, it can also be used with a set of convenience functions to run simulations, which are described in [Simulation with ActionModels](./SimulationActionModels.md).

#### Fitting a Single Subject Model
We have our `Agent` object defined as above. Next, we need to specify priors for the parameters we want to estimate.

For example, let's estimate the action precision parameter `α` and use a Gamma distribution as its prior.

```julia
# Import the Distributions package
using Distributions

# Define the prior distribution for the alpha parameters inside a dictionary
priors = Dict("alpha" => Gamma(1, 1))
```
We can now use the `create_model` function to instantiate a probabilistic model object with data. This function takes the `Agent` object, the priors, and a set of observations and actions as arguments.

First, let's define some observations and actions as vectors:
```julia
# Define observations and actions
observations = [1, 1, 2, 3, 1, 4, 2, 1]
actions = [2, 1, 2, 2, 2, 1, 2, 2]
```

Now we can instantiate the probabilistic model object:
```julia
# Create the model object
single_subject_model = create_model(agent, priors, observations, actions)
```
The `single_subject_model` can be used as a standard Turing object. Performing inference on this model is as simple as:
```julia
results = fit_model(single_subject_model)
```
#### Fitting a Model with Multiple Subjects
Often, we have data from multiple subjects that we would like to fit simultaneously. The good news is that this can be done by instantiating our probabilisitc model on an entire dataset containing data from multiple subjects.

Let's define some dataset with observations and actions for three subjects:

```julia
# Import the DataFrames package
using DataFrames

# Create a DataFrame
data = DataFrame(
   subjectID = [1, 1, 1, 2, 2, 2, 3, 3, 3], # Subject IDs
   observations = [1, 1, 2, 3, 1, 4, 2, 1, 3], # Observations
   actions = [2, 1, 2, 2, 2, 1, 2, 2, 1] # Actions
)
```


```@raw html
<div><div style = "float: left;"><span>9×3 DataFrame</span></div><div style = "clear: both;"></div></div><div class = "data-frame" style = "overflow-x: scroll;"><table class = "data-frame" style = "margin-bottom: 6px;"><thead><tr class = "header"><th class = "rowNumber" style = "font-weight: bold; text-align: right;">Row</th><th style = "text-align: left;">subjectID</th><th style = "text-align: left;">observations</th><th style = "text-align: left;">actions</th></tr><tr class = "subheader headerLastRow"><th class = "rowNumber" style = "font-weight: bold; text-align: right;"></th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Int64" style = "text-align: left;">Int64</th></tr></thead><tbody><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">1</td><td style = "text-align: right;">1</td><td style = "text-align: right;">1</td><td style = "text-align: right;">2</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">2</td><td style = "text-align: right;">1</td><td style = "text-align: right;">1</td><td style = "text-align: right;">1</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">3</td><td style = "text-align: right;">1</td><td style = "text-align: right;">2</td><td style = "text-align: right;">2</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">4</td><td style = "text-align: right;">2</td><td style = "text-align: right;">3</td><td style = "text-align: right;">2</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">5</td><td style = "text-align: right;">2</td><td style = "text-align: right;">1</td><td style = "text-align: right;">2</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">6</td><td style = "text-align: right;">2</td><td style = "text-align: right;">4</td><td style = "text-align: right;">1</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">7</td><td style = "text-align: right;">3</td><td style = "text-align: right;">2</td><td style = "text-align: right;">2</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">8</td><td style = "text-align: right;">3</td><td style = "text-align: right;">1</td><td style = "text-align: right;">2</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">9</td><td style = "text-align: right;">3</td><td style = "text-align: right;">3</td><td style = "text-align: right;">1</td></tr></tbody></table></div>
```

To instantiate the probabilistic model on our dataset, we pass the `data` DataFrame to the `create_model` function along with the names of the columns that contain the subject identifiers, observations, and actions:
```julia
# Create the model object
multi_subject_model = create_model(
    agent,
    priors,
    data; # Dataframe
    grouping_cols = [:subjectID], # Column with subject IDs
    input_cols = ["observations"], # Column with observations
    action_cols = ["actions"] # Column with actions
)
```


To fit the model, we use the `fit_model` function as before:
```julia
results = fit_model(multi_subject_model)
```


#### Customizing the Fitting Procedure
The `fit_model` function has several optional arguments that allow us to customize the fitting procedure. For example, you can specify the number of iterations, the number of chains, the sampling algorithm, or to parallelize over chains:

```julia
results = fit_model(
    multi_subject_model, # The model object
    parallelization = MCMCDistributed(), # Run chains in parallel
    sampler = NUTS(;adtype=AutoReverseDiff(compile=true)), # Specify the type of sampler
    n_itererations = 1000, # Number of iterations,
    n_chains = 4, # Number of chains
)
```
'**Turing**' allows us to run distributed `MCMCDistributed()` or threaded `MCMCThreads()` parallel sampling. The default is to run chains serially `MCMCSerial()`. For information on the available samplers see the [Turing documentation](https://turing.ml/dev/docs/using-turing/samplers/).

#### Results

The output of the `fit_model` function is an object that contains the standard '**Turing**' chains which we can use to extract the summary statistics of the posterior distribution.

Let's extract the chains from the results object:

```julia
chains = results.chains
```

````
Chains MCMC chain (1000×15×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 19.56 seconds
Compute duration  = 19.56 seconds
parameters        = parameters[1, 1], parameters[1, 2], parameters[1, 3]
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
        parameters      mean       std      mcse   ess_bulk   ess_tail      rhat   ess_per_sec
            Symbol   Float64   Float64   Float64    Float64    Float64   Float64       Float64

  parameters[1, 1]    1.0256    0.9533    0.0334   604.0459   456.9243    0.9996       30.8801
  parameters[1, 2]    0.9735    0.9665    0.0330   588.0456   500.3343    1.0008       30.0621
  parameters[1, 3]    1.0613    1.0308    0.0422   452.5146   370.6840    1.0001       23.1335

Quantiles
        parameters      2.5%     25.0%     50.0%     75.0%     97.5%
            Symbol   Float64   Float64   Float64   Float64   Float64

  parameters[1, 1]    0.0468    0.3173    0.7704    1.3994    3.5222
  parameters[1, 2]    0.0272    0.2746    0.6785    1.3652    3.4548
  parameters[1, 3]    0.0468    0.3363    0.7380    1.4649    3.5748

````

Note that the parameter names in the chains are somewhat cryptic. We can use the `rename_chains` function to rename them to something more understandable:

```julia
renamed_chains = rename_chains(chains, multi_subject_model)
```

````
Chains MCMC chain (1000×15×1 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 19.56 seconds
Compute duration  = 19.56 seconds
parameters        = subjectID:1.alpha, subjectID:2.alpha, subjectID:3.alpha
internals         = lp, n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size

Summary Statistics
         parameters      mean       std      mcse   ess_bulk   ess_tail      rhat   ess_per_sec
             Symbol   Float64   Float64   Float64    Float64    Float64   Float64       Float64

  subjectID:1.alpha    1.0256    0.9533    0.0334   604.0459   456.9243    0.9996       30.8801
  subjectID:2.alpha    0.9735    0.9665    0.0330   588.0456   500.3343    1.0008       30.0621
  subjectID:3.alpha    1.0613    1.0308    0.0422   452.5146   370.6840    1.0001       23.1335

Quantiles
         parameters      2.5%     25.0%     50.0%     75.0%     97.5%
             Symbol   Float64   Float64   Float64   Float64   Float64

  subjectID:1.alpha    0.0468    0.3173    0.7704    1.3994    3.5222
  subjectID:2.alpha    0.0272    0.2746    0.6785    1.3652    3.4548
  subjectID:3.alpha    0.0468    0.3363    0.7380    1.4649    3.5748

````

That looks better! We can now use the '**StatsPlots**' package to plot the chain traces and density plots of the posterior distributions for all subjects:
```julia
using StatsPlots # Load the StatsPlots package

plot(renamed_chains)
```

> [!WARNING] Image missing: assets/chain_traces.png
> *(image2)*

We can also visualize the posterior distributions against the priors. This can be done by first taking samples from the prior:
```julia
# Sample from the prior
prior_chains = sample(multi_subject_model, Prior(), 1000)
# Rename parameters in the prior chains
renamed_prior_chains = rename_chains(prior_chains, multi_subject_model)
```
To plot the posterior distributions against the priors, we use the `plot_parameters` function:
```julia
plot_parameters(renamed_prior_chains, renamed_chains)
```

> [!WARNING] Image missing: assets/posteriors.png
> *(image3)*

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*




================================================
FILE: docs/src/GenerativeModelCreation.md
================================================
```@meta
EditURL = "../julia_files/GenerativeModelCreation.jl"
```

# Creating the POMDP Generative Model

In this section we will go through the process of creating a generative model and how it should be structured. In this part, we will show the code necessary for correct typing of the generative model.
For a theoretical explanation of POMDPs look under the "Theory" section further down in the documentation.

## Typing of the POMDP parameters

In ActiveInference.jl, it is important that the parameters describing the generative model is typed correctly.
The correct typing of the generative model parameters, which often take the shapes of matrices, tensors and vectors.
The collections of generative model parameters are colloquially referred to as **A**, **B**, **C**, **D**, and **E**. We will denote these parameters by their letter in bold. For a quick refresher this is the vernacular used to describe these parameter collections:

- **A** - Observation Likelihood Model
- **B** - Transition Likelihood Model
- **C** - Prior over Observations
- **D** - Prior over States
- **E** - Prior over Policies

These should be typed the following way in ActiveInference.jl:

```julia
A = Vector{Array{Float64, 3}}(undef, n_modalities)
B = Vector{Array{Float64, 3}}(undef, n_factors)
C = Vector{Vector{Float64, 3}}(undef, n_modalities)
D = Vector{Vector{Float64, 3}}(undef, n_factors)
D = Vector{Float64, 3}(undef, n_policies)
```

Each of the parameter collections are vectors, where each index in the vector contains the parameters associated with a specific modality or factor.
However, creating these from scratch is not necessary, as we have created a helper function that can create a template for these parameters.

## Helper Function for GM Templates
Luckily, there is a helper function that helps create templates for the generative model parameters. This function is called `create_matrix_templates`.

```julia
A, B, C, D, E = create_matrix_templates(n_states::Vector{Int64}, n_observations::Vector{Int64}, n_controls::Vector{Int64}, policy_length::Int64, template_type::String)
```

This function takes the five arguments `n_states`, `n_observations`, `n_controls`, `policy_length`, and `template_type`, which have all the necessary information to create the
right structure of the generative model parameters. We will go through these arguments one by one:

\

- **n_states** - This is the number of states in the environment. The environment can have different kinds of states, which are often referred to as factors. Could be a location factor and a reward condition factor. It takes a vector of integers, where each integer represents a factor, and the value of the integer is the number of states in that factor. E.g. if we had an environment with two factors, one location factor with 4 states and one reward condition factor with 2 states, the argument would look like this: `[4,2]`
\

- **n_observations** - This is the number of observations the agent can make in the environment. The observations are often referred to as modalities. Could be a location modality, a reward modality and a cue modality. Similarly to the first argument, it takes a vector of integers, where each integer represents a modality, and the value of the integer is the number of observations in that modality. E.g. if we had an environment with three modalities, one location modality with 4 observations, one reward modality with 3 observations and one cue modality with 2 observations, the argument would look like this: `[4,3,2]`
\

- **n_controls** - This is the number of controls the agent have in the environment. The controls are the actions the agent can take in the different factors. Could be moving left or right, or choosing between two different rewards. It has one control integer for each factor, where the integer represents the number of actions in that factor. If the agent cannot control a factor, the integer should be 1. E.g. if we had an environment with two factors, one location factor with 4 actions and one reward condition factor with 1 action, the argument would look like this: `[4,1]`
\

- **policy_length** - This is the length of the policies of the agent, and is taken as an integer. The policy is a sequence of actions the agent can take in the environment. The length of the policy describes how many actions into the future the agent is planning. For example, if the agent is planning two steps into the future, the policy length would be 2, and each policy would consist of 2 actions. In that case the argument would look like this: `2`
\

- **template_type** - This is a string that describes the type of template you want to create, or in other words, the initial filling of the generative model structure. There are three options; `"uniform"`, which is default, `"random"`, and `"zeros"`.

If we were to use the arguments from the examples above, the function call would look like this:

```julia
n_states = [4,2]
n_observations = [4,3,2]
n_controls = [4,1]
policy_length = 2
template_type = "zeros"

A, B, C, D, E = create_matrix_templates(n_states, n_observations, n_controls, policy_length, template_type);
```

When these parameter collections have been made, each factor/modality can be accessed by indexing the collection with the factor/modality index like:

```julia
A[1] # Accesses the first modality in the observation likelihood model
B[2] # Accesses the second factor in the transition likelihood model
C[3] # Accesses the third modality in the prior over observations
D[1] # Accesses the first factor in the prior over states
```

The E-parameters are not a divided into modalities or factors, as they are the prior over policies.

## Populating the Parameters
Now that the generative model parameter templates ahave been created, they can now be filled with the desired values, ie. populating the parameters.
Let's take the example of filling **A** with some valus. To start, let's print out the first modality of the A so we get a sense of the dimensions:

```julia
A[1]
```

````
4×4×2 Array{Float64, 3}:
[:, :, 1] =
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0

[:, :, 2] =
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0
````

For a quick recap on the POMDP generative model parameteres look up the [`POMDP Theory`](@ref "The Generative Model Conceptually") section further down in the documentation.

For now, we'll suffice to say that the first modality of **A** is a 3D tensor, where the first dimension are observations in the first modality, the second dimension the first factor, and the third dimension is the second factor.
Remember **A** maps the agents beliefs on how states generate observations. In this case, we have two 4x4 matrices, one matrix for each state int the second factor. This could be how location observations (1st dimenstion) map onto location states (2nd dimension) and reward condition (3rd dimension).
For the sake of simplicity, let's assume that the agent can infer location states with certainty based on location observations. In this case we could populate the first modality of **A** like this:

```julia
# For reward condition right
A[1][:,:,1] = [ 1.0  0.0  0.0  0.0
                0.0  1.0  0.0  0.0
                0.0  0.0  1.0  0.0
                0.0  0.0  0.0  1.0 ]

# For reward condition left
A[1][:,:,2] = [ 1.0  0.0  0.0  0.0
                0.0  1.0  0.0  0.0
                0.0  0.0  1.0  0.0
                0.0  0.0  0.0  1.0 ]
```

In this case the agent would infer the location state with certainty based on the location observations. One could also make the **A** more noisy in this modality, which could look like:

```julia
# For reward condition right
A[1][:,:,1] = [ 0.7  0.1  0.1  0.1
                0.1  0.7  0.1  0.1
                0.1  0.1  0.7  0.1
                0.1  0.1  0.1  0.7 ]

# For reward condition left
A[1][:,:,2] = [ 0.7  0.1  0.1  0.1
                0.1  0.7  0.1  0.1
                0.1  0.1  0.7  0.1
                0.1  0.1  0.1  0.7 ]
```

Importantly the columns should always add up to 1, as we are here dealing with categorical probability distributions.
For the other parameters, the process is similar, but the dimensions of the matrices will differ. For **B** the dimensions are states to states, and for **C** and **D** the dimensions are states to observations and states to factors respectively.
Look up the `T-Maze Simulation` (insert reference here) example for a full example of how to populate the generative model parameters.

## Creating Dirichlet Priors over Parameters
When learning is included, we create Dirichlet priors over the parameters **A**, **B**, and **D**. We usually do this by taking the created **A**, **B**, and **D** parameters and multiplying them with a scalar, which is the concentration parameter of the Dirichlet distribution.
For more information on the specifics of learning and Dirichlet priors, look under the `Active Inference Theory` section in the documentation. Note here, that when we implement learning of a parameter, the parameter is going to be defined by its prior and no longer the initial
parameter that we specified. This is because the agent will update the parameter based on the prior and the data it receives. An example of how we would create a Dirichlet prior over **A** could look:

```julia
pA = deepcopy(A)
scale_concentration_parameter = 2.0
pA .*= scale_concentration_parameter
```

This is not relevant if learning is not included. If learning is not included, the parameters are fixed and the agent will not update them. The value of the scaling parameter determines how much each data observation impacts the update of the parameter.
If the scaling is high, e.g. 50, then adding one data point will have a small impact on the parameter. If the scaling is low, e.g. 0.1, then adding one data point will have a large impact on the parameter. The update function updates the parameters by normalising the concentration parameters of the Dirichlet distribution.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*




================================================
FILE: docs/src/GenerativeModelTheory.md
================================================
```@meta
EditURL = "../julia_files/GenerativeModelTheory.jl"
```

# The Generative Model Conceptually

The generative model is the parameters that constitute the agent's beliefs on how the hidden states of the environment generates observations based on states, and how hidden underlying states changes over time.
In the generative model is also the beliefs of how the agent through actions can influence the states of the environment. Together this holds the buidling blocks that allows for the perception-action loop.

There are five main buidling blocks of the generative model which are; **A**, **B**, **C**, **D**, and **E**.
Each of these contain parameters that describe the agent's beliefs about the environment.
We will now go through these conecptually one at a time.

## A
**A** is the observation likelihood model, and describes the agent's beliefs about how the hidden states of the environment generates observations.
Practically in this package, and other POMDP implemantations as well, this is described through a series of categorical distributions, meaning that for each observation, there is a categorical probability distribution over how likely each hidden state is to generate that observation.
Let us for example imagine a simple case, where the agent is in a four location state environment, could be a 2x2 gridworld. In this case, there would be one obseration linked to each hidden state, and **A** then maps the agent's belief of how likely each hidden location state is to generate each observation.
The agent can then use this belief to infer what state it is in based on the observation it receives. Let's look at an example **A**, which in this case would be a 4x4 matrix:

```math
A =
\overset{\text{\normalsize States}\vphantom{\begin{array}{c} 0 \\ 0 \end{array}}}{
    \begin{array}{cccc}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & 1 & 0 \\
        0 & 0 & 0 & 1
    \end{array}
}
\quad
\text{\normalsize Observations}
```

In this case, the agent is quite certain about which states produces which observations. This matrix could be made more uncertain to the point of complete uniformity and it could be made certain in the sense of each column being a one-hot vector.
In the case of a certain **A**, the generative model stops being a "partially observable" Markov decision process, and becomes a fully observable one, making it a Markov decision process (MDP). For a more technical and mathematical definition of the observation likelihood model.

## B
**B** is the transition likelihood model that encodes the agent's beliefs about how the hidden states of the environment changes over time.
This is also made up of categorical distributions, though instead of observations to states, it maps states to states.
If we take the same case again, a 2x2 gridworld, we would have a 4x4 matrix that describes how the agent believes the states evolve over time.
An extra addition to **B**, is that it can depend on actions, meaning that it can believe that the hidden states of the environment change differently depending on the action taken by the agent.
Due to this fact, we would the have a matrix for each action, making **B** a 3 dimensional tensor, with 2 dimensions for the "from" state and the "to" state, and then an action dimension.
Let's look at an example of a slice of **B** for the action "down" in the grid world, which in this case would be a 4x4 matrix:

```math
B("down") =
\overset{\text{\normalsize Previous State}\vphantom{\begin{array}{c} 0 \\ 0 \end{array}}}{
    \begin{array}{cccc}
        0 & 0 & 0 & 0 \\
        1 & 1 & 0 & 0 \\
        0 & 0 & 0 & 0 \\
        0 & 0 & 1 & 1
    \end{array}
}
\quad
\text{\normalsize Current State}
```

We could make 3 more similar matrices for the actions "up", "left", and "right", and then we would have the full **B** tensor for the gridworld. But here, the main point is that
**B** decsribes the agent's belief of how hidden states change over time, and this can be dependent on actions, but might also be independent of actions, and thus the agent believes that the changes are out of its control.

## C
**C** is the prior over observations, also called preferences over observations. This is an integral part of the utility of certain observations, i.e. it encodes how much the agent prefers or dislikes certain observations.
**C** is a simple vector over observations, where each entry is a value that describes the utility or preference of that specific observation.
If we continue with the simple 2x2 gridworld example, we would have 4 observations, one for each location state (same amount of observations as in **A**).
Let's say that we would like for the agent to dislike observing the top left location (indexed as 1), and prefer the bottom right location (indexed as 4). We would then create **C** in the following way:

```math
C =
\begin{array}{cccc}
    -2 & 0 & 0 & 2 \\
\end{array}
```

The magnitude of the values in **C** is arbitrary, and denotes a ratio and amount of dislike/preference. Here, we have chosen the value of -2 and 2
to encode that the agent dislikes the top left location just as much as it likes the bottom right location. The zeros in between just means that the agent has not preference or dislike for these locatin observations.
Note that since **C** is not a categorical distribution, it does not need to sum to 1, and the values can be any real number.

## D
**D** is the prior over states, and is the agent's beliefs about the initial state of the environment. This is also a simple vector that is a categorical distribution.
Note that if **A** is certain, then **D** does not matter a lot for the inference process, as the agent can infer the state from the observation. However, if **A** is uncertain,
then **D** becomes very important, as it serves as the agent's anchor point of where it is initially in the environment. In the case of out
2x2 gridworld, we would have a vector with 4 entries, one for each location state. If we assume that the agent correctly infers it's initial location as upper left corner, **D** would look like:

```math
D =
\begin{array}{cccc}
    1 & 0 & 0 & 0 \\
\end{array}
```

## E
**E** is the prior over policies, and can be described as the agent's habits. Policies in Active Inference vernacular are sets of actions, with an action for each step in the future, specified by a policy length.
It is a categorical distribution over policies, with a probability for each policy. This will have an effect on the agent posterior over policies,
which is the probability of taking a certain action at a time step. This will often be set to a uniform distribution, if we are not interested in giving the agent habits.
Let us assume that we will give our agent a uniform **E** for a policy length of 2, this mean that we will have a uniform categorical distribution over 16 possible policies ``(4 (actions) ^ {2 (policy length)})``:

```math
E =
\begin{array}{cccc}
0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 & 0.0625 \\
\end{array}
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*




================================================
FILE: docs/src/index.md
================================================
```@meta
CurrentModule = ActiveInference
```

# ActiveInference

Documentation for [ActiveInference](https://github.com/ilabcode/ActiveInference.jl).

```@index
```

```@autodocs
Modules = [ActiveInference, ActiveInference.Environments]
```



================================================
FILE: docs/src/Introduction.md
================================================
```@meta
EditURL = "../julia_files/Introduction.jl"
```

# Introduction to the ActiveInference.jl package

This package is a Julia implementation of the Active Inference framework, with a specific focus on cognitive modelling.
In its current implementation, the package is designed to handle scenarios that can be modelled as discrete state spaces, with 'partially observable Markov decision process' (POMDP).
In this documentation we will go through the basic concepts of how to use the package for different purposes; simulation and model inversion with Active Inference, also known as parameter estimation.

## Installing Package
Installing the package is done by adding the package from the julia official package registry in the following way:

```julia
using Pkg
Pkg.add("ActiveInference")
```

Now, having added the package, we simply import the package to start using it:

```julia
using ActiveInference
```

In the next section we will go over the basic concepts of how to start using the package. We do this by providing instructions on how to create and design a generative model, that can be used for both simulation and parameter estimation.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*




================================================
FILE: docs/src/Simulation.md
================================================
```@meta
EditURL = "../julia_files/Simulation.jl"
```

# Simulation with ActiveInference.jl
When simulating with active inference we need a perception-action loop. This loop will perform the following steps:
1. The agent will infer the states of the environment based on its generative model and an observation. The inference here is optimized through the minimization of the variational free energy (see `Active Inference Theory Perception`).
2. The agent will infer the best action based on the minimization of the expected free energy (see `Active Inference Theory Action`).
3. The agent will perform the action in the environment and receive an observation for use in the next iteration.

*Note: for learning included, look at the section below.*

#### The Perception-Action loop:
```julia
T = n_iterations

for t = 1:T

    infer_states!(aif_agent, observation)

    infer_policies!(aif_agent)

    chosen_action = sample_action!(aif_agent)

    observation = environment!(env, chosen_action)

end
```

#### The Perception-Action-Learning loop:
When learning is included, the loop is very similar except for the addition of the update functions, which should be implemented at different points in the loop.
Below we will show how to include learning of the parameters. It is important that only the parameters which have been provided to the agent as a prior are being updated.
```julia
T = n_iterations

for t = 1:T

   infer_states!(aif_agent, observation)

   update_parameters!(aif_agent)

   infer_policies!(aif_agent)

   chosen_action = sample_action!(aif_agent)

   observation = environment!(env, chosen_action)

end
```

The only addition here is the `update_parameters!(aif_agent)` function, which updates the parameters of the agent, based on which priors it has been given.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*




================================================
FILE: docs/src/SimulationActionModels.md
================================================
```@meta
EditURL = "../julia_files/SimulationActionModels.jl"
```

# Simulation with ActionModels.jl

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*




================================================
FILE: docs/src/TMazeSimulationExample.md
================================================
```@meta
EditURL = "../julia_files/TMazeSimulationExample.jl"
```

# Simulation Example T-Maze

We will start from the importing of the necessary modules.

```julia
using ActiveInference
using ActiveInference.Environments
```

We will create a T-Maze environment with a probability of 0.9 for reward in the the reward condition arm.
This is a premade environment in the ActiveInference.jl package.

```julia
env = TMazeEnv(0.9)
initialize_gp(env)
```

### Creating the Generative Model
#### The Helper Function

When creating the generative model we can make use of the helper function, making it convenient to create the correct structure for the generative model parameters.

To use the helper function we need to know the following:

- Number of states in each factor of the environment
- Number of observations in each modality
- Number of controls or actions in each factor
- Policy length of the agent
- Initial fill for the parameters

Let's start with the factors of the environment. Let's take a look at the T-Maze environment:

> [!WARNING] Image missing: assets/TMazeIllustrationSmaller.png
> *(image1)*

We here have two factors with the following number of states:

|       | Location Factor   |       | Reward Condition Factor   |
|:------|:------------------|:------|:------------------------- |
| 1.    | Centre            | 1.    | Reward Condition Left     |
| 2.    | Left Arm          | 2.    | Reward Condition Right    |
| 3.    | Right Arm         |       |                           |
| 4.    | Cue               |       |                           |

We will define this as a vector the following way:

```julia
n_states = [4, 2]
```

We will now define the modalities:

|       | Location Modality |       | Reward Modality           |       | Cue Modality    |
|:------|:------------------|:------|:------------------------- |:------|:--------------- |
| 1.    | Centre            | 1.    | No Reward                 | 1.    | Cue Left        |
| 2.    | Left Arm          | 2.    | Reward                    | 2.    | Cue Right       |
| 3.    | Right Arm         | 3.    | Loss                      |       |                 |
| 4.    | Cue               |       |                           |       |                 |

Here we have 3 modalities, with 4, 3, and 2 observations in each. We will define this as a vector the following way:

```julia
n_observations = [4, 3, 2]
```

Now, let's take a look at the actions, or controls:

|       | Controls Location Factor       |    | Controls Reward Condition Factor       |
|:------|:-------------------------------|:-- |:---------------------------------------|
| 1.    | Go to Centre                   | 1. | No Control                                      |
| 2.    | Go to Left Arm                 |    |                                       |
| 3.    | Go to Right Arm                |    |                                       |
| 4.    | Go to Cue                      |    |                                       |

As we see here, the agent cannot control the reward condition factor, and it therefore believes that there is only one way states can transition in this factor, which is independent of the agent's actions.
We will define this as a vector the following way:

```julia
n_controls = [4, 1]
```

Now we can define the policy length of the agent. In this case we will just set it to 2, meaning that the agent plans two timesteps ahead in the future.
We will just specify this as an integer:

```julia
policy_length = 2
```

The last thing we need to define is the initial fill for the parameters. We will just set this to zeros for now.

```julia
template_type = "zeros"
```

Having defined all the arguments that go into the helper function, we can now create the templates for the generative model parameters.

```julia
A, B, C, D, E = create_matrix_templates(n_states, n_observations, n_controls, policy_length, template_type);
```

#### Populating the Generative Model
##### Populating **A**

Let's take a look at the shape of the first modality in the A parameters:

```julia
A[1]
```

````
4×4×2 Array{Float64, 3}:
[:, :, 1] =
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0

[:, :, 2] =
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0
````

For this first modality we provide the agent with certain knowledge on how location observations map onto location states.
We do this the following way:

```julia
# For reward condition right
A[1][:,:,1] = [ 1.0  0.0  0.0  0.0
                0.0  1.0  0.0  0.0
                0.0  0.0  1.0  0.0
                0.0  0.0  0.0  1.0 ]

# For reward condition left
A[1][:,:,2] = [ 1.0  0.0  0.0  0.0
                0.0  1.0  0.0  0.0
                0.0  0.0  1.0  0.0
                0.0  0.0  0.0  1.0 ]
```

For the second modality, the reward modality, we want the agent to be able to infer "no reward" with certainty when in the centre and cue locations.
In the left and right arm though, the agent should be agnostic as to which arm produces reward and loss. This is the modality that will be learned in this example.

```julia
# For reward condition right
A[2][:,:,1] = [ 1.0  0.0  0.0  1.0
                0.0  0.5  0.5  0.0
                0.0  0.5  0.5  0.0 ]

# For reward condition left
A[2][:,:,2] = [ 1.0  0.0  0.0  1.0
                0.0  0.5  0.5  0.0
                0.0  0.5  0.5  0.0 ]
```

In the third modality, we want the agent to infer the reward condition state when in the cue location.
To do this, we give it an uniform probability for all locations except the cue location, where it veridically will observe the reward condition state.

```julia
# For reward condition right
A[3][:,:,1] = [ 0.5  0.5  0.5  1.0
                0.5  0.5  0.5  0.0 ]

# For reward condition left
A[3][:,:,2] = [ 0.5  0.5  0.5  0.0
                0.5  0.5  0.5  1.0 ]
```

##### Populating **B**

For the first factor we populate the **B** with determined beliefs about how the location states change depended on its actions.
For each action, it determines where to go, without having to go through any of the other location states.
We encode this as:

```julia
# For action "Go to Center Location"
B[1][:,:,1] = [ 1.0  1.0  1.0  1.0
                0.0  0.0  0.0  0.0
                0.0  0.0  0.0  0.0
                0.0  0.0  0.0  0.0 ]

# For action "Go to Right Arm"
B[1][:,:,2] = [ 0.0  0.0  0.0  0.0
                1.0  1.0  1.0  1.0
                0.0  0.0  0.0  0.0
                0.0  0.0  0.0  0.0 ]

# For action "Go to Left Arm"
B[1][:,:,3] = [ 0.0  0.0  0.0  0.0
                0.0  0.0  0.0  0.0
                1.0  1.0  1.0  1.0
                0.0  0.0  0.0  0.0 ]

# For action "Go to Cue Location"
B[1][:,:,4] = [ 0.0  0.0  0.0  0.0
                0.0  0.0  0.0  0.0
                0.0  0.0  0.0  0.0
                1.0  1.0  1.0  1.0 ]
```

For the last factor there is no control, so we will just set the **B** to be the identity matrix.

```julia
# For second factor, which is not controlable by the agent
B[2][:,:,1] = [ 1.0  0.0
                0.0  1.0 ]
```

##### Populating **C**
For the preference parameters **C** we are not interested in the first and third modality, which we will just set to a vector of zeros for each observation in that modality.
However, for the second modality, we want the agent to prefer the "reward observation" indexed as 2, and the dislike the "loss observation" indexed as 3.

```julia
# Preference over locations modality
C[1] = [0.0, 0.0, 0.0, 0.0]

# Preference over reward modality
C[2] = [0.0, 3.0, -3.0]

# Preference over cue modality
C[3] = [0.0, 0.0]
```

##### Populating **D**
For the prior over states **D** we will set the agent's belief to be correct in the location state factor and uniform, or agnostic, in the reward condition factor.

```julia
# For the location state factor
D[1] = [1.0, 0.0, 0.0, 0.0]

# For the reward condition state factor
D[2] = [0.5, 0.5]
```

##### Populating **E**
For the prior over policies **E** we will set it to be uniform, meaning that the agent has no prior preference for any policy.

```julia
# Creating a vector of a uniform distribution over the policies. This means no preferences over policies.
E .= 1.0/length(E)
```

##### Creating the prior over **A**
When creating the prior over **A**, we use **A** as a template, by using 'deepcopy()'.
Then we multiply this with a scaling parameter, setting the initial concentration parameters for the Dirichlet prior over **A**, **pA**.

```julia
pA = deepcopy(A)
scale_concentration_parameter = 2.0
pA .*= scale_concentration_parameter
```

#### Creating Settings and Parameters Dictionary

For the settings we set the 'use_param_info_gain' and 'use_states_info_gain' to true, meaning that the agent will take exploration and parameter learning into account when calculating the prior over policies.
We set the policy length to 2, and specify modalities to learn, which in our case is the reward modality, indexed as 2.

```julia
settings = Dict(
    "use_param_info_gain" => true,
    "use_states_info_gain" => true,
    "policy_len" => 2,
    "modalities_to_learn" => [2]
)
```

For the parameters, we just use the default values, but specify the learning rate here, just to point it out.
```julia
parameters = Dict{String, Real}(
    "lr_pA" => 1.0,
)
```

### Initilising the Agent
We can now initialise the agent with the parameters and settings we have just specified.

```julia
aif_agent = init_aif(
    A, B, C = C, D = D, E = E, pA = pA, settings = settings, parameters = parameters
);
```

### Simulation
We are now ready for the perception-action-learning loop:

```julia
# Settting the number of trials
T = 100

# Creating an initial observation and resetting environment (reward condition might change)
obs = reset_TMaze!(Env)

# Creating a for-loop that loops over the perception-action-learning loop T amount of times
for t = 1:T

    # Infer states based on the current observation
    infer_states!(aif_agent, obs)

    # Updates the A parameters
    update_parameters!(aif_agent)

    # Infer policies and calculate expected free energy
    infer_policies!(aif_agent)

    # Sample an action based on the inferred policies
    chosen_action = sample_action!(aif_agent)

    # Feed the action into the environment and get new observation.
    obs = step_TMaze!(Env, chosen_action)
end
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*




================================================
FILE: docs/src/WhyActiveInference.md
================================================
```@meta
EditURL = "../julia_files/WhyActiveInference.jl"
```

# Why Work with Active Inference?

| Pros             | Cons             |
|------------------|------------------|
| Easy to use      | Limited features |
| Widely supported | Not fully customizable |
| Lightweight      | Lacks some advanced formatting |

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*




================================================
FILE: docs/src/WorkflowsUnfinished.md
================================================
```@meta
EditURL = "../julia_files/WorkflowsUnfinished.jl"
```

## Workflows
This package has two main functions that can be used in a variety of workflows; `simulation` and `model fitting`.
We will here outline two different kind of workflows that can be implemented using the ActiveInference.jl package.
The first one will be a simulation workflow, where we are interested in simulating the agent's behaviour in a given environment.
Here, we might be interested in the behevaiour of a simulated active inference agent in an environment, given some specified parameters.
The second is a model fitting workflow, which is interesting for people in computational psychiatry/mathematical psychology. Here, we use observed data to fit an active inference mode and we will use a classical bayesian workflow in this regard.
See [Bayesian Workflow for Generative Modeling in Computational Psychiatry](https://www.biorxiv.org/content/10.1101/2024.02.19.581001v1)

### Simulation
In the simulation workflow, we are interested in simulating the agent's behaviour in a given environment. We might have some question wrt. behaviour expected under active inference,
or we want to figure out whether our experimental task is suitable for active inference modelling. For these purposes, we will use a simple simulation workflow:

- Decide on an environment the agent will interact with
- Create a generative model based on that environment
- Simulate the agent's behaviour in that environment
- Analyse and visualize the agent's behaviour and inferences
- Potential parameter recovery by model fitting on observed data

First, deciding on the environment entails that we have some dynamic that we are interested in from an active inference perspective - a specific research question.
Classical examples of environments are T-Mazes and Multi-Armed Bandits, that often involves some decision-making, explore-exploit and information seeking dynamics. These environments are easy to encode as POMDPs and are therefore suitable for active inference modelling.
Importantly though this can be any kind of environment that provides the active inference agent with observations, and most often will also take actions so that the agent can interact with the environment.

Based on an environment, you then create the generative model of the agent. Look under the [`Creating the POMDP Generative Model`](@ref "Creating the POMDP Generative Model") section for more information on how to do this.

You then simulate the agent's behaviour in that environment through a perception-action-learning loop, as described under the 'Simulation' section.
After this, you can analyse and visualize the agent's behaviour and inferences, and investigate what was important to the research question you had in mind.

Parameter recovery is also a possibility here, if you are interested in seeing whether the parameters you are interested in are in fact recoverable, or there is a dynamic in the agent-environment interaction, where a parameter cannot be specified but only inferred.
For an example of the latter, look up the 'As One and Many: Relating Individual and Emergent Group-Level Generative Models in Active Inference' paper, where parameters are inferred from group-level behaviour.

### Model Fitting with observed data
For

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*





================================================
FILE: src/ActiveInference.jl
================================================
module ActiveInference

using ActionModels
using LinearAlgebra
using IterTools
using Random
using Distributions
using LogExpFunctions
using ReverseDiff

include("utils/maths.jl")
include("pomdp/struct.jl")
include("pomdp/learning.jl")
include("utils/utils.jl")
include("pomdp/inference.jl")
include("ActionModelsExtensions/get_states.jl")
include("ActionModelsExtensions/get_parameters.jl")
include("ActionModelsExtensions/get_history.jl")
include("ActionModelsExtensions/set_parameters.jl")
include("ActionModelsExtensions/reset.jl")
include("ActionModelsExtensions/give_inputs.jl")
include("ActionModelsExtensions/set_save_history.jl")
include("pomdp/POMDP.jl")
include("utils/helper_functions.jl")
include("utils/create_matrix_templates.jl")

export # utils/create_matrix_templates.jl
        create_matrix_templates,
       
       # utils/maths.jl
       normalize_distribution,
       softmax_array,
       normalize_arrays,

       # utils/utils.jl
       array_of_any_zeros, 
       onehot,
       get_model_dimensions,

       # struct.jl
       init_aif,
       infer_states!,
       infer_policies!,
       sample_action!,
       update_A!,
       update_B!,
       update_D!,
       update_parameters!,

       # POMDP.jl
       action_pomdp!,

       # ActionModelsExtensions
       get_states,
       get_parameters,
       get_history,
       set_parameters!,
       reset!,
       single_input!,
       give_inputs!,
       set_save_history!

    module Environments

    using LinearAlgebra
    using ActiveInference
    using Distributions
    
    include("Environments/EpistChainEnv.jl")
    
    export EpistChainEnv, step!, reset_env!

    include("Environments/TMazeEnv.jl")
    include("utils/maths.jl")

    export TMazeEnv, step_TMaze!, reset_TMaze!, initialize_gp
       
    end
end









================================================
FILE: src/ActionModelsExtensions/get_history.jl
================================================
"""
This extends the "get_history" function of the ActionModels package to work specifically with instances of the AIF type.

    get_history(aif::AIF, target_states::Vector{String})
Retrieves a history for multiple states of an AIF agent. 

    get_history(aif::AIF, target_state::String)
Retrieves a single target state history from an AIF agent.

    get_history(aif::AIF)
Retrieves history of all states from an AIF agent.
"""

using ActionModels


# Retrieve multiple states history
function ActionModels.get_history(aif::AIF, target_states::Vector{String})
    history = Dict()

    for target_state in target_states
        try
            history[target_state] = get_history(aif, target_state)
        catch e
            # Catch the error if a specific state does not exist
            if isa(e, ArgumentError)
                throw(ArgumentError("The specified state $target_state does not exist"))
            else
                rethrow(e) 
            end
        end
    end

    return history
end

# Retrieve a history from a single state
function ActionModels.get_history(aif::AIF, target_state::String)
    # Check if the state is in the AIF's states
    if haskey(aif.states, target_state)

        return aif.states[target_state]
    else
        # If the target state is not found, throw an ArgumentError
        throw(ArgumentError("The specified state $target_state does not exist"))
    end
end


# Retrieve all states history
function ActionModels.get_history(aif::AIF)
    return aif.states
end


================================================
FILE: src/ActionModelsExtensions/get_parameters.jl
================================================
"""
This extends the "get_parameters" function of the ActionModels package to work specifically with instances of the AIF type.

    get_parameters(aif::AIF, target_parameters::Vector{String})
Retrieves multiple target parameters from an AIF agent. 

    get_parameters(aif::AIF, target_parameter::String)
Retrieves a single target parameter from an AIF agent.

    get_parameters(aif::AIF)
Retrieves all parameters from an AIF agent.

"""

using ActionModels

# Retrieves multiple target parameters
function ActionModels.get_parameters(aif::AIF, target_parameters::Vector{String})
    parameters = Dict()

    for target_parameter in target_parameters
        try
            parameters[target_parameter] = get_parameters(aif, target_parameter)
        catch e
            if isa(e, ArgumentError)
                throw(ArgumentError("The specified parameter $target_parameter does not exist"))
            else
                rethrow(e)
            end
        end
    end

    return parameters
end

# Retrieves a single parameter
function ActionModels.get_parameters(aif::AIF, target_parameter::String)
    if haskey(aif.parameters, target_parameter)
        return aif.parameters[target_parameter]
    else
        throw(ArgumentError("The specified parameter $target_parameter does not exist"))
    end
end


# Retrieves all parameters 
function ActionModels.get_parameters(aif::AIF)
    return aif.parameters
end


================================================
FILE: src/ActionModelsExtensions/get_states.jl
================================================
"""
This extends the "get_states" function of the ActionModels package to work specifically with instances of the AIF type.

    get_states(aif::AIF, target_states::Vector{String})
Retrieves multiple states from an AIF agent. 

    get_states(aif::AIF, target_state::String)
Retrieves a single target state from an AIF agent.

    get_states(aif::AIF)
Retrieves all states from an AIF agent.
"""

using ActionModels


# Retrieve multiple states
function ActionModels.get_states(aif::AIF, target_states::Vector{String})
    states = Dict()

    for target_state in target_states
        try
            states[target_state] = get_states(aif, target_state)
        catch e
            # Catch the error if a specific state does not exist
            if isa(e, ArgumentError)
                throw(ArgumentError("The specified state $target_state does not exist"))
            else
                rethrow(e) 
            end
        end
    end

    return states
end

# Retrieve a single state
function ActionModels.get_states(aif::AIF, target_state::String)
    if haskey(aif.states, target_state)
        state_history = aif.states[target_state]
        if target_state == "policies"
            return state_history
        else
            # return the latest state or missing
            return isempty(state_history) ? missing : last(state_history)
        end
    else
        throw(ArgumentError("The specified state $target_state does not exist"))
    end
end


# Retrieve all states
function ActionModels.get_states(aif::AIF)
    all_states = Dict()
    for (key, state_history) in aif.states
        if key == "policies"
            all_states[key] = state_history
        else
            # get the latest state or missing
            all_states[key] = isempty(state_history) ? missing : last(state_history)
        end
    end
    return all_states
end






================================================
FILE: src/ActionModelsExtensions/give_inputs.jl
================================================
"""

This is extends the give_inputs! function of ActionsModels.jl to work with instances of the AIF type.

    single_input!(aif::AIF, obs)
Give a single observation to an AIF agent. 


"""

using ActionModels

### Give single observation to the agent
function ActionModels.single_input!(aif::AIF, obs::Vector)

    # Running the action model to retrieve the action distributions
    action_distributions = action_pomdp!(aif, obs)

    # Get number of factors from the action distributions
    num_factors = length(action_distributions)

    # if there is only one factor
    if num_factors == 1
        # Sample action from the action distribution
        action = rand(action_distributions)

        # If the agent has not taken any actions yet
        if isempty(aif.action)
            push!(aif.action, action)
        else
        # Put the action in the last element of the action vector
            aif.action[end] = action
        end

        push!(aif.states["action"], aif.action)

    # if there are multiple factors
    else
        # Initialize a vector for sampled actions 
        sampled_actions = zeros(Real,num_factors)

        # Sample action per factor
        for factor in eachindex(action_distributions)
            sampled_actions[factor] = rand(action_distributions[factor])
        end
        # If the agent has not taken any actions yet
        if isempty(aif.action)
            aif.action = sampled_actions
        else
        # Put the action in the last element of the action vector
            aif.action[end] = sampled_actions
        end
        # Push the action to agent's states
        push!(aif.states["action"], aif.action)
    end

    return aif.action
end

function ActionModels.give_inputs!(aif::AIF, observations::Vector)
    # For each individual observation run single_input! function
    for observation in observations

        ActionModels.single_input!(aif, observation)

    end

    return aif.states["action"]
end


================================================
FILE: src/ActionModelsExtensions/reset.jl
================================================
"""
Resets an AIF type agent to its initial state

    reset!(aif::AIF)

"""

using ActionModels

function ActionModels.reset!(aif::AIF)
    # Reset the agent's state fields to initial conditions
    aif.qs_current = create_matrix_templates([size(aif.B[f], 1) for f in eachindex(aif.B)])
    aif.prior = aif.D
    aif.Q_pi = ones(length(aif.policies)) / length(aif.policies)
    aif.G = zeros(length(aif.policies))
    aif.action = Int[]

    # Clear the history in the states dictionary
    for key in keys(aif.states)

        if key != "policies"
            aif.states[key] = []
        end
    end
    return nothing
end


================================================
FILE: src/ActionModelsExtensions/set_parameters.jl
================================================
"""
This extends the "set_parameters!" function of the ActionModels package to work with instances of the AIF type.

    set_parameters!(aif::AIF, target_param::String, param_value::Real)
Set a single parameter in the AIF agent

    set_parameters!(aif::AIF, parameters::Dict{String, Real})
Set multiple parameters in the AIF agent

"""

using ActionModels

# Setting a single parameter
function ActionModels.set_parameters!(aif::AIF, target_param::String, param_value::Real)
    # Update the parameters dictionary
    aif.parameters[target_param] = param_value

    # Update the struct's field based on the target_param
    if target_param == "alpha"
        aif.alpha = param_value
    elseif target_param == "gamma"
        aif.gamma = param_value
    elseif target_param == "lr_pA"
        aif.lr_pA = param_value
    elseif target_param == "fr_pA"
        aif.fr_pA = param_value
    elseif target_param == "lr_pB"
        aif.lr_pB = param_value
    elseif target_param == "fr_pB"
        aif.fr_pB = param_value
    elseif target_param == "lr_pD"
        aif.lr_pD = param_value
    elseif target_param == "fr_pD"
        aif.fr_pD = param_value
    else
        throw(ArgumentError("The parameter $target_param is not recognized."))
    end
end

# Setting multiple parameters
function ActionModels.set_parameters!(aif::AIF, parameters::Dict)
    # For each parameter in the input dictionary
    for (target_param, param_value) in parameters
        # Directly set each parameter
        set_parameters!(aif, target_param, param_value)
    end
end


================================================
FILE: src/ActionModelsExtensions/set_save_history.jl
================================================
"""
ActionModels - set save history
"""

using ActionModels

function ActionModels.set_save_history!(aif::AIF, save_history::Bool)
    aif.save_history = save_history
end


================================================
FILE: src/Environments/EpistChainEnv.jl
================================================
""" Pre-defined Environment: Epistemic Chaining Grid-World"""

mutable struct EpistChainEnv
    init_loc::Tuple{Int, Int}
    current_loc::Tuple{Int, Int}
    cue1_loc::Tuple{Int, Int}
    cue2::String
    reward_condition::String
    len_y::Int
    len_x::Int

    function EpistChainEnv(starting_loc::Tuple{Int, Int}, cue1_loc::Tuple{Int, Int}, cue2::String, reward_condition::String, grid_locations)
        len_y, len_x = maximum(first.(grid_locations)), maximum(last.(grid_locations))
        new(starting_loc, starting_loc, cue1_loc, cue2, reward_condition, len_y, len_x)
    end
end

function step!(env::EpistChainEnv, action_label::String)
    # Get current location
    y, x = env.current_loc
    next_y, next_x = y, x

    # Update location based on action
    if action_label == "DOWN"
        next_y = y < env.len_y ? y + 1 : y
    elseif action_label == "UP"
        next_y = y > 1 ? y - 1 : y
    elseif action_label == "LEFT"
        next_x = x > 1 ? x - 1 : x
    elseif action_label == "RIGHT"
        next_x = x < env.len_x ? x + 1 : x
    elseif action_label == "STAY"
        # No change in location
    end

    # Set new location
    env.current_loc = (next_y, next_x)

    # Observations
    loc_obs = env.current_loc
    cue2_names = ["Null", "reward_on_top", "reward_on_bottom"]
    cue2_loc_names = ["L1","L2","L3","L4"]
    cue2_locs = [(1, 3), (2, 4), (4, 4), (5, 3)]

    # Map cue2 location names to indices
    cue2_loc_idx = Dict(cue2_loc_names[1] => 1, cue2_loc_names[2] => 2, cue2_loc_names[3] => 3, cue2_loc_names[4] => 4)

    # Get cue2 location
    cue2_loc = cue2_locs[cue2_loc_idx[env.cue2]]

    # Determine cue1 observation
    if env.current_loc == env.cue1_loc
        cue1_obs = env.cue2
    else
        cue1_obs = "Null"
    end

    # Reward conditions and locations
    reward_conditions = ["TOP", "BOTTOM"]
    reward_locations = [(2,6), (4,6)]
    rew_cond_idx = Dict(reward_conditions[1] => 1, reward_conditions[2] => 2)

    # Determine cue2 observation
    if env.current_loc == cue2_loc
        cue2_obs = cue2_names[rew_cond_idx[env.reward_condition] + 1]
    else
        cue2_obs = "Null"
    end

    # Determine reward observation
    if env.current_loc == reward_locations[1]
        if env.reward_condition == "TOP"
            reward_obs = "Cheese"
        else
            reward_obs = "Shock"
        end
    elseif env.current_loc == reward_locations[2]
        if env.reward_condition == "BOTTOM"
            reward_obs = "Cheese"
        else
            reward_obs = "Shock"
        end
    else
        reward_obs = "Null"
    end

    # Return observations
    return loc_obs, cue1_obs, cue2_obs, reward_obs
end

function reset_env!(env::EpistChainEnv)
    # Reset environment to initial location
    env.current_loc = env.init_loc
    println("Re-initialized location to $(env.init_loc)")
    return env.current_loc
end




================================================
FILE: src/Environments/TMazeEnv.jl
================================================
mutable struct TMazeEnv
    reward_prob::Float64
    reward_idx::Int64
    loss_idx::Int64
    location_factor_id::Int64
    trial_factor_id::Int64
    location_modality_id::Int64
    reward_modality_id::Int64
    cue_modality_id::Int64
    num_states::Vector{Int64}
    num_locations::Int64
    num_controls::Vector{Int64}
    num_reward_conditions::Int64
    num_cues::Int64
    num_obs::Vector{Int64}
    num_factors::Int64
    num_modalities::Int64
    reward_probs::Vector{Float64}
    transition_dist::Array{Any, 1}
    likelihood_dist::Array{Any, 1}
    _state::Array{Any, 1}
    _reward_condition_idx::Int64
    reward_condition::Vector{Int64}
    state::Array{Any, 1}

    function TMazeEnv(reward_prob::Float64;

        reward_idx::Int64 = 2,
        loss_idx::Int64 = 3,
        location_factor_id::Int64 = 1,
        trial_factor_id::Int64 = 2,
        location_modality_id::Int64 = 1,
        reward_modality_id::Int64 = 2,
        cue_modality_id::Int64 = 3,
        )
        num_states = [4, 2]
        num_locations = num_states[location_factor_id]
        num_controls = [num_locations, 1]
        num_reward_conditions = num_states[trial_factor_id]
        num_cues = num_reward_conditions
        num_obs = [num_locations, num_reward_conditions + 1, num_cues]
        num_factors = length(num_states)
        num_modalities = length(num_obs)

        reward_probs = [reward_prob, round(1-reward_prob, digits = 6)]

        new(reward_prob, reward_idx, loss_idx, location_factor_id, trial_factor_id, 
        location_modality_id, reward_modality_id, cue_modality_id, num_states, num_locations, num_controls, num_reward_conditions, num_cues, num_obs, num_factors, 
        num_modalities, reward_probs)
    end
end

function initialize_gp(env::TMazeEnv)
    env.transition_dist = construct_transition_dist(env)
    env.likelihood_dist = construct_likelihood_dist(env)
end

function step_TMaze!(env::TMazeEnv, actions)
    prob_states = Array{Any}(undef, env.num_factors)

    # Calculate the state probabilities based on actions and current state
    for factor = 1:env.num_factors
        transition_matrix = env.transition_dist[factor][:, :, Int(actions[factor])]
        current_state_vector = env._state[factor]
        prob_states[factor] = transition_matrix * current_state_vector
    end

    # Sample the next state from the probability distributions
    state = [sample_dist(ps_i) for ps_i in prob_states]

    # Construct the new state
    env._state = construct_state(env, state) 

    # Generate and return the current observation
    return get_observation(env)
end

function reset_TMaze!(env::TMazeEnv; state=nothing)
    if state === nothing
        # Initialize location state
        loc_state = onehot(1, env.num_locations)

        # Randomly select a reward condition
        env._reward_condition_idx = rand(1:env.num_reward_conditions)
        env.reward_condition = onehot(env._reward_condition_idx, env.num_reward_conditions)

        # Initialize the full state array
        full_state = Vector{Any}(undef, env.num_factors)
        full_state[env.location_factor_id] = loc_state
        full_state[env.trial_factor_id] = env.reward_condition

        env._state = full_state
    else
        env._state = state
    end

    # Return the current observation
    return get_observation(env)
end

function construct_transition_dist(env::TMazeEnv)

    B_locs = reshape(Matrix{Float64}(I, env.num_locations, env.num_locations), env.num_locations, env.num_locations, 1)
    B_locs = repeat(B_locs, 1, 1, env.num_locations) 
    B_locs = permutedims(B_locs, [1, 3, 2])
    
    B_trials = reshape(Matrix{Float64}(I, env.num_reward_conditions, env.num_reward_conditions), env.num_reward_conditions, env.num_reward_conditions, 1)

    B = Array{Any}(undef, env.num_factors)
    B[env.location_factor_id] = B_locs
    B[env.trial_factor_id] = B_trials

    return B
end

function construct_likelihood_dist(env::TMazeEnv)

    A_dims = [[obs_dim; env.num_states...] for obs_dim in env.num_obs]
    A = array_of_any_zeros(A_dims)

    for loc in 1:env.num_states[env.location_factor_id]
        for reward_condition in 1:env.num_states[env.trial_factor_id]

            if loc == 1 # When in the center location
                A[env.reward_modality_id][1, loc, reward_condition] = 1.0

                A[env.cue_modality_id][:, loc, reward_condition] .= 1.0 / env.num_obs[env.cue_modality_id]

            elseif loc == 4  # When in the cue location
                A[env.reward_modality_id][1, loc, reward_condition] = 1.0

                A[env.cue_modality_id][reward_condition, loc, reward_condition] = 1.0

            else  # In one of the (potentially) rewarding arms
                if loc == (reward_condition + 1)
                    high_prob_idx = env.reward_idx
                    low_prob_idx = env.loss_idx
                else
                    high_prob_idx = env.loss_idx
                    low_prob_idx = env.reward_idx
                end

                # Assign probabilities based on the reward condition
                A[env.reward_modality_id][high_prob_idx, loc, reward_condition] = env.reward_probs[1]
                A[env.reward_modality_id][low_prob_idx, loc, reward_condition] = env.reward_probs[2]

                # Cue is ambiguous in the reward location
                A[env.cue_modality_id][:, loc, reward_condition] .= 1.0 / env.num_obs[env.cue_modality_id]
            end

            # Location is always observed correctly
            A[env.location_modality_id][loc, loc, reward_condition] = 1.0
        end
    end

    return A
end


function sample_dist(probabilities)

    probabilities = convert(Vector{Float64}, probabilities)

    # Ensure probabilities sum to 1
    probabilities /= sum(probabilities)

    # Julia's Categorical returns a 1-based index
    sample_onehot = rand(Multinomial(1, probabilities))
    return findfirst(sample_onehot .== 1)
end

function get_observation(env::TMazeEnv)

    # Calculate the probability of observations based on the current state and the likelihood distribution
    prob_obs = [dot_product(A_m, env._state) for A_m in env.likelihood_dist]

    # Sample from the probability distributions to get actual observations
    obs = [sample_dist(po_i) for po_i in prob_obs]

    return obs
end

function construct_state(env::TMazeEnv, state_tuple)
    # Create an array of any
    state = Vector{Any}(undef, env.num_factors)

    # Populate the state array with one-hot encoded vectors
    for (f, ns) in enumerate(env.num_states)
        state[f] = onehot(state_tuple[f], ns)
    end

    return state
end


================================================
FILE: src/pomdp/inference.jl
================================================
""" -------- Inference Functions -------- """

#### State Inference #### 

""" Get Expected States """
function get_expected_states(qs::Vector{Vector{T}} where T <: Real, B, policy::Matrix{Int64})
    n_steps, n_factors = size(policy)

    # initializing posterior predictive density as a list of beliefs over time
    qs_pi = [deepcopy(qs) for _ in 1:n_steps+1]

    # expected states over time
    for t in 1:n_steps
        for control_factor in 1:n_factors
            action = policy[t, control_factor]
            
            qs_pi[t+1][control_factor] = B[control_factor][:, :, action] * qs_pi[t][control_factor]
        end
    end

    return qs_pi[2:end]
end

""" 
    Multiple dispatch for all expected states given all policies

Multiple dispatch for getting expected states for all policies based on the agents currently
inferred states and the transition matrices for each factor and action in the policy.

qs::Vector{Vector{Real}} \n
B: Vector{Array{<:Real}} \n
policy: Vector{Matrix{Int64}}

"""
function get_expected_states(qs::Vector{Vector{Float64}}, B, policy::Vector{Matrix{Int64}})
    
    # Extracting the number of steps (policy_length) and factors from the first policy
    n_steps, n_factors = size(policy[1])

    # Number of policies
    n_policies = length(policy)
    
    # Preparing vessel for the expected states for all policies. Has number of undefined entries equal to the
    # number of policies
    qs_pi_all = Vector{Any}(undef, n_policies)

    # Looping through all policies
    for (policy_idx, policy_x) in enumerate(policy)

        # initializing posterior predictive density as a list of beliefs over time
        qs_pi = [deepcopy(qs) for _ in 1:n_steps+1]

        # expected states over time
        for t in 1:n_steps
            for control_factor in 1:n_factors
                action = policy_x[t, control_factor]
                
                qs_pi[t+1][control_factor] = B[control_factor][:, :, action] * qs_pi[t][control_factor]
            end
        end
        qs_pi_all[policy_idx] = qs_pi[2:end]
    end
    return qs_pi_all
end

"""
    process_observation(observation::Int, n_modalities::Int, n_observations::Vector{Int})

Process a single modality observation. Returns a one-hot encoded vector. 

# Arguments
- `observation::Int`: The index of the observed state with a single observation modality.
- `n_modalities::Int`: The number of observation modalities in the observation. 
- `n_observations::Vector{Int}`: A vector containing the number of observations for each modality.

# Returns
- `Vector{Vector{Real}}`: A vector containing a single one-hot encoded observation.
"""
function process_observation(observation::Int, n_modalities::Int, n_observations::Vector{Int})

    # Check if there is only one modality
    if n_modalities == 1
        # Create a one-hot encoded vector for the observation
        processed_observation = onehot(observation, n_observations[1]) 
    end

    # Return the processed observation wrapped in a vector
    return [processed_observation]
end

"""
    process_observation(observation::Union{Array{Int}, Tuple{Vararg{Int}}}, n_modalities::Int, n_observations::Vector{Int})

Process observation with multiple modalities and return them in a one-hot encoded format 

# Arguments
- `observation::Union{Array{Int}, Tuple{Vararg{Int}}}`: A collection of indices of the observed states for each modality.
- `n_modalities::Int`: The number of observation modalities in the observation. 
- `n_observations::Vector{Int}`: A vector containing the number of observations for each modality.

# Returns
- `Vector{Vector{Real}}`: A vector containing one-hot encoded vectors for each modality.
"""
function process_observation(observation::Union{Array{Int}, Tuple{Vararg{Int}}}, n_modalities::Int, n_observations::Vector{Int})

    # Initialize the processed_observation vector
    processed_observation = Vector{Vector{Float64}}(undef, n_modalities)

    # Check if the length of observation matches the number of modalities
    if length(observation) == n_modalities
        for (modality, modality_observation) in enumerate(observation)
            # Create a one-hot encoded vector for the current modality observation
            one_hot = onehot(modality_observation, n_observations[modality])
            # Add the one-hot vector to the processed_observation vector
            processed_observation[modality] = one_hot
        end
    end

    return processed_observation
end

""" Update Posterior States """
function update_posterior_states(
    A::Vector{Array{T,N}} where {T <: Real, N}, 
    obs::Vector{Int64}; 
    prior::Union{Nothing, Vector{Vector{T}}} where T <: Real = nothing, 
    num_iter::Int=num_iter, dF_tol::Float64=dF_tol, kwargs...)
    num_obs, num_states, num_modalities, num_factors = get_model_dimensions(A)

    obs_processed = process_observation(obs, num_modalities, num_obs)
    return fixed_point_iteration(A, obs_processed, num_obs, num_states, prior=prior, num_iter=num_iter, dF_tol = dF_tol)
end


""" Run State Inference via Fixed-Point Iteration """
function fixed_point_iteration(
    A::Vector{Array{T,N}} where {T <: Real, N}, obs::Vector{Vector{Float64}}, num_obs::Vector{Int64}, num_states::Vector{Int64};
    prior::Union{Nothing, Vector{Vector{T}}} where T <: Real = nothing, 
    num_iter::Int=num_iter, dF::Float64=1.0, dF_tol::Float64=dF_tol
)
    # Get model dimensions (NOTE Sam: We need to save model dimensions in the AIF struct in the future)
    n_modalities = length(num_obs)
    n_factors = length(num_states)

    # Get joint likelihood
    likelihood = get_joint_likelihood(A, obs, num_states)
    likelihood = capped_log(likelihood)

    # Initialize posterior and prior
    qs = Vector{Vector{Float64}}(undef, n_factors)
    for factor in 1:n_factors
        qs[factor] = ones(num_states[factor]) / num_states[factor]
    end

    # If no prior is provided, create a default prior with uniform distribution
    if prior === nothing
        prior = create_matrix_templates(num_states)
    end
    
    # Create a copy of the prior to avoid modifying the original
    prior = deepcopy(prior)
    prior = capped_log_array(prior) 

    # Initialize free energy
    prev_vfe = calc_free_energy(qs, prior, n_factors)

    # Single factor condition
    if n_factors == 1
        qL = dot_product(likelihood, qs[1])  
        return [softmax(qL .+ prior[1], dims=1)]

    # If there are more factors
    else
        ### Fixed-Point Iteration ###
        curr_iter = 0
        ### Sam NOTE: We need check if ReverseDiff might potantially have issues with this while loop ###
        while curr_iter < num_iter && dF >= dF_tol
            qs_all = qs[1]
            # Loop over each factor starting from the second one
            for factor in 2:n_factors
                # Reshape and multiply qs_all with the current factor's qs
                qs_all = qs_all .* reshape(qs[factor], tuple(ones(Real, factor - 1)..., :, 1))
            end

            # Compute the log-likelihood
            LL_tensor = likelihood .* qs_all

            # Update each factor's qs
            for factor in 1:n_factors
                # Initialize qL for the current factor
                qL = zeros(Real, size(qs[factor]))

                # Compute qL for each state in the current factor
                for i in 1:size(qs[factor], 1)
                    qL[i] = sum([LL_tensor[indices...] / qs[factor][i] for indices in Iterators.product([1:size(LL_tensor, dim) for dim in 1:n_factors]...) if indices[factor] == i])
                end

                # If qs is tracked by ReverseDiff, get the value
                if ReverseDiff.istracked(softmax(qL .+ prior[factor], dims=1))
                    qs[factor] = ReverseDiff.value(softmax(qL .+ prior[factor], dims=1))
                else
                    # Otherwise, proceed as normal
                    qs[factor] = softmax(qL .+ prior[factor], dims=1)
                end
            end

            # Recompute free energy
            vfe = calc_free_energy(qs, prior, n_factors, likelihood)

            # Update stopping condition
            dF = abs(prev_vfe - vfe)
            prev_vfe = vfe

            # Increment iteration
            curr_iter += 1
        end

        return qs
    end
end



""" Calculate Accuracy Term """
function compute_accuracy(log_likelihood, qs::Vector{Vector{T}} where T <: Real)
    n_factors = length(qs)
    ndims_ll = ndims(log_likelihood)
    dims = (ndims_ll - n_factors + 1) : ndims_ll

    # Calculate the accuracy term
    accuracy = sum(
        log_likelihood[indices...] * prod(qs[factor][indices[dims[factor]]] for factor in 1:n_factors)
        for indices in Iterators.product((1:size(log_likelihood, i) for i in 1:ndims_ll)...)
    )

    return accuracy
end


""" Calculate Free Energy """
function calc_free_energy(qs::Vector{Vector{T}} where T <: Real, prior, n_factors, likelihood=nothing)
    # Initialize free energy
    free_energy = 0.0
    
    # Calculate free energy for each factor
    for factor in 1:n_factors
        # Neg-entropy of posterior marginal
        negH_qs = dot(qs[factor], log.(qs[factor] .+ 1e-16))
        # Cross entropy of posterior marginal with prior marginal
        xH_qp = -dot(qs[factor], prior[factor])
        # Add to total free energy
        free_energy += negH_qs + xH_qp
    end
    
    # Subtract accuracy
    if likelihood !== nothing
        free_energy -= compute_accuracy(likelihood, qs)
    end
    
    return free_energy
end

#### Policy Inference #### 
""" Update Posterior over Policies """
function update_posterior_policies(
    qs::Vector{Vector{T}} where T <: Real,
    A::Vector{Array{T, N}} where {T <: Real, N},
    B::Vector{Array{T, N}} where {T <: Real, N},
    C::Vector{Array{T}} where T <: Real,
    policies::Vector{Matrix{Int64}},
    use_utility::Bool=true,
    use_states_info_gain::Bool=true,
    use_param_info_gain::Bool=false,
    pA = nothing,
    pB = nothing,
    E::Vector{T} where T <: Real = nothing,
    gamma::Real=16.0
)
    n_policies = length(policies)
    G = zeros(n_policies)
    q_pi = Vector{Float64}(undef, n_policies)
    qs_pi = Vector{Float64}[]
    qo_pi = Vector{Float64}[]
    lnE = capped_log(E)

    for (idx, policy) in enumerate(policies)
        qs_pi = get_expected_states(qs, B, policy)
        qo_pi = get_expected_obs(qs_pi, A)

        # Calculate expected utility
        if use_utility
            # If ReverseDiff is tracking the expected utility, get the value
            if ReverseDiff.istracked(calc_expected_utility(qo_pi, C))
                G[idx] += ReverseDiff.value(calc_expected_utility(qo_pi, C))

            # Otherwise calculate the expected utility and add it to the G vector
            else
                G[idx] += calc_expected_utility(qo_pi, C)
            end
        end

        # Calculate expected information gain of states
        if use_states_info_gain
            # If ReverseDiff is tracking the information gain, get the value
            if ReverseDiff.istracked(calc_states_info_gain(A, qs_pi))
                G[idx] += ReverseDiff.value(calc_states_info_gain(A, qs_pi))

            # Otherwise calculate it and add it to the G vector
            else
                G[idx] += calc_states_info_gain(A, qs_pi)
            end
        end

        # Calculate expected information gain of parameters (learning)
        if use_param_info_gain
            if pA !== nothing

                # if ReverseDiff is tracking pA information gain, get the value
                if ReverseDiff.istracked(calc_pA_info_gain(pA, qo_pi, qs_pi))
                    G[idx] += ReverseDiff.value(calc_pA_info_gain(pA, qo_pi, qs_pi))
                # Otherwise calculate it and add it to the G vector
                else
                    G[idx] += calc_pA_info_gain(pA, qo_pi, qs_pi)
                end
            end

            if pB !== nothing
                G[idx] += calc_pB_info_gain(pB, qs_pi, qs, policy)
            end
        end

    end

    
    q_pi = softmax(G * gamma + lnE, dims=1)

    return q_pi, G
end

""" Get Expected Observations """
function get_expected_obs(qs_pi, A::Vector{Array{T,N}} where {T <: Real, N})
    n_steps = length(qs_pi)
    qo_pi = []

    for t in 1:n_steps
        qo_pi_t = Vector{Any}(undef, length(A))
        qo_pi = push!(qo_pi, qo_pi_t)
    end

    for t in 1:n_steps
        for (modality, A_m) in enumerate(A)
            qo_pi[t][modality] = dot_product(A_m, qs_pi[t])
        end
    end

    return qo_pi
end

""" Calculate Expected Utility """
function calc_expected_utility(qo_pi, C)
    n_steps = length(qo_pi)
    expected_utility = 0.0
    num_modalities = length(C)

    modalities_to_tile = [modality_i for modality_i in 1:num_modalities if ndims(C[modality_i]) == 1]

    C_tiled = deepcopy(C)
    for modality in modalities_to_tile
        modality_data = reshape(C_tiled[modality], :, 1)
        C_tiled[modality] = repeat(modality_data, 1, n_steps)
    end
    
    C_prob = softmax_array(C_tiled)
    lnC =[]
    for t in 1:n_steps
        for modality in 1:num_modalities
            lnC = capped_log(C_prob[modality][:, t])
            expected_utility += dot(qo_pi[t][modality], lnC) 
        end
    end

    return expected_utility
end

""" Calculate States Information Gain """
function calc_states_info_gain(A, qs_pi)
    n_steps = length(qs_pi)
    states_surprise = 0.0

    for t in 1:n_steps
        states_surprise += calculate_bayesian_surprise(A, qs_pi[t])
    end

    return states_surprise
end

""" Calculate observation to state info Gain """
function calc_pA_info_gain(pA, qo_pi, qs_pi)

    n_steps = length(qo_pi)
    num_modalities = length(pA)

    wA = Vector{Any}(undef, num_modalities)
    for (modality, pA_m) in enumerate(pA)
        wA[modality] = spm_wnorm(pA[modality])
    end

    pA_info_gain = 0

    for modality in 1:num_modalities
        wA_modality = wA[modality] .* (pA[modality] .> 0)

        for t in 1:n_steps
            pA_info_gain -= dot(qo_pi[t][modality], dot_product(wA_modality, qs_pi[t]))
        end
    end
    return pA_info_gain
end

""" Calculate state to state info Gain """
function calc_pB_info_gain(pB, qs_pi, qs_prev, policy)
    n_steps = length(qs_pi)
    num_factors = length(pB)

    wB = Vector{Any}(undef, num_factors)
    for (factor, pB_f) in enumerate(pB)
        wB[factor] = spm_wnorm(pB_f)
    end

    pB_info_gain = 0

    for t in 1:n_steps
        if t == 1
            previous_qs = qs_prev
        else
            previous_qs = qs_pi[t-1]
        end

        policy_t = policy[t, :]

        for (factor, a_i) in enumerate(policy_t)
            wB_factor_t = wB[factor][:,:,Int(a_i)] .* (pB[factor][:,:,Int(a_i)] .> 0)
            pB_info_gain -= dot(qs_pi[t][factor], wB_factor_t * previous_qs[factor])
        end
    end
    return pB_info_gain
end

### Action Sampling ###
""" Sample Action [Stochastic or Deterministic] """
function sample_action(q_pi, policies::Vector{Matrix{Int64}}, num_controls; action_selection="stochastic", alpha=16.0)
    num_factors = length(num_controls)
    selected_policy = zeros(Real,num_factors)
    
    eltype_q_pi = eltype(q_pi)

    # Initialize action_marginals with the correct element type
    action_marginals = create_matrix_templates(num_controls, "zeros", eltype_q_pi)

    for (pol_idx, policy) in enumerate(policies)
        for (factor_i, action_i) in enumerate(policy[1,:])
            action_marginals[factor_i][action_i] += q_pi[pol_idx]
        end
    end

    action_marginals = normalize_arrays(action_marginals)

    for factor_i in 1:num_factors
        if action_selection == "deterministic"
            selected_policy[factor_i] = select_highest(action_marginals[factor_i])
        elseif action_selection == "stochastic"
            log_marginal_f = capped_log(action_marginals[factor_i])
            p_actions = softmax(log_marginal_f * alpha, dims=1)
            selected_policy[factor_i] = action_select(p_actions)
        end
    end
    return selected_policy
end

""" Edited Compute Accuracy [Still needs to be nested within Fixed-Point Iteration] """
function compute_accuracy_new(log_likelihood, qs::Vector{Vector{Real}})
    n_factors = length(qs)
    ndims_ll = ndims(log_likelihood)
    dims = (ndims_ll - n_factors + 1) : ndims_ll

    result_size = size(log_likelihood, 1) 
    results = zeros(Real,result_size)

    for indices in Iterators.product((1:size(log_likelihood, i) for i in 1:ndims_ll)...)
        product = log_likelihood[indices...] * prod(qs[factor][indices[dims[factor]]] for factor in 1:n_factors)
        results[indices[1]] += product
    end

    return results
end

""" Calculate State-Action Prediction Error """
function calculate_SAPE(aif::AIF)

    qs_pi_all = get_expected_states(aif.qs_current, aif.B, aif.policies)
    qs_bma = bayesian_model_average(qs_pi_all, aif.Q_pi)

    if length(aif.states["bayesian_model_averages"]) != 0
        sape = kl_divergence(qs_bma, aif.states["bayesian_model_averages"][end])
        push!(aif.states["SAPE"], sape)
    end

    push!(aif.states["bayesian_model_averages"], qs_bma)
end



================================================
FILE: src/pomdp/learning.jl
================================================
""" Update obs likelihood matrix """
function update_obs_likelihood_dirichlet(pA, A, obs, qs; lr = 1.0, fr = 1.0, modalities = "all")

    # If reverse diff is tracking the learning rate, get the value
    if ReverseDiff.istracked(lr)
        lr = ReverseDiff.value(lr)
    end
    # If reverse diff is tracking the forgetting rate, get the value
    if ReverseDiff.istracked(fr)
        fr = ReverseDiff.value(fr)
    end

    # Extracting the number of modalities and observations from the dirichlet: pA
    num_modalities = length(pA)
    num_observations = [size(pA[modality + 1], 1) for modality in 0:(num_modalities - 1)]

    obs = process_observation(obs, num_modalities, num_observations)

    if modalities === "all"
        modalities = collect(1:num_modalities)
    end

    qA = deepcopy(pA)

    # Important! Takes first the cross product of the qs itself, so that it matches dimensions with the A and pA matrices
    qs_cross = outer_product(qs)

    for modality in modalities
        dfda = outer_product(obs[modality], qs_cross)
        dfda = dfda .* (A[modality] .> 0)
        qA[modality] = (fr * qA[modality]) + (lr * dfda)
    end

    return qA
end

""" Update state likelihood matrix """
function update_state_likelihood_dirichlet(pB, B, actions, qs::Vector{Vector{T}} where T <: Real, qs_prev; lr = 1.0, fr = 1.0, factors = "all")

    if ReverseDiff.istracked(lr)
        lr = ReverseDiff.value(lr)
    end
    if ReverseDiff.istracked(fr)
        fr = ReverseDiff.value(fr)
    end

    num_factors = length(pB)

    qB = deepcopy(pB)

    if factors === "all"
        factors = collect(1:num_factors)
    end

    for factor in factors
        dfdb = outer_product(qs[factor], qs_prev[factor])
        dfdb .*= (B[factor][:,:,Int(actions[factor])] .> 0)
        qB[factor][:,:,Int(actions[factor])] = qB[factor][:,:,Int(actions[factor])]*fr .+ (lr .* dfdb)
    end

    return qB
end

""" Update prior D matrix """
function update_state_prior_dirichlet(pD, qs::Vector{Vector{T}} where T <: Real; lr = 1.0, fr = 1.0, factors = "all")

    num_factors = length(pD)

    qD = deepcopy(pD)

    if factors == "all"
        factors = collect(1:num_factors)
    end

    for factor in factors
        idx = pD[factor] .> 0
        qD[factor][idx] = (fr * qD[factor][idx]) .+ (lr * qs[factor][idx])
    end  
    
    return qD
end


================================================
FILE: src/pomdp/POMDP.jl
================================================
"""
    action_pomdp!(agent, obs)
This function wraps the POMDP action-perception loop used for simulating and fitting the data.

Arguments:
- `agent::Agent`: An instance of ActionModels `Agent` type, which contains AIF type object as a substruct.
- `obs::Vector{Int64}`: A vector of observations, where each observation is an integer.
- `obs::Tuple{Vararg{Int}}`: A tuple of observations, where each observation is an integer.
- `obs::Int64`: A single observation, which is an integer.
- `aif::AIF`: An instance of the `AIF` type, which contains the agent's state, parameters, and substructures.

Outputs:
- Returns a `Distributions.Categorical` distribution or a vector of distributions, representing the probability distributions for actions per each state factor.
"""

### Action Model:  Returns probability distributions for actions per factor

function action_pomdp!(agent::Agent, obs::Vector{Int64})

    ### Get parameters 
    alpha = agent.substruct.parameters["alpha"]
    n_factors = length(agent.substruct.settings["num_controls"])

    # Initialize empty arrays for action distribution per factor
    action_p = Vector{Any}(undef, n_factors)
    action_distribution = Vector{Distributions.Categorical}(undef, n_factors)

    #If there was a previous action
    if !ismissing(agent.states["action"])

        #Extract it
        previous_action = agent.states["action"]

        # If it is not a vector, make it one
        if !(previous_action isa Vector)
            previous_action = previous_action isa Integer ? [previous_action] : collect(previous_action)
        end
        #Store the action in the AIF substruct
        agent.substruct.action = previous_action
    end

    ### Infer states & policies

    # Run state inference 
    infer_states!(agent.substruct, obs)

    # If action is empty, update D vectors
    if ismissing(agent.states["action"]) && agent.substruct.pD !== nothing
        update_D!(agent.substruct)
    end

    # If learning of the B matrix is enabled and agent has a previous action
    if !ismissing(agent.states["action"]) && agent.substruct.pB !== nothing

        # Update Transition Matrix
        update_B!(agent.substruct)
    end

    # If learning of the A matrix is enabled
    if agent.substruct.pA !== nothing
        update_A!(agent.substruct)
    end

    # Run policy inference 
    infer_policies!(agent.substruct)


    ### Retrieve log marginal probabilities of actions
    log_action_marginals = get_log_action_marginals(agent.substruct)

    ### Pass action marginals through softmax function to get action probabilities
    for factor in 1:n_factors
        action_p[factor] = softmax(log_action_marginals[factor] * alpha, dims=1)
        action_distribution[factor] = Distributions.Categorical(action_p[factor])
    end

    return n_factors == 1 ? action_distribution[1] : action_distribution
end

### Action Model where the observation is a tuple

function action_pomdp!(agent::Agent, obs::Tuple{Vararg{Int}})

    # convert observation to vector
    obs = collect(obs)

    ### Get parameters 
    alpha = agent.substruct.parameters["alpha"]
    n_factors = length(agent.substruct.settings["num_controls"])

    # Initialize empty arrays for action distribution per factor
    action_p = Vector{Any}(undef, n_factors)
    action_distribution = Vector{Distributions.Categorical}(undef, n_factors)

     #If there was a previous action
     if !ismissing(agent.states["action"])

        #Extract it
        previous_action = agent.states["action"]

        # If it is not a vector, make it one
        if !(previous_action isa Vector)
            previous_action = collect(previous_action)
        end

        #Store the action in the AIF substruct
        agent.substruct.action = previous_action
    end

    ### Infer states & policies

    # Run state inference 
    infer_states!(agent.substruct, obs)

    # If action is empty and pD is not nothing, update D vectors
    if ismissing(agent.states["action"]) && agent.substruct.pD !== nothing
        qs_t1 = get_history(agent.substruct)["posterior_states"][1]
        update_D!(agent.substruct, qs_t1)
    end

    # If learning of the B matrix is enabled and agent has a previous action
    if !ismissing(agent.states["action"]) && agent.substruct.pB !== nothing

        # Get the posterior over states from the previous time step
        states_posterior = get_history(agent.substruct)["posterior_states"][end-1]

        # Update Transition Matrix
        update_B!(agent.substruct, states_posterior)
    end

    # If learning of the A matrix is enabled
    if agent.substruct.pA !== nothing
        update_A!(agent.substruct, obs)
    end

    # Run policy inference 
    infer_policies!(agent.substruct)


    ### Retrieve log marginal probabilities of actions
    log_action_marginals = get_log_action_marginals(agent.substruct)
    
    ### Pass action marginals through softmax function to get action probabilities
    for factor in 1:n_factors
        action_p[factor] = softmax(log_action_marginals[factor] * alpha, dims=1)
        action_distribution[factor] = Distributions.Categorical(action_p[factor])
    end

    return n_factors == 1 ? action_distribution[1] : action_distribution
end

function action_pomdp!(aif::AIF, obs::Vector{Int64})

    ### Get parameters 
    alpha = aif.parameters["alpha"]
    n_factors = length(aif.settings["num_controls"])

    # Initialize empty arrays for action distribution per factor
    action_p = Vector{Any}(undef, n_factors)
    action_distribution = Vector{Distributions.Categorical}(undef, n_factors)

    ### Infer states & policies

    # Run state inference 
    infer_states!(aif, obs)

    # If action is empty, update D vectors
    if ismissing(get_states(aif)["action"]) && aif.pD !== nothing
        qs_t1 = get_history(aif)["posterior_states"][1]
        update_D!(aif, qs_t1)
    end

    # If learning of the B matrix is enabled and agent has a previous action
    if !ismissing(get_states(aif)["action"]) && aif.pB !== nothing

        # Get the posterior over states from the previous time step
        states_posterior = get_history(aif)["posterior_states"][end-1]

        # Update Transition Matrix
        update_B!(aif, states_posterior)
    end

    # If learning of the A matrix is enabled
    if aif.pA !== nothing
        update_A!(aif, obs)
    end

    # Run policy inference 
    infer_policies!(aif)


    ### Retrieve log marginal probabilities of actions
    log_action_marginals = get_log_action_marginals(aif)
    
    ### Pass action marginals through softmax function to get action probabilities
    for factor in 1:n_factors
        action_p[factor] = softmax(log_action_marginals[factor] * alpha, dims=1)
        action_distribution[factor] = Distributions.Categorical(action_p[factor])
    end

    return n_factors == 1 ? action_distribution[1] : action_distribution
end

function action_pomdp!(agent::Agent, obs::Int64)
    action_pomdp!(agent::Agent, [obs])
end


================================================
FILE: src/pomdp/struct.jl
================================================
""" -------- AIF Mutable Struct -------- """

mutable struct AIF
    A::Vector{Array{T, N}} where {T <: Real, N} # A-matrix
    B::Vector{Array{T, N}} where {T <: Real, N} # B-matrix
    C::Vector{Array{Real}} # C-vectors
    D::Vector{Vector{Real}} # D-vectors
    E::Vector{T} where T <: Real       # E-vector (Habits)
    pA::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N} # Dirichlet priors for A-matrix
    pB::Union{Vector{Array{T, N}}, Nothing} where {T <: Real, N} # Dirichlet priors for B-matrix
    pD::Union{Vector{Array{Real}}, Nothing} # Dirichlet priors for D-vector
    lr_pA::Real # pA Learning Parameter
    fr_pA::Real # pA Forgetting Parameter,  1.0 for no forgetting
    lr_pB::Real # pB learning Parameter
    fr_pB::Real # pB Forgetting Parameter
    lr_pD::Real # pD Learning parameter
    fr_pD::Real # PD Forgetting parameter
    modalities_to_learn::Union{String, Vector{Int64}} # Modalities can be eithe "all" or "# modality"
    factors_to_learn::Union{String, Vector{Int64}} # Modalities can be either "all" or "# factor"
    gamma::Real # Gamma parameter
    alpha::Real # Alpha parameter
    policies::Vector{Matrix{Int64}} # Inferred from the B matrix
    num_controls::Array{Int,1} # Number of actions per factor
    control_fac_idx::Array{Int,1} # Indices of controllable factors
    policy_len::Int  # Policy length
    qs_current::Vector{Vector{T}} where T <: Real # Current beliefs about states
    obs_current::Vector{T} where T <: Real # Current observation
    prior::Vector{Vector{T}} where T <: Real # Prior beliefs about states
    Q_pi::Vector{T} where T <:Real # Posterior beliefs over policies
    G::Vector{T} where T <:Real # Expected free energy of policies
    action::Vector{Int} # Last action
    use_utility::Bool # Utility Boolean Flag
    use_states_info_gain::Bool # States Information Gain Boolean Flag
    use_param_info_gain::Bool # Include the novelty value in the learning parameters
    action_selection::String # Action selection: can be either "deterministic" or "stochastic"
    FPI_num_iter::Int # Number of iterations stopping condition in the FPI algorithm
    FPI_dF_tol::Float64 # Free energy difference stopping condition in the FPI algorithm
    states::Dict{String,Array{Any,1}} # States Dictionary
    parameters::Dict{String,Real} # Parameters Dictionary
    settings::Dict{String,Any} # Settings Dictionary
    save_history::Bool # Save history boolean flag
end

# Create ActiveInference Agent 
function create_aif(A, B;
                    C = nothing,
                    D = nothing,
                    E = nothing,
                    pA = nothing, 
                    pB = nothing, 
                    pD = nothing, 
                    lr_pA = 1.0, 
                    fr_pA = 1.0, 
                    lr_pB = 1.0, 
                    fr_pB = 1.0, 
                    lr_pD = 1.0, 
                    fr_pD = 1.0, 
                    modalities_to_learn = "all", 
                    factors_to_learn = "all", 
                    gamma=1.0, 
                    alpha=1.0, 
                    policy_len=1, 
                    num_controls=nothing, 
                    control_fac_idx=nothing, 
                    use_utility=true, 
                    use_states_info_gain=true, 
                    use_param_info_gain = false, 
                    action_selection="stochastic",
                    FPI_num_iter=10,
                    FPI_dF_tol=0.001,
                    save_history=true
    )

    num_states = [size(B[f], 1) for f in eachindex(B)]
    num_obs = [size(A[f], 1) for f in eachindex(A)]

    # If C-vectors are not provided
    if isnothing(C)
        C = create_matrix_templates(num_obs, "zeros")
    end

    # If D-vectors are not provided
    if isnothing(D)
        D = create_matrix_templates(num_states)
    end

    # if num_controls are not given, they are inferred from the B matrix
    if isnothing(num_controls)
        num_controls = [size(B[f], 3) for f in eachindex(B)]  
    end

    # Determine which factors are controllable
    if isnothing(control_fac_idx)
        control_fac_idx = [f for f in eachindex(num_controls) if num_controls[f] > 1]
    end

    policies = construct_policies(num_states, n_controls=num_controls, policy_length=policy_len, controllable_factors_indices=control_fac_idx)

    # if E-vector is not provided
    if isnothing(E)
        E = ones(Real, length(policies)) / length(policies)
    end

    # Throw error if the E-vector does not match the length of policies
    if length(E) != length(policies)
        error("Length of E-vector must match the number of policies.")
    end

    qs_current = create_matrix_templates(num_states)
    obs_current = zeros(Int, length(num_obs))
    prior = D
    Q_pi = ones(length(policies)) / length(policies)  
    G = zeros(length(policies))
    action = Int[]

    # initialize states dictionary
    states = Dict(
        "action" => Vector{Real}[],
        "posterior_states" => Vector{Any}[],
        "prior" => Vector{Any}[],
        "posterior_policies" => Vector{Any}[],
        "expected_free_energies" => Vector{Any}[],
        "policies" => policies,
        "bayesian_model_averages" => Vector{Vector{<:Real}}[],
        "SAPE" => Vector{<:Real}[]
    )

    # initialize parameters dictionary
    parameters = Dict(
        "gamma" => gamma,
        "alpha" => alpha,
        "lr_pA" => lr_pA,
        "fr_pA" => fr_pA,
        "lr_pB" => lr_pB,
        "fr_pB" => fr_pB,
        "lr_pD" => lr_pD,
        "fr_pD" => fr_pD
    )

    # initialize settings dictionary
    settings = Dict(
        "policy_len" => policy_len,
        "num_controls" => num_controls,
        "control_fac_idx" => control_fac_idx,
        "use_utility" => use_utility,
        "use_states_info_gain" => use_states_info_gain,
        "use_param_info_gain" => use_param_info_gain,
        "action_selection" => action_selection,
        "modalities_to_learn" => modalities_to_learn,
        "factors_to_learn" => factors_to_learn,
        "FPI_num_iter" => FPI_num_iter,
        "FPI_dF_tol" => FPI_dF_tol
    )

    return AIF( A,
                B,
                C, 
                D, 
                E,
                pA, 
                pB, 
                pD,
                lr_pA, 
                fr_pA, 
                lr_pB, 
                fr_pB, 
                lr_pD, 
                fr_pD, 
                modalities_to_learn, 
                factors_to_learn, 
                gamma, 
                alpha, 
                policies, 
                num_controls, 
                control_fac_idx, 
                policy_len, 
                qs_current, 
                obs_current,
                prior, 
                Q_pi, 
                G, 
                action, 
                use_utility,
                use_states_info_gain, 
                use_param_info_gain, 
                action_selection, 
                FPI_num_iter, 
                FPI_dF_tol,
                states,
                parameters, 
                settings, 
                save_history)
end

"""
Initialize Active Inference Agent
function init_aif(
        A,
        B;
        C=nothing,
        D=nothing,
        E = nothing,
        pA = nothing,
        pB = nothing, 
        pD = nothing,
        parameters::Union{Nothing, Dict{String,Real}} = nothing,
        settings::Union{Nothing, Dict} = nothing,
        save_history::Bool = true)

# Arguments
- 'A': Relationship between hidden states and observations.
- 'B': Transition probabilities.
- 'C = nothing': Prior preferences over observations.
- 'D = nothing': Prior over initial hidden states.
- 'E = nothing': Prior over policies. (habits)
- 'pA = nothing':
- 'pB = nothing':
- 'pD = nothing':
- 'parameters::Union{Nothing, Dict{String,Real}} = nothing':
- 'settings::Union{Nothing, Dict} = nothing':
- 'settings::Union{Nothing, Dict} = nothing':

"""
function init_aif(A, B; C=nothing, D=nothing, E=nothing, pA=nothing, pB=nothing, pD=nothing,
                  parameters::Union{Nothing, Dict{String, T}} where T<:Real = nothing,
                  settings::Union{Nothing, Dict} = nothing,
                  save_history::Bool = true, verbose::Bool = true)

    # Catch error if A, B or D is not a proper probability distribution  
    # Check A matrix
    try
        if !check_probability_distribution(A)
            error("The A matrix is not a proper probability distribution.")
        end
    catch e
        # Add context and rethrow the error
        error("The A matrix is not a proper probability distribution. Details: $(e)")
    end

    # Check B matrix
    try
        if !check_probability_distribution(B)
            error("The B matrix is not a proper probability distribution.")
        end
    catch e
        # Add context and rethrow the error
        error("The B matrix is not a proper probability distribution. Details: $(e)")
    end

    # Check D matrix (if it's not nothing)
    try
        if !isnothing(D) && !check_probability_distribution(D)
            error("The D matrix is not a proper probability distribution.")
        end
    catch e
        # Add context and rethrow the error
        error("The D matrix is not a proper probability distribution. Details: $(e)")
    end

    # Throw warning if no D-vector is provided. 
    if verbose == true && isnothing(C)
        @warn "No C-vector provided, no prior preferences will be used."
    end 

    # Throw warning if no D-vector is provided. 
    if verbose == true && isnothing(D)
        @warn "No D-vector provided, a uniform distribution will be used."
    end 

    # Throw warning if no E-vector is provided. 
    if verbose == true && isnothing(E)
        @warn "No E-vector provided, a uniform distribution will be used."
    end           
    
    # Check if settings are provided or use defaults
    if isnothing(settings)

        if verbose == true
            @warn "No settings provided, default settings will be used."
        end

        settings = Dict(
            "policy_len" => 1, 
            "num_controls" => nothing, 
            "control_fac_idx" => nothing, 
            "use_utility" => true, 
            "use_states_info_gain" => true, 
            "use_param_info_gain" => false,
            "action_selection" => "stochastic", 
            "modalities_to_learn" => "all",
            "factors_to_learn" => "all",
            "FPI_num_iter" => 10,
            "FPI_dF_tol" => 0.001
        )
    end

    # Check if parameters are provided or use defaults
    if isnothing(parameters)

        if verbose == true
            @warn "No parameters provided, default parameters will be used."
        end
        
        parameters = Dict("gamma" => 16.0,
                          "alpha" => 16.0,
                          "lr_pA" => 1.0,
                          "fr_pA" => 1.0,
                          "lr_pB" => 1.0,
                          "fr_pB" => 1.0,
                          "lr_pD" => 1.0,
                          "fr_pD" => 1.0
                          )
    end

    # Extract parameters and settings from the dictionaries or use defaults
    gamma = get(parameters, "gamma", 16.0)  
    alpha = get(parameters, "alpha", 16.0)
    lr_pA = get(parameters, "lr_pA", 1.0)
    fr_pA = get(parameters, "fr_pA", 1.0)
    lr_pB = get(parameters, "lr_pB", 1.0)
    fr_pB = get(parameters, "fr_pB", 1.0)
    lr_pD = get(parameters, "lr_pD", 1.0)
    fr_pD = get(parameters, "fr_pD", 1.0)

    
    policy_len = get(settings, "policy_len", 1)
    num_controls = get(settings, "num_controls", nothing)
    control_fac_idx = get(settings, "control_fac_idx", nothing)
    use_utility = get(settings, "use_utility", true)
    use_states_info_gain = get(settings, "use_states_info_gain", true)
    use_param_info_gain = get(settings, "use_param_info_gain", false)
    action_selection = get(settings, "action_selection", "stochastic")
    modalities_to_learn = get(settings, "modalities_to_learn", "all" )
    factors_to_learn = get(settings, "factors_to_learn", "all" )
    FPI_num_iter = get(settings, "FPI_num_iter", 10 )
    FPI_dF_tol = get(settings, "FPI_dF_tol", 0.001 )

    # Call create_aif 
    aif = create_aif(A, B,
                    C=C,
                    D=D,
                    E=E,
                    pA=pA,
                    pB=pB,
                    pD=pD,
                    lr_pA = lr_pA, 
                    fr_pA = fr_pA, 
                    lr_pB = lr_pB, 
                    fr_pB = fr_pB, 
                    lr_pD = lr_pD, 
                    fr_pD = fr_pD,
                    modalities_to_learn=modalities_to_learn,
                    factors_to_learn=factors_to_learn,
                    gamma=gamma,
                    alpha=alpha, 
                    policy_len=policy_len,
                    num_controls=num_controls,
                    control_fac_idx=control_fac_idx, 
                    use_utility=use_utility, 
                    use_states_info_gain=use_states_info_gain, 
                    use_param_info_gain=use_param_info_gain,
                    action_selection=action_selection,
                    FPI_num_iter=FPI_num_iter,
                    FPI_dF_tol=FPI_dF_tol,
                    save_history=save_history
                    )

    #Print out agent settings
    if verbose == true
        settings_summary = 
        """
        AIF Agent initialized successfully with the following settings and parameters:
        - Gamma (γ): $(aif.gamma)
        - Alpha (α): $(aif.alpha)
        - Policy Length: $(aif.policy_len)
        - Number of Controls: $(aif.num_controls)
        - Controllable Factors Indices: $(aif.control_fac_idx)
        - Use Utility: $(aif.use_utility)
        - Use States Information Gain: $(aif.use_states_info_gain)
        - Use Parameter Information Gain: $(aif.use_param_info_gain)
        - Action Selection: $(aif.action_selection)
        - Modalities to Learn = $(aif.modalities_to_learn)
        - Factors to Learn = $(aif.factors_to_learn)
        """
        println(settings_summary)
    end
    
    return aif
end

### Struct related functions ###

"""
    construct_policies(n_states::Vector{T} where T <: Real; n_controls::Union{Vector{T}, Nothing} where T <: Real=nothing, 
                       policy_length::Int=1, controllable_factors_indices::Union{Vector{Int}, Nothing}=nothing)

Construct policies based on the number of states, controls, policy length, and indices of controllable state factors.

# Arguments
- `n_states::Vector{T} where T <: Real`: A vector containing the number of  states for each factor.
- `n_controls::Union{Vector{T}, Nothing} where T <: Real=nothing`: A vector specifying the number of allowable actions for each state factor. 
- `policy_length::Int=1`: The length of policies. (planning horizon)
- `controllable_factors_indices::Union{Vector{Int}, Nothing}=nothing`: A vector of indices identifying which state factors are controllable.

"""
function construct_policies(
    n_states::Vector{T} where T <: Real; 
    n_controls::Union{Vector{T}, Nothing} where T <: Real=nothing, 
    policy_length::Int=1, 
    controllable_factors_indices::Union{Vector{Int}, Nothing}=nothing
    )

    # Determine the number of state factors
    n_factors = length(n_states)

    # If indices of controllable factors are not given 
    if isnothing(controllable_factors_indices)
        if !isnothing(n_controls)
            # Determine controllable factors based on which factors have more than one control
            controllable_factors_indices = findall(x -> x > 1, n_controls)
        else
            # If no controls are given, assume all factors are controllable
            controllable_factors_indices = 1:n_factors
        end
    end

    # if number of controls is not given, determine it based n_states and controllable_factors_indices
    if isnothing(n_controls)
        n_controls = [in(factor_index, controllable_factors_indices) ? n_states[factor_index] : 1 for factor_index in 1:n_factors]
    end

    # Create a vector of possible actions for each time step
    x = repeat(n_controls, policy_length)

    # Generate all combinations of actions across all time steps
    policies = collect(Iterators.product([1:i for i in x]...))

    # Initialize an empty vector to store transformed policies
    transformed_policies = Vector{Matrix{Int64}}()

    for policy_tuple in policies
        # Convert tuple into a vector
        policy_vector = collect(policy_tuple)
        
        # Reshape the policy vector into a matrix and transpose it
        policy_matrix = reshape(policy_vector, (length(policy_vector) ÷ policy_length, policy_length))'
        
        # Push the reshaped matrix to the vector of transformed policies
        push!(transformed_policies, policy_matrix)
    end

    return transformed_policies
end

""" Update the agents's beliefs over states """
function infer_states!(aif::AIF, obs::Vector{Int64})
    if !isempty(aif.action)
        int_action = round.(Int, aif.action)
        aif.prior = get_expected_states(aif.qs_current, aif.B, reshape(int_action, 1, length(int_action)))[1]
    else
        aif.prior = aif.D
    end

    # Update posterior over states
    aif.qs_current = update_posterior_states(aif.A, obs, prior=aif.prior, num_iter=aif.FPI_num_iter, dF_tol=aif.FPI_dF_tol)

    # Adding the obs to the agent struct
    aif.obs_current = obs

    # Push changes to agent's history
    push!(aif.states["prior"], aif.prior)
    push!(aif.states["posterior_states"], aif.qs_current)

    return aif.qs_current
end

""" Update the agents's beliefs over policies """
function infer_policies!(aif::AIF)
    # Update posterior over policies and expected free energies of policies
    q_pi, G = update_posterior_policies(aif.qs_current, aif.A, aif.B, aif.C, aif.policies, aif.use_utility, aif.use_states_info_gain, aif.use_param_info_gain, aif.pA, aif.pB, aif.E, aif.gamma)

    aif.Q_pi = q_pi
    aif.G = G  

    # Push changes to agent's history
    push!(aif.states["posterior_policies"], copy(aif.Q_pi))
    push!(aif.states["expected_free_energies"], copy(aif.G))

    return q_pi
end

""" Sample action from the beliefs over policies """
function sample_action!(aif::AIF)
    action = sample_action(aif.Q_pi, aif.policies, aif.num_controls; action_selection=aif.action_selection, alpha=aif.alpha)

    aif.action = action 

    # Push action to agent's history
    push!(aif.states["action"], copy(aif.action))


    return action
end

""" Update A-matrix """
function update_A!(aif::AIF)

    qA = update_obs_likelihood_dirichlet(aif.pA, aif.A, aif.obs_current, aif.qs_current, lr = aif.lr_pA, fr = aif.fr_pA, modalities = aif.modalities_to_learn)
    
    aif.pA = deepcopy(qA)
    aif.A = deepcopy(normalize_arrays(qA))

    return qA
end

""" Update B-matrix """
function update_B!(aif::AIF)

    if length(get_history(aif, "posterior_states")) > 1

        qs_prev = get_history(aif, "posterior_states")[end-1]

        qB = update_state_likelihood_dirichlet(aif.pB, aif.B, aif.action, aif.qs_current, qs_prev, lr = aif.lr_pB, fr = aif.fr_pB, factors = aif.factors_to_learn)

        aif.pB = deepcopy(qB)
        aif.B = deepcopy(normalize_arrays(qB))
    else
        qB = nothing
    end

    return qB
end

""" Update D-matrix """
function update_D!(aif::AIF)

    if length(get_history(aif, "posterior_states")) == 1

        qs_t1 = get_history(aif, "posterior_states")[end]
        qD = update_state_prior_dirichlet(aif.pD, qs_t1; lr = aif.lr_pD, fr = aif.fr_pD, factors = aif.factors_to_learn)

        aif.pD = deepcopy(qD)
        aif.D = deepcopy(normalize_arrays(qD))
    else
        qD = nothing
    end
    return qD
end

""" General Learning Update Function """

function update_parameters!(aif::AIF)

    if aif.pA != nothing
        update_A!(aif)
    end

    if aif.pB != nothing
        update_B!(aif)
    end

    if aif.pD != nothing
        update_D!(aif)
    end
    
end

""" Get the history of the agent """





================================================
FILE: src/utils/create_matrix_templates.jl
================================================
######################## Create Templates Based on states, observations, controls and policy length  ########################

"""
    create_matrix_templates(n_states::Vector{Int64}, n_observations::Vector{Int64}, n_controls::Vector{Int64}, policy_length::Int64, template_type::String = "uniform")

Creates templates for the A, B, C, D, and E matrices based on the specified parameters.

# Arguments
- `n_states::Vector{Int64}`: A vector specifying the dimensions and number of states.
- `n_observations::Vector{Int64}`: A vector specifying the dimensions and number of observations.
- `n_controls::Vector{Int64}`: A vector specifying the number of controls per factor.
- `policy_length::Int64`: The length of the policy sequence. 
- `template_type::String`: The type of templates to create. Can be "uniform", "random", or "zeros". Defaults to "uniform".

# Returns
- `A, B, C, D, E`: The generative model as matrices and vectors.

"""
function create_matrix_templates(n_states::Vector{Int64}, n_observations::Vector{Int64}, n_controls::Vector{Int64}, policy_length::Int64)
    
    # Calculate the number of policies based on the policy length
    n_policies = prod(n_controls) ^ policy_length

    # Uniform A matrices
    A = [normalize_distribution(ones(vcat(observation_dimension, n_states)...)) for observation_dimension in n_observations]

    # Uniform B matrices
    B = [normalize_distribution(ones(state_dimension, state_dimension, n_controls[index])) for (index, state_dimension) in enumerate(n_states)]

    # C vectors as zero vectors
    C = [zeros(observation_dimension) for observation_dimension in n_observations]

    # Uniform D vectors
    D = [fill(1.0 / state_dimension, state_dimension) for state_dimension in n_states]

    # Uniform E vector
    E = fill(1.0 / n_policies, n_policies)

    return A, B, C, D, E
end

function create_matrix_templates(n_states::Vector{Int64}, n_observations::Vector{Int64}, n_controls::Vector{Int64}, policy_length::Int64, template_type::String)
    
    # If the template_type is uniform
    if template_type == "uniform"
        return create_matrix_templates(n_states, n_observations, n_controls, policy_length)
    end

    # Calculate the number of policies based on the policy length
    n_policies = prod(n_controls) ^ policy_length

    # If the template type is random, populate the matrices with random values
    if template_type == "random"
        # Random A matrices
        A = [normalize_distribution(rand(vcat(observation_dimension, n_states)...)) for observation_dimension in n_observations]

        # Random B matrices
        B = [normalize_distribution(rand(state_dimension, state_dimension, n_controls[index])) for (index, state_dimension) in enumerate(n_states)]

        # C vectors populated with random integers between -4 and 4
        C = [rand(-4:4, observation_dimension) for observation_dimension in n_observations]

        # Random D vectors
        D = [normalize_distribution(rand(state_dimension)) for state_dimension in n_states]

        # Random E vector
        E = normalize_distribution(rand(n_policies))
    
    # If the template type is zeros, populate the matrices with zeros
    elseif template_type == "zeros"

        A = [zeros(vcat(observation_dimension, n_states)...) for observation_dimension in n_observations]
        B = [zeros(state_dimension, state_dimension, n_controls[index]) for (index, state_dimension) in enumerate(n_states)]
        C = [zeros(observation_dimension) for observation_dimension in n_observations]
        D = [zeros(state_dimension) for state_dimension in n_states]
        E = zeros(n_policies)
    
    else
        # Throw error for invalid template type
        throw(ArgumentError("Invalid type: $template_type. Choose either 'uniform', 'random' or 'zeros'."))
    end

    return A, B, C, D, E
end

######################## Create Templates Based on Shapes ########################

### Single Array Input 

"""
    create_matrix_templates(shapes::Vector{Int64})

Creates uniform templates based on the specified shapes vector.

# Arguments
- `shapes::Vector{Int64}`: A vector specifying the dimensions of each template to create.

# Returns
- A vector of normalized arrays.

"""
function create_matrix_templates(shapes::Vector{Int64})

    # Create arrays filled with ones and then normalize
    return [normalize_distribution(ones(n)) for n in shapes]
end

"""
    create_matrix_templates(shapes::Vector{Int64}, template_type::String)

Creates templates based on the specified shapes vector and template type. Templates can be uniform, random, or filled with zeros.

# Arguments
- `shapes::Vector{Int64}`: A vector specifying the dimensions of each template to create.
- `template_type::String`: The type of templates to create. Can be "uniform" (default), "random", or "zeros".

# Returns
- A vector of arrays, each corresponding to the shape given by the input vector.


"""
function create_matrix_templates(shapes::Vector{Int64}, template_type::String, eltype::Type=Float64)

    if template_type == "uniform"
        # Create arrays filled with ones and then normalize
        return [normalize_distribution(ones(eltype, n)) for n in shapes]

    elseif template_type == "random"
        # Create arrays filled with random values
        return [normalize_distribution(rand(eltype, n)) for n in shapes]

    elseif template_type == "zeros"
        # Create arrays filled with zeros
        return [zeros(eltype, n) for n in shapes]

    else
        # Throw error for invalid template type
        throw(ArgumentError("Invalid type: $template_type. Choose either 'uniform', 'random' or 'zeros'."))
    end
end

### Vector of Arrays Input 

"""
    create_matrix_templates(shapes::Vector{Vector{Int64}})

Creates a uniform, multidimensional template based on the specified shapes vector.

# Arguments
- `shapes::Vector{Vector{Int64}}`: A vector of vectors, where each vector represent a dimension of the template to create.

# Returns
- A vector of normalized arrays (uniform distributions), each having the multi-dimensional shape specified in the input vector.

"""
function create_matrix_templates(shapes::Vector{Vector{Int64}})

    # Create arrays filled with ones and then normalize
    return [normalize_distribution(ones(shape...)) for shape in shapes]
end

"""
    create_matrix_templates(shapes::Vector{Vector{Int64}}, template_type::String)

Creates a multidimensional template based on the specified vector of shape vectors and template type. Templates can be uniform, random, or filled with zeros.

# Arguments
- `shapes::Vector{Vector{Int64}}`: A vector of vectors, where each vector represent a dimension of the template to create.
- `template_type::String`: The type of templates to create. Can be "uniform" (default), "random", or "zeros".

# Returns
- A vector of arrays, each having the multi-dimensional shape specified in the input vector.

"""
function create_matrix_templates(shapes::Vector{Vector{Int64}}, template_type::String)

    if template_type == "uniform"
        # Create arrays filled with ones and then normalize
        return [normalize_distribution(ones(shape...)) for shape in shapes]

    elseif template_type == "random"
        # Create arrays filled with random values
        return [normalize_distribution(rand(shape...)) for shape in shapes]
    
    elseif template_type == "zeros"
        # Create arrays filled with zeros
        return [zeros(shape...) for shape in shapes]

    else
        # Throw error for invalid template type
        throw(ArgumentError("Invalid type: $template_type. Choose either 'uniform', 'random' or 'zeros'."))
    end
end


================================================
FILE: src/utils/helper_functions.jl
================================================



================================================
FILE: src/utils/maths.jl
================================================
"""Normalizes a Categorical probability distribution"""
function normalize_distribution(distribution)
    distribution .= distribution ./ sum(distribution, dims=1)
    return distribution
end


"""
    capped_log(x::Real)

# Arguments
- `x::Real`: A real number.

Return the natural logarithm of x, capped at the machine epsilon value of x.
"""
function capped_log(x::Real)
    return log(max(x, eps(x))) 
end

"""
    capped_log(array::Array{Float64})
"""
function capped_log(array::Array{Float64}) 

    epsilon = oftype(array[1], 1e-16)
    # Return the log of the array values capped at epsilon
    array = log.(max.(array, epsilon))

    return array
end

"""
    capped_log(array::Array{T}) where T <: Real 
"""
function capped_log(array::Array{T}) where T <: Real 

    epsilon = oftype(array[1], 1e-16)
    # Return the log of the array values capped at epsilon
    array = log.(max.(array, epsilon))

    return array
end

"""
    capped_log(array::Vector{Real})
"""
function capped_log(array::Vector{Real})
    epsilon = oftype(array[1], 1e-16)

    array = log.(max.(array, epsilon))
    # Return the log of the array values capped at epsilon
    return array
end

""" Apply capped_log to array of arrays """
function capped_log_array(array)
    
    return map(capped_log, array)
end


""" Get Joint Likelihood """
function get_joint_likelihood(A, obs_processed, num_states)
    ll = ones(Real, num_states...)
    for modality in eachindex(A)
        ll .*= dot_likelihood(A[modality], obs_processed[modality])
    end

    return ll
end

""" Dot-Product Function """
function dot_likelihood(A, obs)
    # Adjust the shape of obs to match A
    reshaped_obs = reshape(obs, (length(obs), 1, 1, 1))  
    # Element-wise multiplication and sum over the first axis
    LL = sum(A .* reshaped_obs, dims=1)
    # Remove singleton dimensions
    LL = dropdims(LL, dims= tuple(findall(size(LL) .== 1)...))
    if prod(size(LL)) == 1
        LL = [LL[]]  
    end
    return LL
end

""" Softmax Function for array of arrays """
function softmax_array(array)
    # Use map to apply softmax to each element of arr
    array .= map(x -> softmax(x, dims=1), array)
    
    return array
end


""" Multi-dimensional outer product """
function outer_product(x, y=nothing; remove_singleton_dims=true, args...)
    # If only x is provided and it is a vector of arrays, recursively call outer_product on its elements.
    if y === nothing && isempty(args)
        if x isa AbstractVector
            return reduce((a, b) -> outer_product(a, b), x)
        elseif typeof(x) <: Number || typeof(x) <: AbstractArray
            return x
        else
            throw(ArgumentError("Invalid input to outer_product (\$x)"))
        end
    end

    # If y is provided, perform the cross multiplication.
    if y !== nothing
        reshape_dims_x = tuple(size(x)..., ones(Real, ndims(y))...)
        A = reshape(x, reshape_dims_x)

        reshape_dims_y = tuple(ones(Real, ndims(x))..., size(y)...)
        B = reshape(y, reshape_dims_y)

        z = A .* B

    else
        z = x
    end

    # Recursively call outer_product for additional arguments
    for arg in args
        z = outer_product(z, arg; remove_singleton_dims=remove_singleton_dims)
    end

    # Remove singleton dimensions if true
    if remove_singleton_dims
        z = dropdims(z, dims = tuple(findall(size(z) .== 1)...))
    end

    return z
end

#Multidimensional inner product
# Instead of summing over all indices, the function sums over only the last three
# dimensions of X while keeping the first dimension separate, creating a sum for each "layer" of X.

function dot_product(X, x)

    if all(isa.(x, AbstractArray))  
        n_factors = length(x)
    else
        x = [x]  
        n_factors = length(x)
    end

    ndims_X = ndims(X)
    dims = collect(ndims_X - n_factors + 1 : ndims_X)
    Y = zeros(Real, size(X, 1))

    for indices in Iterators.product((1:size(X, i) for i in 1:ndims_X)...)
        product = X[indices...] * prod(x[factor][indices[dims[factor]]] for factor in 1:n_factors)
        Y[indices[1]] += product
    end

    if prod(size(Y)) <= 1
        Y = only(Y)
        Y = [float(Y)]  
    end

    return Y
end


""" Calculate Bayesian Surprise """
function calculate_bayesian_surprise(A, x)
    qx = outer_product(x)
    G = 0.0
    qo = Vector{Float64}()
    idx = [collect(Tuple(indices)) for indices in findall(qx .> exp(-16))]
    index_vector = []

    for i in idx   
        po = ones(Real, 1)
        for (_, A_m) in enumerate(A)
            index_vector = (1:size(A_m, 1),)  
            for additional_index in i  
                index_vector = (index_vector..., additional_index)  
            end
            po = outer_product(po, A_m[index_vector...])
        end
        po = vec(po) 
        if isempty(qo)
            qo = zeros(length(po))
        end
        qo += qx[i...] * po
        G += qx[i...] * dot(po, log.(po .+ exp(-16)))
    end
    G = G - dot(qo, capped_log(qo))
    return G
end

""" Normalizes multiple arrays """
function normalize_arrays(array::Vector{<:Array{<:Real}})
    return map(normalize_distribution, array)
end

""" Normalizes multiple arrays """
function normalize_arrays(array::Vector{Any})
    return map(normalize_distribution, array)
end

""" SPM_wnorm """
function spm_wnorm(A)
    EPS_VAL = 1e-16

    A .+= EPS_VAL
    norm = 1.0 ./ sum(A, dims = 1)
    avg = 1 ./ A
    wA = norm .- avg

    return wA
end

"""
    Calculate Bayesian Model Average (BMA)

Calculates the Bayesian Model Average (BMA) which is used for the State Action Prediction Error (SAPE).
It is a weighted average of the expected states for all policies weighted by the posterior over policies.
The `qs_pi_all` should be the collection of expected states given all policies. Can be retrieved with the
`get_expected_states` function.

`qs_pi_all`: Vector{Any} \n
`q_pi`: Vector{Float64}

"""
function bayesian_model_average(qs_pi_all, q_pi)

    # Extracting the number of factors, states, and timesteps (policy length) from the first policy
    n_factors = length(qs_pi_all[1][1])
    n_states = [size(qs_f, 1) for qs_f in qs_pi_all[1][1]]
    n_steps = length(qs_pi_all[1])

    # Preparing vessel for the expected states for all policies. Has number of undefined entries equal to the number of 
    # n_steps with each entry having the entries equal to the number of factors
    qs_bma = [Vector{Vector{Real}}(undef, n_factors) for _ in 1:n_steps]

    # Populating the entries with zeros for each state in each factor for each timestep in policy
    for i in 1:n_steps
        for f in 1:n_factors
            qs_bma[i][f] = zeros(Real, n_states[f])
        end
    end

    # Populating the entries with the expected states for all policies weighted by the posterior over policies
    for i in 1:n_steps
        for (pol_idx, policy_weight) in enumerate(q_pi)
            for f in 1:n_factors
                qs_bma[i][f] .+= policy_weight .* qs_pi_all[pol_idx][i][f]
            end
        end
    end

    return qs_bma
end

"""
    kl_divergence(P::Vector{Vector{Vector{Float64}}}, Q::Vector{Vector{Vector{Float64}}})

# Arguments
- `P::Vector{Vector{Vector{Real}}}`
- `Q::Vector{Vector{Vector{Real}}}`

Return the Kullback-Leibler (KL) divergence between two probability distributions.
"""
function kl_divergence(P::Vector{Vector{Vector{Real}}}, Q::Vector{Vector{Vector{Real}}})
    eps_val = 1e-16  # eps constant to avoid log(0)
    dkl = 0.0  # Initialize KL divergence to zero

    for j in 1:length(P)
        for i in 1:length(P[j])
            # Compute the dot product of P[j][i] and the difference of logs of P[j][i] and Q[j][i]
            dkl += dot(P[j][i], log.(P[j][i] .+ eps_val) .- log.(Q[j][i] .+ eps_val))
        end
    end

    return dkl  # Return KL divergence
end






================================================
FILE: src/utils/utils.jl
================================================
""" -------- Utility Functions -------- """

""" Creates an array of "Any" with the desired number of sub-arrays filled with zeros"""
function array_of_any_zeros(shape_list)
    arr = Array{Any}(undef, length(shape_list))
    for (i, shape) in enumerate(shape_list)
        arr[i] = zeros(Real, shape...)
    end
    return arr
end

""" Creates a onehot encoded vector """
function onehot(index::Int, vector_length::Int)
    vector = zeros(vector_length)
    vector[index] = 1.0
    return vector
end

""" Get Model Dimensions from either A or B Matrix """
function get_model_dimensions(A = nothing, B = nothing)
    if A === nothing && B === nothing
        throw(ArgumentError("Must provide either `A` or `B`"))
    end
    num_obs, num_modalities, num_states, num_factors = nothing, nothing, nothing, nothing

    if A !== nothing
        num_obs = [size(a, 1) for a in A]
        num_modalities = length(num_obs)
    end

    if B !== nothing
        num_states = [size(b, 1) for b in B]
        num_factors = length(num_states)
    elseif A !== nothing
        num_states = [size(A[1], i) for i in 2:ndims(A[1])]
        num_factors = length(num_states)
    end

    return num_obs, num_states, num_modalities, num_factors
end


""" Selects the highest value from Array -- used for deterministic action sampling """
function select_highest(options_array::Vector{T}) where T <: Real
    options_with_idx = [(i, option) for (i, option) in enumerate(options_array)]
    max_value = maximum(value for (idx, value) in options_with_idx)
    same_prob = [idx for (idx, value) in options_with_idx if abs(value - max_value) <= 1e-8]

    if length(same_prob) > 1
        return same_prob[rand(1:length(same_prob))]
    else
        return same_prob[1]
    end
end


""" Selects action from computed actions probabilities -- used for stochastic action sampling """
function action_select(probabilities)
    sample_onehot = rand(Multinomial(1, probabilities))
    return findfirst(sample_onehot .== 1)
end

""" Function to get log marginal probabilities of actions """
function get_log_action_marginals(aif)
    num_factors = length(aif.num_controls)
    q_pi = get_states(aif, "posterior_policies")
    policies = get_states(aif, "policies")
    
    # Determine the element type from q_pi
    eltype_q_pi = eltype(q_pi)

    # Initialize action_marginals with the correct element type
    action_marginals = create_matrix_templates(aif.num_controls, "zeros", eltype_q_pi)
    log_action_marginals = Vector{Any}(undef, num_factors)
    
    for (pol_idx, policy) in enumerate(policies)
        for (factor_i, action_i) in enumerate(policy[1,:])
            action_marginals[factor_i][action_i] += q_pi[pol_idx]
        end
    end

    action_marginals = normalize_arrays(action_marginals)

    for factor_i in 1:num_factors
        log_marginal_f = capped_log(action_marginals[factor_i])
        log_action_marginals[factor_i] = log_marginal_f
    end

    return log_action_marginals
end

"""
Check if the vector of arrays is a proper probability distribution.

# Arguments

- (Array::Vector{<:Array{T}}) where T<:Real

Throws an error if the array is not a valid probability distribution:
- The values must be non-negative.
- The sum of the values must be approximately 1.
"""
function check_probability_distribution(Array::Vector{<:Array{T}}) where T<:Real
    for tensor in Array
        # Check for non-negativity
        if any(tensor .< 0)
            throw(ArgumentError("All elements must be non-negative."))
        end

        # Check for normalization
        if !all(isapprox.(sum(tensor, dims=1), 1.0, rtol=1e-5, atol=1e-8))
            throw(ArgumentError("The array is not normalized."))
        end
    end

    return true
end

"""
Check if the vector of vectors is a proper probability distribution.

# Arguments

- (Array::Vector{Vector{T}}) where T<:Real

Throws an error if the array is not a valid probability distribution:
- The values must be non-negative.
- The sum of the values must be approximately 1.
"""
function check_probability_distribution(Array::Vector{Vector{T}}) where T<:Real
    for vector in Array
        # Check for non-negativity
        if any(vector .< 0)
            throw(ArgumentError("All elements must be non-negative."))
        end

        # Check for normalization
        if !all(isapprox.(sum(vector, dims=1), 1.0, rtol=1e-5, atol=1e-8))
            throw(ArgumentError("The array is not normalized."))
        end
    end

    return true
end

"""
Check if the vector is a proper probability distribution.

# Arguments

- (Vector::Vector{T}) where T<:Real : The vector to be checked.

Throws an error if the array is not a valid probability distribution:
- The values must be non-negative.
- The sum of the values must be approximately 1.
"""
function check_probability_distribution(Vector::Vector{T}) where T<:Real
    # Check for non-negativity
    if any(Vector .< 0)
        throw(ArgumentError("All elements must be non-negative."))
    end

    # Check for normalization
    if !all(isapprox.(sum(Vector, dims=1), 1.0, rtol=1e-5, atol=1e-8))
        throw(ArgumentError("The array is not normalized."))
    end

    return true
end


================================================
FILE: test/Project.toml
================================================
[deps]
ActionModels = "320cf53b-cc3b-4b34-9a10-0ecb113566a3"
Aqua = "4c88cf16-eb10-579e-8560-4a9242c79595"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Glob = "c27321d9-0574-5035-807b-f59d2c89b15c"
HDF5 = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
IterTools = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"



================================================
FILE: test/quicktests.jl
================================================
using IterTools
using LinearAlgebra
using ActiveInference
using Test

""" Quick tests """

@testset "Multiple Factors/Modalities Default Settings" begin

    # Initializse States, Observations, and Controls
    states = [5,2]
    observations = [5, 4, 2]
    controls = [2,1]
    policy_length = 1

    # Generate random Generative Model 
    A,B = create_matrix_templates(states, observations, controls, policy_length, "random");

    # Initialize agent with default settings/parameters
    aif = init_aif(A,B);

    # Give observation to agent and run state inference
    observation = [rand(1:observations[i]) for i in axes(observations, 1)]
    infer_states!(aif, observation)

    # Run policy inference
    infer_policies!(aif)

    # Sample action
    sample_action!(aif)
end


================================================
FILE: test/runtests.jl
================================================
using ActiveInference
using Test
using Glob

ActiveInference_path = dirname(dirname(pathof(ActiveInference)))

@testset "all tests" begin
    test_path = ActiveInference_path * "/test/"

    @testset "quick tests" begin
        # Include quick tests similar to pre-commit tests
        include("quicktests.jl")
    end

    # List the Julia filenames in the testsuite
    filenames = glob("*.jl", test_path * "testsuite")

    # For each file
    for filename in filenames
        include(filename)
    end
end





================================================
FILE: test/pymdp_cross_val/cross_val_complete_run/julia_complete_script/complete_run_julia.jl
================================================
using ActiveInference
using ActiveInference.Environments
using HDF5
using LinearAlgebra

file_path_gm = "ActiveInference.jl/test/pymdp_cross_val/generative_model_creation/gm_data/gm_matrices.h5"

#############################################
### Loading Generative Model from h5 file ###
#############################################

# A-matrix
A_cross = array_of_any(4)
for i in 1:4
    A_cross[i] = h5read(file_path_gm, "A_cross_$i")
end

# pA-matrix
pA_cross = array_of_any(4)
for i in 1:4
    pA_cross[i] = h5read(file_path_gm, "pA_cross_$i")
end

# B-matrix
B_cross = array_of_any(3)
for i in 1:3
    B_cross[i] = h5read(file_path_gm, "B_cross_$i")
end

# pB-matrix
pB_cross = array_of_any(3)
for i in 1:3
    pB_cross[i] = h5read(file_path_gm, "pB_cross_$i")
end

# C-matrix
C_cross = array_of_any(4)
for i in 1:4
    C_cross[i] = h5read(file_path_gm, "C_cross_$i")
end

# D-matrix
D_cross = array_of_any(3)
for i in 1:3
    D_cross[i] = h5read(file_path_gm, "D_cross_$i")
end

# pD-matrix
pD_cross = array_of_any(3)
for i in 1:3
    pD_cross[i] = h5read(file_path_gm, "pD_cross_$i")
end

################################
### Creating cross val agent ###
################################

settings = Dict("use_param_info_gain" => true,
                "use_states_info_gain" => true,
                "action_selection" => "deterministic",
                "policy_len" => 4)

parameters=Dict{String, Real}("lr_pB" => 0.5,
                              "lr_pA" => 0.5,
                              "lr_pD" => 0.5)

cross_agent = init_aif(A_cross, B_cross, C = C_cross, D = D_cross, pA = pA_cross, pB = pB_cross, pD = pD_cross, settings = settings, parameters = parameters);

#############################################
### Creating and initialising environment ###
#############################################

grid_locations = collect(Iterators.product(1:5, 1:7))
start_loc = (1,1)
cue1_location = (3, 1)
cue2_loc = "L4"
reward_cond = ("BOTTOM")
obs = [1, 1, 1, 1]
location_to_index = Dict(loc => idx for (idx, loc) in enumerate(grid_locations))
actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

cue2_loc_names = ["L1","L2","L3","L4"]
cue2_locations = [(1, 3), (2, 4), (4, 4), (5, 3)]

reward_conditions = ["TOP", "BOTTOM"]
reward_locations = [(2,6), (4,6)]

cue1_names = ["Null";cue2_loc_names]
cue2_names = ["Null", "reward_on_top", "reward_on_bottom"]
reward_names = ["Null", "Cheese", "Shock"]

# Initializing environment
env = EpistChainEnv(start_loc, cue1_location, cue2_loc, reward_cond, grid_locations)

# Getting initial obs
obs = h5read(file_path_gm, "obs")

##########################
### Running simulation ###
##########################

# Time step set to 50 trials
T = 50

# Run simulation
for t in 1:T

    qs = infer_states!(cross_agent, obs)

    update_A!(cross_agent, obs)

    if t != 1
        qs_prev = get_history(cross_agent)["posterior_states"][end-1]
        update_B!(cross_agent, qs_prev)
    end

    if t == 1
        qs_t1 = cross_agent.qs_current
        update_D!(cross_agent, qs_t1)
    end

    q_pi, G = infer_policies!(cross_agent)

    chosen_action_id = sample_action!(cross_agent)

    movement_id = Int(chosen_action_id[1])
    choice_action = actions[movement_id]

    loc_obs, cue1_obs, cue2_obs, reward_obs = step!(env, choice_action)
    obs = [location_to_index[loc_obs], findfirst(isequal(cue1_obs), cue1_names), findfirst(isequal(cue2_obs), cue2_names), findfirst(isequal(reward_obs), reward_names)]

end


###########################
### Storing the results ###
###########################

# Saving the agent parameters after run for cross validate with pymdp
file_path_results = "ActiveInference.jl/test/pymdp_cross_val/cross_val_results/complete_run_data.h5"

# Storing the A-matrix
h5write(file_path_results, "julia_A_cross_1", cross_agent.A[1])
h5write(file_path_results, "julia_A_cross_2", cross_agent.A[2])
h5write(file_path_results, "julia_A_cross_3", cross_agent.A[3])
h5write(file_path_results, "julia_A_cross_4", cross_agent.A[4])

# Storing the B-matrix
h5write(file_path_results, "julia_B_cross_1", cross_agent.B[1])
h5write(file_path_results, "julia_B_cross_2", cross_agent.B[2])
h5write(file_path_results, "julia_B_cross_3", cross_agent.B[3])

# Storing the D-matrix
h5write(file_path_results, "julia_D_cross_1", cross_agent.D[1])
h5write(file_path_results, "julia_D_cross_2", cross_agent.D[2])
h5write(file_path_results, "julia_D_cross_3", cross_agent.D[3])

# Storing the posterior states
h5write(file_path_results, "julia_qs_1", Float64.(cross_agent.qs_current[1]))
h5write(file_path_results, "julia_qs_2", Float64.(cross_agent.qs_current[2]))
h5write(file_path_results, "julia_qs_3", Float64.(cross_agent.qs_current[3]))



================================================
FILE: test/pymdp_cross_val/cross_val_complete_run/python_complete_script/complete_run_python.py
================================================
import h5py
import numpy as np

from pymdp import utils, maths, learning, control, inference
from pymdp.agent import Agent
from pymdp.envs import Env

#############################################
### Loading Generative Model from h5 file ###
#############################################

# Path to file with generative model
file_path_gm = "../../generative_model_creation/gm_data/gm_matrices.h5"

# A-matrix
A_cross = utils.obj_array(4)
for i in range(1, 5):
    with h5py.File(file_path_gm, 'r') as file:
        A_cross[i-1] = file[f"A_cross_{i}"][:]
        
    # Converting the column-major julia indexing into the row-major python indexing
    A_cross[i-1] = np.transpose(A_cross[i-1], (3, 2, 1, 0))

# pA-matrix
pA_cross = utils.obj_array(4)
for i in range(1, 5):
    with h5py.File(file_path_gm, 'r') as file:
        pA_cross[i-1] = file[f"pA_cross_{i}"][:]
        
    # Converting the column-major julia indexing into the row-major python indexing
    pA_cross[i-1] = np.transpose(pA_cross[i-1], (3, 2, 1, 0))

# B-matrix
B_cross = utils.obj_array(3)
for i in range(1, 4):
    with h5py.File(file_path_gm, 'r') as file:
        B_cross[i-1] = file[f"B_cross_{i}"][:]
        
    # Converting the column-major julia indexing into the row-major python indexing
    B_cross[i-1] = np.transpose(B_cross[i-1], (2, 1, 0))

# pB-matrix
pB_cross = utils.obj_array(3)
for i in range(1, 4):
    with h5py.File(file_path_gm, 'r') as file:
        pB_cross[i-1] = file[f"pB_cross_{i}"][:]
        
    # Converting the column-major julia indexing into the row-major python indexing
    pB_cross[i-1] = np.transpose(pB_cross[i-1], (2, 1, 0))

# C_matrix
C_cross = utils.obj_array(4)
for i in range(1, 5):
    with h5py.File(file_path_gm, 'r') as file:
        C_cross[i-1] = file[f"C_cross_{i}"][:]

# D-matrix
D_cross = utils.obj_array(3)
for i in range(1, 4):
    with h5py.File(file_path_gm, 'r') as file:
        D_cross[i-1] = file[f"D_cross_{i}"][:]
        
# pD-matrix
pD_cross = utils.obj_array(3)
for i in range(1, 4):
    with h5py.File(file_path_gm, 'r') as file:
        pD_cross[i-1] = file[f"pD_cross_{i}"][:]
        
################################
### Creating cross val agent ###
################################

cross_agent = Agent(A = A_cross, B = B_cross, C = C_cross, D = D_cross, pA = pA_cross, pB = pB_cross, pD = pD_cross, policy_len = 4, action_selection="deterministic", lr_pA = 0.5, lr_pB = 0.5, lr_pD = 0.5, use_states_info_gain=True, use_param_info_gain=True, save_belief_hist=True)

#############################################
### Creating and initialising environment ###
#############################################

grid_dims = [5, 7]
num_grid_points = np.prod(grid_dims) 

grid = np.arange(num_grid_points).reshape(grid_dims, order='F')

it = np.nditer(grid, flags=["multi_index"])

loc_list = []
while not it.finished:
    loc_list.append(it.multi_index)
    it.iternext()
    
cue1_location = (2, 0)

cue2_loc_names = ['L1', 'L2', 'L3', 'L4']
cue2_locations = [(0, 2), (1, 3), (3, 3), (4, 2)]

cue1_names = ['Null'] + cue2_loc_names
cue2_names = ['Null', 'reward_on_top', 'reward_on_bottom']
reward_names = ['Null', 'Cheese', 'Shock']

reward_conditions = ["TOP", "BOTTOM"]
reward_locations = [(1, 5), (3, 5)]

actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

# Using a custom evironment corresponding to the Julia Epistemic chaining env
class GridWorldEnv():
    
    def __init__(self,starting_loc = (0,0), cue1_loc = (2, 0), cue2 = 'L1', reward_condition = 'TOP'):

        self.init_loc = starting_loc
        self.current_location = self.init_loc

        self.cue1_loc = cue1_loc
        self.cue2_name = cue2
        self.cue2_loc_names = ['L1', 'L2', 'L3', 'L4']
        self.cue2_loc = cue2_locations[self.cue2_loc_names.index(self.cue2_name)]

        self.reward_condition = reward_condition
        print(f'Starting location is {self.init_loc}, Reward condition is {self.reward_condition}, cue is located in {self.cue2_name}')
    
    def step(self,action_label):

        (Y, X) = self.current_location

        if action_label == "UP": 
          
          Y_new = Y - 1 if Y > 0 else Y
          X_new = X

        elif action_label == "DOWN": 

          Y_new = Y + 1 if Y < (grid_dims[0]-1) else Y
          X_new = X

        elif action_label == "LEFT": 
          Y_new = Y
          X_new = X - 1 if X > 0 else X

        elif action_label == "RIGHT": 
          Y_new = Y
          X_new = X +1 if X < (grid_dims[1]-1) else X

        elif action_label == "STAY":
          Y_new, X_new = Y, X 
        
        self.current_location = (Y_new, X_new)

        loc_obs = self.current_location 

        if self.current_location == self.cue1_loc:
          cue1_obs = self.cue2_name
        else:
          cue1_obs = 'Null'

        if self.current_location == self.cue2_loc:
          cue2_obs = cue2_names[reward_conditions.index(self.reward_condition)+1]
        else:
          cue2_obs = 'Null'
        
        if self.current_location == reward_locations[0]:
          if self.reward_condition == 'TOP':
            reward_obs = 'Cheese'
          else:
            reward_obs = 'Shock'
        elif self.current_location == reward_locations[1]:
          if self.reward_condition == 'BOTTOM':
            reward_obs = 'Cheese'
          else:
            reward_obs = 'Shock'
        else:
          reward_obs = 'Null'

        return loc_obs, cue1_obs, cue2_obs, reward_obs

    def reset(self):
        self.current_location = self.init_loc
        print(f'Re-initialized location to {self.init_loc}')
        loc_obs = self.current_location
        cue1_obs = 'Null'
        cue2_obs = 'Null'
        reward_obs = 'Null'

        return loc_obs, cue1_obs, cue2_obs, reward_obs

cross_env = GridWorldEnv(starting_loc = (0,0), cue1_loc = (2, 0), cue2 = 'L4', reward_condition = 'BOTTOM')

# Getting initial observation and setting it to correct format
with h5py.File(file_path_gm, 'r') as file:
    obs = file["obs"][:]
obs = obs - 1

obs_obj_array = np.empty(len(obs), dtype=object)
for i, ob in enumerate(obs):
    obs_obj_array[i] = ob
obs = obs_obj_array

obs = obs.tolist()

##########################
### Running simulation ###
##########################

# Time step set to 50 trials
T = 50

# run simulation
for t in range(T):

    qs = cross_agent.infer_states(obs)

    cross_agent.update_A(obs)
    
    if t != 0:
        qs_prev = cross_agent.qs_hist[t-1]
        cross_agent.update_B(qs_prev)
    
    if t == 0:
        qs_t1 = cross_agent.qs
        cross_agent.update_D(qs_t1)
    
    q_pi, G = cross_agent.infer_policies()
    
    chosen_action_id = cross_agent.sample_action()

    movement_id = int(chosen_action_id[0])

    choice_action = actions[movement_id]

    loc_obs, cue1_obs, cue2_obs, reward_obs = cross_env.step(choice_action)

    obs = [loc_list.index(loc_obs), cue1_names.index(cue1_obs), cue2_names.index(cue2_obs), reward_names.index(reward_obs)]


###########################
### Storing the results ###
###########################

# Storing the variables into the comparison hdf5 file
file_path_results = "../../cross_val_results/complete_run_data.h5"

with h5py.File(file_path_results, "w") as hdf:
  # Storing A-matrix results
  hdf.create_dataset("python_A_cross_1", data=cross_agent.A[0])
  hdf.create_dataset("python_A_cross_2", data=cross_agent.A[1])
  hdf.create_dataset("python_A_cross_3", data=cross_agent.A[2])
  hdf.create_dataset("python_A_cross_4", data=cross_agent.A[3])
  
  # Storing B-matrix results
  hdf.create_dataset("python_B_cross_1", data=cross_agent.B[0])
  hdf.create_dataset("python_B_cross_2", data=cross_agent.B[1])
  hdf.create_dataset("python_B_cross_3", data=cross_agent.B[2])
  
  # Storing D-matrix results
  hdf.create_dataset("python_D_cross_1", data=cross_agent.D[0])
  hdf.create_dataset("python_D_cross_2", data=cross_agent.D[1])
  hdf.create_dataset("python_D_cross_3", data=cross_agent.D[2])
  
  # Storing posterior states
  hdf.create_dataset("python_qs_1", data=cross_agent.qs[0])
  hdf.create_dataset("python_qs_2", data=cross_agent.qs[1])
  hdf.create_dataset("python_qs_3", data=cross_agent.qs[2])





================================================
FILE: test/pymdp_cross_val/cross_val_results/complete_run_data.h5
================================================
[Non-text file]


================================================
FILE: test/pymdp_cross_val/cross_val_results/results_comparison.csv
================================================
parameter,equivalence,to_decimal_place
A,true,7
B,true,8
D,true,14
qs,true,9



================================================
FILE: test/pymdp_cross_val/cross_val_results/results_comparison.jl
================================================
using HDF5
using ActiveInference
using DataFrames
using CSV

########################################################
############### Loading the results data ###############
########################################################

file_path_res = "ActiveInference.jl/test/pymdp_cross_val/cross_val_results/complete_run_data.h5"

#--------------- Loading the complete_run_julia result ----------------
# Loading the julia A matrix
A_julia = array_of_any(4)
for i in 1:4
    A_julia[i] = h5read(file_path_res, "julia_A_cross_$i")
end

# Loading the julia B matrix
B_julia = array_of_any(3)
for i in 1:3
    B_julia[i] = h5read(file_path_res, "julia_B_cross_$i")
end

# Loading the julia D matrix
D_julia = array_of_any(3)
for i in 1:3
    D_julia[i] = h5read(file_path_res, "julia_D_cross_$i")
end

# Loading the julia final posterior over states
qs_julia = array_of_any(3)
for i in 1:3
    qs_julia[i] = h5read(file_path_res, "julia_qs_$i")
end

#--------------- Loading the complete_run_python result ----------------
# Loading the python A matrix
A_python = array_of_any(4)
for i in 1:4
    A_python[i] = h5read(file_path_res, "python_A_cross_$i")
    A_python[i] = permutedims(A_python[i], [4, 3, 2, 1])
end

# Loading the python B matrix
B_python = array_of_any(3)
for i in 1:3
    B_python[i] = h5read(file_path_res, "python_B_cross_$i")
    B_python[i] = permutedims(B_python[i], [3, 2, 1])
end

# Loading the python D matrix
D_python = array_of_any(3)
for i in 1:3
    D_python[i] = h5read(file_path_res, "python_D_cross_$i")
end

# Loading the python final posterior over states
qs_python = array_of_any(3)
for i in 1:3
    qs_python[i] = h5read(file_path_res, "python_qs_$i")
end

############################################################
############### cross-validating the results ###############
############################################################
#------------------ Defining decimal place of agreement function ------------------
function round_arrays(arrays, digits)
    [round.(array, digits=digits) for array in arrays]
end

# Rounding to check A
round_n_A = 15
while round_n_A != 0 && !isequal(A_julia, A_python)

    A_julia = round_arrays(A_julia, round_n_A)
    A_python = round_arrays(A_python, round_n_A)
    round_n_A -= 1
end
round_n_A
is_A_equal = isequal(A_julia, A_python)


# Rounding to check B
round_n_B = 15
while round_n_B != 0 && !isequal(B_julia, B_python)

    B_julia = round_arrays(B_julia, round_n_B)
    B_python = round_arrays(B_python, round_n_B)
    round_n_B -= 1
end
round_n_B
is_B_equal = isequal(B_julia, B_python)

# Rounding to check D
round_n_D = 15
while round_n_D != 0 && !isequal(D_julia, D_python)

    D_julia = round_arrays(D_julia, round_n_D)
    D_python = round_arrays(D_python, round_n_D)
    round_n_D -= 1
end
round_n_D
is_D_equal = isequal(D_julia, D_python)

# Rounding to check qs
round_n_qs = 15
while round_n_qs != 0 && !isequal(qs_julia, qs_python)

    qs_julia = round_arrays(qs_julia, round_n_qs)
    qs_python = round_arrays(qs_python, round_n_qs)
    round_n_qs -= 1
end
round_n_qs
is_qs_equal = isequal(qs_julia, qs_python)

#------------------ Creating a DataFrame to store the results ------------------
results_df = DataFrame(
    parameter = ["A", "B", "D", "qs"],
    equivalence = [is_A_equal, is_B_equal, is_D_equal, is_qs_equal],
    to_decimal_place = [round_n_A, round_n_B, round_n_D, round_n_qs]
)

#------------------ Saving the results ------------------
CSV.write("ActiveInference.jl/test/pymdp_cross_val/cross_val_results/results_comparison.csv", results_df)



================================================
FILE: test/pymdp_cross_val/generative_model_creation/rand_generative_model.jl
================================================
using ActiveInference
using HDF5
using Random

# Setting seed for reproducibility purposes
Random.seed!(246)

##########################################
### Generating random generative model ###
##########################################

# Setting number of states, observations and controls for the generative model
n_states = [35, 4, 2]
n_obs = [35, 5, 3, 3]
n_controls = [5, 1, 1]
policy_length = 1

# Using function for generating A and B matrices with random inputs
A_cross, B_cross = create_matrix_templates(n_states, n_obs, n_controls, policy_length, "random");

# Generating random C matrix
C_cross = array_of_any(4)

C_cross[1] = Float64.(rand(1:10, 35))
C_cross[2] = Float64.(rand(1:10, 5))
C_cross[3] = Float64.(rand(1:10, 3))
C_cross[4] = Float64.(rand(1:10, 3))

# Generating random D matrix
D_cross = array_of_any(3)

D_cross[1] = rand(1:10, 35)
D_cross[2] = rand(1:10, 4)
D_cross[3] = rand(1:10, 2)

D_cross = normalize_arrays(D_cross)

# Setting file path for h5 file containing the dataframes

file_path_gm = "ActiveInference.jl/test/pymdp_cross_val/generative_model_creation/gm_data/gm_matrices.h5"

# Storing the layers for each modality in A matrix in an h5 file. HDF5 can't take an array of arrays 
h5write(file_path_gm, "A_cross_1", A_cross[1])
h5write(file_path_gm, "A_cross_2", A_cross[2])
h5write(file_path_gm, "A_cross_3", A_cross[3])
h5write(file_path_gm, "A_cross_4", A_cross[4])

# Storing the layers for each factor in B matrix in an h5 file. HDF5 can't take an array of arrays 
h5write(file_path_gm, "B_cross_1", B_cross[1])
h5write(file_path_gm, "B_cross_2", B_cross[2])
h5write(file_path_gm, "B_cross_3", B_cross[3])

# Storing the layers for each modality in C matrix in an h5 file. HDF5 can't take an array of arrays
h5write(file_path_gm, "C_cross_1", C_cross[1])
h5write(file_path_gm, "C_cross_2", C_cross[2])
h5write(file_path_gm, "C_cross_3", C_cross[3])
h5write(file_path_gm, "C_cross_4", C_cross[4])

# Storing the layers for each factor in D matrix in an h5 file. HDF5 can't take an array of arrays
h5write(file_path_gm, "D_cross_1", D_cross[1])
h5write(file_path_gm, "D_cross_2", D_cross[2])
h5write(file_path_gm, "D_cross_3", D_cross[3])

#####################################
### Generating random observation ###
#####################################
obs = Int[]
for (i, j) in enumerate(n_obs)
    observation = rand(1:j)
    push!(obs, observation) 
end

h5write(file_path_gm, "obs", obs)

#################################################
### Generating random qs_prev + random action ###
#################################################
qs_prev = array_of_any(3)

for (i, j) in enumerate(n_states)
    qs_prev[i] = rand(1:10, j)
end
qs_prev = normalize_arrays(qs_prev)

h5write(file_path_gm, "qs_prev_1", qs_prev[1])
h5write(file_path_gm, "qs_prev_2", qs_prev[2])
h5write(file_path_gm, "qs_prev_3", qs_prev[3])

action = [rand(1:5), 1, 1]
h5write(file_path_gm, "action", action)

#################################################################
### Generating dirichlet distributions for learning functions ###
#################################################################

# pA
# setting the concentration parameter arbitrarily
pA_cross = deepcopy(A_cross)
for i in 1:length(pA_cross)
    pA_cross[i] = pA_cross[i] .* 10
end

h5write(file_path_gm, "pA_cross_1", pA_cross[1])
h5write(file_path_gm, "pA_cross_2", pA_cross[2])
h5write(file_path_gm, "pA_cross_3", pA_cross[3])
h5write(file_path_gm, "pA_cross_4", pA_cross[4])

# pB
pB_cross = deepcopy(B_cross)
for i in 1:length(pB_cross)
    pB_cross[i] = pB_cross[i] .* 10
end

h5write(file_path_gm, "pB_cross_1", pB_cross[1])
h5write(file_path_gm, "pB_cross_2", pB_cross[2])
h5write(file_path_gm, "pB_cross_3", pB_cross[3])

# pD
pD_cross = deepcopy(D_cross)

for (i, j) in enumerate(n_states)
    pD_cross[i] = pD_cross[i] .* 10
end

h5write(file_path_gm, "pD_cross_1", pD_cross[1])
h5write(file_path_gm, "pD_cross_2", pD_cross[2])
h5write(file_path_gm, "pD_cross_3", pD_cross[3])






================================================
FILE: test/pymdp_cross_val/generative_model_creation/gm_data/gm_matrices.h5
================================================
[Non-text file]


================================================
FILE: test/testsuite/aif_tests.jl
================================================
using IterTools
using LinearAlgebra
using ActiveInference
using Test

""" Test Agent """

@testset "Single Factor Condition - Default Settings" begin

    # Initializse States, Observations, and Controls
    states = [25]
    observations = [25]
    controls = [2]
    policy_length = 1

    # Generate random Generative Model 
    A, B = create_matrix_templates(states, observations, controls, policy_length, "random");

    # Initialize agent with default settings/parameters
    aif = init_aif(A, B);

    # Give observation to agent and run state inference
    observation = [rand(1:observations[i]) for i in axes(observations, 1)]
    QS = infer_states!(aif, observation)

    # Run policy inference
    Q_pi, G = infer_policies!(aif)

    # Sample action
    action = sample_action!(aif)
end


@testset "If There are more factors - Default Settings" begin

    # Initializse States, Observations, and Controls
    states = [64,2]
    observations = [64,2]
    controls = [5,1]
    policy_length = 1

    # Generate random Generative Model 
    A, B = create_matrix_templates(states, observations, controls, policy_length, "random");

    # Initialize agent with default settings/parameters
    aif = init_aif(A, B);

    # Give observation to agent and run state inference
    observation = [rand(1:observations[i]) for i in axes(observations, 1)]
    QS = infer_states!(aif, observation)

    # Run policy inference
    Q_pi, G = infer_policies!(aif)

    # Sample action
    action = sample_action!(aif)
end


@testset "Provide custom settings" begin

    # Initializse States, Observations, and Controls
    states = [64,2]
    observations = [64,2]
    controls = [5,2]
    policy_length = 3

    # Generate random Generative Model 
    A, B, C, D = create_matrix_templates(states, observations, controls, policy_length, "random");

    settings = Dict(
    "policy_len"           => 3,
    "use_states_info_gain" => true,
    "action_selection"     => "deterministic",
    "use_utility"          => true)

    # Initialize agent with custom settings
    aif = init_aif(A, B; C=C, D=D, settings = settings);

    # Give observation to agent and run state inference
    observation = [rand(1:observations[i]) for i in axes(observations, 1)]
    QS = infer_states!(aif, observation)

    # Run policy inference
    Q_pi, G = infer_policies!(aif)

    # Sample action deterministically 
    action = sample_action!(aif)

    # And infer new state
    observation = [rand(1:observations[i]) for i in axes(observations, 1)]
    QS_2 = infer_states!(aif, observation)
end


@testset "Learning with custom parameters" begin

    # Initializse States, Observations, and Controls
    states = [64,2]
    observations = [64,2]
    controls = [5,1]
    policy_length = 2

    # Generate random Generative Model 
    A, B, C, D = create_matrix_templates(states, observations, controls, policy_length, "random");

    # pA concentration parameter
    pA = deepcopy(A)
    for i in eachindex(pA)
        pA[i] .= 1.0
    end

    # pB concentration parameter
    pB = deepcopy(B)
    for i in eachindex(pB)
        pB[i] .= 1.0
    end

    # pD concentration parameter
    pD = deepcopy(D)
    for i in 1:length(D)
        pD[i] .= 1.0
    end

    # Give some settings to agent
    settings = Dict(
        "use_param_info_gain" => true,
        "policy_len" => 2
        )

    # Give custom parameters to agent
    parameters = Dict{String, Real}(
        "lr_pA" => 0.5,
        "fr_pA" => 1.0,
        "lr_pB" => 0.6,
        "lr_pD" => 0.7,
        "alpha" => 2.0,
        "gamma" => 2.0,
        "fr_pB" => 1.0,
        "fr_pD" => 1.0,
        )
    # initialize ageent
    aif = init_aif(A,
                   B; 
                   D = D,
                   pA = pA,
                   pB = pB,
                   pD = pD,
                   settings = settings,
                   parameters = parameters);

    ## Run inference with Learning
    for t in 1:2
        # Give observation to agent and run state inference
        observation = [rand(1:observations[i]) for i in axes(observations, 1)]
        QS = infer_states!(aif, observation)
    
        # # If action is empty, update D vectors
        # if ismissing(get_states(aif)["action"])
        #     QS_t1 = get_history(aif)["posterior_states"][1]
        #     update_D!(aif, QS_t1)
        # end

        # # If agent has taken action, update transition matrices
        # if get_states(aif)["action"] !== missing
        #     QS_prev = get_history(aif)["posterior_states"][end-1]
        #     update_B!(aif, QS_prev)
        # end
        # # Update A matrix
        # update_A!(aif, observation)

        update_parameters!(aif)
    
        # Run policy inference
        Q_pi, G = infer_policies!(aif)
    
        # Sample action
        action = sample_action!(aif)
    end
end



================================================
FILE: test/testsuite/aqua.jl
================================================
using ActiveInference
using Aqua

Aqua.test_all(ActiveInference, ambiguities = false)


================================================
FILE: test/testsuite/utils_tests.jl
================================================
using IterTools
using LinearAlgebra
using ActiveInference
using ActionModels
using Test

""" Test Utils & ActionModels.jl Extensions """

@testset "ActionModels Utils" begin

    # Initializse States, Observations, and Controls
    states = [25]
    observations = [25]
    controls = [2]
    policy_length = 1

    # Generate random Generative Model 
    A, B = create_matrix_templates(states, observations, controls, policy_length, "random");

    # Initialize agent with default settings/parameters
    aif = init_aif(A, B);

    # Set Parameters as dictionary
    params=Dict("lr_pA" => 1.0,
                "fr_pA" => 1.0,
                "lr_pB" => 1.0,
                "lr_pD" => 1.0,
                "alpha" => 1.0,
                "gamma" => 1.0,
                "fr_pB" => 1.0,
                "fr_pD" => 1.0)

    set_parameters!(aif, params)

    # Get states
    get_states(aif)
    get_states(aif, "action")
    get_states(aif, ["action", "prior"])

    # Test get_history
    set_save_history!(aif, true)
    get_history(aif)
    get_history(aif,"action")
    get_history(aif,["action", "prior"])


    # Give individual parameters
    set_parameters!(aif, "lr_pA", 0.5)
    set_parameters!(aif, "fr_pA", 0.5)
    set_parameters!(aif, "lr_pB", 0.5)
    set_parameters!(aif, "lr_pD", 0.5)
    set_parameters!(aif, "gamma", 10.0)
    set_parameters!(aif, "alpha", 10.0)
    set_parameters!(aif, "fr_pB", 0.5)
    set_parameters!(aif, "fr_pD", 0.5)

    # test get_parameters
    get_parameters(aif)
    get_parameters(aif, ["alpha", "gamma"])

end

@testset "Give Inputs and Reset" begin

    # Initializse States, Observations, and Controls
    states = [25]
    observations = [25]
    controls = [2]
    policy_length = 1

    # Generate random Generative Model 
    A, B = create_matrix_templates(states, observations, controls, policy_length, "random");

    # Initialize agent with default settings/parameters
    aif = init_aif(A, B);


    observation = [rand(1:observations[i]) for i in axes(observations, 1)]

    single_input!(aif, observation)

    reset!(aif)


end


@testset "ActionModels Agent and Multiple Factors" begin

    # Initializse States, Observations, and Controls
    states = [25,2]
    observations = [25,2]
    controls = [5,1]
    policy_length = 1

    # Generate random Generative Model 
    A, B = create_matrix_templates(states, observations, controls, policy_length, "random");

    # Initialize agent with default settings/parameters
    aif = init_aif(A, B);

    observation = [rand(1:observations[i]) for i in axes(observations, 1)]

    single_input!(aif, observation)

    reset!(aif)

    agent = init_agent(
    action_pomdp!,
    substruct = aif,
    settings = aif.settings,
    parameters= aif.parameters
)
    inputs =   [[25,1],[24,1]]
    give_inputs!(agent, inputs)
    reset!(agent)

    action_pomdp!(agent, observation)

end




================================================
FILE: .github/agent_output.PNG
================================================
[Non-text file]


================================================
FILE: .github/dependabot.yml
================================================
# Docs: https://docs.github.com/en/github/administering-a-repository/keeping-your-dependencies-updated-automatically
version: 2

updates:
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    target-branch: "dev"
    allow:
      - dependency-type: "all"
    commit-message:
      prefix: ":arrow_up:"


================================================
FILE: .github/workflows/CI_full.yml
================================================
name: CI_full
on:
  push:
    branches:
      - master
    tags: ['*']
  pull_request:
    branches:
      - master
  workflow_dispatch:
concurrency:
  # Skip intermediate builds: always.
  # Cancel intermediate builds: only if it is a pull request build.
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ startsWith(github.ref, 'refs/pull/') }}
jobs:
  test:
    name: Julia ${{ matrix.version }} - ${{ matrix.os }} - ${{ matrix.arch }} - ${{ github.event_name }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        version:
          - '1'
          - 'nightly'
        os:
          - ubuntu-latest
          - macOS-latest
          - windows-latest
        arch:
          - x64
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v2
        with:
          version: ${{ matrix.version }}
          arch: ${{ matrix.arch }}
      - uses: julia-actions/cache@v2
      - uses: julia-actions/julia-buildpkg@v1
      - uses: julia-actions/julia-runtest@v1
      - uses: julia-actions/julia-processcoverage@v1
      - uses: codecov/codecov-action@v5
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          files: lcov.info



================================================
FILE: .github/workflows/CI_small.yml
================================================
name: CI_small
on:
  push:
    branches:
      - dev
      - fitting
    tags: ['*']
  pull_request:
    branches:
      - dev
      - fitting
  workflow_dispatch:
concurrency:
  # Skip intermediate builds: always.
  # Cancel intermediate builds: only if it is a pull request build.
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ startsWith(github.ref, 'refs/pull/') }}
jobs:
  test:
    name: Julia ${{ matrix.version }} - ${{ matrix.os }} - ${{ matrix.arch }} - ${{ github.event_name }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        version:
          - '1'
        os:
          - ubuntu-latest
        arch:
          - x64
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v2
        with:
          version: ${{ matrix.version }}
          arch: ${{ matrix.arch }}
      - uses: julia-actions/cache@v2
      - uses: julia-actions/julia-buildpkg@v1
      - uses: julia-actions/julia-runtest@v1
      - uses: julia-actions/julia-processcoverage@v1
      - uses: codecov/codecov-action@v5
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          files: lcov.info


================================================
FILE: .github/workflows/CompatHelper.yml
================================================
name: CompatHelper
on:
  schedule:
    - cron: 0 0 * * *
  workflow_dispatch:
jobs:
  CompatHelper:
    runs-on: ubuntu-latest
    steps:
      - name: Pkg.add("CompatHelper")
        run: julia -e 'using Pkg; Pkg.add("CompatHelper")'
      - name: CompatHelper.main()
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          COMPATHELPER_PRIV: ${{ secrets.DOCUMENTER_KEY }}
        run: julia -e 'using CompatHelper; CompatHelper.main()'



================================================
FILE: .github/workflows/Documenter.yml
================================================
name: Documentation

on:
  push:
    branches:
      - master
      - dev
    tags: '*'
  pull_request:
    branches:
      - master
      - dev
concurrency:
  group: "${{ github.workflow }} @ ${{ github.ref }}"
  cancel-in-progress: true

jobs:
  docs:
    name: Documentation
    runs-on: macOS-latest
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v2
        with:
          version: '1'
      - uses: julia-actions/julia-buildpkg@v1
      - uses: julia-actions/julia-docdeploy@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          DOCUMENTER_KEY: ${{ secrets.DOCUMENTER_KEY }}
      - run: |
           julia --project=docs/ -e '
            using Documenter: DocMeta, doctest
            using ActiveInference
            DocMeta.setdocmeta!(ActiveInference, :DocTestSetup, :(using ActiveInference); recursive=true)
            doctest(ActiveInference)'


================================================
FILE: .github/workflows/register.yml
================================================
name: Register Package
on:
  workflow_dispatch:
    inputs:
      version:
        description: Version to register or component to bump
        required: true
jobs:
  register:
    runs-on: ubuntu-latest
    steps:
      - uses: julia-actions/RegisterAction@latest
        with:
          token: ${{ secrets.GITHUB_TOKEN }}


================================================
FILE: .github/workflows/TagBot.yml
================================================
name: TagBot
on:
  issue_comment:
    types:
      - created
  workflow_dispatch:
    inputs:
      lookback:
        default: 3
permissions:
  actions: read
  checks: read
  contents: write
  deployments: read
  issues: read
  discussions: read
  packages: read
  pages: read
  pull-requests: read
  repository-projects: read
  security-events: read
  statuses: read
jobs:
  TagBot:
    if: github.event_name == 'workflow_dispatch' || github.actor == 'JuliaTagBot'
    runs-on: ubuntu-latest
    steps:
      - uses: JuliaRegistries/TagBot@v1
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          ssh: ${{ secrets.DOCUMENTER_KEY }}

