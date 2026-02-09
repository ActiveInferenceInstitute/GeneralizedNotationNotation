# Muscle Memory for GNN: Caching Behaviors in Generalized Notation Notation Agents and Processes

## Introduction to GNN and the Need for Caching

[Generalized Notation Notation (GNN)](../gnn/gnn_overview.md) is a standardized, text-based language designed to express Active Inference generative models with precision and clarity. It provides a structured framework for representing cognitive models in a way that is human-readable, machine-parsable, and supports interoperability and reproducibility. GNN files typically describe components like state spaces, probabilistic connections, and equations, forming the basis for textual, graphical, and [executable cognitive models](../gnn/about_gnn.md#the-triple-play-modalities-of-gnn).

The GNN framework often involves complex processing pipelines (e.g., parsing, [type checking](../gnn/gnn_tools.md#gnn-type-checker-and-resource-estimator), visualization, [rendering into executable code](../gnn/gnn_implementation.md#organizing-gnn-processing-tools-a-practical-approach)) and can define sophisticated agents, potentially integrating with [Large Language Models (LLMs) in neurosymbolic architectures](../gnn/gnn_llm_neurosymbolic_active_inference.md). Many of these operations can be repetitive.

This is where `muscle-mem` comes in.

`muscle-mem` is a behavior cache for AI agents and processing tools. It records "tool-calling" patterns (which, in the GNN context, could be pipeline script executions or GNN-defined agent actions) as they solve tasks. It can then deterministically replay these learned trajectories whenever the same task (with the same initial conditions) is encountered again, falling back to the original agent/tool logic if novel conditions or edge cases are detected.

The goal of integrating `muscle-mem` with GNN is to get computationally intensive GNN processing steps or frequently repeated GNN agent behaviors out of the "hot path," leading to:

* **Increased Speed**: Avoid re-computation for known inputs/states.
* **Reduced Variability**: Ensure consistent outputs for identical GNN tasks.
* **Lower Costs**: Save computational resources, especially relevant for complex GNN models or LLM-integrated GNN agents.

This document explores how `muscle-mem` could be applied to the GNN ecosystem. It's unexplored territory, so all feedback is welcome!

* Read [Muscle Mem - Removing LLM calls from Agents](https://pig-dot-dev.github.io/muscle-mem-docs/) for more context on the original `muscle-mem` project.
* Join [Muscle Mem discord](https://discord.gg/N2z5aZzHAc) for feedback.

**Dev Log (Original Muscle-Mem Project)**

* May 7, 2025 - First working demo
* May 8, 2025 - Open sourced

## How Muscle-Mem Works with GNN

`muscle-mem` is not another GNN processing framework or agent architecture. Instead, you would implement your GNN processing tools (like those in the `src/` directory of the GNN project) or your GNN-defined agents as usual, and then plug them into `muscle-mem`'s engine.

When given a GNN-related task (e.g., "visualize `model_X.gnn`" or "simulate agent behavior for observation sequence Y"), the `muscle-mem` engine would:

1. Determine if the GNN context (input file, parameters, agent state) has been seen before (cache-hit), or if it's new (cache-miss) using **Checks**.
2. Perform the task, either:
    * Using the retrieved trajectory (e.g., pre-computed visualization data, cached action sequence) on cache-hit.
    * Passing the task to your GNN tool or GNN-defined agent on cache-miss.
3. Collect GNN tool call events (e.g., outputs of a GNN script, actions taken by a GNN agent) to add to the cache as a new trajectory.

## It's all about Cache Validation in the GNN Context

To safely reuse cached operations for GNN models or processes, the critical question is **cache validation**. Ask yourself:

*For each GNN tool or GNN-defined agent action we might cache, what features of the GNN file, its processing environment, or the agent's state can be used to indicate whether or not it's safe to replay a cached action or result?*

If you can answer this, your GNN tools and agents can have Muscle Memory.

**Examples of GNN-Contextual Features for `Check`s:**

* **For GNN Processing Pipeline Tools** (e.g., [visualization from `8_visualization.py`](../gnn/gnn_tools.md#gnn-visualization), [type-checking from `5_type_checker.py`](../gnn/gnn_tools.md#gnn-type-checker-and-resource-estimator)):
  * Hash (e.g., SHA256) of the input `.gnn` file(s).
  * Specific command-line arguments or configuration parameters used for the tool (e.g., output format, strictness flags, resource estimation flags).
  * Version hash of the GNN processing script itself.
  * Hashes of critical dependency files, like the [ontology term definitions (`act_inf_ontology_terms.json`)](../gnn/ontology_system.md) if the tool uses them.
* **For GNN-Defined Agents (Executable Models)**:
  * The current sensory observation(s) `o` provided to the agent.
  * The agent's current internal belief state `Q(s)` (or a hash/summary of it).
  * The specific policy `π` being evaluated or executed.
  * Relevant parameters of the GNN model if they are considered static for the cached behavior (e.g., specific values in A, B, D, or C matrices from the [GNN file structure](../gnn/gnn_file_structure_doc.md)).
  * The current time step `t` in a dynamic simulation if behavior is time-dependent.
* **For Neurosymbolic GNN-LLM Systems**:
  * The specific query or prompt from the LLM to the GNN component.
  * A hash or summary of the relevant LLM dialogue history or context.
  * The current state of the GNN model being interrogated by the LLM.

### Advanced `Check` Strategies for GNN

While full GNN file hashing is a straightforward approach, more nuanced `Check` strategies can provide better cache granularity and robustness:

* **Section-Specific Hashing**: For tools that only operate on or are affected by specific sections of a GNN file (e.g., an equation renderer only caring about the `## Equations` section, or an ontology validator only caring about `## ActInfOntologyAnnotation` and `## StateSpaceBlock`), the `capture` callback could parse the GNN file and hash only the relevant sections. This makes the cache insensitive to changes in irrelevant parts (e.g., `## ModelAnnotation`).
* **Dependency Tracking**:
  * **Included GNN Files**: If GNNs can include/reference other GNN files (a potential future GNN feature), the `Check` mechanism would need to recursively hash all dependent files.
  * **External Resources**: For tools relying on external files like `act_inf_ontology_terms.json` or other configuration data, hashes of these dependencies should be part of the captured context.
* **Handling GNN Dynamics and Time**:
  * If a GNN tool's output is sensitive to the `## Time` section (e.g., `ModelTimeHorizon`), this section must be included in the `Check`.
  * For agent simulations, if caching decisions at specific time steps, the current time `t` is a critical part of the `Check`'s capture.
* **Configuration-Aware Caching**: For GNN processing tools (like those in the `src/main.py` pipeline), the `Check` must capture all command-line flags or configuration parameters that affect the tool's behavior and output (e.g., `--strict` mode for type checking, output formats for visualization).
* **Ignoring "Volatile" GNN Sections**: Some GNN sections like `## Signature` or free-text comments might be intended to change without affecting the functional output of many tools. `Check`s could be designed to explicitly exclude these sections from the captured state.

Developing these advanced `Check`s requires a deeper understanding of each GNN tool's specific dependencies and sensitivities.

## The API (Conceptualized for GNN)

### Installation (Original `muscle-mem`)

```bash
pip install muscle-mem
```

### Engine

The `Engine` wraps your GNN tool/script or GNN-defined agent and serves as the primary executor of GNN-related tasks. It manages its own cache of previous GNN operation trajectories.

```python
from muscle_mem import Engine

engine = Engine()

# Example: Your GNN visualization script/function
# def generate_gnn_visualization(gnn_file_path: str, output_format: str):
#     # ... GNN visualization logic ...
#     print(f"Generating {output_format} for {gnn_file_path}")
#     # ... returns path to output or content ...

# engine.set_agent(generate_gnn_visualization) # 'agent' here means the GNN tool

# Calling the GNN tool directly:
# generate_gnn_visualization("my_model.gnn", "svg")

# Calling through the engine:
# engine("my_model.gnn", "svg") # Potential cache miss first time
# engine("my_model.gnn", "svg") # Potential cache hit second time
```

### Tool

The `@engine.tool()` decorator instruments GNN operations (functions, script entry points, agent actions) so their invocations and results are recorded by the engine.

```python
from muscle_mem import Engine

engine = Engine()

# Example: A hypothetical GNN model processing step
# @engine.tool() 
# def run_gnn_type_check(gnn_file_path: str, is_strict_mode: bool):
#     # ... logic to perform GNN type checking ...
#     result = f"Type check for {gnn_file_path} (strict: {is_strict_mode}): PASSED"
#     print(result)
#     return result
 
# run_gnn_type_check("another_model.gnn", True) 
# Invocation and its result 'PASSED' would be stored by muscle-mem.
```

### Check

The `Check` is fundamental for cache validation in GNN. It determines if it's safe to reuse a cached GNN operation's result or a GNN agent's action sequence.

Each `Check` encapsulates:

* A `capture` callback: To extract relevant features from the current GNN environment (e.g., GNN file content, agent's belief state).
* A `compare` callback: To determine if the current GNN environment matches a cached environment.

```python
Check(
 capture: Callable[P, T], # P are params of the tool, T is the captured state
    compare: Callable[[T, T], Union[bool, float]], # Compares current T with cached T
)
```

You attach `Check`s to each GNN `@engine.tool` to enforce cache validation, either as a `pre_check` (also used for query-time validation) or `post_check`.

**Conceptual GNN `Check` Example:** Caching a GNN file processing step based on file content.

```python
# from dataclasses import dataclass
# import hashlib
# from muscle_mem import Check, Engine

# engine = Engine()

# # Define what features of the GNN context to store for a check
# @dataclass
# class GNNFileContext:
#     file_path: str
#     file_hash: str
#     processing_params: tuple # e.g., output format, flags

# # Capture callback for a GNN file processing tool
# def capture_gnn_file_context(gnn_file_path: str, **kwargs) -> GNNFileContext:
#     try:
#         with open(gnn_file_path, 'rb') as f:
#             content = f.read()
#             current_hash = hashlib.sha256(content).hexdigest()
#     except IOError:
#         current_hash = "ERROR_READING_FILE" # Handle error
#     return GNNFileContext(
#         file_path=gnn_file_path, 
#         file_hash=current_hash,
#         processing_params=tuple(sorted(kwargs.items())) # store other params
#     )

# # Compare callback
# def compare_gnn_file_contexts(current: GNNFileContext, candidate: GNNFileContext) -> bool:
#     # Cache is valid if file hash and processing params match
#     return (current.file_hash == candidate.file_hash and
#             current.processing_params == candidate.processing_params and
#             current.file_hash != "ERROR_READING_FILE")


# # Decorate a hypothetical GNN tool
# @engine.tool(pre_check=Check(capture_gnn_file_context, compare_gnn_file_contexts))
# def process_gnn_file_somehow(gnn_file_path: str, output_format: str = "svg"):
#     # ... actual GNN processing logic ...
#     print(f"Processing {gnn_file_path} to {output_format}...")
#     # time.sleep(0.1) # Simulate work
#     return f"Output for {gnn_file_path} as {output_format}"
```

## Putting It All Together: A GNN Tool Caching Scenario

Let's combine these ideas for a GNN tool. Imagine a function that renders a GNN model's equations using LaTeX, a potentially time-consuming step if the GNN file is large or equations are complex.

```python
# from dataclasses import dataclass
# from muscle_mem import Check, Engine
# import hashlib
# import time # For simulating work and cache expiration logic

# engine = Engine()

# # Environment features for caching a GNN equation rendering tool
# @dataclass
# class GNNEquationRenderContext:
#     gnn_file_hash: str
#     target_section: str # e.g., "Equations"
#     # We could add a timestamp for time-based cache invalidation
#     # captured_at: float 

# def capture_render_context(gnn_file_path: str, section_name: str = "Equations") -> GNNEquationRenderContext:
#     file_hash = "INVALID_HASH"
#     try:
#         with open(gnn_file_path, 'rb') as f:
#             content = f.read()
#             # In a real GNN parser, you'd extract only the 'Equations' section
#             # For simplicity, we'll hash the whole file for this example.
#             file_hash = hashlib.sha256(content).hexdigest()
#     except Exception as e:
#         print(f"Error reading/hashing GNN file: {e}")
#     return GNNEquationRenderContext(
#         gnn_file_hash=file_hash,
#         target_section=section_name,
#         # captured_at=time.time()
#     )

# def compare_render_contexts(current: GNNEquationRenderContext, candidate: GNNEquationRenderContext) -> bool:
#     # Cache is valid if GNN file hash and target section match
#     # and the hash is valid
#     # Example time-based invalidation:
#     # time_diff = current.captured_at - candidate.captured_at
#     # if time_diff > 60: return False # Expire after 60 seconds
#     return (current.gnn_file_hash == candidate.gnn_file_hash and
#             current.target_section == candidate.target_section and
#             current.gnn_file_hash != "INVALID_HASH")

# @engine.tool(pre_check=Check(capture_render_context, compare_render_contexts))
# def render_gnn_equations_to_latex(gnn_file_path: str, section_name: str = "Equations") -> str:
#     print(f"Rendering LaTeX for section '{section_name}' from {gnn_file_path}...")
#     # Simulate actual LaTeX rendering work based on GNN content
#     # In a real scenario, this would involve parsing the GNN, finding the
#     # 'Equations' section (see gnn_dsl_manual.md), and formatting it.
#     # time.sleep(1) # Simulate work
#     # For this example, return a placeholder based on hash and section
#     ctx = capture_render_context(gnn_file_path, section_name)
#     if ctx.gnn_file_hash == "INVALID_HASH":
#         return "Error: Could not render equations."
#     return f"LaTeX_Output_for_{ctx.gnn_file_hash[:8]}_section_{section_name}"

# # Dummy agent function that uses the GNN tool
# def my_gnn_pipeline_agent(gnn_file_to_process: str):
#    print(f"--- Running GNN Pipeline for {gnn_file_to_process} ---")
#    latex_output = render_gnn_equations_to_latex(gnn_file_to_process)
#    print(f"Generated LaTeX: {latex_output}")
#    # ... other pipeline steps could follow ...

# engine.set_agent(my_gnn_pipeline_agent)

# # Create a dummy GNN file for testing
# DUMMY_GNN_FILE = "dummy_model.gnn"
# with open(DUMMY_GNN_FILE, "w") as f:
#     f.write("## ModelName\nMy Dummy Model\n\n## Equations\ns_t = A * o_t + B * s_{t-1}")

# # Run once - expect cache miss for render_gnn_equations_to_latex
# print("\\nRunning pipeline first time:")
# cache_hit = engine(DUMMY_GNN_FILE) # engine calls my_gnn_pipeline_agent
# assert not cache_hit # The 'agent' itself is new, its internal tool call is also new

# # Run again with same file - expect cache hit for render_gnn_equations_to_latex
# print("\\nRunning pipeline second time (expect cache hit for rendering tool):")
# cache_hit = engine(DUMMY_GNN_FILE)
# assert cache_hit # The 'agent' is now cached, so its tool calls can be replayed

# # Modify the GNN file content
# with open(DUMMY_GNN_FILE, "w") as f:
#     f.write("## ModelName\nMy Modified Model\n\n## Equations\ns_t = A * o_t") # Changed equation

# # Run again - expect cache miss for render_gnn_equations_to_latex due to content change
# print("\\nRunning pipeline after GNN file modification (expect cache miss for rendering tool):")
# cache_hit = engine(DUMMY_GNN_FILE)
# assert not cache_hit

# # Clean up default file
# # import os
# # os.remove(DUMMY_GNN_FILE)
```

*(The Python code above is illustrative and simplified. A full GNN parser and specific section extraction would be needed for `capture_render_context` in a real implementation.)*

### Fallback Implementation

## Use Cases for Muscle-Mem in GNN

### 1. Caching GNN Processing Pipeline Steps

Many GNN workflows, like the one orchestrated by `src/main.py` in the GNN project, involve multiple processing steps (parsing, type checking, visualization, export, ontology mapping, rendering).

* **Target**: Individual scripts/tools within the GNN pipeline (e.g., `5_type_checker.py`, `8_visualization.py`, `11_render.py`). If a script takes a GNN file and some parameters, and produces deterministic output files or logs.
* **Benefit**: Avoid re-running computationally expensive analyses if the input GNN file and relevant parameters haven't changed. This can drastically speed up development cycles and batch processing.
* **`Check`s**:
  * `capture`: Hash of the input `.gnn` file content, command-line arguments passed to the script (e.g., `--strict`, `--estimate-resources` for the type checker), version of the script itself.
  * `compare`: Strict equality of these captured features.
  * The "cached trajectory" would be the set of output files and console output generated by the script.

### 2. Caching GNN-Defined Agent Behaviors

GNN can define executable cognitive models, often within an Active Inference loop where agents perceive, infer, and act.

* **Target**: An agent specified by a `.gnn` file that is being simulated. Cache could apply to the agent's action selection given a certain state and observation, or even sequences of state-action pairs.
* **Benefit**: For agents in deterministic environments, if they encounter an identical sequence of observations while in an identical internal belief state, their subsequent actions might also be identical. Caching these action sequences or policy choices can speed up simulations or deployments of GNN agents in repetitive scenarios.
* **`Check`s**:
  * `capture`: The agent's current sensory input `o_t`, its internal belief state `Q(s_t)` (or a compact representation/hash), and potentially fixed parameters of its GNN model (e.g., A, B, D, C matrices if they are static for the episode). For policy evaluation, the specific policy `π` being considered.
  * `compare`: Equivalence of observations and belief states. For policy evaluation, if the chosen policy for a given `Q(s_t)` and `o_t` is already cached.
  * The "cached trajectory" would be the selected action or sequence of actions.

### 3. Caching Interactions in Neurosymbolic GNN-LLM Systems

As described in `gnn_llm_neurosymbolic_active_inference.md`, LLMs can interact with GNN models (e.g., querying the GNN's belief state, using GNN for structured reasoning, or proposing GNN model updates).

* **Target**: The interaction points between an LLM and a GNN-defined model. For example, if the LLM repeatedly asks the GNN to make predictions based on similar inputs, or if the LLM uses the GNN to evaluate similar hypothetical scenarios.
* **Benefit**: Reduce latency and computational cost (both LLM tokens and GNN inference) for recurring patterns of LLM-GNN interaction.
* **`Check`s**:
  * `capture`: The LLM's query to the GNN (potentially embedded or summarized), the relevant GNN state variables being accessed or modified, key elements of the LLM's conversational context.
  * `compare`: Semantic similarity of LLM queries (if a robust measure is available) and equality of relevant GNN states.
  * The "cached trajectory" would be the GNN's response (e.g., predicted state, EFE value) or the outcome of the GNN model update.

### 4. Advanced GNN Agent Caching Scenarios

Beyond caching simple tool outputs or full agent trajectories given identical initial states, `muscle-mem` could be applied to more specific aspects of GNN-defined agent behavior:

* **Caching Policy Evaluation / Expected Free Energy (EFE) Landscapes**:
  * **Context**: In Active Inference agents defined by GNN, calculating Expected Free Energy (EFE) for various policies `π` given a belief state `Q(s)` is a core, often computationally intensive, step. The GNN specifies the components needed: `P(s'|s,π)` (B-matrices), `P(o|s')` (A-matrices), and preferences `P(o)` (C-vector).
  * **Caching Target**: The calculated EFE value `G(π)` for a specific policy `π` and belief state `Q(s)`.
  * **`Check`s**:
    * `capture`: A representation of the current belief state `Q(s)` (e.g., a hash of the probability distribution), the specific policy `π` being evaluated (or its hash), and hashes of the relevant GNN model parameters (A, B, C matrices) if they can change.
    * `compare`: Equivalence of the captured `Q(s)`, `π`, and model parameters.
  * **Benefit**: If an agent frequently re-evaluates the same policies from similar belief states, caching EFE values can significantly speed up decision-making. This is especially relevant if the EFE landscape is complex but relatively stable for parts of the state space.

* **Caching for Multi-Agent GNN Systems**:
  * **Context**: As outlined in `gnn_multiagent.md`, GNN can define systems of interacting agents. An agent's optimal action often depends on the state or actions of other agents and the shared environment.
  * **Caching Target**: An individual agent's action, or a joint action of a subset of agents, given a combined state.
  * **`Check`s**:
    * `capture`: The individual agent's belief state `Q(s_i)`, its observation `o_i`, relevant parts of the shared environment state, and potentially a summary/hash of recent messages received on communication channels (e.g., `GreetingChannel.input` from the example). For joint actions, this would extend to all involved agents.
    * `compare`: Equivalence of all captured elements.
  * **Benefit**: In MAS with recurring interaction patterns or stable sub-systems, caching can reduce redundant computation for established protocols or responses.
  * **Challenge**: The state space for `Check`s can become very large in MAS, requiring careful feature selection for the `capture` callback.

* **Hierarchical Caching of GNN Agent Behaviors**:
  * **Context**: GNN agents might operate with hierarchical policies, where a high-level policy (e.g., "make coffee") decomposes into a sequence of lower-level actions (e.g., "find cup," "go to machine," "press button").
  * **Caching Target**: Both the selection of high-level policies and the execution of their constituent low-level action sequences.
  * **`Check`s**:
    * High-level: Based on broader goals and environmental context.
    * Low-level: Based on the current sub-goal and more immediate sensory inputs/states.
  * **Benefit**: Allows caching at different levels of abstraction, improving efficiency for both strategic and tactical decision-making. If a high-level policy is cached, its entire known successful sub-action sequence could be replayed.

## Challenges and Future Work for GNN `muscle-mem` Integration

While the application of `muscle-mem` to the GNN ecosystem is promising, several challenges and areas for future development exist:

* **Defining Robust and Granular `Check`s**:
  * **Sensitivity vs. Specificity**: GNN's flexibility, particularly in free-text sections (`## ModelAnnotation`, `## InitialParameterization` descriptions, comments) or detailed LaTeX `## Equations`, makes it challenging to design `Check`s. A `Check` that hashes the entire file might be too sensitive, triggering cache misses for irrelevant changes. Conversely, a `Check` that ignores too much might lead to incorrect cache hits.
  * **Structural Awareness**: Ideal `Check`s might need to be "GNN-aware," capable of parsing the GNN structure and focusing on semantically meaningful components (e.g., changes in variable dimensions in `## StateSpaceBlock`, or altered graph structure in `## Connections`). This implies tighter integration with GNN parsing tools.

* **Cache Granularity and Management**:
  * **Whole vs. Partial Caching**: Deciding whether to cache the output of an entire GNN processing script (e.g., a full visualization) or finer-grained results from functions within that script. Finer granularity offers more reuse but increases `muscle-mem` overhead.
  * **Cache Inspection and Debugging**: As the cache grows, tools will be needed for users to inspect cached GNN trajectories, understand why a specific cache entry was used or not, and manually invalidate entries if necessary.
  * **Cache Size**: GNN models and their outputs (e.g., extensive simulation logs, high-resolution visualizations) can be large. Efficient storage and retrieval strategies for the `muscle-mem` cache database will be important.

* **Handling Dynamic GNN Structures and Learning**:
  * If GNN models are themselves being modified by other processes (e.g., structural learning by an LLM agent, parameter updates that significantly alter behavior beyond what a simple `Check` on parameters might capture), `muscle-mem`'s caching logic needs to be carefully considered. Cache entries might need to be invalidated or associated with specific model versions.

* **Semantic Caching for GNN-LLM Systems**:
  * For interactions between LLMs and GNNs, `Check`s based purely on syntactic equality of LLM prompts or GNN state string representations might be insufficient.
  * Future work could explore "semantic `Check`s" that use embedding comparisons or other NLP techniques to determine if a new LLM query is semantically similar enough to a cached one to reuse the GNN's response. This is a complex research area in itself.

* **Impact of GNN and Tool Versioning**:
  * Changes to the GNN specification itself, or to the GNN processing tools being wrapped by `muscle-mem`, could invalidate large portions of the cache. Strategies for version-aware caching will be necessary.

* **User Experience and Integration**:
  * Seamlessly integrating `muscle-mem` into the existing GNN workflow (e.g., `src/main.py` and its constituent scripts) requires careful API design and consideration of how users will enable and configure caching for different GNN tasks.

Addressing these challenges will likely involve a combination of advancing `muscle-mem`'s core capabilities and developing GNN-specific extensions or best practices for its use.

## Conclusion

`muscle-mem` offers a promising approach to optimize GNN-based workflows and agent behaviors by caching repetitive operations. By carefully defining `Check`s based on relevant features of GNN files, processing parameters, and agent states, it's possible to significantly improve the speed, efficiency, and consistency of GNN applications, from development tooling to deployed intelligent agents. The key lies in robust cache validation tailored to the specific GNN context. Further exploration is needed to develop best practices and concrete implementations for these GNN use cases.

---
*Disclaimer: The GNN-specific code examples for `muscle-mem` provided in this document are conceptual and illustrative. Actual implementation would require integration with GNN parsing libraries and careful consideration of the GNN processing lifecycle.*
