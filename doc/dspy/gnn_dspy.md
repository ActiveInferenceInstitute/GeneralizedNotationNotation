# DSPy: A Framework for Programming-not Prompting-Language Models

DSPy represents a paradigm shift in how developers work with Large Language Models (LLMs), moving from traditional prompt engineering to a more systematic, programmatic approach. As a declarative framework developed by Stanford NLP, DSPy enables the building of modular AI software with improved reliability, maintainability, and portability across different language models and strategies. This comprehensive technical report explores the architecture, components, features, and real-world applications of this innovative framework.

## Core Concepts and Architecture

DSPy fundamentally separates the flow of AI programs from their parameters, allowing developers to focus on high-level logic while optimizers handle the fine-tuning details. This separation represents a significant advancement over traditional prompt engineering methods, where developers spend considerable time crafting and tweaking prompts.

### Signatures: Defining Input-Output Behavior

Signatures are foundational to DSPy's declarative approach, serving as specifications that define the semantic roles of inputs and outputs in a clean format[13]. They function similarly to function signatures in traditional programming languages but are specifically designed for natural language tasks.

```python
# Simple question answering signature
"question -> answer"

# Retrieval-based QA signature
"context: list[str], question: str -> answer: str"

# Multiple-choice with reasoning signature
"question, choices: list[str] -> reasoning: str, selection: int"
```

These signatures provide a clear contract for what a module should do, without specifying how it should be accomplished[8]. This decoupling allows DSPy to swap different implementation strategies while maintaining the same interface.

### Modules: Building Blocks for LLM Programs

A DSPy module serves as a building block for programs that use LLMs, abstracting various prompting techniques like chain of thought or ReAct[8]. Each built-in module generalizes a prompting technique to handle any signature, making them remarkably flexible and reusable.

Modules have learnable parameters (the components of prompts and LM weights) and can be called to process inputs and return outputs. Multiple modules can be composed to create larger programs, similar to neural network modules in PyTorch but applied to language model programs[8].

Key modules in the DSPy ecosystem include:

1. **dspy.Predict**: The most fundamental module from which all other DSPy modules are built. It takes a signature and directly maps inputs to outputs[10].

2. **dspy.ChainOfThought**: A module that reasons step by step to predict task outputs, improving performance on complex reasoning tasks[10].

3. **dspy.ProgramOfThought**: Runs Python programs to solve problems, leveraging code execution for computational tasks[11].

4. **dspy.Refine**: Refines predictions by running a module multiple times with different temperatures, selecting the best result based on a reward function[12].

5. **dspy.ReAct**: Implements agent-based interactions, allowing for reasoning and action loops when solving complex tasks[1].

### Optimizers: Tuning Prompts and Weights

DSPy optimizers (formerly called teleprompters) are algorithms that tune the parameters of DSPy programs to maximize specified metrics[9]. They represent one of DSPy's most powerful features, automating what would otherwise be tedious manual prompt engineering.

An optimizer typically takes three inputs:

- A DSPy program (single module or complex multi-module program)
- A metric function that evaluates program output
- Training inputs [which can be as few as 5-10 examples](9)

Different optimizers in DSPy work through various mechanisms:

1. **dspy.BootstrapRS**: Synthesizes effective few-shot examples for modules[9].

2. **dspy.MIPROv2** (Multiprompt Instruction Proposal Optimizer Version 2): Proposes and explores better natural language instructions for prompts through a three-stage process:
   - Bootstrapping stage: Collects traces of input/output behavior
   - Grounded proposal stage: Drafts potential instructions
   - Discrete search stage: Tests combinations of instructions and traces[9][16]

3. **dspy.BootstrapFinetune**: Builds datasets for modules and uses them to finetune LM weights[9][17].

4. **dspy.BetterTogether**: Combines multiple optimization strategies to achieve better results[18].

## Implementation and Technical Details

DSPy's architecture allows for seamless integration with various LLM providers and supports both local and cloud-based model deployments.

### Getting Started and Setup

Installation is straightforward through pip:

```python
uv pip install -U dspy
```

DSPy supports multiple LLM providers, including:

- OpenAI (via API key)
- Anthropic (via API key)
- Databricks (via SDK or API)
- Ollama (local deployment)
- SGLang (local deployment)
- Any provider supported by LiteLLM[1]

The configuration process typically involves setting up a language model and configuring DSPy to use it:

```python
import dspy
lm = dspy.LM('openai/gpt-4o-mini', api_key='YOUR_API_KEY')
dspy.configure(lm=lm)
```

### Working with Modules

Using DSPy modules follows a consistent pattern: declare the module with a signature, call it with input arguments, and access the outputs[8]:

```python
# Declare with a signature
classify = dspy.ChainOfThought('sentence -> sentiment: bool')

# Call with input argument
response = classify(sentence="it's a charming and often affecting journey.")

# Access the output
print(response.sentiment)
```

This pattern makes working with DSPy modules intuitive and consistent across different types of modules and language models.

## Optimization Capabilities

DSPy's optimization capabilities represent a significant advancement over traditional prompt engineering methods.

### The MIPROv2 Optimization Process

The MIPROv2 optimizer exemplifies DSPy's sophisticated approach to prompt optimization. It operates in three distinct stages:

1. **Bootstrapping**: The optimizer takes an unoptimized program and runs it multiple times across different inputs to collect traces of input/output behavior. It then filters these traces to keep only those that appear in trajectories with high metric scores[1][9].

2. **Grounded Proposal**: MIPRO previews the DSPy program's code, data, and execution traces, using them to draft potential instructions for every prompt in the program[1][9].

3. **Discrete Search**: The optimizer samples mini-batches from the training set, proposes combinations of instructions and traces for each prompt in the pipeline, and evaluates candidates on the mini-batch. Using the resulting score, it updates a surrogate model that helps improve proposals over time[1].

A typical optimization run costs around $2 USD and takes approximately 20 minutes, though costs can vary based on the language model, dataset size, and configuration[1].

### Composable Optimization

One powerful feature of DSPy optimizers is their composability. For example, developers can:

- Run `dspy.MIPROv2` and use the output as input to another run of `dspy.MIPROv2`
- Feed the output of `dspy.MIPROv2` into `dspy.BootstrapFinetune` for further improvement
- Extract the top-5 candidate programs from an optimizer and build a `dspy.Ensemble` of them[1]

This allows for scaling both inference-time compute (e.g., ensembles) and DSPy's unique pre-inference time compute (optimization budget) in systematic ways[1].

## Real-World Applications and Use Cases

DSPy has been adopted by numerous organizations for diverse applications, demonstrating its versatility and effectiveness in real-world scenarios[14][15].

### Enterprise Applications

- **JetBlue**: Multiple chatbot use cases
- **Replit**: Synthesizing code diffs using LLMs
- **Databricks**: Research, products, and customer solutions around LM Judges, RAG, and classification
- **Sephora**: Agent use cases
- **VMware**: RAG and prompt optimization applications
- **Zoro UK**: E-commerce applications for structured shopping
- **PingCAP**: Building knowledge graphs[14]

### Specialized Applications

- **Haize Labs**: Automated red-teaming for LLMs
- **Salomatic**: Enriching medical reports
- **TrueLaw**: Bespoke LLM pipelines for law firms
- **STChealth**: Entity resolution with human-readable rationales
- **Moody's**: RAG systems and agentic systems for financial workflows
- **RadiantLogic**: AI Data Assistant that routes queries, extracts context, and handles text-to-SQL conversion[14]

## Benefits and Advantages

DSPy offers several key advantages over traditional approaches to working with LLMs:

### Programmatic Over Prompt Engineering

By shifting from prompt engineering to programmatic definitions, DSPy eliminates the need for brittle, hard-to-maintain prompt strings. This approach makes AI software more reliable and maintainable over time[1][7].

### Portability Across Models

DSPy programs are designed to be portable across different language models and strategies. This allows developers to switch between models without extensive reworking of their applications[1][6].

### Systematic Optimization

Rather than manual trial-and-error prompt tuning, DSPy provides algorithms that systematically optimize prompts and weights based on defined metrics. This leads to more consistent and higher-quality results[9][19].

### Modularity and Composability

The modular architecture of DSPy enables developers to build complex systems from simple components, making development more efficient and maintainable. Modules can be easily composed and reused across different projects[8][21].

## Historical Development and Community

DSPy has evolved significantly since its inception, growing into a vibrant open-source project with substantial community support.

### Origins and Evolution

The DSPy research effort began at Stanford NLP in February 2022, building on earlier systems like ColBERT-QA, Baleen, and Hindsight. Initially released as DSP in December 2022, it evolved into DSPy by October 2023[1].

With 250 contributors, DSPy has introduced tens of thousands of developers to building and optimizing modular LM programs. The community has produced significant work on optimizers (MIPROv2, BetterTogether, LeReT), program architectures (STORM, IReRa, DSPy Assertions), and applications to various problems[1].

### Production Deployment

DSPy is designed for production use with features for:

- Monitoring and observability through MLflow Tracing based on OpenTelemetry
- Reproducibility through logging of programs, metrics, configs, and environments
- Deployment through MLflow Model Serving
- Scalability with thread-safe design and native asynchronous execution support
- Guardrails and controllability through its Signatures, Modules, and Optimizers[15]

## Conclusion

DSPy represents a significant advancement in how developers interact with and build applications on top of large language models. By shifting from brittle prompt engineering to a programmatic, modular approach, it addresses many of the challenges associated with traditional LLM development.

The framework's key innovations-separation of program flow from parameters, declarative signatures, composable modules, and automatic optimization-enable more reliable, maintainable, and portable AI software. These capabilities have made DSPy an increasingly popular choice for both research and production applications across various industries.

As language models continue to evolve, DSPy's approach provides a more systematic and scalable method for leveraging their capabilities while minimizing the brittleness and maintenance challenges of traditional prompt engineering. Its growing community and ecosystem suggest that DSPy will remain an important tool in the AI development landscape for the foreseeable future.

## Integrating DSPy with Generalized Notation Notation (GNN) and Neurosymbolic Active Inference

The principles and tools offered by DSPy align remarkably well with the goals of Generalized Notation Notation (GNN) and the broader framework of Neurosymbolic Active Inference. GNN provides a standardized, text-based language for specifying the formal structure of generative models in Active Inference, facilitating human readability and machine parsability (`gnn_overview.md`). The integration of DSPy can significantly enhance the LLM-driven aspects of such neurosymbolic systems.

As explored in `gnn_llm_neurosymbolic_active_inference.md`, LLMs play crucial roles in a GNN-backed Active Inference agent, including:

- Natural Language Interfacing (understanding instructions, generating explanations).
- Knowledge Grounding and Enrichment (semantic interpretation, common-sense reasoning).
- Parameterization and Initialization suggestions.
- Hypothesis Generation and Model Adaptation proposals.
- Processing Unstructured Data into GNN-compatible formats.

DSPy can provide the programmatic layer to systematically build, optimize, and manage these LLM interactions within the GNN-based cognitive architecture.

### Leveraging DSPy for GNN-LLM Interactions in Active Inference

The core components of DSPy—Signatures, Modules, and Optimizers—can be strategically employed to streamline the integration of LLMs with GNN-specified models.

#### 1. DSPy Signatures for GNN-LLM Interface Definition

GNN defines the structure of generative models, including state spaces, observations, policies, and their relationships (`gnn_syntax.md`). DSPy Signatures can define the expected input-output behavior of LLM components that interface with these GNN structures.

- **Observation Processing**: An LLM tasked with converting raw natural language observations into the symbolic format required by a GNN model (`o` vector) can be defined with a DSPy signature.

    ```python
    # Signature for LLM processing user utterance into GNN observation format
    "user_utterance: str -> symbolic_observation: str" 
    # symbolic_observation would be a string parsable by GNN tools
    ```

- **Goal Translation**: Translating a high-level user goal into GNN-compatible preferences (e.g., the `C` matrix/vector or parameters for the Expected Free Energy calculation) can be managed by an LLM component with a clear signature.

    ```python
    # Signature for translating natural language goal to GNN preference parameters
    "natural_language_goal: str -> gnn_preference_parameters: dict"
    ```

- **Explanation Generation**: An LLM generating natural language explanations from GNN model states or inference traces.

    ```python
    # Signature for LLM explaining GNN model state
    "gnn_model_state: dict, inference_trace: list[str] -> explanation: str"
    ```

These signatures allow for a declarative approach to defining the LLM's role, which DSPy can then optimize.

#### 2. DSPy Modules for Semantic Processing and Reasoning with GNN

DSPy modules (`dspy.Predict`, `dspy.ChainOfThought`, `dspy.ReAct`, etc.) can implement the LLM functionalities that interact with the GNN model at various stages of the Active Inference loop (perception, learning, action selection as per `gnn_llm_neurosymbolic_active_inference.md`).

- **Natural Language Frontend/Backend**:
  - `dspy.Predict` or `dspy.ChainOfThought` can be used to build modules that parse user commands, queries, or general dialogue, translating them into inputs for the GNN model (e.g., setting priors `D`, observations `o`, or context for policy selection).
  - Similarly, these modules can take outputs from the GNN model (e.g., inferred states `s`, chosen policy `π`, or predicted outcomes) and generate natural language responses or explanations.
- **LLM-Assisted GNN Parameterization/Initialization**:
    A DSPy module could take a scenario description and suggest initial parameters for a GNN model (e.g., A, B, D matrices), which could then be formally validated by GNN tools (`gnn_tools.md`).

    ```python
    # Hypothetical DSPy module for GNN initialization
    suggest_gnn_params = dspy.ChainOfThought("scenario_description: str -> gnn_parameters_suggestion: dict")
    ```

- **Semantic Evaluation in Action Selection**:
    As described in `gnn_llm_neurosymbolic_active_inference.md` (Section 4.3.D), LLMs can assist in evaluating the pragmatic and epistemic value components of Expected Free Energy (EFE). DSPy modules can structure these LLM-based evaluations.
  - A module could assess if a GNN-predicted state (under a policy) semantically aligns with a complex user goal.
  - Another module could evaluate which GNN-proposed exploratory action is most likely to resolve semantic ambiguity identified by the LLM.

- **Hypothesis Generation for GNN Model Adaptation**:
    When the GNN model shows persistent errors, an LLM (managed by a DSPy module, possibly `dspy.ProgramOfThought` if it involves suggesting GNN syntax modifications) could propose changes to the GNN structure (new states, new connections) based on its broader knowledge.

#### 3. DSPy Optimizers for Fine-Tuning GNN-LLM Components

DSPy optimizers (e.g., `MIPROv2`, `BootstrapFinetune`) can automatically refine the prompts and even the LLM components that bridge GNN and natural language.

- **Optimizing Observation Parsers**: If an LLM module is responsible for parsing natural language observations into a GNN-compatible format, its prompts can be optimized using examples of (utterance, correct GNN observation string) pairs. The metric could be the accuracy of the GNN parsing or the performance of the Active Inference agent using these parsed observations.
- **Refining Explanation Generators**: The prompts for an LLM module that explains GNN model behavior can be optimized based on human ratings of explanation clarity or completeness.
- **Tuning Goal Translators**: For LLMs translating natural language goals into GNN preference settings, optimizers can improve this translation based on whether the resulting agent behavior (driven by the GNN model with these preferences) successfully achieves the intended high-level goal.

This optimization process moves beyond manual prompt engineering for these interface points, leading to more robust and effective neurosymbolic systems.

### Orchestrating GNN and DSPy Workflows

The GNN project includes a pipeline script (`src/main.py` detailed in `gnn_tools.md`) that orchestrates various processing stages for GNN files (parsing, type checking, visualization, etc.). DSPy-managed LLM programs could be integrated into this pipeline or operate in conjunction with it.

- **LLM-Assisted GNN Authoring**: A DSPy program could assist a user in authoring a GNN file by providing suggestions, auto-completing sections based on high-level descriptions, or translating natural language descriptions of model components into GNN syntax. The `gnn_syntax.md` provides the target syntax for such LLM assistance.
- **Interactive Model Refinement**: A DSPy-powered agent could discuss GNN model validation errors (from `5_type_checker.py`) with a user, helping to debug or refine the GNN specification.
- **Generating GNN from Descriptions**: Inspired by the "Triple Play" concept in `gnn_paper.md` (text, graphical, executable), a DSPy module could attempt to generate a basic GNN text-based model from a detailed natural language description of a cognitive model.

### Benefits of the Integrated GNN-DSPy Approach

Combining GNN's formal model specification with DSPy's programmatic LLM capabilities offers several advantages:

1. **Structured Semantic Grounding**: GNN provides the formal structure, and DSPy-managed LLMs provide the rich semantic grounding and flexible natural language interaction.
2. **Systematic Development and Optimization**: DSPy's optimizers allow for data-driven refinement of the LLM components that interact with GNN, moving beyond ad-hoc prompt engineering.
3. **Enhanced Interpretability and Explainability**: GNN models are inherently interpretable due to their explicit structure. DSPy can manage LLMs that translate this formal structure and its dynamics into human-understandable language.
4. **Modularity and Reusability**: Both GNN and DSPy promote modular design. GNN modules can define parts of a cognitive model, and DSPy modules can define reusable LLM-based reasoning components.
5. **Principled Agent Design**: This integration supports the vision of Neurosymbolic Active Inference agents where formal models (GNN) guide behavior, and LLMs (managed by DSPy) handle the complexities of real-world language and knowledge.

By leveraging DSPy, the development of sophisticated AI agents that combine the strengths of structured probabilistic models (like those specified in GNN for Active Inference) and the versatile capabilities of LLMs becomes more systematic, robust, and powerful.

### Detailed Examples of DSPy Modules for GNN Integration

To further illustrate the synergy, let's consider some more detailed conceptual examples of how DSPy modules could be structured for specific GNN integration tasks. These examples highlight the interplay between DSPy's programmatic LLM approach and GNN's formal model specification.

**1. Observation Parser Module with DSPy and GNN Validation**

- **Goal**: Convert a natural language user observation into a structured GNN observation string, then validate it using GNN tools.
- **DSPy Signature**: `"user_utterance: str -> candidate_gnn_observation_string: str"`
- **DSPy Module**: `dspy.Predict` or `dspy.ChainOfThought` to generate the `candidate_gnn_observation_string`.

    ```python
    # Conceptual DSPy module
    class GNNObservationParser(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate_gnn_string = dspy.ChainOfThought(
                "user_utterance -> candidate_gnn_observation_string",
                # Prompt would instruct LLM on GNN observation syntax, e.g., 'o[idx]={value}'
            )

        def forward(self, user_utterance):
            prediction = self.generate_gnn_string(user_utterance=user_utterance)
            # The candidate_gnn_observation_string is now available
            # Next, this string would be passed to a GNN validation tool/function
            # (e.g., conceptually, validator.validate_observation_string(prediction.candidate_gnn_observation_string))
            # The result of validation could inform further DSPy optimization loops.
            return prediction
    ```

- **GNN Role**: The GNN syntax rules (`gnn_syntax.md`) and validation tools (e.g., part of `5_type_checker.py` in `gnn_tools.md`) define the target format and correctness criteria for the LLM's output. The GNN specification for the particular model would dictate the valid indices and expected value types for observations.

- **DSPy Optimization**: The `GNNObservationParser` module could be optimized using a metric that combines successful GNN parsing and downstream task performance of the Active Inference agent.

**2. LLM-Guided Policy Refinement within GNN-EFE Framework**

- **Goal**: An LLM suggests refinements or priorities for policies being evaluated by a GNN-based Expected Free Energy (EFE) calculation, especially when goals are complex or qualitative.
- **DSPy Signature**: `"gnn_candidate_policies: list[dict], current_world_state_summary: str, high_level_goal: str -> refined_policy_evaluations: list[dict]"` (where `gnn_candidate_policies` might include GNN-calculated EFE components, and `refined_policy_evaluations` would add LLM-derived scores or flags).
- **DSPy Module**: `dspy.Predict` or `dspy.ReAct` (if the LLM needs to ask clarifying questions about the policies or goal).

    ```python
    # Conceptual DSPy module
    class PolicySemanticRefiner(dspy.Module):
        def __init__(self):
            super().__init__()
            # Prompt would instruct LLM to evaluate policies against the semantic goal
            # and how GNN policy descriptions map to real-world actions.
            self.refine_policy_choice = dspy.ChainOfThought(
                "gnn_candidate_policies, current_world_state_summary, high_level_goal -> refined_policy_evaluations"
            )

        def forward(self, gnn_candidate_policies, current_world_state_summary, high_level_goal):
            # GNN provides policies like: {policy_id: 'pi_1', actions: ['action_A', 'action_B'], pragmatic_value_gnn: 0.8, epistemic_value_gnn: 0.3}
            # LLM refines by adding semantic alignment scores or risk assessments.
            # e.g., refined_policy_evaluations could be: [{...policy_1_data, semantic_score: 0.9, safety_flag: 'low_risk'}, ...]
            return self.refine_policy_choice(
                gnn_candidate_policies=str(gnn_candidate_policies), # DSPy expects string inputs for fields generally
                current_world_state_summary=current_world_state_summary,
                high_level_goal=high_level_goal
            )
    ```

- **GNN Role**: The GNN model provides the set of candidate policies, the forward model to predict their outcomes, and the initial EFE calculations. The `gnn_llm_neurosymbolic_active_inference.md` (Section 4.2.C) discusses how GNN defines policies and the EFE components.

- **DSPy Optimization**: Metrics could involve how well the final chosen policy (after LLM refinement) achieves the high-level goal, or user satisfaction with the agent's decision.

**3. Generating GNN Snippets from Natural Language**

- **Goal**: An LLM assists in GNN model authoring by generating a GNN-syntax snippet (e.g., a variable definition, a connection block) from a natural language description.
- **DSPy Signature**: `"component_description: str, gnn_model_context: str -> gnn_snippet_suggestion: str"`
- **DSPy Module**: `dspy.Predict` or `dspy.ProgramOfThought` (as it's generating code-like syntax).

    ```python
    # Conceptual DSPy module
    class GNNSnippetGenerator(dspy.Module):
        def __init__(self):
            super().__init__()
            # Prompt would contain instructions on GNN syntax (`gnn_syntax.md`) and examples.
            self.generate_snippet = dspy.ChainOfThought(
                "component_description, gnn_model_context -> gnn_snippet_suggestion"
            )

        def forward(self, component_description, gnn_model_context):
            # User might describe: "A hidden state for temperature, it's a scalar float."
            # LLM might suggest: "s_temperature[1,1,type=float] ### Represents ambient temperature"
            # The gnn_model_context helps LLM avoid name collisions or suggest relevant connections.
            return self.generate_snippet(
                component_description=component_description, 
                gnn_model_context=gnn_model_context
            )
    ```

- **GNN Role**: `gnn_syntax.md` provides the target language. The GNN parser and validator (`gnn_tools.md`) would check the LLM's output. The broader GNN file provides context.

- **DSPy Optimization**: Metrics could include the syntactic validity of the generated GNN snippet and how well it integrates into the larger GNN model.

These examples demonstrate how DSPy can provide a structured, optimizable layer for the complex LLM reasoning tasks required in a sophisticated GNN-based neurosymbolic Active Inference system.

### Challenges and Future Directions in GNN-DSPy Integration

While the combination of GNN and DSPy holds significant promise, several challenges and exciting future research directions need to be addressed:

1. **Semantic Alignment and Validation**: Ensuring that the LLM's interpretations and generations (managed by DSPy) strictly adhere to the formal semantics of the GNN model is paramount. Misinterpretations can lead to invalid GNN structures or incorrect model dynamics.
    - **Challenge**: How to robustly validate the *semantic* correctness of LLM outputs beyond mere syntactic GNN validity? For instance, an LLM might generate a syntactically correct GNN variable but with a meaning inconsistent with the rest of the GNN model.
    - **Future Work**: Developing DSPy metrics and GNN validation tools that can assess semantic consistency, perhaps by leveraging ontologies (as mentioned in `gnn_overview.md` and `gnn_tools.md` regarding `10_ontology.py`) or by using the LLM itself in a reflective loop to check its own output against the GNN context.

2. **Defining Effective Optimization Metrics for DSPy**: The success of DSPy's optimizers depends on well-defined metrics. In a GNN-DSPy system, these metrics need to capture not just the LLM's performance but also its impact on the GNN model's validity and the overall Active Inference agent's task success.
    - **Challenge**: Crafting metrics that are sensitive to both the LLM's contribution (e.g., quality of a generated GNN snippet) and the GNN model's functional correctness (e.g., does the snippet allow the GNN model to simulate without errors? Does the agent perform better?).
    - **Future Work**: Research into composite metrics, hierarchical optimization strategies (optimizing LLM modules based on GNN validation feedback and then optimizing the whole system on task performance), and methods for credit assignment in these multi-component systems.

3. **Scalability and Computational Cost**: Both LLMs (especially large ones used by DSPy) and complex Active Inference computations (simulating GNN models) can be resource-intensive. Their combination requires careful consideration of efficiency.
    - **Challenge**: Managing the computational load of frequent LLM calls within the GNN processing pipeline or the Active Inference loop.
    - **Future Work**: Exploring techniques for optimizing DSPy programs (e.g., distillation, caching LLM outputs), developing more efficient GNN simulation engines, and designing architectures that strategically invoke LLM components only when necessary (e.g., when GNN model uncertainty is high or when semantic ambiguity is detected).

4. **Structural Learning of GNN Models via DSPy**: A highly ambitious direction is to use DSPy-managed LLMs to propose or even learn the *structure* of GNN models, not just their parameters or interfacing language.
    - **Challenge**: How can an LLM reliably propose meaningful and valid structural changes (new states, factors, or connections) to a GNN model based on high-level descriptions, observed data, or persistent model failures?
    - **Future Work**: Investigating the use of `dspy.ProgramOfThought` or similar modules where the LLM generates GNN syntax representing model modifications. This would require robust GNN validation and simulation tools to evaluate the proposed changes within a DSPy optimization loop. This aligns with ideas of LLMs aiding model adaptation in `gnn_llm_neurosymbolic_active_inference.md`.

5. **Toolchain Interoperability and Workflow Automation**: Seamless integration between DSPy's Python environment and the GNN toolchain (which might involve various scripts and parsers, as seen in `src/main.py` in `gnn_tools.md`) is crucial for practical development.
    - **Challenge**: Ensuring that data flows smoothly between DSPy modules and GNN processing steps, and that DSPy optimizations can effectively use feedback from GNN tools.
    - **Future Work**: Developing standardized APIs or data exchange formats between DSPy and GNN components. Enhancing the GNN `src/main.py` pipeline to allow for easy insertion of DSPy-driven LLM steps.

6. **Handling Partial Specifications and Ambiguity**: Users might provide incomplete or ambiguous descriptions when trying to generate GNN models or components. DSPy-managed LLMs need to handle this gracefully.
    - **Challenge**: How can an LLM query the user for clarification (perhaps using `dspy.ReAct`) in a way that helps resolve ambiguity for GNN specification, without overwhelming the user?
    - **Future Work**: Developing interactive GNN-DSPy systems where the LLM can engage in a dialogue to refine specifications before committing to GNN code generation, ensuring the generated GNN aligns with user intent and GNN's formal requirements.

Addressing these challenges will pave the way for even more powerful and intelligent systems that deeply integrate the strengths of declarative LLM programming with formal, interpretable generative modeling.

Citations:
[1] <https://dspy.ai>
[2] <https://github.com/stanfordnlp/dspy>
[3] <https://dspy.ai/>
[4] <https://dspy.ai>
[5] <https://github.com/stanfordnlp/dspy>
[6] <https://dspy.ai/>
[7] <https://dspy.ai>
[8] <https://docs.databricks.com/aws/en/generative-ai/dspy>
[9] <https://learnbybuilding.ai/tutorials/a-gentle-introduction-to-dspy>
[10] <https://github.com/stanfordnlp/dspy/blob/main/README.md?plain=1>
[11] <https://dspy.ai/learn/programming/modules/>
[12] <https://dspy.ai/learn/optimization/optimizers/>
[13] <https://dspy.ai/api/modules/ChainOfThought/>
[14] <https://dspy.ai/api/modules/ProgramOfThought/>
[15] <https://dspy.ai/api/modules/Refine/>
[16] <https://dspy.ai/api/signatures/Signature/>
[17] <https://dspy.ai/community/use-cases/>
[18] <https://dspy.ai/production/>
[19] <https://dspy.ai/api/optimizers/MIPROv2/>
[20] <https://dspy.ai/api/optimizers/BootstrapFinetune/>
[21] <https://dspy.ai/api/optimizers/BetterTogether/>
[22] <https://www.ibm.com/think/topics/dspy>
[23] <https://github.com/stanfordnlp/dspy>
[24] <https://www.datacamp.com/blog/dspy-introduction>
[25] <https://pyimagesearch.com/2024/09/09/llmops-with-dspy-build-rag-systems-using-declarative-programming/>
[26] <https://dev.to/gabrielvanderlei/dspy-a-new-approach-to-language-model-programming-10lf>
[27] <https://arxiv.org/abs/2310.03714>
[28] <https://dspy.ai/learn/programming/modules/>
[29] <https://www.ibm.com/think/topics/dspy>
[30] <https://cobusgreyling.substack.com/p/an-introduction-to-dspy>
[31] <https://openreview.net/forum?id=sY5N0zY5Od>
[32] <https://tianpan.co/notes/2025-01-24-llm-agents-compound-ai-system-and-dsp>
[33] <https://dspy.ai/learn/>
[34] <https://relevanceai.com/blog/building-self-improving-agentic-systems-in-production-with-dspy>
[35] <https://www.reddit.com/r/LocalLLaMA/comments/1cplfph/who_is_using_dspy/>
[36] <https://dspy.ai/tutorials/>
[37] <https://www.blog.brightcoding.dev/2025/05/03/dspy-programming-language-models-for-modular-ai-systems/>
[38] <https://dspy.ai/learn/programming/overview/>
[39] <https://docs.clarifai.com/integrations/DSPy/modules-signatures/>
[40] <https://mlflow.org/docs/latest/llms/dspy/optimizer>
[41] <https://jamesdhope.com/post/prompt-optimisation/2024-04-7-prompt-optimisation-dspy/>
[42] <https://weave-docs.wandb.ai/guides/integrations/dspy/>
[43] <https://www.youtube.com/watch?v=bzgWbCeC0iw>
[44] <https://x.com/lateinteraction/status/1779348470530441727>
[45] <https://github.com/stanfordnlp/dspy/blob/main/docs/docs/deep-dive/modules/guide.md>
[46] <https://dspy.ai/learn/optimization/overview/>
[47] <https://sourceforge.net/projects/dspy.mirror/files/2.6.17/README.md/download>
[48] <https://docs.databricks.com/aws/en/generative-ai/dspy/>
[49] <https://www.youtube.com/watch?v=BrvVheleOqc>
[50] <https://news.ycombinator.com/item?id=37417698>
[51] <https://dspy.ai/api/modules/ReAct/>
[52] <https://dspy.ai/deep-dive/modules/program-of-thought/>
[53] <https://dspy.ai/api/modules/MultiChainComparison/>
[54] <https://dspy.ai>
[55] <https://github.com/stanfordnlp/dspy/issues/276>
[56] <https://github.com/stanfordnlp/dspy/blob/main/docs/docs/cheatsheet.md>
[57] <https://github.com/stanfordnlp/dspy/blob/main/docs/docs/deep-dive/modules/multi-chain-comparison.md>
[58] <https://www.aidoczh.com/dspy/api/modules/ChainOfThought/ChainOfThought.html>
[59] <https://github.com/stanfordnlp/dspy/blob/main/dspy/predict/react.py>
[60] <https://github.com/stanfordnlp/dspy/blob/main/docs/docs/learn/programming/modules.md>
[61] <https://github.com/stanfordnlp/dspy/blob/main/dspy/predict/multi_chain_comparison.py>
[62] <https://dspy.ai/learn/programming/signatures/>
[63] <https://dspy.ai/api/optimizers/MIPROv2/>
[64] <https://dspy.ai/api/primitives/History/>
[65] <https://dspy.ai/deep-dive/retrieval_models_clients/MyScaleRM/>
[66] <https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine/>
[67] <https://dspy.ai/deep-dive/retrieval_models_clients/FalkordbRM/>
[68] <https://dspy.ai/api/modules/ChainOfThought/>
[69] <https://dspy.ai/api/optimizers/COPRO/>
[70] <https://dspy.ai/api/signatures/InputField/>
[71] <https://dspy.ai/deep-dive/optimizers/Ensemble/>
[72] <https://www.e2enetworks.com/blog/build-with-e2e-cloud-step-by-step-guide-to-use-dspy-to-build-multi-hop-optimized-rag>
[73] <https://learnbybuilding.ai/tutorial/dspy-agents-from-scratch/>
[74] <https://dspy.ai/deep-dive/data-handling/examples/>
[75] <https://github.com/stanfordnlp/dspy>
[76] <https://dspy.ai/tutorials/rag/>
[77] <https://github.com/mbakgun/dspy-examples>
[78] <https://towardsdatascience.com/supercharge-your-llm-apps-using-dspy-and-langfuse-f83c02ba96a1/>
[79] <https://docs.clarifai.com/integrations/DSPy/rag-dspy/>
[80] <https://www.linkedin.com/pulse/ai-agents-agentic-patterns-dspy-mitul-tiwari-kmrsf>
[81] <https://github.com/stanfordnlp/dspy/issues/164>
[82] <https://www.linkedin.com/posts/christianadib_github-stanfordnlpdspy-dspy-the-framework-activity-7178260963731595264-WOoN>
[83] <https://github.com/ganarajpr/awesome-dspy>
[84] <https://github.com/stanfordnlp/dspy/issues>
[85] <https://dspy.ai/tutorials/rl_papillon/>
[86] <https://hai.stanford.edu/research/dspy-compiling-declarative-language-model-calls-into-state-of-the-art-pipelines>
[87] <https://github.com/stanfordnlp/dspy/blob/main/docs/docs/roadmap.md>
[88] <https://pub.towardsai.net/prompt-like-a-pro-using-dspy-a-guide-to-build-a-better-local-rag-model-using-dspy-qdrant-and-d8011a3942d9>
[89] <https://www.reddit.com/r/LocalLLaMA/comments/165c66u/stanford_dspy_the_framework_for_programming_with/>
[90] <https://arxiv.org/pdf/2310.03714.pdf>
[91] <https://github.com/Scale3-Labs/dspy-examples>
[92] <https://www.linkedin.com/posts/nguoithichkhampha_github-stanfordnlpdspy-dspy-the-framework-activity-7225527595684048896-siCO>
[93] <https://www.reddit.com/r/aipromptprogramming/comments/1ixvfcr/miprov2_the_secret_to_significantly_boosting_any/>
[94] <https://dspy.ai/api/optimizers/BootstrapFewShot/>
[95] <https://www.mlflow.org/docs/latest/llms/dspy/notebooks/dspy_quickstart/>
[96] <https://aclanthology.org/2024.emnlp-main.597.pdf>
[97] <https://dspy.ai/learn/optimization/optimizers/>
[98] <https://notes.andymatuschak.org/zGnNuxhdDYDNevzV3dDSm12>
[99] <https://arxiv.org/html/2407.10930v1>
[100] <https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/MIPROv2.md>
[101] <https://mlflow.org/docs/latest/llms/dspy/notebooks/dspy_quickstart/>
[102] <https://dspy.ai/deep-dive/optimizers/BootstrapFinetune/>
[103] <https://arxiv.org/abs/2407.10930>
[104] <https://arxiv.org/abs/2412.15298>
[105] <https://github.com/stanfordnlp/dspy/blob/main/README.md?plain=1>
[106] <https://www.digitalocean.com/community/tutorials/prompting-with-dspy>
[107] <https://www.linkedin.com/posts/marceloriva_github-stanfordnlpdspy-dspy-the-framework-activity-7215699463636471808-FKVy>
[108] <https://www.youtube.com/watch?v=QdA-CRr_oXo>
[109] <https://qdrant.tech/blog/dspy-vs-langchain/>
[110] <https://www.linkedin.com/posts/sameer-sharma-07945734_github-stanfordnlpdspy-dspy-the-framework-activity-7210835622708527105-IePn>
[111] <https://blog.gopenai.com/unveiling-dspy-a-powerful-framework-for-building-intelligent-llm-applications-3875bd0fd80e>
[112] <https://glasp.co/un782hnbkv3syx54/p/aab2d91f53abd98dfb0b>
[113] <https://gist.github.com/jrknox1977/847c869fe9ee3b0723a9007427c38ef6>
[114] <https://github.com/stanfordnlp/dspy/blob/main/docs/docs/deep-dive/modules/chain-of-thought-with-hint.md>
[115] <https://stackoverflow.com/questions/78623315/keyerror-temperature-in-dspy-chainofthought>
[116] <https://docs.egg-ai.com/examples/dspy_react/>
[117] <https://dspy.ai/api/modules/Module/>
[118] <https://dspy.ai/api/modules/ProgramOfThought/>
[119] <https://dspy.ai/deep-dive/retrieval_models_clients/WatsonDiscovery/>
[120] <https://www.datacamp.com/blog/dspy-introduction>
[121] <https://www.linkedin.com/pulse/use-cases-dspy-data-ins-technology-llc-zf9wc>
[122] <https://wandb.ai/byyoung3/ML_NEWS3/reports/Building-and-evaluating-a-RAG-system-with-DSPy-and-W-B-Weave---Vmlldzo5OTE0MzM4>
[123] <https://dspy.ai/tutorials/agents/>
[124] <https://github.com/stanfordnlp/dspy/blob/main/docs/docs/community/how-to-contribute.md>
[125] <https://dspy.ai/community/how-to-contribute/>
[126] <https://www.youtube.com/watch?v=elZ3-aNyQPU>
[127] <https://github.com/stanfordnlp/dspy/issues/url>
[128] <https://github.com/stanfordnlp/dspy/pulls>
[129] <https://x.com/lateinteraction/status/1832058023583641789>
[130] <https://www.langtrace.ai/blog/grokking-miprov2-the-new-optimizer-from-dspy>
[131] <https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/mipro_optimizer_v2.py>
[132] <https://langwatch.ai/blog/the-power-of-miprov2-in-a-low-code-environment-with-langwatch-s-optimization-studio>
[133] <https://www.reddit.com/r/MachineLearning/comments/1cvsviu/d_does_dspy_actually_change_the_lm_weights/>
[134] <https://pypi.org/project/dspy-ai/>
[135] <https://learnbybuilding.ai/tutorials/a-gentle-introduction-to-dspy>
[136] <https://dev.to/gabrielvanderlei/dspy-a-new-approach-to-language-model-programming-10lf>
