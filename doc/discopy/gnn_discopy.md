https://github.com/discopy/discopy

# DisCoPy and GNN: Categorical Modeling of Complex Systems

DisCoPy is an advanced, open-source Python library designed for working with string diagrams in category theory. It provides a robust framework for implementing and manipulating categorical structures with applications ranging from natural language processing to quantum computing. Generalized Notation Notation (GNN) is a specification for describing graphical models, their state spaces, connections, and parameters. This document explores the powerful synergy between DisCoPy and GNN, examining how DisCoPy's categorical framework can provide a formal and computational foundation for representing, analyzing, and manipulating GNN models. We delve into the mathematical underpinnings, key features of both systems, and the potential for their integration in modeling complex systems, from natural language processing to active inference and quantum computing.

## Theoretical Foundations: Bridging GNN and DisCoPy

The integration of DisCoPy and GNN is founded on the principle that GNN's declarative specifications of models can be mapped to DisCoPy's executable categorical structures. String diagrams, the core of DisCoPy, serve as a graphical calculus for monoidal categories, offering a natural language for the connections and compositions inherent in GNN models.

### Core Data Structures in DisCoPy and their GNN Counterparts

The foundational data structure in DisCoPy is the `Diagram` class. A diagram in DisCoPy is characterized by a domain (`dom`) and codomain (`cod`), a list of `boxes` (operations), `offsets`, and an optional list of `layers`. This structure finds direct parallels in GNN:

*   **GNN `StateSpaceBlock` and DisCoPy `Ty`:**
    The `StateSpaceBlock` in a GNN file defines variables and their dimensions (e.g., `X[2,3]`, `Y[5]`). Each such variable can be represented as a DisCoPy `Ty` (type) object.
    *   A simple variable `X` could be `Ty('X')`.
    *   A multi-dimensional variable like `X[2,3]` could be conceptualized as `Ty('X_dim1') @ Ty('X_dim2')` or a custom `Ty` object whose semantics (via a functor) capture this dimensionality. The name of the type can be more descriptive, like `Ty('X[2,3]')`.
    *   The collection of all input variables to a GNN model or a sub-component can form the `dom` (domain type) of a DisCoPy diagram, and output variables the `cod` (codomain type).

*   **GNN `Connections` and DisCoPy `Box`, `Diagram`:**
    The `Connections` section in GNN describes edges (e.g., `A > B`, `C - D`). These are the core of the model's structure and map directly to DisCoPy's `Box` (morphisms/processes) and their composition into `Diagrams`.
    *   A directed connection `A > B` can be a `Box('A_to_B', Ty('A'), Ty('B'))`.
    *   An undirected connection `C - D` might be represented by a symmetric box or a pair of boxes `Box('C_to_D', Ty('C'), Ty('D'))` and `Box('D_to_C', Ty('D'), Ty('C'))`, depending on the intended semantics.
    *   Complex GNN models with multiple interacting variables are formed by composing these boxes sequentially (`>>`) and in parallel (`@`) to create a comprehensive DisCoPy `Diagram` that mirrors the GNN's graph structure. For instance, a GNN model like:
        ```
        # GNN Snippet
        # StateSpaceBlock
        # A[2]
        # B[3]
        # C[2]
        # Connections
        # A > B
        # B > C
        ```
        could translate to:
   ```python
        from discopy.monoidal import Diagram, Ty, Box
        A_type, B_type, C_type = Ty('A[2]'), Ty('B[3]'), Ty('C[2]')
        f_AB = Box('A_to_B', A_type, B_type)
        f_BC = Box('B_to_C', B_type, C_type)
        gnn_diagram = f_AB >> f_BC
        ```

*   **GNN `InitialParameterization` and DisCoPy Functors:**
    The `InitialParameterization` section provides values for model variables. In the DisCoPy framework, these parameters would inform the *semantics* of the `Box` objects. A DisCoPy `Functor` could map the abstract diagram (representing structure) to a concrete computation or tensor network where boxes are functions/tensors initialized with these GNN parameters.

*   **GNN `Equations` and DisCoPy Functors:**
    The `Equations` section in GNN (often in LaTeX) describes the mathematical relationships. DisCoPy diagrams provide the syntax of composition, while these equations provide the semantics. A functor maps each `Box` in a diagram to a function or process that implements the corresponding GNN equation. For instance, if `A > B` is governed by `B = f(A, params)`, the functor would map `Box('A_to_B', ...)` to this function `f`.

*   **GNN `Time` and DisCoPy's Traced Categories/Monoidal Streams:**
    GNN's `Time` section (specifying `Static`, `Dynamic`, `DiscreteTime`, `ContinuousTime`) is crucial. For dynamic models, DisCoPy's support for traced monoidal categories (allowing feedback loops) and especially monoidal streams is highly relevant. A GNN describing a system evolving over `X_t` (e.g., `DiscreteTime=X_t`) can be modeled as a DisCoPy diagram with explicit trace operators or as a monoidal stream, capturing the step-by-step evolution.

*   **GNN `ActInfOntologyAnnotation` and Semantic Labeling in DisCoPy:**
    This GNN section links model variables to Active Inference Ontology terms (e.g., `C=Preference`). These annotations can be carried over to DisCoPy `Ty` or `Box` objects as metadata. Functors can then use these semantic labels to choose specific interpretations or to interface with ontology-aware tools. This is particularly powerful for building complex cognitive models where components have defined roles.

The `Layer` class in DisCoPy, representing a horizontal slice of a string diagram, is an internal detail that facilitates the construction and manipulation of these GNN-derived diagrams, often remaining transparent to the user who interacts at the `Diagram` level. The overall GNN specification can thus be algorithmically translated into a DisCoPy `Diagram`, enabling the use of DisCoPy's rich toolkit for analysis, simplification, visualization, and execution via functors.

## The Hierarchy of Graphical Languages for GNN Models

DisCoPy implements the complete hierarchy of graphical languages for monoidal categories. This hierarchy provides a progressively richer set of tools and structures that can be applied to GNN models:

### 1. Monoidal Categories: Basic GNN Structure

At the foundational level, DisCoPy supports monoidal categories with sequential (`>>`) and parallel (`@`) composition. This is the starting point for representing GNNs:
- GNN variables (from `StateSpaceBlock`) map to `Ty` objects.
- GNN connections/processes (from `Connections` and interpreted by `Equations`) map to `Box` objects.
- The overall GNN model structure is a `Diagram` composed of these boxes and types.
This allows a basic structural representation of any GNN.

### 2. Symmetric and Braided Categories: Reordering GNN Variables

DisCoPy extends to symmetric monoidal categories by adding `Swap` operations (and braided categories with more general `Braid` operations).
- For GNNs, this means that if you have parallel subsystems or variables (e.g., `X @ Y`), you can formally reorder them (`Y @ X`) using swaps. This is useful when interfacing GNN components that have different conventional input/output orders or for diagram simplification.

### 3. Compact Closed and Rigid Categories: Cycles, Adjoints, and GNN Dualities

For more advanced GNN applications, DisCoPy implements compact closed categories (with `Cup` and `Cap` operations for bending wires) and rigid categories (which distinguish left and right adjoints/duals, `.l` and `.r` attributes of types).
- **Feedback and Cycles:** `Cup` and `Cap` can explicitly model cycles or feedback loops in GNNs that might be described in `Connections` or implied by `Equations`. This is an alternative to using traced categories for some types of feedback.
- **Belief Propagation/Message Passing:** In GNNs representing probabilistic models, cups and caps can sometimes model the pairing of messages or beliefs in inference algorithms.
- **Dualities in GNNs:** If a GNN describes processes with natural duals (e.g., state and observation processes in some agent models, or creation/annihilation processes), the rigid category structure with its adjoints (`type.l`, `type.r`) can capture these faithfully. For example, a GNN variable representing a 'query' might be dual to a variable representing 'data'.

### 4. Traced and Feedback Categories: Explicitly Modeling GNN Dynamics

DisCoPy provides traced monoidal structures, which are essential for GNNs with explicit dynamic evolution or feedback loops not easily captured by just cups/caps.
- **Dynamic GNNs:** GNNs with `Time` specifications (e.g., `DiscreteTime=S_t`) often involve states feeding back into themselves or other components over time. The `Trace` operation in DisCoPy directly models this.
- **Embedding in Compact Closed:** DisCoPy's `Int`-construction (Geometry of Interaction) allows embedding traced categories into compact closed ones, offering another way to handle feedback.

### 5. Monoidal Streams for GNN Dataflow and Stochastic Processes

A highly relevant recent addition to DisCoPy is monoidal streams, providing semantics to signal flow diagrams with delayed feedback.
- **GNNs as Dataflow Programs:** When a GNN's `Equations` define a clear dataflow (outputs of some variables become inputs to others, potentially over time), monoidal streams can model this. The `delay` functor (δ) in DisCoPy streams is perfect for `DiscreteTime` GNNs.
- **Probabilistic GNNs:** If a GNN describes a stochastic process (e.g., a Hidden Markov Model or a POMDP defined via GNN syntax), evaluating its corresponding DisCoPy stream diagram in a category of probabilistic functions can yield the stochastic process itself. This is a powerful way to give operational semantics to probabilistic GNNs.

### 6. Hypergraph Categories: Beyond Pairwise GNN Connections

While GNN `Connections` primarily define pairwise relationships, some complex systems might involve multi-way interactions (e.g., a single process taking three inputs simultaneously to produce two outputs, not easily decomposable into a sequence of pairwise interactions).
- DisCoPy's `Hypergraph` data structure can represent diagrams in hypergraph categories, potentially offering a way to extend GNN's expressiveness for such systems or to interface GNNs with hypergraph-based modeling tools. This could be particularly relevant for certain chemical reaction networks or complex causal models.

## Functorial Semantics: Executing and Analyzing GNN Models via DisCoPy

One of DisCoPy's core features is its implementation of monoidal functors. For GNN models translated into DisCoPy diagrams, functors are the mechanism to assign concrete meaning and execute them. A functor maps the abstract types (`Ty`) and boxes (`Box`) of a DisCoPy diagram to objects and morphisms in a target category (e.g., Python functions, matrices, other diagrams).

### 1. Python Functors for GNN Simulation and Execution

A DisCoPy `Diagram` derived from a GNN can be evaluated as Python code using a custom functor:
-   **Wires to Data Types:** `Ty` objects (representing GNN variables) map to Python data types (e.g., `int`, `float`, `numpy.ndarray`). The dimensionality from `StateSpaceBlock` guides this.
-   **Boxes to Python Functions:** Each `Box` (representing a GNN connection or process) maps to a Python function. This function implements the transformation specified by the GNN's `Equations` section, using parameters from `InitialParameterization`.
-   **Composition to Function Calls:** Diagrammatic composition (`>>` for sequential, `@` for parallel) maps to function application and the structuring of data flow between these functions.

For example, a functor could take a GNN diagram representing `A > B` (where `B = func(A)`) and execute `func(data_A)` to get `data_B`. For dynamic GNNs (marked with `Time` attributes), the functor might manage state and iterate computation.

```python
# Conceptual Example:
# from discopy import Functor
#
# class GNNExecutionFunctor(Functor):
#     def __init__(self, gnn_model_params, gnn_model_equations_map):
#         self.params = gnn_model_params
#         self.equations = gnn_model_equations_map # Maps box names to callable functions
#         super().__init__(ob=lambda ty: ty, ar=self._map_box) # Simple type mapping
#
#     def _map_box(self, box):
#         # Fetch the specific function for this box based on its name or GNN origin
#         if box.name in self.equations:
#             # The function might expect current values of its inputs and parameters
#             return self.equations[box.name] # Returns a callable
#         else:
#             raise ValueError(f"No equation defined for GNN component: {box.name}")
#
# # Assuming discopy_gnn_diagram is derived from a GNN file
# # gnn_params = load_gnn_parameters(...)
# # gnn_equations = { "A_to_B": lambda A_val, p: A_val * p['some_param'], ... }
# # executor = GNNExecutionFunctor(gnn_params, gnn_equations)
# # input_data = {"A_type_name": initial_value_A}
# # result = executor(discopy_gnn_diagram).eval(input_data) # .eval() might be specific to functor target
```

### 2. Tensor Network Functors for Probabilistic GNNs and Parameter Learning

Many GNNs, especially those in fields like computational neuroscience or machine learning, describe probabilistic models or systems that can be represented as tensor networks. DisCoPy's ability to evaluate diagrams as tensor networks is highly beneficial:
-   **Wires to Tensor Dimensions:** `Ty` objects (GNN variables) are interpreted as indices or dimensions of tensors.
-   **Boxes to Tensors:** Each `Box` (GNN component/connection) becomes a specific tensor (e.g., a conditional probability table, a weight matrix) initialized using `InitialParameterization`.
-   **Diagram to Tensor Network Contraction:** The DisCoPy diagram defines a tensor network. Evaluating the diagram (via a tensor functor) corresponds to contracting this network. This can compute marginal probabilities, partition functions, or other quantities of interest.
-   **Integration with ML Libraries:** DisCoPy's support for NumPy, PyTorch, TensorFlow, JAX, etc., means that GNNs translated to DisCoPy can be part of larger machine learning pipelines, enabling gradient-based parameter learning directly on the GNN structure represented categorically. The `Equations` can define the structure of these tensors.

This is particularly powerful for GNNs that specify Bayesian networks, Markov networks, or state-space models where parameters need to be learned from data.

### 3. Diagram-Valued Functors for GNN Model Transformation and Abstraction

Diagram-valued functors map primitive elements (boxes, types) of one DisCoPy diagram to more complex DisCoPy diagrams. This enables powerful GNN model manipulation:
-   **Hierarchical GNN Modeling:** A single `Box` in a high-level DisCoPy diagram (representing a complex GNN subsystem) can be expanded by a functor into a detailed DisCoPy diagram representing that subsystem's internal GNN structure. This allows for modular and hierarchical GNN design.
-   **GNN Model Refinement & Simplification:** Functors can implement rewrite rules that transform a GNN-derived diagram into an equivalent but simpler or canonical form based on categorical equivalences (e.g., snake equations for compact closed categories).
-   **Translating Between GNN Variants or to Other Formalisms:** If different GNN "dialects" exist or if a GNN model needs to be translated to another modeling framework (e.g., Petri nets, certain agent architectures) that also has a categorical representation, diagram-valued functors can define these translations.
-   **Automatic Code Generation:** A functor could map a GNN-derived diagram to a diagram representing the control flow or structure of code in a target simulation language (e.g., PyMDP, RxInfer.jl), effectively acting as a compiler step.

## Applications: GNN and DisCoPy in Synergy

The combination of GNN's descriptive power and DisCoPy's compositional framework opens up numerous application areas, extending beyond DisCoPy's traditional strengths.

### 1. Advanced Active Inference Modeling

GNNs are increasingly used to specify Active Inference models. The `ActInfOntologyAnnotation` section is key here.
-   **Categorical Specification of Agents:** An entire Active Inference agent, with its generative model, variational free energy calculations, and policy selection mechanisms (often defined via complex GNN `Connections` and `Equations`), can be translated into a DisCoPy diagram.
-   **Functorial Semantics for Belief Propagation:** Functors can implement the message passing algorithms (belief propagation) on the GNN-derived DisCoPy diagram to compute posterior beliefs and expected free energy.
-   **Composition of Cognitive Modules:** Different cognitive functions (perception, action, learning), each modeled as a GNN and translated to a DisCoPy diagram, can be formally composed using `>>` and `@` to build more complex agents. DisCoPy's traced categories or monoidal streams are ideal for modeling the perception-action loops inherent in Active Inference.

### 2. Modular System Design and Simulation for GNN

-   **Decomposition and Recomposition:** Complex systems specified in GNN can be broken down into smaller, manageable GNN sub-modules. Each sub-module translates to a DisCoPy diagram. These diagrams can then be recomposed using DisCoPy's operators, ensuring interface compatibility through type checking (`Ty` objects).
-   **Multi-Scale Modeling:** Diagram-valued functors can allow a GNN model to be viewed at different levels of abstraction, from high-level components to detailed variable interactions.
-   **Interfacing GNN with Other Categorical Models:** If other parts of a system are already modeled using DisCoPy (e.g., a natural language interface for a GNN-based robot), DisCoPy can serve as the common compositional backbone.

### 3. Enhanced GNN Model Verification and Analysis

-   **Diagrammatic Reasoning:** DisCoPy's grounding in category theory allows for diagrammatic reasoning about GNN models. Certain structural properties or equivalences between different GNN specifications might be provable by manipulating their DisCoPy diagram representations.
-   **Type Checking for GNN Connections:** DisCoPy's strict typing (`Ty` for domain and codomain of `Box`es) acts as a form of static analysis for GNN `Connections`, ensuring that interconnected components have compatible state spaces.

### 4. Cross-Framework GNN Model Translation (via DisCoPy as Intermediate Representation)

If a GNN model needs to be executed in different simulation environments (e.g., PyMDP, RxInfer.jl, or a custom engine), DisCoPy can act as a common intermediate representation.
-   **GNN -> DisCoPy:** Parse GNN into a DisCoPy diagram.
-   **DisCoPy -> Target:** Define functors that map this DisCoPy diagram to the constructs of the target environment (e.g., code, configuration files). This modularizes the GNN `render` pipeline.

## GNN-Specific Extensions to DisCoPy's Tooling (Conceptual)

The integration also suggests potential GNN-specific tooling built upon or alongside DisCoPy:

*   **GNN Parser to DisCoPy Diagram:** A dedicated Python library function that takes a GNN file path (or string content) and directly outputs a DisCoPy `Diagram` object. This would parse `StateSpaceBlock` to `Ty`s, `Connections` to `Box`es and their compositions, and store `Equations` and `InitialParameterization` metadata with the boxes for later functorial interpretation.
*   **GNN-Aware Functors:** Pre-built functors in DisCoPy or an extension library tailored for GNNs, e.g., a functor that automatically sets up a PyMDP simulation from a GNN-derived diagram, or one that generates boilerplate for RxInfer.
*   **Visualization Tools:** Enhancing DisCoPy's `diagram.draw()` to incorporate GNN-specific information, like displaying variable dimensions from `StateSpaceBlock` on wires, or using `ActInfOntologyAnnotation` terms as labels.

## Monoidal Streams and Probabilistic Dataflow for Dynamic GNNs

The `Monoidal Streams` feature in DisCoPy is particularly apt for dynamic GNNs as specified by the `Time` section (e.g., `DiscreteTime=X_t`).
-   **Causal Flow in GNNs:** Monoidal streams formalize dataflow with delayed feedback. A GNN's temporal update rules (often in `Equations`) can be modeled as the function applied at each step of the stream's evolution.
-   **Stochastic GNNs:** For GNNs representing probabilistic state-space models, evaluating their stream diagram in a category of probabilistic functions (e.g., where boxes are stochastic kernels) yields the controlled stochastic process defined by the GNN. This connects GNN to areas like POMDPs and control theory in a formal way.

## Advanced Example: Geometry of Interaction for Complex GNN Feedback

DisCoPy's implementation of the `Int`-construction (Geometry of Interaction) embeds traced categories (which allow feedback) into compact closed ones. For complex GNNs with intricate feedback loops as defined in `Connections` or implied by iterative `Equations`, this offers a powerful theoretical tool:
-   It provides a uniform way to represent feedback using cups and caps, potentially simplifying diagrammatic reasoning about GNNs with loops.
-   Applications could include analyzing the convergence of iterative algorithms specified in GNN or understanding information flow in recurrent GNN models.

```python
# Conceptual: Representing a GNN feedback loop A -> B, B -> A using Int
# from discopy.interaction import Ty as IntTy, IntFunctor
# from discopy.monoidal import Ty, Box, Diagram
# from discopy.symmetric import Id, Swap
#
# # Original GNN components as standard DisCoPy boxes
# A_orig, B_orig = Ty('A'), Ty('B')
# f_AB = Box('f_AB', A_orig, B_orig) # From GNN: A -> B
# f_BA = Box('f_BA', B_orig, A_orig) # From GNN: B -> A (feedback)
#
# # Translate to the Int category
# IntA, IntB = IntTy(A_orig), IntTy(B_orig)
# int_f_AB = Box('f_AB', IntA, IntB) # Abstractly, functor maps f_AB to this
# int_f_BA = Box('f_BA', IntB, IntA) # Abstractly, functor maps f_BA to this
#
# # Diagram with explicit feedback using cups and caps in the target of IntFunctor
# # This is more illustrative; IntFunctor handles the details internally
# # Suppose F = IntFunctor(...)
# # F(f_AB) and F(f_BA) would be diagrams in a compact closed category.
# # A GNN cycle like (A -> B -> A) could be traced:
# loop_on_A = (f_AB >> f_BA).trace() # Using native trace
#
# # Or, via Int, the trace is implemented with cups/caps on F(loop_on_A)
# # This illustrates the principle rather than direct DisCoPy syntax for this specific case
```

## Integration with the GNN Toolkit and Workflow

DisCoPy can be integrated into the existing GNN processing pipeline (`src/main.py` and its numbered steps) to provide enhanced capabilities:

*   **`src/15_audio.py` (New Step):** A dedicated pipeline step could be introduced. This script would:
    *   Take GNN files (e.g., from `output/gnn_exports/` or directly from `src/gnn/gnn_examples/`) as input.
    *   Utilize a GNN parser and a `gnn_to_discopy_diagram` translator.
    *   Generate DisCoPy diagrams.
    *   Output these diagrams in various formats (e.g., serialized DisCoPy objects, Python scripts that reconstruct the diagrams, or images via `diagram.draw()`) to a new directory like `output/discopy_gnn/`.
    *   Optionally, apply predefined functors (e.g., for simplification or basic analysis).

*   **Enhancing Existing Steps:**
    *   **`src/export/`:** Exporters could gain a "DisCoPy diagram" target format.
    *   **`src/visualization/`:** DisCoPy's `diagram.draw()` could be an alternative or supplementary visualization method, potentially offering different aesthetic or informational advantages.
    *   **`src/render/`:** Instead of directly rendering GNN to PyMDP or RxInfer, the pipeline could first render GNN to a DisCoPy diagram. Then, DisCoPy-to-PyMDP or DisCoPy-to-RxInfer functors could perform the final translation. This makes the rendering process more modular and allows for intermediate categorical manipulations.
    *   **`src/gnn_type_checker/`:** DisCoPy's strict typing of diagrams (domains and codomains of boxes must match) can augment GNN type checking by verifying the consistency of composed GNN components at the categorical level.

### Development and Testing (Synergies)

The rigorous development and testing practices of DisCoPy (`pytest`, high coverage, `pycodestyle`, `sphinx` for docs) can serve as a model or even be partially leveraged if DisCoPy becomes a core dependency for certain GNN functionalities. GNN-to-DisCoPy translation logic would itself require thorough testing.

## Academic Context and Citations

The integration of GNN with DisCoPy builds upon the rich academic foundations of both. Key DisCoPy publications include:

1.  G. de Felice, A. Toumi & B. Coecke, "DisCoPy: Monoidal Categories in Python", EPTCS 333, 2021, pp. 183-197[1][3]
2.  A. Toumi, G. de Felice & R. Yeung, "DisCoPy for the quantum computer scientist", arXiv:2205.05190[1]
3.  A. Toumi, R. Yeung, B. Poór & G. de Felice, "DisCoPy: the Hierarchy of Graphical Languages in Python", arXiv:2311.10608[1][5]

Research in GNN, particularly in areas like Active Inference and compositional systems modeling, would provide the domain-specific use cases that drive the practical application of DisCoPy's categorical tools. Future publications could specifically address the GNN-DisCoPy bridge, detailing the translation algorithms and showcasing applications.

## Conclusion: GNN and DisCoPy - A Unified Framework for Compositional Modeling

DisCoPy provides a powerful, Python-native toolkit for applied category theory using string diagrams. GNN offers a structured notation for specifying complex graphical models. Their integration offers a significant advancement: GNN defines *what* a model is, while DisCoPy provides a formal language and computational tools to reason about, manipulate, and execute *how* these models compose and behave.

By translating GNN specifications into DisCoPy diagrams, we can leverage:
-   **Formal Composition:** Using monoidal products (`@`) and composition (`>>`) to build complex GNNs from simpler ones with guaranteed interface consistency.
-   **Multiple Semantic Interpretations:** Using DisCoPy functors to map GNN diagrams to various computational domains (Python execution, tensor networks, other diagrammatic systems).
-   **Advanced Categorical Structures:** Employing traced, compact closed, and other categories to model sophisticated GNN features like feedback, dynamics, and dualities.
-   **Enhanced Tooling:** Potential for more robust GNN analysis, verification, visualization, and transformation tools built on a categorical foundation.

This synergy positions the GNN-DisCoPy framework as a versatile and rigorous environment for researchers and practitioners developing compositional models across diverse fields, from AI and cognitive science to systems biology and quantum computing. The ongoing development of both GNN and DisCoPy promises a continually evolving and increasingly powerful platform for tackling complex, interconnected systems.

Citations:
[1] https://github.com/discopy/discopy
[2] https://github.com/discopy
[3] https://arxiv.org/pdf/2005.02975.pdf
[4] https://docs.discopy.org/en/main/notebooks/diagrams.html
[5] https://arxiv.org/pdf/2311.10608.pdf
[6] https://github.com/rknaebel/discopy (Note: This seems to be a fork or related project, the main one is discopy/discopy)
[7] https://oxford24.github.io/assets/act-papers/46_monoidal_streams_and_probabili.pdf
[8] https://discopy.org
[9] https://github.com/fbngrm/DiscoPy (Another related project/fork)
[10] https://github.com/discopy/discopy/blob/main/CONTRIBUTING.md
[11] https://github.com/rknaebel/discopy-data (Likely related to [6])

## Using the GNN-to-DisCoPy Pipeline Step (15_audio.py)

The GNN processing pipeline includes a dedicated step, `15_audio.py`, for automatically translating GNN model specifications into DisCoPy diagrams and visualizing them. This step leverages the `src.discopy_translator_module.translator` module.

### Purpose

The `15_audio.py` script aims to:
1.  Discover GNN files (typically `.md` or `.gnn.md` files containing GNN specifications) in a specified input directory.
2.  For each GNN file, parse its content to identify `StateSpaceBlock` and `Connections` sections.
3.  Translate these GNN structures into corresponding DisCoPy elements:
    *   State space variables become `discopy.Ty` objects.
    *   Connections between variables become `discopy.Box` objects, composed into a `discopy.Diagram`.
4.  Generate a PNG image visualizing the resulting DisCoPy diagram.
5.  Save these visualizations into a structured output directory.

### Running the Step

This step is typically invoked as part of the main GNN pipeline orchestrated by `src/main.py`.

```bash
python src/main.py --only-steps 12_discopy
```

Key command-line arguments relevant to this step when running via `src/main.py`:

*   `--output-dir <PATH>`: The main output directory for the entire pipeline. The DisCoPy diagrams will be saved under `<PATH>/discopy_gnn/`.
*   `--target-dir <PATH>` (or `--discopy-gnn-input-dir <PATH>`): Specifies the directory containing the input GNN files for the DisCoPy step. If `--discopy-gnn-input-dir` is provided, it takes precedence for this step; otherwise, the general `--target-dir` is used.
*   `--recursive` / `--no-recursive`: Controls whether to search for GNN files recursively in the input directory.
*   `--verbose` / `--no-verbose`: Enables detailed logging for the step.

### Inputs

*   **GNN Files**: Standard GNN markdown files (`.md` or `.gnn.md`) located in the directory specified by `--gnn-input-dir` (or `--target-dir`). These files should contain at least:
    *   A `## StateSpaceBlock` section defining variables (e.g., `MyVar`, `AnotherVar[dim]`).
    *   A `## Connections` section defining relationships (e.g., `MyVar > AnotherVar`).

### Outputs

For each processed GNN file (e.g., `example_model.md`), the script generates:

*   A PNG image of the DisCoPy diagram: `output/discopy_gnn/example_model_diagram.png`.
    *   If the GNN file was in a subdirectory of the input path (e.g., `my_models/example_model.md`), the output structure is preserved: `output/discopy_gnn/my_models/example_model_diagram.png`.

The diagrams visualize the types (`Ty`) as wires and the connections (`Box`) as boxes, illustrating the categorical structure inferred from the GNN specification.

### Current Limitations and Future Work

*   **Parsing**: The GNN parser in `src.discopy_translator_module.translator` is basic and expects GNN sections and syntax to be well-formed.
*   **Connection Complexity**: The current translator primarily handles sequential connections (`A > B`). More complex graph structures (parallel compositions, feedback loops, multi-input/multi-output boxes from a single GNN connection line) are handled with simplifications or might not be fully represented.
*   **Functorial Semantics**: This step focuses on structural translation. Assigning concrete computational semantics (via DisCoPy functors) to these diagrams is a subsequent step, potentially for a different pipeline stage or manual exploration.

Future development could involve enhancing the GNN parser, supporting more sophisticated diagram construction from complex GNN connection patterns, and integrating automated functor application for simulation or analysis.
