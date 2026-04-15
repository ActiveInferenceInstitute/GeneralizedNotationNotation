# EXPLAIN_MODEL

Here is a detailed explanation of the GNN model:

**Model Purpose:** This document provides an overview of the GNN (Generalized Notation Notation) framework for active inference on to-go data. The model represents a hierarchical active inference agent with three temporal scales, each representing a different level of abstraction from the previous level.

**Core Components:**

1. **Input**: A list of input files containing information about the GNN model and its components (ontology annotations). Each file contains one or more layers of data represented by matrices. The input is structured as follows:
   - Input File: A list of input files, each representing a single layer in the GNN hierarchy. Each input file has two columns: `input_file` and `layer`.
   - Layer: A list of layers containing information about the input data (e.g., `input_data`, `output_data`, etc.). Each layer contains one or more input files, each representing a single layer in the GNN hierarchy.

2. **Output**: A list of output files containing information about the model outputs and actions/controls. Each output file has two columns: `outcome` and `action`.
   - Output File: A list of output files, each representing one or more layers in the GNN hierarchy. Each output file contains one or more input files, each representing a single layer in the GNN hierarchy.

3. **Model Parameters**: A dictionary containing information about the model parameters (e.g., `input_data`, `output_data`, etc.). Each parameter is represented by a key-value pair. The keys are strings and the values are lists of input files, each representing one layer in the GNN hierarchy.

**Signature:** This document provides an overview of how the model evolves over time based on observations from the data. It describes what actions/controls (u_c0, π_c0) and beliefs (o_m0, o_m1, etc.) are available in the model at each level of abstraction.

**Key Relationships:**

1. **Input**: A list of input files containing information about the GNN model and its components. Each input file has two columns: `input` and `layer`.
   - Input File: A list of input files, representing one or more layers in the GNN hierarchy. Each input file contains one or more input files, each representing a single layer in the GNN hierarchy.

