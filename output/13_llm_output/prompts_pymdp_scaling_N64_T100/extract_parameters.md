# EXTRACT_PARAMETERS

Based on the provided information, here are the steps to generate a structured annotation of the ontology:

1. **Generate a list of input files**: Use `pymdp` to generate a list of input files for each ontology type (e.g., "ontology_output/10_ontology_input"). For example:
   ```python
import pymdp as mdp
```

   This generates a list of 32 input files, which can be used in the following steps:
   - Generate a list of input files for each ontology type (e.g., "ontology_output/10_ontology_input"). For example:
    ```python
import pymdp as mdp
```

2. **Generate a list of output files**: Use `pymdp` to generate a list of output files for each ontology type (e.g., "ontology_output/10_ontology_output"). For example:
   ```python
import pymdp as mdp
```

   This generates a list of 64 input files, which can be used in the following steps:
   - Generate a list of output files for each ontology type (e.g., "ontology_output/10_ontology_output"). For example:
    ```python
import pymdp as mdp
```

3. **Generate a list of parameter file formats**: Use `pymdp` to generate a list of parameter file format recommendations based on the input files and output files. For example, for each ontology type (e.g., "ontology_output/10_ontology_input"), use:
   - "ontology_input" -> "ontology_input".
   - "ontology_outcome" -> "ontology_outcome".

4. **Generate a list of parameter file names**: Use `pymdp` to generate a list of parameter name and value pairs for each ontology type (e.g., "ontology_output/10_ontology_input"). For example, use:
   - "ontology_input" -> "ontology".
   - "ontology_outcome" -> "ontology".

5. **Generate a list of parameters**: Use `pymdp` to generate a list of parameter file names for each ontology type (e.g., "ontology_output/10_ontology_input"). For example, use:
   - "ontology_input" -> "ontology".
   - "ontology_outcome" -> "ontology