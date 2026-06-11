# EXTRACT_PARAMETERS

Based on the document, here are the steps to generate a structured annotation for Active Inference:

1. **Generate a list of annotations**: Use the `generate_annotation` function from the `pymdp-core` module to generate a list of annotated nodes and edges in the graph. The list is then stored in a dictionary called `annotations`.

2. **Create a list of parameters**: Use the `create_parameters` function from the `pymdp-core` module to create a list of parameter values for each annotation type, including:
   - `generate_parameter`: Generates a parameter value based on the provided input parameters and their corresponding annotations.
   - `generate_param`: Generates a parameter value based on the provided input parameters and their corresponding annotations.

3. **Create a list of error messages**: Use the `create_error` function from the `pymdp-core` module to create a list of error messages for each annotation type, including:
   - `generate_error`: Generates an error message based on the provided input parameters and their corresponding annotations.

4. **Generate a list of validation metrics**: Use the `create_validation_metrics` function from the `pymdp-core` module to create a list of validation metrics for each annotation type, including:
   - `generate_validation_metric`: Generates an error message based on the provided input parameters and their corresponding annotations.

5. **Generate a list of confidence values**: Use the `create_confidence_values` function from the `pymdp-core` module to create a list of confidence values for each annotation type, including:
   - `generate_confidence`: Generates an error message based on the provided input parameters and their corresponding annotations.

6. **Generate a list of validation frequencies**: Use the `create_validation_frequencies` function from the `pymdp-core` module to create a list of validation frequencies for each annotation type, including:
   - `generate_validation_frequency`: Generates an error message based on the provided input parameters and their corresponding annotations.

7. **Generate a list of temporal dependencies**: Use the `create_temporal_dependencies` function from the `pymdp-core` module to create a list of temporal dependencies for each annotation type, including:
   - `generate_temporal_dependency`: Generates an error message based on the provided input parameters and their corresponding annotations.

8. **Generate a