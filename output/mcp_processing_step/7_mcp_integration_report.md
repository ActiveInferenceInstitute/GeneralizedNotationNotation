# 🤖 MCP Integration and API Report

🗓️ Report Generated: 2025-05-14 12:29:43

**MCP Core Directory:** `/home/trim/Documents/GitHub/GeneralizedNotationNotation/src/mcp`
**Project Source Root (for modules):** `/home/trim/Documents/GitHub/GeneralizedNotationNotation/src`
**Output Directory for this report:** `/home/trim/Documents/GitHub/GeneralizedNotationNotation/output/mcp_processing_step`


## 🌐 Global Summary of Registered MCP Tools

This section lists all tools currently registered with the MCP system, along with their defining module, arguments, and description.

- **Tool:** `ensure_directory_exists`
  - **Defined in Module:** `src.setup.mcp`
  - **Arguments:** `(directory_path)`
  - **Description:** "Ensures a directory exists, creating it if necessary. Returns the absolute path."
- **Tool:** `estimate_resources_for_gnn_directory`
  - **Defined in Module:** `src.gnn_type_checker.mcp`
  - **Arguments:** `(dir_path, recursive)`
  - **Description:** "Estimates computational resources for all GNN files in a specified directory."
- **Tool:** `estimate_resources_for_gnn_file`
  - **Defined in Module:** `src.gnn_type_checker.mcp`
  - **Arguments:** `(file_path)`
  - **Description:** "Estimates computational resources (memory, inference, storage) for a GNN model file."
- **Tool:** `export_gnn_to_gexf`
  - **Defined in Module:** `src.export.mcp`
  - **Arguments:** `(gnn_file_path, output_file_path)`
  - **Description:** "Exports a GNN model to GEXF graph format (requires NetworkX)."
- **Tool:** `export_gnn_to_graphml`
  - **Defined in Module:** `src.export.mcp`
  - **Arguments:** `(gnn_file_path, output_file_path)`
  - **Description:** "Exports a GNN model to GraphML graph format (requires NetworkX)."
- **Tool:** `export_gnn_to_json`
  - **Defined in Module:** `src.export.mcp`
  - **Arguments:** `(gnn_file_path, output_file_path)`
  - **Description:** "Exports a GNN model to JSON format."
- **Tool:** `export_gnn_to_json_adjacency_list`
  - **Defined in Module:** `src.export.mcp`
  - **Arguments:** `(gnn_file_path, output_file_path)`
  - **Description:** "Exports a GNN model to JSON Adjacency List graph format (requires NetworkX)."
- **Tool:** `export_gnn_to_plaintext_dsl`
  - **Defined in Module:** `src.export.mcp`
  - **Arguments:** `(gnn_file_path, output_file_path)`
  - **Description:** "Exports a GNN model back to its GNN DSL plain text format."
- **Tool:** `export_gnn_to_plaintext_summary`
  - **Defined in Module:** `src.export.mcp`
  - **Arguments:** `(gnn_file_path, output_file_path)`
  - **Description:** "Exports a GNN model to a human-readable plain text summary."
- **Tool:** `export_gnn_to_python_pickle`
  - **Defined in Module:** `src.export.mcp`
  - **Arguments:** `(gnn_file_path, output_file_path)`
  - **Description:** "Serializes a GNN model to a Python pickle file."
- **Tool:** `export_gnn_to_xml`
  - **Defined in Module:** `src.export.mcp`
  - **Arguments:** `(gnn_file_path, output_file_path)`
  - **Description:** "Exports a GNN model to XML format."
- **Tool:** `find_project_gnn_files`
  - **Defined in Module:** `src.setup.mcp`
  - **Arguments:** `(search_directory, recursive)`
  - **Description:** "Finds all GNN (.md) files in a specified directory within the project."
- **Tool:** `get_gnn_documentation`
  - **Defined in Module:** `src.gnn.mcp`
  - **Arguments:** `(doc_name)`
  - **Description:** "Retrieve the content of a GNN core documentation file (e.g., syntax, file structure)."
- **Tool:** `get_standard_output_paths`
  - **Defined in Module:** `src.setup.mcp`
  - **Arguments:** `(base_output_directory)`
  - **Description:** "Gets a dictionary of standard output directory paths (e.g., for type_check, visualization), creating them if needed."
- **Tool:** `list_render_targets`
  - **Defined in Module:** `src.render.mcp`
  - **Arguments:** `()`
  - **Description:** "Lists the available target formats for GNN rendering (e.g., pymdp, rxinfer)."
- **Tool:** `parse_gnn_file`
  - **Defined in Module:** `src.visualization.mcp`
  - **Arguments:** `(file_path)`
  - **Description:** "Parse a GNN file without visualization"
- **Tool:** `render_gnn_specification`
  - **Defined in Module:** `src.render.mcp`
  - **Arguments:** `(input_data)`
  - **Description:** "Renders a GNN (Generalized Notation Notation) specification into an executable format for a target modeling environment like PyMDP or RxInfer.jl."
- **Tool:** `run_gnn_type_checker`
  - **Defined in Module:** `src.tests.mcp`
  - **Arguments:** `(file_path)`
  - **Description:** "Run the GNN type checker on a specific file (via test module)."
- **Tool:** `run_gnn_type_checker_on_directory`
  - **Defined in Module:** `src.tests.mcp`
  - **Arguments:** `(dir_path, report_file)`
  - **Description:** "Run the GNN type checker on all GNN files in a directory (via test module)."
- **Tool:** `run_gnn_unit_tests`
  - **Defined in Module:** `src.tests.mcp`
  - **Arguments:** `()`
  - **Description:** "Run the GNN unit tests and return results."
- **Tool:** `type_check_gnn_directory`
  - **Defined in Module:** `src.gnn_type_checker.mcp`
  - **Arguments:** `(dir_path, recursive, output_dir_base, report_md_filename)`
  - **Description:** "Runs the GNN type checker on all GNN files in a specified directory. If output_dir_base is provided, reports are generated."
- **Tool:** `type_check_gnn_file`
  - **Defined in Module:** `src.gnn_type_checker.mcp`
  - **Arguments:** `(file_path)`
  - **Description:** "Runs the GNN type checker on a specified GNN model file."
- **Tool:** `visualize_gnn_directory`
  - **Defined in Module:** `src.visualization.mcp`
  - **Arguments:** `(dir_path, output_dir)`
  - **Description:** "Visualize all GNN files in a directory"
- **Tool:** `visualize_gnn_file`
  - **Defined in Module:** `src.visualization.mcp`
  - **Arguments:** `(file_path, output_dir)`
  - **Description:** "Generate visualizations for a specific GNN file."



## 🔬 Core MCP File Check

This section verifies the presence of essential MCP files in the core directory: `/home/trim/Documents/GitHub/GeneralizedNotationNotation/src/mcp`

- ✅ `mcp.py`: Found (12776 bytes)
- ✅ `meta_mcp.py`: Found (4954 bytes)
- ✅ `cli.py`: Found (4644 bytes)
- ✅ `server_stdio.py`: Found (7620 bytes)
- ✅ `server_http.py`: Found (7731 bytes)

**Status:** 5/5 core MCP files found. All core files seem present.

## 🧩 Functional Module MCP Integration & API Check

Checking for `mcp.py` in these subdirectories of `/home/trim/Documents/GitHub/GeneralizedNotationNotation/src`: ['export', 'gnn', 'gnn_type_checker', 'ontology', 'setup', 'tests', 'visualization']

### Module: `export` (at `/home/trim/Documents/GitHub/GeneralizedNotationNotation/src/export`)
- ✅ **`mcp.py` Status:** Found (8469 bytes)
- **Exposed Methods & Tools:**
  - `def _handle_export(export_func, gnn_file_path, output_file_path, format_name, requires_nx)` - *"Generic helper to run an export function and handle common exceptions."...
  - `def export_gnn_to_gexf(gnn_file_path, output_file_path)` - *"Exports a GNN model to GEXF graph format (requires NetworkX)."*
  - `def export_gnn_to_gexf_mcp(gnn_file_path, output_file_path)`
  - `def export_gnn_to_graphml(gnn_file_path, output_file_path)` - *"Exports a GNN model to GraphML graph format (requires NetworkX)."*
  - `def export_gnn_to_graphml_mcp(gnn_file_path, output_file_path)`
  - `def export_gnn_to_json(gnn_file_path, output_file_path)` - *"Exports a GNN model to JSON format."*
  - `def export_gnn_to_json_adjacency_list(gnn_file_path, output_file_path)` - *"Exports a GNN model to JSON Adjacency List graph format (requires NetworkX)."*
  - `def export_gnn_to_json_adjacency_list_mcp(gnn_file_path, output_file_path)`
  - `def export_gnn_to_json_mcp(gnn_file_path, output_file_path)`
  - `def export_gnn_to_plaintext_dsl(gnn_file_path, output_file_path)` - *"Exports a GNN model back to its GNN DSL plain text format."*
  - `def export_gnn_to_plaintext_dsl_mcp(gnn_file_path, output_file_path)`
  - `def export_gnn_to_plaintext_summary(gnn_file_path, output_file_path)` - *"Exports a GNN model to a human-readable plain text summary."*
  - `def export_gnn_to_plaintext_summary_mcp(gnn_file_path, output_file_path)`
  - `def export_gnn_to_python_pickle(gnn_file_path, output_file_path)` - *"Serializes a GNN model to a Python pickle file."*
  - `def export_gnn_to_python_pickle_mcp(gnn_file_path, output_file_path)`
  - `def export_gnn_to_xml(gnn_file_path, output_file_path)` - *"Exports a GNN model to XML format."*
  - `def export_gnn_to_xml_mcp(gnn_file_path, output_file_path)`
  - `def register_tools(mcp_instance)` - *"Registers all GNN export tools with the MCP instance."...

---

### Module: `gnn` (at `/home/trim/Documents/GitHub/GeneralizedNotationNotation/src/gnn`)
- ✅ **`mcp.py` Status:** Found (3380 bytes)
- **Exposed Methods & Tools:**
  - `def _retrieve_gnn_doc_resource(uri)` - *"Retrieve GNN documentation resource by URI."...
  - `def get_gnn_documentation(doc_name)` - *"Retrieve the content of a GNN core documentation file (e.g., syntax, file structure)."*
  - `def register_tools(mcp_instance)` - *"Register GNN documentation tools and resources with the MCP."...

---

### Module: `gnn_type_checker` (at `/home/trim/Documents/GitHub/GeneralizedNotationNotation/src/gnn_type_checker`)
- ✅ **`mcp.py` Status:** Found (10016 bytes)
- **Exposed Methods & Tools:**
  - `def estimate_resources_for_gnn_directory(dir_path, recursive)` - *"Estimates computational resources for all GNN files in a specified directory."*
  - `def estimate_resources_for_gnn_directory_mcp(dir_path, recursive)` - *"Estimate resources for all GNN files in a directory. Exposed via MCP."...
  - `def estimate_resources_for_gnn_file(file_path)` - *"Estimates computational resources (memory, inference, storage) for a GNN model file."*
  - `def estimate_resources_for_gnn_file_mcp(file_path)` - *"Estimate computational resources for a single GNN file. Exposed via MCP."...
  - `def register_tools(mcp_instance)` - *"Register GNN type checker and resource estimator tools with the MCP."...
  - `def type_check_gnn_directory(dir_path, recursive, output_dir_base, report_md_filename)` - *"Runs the GNN type checker on all GNN files in a specified directory. If output_dir_base is provided, reports are generated."*
  - `def type_check_gnn_directory_mcp(dir_path, recursive, output_dir_base, report_md_filename)` - *"Run the GNN type checker on all GNN files in a directory. Exposed via MCP."...
  - `def type_check_gnn_file(file_path)` - *"Runs the GNN type checker on a specified GNN model file."*
  - `def type_check_gnn_file_mcp(file_path)` - *"Run the GNN type checker on a single GNN file. Exposed via MCP."...

---

### Module: `ontology` (at `/home/trim/Documents/GitHub/GeneralizedNotationNotation/src/ontology`)
- ✅ **`mcp.py` Status:** Found (15497 bytes)
- **Exposed Methods & Tools:**
  - `def generate_ontology_report_for_file(gnn_file_path, parsed_annotations, validation_results)` - *"Generates a markdown formatted report string for a single GNN file's ontology annotations."...
  - `def get_mcp_interface()` - *"Returns the MCP interface for the Ontology module."...
  - `def load_defined_ontology_terms(ontology_terms_path, verbose)` - *"Loads defined ontological terms from a JSON file."...
  - `def parse_gnn_ontology_section(gnn_file_content, verbose)` - *"Parses the 'ActInfOntologyAnnotation' section from GNN file content."...
  - `def validate_annotations(parsed_annotations, defined_terms, verbose)` - *"Validates parsed GNN annotations against a set of defined ontological terms."...

---

### Module: `setup` (at `/home/trim/Documents/GitHub/GeneralizedNotationNotation/src/setup`)
- ✅ **`mcp.py` Status:** Found (3826 bytes)
- **Exposed Methods & Tools:**
  - `def ensure_directory_exists(directory_path)` - *"Ensures a directory exists, creating it if necessary. Returns the absolute path."*
  - `def ensure_directory_exists_mcp(directory_path)` - *"Ensure a directory exists, creating it if necessary. Exposed via MCP."...
  - `def find_project_gnn_files(search_directory, recursive)` - *"Finds all GNN (.md) files in a specified directory within the project."*
  - `def find_project_gnn_files_mcp(search_directory, recursive)` - *"Find all GNN (.md) files in a directory. Exposed via MCP."...
  - `def get_standard_output_paths(base_output_directory)` - *"Gets a dictionary of standard output directory paths (e.g., for type_check, visualization), creating them if needed."*
  - `def get_standard_output_paths_mcp(base_output_directory)` - *"Get standard output paths for the pipeline. Exposed via MCP."...
  - `def register_tools(mcp_instance)` - *"Register setup utility tools with the MCP."...

---

### Module: `tests` (at `/home/trim/Documents/GitHub/GeneralizedNotationNotation/src/tests`)
- ✅ **`mcp.py` Status:** Found (6518 bytes)
- **Exposed Methods & Tools:**
  - `def get_test_report(uri)` - *"Retrieve a test report by URI."...
  - `def register_tools(mcp)` - *"Register test tools with the MCP."...
  - `def run_gnn_type_checker(file_path)` - *"Run the GNN type checker on a specific file (via test module)."*
  - `def run_gnn_type_checker_on_directory(dir_path, report_file)` - *"Run the GNN type checker on all GNN files in a directory (via test module)."*
  - `def run_gnn_unit_tests()` - *"Run the GNN unit tests and return results."*
  - `def run_type_checker_on_directory(dir_path, report_file)` - *"Run the GNN type checker on a directory of files."...
  - `def run_type_checker_on_file(file_path)` - *"Run the GNN type checker on a file."...
  - `def run_unit_tests()` - *"Run the GNN unit tests."...

---

### Module: `visualization` (at `/home/trim/Documents/GitHub/GeneralizedNotationNotation/src/visualization`)
- ✅ **`mcp.py` Status:** Found (5345 bytes)
- **Exposed Methods & Tools:**
  - `def get_visualization_results(uri)` - *"Retrieve visualization results by URI."...
  - `def parse_gnn_file(file_path)` - *"Parse a GNN file without visualization"*
  - `def register_tools(mcp)` - *"Register visualization tools with the MCP."...
  - `def visualize_directory(dir_path, output_dir)` - *"Visualize all GNN files in a directory through MCP."...
  - `def visualize_file(file_path, output_dir)` - *"Visualize a GNN file through MCP."...
  - `def visualize_gnn_directory(dir_path, output_dir)` - *"Visualize all GNN files in a directory"*
  - `def visualize_gnn_file(file_path, output_dir)` - *"Generate visualizations for a specific GNN file."*

---


## 📊 Overall Module Integration Summary

- **Modules Checked:** 7
- **`mcp.py` Integrations Found:** 7/7
- **Status:** All expected functional modules appear to have an `mcp.py` integration file.
  Please ensure each functional module that should be exposed via MCP has its own `mcp.py` following the project's MCP architecture.
