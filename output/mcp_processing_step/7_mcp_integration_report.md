# 🤖 MCP Integration and API Report

🗓️ Report Generated: 2025-06-06 13:08:27

**MCP Core Directory:** `/home/trim/Documents/GitHub/GeneralizedNotationNotation/src/mcp`
**Project Source Root (for modules):** `/home/trim/Documents/GitHub/GeneralizedNotationNotation/src`
**Output Directory for this report:** `/home/trim/Documents/GitHub/GeneralizedNotationNotation/output/mcp_processing_step`


## 🌐 Global Summary of Registered MCP Tools

This section lists all tools currently registered with the MCP system, along with their defining module, arguments, and description.

- **Tool:** `ensure_directory_exists`
  - **Defined in Module:** `src.setup.mcp`
  - **Arguments (from signature):** `(directory_path)`
  - **Description:** "Ensures a directory exists, creating it if necessary. Returns the absolute path."
  - **Schema:**
    ```json
    {
        "directory_path": {
            "type": "string",
            "description": "Path of the directory to create if it doesn't exist."
        }
    }
    ```
- **Tool:** `estimate_resources_for_gnn_directory`
  - **Defined in Module:** `src.gnn_type_checker.mcp`
  - **Arguments (from signature):** `(dir_path, recursive)`
  - **Description:** "Estimates computational resources for all GNN files in a specified directory."
  - **Schema:**
    ```json
    {
        "dir_path": {
            "type": "string",
            "description": "Path to the directory for GNN resource estimation."
        },
        "recursive": {
            "type": "boolean",
            "description": "Search directory recursively. Defaults to False.",
            "optional": true
        }
    }
    ```
- **Tool:** `estimate_resources_for_gnn_file`
  - **Defined in Module:** `src.gnn_type_checker.mcp`
  - **Arguments (from signature):** `(file_path)`
  - **Description:** "Estimates computational resources (memory, inference, storage) for a GNN model file."
  - **Schema:**
    ```json
    {
        "file_path": {
            "type": "string",
            "description": "Path to the GNN file for resource estimation."
        }
    }
    ```
- **Tool:** `export_gnn_to_gexf`
  - **Defined in Module:** `src.export.mcp`
  - **Arguments (from signature):** `(gnn_file_path, output_file_path)`
  - **Description:** "Exports a GNN model to GEXF graph format (requires NetworkX)."
  - **Schema:**
    ```json
    {
        "gnn_file_path": {
            "type": "string",
            "description": "Path to the input GNN Markdown file (.gnn.md)."
        },
        "output_file_path": {
            "type": "string",
            "description": "Path where the exported file will be saved."
        }
    }
    ```
- **Tool:** `export_gnn_to_graphml`
  - **Defined in Module:** `src.export.mcp`
  - **Arguments (from signature):** `(gnn_file_path, output_file_path)`
  - **Description:** "Exports a GNN model to GraphML graph format (requires NetworkX)."
  - **Schema:**
    ```json
    {
        "gnn_file_path": {
            "type": "string",
            "description": "Path to the input GNN Markdown file (.gnn.md)."
        },
        "output_file_path": {
            "type": "string",
            "description": "Path where the exported file will be saved."
        }
    }
    ```
- **Tool:** `export_gnn_to_json`
  - **Defined in Module:** `src.export.mcp`
  - **Arguments (from signature):** `(gnn_file_path, output_file_path)`
  - **Description:** "Exports a GNN model to JSON format."
  - **Schema:**
    ```json
    {
        "gnn_file_path": {
            "type": "string",
            "description": "Path to the input GNN Markdown file (.gnn.md)."
        },
        "output_file_path": {
            "type": "string",
            "description": "Path where the exported file will be saved."
        }
    }
    ```
- **Tool:** `export_gnn_to_json_adjacency_list`
  - **Defined in Module:** `src.export.mcp`
  - **Arguments (from signature):** `(gnn_file_path, output_file_path)`
  - **Description:** "Exports a GNN model to JSON Adjacency List graph format (requires NetworkX)."
  - **Schema:**
    ```json
    {
        "gnn_file_path": {
            "type": "string",
            "description": "Path to the input GNN Markdown file (.gnn.md)."
        },
        "output_file_path": {
            "type": "string",
            "description": "Path where the exported file will be saved."
        }
    }
    ```
- **Tool:** `export_gnn_to_plaintext_dsl`
  - **Defined in Module:** `src.export.mcp`
  - **Arguments (from signature):** `(gnn_file_path, output_file_path)`
  - **Description:** "Exports a GNN model back to its GNN DSL plain text format."
  - **Schema:**
    ```json
    {
        "gnn_file_path": {
            "type": "string",
            "description": "Path to the input GNN Markdown file (.gnn.md)."
        },
        "output_file_path": {
            "type": "string",
            "description": "Path where the exported file will be saved."
        }
    }
    ```
- **Tool:** `export_gnn_to_plaintext_summary`
  - **Defined in Module:** `src.export.mcp`
  - **Arguments (from signature):** `(gnn_file_path, output_file_path)`
  - **Description:** "Exports a GNN model to a human-readable plain text summary."
  - **Schema:**
    ```json
    {
        "gnn_file_path": {
            "type": "string",
            "description": "Path to the input GNN Markdown file (.gnn.md)."
        },
        "output_file_path": {
            "type": "string",
            "description": "Path where the exported file will be saved."
        }
    }
    ```
- **Tool:** `export_gnn_to_python_pickle`
  - **Defined in Module:** `src.export.mcp`
  - **Arguments (from signature):** `(gnn_file_path, output_file_path)`
  - **Description:** "Serializes a GNN model to a Python pickle file."
  - **Schema:**
    ```json
    {
        "gnn_file_path": {
            "type": "string",
            "description": "Path to the input GNN Markdown file (.gnn.md)."
        },
        "output_file_path": {
            "type": "string",
            "description": "Path where the exported file will be saved."
        }
    }
    ```
- **Tool:** `export_gnn_to_xml`
  - **Defined in Module:** `src.export.mcp`
  - **Arguments (from signature):** `(gnn_file_path, output_file_path)`
  - **Description:** "Exports a GNN model to XML format."
  - **Schema:**
    ```json
    {
        "gnn_file_path": {
            "type": "string",
            "description": "Path to the input GNN Markdown file (.gnn.md)."
        },
        "output_file_path": {
            "type": "string",
            "description": "Path where the exported file will be saved."
        }
    }
    ```
- **Tool:** `find_project_gnn_files`
  - **Defined in Module:** `src.setup.mcp`
  - **Arguments (from signature):** `(search_directory, recursive)`
  - **Description:** "Finds all GNN (.md) files in a specified directory within the project."
  - **Schema:**
    ```json
    {
        "search_directory": {
            "type": "string",
            "description": "The directory to search for GNN (.md) files."
        },
        "recursive": {
            "type": "boolean",
            "description": "Set to true to search recursively. Defaults to false.",
            "optional": true
        }
    }
    ```
- **Tool:** `generate_pipeline_summary_site`
  - **Defined in Module:** `src.site.mcp`
  - **Arguments (from signature):** `(output_dir, site_output_filename, verbose)`
  - **Description:** "Generates a single HTML website summarizing all contents of the GNN pipeline output directory."
  - **Schema:**
    ```json
    {
        "type": "object",
        "properties": {
            "output_dir": {
                "type": "string",
                "description": "The main pipeline output directory to scan for results."
            },
            "site_output_filename": {
                "type": "string",
                "description": "The filename for the output HTML report (e.g., 'summary.html')."
            },
            "verbose": {
                "type": "boolean",
                "description": "Enable verbose logging for the generator."
            }
        },
        "required": [
            "output_dir",
            "site_output_filename"
        ]
    }
    ```
- **Tool:** `get_gnn_documentation`
  - **Defined in Module:** `src.gnn.mcp`
  - **Arguments (from signature):** `(doc_name)`
  - **Description:** "Retrieve the content of a GNN core documentation file (e.g., syntax, file structure)."
  - **Schema:**
    ```json
    {
        "doc_name": {
            "type": "string",
            "description": "Name of the GNN document (e.g., 'file_structure', 'punctuation')",
            "enum": [
                "file_structure",
                "punctuation"
            ]
        }
    }
    ```
- **Tool:** `get_standard_output_paths`
  - **Defined in Module:** `src.setup.mcp`
  - **Arguments (from signature):** `(base_output_directory)`
  - **Description:** "Gets a dictionary of standard output directory paths (e.g., for type_check, visualization), creating them if needed."
  - **Schema:**
    ```json
    {
        "base_output_directory": {
            "type": "string",
            "description": "The base directory where output subdirectories will be managed."
        }
    }
    ```
- **Tool:** `list_render_targets`
  - **Defined in Module:** `src.render.mcp`
  - **Arguments (from signature):** `()`
  - **Description:** "Lists the available target formats for GNN rendering (e.g., pymdp, rxinfer)."
  - **Schema:**
    ```json
    {
        "properties": {},
        "title": "ListRenderTargetsInput",
        "type": "object"
    }
    ```
- **Tool:** `llm.explain_gnn_file`
  - **Defined in Module:** `src.llm.mcp`
  - **Arguments (from signature):** `(file_path_str, aspect_to_explain)`
  - **Description:** "Reads a GNN specification file and uses an LLM to generate an explanation of its content. Can focus on a specific aspect if provided."
  - **Schema:**
    ```json
    {
        "type": "object",
        "properties": {
            "file_path_str": {
                "type": "string",
                "description": "The absolute or relative path to the GNN file."
            },
            "aspect_to_explain": {
                "type": "string",
                "description": "(Optional) A specific part or concept within the GNN to focus the explanation on."
            }
        },
        "required": [
            "file_path_str"
        ]
    }
    ```
- **Tool:** `llm.generate_professional_summary`
  - **Defined in Module:** `src.llm.mcp`
  - **Arguments (from signature):** `(file_path_str, experiment_details, target_audience)`
  - **Description:** "Reads a GNN file and optional experiment details, then uses an LLM to generate a professional summary suitable for reports or papers."
  - **Schema:**
    ```json
    {
        "type": "object",
        "properties": {
            "file_path_str": {
                "type": "string",
                "description": "The absolute or relative path to the GNN file."
            },
            "experiment_details": {
                "type": "string",
                "description": "(Optional) Text describing the experiments conducted with the model, including setup, results, or observations."
            },
            "target_audience": {
                "type": "string",
                "description": "(Optional) The intended audience for the summary (e.g., 'fellow researchers', 'project managers'). Default: 'fellow researchers'."
            }
        },
        "required": [
            "file_path_str"
        ]
    }
    ```
- **Tool:** `llm.summarize_gnn_file`
  - **Defined in Module:** `src.llm.mcp`
  - **Arguments (from signature):** `(file_path_str, user_prompt_suffix)`
  - **Description:** "Reads a GNN specification file and uses an LLM to generate a concise summary of its content. Optionally, a user prompt suffix can refine the summary focus."
  - **Schema:**
    ```json
    {
        "type": "object",
        "properties": {
            "file_path_str": {
                "type": "string",
                "description": "The absolute or relative path to the GNN file (.md, .gnn.md, .json)."
            },
            "user_prompt_suffix": {
                "type": "string",
                "description": "(Optional) Additional instructions or focus points for the summary."
            }
        },
        "required": [
            "file_path_str"
        ]
    }
    ```
- **Tool:** `parse_gnn_file`
  - **Defined in Module:** `src.visualization.mcp`
  - **Arguments (from signature):** `(file_path)`
  - **Description:** "Parse a GNN file without visualization"
  - **Schema:**
    ```json
    {
        "file_path": {
            "type": "string",
            "description": "Path to the GNN file to parse"
        }
    }
    ```
- **Tool:** `render_gnn_specification`
  - **Defined in Module:** `src.render.mcp`
  - **Arguments (from signature):** `(input_data)`
  - **Description:** "Renders a GNN (Generalized Notation Notation) specification into an executable format for a target modeling environment like PyMDP or RxInfer.jl."
  - **Schema:**
    ```json
    {
        "properties": {
            "gnn_specification": {
                "anyOf": [
                    {
                        "additionalProperties": true,
                        "type": "object"
                    },
                    {
                        "type": "string"
                    }
                ],
                "description": "The GNN specification itself as a dictionary, or a string URI/path to a GNN spec file (e.g., JSON).",
                "title": "Gnn Specification"
            },
            "target_format": {
                "description": "The target format to render the GNN specification to.",
                "enum": [
                    "pymdp",
                    "rxinfer"
                ],
                "title": "Target Format",
                "type": "string"
            },
            "output_filename_base": {
                "anyOf": [
                    {
                        "type": "string"
                    },
                    {
                        "type": "null"
                    }
                ],
                "default": null,
                "description": "Optional desired base name for the output file (e.g., 'my_model'). Extension is added automatically. If None, derived from GNN spec name or input file name.",
                "title": "Output Filename Base"
            },
            "render_options": {
                "anyOf": [
                    {
                        "additionalProperties": true,
                        "type": "object"
                    },
                    {
                        "type": "null"
                    }
                ],
                "default": null,
                "description": "Optional dictionary of specific options for the chosen renderer (e.g., data_bindings for RxInfer).",
                "title": "Render Options"
            }
        },
        "required": [
            "gnn_specification",
            "target_format"
        ],
        "title": "RenderGnnInput",
        "type": "object"
    }
    ```
- **Tool:** `run_gnn_type_checker`
  - **Defined in Module:** `src.tests.mcp`
  - **Arguments (from signature):** `(file_path)`
  - **Description:** "Run the GNN type checker on a specific file (via test module)."
  - **Schema:**
    ```json
    {
        "file_path": {
            "type": "string",
            "description": "Path to the GNN file to check"
        }
    }
    ```
- **Tool:** `run_gnn_type_checker_on_directory`
  - **Defined in Module:** `src.tests.mcp`
  - **Arguments (from signature):** `(dir_path, report_file)`
  - **Description:** "Run the GNN type checker on all GNN files in a directory (via test module)."
  - **Schema:**
    ```json
    {
        "dir_path": {
            "type": "string",
            "description": "Path to directory containing GNN files"
        },
        "report_file": {
            "type": "string",
            "description": "Optional path to save the report"
        }
    }
    ```
- **Tool:** `run_gnn_unit_tests`
  - **Defined in Module:** `src.tests.mcp`
  - **Arguments (from signature):** `()`
  - **Description:** "Run the GNN unit tests and return results."
  - **Schema:**
    ```json
    No schema provided.
    ```
- **Tool:** `sympy_analyze_stability`
  - **Defined in Module:** `src.mcp.sympy_mcp`
  - **Arguments (from signature):** `(transition_matrices)`
  - **Description:** "Analyze system stability using eigenvalue analysis"
  - **Schema:**
    ```json
    {
        "type": "object",
        "properties": {
            "transition_matrices": {
                "type": "array",
                "description": "List of transition matrices to analyze"
            }
        },
        "required": [
            "transition_matrices"
        ]
    }
    ```
- **Tool:** `sympy_cleanup`
  - **Defined in Module:** `src.mcp.sympy_mcp`
  - **Arguments (from signature):** `()`
  - **Description:** "Clean up SymPy MCP integration and reset state"
  - **Schema:**
    ```json
    {
        "type": "object",
        "properties": {}
    }
    ```
- **Tool:** `sympy_get_latex`
  - **Defined in Module:** `src.mcp.sympy_mcp`
  - **Arguments (from signature):** `(expression)`
  - **Description:** "Convert a mathematical expression to LaTeX format"
  - **Schema:**
    ```json
    {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Expression to convert to LaTeX"
            }
        },
        "required": [
            "expression"
        ]
    }
    ```
- **Tool:** `sympy_initialize`
  - **Defined in Module:** `src.mcp.sympy_mcp`
  - **Arguments (from signature):** `(server_executable)`
  - **Description:** "Initialize SymPy MCP integration"
  - **Schema:**
    ```json
    {
        "type": "object",
        "properties": {
            "server_executable": {
                "type": "string",
                "description": "Path to SymPy MCP server executable",
                "default": null
            }
        }
    }
    ```
- **Tool:** `sympy_simplify_expression`
  - **Defined in Module:** `src.mcp.sympy_mcp`
  - **Arguments (from signature):** `(expression)`
  - **Description:** "Simplify a mathematical expression to canonical form"
  - **Schema:**
    ```json
    {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to simplify"
            }
        },
        "required": [
            "expression"
        ]
    }
    ```
- **Tool:** `sympy_solve_equation`
  - **Defined in Module:** `src.mcp.sympy_mcp`
  - **Arguments (from signature):** `(equation, variable, domain)`
  - **Description:** "Solve an equation algebraically for a specified variable"
  - **Schema:**
    ```json
    {
        "type": "object",
        "properties": {
            "equation": {
                "type": "string",
                "description": "Equation to solve"
            },
            "variable": {
                "type": "string",
                "description": "Variable to solve for"
            },
            "domain": {
                "type": "string",
                "description": "Solution domain (COMPLEX, REAL, etc.)",
                "default": "COMPLEX"
            }
        },
        "required": [
            "equation",
            "variable"
        ]
    }
    ```
- **Tool:** `sympy_validate_equation`
  - **Defined in Module:** `src.mcp.sympy_mcp`
  - **Arguments (from signature):** `(equation, context)`
  - **Description:** "Validate a mathematical equation using SymPy symbolic processing"
  - **Schema:**
    ```json
    {
        "type": "object",
        "properties": {
            "equation": {
                "type": "string",
                "description": "Mathematical equation to validate"
            },
            "context": {
                "type": "object",
                "description": "GNN context for variable definitions",
                "default": {}
            }
        },
        "required": [
            "equation"
        ]
    }
    ```
- **Tool:** `sympy_validate_matrix`
  - **Defined in Module:** `src.mcp.sympy_mcp`
  - **Arguments (from signature):** `(matrix_data, matrix_type)`
  - **Description:** "Validate matrix properties including stochasticity constraints"
  - **Schema:**
    ```json
    {
        "type": "object",
        "properties": {
            "matrix_data": {
                "type": "array",
                "description": "Matrix data as array of arrays"
            },
            "matrix_type": {
                "type": "string",
                "description": "Type of matrix (transition, observation, etc.)",
                "default": "transition"
            }
        },
        "required": [
            "matrix_data"
        ]
    }
    ```
- **Tool:** `type_check_gnn_directory`
  - **Defined in Module:** `src.gnn_type_checker.mcp`
  - **Arguments (from signature):** `(dir_path, recursive, output_dir_base, report_md_filename)`
  - **Description:** "Runs the GNN type checker on all GNN files in a specified directory. If output_dir_base is provided, reports are generated."
  - **Schema:**
    ```json
    {
        "dir_path": {
            "type": "string",
            "description": "Path to the directory containing GNN files to be type-checked."
        },
        "recursive": {
            "type": "boolean",
            "description": "Search directory recursively. Defaults to False.",
            "optional": true
        },
        "output_dir_base": {
            "type": "string",
            "description": "Optional base directory to save the report and other artifacts (HTML, JSON).",
            "optional": true
        },
        "report_md_filename": {
            "type": "string",
            "description": "Optional filename for the markdown report (e.g., 'my_report.md'). Defaults to 'type_check_report.md'.",
            "optional": true
        }
    }
    ```
- **Tool:** `type_check_gnn_file`
  - **Defined in Module:** `src.gnn_type_checker.mcp`
  - **Arguments (from signature):** `(file_path)`
  - **Description:** "Runs the GNN type checker on a specified GNN model file."
  - **Schema:**
    ```json
    {
        "file_path": {
            "type": "string",
            "description": "Path to the GNN file to be type-checked."
        }
    }
    ```
- **Tool:** `visualize_gnn_directory`
  - **Defined in Module:** `src.visualization.mcp`
  - **Arguments (from signature):** `(dir_path, output_dir)`
  - **Description:** "Visualize all GNN files in a directory"
  - **Schema:**
    ```json
    {
        "dir_path": {
            "type": "string",
            "description": "Path to directory containing GNN files"
        },
        "output_dir": {
            "type": "string",
            "description": "Optional output directory"
        }
    }
    ```
- **Tool:** `visualize_gnn_file`
  - **Defined in Module:** `src.visualization.mcp`
  - **Arguments (from signature):** `(file_path, output_dir)`
  - **Description:** "Generate visualizations for a specific GNN file."
  - **Schema:**
    ```json
    {
        "file_path": {
            "type": "string",
            "description": "Path to the GNN file to visualize"
        },
        "output_dir": {
            "type": "string",
            "description": "Optional output directory"
        }
    }
    ```



## 🔬 Core MCP File Check

This section verifies the presence of essential MCP files in the core directory: `/home/trim/Documents/GitHub/GeneralizedNotationNotation/src/mcp`

- ✅ `mcp.py`: Found (20304 bytes)
- ✅ `meta_mcp.py`: Found (4954 bytes)
- ✅ `cli.py`: Found (4644 bytes)
- ✅ `server_stdio.py`: Found (7620 bytes)
- ✅ `server_http.py`: Found (7731 bytes)

**Status:** 5/5 core MCP files found. All core files seem present.

## 🧩 Functional Module MCP Integration & API Check

Checking for `mcp.py` in these subdirectories of `/home/trim/Documents/GitHub/GeneralizedNotationNotation/src`: ['export', 'gnn', 'gnn_type_checker', 'ontology', 'setup', 'tests', 'visualization', 'llm']

### Module: `export` (at `/home/trim/Documents/GitHub/GeneralizedNotationNotation/src/export`)
- ✅ **`mcp.py` Status:** Found (7976 bytes)
- **Exposed Methods & Tools:**
  - `def _handle_export(export_func, gnn_file_path, output_file_path, format_name, requires_nx)` (AST parsed) - *"Generic helper to run an export function and handle common exceptions."
  - `def export_gnn_to_gexf(gnn_file_path, output_file_path)` - *Description: "Exports a GNN model to GEXF graph format (requires NetworkX)."
    - Schema:
      ```json
      {
      "gnn_file_path": {
        "type": "string",
        "description": "Path to the input GNN Markdown file (.gnn.md)."
      },
      "output_file_path": {
        "type": "string",
        "description": "Path where the exported file will be saved."
      }
    }
      ```
  - `def export_gnn_to_gexf_mcp(gnn_file_path, output_file_path)` (AST parsed)
  - `def export_gnn_to_graphml(gnn_file_path, output_file_path)` - *Description: "Exports a GNN model to GraphML graph format (requires NetworkX)."
    - Schema:
      ```json
      {
      "gnn_file_path": {
        "type": "string",
        "description": "Path to the input GNN Markdown file (.gnn.md)."
      },
      "output_file_path": {
        "type": "string",
        "description": "Path where the exported file will be saved."
      }
    }
      ```
  - `def export_gnn_to_graphml_mcp(gnn_file_path, output_file_path)` (AST parsed)
  - `def export_gnn_to_json(gnn_file_path, output_file_path)` - *Description: "Exports a GNN model to JSON format."
    - Schema:
      ```json
      {
      "gnn_file_path": {
        "type": "string",
        "description": "Path to the input GNN Markdown file (.gnn.md)."
      },
      "output_file_path": {
        "type": "string",
        "description": "Path where the exported file will be saved."
      }
    }
      ```
  - `def export_gnn_to_json_adjacency_list(gnn_file_path, output_file_path)` - *Description: "Exports a GNN model to JSON Adjacency List graph format (requires NetworkX)."
    - Schema:
      ```json
      {
      "gnn_file_path": {
        "type": "string",
        "description": "Path to the input GNN Markdown file (.gnn.md)."
      },
      "output_file_path": {
        "type": "string",
        "description": "Path where the exported file will be saved."
      }
    }
      ```
  - `def export_gnn_to_json_adjacency_list_mcp(gnn_file_path, output_file_path)` (AST parsed)
  - `def export_gnn_to_json_mcp(gnn_file_path, output_file_path)` (AST parsed)
  - `def export_gnn_to_plaintext_dsl(gnn_file_path, output_file_path)` - *Description: "Exports a GNN model back to its GNN DSL plain text format."
    - Schema:
      ```json
      {
      "gnn_file_path": {
        "type": "string",
        "description": "Path to the input GNN Markdown file (.gnn.md)."
      },
      "output_file_path": {
        "type": "string",
        "description": "Path where the exported file will be saved."
      }
    }
      ```
  - `def export_gnn_to_plaintext_dsl_mcp(gnn_file_path, output_file_path)` (AST parsed)
  - `def export_gnn_to_plaintext_summary(gnn_file_path, output_file_path)` - *Description: "Exports a GNN model to a human-readable plain text summary."
    - Schema:
      ```json
      {
      "gnn_file_path": {
        "type": "string",
        "description": "Path to the input GNN Markdown file (.gnn.md)."
      },
      "output_file_path": {
        "type": "string",
        "description": "Path where the exported file will be saved."
      }
    }
      ```
  - `def export_gnn_to_plaintext_summary_mcp(gnn_file_path, output_file_path)` (AST parsed)
  - `def export_gnn_to_python_pickle(gnn_file_path, output_file_path)` - *Description: "Serializes a GNN model to a Python pickle file."
    - Schema:
      ```json
      {
      "gnn_file_path": {
        "type": "string",
        "description": "Path to the input GNN Markdown file (.gnn.md)."
      },
      "output_file_path": {
        "type": "string",
        "description": "Path where the exported file will be saved."
      }
    }
      ```
  - `def export_gnn_to_python_pickle_mcp(gnn_file_path, output_file_path)` (AST parsed)
  - `def export_gnn_to_xml(gnn_file_path, output_file_path)` - *Description: "Exports a GNN model to XML format."
    - Schema:
      ```json
      {
      "gnn_file_path": {
        "type": "string",
        "description": "Path to the input GNN Markdown file (.gnn.md)."
      },
      "output_file_path": {
        "type": "string",
        "description": "Path where the exported file will be saved."
      }
    }
      ```
  - `def export_gnn_to_xml_mcp(gnn_file_path, output_file_path)` (AST parsed)
  - `def register_tools(mcp_instance)` (AST parsed) - *"Registers all GNN export tools with the MCP instance."

---

### Module: `gnn` (at `/home/trim/Documents/GitHub/GeneralizedNotationNotation/src/gnn`)
- ✅ **`mcp.py` Status:** Found (4122 bytes)
- **Exposed Methods & Tools:**
  - `def _retrieve_gnn_doc_resource(uri)` (AST parsed) - *"Retrieve GNN documentation resource by URI."
  - `def get_gnn_documentation(doc_name)` - *Description: "Retrieve the content of a GNN core documentation file (e.g., syntax, file structure)."
    - Schema:
      ```json
      {
      "doc_name": {
        "type": "string",
        "description": "Name of the GNN document (e.g., 'file_structure', 'punctuation')",
        "enum": [
          "file_structure",
          "punctuation"
        ]
      }
    }
      ```
  - `def register_tools(mcp_instance)` (AST parsed) - *"Register GNN documentation tools and resources with the MCP."

---

### Module: `gnn_type_checker` (at `/home/trim/Documents/GitHub/GeneralizedNotationNotation/src/gnn_type_checker`)
- ✅ **`mcp.py` Status:** Found (10921 bytes)
- **Exposed Methods & Tools:**
  - `def estimate_resources_for_gnn_directory(dir_path, recursive)` - *Description: "Estimates computational resources for all GNN files in a specified directory."
    - Schema:
      ```json
      {
      "dir_path": {
        "type": "string",
        "description": "Path to the directory for GNN resource estimation."
      },
      "recursive": {
        "type": "boolean",
        "description": "Search directory recursively. Defaults to False.",
        "optional": true
      }
    }
      ```
  - `def estimate_resources_for_gnn_directory_mcp(dir_path, recursive)` (AST parsed) - *"Estimate resources for all GNN files in a directory. Exposed via MCP."
  - `def estimate_resources_for_gnn_file(file_path)` - *Description: "Estimates computational resources (memory, inference, storage) for a GNN model file."
    - Schema:
      ```json
      {
      "file_path": {
        "type": "string",
        "description": "Path to the GNN file for resource estimation."
      }
    }
      ```
  - `def estimate_resources_for_gnn_file_mcp(file_path)` (AST parsed) - *"Estimate computational resources for a single GNN file. Exposed via MCP."
  - `def register_tools(mcp_instance)` (AST parsed) - *"Register GNN type checker and resource estimator tools with the MCP."
  - `def type_check_gnn_directory(dir_path, recursive, output_dir_base, report_md_filename)` - *Description: "Runs the GNN type checker on all GNN files in a specified directory. If output_dir_base is provided, reports are generated."
    - Schema:
      ```json
      {
      "dir_path": {
        "type": "string",
        "description": "Path to the directory containing GNN files to be type-checked."
      },
      "recursive": {
        "type": "boolean",
        "description": "Search directory recursively. Defaults to False.",
        "optional": true
      },
      "output_dir_base": {
        "type": "string",
        "description": "Optional base directory to save the report and other artifacts (HTML, JSON).",
        "optional": true
      },
      "report_md_filename": {
        "type": "string",
        "description": "Optional filename for the markdown report (e.g., 'my_report.md'). Defaults to 'type_check_report.md'.",
        "optional": true
      }
    }
      ```
  - `def type_check_gnn_directory_mcp(dir_path, recursive, output_dir_base, report_md_filename)` (AST parsed) - *"Run the GNN type checker on all GNN files in a directory. Exposed via MCP."
  - `def type_check_gnn_file(file_path)` - *Description: "Runs the GNN type checker on a specified GNN model file."
    - Schema:
      ```json
      {
      "file_path": {
        "type": "string",
        "description": "Path to the GNN file to be type-checked."
      }
    }
      ```
  - `def type_check_gnn_file_mcp(file_path)` (AST parsed) - *"Run the GNN type checker on a single GNN file. Exposed via MCP."

---

### Module: `ontology` (at `/home/trim/Documents/GitHub/GeneralizedNotationNotation/src/ontology`)
- ✅ **`mcp.py` Status:** Found (13473 bytes)
- **Exposed Methods & Tools:**
  - `def generate_ontology_report_for_file(gnn_file_path, parsed_annotations, validation_results)` (AST parsed) - *"Generates a markdown formatted report string for a single GNN file's ontology annotations."
  - `def get_mcp_interface()` (AST parsed) - *"Returns the MCP interface for the Ontology module."
  - `def load_defined_ontology_terms(ontology_terms_path, verbose)` (AST parsed) - *"Loads defined ontological terms from a JSON file."
  - `def parse_gnn_ontology_section(gnn_file_content, verbose)` (AST parsed) - *"Parses the 'ActInfOntologyAnnotation' section from GNN file content."
  - `def validate_annotations(parsed_annotations, defined_terms, verbose)` (AST parsed) - *"Validates parsed GNN annotations against a set of defined ontological terms."

---

### Module: `setup` (at `/home/trim/Documents/GitHub/GeneralizedNotationNotation/src/setup`)
- ✅ **`mcp.py` Status:** Found (4257 bytes)
- **Exposed Methods & Tools:**
  - `def ensure_directory_exists(directory_path)` - *Description: "Ensures a directory exists, creating it if necessary. Returns the absolute path."
    - Schema:
      ```json
      {
      "directory_path": {
        "type": "string",
        "description": "Path of the directory to create if it doesn't exist."
      }
    }
      ```
  - `def ensure_directory_exists_mcp(directory_path)` (AST parsed) - *"Ensure a directory exists, creating it if necessary. Exposed via MCP."
  - `def find_project_gnn_files(search_directory, recursive)` - *Description: "Finds all GNN (.md) files in a specified directory within the project."
    - Schema:
      ```json
      {
      "search_directory": {
        "type": "string",
        "description": "The directory to search for GNN (.md) files."
      },
      "recursive": {
        "type": "boolean",
        "description": "Set to true to search recursively. Defaults to false.",
        "optional": true
      }
    }
      ```
  - `def find_project_gnn_files_mcp(search_directory, recursive)` (AST parsed) - *"Find all GNN (.md) files in a directory. Exposed via MCP."
  - `def get_standard_output_paths(base_output_directory)` - *Description: "Gets a dictionary of standard output directory paths (e.g., for type_check, visualization), creating them if needed."
    - Schema:
      ```json
      {
      "base_output_directory": {
        "type": "string",
        "description": "The base directory where output subdirectories will be managed."
      }
    }
      ```
  - `def get_standard_output_paths_mcp(base_output_directory)` (AST parsed) - *"Get standard output paths for the pipeline. Exposed via MCP."
  - `def register_tools(mcp_instance)` (AST parsed) - *"Register setup utility tools with the MCP."

---

### Module: `tests` (at `/home/trim/Documents/GitHub/GeneralizedNotationNotation/src/tests`)
- ✅ **`mcp.py` Status:** Found (7083 bytes)
- **Exposed Methods & Tools:**
  - `def get_test_report(uri)` (AST parsed) - *"Retrieve a test report by URI."
  - `def register_tools(mcp)` (AST parsed) - *"Register test tools with the MCP."
  - `def run_gnn_type_checker(file_path)` - *Description: "Run the GNN type checker on a specific file (via test module)."
    - Schema:
      ```json
      {
      "file_path": {
        "type": "string",
        "description": "Path to the GNN file to check"
      }
    }
      ```
  - `def run_gnn_type_checker_on_directory(dir_path, report_file)` - *Description: "Run the GNN type checker on all GNN files in a directory (via test module)."
    - Schema:
      ```json
      {
      "dir_path": {
        "type": "string",
        "description": "Path to directory containing GNN files"
      },
      "report_file": {
        "type": "string",
        "description": "Optional path to save the report"
      }
    }
      ```
  - `def run_gnn_unit_tests()` - *Description: "Run the GNN unit tests and return results."
  - `def run_type_checker_on_directory(dir_path, report_file)` (AST parsed) - *"Run the GNN type checker on a directory of files."
  - `def run_type_checker_on_file(file_path)` (AST parsed) - *"Run the GNN type checker on a file."
  - `def run_unit_tests()` (AST parsed) - *"Run the GNN unit tests."

---

### Module: `visualization` (at `/home/trim/Documents/GitHub/GeneralizedNotationNotation/src/visualization`)
- ✅ **`mcp.py` Status:** Found (5934 bytes)
- **Exposed Methods & Tools:**
  - `def get_visualization_results(uri)` (AST parsed) - *"Retrieve visualization results by URI."
  - `def parse_gnn_file(file_path)` - *Description: "Parse a GNN file without visualization"
    - Schema:
      ```json
      {
      "file_path": {
        "type": "string",
        "description": "Path to the GNN file to parse"
      }
    }
      ```
  - `def register_tools(mcp)` (AST parsed) - *"Register visualization tools with the MCP."
  - `def visualize_directory(dir_path, output_dir)` (AST parsed) - *"Visualize all GNN files in a directory through MCP."
  - `def visualize_file(file_path, output_dir)` (AST parsed) - *"Visualize a GNN file through MCP."
  - `def visualize_gnn_directory(dir_path, output_dir)` - *Description: "Visualize all GNN files in a directory"
    - Schema:
      ```json
      {
      "dir_path": {
        "type": "string",
        "description": "Path to directory containing GNN files"
      },
      "output_dir": {
        "type": "string",
        "description": "Optional output directory"
      }
    }
      ```
  - `def visualize_gnn_file(file_path, output_dir)` - *Description: "Generate visualizations for a specific GNN file."
    - Schema:
      ```json
      {
      "file_path": {
        "type": "string",
        "description": "Path to the GNN file to visualize"
      },
      "output_dir": {
        "type": "string",
        "description": "Optional output directory"
      }
    }
      ```

---

### Module: `llm` (at `/home/trim/Documents/GitHub/GeneralizedNotationNotation/src/llm`)
- ✅ **`mcp.py` Status:** Found (19238 bytes)
- **Exposed Methods & Tools:**
  - `def ensure_llm_tools_registered(mcp_instance_ref)` (AST parsed) - *"Ensures that LLM tools are registered with the provided MCP instance."
  - `def explain_gnn_file_content(file_path_str, aspect_to_explain)` (AST parsed) - *"Reads a GNN file, sends its content to an LLM, and returns an explanation."
  - `def generate_professional_summary_from_gnn(file_path_str, experiment_details, target_audience)` (AST parsed) - *"Generates a professional summary of a GNN model and its experimental context."
  - `def initialize_llm_module(mcp_instance_ref)` (AST parsed) - *"Initializes the LLM module, loads API key, and updates MCP status."
  - `def llm.explain_gnn_file(file_path_str, aspect_to_explain)` - *Description: "Reads a GNN specification file and uses an LLM to generate an explanation of its content. Can focus on a specific aspect if provided."
    - Schema:
      ```json
      {
      "type": "object",
      "properties": {
        "file_path_str": {
          "type": "string",
          "description": "The absolute or relative path to the GNN file."
        },
        "aspect_to_explain": {
          "type": "string",
          "description": "(Optional) A specific part or concept within the GNN to focus the explanation on."
        }
      },
      "required": [
        "file_path_str"
      ]
    }
      ```
  - `def llm.generate_professional_summary(file_path_str, experiment_details, target_audience)` - *Description: "Reads a GNN file and optional experiment details, then uses an LLM to generate a professional summary suitable for reports or papers."
    - Schema:
      ```json
      {
      "type": "object",
      "properties": {
        "file_path_str": {
          "type": "string",
          "description": "The absolute or relative path to the GNN file."
        },
        "experiment_details": {
          "type": "string",
          "description": "(Optional) Text describing the experiments conducted with the model, including setup, results, or observations."
        },
        "target_audience": {
          "type": "string",
          "description": "(Optional) The intended audience for the summary (e.g., 'fellow researchers', 'project managers'). Default: 'fellow researchers'."
        }
      },
      "required": [
        "file_path_str"
      ]
    }
      ```
  - `def llm.summarize_gnn_file(file_path_str, user_prompt_suffix)` - *Description: "Reads a GNN specification file and uses an LLM to generate a concise summary of its content. Optionally, a user prompt suffix can refine the summary focus."
    - Schema:
      ```json
      {
      "type": "object",
      "properties": {
        "file_path_str": {
          "type": "string",
          "description": "The absolute or relative path to the GNN file (.md, .gnn.md, .json)."
        },
        "user_prompt_suffix": {
          "type": "string",
          "description": "(Optional) Additional instructions or focus points for the summary."
        }
      },
      "required": [
        "file_path_str"
      ]
    }
      ```
  - `def register_tools(mcp_instance_ref)` (AST parsed)
  - `def summarize_gnn_file_content(file_path_str, user_prompt_suffix)` (AST parsed) - *"Reads a GNN file, sends its content to an LLM, and returns a summary."

---


## 📊 Overall Module Integration Summary

- **Modules Checked:** 8
- **`mcp.py` Integrations Found:** 8/8
- **Status:** All expected functional modules appear to have an `mcp.py` integration file.
  Please ensure each functional module that should be exposed via MCP has its own `mcp.py` following the project's MCP architecture.
