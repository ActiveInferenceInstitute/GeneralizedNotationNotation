# visualization.matrix

| Symbol | Location |
|--------|----------|
| `MatrixVisualizer`, `generate_matrix_visualizations`, `process_matrix_visualization` | `visualizer.py` |
| `convert_to_matrix`, `extract_matrix_data_from_parameters` | `extract.py` |
| `parse_matrix_data`, `generate_matrix_visualizations` | `compat.py` — string parse + batch matrix plots |

Root `matrix_compat.py` re-exports `compat`. Shared plotting imports: `visualization.compat.viz_compat`. Tight layout: `visualization.plotting.utils.safe_tight_layout`.

Root `matrix_visualizer.py` re-exports the public class and module functions.
