# GNN Pipeline Analysis Report

**Generated:** 2025-09-15T13:31:57.076172

## Executive Summary

- **Health Score:** 94.0/100
- **Total Steps:** 7
- **Successful Steps:** 7
- **Failed Steps:** 0
- **Warnings:** 2
- **Files Generated:** 101
- **Total Size:** 1.88 MB

## Pipeline Steps Analysis


### ✅ 3_gnn.py
- **Status:** SUCCESS
- **Duration:** 0.14s
- **Memory:** 23.8 MB
- **Description:** GNN file processing


### ⚠️ 5_type_checker.py
- **Status:** SUCCESS_WITH_WARNINGS
- **Duration:** 0.09s
- **Memory:** 23.8 MB
- **Description:** Type checking


### ✅ 7_export.py
- **Status:** SUCCESS
- **Duration:** 0.11s
- **Memory:** 23.8 MB
- **Description:** Multi-format export


### ✅ 8_visualization.py
- **Status:** SUCCESS
- **Duration:** 5.78s
- **Memory:** 23.8 MB
- **Description:** Visualization


### ✅ 11_render.py
- **Status:** SUCCESS
- **Duration:** 0.15s
- **Memory:** 23.8 MB
- **Description:** Code rendering


### ⚠️ 12_execute.py
- **Status:** SUCCESS_WITH_WARNINGS
- **Duration:** 8.23s
- **Memory:** 23.8 MB
- **Description:** Execution
- **Errors:**
  - ❌ 2025-09-15 13:31:32,730 - execute - ERROR - Error output: Traceback (most recent call last):
  - ❌ 2025-09-15 13:31:34,040 - execute - ERROR - Error output: ERROR: LoadError: ArgumentError: Package RxInfer not found in current path.
  - ❌ 2025-09-15 13:31:35,353 - execute - ERROR - Error output: ERROR: LoadError: ArgumentError: Package ActiveInference not found in current path.


### ✅ 15_audio.py
- **Status:** SUCCESS
- **Duration:** 0.13s
- **Memory:** 23.8 MB
- **Description:** Audio processing


## File Generation Analysis

- **Total Files:** 101
- **Total Size:** 1.88 MB

### File Types

| Type | Count | Total Size (MB) |
|------|-------|----------------|
| .json | 36 | 0.13 |
| .md | 12 | 0.03 |
| .yaml | 2 | 0.01 |
| no extension | 1 | 0.00 |
| .py | 5 | 0.03 |
| .log | 7 | 0.03 |
| .scala | 1 | 0.00 |
| .lean | 1 | 0.01 |
| .v | 1 | 0.01 |
| .bnf | 1 | 0.01 |
| .ebnf | 1 | 0.01 |
| .thy | 1 | 0.01 |
| .max | 1 | 0.01 |
| .xml | 2 | 0.01 |
| .proto | 1 | 0.01 |
| .xsd | 1 | 0.01 |
| .asn1 | 1 | 0.01 |
| .pkl | 3 | 0.03 |
| .als | 1 | 0.01 |
| .z | 1 | 0.01 |
| .tla | 1 | 0.00 |
| .agda | 1 | 0.00 |
| .hs | 1 | 0.01 |
| .graphml | 1 | 0.00 |
| .gexf | 1 | 0.00 |
| .png | 7 | 1.51 |
| .html | 2 | 0.00 |
| .jl | 2 | 0.01 |
| .txt | 5 | 0.00 |

### Largest Files

- actinf_pomdp_agent_combined_analysis.png (0.38 MB)
- actinf_pomdp_agent_network_graph.png (0.28 MB)
- matrix_analysis.png (0.25 MB)
- actinf_pomdp_agent_combined_analysis.png (0.24 MB)
- matrix_statistics.png (0.19 MB)
- actinf_pomdp_agent_network_graph.png (0.11 MB)
- pomdp_transition_analysis.png (0.05 MB)
- llm_results.json (0.03 MB)
- pytest_stdout.log (0.02 MB)
- actinf_pomdp_agent_parsed.json (0.02 MB)


## Error Analysis

- **Total Errors:** 3
- **Total Warnings:** 0

### Common Issues

| Issue Type | Count |
|------------|-------|
| Other error | 1 |
| File not found | 2 |


## Performance Analysis

- **Total Duration:** 14.63 seconds
- **Average Step Duration:** 2.09 seconds
- **Efficiency Score:** 99.9/100

- **Average Memory:** 23.8 MB
- **Maximum Memory:** 23.8 MB
- **Minimum Memory:** 23.8 MB

### Slowest Steps

| Step | Duration (s) | Description |
|------|--------------|-------------|
| 12_execute.py | 8.23 | Execution |
| 8_visualization.py | 5.78 | Visualization |


## Recommendations

1. Review 2 steps with warnings to prevent future issues
2. Optimize 12_execute.py which took 8.2s
3. Pipeline is performing well - consider optimization opportunities

