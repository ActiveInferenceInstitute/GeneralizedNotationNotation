# Pipeline Intelligent Analysis Report

**Generated**: 2026-03-24T14:13:06.912488

**Status**: ✅ SUCCESS

**Health Score**: 100.0/100


## Quick Overview

| Metric | Value |
|--------|-------|
| Total Steps | 24 |
| Duration | 1087.40s |
| Peak Memory | 88.5 MB |
| 🔴 Red Flags | 0 |
| 🟡 Yellow Flags | 4 |
| ✅ Green (Clean) | 20 |

## AI-Powered Analysis

✅ **0_template.py** (Template initialization): Completed successfully in 0.90s
✅ **1_setup.py** (Environment setup): Completed successfully in 14.83s | Flags: Very slow: 120.0s (>120.0s threshold)
🟡 **2_tests.py** (Test suite execution): Completed with 1 flag(s) in 120.95s | Flags: Very slow: 120.9s (>120.0s threshold)
✅ **3_gnn.py** (GNN file processing): Completed successfully in 3.22s | Flags: Very slow: 120.9s (>120.0s threshold)
✅ **4_model_registry.py** (Model registry): Completed successfully in 0.46s | Flags: Very slow: 512.0s (>120.0s threshold)
✅ **5_type_checker.py** (Type checking): Completed successfully in 2.94s | Flags: Very slow: 281.56s
✅ **6_validation.py** (Validation): Completed successfully in 2.86s | Flags: Very slow: 27.91s (>120.0s threshold)
✅ **7_export.py** (Multi-format export): Completed successfully in 0.47s | Flags: Very slow: 512.0s (>120.0s threshold)
✅ **8_visualization.py** (Visualization): Completed successfully in 27.91s | Flags: Very slow: 281.56s
✅ **9_advanced_viz.py** (Advanced visualization): Completed successfully in 27.91s | Flags: Very slow: 281.56s
✅ **10_ontology.py** (Ontology processing): Completed successfully in 2.90s | Flags: Very slow: 281.56s
✅ **11_render.py** (Code rendering): Completed successfully in 1.91s | Flags: Very slow: 27.91s (>120.0s threshold)
🟡 **12_execute.py** (Execution): Completed with 1 flag(s) in 512.00s | Flags: Very slow: 281.56s (<120.0s threshold)
✅ **13_llm.py** (LLM processing): Completed successfully in 3.01s | Flags: Very slow: 77.5s (>60.0s threshold)
✅ **14_ml_integration.**: Completed successfully in 0.66s | Flags: Very slow: 77.5s (>60.0s threshold)
✅ **15_audio.py** (Audio processing): Completed successfully in 2.97s | Flags: Very slow: 281.56s (<120.0s threshold)
✅ **16_analysis.**(Analysis): Completed with 1 flag(s) in 77.55s | Flags: Very slow: 77.55s (>120.0s threshold)
✅ **17_integration.**(Integration): Completed successfully in 0.66s | Flags: Very slow: 281.56s (<120.0s threshold)
✅ **19_security.**(Security): Completed successfully in 0.40s | Flags: Very slow: 47.3 seconds (time to complete all steps)
✅ **20_website.**(Website generation): Completed successfully in 0.65s | Flags: Very slow: 198.0 seconds (<120.0s threshold)
✅ **21_mcp.**(Model context protocol processing): Completed successfully in 3.01s | Flags: Very slow: 47.3 seconds (time to complete all steps)
✅ **22_gui.**(GUI (Interactive GNN Constructor)): Completed successfully in 0.51s | Flags: Very slow: 69.8 seconds (<120.0s threshold)
✅ **23_report.**(Report generation): Completed successfully in 0.63s | Flags: Very slow: 47.3 seconds (time to complete all steps)

## Key Performance Indicators (KPIs)
### Total Steps
- **Total time**: 1087.40ms
- **Time spent on each task**: 24msecs
- **Total number of tasks**: 5693 seconds
- **Average time per task**: 2.94seconds (average)
### Total Time Spent in Tasks
- **Total time spent**: 1087ms
- **Time spent on each task**: 24msecs
- **Total number of tasks**: 5693 seconds
- **Average time per task**: 2.94seconds (average)

## 🟡 Yellow Flags (Warnings)

| Step | Duration | Memory | Issues |
|------|----------|--------|--------|
| 2_tests.py | 120.9s | 30MB | Very slow: 120.9s (>120.0s threshold) |
| 12_execute.py | 512.0s | 34MB | Very slow: 512.0s (>120.0s threshold) |
| 13_llm.py | 281.5s | 19MB | Very slow: 281.5s (>120.0s threshold) |
| 16_analysis.py | 77.5s | 24MB | Slow: 77.5s (>60.0s threshold) |

## Per-Step Execution Details

| # | Step | Status | Duration | Memory | Flags |
|---|------|--------|----------|--------|-------|
| 1 | 0_template.py | ✅ SUCCESS | 0.90s | 88MB | - |
| 2 | 1_setup.py | ✅ SUCCESS | 14.83s | 88MB | - |
| 3 | 2_tests.py | 🟡 SUCCESS | 120.95s | 30MB | 1 |
| 4 | 3_gnn.py | ✅ SUCCESS | 3.22s | 34MB | - |
| 5 | 4_model_registry.py | ✅ SUCCESS | 0.46s | 34MB | - |
| 6 | 5_type_checker.py | ✅ SUCCESS | 2.94s | 35MB | - |
| 7 | 6_validation.py | ✅ SUCCESS | 2.86s | 35MB | - |
| 8 | 7_export.py | ✅ SUCCESS | 0.47s | 35MB | - |
| 9 | 8_visualization.py | ✅ SUCCESS | 27.12s | 35MB | - |
| 10 | 9_advanced_viz.py | ✅ SUCCESS | 27.91s | 30MB | - |
| 11 | 10_ontology.py | ✅ SUCCESS | 2.90s | 34MB | - |
| 12 | 11_render.py | ✅ SUCCESS | 1.91s | 34MB | - |
| 13 | 12_execute.py | 🟡 SUCCESS | 512.00s | 34MB | 1 |
| 14 | 13_llm.py | 🟡 SUCCESS | 281.46s | 19MB | 1 |
| 15 | 14_ml_integration.py | ✅ SUCCESS | 0.56s | 24MB | - |
| 16 | 15_audio.py | ✅ SUCCESS | 2.97s | 24MB | - |
| 17 | 16_analysis.py | 🟡 SUCCESS | 77.55s | 24MB | 1 |
| 18 | 17_integration.py | ✅ SUCCESS | 0.66s | 23MB | - |
| 19 | 18_security.py | ✅ SUCCESS | 0.40s | 23MB | - |
| 20 | 19_research.py | ✅ SUCCESS | 0.45s | 23MB | - |
| 21 | 20_website.py | ✅ SUCCESS | 0.65s | 24MB | - |
| 22 | 21_mcp.py | ✅ SUCCESS | 3.01s | 24MB | - |
| 23 | 22_gui.py | ✅ SUCCESS | 0.51s | 24MB | - |
| 24 | 23_report.py | ✅ SUCCESS | 0.63s | 25MB | - |

## Detailed Step Output (Flagged Steps)

### 2_tests.py

**Test suite execution**

- Status: SUCCESS
- Duration: 120.95s
- Memory: 30MB
- Flags: Very slow: 120.9s (>120.0s threshold)

**Output Snippet**:
```
2026-03-24 13:55:11,978 - 2_tests.py - INFO - Executing fast tests: /Users/4d/Documents/GitHub/generalizednotationnotation/.venv/bin/python -m pytest --tb=short --maxfail=5 --durations=10 -ra --timeout 600 -q -m not slow --ignore=src/tests/test_llm_ollama.py --ignore=src/tests/test_llm_ollama_integration.py --ignore=src/tests/test_pipeline_performance.py --ignore=src/tests/test_pipeline_recovery.py --ignore=src/tests/test_report_integration.py src/tests/
```

**Error Output**:
```
2026-03-24 13:55:11,978 - 2_tests.py - INFO - Executing fast tests: /Users/4d/Documents/GitHub/generalizednotationnotation/.venv/bin/python -m pytest --tb=short --maxfail=5 --durations=10 -ra --timeout 600 -q -m not slow --ignore=src/tests/test_llm_ollama.py --ignore=src/tests/test_llm_ollama_integration.py --ignore=src/tests/test_pipeline_performance.py --ignore=src/tests/test_pipeline_recovery.py --ignore=src/tests/test_report_integration.py src/tests/
```

### 12_execute.py

**Execution**

- Status: SUCCESS
- Duration: 512.00s
- Memory: 34MB
- Flags: Very slow: 512.0s (>120.0s threshold)

**Output Snippet**:
```
--- Output for discrete ---
```

**Error Output**:
```
2026-03-24 13:58:38,413 - execute - INFO - ✅ Successfully executed Simple MDP Agent_numpyro.py
2026-03-24 13:58:41,522 - execute - INFO - ✅ Successfully executed Simple MDP Agent_pymdp.py
2026-03-24 13:58:43,764 - execute - INFO - ✅ Successfully executed Simple MDP Agent_pytorch.py
2026-03-24 13:58:45,483 - execute - INFO - ✅ Successfully executed Simple MDP Agent_jax.py
2026-03-24 13:58:46,400 - execute - INFO - ✅ Successfully executed Simple MDP Agent_discopy.py
```

### 13_llm.py

**LLM processing**

- Status: SUCCESS
- Duration: 281.46s
- Memory: 19MB
- Flags: Very slow: 281.5s (>120.0s threshold)

**Error Output**:
```
2026-03-24 14:06:54,890 - llm.providers.openai_provider - INFO - OpenAI provider initialized successfully
2026-03-24 14:06:54,901 - llm.providers.openai_provider - INFO - OpenAI provider initialized successfully
2026-03-24 14:06:57,038 - llm.providers.openai_provider - ERROR - OpenAI API call failed: Error code: 429 - {'error': {'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, read the docs: https://platform.openai.com/d
```

### 16_analysis.py

**Analysis**

- Status: SUCCESS
- Duration: 77.55s
- Memory: 24MB
- Flags: Slow: 77.5s (>60.0s threshold)

**Output Snippet**:
```
--- Output for discrete ---
```

**Error Output**:
```
2026-03-24 14:12:55,080 - analysis - INFO - ✅ Analysis processing completed successfully
```

## Performance Bottlenecks

| Step | Duration (s) | Memory (MB) | Above Avg Ratio |
|------|-------------|-------------|-----------------|
| 12_execute.py | 512.0 | 34 | 11.3x |
| 13_llm.py | 281.5 | 19 | 6.2x |
| 2_tests.py | 120.9 | 30 | 2.7x |
| 16_analysis.py | 77.5 | 24 | 1.7x |

## Recommendations

- 🟡 **WARNINGS**: 4 step(s) have yellow flags that should be reviewed.
-    ↳ **2_tests.py**: Very slow: 120.9s (>120.0s threshold)
-    ↳ **12_execute.py**: Very slow: 512.0s (>120.0s threshold)
-    ↳ **13_llm.py**: Very slow: 281.5s (>120.0s threshold)
- ⚡ **Performance**: Slowest step is **12_execute.py** (512.0s). Consider parallelization or caching.
- ✅ **Health**: Pipeline is healthy (100/100). All systems nominal.

## Pipeline Configuration

```json
{
  "target_dir": "input/gnn_files",
  "output_dir": "output",
  "verbose": false,
  "skip_steps": null,
  "only_steps": null,
  "strict": false,
  "frameworks": "all"
}
```
