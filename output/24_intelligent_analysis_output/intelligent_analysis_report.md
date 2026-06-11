# Pipeline Intelligent Analysis Report

**Generated**: 2026-05-22T06:35:10.993320

**Status**: ✅ SUCCESS

**Health Score**: 85.0/100


## Quick Overview

| Metric | Value |
|--------|-------|
| Total Steps | 24 |
| Duration | 1482.19s |
| Peak Memory | 83.3 MB |
| 🔴 Red Flags | 0 |
| 🟡 Yellow Flags | 7 |
| ✅ Green (Clean) | 17 |

## AI-Powered Analysis

### Executive Summary
Pipeline completed successfully with a health score of 85/100. There are 7 yellow flag(s) to review for optimization opportunities.

### Red Flags (Critical Issues)
None - No critical issues detected.

### Yellow Flags (Warnings)
- **2_tests.py**: Very slow: 329.3s (>120.0s threshold)
- **3_gnn.py**: Slow: 78.8s (>60.0s threshold)
- **12_execute.py**: Very slow: 321.7s (>120.0s threshold)
- **13_llm.py**: Very slow: 571.4s (>120.0s threshold), Step completed with warnings
- **16_analysis.py**: Slow: 113.9s (>60.0s threshold)
- ...and 2 more

### Action Items
1. **Review**: Investigate yellow flag warnings


## 🟡 Yellow Flags (Warnings)

| Step | Duration | Memory | Issues |
|------|----------|--------|--------|
| 2_tests.py | 329.3s | 83MB | Very slow: 329.3s (>120.0s threshold) |
| 3_gnn.py | 78.8s | 27MB | Slow: 78.8s (>60.0s threshold) |
| 12_execute.py | 321.7s | 29MB | Very slow: 321.7s (>120.0s threshold) |
| 13_llm.py | 571.4s | 29MB | Very slow: 571.4s (>120.0s threshold); Step completed with warnings |
| 16_analysis.py | 113.9s | 28MB | Slow: 113.9s (>60.0s threshold) |
| 17_integration.py | 0.4s | 28MB | Step completed with warnings |
| 22_gui.py | 0.2s | 28MB | Step completed with warnings |

## Per-Step Execution Details

| # | Step | Status | Duration | Memory | Flags |
|---|------|--------|----------|--------|-------|
| 1 | 0_template.py | ✅ SUCCESS | 0.18s | 83MB | - |
| 2 | 1_setup.py | ✅ SUCCESS | 4.75s | 83MB | - |
| 3 | 2_tests.py | 🟡 SUCCESS | 329.33s | 83MB | 1 |
| 4 | 3_gnn.py | 🟡 SUCCESS | 78.81s | 27MB | 1 |
| 5 | 4_model_registry.py | ✅ SUCCESS | 0.09s | 28MB | - |
| 6 | 5_type_checker.py | ✅ SUCCESS | 22.39s | 28MB | - |
| 7 | 6_validation.py | ✅ SUCCESS | 0.82s | 28MB | - |
| 8 | 7_export.py | ✅ SUCCESS | 0.09s | 28MB | - |
| 9 | 8_visualization.py | ✅ SUCCESS | 20.90s | 28MB | - |
| 10 | 9_advanced_viz.py | ✅ SUCCESS | 11.20s | 28MB | - |
| 11 | 10_ontology.py | ✅ SUCCESS | 0.45s | 28MB | - |
| 12 | 11_render.py | ✅ SUCCESS | 0.21s | 29MB | - |
| 13 | 12_execute.py | 🟡 SUCCESS | 321.68s | 29MB | 1 |
| 14 | 13_llm.py | 🟡 SUCCESS_WITH_WARNINGS | 571.43s | 29MB | 2 |
| 15 | 14_ml_integration.py | ✅ SUCCESS | 1.12s | 28MB | - |
| 16 | 15_audio.py | ✅ SUCCESS | 1.48s | 28MB | - |
| 17 | 16_analysis.py | 🟡 SUCCESS | 113.88s | 28MB | 1 |
| 18 | 17_integration.py | 🟡 SUCCESS_WITH_WARNINGS | 0.41s | 28MB | 1 |
| 19 | 18_security.py | ✅ SUCCESS | 0.09s | 28MB | - |
| 20 | 19_research.py | ✅ SUCCESS | 0.08s | 28MB | - |
| 21 | 20_website.py | ✅ SUCCESS | 0.14s | 28MB | - |
| 22 | 21_mcp.py | ✅ SUCCESS | 2.12s | 28MB | - |
| 23 | 22_gui.py | 🟡 SUCCESS_WITH_WARNINGS | 0.19s | 28MB | 1 |
| 24 | 23_report.py | ✅ SUCCESS | 0.29s | 30MB | - |

## Detailed Step Output (Flagged Steps)

### 2_tests.py

**Test suite execution**

- Status: SUCCESS
- Duration: 329.33s
- Memory: 83MB
- Flags: Very slow: 329.3s (>120.0s threshold)

**Output Snippet**:
```
2026-05-22 06:10:31,669 [5a767d42:2_tests] 2_tests.py - INFO - Executing fast tests: /Users/mini/Documents/GitHub/GeneralizedNotationNotation/.venv/bin/python -m pytest --tb=short --maxfail=5 --durations=10 -ra --timeout 600 -q -m not slow --ignore=src/tests/llm/test_llm_ollama.py --ignore=src/tests/llm/test_llm_ollama_integration.py --ignore=src/tests/test_pipeline_performance.py --ignore=src/tests/test_pipeline_recovery.py --ignore=src/tests/test_report_integration.py src/tests/
```

### 3_gnn.py

**GNN file processing**

- Status: SUCCESS
- Duration: 78.81s
- Memory: 27MB
- Flags: Slow: 78.8s (>60.0s threshold)

**Output Snippet**:
```
2026-05-22 06:17:19,368 [94fb85e4:3_gnn] 3_gnn.py - INFO - Processing: input/gnn_files/continuous/stochastic_dynamics.md
2026-05-22 06:17:19,375 [94fb85e4:3_gnn] 3_gnn.py - INFO - Generated 22 formats for stochastic_dynamics.md
2026-05-22 06:17:19,375 [94fb85e4:3_gnn] 3_gnn.py - INFO - Processing: input/gnn_files/continuous/continuous_navigation.md
2026-05-22 06:17:19,385 [94fb85e4:3_gnn] 3_gnn.py - INFO - Generated 22 formats for continuous_navigation.md
2026-05-22 06:17:19,385 [94fb85e4:3_gnn]
```

### 12_execute.py

**Execution**

- Status: SUCCESS
- Duration: 321.68s
- Memory: 29MB
- Flags: Very slow: 321.7s (>120.0s threshold)

**Output Snippet**:
```
2026-05-22 06:18:19,951 [49c4e484:12_execute] execute - INFO - ✅ Successfully executed Simple Markov Chain_pymdp.py
2026-05-22 06:18:21,162 [49c4e484:12_execute] execute - INFO - ✅ Successfully executed Simple Markov Chain_pytorch.py
2026-05-22 06:18:22,268 [49c4e484:12_execute] execute - INFO - ✅ Successfully executed Simple Markov Chain_jax.py
2026-05-22 06:18:22,763 [49c4e484:12_execute] execute - INFO - ✅ Successfully executed Simple Markov Chain_discopy.py
2026-05-22 06:18:27,062 [49c4e484:
```

### 13_llm.py

**LLM processing**

- Status: SUCCESS_WITH_WARNINGS
- Duration: 571.43s
- Memory: 29MB
- Flags: Very slow: 571.4s (>120.0s threshold), Step completed with warnings

**Output Snippet**:
```
2026-05-22 06:23:38,443 [70fc6d7a:13_llm] llm.providers.openai_provider - INFO - OpenAI provider initialized successfully
2026-05-22 06:23:38,452 [MAIN:pipeline] llm.providers.openai_provider - INFO - OpenAI provider initialized successfully
2026-05-22 06:23:52,789 [MAIN:pipeline] llm.providers.openai_provider - INFO - OpenAI provider initialized successfully
2026-05-22 06:24:04,352 [MAIN:pipeline] llm.providers.openai_provider - INFO - OpenAI provider initialized successfully
2026-05-22 06:24:2
```

### 16_analysis.py

**Analysis**

- Status: SUCCESS
- Duration: 113.88s
- Memory: 28MB
- Flags: Slow: 113.9s (>60.0s threshold)

**Output Snippet**:
```
2026-05-22 06:35:05,027 [81f1c44c:16_analysis] analysis - INFO - ✅ Analysis processing completed successfully [step_number=None]
```

### 17_integration.py

**Integration**

- Status: SUCCESS_WITH_WARNINGS
- Duration: 0.41s
- Memory: 28MB
- Flags: Step completed with warnings

**Output Snippet**:
```
2026-05-22 06:35:05,547 [4486b395:17_integration] integration - WARNING - No sweep-parameterized records found
```

### 22_gui.py

**GUI (Interactive GNN Constructor)**

- Status: SUCCESS_WITH_WARNINGS
- Duration: 0.19s
- Memory: 28MB
- Flags: Step completed with warnings

**Output Snippet**:
```
2026-05-22 06:35:08,108 [328d39eb:22_gui] gui.processor - WARNING - Gradio not available - generating recovery artifacts only
2026-05-22 06:35:08,108 [328d39eb:22_gui] gui.processor - INFO - ✅ GUI 1 completed successfully
2026-05-22 06:35:08,108 [328d39eb:22_gui] gui.processor - INFO - ✅ Successfully compiled gui_1 visual DOM artifacts.
2026-05-22 06:35:08,108 [328d39eb:22_gui] gui.processor - WARNING - Gradio not available - generating recovery artifacts only
2026-05-22 06:35:08,109 [328d39eb:2
```

## Performance Bottlenecks

| Step | Duration (s) | Memory (MB) | Above Avg Ratio |
|------|-------------|-------------|-----------------|
| 13_llm.py | 571.4 | 29 | 9.3x |
| 2_tests.py | 329.3 | 83 | 5.3x |
| 12_execute.py | 321.7 | 29 | 5.2x |
| 16_analysis.py | 113.9 | 28 | 1.8x |
| 3_gnn.py | 78.8 | 27 | 1.3x |

## Recommendations

- 🟡 **WARNINGS**: 7 step(s) have yellow flags that should be reviewed.
-    ↳ **2_tests.py**: Very slow: 329.3s (>120.0s threshold)
-    ↳ **3_gnn.py**: Slow: 78.8s (>60.0s threshold)
-    ↳ **12_execute.py**: Very slow: 321.7s (>120.0s threshold)
- ⚡ **Performance**: Slowest step is **13_llm.py** (571.4s). Consider parallelization or caching.
- ⚠️ **Health**: Pipeline health needs attention (85/100).

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
