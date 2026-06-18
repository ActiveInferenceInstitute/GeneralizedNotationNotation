# Pipeline Intelligent Analysis Report

**Generated**: 2026-06-18T07:59:03.255752

**Status**: ✅ SUCCESS

**Health Score**: 85.0/100


## Quick Overview

| Metric | Value |
|--------|-------|
| Total Steps | 24 |
| Duration | 481.42s |
| Peak Memory | 179.8 MB |
| 🔴 Red Flags | 0 |
| 🟡 Yellow Flags | 5 |
| ✅ Green (Clean) | 19 |

## AI-Powered Analysis

### Executive Summary
Pipeline completed successfully with a health score of 85/100. There are 5 yellow flag(s) to review for optimization opportunities.

### Red Flags (Critical Issues)
None - No critical issues detected.

### Yellow Flags (Warnings)
- **2_tests.py**: Slow: 105.4s (>60.0s threshold)
- **13_llm.py**: Very slow: 299.2s (>120.0s threshold)
- **15_audio.py**: Step completed with warnings
- **16_analysis.py**: Step completed with warnings
- **22_gui.py**: Step completed with warnings

### Action Items
1. **Review**: Investigate yellow flag warnings


## 🟡 Yellow Flags (Warnings)

| Step | Duration | Memory | Issues |
|------|----------|--------|--------|
| 2_tests.py | 105.4s | 169MB | Slow: 105.4s (>60.0s threshold) |
| 13_llm.py | 299.2s | 64MB | Very slow: 299.2s (>120.0s threshold) |
| 15_audio.py | 0.3s | 28MB | Step completed with warnings |
| 16_analysis.py | 17.0s | 28MB | Step completed with warnings |
| 22_gui.py | 0.1s | 27MB | Step completed with warnings |

## Per-Step Execution Details

| # | Step | Status | Duration | Memory | Flags |
|---|------|--------|----------|--------|-------|
| 1 | 0_template.py | ✅ SUCCESS | 0.09s | 180MB | - |
| 2 | 1_setup.py | ✅ SUCCESS | 4.36s | 180MB | - |
| 3 | 2_tests.py | 🟡 SUCCESS | 105.36s | 169MB | 1 |
| 4 | 3_gnn.py | ✅ SUCCESS | 0.13s | 66MB | - |
| 5 | 4_model_registry.py | ✅ SUCCESS | 0.08s | 67MB | - |
| 6 | 5_type_checker.py | ✅ SUCCESS | 0.83s | 68MB | - |
| 7 | 6_validation.py | ✅ SUCCESS | 0.08s | 68MB | - |
| 8 | 7_export.py | ✅ SUCCESS | 0.13s | 68MB | - |
| 9 | 8_visualization.py | ✅ SUCCESS | 3.73s | 68MB | - |
| 10 | 9_advanced_viz.py | ✅ SUCCESS | 9.29s | 68MB | - |
| 11 | 10_ontology.py | ✅ SUCCESS | 0.08s | 68MB | - |
| 12 | 11_render.py | ✅ SUCCESS | 0.14s | 68MB | - |
| 13 | 12_execute.py | ✅ SUCCESS | 36.99s | 68MB | - |
| 14 | 13_llm.py | 🟡 SUCCESS | 299.21s | 64MB | 1 |
| 15 | 14_ml_integration.py | ✅ SUCCESS | 0.20s | 26MB | - |
| 16 | 15_audio.py | 🟡 SUCCESS_WITH_WARNINGS | 0.32s | 28MB | 1 |
| 17 | 16_analysis.py | 🟡 SUCCESS_WITH_WARNINGS | 17.05s | 28MB | 1 |
| 18 | 17_integration.py | ✅ SUCCESS | 0.44s | 26MB | - |
| 19 | 18_security.py | ✅ SUCCESS | 0.08s | 26MB | - |
| 20 | 19_research.py | ✅ SUCCESS | 0.08s | 26MB | - |
| 21 | 20_website.py | ✅ SUCCESS | 0.13s | 26MB | - |
| 22 | 21_mcp.py | ✅ SUCCESS | 2.31s | 26MB | - |
| 23 | 22_gui.py | 🟡 SUCCESS_WITH_WARNINGS | 0.13s | 27MB | 1 |
| 24 | 23_report.py | ✅ SUCCESS | 0.15s | 29MB | - |

## Detailed Step Output (Flagged Steps)

### 2_tests.py

**Test suite execution**

- Status: SUCCESS
- Duration: 105.36s
- Memory: 169MB
- Flags: Slow: 105.4s (>60.0s threshold)

**Output Snippet**:
```
2026-06-18 07:51:05,556 [f87d1831:2_tests] 2_tests.py - INFO - Executing fast tests: /Users/4d/Documents/GitHub/GeneralizedNotationNotation/.venv/bin/python -m pytest --tb=short --maxfail=5 --durations=10 -ra --timeout 600 -v -m not slow --ignore=src/tests/llm/test_llm_ollama.py --ignore=src/tests/llm/test_llm_ollama_integration.py --ignore=src/tests/test_pipeline_performance.py --ignore=src/tests/test_pipeline_recovery.py --ignore=src/tests/test_report_integration.py src/tests/
```

### 13_llm.py

**LLM processing**

- Status: SUCCESS
- Duration: 299.21s
- Memory: 64MB
- Flags: Very slow: 299.2s (>120.0s threshold)

**Output Snippet**:
```
2026-06-18 07:54:01,054 [bff8a973:13_llm] llm - DEBUG -   ✅ Prompt completed successfully
2026-06-18 07:54:09,731 [bff8a973:13_llm] llm - DEBUG -   ✅ Prompt completed successfully
2026-06-18 07:54:18,451 [bff8a973:13_llm] llm - DEBUG -   ✅ Prompt completed successfully
2026-06-18 07:54:26,874 [bff8a973:13_llm] llm - DEBUG -   ✅ Prompt completed successfully
2026-06-18 07:54:35,316 [bff8a973:13_llm] llm - DEBUG -   ✅ Prompt completed successfully
```

### 15_audio.py

**Audio processing**

- Status: SUCCESS_WITH_WARNINGS
- Duration: 0.32s
- Memory: 28MB
- Flags: Step completed with warnings

**Output Snippet**:
```
2026-06-18 07:58:41,786 [2589e2ec:15_audio] audio - WARNING - Could not read audio telemetry file /Users/4d/Documents/GitHub/GeneralizedNotationNotation/output/12_execute_output/summaries/output/12_execute_output/pomdp_gridworld_3x3/numpyro/execution_logs/POMDP GridWorld 3x3_numpyro.py_results.json: [Errno 2] No such file or directory: '/Users/4d/Documents/GitHub/GeneralizedNotationNotation/output/12_execute_output/summaries/output/12_execute_output/pomdp_gridworld_3x3/numpyro/execution_logs/POM
```

### 16_analysis.py

**Analysis**

- Status: SUCCESS_WITH_WARNINGS
- Duration: 17.05s
- Memory: 28MB
- Flags: Step completed with warnings

**Output Snippet**:
```
2026-06-18 07:58:43,236 [2484f8dc:16_analysis] analysis.analyzer - DEBUG - Correlation computation failed, defaulting to 0.0: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 58 and the array at index 1 has size 11
2026-06-18 07:58:49,902 [2484f8dc:16_analysis] analysis - WARNING -   [numpyro] No simulation data found
2026-06-18 07:58:49,902 [2484f8dc:16_analysis] analysis - WARNING -   [pymdp] No simulation
```

### 22_gui.py

**GUI (Interactive GNN Constructor)**

- Status: SUCCESS_WITH_WARNINGS
- Duration: 0.13s
- Memory: 27MB
- Flags: Step completed with warnings

**Output Snippet**:
```
2026-06-18 07:59:01,998 [c90d184b:22_gui] gui.processor - WARNING - Gradio not available - generating recovery artifacts only
2026-06-18 07:59:01,999 [c90d184b:22_gui] gui.processor - INFO - ✅ GUI 1 completed successfully
2026-06-18 07:59:01,999 [c90d184b:22_gui] gui.processor - INFO - ✅ Successfully compiled gui_1 visual DOM artifacts.
2026-06-18 07:59:01,999 [c90d184b:22_gui] gui.processor - WARNING - Gradio not available - generating recovery artifacts only
2026-06-18 07:59:02,001 [c90d184b:2
```

## Performance Bottlenecks

| Step | Duration (s) | Memory (MB) | Above Avg Ratio |
|------|-------------|-------------|-----------------|
| 13_llm.py | 299.2 | 64 | 14.9x |
| 2_tests.py | 105.4 | 169 | 5.3x |

## Recommendations

- 🟡 **WARNINGS**: 5 step(s) have yellow flags that should be reviewed.
-    ↳ **2_tests.py**: Slow: 105.4s (>60.0s threshold)
-    ↳ **13_llm.py**: Very slow: 299.2s (>120.0s threshold)
-    ↳ **15_audio.py**: Step completed with warnings
- ⚡ **Performance**: Slowest step is **13_llm.py** (299.2s). Consider parallelization or caching.
- ⚠️ **Health**: Pipeline health needs attention (85/100).

## Pipeline Configuration

```json
{
  "target_dir": "input/gnn_files/pomdp_gridworld",
  "output_dir": "output",
  "verbose": true,
  "skip_steps": null,
  "only_steps": null,
  "strict": true,
  "frameworks": "all"
}
```
