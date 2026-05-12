# Pipeline Intelligent Analysis Report

**Generated**: 2026-05-12T07:47:06.102854

**Status**: ✅ SUCCESS

**Health Score**: 85.0/100


## Quick Overview

| Metric | Value |
|--------|-------|
| Total Steps | 24 |
| Duration | 1396.05s |
| Peak Memory | 378.4 MB |
| 🔴 Red Flags | 0 |
| 🟡 Yellow Flags | 7 |
| ✅ Green (Clean) | 17 |

## AI-Powered Analysis

### Executive Summary
Pipeline completed successfully with a health score of 85/100. There are 7 yellow flag(s) to review for optimization opportunities.

### Red Flags (Critical Issues)
None - No critical issues detected.

### Yellow Flags (Warnings)
- **2_tests.py**: Very slow: 157.5s (>120.0s threshold)
- **3_gnn.py**: Slow: 78.6s (>60.0s threshold)
- **12_execute.py**: Very slow: 381.9s (>120.0s threshold)
- **13_llm.py**: Very slow: 572.2s (>120.0s threshold), Step completed with warnings
- **16_analysis.py**: Very slow: 122.8s (>120.0s threshold)
- ...and 2 more

### Action Items
1. **Review**: Investigate yellow flag warnings


## 🟡 Yellow Flags (Warnings)

| Step | Duration | Memory | Issues |
|------|----------|--------|--------|
| 2_tests.py | 157.5s | 378MB | Very slow: 157.5s (>120.0s threshold) |
| 3_gnn.py | 78.6s | 107MB | Slow: 78.6s (>60.0s threshold) |
| 12_execute.py | 381.9s | 111MB | Very slow: 381.9s (>120.0s threshold) |
| 13_llm.py | 572.2s | 111MB | Very slow: 572.2s (>120.0s threshold); Step completed with warnings |
| 16_analysis.py | 122.8s | 109MB | Very slow: 122.8s (>120.0s threshold) |
| 17_integration.py | 0.4s | 109MB | Step completed with warnings |
| 21_mcp.py | 2.0s | 111MB | Step completed with warnings |

## Per-Step Execution Details

| # | Step | Status | Duration | Memory | Flags |
|---|------|--------|----------|--------|-------|
| 1 | 0_template.py | ✅ SUCCESS | 0.29s | 378MB | - |
| 2 | 1_setup.py | ✅ SUCCESS | 17.95s | 378MB | - |
| 3 | 2_tests.py | 🟡 SUCCESS | 157.49s | 378MB | 1 |
| 4 | 3_gnn.py | 🟡 SUCCESS | 78.62s | 107MB | 1 |
| 5 | 4_model_registry.py | ✅ SUCCESS | 0.24s | 107MB | - |
| 6 | 5_type_checker.py | ✅ SUCCESS | 21.34s | 108MB | - |
| 7 | 6_validation.py | ✅ SUCCESS | 1.89s | 109MB | - |
| 8 | 7_export.py | ✅ SUCCESS | 0.24s | 109MB | - |
| 9 | 8_visualization.py | ✅ SUCCESS | 20.34s | 109MB | - |
| 10 | 9_advanced_viz.py | ✅ SUCCESS | 12.65s | 109MB | - |
| 11 | 10_ontology.py | ✅ SUCCESS | 0.82s | 109MB | - |
| 12 | 11_render.py | ✅ SUCCESS | 0.30s | 110MB | - |
| 13 | 12_execute.py | 🟡 SUCCESS | 381.88s | 111MB | 1 |
| 14 | 13_llm.py | 🟡 SUCCESS_WITH_WARNINGS | 572.25s | 111MB | 2 |
| 15 | 14_ml_integration.py | ✅ SUCCESS | 1.31s | 107MB | - |
| 16 | 15_audio.py | ✅ SUCCESS | 1.63s | 108MB | - |
| 17 | 16_analysis.py | 🟡 SUCCESS | 122.77s | 109MB | 1 |
| 18 | 17_integration.py | 🟡 SUCCESS_WITH_WARNINGS | 0.41s | 109MB | 1 |
| 19 | 18_security.py | ✅ SUCCESS | 0.25s | 110MB | - |
| 20 | 19_research.py | ✅ SUCCESS | 0.25s | 110MB | - |
| 21 | 20_website.py | ✅ SUCCESS | 0.31s | 110MB | - |
| 22 | 21_mcp.py | 🟡 SUCCESS_WITH_WARNINGS | 1.96s | 111MB | 1 |
| 23 | 22_gui.py | ✅ SUCCESS | 0.32s | 111MB | - |
| 24 | 23_report.py | ✅ SUCCESS | 0.43s | 113MB | - |

## Detailed Step Output (Flagged Steps)

### 2_tests.py

**Test suite execution**

- Status: SUCCESS
- Duration: 157.49s
- Memory: 378MB
- Flags: Very slow: 157.5s (>120.0s threshold)

**Output Snippet**:
```
2026-05-12 07:24:07,033 [ad8dfc8f:2_tests] 2_tests.py - INFO - Executing fast tests: /Users/mini/Documents/GitHub/GeneralizedNotationNotation/.venv/bin/python -m pytest --tb=short --maxfail=5 --durations=10 -ra --timeout 600 -v -m not slow --ignore=src/tests/test_llm_ollama.py --ignore=src/tests/test_llm_ollama_integration.py --ignore=src/tests/test_pipeline_performance.py --ignore=src/tests/test_pipeline_recovery.py --ignore=src/tests/test_report_integration.py src/tests/
```

### 3_gnn.py

**GNN file processing**

- Status: SUCCESS
- Duration: 78.62s
- Memory: 107MB
- Flags: Slow: 78.6s (>60.0s threshold)

**Output Snippet**:
```
2026-05-12 07:28:02,465 [c96c353d:3_gnn] 3_gnn.py - INFO -   Generated agda: 6900 bytes
2026-05-12 07:28:02,465 [c96c353d:3_gnn] 3_gnn.py - INFO -   Generated haskell: 7796 bytes
2026-05-12 07:28:02,466 [c96c353d:3_gnn] 3_gnn.py - INFO -   Generated pickle: 4664 bytes
2026-05-12 07:28:02,466 [c96c353d:3_gnn] 3_gnn.py - INFO - Generated 22 formats for continuous_navigation.md
2026-05-12 07:28:02,466 [c96c353d:3_gnn] 3_gnn.py - INFO - ✅ Processed 3 files, generated 66 format instances across 22 fo
```

### 12_execute.py

**Execution**

- Status: SUCCESS
- Duration: 381.88s
- Memory: 111MB
- Flags: Very slow: 381.9s (>120.0s threshold)

**Output Snippet**:
```
2026-05-12 07:29:05,697 [28b502ec:12_execute] execute - INFO - ✅ Successfully executed Simple Markov Chain_pymdp.py
2026-05-12 07:29:06,774 [28b502ec:12_execute] execute - INFO - ✅ Successfully executed Simple Markov Chain_pytorch.py
2026-05-12 07:29:07,841 [28b502ec:12_execute] execute - INFO - ✅ Successfully executed Simple Markov Chain_jax.py
2026-05-12 07:29:08,320 [28b502ec:12_execute] execute - INFO - ✅ Successfully executed Simple Markov Chain_discopy.py
2026-05-12 07:29:12,446 [28b502ec:
```

### 13_llm.py

**LLM processing**

- Status: SUCCESS_WITH_WARNINGS
- Duration: 572.25s
- Memory: 111MB
- Flags: Very slow: 572.2s (>120.0s threshold), Step completed with warnings

**Output Snippet**:
```
2026-05-12 07:35:23,719 [d466bd18:13_llm] llm.providers.openai_provider - INFO - OpenAI provider initialized successfully
2026-05-12 07:35:23,728 [MAIN:pipeline] llm.providers.openai_provider - INFO - OpenAI provider initialized successfully
2026-05-12 07:35:26,948 [d466bd18:13_llm] llm - DEBUG -   ✅ Prompt completed successfully
2026-05-12 07:35:28,299 [d466bd18:13_llm] llm - DEBUG -   ✅ Prompt completed successfully
2026-05-12 07:35:30,122 [d466bd18:13_llm] llm - DEBUG -   ✅ Prompt completed s
```

### 16_analysis.py

**Analysis**

- Status: SUCCESS
- Duration: 122.77s
- Memory: 109MB
- Flags: Very slow: 122.8s (>120.0s threshold)

**Output Snippet**:
```
2026-05-12 07:44:58,468 [8e73b2e2:16_analysis] analysis.analyzer - DEBUG - Correlation computation failed, defaulting to 0.0: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 67 and the array at index 1 has size 10
2026-05-12 07:44:58,472 [8e73b2e2:16_analysis] analysis.analyzer - DEBUG - Correlation computation failed, defaulting to 0.0: all the input array dimensions except for the concatenation axis must 
```

### 17_integration.py

**Integration**

- Status: SUCCESS_WITH_WARNINGS
- Duration: 0.41s
- Memory: 109MB
- Flags: Step completed with warnings

**Output Snippet**:
```
2026-05-12 07:47:00,617 [d393409f:17_integration] integration - WARNING - No sweep-parameterized records found
```

### 21_mcp.py

**Model Context Protocol processing**

- Status: SUCCESS_WITH_WARNINGS
- Duration: 1.96s
- Memory: 111MB
- Flags: Step completed with warnings

**Output Snippet**:
```
2026-05-12 07:47:01,683 [MAIN:pipeline] mcp - INFO - Successfully loaded module research: 4 tools, 0 resources (load: 0.002s, import: 0.002s, register: 0.000s)
2026-05-12 07:47:01,684 [MAIN:pipeline] mcp - INFO - Successfully loaded module ml_integration: 4 tools, 0 resources (load: 0.002s, import: 0.002s, register: 0.000s)
2026-05-12 07:47:01,684 [MAIN:pipeline] src.pipeline.mcp - INFO - Successfully registered pipeline MCP tools
2026-05-12 07:47:01,684 [MAIN:pipeline] mcp - INFO - Successfully
```

## Performance Bottlenecks

| Step | Duration (s) | Memory (MB) | Above Avg Ratio |
|------|-------------|-------------|-----------------|
| 13_llm.py | 572.2 | 111 | 9.8x |
| 12_execute.py | 381.9 | 111 | 6.6x |
| 2_tests.py | 157.5 | 378 | 2.7x |
| 16_analysis.py | 122.8 | 109 | 2.1x |
| 3_gnn.py | 78.6 | 107 | 1.4x |

## Recommendations

- 🟡 **WARNINGS**: 7 step(s) have yellow flags that should be reviewed.
-    ↳ **2_tests.py**: Very slow: 157.5s (>120.0s threshold)
-    ↳ **3_gnn.py**: Slow: 78.6s (>60.0s threshold)
-    ↳ **12_execute.py**: Very slow: 381.9s (>120.0s threshold)
- ⚡ **Performance**: Slowest step is **13_llm.py** (572.2s). Consider parallelization or caching.
- ⚠️ **Health**: Pipeline health needs attention (85/100).

## Pipeline Configuration

```json
{
  "target_dir": "input/gnn_files",
  "output_dir": "output",
  "verbose": true,
  "skip_steps": null,
  "only_steps": null,
  "strict": false,
  "frameworks": "all"
}
```
