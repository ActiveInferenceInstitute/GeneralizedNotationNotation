# Pipeline Intelligent Analysis Report

**Generated**: 2026-06-18T09:11:01.214307

**Status**: ✅ SUCCESS

**Health Score**: 100.0/100


## Quick Overview

| Metric | Value |
|--------|-------|
| Total Steps | 24 |
| Duration | 493.83s |
| Peak Memory | 181.8 MB |
| 🔴 Red Flags | 0 |
| 🟡 Yellow Flags | 2 |
| ✅ Green (Clean) | 22 |

## AI-Powered Analysis

### Executive Summary
Pipeline completed successfully with a health score of 100/100. There are 2 yellow flag(s) to review for optimization opportunities.

### Red Flags (Critical Issues)
None - No critical issues detected.

### Yellow Flags (Warnings)
- **2_tests.py**: Slow: 106.7s (>60.0s threshold)
- **13_llm.py**: Very slow: 308.7s (>120.0s threshold)

### Action Items
1. **Review**: Investigate yellow flag warnings


## 🟡 Yellow Flags (Warnings)

| Step | Duration | Memory | Issues |
|------|----------|--------|--------|
| 2_tests.py | 106.7s | 182MB | Slow: 106.7s (>60.0s threshold) |
| 13_llm.py | 308.7s | 146MB | Very slow: 308.7s (>120.0s threshold) |

## Per-Step Execution Details

| # | Step | Status | Duration | Memory | Flags |
|---|------|--------|----------|--------|-------|
| 1 | 0_template.py | ✅ SUCCESS | 0.08s | 182MB | - |
| 2 | 1_setup.py | ✅ SUCCESS | 4.04s | 182MB | - |
| 3 | 2_tests.py | 🟡 SUCCESS | 106.68s | 182MB | 1 |
| 4 | 3_gnn.py | ✅ SUCCESS | 0.14s | 145MB | - |
| 5 | 4_model_registry.py | ✅ SUCCESS | 0.08s | 145MB | - |
| 6 | 5_type_checker.py | ✅ SUCCESS | 0.77s | 145MB | - |
| 7 | 6_validation.py | ✅ SUCCESS | 0.08s | 146MB | - |
| 8 | 7_export.py | ✅ SUCCESS | 0.13s | 146MB | - |
| 9 | 8_visualization.py | ✅ SUCCESS | 4.59s | 146MB | - |
| 10 | 9_advanced_viz.py | ✅ SUCCESS | 9.40s | 146MB | - |
| 11 | 10_ontology.py | ✅ SUCCESS | 0.08s | 146MB | - |
| 12 | 11_render.py | ✅ SUCCESS | 0.19s | 146MB | - |
| 13 | 12_execute.py | ✅ SUCCESS | 34.22s | 146MB | - |
| 14 | 13_llm.py | 🟡 SUCCESS | 308.70s | 146MB | 1 |
| 15 | 14_ml_integration.py | ✅ SUCCESS | 0.21s | 28MB | - |
| 16 | 15_audio.py | ✅ SUCCESS | 0.38s | 31MB | - |
| 17 | 16_analysis.py | ✅ SUCCESS | 16.79s | 31MB | - |
| 18 | 17_integration.py | ✅ SUCCESS | 0.51s | 29MB | - |
| 19 | 18_security.py | ✅ SUCCESS | 0.13s | 29MB | - |
| 20 | 19_research.py | ✅ SUCCESS | 0.14s | 29MB | - |
| 21 | 20_website.py | ✅ SUCCESS | 0.24s | 29MB | - |
| 22 | 21_mcp.py | ✅ SUCCESS | 5.82s | 29MB | - |
| 23 | 22_gui.py | ✅ SUCCESS | 0.19s | 28MB | - |
| 24 | 23_report.py | ✅ SUCCESS | 0.19s | 30MB | - |

## Detailed Step Output (Flagged Steps)

### 2_tests.py

**Test suite execution**

- Status: SUCCESS
- Duration: 106.68s
- Memory: 182MB
- Flags: Slow: 106.7s (>60.0s threshold)

**Output Snippet**:
```
2026-06-18 09:02:49,680 [c3338a84:2_tests] 2_tests.py - INFO - Executing fast tests: /Users/4d/Documents/GitHub/GeneralizedNotationNotation/.venv/bin/python -m pytest --tb=short --maxfail=5 --durations=10 -ra --timeout 600 -v -m not slow --ignore=src/tests/llm/test_llm_ollama.py --ignore=src/tests/llm/test_llm_ollama_integration.py --ignore=src/tests/test_pipeline_performance.py --ignore=src/tests/test_pipeline_recovery.py --ignore=src/tests/test_report_integration.py src/tests/
```

### 13_llm.py

**LLM processing**

- Status: SUCCESS
- Duration: 308.70s
- Memory: 146MB
- Flags: Very slow: 308.7s (>120.0s threshold)

**Output Snippet**:
```
2026-06-18 09:05:44,422 [8d1bc7e0:13_llm] llm - DEBUG -   ✅ Prompt completed successfully
2026-06-18 09:05:52,818 [8d1bc7e0:13_llm] llm - DEBUG -   ✅ Prompt completed successfully
2026-06-18 09:06:01,388 [8d1bc7e0:13_llm] llm - DEBUG -   ✅ Prompt completed successfully
2026-06-18 09:06:09,715 [8d1bc7e0:13_llm] llm - DEBUG -   ✅ Prompt completed successfully
2026-06-18 09:06:18,063 [8d1bc7e0:13_llm] llm - DEBUG -   ✅ Prompt completed successfully
```

## Performance Bottlenecks

| Step | Duration (s) | Memory (MB) | Above Avg Ratio |
|------|-------------|-------------|-----------------|
| 13_llm.py | 308.7 | 146 | 15.0x |
| 2_tests.py | 106.7 | 182 | 5.2x |

## Recommendations

- 🟡 **WARNINGS**: 2 step(s) have yellow flags that should be reviewed.
-    ↳ **2_tests.py**: Slow: 106.7s (>60.0s threshold)
-    ↳ **13_llm.py**: Very slow: 308.7s (>120.0s threshold)
- ⚡ **Performance**: Slowest step is **13_llm.py** (308.7s). Consider parallelization or caching.
- ✅ **Health**: Pipeline is healthy (100/100). All systems nominal.

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
