# Pipeline Intelligent Analysis Report

**Generated**: 2026-04-15T12:39:21.311615

**Status**: ✅ SUCCESS

**Health Score**: 100.0/100


## Quick Overview

| Metric | Value |
|--------|-------|
| Total Steps | 24 |
| Duration | 883.51s |
| Peak Memory | 37.9 MB |
| 🔴 Red Flags | 0 |
| 🟡 Yellow Flags | 3 |
| ✅ Green (Clean) | 21 |

## AI-Powered Analysis

### Executive Summary
Pipeline completed successfully with a health score of 100/100. There are 3 yellow flag(s) to review for optimization opportunities.

### Red Flags (Critical Issues)
None - No critical issues detected.

### Yellow Flags (Warnings)
- **2_tests.py**: Slow: 74.0s (>60.0s threshold)
- **12_execute.py**: Very slow: 394.8s (>120.0s threshold)
- **13_llm.py**: Very slow: 307.8s (>120.0s threshold)

### Action Items
1. **Review**: Investigate yellow flag warnings


## 🟡 Yellow Flags (Warnings)

| Step | Duration | Memory | Issues |
|------|----------|--------|--------|
| 2_tests.py | 74.0s | 30MB | Slow: 74.0s (>60.0s threshold) |
| 12_execute.py | 394.8s | 28MB | Very slow: 394.8s (>120.0s threshold) |
| 13_llm.py | 307.8s | 27MB | Very slow: 307.8s (>120.0s threshold) |

## Per-Step Execution Details

| # | Step | Status | Duration | Memory | Flags |
|---|------|--------|----------|--------|-------|
| 1 | 0_template.py | ✅ SUCCESS | 0.34s | 38MB | - |
| 2 | 1_setup.py | ✅ SUCCESS | 4.39s | 38MB | - |
| 3 | 2_tests.py | 🟡 SUCCESS | 74.03s | 30MB | 1 |
| 4 | 3_gnn.py | ✅ SUCCESS | 3.31s | 28MB | - |
| 5 | 4_model_registry.py | ✅ SUCCESS | 0.61s | 28MB | - |
| 6 | 5_type_checker.py | ✅ SUCCESS | 5.01s | 28MB | - |
| 7 | 6_validation.py | ✅ SUCCESS | 1.68s | 28MB | - |
| 8 | 7_export.py | ✅ SUCCESS | 0.24s | 28MB | - |
| 9 | 8_visualization.py | ✅ SUCCESS | 21.27s | 29MB | - |
| 10 | 9_advanced_viz.py | ✅ SUCCESS | 9.18s | 28MB | - |
| 11 | 10_ontology.py | ✅ SUCCESS | 0.54s | 28MB | - |
| 12 | 11_render.py | ✅ SUCCESS | 0.29s | 28MB | - |
| 13 | 12_execute.py | 🟡 SUCCESS | 394.79s | 28MB | 1 |
| 14 | 13_llm.py | 🟡 SUCCESS | 307.83s | 27MB | 1 |
| 15 | 14_ml_integration.py | ✅ SUCCESS | 1.42s | 28MB | - |
| 16 | 15_audio.py | ✅ SUCCESS | 2.14s | 28MB | - |
| 17 | 16_analysis.py | ✅ SUCCESS | 50.49s | 28MB | - |
| 18 | 17_integration.py | ✅ SUCCESS | 0.36s | 27MB | - |
| 19 | 18_security.py | ✅ SUCCESS | 0.25s | 27MB | - |
| 20 | 19_research.py | ✅ SUCCESS | 0.25s | 27MB | - |
| 21 | 20_website.py | ✅ SUCCESS | 0.31s | 27MB | - |
| 22 | 21_mcp.py | ✅ SUCCESS | 3.14s | 28MB | - |
| 23 | 22_gui.py | ✅ SUCCESS | 1.14s | 28MB | - |
| 24 | 23_report.py | ✅ SUCCESS | 0.47s | 29MB | - |

## Detailed Step Output (Flagged Steps)

### 2_tests.py

**Test suite execution**

- Status: SUCCESS
- Duration: 74.03s
- Memory: 30MB
- Flags: Slow: 74.0s (>60.0s threshold)

**Output Snippet**:
```
2026-04-15 12:24:38,991 - 2_tests.py - INFO - Executing fast tests: /Users/mini/Documents/GitHub/GeneralizedNotationNotation/.venv/bin/python -m pytest --tb=short --maxfail=5 --durations=10 -ra --timeout 600 -q -m not slow --ignore=src/tests/test_llm_ollama.py --ignore=src/tests/test_llm_ollama_integration.py --ignore=src/tests/test_pipeline_performance.py --ignore=src/tests/test_pipeline_recovery.py --ignore=src/tests/test_report_integration.py src/tests/
```

**Error Output**:
```
2026-04-15 12:24:38,991 - 2_tests.py - INFO - Executing fast tests: /Users/mini/Documents/GitHub/GeneralizedNotationNotation/.venv/bin/python -m pytest --tb=short --maxfail=5 --durations=10 -ra --timeout 600 -q -m not slow --ignore=src/tests/test_llm_ollama.py --ignore=src/tests/test_llm_ollama_integration.py --ignore=src/tests/test_pipeline_performance.py --ignore=src/tests/test_pipeline_recovery.py --ignore=src/tests/test_report_integration.py src/tests/
```

### 12_execute.py

**Execution**

- Status: SUCCESS
- Duration: 394.79s
- Memory: 28MB
- Flags: Very slow: 394.8s (>120.0s threshold)

**Output Snippet**:
```
--- Output for discrete ---
```

**Error Output**:
```
2026-04-15 12:26:37,289 - execute - INFO - ✅ Successfully executed Simple Markov Chain_numpyro.py
2026-04-15 12:26:40,010 - execute - INFO - ✅ Successfully executed Simple Markov Chain_pymdp.py
2026-04-15 12:26:41,233 - execute - INFO - ✅ Successfully executed Simple Markov Chain_pytorch.py
2026-04-15 12:26:42,108 - execute - INFO - ✅ Successfully executed Simple Markov Chain_jax.py
2026-04-15 12:26:42,598 - execute - INFO - ✅ Successfully executed Simple Markov Chain_discopy.py
```

### 13_llm.py

**LLM processing**

- Status: SUCCESS
- Duration: 307.83s
- Memory: 27MB
- Flags: Very slow: 307.8s (>120.0s threshold)

**Error Output**:
```
2026-04-15 12:33:11,073 - llm.providers.openai_provider - INFO - OpenAI provider initialized successfully
2026-04-15 12:33:11,084 - llm.providers.openai_provider - INFO - OpenAI provider initialized successfully
2026-04-15 12:33:25,430 - llm.providers.openai_provider - INFO - OpenAI provider initialized successfully
2026-04-15 12:33:40,514 - llm.providers.openai_provider - INFO - OpenAI provider initialized successfully
2026-04-15 12:33:59,042 - llm.providers.openai_provider - INFO - OpenAI prov
```

## Performance Bottlenecks

| Step | Duration (s) | Memory (MB) | Above Avg Ratio |
|------|-------------|-------------|-----------------|
| 12_execute.py | 394.8 | 28 | 10.7x |
| 13_llm.py | 307.8 | 27 | 8.4x |
| 2_tests.py | 74.0 | 30 | 2.0x |

## Recommendations

- 🟡 **WARNINGS**: 3 step(s) have yellow flags that should be reviewed.
-    ↳ **2_tests.py**: Slow: 74.0s (>60.0s threshold)
-    ↳ **12_execute.py**: Very slow: 394.8s (>120.0s threshold)
-    ↳ **13_llm.py**: Very slow: 307.8s (>120.0s threshold)
- ⚡ **Performance**: Slowest step is **12_execute.py** (394.8s). Consider parallelization or caching.
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
