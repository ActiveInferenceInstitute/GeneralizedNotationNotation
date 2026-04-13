# Pipeline Intelligent Analysis Report

**Generated**: 2026-04-12T17:35:07.960886

**Status**: ✅ SUCCESS

**Health Score**: 100.0/100


## Quick Overview

| Metric | Value |
|--------|-------|
| Total Steps | 24 |
| Duration | 804.66s |
| Peak Memory | 37.9 MB |
| 🔴 Red Flags | 0 |
| 🟡 Yellow Flags | 3 |
| ✅ Green (Clean) | 21 |

## AI-Powered Analysis

✅ **0_template.py** (Template initialization): Completed successfully in 0.24s
✅ **1_setup.py** (Environment setup): Completed successfully in 5.71s
🟡 **2_tests.py** (Test suite execution): Completed with 1 flag(s) in 73.82s | Flags: Slow: 73.8s (>60.0s threshold)
✅ **3_gnn.py** (GNN file processing): Completed successfully in 2.83s
✅ **4_model_registry.py** (Model registry): Completed successfully in 0.48s
✅ **5_type_checker.py** (Type checking): Completed successfully in 1.95s
✅ **6_validation.py** (Validation): Completed successfully in 1.65s
✅ **7_export.py** (Multi-format export): Completed successfully in 0.29s
✅ **8_visualization.py** (Visualization): Completed successfully in 18.17s
✅ **9_advanced_viz.py** (Advanced visualization): Completed successfully in 8.81s
✅ **10_ontology.py** (Ontology processing): Completed successfully in 1.31s
✅ **11_render.py** (Code rendering): Completed successfully in 1.99s
✅ **12_execute.py** (Execution): Completed with 1 flag(s) in 341.27s | Flags: Very slow: 341.3s (>120.0s threshold)
✅ **13_llm.py** (LLM processing): Completed successfully in 1.99s
✅ **14_ml_integration.**: Completed successfully in 1.31s
✅ **15_audio.py**.*: Completed successfully in 0.29s
✅ **16_analysis.**: Completed successfully in 41.65s
✅ **17_integration.**: Completed successfully in 0.39s
✅ **18_security.**: Completed successfully in 0.24s
✅ **19_research.**: Completed successfully in 0.24s
✅ **20_website.**: Completed successfully in 0.21s
✅ **21_mcp.**: Completed successfully in 0.21s
✅ **23_gui.**: Completed successfully in 0.34s
✅ **25_report.**: Completed successfully in 0.45s

## 🟡 Yellow Flags (Warnings)

| Step | Duration | Memory | Issues |
|------|----------|--------|--------|
| 2_tests.py | 73.8s | 25MB | Slow: 73.8s (>60.0s threshold) |
| 12_execute.py | 341.3s | 29MB | Very slow: 341.3s (>120.0s threshold) |
| 13_llm.py | 300.2s | 29MB | Very slow: 300.2s (>120.0s threshold) |

## Per-Step Execution Details

| # | Step | Status | Duration | Memory | Flags |
|---|------|--------|----------|--------|-------|
| 1 | 0_template.py | ✅ SUCCESS | 0.24s | 38MB | - |
| 2 | 1_setup.py | ✅ SUCCESS | 5.71s | 38MB | - |
| 3 | 2_tests.py | 🟡 SUCCESS | 73.82s | 25MB | 1 |
| 4 | 3_gnn.py | ✅ SUCCESS | 2.83s | 28MB | - |
| 5 | 4_model_registry.py | ✅ SUCCESS | 0.48s | 28MB | - |
| 6 | 5_type_checker.py | ✅ SUCCESS | 1.95s | 28MB | - |
| 7 | 6_validation.py | ✅ SUCCESS | 1.65s | 28MB | - |
| 8 | 7_export.py | ✅ SUCCESS | 0.29s | 29MB | - |
| 9 | 8_visualization.py | ✅ SUCCESS | 18.17s | 29MB | - |
| 10 | 9_advanced_viz.py | ✅ SUCCESS | 8.81s | 29MB | - |
| 11 | 10_ontology.py | ✅ SUCCESS | 0.48s | 29MB | - |
| 12 | 11_render.py | ✅ SUCCESS | 0.29s | 29MB | - |
| 13 | 12_execute.py | 🟡 SUCCESS | 341.27s | 29MB | 1 |
| 14 | 13_llm.py | 🟡 SUCCESS | 300.21s | 29MB | 1 |
| 15 | 14_ml_integration.py | ✅ SUCCESS | 1.31s | 29MB | - |
| 16 | 15_audio.py | ✅ SUCCESS | 1.52s | 29MB | - |
| 17 | 16_analysis.py | ✅ SUCCESS | 41.65s | 29MB | - |
| 18 | 17_integration.py | ✅ SUCCESS | 0.39s | 28MB | - |
| 19 | 18_security.py | ✅ SUCCESS | 0.24s | 29MB | - |
| 20 | 19_research.py | ✅ SUCCESS | 0.24s | 29MB | - |
| 21 | 20_website.py | ✅ SUCCESS | 0.29s | 29MB | - |
| 22 | 21_mcp.py | ✅ SUCCESS | 1.99s | 29MB | - |
| 23 | 22_gui.py | ✅ SUCCESS | 0.34s | 29MB | - |
| 24 | 23_report.py | ✅ SUCCESS | 0.45s | 30MB | - |

## Detailed Step Output (Flagged Steps)

### 2_tests.py

**Test suite execution**

- Status: SUCCESS
- Duration: 73.82s
- Memory: 25MB
- Flags: Slow: 73.8s (>60.0s threshold)

**Output Snippet**:
```
2026-04-12 17:21:46,873 - 2_tests.py - INFO - Executing fast tests: /Users/mini/Documents/GitHub/GeneralizedNotationNotation/.venv/bin/python -m pytest --tb=short --maxfail=5 --durations=10 -ra --timeout 600 -q -m not slow --ignore=src/tests/test_llm_ollama.py --ignore=src/tests/test_llm_ollama_integration.py --ignore=src/tests/test_pipeline_performance.py --ignore=src/tests/test_pipeline_recovery.py --ignore=src/tests/test_report_integration.py src/tests/
```

**Error Output**:
```
2026-04-12 17:21:46,873 - 2_tests.py - INFO - Executing fast tests: /Users/mini/Documents/GitHub/GeneralizedNotationNotation/.venv/bin/python -m pytest --tb=short --maxfail=5 --durations=10 -ra --timeout 600 -q -m not slow --ignore=src/tests/test_llm_ollama.py --ignore=src/tests/test_llm_ollama_integration.py --ignore=src/tests/test_pipeline_performance.py --ignore=src/tests/test_pipeline_recovery.py --ignore=src/tests/test_report_integration.py src/tests/
```

### 12_execute.py

**Execution**

- Status: SUCCESS
- Duration: 341.27s
- Memory: 29MB
- Flags: Very slow: 341.3s (>120.0s threshold)

**Output Snippet**:
```
--- Output for discrete ---
```

**Error Output**:
```
2026-04-12 17:23:37,526 - execute - INFO - ✅ Successfully executed Simple Markov Chain_numpyro.py
2026-04-12 17:23:40,207 - execute - INFO - ✅ Successfully executed Simple Markov Chain_pymdp.py
2026-04-12 17:23:41,396 - execute - INFO - ✅ Successfully executed Simple Markov Chain_pytorch.py
2026-04-12 17:23:42,305 - execute - INFO - ✅ Successfully executed Simple Markov Chain_jax.py
2026-04-12 17:23:42,783 - execute - INFO - ✅ Successfully executed Simple Markov Chain_discopy.py
```

### 13_llm.py

**LLM processing**

- Status: SUCCESS
- Duration: 300.21s
- Memory: 29MB
- Flags: Very slow: 300.2s (>120.0s threshold)

**Error Output**:
```
2026-04-12 17:29:17,528 - llm.providers.openai_provider - INFO - OpenAI provider initialized successfully
2026-04-12 17:29:17,537 - llm.providers.openai_provider - INFO - OpenAI provider initialized successfully
2026-04-12 17:29:29,156 - llm.providers.openai_provider - INFO - OpenAI provider initialized successfully
2026-04-12 17:29:44,427 - llm.providers.openai_provider - INFO - OpenAI provider initialized successfully
2026-04-12 17:30:01,013 - llm.providers.openai_provider - INFO - OpenAI prov
```

## Performance Bottlenecks

| Step | Duration (s) | Memory (MB) | Above Avg Ratio |
|------|-------------|-------------|-----------------|
| 12_execute.py | 341.3 | 29 | 10.2x |
| 13_llm.py | 300.2 | 29 | 9.0x |
| 2_tests.py | 73.8 | 25 | 2.2x |

## Recommendations

- 🟡 **WARNINGS**: 3 step(s) have yellow flags that should be reviewed.
-    ↳ **2_tests.py**: Slow: 73.8s (>60.0s threshold)
-    ↳ **12_execute.py**: Very slow: 341.3s (>120.0s threshold)
-    ↳ **13_llm.py**: Very slow: 300.2s (>120.0s threshold)
- ⚡ **Performance**: Slowest step is **12_execute.py** (341.3s). Consider parallelization or caching.
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
