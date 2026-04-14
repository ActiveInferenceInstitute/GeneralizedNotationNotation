# Pipeline Intelligent Analysis Report

**Generated**: 2026-04-14T11:15:35.719055

**Status**: ✅ SUCCESS

**Health Score**: 95.0/100


## Quick Overview

| Metric | Value |
|--------|-------|
| Total Steps | 25 |
| Duration | 878.53s |
| Peak Memory | 37.8 MB |
| 🔴 Red Flags | 0 |
| 🟡 Yellow Flags | 4 |
| ✅ Green (Clean) | 21 |

## AI-Powered Analysis

🟡 **0_template.py** (Template initialization): Completed successfully in 0.29s
✅ **1_setup.py** (Environment setup): Completed successfully in 7.12s
🟡 **2_tests.py** (Test suite execution): Completed with 1 flag(s) in 83.36s | Flags: Slow: 83.4s (>60.0s threshold)
✅ **3_gnn.py** (GNN file processing): Completed successfully in 3.22s
🟡 **4_model_registry.py** (Model registry): Completed successfully in 0.66s
✅ **5_type_checker.py** (Type checking): Completed successfully in 6.58s
✅ **6_validation.py** (Validation): Completed successfully in 2.81s
🟡 **7_export.py** (Multi-format export): Completed successfully in 0.40s
✅ **8_visualization.py** (Visualization): Completed successfully in 9.29s
✅ **9_advanced_viz.py** (Advanced visualization): Completed successfully in 1.31s
🟡 **10_ontology.py** (Ontology processing): Completed successfully in 0.57s
✅ **11_render.py** (Code rendering): Completed successfully in 0.29s
🟡 **12_execute.py** (Execution): Completed with 1 flag(s) in 371.94s | Flags: Very slow: 371.9s (>120.0s threshold)
✅ **13_llm.py** (LLM processing): Completed with 1 flag(s) in 297.08s | Flags: Very slow: 297.1s (>120.0s threshold)
✅ **14_ml_integration.**: Completed successfully in 1.31s
🟡 **15_airaction.py** (AIRATION): Completed successfully in 1.08s
✅ **16_analysis.**: Completed with 1 flag(s) in 297.08s | Flags: Very slow: 371.9s (>120.0s threshold)
🟡 **17_integration.**: Completed successfully in 0.40s
✅ **18_security.**: Completed with 1 flag(s) in 25 seconds
✅ **19_research.**: Completed with 1 flag(s) in 36 seconds
🟡 **20_website.**: Completed successfully in 0.40s
✅ **21_mcp.**: Completed successfully in 1 flag(s) in 57 seconds
✅ **22_gui.**: Completed with 1 flag(s) in 36 seconds
🟡 **23_report.**: Completed with 1 flag(s) in 40 seconds
🟡 **24_intelligent_analysis.**: Status: SUCCESS_WITH_WARNINGS (10.43s) | Flags: Step completed with warnings
```

## 🟡 Yellow Flags (Warnings)

| Step | Duration | Memory | Issues |
|------|----------|--------|--------|
| 2_tests.py | 83.4s | 24MB | Slow: 83.4s (>60.0s threshold) |
| 12_execute.py | 371.9s | 28MB | Very slow: 371.9s (>120.0s threshold) |
| 13_llm.py | 297.1s | 25MB | Very slow: 297.1s (>120.0s threshold) |
| 24_intelligent_analysis.py | 10.4s | 27MB | Step completed with warnings |

## Per-Step Execution Details

| # | Step | Status | Duration | Memory | Flags |
|---|------|--------|----------|--------|-------|
| 1 | 0_template.py | ✅ SUCCESS | 0.29s | 38MB | - |
| 2 | 1_setup.py | ✅ SUCCESS | 7.12s | 38MB | - |
| 3 | 2_tests.py | 🟡 SUCCESS | 83.36s | 24MB | 1 |
| 4 | 3_gnn.py | ✅ SUCCESS | 3.16s | 24MB | - |
| 5 | 4_model_registry.py | ✅ SUCCESS | 0.66s | 28MB | - |
| 6 | 5_type_checker.py | ✅ SUCCESS | 6.58s | 28MB | - |
| 7 | 6_validation.py | ✅ SUCCESS | 2.81s | 28MB | - |
| 8 | 7_export.py | ✅ SUCCESS | 0.40s | 28MB | - |
| 9 | 8_visualization.py | ✅ SUCCESS | 23.09s | 28MB | - |
| 10 | 9_advanced_viz.py | ✅ SUCCESS | 9.29s | 27MB | - |
| 11 | 10_ontology.py | ✅ SUCCESS | 0.57s | 28MB | - |
| 12 | 11_render.py | ✅ SUCCESS | 0.29s | 28MB | - |
| 13 | 12_execute.py | 🟡 SUCCESS | 371.94s | 28MB | 1 |
| 14 | 13_llm.py | 🟡 SUCCESS | 297.08s | 25MB | 1 |
| 15 | 14_ml_integration.py | ✅ SUCCESS | 1.31s | 26MB | - |
| 16 | 15_audio.py | ✅ SUCCESS | 2.12s | 26MB | - |
| 17 | 16_analysis.py | ✅ SUCCESS | 52.06s | 26MB | - |
| 18 | 17_integration.py | ✅ SUCCESS | 0.40s | 26MB | - |
| 19 | 18_security.py | ✅ SUCCESS | 0.24s | 26MB | - |
| 20 | 19_research.py | ✅ SUCCESS | 0.24s | 26MB | - |
| 21 | 20_website.py | ✅ SUCCESS | 0.29s | 26MB | - |
| 22 | 21_mcp.py | ✅ SUCCESS | 3.22s | 26MB | - |
| 23 | 22_gui.py | ✅ SUCCESS | 1.08s | 26MB | - |
| 24 | 23_report.py | ✅ SUCCESS | 0.45s | 27MB | - |
| 25 | 24_intelligent_analysis.py | 🟡 SUCCESS_WITH_WARNINGS | 10.43s | 27MB | 1 |

## Detailed Step Output (Flagged Steps)

### 2_tests.py

**Test suite execution**

- Status: SUCCESS
- Duration: 83.36s
- Memory: 24MB
- Flags: Slow: 83.4s (>60.0s threshold)

**Output Snippet**:
```
2026-04-14 10:56:48,405 - 2_tests.py - INFO - Executing fast tests: /Users/mini/Documents/GitHub/GeneralizedNotationNotation/.venv/bin/python -m pytest --tb=short --maxfail=5 --durations=10 -ra --timeout 600 -q -m not slow --ignore=src/tests/test_llm_ollama.py --ignore=src/tests/test_llm_ollama_integration.py --ignore=src/tests/test_pipeline_performance.py --ignore=src/tests/test_pipeline_recovery.py --ignore=src/tests/test_report_integration.py src/tests/
```

**Error Output**:
```
2026-04-14 10:56:48,405 - 2_tests.py - INFO - Executing fast tests: /Users/mini/Documents/GitHub/GeneralizedNotationNotation/.venv/bin/python -m pytest --tb=short --maxfail=5 --durations=10 -ra --timeout 600 -q -m not slow --ignore=src/tests/test_llm_ollama.py --ignore=src/tests/test_llm_ollama_integration.py --ignore=src/tests/test_pipeline_performance.py --ignore=src/tests/test_pipeline_recovery.py --ignore=src/tests/test_report_integration.py src/tests/
```

### 12_execute.py

**Execution**

- Status: SUCCESS
- Duration: 371.94s
- Memory: 28MB
- Flags: Very slow: 371.9s (>120.0s threshold)

**Output Snippet**:
```
--- Output for discrete ---
```

**Error Output**:
```
2026-04-14 10:59:00,691 - execute - INFO - ✅ Successfully executed Simple Markov Chain_numpyro.py
2026-04-14 10:59:03,433 - execute - INFO - ✅ Successfully executed Simple Markov Chain_pymdp.py
2026-04-14 10:59:04,722 - execute - INFO - ✅ Successfully executed Simple Markov Chain_pytorch.py
2026-04-14 10:59:05,587 - execute - INFO - ✅ Successfully executed Simple Markov Chain_jax.py
2026-04-14 10:59:06,079 - execute - INFO - ✅ Successfully executed Simple Markov Chain_discopy.py
```

### 13_llm.py

**LLM processing**

- Status: SUCCESS
- Duration: 297.08s
- Memory: 25MB
- Flags: Very slow: 297.1s (>120.0s threshold)

**Error Output**:
```
2026-04-14 11:05:11,630 - llm.providers.openai_provider - INFO - OpenAI provider initialized successfully
2026-04-14 11:05:11,641 - llm.providers.openai_provider - INFO - OpenAI provider initialized successfully
2026-04-14 11:05:25,302 - llm.providers.openai_provider - INFO - OpenAI provider initialized successfully
2026-04-14 11:05:36,140 - llm.providers.openai_provider - INFO - OpenAI provider initialized successfully
2026-04-14 11:05:52,580 - llm.providers.openai_provider - INFO - OpenAI prov
```

### 24_intelligent_analysis.py

**Intelligent pipeline analysis**

- Status: SUCCESS_WITH_WARNINGS
- Duration: 10.43s
- Memory: 27MB
- Flags: Step completed with warnings

**Output Snippet**:
```
2026-04-14 11:11:08,617 - 24_intelligent_analysis.py - INFO - Analysis complete: Status=SUCCESS, Failures=0, Health=100.0
2026-04-14 11:11:18,558 - 24_intelligent_analysis.py - WARNING - LLM hallucinated python code instead of prose. Using robust rule-based summary.
2026-04-14 11:11:18,563 - 24_intelligent_analysis.py - INFO - ✅ Intelligent analysis completed successfully
```

**Error Output**:
```
2026-04-14 11:11:08,617 - 24_intelligent_analysis.py - INFO - Analysis complete: Status=SUCCESS, Failures=0, Health=100.0
2026-04-14 11:11:09,540 - llm.providers.openai_provider - INFO - OpenAI provider initialized successfully
2026-04-14 11:11:18,558 - 24_intelligent_analysis.py - WARNING - LLM hallucinated python code instead of prose. Using robust rule-based summary.
2026-04-14 11:11:18,563 - 24_intelligent_analysis.py - INFO - ✅ Intelligent analysis completed successfully
```

## Performance Bottlenecks

| Step | Duration (s) | Memory (MB) | Above Avg Ratio |
|------|-------------|-------------|-----------------|
| 12_execute.py | 371.9 | 28 | 10.6x |
| 13_llm.py | 297.1 | 25 | 8.5x |
| 2_tests.py | 83.4 | 24 | 2.4x |

## Recommendations

- 🟡 **WARNINGS**: 4 step(s) have yellow flags that should be reviewed.
-    ↳ **2_tests.py**: Slow: 83.4s (>60.0s threshold)
-    ↳ **12_execute.py**: Very slow: 371.9s (>120.0s threshold)
-    ↳ **13_llm.py**: Very slow: 297.1s (>120.0s threshold)
- ⚡ **Performance**: Slowest step is **12_execute.py** (371.9s). Consider parallelization or caching.
- ✅ **Health**: Pipeline health is good (95/100).

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
