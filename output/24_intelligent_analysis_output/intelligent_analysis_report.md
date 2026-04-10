# Pipeline Intelligent Analysis Report

**Generated**: 2026-04-10T10:35:43.605840

**Status**: ✅ SUCCESS

**Health Score**: 100.0/100


## Quick Overview

| Metric | Value |
|--------|-------|
| Total Steps | 24 |
| Duration | 726.55s |
| Peak Memory | 39.2 MB |
| 🔴 Red Flags | 0 |
| 🟡 Yellow Flags | 2 |
| ✅ Green (Clean) | 22 |

## AI-Powered Analysis

🟡 **0_template.py** (Template initialization): Completed successfully in 0.30s
✅ **1_setup.py** (Environment setup): Completed successfully in 3.28s
✅ **2_tests.py** (Test suite execution): Completed successfully in 54.11s
🟡 **1_type_checker.py** (Type checking): Completed successfully in 1.85s
🟡 **6_validation.py** (Validation): Completed successfully in 1.67s
✅ **7_export.py** (Multi-format export): Completed successfully in 0.24s
🟡 **8_visualization.py** (Visualization): Completed successfully in 17.09s
✅ **9_advanced_viz.py** (Advanced visualization): Completed successfully in 8.55s
🟡 **10_ontology.py** (Ontology processing): Completed successfully in 0.49s
🟡 **11_render.py** (Code rendering): Completed successfully in 0.29s
✅ **12_execute.py** (Execution): Completed with 1 flag(s) in 296.11s | Flags: Very slow: 296.5s (>120.0s threshold)
🟡 **13_llm.py** (LLM processing): Completed successfully in 2.48s
✅ **14_ml_integration.**: Completed successfully in 0.29s
✅ **15_audio.py** (Audio processing): Completed successfully in 1.53s
🟡 **16_analysis.**: Completed successfully in 17.09s
✅ **17_integration.**: Completed successfully in 0.29s
✅ **18_security.**: Completed successfully in 0.24s
✅ **19_research.**: Completed successfully in 0.24s
🟡 **20_website.**: Completed successfully in 0.21s
✅ **21_mcp.**: Completed successfully in 0.36s
✅ **22_gui.**: Completed successfully in 0.34s
✅ **23_report.**: Completed successfully in 0.41s

## 🟡 Yellow Flags (Warnings)

| Step | Duration | Memory | Issues |
|------|----------|--------|--------|
| 12_execute.py | 296.1s | 39MB | Very slow: 296.1s (>120.0s threshold) |
| 13_llm.py | 296.5s | 39MB | Very slow: 296.5s (>120.0s threshold) |

## Per-Step Execution Details

| # | Step | Status | Duration | Memory | Flags |
|---|------|--------|----------|--------|-------|
| 1 | 0_template.py | ✅ SUCCESS | 0.30s | 38MB | - |
| 2 | 1_setup.py | ✅ SUCCESS | 3.28s | 38MB | - |
| 3 | 2_tests.py | ✅ SUCCESS | 54.11s | 38MB | - |
| 4 | 3_gnn.py | ✅ SUCCESS | 2.48s | 38MB | - |
| 5 | 4_model_registry.py | ✅ SUCCESS | 0.30s | 38MB | - |
| 6 | 5_type_checker.py | ✅ SUCCESS | 1.85s | 38MB | - |
| 7 | 6_validation.py | ✅ SUCCESS | 1.67s | 38MB | - |
| 8 | 7_export.py | ✅ SUCCESS | 0.24s | 38MB | - |
| 9 | 8_visualization.py | ✅ SUCCESS | 17.09s | 38MB | - |
| 10 | 9_advanced_viz.py | ✅ SUCCESS | 8.55s | 38MB | - |
| 11 | 10_ontology.py | ✅ SUCCESS | 0.49s | 38MB | - |
| 12 | 11_render.py | ✅ SUCCESS | 0.29s | 38MB | - |
| 13 | 12_execute.py | 🟡 SUCCESS | 296.11s | 39MB | 1 |
| 14 | 13_llm.py | 🟡 SUCCESS | 296.48s | 39MB | 1 |
| 15 | 14_ml_integration.py | ✅ SUCCESS | 0.95s | 39MB | - |
| 16 | 15_audio.py | ✅ SUCCESS | 1.53s | 39MB | - |
| 17 | 16_analysis.py | ✅ SUCCESS | 37.24s | 39MB | - |
| 18 | 17_integration.py | ✅ SUCCESS | 0.29s | 39MB | - |
| 19 | 18_security.py | ✅ SUCCESS | 0.24s | 39MB | - |
| 20 | 19_research.py | ✅ SUCCESS | 0.24s | 39MB | - |
| 21 | 20_website.py | ✅ SUCCESS | 0.29s | 39MB | - |
| 22 | 21_mcp.py | ✅ SUCCESS | 1.74s | 39MB | - |
| 23 | 22_gui.py | ✅ SUCCESS | 0.34s | 39MB | - |
| 24 | 23_report.py | ✅ SUCCESS | 0.41s | 39MB | - |

## Detailed Step Output (Flagged Steps)

### 12_execute.py

**Execution**

- Status: SUCCESS
- Duration: 296.11s
- Memory: 39MB
- Flags: Very slow: 296.1s (>120.0s threshold)

**Output Snippet**:
```
--- Output for discrete ---
```

**Error Output**:
```
2026-04-10 10:25:06,743 - execute - INFO - ✅ Successfully executed Simple Markov Chain_numpyro.py
2026-04-10 10:25:09,342 - execute - INFO - ✅ Successfully executed Simple Markov Chain_pymdp.py
2026-04-10 10:25:10,293 - execute - INFO - ✅ Successfully executed Simple Markov Chain_pytorch.py
2026-04-10 10:25:11,230 - execute - INFO - ✅ Successfully executed Simple Markov Chain_jax.py
2026-04-10 10:25:11,750 - execute - INFO - ✅ Successfully executed Simple Markov Chain_discopy.py
```

### 13_llm.py

**LLM processing**

- Status: SUCCESS
- Duration: 296.48s
- Memory: 39MB
- Flags: Very slow: 296.5s (>120.0s threshold)

**Error Output**:
```
2026-04-10 10:30:01,835 - llm.providers.openai_provider - INFO - OpenAI provider initialized successfully
2026-04-10 10:30:01,844 - llm.providers.openai_provider - INFO - OpenAI provider initialized successfully
2026-04-10 10:30:14,209 - llm.providers.openai_provider - INFO - OpenAI provider initialized successfully
2026-04-10 10:30:25,261 - llm.providers.openai_provider - INFO - OpenAI provider initialized successfully
2026-04-10 10:30:42,520 - llm.providers.openai_provider - INFO - OpenAI prov
```

## Performance Bottlenecks

| Step | Duration (s) | Memory (MB) | Above Avg Ratio |
|------|-------------|-------------|-----------------|
| 13_llm.py | 296.5 | 39 | 9.8x |
| 12_execute.py | 296.1 | 39 | 9.8x |

## Recommendations

- 🟡 **WARNINGS**: 2 step(s) have yellow flags that should be reviewed.
-    ↳ **12_execute.py**: Very slow: 296.1s (>120.0s threshold)
-    ↳ **13_llm.py**: Very slow: 296.5s (>120.0s threshold)
- ⚡ **Performance**: Slowest step is **13_llm.py** (296.5s). Consider parallelization or caching.
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
