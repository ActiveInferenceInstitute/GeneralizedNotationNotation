# Pipeline Intelligent Analysis Report

**Generated**: 2026-02-13T11:20:55.246261

**Status**: ✅ SUCCESS

**Health Score**: 100.0/100


## Quick Overview

| Metric | Value |
|--------|-------|
| Total Steps | 7 |
| Duration | 87.26s |
| Peak Memory | 74.2 MB |
| 🔴 Red Flags | 0 |
| 🟡 Yellow Flags | 1 |
| ✅ Green (Clean) | 6 |

## AI-Powered Analysis

### Executive Summary
Pipeline completed successfully with a health score of 100/100. There are 1 yellow flag(s) to review for optimization opportunities.

### Red Flags (Critical Issues)
None - No critical issues detected.

### Yellow Flags (Warnings)
- **12_execute.py**: Slow: 81.0s (>60.0s threshold)

### Action Items
1. **Review**: Investigate yellow flag warnings


## 🟡 Yellow Flags (Warnings)

| Step | Duration | Memory | Issues |
|------|----------|--------|--------|
| 12_execute.py | 81.0s | 28MB | Slow: 81.0s (>60.0s threshold) |

## Per-Step Execution Details

| # | Step | Status | Duration | Memory | Flags |
|---|------|--------|----------|--------|-------|
| 1 | 3_gnn.py | ✅ SUCCESS | 0.43s | 74MB | - |
| 2 | 5_type_checker.py | ✅ SUCCESS | 0.44s | 69MB | - |
| 3 | 7_export.py | ✅ SUCCESS | 0.49s | 60MB | - |
| 4 | 8_visualization.py | ✅ SUCCESS | 3.20s | 60MB | - |
| 5 | 11_render.py | ✅ SUCCESS | 0.54s | 28MB | - |
| 6 | 12_execute.py | 🟡 SUCCESS | 81.02s | 28MB | 1 |
| 7 | 15_audio.py | ✅ SUCCESS | 1.13s | 28MB | - |

## Detailed Step Output (Flagged Steps)

### 12_execute.py

**Execution**

- Status: SUCCESS
- Duration: 81.02s
- Memory: 28MB
- Flags: Slow: 81.0s (>60.0s threshold)

## Performance Bottlenecks

| Step | Duration (s) | Memory (MB) | Above Avg Ratio |
|------|-------------|-------------|-----------------|
| 12_execute.py | 81.0 | 28 | 6.5x |

## Recommendations

- 🟡 **WARNINGS**: 1 step(s) have yellow flags that should be reviewed.
-    ↳ **12_execute.py**: Slow: 81.0s (>60.0s threshold)
- ⚡ **Performance**: Slowest step is **12_execute.py** (81.0s). Consider parallelization or caching.
- ✅ **Health**: Pipeline is healthy (100/100). All systems nominal.

## Pipeline Configuration

```json
{
  "target_dir": "/Users/4d/Documents/GitHub/generalizednotationnotation/input/gnn_files",
  "output_dir": "/Users/4d/Documents/GitHub/generalizednotationnotation/output",
  "verbose": true,
  "skip_steps": [],
  "only_steps": "3,5,7,8,12,15",
  "strict": false,
  "frameworks": "all"
}
```
