# Pipeline Intelligent Analysis Report

**Generated**: 2026-02-09T16:13:40.502615

**Status**: ✅ SUCCESS

**Health Score**: 100.0/100


## Quick Overview

| Metric | Value |
|--------|-------|
| Total Steps | 7 |
| Duration | 63.52s |
| Peak Memory | 78.9 MB |
| 🔴 Red Flags | 0 |
| 🟡 Yellow Flags | 1 |
| ✅ Green (Clean) | 6 |

## AI-Powered Analysis

### Executive Summary
Pipeline completed successfully with a health score of 100/100. There are 1 yellow flag(s) to review for optimization opportunities.

### Red Flags (Critical Issues)
None - No critical issues detected.

### Yellow Flags (Warnings)
- **12_execute.py**: Significantly above average: 57.2s (6.3x avg)

### Action Items
1. **Review**: Investigate yellow flag warnings


## 🟡 Yellow Flags (Warnings)

| Step | Duration | Memory | Issues |
|------|----------|--------|--------|
| 12_execute.py | 57.2s | 66MB | Significantly above average: 57.2s (6.3x avg) |

## Per-Step Execution Details

| # | Step | Status | Duration | Memory | Flags |
|---|------|--------|----------|--------|-------|
| 1 | 3_gnn.py | ✅ SUCCESS | 0.34s | 79MB | - |
| 2 | 5_type_checker.py | ✅ SUCCESS | 0.34s | 79MB | - |
| 3 | 7_export.py | ✅ SUCCESS | 0.33s | 79MB | - |
| 4 | 8_visualization.py | ✅ SUCCESS | 4.00s | 79MB | - |
| 5 | 11_render.py | ✅ SUCCESS | 0.34s | 66MB | - |
| 6 | 12_execute.py | 🟡 SUCCESS | 57.22s | 66MB | 1 |
| 7 | 15_audio.py | ✅ SUCCESS | 0.95s | 34MB | - |

## Detailed Step Output (Flagged Steps)

### 12_execute.py

**Execution**

- Status: SUCCESS
- Duration: 57.22s
- Memory: 66MB
- Flags: Significantly above average: 57.2s (6.3x avg)

## Performance Bottlenecks

| Step | Duration (s) | Memory (MB) | Above Avg Ratio |
|------|-------------|-------------|-----------------|
| 12_execute.py | 57.2 | 66 | 6.3x |

## Recommendations

- 🟡 **WARNINGS**: 1 step(s) have yellow flags that should be reviewed.
-    ↳ **12_execute.py**: Significantly above average: 57.2s (6.3x avg)
- ⚡ **Performance**: Slowest step is **12_execute.py** (57.2s). Consider parallelization or caching.
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
