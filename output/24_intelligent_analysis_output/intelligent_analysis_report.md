# Pipeline Intelligent Analysis Report

**Generated**: 2026-02-15T14:33:13.474639

**Status**: ✅ SUCCESS

**Health Score**: 100.0/100


## Quick Overview

| Metric | Value |
|--------|-------|
| Total Steps | 7 |
| Duration | 58.79s |
| Peak Memory | 71.7 MB |
| 🔴 Red Flags | 0 |
| 🟡 Yellow Flags | 1 |
| ✅ Green (Clean) | 6 |

## 🟡 Yellow Flags (Warnings)

| Step | Duration | Memory | Issues |
|------|----------|--------|--------|
| 12_execute.py | 52.7s | 72MB | Significantly above average: 52.7s (6.3x avg) |

## Per-Step Execution Details

| # | Step | Status | Duration | Memory | Flags |
|---|------|--------|----------|--------|-------|
| 1 | 3_gnn.py | ✅ SUCCESS | 0.34s | 71MB | - |
| 2 | 5_type_checker.py | ✅ SUCCESS | 0.28s | 71MB | - |
| 3 | 7_export.py | ✅ SUCCESS | 0.29s | 71MB | - |
| 4 | 8_visualization.py | ✅ SUCCESS | 3.83s | 71MB | - |
| 5 | 11_render.py | ✅ SUCCESS | 0.33s | 72MB | - |
| 6 | 12_execute.py | 🟡 SUCCESS | 52.73s | 72MB | 1 |
| 7 | 15_audio.py | ✅ SUCCESS | 0.98s | 27MB | - |

## Detailed Step Output (Flagged Steps)

### 12_execute.py

**Execution**

- Status: SUCCESS
- Duration: 52.73s
- Memory: 72MB
- Flags: Significantly above average: 52.7s (6.3x avg)

## Performance Bottlenecks

| Step | Duration (s) | Memory (MB) | Above Avg Ratio |
|------|-------------|-------------|-----------------|
| 12_execute.py | 52.7 | 72 | 6.3x |

## Recommendations

- 🟡 **WARNINGS**: 1 step(s) have yellow flags that should be reviewed.
-    ↳ **12_execute.py**: Significantly above average: 52.7s (6.3x avg)
- ⚡ **Performance**: Slowest step is **12_execute.py** (52.7s). Consider parallelization or caching.
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
