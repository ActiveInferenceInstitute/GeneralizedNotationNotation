# GNN Troubleshooting Guide

## Overview

This guide addresses common issues encountered when working with Generalized Notation Notation (GNN) files and the GNN processing pipeline. It covers syntax errors, pipeline failures, framework integration issues, and validation warnings.

## 🔍 Diagnostic Tools

Before diving into specific errors, use these tools to diagnose issues:

### 1. Verbose Mode
Run the pipeline with `--verbose` to see detailed logs:
```bash
python src/main.py --verbose
```

### 2. Step-Specific Checks
Run individual steps to isolate the problem:
```bash
# Check parsing
python src/3_gnn.py --target-dir input/gnn_files --verbose

# Check type validity
python src/5_type_checker.py --target-dir input/gnn_files --verbose
```

### 3. Output Inspection
Check the `output/` directory for intermediate results and error logs:
- `output/pipeline_execution_summary.json`: Overall status
- `output/3_gnn_output/gnn_processing_results.json`: Parsing details
- `output/5_type_checker_output/type_check_results.json`: Type errors

---

## Common Issues and Solutions

### 1. Parsing Errors (Step 3)

#### "Failed to parse GNN file at line X"
**Symptom**: The pipeline stops or logs an error during Step 3.
**Cause**: Invalid Markdown syntax or GNN structural violation.
**Common Triggers**:
- Missing section headers (e.g., `## State Space Block`)
- Incorrect indentation in variable definitions
- Using unsupported characters in variable names (only alphanumeric, `_`, `^`, `+` allowed)

**Solution**:
1. Check the line number in the error log.
2. Verify the section header format (must start with `##`).
3. Ensure variable definitions follow `Name[Dimensions,type=Type]`.

**Example Fix**:
```markdown
# BAD
MyVariable: [2, 1]

# GOOD
MyVariable[2,1,type=float]
```

#### "Unknown section: X"
**Symptom**: Warning about an unrecognized section.
**Cause**: Misspelled or non-standard section header.
**Solution**: Refer to the [GNN File Structure Guide](gnn_file_structure_doc.md) for valid section names.

### 2. Type Checking Errors (Step 5)

#### "Dimension mismatch"
**Symptom**: Validation fails with matrix dimension errors.
**Cause**: The dimensions of connected variables do not match matrix requirements.
**Solution**:
- For `A` matrix (Likelihood): `A[No,Ns]` where `No` is observation dim and `Ns` is state dim.
- For `B` matrix (Transition): `B[Ns,Ns]` or `B[Ns,Ns,Na]` (with action).
- Check `State Space Block` definitions against `Initial Parameterization`.

#### "Invalid type assignment"
**Symptom**: Warning about type incompatibility.
**Cause**: Assigning a float value to an integer-typed variable or vice-versa.
**Solution**: Ensure `type=int` or `type=float` matches the values in `Initial Parameterization`.

### 3. Framework Integration Issues (Steps 11 & 12)

#### "PyMDP generation failed"
**Symptom**: Step 11 succeeds but PyMDP code generation is incomplete or errors.
**Cause**: Missing required matrices (A, B, C, D) or incorrect variable naming for PyMDP conventions.
**Solution**:
- PyMDP requires specific matrix names (A, B, C, D, E).
- Ensure `ActInfOntologyAnnotation` maps your variables to `RecognitionMatrix`, `TransitionMatrix`, etc.

#### "RxInfer execution error"
**Symptom**: Step 12 fails when running Julia code.
**Cause**: Julia environment issues or RxInfer syntax mismatch.
**Solution**:
- Run `python src/1_setup.py --check-julia` to verify Julia setup.
- Check `output/11_render_output/*.jl` for syntax correctness.

### 4. Visualization Issues (Steps 8 & 9)

#### "Graph generation failed"
**Symptom**: No graph outputs in `output/8_visualization_output/`.
**Cause**: `graphviz` or `networkx` issues, or disconnected graph structure.
**Solution**:
- Ensure dependencies are installed: `pip install graphviz networkx`.
- Verify `Connections` section in GNN file is not empty.

---

## Advanced Debugging

### Correlation IDs
The pipeline assigns a unique `correlation_id` to each run. Use this ID to grep through logs if you are running multiple concurrent pipelines or using a centralized logging system.

### Pipeline Recovery
If a step fails, the pipeline creates a recovery checkpoint.
**To resume**:
The current pipeline design is stateless between runs, but you can skip successful steps using `--only-steps`.

```bash
# Skip steps 0-3 if they passed
python src/main.py --only-steps "4,5,6,7,8..."
```

---

## Getting Help

If you encounter an issue not listed here:

1. Check the **[Technical Reference](technical_reference.md)** for detailed data flow.
2. Review **[GNN Standards](gnn_standards.md)** for compliance.
3. Open an issue on the [GitHub Repository](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation) with your GNN file and the error log.




