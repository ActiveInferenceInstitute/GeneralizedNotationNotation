# Common GNN Errors and Solutions

This guide helps you diagnose and fix common issues when working with GNN models.

## 🔍 Quick Diagnosis

| Error Type | Symptoms | Quick Fix |
|------------|----------|-----------|
| **Syntax Error** | Parser fails, invalid character warnings | Check [GNN Syntax Reference](../gnn_syntax.md) |
| **Dimension Mismatch** | Type checker fails, matrix incompatibility | Verify matrix dimensions in StateSpaceBlock |
| **Connection Error** | Invalid variable references | Ensure all connected variables are defined |
| **Parameterization Error** | Probabilities don't sum to 1 | Normalize probability distributions |
| **Rendering Error** | Code generation fails | Check variable naming and matrix structures |

## 📝 Syntax Errors

### Problem: "Invalid GNN syntax"
```
Error: Unexpected character '[' at line 15
```

**Common Causes:**
- Missing commas in variable definitions
- Incorrect bracket usage `[]` vs `{}` vs `()`
- Invalid variable naming (spaces, special characters)

**Solutions:**
1. **Check variable definitions:**
   ```gnn
   # ❌ Wrong
   s f0[2,1,type=int]  # Space in variable name
   
   # ✅ Correct  
   s_f0[2,1,type=int]  # Underscore for subscripts
   ```

2. **Verify bracket usage:**
   ```gnn
   # ❌ Wrong
   s_f0{2,1,type=int}  # Curly braces for dimensions
   
   # ✅ Correct
   s_f0[2,1,type=int]  # Square brackets for dimensions
   ```

3. **Check connection syntax:**
   ```gnn
   # ❌ Wrong
   s_f0 -> A_m0 -> o_m0  # Chain notation not supported
   
   # ✅ Correct
   (s_f0) -> (A_m0)      # Each connection separately
   (A_m0) -> (o_m0)
   ```

### Problem: "Unrecognized section header"
```
Error: Unknown section "StateSpace" at line 8
```

**Solution:** Use exact section names from the [GNN File Structure](../gnn_file_structure_doc.md):
```gnn
# ❌ Wrong
## StateSpace

# ✅ Correct  
## StateSpaceBlock
```

## 🔢 Dimension and Type Errors

### Problem: "Matrix dimension mismatch"
```
Error: A_m0 expects dimensions [3,2] but got [2,3]
```

**Diagnosis:**
1. Check your StateSpaceBlock definitions
2. Verify matrix structure in InitialParameterization
3. Ensure observation outcomes match matrix rows

**Solution:**
```gnn
## StateSpaceBlock
o_m0[3,1,type=int]   # 3 possible observations
s_f0[2,1,type=int]   # 2 possible states

# A_m0 should be [observations × states] = [3,2]
A_m0[3,2,type=float] # ✅ Correct dimensions

## InitialParameterization
A_m0={
  # 3 rows (observations) × 2 columns (states)
  ((0.9, 0.1),   # P(o=0|s=0), P(o=0|s=1)  
   (0.1, 0.8),   # P(o=1|s=0), P(o=1|s=1)
   (0.0, 0.1))   # P(o=2|s=0), P(o=2|s=1)
}
```

### Problem: "Probability distributions don't sum to 1"
```
Error: B_f0 column 0 sums to 0.85, expected 1.0
```

**Solution:**
1. **Check each column sums to 1:**
   ```gnn
   # ❌ Wrong - columns don't sum to 1
   B_f0={
     ((0.7, 0.3),    # Column 0: 0.7 + 0.2 = 0.9 ≠ 1.0
      (0.2, 0.7))    # Column 1: 0.3 + 0.7 = 1.0 ✓
   }
   
   # ✅ Correct - all columns sum to 1
   B_f0={
     ((0.8, 0.3),    # Column 0: 0.8 + 0.2 = 1.0 ✓
      (0.2, 0.7))    # Column 1: 0.3 + 0.7 = 1.0 ✓
   }
   ```

2. **Use normalization helper:**
   ```python
   # Python helper for normalization
   import numpy as np
   
   # Your unnormalized matrix
   B = np.array([[0.7, 0.3], [0.2, 0.7]])
   
   # Normalize columns to sum to 1
   B_normalized = B / B.sum(axis=0)
   print(B_normalized)
   ```

## 🔗 Connection Errors

### Problem: "Undefined variable in connections"
```
Error: Variable 'G' referenced in connections but not defined in StateSpaceBlock
```

**Solution:**
1. **Add missing variables to StateSpaceBlock:**
   ```gnn
   ## StateSpaceBlock
   # Add the missing variable
   G[1,type=float]  # Expected Free Energy
   
   ## Connections
   # Now this connection is valid
   (C_m0, A_m0, B_f0) > G
   ```

2. **Check for typos in variable names:**
   ```gnn
   # ❌ Typo in connection
   (s_f0) -> (A_m0)
   (A_m0) -> (o_m0)
   (s_f0) -> (B_f0)  # Should be s_f0, not s_f1
   
   # ✅ Correct
   (s_f0) -> (A_m0)
   (A_m0) -> (o_m0)
   (s_f0) -> (B_f0)
   ```

### Problem: "Circular dependency detected"
```
Error: Circular dependency: s_f0 -> A_m0 -> s_f0
```

**Solution:**
Review your model structure. Circular dependencies usually indicate:
1. **Incorrect causality direction**
2. **Missing temporal distinction** (use `s_f0_next` for future states)
3. **Conceptual modeling error**

```gnn
# ❌ Circular
(s_f0) -> (A_m0)
(A_m0) -> (s_f0)  # Creates cycle

# ✅ Correct - temporal distinction
(s_f0) -> (A_m0)
(A_m0) -> (o_m0)
(s_f0) -> (B_f0)  
(B_f0) -> s_f0_next  # Next time step
```

## 🎯 Rendering and Code Generation Errors

### Problem: "Cannot generate PyMDP code"
```
Error: Variable naming conflicts with PyMDP reserved words
```

**Solutions:**
1. **Avoid reserved words:**
   ```gnn
   # ❌ Problematic names
   A[2,2,type=float]      # 'A' might conflict with numpy
   class[3,1,type=int]    # 'class' is Python keyword
   
   # ✅ Better names
   A_m0[2,2,type=float]   # Explicit modality naming
   object_class[3,1,type=int]  # Descriptive name
   ```

2. **Check matrix structure compatibility:**
   ```gnn
   # Ensure matrices are properly structured for target framework
   # PyMDP expects specific conventions for A, B, C, D matrices
   ```

### Problem: "LaTeX rendering fails"
```
Error: Invalid LaTeX syntax in equations section
```

**Solution:**
1. **Escape special characters:**
   ```gnn
   ## Equations
   # ❌ Unescaped underscore
   s_t = softmax(ln(D) + ln(A^T * o_t))
   
   # ✅ Properly escaped
   s\_t = \text{softmax}(\ln(D) + \ln(A^T \cdot o\_t))
   ```

2. **Use supported LaTeX commands:**
   ```gnn
   # ✅ Standard mathematical notation
   \mathbf{A}          # Bold matrix
   \mathcal{D}         # Calligraphic
   \text{softmax}      # Text function names
   ```

## 🛠️ Debugging Workflow

### Step 1: Validate Syntax
```bash
# Run the GNN type checker
python src/4_gnn_type_checker.py --target-dir your_model_directory
```

### Step 2: Check Individual Sections
1. **StateSpaceBlock**: Verify all variables are properly defined
2. **Connections**: Ensure all referenced variables exist
3. **InitialParameterization**: Check matrix dimensions and probability constraints
4. **Equations**: Validate mathematical notation

### Step 3: Test Incremental Complexity
1. Start with a minimal working model
2. Add one component at a time
3. Test after each addition
4. Isolate the problematic component

### Step 4: Use Validation Tools
```python
# Python validation script
from src.gnn_type_checker import validate_gnn_file

result = validate_gnn_file("your_model.gnn")
if not result.is_valid:
    for error in result.errors:
        print(f"Error at line {error.line_number}: {error.message}")
```

## 📋 Preventive Best Practices

### 1. Use Consistent Naming
- Follow `s_f0`, `o_m0`, `A_m0` conventions
- Use descriptive comments
- Avoid special characters

### 2. Validate Early and Often
- Run type checker after major changes
- Test with simple examples first
- Use templates for new models

### 3. Document Your Model
- Add clear ModelAnnotation
- Comment complex parameterizations
- Include usage examples

### 4. Version Control
- Track changes to your GNN files
- Tag working versions
- Document breaking changes

## 🆘 Getting Help

If you're still stuck:

1. **Check the examples** in `doc/archive/` for similar patterns
2. **Search GitHub Issues** for related problems
3. **Post in GitHub Discussions** with:
   - Your GNN file (or minimal reproducing example)
   - Error messages
   - What you've already tried
4. **Review the specification** in [GNN Syntax](../gnn_syntax.md) and [File Structure](../gnn_file_structure_doc.md)

## 🔄 Error Recovery Templates

### Quick Fix: Basic POMDP Model
```gnn
## GNNVersionAndFlags
GNN v1

## ModelName
Debug Test Model

## ModelAnnotation
Minimal model for debugging

## StateSpaceBlock
s_f0[2,1,type=int]
o_m0[2,1,type=int]
A_m0[2,2,type=float]
D_f0[2,type=float]

## Connections
(D_f0) -> (s_f0)
(s_f0) -> (A_m0)
(A_m0) -> (o_m0)

## InitialParameterization
A_m0={((0.9,0.1),(0.1,0.9))}
D_f0={(0.5,0.5)}

## Time
Static

## Footer
Debug Test Model
```

This minimal model should always parse correctly and can serve as a baseline for debugging more complex models. 