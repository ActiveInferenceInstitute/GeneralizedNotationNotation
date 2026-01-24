# Security Module - PAI Context

## Quick Reference

**Purpose:** Security validation and safe execution of generated code.

**When to use this module:**
- Validate code safety
- Sandbox execution
- Security audit generated code

## Common Operations

```python
# Run security checks
from security.processor import SecurityProcessor
processor = SecurityProcessor(input_dir, output_dir)
results = processor.process(verbose=True)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | render | Generated code |
| **Output** | execute | Security status |

## Key Files

- `processor.py` - Main processor class
- `validator.py` - Security validation
- `__init__.py` - Public API exports

## Tips for AI Assistants

1. **Step 18:** Security is Step 18
2. **Sandboxing:** Isolated code execution
3. **Output Location:** `output/18_security_output/`
4. **Validation:** Checks for dangerous patterns

---

**Version:** 1.1.3 | **Step:** 18 (Security)
