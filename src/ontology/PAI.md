# Ontology Module - PAI Context

## Quick Reference

**Purpose:** Manage Active Inference ontology terms and semantic mappings.

**When to use this module:**
- Map GNN terms to formal ontology
- Validate semantic consistency
- Generate ontology documentation

## Common Operations

```python
# Process ontology
from ontology.processor import OntologyProcessor
processor = OntologyProcessor(input_dir, output_dir)
results = processor.process(verbose=True)
```

## Integration Points

| Direction | Module | Data |
|-----------|--------|------|
| **Input** | gnn | Model terms |
| **Output** | validation | Semantic mappings |

## Key Files

- `processor.py` - Main processor class
- `terms.py` - Ontology term definitions
- `__init__.py` - Public API exports

## Tips for AI Assistants

1. **Step 10:** Ontology processing is Step 10
2. **Semantics:** Maps informal to formal terms
3. **Output Location:** `output/10_ontology_output/`
4. **Active Inference:** Based on AI ontology standards

---

**Version:** 1.1.3 | **Step:** 10 (Ontology)
