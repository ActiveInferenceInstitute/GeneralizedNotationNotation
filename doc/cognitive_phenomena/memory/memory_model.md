# Memory Systems - GNN Implementation

> **Status**: Production Ready | **Version**: 1.0

## Overview

Memory in Active Inference: Episodic, Semantic, Working, and Procedural memory systems.

## Working Memory Model

```markdown
# Working Memory POMDP
## GNNSection
WorkingMemoryPOMDP

## StateSpaceBlock
w[3,1,type=float]     # Items in working memory
c[2,type=float]     # Task context

o[5,type=float]    # Observations

## Connections
D>w
D>c
w-A
A-o
```

## Memory Types

| Type | Function |
|------|----------|
| **Working Memory** | Active maintenance of information |
| **Episodic Memory** | Storage of specific experiences |
| **Semantic Memory** | Factual knowledge |
| **Procedural Memory** | Motor skills and habits |

## References

- Working Memory: Baddeley & Hitch (1974)
- Episodic Memory: Tulving (1972)

---

*See also: [doc/cognitive_phenomena/perception/](../perception/) for perception models*
