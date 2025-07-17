# GNN File: test_ontology_validation.md\n\n## Raw File Content\n\n```\n# Test Ontology Validation

This is a test file to validate enhanced ontology processing features.

## StateSpaceBlock
A[3,3,type=float]
B[3,3,type=float]
invalidVar[2,1,type=float]

## ActInfOntologyAnnotation
A=LikelihoodMatrix
B=transitionmatrix  # Case mismatch test
invalidVar=InvalidTerm  # Should be flagged as invalid
C=HiddenStae  # Typo that should trigger fuzzy matching (should suggest HiddenState)
D=SomethingCompletelyWrong  # Should be flagged as invalid
E=  # Empty value test
# F=CommentedOut  # Should be ignored
validTerm=Action \n```\n\n## Parsed Sections

### _HeaderComments

```
# Test Ontology Validation

This is a test file to validate enhanced ontology processing features.
```

### StateSpaceBlock

```
A[3,3,type=float]
B[3,3,type=float]
invalidVar[2,1,type=float]
```

### ActInfOntologyAnnotation

```
A=LikelihoodMatrix
B=transitionmatrix  # Case mismatch test
invalidVar=InvalidTerm  # Should be flagged as invalid
C=HiddenStae  # Typo that should trigger fuzzy matching (should suggest HiddenState)
D=SomethingCompletelyWrong  # Should be flagged as invalid
E=  # Empty value test
# F=CommentedOut  # Should be ignored
validTerm=Action
```

### ModelName

```
Test Ontology Validation
```

