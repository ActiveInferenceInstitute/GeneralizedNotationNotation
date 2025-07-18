# GNN Round-Trip Testing Report
**Generated:** 2025-07-17 19:04:09
**Reference File:** `/home/trim/Documents/GitHub/GeneralizedNotationNotation/src/gnn/gnn_examples/actinf_pomdp_agent.md`

## Summary
- **Total Tests:** 21
- **Successful:** 16
- **Failed:** 5
- **Success Rate:** 76.2%

## Format Summary
- **json** ✅: 1/1 (100.0%)
- **xml** ✅: 1/1 (100.0%)
- **yaml** ✅: 1/1 (100.0%)
- **protobuf** ✅: 1/1 (100.0%)
- **xsd** ✅: 1/1 (100.0%)
- **asn1** ✅: 1/1 (100.0%)
- **pkl** ✅: 1/1 (100.0%)
- **scala** ✅: 1/1 (100.0%)
- **lean** ✅: 1/1 (100.0%)
- **coq** ✅: 1/1 (100.0%)
- **python** ✅: 1/1 (100.0%)
- **isabelle** ✅: 1/1 (100.0%)
- **haskell** ✅: 1/1 (100.0%)
- **tla_plus** ✅: 1/1 (100.0%)
- **agda** ✅: 1/1 (100.0%)
- **alloy** ✅: 1/1 (100.0%)
- **z_notation** ❌: 0/1 (0.0%)
- **bnf** ❌: 0/1 (0.0%)
- **ebnf** ❌: 0/1 (0.0%)
- **maxima** ❌: 0/1 (0.0%)
- **pickle** ❌: 0/1 (0.0%)

## Detailed Results
### json ✅ PASS
- **Semantic Checksum:** ✅

### xml ✅ PASS
- **Semantic Checksum:** ❌
- **Warnings:**
  - Semantic checksums don't match (may indicate data loss)

### yaml ✅ PASS
- **Semantic Checksum:** ✅

### protobuf ✅ PASS
- **Semantic Checksum:** ✅

### xsd ✅ PASS
- **Semantic Checksum:** ✅

### asn1 ✅ PASS
- **Semantic Checksum:** ✅

### pkl ✅ PASS
- **Semantic Checksum:** ✅

### scala ✅ PASS
- **Semantic Checksum:** ✅

### lean ✅ PASS
- **Semantic Checksum:** ✅

### coq ✅ PASS
- **Semantic Checksum:** ✅

### python ✅ PASS
- **Semantic Checksum:** ✅

### isabelle ✅ PASS
- **Semantic Checksum:** ✅

### haskell ✅ PASS
- **Semantic Checksum:** ✅

### tla_plus ✅ PASS
- **Semantic Checksum:** ✅

### agda ✅ PASS
- **Semantic Checksum:** ✅

### alloy ✅ PASS
- **Semantic Checksum:** ✅

### z_notation ❌ FAIL
- **Semantic Checksum:** ❌
- **Differences:**
  - Model name mismatch: 'Classic Active Inference POMDP Agent v1' vs 'ZNotationModel'
  - Annotation mismatch
  - Variable A type mismatch: VariableType.LIKELIHOOD_MATRIX vs VariableType.HIDDEN_STATE
  - Variable A data type mismatch: DataType.FLOAT vs DataType.INTEGER
  - Variable A dimensions mismatch: [3, 3] vs [1]
  - Variable B type mismatch: VariableType.TRANSITION_MATRIX vs VariableType.HIDDEN_STATE
  - Variable B data type mismatch: DataType.FLOAT vs DataType.INTEGER
  - Variable B dimensions mismatch: [3, 3, 3] vs [1]
  - Variable D type mismatch: VariableType.PRIOR_VECTOR vs VariableType.HIDDEN_STATE
  - Variable D data type mismatch: DataType.FLOAT vs DataType.INTEGER
  - Variable D dimensions mismatch: [3] vs [1]
  - Variable o type mismatch: VariableType.OBSERVATION vs VariableType.HIDDEN_STATE
  - Variable o dimensions mismatch: [3, 1] vs [1]
  - Variable s_prime data type mismatch: DataType.FLOAT vs DataType.INTEGER
  - Variable s_prime dimensions mismatch: [3, 1] vs [1]
  - Variable E type mismatch: VariableType.POLICY vs VariableType.HIDDEN_STATE
  - Variable E data type mismatch: DataType.FLOAT vs DataType.INTEGER
  - Variable E dimensions mismatch: [3] vs [1]
  - Variable u type mismatch: VariableType.ACTION vs VariableType.HIDDEN_STATE
  - Variable s data type mismatch: DataType.FLOAT vs DataType.INTEGER
  - Variable s dimensions mismatch: [3, 1] vs [1]
  - Variable C type mismatch: VariableType.PREFERENCE_VECTOR vs VariableType.HIDDEN_STATE
  - Variable C data type mismatch: DataType.FLOAT vs DataType.INTEGER
  - Variable C dimensions mismatch: [3] vs [1]
  - Variable G type mismatch: VariableType.POLICY vs VariableType.HIDDEN_STATE
  - Variable G data type mismatch: DataType.FLOAT vs DataType.INTEGER
  - Variable π type mismatch: VariableType.POLICY vs VariableType.HIDDEN_STATE
  - Variable π data type mismatch: DataType.FLOAT vs DataType.INTEGER
  - Variable π dimensions mismatch: [3] vs [1]
  - Connection count mismatch: 11 vs 0
  - Missing connection: s--undirected-->B
  - Missing connection: G--directed-->π
  - Missing connection: E--directed-->π
  - Missing connection: D--directed-->s
  - Missing connection: A--undirected-->o
  - Missing connection: s--directed-->s_prime
  - Missing connection: C--directed-->G
  - Missing connection: π--directed-->u
  - Missing connection: B--directed-->u
  - Missing connection: s--undirected-->A
  - Missing connection: u--directed-->s_prime
  - Missing parameter: E
  - Missing parameter: num_actions: 3       # B actions_dim
  - Missing parameter: A
  - Missing parameter: C
  - Missing parameter: B
  - Missing parameter: D
  - Time specification presence mismatch
  - Ontology mappings mismatch
- **Warnings:**
  - Semantic checksums don't match (may indicate data loss)

### bnf ❌ FAIL
- **Semantic Checksum:** ❌
- **Differences:**
  - Model name mismatch: 'Classic Active Inference POMDP Agent v1' vs 'GNN Model: Classic Active Inference POMDP Agent v1'
  - Annotation mismatch
  - Variable missing in converted: t
  - Variable missing in converted: A
  - Variable missing in converted: B
  - Variable missing in converted: D
  - Variable missing in converted: o
  - Variable missing in converted: s_prime
  - Variable missing in converted: E
  - Variable missing in converted: u
  - Variable missing in converted: s
  - Variable missing in converted: C
  - Variable missing in converted: G
  - Variable missing in converted: π
  - Extra variable in converted: variables
  - Extra variable in converted: variable_name
  - Extra variable in converted: variable
  - Connection count mismatch: 11 vs 16
  - Missing connection: s--undirected-->B
  - Missing connection: G--directed-->π
  - Missing connection: E--directed-->π
  - Missing connection: D--directed-->s
  - Missing connection: A--undirected-->o
  - Missing connection: s--directed-->s_prime
  - Missing connection: C--directed-->G
  - Missing connection: π--directed-->u
  - Missing connection: B--directed-->u
  - Missing connection: s--undirected-->A
  - Missing connection: u--directed-->s_prime
  - Extra connection: variables--directed-->gnn_model
  - Extra connection: variable_name--directed-->variable
  - Extra connection: parameter--directed-->parameters
  - Extra connection: parameters--directed-->gnn_model
  - Extra connection: connection--directed-->connections
  - Extra connection: connection_op--directed-->connection
  - Extra connection: connections--directed-->gnn_model
  - Extra connection: variable_type--directed-->variable
  - Extra connection: param_name--directed-->parameter
  - Extra connection: variable--directed-->variables
  - Extra connection: source_var--directed-->connection
  - Extra connection: target_var--directed-->connection
  - Extra connection: param_value--directed-->parameter
  - Missing parameter: E
  - Missing parameter: num_actions: 3       # B actions_dim
  - Missing parameter: A
  - Missing parameter: C
  - Missing parameter: B
  - Missing parameter: D
  - Time specification presence mismatch
  - Ontology mappings mismatch
- **Warnings:**
  - Semantic checksums don't match (may indicate data loss)

### ebnf ❌ FAIL
- **Semantic Checksum:** ❌
- **Differences:**
  - Model name mismatch: 'Classic Active Inference POMDP Agent v1' vs 'GNN Model: Classic Active Inference POMDP Agent v1'
  - Annotation mismatch
  - Variable missing in converted: t
  - Variable missing in converted: A
  - Variable missing in converted: B
  - Variable missing in converted: D
  - Variable missing in converted: o
  - Variable missing in converted: s_prime
  - Variable missing in converted: E
  - Variable missing in converted: u
  - Variable missing in converted: s
  - Variable missing in converted: C
  - Variable missing in converted: G
  - Variable missing in converted: π
  - Connection count mismatch: 11 vs 0
  - Missing connection: s--undirected-->B
  - Missing connection: G--directed-->π
  - Missing connection: E--directed-->π
  - Missing connection: D--directed-->s
  - Missing connection: A--undirected-->o
  - Missing connection: s--directed-->s_prime
  - Missing connection: C--directed-->G
  - Missing connection: π--directed-->u
  - Missing connection: B--directed-->u
  - Missing connection: s--undirected-->A
  - Missing connection: u--directed-->s_prime
  - Missing parameter: E
  - Missing parameter: num_actions: 3       # B actions_dim
  - Missing parameter: A
  - Missing parameter: C
  - Missing parameter: B
  - Missing parameter: D
  - Time specification presence mismatch
  - Ontology mappings mismatch
- **Warnings:**
  - Semantic checksums don't match (may indicate data loss)

### maxima ❌ FAIL
- **Semantic Checksum:** ✅
- **Differences:**
  - Time specification presence mismatch
  - Ontology mappings mismatch

### pickle ❌ FAIL
- **Semantic Checksum:** ✅
- **Differences:**
  - Time specification presence mismatch
  - Ontology mappings mismatch
- **Warnings:**
  - Validation error: File error: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte

## Recommendations
⚠️ **Some tests failed.** Review the failed formats and address the differences:
  - Fix serialization/parsing for z_notation
  - Fix serialization/parsing for bnf
  - Fix serialization/parsing for ebnf
  - Fix serialization/parsing for maxima
  - Fix serialization/parsing for pickle