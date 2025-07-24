"""
Semantic Validator

This module provides semantic validation for GNN models beyond basic syntax checking.
It validates compliance with Active Inference principles, mathematical consistency,
and causal relationship validation.
"""

import re
from typing import Dict, Any, List, Optional, Union

class SemanticValidator:
    """Validator for semantic aspects of GNN models."""
    
    def __init__(self, validation_level: str = "standard"):
        """
        Initialize the semantic validator.
        
        Args:
            validation_level: Validation level (basic, standard, strict, research)
        """
        self.validation_level = validation_level
        self.validation_rules = self._get_validation_rules()
    
    def validate(self, content: str) -> Dict[str, Any]:
        """
        Validate the semantic aspects of a GNN model.
        
        Args:
            content: GNN model content
            
        Returns:
            Validation result with errors and warnings
        """
        errors = []
        warnings = []
        
        # Apply validation rules based on level
        for rule in self.validation_rules:
            if rule["level"] <= self._get_level_value():
                result = rule["validator"](content)
                errors.extend(result.get("errors", []))
                warnings.extend(result.get("warnings", []))
        
        return {
            "is_valid": len(errors) == 0,
            "validation_level": self.validation_level,
            "errors": errors,
            "warnings": warnings
        }
    
    def _get_level_value(self) -> int:
        """Convert validation level string to numeric value."""
        levels = {
            "basic": 1,
            "standard": 2,
            "strict": 3,
            "research": 4
        }
        return levels.get(self.validation_level.lower(), 2)  # Default to standard
    
    def _get_validation_rules(self) -> List[Dict[str, Any]]:
        """Get validation rules based on validation level."""
        return [
            {
                "name": "basic_structure",
                "level": 1,  # Basic
                "validator": self._validate_basic_structure
            },
            {
                "name": "state_space_definitions",
                "level": 1,  # Basic
                "validator": self._validate_state_space_definitions
            },
            {
                "name": "connection_integrity",
                "level": 2,  # Standard
                "validator": self._validate_connection_integrity
            },
            {
                "name": "mathematical_consistency",
                "level": 2,  # Standard
                "validator": self._validate_mathematical_consistency
            },
            {
                "name": "active_inference_principles",
                "level": 3,  # Strict
                "validator": self._validate_active_inference_principles
            },
            {
                "name": "causal_relationships",
                "level": 3,  # Strict
                "validator": self._validate_causal_relationships
            },
            {
                "name": "advanced_mathematical_properties",
                "level": 4,  # Research
                "validator": self._validate_advanced_mathematical_properties
            }
        ]
    
    def _validate_basic_structure(self, content: str) -> Dict[str, Any]:
        """Validate basic structure of the GNN model."""
        errors = []
        warnings = []
        
        # Check for required elements
        if not re.search(r'ModelName:', content):
            warnings.append("Missing ModelName definition")
        
        if not re.search(r'StateSpaceBlock', content):
            errors.append("Missing StateSpaceBlock definition")
        
        if not re.search(r'Connection', content) and re.search(r'StateSpaceBlock', content):
            warnings.append("Model has StateSpaceBlocks but no Connections")
        
        # Check for proper block structure
        state_blocks = re.findall(r'StateSpaceBlock\s*\{([^}]*)\}', content)
        for i, block in enumerate(state_blocks):
            if not block.strip():
                errors.append(f"Empty StateSpaceBlock at index {i}")
        
        return {
            "errors": errors,
            "warnings": warnings
        }
    
    def _validate_state_space_definitions(self, content: str) -> Dict[str, Any]:
        """Validate state space definitions."""
        errors = []
        warnings = []
        
        # Extract state blocks
        state_blocks = re.findall(r'StateSpaceBlock\s*\{([^}]*)\}', content)
        
        for i, block in enumerate(state_blocks):
            # Check for required fields
            if not re.search(r'Name:', block):
                errors.append(f"StateSpaceBlock at index {i} missing Name field")
            
            if not re.search(r'Dimensions:', block):
                errors.append(f"StateSpaceBlock at index {i} missing Dimensions field")
            
            # Check dimensions format
            dim_match = re.search(r'Dimensions:\s*([^\n]+)', block)
            if dim_match:
                dims_str = dim_match.group(1).strip()
                try:
                    dims = [int(d.strip()) for d in dims_str.split(',')]
                    if any(d <= 0 for d in dims):
                        errors.append(f"StateSpaceBlock at index {i} has non-positive dimensions: {dims_str}")
                except ValueError:
                    errors.append(f"StateSpaceBlock at index {i} has invalid dimensions format: {dims_str}")
        
        return {
            "errors": errors,
            "warnings": warnings
        }
    
    def _validate_connection_integrity(self, content: str) -> Dict[str, Any]:
        """Validate connection integrity."""
        errors = []
        warnings = []
        
        # Extract state blocks and connections
        state_blocks = re.findall(r'StateSpaceBlock\s*\{([^}]*)\}', content)
        connections = re.findall(r'Connection\s*\{([^}]*)\}', content)
        
        # Extract block names
        block_names = []
        for block in state_blocks:
            name_match = re.search(r'Name:\s*([^\n]+)', block)
            if name_match:
                block_names.append(name_match.group(1).strip())
        
        # Check connections
        for i, conn in enumerate(connections):
            # Check for required fields
            if not re.search(r'From:', conn):
                errors.append(f"Connection at index {i} missing From field")
            
            if not re.search(r'To:', conn):
                errors.append(f"Connection at index {i} missing To field")
            
            # Check if referenced blocks exist
            from_match = re.search(r'From:\s*([^\n]+)', conn)
            to_match = re.search(r'To:\s*([^\n]+)', conn)
            
            if from_match and from_match.group(1).strip() not in block_names:
                errors.append(f"Connection {i} references non-existent 'From' block: {from_match.group(1).strip()}")
            
            if to_match and to_match.group(1).strip() not in block_names:
                errors.append(f"Connection {i} references non-existent 'To' block: {to_match.group(1).strip()}")
        
        # Check for isolated blocks
        connected_blocks = set()
        for conn in connections:
            from_match = re.search(r'From:\s*([^\n]+)', conn)
            to_match = re.search(r'To:\s*([^\n]+)', conn)
            
            if from_match:
                connected_blocks.add(from_match.group(1).strip())
            if to_match:
                connected_blocks.add(to_match.group(1).strip())
        
        isolated_blocks = [name for name in block_names if name not in connected_blocks]
        if isolated_blocks:
            warnings.append(f"Isolated blocks with no connections: {', '.join(isolated_blocks)}")
        
        return {
            "errors": errors,
            "warnings": warnings
        }
    
    def _validate_mathematical_consistency(self, content: str) -> Dict[str, Any]:
        """Validate mathematical consistency."""
        errors = []
        warnings = []
        
        # Extract state blocks and connections
        state_blocks = re.findall(r'StateSpaceBlock\s*\{([^}]*)\}', content)
        connections = re.findall(r'Connection\s*\{([^}]*)\}', content)
        
        # Extract block dimensions
        block_dims = {}
        for block in state_blocks:
            name_match = re.search(r'Name:\s*([^\n]+)', block)
            dim_match = re.search(r'Dimensions:\s*([^\n]+)', block)
            
            if name_match and dim_match:
                name = name_match.group(1).strip()
                try:
                    dims = [int(d.strip()) for d in dim_match.group(1).strip().split(',')]
                    block_dims[name] = dims
                except ValueError:
                    pass  # Already caught in state space validation
        
        # Check connection dimension compatibility
        for i, conn in enumerate(connections):
            from_match = re.search(r'From:\s*([^\n]+)', conn)
            to_match = re.search(r'To:\s*([^\n]+)', conn)
            mapping_match = re.search(r'Mapping:\s*([^\n]+)', conn)
            
            if from_match and to_match:
                from_block = from_match.group(1).strip()
                to_block = to_match.group(1).strip()
                
                if from_block in block_dims and to_block in block_dims:
                    from_dims = block_dims[from_block]
                    to_dims = block_dims[to_block]
                    
                    # Check if mapping is specified
                    if mapping_match:
                        mapping = mapping_match.group(1).strip()
                        # TODO: Implement mapping validation
                    else:
                        # Default mapping: check if dimensions match
                        if len(from_dims) != len(to_dims):
                            warnings.append(f"Connection {i} between blocks with different dimension counts: {from_block}({len(from_dims)}) -> {to_block}({len(to_dims)})")
                        
                        # Check if dimensions are compatible
                        for j, (from_dim, to_dim) in enumerate(zip(from_dims, to_dims)):
                            if from_dim != to_dim:
                                warnings.append(f"Connection {i} has mismatched dimensions at index {j}: {from_dim} -> {to_dim}")
        
        return {
            "errors": errors,
            "warnings": warnings
        }
    
    def _validate_active_inference_principles(self, content: str) -> Dict[str, Any]:
        """Validate compliance with Active Inference principles."""
        errors = []
        warnings = []
        
        # Check for required components in Active Inference models
        has_observation = bool(re.search(r'ObservationModel|Observation', content, re.IGNORECASE))
        has_transition = bool(re.search(r'TransitionModel|Dynamics', content, re.IGNORECASE))
        has_prior = bool(re.search(r'Prior|PriorPreferences', content, re.IGNORECASE))
        
        if not has_observation:
            warnings.append("Active Inference model missing explicit observation model")
        
        if not has_transition:
            warnings.append("Active Inference model missing explicit transition dynamics")
        
        if not has_prior:
            warnings.append("Active Inference model missing explicit priors or preferences")
        
        # Check for free energy minimization principle
        has_free_energy = bool(re.search(r'FreeEnergy|VariationalFreeEnergy|VFE', content, re.IGNORECASE))
        if not has_free_energy:
            warnings.append("Active Inference model does not explicitly mention free energy minimization")
        
        return {
            "errors": errors,
            "warnings": warnings
        }
    
    def _validate_causal_relationships(self, content: str) -> Dict[str, Any]:
        """Validate causal relationships in the model."""
        errors = []
        warnings = []
        
        # Extract connections
        connections = re.findall(r'Connection\s*\{([^}]*)\}', content)
        
        # Check for circular dependencies
        graph = {}
        for conn in connections:
            from_match = re.search(r'From:\s*([^\n]+)', conn)
            to_match = re.search(r'To:\s*([^\n]+)', conn)
            
            if from_match and to_match:
                from_block = from_match.group(1).strip()
                to_block = to_match.group(1).strip()
                
                if from_block not in graph:
                    graph[from_block] = []
                graph[from_block].append(to_block)
        
        # Check for cycles
        visited = set()
        path = set()
        
        def has_cycle(node):
            if node in path:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            path.add(node)
            
            for neighbor in graph.get(node, []):
                if has_cycle(neighbor):
                    return True
            
            path.remove(node)
            return False
        
        cycles = []
        for node in graph:
            if has_cycle(node):
                cycles.append(node)
        
        if cycles:
            warnings.append(f"Potential circular dependencies detected in blocks: {', '.join(cycles)}")
        
        return {
            "errors": errors,
            "warnings": warnings
        }
    
    def _validate_advanced_mathematical_properties(self, content: str) -> Dict[str, Any]:
        """Validate advanced mathematical properties."""
        errors = []
        warnings = []
        
        # This is a placeholder for advanced mathematical validation
        # In a real implementation, this would include:
        # - Markov blanket validation
        # - Information geometry validation
        # - Ergodicity validation
        # - Convergence properties
        
        # For now, just check for some advanced concepts
        has_markov_blanket = bool(re.search(r'MarkovBlanket', content, re.IGNORECASE))
        has_information_geometry = bool(re.search(r'InformationGeometry|FisherInformation', content, re.IGNORECASE))
        has_ergodicity = bool(re.search(r'Ergodic|Ergodicity', content, re.IGNORECASE))
        
        if not has_markov_blanket:
            warnings.append("Advanced model does not explicitly define Markov blankets")
        
        if not has_information_geometry:
            warnings.append("Advanced model does not reference information geometry concepts")
        
        if not has_ergodicity:
            warnings.append("Advanced model does not address ergodicity assumptions")
        
        return {
            "errors": errors,
            "warnings": warnings
        } 