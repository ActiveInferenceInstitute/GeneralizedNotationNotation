"""
Consistency Checker

This module provides consistency checking for GNN models, including
naming conventions, style consistency, and structural integrity.
"""

import re
from typing import Dict, Any, List, Optional, Union

class ConsistencyChecker:
    """Checker for consistency aspects of GNN models."""
    
    def __init__(self):
        """Initialize the consistency checker."""
        pass
    
    def check(self, content: str) -> Dict[str, Any]:
        """
        Check the consistency of a GNN model.
        
        Args:
            content: GNN model content
            
        Returns:
            Consistency check result with warnings
        """
        # Run all consistency checks
        naming_result = self._check_naming_conventions(content)
        style_result = self._check_style_consistency(content)
        structure_result = self._check_structural_integrity(content)
        reference_result = self._check_reference_consistency(content)
        
        # Combine warnings
        warnings = []
        warnings.extend(naming_result.get("warnings", []))
        warnings.extend(style_result.get("warnings", []))
        warnings.extend(structure_result.get("warnings", []))
        warnings.extend(reference_result.get("warnings", []))
        
        # Determine overall consistency
        is_consistent = (
            naming_result.get("is_consistent", True) and
            style_result.get("is_consistent", True) and
            structure_result.get("is_consistent", True) and
            reference_result.get("is_consistent", True)
        )
        
        return {
            "is_consistent": is_consistent,
            "warnings": warnings,
            "checks": {
                "naming_conventions": naming_result,
                "style_consistency": style_result,
                "structural_integrity": structure_result,
                "reference_consistency": reference_result
            }
        }
    
    def _check_naming_conventions(self, content: str) -> Dict[str, Any]:
        """Check naming conventions."""
        warnings = []
        
        # Extract named elements
        state_blocks = re.findall(r'StateSpaceBlock\s*\{([^}]*)\}', content)
        block_names = []
        
        for block in state_blocks:
            name_match = re.search(r'Name:\s*([^\n]+)', block)
            if name_match:
                block_names.append(name_match.group(1).strip())
        
        # Check for naming consistency
        camel_case = sum(1 for name in block_names if name and name[0].isupper() and '_' not in name)
        snake_case = sum(1 for name in block_names if '_' in name)
        pascal_case = sum(1 for name in block_names if name and name[0].isupper() and not any(c.islower() for c in name))
        
        # Determine dominant naming convention
        naming_styles = {"camelCase": camel_case, "snake_case": snake_case, "PascalCase": pascal_case}
        dominant_style = max(naming_styles.items(), key=lambda x: x[1])[0] if naming_styles else None
        
        # Check for consistency
        mixed_styles = sum(count > 0 for count in naming_styles.values()) > 1
        if mixed_styles:
            warnings.append(f"Inconsistent naming conventions: mix of {', '.join(style for style, count in naming_styles.items() if count > 0)}")
        
        # Check for duplicate names
        duplicate_names = set([name for name in block_names if block_names.count(name) > 1])
        if duplicate_names:
            warnings.append(f"Duplicate block names found: {', '.join(duplicate_names)}")
        
        # Check for descriptive names
        non_descriptive_names = [name for name in block_names if len(name) < 3]
        if non_descriptive_names:
            warnings.append(f"Non-descriptive block names found: {', '.join(non_descriptive_names)}")
        
        return {
            "is_consistent": len(warnings) == 0,
            "warnings": warnings,
            "dominant_style": dominant_style,
            "mixed_styles": mixed_styles,
            "duplicate_names": list(duplicate_names),
            "non_descriptive_names": non_descriptive_names
        }
    
    def _check_style_consistency(self, content: str) -> Dict[str, Any]:
        """Check style consistency."""
        warnings = []
        
        # Check for consistent indentation
        lines = content.split('\n')
        indentation_patterns = {}
        
        for line in lines:
            if line.strip() and line.startswith(' '):
                indent = len(line) - len(line.lstrip(' '))
                if indent not in indentation_patterns:
                    indentation_patterns[indent] = 0
                indentation_patterns[indent] += 1
        
        # Determine if there are inconsistent indentation patterns
        if len(indentation_patterns) > 2:
            warnings.append(f"Inconsistent indentation patterns: {len(indentation_patterns)} different patterns detected")
        
        # Check for consistent block formatting
        state_block_formats = set()
        state_blocks = re.findall(r'StateSpaceBlock\s*\{([^}]*)\}', content)
        
        for block in state_blocks:
            # Check if fields are consistently formatted
            fields = re.findall(r'([A-Za-z]+):\s*([^\n]+)', block)
            if fields:
                format_style = "inline" if len(fields) == 1 and len(block.strip().split('\n')) == 1 else "multiline"
                state_block_formats.add(format_style)
        
        if len(state_block_formats) > 1:
            warnings.append("Inconsistent block formatting: mix of inline and multiline formats")
        
        # Check for consistent field ordering
        field_orders = []
        for block in state_blocks:
            fields = re.findall(r'([A-Za-z]+):', block)
            if fields:
                field_orders.append(tuple(fields))
        
        unique_orders = set(field_orders)
        if len(unique_orders) > 1:
            warnings.append("Inconsistent field ordering across blocks")
        
        return {
            "is_consistent": len(warnings) == 0,
            "warnings": warnings,
            "indentation_patterns": len(indentation_patterns),
            "block_format_styles": list(state_block_formats),
            "field_order_consistency": len(unique_orders) == 1
        }
    
    def _check_structural_integrity(self, content: str) -> Dict[str, Any]:
        """Check structural integrity."""
        warnings = []
        
        # Check for balanced braces
        open_braces = content.count('{')
        close_braces = content.count('}')
        if open_braces != close_braces:
            warnings.append(f"Unbalanced braces: {open_braces} opening vs {close_braces} closing")
        
        # Check for proper block structure
        state_blocks = re.findall(r'StateSpaceBlock\s*\{([^}]*)\}', content)
        connections = re.findall(r'Connection\s*\{([^}]*)\}', content)
        
        # Check for empty blocks
        empty_blocks = sum(1 for block in state_blocks if not block.strip())
        if empty_blocks > 0:
            warnings.append(f"Empty StateSpaceBlock definitions found: {empty_blocks}")
        
        empty_connections = sum(1 for conn in connections if not conn.strip())
        if empty_connections > 0:
            warnings.append(f"Empty Connection definitions found: {empty_connections}")
        
        # Check for consistent field presence
        required_state_fields = ["Name", "Dimensions"]
        required_connection_fields = ["From", "To"]
        
        missing_state_fields = []
        for i, block in enumerate(state_blocks):
            for field in required_state_fields:
                if not re.search(f'{field}:', block):
                    missing_state_fields.append((i, field))
        
        if missing_state_fields:
            field_warnings = [f"Block {i} missing {field}" for i, field in missing_state_fields]
            warnings.append(f"Missing required fields in StateSpaceBlocks: {', '.join(field_warnings)}")
        
        missing_connection_fields = []
        for i, conn in enumerate(connections):
            for field in required_connection_fields:
                if not re.search(f'{field}:', conn):
                    missing_connection_fields.append((i, field))
        
        if missing_connection_fields:
            field_warnings = [f"Connection {i} missing {field}" for i, field in missing_connection_fields]
            warnings.append(f"Missing required fields in Connections: {', '.join(field_warnings)}")
        
        return {
            "is_consistent": len(warnings) == 0,
            "warnings": warnings,
            "balanced_braces": open_braces == close_braces,
            "empty_blocks": empty_blocks,
            "empty_connections": empty_connections,
            "missing_state_fields": missing_state_fields,
            "missing_connection_fields": missing_connection_fields
        }
    
    def _check_reference_consistency(self, content: str) -> Dict[str, Any]:
        """Check reference consistency."""
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
        
        # Check for references to non-existent blocks
        invalid_references = []
        for i, conn in enumerate(connections):
            from_match = re.search(r'From:\s*([^\n]+)', conn)
            to_match = re.search(r'To:\s*([^\n]+)', conn)
            
            if from_match and from_match.group(1).strip() not in block_names:
                invalid_references.append((i, "From", from_match.group(1).strip()))
            
            if to_match and to_match.group(1).strip() not in block_names:
                invalid_references.append((i, "To", to_match.group(1).strip()))
        
        if invalid_references:
            ref_warnings = [f"Connection {i} references non-existent {field} block: '{ref}'" 
                           for i, field, ref in invalid_references]
            warnings.append(f"Invalid block references: {', '.join(ref_warnings)}")
        
        # Check for isolated blocks (no incoming or outgoing connections)
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
        
        # Check for circular references
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
            warnings.append(f"Circular dependencies detected in blocks: {', '.join(cycles)}")
        
        return {
            "is_consistent": len(warnings) == 0,
            "warnings": warnings,
            "invalid_references": invalid_references,
            "isolated_blocks": isolated_blocks,
            "circular_references": cycles
        } 

def check_consistency(content: str) -> Dict[str, Any]:
    """
    Convenience function for consistency checking.
    
    Args:
        content: GNN model content string
        
    Returns:
        Dictionary with consistency check results
    """
    checker = ConsistencyChecker()
    return checker.check(content)