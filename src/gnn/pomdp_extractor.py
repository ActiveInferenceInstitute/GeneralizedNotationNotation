#!/usr/bin/env python3
"""
POMDP State Space Extractor for GNN Active Inference Models

This module provides specialized parsing and extraction capabilities for POMDP 
(Partially Observable Markov Decision Process) state spaces from GNN specifications,
with focus on Active Inference model structures.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class POMDPStateSpace:
    """Represents extracted POMDP state space information."""
    
    # Core dimensions
    num_states: int
    num_observations: int  
    num_actions: int
    
    # Active Inference matrices and vectors
    A_matrix: Optional[List[List[float]]] = None  # Likelihood: P(o|s)
    B_matrix: Optional[List[List[List[float]]]] = None  # Transition: P(s'|s,a)
    C_vector: Optional[List[float]] = None  # Preferences over observations
    D_vector: Optional[List[float]] = None  # Prior beliefs over states
    E_vector: Optional[List[float]] = None  # Policy priors
    
    # State space variables
    state_variables: Optional[List[Dict[str, Any]]] = None
    observation_variables: Optional[List[Dict[str, Any]]] = None
    action_variables: Optional[List[Dict[str, Any]]] = None
    
    # Connections/relationships
    connections: Optional[List[Tuple[str, str, str]]] = None  # (source, relation, target)
    
    # Metadata
    model_name: Optional[str] = None
    model_annotation: Optional[str] = None
    ontology_mapping: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'num_states': self.num_states,
            'num_observations': self.num_observations,
            'num_actions': self.num_actions,
            'A_matrix': self.A_matrix,
            'B_matrix': self.B_matrix,
            'C_vector': self.C_vector,
            'D_vector': self.D_vector,
            'E_vector': self.E_vector,
            'state_variables': self.state_variables,
            'observation_variables': self.observation_variables,
            'action_variables': self.action_variables,
            'connections': self.connections,
            'model_name': self.model_name,
            'model_annotation': self.model_annotation,
            'ontology_mapping': self.ontology_mapping
        }


class POMDPExtractor:
    """
    Specialized extractor for POMDP state spaces from GNN specifications.
    
    Features:
    - Parses Active Inference matrix structures (A, B, C, D, E)
    - Extracts state space dimensions and variable definitions
    - Handles initial parameterization with matrix values
    - Maps ontology annotations to Active Inference concepts
    - Validates POMDP structural consistency
    """
    
    def __init__(self, strict_validation: bool = True):
        """
        Initialize POMDP extractor.
        
        Args:
            strict_validation: Enable strict validation of POMDP structure
        """
        self.strict_validation = strict_validation
        self.logger = logging.getLogger(__name__)
        
        # Patterns for parsing GNN content
        self.SECTION_PATTERN = re.compile(r'^##\s+(.+)$', re.MULTILINE)
        self.VARIABLE_PATTERN = re.compile(r'^([A-Za-z_π][A-Za-z0-9_π]*)\[([^\]]+)\](?:,type=([a-zA-Z]+))?(?:\s*#\s*(.*))?$')
        self.CONNECTION_PATTERN = re.compile(r'^(.+?)\s*(>|->|-|\|)\s*(.+?)(?:\s*#\s*(.*))?$')
        self.PARAMETER_PATTERN = re.compile(r'^([A-Za-z_π][A-Za-z0-9_π]*)\s*=\s*\{(.+)\}', re.MULTILINE | re.DOTALL)
        
    def extract_from_gnn_content(self, content: str) -> Optional[POMDPStateSpace]:
        """
        Extract POMDP state space from GNN content.
        
        Args:
            content: Raw GNN file content
            
        Returns:
            POMDPStateSpace object or None if extraction fails
        """
        try:
            sections = self._parse_sections(content)
            
            # Extract basic information
            model_name = self._extract_model_name(sections)
            model_annotation = self._extract_model_annotation(sections)
            
            # Parse state space block
            state_space_info = self._parse_state_space_block(sections.get('StateSpaceBlock', ''))
            
            # Extract dimensions
            num_states, num_observations, num_actions = self._extract_dimensions(state_space_info)
            
            # Parse initial parameterization
            initial_params = self._parse_initial_parameterization(sections.get('InitialParameterization', ''))
            
            # Parse connections
            connections = self._parse_connections(sections.get('Connections', ''))
            
            # Parse ontology mapping
            ontology_mapping = self._parse_ontology_annotations(sections.get('ActInfOntologyAnnotation', ''))
            
            # Create POMDP state space
            pomdp_space = POMDPStateSpace(
                num_states=num_states,
                num_observations=num_observations,
                num_actions=num_actions,
                A_matrix=initial_params.get('A'),
                B_matrix=initial_params.get('B'), 
                C_vector=initial_params.get('C'),
                D_vector=initial_params.get('D'),
                E_vector=initial_params.get('E'),
                state_variables=state_space_info.get('state_variables'),
                observation_variables=state_space_info.get('observation_variables'),
                action_variables=state_space_info.get('action_variables'),
                connections=connections,
                model_name=model_name,
                model_annotation=model_annotation,
                ontology_mapping=ontology_mapping
            )
            
            # Validate if strict validation enabled
            if self.strict_validation:
                validation_result = self._validate_pomdp_structure(pomdp_space)
                if not validation_result['valid']:
                    self.logger.warning(f"POMDP validation warnings: {validation_result['warnings']}")
            
            return pomdp_space
            
        except Exception as e:
            self.logger.error(f"Failed to extract POMDP state space: {e}")
            return None
    
    def extract_from_file(self, file_path: Union[str, Path]) -> Optional[POMDPStateSpace]:
        """
        Extract POMDP state space from GNN file.
        
        Args:
            file_path: Path to GNN file
            
        Returns:
            POMDPStateSpace object or None if extraction fails
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                self.logger.error(f"File not found: {file_path}")
                return None
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            return self.extract_from_gnn_content(content)
            
        except Exception as e:
            self.logger.error(f"Failed to read file {file_path}: {e}")
            return None
    
    def _parse_sections(self, content: str) -> Dict[str, str]:
        """Parse GNN content into sections."""
        sections = {}
        current_section = None
        current_content = []
        
        for line in content.split('\n'):
            line = line.strip()
            
            # Check for section header
            section_match = self.SECTION_PATTERN.match(line)
            if section_match:
                # Save previous section
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                
                # Start new section
                current_section = section_match.group(1).strip()
                current_content = []
            else:
                # Add line to current section
                if current_section and line:
                    current_content.append(line)
        
        # Save final section
        if current_section:
            sections[current_section] = '\n'.join(current_content)
            
        return sections
    
    def _extract_model_name(self, sections: Dict[str, str]) -> Optional[str]:
        """Extract model name from sections."""
        return sections.get('ModelName', '').strip() or None
    
    def _extract_model_annotation(self, sections: Dict[str, str]) -> Optional[str]:
        """Extract model annotation from sections."""
        return sections.get('ModelAnnotation', '').strip() or None
    
    def _parse_state_space_block(self, content: str) -> Dict[str, Any]:
        """Parse StateSpaceBlock section."""
        variables = {
            'state_variables': [],
            'observation_variables': [],
            'action_variables': []
        }
        
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            match = self.VARIABLE_PATTERN.match(line)
            if match:
                var_name = match.group(1)
                dimensions_str = match.group(2)
                var_type = match.group(3) or 'float'
                comment = match.group(4)
                
                # Parse dimensions
                dimensions = []
                for dim in dimensions_str.split(','):
                    dim = dim.strip()
                    if '=' not in dim:  # Skip type specifications
                        try:
                            if dim == 'π':  # Special handling for π
                                dimensions.append('π')
                            else:
                                dimensions.append(int(dim))
                        except ValueError:
                            dimensions.append(dim)  # Keep as string if not integer
                
                var_info = {
                    'name': var_name,
                    'dimensions': dimensions,
                    'type': var_type,
                    'comment': comment
                }
                
                # Categorize variables
                if var_name.lower() in ['s', 's_prime'] or 'state' in (comment or '').lower():
                    variables['state_variables'].append(var_info)
                elif var_name.lower() in ['o'] or 'observation' in (comment or '').lower():
                    variables['observation_variables'].append(var_info)
                elif var_name.lower() in ['u', 'π'] or 'action' in (comment or '').lower() or 'policy' in (comment or '').lower():
                    variables['action_variables'].append(var_info)
                else:
                    # Default categorization based on typical Active Inference naming
                    if var_name in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
                        # These are matrix/vector parameters, not state space variables
                        pass
                    else:
                        variables['state_variables'].append(var_info)
        
        return variables
    
    def _extract_dimensions(self, state_space_info: Dict[str, Any]) -> Tuple[int, int, int]:
        """Extract core dimensions from state space information."""
        num_states = 3  # Default
        num_observations = 3  # Default  
        num_actions = 3  # Default
        
        # Try to extract from state variables
        for var in state_space_info.get('state_variables', []):
            if var['name'].lower() == 's':
                if len(var['dimensions']) > 0 and isinstance(var['dimensions'][0], int):
                    num_states = var['dimensions'][0]
        
        # Try to extract from observation variables
        for var in state_space_info.get('observation_variables', []):
            if var['name'].lower() == 'o':
                if len(var['dimensions']) > 0 and isinstance(var['dimensions'][0], int):
                    num_observations = var['dimensions'][0]
        
        # Try to extract from action variables
        for var in state_space_info.get('action_variables', []):
            if var['name'].lower() in ['u', 'π']:
                if len(var['dimensions']) > 0 and isinstance(var['dimensions'][0], int):
                    num_actions = var['dimensions'][0]
        
        return num_states, num_observations, num_actions
    
    def _parse_initial_parameterization(self, content: str) -> Dict[str, Any]:
        """Parse InitialParameterization section."""
        params = {}
        
        # Split content into lines and process each parameter block
        lines = content.split('\n')
        current_param = None
        current_value = ""
        in_param_block = False
        
        for line in lines:
            line = line.strip()
            
            # Skip comments
            if line.startswith('#') or not line:
                continue
            
            # Check if this line starts a parameter definition
            if '={' in line and not in_param_block:
                # Start of parameter block
                param_name = line.split('={')[0].strip()
                current_param = param_name
                current_value = line.split('={')[1]
                
                # Check if parameter ends on the same line
                if '}' in current_value:
                    # Single-line parameter
                    current_value = current_value.split('}')[0]
                    try:
                        parsed_value = self._parse_parameter_value(current_value)
                        params[current_param] = parsed_value
                    except Exception as e:
                        self.logger.warning(f"Failed to parse parameter {current_param}: {e}")
                    current_param = None
                    current_value = ""
                else:
                    # Multi-line parameter
                    in_param_block = True
                    
            elif in_param_block and current_param:
                # Continue collecting parameter value
                if '}' in line:
                    # End of parameter block
                    current_value += " " + line.split('}')[0]
                    try:
                        parsed_value = self._parse_parameter_value(current_value)
                        params[current_param] = parsed_value
                    except Exception as e:
                        self.logger.warning(f"Failed to parse parameter {current_param}: {e}")
                    in_param_block = False
                    current_param = None
                    current_value = ""
                else:
                    # Add line to current value
                    current_value += " " + line
        
        return params
    
    def _parse_parameter_value(self, value_str: str) -> Union[List, float, int]:
        """Parse parameter value string into appropriate data structure."""
        value_str = value_str.strip()
        
        # Handle the specific format used in the GNN file: (val1, val2, val3)
        # First check if it's a simple tuple
        if value_str.startswith('(') and value_str.endswith(')') and value_str.count('(') == 1:
            # Simple tuple: (0.1, 0.1, 1.0)
            inner = value_str[1:-1].strip()
            values = []
            for item in inner.split(','):
                item = item.strip()
                try:
                    if '.' in item:
                        values.append(float(item))
                    else:
                        values.append(int(item))
                except ValueError:
                    values.append(item)
            return values
        
        # Handle nested structure for matrices using safer parsing
        elif '(' in value_str and ')' in value_str:
            # This is a multi-dimensional structure - use safe parsing
            try:
                return self._parse_nested_structure_safe(value_str)
            except RecursionError:
                # Fallback to eval for complex nested structures
                try:
                    # Convert to Python-evaluable format
                    python_str = value_str.replace('(', '[').replace(')', ']')
                    result = eval(python_str)
                    return result
                except Exception as e:
                    self.logger.warning(f"Failed to parse complex structure, using fallback: {e}")
                    return []
        
        else:
            # Simple comma-separated values
            values = []
            for item in value_str.split(','):
                item = item.strip()
                try:
                    if '.' in item:
                        values.append(float(item))
                    else:
                        values.append(int(item))
                except ValueError:
                    values.append(item)
            return values
    
    def _parse_nested_structure(self, value_str: str) -> List:
        """Parse nested parameter structure (for matrices)."""
        result = []
        current = ''
        paren_depth = 0
        
        # Clean up the value string
        value_str = value_str.strip()
        
        for char in value_str:
            if char == '(':
                paren_depth += 1
                if paren_depth > 1:  # Don't include outermost parentheses
                    current += char
            elif char == ')':
                paren_depth -= 1
                if paren_depth >= 1:  # Don't include outermost parentheses
                    current += char
                elif paren_depth == 0:
                    # End of a group
                    if current.strip():
                        # Parse the current group as a nested structure or simple values
                        if '(' in current:
                            parsed = self._parse_nested_structure(f"({current})")
                        else:
                            # Simple comma-separated values
                            values = []
                            for item in current.split(','):
                                item = item.strip()
                                try:
                                    if '.' in item:
                                        values.append(float(item))
                                    else:
                                        values.append(int(item))
                                except ValueError:
                                    values.append(item)
                            parsed = values
                        result.append(parsed)
                        current = ''
            elif char == ',' and paren_depth <= 1:
                # Top-level separator (or within first level)
                if current.strip() and paren_depth == 1:
                    # We're inside the outer parentheses, this is an element separator
                    if '(' in current and ')' in current:
                        # This is a complete sub-element
                        parsed = self._parse_parameter_value(current.strip())
                        result.append(parsed)
                        current = ''
                    else:
                        # Add comma to current (it's part of a sub-element)
                        current += char
                elif paren_depth == 0:
                    # Top-level separator
                    if current.strip():
                        parsed = self._parse_parameter_value(current)
                        result.append(parsed)
                        current = ''
                else:
                    current += char
            else:
                if paren_depth >= 1:  # Only add characters when we're inside parentheses
                    current += char
        
        # Handle final group
        if current.strip():
            if '(' in current:
                parsed = self._parse_nested_structure(f"({current})")
            else:
                values = []
                for item in current.split(','):
                    item = item.strip()
                    try:
                        if '.' in item:
                            values.append(float(item))
                        else:
                            values.append(int(item))
                    except ValueError:
                        values.append(item)
                parsed = values
            result.append(parsed)
            
        return result
    
    def _parse_nested_structure_safe(self, value_str: str, max_depth: int = 10) -> List:
        """Parse nested parameter structure safely with recursion limit."""
        if max_depth <= 0:
            raise RecursionError("Maximum parsing depth exceeded")
            
        # Remove outer curly braces if present
        value_str = value_str.strip()
        if value_str.startswith('{') and value_str.endswith('}'):
            value_str = value_str[1:-1].strip()
        
        # For B matrix format like: ((1.0,0.0,0.0), (0.0,1.0,0.0), (0.0,0.0,1.0))
        result = []
        current = ''
        paren_depth = 0
        i = 0
        
        while i < len(value_str):
            char = value_str[i]
            
            if char == '(':
                paren_depth += 1
                if paren_depth == 1:
                    # Start of a new group - don't include the opening parenthesis
                    current = ''
                else:
                    current += char
            elif char == ')':
                paren_depth -= 1
                if paren_depth == 0:
                    # End of a complete group
                    if current.strip():
                        # Parse simple numeric tuple
                        values = []
                        for item in current.split(','):
                            item = item.strip()
                            if item:
                                try:
                                    if '.' in item:
                                        values.append(float(item))
                                    else:
                                        values.append(int(item))
                                except ValueError:
                                    values.append(item)
                        result.append(values)
                        current = ''
                else:
                    current += char
            elif char == ',' and paren_depth == 0:
                # Top-level separator between groups - ignore
                pass
            else:
                if paren_depth > 0:
                    current += char
            
            i += 1
        
        return result
    
    def _parse_connections(self, content: str) -> List[Tuple[str, str, str]]:
        """Parse Connections section."""
        connections = []
        
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            match = self.CONNECTION_PATTERN.match(line)
            if match:
                source = match.group(1).strip()
                relation = match.group(2).strip()
                target = match.group(3).strip()
                connections.append((source, relation, target))
        
        return connections
    
    def _parse_ontology_annotations(self, content: str) -> Dict[str, str]:
        """Parse ActInfOntologyAnnotation section."""
        mapping = {}
        
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            if '=' in line:
                parts = line.split('=', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    mapping[key] = value
        
        return mapping
    
    def _validate_pomdp_structure(self, pomdp_space: POMDPStateSpace) -> Dict[str, Any]:
        """Validate POMDP structure for consistency."""
        warnings = []
        
        # Check dimension consistency
        try:
            if pomdp_space.A_matrix and isinstance(pomdp_space.A_matrix, list):
                if len(pomdp_space.A_matrix) > 0 and isinstance(pomdp_space.A_matrix[0], list):
                    expected_a_dims = (pomdp_space.num_observations, pomdp_space.num_states)
                    actual_a_dims = (len(pomdp_space.A_matrix), len(pomdp_space.A_matrix[0]))
                    if expected_a_dims != actual_a_dims:
                        warnings.append(f"A matrix dimensions {actual_a_dims} don't match expected {expected_a_dims}")
        except (TypeError, IndexError) as e:
            warnings.append(f"A matrix has invalid structure: {e}")
        
        try:
            if pomdp_space.B_matrix and isinstance(pomdp_space.B_matrix, list):
                if (len(pomdp_space.B_matrix) > 0 and 
                    isinstance(pomdp_space.B_matrix[0], list) and 
                    len(pomdp_space.B_matrix[0]) > 0 and
                    isinstance(pomdp_space.B_matrix[0][0], list)):
                    expected_b_dims = (pomdp_space.num_states, pomdp_space.num_states, pomdp_space.num_actions)
                    actual_b_dims = (
                        len(pomdp_space.B_matrix[0]),
                        len(pomdp_space.B_matrix[0][0]),
                        len(pomdp_space.B_matrix)
                    )
                    if expected_b_dims != actual_b_dims:
                        warnings.append(f"B matrix dimensions {actual_b_dims} don't match expected {expected_b_dims}")
        except (TypeError, IndexError) as e:
            warnings.append(f"B matrix has invalid structure: {e}")
        
        try:
            if pomdp_space.C_vector and isinstance(pomdp_space.C_vector, list):
                if len(pomdp_space.C_vector) != pomdp_space.num_observations:
                    warnings.append(f"C vector length {len(pomdp_space.C_vector)} doesn't match num_observations {pomdp_space.num_observations}")
        except (TypeError) as e:
            warnings.append(f"C vector has invalid structure: {e}")
        
        try:
            if pomdp_space.D_vector and isinstance(pomdp_space.D_vector, list):
                if len(pomdp_space.D_vector) != pomdp_space.num_states:
                    warnings.append(f"D vector length {len(pomdp_space.D_vector)} doesn't match num_states {pomdp_space.num_states}")
        except (TypeError) as e:
            warnings.append(f"D vector has invalid structure: {e}")
        
        return {
            'valid': len(warnings) == 0,
            'warnings': warnings
        }


def extract_pomdp_from_file(file_path: Union[str, Path], 
                           strict_validation: bool = True) -> Optional[POMDPStateSpace]:
    """
    Convenience function to extract POMDP state space from a GNN file.
    
    Args:
        file_path: Path to GNN file
        strict_validation: Enable strict validation
        
    Returns:
        POMDPStateSpace object or None if extraction fails
    """
    extractor = POMDPExtractor(strict_validation=strict_validation)
    return extractor.extract_from_file(file_path)


def extract_pomdp_from_content(content: str, 
                              strict_validation: bool = True) -> Optional[POMDPStateSpace]:
    """
    Convenience function to extract POMDP state space from GNN content.
    
    Args:
        content: GNN file content
        strict_validation: Enable strict validation
        
    Returns:
        POMDPStateSpace object or None if extraction fails
    """
    extractor = POMDPExtractor(strict_validation=strict_validation)
    return extractor.extract_from_gnn_content(content)
