"""
Scala GNN Parser

This module provides parsing for Scala-based categorical GNN specifications.
It extracts Active Inference model information from Scala code using 
category theory constructs.

Author: @docxology
Date: 2025-01-11
License: MIT
"""

import re
import ast
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import logging

from .common import (
    BaseGNNParser, ParseResult, GNNInternalRepresentation, ParseError,
    Variable, Connection, Parameter, Equation, TimeSpecification, OntologyMapping,
    VariableType, DataType, ConnectionType, 
    normalize_variable_name, parse_dimensions, infer_variable_type, parse_connection_operator
)

logger = logging.getLogger(__name__)

class ScalaGNNParser(BaseGNNParser):
    """
    Parser for Scala categorical GNN specifications.
    
    This parser extracts Active Inference model information from Scala code
    that uses category theory constructs like functors, natural transformations,
    and monads.
    """
    
    def __init__(self):
        super().__init__()
        self.variable_patterns = {
            'case_class': re.compile(r'case class\s+(\w+)\s*\([^)]*\)'),
            'val_definition': re.compile(r'val\s+(\w+)\s*:\s*([^=]+)=?'),
            'def_function': re.compile(r'def\s+(\w+)\s*\([^)]*\)\s*:\s*([^=]+)=?'),
            'type_alias': re.compile(r'type\s+(\w+)\s*=\s*(.+)')
        }
        
        self.active_inference_patterns = {
            'state_space': re.compile(r'StateSpace\s*\(\s*([^)]+)\)'),
            'observation_space': re.compile(r'ObservationSpace\s*\(\s*([^)]+)\)'),
            'action_space': re.compile(r'ActionSpace\s*\(\s*([^)]+)\)'),
            'likelihood': re.compile(r'LikelihoodMapping\s*\(\s*([^)]+)\)'),
            'transition': re.compile(r'TransitionMapping\s*\(\s*([^)]+)\)'),
            'preferences': re.compile(r'PreferenceMapping\s*\(\s*([^)]+)\)'),
            'priors': re.compile(r'PriorMapping\s*\(\s*([^)]+)\)')
        }
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions."""
        return ['.scala']
    
    def parse_file(self, file_path: str) -> ParseResult:
        """Parse a Scala GNN file."""
        self.current_file = file_path
        file_path = Path(file_path)
        
        try:
            content = file_path.read_text(encoding='utf-8')
            return self.parse_string(content)
        except Exception as e:
            raise ParseError(f"Failed to read file {file_path}: {e}")
    
    def parse_string(self, content: str) -> ParseResult:
        """Parse Scala GNN content from string."""
        try:
            # Create result container
            result = ParseResult(model=self.create_empty_model())
            
            # Extract model name from package or object name
            model_name = self._extract_model_name(content)
            result.model.model_name = model_name
            
            # Parse Scala constructs
            self._parse_scala_constructs(content, result.model)
            
            # Extract Active Inference components
            self._parse_active_inference_components(content, result.model)
            
            # Parse categorical structures
            self._parse_categorical_structures(content, result.model)
            
            # Extract equations from comments and definitions
            self._parse_equations_from_scala(content, result.model)
            
            return result
            
        except Exception as e:
            result = ParseResult(
                model=self.create_empty_model("Parse Error"),
                success=False
            )
            result.add_error(str(e))
            return result
    
    def _extract_model_name(self, content: str) -> str:
        """Extract model name from package, object, or class definitions."""
        # Try object name first
        object_match = re.search(r'object\s+(\w+)', content)
        if object_match:
            return object_match.group(1)
        
        # Try class name
        class_match = re.search(r'class\s+(\w+)', content)
        if class_match:
            return class_match.group(1)
        
        # Try package name
        package_match = re.search(r'package\s+([\w.]+)', content)
        if package_match:
            return package_match.group(1).split('.')[-1]
        
        return "Scala GNN Model"
    
    def _parse_scala_constructs(self, content: str, model: GNNInternalRepresentation):
        """Parse general Scala constructs for variables and types."""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            self.current_line = i + 1
            line = line.strip()
            
            if not line or line.startswith('//') or line.startswith('/*'):
                continue
            
            # Parse case classes as potential variables
            if line.startswith('case class'):
                variable = self._parse_case_class(line)
                if variable:
                    model.variables.append(variable)
            
            # Parse val definitions
            elif line.startswith('val '):
                variable = self._parse_val_definition(line)
                if variable:
                    model.variables.append(variable)
            
            # Parse type aliases
            elif line.startswith('type '):
                self._parse_type_alias(line, model)
    
    def _parse_case_class(self, line: str) -> Optional[Variable]:
        """Parse a case class definition as a variable."""
        try:
            # Extract case class name and parameters
            match = re.match(r'case class\s+(\w+)\s*\(([^)]*)\)', line)
            if not match:
                return None
            
            class_name = match.group(1)
            params_str = match.group(2)
            
            # Determine variable type from name patterns
            var_type = infer_variable_type(class_name)
            
            # Parse dimensions from parameters
            dimensions = self._extract_dimensions_from_params(params_str)
            
            # Determine data type
            data_type = DataType.CATEGORICAL  # Default for Scala categorical specs
            
            return Variable(
                name=normalize_variable_name(class_name),
                var_type=var_type,
                dimensions=dimensions,
                data_type=data_type,
                description=f"Scala case class: {line}"
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse case class '{line}': {e}")
            return None
    
    def _parse_val_definition(self, line: str) -> Optional[Variable]:
        """Parse a val definition as a variable."""
        try:
            # Extract val name and type
            match = re.match(r'val\s+(\w+)\s*:\s*([^=]+)', line)
            if not match:
                return None
            
            val_name = match.group(1)
            type_str = match.group(2).strip()
            
            # Skip non-AI related vals
            if not self._is_ai_related_name(val_name):
                return None
            
            # Determine variable type
            var_type = infer_variable_type(val_name)
            
            # Extract dimensions from type
            dimensions = self._extract_dimensions_from_type(type_str)
            
            # Determine data type from Scala type
            data_type = self._scala_type_to_data_type(type_str)
            
            return Variable(
                name=normalize_variable_name(val_name),
                var_type=var_type,
                dimensions=dimensions,
                data_type=data_type,
                description=f"Scala val: {line}"
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse val definition '{line}': {e}")
            return None
    
    def _parse_type_alias(self, line: str, model: GNNInternalRepresentation):
        """Parse a type alias definition."""
        try:
            match = re.match(r'type\s+(\w+)\s*=\s*(.+)', line)
            if match:
                type_name = match.group(1)
                type_def = match.group(2).strip()
                
                # Store as extension
                model.extensions.setdefault('type_aliases', {})[type_name] = type_def
                
        except Exception as e:
            logger.warning(f"Failed to parse type alias '{line}': {e}")
    
    def _parse_active_inference_components(self, content: str, model: GNNInternalRepresentation):
        """Parse Active Inference specific components."""
        for component, pattern in self.active_inference_patterns.items():
            matches = pattern.finditer(content)
            
            for match in matches:
                params_str = match.group(1)
                self._process_ai_component(component, params_str, model)
    
    def _process_ai_component(self, component_type: str, params_str: str, 
                            model: GNNInternalRepresentation):
        """Process a specific Active Inference component."""
        try:
            if component_type == 'state_space':
                self._parse_state_space_scala(params_str, model)
            elif component_type == 'observation_space':
                self._parse_observation_space_scala(params_str, model)
            elif component_type == 'action_space':
                self._parse_action_space_scala(params_str, model)
            elif component_type in ['likelihood', 'transition', 'preferences', 'priors']:
                self._parse_ai_mapping(component_type, params_str, model)
                
        except Exception as e:
            logger.warning(f"Failed to process AI component '{component_type}': {e}")
    
    def _parse_state_space_scala(self, params_str: str, model: GNNInternalRepresentation):
        """Parse StateSpace definition from Scala."""
        # Extract factors and dimensions
        factors_match = re.search(r'factors\s*=\s*List\(([^)]+)\)', params_str)
        if factors_match:
            factors_str = factors_match.group(1)
            factor_names = [name.strip().strip('"\'') for name in factors_str.split(',')]
            
            for factor_name in factor_names:
                if factor_name:
                    variable = Variable(
                        name=normalize_variable_name(factor_name),
                        var_type=VariableType.HIDDEN_STATE,
                        dimensions=[2],  # Default dimension
                        data_type=DataType.CATEGORICAL
                    )
                    model.variables.append(variable)
    
    def _parse_observation_space_scala(self, params_str: str, model: GNNInternalRepresentation):
        """Parse ObservationSpace definition from Scala."""
        # Extract modalities
        modalities_match = re.search(r'modalities\s*=\s*List\(([^)]+)\)', params_str)
        if modalities_match:
            modalities_str = modalities_match.group(1)
            modality_names = [name.strip().strip('"\'') for name in modalities_str.split(',')]
            
            for modality_name in modality_names:
                if modality_name:
                    variable = Variable(
                        name=normalize_variable_name(modality_name),
                        var_type=VariableType.OBSERVATION,
                        dimensions=[2],  # Default dimension
                        data_type=DataType.CATEGORICAL
                    )
                    model.variables.append(variable)
    
    def _parse_action_space_scala(self, params_str: str, model: GNNInternalRepresentation):
        """Parse ActionSpace definition from Scala."""
        # Extract controls
        controls_match = re.search(r'controls\s*=\s*List\(([^)]+)\)', params_str)
        if controls_match:
            controls_str = controls_match.group(1)
            control_names = [name.strip().strip('"\'') for name in controls_str.split(',')]
            
            for control_name in control_names:
                if control_name:
                    variable = Variable(
                        name=normalize_variable_name(control_name),
                        var_type=VariableType.ACTION,
                        dimensions=[2],  # Default dimension
                        data_type=DataType.CATEGORICAL
                    )
                    model.variables.append(variable)
    
    def _parse_ai_mapping(self, mapping_type: str, params_str: str, 
                         model: GNNInternalRepresentation):
        """Parse Active Inference mapping (A, B, C, D matrices)."""
        # Store mapping information as parameters
        parameter = Parameter(
            name=f"{mapping_type}_mapping",
            value=params_str,
            description=f"Scala {mapping_type} mapping definition"
        )
        model.parameters.append(parameter)
    
    def _parse_categorical_structures(self, content: str, model: GNNInternalRepresentation):
        """Parse categorical theory structures from Scala code."""
        # Look for functor definitions
        functor_pattern = re.compile(r'implicit\s+def\s+(\w+Functor).*?Functor\[([^\]]+)\]')
        functors = functor_pattern.findall(content)
        
        for functor_name, functor_type in functors:
            model.extensions.setdefault('functors', {})[functor_name] = functor_type
        
        # Look for natural transformations
        nat_trans_pattern = re.compile(r'def\s+(\w+)\s*:\s*([^=]+)~>\s*([^=]+)')
        nat_trans = nat_trans_pattern.findall(content)
        
        for trans_name, from_type, to_type in nat_trans:
            model.extensions.setdefault('natural_transformations', {})[trans_name] = {
                'from': from_type.strip(),
                'to': to_type.strip()
            }
        
        # Look for monads
        monad_pattern = re.compile(r'implicit\s+def\s+(\w+Monad).*?Monad\[([^\]]+)\]')
        monads = monad_pattern.findall(content)
        
        for monad_name, monad_type in monads:
            model.extensions.setdefault('monads', {})[monad_name] = monad_type
    
    def _parse_equations_from_scala(self, content: str, model: GNNInternalRepresentation):
        """Parse mathematical equations from Scala comments and definitions."""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Look for equations in comments
            if '//' in line:
                comment = line.split('//', 1)[1].strip()
                if self._contains_equation(comment):
                    equation = Equation(
                        label=None,
                        content=comment,
                        format="ascii",
                        description=f"From Scala comment at line {i+1}"
                    )
                    model.equations.append(equation)
            
            # Look for equations in multiline comments
            elif '/*' in line and '*/' in line:
                comment = line.split('/*', 1)[1].split('*/', 1)[0].strip()
                if self._contains_equation(comment):
                    equation = Equation(
                        label=None,
                        content=comment,
                        format="ascii",
                        description=f"From Scala comment at line {i+1}"
                    )
                    model.equations.append(equation)
    
    def _contains_equation(self, text: str) -> bool:
        """Check if text contains mathematical equations."""
        equation_indicators = ['=', '∝', '∀', '∃', '∈', '→', '⟹', '∑', '∏', '∫']
        return any(indicator in text for indicator in equation_indicators)
    
    def _extract_dimensions_from_params(self, params_str: str) -> List[int]:
        """Extract dimensions from case class parameters."""
        # Look for dimension-related parameters
        dim_patterns = [r'dim\s*:\s*Int', r'dimensions\s*:\s*List\[Int\]', r'size\s*:\s*Int']
        
        for pattern in dim_patterns:
            if re.search(pattern, params_str):
                # Try to extract numeric values
                numbers = re.findall(r'\b\d+\b', params_str)
                if numbers:
                    return [int(n) for n in numbers]
        
        return [1]  # Default dimension
    
    def _extract_dimensions_from_type(self, type_str: str) -> List[int]:
        """Extract dimensions from Scala type information."""
        # Look for Matrix[Fin m, Fin n] patterns
        matrix_match = re.search(r'Matrix\s*\(\s*Fin\s+(\d+)\s*,\s*Fin\s+(\d+)\s*\)', type_str)
        if matrix_match:
            return [int(matrix_match.group(1)), int(matrix_match.group(2))]
        
        # Look for Vector[Fin n] patterns
        vector_match = re.search(r'Vector\s*\[\s*Fin\s+(\d+)\s*\]', type_str)
        if vector_match:
            return [int(vector_match.group(1))]
        
        # Look for numeric literals
        numbers = re.findall(r'\b\d+\b', type_str)
        if numbers:
            return [int(n) for n in numbers[:2]]  # Take up to 2 dimensions
        
        return [1]  # Default
    
    def _scala_type_to_data_type(self, type_str: str) -> DataType:
        """Convert Scala type to GNN DataType."""
        type_str = type_str.lower()
        
        if 'double' in type_str or 'float' in type_str or 'real' in type_str:
            return DataType.FLOAT
        elif 'int' in type_str or 'long' in type_str:
            return DataType.INTEGER
        elif 'boolean' in type_str or 'bool' in type_str:
            return DataType.BINARY
        elif 'string' in type_str:
            return DataType.CATEGORICAL
        elif 'complex' in type_str:
            return DataType.COMPLEX
        else:
            return DataType.CATEGORICAL  # Default for categorical specs
    
    def _is_ai_related_name(self, name: str) -> bool:
        """Check if a name is related to Active Inference."""
        ai_keywords = [
            's_f', 'o_m', 'u_c', 'pi_c', 'a_m', 'b_f', 'c_m', 'd_f',
            'state', 'observation', 'action', 'policy', 'belief',
            'likelihood', 'transition', 'preference', 'prior',
            'efe', 'vfe', 'free_energy'
        ]
        
        name_lower = name.lower()
        return any(keyword in name_lower for keyword in ai_keywords)

# Export the parser class
__all__ = ['ScalaGNNParser'] 