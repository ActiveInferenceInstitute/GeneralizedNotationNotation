"""
Binary Parser for GNN - Pickle

This module provides parsing capabilities for Python pickle serialization
format specifications of GNN models.

Author: @docxology
Date: 2025-01-11
License: MIT
"""

import pickle
from typing import List, Dict, Any, Optional
from pathlib import Path

from .common import (
    BaseGNNParser, ParseResult, GNNInternalRepresentation,
    Variable, Connection, Parameter, Equation, TimeSpecification,
    VariableType, DataType, ConnectionType, ParseError
)

class PickleGNNParser(BaseGNNParser):
    """Parser for Python pickle binary format."""
    
    def __init__(self):
        super().__init__()
        
    def get_supported_extensions(self) -> List[str]:
        return ['.pkl', '.pickle']
    
    def parse_file(self, file_path: str) -> ParseResult:
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            return self._parse_pickle_data(data)
        except Exception as e:
            result = ParseResult(model=self.create_empty_model())
            result.add_error(f"Failed to read pickle file: {e}")
            return result
    
    def parse_string(self, content: str) -> ParseResult:
        # For pickle, string parsing doesn't make sense - return empty model
        result = ParseResult(model=self.create_empty_model())
        result.add_warning("Pickle parser requires binary file, not string content")
        return result
    
    def _parse_pickle_data(self, data: Any) -> ParseResult:
        """Parse pickled data into GNN representation."""
        result = ParseResult(model=self.create_empty_model())
        result.model.model_name = "PickleModel"
        
        try:
            if isinstance(data, dict):
                self._parse_dict_data(data, result)
            elif isinstance(data, GNNInternalRepresentation):
                # Already a GNN model
                result.model = data
            elif hasattr(data, '__dict__'):
                # Object with attributes
                self._parse_object_data(data, result)
            else:
                # Simple value
                parameter = Parameter(
                    name="pickle_value",
                    value=data,
                    description=f"Pickled {type(data).__name__}"
                )
                result.model.parameters.append(parameter)
            
            result.model.annotation = "Parsed from Python pickle file"
            
        except Exception as e:
            result.add_error(f"Error parsing pickle data: {e}")
        
        return result
    
    def _parse_dict_data(self, data: Dict[str, Any], result: ParseResult):
        """Parse dictionary data."""
        # Look for common GNN structure patterns
        if 'model_name' in data:
            result.model.model_name = str(data['model_name'])
        
        if 'variables' in data:
            self._parse_variables_list(data['variables'], result)
        
        if 'parameters' in data:
            self._parse_parameters_dict(data['parameters'], result)
        
        # Parse other dictionary items as parameters
        for key, value in data.items():
            if key not in ['model_name', 'variables', 'parameters']:
                parameter = Parameter(
                    name=key,
                    value=value,
                    description=f"Pickled parameter: {type(value).__name__}"
                )
                result.model.parameters.append(parameter)
    
    def _parse_object_data(self, obj: Any, result: ParseResult):
        """Parse object data."""
        result.model.model_name = f"{type(obj).__name__}Model"
        
        for attr_name in dir(obj):
            if not attr_name.startswith('_'):
                try:
                    attr_value = getattr(obj, attr_name)
                    if not callable(attr_value):
                        parameter = Parameter(
                            name=attr_name,
                            value=attr_value,
                            description=f"Object attribute: {type(attr_value).__name__}"
                        )
                        result.model.parameters.append(parameter)
                except:
                    pass
    
    def _parse_variables_list(self, variables: List[Any], result: ParseResult):
        """Parse variables from list."""
        for var_data in variables:
            if isinstance(var_data, dict):
                variable = Variable(
                    name=var_data.get('name', f'var_{len(result.model.variables)}'),
                    var_type=self._parse_variable_type(var_data.get('type', 'hidden_state')),
                    dimensions=var_data.get('dimensions', [1]),
                    data_type=self._parse_data_type(var_data.get('data_type', 'categorical')),
                    description=var_data.get('description', 'Pickled variable')
                )
                result.model.variables.append(variable)
    
    def _parse_parameters_dict(self, parameters: Dict[str, Any], result: ParseResult):
        """Parse parameters from dictionary."""
        for param_name, param_value in parameters.items():
            parameter = Parameter(
                name=param_name,
                value=param_value,
                description=f"Pickled parameter: {type(param_value).__name__}"
            )
            result.model.parameters.append(parameter)
    
    def _parse_variable_type(self, type_str: str) -> VariableType:
        """Parse variable type from string."""
        type_mapping = {
            'hidden_state': VariableType.HIDDEN_STATE,
            'observation': VariableType.OBSERVATION,
            'action': VariableType.ACTION,
            'policy': VariableType.POLICY,
            'likelihood_matrix': VariableType.LIKELIHOOD_MATRIX,
            'transition_matrix': VariableType.TRANSITION_MATRIX,
            'preference_vector': VariableType.PREFERENCE_VECTOR,
            'prior_vector': VariableType.PRIOR_VECTOR
        }
        return type_mapping.get(type_str.lower(), VariableType.HIDDEN_STATE)
    
    def _parse_data_type(self, type_str: str) -> DataType:
        """Parse data type from string."""
        type_mapping = {
            'categorical': DataType.CATEGORICAL,
            'continuous': DataType.CONTINUOUS,
            'binary': DataType.BINARY,
            'integer': DataType.INTEGER,
            'float': DataType.FLOAT,
            'complex': DataType.COMPLEX
        }
        return type_mapping.get(type_str.lower(), DataType.CATEGORICAL)


# Compatibility alias
PickleParser = PickleGNNParser 