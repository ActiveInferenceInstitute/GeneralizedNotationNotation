"""
Functional Parser for GNN - Haskell

This module provides parsing capabilities for Haskell functional programming
specifications of GNN models.

Author: @docxology
Date: 2025-01-11
License: MIT
"""

import re
from typing import List, Dict, Any, Optional
from pathlib import Path

from .common import (
    BaseGNNParser, ParseResult, GNNInternalRepresentation,
    Variable, Connection, Parameter, Equation, TimeSpecification,
    VariableType, DataType, ConnectionType, ParseError
)

class HaskellGNNParser(BaseGNNParser):
    """Parser for Haskell functional specifications."""
    
    def __init__(self):
        super().__init__()
        self.module_pattern = re.compile(r'module\s+([\w.]+)\s*(?:\([^)]*\))?\s*where', re.IGNORECASE)
        self.data_pattern = re.compile(r'data\s+(\w+).*?=\s*(.+?)(?=\ndata|\n\w+\s*::|\Z)', re.DOTALL)
        self.type_pattern = re.compile(r'(\w+)\s*::\s*(.+?)(?=\n\w+\s*::|\n\w+\s*=|\Z)', re.DOTALL)
        self.function_pattern = re.compile(r'(\w+)\s+(.+?)\s*=\s*(.+?)(?=\n\w+\s|$)', re.MULTILINE | re.DOTALL)
        
    def get_supported_extensions(self) -> List[str]:
        return ['.hs', '.lhs']
    
    def parse_file(self, file_path: str) -> ParseResult:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.parse_string(content)
        except Exception as e:
            result = ParseResult(model=self.create_empty_model())
            result.add_error(f"Failed to read Haskell file: {e}")
            return result
    
    def parse_string(self, content: str) -> ParseResult:
        # First try to extract embedded JSON model data
        embedded_data = self._extract_embedded_json_data(content)
        if embedded_data:
            result = ParseResult(model=self.create_empty_model())
            return self._parse_from_embedded_data(embedded_data, result)
        
        result = ParseResult(model=self.create_empty_model())
        
        try:
            # Parse module name
            module_match = self.module_pattern.search(content)
            if module_match:
                result.model.model_name = module_match.group(1).replace('.', '_')
            else:
                result.model.model_name = "HaskellModel"
            
            # Parse data types
            for match in self.data_pattern.finditer(content):
                data_name = match.group(1)
                data_def = match.group(2).strip()
                
                variable = Variable(
                    name=data_name,
                    var_type=self._infer_variable_type(data_name),
                    dimensions=self._parse_haskell_dimensions(data_def),
                    data_type=self._infer_haskell_data_type(data_def),
                    description=f"Haskell data type: {data_def[:50]}..."
                )
                result.model.variables.append(variable)
            
            # Parse type signatures and functions
            type_sigs = {}
            for match in self.type_pattern.finditer(content):
                func_name = match.group(1)
                func_type = match.group(2).strip()
                type_sigs[func_name] = func_type
            
            # Parse function definitions
            for match in self.function_pattern.finditer(content):
                func_name = match.group(1)
                func_args = match.group(2)
                func_body = match.group(3).strip()
                
                func_type = type_sigs.get(func_name, "")
                
                if self._is_haskell_parameter(func_type, func_body):
                    parameter = Parameter(
                        name=func_name,
                        value=self._extract_haskell_value(func_body),
                        type_hint=func_type,
                        description=f"Haskell definition: {func_type}"
                    )
                    result.model.parameters.append(parameter)
                else:
                    equation = Equation(
                        label=func_name,
                        content=f"{func_name} {func_args} = {func_body}",
                        format="haskell",
                        description=f"Haskell function with type {func_type}"
                    )
                    result.model.equations.append(equation)
            
            result.model.annotation = "Parsed from Haskell functional specification"
            
        except Exception as e:
            result.add_error(f"Parsing error: {e}")
        
        return result
    
    def _extract_embedded_json_data(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract embedded JSON model data from Haskell comments."""
        import json
        # Look for JSON data in Haskell comments
        patterns = [
            r'--\s*MODEL_DATA:\s*(\{.+\})',  # -- MODEL_DATA: {...}
            r'\{-\s*MODEL_DATA:\s*(\{.+?\})\s*-\}',  # {- MODEL_DATA: {...} -}
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
        return None
    
    def _parse_from_embedded_data(self, embedded_data: Dict[str, Any], result: ParseResult) -> ParseResult:
        """Parse model from embedded JSON data."""
        try:
            from .common import Variable, Connection, Parameter, VariableType, DataType, ConnectionType
            
            # Create model from embedded data
            model = GNNInternalRepresentation(
                model_name=embedded_data.get('model_name', 'Unknown Model'),
                annotation=embedded_data.get('annotation', ''),
            )
            
            # Parse variables
            for var_data in embedded_data.get('variables', []):
                var = Variable(
                    name=var_data['name'],
                    var_type=VariableType(var_data['var_type']),
                    data_type=DataType(var_data['data_type']),
                    dimensions=var_data.get('dimensions', [])
                )
                model.variables.append(var)
            
            # Parse connections
            for conn_data in embedded_data.get('connections', []):
                conn = Connection(
                    source_variables=conn_data['source_variables'],
                    target_variables=conn_data['target_variables'],
                    connection_type=ConnectionType(conn_data['connection_type'])
                )
                model.connections.append(conn)
            
            # Parse parameters
            for param_data in embedded_data.get('parameters', []):
                param = Parameter(
                    name=param_data['name'],
                    value=param_data['value'],
                    type_hint=param_data.get('param_type', 'constant')
                )
                model.parameters.append(param)
            
            # Set time specification if present
            if embedded_data.get('time_specification'):
                time_data = embedded_data['time_specification']
                from .common import TimeSpecification
                model.time_specification = TimeSpecification(
                    time_type=time_data.get('time_type', 'dynamic'),
                    discretization=time_data.get('discretization'),
                    horizon=time_data.get('horizon'),
                    step_size=time_data.get('step_size')
                )
            
            # Set ontology mappings if present
            if embedded_data.get('ontology_mappings'):
                from .common import OntologyMapping
                for mapping_data in embedded_data['ontology_mappings']:
                    mapping = OntologyMapping(
                        variable_name=mapping_data['variable_name'],
                        ontology_term=mapping_data['ontology_term'],
                        description=mapping_data.get('description')
                    )
                    model.ontology_mappings.append(mapping)
            
            result.model = model
            result.success = True
            return result
            
        except Exception as e:
            result.add_error(f"Failed to parse embedded data: {e}")
            return result
    
    def _parse_haskell_dimensions(self, data_def: str) -> List[int]:
        """Parse dimensions from Haskell data definition."""
        constructors = data_def.split('|')
        return [len(constructors)]
    
    def _infer_haskell_data_type(self, data_def: str) -> DataType:
        """Infer data type from Haskell data definition."""
        if 'Int' in data_def or 'Integer' in data_def:
            return DataType.INTEGER
        elif 'Float' in data_def or 'Double' in data_def:
            return DataType.FLOAT
        elif 'Bool' in data_def:
            return DataType.BINARY
        else:
            return DataType.CATEGORICAL
    
    def _is_haskell_parameter(self, func_type: str, func_body: str) -> bool:
        """Check if function represents a parameter."""
        return (func_body.replace(' ', '').isdigit() or
                func_body in ['True', 'False'] or
                func_type in ['Int', 'Integer', 'Float', 'Double', 'Bool'])
    
    def _extract_haskell_value(self, func_body: str) -> Any:
        """Extract value from Haskell function body."""
        if func_body.isdigit():
            return int(func_body)
        elif func_body in ['True', 'False']:
            return func_body == 'True'
        else:
            try:
                return float(func_body)
            except:
                return func_body
    
    def _infer_variable_type(self, name: str) -> VariableType:
        """Infer variable type from name."""
        name_lower = name.lower()
        if 'state' in name_lower:
            return VariableType.HIDDEN_STATE
        elif 'observation' in name_lower:
            return VariableType.OBSERVATION
        else:
            return VariableType.HIDDEN_STATE


# Compatibility alias
HaskellParser = HaskellGNNParser 