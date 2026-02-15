"""
Isabelle/HOL Parser for GNN

This module provides parsing capabilities for Isabelle/HOL theorem proving
language specifications of GNN models.

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

class IsabelleParser(BaseGNNParser):
    """Parser for Isabelle/HOL theorem proving specifications."""
    
    def __init__(self):
        super().__init__()
        self.theory_pattern = re.compile(r'theory\s+(\w+)\s+imports\s+([\w\s]+)\s+begin', re.IGNORECASE)
        self.datatype_pattern = re.compile(r'datatype\s+(\w+)\s*=\s*(.+)', re.IGNORECASE)
        self.definition_pattern = re.compile(r'definition\s+(\w+)\s*::\s*"([^"]+)"\s+where\s+"([^"]+)"', re.IGNORECASE)
        self.lemma_pattern = re.compile(r'lemma\s+(\w+)\s*:\s*"([^"]+)"', re.IGNORECASE)
        self.theorem_pattern = re.compile(r'theorem\s+(\w+)\s*:\s*"([^"]+)"', re.IGNORECASE)
        
    def get_supported_extensions(self) -> List[str]:
        """Get file extensions supported by this parser."""
        return ['.thy']
    
    def parse_file(self, file_path: str) -> ParseResult:
        """Parse an Isabelle/HOL file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.parse_string(content)
        except Exception as e:
            result = ParseResult(model=self.create_empty_model())
            result.add_error(f"Failed to read file: {e}")
            return result
    
    def parse_string(self, content: str) -> ParseResult:
        """Parse Isabelle/HOL content string."""
        # First try to extract embedded JSON model data
        embedded_data = self._extract_embedded_json_data(content)
        if embedded_data:
            result = ParseResult(model=self.create_empty_model())
            return self._parse_from_embedded_data(embedded_data, result)
        
        result = ParseResult(model=self.create_empty_model())
        
        try:
            # Parse theory name
            theory_match = self.theory_pattern.search(content)
            if theory_match:
                model_name = theory_match.group(1)
                result.model.model_name = model_name
            else:
                result.model.model_name = "IsabelleModel"
            
            # Parse datatypes (variables)
            for match in self.datatype_pattern.finditer(content):
                var_name = match.group(1)
                var_def = match.group(2)
                
                variable = Variable(
                    name=var_name,
                    var_type=self._infer_variable_type(var_name),
                    dimensions=self._parse_datatype_dimensions(var_def),
                    data_type=self._infer_data_type(var_def),
                    description=f"Isabelle datatype: {var_def}"
                )
                result.model.variables.append(variable)
            
            # Parse definitions (parameters/equations)
            for match in self.definition_pattern.finditer(content):
                def_name = match.group(1)
                def_type = match.group(2)
                def_body = match.group(3)
                
                if self._is_parameter_definition(def_body):
                    parameter = Parameter(
                        name=def_name,
                        value=self._extract_parameter_value(def_body),
                        type_hint=def_type,
                        description=f"Isabelle definition: {def_body}"
                    )
                    result.model.parameters.append(parameter)
                else:
                    equation = Equation(
                        label=def_name,
                        content=def_body,
                        format="isabelle",
                        description=f"Type: {def_type}"
                    )
                    result.model.equations.append(equation)
            
            # Parse lemmas and theorems as equations
            for pattern, eq_type in [(self.lemma_pattern, "lemma"), (self.theorem_pattern, "theorem")]:
                for match in pattern.finditer(content):
                    eq_name = match.group(1)
                    eq_content = match.group(2)
                    
                    equation = Equation(
                        label=eq_name,
                        content=eq_content,
                        format="isabelle",
                        description=f"Isabelle {eq_type}"
                    )
                    result.model.equations.append(equation)
            
            # Extract connections from function applications
            connections = self._extract_connections(content)
            result.model.connections.extend(connections)
            
            result.model.annotation = f"Parsed from Isabelle/HOL theory"
            
        except Exception as e:
            result.add_error(f"Parsing error: {e}")
        
        return result
    
    def _extract_embedded_json_data(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract embedded JSON model data from Isabelle comments."""
        import json
        # Look for JSON data in Isabelle comments
        patterns = [
            r'\(\*\s*MODEL_DATA:\s*(\{.+?\})\s*\*\)',  # (* MODEL_DATA: {...} *)
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
    
    def _infer_variable_type(self, name: str) -> VariableType:
        """Infer variable type from name."""
        name_lower = name.lower()
        if 'state' in name_lower or 's_' in name_lower:
            return VariableType.HIDDEN_STATE
        elif 'obs' in name_lower or 'o_' in name_lower:
            return VariableType.OBSERVATION
        elif 'action' in name_lower or 'u_' in name_lower:
            return VariableType.ACTION
        elif 'policy' in name_lower or 'pi_' in name_lower:
            return VariableType.POLICY
        else:
            return VariableType.HIDDEN_STATE
    
    def _parse_datatype_dimensions(self, datatype_def: str) -> List[int]:
        """Parse dimensions from datatype definition."""
        # Simple heuristic for common patterns
        if 'list' in datatype_def.lower():
            return [1]  # Assume 1D list
        elif '*' in datatype_def:
            # Count product types
            return [len(datatype_def.split('*'))]
        else:
            return [1]
    
    def _infer_data_type(self, datatype_def: str) -> DataType:
        """Infer data type from datatype definition."""
        def_lower = datatype_def.lower()
        if 'nat' in def_lower or 'int' in def_lower:
            return DataType.INTEGER
        elif 'real' in def_lower:
            return DataType.FLOAT
        elif 'bool' in def_lower:
            return DataType.BINARY
        else:
            return DataType.CATEGORICAL
    
    def _is_parameter_definition(self, body: str) -> bool:
        """Check if definition body represents a parameter."""
        # Simple heuristics
        body_clean = body.strip()
        return (body_clean.replace(' ', '').replace('.', '').replace(',', '').replace('_', '').isdigit() or
                any(op in body_clean for op in ['=', '+', '-', '*', '/', '^']))
    
    def _extract_parameter_value(self, body: str) -> Any:
        """Extract parameter value from definition body."""
        body_clean = body.strip()
        try:
            # Try to extract numeric values
            if body_clean.replace('.', '').isdigit():
                return float(body_clean) if '.' in body_clean else int(body_clean)
            else:
                return body_clean
        except Exception:
            return body_clean
    
    def _extract_connections(self, content: str) -> List[Connection]:
        """Extract connections from function applications."""
        connections = []
        
        # Look for function application patterns that might indicate connections
        app_pattern = re.compile(r'(\w+)\s+(\w+)', re.IGNORECASE)
        
        for match in app_pattern.finditer(content):
            func_name = match.group(1)
            arg_name = match.group(2)
            
            # Simple heuristic: if both are likely variable names
            if (any(prefix in func_name.lower() for prefix in ['s_', 'o_', 'u_', 'pi_']) and
                any(prefix in arg_name.lower() for prefix in ['s_', 'o_', 'u_', 'pi_'])):
                
                connection = Connection(
                    source_variables=[arg_name],
                    target_variables=[func_name],
                    connection_type=ConnectionType.DIRECTED,
                    description="Inferred from Isabelle function application"
                )
                connections.append(connection)
        
        return connections


# Compatibility alias
IsabelleGNNParser = IsabelleParser 