"""
Grammar Parser for GNN BNF/EBNF Specifications

This module provides parsing capabilities for BNF and EBNF files that specify
GNN models using formal grammar definitions.

Author: @docxology
Date: 2025-01-11
License: MIT
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from .common import (
    BaseGNNParser, ParseResult, GNNInternalRepresentation, ParseError,
    Variable, Connection, Parameter, VariableType, DataType, ConnectionType
)

logger = logging.getLogger(__name__)

class BNFParser(BaseGNNParser):
    """Parser for BNF grammar specifications with embedded data support."""
    
    def __init__(self):
        """Initialize the BNF parser."""
        super().__init__()
        self.rule_pattern = re.compile(r'<([^>]+)>\s*::=\s*([^\n]+)')
        self.terminal_pattern = re.compile(r'"([^"]*)"')
        self.non_terminal_pattern = re.compile(r'<([^>]+)>')
    
    def _extract_embedded_json_data(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract embedded JSON model data from BNF comments."""
        import json
        # Look for JSON data in # MODEL_DATA: {...} comments
        # Use a more robust pattern for multi-line JSON
        pattern = r'#\s*MODEL_DATA:\s*(\{.*\})'
        match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
        if match:
            try:
                json_data = match.group(1)
                return json.loads(json_data)
            except json.JSONDecodeError:
                pass
        return None
    
    def _parse_from_embedded_data(self, embedded_data: Dict[str, Any], result: ParseResult) -> ParseResult:
        """Parse model from embedded JSON data for perfect round-trip fidelity."""
        from .common import Variable, Connection, Parameter, VariableType, DataType, ConnectionType
        
        try:
            result.model.model_name = embedded_data.get('model_name', 'BNFGNNModel')
            result.model.annotation = embedded_data.get('annotation', '')
            
            # Restore variables
            for var_data in embedded_data.get('variables', []):
                var = Variable(
                    name=var_data['name'],
                    var_type=VariableType(var_data.get('var_type', 'hidden_state')),
                    data_type=DataType(var_data.get('data_type', 'categorical')),
                    dimensions=var_data.get('dimensions', [])
                )
                result.model.variables.append(var)
            
            # Restore connections
            for conn_data in embedded_data.get('connections', []):
                conn = Connection(
                    source_variables=conn_data.get('source_variables', []),
                    target_variables=conn_data.get('target_variables', []),
                    connection_type=ConnectionType(conn_data.get('connection_type', 'directed'))
                )
                result.model.connections.append(conn)
            
            # Restore parameters
            for param_data in embedded_data.get('parameters', []):
                param = Parameter(
                    name=param_data['name'],
                    value=param_data['value']
                )
                result.model.parameters.append(param)
            
            # Restore time specification
            if embedded_data.get('time_specification'):
                from .common import TimeSpecification
                time_data = embedded_data['time_specification']
                result.model.time_specification = TimeSpecification(
                    time_type=time_data.get('time_type', 'dynamic'),
                    discretization=time_data.get('discretization'),
                    horizon=time_data.get('horizon'),
                    step_size=time_data.get('step_size')
                )
            
            # Restore ontology mappings
            for mapping_data in embedded_data.get('ontology_mappings', []):
                from .common import OntologyMapping
                mapping = OntologyMapping(
                    variable_name=mapping_data.get('variable_name', ''),
                    ontology_term=mapping_data.get('ontology_term', ''),
                    description=mapping_data.get('description')
                )
                result.model.ontology_mappings.append(mapping)
            
            return result
            
        except Exception as e:
            result.add_error(f"Failed to parse embedded data: {e}")
            return result
        
    def parse_file(self, file_path: str) -> ParseResult:
        """Parse a BNF file containing GNN grammar specifications."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return self.parse_string(content)
            
        except Exception as e:
            logger.error(f"Error parsing BNF file {file_path}: {e}")
            result = ParseResult(
                model=self.create_empty_model("Failed BNF Parse"),
                success=False
            )
            result.add_error(f"Failed to parse BNF file: {e}")
            return result
    
    def parse_string(self, content: str) -> ParseResult:
        """Parse BNF content from string."""
        result = ParseResult(model=self.create_empty_model("Failed BNF Parse"), success=False)
        
        # First, try to extract embedded JSON data for perfect round-trip
        embedded_data = self._extract_embedded_json_data(content)
        if embedded_data:
            result.success = True
            return self._parse_from_embedded_data(embedded_data, result)
        
        # Fallback to standard parsing
        try:
            model = self._parse_bnf_content(content)
            return ParseResult(model=model, success=True)
            
        except Exception as e:
            logger.error(f"Error parsing BNF content: {e}")
            result.add_error(f"Failed to parse BNF content: {e}")
            return result
    
    def _parse_bnf_content(self, content: str) -> GNNInternalRepresentation:
        """Parse the main BNF content."""
        model_name = self._extract_model_name(content)
        
        model = GNNInternalRepresentation(
            model_name=model_name,
            annotation="Parsed from BNF grammar specification"
        )
        
        # Parse grammar rules
        self._parse_grammar_rules(content, model)
        
        return model
    
    def _extract_model_name(self, content: str) -> str:
        """Extract model name from BNF content."""
        # Look for comments with model name
        comment_patterns = [
            r'#\s*Grammar\s+for\s+([^\n]+)',
            r'#\s*([^\n]*Model[^\n]*)',
            r'//\s*([^\n]*Model[^\n]*)'
        ]
        
        for pattern in comment_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Look for a root rule that might indicate the model
        rules = self.rule_pattern.findall(content)
        if rules:
            root_rule = rules[0][0]
            if 'model' in root_rule.lower():
                return root_rule.replace('_', ' ').title()
        
        return "BNFGNNModel"
    
    def _parse_grammar_rules(self, content: str, model: GNNInternalRepresentation):
        """Parse BNF grammar rules to extract GNN components."""
        rules = self.rule_pattern.findall(content)
        
        for non_terminal, production in rules:
            # Treat non-terminals as potential variables
            var_type = self._infer_variable_type_from_rule(non_terminal, production)
            
            if var_type:
                variable = Variable(
                    name=non_terminal,
                    var_type=var_type,
                    dimensions=[],
                    data_type=self._infer_data_type_from_production(production),
                    description=f"Grammar rule: {non_terminal} ::= {production}"
                )
                
                model.variables.append(variable)
            
            # Extract connections from production rules
            self._extract_connections_from_production(non_terminal, production, model)
    
    def _infer_variable_type_from_rule(self, non_terminal: str, production: str) -> Optional[VariableType]:
        """Infer variable type from grammar rule."""
        nt_lower = non_terminal.lower()
        prod_lower = production.lower()
        
        if any(keyword in nt_lower for keyword in ['state', 'hidden']):
            return VariableType.HIDDEN_STATE
        elif any(keyword in nt_lower for keyword in ['observation', 'obs']):
            return VariableType.OBSERVATION
        elif any(keyword in nt_lower for keyword in ['action', 'control']):
            return VariableType.ACTION
        elif any(keyword in nt_lower for keyword in ['policy']):
            return VariableType.POLICY
        elif 'matrix' in nt_lower:
            if 'a' in nt_lower:
                return VariableType.LIKELIHOOD_MATRIX
            elif 'b' in nt_lower:
                return VariableType.TRANSITION_MATRIX
        elif 'vector' in nt_lower:
            if 'c' in nt_lower:
                return VariableType.PREFERENCE_VECTOR
            elif 'd' in nt_lower:
                return VariableType.PRIOR_VECTOR
        
        # Only return type for GNN-relevant rules
        gnn_keywords = ['state', 'observation', 'action', 'policy', 'matrix', 'vector', 'variable']
        if any(keyword in nt_lower for keyword in gnn_keywords):
            return VariableType.HIDDEN_STATE
        
        return None
    
    def _infer_data_type_from_production(self, production: str) -> DataType:
        """Infer data type from production rule."""
        prod_lower = production.lower()
        
        if any(keyword in prod_lower for keyword in ['real', 'float', 'double']):
            return DataType.CONTINUOUS
        elif any(keyword in prod_lower for keyword in ['int', 'integer', 'nat']):
            return DataType.INTEGER
        elif any(keyword in prod_lower for keyword in ['bool', 'boolean']):
            return DataType.BINARY
        elif any(keyword in prod_lower for keyword in ['list', 'array', 'vector']):
            return DataType.CATEGORICAL
        
        return DataType.CONTINUOUS
    
    def _extract_connections_from_production(self, non_terminal: str, production: str, 
                                           model: GNNInternalRepresentation):
        """Extract connections from production rules."""
        # Find other non-terminals referenced in the production
        referenced_nts = self.non_terminal_pattern.findall(production)
        
        for ref_nt in referenced_nts:
            if ref_nt != non_terminal:  # Avoid self-references
                # Create a connection
                connection = Connection(
                    source_variables=[ref_nt],
                    target_variables=[non_terminal],
                    connection_type=ConnectionType.DIRECTED,
                    description=f"Grammar dependency: {ref_nt} -> {non_terminal}"
                )
                
                model.connections.append(connection)
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions."""
        return ['.bnf']


class EBNFParser(BNFParser):
    """Parser for EBNF (Extended BNF) grammar specifications with embedded data support."""
    
    def __init__(self):
        """Initialize the EBNF parser."""
        super().__init__()
        # EBNF uses = instead of ::= 
        self.rule_pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^;]+);?')
        self.optional_pattern = re.compile(r'\[([^\]]+)\]')
        self.repetition_pattern = re.compile(r'\{([^}]+)\}')
        self.grouping_pattern = re.compile(r'\(([^)]+)\)')
    
    def _parse_from_embedded_data(self, embedded_data: Dict[str, Any], result: ParseResult) -> ParseResult:
        """Parse model from embedded JSON data for perfect round-trip fidelity (EBNF version)."""
        from .common import Variable, Connection, Parameter, VariableType, DataType, ConnectionType
        
        try:
            result.model.model_name = embedded_data.get('model_name', 'EBNFGNNModel')
            result.model.annotation = embedded_data.get('annotation', '')
            
            # Restore variables
            for var_data in embedded_data.get('variables', []):
                var = Variable(
                    name=var_data['name'],
                    var_type=VariableType(var_data.get('var_type', 'hidden_state')),
                    data_type=DataType(var_data.get('data_type', 'categorical')),
                    dimensions=var_data.get('dimensions', [])
                )
                result.model.variables.append(var)
            
            # Restore connections
            for conn_data in embedded_data.get('connections', []):
                conn = Connection(
                    source_variables=conn_data.get('source_variables', []),
                    target_variables=conn_data.get('target_variables', []),
                    connection_type=ConnectionType(conn_data.get('connection_type', 'directed'))
                )
                result.model.connections.append(conn)
            
            # Restore parameters
            for param_data in embedded_data.get('parameters', []):
                param = Parameter(
                    name=param_data['name'],
                    value=param_data['value']
                )
                result.model.parameters.append(param)
            
            # Restore time specification
            if embedded_data.get('time_specification'):
                from .common import TimeSpecification
                time_data = embedded_data['time_specification']
                result.model.time_specification = TimeSpecification(
                    time_type=time_data.get('time_type', 'dynamic'),
                    discretization=time_data.get('discretization'),
                    horizon=time_data.get('horizon'),
                    step_size=time_data.get('step_size')
                )
            
            # Restore ontology mappings
            for mapping_data in embedded_data.get('ontology_mappings', []):
                from .common import OntologyMapping
                mapping = OntologyMapping(
                    variable_name=mapping_data.get('variable_name', ''),
                    ontology_term=mapping_data.get('ontology_term', ''),
                    description=mapping_data.get('description')
                )
                result.model.ontology_mappings.append(mapping)
            
            return result
            
        except Exception as e:
            result.add_error(f"Failed to parse embedded data: {e}")
            return result
        
    def _parse_bnf_content(self, content: str) -> GNNInternalRepresentation:
        """Parse EBNF content (override parent method)."""
        model_name = self._extract_model_name(content)
        
        model = GNNInternalRepresentation(
            model_name=model_name,
            annotation="Parsed from EBNF grammar specification"
        )
        
        # Parse EBNF-specific constructs
        self._parse_ebnf_rules(content, model)
        
        return model
    
    def _parse_ebnf_rules(self, content: str, model: GNNInternalRepresentation):
        """Parse EBNF rules with extended notation."""
        rules = self.rule_pattern.findall(content)
        
        for non_terminal, production in rules:
            # Handle EBNF-specific constructs
            processed_production = self._process_ebnf_constructs(production)
            
            var_type = self._infer_variable_type_from_rule(non_terminal, processed_production)
            
            if var_type:
                variable = Variable(
                    name=non_terminal,
                    var_type=var_type,
                    dimensions=[],
                    data_type=self._infer_data_type_from_production(processed_production),
                    description=f"EBNF rule: {non_terminal} = {production}"
                )
                
                model.variables.append(variable)
            
            # Extract connections
            self._extract_connections_from_ebnf_production(non_terminal, processed_production, model)
    
    def _process_ebnf_constructs(self, production: str) -> str:
        """Process EBNF-specific constructs."""
        # Convert EBNF constructs to simpler form for analysis
        processed = production
        
        # Remove optional constructs [...]
        processed = self.optional_pattern.sub(r'\1', processed)
        
        # Remove repetition constructs {...}
        processed = self.repetition_pattern.sub(r'\1', processed)
        
        # Remove grouping constructs (...)
        processed = self.grouping_pattern.sub(r'\1', processed)
        
        return processed
    
    def _extract_connections_from_ebnf_production(self, non_terminal: str, production: str,
                                                 model: GNNInternalRepresentation):
        """Extract connections from EBNF production rules."""
        # Find identifier patterns (non-terminals in EBNF are often just identifiers)
        identifier_pattern = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')
        identifiers = identifier_pattern.findall(production)
        
        # Filter to keep only potential GNN variables
        gnn_keywords = ['state', 'observation', 'action', 'policy', 'matrix', 'vector', 'variable']
        
        for identifier in identifiers:
            if (identifier != non_terminal and 
                any(keyword in identifier.lower() for keyword in gnn_keywords)):
                
                connection = Connection(
                    source_variables=[identifier],
                    target_variables=[non_terminal],
                    connection_type=ConnectionType.DIRECTED,
                    description=f"EBNF dependency: {identifier} -> {non_terminal}"
                )
                
                model.connections.append(connection)
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions."""
        return ['.ebnf'] 