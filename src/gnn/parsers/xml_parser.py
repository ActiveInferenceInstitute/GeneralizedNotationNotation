"""
XML Parser for GNN XML and PNML Formats

This module provides parsing capabilities for XML and PNML files that specify
GNN models using XML-based representations including Petri nets.

Author: @docxology
Date: 2025-01-11
License: MIT
"""

import xml.etree.ElementTree as ET
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from .common import (
    BaseGNNParser, ParseResult, GNNInternalRepresentation, ParseError,
    Variable, Connection, Parameter, VariableType, DataType, ConnectionType,
    Equation, TimeSpecification, OntologyMapping
)

logger = logging.getLogger(__name__)

class XMLGNNParser(BaseGNNParser):
    """Parser for XML format GNN specifications."""
    
    def __init__(self):
        """Initialize the XML parser."""
        super().__init__()
        
    def parse_file(self, file_path: str) -> ParseResult:
        """Parse an XML file containing GNN specifications."""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            return self._parse_xml_root(root)
            
        except ET.ParseError as e:
            logger.error(f"XML parse error in {file_path}: {e}")
            result = ParseResult(
                model=self.create_empty_model("Failed XML Parse"),
                success=False
            )
            result.add_error(f"Invalid XML format: {e}")
            return result
            
        except Exception as e:
            logger.error(f"Error parsing XML file {file_path}: {e}")
            result = ParseResult(
                model=self.create_empty_model("Failed XML Parse"),
                success=False
            )
            result.add_error(f"Failed to parse XML file: {e}")
            return result
    
    def parse_string(self, content: str) -> ParseResult:
        """Parse XML content from string."""
        try:
            # Quick check if this even looks like XML
            content = content.strip()
            if not content.startswith('<') or '<?xml' not in content and '<gnn' not in content:
                result = ParseResult(
                    model=self.create_empty_model("Invalid XML"),
                    success=False
                )
                result.add_error("Content doesn't appear to be valid XML (missing XML declaration or root element)")
                return result
                
            root = ET.fromstring(content)
            return self._parse_xml_root(root)
            
        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")
            result = ParseResult(
                model=self.create_empty_model("Failed XML Parse"),
                success=False
            )
            result.add_error(f"Invalid XML format: {e}")
            return result
            
        except Exception as e:
            logger.error(f"Error parsing XML content: {e}")
            result = ParseResult(
                model=self.create_empty_model("Failed XML Parse"),
                success=False
            )
            result.add_error(f"Failed to parse XML content: {e}")
            return result
    
    def _parse_xml_root(self, root: ET.Element) -> ParseResult:
        """Parse XML root element."""
        try:
            model = self._convert_xml_to_model(root)
            return ParseResult(model=model, success=True)
            
        except Exception as e:
            logger.error(f"Error converting XML to model: {e}")
            result = ParseResult(
                model=self.create_empty_model("Failed XML Conversion"),
                success=False
            )
            result.add_error(f"Failed to convert XML to model: {e}")
            return result
    
    def _convert_xml_to_model(self, root: ET.Element) -> GNNInternalRepresentation:
        """Convert XML element to GNN internal representation."""
        
        # Extract model name from root element or attributes
        model_name = (root.get('name') or 
                     root.get('model_name') or 
                     root.tag.replace('_', ' ').title() or 
                     'XMLGNNModel')
        
        version = root.get('version', '1.0')
        
        model = GNNInternalRepresentation(
            model_name=model_name,
            version=version,
            annotation="Parsed from XML specification"
        )
        
        # Parse metadata
        self._parse_metadata(root, model)
        
        # Parse variables
        self._parse_xml_variables(root, model)
        
        # Parse connections
        self._parse_xml_connections(root, model)
        
        # Parse parameters
        self._parse_xml_parameters(root, model)
        
        # Parse equations
        self._parse_xml_equations(root, model)
        
        # Parse time specification
        self._parse_xml_time_specification(root, model)
        
        # Parse ontology mappings
        self._parse_xml_ontology_mappings(root, model)
        
        return model
    
    def _parse_metadata(self, root: ET.Element, model: GNNInternalRepresentation):
        """Parse metadata from XML."""
        metadata_elem = root.find('.//metadata')
        if metadata_elem is not None:
            annotation_elem = metadata_elem.find('annotation')
            if annotation_elem is not None and annotation_elem.text:
                model.annotation = annotation_elem.text
    
    def _parse_xml_variables(self, root: ET.Element, model: GNNInternalRepresentation):
        """Parse variables from XML."""
        # Look for variables in multiple possible locations
        variables_containers = [
            root.findall('.//variables/variable'),
            root.findall('.//variable'),
            root.findall('.//state'),
            root.findall('.//node')  # For more generic XML
        ]
        
        for variables in variables_containers:
            for var_elem in variables:
                variable = self._parse_xml_variable(var_elem)
                if variable:
                    model.variables.append(variable)
    
    def _parse_xml_variable(self, var_elem: ET.Element) -> Optional[Variable]:
        """Parse a single variable from XML element."""
        try:
            name = var_elem.get('name') or var_elem.get('id') or var_elem.tag
            
            # Get variable type
            var_type_str = var_elem.get('type', 'hidden_state')
            try:
                var_type = VariableType(var_type_str)
            except ValueError:
                var_type = self._infer_variable_type_from_name(name)
            
            # Get data type
            data_type_str = var_elem.get('data_type', 'continuous')
            try:
                data_type = DataType(data_type_str)
            except ValueError:
                data_type = DataType.CONTINUOUS
            
            # Get dimensions
            dimensions = []
            dims_str = var_elem.get('dimensions')
            if dims_str:
                try:
                    dimensions = [int(d.strip()) for d in dims_str.split(',')]
                except ValueError:
                    pass
            
            # Get description
            description = var_elem.get('description', '')
            if not description:
                desc_elem = var_elem.find('description')
                if desc_elem is not None and desc_elem.text:
                    description = desc_elem.text
            
            return Variable(
                name=name,
                var_type=var_type,
                dimensions=dimensions,
                data_type=data_type,
                description=description
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse XML variable {var_elem.tag}: {e}")
            return None
    
    def _parse_xml_connections(self, root: ET.Element, model: GNNInternalRepresentation):
        """Parse connections from XML."""
        # Look for connections in prioritized order to avoid duplicates
        connection_elements = []
        
        # First try structured format
        connections_in_container = root.findall('.//connections/connection')
        if connections_in_container:
            connection_elements = connections_in_container
        else:
            # Fall back to other formats if no structured format found
            connection_elements = (root.findall('.//connection') + 
                                 root.findall('.//edge') + 
                                 root.findall('.//arc'))
        
        for conn_elem in connection_elements:
            connection = self._parse_xml_connection(conn_elem)
            if connection:
                model.connections.append(connection)
    
    def _parse_xml_connection(self, conn_elem: ET.Element) -> Optional[Connection]:
        """Parse a single connection from XML element."""
        try:
            # Get source and target variables
            source_vars = []
            target_vars = []
            
            # Multiple ways to specify sources and targets
            source_attr = conn_elem.get('source') or conn_elem.get('from')
            target_attr = conn_elem.get('target') or conn_elem.get('to')
            
            if source_attr:
                source_vars = [s.strip() for s in source_attr.split(',')]
            if target_attr:
                target_vars = [t.strip() for t in target_attr.split(',')]
            
            # Try subelements
            sources_elem = conn_elem.find('sources')
            if sources_elem is not None and sources_elem.text:
                source_vars = [s.strip() for s in sources_elem.text.split(',')]
            
            targets_elem = conn_elem.find('targets')
            if targets_elem is not None and targets_elem.text:
                target_vars = [t.strip() for t in targets_elem.text.split(',')]
            
            if not source_vars or not target_vars:
                return None
            
            # Get connection type
            conn_type_str = conn_elem.get('type', 'directed')
            try:
                conn_type = ConnectionType(conn_type_str)
            except ValueError:
                conn_type = ConnectionType.DIRECTED
            
            # Get weight
            weight = None
            weight_str = conn_elem.get('weight')
            if weight_str:
                try:
                    weight = float(weight_str)
                except ValueError:
                    pass
            
            # Get description
            description = conn_elem.get('description', '')
            
            return Connection(
                source_variables=source_vars,
                target_variables=target_vars,
                connection_type=conn_type,
                weight=weight,
                description=description
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse XML connection {conn_elem.tag}: {e}")
            return None
    
    def _parse_xml_parameters(self, root: ET.Element, model: GNNInternalRepresentation):
        """Parse parameters from XML."""
        # Look for parameters in multiple locations
        params_containers = [
            root.findall('.//parameters/parameter'),
            root.findall('.//parameter'),
            root.findall('.//param')
        ]
        
        for params in params_containers:
            for param_elem in params:
                parameter = self._parse_xml_parameter(param_elem)
                if parameter:
                    model.parameters.append(parameter)
    
    def _parse_xml_parameter(self, param_elem: ET.Element) -> Optional[Parameter]:
        """Parse a single parameter from XML element."""
        try:
            name = param_elem.get('name') or param_elem.get('id')
            if not name:
                return None
            
            value = param_elem.get('value') or param_elem.text
            type_hint = param_elem.get('type')
            description = param_elem.get('description', '')
            
            return Parameter(
                name=name,
                value=value,
                type_hint=type_hint,
                description=description
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse XML parameter {param_elem.tag}: {e}")
            return None
    
    def _infer_variable_type_from_name(self, name: str) -> VariableType:
        """Infer variable type from name."""
        name_lower = name.lower()
        
        if any(keyword in name_lower for keyword in ['state', 'hidden', 's_']):
            return VariableType.HIDDEN_STATE
        elif any(keyword in name_lower for keyword in ['obs', 'observation', 'o_']):
            return VariableType.OBSERVATION
        elif any(keyword in name_lower for keyword in ['action', 'control', 'u_']):
            return VariableType.ACTION
        elif any(keyword in name_lower for keyword in ['policy', 'pi_']):
            return VariableType.POLICY
        
        return VariableType.HIDDEN_STATE
    
    def _parse_xml_equations(self, root: ET.Element, model: GNNInternalRepresentation):
        """Parse equations from XML."""
        equations_containers = [
            root.findall('.//equations/equation'),
            root.findall('.//equation')
        ]
        
        for equations in equations_containers:
            for eq_elem in equations:
                equation = self._parse_xml_equation(eq_elem)
                if equation:
                    model.equations.append(equation)
    
    def _parse_xml_equation(self, eq_elem: ET.Element) -> Optional[Equation]:
        """Parse a single equation from XML element."""
        try:
            content = eq_elem.text or ""
            label = eq_elem.get('label')
            format_type = eq_elem.get('format', 'latex')
            description = eq_elem.get('description', '')
            
            return Equation(
                content=content,
                label=label,
                format=format_type,
                description=description
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse XML equation {eq_elem.tag}: {e}")
            return None
    
    def _parse_xml_time_specification(self, root: ET.Element, model: GNNInternalRepresentation):
        """Parse time specification from XML."""
        time_elem = root.find('.//time_specification')
        if time_elem is not None:
            time_type = time_elem.get('type', 'Static')
            discretization = time_elem.get('discretization')
            horizon = time_elem.get('horizon')
            
            # Convert horizon to appropriate type
            if horizon:
                try:
                    if horizon.isdigit():
                        horizon = int(horizon)
                except:
                    pass  # Keep as string
            
            model.time_specification = TimeSpecification(
                time_type=time_type,
                discretization=discretization,
                horizon=horizon
            )
    
    def _parse_xml_ontology_mappings(self, root: ET.Element, model: GNNInternalRepresentation):
        """Parse ontology mappings from XML."""
        ontology_containers = [
            root.findall('.//ontology_mappings/mapping'),
            root.findall('.//ontology/mapping'),
            root.findall('.//mapping')
        ]
        
        for mappings in ontology_containers:
            for mapping_elem in mappings:
                mapping = self._parse_xml_ontology_mapping(mapping_elem)
                if mapping:
                    model.ontology_mappings.append(mapping)
    
    def _parse_xml_ontology_mapping(self, mapping_elem: ET.Element) -> Optional[OntologyMapping]:
        """Parse a single ontology mapping from XML element."""
        try:
            variable_name = mapping_elem.get('variable')
            ontology_term = mapping_elem.get('term')
            description = mapping_elem.get('description', '')
            
            if not variable_name or not ontology_term:
                return None
            
            return OntologyMapping(
                variable_name=variable_name,
                ontology_term=ontology_term,
                description=description
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse XML ontology mapping {mapping_elem.tag}: {e}")
            return None
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions."""
        return ['.xml']


class PNMLParser(XMLGNNParser):
    """Parser for PNML (Petri Net Markup Language) format."""
    
    def __init__(self):
        """Initialize the PNML parser."""
        super().__init__()
        
    def _convert_xml_to_model(self, root: ET.Element) -> GNNInternalRepresentation:
        """Convert PNML to GNN internal representation."""
        
        # PNML has specific structure
        model_name = root.get('id', 'PNMLGNNModel')
        
        model = GNNInternalRepresentation(
            model_name=model_name,
            annotation="Parsed from PNML (Petri Net) specification"
        )
        
        # Parse nets
        nets = root.findall('.//net')
        for net in nets:
            self._parse_pnml_net(net, model)
        
        return model
    
    def _parse_pnml_net(self, net_elem: ET.Element, model: GNNInternalRepresentation):
        """Parse a PNML net element."""
        # Parse places as variables
        places = net_elem.findall('.//place')
        for place in places:
            variable = self._parse_pnml_place(place)
            if variable:
                model.variables.append(variable)
        
        # Parse transitions as variables
        transitions = net_elem.findall('.//transition')
        for transition in transitions:
            variable = self._parse_pnml_transition(transition)
            if variable:
                model.variables.append(variable)
        
        # Parse arcs as connections
        arcs = net_elem.findall('.//arc')
        for arc in arcs:
            connection = self._parse_pnml_arc(arc)
            if connection:
                model.connections.append(connection)
    
    def _parse_pnml_place(self, place_elem: ET.Element) -> Optional[Variable]:
        """Parse a PNML place as a variable."""
        place_id = place_elem.get('id')
        if not place_id:
            return None
        
        # Get name from text element
        name_elem = place_elem.find('.//text')
        name = name_elem.text if name_elem is not None and name_elem.text else place_id
        
        return Variable(
            name=name,
            var_type=VariableType.HIDDEN_STATE,  # Places represent states
            dimensions=[],
            data_type=DataType.INTEGER,  # Petri net places typically hold tokens (integers)
            description=f"PNML place: {place_id}"
        )
    
    def _parse_pnml_transition(self, trans_elem: ET.Element) -> Optional[Variable]:
        """Parse a PNML transition as a variable."""
        trans_id = trans_elem.get('id')
        if not trans_id:
            return None
        
        # Get name from text element
        name_elem = trans_elem.find('.//text')
        name = name_elem.text if name_elem is not None and name_elem.text else trans_id
        
        return Variable(
            name=name,
            var_type=VariableType.ACTION,  # Transitions represent actions
            dimensions=[],
            data_type=DataType.BINARY,  # Transitions fire or don't fire
            description=f"PNML transition: {trans_id}"
        )
    
    def _parse_pnml_arc(self, arc_elem: ET.Element) -> Optional[Connection]:
        """Parse a PNML arc as a connection."""
        source = arc_elem.get('source')
        target = arc_elem.get('target')
        
        if not source or not target:
            return None
        
        return Connection(
            source_variables=[source],
            target_variables=[target],
            connection_type=ConnectionType.DIRECTED,
            description=f"PNML arc: {source} -> {target}"
        )
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions."""
        return ['.pnml'] 