from typing import Dict, Any, List, Optional, Union, Protocol
from abc import ABC, abstractmethod
import json
from datetime import datetime
from .common import GNNInternalRepresentation, GNNFormat
from .base_serializer import BaseGNNSerializer

class ASN1Serializer(BaseGNNSerializer):
    """Serializer for ASN.1 format."""
    
    def __init__(self):
        super().__init__()
        self.format_name = 'asn1'
    
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to ASN.1 format."""
        lines = []
        
        # Module header
        model_name_clean = self._sanitize_asn1_name(model.model_name)
        lines.append(f"{model_name_clean}-Schema DEFINITIONS AUTOMATIC TAGS ::= BEGIN")
        lines.append("")
        
        # Module comment
        lines.append("-- GNN Model Schema Definition")
        lines.append(f"-- Model: {model.model_name}")
        if model.annotation:
            lines.append(f"-- {model.annotation.replace(chr(10), chr(10) + '-- ')}")
        lines.append("")
        
        # Main model structure
        lines.append("GNNModel ::= SEQUENCE {")
        lines.append("    modelName       UTF8String,")
        lines.append("    version         UTF8String,")
        lines.append("    annotation      UTF8String OPTIONAL,")
        
        if model.variables:
            lines.append("    variables       SEQUENCE OF Variable OPTIONAL,")
        if model.connections:
            lines.append("    connections     SEQUENCE OF Connection OPTIONAL,")
        if model.parameters:
            lines.append("    parameters      SEQUENCE OF Parameter OPTIONAL")
        
        lines.append("}")
        lines.append("")
        
        # Variable definition
        if model.variables:
            lines.append("Variable ::= SEQUENCE {")
            lines.append("    name            UTF8String,")
            lines.append("    varType         UTF8String,")
            lines.append("    dataType        UTF8String,")
            lines.append("    dimensions      SEQUENCE OF INTEGER OPTIONAL,")
            lines.append("    description     UTF8String OPTIONAL")
            lines.append("}")
            lines.append("")
        
        # Connection definition
        if model.connections:
            lines.append("Connection ::= SEQUENCE {")
            lines.append("    sourceVariables SEQUENCE OF UTF8String,")
            lines.append("    targetVariables SEQUENCE OF UTF8String,")
            lines.append("    connectionType  UTF8String,")
            lines.append("    weight          REAL OPTIONAL,")
            lines.append("    description     UTF8String OPTIONAL")
            lines.append("}")
            lines.append("")
        
        # Parameter definition
        if model.parameters:
            lines.append("Parameter ::= SEQUENCE {")
            lines.append("    name            UTF8String,")
            lines.append("    value           UTF8String,")
            lines.append("    typeHint        UTF8String OPTIONAL,")
            lines.append("    description     UTF8String OPTIONAL")
            lines.append("}")
            lines.append("")
        
        # Add example values section
        lines.append("-- Example Values")
        lines.append("exampleModel GNNModel ::= {")
        lines.append(f"    modelName \"{model.model_name}\",")
        lines.append("    version \"1.0\",")
        if model.annotation:
            lines.append(f"    annotation \"{self._escape_string(model.annotation)}\",")
        
        if model.variables:
            lines.append("    variables {")
            for i, var in enumerate(model.variables[:3]):  # Show first 3 variables as examples
                var_type = var.var_type.value if hasattr(var, 'var_type') else 'hidden_state'
                data_type = var.data_type.value if hasattr(var, 'data_type') else 'categorical'
                
                lines.append("        {")
                lines.append(f"            name \"{var.name}\",")
                lines.append(f"            varType \"{var_type}\",")
                lines.append(f"            dataType \"{data_type}\"")
                if hasattr(var, 'dimensions') and var.dimensions:
                    dims_str = ", ".join(str(d) for d in var.dimensions)
                    lines.append(f"            dimensions {{ {dims_str} }}")
                
                if i < min(len(model.variables), 3) - 1:
                    lines.append("        },")
                else:
                    lines.append("        }")
            lines.append("    }")
        
        lines.append("}")
        lines.append("")
        
        lines.append(f"END -- {model_name_clean}-Schema")
        lines.append("")
        
        # Join all lines
        content = '\n'.join(lines)
        
        # Add embedded model data for round-trip fidelity
        return self._add_embedded_model_data(content, model)
    
    def _sanitize_asn1_name(self, name: str) -> str:
        """Sanitize names for ASN.1 syntax."""
        import re
        # ASN.1 module names must start with uppercase letter
        sanitized = re.sub(r'[^a-zA-Z0-9\-]', '', name)
        if sanitized and sanitized[0].islower():
            sanitized = sanitized[0].upper() + sanitized[1:]
        elif not sanitized or not sanitized[0].isalpha():
            sanitized = 'GNNModel' + sanitized
        return sanitized or 'GNNModel'
    
    def _escape_string(self, text: str) -> str:
        """Escape string for ASN.1 format."""
        return text.replace('"', '""').replace('\n', ' ').replace('\r', ' ') 