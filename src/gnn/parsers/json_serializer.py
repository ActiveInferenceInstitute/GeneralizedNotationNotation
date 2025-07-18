from typing import Dict, Any, List, Optional, Union, Protocol
from abc import ABC, abstractmethod
import json
from .common import GNNInternalRepresentation, GNNFormat
from .base_serializer import BaseGNNSerializer

class JSONSerializer(BaseGNNSerializer):
    """Serializer for JSON data interchange format."""
    
    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to JSON format."""
        return json.dumps(model.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) 