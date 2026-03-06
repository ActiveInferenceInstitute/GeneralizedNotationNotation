from .common import GNNInternalRepresentation
from .base_serializer import BaseGNNSerializer
from .xsd_serializer import XSDSerializer

class SchemaSerializer(BaseGNNSerializer):
    """Serializer for formal schema languages."""

    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to generic schema format."""
        # Default to XSD format
        return XSDSerializer().serialize(model)
