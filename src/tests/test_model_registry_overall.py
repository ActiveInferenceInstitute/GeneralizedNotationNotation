import pytest
import json
from pathlib import Path
from model_registry.registry import ModelRegistry, ModelEntry, ModelVersion

class TestModelRegistryOverall:
    """Test suite for Model Registry module."""

    @pytest.fixture
    def registry_file(self, safe_filesystem):
        return safe_filesystem.temp_dir / "registry.json"

    @pytest.fixture
    def sample_model_file(self, safe_filesystem):
        content = """
# Test Model
ModelName: MyRegistryModel
Version: 1.2.3
Tags: tag1, tag2
Description: A test description.
Author: Me
"""
        return safe_filesystem.create_file("my_model.md", content)

    def test_registry_initialization(self, registry_file):
        """Test registry creation."""
        registry = ModelRegistry(registry_file)
        assert registry.registry_path == registry_file
        assert len(registry.models) == 0

    def test_register_model_flow(self, registry_file, sample_model_file):
        """Test registering a new model."""
        registry = ModelRegistry(registry_file)
        
        success = registry.register_model(sample_model_file)
        assert success is True
        
        # Verify model entry
        model_id = sample_model_file.stem
        assert model_id in registry.models
        model = registry.models[model_id]
        
        assert model.name == "MyRegistryModel"
        assert model.current_version == "1.2.3"
        assert "tag1" in model.tags
        # Author may be in model metadata or version metadata
        author = model.metadata.get("author") or model.get_version().metadata.get("author", "")
        assert "Me" in author, f"Expected 'Me' in author, got: {author}"
        
        # Verify persistence
        registry.save()
        assert registry_file.exists()
        
        # Load fresh
        registry2 = ModelRegistry(registry_file)
        assert model_id in registry2.models
        assert registry2.models[model_id].name == "MyRegistryModel"

    def test_search_models(self, registry_file, sample_model_file):
        """Test searching functionality."""
        registry = ModelRegistry(registry_file)
        registry.register_model(sample_model_file)
        
        # Search by name
        results = registry.search_models("RegistryModel")
        assert len(results) == 1
        assert results[0].name == "MyRegistryModel"
        
        # Search by tag
        results = registry.search_models("tag2")
        assert len(results) == 1

    def test_delete_model(self, registry_file, sample_model_file):
        """Test deletion."""
        registry = ModelRegistry(registry_file)
        registry.register_model(sample_model_file)
        
        assert len(registry.models) == 1
        
        success = registry.delete_model(sample_model_file.stem)
        assert success is True
        assert len(registry.models) == 0
