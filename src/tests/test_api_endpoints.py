"""Tests for the GNN API endpoints."""

from unittest.mock import patch

import pytest

# Import the API app factory
try:
    from api.app import FASTAPI_AVAILABLE, create_app
except ImportError:
    try:
        from src.api.app import FASTAPI_AVAILABLE, create_app
    except ImportError:
        FASTAPI_AVAILABLE = False

@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
def test_api_health_endpoint():
    """Test the health check endpoint."""
    from fastapi.testclient import TestClient
    
    app = create_app()
    client = TestClient(app)
    
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "renderers" in data
    assert "version" in data

@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
def test_api_list_runs_empty():
    """Test listing runs when none have been submitted."""
    from fastapi.testclient import TestClient
    
    # We need to clear the global _runs dict if we want a clean state,
    # but since it's module-level in app.py, isolations is tricky without reloading.
    # For baseline, we just check it returns a dict.
    app = create_app()
    client = TestClient(app)
    
    response = client.get("/api/v1/runs")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)

@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
def test_api_submit_run_invalid_payload():
    """Test submitting a run with invalid JSON."""
    from fastapi.testclient import TestClient
    
    app = create_app()
    client = TestClient(app)
    
    response = client.post("/api/v1/run", json={"skip_steps": ["not_an_int"]})
    # FastAPI returns 422 Unprocessable Entity for schema validation errors
    assert response.status_code == 422

@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
def test_api_submit_run_success():
    """Test successful run submission."""
    from fastapi.testclient import TestClient
    
    app = create_app()
    client = TestClient(app)
    
    # Mock compute_run_hash to avoid real hashing logic
    with patch('pipeline.hasher.compute_run_hash', return_value="test_hash_123"):
        # Mock background task execution
        with patch('api.app._execute_pipeline'):
            response = client.post("/api/v1/run", json={
                "target_dir": "input/gnn_files",
                "output_dir": "output",
                "skip_steps": [13]
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["run_hash"] == "test_hash_123"
            assert data["status"] == "queued"

def test_api_availability_flag():
    """Verify that the availability flag matches reality."""
    try:
        import fastapi
        expected = True
    except ImportError:
        expected = False
    
    assert FASTAPI_AVAILABLE == expected
