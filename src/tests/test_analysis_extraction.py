
import pytest
from pathlib import Path
import json
import csv
from analysis.post_simulation import extract_activeinference_jl_data, extract_discopy_data


@pytest.fixture
def temp_activeinference_output(tmp_path):
    """Create a temporary ActiveInference.jl output structure."""
    impl_dir = tmp_path / "activeinference_jl"
    impl_dir.mkdir()
    
    output_dir = impl_dir / "activeinference_outputs_2026-01-23_12-00-00"
    output_dir.mkdir()
    
    # Create CSV data
    csv_file = output_dir / "simulation_results.csv"
    with open(csv_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["# ActiveInference.jl Simulation Results"])
        writer.writerow(["# Generated: ..."])
        writer.writerow(["# Model: test_model"])
        writer.writerow(["# Steps: 10"])
        writer.writerow(["# Columns: step, observation, action, belief_state_1"])
        
        # Add 10 steps of data
        for i in range(1, 11):
            writer.writerow([i, 1.0, 2.0, 0.9, 0.1, 0.0]) # step, obs, act, beliefs...
            
    return impl_dir

@pytest.fixture
def temp_discopy_output(tmp_path):
    """Create a temporary DisCoPy output structure."""
    impl_dir = tmp_path / "discopy"
    impl_dir.mkdir()
    
    report_file = impl_dir / "discopy_execution_report.json"
    
    data = {
        "analysis_summary": {
            "total_files_processed": 5,
            "jax_outputs_analyzed": 2,
            "diagrams_validated": 3
        },
        "executions": [
            {
                "script": "diagram1.png",
                "type": "diagram_validation",
                "status": "SUCCESS",
                "file_path": "/path/to/diagram1.png"
            },
               {
                "script": "jax1.png",
                "type": "jax_analysis",
                "status": "SUCCESS",
                "file_path": "/path/to/jax1.png"
            }
        ]
    }
    
    with open(report_file, 'w') as f:
        json.dump(data, f)
        
    return impl_dir

def test_extract_activeinference_jl_data(temp_activeinference_output):
    """Test extracting data from ActiveInference.jl CSV."""
    execution_result = {
        "implementation_directory": str(temp_activeinference_output),
        "simulation_data": {} # Initial empty data
    }
    
    extracted = extract_activeinference_jl_data(execution_result)
    
    assert extracted["num_timesteps"] == 10
    assert len(extracted["observations"]) == 10
    assert len(extracted["actions"]) == 10
    assert len(extracted["beliefs"]) == 10
    assert extracted["observations"][0] == 1.0
    assert extracted["actions"][0] == 2.0
    assert extracted["beliefs"][0] == [0.9, 0.1, 0.0]

def test_extract_discopy_data(temp_discopy_output):
    """Test extracting data from DisCoPy report."""
    execution_result = {
        "implementation_directory": str(temp_discopy_output),
        "simulation_data": {}
    }
    
    extracted = extract_discopy_data(execution_result)
    
    assert len(extracted["diagrams"]) == 1 # Only 1 SUCCESS diagram
    assert extracted["visualization_count"] == 1
    assert len(extracted["traces"]) == 2 # 2 JAX analyzed = 2 steps
