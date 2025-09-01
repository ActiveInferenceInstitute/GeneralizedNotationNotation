#!/usr/bin/env python3
"""
Real Simulation Monitoring System
Tracks actual simulation execution and logs failures as suggested in web search results
"""

import logging
import functools
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock
import json
from typing import Dict, List, Any, Callable

class SimulationMonitor:
    """
    Monitors real simulation execution with function call tracking and logging
    Based on web search results for simulation monitoring
    """
    
    def __init__(self, log_file: Path = None):
        """Initialize simulation monitor with logging"""
        self.log_file = log_file or Path("simulation_execution.log")
        self.execution_data = {
            "timestamp": datetime.now().isoformat(),
            "simulations": {},
            "failures": {},
            "total_attempted": 0,
            "total_successful": 0,
            "total_failed": 0
        }
        
        # Configure logging as suggested in web results
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("SimulationMonitor")
        self.logger.info("🔍 SimulationMonitor initialized")
    
    def track_simulation(self, simulation_name: str):
        """
        Decorator to track simulation function execution
        Uses mock-based tracking as suggested in web results
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                self.execution_data["total_attempted"] += 1
                
                # Create mock to track function calls (from web results)
                simulation_mock = Mock(side_effect=func)
                
                try:
                    self.logger.info(f"🚀 Starting simulation: {simulation_name}")
                    
                    # Execute the actual simulation function
                    result = simulation_mock(*args, **kwargs)
                    
                    # Check if simulation was actually executed (from web results)
                    if simulation_mock.called:
                        self.logger.info(f"✅ Simulation '{simulation_name}' executed successfully")
                        self.execution_data["simulations"][simulation_name] = {
                            "status": "success",
                            "timestamp": datetime.now().isoformat(),
                            "call_count": simulation_mock.call_count,
                            "result_type": type(result).__name__ if result is not None else "None"
                        }
                        self.execution_data["total_successful"] += 1
                    else:
                        self.logger.error(f"❌ Simulation '{simulation_name}' was NOT executed")
                        self.execution_data["failures"][simulation_name] = {
                            "error": "Function not called",
                            "timestamp": datetime.now().isoformat()
                        }
                        self.execution_data["total_failed"] += 1
                    
                    return result
                    
                except Exception as e:
                    self.logger.error(f"❌ Simulation '{simulation_name}' failed: {str(e)}")
                    self.execution_data["failures"][simulation_name] = {
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "timestamp": datetime.now().isoformat()
                    }
                    self.execution_data["total_failed"] += 1
                    raise
                    
            return wrapper
        return decorator
    
    def check_function_exists(self, func_name: str, func_obj: Callable) -> bool:
        """
        Check if simulation function exists and is callable
        Based on web search results
        """
        exists = callable(func_obj)
        if exists:
            self.logger.info(f"✅ Simulation function '{func_name}' exists and is callable")
        else:
            self.logger.error(f"❌ Simulation function '{func_name}' does not exist or is not callable")
            self.execution_data["failures"][func_name] = {
                "error": "Function not callable",
                "timestamp": datetime.now().isoformat()
            }
        return exists
    
    def monitor_data_collection(self, data_list: List, simulation_name: str) -> bool:
        """
        Monitor if simulation data was collected (SimPy-style from web results)
        """
        if data_list:
            self.logger.info(f"✅ Data collection successful for '{simulation_name}': {len(data_list)} items")
            return True
        else:
            self.logger.error(f"❌ No data collected for simulation '{simulation_name}'")
            self.execution_data["failures"][simulation_name] = {
                "error": "No data collected", 
                "timestamp": datetime.now().isoformat()
            }
            return False
    
    def log_simulation_step(self, simulation_name: str, step: int, data: Dict[str, Any]):
        """Log individual simulation steps for monitoring"""
        self.logger.info(f"📊 {simulation_name} Step {step}: {data}")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive simulation execution report"""
        self.execution_data["completion_timestamp"] = datetime.now().isoformat()
        self.execution_data["success_rate"] = (
            self.execution_data["total_successful"] / max(1, self.execution_data["total_attempted"])
        ) * 100
        
        # Save report to JSON
        report_file = self.log_file.parent / f"simulation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.execution_data, f, indent=2)
        
        self.logger.info(f"📊 Simulation Report Generated:")
        self.logger.info(f"  - Total Attempted: {self.execution_data['total_attempted']}")
        self.logger.info(f"  - Successful: {self.execution_data['total_successful']}")
        self.logger.info(f"  - Failed: {self.execution_data['total_failed']}")
        self.logger.info(f"  - Success Rate: {self.execution_data['success_rate']:.1f}%")
        self.logger.info(f"  - Report saved: {report_file}")
        
        return self.execution_data

# Global monitor instance
global_monitor = SimulationMonitor(Path("output") / "12_execute_output" / "real_simulation_monitor.log")

def track_simulation(simulation_name: str):
    """Convenience decorator using global monitor"""
    return global_monitor.track_simulation(simulation_name)

def log_simulation_failure(simulation_name: str, error: str):
    """Log when a simulation fails to execute"""
    global_monitor.logger.error(f"❌ SIMULATION FAILURE - {simulation_name}: {error}")
    global_monitor.execution_data["failures"][simulation_name] = {
        "error": error,
        "timestamp": datetime.now().isoformat()
    }
    global_monitor.execution_data["total_failed"] += 1

