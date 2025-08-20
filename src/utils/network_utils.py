"""
Network utilities for the GNN Processing Pipeline.

This module provides network-related utilities for API requests,
batch operations, and network timing measurements.
"""

import time
import logging
try:
    import requests
except Exception:
    # Provide minimal in-repo fallback for tests that don't perform real HTTP calls
    class _DummyResponse:
        status_code = 0
        content = b''
        headers = {}

    class _DummyRequests:
        @staticmethod
        def request(method, url, **kwargs):
            raise RuntimeError('requests library not available in test environment')

    requests = _DummyRequests()
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json

logger = logging.getLogger(__name__)

def timed_request(url: str, method: str = "GET", **kwargs) -> Dict[str, Any]:
    """
    Make a timed HTTP request and return timing information.
    
    Args:
        url: The URL to request
        method: HTTP method (GET, POST, etc.)
        **kwargs: Additional arguments for requests
        
    Returns:
        Dictionary with response data and timing information
    """
    start_time = time.time()
    
    try:
        response = requests.request(method, url, **kwargs)
        end_time = time.time()
        
        return {
            "success": True,
            "status_code": response.status_code,
            "response_time": end_time - start_time,
            "url": url,
            "method": method,
            "content_length": len(response.content) if response.content else 0,
            "headers": dict(response.headers)
        }
    except Exception as e:
        end_time = time.time()
        return {
            "success": False,
            "error": str(e),
            "response_time": end_time - start_time,
            "url": url,
            "method": method
        }

def batch_request(urls: List[str], method: str = "GET", **kwargs) -> List[Dict[str, Any]]:
    """
    Make batch HTTP requests and return results.
    
    Args:
        urls: List of URLs to request
        method: HTTP method (GET, POST, etc.)
        **kwargs: Additional arguments for requests
        
    Returns:
        List of response dictionaries
    """
    results = []
    
    for url in urls:
        result = timed_request(url, method, **kwargs)
        results.append(result)
        
    return results

def validate_api_endpoint(url: str, expected_status: int = 200) -> Dict[str, Any]:
    """
    Validate an API endpoint by making a test request.
    
    Args:
        url: The URL to validate
        expected_status: Expected HTTP status code
        
    Returns:
        Dictionary with validation results
    """
    result = timed_request(url)
    
    validation_result = {
        "url": url,
        "accessible": result["success"],
        "status_code": result.get("status_code"),
        "response_time": result.get("response_time", 0),
        "valid": result["success"] and result.get("status_code") == expected_status
    }
    
    return validation_result

def get_network_performance_metrics(urls: List[str]) -> Dict[str, Any]:
    """
    Get network performance metrics for a list of URLs.
    
    Args:
        urls: List of URLs to test
        
    Returns:
        Dictionary with performance metrics
    """
    results = batch_request(urls)
    
    successful_requests = [r for r in results if r["success"]]
    failed_requests = [r for r in results if not r["success"]]
    
    response_times = [r["response_time"] for r in successful_requests]
    
    metrics = {
        "total_requests": len(results),
        "successful_requests": len(successful_requests),
        "failed_requests": len(failed_requests),
        "success_rate": len(successful_requests) / len(results) if results else 0,
        "average_response_time": sum(response_times) / len(response_times) if response_times else 0,
        "min_response_time": min(response_times) if response_times else 0,
        "max_response_time": max(response_times) if response_times else 0,
        "total_response_time": sum(response_times)
    }
    
    return metrics 