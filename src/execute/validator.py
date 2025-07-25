#!/usr/bin/env python3
"""
Execution Environment Validator

This module provides comprehensive validation of the execution environment
to ensure safe and reliable execution of GNN pipeline simulations.
"""

import sys
import os
import platform
import subprocess
import psutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class ValidationResult:
    """Results of environment validation."""
    component: str
    status: str  # "pass", "warn", "fail"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    suggestion: Optional[str] = None

@dataclass
class EnvironmentValidation:
    """Complete environment validation results."""
    timestamp: datetime
    overall_status: str  # "healthy", "degraded", "failed"
    results: List[ValidationResult] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    
    def add_result(self, result: ValidationResult):
        """Add a validation result."""
        self.results.append(result)
        self.summary[result.status] = self.summary.get(result.status, 0) + 1
        
    def get_failed_components(self) -> List[str]:
        """Get list of failed components."""
        return [r.component for r in self.results if r.status == "fail"]
        
    def get_warnings(self) -> List[str]:
        """Get list of components with warnings."""
        return [r.component for r in self.results if r.status == "warn"]

def check_python_environment() -> ValidationResult:
    """Validate Python environment and version."""
    try:
        version = sys.version_info
        
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            return ValidationResult(
                component="python_version",
                status="fail",
                message=f"Python {version.major}.{version.minor} is too old",
                details={"version": f"{version.major}.{version.minor}.{version.micro}"},
                suggestion="Upgrade to Python 3.8 or later"
            )
        elif version.minor < 9:
            return ValidationResult(
                component="python_version",
                status="warn",
                message=f"Python {version.major}.{version.minor} is supported but not optimal",
                details={"version": f"{version.major}.{version.minor}.{version.micro}"},
                suggestion="Consider upgrading to Python 3.9+ for better performance"
            )
        else:
            return ValidationResult(
                component="python_version",
                status="pass",
                message=f"Python {version.major}.{version.minor} is supported",
                details={"version": f"{version.major}.{version.minor}.{version.micro}"}
            )
            
    except Exception as e:
        return ValidationResult(
            component="python_version",
            status="fail",
            message=f"Failed to check Python version: {e}",
            suggestion="Verify Python installation"
        )

def check_system_resources() -> List[ValidationResult]:
    """Validate system resources (memory, disk, CPU)."""
    results = []
    
    try:
        # Memory check
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024 ** 3)
        
        if memory_gb < 2:
            results.append(ValidationResult(
                component="memory",
                status="fail",
                message=f"Insufficient memory: {memory_gb:.1f}GB available",
                details={"total_gb": memory_gb, "available_gb": memory.available / (1024 ** 3)},
                suggestion="Increase system memory to at least 4GB"
            ))
        elif memory_gb < 4:
            results.append(ValidationResult(
                component="memory",
                status="warn",
                message=f"Low memory: {memory_gb:.1f}GB available",
                details={"total_gb": memory_gb, "available_gb": memory.available / (1024 ** 3)},
                suggestion="Consider increasing memory for better performance"
            ))
        else:
            results.append(ValidationResult(
                component="memory",
                status="pass",
                message=f"Sufficient memory: {memory_gb:.1f}GB available",
                details={"total_gb": memory_gb, "available_gb": memory.available / (1024 ** 3)}
            ))
        
        # Disk space check
        disk = psutil.disk_usage('/')
        disk_free_gb = disk.free / (1024 ** 3)
        
        if disk_free_gb < 1:
            results.append(ValidationResult(
                component="disk_space",
                status="fail",
                message=f"Insufficient disk space: {disk_free_gb:.1f}GB free",
                details={"free_gb": disk_free_gb, "total_gb": disk.total / (1024 ** 3)},
                suggestion="Free up at least 2GB of disk space"
            ))
        elif disk_free_gb < 5:
            results.append(ValidationResult(
                component="disk_space",
                status="warn",
                message=f"Low disk space: {disk_free_gb:.1f}GB free",
                details={"free_gb": disk_free_gb, "total_gb": disk.total / (1024 ** 3)},
                suggestion="Consider freeing up more disk space"
            ))
        else:
            results.append(ValidationResult(
                component="disk_space",
                status="pass",
                message=f"Sufficient disk space: {disk_free_gb:.1f}GB free",
                details={"free_gb": disk_free_gb, "total_gb": disk.total / (1024 ** 3)}
            ))
        
        # CPU check
        cpu_count = psutil.cpu_count(logical=False)
        cpu_percent = psutil.cpu_percent(interval=1)
        
        if cpu_percent > 90:
            results.append(ValidationResult(
                component="cpu_usage",
                status="warn",
                message=f"High CPU usage: {cpu_percent}%",
                details={"cpu_count": cpu_count, "cpu_percent": cpu_percent},
                suggestion="Wait for CPU usage to decrease before running intensive tasks"
            ))
        else:
            results.append(ValidationResult(
                component="cpu_usage",
                status="pass",
                message=f"Normal CPU usage: {cpu_percent}%",
                details={"cpu_count": cpu_count, "cpu_percent": cpu_percent}
            ))
            
    except Exception as e:
        results.append(ValidationResult(
            component="system_resources",
            status="fail",
            message=f"Failed to check system resources: {e}",
            suggestion="Verify system monitoring tools are available"
        ))
    
    return results

def check_dependencies() -> List[ValidationResult]:
    """Check for required Python packages."""
    results = []
    
    required_packages = [
        ("numpy", "1.19.0"),
        ("matplotlib", "3.3.0"),
        ("networkx", "2.5"),
        ("pandas", "1.1.0"),
        ("pyyaml", "5.4.0"),
        ("scipy", "1.5.0"),
        ("scikit-learn", "0.24.0")
    ]
    
    for package, min_version in required_packages:
        try:
            # Try to import the package
            if package == "pyyaml":
                # Special case: pyyaml installs as 'yaml' module
                import yaml
                module = yaml
            elif package == "scikit-learn":
                # Special case: scikit-learn installs as 'sklearn' module  
                import sklearn
                module = sklearn
            else:
                module = __import__(package)
            
            # Try to get version
            try:
                version = getattr(module, '__version__', 'unknown')
                
                results.append(ValidationResult(
                    component=f"dependency_{package}",
                    status="pass",
                    message=f"{package} {version} is available",
                    details={"package": package, "version": version, "min_version": min_version}
                ))
                
            except Exception:
                results.append(ValidationResult(
                    component=f"dependency_{package}",
                    status="warn",
                    message=f"{package} is available but version unknown",
                    details={"package": package, "min_version": min_version},
                    suggestion=f"Verify {package} version meets minimum requirement"
                ))
                
        except ImportError:
            results.append(ValidationResult(
                component=f"dependency_{package}",
                status="fail",
                message=f"{package} is not available",
                details={"package": package, "min_version": min_version},
                suggestion=f"Install {package}>={min_version} with: pip install {package}>={min_version}"
            ))
    
    return results

def check_file_permissions() -> List[ValidationResult]:
    """Check file system permissions for execution."""
    results = []
    
    try:
        # Check write permissions in current directory
        test_file = Path("test_write_permission.tmp")
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            test_file.unlink()
            
            results.append(ValidationResult(
                component="write_permissions",
                status="pass",
                message="Write permissions verified",
                details={"location": str(Path.cwd())}
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                component="write_permissions",
                status="fail",
                message=f"Cannot write to current directory: {e}",
                details={"location": str(Path.cwd())},
                suggestion="Ensure write permissions in working directory"
            ))
        
        # Check execution permissions
        if platform.system() != "Windows":
            # On Unix-like systems, check if we can execute files
            try:
                result = subprocess.run(["echo", "test"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    results.append(ValidationResult(
                        component="execute_permissions",
                        status="pass",
                        message="Execute permissions verified"
                    ))
                else:
                    results.append(ValidationResult(
                        component="execute_permissions",
                        status="warn",
                        message="Execute permissions may be limited",
                        suggestion="Verify script execution permissions"
                    ))
            except Exception as e:
                results.append(ValidationResult(
                    component="execute_permissions",
                    status="warn",
                    message=f"Cannot verify execute permissions: {e}",
                    suggestion="Manually verify script execution permissions"
                ))
        
    except Exception as e:
        results.append(ValidationResult(
            component="file_permissions",
            status="fail",
            message=f"Failed to check file permissions: {e}",
            suggestion="Verify file system access and permissions"
        ))
    
    return results

def check_network_connectivity() -> ValidationResult:
    """Check basic network connectivity (optional)."""
    try:
        # Try to resolve DNS
        import socket
        socket.gethostbyname("google.com")
        
        return ValidationResult(
            component="network_connectivity",
            status="pass",
            message="Network connectivity verified"
        )
        
    except Exception as e:
        return ValidationResult(
            component="network_connectivity",
            status="warn",
            message=f"Network connectivity limited: {e}",
            suggestion="Some features requiring network access may not work"
        )

def validate_execution_environment() -> Dict[str, Any]:
    """Perform comprehensive execution environment validation."""
    validation = EnvironmentValidation(
        timestamp=datetime.now(),
        overall_status="healthy"
    )
    
    # Run all validation checks
    validation.add_result(check_python_environment())
    
    for result in check_system_resources():
        validation.add_result(result)
    
    for result in check_dependencies():
        validation.add_result(result)
    
    for result in check_file_permissions():
        validation.add_result(result)
    
    validation.add_result(check_network_connectivity())
    
    # Calculate summary
    validation.summary = {
        "total_checks": len(validation.results),
        "total_passed": len([r for r in validation.results if r.status == "pass"]),
        "total_warnings": len([r for r in validation.results if r.status == "warn"]),
        "total_errors": len([r for r in validation.results if r.status == "fail"])
    }
    
    # Determine overall status
    if validation.summary["total_errors"] > 0:
        validation.overall_status = "failed"
    elif validation.summary["total_warnings"] > 0:
        validation.overall_status = "degraded"
    else:
        validation.overall_status = "healthy"
    
    # Convert to dict for JSON serialization
    return {
        "timestamp": validation.timestamp.isoformat(),
        "overall_status": validation.overall_status,
        "summary": validation.summary,
        "results": [
            {
                "component": r.component,
                "status": r.status,
                "message": r.message,
                "details": r.details,
                "suggestion": r.suggestion
            }
            for r in validation.results
        ]
    }

def log_validation_results(validation_results: Dict[str, Any], logger: logging.Logger):
    """Log validation results in a structured format."""
    logger.info(f"Environment validation completed: {validation_results['overall_status']}")
    
    summary = validation_results["summary"]
    logger.info(f"Validation summary: {summary['total_passed']} passed, "
                f"{summary['total_warnings']} warnings, {summary['total_errors']} errors")
    
    # Log failures first
    for result in validation_results["results"]:
        if result["status"] == "fail":
            logger.error(f"❌ {result['component']}: {result['message']}")
            if result.get("suggestion"):
                logger.error(f"   💡 Suggestion: {result['suggestion']}")
    
    # Log warnings
    for result in validation_results["results"]:
        if result["status"] == "warn":
            logger.warning(f"⚠️ {result['component']}: {result['message']}")
            if result.get("suggestion"):
                logger.warning(f"   💡 Suggestion: {result['suggestion']}")
    
    # Log successes (debug level)
    passed_count = 0
    for result in validation_results["results"]:
        if result["status"] == "pass":
            passed_count += 1
            logger.debug(f"✅ {result['component']}: {result['message']}")
    
    if passed_count > 0:
        logger.info(f"✅ {passed_count} components passed validation")

if __name__ == "__main__":
    # Standalone validation for testing
    results = validate_execution_environment()
    print(f"Environment Status: {results['overall_status']}")
    print(f"Summary: {results['summary']}")
    
    for result in results["results"]:
        status_emoji = {"pass": "✅", "warn": "⚠️", "fail": "❌"}[result["status"]]
        print(f"{status_emoji} {result['component']}: {result['message']}")
        if result.get("suggestion"):
            print(f"   💡 {result['suggestion']}") 