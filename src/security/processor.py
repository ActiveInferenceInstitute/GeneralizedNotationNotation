#!/usr/bin/env python3
"""
Security processor module for GNN pipeline.
"""

from pathlib import Path
from typing import Dict, Any, List
import logging
import json
import hashlib
import re
from datetime import datetime

from utils.pipeline_template import (
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)

def process_security(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Process security validation for GNN files.
    
    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Directory to save results
        verbose: Enable verbose output
        **kwargs: Additional arguments
        
    Returns:
        True if processing successful, False otherwise
    """
    logger = logging.getLogger("security")
    
    try:
        log_step_start(logger, "Processing security")
        
        # Create results directory
        results_dir = output_dir / "security_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results
        results = {
            "timestamp": datetime.now().isoformat(),
            "processed_files": 0,
            "success": True,
            "errors": [],
            "security_checks": [],
            "vulnerabilities": [],
            "recommendations": []
        }
        
        # Find GNN files
        gnn_files = list(target_dir.glob("*.md"))
        if not gnn_files:
            logger.warning("No GNN files found for security processing")
            results["success"] = False
            results["errors"].append("No GNN files found")
        else:
            results["processed_files"] = len(gnn_files)
            
            # Process each GNN file
            for gnn_file in gnn_files:
                try:
                    # Perform security checks
                    security_check = perform_security_check(gnn_file, verbose)
                    results["security_checks"].append(security_check)
                    
                    # Check for vulnerabilities
                    vulnerabilities = check_vulnerabilities(gnn_file, verbose)
                    results["vulnerabilities"].extend(vulnerabilities)
                    
                    # Generate security recommendations
                    recommendations = generate_security_recommendations(gnn_file, verbose)
                    results["recommendations"].extend(recommendations)
                    
                except Exception as e:
                    error_info = {
                        "file": str(gnn_file),
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                    results["errors"].append(error_info)
                    logger.error(f"Error processing {gnn_file}: {e}")
        
        # Save detailed results
        results_file = results_dir / "security_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate security summary
        summary = generate_security_summary(results)
        summary_file = results_dir / "security_summary.md"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        if results["success"]:
            log_step_success(logger, "Security processing completed successfully")
        else:
            log_step_error(logger, "Security processing failed")
        
        return results["success"]
        
    except Exception as e:
        log_step_error(logger, "Security processing failed", {"error": str(e)})
        return False

def perform_security_check(file_path: Path, verbose: bool = False) -> Dict[str, Any]:
    """Perform security checks on a GNN file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Calculate file hash
        file_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Check for sensitive patterns
        sensitive_patterns = [
            r'password\s*[:=]',
            r'secret\s*[:=]',
            r'api_key\s*[:=]',
            r'token\s*[:=]',
            r'private_key\s*[:=]'
        ]
        
        found_patterns = []
        for pattern in sensitive_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                found_patterns.append({
                    "pattern": pattern,
                    "line": content[:match.start()].count('\n') + 1,
                    "context": match.group(0)
                })
        
        # Check file permissions (simplified)
        file_permissions = "readable"
        
        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_hash": file_hash,
            "file_size": file_path.stat().st_size,
            "sensitive_patterns": found_patterns,
            "file_permissions": file_permissions,
            "security_score": calculate_security_score(found_patterns),
            "check_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise Exception(f"Failed to perform security check on {file_path}: {e}")

def check_vulnerabilities(file_path: Path, verbose: bool = False) -> List[Dict[str, Any]]:
    """Check for security vulnerabilities in a GNN file."""
    vulnerabilities = []
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for common vulnerabilities
        vuln_patterns = [
            (r'eval\s*\(', "Code injection vulnerability"),
            (r'exec\s*\(', "Code execution vulnerability"),
            (r'import\s+os\s*', "OS command injection risk"),
            (r'subprocess\s*\.', "Subprocess execution risk"),
            (r'file\s*\(', "File operation risk"),
            (r'open\s*\(', "File operation risk")
        ]
        
        for pattern, description in vuln_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                vulnerabilities.append({
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "vulnerability_type": description,
                    "pattern": pattern,
                    "line": content[:match.start()].count('\n') + 1,
                    "context": match.group(0),
                    "severity": "medium" if "risk" in description else "high"
                })
        
        # Check for hardcoded credentials
        credential_patterns = [
            r'password\s*[:=]\s*["\'][^"\']+["\']',
            r'secret\s*[:=]\s*["\'][^"\']+["\']',
            r'api_key\s*[:=]\s*["\'][^"\']+["\']'
        ]
        
        for pattern in credential_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                vulnerabilities.append({
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "vulnerability_type": "Hardcoded credentials",
                    "pattern": pattern,
                    "line": content[:match.start()].count('\n') + 1,
                    "context": match.group(0),
                    "severity": "high"
                })
        
    except Exception as e:
        vulnerabilities.append({
            "file_path": str(file_path),
            "file_name": file_path.name,
            "vulnerability_type": "File access error",
            "error": str(e),
            "severity": "low"
        })
    
    return vulnerabilities

def generate_security_recommendations(file_path: Path, verbose: bool = False) -> List[Dict[str, Any]]:
    """Generate security recommendations for a GNN file."""
    recommendations = []
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for basic security practices
        if not re.search(r'#\s*Security', content, re.IGNORECASE):
            recommendations.append({
                "file_path": str(file_path),
                "file_name": file_path.name,
                "recommendation": "Add security documentation section",
                "priority": "medium",
                "description": "Consider adding a security section to document security considerations"
            })
        
        # Check for input validation
        if re.search(r'input\s*[:=]', content, re.IGNORECASE):
            if not re.search(r'validate|check|verify', content, re.IGNORECASE):
                recommendations.append({
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "recommendation": "Add input validation",
                    "priority": "high",
                    "description": "Input validation should be implemented for all user inputs"
                })
        
        # Check for error handling
        if not re.search(r'try\s*:|except\s*:', content, re.IGNORECASE):
            recommendations.append({
                "file_path": str(file_path),
                "file_name": file_path.name,
                "recommendation": "Add error handling",
                "priority": "medium",
                "description": "Implement proper error handling for robust security"
            })
        
        # Check for logging
        if not re.search(r'log|logging', content, re.IGNORECASE):
            recommendations.append({
                "file_path": str(file_path),
                "file_name": file_path.name,
                "recommendation": "Add security logging",
                "priority": "medium",
                "description": "Implement security event logging for monitoring"
            })
        
    except Exception as e:
        recommendations.append({
            "file_path": str(file_path),
            "file_name": file_path.name,
            "recommendation": "File access error",
            "priority": "low",
            "description": f"Could not analyze file: {e}"
        })
    
    return recommendations

def calculate_security_score(vulnerabilities: List[Dict]) -> float:
    """Calculate a security score based on vulnerabilities."""
    if not vulnerabilities:
        return 100.0
    
    # Weight vulnerabilities by severity
    severity_weights = {
        "high": 10.0,
        "medium": 5.0,
        "low": 1.0
    }
    
    total_score = 0.0
    for vuln in vulnerabilities:
        severity = vuln.get("severity", "medium")
        total_score += severity_weights.get(severity, 5.0)
    
    # Convert to 0-100 scale (higher is better)
    max_possible_score = len(vulnerabilities) * 10.0
    if max_possible_score == 0:
        return 100.0
    
    score = max(0.0, 100.0 - (total_score / max_possible_score) * 100.0)
    return score

def generate_security_summary(results: Dict[str, Any]) -> str:
    """Generate a security summary report."""
    summary = f"""
# Security Analysis Summary

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Processing Results
- **Files Processed**: {results.get('processed_files', 0)}
- **Success**: {results.get('success', False)}
- **Errors**: {len(results.get('errors', []))}

## Security Results
- **Security Checks**: {len(results.get('security_checks', []))}
- **Vulnerabilities Found**: {len(results.get('vulnerabilities', []))}
- **Recommendations**: {len(results.get('recommendations', []))}

## Vulnerability Summary
"""
    
    vulnerabilities = results.get('vulnerabilities', [])
    if vulnerabilities:
        high_vulns = [v for v in vulnerabilities if v.get('severity') == 'high']
        medium_vulns = [v for v in vulnerabilities if v.get('severity') == 'medium']
        low_vulns = [v for v in vulnerabilities if v.get('severity') == 'low']
        
        summary += f"- **High Severity**: {len(high_vulns)}\n"
        summary += f"- **Medium Severity**: {len(medium_vulns)}\n"
        summary += f"- **Low Severity**: {len(low_vulns)}\n"
        
        if high_vulns:
            summary += "\n### High Severity Vulnerabilities\n"
            for vuln in high_vulns[:5]:  # Show first 5
                summary += f"- **{vuln.get('file_name', 'Unknown')}**: {vuln.get('vulnerability_type', 'Unknown')}\n"
    else:
        summary += "- No vulnerabilities found\n"
    
    summary += "\n## Recommendations\n"
    
    recommendations = results.get('recommendations', [])
    if recommendations:
        high_recs = [r for r in recommendations if r.get('priority') == 'high']
        medium_recs = [r for r in recommendations if r.get('priority') == 'medium']
        
        if high_recs:
            summary += "\n### High Priority Recommendations\n"
            for rec in high_recs[:3]:  # Show first 3
                summary += f"- **{rec.get('file_name', 'Unknown')}**: {rec.get('recommendation', 'Unknown')}\n"
        
        if medium_recs:
            summary += "\n### Medium Priority Recommendations\n"
            for rec in medium_recs[:3]:  # Show first 3
                summary += f"- **{rec.get('file_name', 'Unknown')}**: {rec.get('recommendation', 'Unknown')}\n"
    else:
        summary += "- No recommendations generated\n"
    
    return summary
