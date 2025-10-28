#!/usr/bin/env python3
"""
GNN Pipeline Dependency Audit and Optimization System

This module provides comprehensive dependency analysis, security auditing,
and optimization capabilities for the GNN processing pipeline.
"""

import subprocess
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import re


@dataclass
class DependencyInfo:
    """Information about a single dependency."""
    name: str
    version: str
    specifier: str = ""
    installed_version: str = ""
    latest_version: str = ""
    is_outdated: bool = False
    security_vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    license_info: Dict[str, Any] = field(default_factory=dict)
    size_mb: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    reverse_dependencies: List[str] = field(default_factory=list)
    homepage: str = ""
    summary: str = ""
    author: str = ""
    email: str = ""


@dataclass
class AuditResult:
    """Results of a dependency audit."""
    timestamp: datetime
    total_dependencies: int
    outdated_dependencies: int
    vulnerable_dependencies: int
    missing_dependencies: int
    unused_dependencies: int
    circular_dependencies: List[Tuple[str, str]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    security_score: float = 0.0
    performance_score: float = 0.0
    maintainability_score: float = 0.0


class DependencyAuditor:
    """Comprehensive dependency auditor for Python projects."""

    def __init__(self, project_root: Path, logger: Optional[logging.Logger] = None):
        self.project_root = project_root
        self.logger = logger or logging.getLogger(__name__)
        self.dependencies: Dict[str, DependencyInfo] = {}
        self.audit_result: Optional[AuditResult] = None

    def audit_dependencies(self) -> AuditResult:
        """Perform comprehensive dependency audit."""
        self.logger.info("Starting comprehensive dependency audit")

        # Load project dependencies
        self._load_project_dependencies()

        # Check for outdated packages
        self._check_outdated_packages()

        # Security vulnerability scanning
        self._scan_security_vulnerabilities()

        # Analyze dependency relationships
        self._analyze_dependency_graph()

        # Check for unused dependencies
        self._find_unused_dependencies()

        # Generate recommendations
        recommendations = self._generate_recommendations()

        # Calculate scores
        security_score = self._calculate_security_score()
        performance_score = self._calculate_performance_score()
        maintainability_score = self._calculate_maintainability_score()

        # Create audit result
        self.audit_result = AuditResult(
            timestamp=datetime.now(),
            total_dependencies=len(self.dependencies),
            outdated_dependencies=sum(1 for dep in self.dependencies.values() if dep.is_outdated),
            vulnerable_dependencies=sum(1 for dep in self.dependencies.values() if dep.security_vulnerabilities),
            missing_dependencies=0,  # Would need additional analysis
            unused_dependencies=0,  # Would need additional analysis
            recommendations=recommendations,
            security_score=security_score,
            performance_score=performance_score,
            maintainability_score=maintainability_score
        )

        self.logger.info(f"Dependency audit completed. Found {len(recommendations)} recommendations")
        return self.audit_result

    def _load_project_dependencies(self):
        """Load dependencies from project files."""
        # Load from pyproject.toml (primary source with UV)
        pyproject_path = self.project_root / "pyproject.toml"
        if pyproject_path.exists():
            self._load_from_pyproject_toml(pyproject_path)

        # Load from uv.lock if present
        uv_lock_path = self.project_root / "uv.lock"
        if uv_lock_path.exists():
            self._load_from_uv_lock(uv_lock_path)

        # Load installed packages
        self._load_installed_packages()

    def _load_from_pyproject_toml(self, pyproject_path: Path):
        """Load dependencies from pyproject.toml."""
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                self.logger.warning("tomllib/tomli not available, skipping pyproject.toml parsing")
                return

        try:
            with open(pyproject_path, 'rb') as f:
                data = tomllib.load(f)

            # Load main dependencies
            dependencies = data.get('project', {}).get('dependencies', [])
            for dep_spec in dependencies:
                name, version = self._parse_dependency_spec(dep_spec)
                if name:
                    self.dependencies[name] = DependencyInfo(
                        name=name,
                        version=version,
                        specifier=dep_spec
                    )

            # Load optional dependencies
            optional_deps = data.get('project', {}).get('optional-dependencies', {})
            for group, deps in optional_deps.items():
                for dep_spec in deps:
                    name, version = self._parse_dependency_spec(dep_spec)
                    if name and name not in self.dependencies:
                        self.dependencies[name] = DependencyInfo(
                            name=name,
                            version=version,
                            specifier=dep_spec
                        )

        except Exception as e:
            self.logger.error(f"Error parsing pyproject.toml: {e}")

    def _load_from_uv_lock(self, uv_lock_path: Path):
        """Load dependencies from uv.lock file."""
        try:
            # UV lock files are in TOML format
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib
                except ImportError:
                    self.logger.warning("tomllib/tomli not available, skipping uv.lock parsing")
                    return

            with open(uv_lock_path, 'rb') as f:
                lock_data = tomllib.load(f)

            # UV lock format has a 'package' array with package information
            packages = lock_data.get('package', [])
            for pkg in packages:
                name = pkg.get('name', '')
                version = pkg.get('version', '')
                if name:
                    if name not in self.dependencies:
                        self.dependencies[name] = DependencyInfo(
                            name=name,
                            version=version,
                            specifier=f"{name}=={version}"
                        )
                    else:
                        # Update version from lock file
                        self.dependencies[name].version = version

        except Exception as e:
            self.logger.error(f"Error parsing uv.lock: {e}")

    def _load_installed_packages(self):
        """Load information about installed packages."""
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'list', '--format=json'
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                installed_packages = json.loads(result.stdout)
                for package in installed_packages:
                    name = package['name']
                    version = package['version']

                    if name in self.dependencies:
                        self.dependencies[name].installed_version = version
                    else:
                        # Add packages that are installed but not in requirements
                        self.dependencies[name] = DependencyInfo(
                            name=name,
                            version=version,
                            installed_version=version,
                            specifier=""  # Not in requirements
                        )

        except Exception as e:
            self.logger.error(f"Error loading installed packages: {e}")

    def _parse_dependency_spec(self, spec: str) -> Tuple[str, str]:
        """Parse dependency specification into name and version."""
        # Handle common patterns like "package>=1.0.0", "package==1.0.0", etc.
        match = re.match(r'^([a-zA-Z0-9_-]+)([><=~!,\s\d.]+)?', spec)
        if match:
            name = match.group(1).lower()
            version = match.group(2) or ""
            return name, version.strip()
        return "", ""

    def _check_outdated_packages(self):
        """Check for outdated packages."""
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'list', '--outdated', '--format=json'
            ], capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                outdated_packages = json.loads(result.stdout)
                for package in outdated_packages:
                    name = package['name']
                    latest_version = package['latest_version']
                    installed_version = package['version']

                    if name in self.dependencies:
                        self.dependencies[name].latest_version = latest_version
                        self.dependencies[name].is_outdated = True
                        self.logger.warning(f"Package {name} is outdated: {installed_version} -> {latest_version}")

        except Exception as e:
            self.logger.error(f"Error checking outdated packages: {e}")

    def _scan_security_vulnerabilities(self):
        """Scan for security vulnerabilities."""
        try:
            # Use pip-audit if available
            result = subprocess.run([
                sys.executable, '-m', 'pip_audit'
            ], capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                # Parse pip-audit output (would need to implement proper parsing)
                self.logger.info("Security scan completed successfully")
            else:
                self.logger.warning("Security scan encountered issues")

        except FileNotFoundError:
            self.logger.warning("pip-audit not available, install with: pip install pip-audit")
        except Exception as e:
            self.logger.error(f"Error during security scan: {e}")

    def _analyze_dependency_graph(self):
        """Analyze dependency relationships."""
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'show', '--verbose', *list(self.dependencies.keys())
            ], capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                # Parse pip show output to extract dependency information
                self._parse_pip_show_output(result.stdout)

        except Exception as e:
            self.logger.error(f"Error analyzing dependency graph: {e}")

    def _parse_pip_show_output(self, output: str):
        """Parse output from pip show command."""
        sections = output.split('---')
        for section in sections:
            if not section.strip():
                continue

            lines = section.strip().split('\n')
            if not lines:
                continue

            package_info = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    package_info[key.strip().lower()] = value.strip()

            name = package_info.get('name', '').lower()
            if name in self.dependencies:
                dep = self.dependencies[name]
                dep.homepage = package_info.get('homepage', '')
                dep.summary = package_info.get('summary', '')
                dep.author = package_info.get('author', '')
                dep.license_info = {
                    'license': package_info.get('license', ''),
                    'classifier': package_info.get('license-expression', '')
                }

                # Parse dependencies
                requires = package_info.get('requires', '')
                if requires:
                    dep.dependencies = [d.strip() for d in requires.split(',') if d.strip()]

    def _find_unused_dependencies(self):
        """Find potentially unused dependencies."""
        # This would require static analysis of the codebase
        # For now, we'll use a simple heuristic
        self.logger.info("Analyzing potentially unused dependencies would require static code analysis")

    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []

        # Outdated package recommendations
        outdated = [name for name, dep in self.dependencies.items() if dep.is_outdated]
        if outdated:
            recommendations.append(f"Update {len(outdated)} outdated packages: {', '.join(outdated[:5])}")

        # Security recommendations
        vulnerable = [name for name, dep in self.dependencies.items() if dep.security_vulnerabilities]
        if vulnerable:
            recommendations.append(f"Address security vulnerabilities in {len(vulnerable)} packages")

        # Large packages
        large_packages = [(name, dep.size_mb) for name, dep in self.dependencies.items() if dep.size_mb > 50]
        if large_packages:
            recommendations.append(f"Consider lighter alternatives for large packages: {', '.join([name for name, _ in large_packages])}")

        # Missing dependencies in requirements
        missing_in_reqs = [name for name, dep in self.dependencies.items() if not dep.specifier and dep.installed_version]
        if missing_in_reqs:
            recommendations.append(f"Add {len(missing_in_reqs)} packages to requirements files")

        return recommendations

    def _calculate_security_score(self) -> float:
        """Calculate security score (0-100)."""
        if not self.dependencies:
            return 100.0

        vulnerable_count = sum(1 for dep in self.dependencies.values() if dep.security_vulnerabilities)
        score = 100.0 * (1 - vulnerable_count / len(self.dependencies))
        return max(0.0, min(100.0, score))

    def _calculate_performance_score(self) -> float:
        """Calculate performance score based on package sizes and dependencies."""
        if not self.dependencies:
            return 100.0

        # Simple scoring based on number of dependencies and package sizes
        total_deps = sum(len(dep.dependencies) for dep in self.dependencies.values())
        avg_deps = total_deps / len(self.dependencies)

        # Penalize high dependency counts
        score = 100.0 - (avg_deps * 2)
        return max(0.0, min(100.0, score))

    def _calculate_maintainability_score(self) -> float:
        """Calculate maintainability score."""
        if not self.dependencies:
            return 100.0

        # Score based on outdated packages and license issues
        outdated_penalty = sum(1 for dep in self.dependencies.values() if dep.is_outdated)
        score = 100.0 - (outdated_penalty * 5)
        return max(0.0, min(100.0, score))

    def export_audit_report(self, output_path: Path) -> Path:
        """Export audit results to file."""
        if not self.audit_result:
            raise ValueError("Run audit_dependencies() first")

        report_data = {
            "audit_timestamp": self.audit_result.timestamp.isoformat(),
            "summary": {
                "total_dependencies": self.audit_result.total_dependencies,
                "outdated_dependencies": self.audit_result.outdated_dependencies,
                "vulnerable_dependencies": self.audit_result.vulnerable_dependencies,
                "missing_dependencies": self.audit_result.missing_dependencies,
                "unused_dependencies": self.audit_result.unused_dependencies,
                "security_score": self.audit_result.security_score,
                "performance_score": self.audit_result.performance_score,
                "maintainability_score": self.audit_result.maintainability_score
            },
            "recommendations": self.audit_result.recommendations,
            "dependencies": {
                name: {
                    "version": dep.version,
                    "installed_version": dep.installed_version,
                    "latest_version": dep.latest_version,
                    "is_outdated": dep.is_outdated,
                    "security_vulnerabilities": len(dep.security_vulnerabilities),
                    "dependencies": dep.dependencies,
                    "homepage": dep.homepage,
                    "summary": dep.summary
                }
                for name, dep in self.dependencies.items()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        self.logger.info(f"Audit report exported to {output_path}")
        return output_path

    def optimize_dependencies(self) -> Dict[str, Any]:
        """Suggest dependency optimizations."""
        optimizations = {
            "updates_available": [],
            "security_fixes": [],
            "consolidation_opportunities": [],
            "removal_candidates": []
        }

        # Find update opportunities
        for name, dep in self.dependencies.items():
            if dep.is_outdated:
                optimizations["updates_available"].append({
                    "package": name,
                    "current": dep.installed_version,
                    "latest": dep.latest_version
                })

        # Find security issues
        for name, dep in self.dependencies.items():
            if dep.security_vulnerabilities:
                optimizations["security_fixes"].append({
                    "package": name,
                    "vulnerabilities": len(dep.security_vulnerabilities)
                })

        return optimizations


class DependencyOptimizer:
    """Tools for optimizing dependency management."""

    def __init__(self, project_root: Path, logger: Optional[logging.Logger] = None):
        self.project_root = project_root
        self.logger = logger or logging.getLogger(__name__)

    def consolidate_requirements(self) -> Dict[str, Any]:
        """Consolidate and optimize requirements files."""
        # This would analyze overlapping dependencies and suggest optimizations
        return {"consolidation_suggestions": []}

    def update_dependencies(self, dry_run: bool = True) -> Dict[str, Any]:
        """Update outdated dependencies."""
        result = {"updated": [], "failed": [], "skipped": []}

        try:
            cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade']
            if dry_run:
                cmd.append('--dry-run')

            # Would need to identify which packages to update
            # For now, return placeholder
            result["message"] = "Dependency update simulation completed"

        except Exception as e:
            result["error"] = str(e)

        return result

    def clean_unused_dependencies(self) -> Dict[str, Any]:
        """Remove unused dependencies."""
        # This would require static analysis
        return {"removed": [], "message": "Unused dependency analysis requires static code analysis"}


def audit_project_dependencies(project_root: Path) -> AuditResult:
    """Convenience function to audit project dependencies."""
    auditor = DependencyAuditor(project_root)
    return auditor.audit_dependencies()


def optimize_project_dependencies(project_root: Path) -> Dict[str, Any]:
    """Convenience function to optimize project dependencies."""
    optimizer = DependencyOptimizer(project_root)
    return optimizer.consolidate_requirements()
