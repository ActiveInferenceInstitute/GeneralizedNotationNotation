#!/usr/bin/env python3
"""
Comprehensive Dependency Installer for GNN Pipeline

This module automatically installs missing optional dependencies to eliminate
warnings and achieve full pipeline functionality.
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import importlib

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils.pipeline_dependencies import get_pipeline_dependency_manager
except ImportError:
    print("Warning: Could not import pipeline dependency manager")


class DependencyInstaller:
    """Comprehensive dependency installer for the GNN pipeline."""
    
    def __init__(self, use_uv: bool = True, verbose: bool = True):
        self.use_uv = use_uv
        self.verbose = verbose
        self.logger = self._setup_logging()
        self.dependency_manager = get_pipeline_dependency_manager()
        
        # Define optional dependencies that should be installed to eliminate warnings
        self.install_targets = {
            "visualization": {
                "seaborn": "Enhanced statistical visualizations",
                "networkx": "Network and graph visualizations", 
                "plotly": "Interactive plotting capabilities"
            },
            "audio": {
                "librosa": "Advanced audio analysis",
                "soundfile": "Audio file I/O operations",
                "pedalboard": "Audio effects processing",
                "pydub": "Audio manipulation utilities"
            },
            "execution": {
                "pymdp": "PyMDP Active Inference simulations",
                "jax": "JAX-based high-performance computing",
                "flax": "Neural network library for JAX",
                "discopy": "Categorical diagrams and monoidal categories"
            },
            "scientific": {
                "scipy": "Scientific computing utilities",
                "sympy": "Symbolic mathematics",
                "numpy": "Numerical computing (should already be installed)"
            },
            "development": {
                "pytest": "Testing framework",
                "black": "Code formatting",
                "mypy": "Static type checking"
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the installer."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        return logger
    
    def check_missing_dependencies(self) -> Dict[str, List[str]]:
        """Check which dependencies are missing for each category."""
        missing = {}
        
        for category, deps in self.install_targets.items():
            missing_in_category = []
            for dep_name, description in deps.items():
                try:
                    importlib.import_module(dep_name)
                    if self.verbose:
                        self.logger.info(f"✅ {dep_name} already available")
                except ImportError:
                    missing_in_category.append(dep_name)
                    if self.verbose:
                        self.logger.warning(f"❌ {dep_name} missing: {description}")
            
            if missing_in_category:
                missing[category] = missing_in_category
        
        return missing
    
    def install_dependency(self, package_name: str, category: str = "") -> bool:
        """
        Install a single dependency using uv.
        
        Args:
            package_name: Name of the package to install
            category: Category for context (optional)
            
        Returns:
            True if installation successful, False otherwise
        """
        try:
            if self.use_uv:
                cmd = ["uv", "pip", "install", package_name]
            else:
                cmd = [sys.executable, "-m", "pip", "install", package_name]
            
            self.logger.info(f"🔧 Installing {package_name} ({category})...")
            
            # Handle special cases that need different package names
            special_packages = {
                "discopy": "discopy[all]",  # Install with all optional features
                "jax": "jax[cpu]",  # Install CPU version by default
                "flax": "flax",
                "librosa": "librosa",
                "soundfile": "soundfile",
                "pedalboard": "pedalboard",
                "pydub": "pydub",
                "pymdp": "pymdp",
                "seaborn": "seaborn",
                "networkx": "networkx[default]",
                "plotly": "plotly",
                "scipy": "scipy",
                "sympy": "sympy"
            }
            
            actual_package = special_packages.get(package_name, package_name)
            cmd[-1] = actual_package
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per package
            )
            
            if result.returncode == 0:
                self.logger.info(f"✅ Successfully installed {package_name}")
                return True
            else:
                self.logger.error(f"❌ Failed to install {package_name}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"⏰ Installation of {package_name} timed out")
            return False
        except Exception as e:
            self.logger.error(f"💥 Error installing {package_name}: {e}")
            return False
    
    def install_category(self, category: str, missing_deps: List[str]) -> Dict[str, bool]:
        """Install all missing dependencies in a category."""
        results = {}
        
        self.logger.info(f"📦 Installing {category} dependencies...")
        
        for dep in missing_deps:
            success = self.install_dependency(dep, category)
            results[dep] = success
            
            # Clear import cache so we can re-check
            if success and dep in sys.modules:
                del sys.modules[dep]
        
        return results
    
    def install_all_missing(self, categories: List[str] = None) -> Dict[str, Dict[str, bool]]:
        """
        Install all missing dependencies.
        
        Args:
            categories: List of categories to install, or None for all
            
        Returns:
            Dictionary mapping categories to installation results
        """
        missing = self.check_missing_dependencies()
        
        if not missing:
            self.logger.info("🎉 All dependencies already installed!")
            return {}
        
        if categories:
            missing = {cat: deps for cat, deps in missing.items() if cat in categories}
        
        self.logger.info("🚀 Starting comprehensive dependency installation...")
        self.logger.info(f"📋 Categories to install: {list(missing.keys())}")
        
        all_results = {}
        
        # Install in order of importance
        install_order = ["scientific", "visualization", "execution", "audio", "development"]
        
        for category in install_order:
            if category in missing:
                category_results = self.install_category(category, missing[category])
                all_results[category] = category_results
        
        # Install remaining categories not in the ordered list
        for category, deps in missing.items():
            if category not in all_results:
                category_results = self.install_category(category, deps)
                all_results[category] = category_results
        
        return all_results
    
    def verify_installations(self) -> Dict[str, bool]:
        """Verify that all installations were successful."""
        verification_results = {}
        
        self.logger.info("🔍 Verifying installations...")
        
        for category, deps in self.install_targets.items():
            for dep_name in deps.keys():
                try:
                    importlib.import_module(dep_name)
                    verification_results[dep_name] = True
                    self.logger.info(f"✅ {dep_name} verified")
                except ImportError:
                    verification_results[dep_name] = False
                    self.logger.warning(f"❌ {dep_name} verification failed")
        
        return verification_results
    
    def generate_installation_report(self, installation_results: Dict[str, Dict[str, bool]]) -> Dict:
        """Generate comprehensive installation report."""
        report = {
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "installer_version": "1.0.0",
            "package_manager": "uv" if self.use_uv else "pip",
            "categories": installation_results,
            "summary": {
                "total_categories": len(installation_results),
                "total_packages": sum(len(deps) for deps in installation_results.values()),
                "successful_packages": sum(
                    sum(1 for success in deps.values() if success) 
                    for deps in installation_results.values()
                ),
                "failed_packages": sum(
                    sum(1 for success in deps.values() if not success) 
                    for deps in installation_results.values()
                )
            }
        }
        
        # Calculate success rate
        if report["summary"]["total_packages"] > 0:
            report["summary"]["success_rate"] = (
                report["summary"]["successful_packages"] / 
                report["summary"]["total_packages"] * 100
            )
        else:
            report["summary"]["success_rate"] = 100.0
        
        return report
    
    def save_report(self, report: Dict, output_path: Path = None) -> Path:
        """Save installation report to file."""
        if output_path is None:
            output_path = Path("output/dependency_installation_report.json")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"📄 Installation report saved to: {output_path}")
        return output_path


def main():
    """Main function to install all missing dependencies."""
    installer = DependencyInstaller(use_uv=True, verbose=True)
    
    print("🔧 GNN Pipeline Dependency Installer")
    print("=" * 50)
    
    # Check what's missing
    missing = installer.check_missing_dependencies()
    
    if not missing:
        print("🎉 All dependencies already installed!")
        return 0
    
    print(f"\n📋 Missing dependencies found in {len(missing)} categories:")
    for category, deps in missing.items():
        print(f"  {category}: {', '.join(deps)}")
    
    # Prompt for confirmation
    response = input("\n🤔 Install all missing dependencies? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Installation cancelled by user")
        return 1
    
    # Install all missing dependencies
    installation_results = installer.install_all_missing()
    
    # Verify installations
    verification_results = installer.verify_installations()
    
    # Generate and save report
    report = installer.generate_installation_report(installation_results)
    report["verification_results"] = verification_results
    
    report_path = installer.save_report(report)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 INSTALLATION SUMMARY")
    print("=" * 50)
    print(f"Total packages: {report['summary']['total_packages']}")
    print(f"Successful: {report['summary']['successful_packages']}")
    print(f"Failed: {report['summary']['failed_packages']}")
    print(f"Success rate: {report['summary']['success_rate']:.1f}%")
    print(f"Report saved: {report_path}")
    
    # Verify at least the critical ones are installed
    critical_deps = ["numpy", "matplotlib", "pathlib"]
    all_critical_ok = all(verification_results.get(dep, False) for dep in critical_deps)
    
    if report['summary']['success_rate'] >= 80 and all_critical_ok:
        print("🌟 Installation completed successfully!")
        return 0
    else:
        print("⚠️ Some installations failed - pipeline may still have warnings")
        return 1


if __name__ == "__main__":
    sys.exit(main())

