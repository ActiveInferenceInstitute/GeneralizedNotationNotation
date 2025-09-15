"""
Dependency Installer for GNN Pipeline

This module handles the installation of missing dependencies including:
- Julia runtime and packages
- PyMDP and other Python packages
- System-level dependencies
- API key configuration guidance
"""

import os
import subprocess
import sys
import platform
import shutil
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

logger = logging.getLogger(__name__)

class DependencyInstaller:
    """Handles installation of missing dependencies for the GNN pipeline."""
    
    def __init__(self, project_root: Path, verbose: bool = False):
        self.project_root = project_root
        self.verbose = verbose
        self.install_log = []
        
    def log_install(self, message: str, level: str = "INFO"):
        """Log installation messages with timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.install_log.append(log_entry)
        
        if level == "ERROR":
            logger.error(message)
        elif level == "WARNING":
            logger.warning(message)
        else:
            logger.info(message)
    
    def check_command(self, command: str) -> bool:
        """Check if a command is available in PATH."""
        try:
            subprocess.run([command, "--version"], 
                         capture_output=True, check=True, timeout=10)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def run_command(self, command: List[str], cwd: Optional[Path] = None, 
                   check: bool = True, timeout: int = 300) -> subprocess.CompletedProcess:
        """Run a command with proper error handling and logging."""
        if cwd is None:
            cwd = self.project_root
            
        self.log_install(f"Running: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command, 
                cwd=cwd, 
                check=check, 
                capture_output=True, 
                text=True, 
                timeout=timeout
            )
            
            if result.stdout and self.verbose:
                self.log_install(f"STDOUT: {result.stdout.strip()}")
            if result.stderr and self.verbose:
                self.log_install(f"STDERR: {result.stderr.strip()}")
                
            return result
        except subprocess.TimeoutExpired:
            self.log_install(f"Command timed out after {timeout}s: {' '.join(command)}", "ERROR")
            raise
        except subprocess.CalledProcessError as e:
            self.log_install(f"Command failed with code {e.returncode}: {' '.join(command)}", "ERROR")
            if e.stdout:
                self.log_install(f"STDOUT: {e.stdout.strip()}")
            if e.stderr:
                self.log_install(f"STDERR: {e.stderr.strip()}")
            raise
    
    def install_julia(self) -> bool:
        """Install Julia runtime if not available."""
        if self.check_command("julia"):
            self.log_install("Julia is already installed")
            return True
            
        self.log_install("Julia not found. Installing Julia...")
        
        try:
            if platform.system() == "Linux":
                return self._install_julia_linux()
            elif platform.system() == "Darwin":
                return self._install_julia_macos()
            elif platform.system() == "Windows":
                return self._install_julia_windows()
            else:
                self.log_install(f"Unsupported platform: {platform.system()}", "ERROR")
                return False
        except Exception as e:
            self.log_install(f"Failed to install Julia: {e}", "ERROR")
            return False
    
    def _install_julia_linux(self) -> bool:
        """Install Julia on Linux using the official installer."""
        try:
            # Download and install Julia
            self.log_install("Downloading Julia installer...")
            
            # Use the official Julia installer script
            install_script = """
            set -e
            JULIA_VERSION="1.10.2"
            JULIA_URL="https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-${JULIA_VERSION}-linux-x86_64.tar.gz"
            JULIA_DIR="/opt/julia"
            
            # Create directory
            sudo mkdir -p $JULIA_DIR
            
            # Download and extract
            cd /tmp
            wget -q $JULIA_URL
            tar -xzf julia-${JULIA_VERSION}-linux-x86_64.tar.gz
            sudo mv julia-${JULIA_VERSION}/* $JULIA_DIR/
            
            # Create symlink
            sudo ln -sf $JULIA_DIR/bin/julia /usr/local/bin/julia
            
            # Cleanup
            rm -rf julia-${JULIA_VERSION}*
            
            echo "Julia installed successfully"
            """
            
            # Write and execute script
            script_path = self.project_root / "install_julia.sh"
            with open(script_path, 'w') as f:
                f.write(install_script)
            
            os.chmod(script_path, 0o755)
            self.run_command(["bash", str(script_path)])
            
            # Cleanup
            script_path.unlink()
            
            # Verify installation
            if self.check_command("julia"):
                self.log_install("Julia installed successfully")
                return True
            else:
                self.log_install("Julia installation verification failed", "ERROR")
                return False
                
        except Exception as e:
            self.log_install(f"Failed to install Julia on Linux: {e}", "ERROR")
            return False
    
    def _install_julia_macos(self) -> bool:
        """Install Julia on macOS using Homebrew."""
        try:
            if not self.check_command("brew"):
                self.log_install("Homebrew not found. Please install Homebrew first.", "ERROR")
                return False
            
            self.run_command(["brew", "install", "julia"])
            
            if self.check_command("julia"):
                self.log_install("Julia installed successfully via Homebrew")
                return True
            else:
                self.log_install("Julia installation verification failed", "ERROR")
                return False
                
        except Exception as e:
            self.log_install(f"Failed to install Julia on macOS: {e}", "ERROR")
            return False
    
    def _install_julia_windows(self) -> bool:
        """Install Julia on Windows using winget or direct download."""
        try:
            # Try winget first
            if self.check_command("winget"):
                self.run_command(["winget", "install", "Julia.Julia"])
            else:
                # Fallback to direct download
                self.log_install("winget not available. Please install Julia manually from https://julialang.org/downloads/", "WARNING")
                return False
            
            if self.check_command("julia"):
                self.log_install("Julia installed successfully")
                return True
            else:
                self.log_install("Julia installation verification failed", "ERROR")
                return False
                
        except Exception as e:
            self.log_install(f"Failed to install Julia on Windows: {e}", "ERROR")
            return False
    
    def install_pymdp(self) -> bool:
        """Install PyMDP package."""
        try:
            self.log_install("Installing PyMDP...")
            
            # Try installing via pip first
            try:
                self.run_command([sys.executable, "-m", "pip", "install", "pymdp"])
                self.log_install("PyMDP installed successfully via pip")
                return True
            except subprocess.CalledProcessError:
                pass
            
            # Try installing from source
            try:
                self.run_command([
                    sys.executable, "-m", "pip", "install", 
                    "git+https://github.com/infer-actively/pymdp.git"
                ])
                self.log_install("PyMDP installed successfully from source")
                return True
            except subprocess.CalledProcessError:
                pass
            
            self.log_install("Failed to install PyMDP. Please install manually.", "WARNING")
            return False
            
        except Exception as e:
            self.log_install(f"Failed to install PyMDP: {e}", "ERROR")
            return False
    
    def install_missing_python_packages(self) -> bool:
        """Install missing Python packages."""
        try:
            self.log_install("Installing missing Python packages...")
            
            # List of packages that might be missing
            packages = [
                "aiohttp>=3.9.0",  # For LLM providers
                "ollama",           # For local LLM
                "pymdp",            # For Active Inference
            ]
            
            for package in packages:
                try:
                    self.run_command([sys.executable, "-m", "pip", "install", package])
                    self.log_install(f"Installed {package}")
                except subprocess.CalledProcessError as e:
                    self.log_install(f"Failed to install {package}: {e}", "WARNING")
            
            return True
            
        except Exception as e:
            self.log_install(f"Failed to install Python packages: {e}", "ERROR")
            return False
    
    def setup_api_key_guidance(self) -> None:
        """Provide clear guidance on API key setup."""
        self.log_install("=" * 60)
        self.log_install("API KEY CONFIGURATION GUIDANCE")
        self.log_install("=" * 60)
        
        # Create API key template file
        api_key_template = self.project_root / "api_keys_template.env"
        with open(api_key_template, 'w') as f:
            f.write("""# API Keys Configuration Template
# Copy this file to .env and fill in your API keys
# Never commit .env files to version control!

# OpenAI API Key (for GPT models)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key (for Claude models)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# OpenRouter API Key (for multiple providers)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Perplexity API Key
PERPLEXITY_API_KEY=your_perplexity_api_key_here

# Ollama Configuration (for local models)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# Environment Variables
export OPENAI_API_KEY="your_openai_api_key_here"
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
export OPENROUTER_API_KEY="your_openrouter_api_key_here"
export PERPLEXITY_API_KEY="your_perplexity_api_key_here"
""")
        
        self.log_install(f"Created API key template: {api_key_template}")
        
        # Create .env file if it doesn't exist
        env_file = self.project_root / ".env"
        if not env_file.exists():
            env_file.write_text("# Add your API keys here\n")
            self.log_install(f"Created .env file: {env_file}")
        
        # Provide instructions
        self.log_install("")
        self.log_install("To configure API keys for LLM providers:")
        self.log_install("1. Copy api_keys_template.env to .env")
        self.log_install("2. Edit .env and add your actual API keys")
        self.log_install("3. Or set environment variables:")
        self.log_install("   export OPENAI_API_KEY='your_key_here'")
        self.log_install("   export ANTHROPIC_API_KEY='your_key_here'")
        self.log_install("   export OPENROUTER_API_KEY='your_key_here'")
        self.log_install("   export PERPLEXITY_API_KEY='your_key_here'")
        self.log_install("")
        self.log_install("For local LLM (Ollama):")
        self.log_install("1. Install Ollama: https://ollama.ai")
        self.log_install("2. Pull a model: ollama pull llama2")
        self.log_install("3. Set OLLAMA_BASE_URL=http://localhost:11434")
        self.log_install("")
        self.log_install("Security best practices:")
        self.log_install("- Never commit .env files to version control")
        self.log_install("- Use environment variables in production")
        self.log_install("- Rotate API keys regularly")
        self.log_install("- Use least privilege principle")
        self.log_install("=" * 60)
    
    def install_all_dependencies(self) -> Dict[str, bool]:
        """Install all missing dependencies."""
        results = {}
        
        self.log_install("Starting dependency installation...")
        
        # Install Julia
        results['julia'] = self.install_julia()
        
        # Install PyMDP
        results['pymdp'] = self.install_pymdp()
        
        # Install missing Python packages
        results['python_packages'] = self.install_missing_python_packages()
        
        # Setup API key guidance
        self.setup_api_key_guidance()
        
        # Save installation log
        log_file = self.project_root / "dependency_installation.log"
        with open(log_file, 'w') as f:
            f.write('\n'.join(self.install_log))
        
        self.log_install(f"Installation log saved to: {log_file}")
        
        return results
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check which dependencies are available."""
        return {
            'julia': self.check_command('julia'),
            'pymdp': self._check_python_package('pymdp'),
            'aiohttp': self._check_python_package('aiohttp'),
            'ollama': self._check_python_package('ollama'),
        }
    
    def _check_python_package(self, package_name: str) -> bool:
        """Check if a Python package is installed."""
        try:
            __import__(package_name)
            return True
        except ImportError:
            return False
