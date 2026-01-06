#!/usr/bin/env python3
"""
Comprehensive Filepath and Reference Audit Script

This script systematically audits all filepaths, signposts, folder structures,
and file references across the GNN codebase for accuracy and completeness.
"""

import sys
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import ast

# Add src to path (now in src/pipeline/, so go up two levels to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

class FilepathAuditor:
    """Comprehensive filepath and reference auditor"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.issues = {
            "broken_links": [],
            "missing_files": [],
            "incorrect_paths": [],
            "outdated_references": [],
            "import_errors": [],
            "output_dir_mismatches": [],
            "script_name_errors": []
        }
        self.fixes_applied = []
        
        # Expected numbered scripts (0-23)
        self.expected_scripts = {
            f"{i}_{name}.py": i for i, name in [
                (0, "template"), (1, "setup"), (2, "tests"), (3, "gnn"),
                (4, "model_registry"), (5, "type_checker"), (6, "validation"),
                (7, "export"), (8, "visualization"), (9, "advanced_viz"),
                (10, "ontology"), (11, "render"), (12, "execute"),
                (13, "llm"), (14, "ml_integration"), (15, "audio"),
                (16, "analysis"), (17, "integration"), (18, "security"),
                (19, "research"), (20, "website"), (21, "mcp"),
                (22, "gui"), (23, "report")
            ]
        }
        
        # Expected output directories
        self.expected_output_dirs = {
            f"{i}_{name}_output": i for i, name in [
                (0, "template"), (1, "setup"), (2, "tests"), (3, "gnn"),
                (4, "model_registry"), (5, "type_checker"), (6, "validation"),
                (7, "export"), (8, "visualization"), (9, "advanced_viz"),
                (10, "ontology"), (11, "render"), (12, "execute"),
                (13, "llm"), (14, "ml_integration"), (15, "audio"),
                (16, "analysis"), (17, "integration"), (18, "security"),
                (19, "research"), (20, "website"), (21, "mcp"),
                (22, "gui"), (23, "report")
            ]
        }
        
    def scan_markdown_links(self, file_path: Path) -> List[Tuple[str, str, int]]:
        """Extract all markdown links from a file"""
        links = []
        try:
            content = file_path.read_text(encoding='utf-8')
            # Match markdown links: [text](path)
            pattern = r'\[([^\]]+)\]\(([^)]+)\)'
            for match in re.finditer(pattern, content):
                link_text = match.group(1)
                link_path = match.group(2)
                line_num = content[:match.start()].count('\n') + 1
                links.append((link_text, link_path, line_num))
        except Exception as e:
            self.issues["broken_links"].append(f"Error reading {file_path}: {e}")
        return links
    
    def validate_markdown_link(self, source_file: Path, link_path: str, line_num: int) -> bool:
        """Validate that a markdown link resolves to an existing file"""
        # Skip external links
        if link_path.startswith(('http://', 'https://', 'mailto:')):
            return True
        
        # Handle anchor links
        if link_path.startswith('#'):
            return True
        
        # Resolve relative path
        if link_path.startswith('/'):
            target = self.project_root / link_path.lstrip('/')
        else:
            target = (source_file.parent / link_path).resolve()
        
        # Only validate paths within project root
        try:
            target_rel = target.relative_to(self.project_root)
        except ValueError:
            # Path is outside project root, skip validation
            return True
        
        # Check if file exists
        if not target.exists():
            # Check if it's a directory (might be valid)
            if not target.is_dir():
                self.issues["missing_files"].append({
                    "source": str(source_file.relative_to(self.project_root)),
                    "line": line_num,
                    "link": link_path,
                    "resolved": str(target_rel)
                })
                return False
        return True
    
    def scan_python_imports(self, file_path: Path) -> List[Tuple[str, int, str]]:
        """Extract all import statements from a Python file"""
        imports = []
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content, filename=str(file_path))
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append((alias.name, node.lineno, "import"))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append((f"{module}.{alias.name}", node.lineno, "from"))
        except Exception as e:
            self.issues["import_errors"].append(f"Error parsing {file_path}: {e}")
        return imports
    
    def validate_output_directory_reference(self, file_path: Path, ref: str) -> bool:
        """Validate output directory references match standard pattern"""
        # Check for output directory patterns
        pattern = r'output/(\d+)_([a-z_]+)_output'
        match = re.search(pattern, ref)
        if match:
            step_num = int(match.group(1))
            dir_name = match.group(2)
            expected_name = self.expected_output_dirs.get(f"{step_num}_{dir_name}_output")
            
            if expected_name is None or expected_name != step_num:
                self.issues["output_dir_mismatches"].append({
                    "file": str(file_path.relative_to(self.project_root)),
                    "reference": ref,
                    "expected": f"output/{step_num}_{dir_name}_output",
                    "found": ref
                })
                return False
        return True
    
    def validate_script_reference(self, file_path: Path, ref: str) -> bool:
        """Validate numbered script references"""
        # Check for script references like "3_gnn.py" or "src/3_gnn.py"
        pattern = r'(\d+)_([a-z_]+)\.py'
        match = re.search(pattern, ref)
        if match:
            step_num = int(match.group(1))
            script_name = match.group(2)
            expected_script = f"{step_num}_{script_name}.py"
            
            # Check if script exists
            script_path = self.project_root / "src" / expected_script
            if not script_path.exists():
                self.issues["script_name_errors"].append({
                    "file": str(file_path.relative_to(self.project_root)),
                    "reference": ref,
                    "expected": expected_script,
                    "exists": False
                })
                return False
            
            # Check for outdated references (old step numbers)
            if step_num < 0 or step_num > 23:
                self.issues["outdated_references"].append({
                    "file": str(file_path.relative_to(self.project_root)),
                    "reference": ref,
                    "issue": f"Step number {step_num} out of range (0-23)"
                })
                return False
        return True
    
    def scan_file(self, file_path: Path):
        """Scan a single file for all reference issues"""
        if file_path.suffix == '.md':
            # Scan markdown links
            links = self.scan_markdown_links(file_path)
            for link_text, link_path, line_num in links:
                self.validate_markdown_link(file_path, link_path, line_num)
                # Check for output directory references
                if 'output/' in link_path:
                    self.validate_output_directory_reference(file_path, link_path)
                # Check for script references
                if '.py' in link_path:
                    self.validate_script_reference(file_path, link_path)
        
        elif file_path.suffix == '.py':
            # Scan Python imports
            imports = self.scan_python_imports(file_path)
            # Check for output directory references in strings/comments
            try:
                content = file_path.read_text(encoding='utf-8')
                # Find output directory references
                output_pattern = r'output/(\d+)_([a-z_]+)_output'
                for match in re.finditer(output_pattern, content):
                    self.validate_output_directory_reference(file_path, match.group(0))
                # Find script references
                script_pattern = r'(\d+)_([a-z_]+)\.py'
                for match in re.finditer(script_pattern, content):
                    self.validate_script_reference(file_path, match.group(0))
            except Exception as e:
                pass
    
    def verify_numbered_scripts(self):
        """Verify all numbered scripts 0-23 exist"""
        src_dir = self.project_root / "src"
        for script_name, step_num in self.expected_scripts.items():
            script_path = src_dir / script_name
            if not script_path.exists():
                self.issues["missing_files"].append({
                    "type": "numbered_script",
                    "expected": script_name,
                    "step": step_num,
                    "path": str(script_path.relative_to(self.project_root))
                })
    
    def verify_output_directories(self):
        """Verify output directory structure matches expected pattern"""
        output_dir = self.project_root / "output"
        if not output_dir.exists():
            return
        
        for dir_name, step_num in self.expected_output_dirs.items():
            dir_path = output_dir / dir_name
            # Note: directories might not exist if pipeline hasn't run, so this is just a check
    
    def scan_all_files(self):
        """Scan all relevant files in the project"""
        print("Scanning markdown files...")
        md_files = list(self.project_root.rglob("*.md"))
        print(f"Found {len(md_files)} markdown files")
        
        print("Scanning Python files...")
        py_files = list((self.project_root / "src").rglob("*.py"))
        print(f"Found {len(py_files)} Python files")
        
        print("Verifying numbered scripts...")
        self.verify_numbered_scripts()
        
        print("Scanning files for references...")
        all_files = md_files + py_files
        for i, file_path in enumerate(all_files):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(all_files)}")
            # Skip certain directories
            if any(skip in str(file_path) for skip in ['__pycache__', '.git', 'node_modules', '.venv']):
                continue
            self.scan_file(file_path)
    
    def generate_report(self) -> Dict:
        """Generate comprehensive audit report"""
        total_issues = sum(len(issues) for issues in self.issues.values())
        
        return {
            "summary": {
                "total_issues": total_issues,
                "broken_links": len(self.issues["broken_links"]),
                "missing_files": len(self.issues["missing_files"]),
                "incorrect_paths": len(self.issues["incorrect_paths"]),
                "outdated_references": len(self.issues["outdated_references"]),
                "import_errors": len(self.issues["import_errors"]),
                "output_dir_mismatches": len(self.issues["output_dir_mismatches"]),
                "script_name_errors": len(self.issues["script_name_errors"])
            },
            "issues": self.issues,
            "fixes_applied": self.fixes_applied
        }
    
    def save_report(self, output_path: Path):
        """Save audit report to file"""
        report = self.generate_report()
        output_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f"\nAudit report saved to: {output_path}")

def main():
    """Main execution"""
    project_root = Path(__file__).parent.parent.parent
    auditor = FilepathAuditor(project_root)
    
    print("=" * 60)
    print("GNN Filepath and Reference Audit")
    print("=" * 60)
    
    auditor.scan_all_files()
    
    report = auditor.generate_report()
    print("\n" + "=" * 60)
    print("Audit Summary")
    print("=" * 60)
    print(f"Total issues found: {report['summary']['total_issues']}")
    print(f"  - Broken links: {report['summary']['broken_links']}")
    print(f"  - Missing files: {report['summary']['missing_files']}")
    print(f"  - Outdated references: {report['summary']['outdated_references']}")
    print(f"  - Output dir mismatches: {report['summary']['output_dir_mismatches']}")
    print(f"  - Script name errors: {report['summary']['script_name_errors']}")
    
    # Save report
    output_path = project_root / "output" / "filepath_audit_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    auditor.save_report(output_path)
    
    return 0 if report['summary']['total_issues'] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

