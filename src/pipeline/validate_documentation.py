#!/usr/bin/env python3
"""
Documentation Validation Script for GNN Project

This script performs comprehensive validation of GNN documentation including:
- Broken link detection
- Pipeline step reference validation  
- Cross-reference consistency
- Metadata validation
- File existence checks

Usage:
    python src/pipeline/validate_documentation.py
    python src/pipeline/validate_documentation.py --fix-issues
    python src/pipeline/validate_documentation.py --verbose
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class ValidationResult:
    """Container for validation results"""
    errors: List[str]
    warnings: List[str]
    fixes_applied: List[str]
    
    @property
    def has_issues(self) -> bool:
        return len(self.errors) > 0 or len(self.warnings) > 0

class DocumentationValidator:
    """Main documentation validation class"""
    
    def __init__(self, project_root: Path, verbose: bool = False):
        self.project_root = project_root
        self.doc_root = project_root / "doc"
        self.verbose = verbose
        self.results = ValidationResult([], [], [])
        
        # Define actual pipeline steps (25 steps: 0-24)
        self.actual_pipeline_steps = list(range(0, 25))  # 0-24
        # Step number to script name mapping
        self.step_mapping = {
            0: "0_template.py",
            1: "1_setup.py",
            2: "2_tests.py",
            3: "3_gnn.py",
            4: "4_model_registry.py",
            5: "5_type_checker.py",
            6: "6_validation.py",
            7: "7_export.py",
            8: "8_visualization.py",
            9: "9_advanced_viz.py",
            10: "10_ontology.py",
            11: "11_render.py",
            12: "12_execute.py",
            13: "13_llm.py",
            14: "14_ml_integration.py",
            15: "15_audio.py",
            16: "16_analysis.py",
            17: "17_integration.py",
            18: "18_security.py",
            19: "19_research.py",
            20: "20_website.py",
            21: "21_mcp.py",
            22: "22_gui.py",
            23: "23_report.py",
            24: "24_intelligent_analysis.py"
        }
        
        # Patterns to search for broken links
        self.link_patterns = [
            r'\[([^\]]+)\]\(([^)]+)\)',  # [text](link)
            r'\[([^\]]+)\]\(([^)]+\.md(?:#[^)]*)?)\)',  # Markdown file links
            r'see:\s*([^\s]+\.md)',  # See: filename.md
            r'doc/([^\s]+\.md)',  # doc/filename.md
        ]
        
        # Patterns for pipeline step references
        self.pipeline_patterns = [
            r'step\s+(\d+)',
            r'(\d+)[-_]step',
            r'steps?\s*(\d+)',
            r'pipeline.*?(\d+)\s*steps?',
            r'(\d+)\s*-step',
            r'step.*?(\d+)',
        ]
        
    def log(self, message: str, level: str = "INFO"):
        """Log message if verbose mode enabled"""
        if self.verbose or level in ["ERROR", "WARNING"]:
            print(f"[{level}] {message}")
    
    def find_markdown_files(self) -> List[Path]:
        """Find all markdown files in documentation"""
        md_files = []
        for root, dirs, files in os.walk(self.doc_root):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            for file in files:
                if file.endswith('.md'):
                    md_files.append(Path(root) / file)
        return md_files
    
    def validate_links(self, file_path: Path) -> None:
        """Validate all links in a markdown file"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        for line_no, line in enumerate(lines, 1):
            # Find all markdown links in the line
            for pattern in self.link_patterns:
                matches = re.findall(pattern, line, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        link_text, link_url = match
                    else:
                        link_text, link_url = "", match
                    
                    self._validate_single_link(file_path, line_no, link_text, link_url)
    
    def _validate_single_link(self, file_path: Path, line_no: int, text: str, url: str) -> None:
        """Validate a single link"""
        # Skip external URLs, anchors only, and special protocols
        if any(url.startswith(prefix) for prefix in ['http://', 'https://', 'ftp://', 'mailto:', '#']):
            return
        
        # Handle relative paths
        if url.startswith('../'):
            # Relative to file location
            target_path = file_path.parent / url
        elif url.startswith('./'):
            # Relative to file location
            target_path = file_path.parent / url[2:]
        elif url.startswith('/'):
            # Absolute from project root
            target_path = self.project_root / url[1:]
        else:
            # Relative to file location
            target_path = file_path.parent / url
        
        # Remove fragment identifier
        if '#' in url:
            target_path = Path(str(target_path).split('#')[0])
        
        # Resolve path
        try:
            target_path = target_path.resolve()
            if not target_path.exists():
                rel_file_path = file_path.relative_to(self.project_root)
                self.results.errors.append(
                    f"Broken link in {rel_file_path}:{line_no}: '{text}' -> '{url}' (target not found: {target_path})"
                )
        except (OSError, ValueError) as e:
            rel_file_path = file_path.relative_to(self.project_root)
            self.results.errors.append(
                f"Invalid link in {rel_file_path}:{line_no}: '{text}' -> '{url}' (error: {e})"
            )
    
    def validate_pipeline_references(self, file_path: Path) -> None:
        """Validate pipeline step references"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        for line_no, line in enumerate(lines, 1):
            # Look for pipeline step references
            for pattern in self.pipeline_patterns:
                matches = re.findall(pattern, line, re.IGNORECASE)
                for match in matches:
                    try:
                        step_num = int(match)
                        if step_num < 0 or step_num > 24:
                            rel_file_path = file_path.relative_to(self.project_root)
                            self.results.warnings.append(
                                f"Invalid pipeline step in {rel_file_path}:{line_no}: "
                                f"Step {step_num} is outside valid range 0-24: '{line.strip()}'"
                            )
                    except ValueError:
                        continue
            
            # Check for specific incorrect references (old pipeline references)
            incorrect_refs = [
                "15_audio.py", "20_website.py", "20_website.py",
                "3_gnn.py",  # Should be 3_gnn.py
                "2_tests.py",  # Should be 2_tests.py
                "14-step", "14 steps", "steps 1-14", "fourteen steps",
                "13-step", "13 steps", "steps 1-13", "thirteen steps"
            ]
            for ref in incorrect_refs:
                if ref.lower() in line.lower():
                    rel_file_path = file_path.relative_to(self.project_root)
                    self.results.warnings.append(
                        f"Potentially outdated pipeline reference in {rel_file_path}:{line_no}: "
                        f"'{ref}' found in: '{line.strip()}' (pipeline has 25 steps: 0-24)"
                    )
    
    def validate_cross_references(self, file_path: Path) -> None:
        """Validate cross-references and metadata"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Check for metadata blocks
        if file_path.name == "README.md" and "Document Metadata" not in content:
            rel_file_path = file_path.relative_to(self.project_root)
            self.results.warnings.append(
                f"Missing metadata block in major README: {rel_file_path}"
            )
        
        # Check for cross-reference sections
        if "Cross-References" in content or "Cross-refs" in content:
            # Extract cross-references and validate them
            cross_ref_pattern = r'\*\*?Cross-References?\*\*?:?\s*(.+?)(?:\n|$)'
            matches = re.findall(cross_ref_pattern, content, re.IGNORECASE)
            for match in matches:
                # Find links in cross-reference line
                link_matches = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', match)
                for link_text, link_url in link_matches:
                    self._validate_single_link(file_path, 0, link_text, link_url)
    
    def validate_file_existence(self) -> None:
        """Validate that referenced files exist"""
        # Check that all pipeline script files exist
        src_dir = self.project_root / "src"
        for step_num, script_name in self.step_mapping.items():
            script_path = src_dir / script_name
            if not script_path.exists():
                self.results.errors.append(
                    f"Missing pipeline script: {script_name} (step {step_num})"
                )
        
        # Check for existence of commonly referenced directories
        important_dirs = [
            "doc/gnn", "doc/templates", "doc/archive", "doc/pipeline",
            "doc/troubleshooting", "src/gnn", "src/export", "src/render"
        ]
        for dir_path in important_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                self.results.warnings.append(f"Important directory missing: {dir_path}")
    
    def fix_common_issues(self, file_path: Path) -> bool:
        """Attempt to fix common documentation issues"""
        if not file_path.exists():
            return False
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        fixed = False
        
        # Fix common pipeline reference issues
        fixes = {
            # Old 14-step references -> 24-step
            r'\b14[_-]step\b': '24-step',
            r'\bstep[_\s]+14\b': 'step 23',
            r'\b14\s+steps?\b': '24 steps',
            r'\bfourteen\s+steps?\b': 'twenty-four steps',
            r'\bsteps?\s+1[-\s]*14\b': 'steps 0-23',
            r'\b1[-\s]*14\s+steps?\b': '0-23 steps',
            
            # Old 13-step references -> 24-step
            r'\b13[_-]step\b': '24-step',
            r'\bstep[_\s]+13\b': 'step 23',
            r'\b13\s+steps?\b': '24 steps',
            r'\bthirteen\s+steps?\b': 'twenty-four steps',
            r'\bsteps?\s+1[-\s]*13\b': 'steps 0-23',
            r'\b1[-\s]*13\s+steps?\b': '0-23 steps',
            
            # Incorrect script references (old pipeline)
            r'\b12_discopy\.py\b': '15_audio.py',
            r'\b13_discopy_jax_eval\.py\b': '20_website.py',
            r'\b14_site\.py\b': '20_website.py',
            r'\b2_gnn\.py\b': '3_gnn.py',
            r'\b3_tests\.py\b': '2_tests.py',
            
            # Maximum value fixes
            r'\bmaximum:\s*1[34]\b': 'maximum: 23',
            r'\bmax.*?1[34]\b': 'max: 23',
        }
        
        for pattern, replacement in fixes.items():
            if re.search(pattern, content, re.IGNORECASE):
                content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
                fixed = True
                self.results.fixes_applied.append(
                    f"Fixed '{pattern}' -> '{replacement}' in {file_path.relative_to(self.project_root)}"
                )
        
        # Write back if changes were made
        if fixed and content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
    
    def validate_all(self, fix_issues: bool = False) -> ValidationResult:
        """Run all validation checks"""
        self.log("Starting comprehensive documentation validation...")
        
        # Find all markdown files
        md_files = self.find_markdown_files()
        self.log(f"Found {len(md_files)} markdown files to validate")
        
        # Validate file existence first
        self.validate_file_existence()
        
        # Process each markdown file
        for file_path in md_files:
            self.log(f"Validating {file_path.relative_to(self.project_root)}")
            
            # Try to fix issues if requested
            if fix_issues:
                self.fix_common_issues(file_path)
            
            # Run validation checks
            self.validate_links(file_path)
            self.validate_pipeline_references(file_path)
            self.validate_cross_references(file_path)
        
        # Generate summary
        total_issues = len(self.results.errors) + len(self.results.warnings)
        if total_issues == 0:
            self.log("✅ Documentation validation completed successfully - no issues found!")
        else:
            self.log(f"❌ Documentation validation found {len(self.results.errors)} errors and {len(self.results.warnings)} warnings")
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate a detailed validation report"""
        report = ["# GNN Documentation Validation Report", ""]
        
        # Summary
        report.append(f"**Total Files Checked**: {len(self.find_markdown_files())}")
        report.append(f"**Errors Found**: {len(self.results.errors)}")
        report.append(f"**Warnings**: {len(self.results.warnings)}")
        report.append(f"**Fixes Applied**: {len(self.results.fixes_applied)}")
        report.append("")
        
        # Errors
        if self.results.errors:
            report.append("## ❌ Errors")
            for error in self.results.errors:
                report.append(f"- {error}")
            report.append("")
        
        # Warnings
        if self.results.warnings:
            report.append("## ⚠️ Warnings")
            for warning in self.results.warnings:
                report.append(f"- {warning}")
            report.append("")
        
        # Fixes
        if self.results.fixes_applied:
            report.append("## 🔧 Fixes Applied")
            for fix in self.results.fixes_applied:
                report.append(f"- {fix}")
            report.append("")
        
        return "\n".join(report)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Validate GNN documentation")
    parser.add_argument("--fix-issues", action="store_true", 
                       help="Attempt to automatically fix common issues")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--output-report", type=str,
                       help="Output validation report to file")
    parser.add_argument("--project-root", type=str, default=".",
                       help="Project root directory")
    
    args = parser.parse_args()
    
    # Initialize validator
    project_root = Path(args.project_root).resolve()
    if not (project_root / "doc").exists():
        print(f"Error: doc directory not found in {project_root}")
        sys.exit(1)
    
    validator = DocumentationValidator(project_root, verbose=args.verbose)
    
    # Run validation
    results = validator.validate_all(fix_issues=args.fix_issues)
    
    # Generate and output report
    report = validator.generate_report()
    
    if args.output_report:
        with open(args.output_report, 'w') as f:
            f.write(report)
        print(f"Validation report written to {args.output_report}")
    else:
        print(report)
    
    # Exit with error code if issues found
    if results.has_issues and not args.fix_issues:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main() 