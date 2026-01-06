#!/usr/bin/env python3
"""
Fix Filepath and Reference Issues

This script fixes broken links, outdated script references, and path issues
identified by the audit.
"""

import sys
import re
import json
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Mapping of old script names to new script names
SCRIPT_NAME_FIXES = {
    # Old -> New
    r'\b2_gnn\.py\b': '3_gnn.py',
    r'\b3_tests\.py\b': '2_tests.py',
    r'\b4_type_checker\.py\b': '5_type_checker.py',
    r'\b5_export\.py\b': '7_export.py',
    r'\b7_mcp\.py\b': '21_mcp.py',
    r'\b8_ontology\.py\b': '10_ontology.py',
    r'\b9_render\.py\b': '11_render.py',
    r'\b10_execute\.py\b': '12_execute.py',
    r'\b12_discopy\.py\b': '15_audio.py',  # Old discopy step
    r'\b13_discopy_jax_eval\.py\b': '20_website.py',  # Old eval step
    r'\b14_site\.py\b': '20_website.py',
}

# Files that don't exist but are referenced (optional files)
OPTIONAL_FILES = {
    'CONTRIBUTING.md',
    'CHANGELOG.md',
    '.env',  # May not exist in repo
}

# Path fixes for markdown links
PATH_FIXES = [
    # Fix double "doc/doc/" paths
    (r'\(doc/doc/', '(doc/'),
    # Fix releases/ references
    (r'\.\./gnn/', 'doc/gnn/'),
    # Fix anchor-only links that should be file+anchor
    (r'\(#([^)]+)\)', r'(README.md#\1)'),  # Only in specific contexts
]

def fix_script_references(file_path: Path) -> bool:
    """Fix outdated script name references in a file"""
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        fixed = False
        
        for pattern, replacement in SCRIPT_NAME_FIXES.items():
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                fixed = True
        
        if fixed and content != original_content:
            file_path.write_text(content, encoding='utf-8')
            return True
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
    return False

def fix_markdown_paths(file_path: Path) -> bool:
    """Fix markdown link path issues"""
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        fixed = False
        
        # Fix double doc/doc paths
        if 'doc/doc/' in content:
            content = content.replace('doc/doc/', 'doc/')
            fixed = True
        
        # Fix releases/ references to doc/gnn/
        if file_path.parent.name == 'releases':
            content = re.sub(r'\.\./gnn/', 'doc/gnn/', content)
            if 'doc/gnn/' in content:
                fixed = True
        
        if fixed and content != original_content:
            file_path.write_text(content, encoding='utf-8')
            return True
    except Exception as e:
        print(f"Error fixing markdown paths in {file_path}: {e}")
    return False

def process_file(file_path: Path) -> Dict[str, int]:
    """Process a single file and return fix counts"""
    fixes = {"script_refs": 0, "paths": 0}
    
    if file_path.suffix == '.py':
        if fix_script_references(file_path):
            fixes["script_refs"] = 1
    elif file_path.suffix == '.md':
        if fix_script_references(file_path):
            fixes["script_refs"] = 1
        if fix_markdown_paths(file_path):
            fixes["paths"] = 1
    
    return fixes

def main():
    """Main execution"""
    print("=" * 60)
    print("Fixing Filepath and Reference Issues")
    print("=" * 60)
    
    # Load audit report
    audit_report_path = PROJECT_ROOT / "output" / "filepath_audit_report.json"
    if not audit_report_path.exists():
        print("Error: Audit report not found. Run audit_filepaths.py first.")
        return 1
    
    with open(audit_report_path, 'r') as f:
        audit_data = json.load(f)
    
    total_fixes = {"script_refs": 0, "paths": 0}
    files_fixed = []
    
    # Process Python files
    print("\nProcessing Python files...")
    py_files = list((PROJECT_ROOT / "src").rglob("*.py"))
    for file_path in py_files:
        if any(skip in str(file_path) for skip in ['__pycache__', '.git']):
            continue
        fixes = process_file(file_path)
        if sum(fixes.values()) > 0:
            files_fixed.append(str(file_path.relative_to(PROJECT_ROOT)))
            total_fixes["script_refs"] += fixes["script_refs"]
            total_fixes["paths"] += fixes["paths"]
    
    # Process markdown files
    print("Processing markdown files...")
    md_files = list(PROJECT_ROOT.rglob("*.md"))
    for file_path in md_files:
        if any(skip in str(file_path) for skip in ['__pycache__', '.git', 'node_modules']):
            continue
        fixes = process_file(file_path)
        if sum(fixes.values()) > 0:
            files_fixed.append(str(file_path.relative_to(PROJECT_ROOT)))
            total_fixes["script_refs"] += fixes["script_refs"]
            total_fixes["paths"] += fixes["paths"]
    
    print("\n" + "=" * 60)
    print("Fix Summary")
    print("=" * 60)
    print(f"Files fixed: {len(files_fixed)}")
    print(f"Script reference fixes: {total_fixes['script_refs']}")
    print(f"Path fixes: {total_fixes['paths']}")
    
    if files_fixed:
        print(f"\nFixed files (first 20):")
        for f in files_fixed[:20]:
            print(f"  - {f}")
        if len(files_fixed) > 20:
            print(f"  ... and {len(files_fixed) - 20} more")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

