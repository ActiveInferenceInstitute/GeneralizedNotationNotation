#!/usr/bin/env python3
"""
Fix Path Issues in Documentation

Fixes common path issues identified in the audit:
- Double doc/doc/ paths
- Incorrect relative paths
- Missing file references that should be removed or updated
"""

import sys
import re
import json
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Path fixes: (pattern, replacement, description)
PATH_FIXES: List[Tuple[str, str, str]] = [
    # Fix double doc/doc/ paths
    (r'\(doc/doc/', '(doc/', 'Fix double doc/doc/ paths'),
    (r'\[([^\]]+)\]\(doc/doc/', r'[\1](doc/', 'Fix double doc/doc/ in markdown links'),
    
    # Fix releases/ references (already fixed, but keep for safety)
    (r'\.\./gnn/', 'doc/gnn/', 'Fix releases/gnn/ references'),
    
    # Remove references to non-existent security docs (comment them out or remove)
    # We'll handle these separately as they may need content creation
]

# Files that reference non-existent files - we'll comment them out
NON_EXISTENT_FILE_REFERENCES = {
    'doc/llm/security_guidelines.md',
    'doc/security/incident_response.md',
    'doc/security/vulnerability_assessment.md',
    'doc/security/monitoring.md',
}

def fix_file_paths(file_path: Path) -> Tuple[int, List[str]]:
    """Fix path issues in a file"""
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        fixes_applied = []
        
        # Apply path fixes
        for pattern, replacement, description in PATH_FIXES:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                fixes_applied.append(description)
        
        # Comment out references to non-existent files (optional - we'll skip for now)
        # This would require more careful handling
        
        if fixes_applied and content != original_content:
            file_path.write_text(content, encoding='utf-8')
            return len(fixes_applied), fixes_applied
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
    return 0, []

def main():
    """Main execution"""
    print("=" * 60)
    print("Fixing Path Issues in Documentation")
    print("=" * 60)
    
    total_fixes = 0
    files_fixed = []
    
    # Process markdown files
    print("\nProcessing markdown files...")
    md_files = list(PROJECT_ROOT.rglob("*.md"))
    for file_path in md_files:
        if any(skip in str(file_path) for skip in ['__pycache__', '.git', 'node_modules', 'output/']):
            continue
        fixes_count, fixes = fix_file_paths(file_path)
        if fixes_count > 0:
            files_fixed.append((str(file_path.relative_to(PROJECT_ROOT)), fixes))
            total_fixes += fixes_count
    
    print("\n" + "=" * 60)
    print("Fix Summary")
    print("=" * 60)
    print(f"Total fixes applied: {total_fixes}")
    print(f"Files modified: {len(files_fixed)}")
    
    if files_fixed:
        print(f"\nFixed files (first 20):")
        for file_path, fixes in files_fixed[:20]:
            print(f"  - {file_path}: {', '.join(fixes)}")
        if len(files_fixed) > 20:
            print(f"  ... and {len(files_fixed) - 20} more")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


