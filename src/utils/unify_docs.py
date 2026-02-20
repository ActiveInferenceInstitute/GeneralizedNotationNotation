#!/usr/bin/env python3
"""
Utility script to unify headers across all GNN documentation files.
Enforces the standard:
**Version**: v1.1.0
**Last Updated**: February 9, 2026
**Status**: ✅ Production Ready
**Test Count**: 1,127 Tests Passing
"""

import os
import re
from pathlib import Path

STANDARD_HEADER = """**Version**: v1.1.0  
**Last Updated**: February 9, 2026  
**Status**: ✅ Production Ready  
**Test Count**: 1,127 Tests Passing  

"""

def unify_headers(doc_dir: Path):
    unified_count = 0
    
    for md_file in doc_dir.glob("**/*.md"):
        content = md_file.read_text(encoding="utf-8")
        
        # Split by the first header
        lines = content.splitlines(keepends=True)
        if not lines:
            continue
            
        if not lines[0].startswith("# "):
            continue  # Doesn't start with H1
            
        h1_line = lines[0]
        
        # Find where the actual content starts (skip existing metadata)
        start_idx = 1
        
        # Skip empty lines after H1
        while start_idx < len(lines) and lines[start_idx].strip() == "":
            start_idx += 1
            
        # Detect and skip existing metadata block
        in_metadata = False
        while start_idx < len(lines):
            line = lines[start_idx].strip()
            if line.startswith("**Version") or line.startswith("**Last Updated") or \
               line.startswith("**Status") or line.startswith("**Test Count") or \
               line.startswith("Version:"):
                in_metadata = True
                start_idx += 1
            elif in_metadata and line == "":
                start_idx += 1
            elif line.startswith("---") and in_metadata:
                start_idx += 1
            else:
                break
                
        # Skip the next empty line if there is one
        while start_idx < len(lines) and lines[start_idx].strip() == "":
            start_idx += 1
            
        new_content = h1_line + "\n" + STANDARD_HEADER + "".join(lines[start_idx:])
        
        if content != new_content:
            md_file.write_text(new_content, encoding="utf-8")
            print(f"Updated header in: {md_file.name}")
            unified_count += 1
            
    print(f"Total files updated: {unified_count}")

if __name__ == "__main__":
    docs_dir = Path(__file__).parent.parent.parent / "doc" / "gnn"
    if docs_dir.exists():
        unify_headers(docs_dir)
    else:
        print(f"Docs directory not found: {docs_dir}")
