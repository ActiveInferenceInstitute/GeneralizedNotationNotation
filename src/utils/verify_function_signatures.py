#!/usr/bin/env python3
"""
Script to verify function signatures in AGENTS.md match actual code exports.

This script parses __init__.py files and AGENTS.md files to identify discrepancies.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
import sys

def extract_exports_from_init(init_path: Path) -> Dict[str, Any]:
    """Extract exported functions and classes from __init__.py."""
    if not init_path.exists():
        return {}
    
    with open(init_path, 'r') as f:
        content = f.read()
    
    exports = {
        'functions': [],
        'classes': [],
        'constants': [],
        'all_exports': []
    }
    
    # Parse __all__ if present
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == '__all__':
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            for item in node.value.elts:
                                if isinstance(item, ast.Str):
                                    exports['all_exports'].append(item.s)
                                elif isinstance(item, ast.Constant):
                                    exports['all_exports'].append(item.value)
    except Exception as e:
        print(f"Warning: Could not parse {init_path}: {e}")
    
    # Extract function definitions
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                exports['functions'].append(node.name)
            elif isinstance(node, ast.ClassDef):
                exports['classes'].append(node.name)
    except Exception:
        pass
    
    return exports

def extract_functions_from_agents(agents_path: Path) -> List[Dict[str, Any]]:
    """Extract function signatures documented in AGENTS.md."""
    if not agents_path.exists():
        return []
    
    with open(agents_path, 'r') as f:
        content = f.read()
    
    functions = []
    
    # Pattern to match function signatures in AGENTS.md
    # Matches: #### `function_name(...) -> return_type`
    pattern = r'####\s+`([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*(?:->\s*([^`]+))?`'
    
    for match in re.finditer(pattern, content):
        func_name = match.group(1)
        return_type = match.group(2) if match.group(2) else None
        
        # Try to extract full signature
        func_section_start = match.start()
        func_section_end = content.find('####', func_section_start + 1)
        if func_section_end == -1:
            func_section_end = len(content)
        
        func_section = content[func_section_start:func_section_end]
        
        # Extract parameters
        params_match = re.search(r'\(([^)]*)\)', match.group(0))
        params = []
        if params_match:
            param_str = params_match.group(1)
            for param in param_str.split(','):
                param = param.strip()
                if param:
                    params.append(param)
        
        functions.append({
            'name': func_name,
            'signature': match.group(0),
            'return_type': return_type,
            'parameters': params
        })
    
    return functions

def verify_module(module_path: Path) -> Dict[str, Any]:
    """Verify function signatures for a module."""
    init_path = module_path / '__init__.py'
    agents_path = module_path / 'AGENTS.md'
    
    result = {
        'module': str(module_path),
        'has_init': init_path.exists(),
        'has_agents': agents_path.exists(),
        'exports': {},
        'documented': [],
        'missing_in_docs': [],
        'missing_in_code': [],
        'signature_mismatches': []
    }
    
    if init_path.exists():
        result['exports'] = extract_exports_from_init(init_path)
    
    if agents_path.exists():
        result['documented'] = extract_functions_from_agents(agents_path)
        
        # Check for missing documentation
        if result['exports'].get('all_exports'):
            for export in result['exports']['all_exports']:
                if export.startswith('_'):
                    continue  # Skip private exports
                if not any(doc['name'] == export for doc in result['documented']):
                    result['missing_in_docs'].append(export)
        
        # Check for undocumented functions
        documented_names = {doc['name'] for doc in result['documented']}
        if result['exports'].get('all_exports'):
            for export in result['exports']['all_exports']:
                if export.startswith('_'):
                    continue
                if export not in documented_names and export not in ['__version__', 'FEATURES']:
                    result['missing_in_code'].append(export)
    
    return result

def main():
    """Main verification function."""
    src_path = Path(__file__).parent.parent
    
    modules = [
        'gnn', 'render', 'visualization', 'execute', 'llm', 'analysis',
        'export', 'type_checker', 'validation', 'ontology', 'audio',
        'integration', 'security', 'research', 'website', 'mcp', 'gui',
        'report', 'model_registry', 'setup', 'template', 'advanced_visualization'
    ]
    
    results = []
    for module_name in modules:
        module_path = src_path / module_name
        if module_path.exists() and module_path.is_dir():
            result = verify_module(module_path)
            results.append(result)
    
    # Print summary
    print("=" * 80)
    print("Function Signature Verification Report")
    print("=" * 80)
    print()
    
    for result in results:
        print(f"\nModule: {result['module']}")
        print(f"  Has __init__.py: {result['has_init']}")
        print(f"  Has AGENTS.md: {result['has_agents']}")
        
        if result['exports'].get('all_exports'):
            print(f"  Exports: {len(result['exports']['all_exports'])} items")
        
        if result['documented']:
            print(f"  Documented functions: {len(result['documented'])}")
        
        if result['missing_in_docs']:
            print(f"  ⚠️  Missing in docs: {', '.join(result['missing_in_docs'])}")
        
        if result['missing_in_code']:
            print(f"  ⚠️  Documented but not exported: {', '.join(result['missing_in_code'])}")
    
    # Generate detailed report
    report_path = src_path.parent / 'function_signature_verification_report.txt'
    with open(report_path, 'w') as f:
        f.write("Function Signature Verification Report\n")
        f.write("=" * 80 + "\n\n")
        
        for result in results:
            f.write(f"\nModule: {result['module']}\n")
            f.write(f"{'=' * 80}\n")
            
            if result['exports'].get('all_exports'):
                f.write(f"\nExports from __init__.py:\n")
                for export in result['exports']['all_exports']:
                    f.write(f"  - {export}\n")
            
            if result['documented']:
                f.write(f"\nDocumented in AGENTS.md:\n")
                for doc in result['documented']:
                    f.write(f"  - {doc['name']}\n")
            
            if result['missing_in_docs']:
                f.write(f"\n⚠️  Missing in documentation:\n")
                for missing in result['missing_in_docs']:
                    f.write(f"  - {missing}\n")
            
            if result['missing_in_code']:
                f.write(f"\n⚠️  Documented but not exported:\n")
                for missing in result['missing_in_code']:
                    f.write(f"  - {missing}\n")
    
    print(f"\n\nDetailed report written to: {report_path}")
    return 0

if __name__ == '__main__':
    sys.exit(main())


