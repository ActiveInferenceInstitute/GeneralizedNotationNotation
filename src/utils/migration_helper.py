#!/usr/bin/env python3
"""
Pipeline Migration Helper

This utility helps automate the implementation of pipeline improvements
by analyzing and updating code patterns across pipeline modules.

Usage:
    python -m utils.migration_helper --analyze
    python -m utils.migration_helper --apply-improvements
    python -m utils.migration_helper --module 7_export.py --dry-run
"""

import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class PipelineMigrationHelper:
    """Helper class for migrating pipeline modules to new patterns."""
    
    def __init__(self, src_dir: Path):
        self.src_dir = src_dir
        self.changes_made = []
        
    def analyze_module(self, module_path: Path) -> Dict[str, List[str]]:
        """Analyze a module for potential improvements."""
        issues = {
            "redundant_fallbacks": [],
            "missing_enhanced_imports": [],
            "hardcoded_paths": [],
            "missing_performance_tracking": []
        }
        
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for redundant fallback patterns
            if self._has_redundant_fallbacks(content):
                issues["redundant_fallbacks"].append("Has redundant import fallbacks")
            
            # Check for enhanced argument parsing opportunities
            if self._needs_enhanced_argument_parsing(content, module_path):
                issues["missing_enhanced_imports"].append("Could use ArgumentParser")
            
            # Check for hardcoded paths
            hardcoded = self._find_hardcoded_paths(content)
            if hardcoded:
                issues["hardcoded_paths"].extend(hardcoded)
            
            # Check for performance tracking opportunities
            if self._needs_performance_tracking(module_path):
                issues["missing_performance_tracking"].append("Compute-intensive step without performance tracking")
                
        except Exception as e:
            logger.error(f"Error analyzing {module_path}: {e}")
            
        return issues
    
    def apply_improvements(self, module_path: Path, dry_run: bool = True) -> List[str]:
        """Apply automatic improvements to a module."""
        changes = []
        
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            content = original_content
            
            # Remove redundant fallbacks
            if self._has_redundant_fallbacks(content):
                content, fallback_changes = self._remove_redundant_fallbacks(content)
                changes.extend(fallback_changes)
            
            # Add missing imports
            content, import_changes = self._add_missing_imports(content, module_path)
            changes.extend(import_changes)
            
            # Update hardcoded paths (basic cases)
            content, path_changes = self._fix_hardcoded_paths(content)
            changes.extend(path_changes)
            
            # Write changes if not dry run
            if not dry_run and content != original_content:
                # Create backup
                backup_path = module_path.with_suffix(module_path.suffix + '.backup')
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                
                # Write updated content
                with open(module_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                changes.append(f"✅ Applied changes (backup: {backup_path.name})")
            elif dry_run and content != original_content:
                changes.append("🔍 [DRY RUN] Changes would be applied")
                
        except Exception as e:
            logger.error(f"Error applying improvements to {module_path}: {e}")
            changes.append(f"❌ Error: {e}")
            
        return changes
    
    def _has_redundant_fallbacks(self, content: str) -> bool:
        """Check if module has redundant fallback imports."""
        patterns = [
            r'try:\s+from utils import.*?except ImportError.*?def setup_step_logging',
            r'except ImportError as e:.*?logging\.basicConfig',
            r'try:\s+.*?UTILS_AVAILABLE.*?except ImportError.*?UTILS_AVAILABLE = False'
        ]
        
        for pattern in patterns:
            if re.search(pattern, content, re.DOTALL):
                return True
        return False
    
    def _needs_enhanced_argument_parsing(self, content: str, module_path: Path) -> bool:
        """Check if module could benefit from enhanced argument parsing."""
        # Skip if already using enhanced parser
        if "ArgumentParser" in content:
            return False
            
        # Check if it's a pipeline step that parses arguments
        if module_path.name.startswith(('1_', '2_', '3_', '4_', '5_', '6_', '7_', '8_', '9_', '10_', '11_', '12_', '13_', '14_')):
            return "argparse.ArgumentParser" in content
        
        # Check main.py specifically
        if module_path.name == "main.py":
            return "ArgumentParser" in content
            
        return False
    
    def _find_hardcoded_paths(self, content: str) -> List[str]:
        """Find hardcoded path patterns."""
        issues = []
        
        # Look for hardcoded relative paths
        hardcoded_patterns = [
            (r'Path\(["\']src/', "Hardcoded 'src/' path"),
            (r'Path\(["\']output/', "Hardcoded 'output/' path"),
            (r'["\']input/gnn_files["\']', "Hardcoded input GNN files path"),
            (r'project_root / ["\']src["\'] / ["\']gnn["\']', "Could use DEFAULT_PATHS")
        ]
        
        for pattern, description in hardcoded_patterns:
            if re.search(pattern, content):
                issues.append(description)
                
        return issues
    
    def _needs_performance_tracking(self, module_path: Path) -> bool:
        """Check if module should have performance tracking."""
        compute_intensive = ['7_export.py', '8_visualization.py', '11_render.py', '12_execute.py', '13_llm.py']
        return module_path.name in compute_intensive
    
    def _remove_redundant_fallbacks(self, content: str) -> Tuple[str, List[str]]:
        """Remove redundant fallback import patterns."""
        changes = []
        
        # Pattern 1: Remove try/except around utils import with custom fallbacks
        pattern1 = r'try:\s+(from utils import[^}]+})\s+except ImportError as e:.*?(?=\n\w|\n#|\nif|\ndef|\nclass|\Z)'
        
        def replace_fallback(match):
            utils_import = match.group(1)
            changes.append("Removed redundant import fallback - utils provides graceful fallbacks")
            return utils_import + "\n"
        
        content = re.sub(pattern1, replace_fallback, content, flags=re.DOTALL)
        
        # Pattern 2: Remove UTILS_AVAILABLE fallback definitions
        pattern2 = r'except ImportError.*?UTILS_AVAILABLE = False.*?(?=\n\w|\n#|\Z)'
        if re.search(pattern2, content, re.DOTALL):
            content = re.sub(pattern2, '', content, flags=re.DOTALL)
            changes.append("Removed redundant UTILS_AVAILABLE fallback")
        
        return content, changes
    
    def _add_missing_imports(self, content: str, module_path: Path) -> Tuple[str, List[str]]:
        """Add missing standard imports."""
        changes = []
        
        # For main.py, suggest enhanced imports
        if module_path.name == "main.py":
            if "from utils import" in content and "setup_main_logging" not in content:
                # This is a complex change, just suggest it
                changes.append("💡 Consider adding: setup_main_logging, ArgumentParser")
        
        return content, changes
    
    def _fix_hardcoded_paths(self, content: str) -> Tuple[str, List[str]]:
        """Fix simple hardcoded path patterns."""
        changes = []
        
        # Replace hardcoded input/gnn_files with DEFAULT_PATHS reference
        if 'src" / "gnn" / "examples"' in content and "DEFAULT_PATHS" not in content:
            changes.append("💡 Consider using DEFAULT_PATHS['target_dir'] instead of hardcoded paths")
        
        return content, changes

def main():
    parser = argparse.ArgumentParser(description="Pipeline Migration Helper")
    parser.add_argument("--analyze", action="store_true", 
                       help="Analyze all modules for improvement opportunities")
    parser.add_argument("--apply-improvements", action="store_true",
                       help="Apply automatic improvements to all modules")
    parser.add_argument("--module", type=str,
                       help="Specific module to process")
    parser.add_argument("--dry-run", action="store_true", default=True,
                       help="Show what would be changed without applying (default)")
    parser.add_argument("--apply", action="store_true",
                       help="Actually apply changes (removes dry-run)")
    
    args = parser.parse_args()
    
    # Determine src directory
    script_path = Path(__file__).resolve()
    src_dir = script_path.parent.parent  # utils -> src
    
    migration_helper = PipelineMigrationHelper(src_dir)
    
    # Get modules to process
    if args.module:
        modules = [src_dir / args.module]
        if not modules[0].exists():
            logger.error(f"Module not found: {modules[0]}")
            return 1
    else:
        # Get all pipeline modules
        pattern = "*_*.py"  # Numbered pipeline modules
        modules = list(src_dir.glob(pattern))
        modules.append(src_dir / "main.py")
        modules = [m for m in modules if m.is_file()]
    
    logger.info(f"Processing {len(modules)} modules...")
    
    if args.analyze:
        # Analysis mode
        print("\n" + "="*80)
        print("PIPELINE MIGRATION ANALYSIS")
        print("="*80)
        
        total_issues = 0
        for module in modules:
            issues = migration_helper.analyze_module(module)
            module_issues = sum(len(v) for v in issues.values())
            
            if module_issues > 0:
                print(f"\n📁 {module.name}:")
                total_issues += module_issues
                
                for category, issue_list in issues.items():
                    if issue_list:
                        print(f"  {category.replace('_', ' ').title()}:")
                        for issue in issue_list:
                            print(f"    - {issue}")
        
        print(f"\n📊 Summary: {total_issues} improvement opportunities found across {len(modules)} modules")
        
    if args.apply_improvements:
        # Apply improvements mode
        dry_run = not args.apply
        mode = "DRY RUN" if dry_run else "APPLYING CHANGES"
        
        print(f"\n{mode}: Applying improvements...")
        
        total_changes = 0
        for module in modules:
            changes = migration_helper.apply_improvements(module, dry_run=dry_run)
            
            if changes:
                print(f"\n📁 {module.name}:")
                for change in changes:
                    print(f"  {change}")
                total_changes += len(changes)
        
        if dry_run:
            print(f"\n🔍 DRY RUN COMPLETE: {total_changes} potential changes identified")
            print("Use --apply to actually make changes")
        else:
            print(f"\n✅ MIGRATION COMPLETE: {total_changes} changes applied")
            print("Run the pipeline validation to see improvements:")
            print("  python src/pipeline_validation.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 