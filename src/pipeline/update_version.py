#!/usr/bin/env python3
"""
Version Update Script for GeneralizedNotationNotation (GNN)

This script updates version numbers consistently across all project files.
"""

import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

class VersionUpdater:
    """Updates version numbers across GNN project files."""
    
    def __init__(self, new_version: str, project_root: Path = None):
        self.new_version = new_version
        self.project_root = project_root or Path(__file__).parent.parent
        self.updated_files: List[str] = []
        self.errors: List[str] = []
        
    def validate_version(self) -> bool:
        """Validate version format (semantic versioning)."""
        pattern = r'^\d+\.\d+\.\d+(?:-[a-zA-Z0-9\-\.]+)?(?:\+[a-zA-Z0-9\-\.]+)?$'
        if not re.match(pattern, self.new_version):
            self.errors.append(f"Invalid version format: {self.new_version}")
            return False
        return True
    
    def update_citation_file(self) -> bool:
        """Update CITATION.cff file."""
        try:
            file_path = self.project_root / "CITATION.cff"
            if not file_path.exists():
                self.errors.append(f"File not found: {file_path}")
                return False
                
            content = file_path.read_text()
            
            # Update version
            content = re.sub(
                r'version: \d+\.\d+\.\d+.*',
                f'version: {self.new_version} # Current stable release',
                content
            )
            
            # Update date
            today = datetime.now().strftime('%Y-%m-%d')
            content = re.sub(
                r'date-released: \d{4}-\d{2}-\d{2}',
                f'date-released: {today}',
                content
            )
            
            file_path.write_text(content)
            self.updated_files.append(str(file_path))
            return True
            
        except Exception as e:
            self.errors.append(f"Error updating {file_path}: {e}")
            return False
    
    def update_module_init_files(self) -> bool:
        """Update all module __init__.py files."""
        success = True
        
        init_files = [
            "src/__init__.py",
            "src/gnn/__init__.py", 
            "src/execute/__init__.py",
            "src/sapf/__init__.py",
            "src/type_checker/__init__.py",
            "src/export/__init__.py",
            "src/website/__init__.py"
        ]
        
        for file_path_str in init_files:
            try:
                file_path = self.project_root / file_path_str
                if not file_path.exists():
                    self.errors.append(f"File not found: {file_path}")
                    success = False
                    continue
                    
                content = file_path.read_text()
                content = re.sub(
                    r'__version__ = ["\'][^"\']*["\']',
                    f'__version__ = "{self.new_version}"',
                    content
                )
                
                file_path.write_text(content)
                self.updated_files.append(str(file_path))
                
            except Exception as e:
                self.errors.append(f"Error updating {file_path}: {e}")
                success = False
                
        return success
    
    def update_julia_scripts(self) -> bool:
        """Update version in Julia scripts."""
        try:
            julia_file = self.project_root / "src/execute/activeinference_jl/activeinference_runner.jl"
            if not julia_file.exists():
                return True  # Not an error if file doesn't exist
                
            content = julia_file.read_text()
            content = re.sub(
                r'const SCRIPT_VERSION = "[^"]*"',
                f'const SCRIPT_VERSION = "{self.new_version}"',
                content
            )
            
            julia_file.write_text(content)
            self.updated_files.append(str(julia_file))
            return True
            
        except Exception as e:
            self.errors.append(f"Error updating Julia script: {e}")
            return False
    
    def update_changelog(self) -> bool:
        """Update changelog with new version if it's a release."""
        try:
            changelog_path = self.project_root / "CHANGELOG.md"
            if not changelog_path.exists():
                return True  # Not an error if file doesn't exist
                
            content = changelog_path.read_text()
            
            # Only update if this is a release version (not pre-release)
            if '-' not in self.new_version and '+' not in self.new_version:
                today = datetime.now().strftime('%Y-%m-%d')
                
                # Add new version entry after [Unreleased]
                unreleased_pattern = r'(## \[Unreleased\].*?\n)'
                replacement = f'\\1\n## [{self.new_version}] - {today}\n\n### Changed\n- Version update to {self.new_version}\n'
                
                content = re.sub(unreleased_pattern, replacement, content, flags=re.DOTALL)
                
                # Update comparison links at bottom
                if '[Unreleased]' in content:
                    content = re.sub(
                        r'\[Unreleased\]: https://github\.com/ActiveInferenceInstitute/GeneralizedNotationNotation/compare/v[^.]+\.[^.]+\.[^.]+(.*?)HEAD',
                        f'[Unreleased]: https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/compare/v{self.new_version}...HEAD',
                        content
                    )
                
                changelog_path.write_text(content)
                self.updated_files.append(str(changelog_path))
                
            return True
            
        except Exception as e:
            self.errors.append(f"Error updating changelog: {e}")
            return False
    
    def run_update(self) -> bool:
        """Run the complete version update process."""
        if not self.validate_version():
            return False
            
        print(f"Updating version to {self.new_version}...")
        
        success = True
        success &= self.update_citation_file()
        success &= self.update_module_init_files()
        success &= self.update_julia_scripts()
        success &= self.update_changelog()
        
        return success
    
    def print_summary(self):
        """Print summary of update results."""
        print(f"\nVersion Update Summary:")
        print(f"Target Version: {self.new_version}")
        print(f"Updated Files: {len(self.updated_files)}")
        
        if self.updated_files:
            print("\nSuccessfully Updated:")
            for file in self.updated_files:
                print(f"  ✅ {file}")
        
        if self.errors:
            print(f"\nErrors: {len(self.errors)}")
            for error in self.errors:
                print(f"  ❌ {error}")
                
        return len(self.errors) == 0

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Update version numbers across GNN project files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/update_version.py 1.2.0
  python scripts/update_version.py 1.2.0-rc.1
  python scripts/update_version.py 1.2.0+build.123
        """
    )
    
    parser.add_argument(
        'version',
        help='New version number (semantic versioning format)'
    )
    
    parser.add_argument(
        '--project-root',
        type=Path,
        help='Path to project root (default: auto-detect)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be updated without making changes'
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
        print(f"Would update to version: {args.version}")
        return 0
    
    updater = VersionUpdater(args.version, args.project_root)
    success = updater.run_update()
    
    if updater.print_summary():
        print(f"\n🎉 Successfully updated project to version {args.version}")
        print("\nNext steps:")
        print("1. Review the changes: git diff")
        print("2. Test the build: python src/main.py --only-steps 2,3")
        print("3. Commit changes: git add . && git commit -m 'Bump version to {args.version}'")
        print("4. Create tag: git tag -a v{args.version} -m 'Release v{args.version}'")
        return 0
    else:
        print(f"\n❌ Version update completed with errors")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 