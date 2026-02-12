
import os
import re
from pathlib import Path

def audit_documentation(root_dir: Path):
    issues = []
    
    # Define patterns to look for
    placeholder_patterns = [
        r"\[Module Name\]",
        r"\[Brief.*?\]",
        r"\[Main responsibility.*?\]",
        r"\[Capability.*?\]",
        r"\[What this agent does.*?\]",
        r"TODO",
        r"TBD",
        # r"FIXME" # FIXME might be code related, let's keep it to doc specific
    ]
    
    # Traverse directories
    for root, dirs, files in os.walk(root_dir):
        if ".git" in root or "__pycache__" in root or "node_modules" in root or ".venv" in root:
            continue
            
        for file in files:
            if file in ["README.md", "AGENTS.md"]:
                file_path = Path(root) / file
                try:
                    content = file_path.read_text(encoding="utf-8")
                except Exception as e:
                    issues.append(f"ERROR: Could not read {file_path}: {e}")
                    continue
                
                # Check for placeholders
                for pattern in placeholder_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        issues.append(f"ISSUE: {file_path} contains '{pattern}' (Matches: {len(matches)})")
                
                # Check for empty sections (basic check)
                # Matches "## Section Name" followed immediately by another header or end of file
                # This is a bit rough, but might catch completely empty templates
                empty_section_pattern = r"(^##\s+.*$)\n+(\s*^##\s+.*$)" 
                # Doing this reliably with regex is hard, maybe just rely on placeholders for now.
                
                # Check for broken relative links
                links = re.findall(r"\[.*?\]\((.*?)\)", content)
                for link in links:
                    if link.startswith("http") or link.startswith("#") or link.startswith("mailto:"):
                        continue
                    
                    # Ignore mailto and simple anchors
                    if "@" in link and not "/" in link: 
                        continue

                    # Clean link (remove anchors and queries)
                    link_path_str = link.split("#")[0].split("?")[0]
                    if not link_path_str:
                        continue
                        
                    # Resolve path
                    try:
                        # Assuming relative to file location
                        target_path = (file_path.parent / link_path_str).resolve()
                        if not target_path.exists():
                           issues.append(f"BROKEN LINK: {file_path} -> {link} (Target not found: {target_path})")
                    except Exception:
                        pass

    return issues

if __name__ == "__main__":
    root_dir = Path(".")
    print(f"Auditing documentation in {root_dir.resolve()}...")
    found_issues = audit_documentation(root_dir)
    
    if found_issues:
        print(f"\nFound {len(found_issues)} issues:")
        for issue in found_issues:
            print(issue)
        exit(1)
    else:
        print("\nNo obvious documentation issues found!")
        exit(0)
