import re
from pathlib import Path

repo_root = Path("/Users/4d/Documents/GitHub/generalizednotationnotation")

results = []
checked = 0

for p in repo_root.rglob("*.md"):
    if p.name not in ["AGENTS.md", "README.md"]:
        continue
    if any(x in p.parts for x in ["output", "__pycache__", ".git", ".venv"]):
        continue
    
    checked += 1
    content = p.read_text(errors="ignore")
    issues = []
    
    # Check for unreplaced template variables
    if re.search(r'\{[a-zA-Z0-9_]+\}', content):
        variables = set(re.findall(r'\{[a-zA-Z0-9_]+\}', content))
        if variables != {"{module}", "{module_name}"}: # Sometimes curly braces are used in code blocks, so limit what we look for
            issues.append(f"Unreplaced template variables: {variables}")
            
    # Check if file has "TODO", "TBD", "Pending"
    if "TODO" in content or "TBD" in content or "Pending implementation" in content:
        issues.append("Contains TODO/TBD placeholders")
        
    # Check if the title is literally "Module_Name" or similar
    lines = content.splitlines()
    if lines and len(lines) > 0 and 'Module_Name' in lines[0]:
         issues.append("Title seems to be a template: " + lines[0])

    if p.name == "AGENTS.md":
        # Check standard sections
        if "Overview" not in content:
            issues.append("Missing Overview section")
        
    if issues:
        results.append((str(p.relative_to(repo_root)), issues))

print(f"Checked {checked} files. Found {len(results)} files with issues.")
for f, iss in sorted(results):
    print(f"{f}: {iss}")
