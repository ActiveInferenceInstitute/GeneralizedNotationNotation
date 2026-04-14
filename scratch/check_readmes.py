from pathlib import Path
import json

results = []
for p in Path('.').rglob("README.md"):
    if "node_modules" in p.parts or ".venv" in p.parts or "output" in p.parts:
        continue
    text = p.read_text()
    lines = text.split('\n')
    
    issues = []
    if len(lines) < 10:
        issues.append(f"Too short ({len(lines)} lines)")
    if "TODO" in text:
        issues.append("Contains TODO")
    if "This directory serves" in text and len(lines) < 15:
        issues.append("Generic boilerplate")
        
    if issues:
        results.append({"file": str(p), "issues": issues})
        
print(json.dumps(results, indent=2))
