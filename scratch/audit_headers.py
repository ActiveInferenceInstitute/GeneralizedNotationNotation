import re
from pathlib import Path

repo_root = Path("/Users/4d/Documents/GitHub/generalizednotationnotation")

results = []
for p in repo_root.glob("src/*/AGENTS.md"):
    if not p.is_file():
        continue
    content = p.read_text(errors="ignore")
    # extract "Pipeline Step" line
    step_match = re.search(r'\*\*Pipeline Step\*\*: (.+)', content)
    version_match = re.search(r'\*\*Version\*\*: (.+)', content)
    updated_match = re.search(r'\*\*Last Updated\*\*: (.+)', content)
    
    # We expect all to be aligned to some common version/date
    results.append({
        'module': p.parent.name,
        'step': step_match.group(1) if step_match else "MISSING",
        'version': version_match.group(1) if version_match else "MISSING",
        'updated': updated_match.group(1) if updated_match else "MISSING"
    })

print(f"{'Module':<20} | {'Step':<30} | {'Version':<10} | {'Updated'}")
print("-" * 80)
for r in sorted(results, key=lambda x: x['module']):
    print(f"{r['module']:<20} | {r['step'][:30]:<30} | {r['version']:<10} | {r['updated']}")
