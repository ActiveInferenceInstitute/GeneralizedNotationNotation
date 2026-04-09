import os
import re
from pathlib import Path
import json

repo_root = Path("/Users/4d/Documents/GitHub/generalizednotationnotation")

modules = [
    ("template", 0), ("setup", 1), ("tests", 2), ("gnn", 3), ("model_registry", 4),
    ("type_checker", 5), ("validation", 6), ("export", 7), ("visualization", 8),
    ("advanced_visualization", 9), ("ontology", 10), ("render", 11), ("execute", 12),
    ("llm", 13), ("ml_integration", 14), ("audio", 15), ("analysis", 16),
    ("integration", 17), ("security", 18), ("research", 19), ("website", 20),
    ("mcp", 21), ("gui", 22), ("report", 23), ("intelligent_analysis", 24)
]

results = []

for mod, step in modules:
    agent_path = repo_root / f"src/{mod}/AGENTS.md"
    readme_path = repo_root / f"src/{mod}/README.md"
    
    mod_result = {"module": mod, "step": step, "agents_exists": agent_path.exists(), "readme_exists": readme_path.exists(), "issues": []}
    
    if agent_path.exists():
        content = agent_path.read_text(errors="ignore")
        if f"Step {step}" not in content:
            mod_result["issues"].append(f"AGENTS.md missing correct 'Step {step}' reference.")
        
        # Check standard sections
        if "## Module Overview" not in content and "Overview" not in content:
            mod_result["issues"].append("AGENTS.md missing Overview section.")
        if "**Status**:" not in content and "Status:" not in content:
            mod_result["issues"].append("AGENTS.md missing Status.")
    else:
        mod_result["issues"].append("AGENTS.md missing entirely.")

    if readme_path.exists():
        content = readme_path.read_text(errors="ignore")
        if "Pipeline Step" in content and f"Step {step}" not in content:
            mod_result["issues"].append(f"README.md has incorrect 'Step {step}' reference.")
        if len(content.splitlines()) < 10:
             mod_result["issues"].append("README.md is very short/stubbed.")
    else:
        mod_result["issues"].append("README.md missing entirely.")
        
    results.append(mod_result)

# Also let's check submodules like src/execute/jax/AGENTS.md
sub_agents = list(repo_root.rglob("src/*/*/AGENTS.md"))
broken_sub = []
for p in sub_agents:
    if "output" in p.parts or "__pycache__" in p.parts:
        continue
    c = p.read_text(errors="ignore")
    if len(c.splitlines()) < 5:
        broken_sub.append(str(p.relative_to(repo_root)) + " (too short)")
    elif "TODO" in c or "TBD" in c:
        broken_sub.append(str(p.relative_to(repo_root)) + " (contains TODO/TBD)")

print("=== MODULE ISSUES ===")
for r in results:
    if r["issues"]:
         print(f"{r['module']} (Step {r['step']}): {r['issues']}")

print("\n=== SUBMODULE AGENTS ISSUES ===")
for b in broken_sub:
    print(b)

