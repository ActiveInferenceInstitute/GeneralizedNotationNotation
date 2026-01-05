import os
import re
from pathlib import Path

REPO_ROOT = "/Users/mini/Documents/GitHub/GeneralizedNotationNotation"
SRC_DIR = os.path.join(REPO_ROOT, "src")
DOC_DIR = os.path.join(REPO_ROOT, "doc")

def check_directory(dir_path, is_src=False):
    dirname = os.path.basename(dir_path)
    if dirname.startswith("__") or dirname.startswith("."):
        return None

    agents_md = os.path.join(dir_path, "AGENTS.md")
    readme_md = os.path.join(dir_path, "README.md")
    
    has_agents = os.path.exists(agents_md)
    has_readme = os.path.exists(readme_md)
    has_mermaid = False
    has_signatures = False
    
    if has_readme:
        with open(readme_md, 'r') as f:
            content = f.read()
            if "```mermaid" in content:
                has_mermaid = True

    if has_agents:
        with open(agents_md, 'r') as f:
            content = f.read()
            # Check for function signature markers
            if "def " in content or "->" in content or "### `" in content:
                has_signatures = True
            
            # Simple heuristic: Does it mention "Public Functions" or similar?
            if "Public Functions" not in content and "API Reference" not in content:
                has_signatures = False
    
    return {
        "path": dir_path,
        "is_src": is_src,
        "has_agents": has_agents,
        "has_readme": has_readme,
        "has_mermaid": has_mermaid,
        "has_signatures": has_signatures
    }

def scan_root(root_dir, is_src=False):
    results = []
    try:
        items = os.listdir(root_dir)
    except FileNotFoundError:
        return results

    for item in items:
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            if item.startswith("__") or item.startswith("."):
                continue
            res = check_directory(item_path, is_src)
            if res:
                results.append(res)
    return results

print("Scanning src/...")
src_results = scan_root(SRC_DIR, is_src=True)

print("Scanning doc/...")
doc_results = scan_root(DOC_DIR, is_src=False)

print("\n--- Audit Report ---")
missing_agents_src = [r["path"] for r in src_results if not r["has_agents"]]
missing_agents_doc = [r["path"] for r in doc_results if not r["has_agents"]]
missing_readme = [r["path"] for r in src_results + doc_results if not r["has_readme"]]
missing_mermaid = [r["path"] for r in src_results if r["has_readme"] and not r["has_mermaid"]]
missing_signatures = [r["path"] for r in src_results if r["has_agents"] and not r["has_signatures"]]

print(f"Total Source Modules: {len(src_results)}")
print(f"Total Doc Directories: {len(doc_results)}")
print(f"\nMissing AGENTS.md in src/: {len(missing_agents_src)}")
for p in missing_agents_src:
    print(f"  - {os.path.basename(p)}")

print(f"\nMissing AGENTS.md in doc/: {len(missing_agents_doc)}")
for p in missing_agents_doc:
    print(f"  - {os.path.basename(p)}")

print(f"\nMissing README.md (All): {len(missing_readme)}")
for p in missing_readme:
    print(f"  - {os.path.basename(p)}")

print(f"\nMissing Mermaid in README.md (src/): {len(missing_mermaid)}")
for p in missing_mermaid:
    print(f"  - {os.path.basename(p)}")

print(f"\nMissing Function Signatures in AGENTS.md (src/): {len(missing_signatures)}")
for p in missing_signatures:
    print(f"  - {os.path.basename(p)}")
