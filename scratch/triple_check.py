#!/usr/bin/env python3
"""
Triple-check audit: cross-reference orchestrator scripts ↔ module directories ↔ AGENTS.md ↔ README.md
"""
import re, ast, os
from pathlib import Path

repo = Path("/Users/4d/Documents/GitHub/generalizednotationnotation")
src = repo / "src"

# ── 1. Enumerate all orchestrator scripts ──
orchestrators = sorted(src.glob("[0-9]*_*.py"))
print("=" * 80)
print(f"PHASE 1: Orchestrator Scripts Found ({len(orchestrators)})")
print("=" * 80)
for o in orchestrators:
    print(f"  {o.name}")

# ── 2. Parse each orchestrator: extract module import, function call, step number ──
print("\n" + "=" * 80)
print("PHASE 2: Orchestrator → Module Import Cross-Reference")
print("=" * 80)

issues = []

for o in orchestrators:
    stem = o.stem  # e.g. "3_gnn"
    step_num = int(stem.split("_")[0])
    content = o.read_text(errors="ignore")
    
    # Find module imports
    module_imports = re.findall(r'from\s+(\w+)\s+import', content)
    module_imports += re.findall(r'import\s+(\w+)', content)
    # Filter to likely module names (not stdlib)
    stdlib = {'sys', 'os', 'pathlib', 'argparse', 'logging', 'json', 'time', 'traceback', 'datetime', 'importlib', 'subprocess'}
    module_imports = [m for m in module_imports if m not in stdlib]
    
    # Check if the module directory exists
    module_name_from_script = "_".join(stem.split("_")[1:])  # e.g. "gnn", "model_registry"
    module_dir = src / module_name_from_script
    
    # Some scripts map to differently-named directories
    alt_names = {
        "advanced_viz": "advanced_visualization",
    }
    if module_name_from_script in alt_names:
        module_dir = src / alt_names[module_name_from_script]
        module_name_from_script = alt_names[module_name_from_script]
    
    has_dir = module_dir.is_dir()
    has_init = (module_dir / "__init__.py").exists() if has_dir else False
    has_agents = (module_dir / "AGENTS.md").exists() if has_dir else False
    has_readme = (module_dir / "README.md").exists() if has_dir else False
    
    status = "✅" if all([has_dir, has_init, has_agents, has_readme]) else "❌"
    print(f"\n  {status} Step {step_num}: {o.name} → {module_name_from_script}/")
    
    if not has_dir:
        issues.append(f"Step {step_num}: Module directory '{module_name_from_script}/' MISSING")
    else:
        if not has_init:
            issues.append(f"Step {step_num}: {module_name_from_script}/__init__.py MISSING")
        if not has_agents:
            issues.append(f"Step {step_num}: {module_name_from_script}/AGENTS.md MISSING")
        if not has_readme:
            issues.append(f"Step {step_num}: {module_name_from_script}/README.md MISSING")
    
    # Cross-check: does the AGENTS.md reference the correct step number?
    if has_agents:
        agents_content = (module_dir / "AGENTS.md").read_text(errors="ignore")
        if f"Step {step_num}" not in agents_content:
            issues.append(f"Step {step_num}: {module_name_from_script}/AGENTS.md does NOT mention 'Step {step_num}'")
            print(f"    ⚠️  AGENTS.md does not mention 'Step {step_num}'")
        
        # Check for standard metadata
        for field in ["**Purpose**:", "**Status**:", "**Version**:"]:
            if field not in agents_content:
                issues.append(f"Step {step_num}: {module_name_from_script}/AGENTS.md missing '{field}'")
                print(f"    ⚠️  AGENTS.md missing '{field}'")
    
    print(f"    dir={has_dir} init={has_init} agents={has_agents} readme={has_readme}")
    print(f"    Imports detected: {module_imports[:5]}")

# ── 3. Check infrastructure modules ──
print("\n" + "=" * 80)
print("PHASE 3: Infrastructure Module Verification")
print("=" * 80)

infra_modules = ["api", "cli", "lsp", "pipeline", "utils", "sapf", "doc", "output"]
for mod in infra_modules:
    mod_dir = src / mod
    has_dir = mod_dir.is_dir()
    has_init = (mod_dir / "__init__.py").exists() if has_dir else False
    has_agents = (mod_dir / "AGENTS.md").exists() if has_dir else False
    has_readme = (mod_dir / "README.md").exists() if has_dir else False
    
    status = "✅" if all([has_dir, has_agents, has_readme]) else "❌"
    print(f"  {status} {mod}/: dir={has_dir} init={has_init} agents={has_agents} readme={has_readme}")
    
    if has_agents:
        agents_content = (mod_dir / "AGENTS.md").read_text(errors="ignore")
        for field in ["**Purpose**:", "**Status**:", "**Version**:"]:
            if field not in agents_content:
                issues.append(f"Infra {mod}: AGENTS.md missing '{field}'")
                print(f"    ⚠️  AGENTS.md missing '{field}'")

# ── 4. Check top-level src/ files ──
print("\n" + "=" * 80)
print("PHASE 4: Top-Level src/ Files")
print("=" * 80)

top_files = ["__init__.py", "main.py", "AGENTS.md", "README.md", "SPEC.md", "STEP_INDEX.md"]
for f in top_files:
    fp = src / f
    exists = fp.exists()
    size = fp.stat().st_size if exists else 0
    status = "✅" if exists and size > 10 else "❌"
    print(f"  {status} src/{f}: exists={exists} size={size}B")
    if not exists:
        issues.append(f"src/{f} MISSING")

# ── 5. Cross-check STEP_INDEX.md for all 25 steps ──
print("\n" + "=" * 80)
print("PHASE 5: STEP_INDEX.md Completeness")
print("=" * 80)

step_index = src / "STEP_INDEX.md"
if step_index.exists():
    si_content = step_index.read_text(errors="ignore")
    for i in range(25):
        if f"Step {i}" not in si_content and f"step {i}" not in si_content.lower() and str(i) not in si_content:
            issues.append(f"STEP_INDEX.md missing reference to Step {i}")
            print(f"  ⚠️  Missing Step {i}")
        else:
            print(f"  ✅ Step {i} referenced")
else:
    issues.append("STEP_INDEX.md does not exist")
    print("  ❌ STEP_INDEX.md not found")

# ── 6. Cross-check main.py step count ──
print("\n" + "=" * 80)
print("PHASE 6: main.py Step Registration")
print("=" * 80)

main_py = src / "main.py"
if main_py.exists():
    main_content = main_py.read_text(errors="ignore")
    # Count step references
    step_refs = set()
    for m in re.finditer(r'(\d+)_\w+', main_content):
        num = int(m.group(1))
        if 0 <= num <= 24:
            step_refs.add(num)
    
    missing_steps = set(range(25)) - step_refs
    print(f"  Steps referenced in main.py: {sorted(step_refs)}")
    if missing_steps:
        print(f"  ⚠️  Missing steps: {sorted(missing_steps)}")
        for ms in missing_steps:
            issues.append(f"main.py may not reference Step {ms}")
    else:
        print(f"  ✅ All 25 steps (0-24) referenced")

# ── 7. Cross-check src/AGENTS.md module registry ──
print("\n" + "=" * 80)
print("PHASE 7: src/AGENTS.md Module Registry Completeness")
print("=" * 80)

src_agents = src / "AGENTS.md"
if src_agents.exists():
    sa_content = src_agents.read_text(errors="ignore")
    expected_modules = [
        ("template", 0), ("setup", 1), ("tests", 2), ("gnn", 3), ("model_registry", 4),
        ("type_checker", 5), ("validation", 6), ("export", 7), ("visualization", 8),
        ("advanced_visualization", 9), ("ontology", 10), ("render", 11), ("execute", 12),
        ("llm", 13), ("ml_integration", 14), ("audio", 15), ("analysis", 16),
        ("integration", 17), ("security", 18), ("research", 19), ("website", 20),
        ("mcp", 21), ("gui", 22), ("report", 23), ("intelligent_analysis", 24)
    ]
    for mod, step in expected_modules:
        if mod not in sa_content:
            issues.append(f"src/AGENTS.md missing module '{mod}' (Step {step})")
            print(f"  ⚠️  Missing '{mod}' (Step {step})")
        else:
            print(f"  ✅ '{mod}' (Step {step}) present")

# ── 8. Sub-module AGENTS.md check ──
print("\n" + "=" * 80)
print("PHASE 8: Sub-Module Documentation Presence")
print("=" * 80)

for mod_dir in sorted(src.iterdir()):
    if not mod_dir.is_dir() or mod_dir.name.startswith(('.', '_')):
        continue
    if mod_dir.name in ('output', 'tests', '__pycache__', '.benchmarks'):
        continue
    
    subdirs = [d for d in mod_dir.iterdir() if d.is_dir() and not d.name.startswith(('.', '_'))]
    for sd in sorted(subdirs):
        has_agents = (sd / "AGENTS.md").exists()
        has_readme = (sd / "README.md").exists()
        if not has_agents and not has_readme:
            # Only flag if it has Python files (actual code module)
            py_files = list(sd.glob("*.py"))
            if py_files:
                print(f"  ⚠️  {sd.relative_to(src)}: No AGENTS.md or README.md (has {len(py_files)} .py files)")
                issues.append(f"Sub-module {sd.relative_to(src)} has code but no documentation")

# ── Summary ──
print("\n" + "=" * 80)
print(f"SUMMARY: {len(issues)} issues found")
print("=" * 80)
for i, issue in enumerate(issues, 1):
    print(f"  {i}. {issue}")

if not issues:
    print("  🎉 ALL CHECKS PASSED — Perfect documentation-to-code parity!")
