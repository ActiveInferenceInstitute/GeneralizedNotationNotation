from pathlib import Path
import re

count = 0
for md in Path("doc/gnn/modules").rglob("*.md"):
    text = md.read_text()
    
    # Fix web url links wrapped with src references
    pattern = r"\.\./\.\./\.\./src/[^/]+/(https?://)"
    fixed_text = re.sub(pattern, r"\1", text)
    
    # Fix internal folder links in visualization that got ported out:
    # We should point them to the src/visualization ones, e.g., `../../../src/visualization/core/README.md`
    viz_pattern = r"\]\((core/README\.md|parse/README\.md|plotting/README\.md|graph/README\.md|matrix/README\.md|analysis/README\.md|ontology/README\.md)\)"
    fixed_text = re.sub(viz_pattern, r"](../../../src/visualization/\1)", fixed_text)
    
    # Fix testing/test_round_trip.py in 03_gnn.md
    if md.name == "03_gnn.md":
        fixed_text = fixed_text.replace("](testing/test_round_trip.py)", "](../../../src/gnn/testing/test_round_trip.py)")
        
    if fixed_text != text:
        md.write_text(fixed_text)
        count += 1
        print(f"Fixed URL artifacts in {md.name}")

print(f"Updated {count} files.")
