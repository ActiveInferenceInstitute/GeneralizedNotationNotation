from pathlib import Path
import re

count = 0
for ext in ["AGENTS.md", "SPEC.md"]:
    for md in Path('.').rglob(ext):
        if "node_modules" in md.parts or ".venv" in md.parts or "output" in md.parts or ".git" in md.parts:
            continue
        
        text = md.read_text()
        
        modified = False
        
        new_text = re.sub(r'(Last [uU]pdated\*\*:?\s*)2026-\d{2}-\d{2}', r'\g<1>2026-04-14', text)
        if new_text != text:
            text = new_text
            modified = True
            
        new_text_v = re.sub(r'(\*\*Version\*\*:?\s+)1\.[0-2]\.\d+', r'\g<1>1.3.0', text)
        if new_text_v != text:
            text = new_text_v
            modified = True

        new_text_v2 = re.sub(r'(Pipeline Version\*\*:?\s+)1\.[0-2]\.\d+', r'\g<1>1.3.0', text)
        if new_text_v2 != text:
            text = new_text_v2
            modified = True

        if modified:
            md.write_text(text)
            count += 1
            print(f"Bumped metadata in {md}")

print(f"Updated {count} metadata files.")
