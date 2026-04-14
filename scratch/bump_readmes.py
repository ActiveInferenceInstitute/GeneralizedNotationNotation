from pathlib import Path
import re

count = 0
for md in Path('.').rglob("README.md"):
    if "node_modules" in md.parts or ".venv" in md.parts or "output" in md.parts:
        continue
    
    text = md.read_text()
    
    # Let's replace 'v1.2.0', 'v1.1.2', 'v1.2.2' etc to 'v1.3.0' 
    # ONLY where it says "Version" or "v1." in the status block under the title
    
    # We can also fix "Last Updated: 2026-03-" or "2026-04-12" to "2026-04-14"
    modified = False
    
    new_text = re.sub(r'(Last [uU]pdated\*\*:?\s*)2026-\d{2}-\d{2}', r'\g<1>2026-04-14', text)
    if new_text != text:
        text = new_text
        modified = True
        
    # Bump version where it specifically denotes pipeline version
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

print(f"Updated {count} README files.")
