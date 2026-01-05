import os
import re
from pathlib import Path
import urllib.parse

def check_links():
    root_dir = Path(".")
    broken_links = []
    
    # improved regex to capture markdown links: [text](path)
    # handling possible titles and extra spaces
    link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    
    for markdown_file in root_dir.rglob("*.md"):
        # skip virtual envs and hidden dirs
        if any(part.startswith(".") for part in markdown_file.parts):
            continue
            
        try:
            content = markdown_file.read_text(encoding='utf-8')
        except Exception as e:
            print(f"Error reading {markdown_file}: {e}")
            continue
            
        for match in link_pattern.finditer(content):
            link_text = match.group(1)
            link_target = match.group(2)
            
            # Clean up link target (remove title "...", remove fragments #...)
            target_path_raw = link_target.split()[0]  # remove title if any
            target_path_nobdry = target_path_raw.strip('<>') # remove angle brackets if used
            
            # Parse URL to separate path and fragment
            parsed = urllib.parse.urlparse(target_path_nobdry)
            if parsed.scheme: # Skip external links and absolute file:// links (checked separately)
                continue
                
            path_part = parsed.path
            if not path_part:
                continue # Internal link to same page anchor
                
            # Resolve relative path
            # If starts with /, it's relative to repo root (usually not valid in standard markdown but handled by some viewers)
            # We assume relative to file location for standard markdown
            
            if path_part.startswith("/"):
                # verify from root
                resolved_path = root_dir / path_part.lstrip("/")
            else:
                resolved_path = markdown_file.parent / path_part
                
            resolved_path = resolved_path.resolve()
            
            # Check existence
            if not os.path.exists(resolved_path):
                # Try handling local file without anchor (already stripped fragment, but maybe it was a file path that looked like fragment?)
                # Actually, resolve() handles .. and .
                
                # Check directly if it's a directory or file
                # Sometimes people link to folders
                
                # Report broken
                broken_links.append({
                    "source": str(markdown_file),
                    "link_text": link_text,
                    "target": target_path_raw,
                    "resolved": str(resolved_path)
                })

    print(f"Found {len(broken_links)} broken relative links.")
    for link in broken_links:
        print(f"Source: {link['source']}")
        print(f"  Link: [{link['link_text']}]({link['target']})")
        print(f"  Resolved: {link['resolved']}")
        print("---")

if __name__ == "__main__":
    check_links()
