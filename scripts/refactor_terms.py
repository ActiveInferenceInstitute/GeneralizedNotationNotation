#!/usr/bin/env python3
import os
import re
from pathlib import Path

TARGET_DIR = Path('src')

RULES = [
    (re.compile(r'\bfallback\b'), 'recovery'),
    (re.compile(r'\bFallback\b'), 'Recovery'),
    (re.compile(r'\bFALLBACK\b'), 'RECOVERY'),
    
    (re.compile(r'\bmock\b'), 'simulated'),
    (re.compile(r'\bMock\b'), 'Simulated'),
    (re.compile(r'\bMOCK\b'), 'SIMULATED'),
    
    (re.compile(r'\blegacy\b'), 'previous'),
    (re.compile(r'\bLegacy\b'), 'Previous'),
    (re.compile(r'\bLEGACY\b'), 'PREVIOUS'),
    
    (re.compile(r'\bstub\b'), 'placeholder'),
    (re.compile(r'\bStub\b'), 'Placeholder'),
    (re.compile(r'\bSTUB\b'), 'PLACEHOLDER'),
    
    (re.compile(r'\bfake\b'), 'simulated'),
    (re.compile(r'\bFake\b'), 'Simulated'),
    (re.compile(r'\bFAKE\b'), 'SIMULATED'),
]

def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        new_content = content
        for pattern, replacement in RULES:
            new_content = pattern.sub(replacement, new_content)
            
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated: {filepath}")
            return True
    except (OSError, UnicodeDecodeError) as e:
        print(f"Error processing {filepath}: {e}")
    return False

def main():
    updated_files = 0
    for root, _, files in os.walk(TARGET_DIR):
        for file in files:
            if file.endswith(('.py', '.md', '.json', '.yaml', '.txt')):
                filepath = Path(root) / file
                if process_file(filepath):
                    updated_files += 1
                    
    print(f"\nCompleted! Modified {updated_files} files.")

if __name__ == "__main__":
    main()
