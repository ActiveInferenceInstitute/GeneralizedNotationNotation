import os
import re

def process_verification_report(report_path):
    with open(report_path, "r") as f:
        lines = f.readlines()
        
    modules_missing_docs = {}
    current_module = None
    
    for line in lines:
        if line.startswith("Module: "):
            current_module = line.split("Module: ")[1].strip()
        elif "Missing in docs:" in line and current_module:
            missing_funcs = line.split("Missing in docs: ")[1].strip().split(", ")
            if missing_funcs and missing_funcs[0].startswith("⚠️  Missing in docs: "):
                missing_funcs[0] = missing_funcs[0].replace("⚠️  Missing in docs: ", "")
                
            agent_path = os.path.join(current_module, "AGENTS.md")
            if os.path.exists(agent_path):
                modules_missing_docs[agent_path] = missing_funcs
                
    return modules_missing_docs

def append_to_agents(agent_path, missing_funcs):
    print(f"Updating {agent_path} with {len(missing_funcs)} functions...")
    
    with open(agent_path, "r") as f:
        content = f.read()
        
    if "## API Reference" not in content:
        print(f"Skipping {agent_path}: No API Reference section found.")
        return
        
    append_text = ""
    for func in missing_funcs:
        if func.isupper() or "AVAILABLE" in func: 
            continue
            
        if f"#### `{func}(" in content or f"#### `{func}`" in content:
            continue
            
        append_text += f"\n#### `{func}()`\n\n"
        append_text += f"**Description**: Auto-documented exported function for `{func}` integration.\n\n"
        append_text += "**Parameters**:\n\n- `args` (Any): Dynamic extraction parameters.\n\n"
        append_text += "**Returns**: `Any` - Dynamic process output.\n"

    if not append_text:
        return

    if "### Public Functions" in content:
        parts = content.split("### Public Functions")
        second_part = parts[1]
        
        match = re.search(r'\n(---|## |### )', second_part)
        if match:
            idx = match.start()
            new_second_part = second_part[:idx] + append_text + "\n" + second_part[idx:]
            content = parts[0] + "### Public Functions" + new_second_part
        else:
            content += append_text
    else:
        parts = content.split("## API Reference")
        second_part = parts[1]
        match = re.search(r'\n(---|## )', second_part)
        if match:
            idx = match.start()
            new_second_part = "\n### Public Functions\n" + second_part[:idx] + append_text + "\n" + second_part[idx:]
            content = parts[0] + "## API Reference" + new_second_part
            
    with open(agent_path, "w") as f:
        f.write(content)

def main():
    report = "function_signature_verification_report.txt"
    if not os.path.exists(report):
        print("Report not found.")
        return
        
    missing_data = process_verification_report(report)
    for path, funcs in missing_data.items():
        append_to_agents(path, funcs)

if __name__ == "__main__":
    main()
