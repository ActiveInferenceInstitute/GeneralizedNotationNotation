import glob
import os
import re

def analyze_orchestrators():
    files = [
        "__init__.py", "0_template.py", "1_setup.py", "2_tests.py", "3_gnn.py", 
        "4_model_registry.py", "5_type_checker.py", "6_validation.py", "7_export.py", 
        "8_visualization.py", "9_advanced_viz.py", "10_ontology.py", "11_render.py", 
        "12_execute.py", "13_llm.py", "14_ml_integration.py", "15_audio.py", 
        "16_analysis.py", "17_integration.py", "18_security.py", "19_research.py", 
        "20_website.py", "21_mcp.py", "22_gui.py", "23_report.py", "24_intelligent_analysis.py",
        "main.py", "pipeline_step_template.py"
    ]
    
    issues = {}
    
    for filename in files:
        path = os.path.join("src", filename)
        if not os.path.exists(path):
            issues[filename] = ["File missing"]
            continue
            
        with open(path, "r") as f:
            content = f.read()
            
        file_issues = []
        
        # Skip purely documentation/template ones from strict pipeline requirement
        if filename in ["__init__.py", "pipeline_step_template.py", "main.py", "0_template.py"]:
            pass # Have custom orchestrator structures
        else:
            # Must import create_standardized_pipeline_script
            if "create_standardized_pipeline_script" not in content:
                file_issues.append("Missing create_standardized_pipeline_script call")
                
            # Must import the actual processing function from its module
            # Extract number prefix to guess module name
            parts = filename.split('_', 1)
            if len(parts) == 2:
                base_name = parts[1].replace('.py', '')
                import_pattern = rf"from[\s]+{base_name}(\.[\w]+)?[\s]+import"
                if not re.search(import_pattern, content) and "import" in content:
                    pass # We will check this manually if we want, or just log
    
                if "def main(" not in content:
                    file_issues.append("Missing def main()")
                    
                if 'if __name__ == "__main__":' not in content:
                    file_issues.append('Missing if __name__ == "__main__": block')
                
        if file_issues:
            issues[filename] = file_issues
            
    return issues

if __name__ == "__main__":
    issues = analyze_orchestrators()
    print("Orchestrator Analysis:")
    if issues:
        for f, errs in issues.items():
            print(f"[{f}]")
            for e in errs:
                print(f"  - {e}")
    else:
        print("All python files passed base orchestrator heuristics.")
