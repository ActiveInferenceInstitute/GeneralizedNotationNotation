#!/usr/bin/env python3
"""
Generate Comprehensive Visualization Inventory Report
Scans all output directories and catalogs all visualization files with metadata
"""

import json
from pathlib import Path
from datetime import datetime
from PIL import Image
import hashlib
from typing import Dict, List, Any

def get_image_metadata(image_path: Path) -> Dict[str, Any]:
    """Extract metadata from image file."""
    try:
        img = Image.open(image_path)
        return {
            "format": img.format,
            "mode": img.mode,
            "size": list(img.size),
            "width": img.size[0],
            "height": img.size[1],
            "file_size_bytes": image_path.stat().st_size,
            "file_size_kb": round(image_path.stat().st_size / 1024, 2)
        }
    except Exception as e:
        return {"error": str(e)}

def get_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of file."""
    try:
        return hashlib.sha256(file_path.read_bytes()).hexdigest()[:16]
    except:
        return "error"

def scan_visualizations(root_dir: Path) -> List[Dict[str, Any]]:
    """Scan directory tree for all visualization files."""
    viz_extensions = {'.png', '.jpg', '.jpeg', '.svg', '.pdf', '.html'}
    visualizations = []
    
    for viz_file in root_dir.rglob('*'):
        if viz_file.is_file() and viz_file.suffix.lower() in viz_extensions:
            # Determine category based on path
            path_str = str(viz_file.relative_to(root_dir))
            
            if 'pymdp' in path_str.lower():
                framework = 'PyMDP'
                category = 'execution'
            elif 'jax' in path_str.lower():
                framework = 'JAX'
                category = 'execution'
            elif 'discopy' in path_str.lower():
                framework = 'DisCoPy'
                category = 'execution'
            elif 'activeinference' in path_str.lower():
                framework = 'ActiveInference.jl'
                category = 'execution'
            elif 'rxinfer' in path_str.lower():
                framework = 'RxInfer.jl'
                category = 'execution'
            elif '8_visualization' in path_str:
                framework = 'Core'
                category = 'network_analysis'
            elif '9_advanced' in path_str:
                framework = 'Advanced'
                category = 'advanced_analysis'
            else:
                framework = 'Other'
                category = 'other'
            
            # Determine visualization type from filename
            filename_lower = viz_file.stem.lower()
            if 'belief' in filename_lower and 'evolution' in filename_lower:
                viz_type = 'belief_evolution'
            elif 'action' in filename_lower:
                viz_type = 'action_analysis'
            elif 'matrix' in filename_lower or 'heatmap' in filename_lower:
                viz_type = 'matrix_heatmap'
            elif 'network' in filename_lower or 'graph' in filename_lower:
                viz_type = 'network_graph'
            elif 'dashboard' in filename_lower:
                viz_type = 'dashboard'
            elif '3d' in filename_lower or 'trajectory' in filename_lower:
                viz_type = '3d_visualization'
            elif 'preference' in filename_lower or 'prior' in filename_lower:
                viz_type = 'parameter_visualization'
            elif 'diagram' in filename_lower:
                viz_type = 'categorical_diagram'
            else:
                viz_type = 'other'
            
            viz_info = {
                "file_path": str(viz_file.relative_to(root_dir)),
                "filename": viz_file.name,
                "framework": framework,
                "category": category,
                "type": viz_type,
                "extension": viz_file.suffix.lower(),
                "created": datetime.fromtimestamp(viz_file.stat().st_mtime).isoformat(),
                "hash": get_file_hash(viz_file)
            }
            
            # Add image-specific metadata
            if viz_file.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                viz_info["metadata"] = get_image_metadata(viz_file)
            
            visualizations.append(viz_info)
    
    return visualizations

def generate_report(visualizations: List[Dict[str, Any]], output_file: Path):
    """Generate comprehensive report."""
    
    # Calculate statistics
    total_count = len(visualizations)
    by_framework = {}
    by_category = {}
    by_type = {}
    total_size_bytes = 0
    
    for viz in visualizations:
        # Count by framework
        fw = viz['framework']
        by_framework[fw] = by_framework.get(fw, 0) + 1
        
        # Count by category
        cat = viz['category']
        by_category[cat] = by_category.get(cat, 0) + 1
        
        # Count by type
        vtype = viz['type']
        by_type[vtype] = by_type.get(vtype, 0) + 1
        
        # Sum file sizes
        if 'metadata' in viz and 'file_size_bytes' in viz['metadata']:
            total_size_bytes += viz['metadata']['file_size_bytes']
    
    report = {
        "report_generated": datetime.now().isoformat(),
        "summary": {
            "total_visualizations": total_count,
            "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
            "by_framework": by_framework,
            "by_category": by_category,
            "by_type": by_type
        },
        "visualizations": visualizations
    }
    
    # Write JSON report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Generated visualization inventory: {output_file}")
    print(f"📊 Total visualizations found: {total_count}")
    print(f"💾 Total size: {report['summary']['total_size_mb']} MB")
    print(f"\n🔍 By Framework:")
    for fw, count in sorted(by_framework.items(), key=lambda x: x[1], reverse=True):
        print(f"  {fw}: {count}")
    print(f"\n📁 By Category:")
    for cat, count in sorted(by_category.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")
    
    # Generate markdown report
    md_file = output_file.with_suffix('.md')
    with open(md_file, 'w') as f:
        f.write(f"# Visualization Inventory Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- **Total Visualizations**: {total_count}\n")
        f.write(f"- **Total Size**: {report['summary']['total_size_mb']} MB\n\n")
        
        f.write(f"### By Framework\n\n")
        f.write(f"| Framework | Count | Percentage |\n")
        f.write(f"|-----------|-------|------------|\n")
        for fw, count in sorted(by_framework.items(), key=lambda x: x[1], reverse=True):
            pct = round(100 * count / total_count, 1)
            f.write(f"| {fw} | {count} | {pct}% |\n")
        
        f.write(f"\n### By Type\n\n")
        f.write(f"| Type | Count |\n")
        f.write(f"|------|-------|\n")
        for vtype, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
            f.write(f"| {vtype} | {count} |\n")
        
        f.write(f"\n## Detailed Inventory\n\n")
        for viz in sorted(visualizations, key=lambda x: (x['framework'], x['type'])):
            f.write(f"### {viz['filename']}\n\n")
            f.write(f"- **Path**: `{viz['file_path']}`\n")
            f.write(f"- **Framework**: {viz['framework']}\n")
            f.write(f"- **Category**: {viz['category']}\n")
            f.write(f"- **Type**: {viz['type']}\n")
            f.write(f"- **Created**: {viz['created']}\n")
            if 'metadata' in viz:
                meta = viz['metadata']
                if 'width' in meta:
                    f.write(f"- **Dimensions**: {meta['width']}x{meta['height']} px\n")
                if 'file_size_kb' in meta:
                    f.write(f"- **Size**: {meta['file_size_kb']} KB\n")
            f.write(f"\n")
    
    print(f"📄 Generated markdown report: {md_file}")

if __name__ == "__main__":
    root = Path("/home/q/Documents/GitHub/GeneralizedNotationNotation/output")
    output_file = root / "VISUALIZATION_INVENTORY.json"
    
    print("🔍 Scanning for visualizations...")
    visualizations = scan_visualizations(root)
    
    print(f"📊 Found {len(visualizations)} visualization files")
    
    generate_report(visualizations, output_file)
    
    print("\n✅ Visualization inventory complete!")

