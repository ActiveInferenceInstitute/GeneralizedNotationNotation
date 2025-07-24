#!/usr/bin/env python3
"""
Step 3: GNN File Discovery and Parsing

This step discovers and parses GNN files from the target directory.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import (
    setup_step_logging,
    log_step_start,
    log_step_success,
    log_step_error
)
from utils.argument_utils import EnhancedArgumentParser
from pipeline.config import get_output_dir_for_script, get_pipeline_config

def main():
    """Main GNN processing function."""
    args = EnhancedArgumentParser.parse_step_arguments("3_gnn.py")
    
    # Setup logging
    logger = setup_step_logging("gnn_processing", args)
    
    try:
        # Get pipeline configuration
        config = get_pipeline_config()
        output_dir = get_output_dir_for_script("3_gnn.py", Path(args.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        log_step_start(logger, "Processing GNN files with full multi-format generation")
        
        # Import the comprehensive GNN parsing system
        from gnn.parsers import GNNParsingSystem, GNNFormat
        
        # Initialize the parsing system
        parsing_system = GNNParsingSystem(strict_validation=True)
        
        # Get all supported formats
        supported_formats = parsing_system.get_supported_formats()
        available_serializers = parsing_system.get_available_serializers()
        
        logger.info(f"Initialized GNN parsing system with {len(supported_formats)} supported formats")
        logger.info(f"Available serializers: {list(available_serializers.keys())}")
        
        # Find GNN files
        target_path = Path(args.target_dir)
        gnn_files = []
        
        if args.recursive:
            # Recursive search for GNN files
            for ext in ['.md', '.markdown', '.json', '.xml', '.yaml', '.yml', '.py', '.jl', '.hs', '.lean', '.v', '.thy', '.max', '.proto', '.xsd', '.asn1', '.pkl', '.als', '.z', '.tla', '.agda', '.bnf', '.ebnf']:
                gnn_files.extend(target_path.rglob(f"*{ext}"))
        else:
            # Non-recursive search
            for ext in ['.md', '.markdown', '.json', '.xml', '.yaml', '.yml', '.py', '.jl', '.hs', '.lean', '.v', '.thy', '.max', '.proto', '.xsd', '.asn1', '.pkl', '.als', '.z', '.tla', '.agda', '.bnf', '.ebnf']:
                gnn_files.extend(target_path.glob(f"*{ext}"))
        
        logger.info(f"Found {len(gnn_files)} potential GNN files")
        
        # Processing results
        processing_results = {
            "timestamp": datetime.now().isoformat(),
            "source_directory": str(args.target_dir),
            "output_directory": str(output_dir),
            "processing_config": {
                "recursive": args.recursive,
                "verbose": args.verbose,
                "supported_formats": [fmt.value for fmt in supported_formats],
                "available_serializers": {fmt.value: name for fmt, name in available_serializers.items()}
            },
            "processed_files": [],
            "summary": {
                "total_files": 0,
                "successful_parses": 0,
                "failed_parses": 0,
                "total_formats_generated": 0,
                "formats_per_file": {}
            }
        }
        
        # Process each file
        for file_path in gnn_files:
            try:
                logger.info(f"Processing: {file_path}")
                
                # Try to parse the file
                parse_result = parsing_system.parse_file(file_path)
                
                if not parse_result.success:
                    logger.warning(f"Failed to parse {file_path}: {parse_result.errors}")
                    processing_results["processed_files"].append({
                        "file_name": file_path.name,
                        "file_path": str(file_path),
                        "parse_success": False,
                        "errors": parse_result.errors,
                        "warnings": parse_result.warnings
                    })
                    processing_results["summary"]["failed_parses"] += 1
                    continue
                
                # Create file-specific output directory
                file_output_dir = output_dir / file_path.stem
                file_output_dir.mkdir(exist_ok=True)
                
                # Save the parsed model as JSON
                parsed_model_file = file_output_dir / f"{file_path.stem}_parsed.json"
                with open(parsed_model_file, 'w') as f:
                    json.dump(parse_result.model.to_dict(), f, indent=2, default=str)
                
                # Generate ALL 21 formats
                file_result = {
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "parse_success": True,
                    "parsed_model_file": str(parsed_model_file),
                    "source_format": parse_result.model.source_format.value if parse_result.model.source_format else None,
                    "model_info": {
                        "name": parse_result.model.model_name,
                        "version": parse_result.model.version,
                        "variables_count": len(parse_result.model.variables),
                        "connections_count": len(parse_result.model.connections),
                        "parameters_count": len(parse_result.model.parameters),
                        "equations_count": len(parse_result.model.equations)
                    },
                    "generated_formats": {},
                    "errors": parse_result.errors,
                    "warnings": parse_result.warnings
                }
                
                formats_generated = 0
                
                # Generate each format
                for format_enum in supported_formats:
                    try:
                        # Get the appropriate file extension
                        extension_map = {
                            GNNFormat.MARKDOWN: ".md",
                            GNNFormat.JSON: ".json",
                            GNNFormat.XML: ".xml",
                            GNNFormat.PNML: ".pnml",
                            GNNFormat.YAML: ".yaml",
                            GNNFormat.SCALA: ".scala",
                            GNNFormat.PROTOBUF: ".proto",
                            GNNFormat.PKL: ".pkl",
                            GNNFormat.XSD: ".xsd",
                            GNNFormat.ASN1: ".asn1",
                            GNNFormat.LEAN: ".lean",
                            GNNFormat.COQ: ".v",
                            GNNFormat.PYTHON: ".py",
                            GNNFormat.BNF: ".bnf",
                            GNNFormat.EBNF: ".ebnf",
                            GNNFormat.ISABELLE: ".thy",
                            GNNFormat.MAXIMA: ".max",
                            GNNFormat.ALLOY: ".als",
                            GNNFormat.Z_NOTATION: ".z",
                            GNNFormat.TLA_PLUS: ".tla",
                            GNNFormat.AGDA: ".agda",
                            GNNFormat.HASKELL: ".hs",
                            GNNFormat.PICKLE: ".pkl"
                        }
                        
                        extension = extension_map.get(format_enum, f".{format_enum.value}")
                        output_file = file_output_dir / f"{file_path.stem}_{format_enum.value}{extension}"
                        
                        # Serialize to the specific format
                        serialized_content = parsing_system.serialize(parse_result.model, format_enum)
                        
                        # Write the file
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(serialized_content)
                        
                        # Record the generation
                        file_result["generated_formats"][format_enum.value] = {
                            "output_file": str(output_file),
                            "file_size": output_file.stat().st_size,
                            "success": True
                        }
                        
                        formats_generated += 1
                        
                        if args.verbose:
                            logger.info(f"  Generated {format_enum.value}: {output_file.stat().st_size} bytes")
                        
                    except Exception as e:
                        logger.warning(f"  Failed to generate {format_enum.value}: {e}")
                        file_result["generated_formats"][format_enum.value] = {
                            "output_file": str(file_output_dir / f"{file_path.stem}_{format_enum.value}{extension}"),
                            "error": str(e),
                            "success": False
                        }
                
                # Update summary
                processing_results["summary"]["formats_per_file"][file_path.name] = formats_generated
                processing_results["summary"]["total_formats_generated"] += formats_generated
                
                logger.info(f"Generated {formats_generated} formats for {file_path.name}")
                
                processing_results["processed_files"].append(file_result)
                processing_results["summary"]["successful_parses"] += 1
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                processing_results["processed_files"].append({
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "parse_success": False,
                    "error": str(e)
                })
                processing_results["summary"]["failed_parses"] += 1
        
        # Update total count
        processing_results["summary"]["total_files"] = len(gnn_files)
        
        # Save comprehensive results
        results_file = output_dir / "gnn_processing_results.json"
        with open(results_file, 'w') as f:
            json.dump(processing_results, f, indent=2, default=str)
        
        # Save summary
        summary_file = output_dir / "gnn_processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(processing_results["summary"], f, indent=2)
        
        # Generate format statistics
        format_stats = {}
        for file_result in processing_results["processed_files"]:
            if file_result["parse_success"]:
                for format_name, format_info in file_result["generated_formats"].items():
                    if format_name not in format_stats:
                        format_stats[format_name] = {"successful": 0, "failed": 0, "total_size": 0}
                    
                    if format_info["success"]:
                        format_stats[format_name]["successful"] += 1
                        format_stats[format_name]["total_size"] += format_info.get("file_size", 0)
                    else:
                        format_stats[format_name]["failed"] += 1
        
        # Save format statistics
        stats_file = output_dir / "format_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(format_stats, f, indent=2)
        
        # Determine success
        success = processing_results["summary"]["successful_parses"] > 0
        
        if success:
            total_formats = processing_results["summary"]["total_formats_generated"]
            log_step_success(logger, f"Processed {processing_results['summary']['successful_parses']} files, generated {total_formats} format instances across {len(supported_formats)} formats")
            return 0
        else:
            log_step_error(logger, "No files were successfully processed")
            return 1
            
    except Exception as e:
        log_step_error(logger, f"GNN processing failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 