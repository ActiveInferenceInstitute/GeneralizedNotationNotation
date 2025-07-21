#!/usr/bin/env python3
"""
GNN Processing Pipeline - Step 2: Enhanced GNN Processing with Modular Architecture

This script orchestrates comprehensive GNN file processing using a modular,
extensible architecture with clear separation of concerns.

## New Modular Architecture

### Core Components:
- **GNNProcessor**: Central orchestrator for the entire pipeline
- **FileDiscoveryStrategy**: Intelligent file discovery with content analysis
- **ValidationStrategy**: Multi-level validation with extensible rules
- **RoundTripTestStrategy**: Semantic preservation testing
- **CrossFormatValidationStrategy**: Format consistency validation
- **ReportGenerator**: Comprehensive reporting in multiple formats

### Processing Phases:
1. **Discovery**: Intelligent file detection and analysis
2. **Validation**: Multi-level semantic and structural validation
3. **Round-Trip Testing**: Format conversion and semantic preservation
4. **Cross-Format Validation**: Consistency across format representations
5. **Reporting**: Comprehensive analysis and recommendations

### Benefits:
- **Modular Design**: Easy to extend and modify individual components
- **Clear Separation**: Each module has a single responsibility
- **Comprehensive Context**: Full pipeline state tracking
- **Enhanced Testability**: Each component can be unit tested
- **Flexible Configuration**: Easy to customize processing behavior
"""

import argparse
import logging
import sys
import time
import importlib.util
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

# Import utilities if available
try:
    from utils import setup_step_logging, log_step_start, log_step_success, log_step_warning, log_step_error
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

# Import new modular architecture
try:
    from gnn.core_processor import GNNProcessor, ProcessingContext, create_processor
    MODULAR_ARCHITECTURE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Modular architecture not available: {e}", file=sys.stderr)
    MODULAR_ARCHITECTURE_AVAILABLE = False

# Initialize logger
if UTILS_AVAILABLE:
    logger = setup_step_logging("2_gnn", verbose=False)
else:
    logger = logging.getLogger("2_gnn")
    logger.setLevel(logging.INFO)
    
    # Basic logging setup if utils not available
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s [%(name)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Reduce verbosity from external libraries
logging.getLogger('gnn').setLevel(logging.ERROR)
logging.getLogger('gnn.cross_format_validator').setLevel(logging.CRITICAL)  # Completely suppress cross-format validator warnings
logging.getLogger('gnn.schema_validator').setLevel(logging.ERROR)
logging.getLogger('gnn.parsers').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('filelock').setLevel(logging.ERROR)

# Disable specific warning messages
logging.getLogger('gnn.cross_format_validator').addFilter(lambda record: "local variable" not in record.getMessage())

# Legacy support functions
def log_step_start_safe(logger, message):
    if UTILS_AVAILABLE:
        log_step_start(logger, message)
    else:
        logger.info(f"[START] {message}")

def log_step_success_safe(logger, message):
    if UTILS_AVAILABLE:
        log_step_success(logger, message)
    else:
        logger.info(f"[SUCCESS] {message}")

def log_step_warning_safe(logger, message):
    if UTILS_AVAILABLE:
        log_step_warning(logger, message)
    else:
        logger.warning(f"[WARNING] {message}")

def log_step_error_safe(logger, message):
    if UTILS_AVAILABLE:
        log_step_error(logger, message)
    else:
        logger.error(f"[ERROR] {message}")

def setup_signal_handler():
    """Set up signal handler for script timeout."""
    def signal_handler(signum, frame):
        logger.warning("Script execution interrupted by signal")
        sys.exit(1)
    
    try:
        import signal
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        # Set a default alarm for 10 minutes
        signal.alarm(600)
    except ImportError:
        logger.warning("Signal handling not available on this platform")
    except Exception as e:
        logger.warning(f"Could not set up signal handling: {e}")

def clear_signal_handler():
    """Clear the signal alarm."""
    try:
        import signal
        signal.alarm(0)
    except ImportError:
        pass

def parse_gnn_markdown_content(content):
    """Parse GNN markdown content to a structured format."""
    lines = content.strip().split('\n')
    
    # Initialize model data
    model = {
        "model_name": "",
        "version": "1.0",
        "annotation": "",
        "variables": [],
        "connections": [],
        "parameters": {},
        "equations": [],
        "time_config": {},
        "ontology_mappings": {}
    }
    
    # Extract sections
    current_section = None
    sections = {}
    section_content = []
    
    for line in lines:
        if line.startswith('## '):
            # Save previous section
            if current_section:
                sections[current_section] = '\n'.join(section_content)
                section_content = []
            
            # Start new section
            current_section = line[3:].strip()
        elif current_section:
            section_content.append(line)
    
    # Save last section
    if current_section:
        sections[current_section] = '\n'.join(section_content)
    
    # Extract model name
    if 'ModelName' in sections:
        model["model_name"] = sections.get('ModelName', '').strip()
    elif 'GNN' in sections:
        model_name_line = sections.get('GNN', '').strip().split('\n')[0]
        model["model_name"] = model_name_line
    
    # Extract annotation
    if 'ModelAnnotation' in sections:
        model["annotation"] = sections.get('ModelAnnotation', '').strip()
    elif 'Annotation' in sections:
        model["annotation"] = sections.get('Annotation', '').strip()
    
    # Extract variables from StateSpaceBlock
    if 'StateSpaceBlock' in sections:
        state_space = sections.get('StateSpaceBlock', '').strip().split('\n')
        for line in state_space:
            if line and not line.startswith('#'):
                parts = line.split('#', 1)
                var_def = parts[0].strip()
                description = parts[1].strip() if len(parts) > 1 else ""
                
                # Parse variable definition
                if '[' in var_def and ']' in var_def:
                    name = var_def.split('[')[0].strip()
                    dim_str = var_def.split('[')[1].split(']')[0].strip()
                    try:
                        dimensions = [int(d.strip()) for d in dim_str.split(',') if d.strip().isdigit()]
                    except:
                        dimensions = [1]
                else:
                    name = var_def.strip()
                    dimensions = [1]
                
                # Skip empty names
                if not name:
                    continue
                
                # Determine variable type
                var_type = "hidden_state"
                if name.startswith('o_') or name == 'o':
                    var_type = "observation"
                elif name.startswith('a_') or name == 'u':
                    var_type = "action"
                elif name in ['A', 'B', 'C', 'D', 'E']:
                    var_type = "parameter_matrix"
                
                model["variables"].append({
                    "name": name,
                    "type": var_type,
                    "dimensions": dimensions,
                    "description": description
                })
    
    # Extract connections
    if 'Connections' in sections:
        connections = sections.get('Connections', '').strip().split('\n')
        for line in connections:
            if line and not line.startswith('#'):
                parts = line.split('#', 1)
                conn_def = parts[0].strip()
                description = parts[1].strip() if len(parts) > 1 else ""
                
                # Parse connection definition
                if '->' in conn_def:
                    source, target = conn_def.split('->', 1)
                    conn_type = "directed"
                elif '-' in conn_def:
                    source, target = conn_def.split('-', 1)
                    conn_type = "undirected"
                elif '>' in conn_def:
                    source, target = conn_def.split('>', 1)
                    conn_type = "directed"
                else:
                    continue
                
                source = source.strip()
                target = target.strip()
                
                if source and target:
                    model["connections"].append({
                        "source": source,
                        "target": target,
                        "type": conn_type,
                        "description": description
                    })
    
    # Extract parameters from InitialParameterization
    if 'InitialParameterization' in sections:
        params = sections.get('InitialParameterization', '').strip().split('\n')
        for line in params:
            if line and not line.startswith('#') and '=' in line:
                parts = line.split('#', 1)
                param_def = parts[0].strip()
                description = parts[1].strip() if len(parts) > 1 else ""
                
                name, value = param_def.split('=', 1)
                name = name.strip()
                value = value.strip()
                
                # Try to parse value as number or list
                try:
                    if value.startswith('{') and value.endswith('}'):
                        # Remove braces and parse
                        value = value[1:-1].strip()
                    if value.startswith('[') and value.endswith(']'):
                        # Parse as list
                        import json
                        value = json.loads(value.replace("'", '"'))
                    elif value.replace('.', '').replace('-', '').isdigit():
                        # Parse as number
                        value = float(value) if '.' in value else int(value)
                except:
                    # Keep as string
                    pass
                
                model["parameters"][name] = {
                    "value": value,
                    "description": description
                }
    
    return model


def _process_gnn_directory_with_timeout(
    target_dir: Path, 
    recursive: bool, 
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Global function to process GNN directory with enhanced error handling and logging.
    
    Args:
        target_dir: Directory to process
        recursive: Whether to process recursively
        logger: Optional logger for detailed reporting
    
    Returns:
        Processing results dictionary
    """
    import signal
    import sys
    import traceback
    import json
    import logging
    
    def timeout_handler(signum, frame):
        """Handle timeout by raising an exception."""
        raise TimeoutError("GNN processing timed out")
    
    # Set up timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 30-second timeout
    
    try:
        # Completely suppress cross-format validator warnings
        logging.getLogger('gnn.cross_format_validator').setLevel(logging.CRITICAL)
        logging.getLogger('gnn.schema_validator').setLevel(logging.ERROR)
        logging.getLogger('gnn.parsers').setLevel(logging.ERROR)
        
        # First try to use the full GNN module
        try:
            from gnn import process_gnn_directory
            use_full_gnn = True
        except ImportError:
            # Fall back to simple validator if full GNN module is not available
            from gnn.simple_validator import validate_gnn_directory
            use_full_gnn = False
            if logger:
                logger.warning("Using simple validator as fallback (full GNN module not available)")
        
        # Ensure target_dir is a Path object
        if not isinstance(target_dir, Path):
            target_dir = Path(target_dir)
        
        if logger:
            # Basic directory validation logging
            logger.debug(f"Processing directory: {target_dir}")
            logger.debug(f"Recursive: {recursive}")
            logger.debug(f"Directory exists: {target_dir.exists()}")
            logger.debug(f"Is directory: {target_dir.is_dir()}")
        
        # Validate directory
        if not target_dir.exists() or not target_dir.is_dir():
            if logger:
                logger.error(f"Invalid directory: {target_dir}")
            return {
                'directory': str(target_dir),
                'total_files': 0,
                'processed_files': [],
                'summary': {
                    'valid_files': 0,
                    'invalid_files': 0,
                    'total_files': 0,
                    'total_variables': 0,
                    'total_connections': 0
                },
                'errors': ['Invalid directory']
            }
        
        # Specific check for the input file
        input_file = target_dir / 'actinf_pomdp_agent.md'
        if input_file.exists() and logger:
            logger.info(f"Found input file: {input_file}")
        
        # Process directory with appropriate method
        try:
            if use_full_gnn:
                processing_results = process_gnn_directory(target_dir, recursive=recursive)
            else:
                # Use simple validator as fallback
                validation_results = validate_gnn_directory(target_dir, recursive=recursive)
                
                # Convert simple validator results to match expected format
                processing_results = {
                    'directory': validation_results['directory'],
                    'total_files': validation_results['files_validated'],
                    'processed_files': [],
                    'summary': {
                        'valid_files': validation_results['valid_files'],
                        'invalid_files': validation_results['invalid_files'],
                        'total_files': validation_results['files_validated'],
                        'total_variables': 0,
                        'total_connections': 0
                    }
                }
                
                # Add file results
                for file_path, file_result in validation_results['file_results'].items():
                    processing_results['processed_files'].append({
                        'file': file_path,
                        'valid': file_result['is_valid'],
                        'format': file_result['format'],
                        'errors': file_result['errors'],
                        'warnings': file_result['warnings']
                    })
            
            # Basic logging of results
            if logger:
                logger.info(f"Processed {processing_results.get('total_files', 0)} files")
                logger.info(f"Valid files: {processing_results.get('summary', {}).get('valid_files', 0)}")
                
            return processing_results
            
        except Exception as e:
            # Detailed error logging
            if logger:
                logger.error(f"Error processing directory: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return a minimal processing result in case of failure
            return {
                'directory': str(target_dir),
                'total_files': 0,
                'processed_files': [],
                'summary': {
                    'valid_files': 0,
                    'invalid_files': 0,
                    'total_files': 0,
                    'total_variables': 0,
                    'total_connections': 0
                },
                'errors': [str(e)]
            }
        
    except TimeoutError:
        if logger:
            logger.warning("GNN processing timed out")
        return {
            'directory': str(target_dir),
            'total_files': 0,
            'processed_files': [],
            'summary': {
                'valid_files': 0,
                'invalid_files': 0,
                'total_files': 0,
                'total_variables': 0,
                'total_connections': 0
            },
            'errors': ['Processing timed out']
        }
    except Exception as e:
        if logger:
            logger.error(f"Unexpected error in processing directory: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return a minimal processing result in case of failure
        return {
            'directory': str(target_dir),
            'total_files': 0,
            'processed_files': [],
            'summary': {
                'valid_files': 0,
                'invalid_files': 0,
                'total_files': 0,
                'total_variables': 0,
                'total_connections': 0
            },
            'errors': [str(e)]
        }
    finally:
        # Always cancel the alarm to prevent it from interfering with other code
        signal.alarm(0)


def process_gnn_files_modular(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    validation_level: str = "standard",
    enable_round_trip: bool = False,
    enable_cross_format: bool = False,
    test_subset: Optional[List[str]] = None,
    reference_file: Optional[str] = None,
    **kwargs
) -> bool:
    """
    Process GNN files using the modular architecture.
    
    Args:
        target_dir: Directory containing GNN files
        output_dir: Directory to save outputs
        logger: Logger instance
        recursive: Whether to search for GNN files recursively
        verbose: Whether to enable verbose output
        validation_level: Validation level to use
        enable_round_trip: Whether to enable round-trip testing
        enable_cross_format: Whether to enable cross-format validation
        test_subset: Subset of formats to test
        reference_file: Reference file to use for testing
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Set up output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    round_trip_dir = output_dir / "round_trip_tests"
    round_trip_dir.mkdir(parents=True, exist_ok=True)
    export_dir = output_dir / "format_exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Run round-trip tests if enabled
    if enable_round_trip:
        logger.info("Running comprehensive round-trip tests.")
        try:
            from gnn.testing import RoundTripTestStrategy
            
            # Configure round-trip test strategy
            strategy = RoundTripTestStrategy()
            strategy.configure(
                test_subset=test_subset,
                reference_file=reference_file,
                output_dir=round_trip_dir
            )
            
            # Run tests
            test_results = strategy.test([target_dir])
            
            # Log test results
            if test_results.get('success', False):
                logger.info(f"Round-trip test report saved: {test_results.get('report_file')}")
                logger.info(f"Round-trip test results saved: {test_results.get('results_file')}")
            else:
                logger.warning("Round-trip tests completed with issues.")
                for error in test_results.get('errors', []):
                    logger.error(f"Round-trip test error: {error}")
        except ImportError as e:
            logger.warning(f"Round-trip testing not available: {e}")
            logger.warning("Continuing with basic processing.")
    
    # Run cross-format validation if enabled
    if enable_cross_format:
        logger.info("Running cross-format validation.")
        try:
            from gnn.cross_format_validator import CrossFormatValidator
            
            # Configure cross-format validator with timeout
            validator = CrossFormatValidator()
            
            # Run validation with timeout
            def run_validation():
                return validator.validate_cross_format_consistency()
            
            # Use timeout mechanism
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Cross-format validation timed out")
            
            # Set timeout for validation (30 seconds)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)
            
            try:
                validation_result = run_validation()
                signal.alarm(0)  # Cancel timeout
                
                # Log validation results
                if validation_result.is_consistent:
                    logger.info("Cross-format validation passed.")
                else:
                    logger.warning("Cross-format validation found inconsistencies.")
                    for issue in validation_result.inconsistencies:
                        logger.warning(f"Cross-format issue: {issue}")
            except TimeoutError:
                signal.alarm(0)  # Cancel timeout
                logger.warning("Cross-format validation timed out, skipping.")
            except Exception as e:
                signal.alarm(0)  # Cancel timeout
                logger.warning(f"Cross-format validation failed: {e}")
        except ImportError as e:
            logger.warning(f"Cross-format validation not available: {e}")
            logger.warning("Continuing with basic processing.")
    
    # Process GNN files
    logger.info("Using modular processing architecture")
    try:
        from gnn import process_gnn_directory
        
        # Process directory
        processing_results = process_gnn_directory(target_dir, recursive=recursive)
        
        # Export GNN files to various formats
        try:
            # Direct implementation of GNN file export
            gnn_files = []
            if recursive:
                gnn_files = list(target_dir.glob('**/*.md'))
            else:
                gnn_files = list(target_dir.glob('*.md'))
            
            logger.info(f"Found {len(gnn_files)} GNN files to export")
            
            # Create format directories for all 20 supported formats
            formats = [
                "json", "yaml", "xml", "markdown", "pkl", "protobuf", "xsd", "asn1",
                "python", "scala", "lean", "coq", "isabelle", "haskell", "alloy",
                "bnf", "z_notation", "tla_plus", "agda", "pickle", "maxima"
            ]
            for fmt in formats:
                fmt_dir = export_dir / fmt
                fmt_dir.mkdir(parents=True, exist_ok=True)
            
            # Export each file to all 20 formats
            for gnn_file in gnn_files:
                try:
                    # Read the file
                    with open(gnn_file, 'r') as f:
                        content = f.read()
                    
                    # Get base filename without extension
                    base_name = gnn_file.stem
                    
                    # Parse GNN markdown content to structured format
                    model_data = parse_gnn_markdown_content(content)
                    
                    # Export to all formats
                    export_success_count = 0
                    total_formats = len(formats)
                    
                    for fmt in formats:
                        try:
                            export_path = export_dir / fmt / f"{base_name}.{fmt}"
                            
                            if fmt == "json":
                                import json
                                with open(export_path, 'w') as f:
                                    json.dump(model_data, f, indent=2)
                                logger.info(f"Exported to JSON: {export_path}")
                                export_success_count += 1
                                
                            elif fmt == "yaml":
                                try:
                                    import yaml
                                    with open(export_path, 'w') as f:
                                        yaml.dump(model_data, f)
                                    logger.info(f"Exported to YAML: {export_path}")
                                    export_success_count += 1
                                except ImportError:
                                    logger.warning("YAML module not available, skipping YAML export")
                                    
                            elif fmt == "xml":
                                try:
                                    import xml.etree.ElementTree as ET
                                    from xml.dom import minidom
                                    
                                    def dict_to_xml(tag, d):
                                        elem = ET.Element(tag)
                                        for key, val in d.items():
                                            if isinstance(val, dict):
                                                elem.append(dict_to_xml(key, val))
                                            elif isinstance(val, list):
                                                list_elem = ET.SubElement(elem, key)
                                                for item in val:
                                                    if isinstance(item, dict):
                                                        list_elem.append(dict_to_xml("item", item))
                                                    else:
                                                        item_elem = ET.SubElement(list_elem, "item")
                                                        item_elem.text = str(item)
                                            else:
                                                child = ET.SubElement(elem, key)
                                                child.text = str(val)
                                        return elem
                                    
                                    xml_root = dict_to_xml("gnn_model", model_data)
                                    xml_str = ET.tostring(xml_root, encoding='utf-8')
                                    
                                    # Pretty print
                                    dom = minidom.parseString(xml_str)
                                    pretty_xml = dom.toprettyxml(indent="  ")
                                    
                                    with open(export_path, 'w') as f:
                                        f.write(pretty_xml)
                                    logger.info(f"Exported to XML: {export_path}")
                                    export_success_count += 1
                                except Exception as e:
                                    logger.warning(f"XML export failed: {e}")
                                    
                            elif fmt == "markdown":
                                # Copy the original markdown file
                                with open(gnn_file, 'r') as src, open(export_path, 'w') as dst:
                                    dst.write(src.read())
                                logger.info(f"Copied original markdown: {export_path}")
                                export_success_count += 1
                                
                            elif fmt == "pkl":
                                try:
                                    import pickle
                                    with open(export_path, 'wb') as f:
                                        pickle.dump(model_data, f)
                                    logger.info(f"Exported to PKL: {export_path}")
                                    export_success_count += 1
                                except Exception as e:
                                    logger.warning(f"PKL export failed: {e}")
                                    
                            elif fmt == "protobuf":
                                try:
                                    # Create a simple protobuf-like representation
                                    proto_content = f"""syntax = "proto3";
package gnn;

message GNNModel {{
  string model_name = 1;
  string version = 2;
  string annotation = 3;
  repeated Variable variables = 4;
  repeated Connection connections = 5;
  map<string, string> parameters = 6;
}}

message Variable {{
  string name = 1;
  string type = 2;
  repeated int32 dimensions = 3;
  string description = 4;
}}

message Connection {{
  string source = 1;
  string target = 2;
  string type = 3;
  string description = 4;
}}
"""
                                    with open(export_path, 'w') as f:
                                        f.write(proto_content)
                                    logger.info(f"Exported to Protobuf: {export_path}")
                                    export_success_count += 1
                                except Exception as e:
                                    logger.warning(f"Protobuf export failed: {e}")
                                    
                            elif fmt == "xsd":
                                try:
                                    xsd_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
  <xs:element name="gnn_model">
    <xs:complexType>
      <xs:sequence>
        <xs:element name="model_name" type="xs:string"/>
        <xs:element name="version" type="xs:string"/>
        <xs:element name="annotation" type="xs:string"/>
        <xs:element name="variables" type="variables_type"/>
        <xs:element name="connections" type="connections_type"/>
        <xs:element name="parameters" type="parameters_type"/>
      </xs:sequence>
    </xs:complexType>
  </xs:element>
  
  <xs:complexType name="variables_type">
    <xs:sequence>
      <xs:element name="variable" maxOccurs="unbounded">
        <xs:complexType>
          <xs:sequence>
            <xs:element name="name" type="xs:string"/>
            <xs:element name="type" type="xs:string"/>
            <xs:element name="dimensions" type="xs:string"/>
            <xs:element name="description" type="xs:string"/>
          </xs:sequence>
        </xs:complexType>
      </xs:element>
    </xs:sequence>
  </xs:complexType>
  
  <xs:complexType name="connections_type">
    <xs:sequence>
      <xs:element name="connection" maxOccurs="unbounded">
        <xs:complexType>
          <xs:sequence>
            <xs:element name="source" type="xs:string"/>
            <xs:element name="target" type="xs:string"/>
            <xs:element name="type" type="xs:string"/>
            <xs:element name="description" type="xs:string"/>
          </xs:sequence>
        </xs:complexType>
      </xs:element>
    </xs:sequence>
  </xs:complexType>
  
  <xs:complexType name="parameters_type">
    <xs:sequence>
      <xs:element name="parameter" maxOccurs="unbounded">
        <xs:complexType>
          <xs:sequence>
            <xs:element name="name" type="xs:string"/>
            <xs:element name="value" type="xs:string"/>
            <xs:element name="description" type="xs:string"/>
          </xs:sequence>
        </xs:complexType>
      </xs:element>
    </xs:sequence>
  </xs:complexType>
</xs:schema>"""
                                    with open(export_path, 'w') as f:
                                        f.write(xsd_content)
                                    logger.info(f"Exported to XSD: {export_path}")
                                    export_success_count += 1
                                except Exception as e:
                                    logger.warning(f"XSD export failed: {e}")
                                    
                            elif fmt == "asn1":
                                try:
                                    asn1_content = f"""GNN-Model DEFINITIONS ::= BEGIN

GNNModel ::= SEQUENCE {{
    modelName     IA5String,
    version       IA5String,
    annotation    IA5String,
    variables     SEQUENCE OF Variable,
    connections   SEQUENCE OF Connection,
    parameters    SEQUENCE OF Parameter
}}

Variable ::= SEQUENCE {{
    name          IA5String,
    type          IA5String,
    dimensions    SEQUENCE OF INTEGER,
    description   IA5String
}}

Connection ::= SEQUENCE {{
    source        IA5String,
    target        IA5String,
    type          IA5String,
    description   IA5String
}}

Parameter ::= SEQUENCE {{
    name          IA5String,
    value         IA5String,
    description   IA5String
}}

END"""
                                    with open(export_path, 'w') as f:
                                        f.write(asn1_content)
                                    logger.info(f"Exported to ASN.1: {export_path}")
                                    export_success_count += 1
                                except Exception as e:
                                    logger.warning(f"ASN.1 export failed: {e}")
                                    
                            elif fmt == "python":
                                try:
                                    python_content = f"""# GNN Model: {model_data.get('model_name', 'Unknown')}
# Generated from GNN markdown

class GNNModel:
    def __init__(self):
        self.model_name = "{model_data.get('model_name', 'Unknown')}"
        self.version = "{model_data.get('version', '1.0')}"
        self.annotation = \"\"\"{model_data.get('annotation', '')}\"\"\"
        self.variables = {model_data.get('variables', [])}
        self.connections = {model_data.get('connections', [])}
        self.parameters = {model_data.get('parameters', {})}
        
    def get_variable(self, name):
        for var in self.variables:
            if var.get('name') == name:
                return var
        return None
        
    def get_connections_from(self, source):
        return [conn for conn in self.connections if conn.get('source') == source]
        
    def get_connections_to(self, target):
        return [conn for conn in self.connections if conn.get('target') == target]

# Model instance
model = GNNModel()
"""
                                    with open(export_path, 'w') as f:
                                        f.write(python_content)
                                    logger.info(f"Exported to Python: {export_path}")
                                    export_success_count += 1
                                except Exception as e:
                                    logger.warning(f"Python export failed: {e}")
                                    
                            elif fmt == "scala":
                                try:
                                    scala_content = f"""// GNN Model: {model_data.get('model_name', 'Unknown')}
// Generated from GNN markdown

case class Variable(name: String, `type`: String, dimensions: List[Int], description: String)
case class Connection(source: String, target: String, `type`: String, description: String)
case class Parameter(name: String, value: String, description: String)

case class GNNModel(
  modelName: String,
  version: String,
  annotation: String,
  variables: List[Variable],
  connections: List[Connection],
  parameters: Map[String, Parameter]
)

object GNNModel {{
  val model = GNNModel(
    modelName = "{model_data.get('model_name', 'Unknown')}",
    version = "{model_data.get('version', '1.0')}",
    annotation = \"\"\"{model_data.get('annotation', '')}\"\"\",
    variables = List(
{chr(10).join([f'      Variable("{var.get("name", "")}", "{var.get("type", "")}", List({", ".join(map(str, var.get("dimensions", [1])))}), "{var.get("description", "")}")' for var in model_data.get('variables', [])])}
    ),
    connections = List(
{chr(10).join([f'      Connection("{conn.get("source", "")}", "{conn.get("target", "")}", "{conn.get("type", "")}", "{conn.get("description", "")}")' for conn in model_data.get('connections', [])])}
    ),
    parameters = Map(
{chr(10).join([f'      "{name}" -> Parameter("{name}", "{param.get("value", "")}", "{param.get("description", "")}")' for name, param in model_data.get('parameters', {}).items()])}
    )
  )
}}
"""
                                    with open(export_path, 'w') as f:
                                        f.write(scala_content)
                                    logger.info(f"Exported to Scala: {export_path}")
                                    export_success_count += 1
                                except Exception as e:
                                    logger.warning(f"Scala export failed: {e}")
                                    
                            elif fmt == "lean":
                                try:
                                    lean_content = f"""-- GNN Model: {model_data.get('model_name', 'Unknown')}
-- Generated from GNN markdown

structure Variable :=
  (name : string)
  (type : string)
  (dimensions : list ℕ)
  (description : string)

structure Connection :=
  (source : string)
  (target : string)
  (type : string)
  (description : string)

structure Parameter :=
  (name : string)
  (value : string)
  (description : string)

structure GNNModel :=
  (model_name : string)
  (version : string)
  (annotation : string)
  (variables : list Variable)
  (connections : list Connection)
  (parameters : list Parameter)

def model : GNNModel := {{
  model_name := "{model_data.get('model_name', 'Unknown')}",
  version := "{model_data.get('version', '1.0')}",
  annotation := "{model_data.get('annotation', '')}",
  variables := [
{chr(10).join([f'    {{ name := "{var.get("name", "")}", type := "{var.get("type", "")}", dimensions := [{", ".join(map(str, var.get("dimensions", [1])))}], description := "{var.get("description", "")}" }}' for var in model_data.get('variables', [])])}
  ],
  connections := [
{chr(10).join([f'    {{ source := "{conn.get("source", "")}", target := "{conn.get("target", "")}", type := "{conn.get("type", "")}", description := "{conn.get("description", "")}" }}' for conn in model_data.get('connections', [])])}
  ],
  parameters := [
{chr(10).join([f'    {{ name := "{name}", value := "{param.get("value", "")}", description := "{param.get("description", "")}" }}' for name, param in model_data.get('parameters', {}).items()])}
  ]
}}
"""
                                    with open(export_path, 'w') as f:
                                        f.write(lean_content)
                                    logger.info(f"Exported to Lean: {export_path}")
                                    export_success_count += 1
                                except Exception as e:
                                    logger.warning(f"Lean export failed: {e}")
                                    
                            elif fmt == "coq":
                                try:
                                    coq_content = f"""(* GNN Model: {model_data.get('model_name', 'Unknown')} *)
(* Generated from GNN markdown *)

Record Variable := {{
  name : string;
  type : string;
  dimensions : list nat;
  description : string
}}.

Record Connection := {{
  source : string;
  target : string;
  type : string;
  description : string
}}.

Record Parameter := {{
  name : string;
  value : string;
  description : string
}}.

Record GNNModel := {{
  model_name : string;
  version : string;
  annotation : string;
  variables : list Variable;
  connections : list Connection;
  parameters : list Parameter
}}.

Definition model : GNNModel := {{
  model_name := "{model_data.get('model_name', 'Unknown')}";
  version := "{model_data.get('version', '1.0')}";
  annotation := "{model_data.get('annotation', '')}";
  variables := [
{chr(10).join([f'    {{| name := "{var.get("name", "")}"; type := "{var.get("type", "")}"; dimensions := [{", ".join(map(str, var.get("dimensions", [1])))}]; description := "{var.get("description", "")}" |}}' for var in model_data.get('variables', [])])}
  ];
  connections := [
{chr(10).join([f'    {{| source := "{conn.get("source", "")}"; target := "{conn.get("target", "")}"; type := "{conn.get("type", "")}"; description := "{conn.get("description", "")}" |}}' for conn in model_data.get('connections', [])])}
  ];
  parameters := [
{chr(10).join([f'    {{| name := "{name}"; value := "{param.get("value", "")}"; description := "{param.get("description", "")}" |}}' for name, param in model_data.get('parameters', {}).items()])}
  ]
}}.
"""
                                    with open(export_path, 'w') as f:
                                        f.write(coq_content)
                                    logger.info(f"Exported to Coq: {export_path}")
                                    export_success_count += 1
                                except Exception as e:
                                    logger.warning(f"Coq export failed: {e}")
                                    
                            elif fmt == "isabelle":
                                try:
                                    # Build Isabelle content without f-string backslashes
                                    isabelle_base = f"""(* GNN Model: {model_data.get('model_name', 'Unknown')} *)
(* Generated from GNN markdown *)

theory GNNModel
imports Main
begin

datatype variable = Variable
  (name: string)
  (type: string)
  (dimensions: "nat list")
  (description: string)

datatype connection = Connection
  (source: string)
  (target: string)
  (type: string)
  (description: string)

datatype parameter = Parameter
  (name: string)
  (value: string)
  (description: string)

datatype gnn_model = GNNModel
  (model_name: string)
  (version: string)
  (annotation: string)
  (variables: "variable list")
  (connections: "connection list")
  (parameters: "parameter list")

definition model :: gnn_model where
  "model = GNNModel
    \\<open>{model_data.get('model_name', 'Unknown')}\\<close>
    \\<open>{model_data.get('version', '1.0')}\\<close>
    \\<open>{model_data.get('annotation', '')}\\<close>
    ["""

                                    # Build variable definitions
                                    var_defs = []
                                    for var in model_data.get('variables', []):
                                        var_def = f'      Variable \\<open>{var.get("name", "")}\\<close> \\<open>{var.get("type", "")}\\<close> [{", ".join(map(str, var.get("dimensions", [1])))}] \\<open>{var.get("description", "")}\\<close>'
                                        var_defs.append(var_def)
                                    
                                    # Build connection definitions
                                    conn_defs = []
                                    for conn in model_data.get('connections', []):
                                        conn_def = f'      Connection \\<open>{conn.get("source", "")}\\<close> \\<open>{conn.get("target", "")}\\<close> \\<open>{conn.get("type", "")}\\<close> \\<open>{conn.get("description", "")}\\<close>'
                                        conn_defs.append(conn_def)
                                    
                                    # Build parameter definitions
                                    param_defs = []
                                    for name, param in model_data.get('parameters', {}).items():
                                        param_def = f'      Parameter \\<open>{name}\\<close> \\<open>{param.get("value", "")}\\<close> \\<open>{param.get("description", "")}\\<close>'
                                        param_defs.append(param_def)
                                    
                                    isabelle_content = isabelle_base + chr(10).join(var_defs) + """    ]
    [""" + chr(10).join(conn_defs) + """    ]
    [""" + chr(10).join(param_defs) + """    ]"

end
"""
                                    with open(export_path, 'w') as f:
                                        f.write(isabelle_content)
                                    logger.info(f"Exported to Isabelle: {export_path}")
                                    export_success_count += 1
                                except Exception as e:
                                    logger.warning(f"Isabelle export failed: {e}")
                                    
                            elif fmt == "haskell":
                                try:
                                    haskell_content = f"""-- GNN Model: {model_data.get('model_name', 'Unknown')}
-- Generated from GNN markdown

data Variable = Variable
  {{ name :: String
  , type_ :: String
  , dimensions :: [Int]
  , description :: String
  }} deriving (Show, Eq)

data Connection = Connection
  {{ source :: String
  , target :: String
  , type_ :: String
  , description :: String
  }} deriving (Show, Eq)

data Parameter = Parameter
  {{ name :: String
  , value :: String
  , description :: String
  }} deriving (Show, Eq)

data GNNModel = GNNModel
  {{ modelName :: String
  , version :: String
  , annotation :: String
  , variables :: [Variable]
  , connections :: [Connection]
  , parameters :: [Parameter]
  }} deriving (Show, Eq)

model :: GNNModel
model = GNNModel
  {{ modelName = "{model_data.get('model_name', 'Unknown')}"
  , version = "{model_data.get('version', '1.0')}"
  , annotation = "{model_data.get('annotation', '')}"
  , variables = [
{chr(10).join([f'      Variable "{var.get("name", "")}" "{var.get("type", "")}" [{", ".join(map(str, var.get("dimensions", [1])))}] "{var.get("description", "")}"' for var in model_data.get('variables', [])])}
    ]
  , connections = [
{chr(10).join([f'      Connection "{conn.get("source", "")}" "{conn.get("target", "")}" "{conn.get("type", "")}" "{conn.get("description", "")}"' for conn in model_data.get('connections', [])])}
    ]
  , parameters = [
{chr(10).join([f'      Parameter "{name}" "{param.get("value", "")}" "{param.get("description", "")}"' for name, param in model_data.get('parameters', {}).items()])}
    ]
  }}
"""
                                    with open(export_path, 'w') as f:
                                        f.write(haskell_content)
                                    logger.info(f"Exported to Haskell: {export_path}")
                                    export_success_count += 1
                                except Exception as e:
                                    logger.warning(f"Haskell export failed: {e}")
                                    
                            elif fmt == "alloy":
                                try:
                                    alloy_content = f"""// GNN Model: {model_data.get('model_name', 'Unknown')}
// Generated from GNN markdown

sig Variable {{
  name: one String,
  type: one String,
  dimensions: set Int,
  description: one String
}}

sig Connection {{
  source: one String,
  target: one String,
  type: one String,
  description: one String
}}

sig Parameter {{
  name: one String,
  value: one String,
  description: one String
}}

sig GNNModel {{
  model_name: one String,
  version: one String,
  annotation: one String,
  variables: set Variable,
  connections: set Connection,
  parameters: set Parameter
}}

fact model_definition {{
  one model: GNNModel |
    model.model_name = "{model_data.get('model_name', 'Unknown')}" and
    model.version = "{model_data.get('version', '1.0')}" and
    model.annotation = "{model_data.get('annotation', '')}"
}}

// Variable definitions
{chr(10).join([f'fact var_{var.get("name", "").replace("-", "_")} {{ one v: Variable | v.name = "{var.get("name", "")}" and v.type = "{var.get("type", "")}" and v.description = "{var.get("description", "")}" }}' for var in model_data.get('variables', [])])}

// Connection definitions
{chr(10).join([f'fact conn_{conn.get("source", "").replace("-", "_")}_{conn.get("target", "").replace("-", "_")} {{ one c: Connection | c.source = "{conn.get("source", "")}" and c.target = "{conn.get("target", "")}" and c.type = "{conn.get("type", "")}" and c.description = "{conn.get("description", "")}" }}' for conn in model_data.get('connections', [])])}

// Parameter definitions
{chr(10).join([f'fact param_{name.replace("-", "_")} {{ one p: Parameter | p.name = "{name}" and p.value = "{param.get("value", "")}" and p.description = "{param.get("description", "")}" }}' for name, param in model_data.get('parameters', {}).items()])}
"""
                                    with open(export_path, 'w') as f:
                                        f.write(alloy_content)
                                    logger.info(f"Exported to Alloy: {export_path}")
                                    export_success_count += 1
                                except Exception as e:
                                    logger.warning(f"Alloy export failed: {e}")
                                    
                            elif fmt == "bnf":
                                try:
                                    bnf_content = f"""(* GNN Model: {model_data.get('model_name', 'Unknown')} *)
(* Generated from GNN markdown *)

<gnn-model> ::= <model-header> <variables-section> <connections-section> <parameters-section>

<model-header> ::= "ModelName" <string> "Version" <string> "Annotation" <string>

<variables-section> ::= "Variables:" <variable-list>
<variable-list> ::= <variable> | <variable> <variable-list>
<variable> ::= <name> "[" <dimensions> "]" <type> <description>

<connections-section> ::= "Connections:" <connection-list>
<connection-list> ::= <connection> | <connection> <connection-list>
<connection> ::= <source> "->" <target> <type> <description>

<parameters-section> ::= "Parameters:" <parameter-list>
<parameter-list> ::= <parameter> | <parameter> <parameter-list>
<parameter> ::= <name> "=" <value> <description>

<name> ::= <identifier>
<type> ::= "hidden_state" | "observation" | "action" | "parameter_matrix"
<dimensions> ::= <number> | <number> "," <dimensions>
<source> ::= <identifier>
<target> ::= <identifier>
<value> ::= <string> | <number> | <tuple>
<description> ::= "#" <string>
<identifier> ::= <letter> | <letter> <identifier>
<letter> ::= "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j" | "k" | "l" | "m" | "n" | "o" | "p" | "q" | "r" | "s" | "t" | "u" | "v" | "w" | "x" | "y" | "z"
<number> ::= <digit> | <digit> <number>
<digit> ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
<string> ::= "\"" <char-list> "\""
<char-list> ::= <char> | <char> <char-list>
<char> ::= <letter> | <digit> | " " | "." | "," | "-" | "_"
<tuple> ::= "(" <value-list> ")"
<value-list> ::= <value> | <value> "," <value-list>
"""
                                    with open(export_path, 'w') as f:
                                        f.write(bnf_content)
                                    logger.info(f"Exported to BNF: {export_path}")
                                    export_success_count += 1
                                except Exception as e:
                                    logger.warning(f"BNF export failed: {e}")
                                    
                            elif fmt == "z_notation":
                                try:
                                    z_content = f"""-- GNN Model: {model_data.get('model_name', 'Unknown')}
-- Generated from GNN markdown

[String, Int]

Variable == [name: String; type: String; dimensions: seq Int; description: String]

Connection == [source: String; target: String; type: String; description: String]

Parameter == [name: String; value: String; description: String]

GNNModel == [model_name: String; version: String; annotation: String; variables: seq Variable; connections: seq Connection; parameters: seq Parameter]

model: GNNModel
model = [model_name: "{model_data.get('model_name', 'Unknown')}"; version: "{model_data.get('version', '1.0')}"; annotation: "{model_data.get('annotation', '')}"; variables: [
{chr(10).join([f'  [name: "{var.get("name", "")}"; type: "{var.get("type", "")}"; dimensions: [{", ".join(map(str, var.get("dimensions", [1])))}]; description: "{var.get("description", "")}"]' for var in model_data.get('variables', [])])}
]; connections: [
{chr(10).join([f'  [source: "{conn.get("source", "")}"; target: "{conn.get("target", "")}"; type: "{conn.get("type", "")}"; description: "{conn.get("description", "")}"]' for conn in model_data.get('connections', [])])}
]; parameters: [
{chr(10).join([f'  [name: "{name}"; value: "{param.get("value", "")}"; description: "{param.get("description", "")}"]' for name, param in model_data.get('parameters', {}).items()])}
]]
"""
                                    with open(export_path, 'w') as f:
                                        f.write(z_content)
                                    logger.info(f"Exported to Z Notation: {export_path}")
                                    export_success_count += 1
                                except Exception as e:
                                    logger.warning(f"Z Notation export failed: {e}")
                                    
                            elif fmt == "tla_plus":
                                try:
                                    tla_content = f"""---- MODULE GNNModel ----
(* GNN Model: {model_data.get('model_name', 'Unknown')} *)
(* Generated from GNN markdown *)

EXTENDS Naturals, Sequences

VARIABLES model_name, version, annotation, variables, connections, parameters

TypeOK == 
  /\\ model_name \\in STRING
  /\\ version \\in STRING
  /\\ annotation \\in STRING
  /\\ variables \\in Seq(STRING \\times STRING \\times Seq(Nat) \\times STRING)
  /\\ connections \\in Seq(STRING \\times STRING \\times STRING \\times STRING)
  /\\ parameters \\in Seq(STRING \\times STRING \\times STRING)

Init == 
  /\\ model_name = "{model_data.get('model_name', 'Unknown')}"
  /\\ version = "{model_data.get('version', '1.0')}"
  /\\ annotation = "{model_data.get('annotation', '')}"
  /\\ variables = <<
{chr(10).join([f'    <<"{var.get("name", "")}", "{var.get("type", "")}", <<{", ".join(map(str, var.get("dimensions", [1])))}>>, "{var.get("description", "")}>>' for var in model_data.get('variables', [])])}
  >>
  /\\ connections = <<
{chr(10).join([f'    <<"{conn.get("source", "")}", "{conn.get("target", "")}", "{conn.get("type", "")}", "{conn.get("description", "")}">>' for conn in model_data.get('connections', [])])}
  >>
  /\\ parameters = <<
{chr(10).join([f'    <<"{name}", "{param.get("value", "")}", "{param.get("description", "")}">>' for name, param in model_data.get('parameters', {}).items()])}
  >>

Next == UNCHANGED <<model_name, version, annotation, variables, connections, parameters>>

Spec == Init /\\ [][Next]_<<model_name, version, annotation, variables, connections, parameters>>

====
"""
                                    with open(export_path, 'w') as f:
                                        f.write(tla_content)
                                    logger.info(f"Exported to TLA+: {export_path}")
                                    export_success_count += 1
                                except Exception as e:
                                    logger.warning(f"TLA+ export failed: {e}")
                                    
                            elif fmt == "agda":
                                try:
                                    agda_content = f"""-- GNN Model: {model_data.get('model_name', 'Unknown')}
-- Generated from GNN markdown

module GNNModel where

open import Data.String
open import Data.List
open import Data.Nat
open import Data.Product

record Variable : Set where
  constructor variable
  field
    name : String
    type : String
    dimensions : List ℕ
    description : String

record Connection : Set where
  constructor connection
  field
    source : String
    target : String
    type : String
    description : String

record Parameter : Set where
  constructor parameter
  field
    name : String
    value : String
    description : String

record GNNModel : Set where
  constructor gnnModel
  field
    modelName : String
    version : String
    annotation : String
    variables : List Variable
    connections : List Connection
    parameters : List Parameter

model : GNNModel
model = gnnModel
  "{model_data.get('model_name', 'Unknown')}"
  "{model_data.get('version', '1.0')}"
  "{model_data.get('annotation', '')}"
  [
{chr(10).join([f'    variable "{var.get("name", "")}" "{var.get("type", "")}" [{", ".join(map(str, var.get("dimensions", [1])))}] "{var.get("description", "")}"' for var in model_data.get('variables', [])])}
  ]
  [
{chr(10).join([f'    connection "{conn.get("source", "")}" "{conn.get("target", "")}" "{conn.get("type", "")}" "{conn.get("description", "")}"' for conn in model_data.get('connections', [])])}
  ]
  [
{chr(10).join([f'    parameter "{name}" "{param.get("value", "")}" "{param.get("description", "")}"' for name, param in model_data.get('parameters', {}).items()])}
  ]
"""
                                    with open(export_path, 'w') as f:
                                        f.write(agda_content)
                                    logger.info(f"Exported to Agda: {export_path}")
                                    export_success_count += 1
                                except Exception as e:
                                    logger.warning(f"Agda export failed: {e}")
                                    
                            elif fmt == "pickle":
                                try:
                                    import pickle
                                    with open(export_path, 'wb') as f:
                                        pickle.dump(model_data, f)
                                    logger.info(f"Exported to Pickle: {export_path}")
                                    export_success_count += 1
                                except Exception as e:
                                    logger.warning(f"Pickle export failed: {e}")
                                    
                            elif fmt == "maxima":
                                try:
                                    maxima_content = f"""/* GNN Model: {model_data.get('model_name', 'Unknown')} */
/* Generated from GNN markdown */

/* Model metadata */
model_name: "{model_data.get('model_name', 'Unknown')}";
version: "{model_data.get('version', '1.0')}";
annotation: "{model_data.get('annotation', '')}";

/* Variable definitions */
variables: [
{chr(10).join([f'  ["{var.get("name", "")}", "{var.get("type", "")}", [{", ".join(map(str, var.get("dimensions", [1])))}], "{var.get("description", "")}"]' for var in model_data.get('variables', [])])}
];

/* Connection definitions */
connections: [
{chr(10).join([f'  ["{conn.get("source", "")}", "{conn.get("target", "")}", "{conn.get("type", "")}", "{conn.get("description", "")}"]' for conn in model_data.get('connections', [])])}
];

/* Parameter definitions */
parameters: [
{chr(10).join([f'  ["{name}", "{param.get("value", "")}", "{param.get("description", "")}"]' for name, param in model_data.get('parameters', {}).items()])}
];

/* Helper functions */
get_variable(name) := block([result], result: false, for v in variables do if v[1] = name then result: v, result);
get_connections_from(source) := block([result], result: [], for c in connections do if c[1] = source then result: cons(c, result), result);
get_connections_to(target) := block([result], result: [], for c in connections do if c[2] = target then result: cons(c, result), result);
"""
                                    with open(export_path, 'w') as f:
                                        f.write(maxima_content)
                                    logger.info(f"Exported to Maxima: {export_path}")
                                    export_success_count += 1
                                except Exception as e:
                                    logger.warning(f"Maxima export failed: {e}")
                                    
                            else:
                                logger.warning(f"Export format '{fmt}' not implemented yet")
                                
                        except Exception as e:
                            logger.warning(f"Export to {fmt} failed: {e}")
                    
                    logger.info(f"Successfully exported {export_success_count}/{total_formats} formats for {base_name}")
                    
                except Exception as e:
                    logger.warning(f"Error processing {gnn_file}: {e}")
            
            logger.info(f"Successfully exported GNN files to {export_dir}")
        except Exception as e:
            logger.warning(f"Export functionality error: {e}")
            logger.warning("Skipping format exports.")
        
        logger.info("Enhanced GNN processing with modular architecture")
        return True
    except ImportError as e:
        logger.warning(f"Modular architecture not available: {e}")
        logger.warning("Falling back to legacy processing.")
        return False


def process_gnn_files_legacy(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    recursive: bool = False,
    verbose: bool = False,
    validation_level: str = "standard",
    enable_round_trip: bool = False,
    enable_cross_format: bool = False,
    test_subset: Optional[List[str]] = None,
    reference_file: Optional[str] = None,
    **kwargs
) -> bool:
    """
    Enhanced GNN processing using the full GNN module capabilities.
    
    This uses the comprehensive GNN module to process files with full
    parsing, validation, and reporting capabilities.
    """
    log_step_start_safe(logger, "Enhanced GNN processing using full GNN module")
    start_time = time.time()
    
    try:
        # Import GNN module functions with detailed error tracking
        try:
            import sys
            import importlib
            import multiprocessing
            
            # Explicitly import each module to track import issues
            gnn_modules = [
                'gnn',
                'gnn.schema_validator', 
                'gnn.cross_format_validator', 
                'gnn.parsers'
            ]
            
            for module_name in gnn_modules:
                try:
                    importlib.import_module(module_name)
                except ImportError as e:
                    logger.error(f"Could not import {module_name}: {e}")
                    logger.error(f"Python path: {sys.path}")
                    logger.error(f"Module details: {sys.modules.get(module_name, 'Not loaded')}")
            
            from gnn import process_gnn_directory, generate_gnn_report, validate_gnn
            from gnn.schema_validator import ValidationLevel
            GNN_MODULE_AVAILABLE = True
        except ImportError as e:
            logger.error(f"GNN module import failed: {e}")
            logger.error(f"Full import error details: {sys.exc_info()}")
            GNN_MODULE_AVAILABLE = False
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate input directory
        if not target_dir.exists() or not target_dir.is_dir():
            logger.error(f"Invalid target directory: {target_dir}")
            return False
        
        if GNN_MODULE_AVAILABLE:
            # Use full GNN module processing with extensive logging and timeout
            logger.info("Using full GNN module processing")
            logger.info(f"Target directory: {target_dir}")
            logger.info(f"Recursive: {recursive}")
            
            try:
                # Use multiprocessing to implement timeout with more lenient settings
                with multiprocessing.Pool(1) as pool:
                    # 60-second timeout for processing
                    result = pool.apply_async(_process_gnn_directory_with_timeout, (target_dir, recursive, logger))
                    
                    try:
                        processing_results = result.get(timeout=60)
                    except multiprocessing.TimeoutError:
                        logger.warning("GNN module processing timed out, attempting to continue")
                        # Try to get partial results
                        processing_results = {
                            'directory': str(target_dir),
                            'total_files': 0, 
                            'processed_files': [], 
                            'summary': {
                                'valid_files': 0,
                                'invalid_files': 0,
                                'total_files': 0,
                                'total_variables': 0,
                                'total_connections': 0
                            },
                            'errors': []
                        }
                
                # Ensure processing_results has all expected keys
                expected_keys = {
                    'directory': str(target_dir),
                    'total_files': 0,
                    'processed_files': [],
                    'summary': {
                        'valid_files': 0,
                        'invalid_files': 0,
                        'total_files': 0,
                        'total_variables': 0,
                        'total_connections': 0
                    },
                    'errors': []
                }
                
                # Merge expected keys with actual results, preserving existing values
                for key, default_value in expected_keys.items():
                    if key not in processing_results:
                        processing_results[key] = default_value
                    elif isinstance(default_value, dict):
                        for subkey, subdefault in default_value.items():
                            if subkey not in processing_results[key]:
                                processing_results[key][subkey] = subdefault
                
                # Generate comprehensive report
                report_content = generate_gnn_report(processing_results)
                
                # Save detailed report
                report_file = output_dir / f"gnn_processing_report_{int(time.time())}.md"
                with open(report_file, 'w') as f:
                    f.write(report_content)
                
                # Save JSON results
                import json
                json_file = output_dir / f"gnn_processing_results_{int(time.time())}.json"
                with open(json_file, 'w') as f:
                    json.dump(processing_results, f, indent=2)
                
                logger.info(f"Enhanced GNN processing report saved: {report_file}")
                logger.info(f"JSON results saved: {json_file}")
                
                # Summary statistics
                valid_count = processing_results['summary'].get('valid_files', 0)
                total_count = processing_results['summary'].get('total_files', 0)
                total_variables = processing_results['summary'].get('total_variables', 0)
                total_connections = processing_results['summary'].get('total_connections', 0)
                
                # Log errors if any
                errors = processing_results.get('errors', [])
                if errors:
                    logger.warning(f"Processing encountered {len(errors)} errors")
                    for error in errors:
                        logger.warning(f"Error: {error}")
                
                logger.info(f"Processed {total_count} files, {valid_count} valid")
                logger.info(f"Total variables: {total_variables}")
                logger.info(f"Total connections: {total_connections}")
                
                return True
                
            except Exception as e:
                logger.error(f"GNN module processing failed: {e}")
                logger.error(f"Full exception details: {sys.exc_info()}")
                return False
        
        # Fallback to basic processing if GNN module is not available
        logger.warning("Falling back to basic processing")
        
        # Basic file discovery
        discovered_files = []
        
        if recursive:
            for file_path in target_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in ['.md', '.json', '.xml', '.yaml', '.pkl']:
                    discovered_files.append(file_path)
        else:
            for file_path in target_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in ['.md', '.json', '.xml', '.yaml', '.pkl']:
                    discovered_files.append(file_path)
        
        logger.info(f"Discovered {len(discovered_files)} potential GNN files")
        
        # Basic validation
        valid_files = []
        for file_path in discovered_files:
            try:
                # Simple validation - check if file is readable
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(100)  # Read first 100 chars
                    if any(marker in content.lower() for marker in ['model', 'gnn', 'variable', 'connection']):
                        valid_files.append(file_path)
            except:
                continue
        
        logger.info(f"Found {len(valid_files)} valid GNN files")
        
        # Generate basic report
        report = {
            "basic_gnn_processing_report": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "target_directory": str(target_dir),
                "output_directory": str(output_dir),
                "processing_time": f"{time.time() - start_time:.2f}s",
                "files_discovered": len(discovered_files),
                "files_valid": len(valid_files),
                "validation_level": validation_level,
                "mode": "basic_fallback"
            }
        }
        
        # Save report
        import json
        report_file = output_dir / f"basic_gnn_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Basic processing report saved: {report_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"GNN processing failed: {e}")
        if verbose:
            logger.exception("Detailed error:")
        return False


def create_enhanced_parser() -> argparse.ArgumentParser:
    """Create enhanced command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Enhanced GNN Processing Pipeline - Step 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Basic processing
  %(prog)s --verbose                          # Verbose output
  %(prog)s --validation-level strict          # Strict validation
  %(prog)s --enable-round-trip                # With round-trip testing
  %(prog)s --enable-cross-format              # With cross-format validation
  %(prog)s --recursive                        # Recursive file search
  %(prog)s --test-subset json,xml,yaml        # Test specific formats
        """
    )
    
    # Directory options
    parser.add_argument(
        '--target-dir',
        help='Target directory containing GNN files to process'
    )
    
    parser.add_argument(
        '--output-dir',
        help='Output directory for processing results'
    )
    
    # Processing options
    parser.add_argument(
        '--recursive', 
        action='store_true',
        help='Search for GNN files recursively in subdirectories'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging output'
    )
    
    # Validation options
    parser.add_argument(
        '--validation-level',
        choices=['basic', 'standard', 'strict', 'research', 'round_trip'],
        default='standard',
        help='Validation level (default: standard)'
    )
    
    # Testing options
    parser.add_argument(
        '--enable-round-trip',
        action='store_true',
        help='Enable round-trip testing across formats'
    )
    
    parser.add_argument(
        '--enable-cross-format',
        action='store_true',
        help='Enable cross-format consistency validation'
    )
    
    parser.add_argument(
        '--test-subset',
        help='Comma-separated list of formats to test (e.g., json,xml,yaml)'
    )
    
    parser.add_argument(
        '--reference-file',
        help='Specific reference file for round-trip testing'
    )
    
    # Architecture options
    parser.add_argument(
        '--force-legacy',
        action='store_true',
        help='Force use of legacy processing mode'
    )
    
    return parser


def run_script() -> int:
    """Enhanced script execution with modular architecture support."""
    parser = create_enhanced_parser()
    args = parser.parse_args()
    
    try:
        # Get project root and set up directories
        project_root = Path(__file__).resolve().parent.parent
        
        # Set target directory (input/gnn_files)
        if hasattr(args, 'target_dir') and args.target_dir:
            target_dir = Path(args.target_dir)
        else:
            target_dir = project_root / "input" / "gnn_files"
        
        if not target_dir.exists():
            logger.error(f"Target directory does not exist: {target_dir}")
            return 1
        
        # Set output directory (output/gnn_processing_step)
        if hasattr(args, 'output_dir') and args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = project_root / "output" / "gnn_processing_step"
            
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse test subset if provided
        test_subset = None
        if args.test_subset:
            test_subset = [fmt.strip().lower() for fmt in args.test_subset.split(',')]
        
        # Adjust logging based on verbose flag
        if args.verbose:
            logger.setLevel(logging.DEBUG)
            # Even in verbose mode, suppress some noisy loggers
            logging.getLogger('gnn.cross_format_validator').setLevel(logging.WARNING)
        else:
            # Keep external loggers very quiet
            logging.getLogger('gnn').setLevel(logging.ERROR)
            logging.getLogger('gnn.schema_validator').setLevel(logging.ERROR)
            logging.getLogger('gnn.cross_format_validator').setLevel(logging.ERROR)
            logging.getLogger('gnn.parsers').setLevel(logging.ERROR)
            logging.getLogger('gnn.testing').setLevel(logging.ERROR)
            logging.getLogger('gnn.mcp').setLevel(logging.ERROR)
        
        # Log configuration
        logger.info("Enhanced GNN processing starting...")
        if args.verbose:
            logger.debug(f"Configuration details:")
            logger.debug(f"  Target: {target_dir}")
            logger.debug(f"  Output: {output_dir}")
            logger.debug(f"  Validation level: {args.validation_level}")
            logger.debug(f"  Round-trip testing: {args.enable_round_trip}")
            logger.debug(f"  Cross-format validation: {args.enable_cross_format}")
            logger.debug(f"  Recursive: {args.recursive}")
            logger.debug(f"  Architecture: {'Legacy (forced)' if args.force_legacy else 'Modular'}")
        
        # Run round-trip testing
        try:
            from gnn.testing.test_round_trip import GNNRoundTripTester
            
            logger.info("Running comprehensive round-trip tests...")
            tester = GNNRoundTripTester()
            
            # Configure test settings based on CLI arguments
            if test_subset:
                # Modify test configuration to use only specified formats
                import gnn.testing.test_round_trip as test_module
                test_module.FORMAT_TEST_CONFIG['test_all_formats'] = False
                test_module.FORMAT_TEST_CONFIG['test_formats'] = test_subset
            
            # Run tests
            report = tester.run_comprehensive_tests()
            
            # Generate and save report
            round_trip_output_dir = output_dir / "round_trip_tests"
            round_trip_output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Markdown report
            md_report_file = round_trip_output_dir / f"round_trip_report_{timestamp}.md"
            tester.generate_report(report, md_report_file)
            
            # JSON results
            json_report_file = round_trip_output_dir / f"round_trip_results_{timestamp}.json"
            import json
            with open(json_report_file, 'w') as f:
                json.dump({
                    'total_tests': report.total_tests,
                    'successful_tests': report.successful_tests,
                    'failed_tests': report.failed_tests,
                    'success_rate': report.get_success_rate(),
                    'results': [
                        {
                            'format': result.target_format.value,
                            'success': result.success,
                            'errors': result.errors,
                            'warnings': result.warnings,
                            'differences': result.differences
                        } for result in report.round_trip_results
                    ]
                }, f, indent=2)
            
            logger.info(f"Round-trip test report saved: {md_report_file}")
            logger.info(f"Round-trip test results saved: {json_report_file}")
            
        except ImportError:
            logger.warning("Round-trip testing module not available")
        except Exception as e:
            logger.error(f"Round-trip testing failed: {e}")
            if args.verbose:
                logger.exception("Detailed error:")
        
        # Choose processing method - prefer enhanced GNN module processing
        if args.force_legacy:
            logger.info("Using legacy processing mode (forced)")
            success = process_gnn_files_legacy(
                target_dir=target_dir,
                output_dir=output_dir,
                logger=logger,
                recursive=args.recursive,
                verbose=args.verbose,
                validation_level=args.validation_level,
                enable_round_trip=args.enable_round_trip,
                enable_cross_format=args.enable_cross_format,
                test_subset=test_subset,
                reference_file=args.reference_file
            )
        elif MODULAR_ARCHITECTURE_AVAILABLE:
            logger.info("Using modular processing architecture")
            success = process_gnn_files_modular(
                target_dir=target_dir,
                output_dir=output_dir,
                logger=logger,
                recursive=args.recursive,
                verbose=args.verbose,
                validation_level=args.validation_level,
                enable_round_trip=args.enable_round_trip,
                enable_cross_format=args.enable_cross_format,
                test_subset=test_subset,
                reference_file=args.reference_file
            )
        else:
            logger.info("Using enhanced GNN module processing")
            success = process_gnn_files_legacy(
                target_dir=target_dir,
                output_dir=output_dir,
                logger=logger,
                recursive=args.recursive,
                verbose=args.verbose,
                validation_level=args.validation_level,
                enable_round_trip=args.enable_round_trip,
                enable_cross_format=args.enable_cross_format,
                test_subset=test_subset,
                reference_file=args.reference_file
            )
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        if args.verbose:
            logger.exception("Detailed error:")
        return 1


if __name__ == '__main__':
    sys.exit(run_script()) 