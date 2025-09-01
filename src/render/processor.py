#!/usr/bin/env python3
"""
Enhanced Render processor module for GNN code generation with POMDP-aware processing.

This module provides comprehensive rendering capabilities that:
1. Extract POMDP state spaces from GNN specifications
2. Modularly inject them into framework-specific renderers
3. Create implementation-specific output subfolders
4. Provide structured documentation and results
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
from datetime import datetime

logger = logging.getLogger(__name__)

def process_render(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Process render for GNN specifications with POMDP-aware processing.
    
    This enhanced processor:
    1. Extracts POMDP state spaces from GNN files
    2. Modularly injects them into framework-specific renderers  
    3. Creates implementation-specific output subfolders
    4. Provides structured documentation and results
    
    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Output directory for rendered files
        verbose: Enable verbose logging
        **kwargs: Additional processing options including:
            - frameworks: List of frameworks to render for (default: all)
            - strict_validation: Enable strict POMDP validation
            - include_documentation: Generate framework documentation
        
    Returns:
        True if processing succeeded, False otherwise
    """
    try:
        logger.info(f"🚀 Starting POMDP-aware render processing")
        logger.info(f"Processing GNN files in: {target_dir}")
        logger.info(f"Output directory: {output_dir}")
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import POMDP processing capabilities
        try:
            from ..gnn.pomdp_extractor import extract_pomdp_from_file
            from .pomdp_processor import POMDPRenderProcessor
            pomdp_available = True
        except ImportError as e:
            logger.warning(f"POMDP processing modules not available: {e}")
            logger.info("Falling back to basic rendering")
            pomdp_available = False
        
        # Find GNN files
        gnn_files = []
        for pattern in ['*.md', '*.json', '*.yaml', '*.yml']:
            gnn_files.extend(target_dir.glob(pattern))
        
        if not gnn_files:
            logger.warning(f"No GNN files found in {target_dir}")
            return True
        
        logger.info(f"Found {len(gnn_files)} GNN files to process")
        
        # Processing configuration
        frameworks = kwargs.get('frameworks', None)  # None means all frameworks
        strict_validation = kwargs.get('strict_validation', True)
        
        if frameworks:
            logger.info(f"Target frameworks: {frameworks}")
        else:
            logger.info("Target frameworks: all available")
        
        results = {}
        success_count = 0
        total_framework_successes = 0
        total_framework_attempts = 0
        
        if pomdp_available:
            # Use POMDP-aware processing
            pomdp_processor = POMDPRenderProcessor(output_dir)
            
            for gnn_file in gnn_files:
                try:
                    logger.info(f"📁 Processing: {gnn_file}")
                    
                    # Extract POMDP state space from GNN file
                    pomdp_space = extract_pomdp_from_file(gnn_file, strict_validation=strict_validation)
                    
                    if pomdp_space is None:
                        logger.warning(f"Could not extract POMDP from {gnn_file}, trying basic rendering")
                        # Fall back to basic processing for this file
                        file_result = _process_single_gnn_file_basic(gnn_file, output_dir, verbose, **kwargs)
                        results[str(gnn_file)] = file_result
                        if file_result['success']:
                            success_count += 1
                        continue
                    
                    logger.info(f"✅ Extracted POMDP '{pomdp_space.model_name}' with {pomdp_space.num_states} states, {pomdp_space.num_observations} observations, {pomdp_space.num_actions} actions")
                    
                    # Create file-specific output directory
                    file_output_dir = output_dir / gnn_file.stem
                    
                    # Create processor with file-specific directory
                    file_processor = POMDPRenderProcessor(file_output_dir)
                    
                    # Process POMDP for all frameworks
                    processing_result = file_processor.process_pomdp_for_all_frameworks(
                        pomdp_space, gnn_file_path=gnn_file, frameworks=frameworks, **kwargs
                    )
                    processing_result['base_output_dir'] = str(file_output_dir)
                    
                    results[str(gnn_file)] = processing_result
                    
                    if processing_result['overall_success']:
                        success_count += 1
                        logger.info(f"✅ Successfully processed {gnn_file.name}")
                    else:
                        logger.error(f"❌ Failed to process {gnn_file.name}")
                    
                    # Count framework-level successes
                    for framework, result in processing_result['framework_results'].items():
                        total_framework_attempts += 1
                        if result['success']:
                            total_framework_successes += 1
                            
                except Exception as e:
                    error_msg = f"Error processing {gnn_file}: {e}"
                    logger.error(error_msg)
                    results[str(gnn_file)] = {
                        'overall_success': False,
                        'framework_results': {},
                        'error': error_msg
                    }
        else:
            # Use basic rendering
            for gnn_file in gnn_files:
                try:
                    logger.info(f"📁 Processing (basic): {gnn_file}")
                    file_result = _process_single_gnn_file_basic(gnn_file, output_dir, verbose, **kwargs)
                    results[str(gnn_file)] = file_result
                    if file_result['success']:
                        success_count += 1
                        logger.info(f"✅ Processed {gnn_file.name}")
                    else:
                        logger.error(f"❌ Failed to process {gnn_file.name}")
                        
                except Exception as e:
                    error_msg = f"Error processing {gnn_file}: {e}"
                    logger.error(error_msg)
                    results[str(gnn_file)] = {
                        'success': False,
                        'error': error_msg
                    }
        
        # Create overall processing summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'processing_type': 'POMDP-aware rendering' if pomdp_available else 'Basic rendering',
            'total_files': len(gnn_files),
            'successful_files': success_count,
            'failed_files': len(gnn_files) - success_count,
            'total_framework_attempts': total_framework_attempts,
            'successful_framework_renderings': total_framework_successes,
            'framework_success_rate': (total_framework_successes / total_framework_attempts * 100) if total_framework_attempts > 0 else 0,
            'configuration': {
                'frameworks': frameworks or 'all',
                'strict_validation': strict_validation,
                'verbose': verbose,
                'pomdp_processing_available': pomdp_available
            },
            'file_results': results
        }
        
        summary_file = output_dir / 'render_processing_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create overview documentation
        _create_overview_documentation(output_dir, summary)
        
        logger.info(f"🎉 Render processing completed!")
        logger.info(f"📊 Files: {success_count}/{len(gnn_files)} successful")
        if pomdp_available:
            logger.info(f"🧠 Framework renderings: {total_framework_successes}/{total_framework_attempts} successful ({summary['framework_success_rate']:.1f}%)")
        logger.info(f"📄 Summary saved to: {summary_file}")
        
        return success_count == len(gnn_files)
        
    except Exception as e:
        logger.error(f"Render processing failed: {e}")
        return False

def _process_single_gnn_file_basic(gnn_file: Path, output_dir: Path, verbose: bool, **kwargs) -> Dict[str, Any]:
    """
    Basic processing for a single GNN file without POMDP extraction.
    
    Args:
        gnn_file: GNN file to process
        output_dir: Output directory
        verbose: Enable verbose logging
        **kwargs: Additional processing options
        
    Returns:
        Processing result dictionary
    """
    try:
        # Import basic generators
        from .generators import (
            generate_pymdp_code, generate_rxinfer_code,
            generate_activeinference_jl_code, generate_discopy_code
        )
        
        # Create basic model data from filename
        model_data = {
            'model_name': gnn_file.stem,
            'variables': [],
            'connections': []
        }
        
        # Create file-specific output directory
        file_output_dir = output_dir / gnn_file.stem
        file_output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        # Generate code for each framework
        frameworks = {
            'pymdp': (generate_pymdp_code, '.py'),
            'rxinfer': (generate_rxinfer_code, '.jl'),
            'activeinference_jl': (generate_activeinference_jl_code, '.jl'),
            'discopy': (generate_discopy_code, '.py')
        }
        
        for framework_name, (generator_func, extension) in frameworks.items():
            try:
                # Create framework subdirectory
                framework_dir = file_output_dir / framework_name
                framework_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate code
                code = generator_func(model_data)
                if code:
                    output_file = framework_dir / f"{gnn_file.stem}_{framework_name}{extension}"
                    with open(output_file, 'w') as f:
                        f.write(code)
                    generated_files.append(str(output_file))
                    
            except Exception as e:
                logger.warning(f"Failed to generate {framework_name} code for {gnn_file}: {e}")
        
        return {
            'success': len(generated_files) > 0,
            'message': f"Generated {len(generated_files)} files" if generated_files else "No files generated",
            'generated_files': generated_files,
            'output_directory': str(file_output_dir)
        }
        
    except Exception as e:
        return {
            'success': False,
            'message': f"Basic processing failed: {e}",
            'generated_files': []
        }

def _create_overview_documentation(output_dir: Path, summary: Dict[str, Any]) -> None:
    """
    Create overview documentation for the rendering results.
    
    Args:
        output_dir: Output directory
        summary: Processing summary data
    """
    try:
        doc_content = f"""# GNN Rendering Results

Generated: {summary['timestamp']}
Processing Type: **{summary['processing_type']}**

## Summary

- **Total Files**: {summary['total_files']}
- **Successfully Processed**: {summary['successful_files']}
- **Failed**: {summary['failed_files']}
"""

        if summary['total_framework_attempts'] > 0:
            doc_content += f"""- **Framework Renderings**: {summary['successful_framework_renderings']}/{summary['total_framework_attempts']} ({summary['framework_success_rate']:.1f}% success rate)
"""

        doc_content += f"""
## Configuration

- **Frameworks**: {summary['configuration']['frameworks']}
- **Strict Validation**: {summary['configuration']['strict_validation']}
- **Verbose**: {summary['configuration']['verbose']}
- **POMDP Processing**: {'✅ Available' if summary['configuration'].get('pomdp_processing_available', False) else '❌ Not Available'}

## File Results

"""

        for file_path, result in summary['file_results'].items():
            file_name = Path(file_path).name
            if result.get('overall_success', result.get('success', False)):
                doc_content += f"- ✅ **{file_name}** - Successfully processed\n"
                
                # Add framework details if available
                if 'framework_results' in result:
                    for framework, framework_result in result['framework_results'].items():
                        status = "✅" if framework_result['success'] else "❌"
                        doc_content += f"  - {status} {framework}: {framework_result.get('message', 'N/A')}\n"
            else:
                error_msg = result.get('error', result.get('message', 'Unknown error'))
                doc_content += f"- ❌ **{file_name}** - {error_msg}\n"

        doc_content += f"""

## Output Structure

The rendered files are organized in implementation-specific subfolders:

```
{output_dir}/
├── [model_name]/
│   ├── pymdp/              # PyMDP Python simulations
│   ├── rxinfer/            # RxInfer.jl Julia simulations
│   ├── activeinference_jl/ # ActiveInference.jl Julia simulations
│   ├── jax/                # JAX Python simulations
│   └── discopy/            # DisCoPy categorical diagrams
└── render_processing_summary.json  # Detailed results
```

## Generated Files

Each framework subdirectory contains:
- Main simulation/diagram script
- Framework-specific README.md with model details
- Configuration files (if applicable)

## Next Steps

1. Navigate to specific framework directories to find generated code
2. Follow framework-specific READMEs for execution instructions  
3. Check the processing summary JSON for detailed results and any warnings

---

*Generated by GNN Render Processor v1.0*
"""

        doc_file = output_dir / 'README.md'
        with open(doc_file, 'w') as f:
            f.write(doc_content)
        
        logger.info(f"Created overview documentation: {doc_file}")
        
    except Exception as e:
        logger.warning(f"Failed to create overview documentation: {e}")

def render_gnn_spec(
    gnn_spec: Dict[str, Any],
    target: str,
    output_directory: Union[str, Path],
    options: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, List[str]]:
    """
    Render a GNN specification to a target language with POMDP awareness.
    
    Args:
        gnn_spec: GNN specification dictionary
        target: Target language/environment
        output_directory: Output directory for generated code
        options: Additional options
        
    Returns:
        Tuple of (success, message, warnings)
    """
    try:
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to use POMDP-aware processing first
        try:
            from ..gnn.pomdp_extractor import POMDPExtractor
            from .pomdp_processor import POMDPRenderProcessor
            
            # Create a dummy POMDP space from GNN spec
            # This is a simplified conversion - real usage should use proper extraction
            pomdp_extractor = POMDPExtractor()
            
            # For now, create basic model data and use generators
            if target.lower() == "pymdp":
                from .generators import generate_pymdp_code
                code = generate_pymdp_code(gnn_spec)
                output_file = output_dir / f"{gnn_spec.get('model_name', 'model')}_pymdp.py"
            elif target.lower() == "rxinfer":
                from .generators import generate_rxinfer_code
                code = generate_rxinfer_code(gnn_spec)
                output_file = output_dir / f"{gnn_spec.get('model_name', 'model')}_rxinfer.jl"
            elif target.lower() == "activeinference_jl":
                from .generators import generate_activeinference_jl_code
                code = generate_activeinference_jl_code(gnn_spec)
                output_file = output_dir / f"{gnn_spec.get('model_name', 'model')}_activeinference.jl"
            elif target.lower() == "discopy":
                from .generators import generate_discopy_code
                code = generate_discopy_code(gnn_spec)
                output_file = output_dir / f"{gnn_spec.get('model_name', 'model')}_discopy.py"
            else:
                return False, f"Unsupported target: {target}", []
            
            if code:
                with open(output_file, 'w') as f:
                    f.write(code)
                return True, f"Successfully generated {target} code", []
            else:
                return False, f"Failed to generate {target} code", []
                
        except ImportError:
            # Fall back to basic rendering
            return _render_gnn_spec_basic(gnn_spec, target, output_directory, options)
            
    except Exception as e:
        return False, f"Error rendering {target}: {e}", []

def _render_gnn_spec_basic(
    gnn_spec: Dict[str, Any],
    target: str,
    output_directory: Union[str, Path],
    options: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, List[str]]:
    """Basic GNN spec rendering without POMDP awareness."""
    try:
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        from .generators import (
            generate_pymdp_code, generate_rxinfer_code,
            generate_activeinference_jl_code, generate_discopy_code
        )
        
        if target.lower() == "pymdp":
            code = generate_pymdp_code(gnn_spec)
            output_file = output_dir / f"{gnn_spec.get('model_name', 'model')}_pymdp.py"
        elif target.lower() == "rxinfer":
            code = generate_rxinfer_code(gnn_spec)
            output_file = output_dir / f"{gnn_spec.get('model_name', 'model')}_rxinfer.jl"
        elif target.lower() == "activeinference_jl":
            code = generate_activeinference_jl_code(gnn_spec)
            output_file = output_dir / f"{gnn_spec.get('model_name', 'model')}_activeinference.jl"
        elif target.lower() == "discopy":
            code = generate_discopy_code(gnn_spec)
            output_file = output_dir / f"{gnn_spec.get('model_name', 'model')}_discopy.py"
        else:
            return False, f"Unsupported target: {target}", []
        
        if code:
            with open(output_file, 'w') as f:
                f.write(code)
            return True, f"Successfully generated {target} code", []
        else:
            return False, f"Failed to generate {target} code", []
            
    except Exception as e:
        return False, f"Error rendering {target}: {e}", []

def get_module_info() -> Dict[str, Any]:
    """Get information about the enhanced render module."""
    return {
        "name": "Enhanced Render Module",
        "version": "2.0.0",
        "description": "POMDP-aware code generation for GNN specifications",
        "supported_targets": ["pymdp", "rxinfer", "activeinference_jl", "jax", "discopy"],
        "available_targets": ["pymdp", "rxinfer", "activeinference_jl", "discopy"],
        "features": [
            "POMDP state space extraction",
            "Modular framework injection",
            "Implementation-specific output directories", 
            "PyMDP code generation",
            "RxInfer.jl code generation", 
            "ActiveInference.jl code generation",
            "JAX code generation",
            "DisCoPy categorical diagram generation",
            "Structured documentation generation"
        ],
        "supported_formats": ["python", "julia", "python_script"],
        "processing_modes": ["basic", "pomdp_aware"]
    }

def get_available_renderers() -> Dict[str, Dict[str, Any]]:
    """Get information about available renderers."""
    return {
        "pymdp": {
            "name": "PyMDP",
            "description": "Python Markov Decision Process library for Active Inference",
            "language": "Python",
            "file_extension": ".py",
            "supported_features": ["POMDP", "MDP", "Belief State Updates", "Active Inference"],
            "function": "render_gnn_to_pymdp",
            "output_format": "python",
            "pomdp_compatible": True
        },
        "rxinfer": {
            "name": "RxInfer.jl",
            "description": "Julia reactive message passing inference engine",
            "language": "Julia",
            "file_extension": ".jl",
            "supported_features": ["Message Passing", "Probabilistic Programming", "Bayesian Inference"],
            "function": "render_gnn_to_rxinfer",
            "output_format": "julia",
            "pomdp_compatible": True
        },
        "activeinference_jl": {
            "name": "ActiveInference.jl",
            "description": "Julia Active Inference library",
            "language": "Julia", 
            "file_extension": ".jl",
            "supported_features": ["Free Energy Minimization", "Active Inference", "POMDP"],
            "function": "render_gnn_to_activeinference_jl",
            "output_format": "julia",
            "pomdp_compatible": True
        },
        "jax": {
            "name": "JAX",
            "description": "High-performance numerical computing with automatic differentiation",
            "language": "Python",
            "file_extension": ".py", 
            "supported_features": ["GPU Acceleration", "Automatic Differentiation", "JIT Compilation"],
            "function": "render_gnn_to_jax",
            "output_format": "python",
            "pomdp_compatible": True
        },
        "discopy": {
            "name": "DisCoPy",
            "description": "Python library for computing with string diagrams",
            "language": "Python",
            "file_extension": ".py",
            "supported_features": ["Categorical Diagrams", "String Diagrams", "Compositional Models"],
            "function": "render_gnn_to_discopy",
            "output_format": "python",
            "pomdp_compatible": True
        }
    }