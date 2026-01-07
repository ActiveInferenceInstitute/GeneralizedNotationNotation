#!/usr/bin/env python3
"""
POMDP Processor for Render Module

This module provides specialized processing capabilities for injecting POMDP state spaces
into various rendering implementations (PyMDP, RxInfer, ActiveInference.jl, etc.).
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from ..gnn.pomdp_extractor import POMDPStateSpace

logger = logging.getLogger(__name__)

class POMDPRenderProcessor:
    """
    Processes POMDP state spaces and injects them into framework-specific renderers.
    
    Features:
    - Modular injection of POMDP state spaces into renderers
    - Framework-specific output directory management
    - Structured approach to render coordination
    - Validation of POMDP-renderer compatibility
    """
    
    def __init__(self, base_output_dir: Path):
        """
        Initialize POMDP render processor.
        
        Args:
            base_output_dir: Base output directory for all renderers
        """
        self.base_output_dir = Path(base_output_dir)
        self.logger = logging.getLogger(__name__)
        
        # Framework-specific configurations
        self.framework_configs = {
            'pymdp': {
                'output_subdir': 'pymdp_gen',
                'file_extension': '.py',
                'requires_matrices': ['A', 'B', 'C', 'D'],
                'optional_matrices': ['E'],
                'supports_multi_modality': True,
                'supports_multi_factor': True
            },
            'rxinfer': {
                'output_subdir': 'rxinfer',
                'file_extension': '.jl',
                'requires_matrices': ['A', 'B', 'C', 'D'],
                'optional_matrices': ['E'],
                'supports_multi_modality': False,
                'supports_multi_factor': False
            },
            'activeinference_jl': {
                'output_subdir': 'activeinference_jl',
                'file_extension': '.jl',
                'requires_matrices': ['A', 'B', 'C', 'D'],
                'optional_matrices': ['E'],
                'supports_multi_modality': True,
                'supports_multi_factor': True
            },
            'jax': {
                'output_subdir': 'jax',
                'file_extension': '.py',
                'requires_matrices': ['A', 'B', 'C', 'D'],
                'optional_matrices': ['E'],
                'supports_multi_modality': True,
                'supports_multi_factor': True
            },
            'discopy': {
                'output_subdir': 'discopy',
                'file_extension': '.py',
                'requires_matrices': [],
                'optional_matrices': ['A', 'B', 'C', 'D', 'E'],
                'supports_multi_modality': True,
                'supports_multi_factor': True
            }
        }
    
    def process_pomdp_for_all_frameworks(self, 
                                       pomdp_space: 'POMDPStateSpace',
                                       gnn_file_path: Optional[Path] = None,
                                       frameworks: Optional[List[str]] = None,
                                       **kwargs) -> Dict[str, Any]:
        """
        Process POMDP state space for all or specified frameworks.
        
        Args:
            pomdp_space: Extracted POMDP state space
            gnn_file_path: Original GNN file path (for reference)
            frameworks: List of frameworks to render for (default: all)
            **kwargs: Additional processing options
            
        Returns:
            Dictionary with processing results for each framework
        """
        if frameworks is None:
            frameworks = list(self.framework_configs.keys())
        
        results = {}
        overall_success = True
        
        # Create base output directory
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create processing summary
        processing_summary = {
            'timestamp': datetime.now().isoformat(),
            'source_file': str(gnn_file_path) if gnn_file_path else None,
            'model_name': pomdp_space.model_name,
            'pomdp_dimensions': {
                'num_states': pomdp_space.num_states,
                'num_observations': pomdp_space.num_observations,
                'num_actions': pomdp_space.num_actions
            },
            'frameworks_requested': frameworks,
            'frameworks_processed': [],
            'frameworks_failed': []
        }
        
        self.logger.info(f"Processing POMDP '{pomdp_space.model_name}' for frameworks: {frameworks}")
        
        for framework in frameworks:
            try:
                self.logger.info(f"Processing framework: {framework}")
                framework_result = self._process_single_framework(
                    pomdp_space, framework, gnn_file_path, **kwargs
                )
                
                results[framework] = framework_result
                
                if framework_result['success']:
                    processing_summary['frameworks_processed'].append(framework)
                    self.logger.info(f"✅ {framework}: {framework_result['message']}")
                else:
                    processing_summary['frameworks_failed'].append(framework)
                    self.logger.error(f"❌ {framework}: {framework_result['message']}")
                    
            except Exception as e:
                error_msg = f"Unexpected error processing {framework}: {e}"
                self.logger.error(error_msg)
                results[framework] = {
                    'success': False,
                    'message': error_msg,
                    'output_files': [],
                    'warnings': []
                }
                processing_summary['frameworks_failed'].append(framework)
        
        # Determine overall success:
        # Consider successful if at least 60% of frameworks succeeded OR at least one succeeded
        total_frameworks = len(frameworks)
        successful_frameworks = len(processing_summary['frameworks_processed'])
        success_rate = successful_frameworks / total_frameworks if total_frameworks > 0 else 0
        overall_success = success_rate >= 0.6 or successful_frameworks > 0
        
        if not overall_success:
            self.logger.warning(f"⚠️ Low framework success rate: {successful_frameworks}/{total_frameworks} ({success_rate*100:.1f}%)")
        
        # Save processing summary
        summary_file = self.base_output_dir / 'processing_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(processing_summary, f, indent=2)
        
        return {
            'overall_success': overall_success,
            'framework_results': results,
            'summary_file': str(summary_file),
            'output_directory': str(self.base_output_dir)
        }
    
    def _process_single_framework(self, 
                                 pomdp_space: 'POMDPStateSpace',
                                 framework: str,
                                 gnn_file_path: Optional[Path] = None,
                                 **kwargs) -> Dict[str, Any]:
        """
        Process POMDP state space for a single framework.
        
        Args:
            pomdp_space: POMDP state space data
            framework: Target framework name
            gnn_file_path: Original GNN file path
            **kwargs: Additional options
            
        Returns:
            Processing result dictionary
        """
        if framework not in self.framework_configs:
            return {
                'success': False,
                'message': f"Unknown framework: {framework}",
                'output_files': [],
                'warnings': []
            }
        
        config = self.framework_configs[framework]
        
        # Validate POMDP compatibility with framework
        validation_result = self._validate_pomdp_framework_compatibility(pomdp_space, framework)
        if not validation_result['compatible']:
            return {
                'success': False,
                'message': f"POMDP not compatible with {framework}: {validation_result['reason']}",
                'output_files': [],
                'warnings': validation_result.get('warnings', [])
            }
        
        # Create framework-specific output directory
        framework_output_dir = self.base_output_dir / config['output_subdir']
        framework_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert POMDP to GNN spec format expected by renderers
        gnn_spec = self._pomdp_to_gnn_spec(pomdp_space)
        
        # Get framework-specific renderer
        try:
            renderer_result = self._call_framework_renderer(
                framework, gnn_spec, framework_output_dir, **kwargs
            )
            
            if renderer_result['success']:
                # Create framework-specific documentation
                self._create_framework_documentation(
                    framework, pomdp_space, framework_output_dir, renderer_result
                )
                
                return {
                    'success': True,
                    'message': renderer_result['message'],
                    'output_files': renderer_result.get('artifacts', []),
                    'output_directory': str(framework_output_dir),
                    'warnings': validation_result.get('warnings', [])
                }
            else:
                return {
                    'success': False,
                    'message': renderer_result['message'],
                    'output_files': [],
                    'warnings': validation_result.get('warnings', [])
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f"Framework renderer failed: {e}",
                'output_files': [],
                'warnings': validation_result.get('warnings', [])
            }
    
    def _validate_pomdp_framework_compatibility(self, 
                                              pomdp_space: 'POMDPStateSpace',
                                              framework: str) -> Dict[str, Any]:
        """
        Validate that POMDP is compatible with target framework.
        
        Args:
            pomdp_space: POMDP state space data
            framework: Target framework name
            
        Returns:
            Validation result dictionary
        """
        config = self.framework_configs[framework]
        warnings = []
        
        # Check required matrices are present
        missing_matrices = []
        for required_matrix in config['requires_matrices']:
            matrix_attr = f"{required_matrix}_matrix" if required_matrix in ['A', 'B'] else f"{required_matrix}_vector"
            if getattr(pomdp_space, matrix_attr, None) is None:
                missing_matrices.append(required_matrix)
        
        if missing_matrices:
            return {
                'compatible': False,
                'reason': f"Missing required matrices: {missing_matrices}",
                'warnings': warnings
            }
        
        # Framework-specific checks
        if framework == 'rxinfer' and not config['supports_multi_modality']:
            if pomdp_space.num_observations > 1:  # This is a simplistic check
                warnings.append(f"{framework} has limited multi-modality support")
        
        # Check dimension limits
        max_reasonable_dim = 100  # Reasonable limit for most frameworks
        if (pomdp_space.num_states > max_reasonable_dim or 
            pomdp_space.num_observations > max_reasonable_dim or
            pomdp_space.num_actions > max_reasonable_dim):
            warnings.append("Large state spaces may cause performance issues")
        
        return {
            'compatible': True,
            'reason': None,
            'warnings': warnings
        }
    
    def _pomdp_to_gnn_spec(self, pomdp_space: 'POMDPStateSpace') -> Dict[str, Any]:
        """
        Convert POMDP state space to GNN spec format expected by renderers.
        
        Args:
            pomdp_space: POMDP state space data
            
        Returns:
            GNN specification dictionary
        """
        gnn_spec = {
            'name': pomdp_space.model_name or 'POMDP_Model',
            'model_name': pomdp_space.model_name or 'POMDP_Model',
            'description': pomdp_space.model_annotation or 'Extracted POMDP model',
            'model_parameters': {
                'num_hidden_states': pomdp_space.num_states,
                'num_obs': pomdp_space.num_observations,
                'num_actions': pomdp_space.num_actions
            },
            'initialparameterization': {},
            'variables': [],
            'connections': []
        }
        
        # Add matrices to initial parameterization
        if pomdp_space.A_matrix:
            gnn_spec['initialparameterization']['A'] = pomdp_space.A_matrix
        if pomdp_space.B_matrix:
            gnn_spec['initialparameterization']['B'] = pomdp_space.B_matrix
        if pomdp_space.C_vector:
            gnn_spec['initialparameterization']['C'] = pomdp_space.C_vector
        if pomdp_space.D_vector:
            gnn_spec['initialparameterization']['D'] = pomdp_space.D_vector
        if pomdp_space.E_vector:
            gnn_spec['initialparameterization']['E'] = pomdp_space.E_vector
        
        # Add variable definitions
        if pomdp_space.state_variables:
            gnn_spec['variables'].extend(pomdp_space.state_variables)
        if pomdp_space.observation_variables:
            gnn_spec['variables'].extend(pomdp_space.observation_variables)
        if pomdp_space.action_variables:
            gnn_spec['variables'].extend(pomdp_space.action_variables)
        
        # Add connections
        if pomdp_space.connections:
            gnn_spec['connections'] = [
                {'source': conn[0], 'relation': conn[1], 'target': conn[2]}
                for conn in pomdp_space.connections
            ]
        
        # Add ontology mapping if available
        if pomdp_space.ontology_mapping:
            gnn_spec['ontology_mapping'] = pomdp_space.ontology_mapping
        
        return gnn_spec
    
    def _call_framework_renderer(self, 
                                framework: str,
                                gnn_spec: Dict[str, Any],
                                output_dir: Path,
                                **kwargs) -> Dict[str, Any]:
        """
        Call the appropriate framework renderer.
        
        Args:
            framework: Target framework name
            gnn_spec: GNN specification
            output_dir: Output directory for this framework
            **kwargs: Additional renderer options
            
        Returns:
            Renderer result dictionary
        """
        config = self.framework_configs[framework]
        model_name = gnn_spec.get('name', 'pomdp_model')
        
        if framework == 'pymdp':
            return self._call_pymdp_renderer(gnn_spec, output_dir, **kwargs)
        elif framework == 'rxinfer':
            return self._call_rxinfer_renderer(gnn_spec, output_dir, **kwargs)
        elif framework == 'activeinference_jl':
            return self._call_activeinference_jl_renderer(gnn_spec, output_dir, **kwargs)
        elif framework == 'jax':
            return self._call_jax_renderer(gnn_spec, output_dir, **kwargs)
        elif framework == 'discopy':
            return self._call_discopy_renderer(gnn_spec, output_dir, **kwargs)
        else:
            return {
                'success': False,
                'message': f"No renderer implemented for {framework}",
                'artifacts': []
            }
    
    def _call_pymdp_renderer(self, gnn_spec: Dict[str, Any], output_dir: Path, **kwargs) -> Dict[str, Any]:
        """Call PyMDP renderer."""
        try:
            from .pymdp.pymdp_renderer import render_gnn_to_pymdp
            
            model_name = gnn_spec.get('name', 'pomdp_model')
            output_file = output_dir / f"{model_name}_pymdp.py"
            
            # Validate state spaces are present before rendering
            validation_result = self._validate_state_spaces_in_spec(gnn_spec, 'pymdp')
            if not validation_result['valid']:
                warnings = validation_result.get('warnings', [])
                if validation_result.get('critical', False):
                    return {
                        'success': False,
                        'message': f"State space validation failed: {validation_result.get('reason', 'Unknown')}",
                        'artifacts': [],
                        'warnings': warnings
                    }
            
            success, message, warnings = render_gnn_to_pymdp(gnn_spec, output_file, kwargs)
            
            # Post-render validation: verify state spaces are in generated script
            if success and output_file.exists():
                post_validation = self._validate_state_spaces_in_script(output_file, gnn_spec)
                if not post_validation['valid']:
                    warnings.extend(post_validation.get('warnings', []))
            
            return {
                'success': success,
                'message': message,
                'artifacts': [str(output_file)] if success else [],
                'warnings': warnings
            }
            
        except ImportError:
            return {
                'success': False,
                'message': "PyMDP renderer not available",
                'artifacts': []
            }
    
    def _call_rxinfer_renderer(self, gnn_spec: Dict[str, Any], output_dir: Path, **kwargs) -> Dict[str, Any]:
        """Call RxInfer renderer."""
        try:
            from .rxinfer.rxinfer_renderer import render_gnn_to_rxinfer
            
            model_name = gnn_spec.get('name', 'pomdp_model')
            output_file = output_dir / f"{model_name}_rxinfer.jl"
            
            # Validate state spaces are present before rendering
            validation_result = self._validate_state_spaces_in_spec(gnn_spec, 'rxinfer')
            if not validation_result['valid']:
                warnings = validation_result.get('warnings', [])
                if validation_result.get('critical', False):
                    return {
                        'success': False,
                        'message': f"State space validation failed: {validation_result.get('reason', 'Unknown')}",
                        'artifacts': [],
                        'warnings': warnings
                    }
            
            success, message, warnings = render_gnn_to_rxinfer(gnn_spec, output_file, kwargs)
            
            # Post-render validation
            if success and output_file.exists():
                post_validation = self._validate_state_spaces_in_script(output_file, gnn_spec)
                if not post_validation['valid']:
                    warnings.extend(post_validation.get('warnings', []))
            
            return {
                'success': success,
                'message': message,
                'artifacts': [str(output_file)] if success else [],
                'warnings': warnings
            }
            
        except ImportError:
            return {
                'success': False,
                'message': "RxInfer renderer not available",
                'artifacts': []
            }
    
    def _call_activeinference_jl_renderer(self, gnn_spec: Dict[str, Any], output_dir: Path, **kwargs) -> Dict[str, Any]:
        """Call ActiveInference.jl renderer."""
        try:
            from .activeinference_jl.activeinference_renderer import render_gnn_to_activeinference_jl
            
            model_name = gnn_spec.get('name', 'pomdp_model')
            output_file = output_dir / f"{model_name}_activeinference.jl"
            
            # Validate state spaces are present before rendering
            validation_result = self._validate_state_spaces_in_spec(gnn_spec, 'activeinference_jl')
            warnings = []
            if not validation_result['valid']:
                warnings = validation_result.get('warnings', [])
                if validation_result.get('critical', False):
                    return {
                        'success': False,
                        'message': f"State space validation failed: {validation_result.get('reason', 'Unknown')}",
                        'artifacts': [],
                        'warnings': warnings
                    }
            
            success, message, artifacts = render_gnn_to_activeinference_jl(gnn_spec, output_file, kwargs)
            
            # Post-render validation
            if success and output_file.exists():
                post_validation = self._validate_state_spaces_in_script(output_file, gnn_spec)
                if not post_validation['valid']:
                    warnings.extend(post_validation.get('warnings', []))
            
            return {
                'success': success,
                'message': message,
                'artifacts': artifacts,
                'warnings': warnings
            }
            
        except ImportError:
            return {
                'success': False,
                'message': "ActiveInference.jl renderer not available",
                'artifacts': []
            }
    
    def _call_jax_renderer(self, gnn_spec: Dict[str, Any], output_dir: Path, **kwargs) -> Dict[str, Any]:
        """Call JAX renderer."""
        try:
            from .jax.jax_renderer import render_gnn_to_jax
            
            model_name = gnn_spec.get('name', 'pomdp_model')
            output_file = output_dir / f"{model_name}_jax.py"
            
            success, message, artifacts = render_gnn_to_jax(gnn_spec, output_file, kwargs)
            
            return {
                'success': success,
                'message': message,
                'artifacts': artifacts,
                'warnings': []
            }
            
        except ImportError:
            return {
                'success': False,
                'message': "JAX renderer not available",
                'artifacts': []
            }
    
    def _call_discopy_renderer(self, gnn_spec: Dict[str, Any], output_dir: Path, **kwargs) -> Dict[str, Any]:
        """Call DisCoPy renderer."""
        try:
            from .discopy.discopy_renderer import render_gnn_to_discopy
            
            model_name = gnn_spec.get('name', 'pomdp_model')
            output_file = output_dir / f"{model_name}_discopy.py"
            
            success, message, warnings = render_gnn_to_discopy(gnn_spec, output_file, kwargs)
            
            return {
                'success': success,
                'message': message,
                'artifacts': [str(output_file)] if success else [],
                'warnings': warnings
            }
            
        except ImportError:
            return {
                'success': False,
                'message': "DisCoPy renderer not available",
                'artifacts': []
            }
    
    def _validate_state_spaces_in_spec(self, gnn_spec: Dict[str, Any], framework: str) -> Dict[str, Any]:
        """
        Validate that state spaces are present in GNN spec.
        
        Args:
            gnn_spec: GNN specification dictionary
            framework: Target framework name
            
        Returns:
            Validation result dictionary
        """
        warnings = []
        initial_params = gnn_spec.get('initialparameterization', {})
        config = self.framework_configs[framework]
        
        # Check required matrices
        missing_required = []
        for required_matrix in config['requires_matrices']:
            if required_matrix not in initial_params:
                missing_required.append(required_matrix)
        
        if missing_required:
            return {
                'valid': False,
                'critical': True,
                'reason': f"Missing required matrices: {missing_required}",
                'warnings': warnings
            }
        
        # Check optional matrices
        for optional_matrix in config.get('optional_matrices', []):
            if optional_matrix not in initial_params:
                warnings.append(f"Optional matrix {optional_matrix} not found")
        
        return {
            'valid': True,
            'critical': False,
            'warnings': warnings
        }
    
    def _validate_state_spaces_in_script(self, script_path: Path, gnn_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that state spaces are present in generated script.
        
        Args:
            script_path: Path to generated script
            gnn_spec: Original GNN specification
            
        Returns:
            Validation result dictionary
        """
        warnings = []
        
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
            
            initial_params = gnn_spec.get('initialparameterization', {})
            
            # Check if matrices are referenced in script
            for matrix_name in ['A', 'B', 'C', 'D', 'E']:
                if matrix_name in initial_params:
                    # Check if matrix is present in script (as variable or in data structure)
                    if matrix_name not in script_content and f'"{matrix_name}"' not in script_content:
                        warnings.append(f"Matrix {matrix_name} may not be properly injected into script")
            
            return {
                'valid': len(warnings) == 0,
                'warnings': warnings
            }
        except Exception as e:
            return {
                'valid': False,
                'warnings': [f"Failed to validate script: {e}"]
            }
    
    def _create_framework_documentation(self, 
                                      framework: str,
                                      pomdp_space: 'POMDPStateSpace',
                                      output_dir: Path,
                                      render_result: Dict[str, Any]) -> None:
        """
        Create framework-specific documentation.
        
        Args:
            framework: Framework name
            pomdp_space: POMDP state space
            output_dir: Output directory
            render_result: Rendering result
        """
        try:
            doc_file = output_dir / 'README.md'
            
            # Get model annotation safely
            model_annotation = getattr(pomdp_space, 'model_annotation', None) or 'N/A'
            
            doc_content = f"""# {framework.upper()} Rendering Results

Generated from GNN POMDP Model: **{pomdp_space.model_name}**

## Model Information

- **Model Name**: {pomdp_space.model_name}
- **Model Description**: {model_annotation}
- **Generation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## POMDP Dimensions

- **Number of States**: {pomdp_space.num_states}
- **Number of Observations**: {pomdp_space.num_observations}
- **Number of Actions**: {pomdp_space.num_actions}

## Active Inference Matrices

### Available Matrices/Vectors:
"""
            
            # Safely check for matrices/vectors
            A_matrix = getattr(pomdp_space, 'A_matrix', None)
            if A_matrix and len(A_matrix) > 0 and len(A_matrix[0]) > 0:
                doc_content += f"- **A Matrix (Likelihood)**: {len(A_matrix)}×{len(A_matrix[0])} - Maps hidden states to observations\n"
            
            B_matrix = getattr(pomdp_space, 'B_matrix', None)
            if B_matrix and len(B_matrix) > 0 and len(B_matrix[0]) > 0:
                try:
                    doc_content += f"- **B Matrix (Transition)**: {len(B_matrix[0])}×{len(B_matrix[0][0])}×{len(B_matrix)} - State transitions given actions\n"
                except (IndexError, TypeError):
                    doc_content += f"- **B Matrix (Transition)**: Present - State transitions given actions\n"
            
            C_vector = getattr(pomdp_space, 'C_vector', None)
            if C_vector and len(C_vector) > 0:
                doc_content += f"- **C Vector (Preferences)**: Length {len(C_vector)} - Preferences over observations\n"
            
            D_vector = getattr(pomdp_space, 'D_vector', None)
            if D_vector and len(D_vector) > 0:
                doc_content += f"- **D Vector (Prior)**: Length {len(D_vector)} - Prior beliefs over states\n"
            
            E_vector = getattr(pomdp_space, 'E_vector', None)
            if E_vector and len(E_vector) > 0:
                doc_content += f"- **E Vector (Habits)**: Length {len(E_vector)} - Policy priors\n"
            
            doc_content += f"""

## Generated Files

"""
            
            for artifact in render_result.get('artifacts', []):
                artifact_path = Path(artifact)
                doc_content += f"- `{artifact_path.name}` - {framework} simulation script\n"
            
            if render_result.get('warnings'):
                doc_content += f"""

## Warnings

"""
                for warning in render_result['warnings']:
                    doc_content += f"- ⚠️ {warning}\n"
            
            doc_content += f"""

## Usage

Refer to the main {framework} documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: {framework}
- **File Extension**: {self.framework_configs[framework]['file_extension']}
- **Multi-Modality Support**: {'✅' if self.framework_configs[framework]['supports_multi_modality'] else '❌'}
- **Multi-Factor Support**: {'✅' if self.framework_configs[framework]['supports_multi_factor'] else '❌'}
"""
            
            with open(doc_file, 'w') as f:
                f.write(doc_content)
                
            self.logger.info(f"Created documentation: {doc_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create documentation for {framework}: {e}")


def process_pomdp_for_frameworks(pomdp_space: 'POMDPStateSpace',
                               output_dir: Union[str, Path],
                               frameworks: Optional[List[str]] = None,
                               gnn_file_path: Optional[Path] = None,
                               **kwargs) -> Dict[str, Any]:
    """
    Convenience function to process POMDP for multiple frameworks.
    
    Args:
        pomdp_space: POMDP state space data
        output_dir: Base output directory
        frameworks: List of frameworks to process (default: all)
        gnn_file_path: Original GNN file path
        **kwargs: Additional processing options
        
    Returns:
        Processing results dictionary
    """
    processor = POMDPRenderProcessor(Path(output_dir))
    return processor.process_pomdp_for_all_frameworks(
        pomdp_space, gnn_file_path, frameworks, **kwargs
    )
