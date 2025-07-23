"""
SAPF-GNN Processor

Core module for converting GNN (Generalized Notation Notation) models to 
SAPF (Sound As Pure Form) audio representations.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

class SAPFGNNProcessor:
    """
    Main processor for converting GNN models to SAPF audio representations.
    """
    
    def __init__(self):
        self.base_frequency = 261.63  # C4
        self.sample_rate = 44100
        self.default_duration = 10.0
        
    def parse_gnn_sections(self, gnn_content: str) -> Dict[str, Any]:
        """
        Parse GNN content into structured sections.
        
        Args:
            gnn_content: Raw GNN file content
            
        Returns:
            Dictionary containing parsed sections
        """
        sections = {}
        
        # Extract ModelName
        model_name_match = re.search(r'## ModelName\s*\n([^\n]+)', gnn_content)
        if model_name_match:
            sections['ModelName'] = model_name_match.group(1).strip()
        
        # Extract StateSpaceBlock
        state_space_match = re.search(
            r'## StateSpaceBlock\s*\n(.*?)(?=\n## |\n# |\Z)', 
            gnn_content, 
            re.DOTALL
        )
        if state_space_match:
            sections['StateSpaceBlock'] = self._parse_state_space(state_space_match.group(1))
        
        # Extract Connections
        connections_match = re.search(
            r'## Connections\s*\n(.*?)(?=\n## |\n# |\Z)', 
            gnn_content, 
            re.DOTALL
        )
        if connections_match:
            sections['Connections'] = self._parse_connections(connections_match.group(1))
        
        # Extract InitialParameterization
        params_match = re.search(
            r'## InitialParameterization\s*\n(.*?)(?=\n## |\n# |\Z)', 
            gnn_content, 
            re.DOTALL
        )
        if params_match:
            sections['InitialParameterization'] = self._parse_parameters(params_match.group(1))
        
        # Extract Time configuration
        time_match = re.search(
            r'## Time\s*\n(.*?)(?=\n## |\n# |\Z)', 
            gnn_content, 
            re.DOTALL
        )
        if time_match:
            sections['Time'] = self._parse_time_config(time_match.group(1))
        
        return sections
    
    def _parse_state_space(self, content: str) -> List[Dict[str, Any]]:
        """Parse StateSpaceBlock content including matrices, states, observations, and policies."""
        states = []
        lines = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]
        
        for line in lines:
            # Parse any variable with dimensions and optional type
            var_match = re.match(r'(\w+)\[([^\]]+)\](?:.*?type=(\w+))?', line)
            if var_match:
                var_name = var_match.group(1)
                dimensions_str = var_match.group(2)
                var_type_match = var_match.group(3)
                
                # Determine variable category and type
                if var_name.startswith('A_'):
                    var_category = 'likelihood_matrix'
                    var_type = 'matrix'
                elif var_name.startswith('B_'):
                    var_category = 'transition_matrix'
                    var_type = 'matrix'
                elif var_name.startswith('C_'):
                    var_category = 'preference_vector'
                    var_type = 'vector'
                elif var_name.startswith('D_'):
                    var_category = 'prior_vector'
                    var_type = 'vector'
                elif var_name.startswith('s_'):
                    var_category = 'hidden_state'
                    var_type = var_type_match or 'continuous'
                elif var_name.startswith('o_'):
                    var_category = 'observation'
                    var_type = var_type_match or 'discrete'
                elif var_name.startswith('π_') or var_name.startswith('u_'):
                    var_category = 'policy_control'
                    var_type = var_type_match or 'discrete'
                else:
                    var_category = 'other'
                    var_type = var_type_match or 'continuous'
                
                # Parse dimensions
                dimensions = []
                dim_parts = dimensions_str.split(',')
                for dim in dim_parts:
                    dim_clean = dim.strip()
                    # Extract numeric part, ignoring type specifications
                    numeric_match = re.match(r'(\d+)', dim_clean)
                    if numeric_match:
                        dimensions.append(int(numeric_match.group(1)))
                    else:
                        dimensions.append(1)
                
                states.append({
                    'name': var_name,
                    'dimensions': dimensions,
                    'type': var_type,
                    'category': var_category
                })
        
        return states
    
    def _parse_connections(self, content: str) -> List[Dict[str, str]]:
        """Parse Connections content."""
        connections = []
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        for line in lines:
            # Parse connection patterns like "s_f0 > o_m0" or "s_f0 - s_f1"
            conn_match = re.match(r'(\w+)\s*([>-])\s*(\w+)', line)
            if conn_match:
                source = conn_match.group(1)
                conn_type = conn_match.group(2)
                target = conn_match.group(3)
                
                connections.append({
                    'source': source,
                    'target': target,
                    'type': conn_type,
                    'directed': conn_type == '>'
                })
        
        return connections
    
    def _parse_parameters(self, content: str) -> Dict[str, Any]:
        """Parse InitialParameterization content."""
        params = {}
        
        # Look for matrix definitions (A, B, C, D)
        matrix_patterns = {
            'A': r'A\s*=\s*\[(.*?)\]',
            'B': r'B\s*=\s*\[(.*?)\]', 
            'C': r'C\s*=\s*\[(.*?)\]',
            'D': r'D\s*=\s*\[(.*?)\]'
        }
        
        for matrix_name, pattern in matrix_patterns.items():
            match = re.search(pattern, content, re.DOTALL)
            if match:
                matrix_content = match.group(1)
                params[matrix_name] = self._parse_matrix(matrix_content)
        
        return params
    
    def _parse_matrix(self, matrix_str: str) -> List[List[float]]:
        """Parse matrix string into numerical array."""
        matrix = []
        lines = [line.strip() for line in matrix_str.split('\n') if line.strip()]
        
        for line in lines:
            # Remove semicolons and split by whitespace/commas
            row_str = line.replace(';', '').replace(',', ' ')
            row_values = []
            
            for val_str in row_str.split():
                try:
                    row_values.append(float(val_str))
                except ValueError:
                    continue
            
            if row_values:
                matrix.append(row_values)
        
        return matrix
    
    def _parse_time_config(self, content: str) -> Dict[str, Any]:
        """Parse Time configuration."""
        config = {}
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        for line in lines:
            if 'ModelTimeHorizon' in line:
                horizon_match = re.search(r'ModelTimeHorizon\s*:\s*(\d+)', line)
                if horizon_match:
                    config['ModelTimeHorizon'] = int(horizon_match.group(1))
            
            if 'DiscreteTime' in line:
                config['DiscreteTime'] = 'DiscreteTime' in line
        
        return config
    
    def convert_to_sapf(self, gnn_sections: Dict[str, Any], model_name: str) -> str:
        """
        Convert parsed GNN sections to SAPF code.
        
        Args:
            gnn_sections: Parsed GNN sections
            model_name: Name of the model
            
        Returns:
            Generated SAPF code
        """
        # Calculate model characteristics for strong audio differentiation
        state_count = len(gnn_sections.get('StateSpaceBlock', []))
        connection_count = len(gnn_sections.get('Connections', []))
        matrix_count = len(gnn_sections.get('InitialParameterization', {}))
        time_horizon = gnn_sections.get('Time', {}).get('ModelTimeHorizon', 10)
        
        # Create distinctive model signature and characteristics
        model_signature = self._get_model_signature(model_name, gnn_sections)
        complexity = self._get_complexity_level(gnn_sections)
        
        # Much stronger base frequency differentiation
        base_freq = self.base_frequency + (model_signature * 50) + (state_count * 25)
        model_scale = self._get_model_scale(model_name)
        model_tempo = self._get_model_tempo(complexity)
        
        sapf_lines = [
            f"; SAPF Audio Representation for GNN Model: {model_name}",
            f"; Generated by SAPF-GNN Processor with Distinctive Audio Mapping",
            f"; Model characteristics: {state_count} states, {connection_count} connections, {matrix_count} matrices",
            f"; Model signature: {model_signature}, Scale: {model_scale}, Tempo: {model_tempo}",
            "",
            "; Model Schema with Distinctive Audio Characteristics",
            f'{{ :model_name "{model_name}" :gnn_version "1.4" :complexity "{complexity}" :signature {model_signature} }} = model_schema',
            f'{base_freq:.2f} = base_freq',
            f'{model_scale} = model_scale',
            f'{model_tempo} = model_tempo',
            f'{self._get_model_reverb(model_name)} = model_reverb',
            ""
        ]
        
        # Generate state space oscillators
        if 'StateSpaceBlock' in gnn_sections:
            sapf_lines.extend(self._generate_state_oscillators(gnn_sections['StateSpaceBlock'], base_freq))
        
        # Generate connection routing
        if 'Connections' in gnn_sections:
            sapf_lines.extend(self._generate_connection_routing(gnn_sections['Connections']))
        
        # Generate matrix processing
        if 'InitialParameterization' in gnn_sections:
            sapf_lines.extend(self._generate_matrix_processing(gnn_sections['InitialParameterization']))
        
        # Generate temporal structure
        if 'Time' in gnn_sections:
            sapf_lines.extend(self._generate_temporal_structure(gnn_sections['Time']))
        
        # Final audio combination with strong model-specific processing
        duration = max(5, min(20, time_horizon))
        attack = 0.1 + (state_count * 0.02)
        release = 0.2 + (connection_count * 0.05)
        
        sapf_lines.extend([
            "",
            "; Final Audio Generation with Model-Specific Processing",
            "\\generate_final_audio [",
            "  ; Combine all audio elements with model characteristics",
            "  state_oscillators connection_audio + matrix_audio + temporal_audio +",
            "  ",
            "  ; Apply model-specific scale quantization",
            f"  model_scale quantize",
            "  ",
            "  ; Apply model-specific tempo modulation",
            f"  dup model_tempo lfnoise * +",
            "  ",
            "  ; Apply model-specific reverb",
            f"  dup model_reverb 0.2 reverb +",
            "  ",
            "  ; Apply model-specific volume and envelope",
            f"  .4 * {duration} sec {attack:.2f} 1 0.8 {release:.2f} env *",
            "  ",
            "  ; Final model signature filtering",
            f"  dup {base_freq * 0.5:.1f} highpass 0.3 * +",
            "] = final_audio",
            "",
            "; Execute and play audio",
            "generate_final_audio !",
            "final_audio play"
        ])
        
        return "\n".join(sapf_lines)
    
    def _get_complexity_level(self, gnn_sections: Dict[str, Any]) -> str:
        """Determine complexity level based on model characteristics."""
        state_count = len(gnn_sections.get('StateSpaceBlock', []))
        connection_count = len(gnn_sections.get('Connections', []))
        matrix_count = len(gnn_sections.get('InitialParameterization', {}))
        
        complexity_score = state_count + (connection_count * 0.5) + (matrix_count * 2)
        
        if complexity_score < 3:
            return "simple"
        elif complexity_score < 8:
            return "moderate"
        else:
            return "complex"
    
    def _get_model_signature(self, model_name: str, gnn_sections: Dict[str, Any]) -> int:
        """Generate a unique signature for each model based on name and structure."""
        # Create hash-like signature from model characteristics
        name_hash = sum(ord(c) for c in model_name.lower()) % 100
        state_signature = len(gnn_sections.get('StateSpaceBlock', [])) * 7
        connection_signature = len(gnn_sections.get('Connections', [])) * 13
        
        # Include variable type diversity
        state_types = set()
        for state in gnn_sections.get('StateSpaceBlock', []):
            state_types.add(state.get('category', 'unknown'))
        type_signature = len(state_types) * 11
        
        return (name_hash + state_signature + connection_signature + type_signature) % 255
    
    def _get_model_scale(self, model_name: str) -> str:
        """Get musical scale based on model name for tonal differentiation."""
        scales = {
            'pymdp': '[0 2 4 5 7 9 11]',     # Major scale - bright, optimistic
            'rxinfer': '[0 2 3 5 7 8 10]',   # Minor scale - contemplative
            'multiagent': '[0 1 4 6 7 10]',  # Blues scale - complex interactions
            'hidden': '[0 3 5 6 7 10]',      # Dorian mode - mysterious
            'markov': '[0 2 3 6 7 8 11]',    # Harmonic minor - dramatic
            'agent': '[0 2 4 6 8 10]',       # Whole tone - ethereal
        }
        
        for key, scale in scales.items():
            if key in model_name.lower():
                return scale
        
        # Default pentatonic scale
        return '[0 2 4 7 9]'
    
    def _get_model_tempo(self, complexity: str) -> float:
        """Get tempo multiplier based on complexity."""
        tempo_map = {
            'simple': 1.2,      # Faster for simple models
            'moderate': 1.0,    # Normal tempo
            'complex': 0.8      # Slower for complex models
        }
        return tempo_map.get(complexity, 1.0)
    
    def _get_model_reverb(self, model_name: str) -> float:
        """Get reverb amount based on model characteristics."""
        if 'multiagent' in model_name.lower():
            return 0.4  # High reverb for multiagent (spatial)
        elif 'hidden' in model_name.lower():
            return 0.6  # More reverb for hidden states (mystery)
        elif 'pymdp' in model_name.lower():
            return 0.2  # Clean sound for PyMDP (precise)
        else:
            return 0.3  # Moderate reverb
    
    def _generate_state_oscillators(self, state_space: List[Dict[str, Any]], base_freq: float) -> List[str]:
        """Generate SAPF code for state space oscillators."""
        lines = [
            "; State Space Oscillators",
            "\\generate_state_oscillators [",
            ""
        ]
        
        for i, state in enumerate(state_space):
            var_name = state['name']
            dimensions = state.get('dimensions', [1])
            var_type = state.get('type', 'continuous')
            var_category = state.get('category', 'other')
            
            # Create much stronger frequency patterns based on variable category and characteristics
            category_offset = {
                'likelihood_matrix': 200,   # Doubled offsets for stronger differentiation
                'transition_matrix': 400,
                'preference_vector': 600,
                'prior_vector': 800,
                'hidden_state': 1000,
                'observation': 1200,
                'policy_control': 1400,
                'other': 1600
            }.get(var_category, 0)
            
            # Much stronger frequency separation
            freq_offset = category_offset + (i * 75) + (sum(dimensions) * 25)
            
            # Strongly differentiated amplitudes, waveforms, and effects by category
            if var_category in ['likelihood_matrix', 'transition_matrix']:
                amplitude = 0.5
                osc_type = "sinosc"
                effect = "dup 0.3 0.1 delay +"  # Echo effect for matrices
            elif var_category in ['preference_vector', 'prior_vector']:
                amplitude = 0.4
                osc_type = "lfsaw"
                effect = "dup 0.2 lfnoise *"  # Noise modulation for preferences
            elif var_category == 'hidden_state':
                amplitude = 0.45
                osc_type = "sinosc" if var_type == 'continuous' else "tri"
                effect = "dup 0.15 0.05 delay 0.5 * +"  # Subtle delay for mystery
            elif var_category == 'observation':
                amplitude = 0.35
                osc_type = "square"  # Sharp square waves for observations
                effect = "4 lowpass"  # Low-pass filter to soften
            elif var_category == 'policy_control':
                amplitude = 0.3
                osc_type = "lfpulse"  # Pulse waves for control
                effect = "dup 0.1 lfnoise 0.1 * +"  # Small noise addition
            else:
                amplitude = 0.4
                osc_type = "sinosc"
                effect = ""  # Clean for others
            
            total_dims = sum(dimensions) if dimensions else 1
            
            if total_dims <= 6:
                # Simple variable - single oscillator with effects
                lines.append(f"  ; {var_name} ({var_category}, {var_type}, dims={dimensions})")
                osc_line = f"  base_freq {freq_offset} + 0 {osc_type} {amplitude} *"
                if effect:
                    osc_line += f" {effect}"
                osc_line += f" = {var_name}_osc"
                lines.append(osc_line)
            else:
                # Complex variable - harmonic series with audio effects
                harmonic_count = min(total_dims, 8)
                detune = (i * 5) + (len(var_category) % 7)  # Stronger category-based detuning
                lines.append(f"  ; {var_name} ({var_category}, dims={dimensions}) harmonic series")
                lines.append(f"  [")
                lines.append(f"    {harmonic_count} 1 to [")
                harmonic_line = f"      base_freq {freq_offset} + {detune} + i * 2 + {osc_type} i 1 + / {amplitude:.1f} *"
                if effect:
                    harmonic_line += f" {effect}"
                lines.append(harmonic_line)
                lines.append(f"    ] +/")
                lines.append(f"  ] = {var_name}_osc")
            
            lines.append("")
        
        lines.extend([
            "  ; Mix all state oscillators",
            "  [" + " ".join(f"{state['name']}_osc" for state in state_space) + "] +/",
            "] = state_oscillators",
            ""
        ])
        
        return lines
    
    def _generate_connection_routing(self, connections: List[Dict[str, str]]) -> List[str]:
        """Generate SAPF code for connection routing."""
        lines = [
            "; Connection Routing",
            "\\generate_connection_audio [",
            "  state_oscillators = base_signal",
            ""
        ]
        
        for conn in connections:
            source = conn['source']
            target = conn['target']
            conn_type = conn['type']
            
            if conn_type == '>':
                # Directed connection - modulation
                lines.append(f"  ; {source} -> {target} (directed)")
                lines.append(f"  base_signal .1 0 lfsaw .2 * 1 + * = base_signal")
            else:
                # Undirected connection - mixing
                lines.append(f"  ; {source} - {target} (undirected)")
                lines.append(f"  base_signal dup .5 * + = base_signal")
        
        lines.extend([
            "  base_signal",
            "] = connection_audio",
            ""
        ])
        
        return lines
    
    def _generate_matrix_processing(self, params: Dict[str, Any]) -> List[str]:
        """Generate SAPF code for matrix-based processing."""
        lines = [
            "; Matrix-based Audio Processing",
            "\\generate_matrix_audio [",
            "  connection_audio = signal",
            ""
        ]
        
        if 'A' in params:
            lines.extend([
                "  ; A-matrix (likelihood) -> spectral filtering",
                "  signal dup 800 0 lpf .2 * + = signal"
            ])
        
        if 'B' in params:
            lines.extend([
                "  ; B-matrix (transitions) -> temporal modulation",
                "  signal .5 0 lfsaw .3 * 1 + * = signal"
            ])
        
        if 'C' in params:
            lines.extend([
                "  ; C-matrix (preferences) -> harmonic enhancement",
                "  signal dup 2 * 0 sinosc .15 * + = signal"
            ])
        
        if 'D' in params:
            lines.extend([
                "  ; D-matrix (priors) -> bass foundation",
                "  signal base_freq 2 / 0 sinosc .2 * + = signal"
            ])
        
        lines.extend([
            "  signal",
            "] = matrix_audio",
            ""
        ])
        
        return lines
    
    def _generate_temporal_structure(self, time_config: Dict[str, Any]) -> List[str]:
        """Generate SAPF code for temporal structure."""
        horizon = time_config.get('ModelTimeHorizon', 10)
        discrete = time_config.get('DiscreteTime', False)
        
        lines = [
            "; Temporal Structure",
            "\\generate_temporal_audio [",
            f"  {horizon} = time_horizon",
            ""
        ]
        
        if discrete:
            lines.extend([
                "  ; Discrete time - stepped envelope",
                "  time_horizon sec 0 1 1 0 env = time_env"
            ])
        else:
            lines.extend([
                "  ; Continuous time - smooth envelope", 
                "  time_horizon sec 0 1 0.8 0.2 env = time_env"
            ])
        
        lines.extend([
            "  matrix_audio time_env *",
            "] = temporal_audio",
            ""
        ])
        
        return lines

def convert_gnn_to_sapf(gnn_content: str, model_name: str) -> str:
    """
    Convert GNN content to SAPF code.
    
    Args:
        gnn_content: Raw GNN file content
        model_name: Name of the model
        
    Returns:
        Generated SAPF code
    """
    processor = SAPFGNNProcessor()
    sections = processor.parse_gnn_sections(gnn_content)
    return processor.convert_to_sapf(sections, model_name)

def validate_sapf_code(sapf_code: str) -> Tuple[bool, List[str]]:
    """
    Validate SAPF code for basic syntax issues.
    
    Args:
        sapf_code: SAPF code to validate
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Basic validation checks
    if not sapf_code.strip():
        issues.append("Empty SAPF code")
        return False, issues
    
    # Check for balanced brackets
    bracket_count = 0
    for char in sapf_code:
        if char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
    
    if bracket_count != 0:
        issues.append(f"Unbalanced brackets: {bracket_count}")
    
    # Check for required elements
    if 'play' not in sapf_code:
        issues.append("No 'play' command found")
    
    if not re.search(r'= \w+', sapf_code):
        issues.append("No variable assignments found")
    
    return len(issues) == 0, issues

def generate_audio_from_sapf(sapf_code: str, output_file: Path, duration: float = 10.0) -> bool:
    """
    Generate audio file from SAPF code (using Python synthesis).
    
    Args:
        sapf_code: SAPF code to execute
        output_file: Output audio file path
        duration: Audio duration in seconds
        
    Returns:
        True if successful
    """
    try:
        from .audio_generators import SyntheticAudioGenerator
        
        generator = SyntheticAudioGenerator()
        return generator.generate_from_sapf(sapf_code, output_file, duration)
        
    except Exception as e:
        logger.error(f"Failed to generate audio: {e}")
        return False 