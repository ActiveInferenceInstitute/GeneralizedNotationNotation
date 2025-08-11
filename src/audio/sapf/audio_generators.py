"""
SAPF Audio Generators

This module provides audio generation capabilities for SAPF code,
including synthetic oscillators, envelopes, and audio file output.
"""

import wave
import struct
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# Use NumPy FFT implementations to avoid optional SciPy dependency at import-time

logger = logging.getLogger(__name__)

class SyntheticAudioGenerator:
    """
    Generates synthetic audio based on SAPF code analysis.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.base_frequency = 261.63  # C4
        
    def generate_from_sapf(self, sapf_code: str, output_file: Path, duration: float, 
                          create_visualization: bool = True) -> bool:
        """
        Generate audio file from SAPF code analysis with optional waveform visualization.
        
        Args:
            sapf_code: SAPF code to analyze and convert
            output_file: Output audio file path
            duration: Audio duration in seconds
            create_visualization: Whether to create waveform and spectrum visualizations
            
        Returns:
            True if successful
        """
        try:
            # Analyze SAPF code for audio parameters
            audio_params = self._analyze_sapf_code(sapf_code)
            
            # Generate audio based on parameters
            audio_data = self._generate_audio(audio_params, duration)
            
            # Create visualizations if requested
            if create_visualization:
                self._create_waveform_visualizations(audio_data, output_file, audio_params)
            
            # Write to file
            return self._write_wav_file(audio_data, output_file)
            
        except Exception as e:
            logger.error(f"Failed to generate audio from SAPF: {e}")
            return False
    
    def _analyze_sapf_code(self, sapf_code: str) -> Dict[str, Any]:
        """
        Analyze SAPF code to extract audio generation parameters.
        
        Args:
            sapf_code: SAPF code to analyze
            
        Returns:
            Dictionary of audio parameters
        """
        params = {
            'base_frequency': self.base_frequency,
            'oscillators': [],
            'filters': [],
            'envelopes': [],
            'effects': [],
            'complexity': 'moderate'
        }
        
        lines = sapf_code.split('\n')
        
        # Extract base frequency if specified
        for line in lines:
            base_freq_match = re.search(r'(\d+\.?\d*)\s+=\s+base_freq', line)
            if base_freq_match:
                params['base_frequency'] = float(base_freq_match.group(1))
                break
        
        # Extract complexity level
        complexity_match = re.search(r':complexity\s+"(\w+)"', sapf_code)
        if complexity_match:
            params['complexity'] = complexity_match.group(1)
        
        for line in lines:
            line = line.strip()
            
            # Extract different oscillator types and frequencies
            # Sine oscillators
            sine_match = re.search(r'base_freq\s+(\d+\.?\d*)\s+\+.*?sinosc\s+([\d\.]+)\s*\*', line)
            if sine_match:
                freq_offset = float(sine_match.group(1))
                amplitude = float(sine_match.group(2))
                params['oscillators'].append({
                    'type': 'sine',
                    'frequency': params['base_frequency'] + freq_offset,
                    'amplitude': amplitude
                })
            
            # Sawtooth oscillators (lfsaw)
            saw_match = re.search(r'base_freq\s+(\d+\.?\d*)\s+\+.*?lfsaw\s+([\d\.]+)\s*\*', line)
            if saw_match:
                freq_offset = float(saw_match.group(1))
                amplitude = float(saw_match.group(2))
                params['oscillators'].append({
                    'type': 'saw',
                    'frequency': params['base_frequency'] + freq_offset,
                    'amplitude': amplitude
                })
            
            # Simple frequency extraction (fallback)
            freq_match = re.search(r'(\d+\.?\d*)\s+0\s+(sinosc|lfsaw)', line)
            if freq_match and not sine_match and not saw_match:
                freq = float(freq_match.group(1))
                osc_type = 'sine' if freq_match.group(2) == 'sinosc' else 'saw'
                # Extract amplitude if present
                amp_match = re.search(r'([\d\.]+)\s*\*', line)
                amplitude = float(amp_match.group(1)) if amp_match else 0.3
                
                params['oscillators'].append({
                    'type': osc_type,
                    'frequency': freq,
                    'amplitude': amplitude
                })
            
            # Extract LFO modulation
            lfo_match = re.search(r'\.(\d+)\s+0\s+lfsaw.*?(\d+\.?\d*)\s+\*', line)
            if lfo_match:
                lfo_freq = float(f"0.{lfo_match.group(1)}")
                amount = float(lfo_match.group(2)) if lfo_match.group(2) else 0.2
                params['effects'].append({
                    'type': 'lfo',
                    'frequency': lfo_freq,
                    'amount': amount
                })
            
            # Extract filter information
            lpf_match = re.search(r'(\d+)\s+0\s+lpf', line)
            if lpf_match:
                cutoff = float(lpf_match.group(1))
                params['filters'].append({
                    'type': 'lowpass',
                    'cutoff': cutoff,
                    'resonance': 0.1
                })
            
            # Extract envelope information with model-specific parameters
            env_match = re.search(r'(\d+\.?\d*)\s+sec\s+0\s+1\s+([\d\.]+)\s+([\d\.]+)\s+env', line)
            if env_match:
                duration = float(env_match.group(1))
                sustain = float(env_match.group(2))
                release = float(env_match.group(3))
                
                # Adjust attack based on complexity
                attack = 0.05 if params['complexity'] == 'simple' else 0.15 if params['complexity'] == 'complex' else 0.1
                
                params['envelopes'].append({
                    'attack': attack,
                    'decay': 0.1,
                    'sustain': sustain,
                    'release': release,
                    'duration': duration
                })
        
        # Create oscillators based on complexity if none found
        if not params['oscillators']:
            if params['complexity'] == 'simple':
                params['oscillators'].append({
                    'type': 'sine',
                    'frequency': params['base_frequency'],
                    'amplitude': 0.4
                })
            elif params['complexity'] == 'complex':
                # Multiple oscillators for complex models
                for i in range(3):
                    params['oscillators'].append({
                        'type': 'sine' if i == 0 else 'saw',
                        'frequency': params['base_frequency'] + (i * 100),
                        'amplitude': 0.2 - (i * 0.05)
                    })
            else:  # moderate
                params['oscillators'].extend([
                    {
                        'type': 'sine',
                        'frequency': params['base_frequency'],
                        'amplitude': 0.3
                    },
                    {
                        'type': 'sine',
                        'frequency': params['base_frequency'] * 1.5,
                        'amplitude': 0.15
                    }
                ])
        
        # Create envelope based on complexity if none found
        if not params['envelopes']:
            if params['complexity'] == 'simple':
                params['envelopes'].append({
                    'attack': 0.05,
                    'decay': 0.1,
                    'sustain': 0.9,
                    'release': 0.1,
                    'duration': 6.0
                })
            elif params['complexity'] == 'complex':
                params['envelopes'].append({
                    'attack': 0.2,
                    'decay': 0.3,
                    'sustain': 0.6,
                    'release': 0.5,
                    'duration': 15.0
                })
            else:  # moderate
                params['envelopes'].append({
                    'attack': 0.1,
                    'decay': 0.1,
                    'sustain': 0.8,
                    'release': 0.2,
                    'duration': 10.0
                })
        
        return params
    
    def _generate_audio(self, params: Dict[str, Any], duration: float) -> List[int]:
        """
        Generate audio data based on parameters.
        
        Args:
            params: Audio generation parameters
            duration: Audio duration in seconds
            
        Returns:
            List of 16-bit audio samples
        """
        samples = int(self.sample_rate * duration)
        audio_data = np.zeros(samples, dtype=np.float32)
        
        # Generate oscillators
        for osc in params['oscillators']:
            osc_audio = self._generate_oscillator(
                osc['frequency'],
                osc['amplitude'],
                osc['type'],
                samples
            )
            audio_data += osc_audio
        
        # Apply LFO modulation
        for effect in params['effects']:
            if effect['type'] == 'lfo':
                lfo_audio = self._generate_lfo(
                    effect['frequency'],
                    effect['amount'],
                    samples
                )
                audio_data *= (1.0 + lfo_audio)
        
        # Apply filters
        for filt in params['filters']:
            if filt['type'] == 'lowpass':
                audio_data = self._apply_lowpass_filter(
                    audio_data,
                    filt['cutoff']
                )
        
        # Apply envelope
        if params['envelopes']:
            envelope = self._generate_envelope(
                params['envelopes'][0],
                samples
            )
            audio_data *= envelope
        
        # Normalize and convert to 16-bit integers
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_data_int = (audio_data * 32767).astype(np.int16)
        
        return audio_data_int.tolist()
    
    def _generate_oscillator(self, frequency: float, amplitude: float, 
                           osc_type: str, samples: int) -> np.ndarray:
        """Generate oscillator audio."""
        t = np.arange(samples) / self.sample_rate
        
        if osc_type == 'sine':
            return amplitude * np.sin(2 * np.pi * frequency * t)
        elif osc_type == 'saw':
            return amplitude * 2 * (t * frequency - np.floor(t * frequency + 0.5))
        elif osc_type == 'square':
            return amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
        else:
            # Default to sine
            return amplitude * np.sin(2 * np.pi * frequency * t)
    
    def _generate_lfo(self, frequency: float, amount: float, samples: int) -> np.ndarray:
        """Generate LFO (Low Frequency Oscillator) modulation."""
        t = np.arange(samples) / self.sample_rate
        return amount * np.sin(2 * np.pi * frequency * t)
    
    def _apply_lowpass_filter(self, audio: np.ndarray, cutoff: float) -> np.ndarray:
        """Apply simple lowpass filter (basic implementation)."""
        # Simple one-pole lowpass filter
        cutoff_normalized = cutoff / (self.sample_rate / 2)
        cutoff_normalized = np.clip(cutoff_normalized, 0.001, 0.999)
        
        alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff_normalized)
        filtered = np.zeros_like(audio)
        
        for i in range(1, len(audio)):
            filtered[i] = alpha * audio[i] + (1 - alpha) * filtered[i-1]
        
        return filtered
    
    def _generate_envelope(self, env_params: Dict[str, float], samples: int) -> np.ndarray:
        """Generate ADSR envelope."""
        attack = env_params['attack']
        decay = env_params['decay']
        sustain = env_params['sustain']
        release = env_params['release']
        
        envelope = np.zeros(samples)
        sample_rate = self.sample_rate
        
        # Attack phase
        attack_samples = int(attack * sample_rate)
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay phase
        decay_samples = int(decay * sample_rate)
        decay_end = attack_samples + decay_samples
        if decay_samples > 0 and decay_end < samples:
            envelope[attack_samples:decay_end] = np.linspace(1, sustain, decay_samples)
        
        # Sustain phase
        release_samples = int(release * sample_rate)
        sustain_end = max(samples - release_samples, decay_end)
        envelope[decay_end:sustain_end] = sustain
        
        # Release phase
        if release_samples > 0 and sustain_end < samples:
            envelope[sustain_end:] = np.linspace(sustain, 0, samples - sustain_end)
        
        return envelope
    
    def _write_wav_file(self, audio_data: List[int], output_file: Path) -> bool:
        """Write audio data to WAV file."""
        try:
            with wave.open(str(output_file), 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                
                # Convert to bytes
                audio_bytes = struct.pack(f'{len(audio_data)}h', *audio_data)
                wav_file.writeframes(audio_bytes)
            
            logger.debug(f"Audio file written: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write audio file: {e}")
            return False
    
    def _create_waveform_visualizations(self, audio_data: List[int], 
                                      output_file: Path, 
                                      audio_params: Dict[str, Any]) -> None:
        """
        Create comprehensive waveform and frequency visualizations.
        
        Args:
            audio_data: Generated audio data
            output_file: Base output file path for naming visualizations
            audio_params: Audio parameters for visualization context
        """
        try:
            # Convert audio data to normalized numpy array
            audio_array = np.array(audio_data, dtype=np.float32) / 32767.0
            time_axis = np.arange(len(audio_array)) / self.sample_rate
            
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Audio Analysis: {output_file.stem}', fontsize=16, fontweight='bold')
            
            # 1. Full waveform view
            ax1.plot(time_axis, audio_array, color='#2E86AB', linewidth=0.8, alpha=0.8)
            ax1.set_title('Waveform (Full Duration)', fontweight='bold')
            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel('Amplitude')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, time_axis[-1])
            
            # 2. Zoomed waveform view (first 0.1 seconds)
            zoom_samples = int(0.1 * self.sample_rate)
            zoom_end = min(zoom_samples, len(audio_array))
            ax2.plot(time_axis[:zoom_end], audio_array[:zoom_end], 
                    color='#A23B72', linewidth=1.2)
            ax2.set_title('Waveform Detail (First 0.1s)', fontweight='bold')
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Amplitude')
            ax2.grid(True, alpha=0.3)
            
            # 3. Frequency spectrum
            # Use only first second for FFT to avoid memory issues
            fft_samples = min(self.sample_rate, len(audio_array))
            frequencies = np.fft.rfftfreq(fft_samples, 1/self.sample_rate)
            fft_magnitude = np.abs(np.fft.rfft(audio_array[:fft_samples]))
            
            # Only plot positive frequencies up to Nyquist
            ax3.loglog(frequencies[1:], fft_magnitude[1:], color='#F18F01', linewidth=1.5)
            ax3.set_title('Frequency Spectrum', fontweight='bold')
            ax3.set_xlabel('Frequency (Hz)')
            ax3.set_ylabel('Magnitude')
            ax3.grid(True, alpha=0.3)
            
            # Mark expected frequencies from oscillators
            for osc in audio_params.get('oscillators', []):
                freq = osc.get('frequency', 0)
                if freq > 0 and freq < self.sample_rate/2:
                    ax3.axvline(freq, color='red', linestyle='--', alpha=0.7, linewidth=1)
            
            # 4. Spectrogram
            if len(audio_array) > 1024:  # Only if we have enough samples
                # Downsample for spectrogram if needed
                downsample_factor = max(1, len(audio_array) // 50000)
                spec_audio = audio_array[::downsample_factor]
                spec_sample_rate = self.sample_rate // downsample_factor

                nfft = min(1024, max(64, len(spec_audio)//4))
                noverlap = int(nfft * 0.5)

                Pxx, f_spec, t_spec, im = ax4.specgram(
                    spec_audio,
                    NFFT=nfft,
                    Fs=spec_sample_rate,
                    noverlap=noverlap,
                    cmap='plasma'
                )

                ax4.set_title('Spectrogram', fontweight='bold')
                ax4.set_xlabel('Time (seconds)')
                ax4.set_ylabel('Frequency (Hz)')
                ax4.set_ylim(0, min(2000, spec_sample_rate/2))  # Limit to 2kHz for clarity

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax4)
                cbar.set_label('Power (dB)', rotation=270, labelpad=15)
            else:
                ax4.text(0.5, 0.5, 'Insufficient data\\nfor spectrogram', 
                        ha='center', va='center', transform=ax4.transAxes,
                        fontsize=12, style='italic')
                ax4.set_title('Spectrogram (N/A)', fontweight='bold')
            
            # Add model information text
            info_text = f"Base Frequency: {audio_params.get('base_frequency', 'Unknown')} Hz\\n"
            info_text += f"Complexity: {audio_params.get('complexity', 'Unknown')}\\n"
            info_text += f"Oscillators: {len(audio_params.get('oscillators', []))}\\n"
            info_text += f"Sample Rate: {self.sample_rate} Hz\\n"
            info_text += f"Duration: {len(audio_array)/self.sample_rate:.2f}s"
            
            fig.text(0.02, 0.02, info_text, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.93, bottom=0.15)
            
            # Save visualization
            viz_file = output_file.parent / f"{output_file.stem}_waveform_analysis.png"
            plt.savefig(viz_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            logger.info(f"Created waveform visualization: {viz_file}")
            
        except Exception as e:
            logger.error(f"Failed to create waveform visualization: {e}")

def generate_oscillator_audio(frequency: float, amplitude: float, 
                             duration: float, sample_rate: int = 44100,
                             osc_type: str = 'sine') -> np.ndarray:
    """
    Generate oscillator audio.
    
    Args:
        frequency: Oscillator frequency in Hz
        amplitude: Amplitude (0.0 to 1.0)
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        osc_type: Oscillator type ('sine', 'saw', 'square')
        
    Returns:
        Audio samples as numpy array
    """
    samples = int(sample_rate * duration)
    t = np.arange(samples) / sample_rate
    
    if osc_type == 'sine':
        return amplitude * np.sin(2 * np.pi * frequency * t)
    elif osc_type == 'saw':
        return amplitude * 2 * (t * frequency - np.floor(t * frequency + 0.5))
    elif osc_type == 'square':
        return amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
    else:
        return amplitude * np.sin(2 * np.pi * frequency * t)

def apply_envelope(audio: np.ndarray, attack: float, decay: float, 
                   sustain: float, release: float, sample_rate: int = 44100) -> np.ndarray:
    """
    Apply ADSR envelope to audio.
    
    Args:
        audio: Input audio samples
        attack: Attack time in seconds
        decay: Decay time in seconds  
        sustain: Sustain level (0.0 to 1.0)
        release: Release time in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Audio with envelope applied
    """
    samples = len(audio)
    envelope = np.zeros(samples)
    
    # Attack phase
    attack_samples = int(attack * sample_rate)
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    
    # Decay phase
    decay_samples = int(decay * sample_rate)
    decay_end = attack_samples + decay_samples
    if decay_samples > 0 and decay_end < samples:
        envelope[attack_samples:decay_end] = np.linspace(1, sustain, decay_samples)
    
    # Sustain phase
    release_samples = int(release * sample_rate)
    sustain_end = max(samples - release_samples, decay_end)
    envelope[decay_end:sustain_end] = sustain
    
    # Release phase
    if release_samples > 0 and sustain_end < samples:
        envelope[sustain_end:] = np.linspace(sustain, 0, samples - sustain_end)
    
    return audio * envelope

def mix_audio_channels(audio_list: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
    """
    Mix multiple audio channels together.
    
    Args:
        audio_list: List of audio arrays to mix
        weights: Optional list of mixing weights
        
    Returns:
        Mixed audio
    """
    if not audio_list:
        return np.array([])
    
    if weights is None:
        weights = [1.0 / len(audio_list)] * len(audio_list)
    
    # Ensure all arrays are the same length
    max_length = max(len(audio) for audio in audio_list)
    padded_audio = []
    
    for audio in audio_list:
        if len(audio) < max_length:
            padded = np.zeros(max_length)
            padded[:len(audio)] = audio
            padded_audio.append(padded)
        else:
            padded_audio.append(audio[:max_length])
    
    # Mix with weights
    mixed = np.zeros(max_length)
    for audio, weight in zip(padded_audio, weights):
        mixed += audio * weight
    
    return mixed 