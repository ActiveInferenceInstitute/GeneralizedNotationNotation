# SAPF Audio Processing — Technical Specification

**Version**: 3.2.0

## Purpose

Sonification of GNN model structure by generating SAPF (Sound As Pure Form) code from the parsed model and synthesizing that code to audio in Python.

## Processing Pipeline

1. Parse GNN model sections (`SAPFGNNProcessor.parse_gnn_sections`)
2. Map state space, connections, parameters, and time configuration to SAPF oscillators, routing, and processing chains (`convert_to_sapf`)
3. Optionally validate the SAPF code (`validate_sapf_code`)
4. Synthesize audio from the SAPF code (`SyntheticAudioGenerator.generate_from_sapf`)
5. Export a WAV file and a waveform/spectrum analysis PNG

## Input

- GNN model content (Markdown) and a model name

## Output

- `{model}_sapf_audio.wav` (44.1kHz, 16-bit mono, written with the stdlib `wave` module)
- `{model}_sapf_audio_waveform_analysis.png` (waveform, detail, spectrum, and spectrogram panels) when `create_visualization` is left enabled
- Optional JSON from `create_sapf_visualization` (parsed components) and `generate_sapf_report` (results summary) when an output path is supplied

Step 12 execution telemetry is consumed by `audio/processor.py` streaming, not by this sub-package.

## Dependencies

- `numpy`, `matplotlib` (imported unconditionally by `audio_generators.py`)
