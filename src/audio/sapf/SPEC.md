# SAPF Audio Processing — Technical Specification

**Version**: 1.6.0

## Purpose

Sonification of Active Inference model dynamics using the Sonification of Active Inference Processes Framework (SAPF).

## Processing Pipeline

1. Parse GNN model structure
2. Map model variables to audio parameters
3. Generate audio waveforms (sine, noise, AM/FM synthesis)
4. Apply temporal dynamics from simulation results
5. Mix and export WAV files

## Input

- Parsed GNN models from Step 3
- Optional: simulation results from Step 12

## Output

- WAV audio files (44.1kHz, 16-bit)
- Sonification metadata (JSON)
- Spectrogram visualizations (PNG)

## Dependencies

- `soundfile`, `numpy` (required)
- `scipy` (for signal processing)
