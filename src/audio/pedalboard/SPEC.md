# Pedalboard Audio Processing — Technical Specification

**Version**: 1.6.0

## Purpose

Audio effects processing using Spotify's Pedalboard library for post-processing generated audio.

## Effects Pipeline

- Reverb, delay, compression, EQ
- Configurable effect chains per model type

## Input

- WAV audio files from SAPF generation

## Output

- Processed WAV files with applied effects
- Effects metadata (JSON)

## Dependencies

- `pedalboard >= 0.7.0` (optional, graceful skip)
