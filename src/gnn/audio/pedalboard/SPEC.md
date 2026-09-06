# Pedalboard Audio Processing — Technical Specification

**Version**: 3.2.0

**Status**: Planned. This directory is a documentation-only scaffold with no Python code (see [README.md](README.md)).

## Purpose

Planned audio effects processing using Spotify's Pedalboard library for post-processing generated audio.

## Effects Pipeline (planned)

- Reverb, delay, compression, EQ
- Configurable effect chains per model type

## Input (planned)

- WAV audio files from `audio/` and `audio/sapf/` generation

## Output (planned)

- Processed WAV files with applied effects
- Effects metadata (JSON)

No code path produces these artifacts today.

## Dependencies

- `pedalboard >= 0.7.0` (optional `audio` extra; availability reported by `audio.check_audio_backends()`)
