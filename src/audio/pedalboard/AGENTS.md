# Pedalboard Audio Submodule - Agent Scaffolding

## Module Overview

**Purpose**: Planned audio effects post-processing for GNN sonification output using Spotify's Pedalboard library

**Parent Module**: Audio Module (Step 15: Audio processing)

**Category**: Audio Framework / Effects (planned)

**Status**: Documentation-only scaffold. This directory holds `AGENTS.md`, `README.md`, and `SPEC.md` and no Python code. Nothing in the pipeline imports `audio.pedalboard`; the only live reference to Pedalboard is the availability probe in `audio.check_audio_backends()`.

---

## Planned Design

### Intended Responsibilities
1. Apply effects chains (reverb, delay, compression, EQ) to WAV files produced by `audio/` and `audio/sapf/`
2. Configure effect chains per model type
3. Record the applied chain as JSON metadata next to the processed audio

### Intended Inputs
- WAV files from `output/15_audio_output/` (`{model}_tonal.wav`, `{model}_sapf_audio.wav`, ...)

### Intended Outputs
- Processed WAV files with effects applied
- Effects-chain metadata (JSON)

None of these are produced today; see [README.md](README.md) for the current status.

---

## Dependencies

- `pedalboard >= 0.7.0` (optional `audio` extra in `pyproject.toml`); its presence is reported by `audio.check_audio_backends()`
- `numpy` and `soundfile` for the eventual read/write path

---

## Integration Points

- **Parent Module**: `src/audio/` (Step 15)
- **Sibling**: `src/audio/sapf/` produces the WAV files this module is meant to post-process
- **Tests**: none exist for this directory; audio tests live in `src/tests/audio/`

---

## Development Guidelines

When implementing this module:
1. Add `__init__.py`, an `effects.py` with chain construction, and a processor entry point that mirrors `audio.processor.process_audio` conventions (accept `target_dir`, `output_dir`, `verbose`, `**kwargs`; return `bool`)
2. Keep `pedalboard` an optional import with a clear skip path when it is missing
3. Add tests under `src/tests/audio/` and update `README.md`, `SPEC.md`, and this file in the same change

---

## References

- [Audio Module](../AGENTS.md) - Parent audio module
- [Pedalboard Documentation](https://spotify.github.io/pedalboard/) - Official Pedalboard docs
- [Pipeline Overview](../../../README.md) - Main pipeline documentation

---

**Last Updated**: 2026-09-02
**Maintainer**: Audio Processing Team
**Status**: Planned (documentation-only scaffold)
