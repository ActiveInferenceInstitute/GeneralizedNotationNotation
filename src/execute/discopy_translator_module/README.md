# DisCoPy Translator Module

Translates between DisCoPy categorical representations and JAX tensor computations.

## Usage

```python
from execute.discopy_translator_module import translate_and_visualize

translate_and_visualize(diagram_path, jax_output, output_dir)
```

## Components

- `translator.py` — DisCoPy ↔ JAX translation (234 lines)
- `visualize_jax_output.py` — JAX result visualization (270 lines)

## See Also

- [Parent: execute/README.md](../README.md)
- [AGENTS.md](AGENTS.md) — Architecture documentation
