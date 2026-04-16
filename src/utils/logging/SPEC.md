# Logging Utilities — Technical Specification

**Version**: 1.6.0

## Logging Architecture

- **Handler hierarchy**: Console (colored) → File (plain text) → Structured (JSON)
- **Log levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL (standard Python)
- **Correlation IDs**: UUID-based tracking across pipeline steps

## Visual Output

- Step progress: `[3/25] ⚙️ GNN Processing...`
- Success: green checkmarks with timing
- Warnings: yellow with context
- Errors: red with traceback

## Configuration

- Verbose mode: `--verbose` flag enables DEBUG level
- Structured mode: `--structured-logging` outputs JSON lines
- Accessible mode: `--accessible` disables emojis for screen readers
