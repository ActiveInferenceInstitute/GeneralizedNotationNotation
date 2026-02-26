# Step 20: Website — Static Website Generation

## Overview

Generates a static HTML summary website from pipeline outputs, aggregating visualizations, analysis results, and reports into a browsable format.

## Usage

```bash
python src/20_website.py --target-dir input/gnn_files --output-dir output --verbose
python src/20_website.py --website-html-filename custom_report.html --verbose
```

## Architecture

| Component | Path |
|-----------|------|
| Orchestrator | `src/20_website.py` (66 lines) |
| Module | `src/website/` |
| Module function | `process_website()` |

## CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--website-html-filename` | `str` | `gnn_pipeline_summary_website.html` | Output HTML filename |

## Output

- **Directory**: `output/20_website_output/`
- HTML website, embedded visualizations, and summary dashboard

## Source

- **Script**: [src/20_website.py](#placeholder)
