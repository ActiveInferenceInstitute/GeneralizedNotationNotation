# Logging Utilities Specification

## Overview

`utils.logging` provides shared logging configuration utilities used throughout the pipeline.

## Scope

- standardized logger setup for step scripts and modules
- consistent formatting and handler configuration (console + file)

## Public API

The public API is defined by `src/utils/logging/__init__.py` and `logging_utils.py`.

## Non-goals

- implementing a custom logging framework
- embedding runtime metrics that are not directly observed by the pipeline

