# Tests Infrastructure Specification

## Overview

This folder provides a small support library for consistent, observable test execution.

## Public API

The authoritative export surface is `src/tests/infrastructure/__init__.py`.

Consumers should prefer importing from `tests.infrastructure` rather than individual files.

## Non-goals

- defining test cases (lives in `src/tests/test_*.py`)
- replacing pytest

