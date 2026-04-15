# Core Skill: `output_fixtures`

**Function**: Test fixture storage for pipeline output artifacts. Not a runtime module.

## Scope

This directory holds reference copies of pipeline output structures used for test validation. It does not export any Python functionality.

## Contents

| Directory | Purpose |
|-----------|---------|
| `2_tests_output/` | Test execution fixture data |
| `12_execute_output/` | Execution result fixture data |
| `16_analysis_output/` | Analysis result fixture data |

## Policy

- Primary pipeline output target is `output/` at repository root
- Contents may be overwritten by test runs
- Not subject to standard module SKILL documentation expectations
