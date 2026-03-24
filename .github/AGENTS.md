# .github Agent Guide

## Purpose

Repository automation: Dependabot and GitHub Actions. **Start here for humans**: [README.md](README.md) (full file index, triggers, local commands). Workflow-focused summary: [workflows/README.md](workflows/README.md).

## Files

- [dependabot.yml](dependabot.yml): Dependabot for pip and GitHub Actions (weekly Monday UTC).
- [workflows/](workflows/): CI (test / lint / security), docs audit, actionlint, dependency review, CodeQL, scheduled supply-chain `pip-audit`.

## Operating rules

- Keep permissions least-privilege at workflow and job level.
- Keep dependency operations deterministic (`--frozen` where lockfile behavior matters).
- Validate workflow changes with actionlint before merge (see [workflows/actionlint.yml](workflows/actionlint.yml)).
- Keep `output/` tracked in git; do not use `.github` automation to alter that policy.

## Maintenance checklist

- Review workflow action major versions periodically.
- Review Dependabot grouping and cadence when PR volume changes.
- Ensure new workflows include `concurrency`, `permissions`, and `timeout-minutes`.
