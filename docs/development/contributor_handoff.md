# Contributor handoff

Use this checklist to transfer a specific change or investigation. Record current
observations instead of copying old repository-wide pass counts.

## Establish the baseline

Record the repository, checkout path, branch, full commit, dirty paths and active
work. Identify any task-owned environments, output directories and running jobs.
Preserve other contributors' changes and keep normal producer ancestry.

Read [AGENTS.md](../../AGENTS.md), the closest module guide and the
[workflow rules](../../.agent_rules/workflow.md). The
[roadmap](../../TO-DO.md) owns remaining work; the
[changelog](../../CHANGELOG.md) owns completed release history.

## Describe the change and its evidence

- State the concrete problem, resulting behavior and supported public entrypoint.
- Name changed files, contracts, examples and documentation.
- Record exact commands, environments, source commits and terminal outcomes.
- Link test reports and artifacts, including failed attempts and known limits.
- Distinguish local tests, hosted checks, native numerical results, installation
  checks and manuscript or companion custody. Keep overlapping selections separate.

Choose checks from the [verification commands](../../TO-DO.md#verification-and-execution-rules)
and [workflow guide](../../.github/workflows/README.md). A pending check remains
pending. A successful command applies only to its recorded selection and source.

## Integrate and close out

Inspect the final diff and reconcile upstream without overwriting other work.
Determine whether manuscript inputs or companion-owned files changed; follow
the [release/custody procedure](../../.agent_rules/release.md) when they do.
After authorized publication, verify remote source parity and required terminal
checks. Report what remains, who owns it and the evidence needed to finish.
