# Test suite guide

The suite exercises public GNN behavior, scientific contracts, native adapters,
installation boundaries and repository tooling. Test outcomes belong to the
exact source, environment and selection that produced them.

## Choose the owning profile

| Task | Owner |
| --- | --- |
| Ordinary Python matrix, quality and pipeline contracts | [CI workflow](../.github/workflows/ci.yml) |
| Per-interpreter comprehensive native coverage | [Native coverage guide](../docs/development/native_comprehensive_coverage.md) |
| Installed wheel and container behavior | [Installed-platform guide](../docs/development/installed_platform_acceptance.md) |
| Optional-package acceptance | [All-extras workflow](../.github/workflows/full-extras.yml) |
| Julia/RxInfer model tests | [RxInfer workflow](../.github/workflows/rxinfer.yml) |
| Documentation and repository gates | [Workflow guide](../.github/workflows/README.md) |
| Local selections and release verification | [Verification commands](../TO-DO.md#verification-and-execution-rules) |

The workflows define their markers, dependencies, toolchains, timeouts and
coverage floors. Preserve each profile's selection and denominator. Record
optional-runtime skips and unsupported scientific cases with their actual
reasons; passing a focused selection does not qualify a broader environment.

## Run a focused check

After installing the locked development profile:

```bash
uv sync --frozen --extra dev --python 3.14
uv run --frozen --no-sync python -m pytest tests/docs -q --no-cov
```

Choose the real test directory for the changed module and follow
[AGENTS.md](AGENTS.md) and the [testing rules](../.agent_rules/testing.md).
Public imports use `gnn.*`; tests should exercise concrete inputs, failures and
consumer contracts. [pytest.ini](../pytest.ini) owns registered markers and
[pyproject.toml](../pyproject.toml) owns development dependencies.

## Report results accurately

Record the full commit, Python/native package versions, command, selected tests,
terminal exit status and retained logs or receipts. Keep ordinary CI, focused
native tests, installed-package checks and comprehensive coverage separate;
overlapping selections are not additive. Link current hosted results from the
[Actions page](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/actions)
and historical qualification from the appropriate [release](../CHANGELOG.md).

Changes that affect manuscript counts or scientific owners follow the
[custody procedure](../.agent_rules/release.md). Review the
[handoff checklist](../docs/development/contributor_handoff.md) before integration.
