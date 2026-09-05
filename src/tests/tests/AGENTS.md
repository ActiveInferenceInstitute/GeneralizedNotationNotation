# Test Infrastructure Tests

This directory tests the test runner, category definitions, helper contracts,
infrastructure exports and Step 2 wrapper. Keep tests aligned with the public
`tests` package and pipeline wrapper behavior. Production code belongs in its
own module.

Run `uv run --extra dev python -m pytest src/tests/tests/ -q`.
