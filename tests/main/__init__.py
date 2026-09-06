"""Tests for the src-root entry surfaces owned by the srcroot fleet worker.

Covers the composable step-selection core in ``main.py``
(``parse_step_list_strict``, ``select_pipeline_steps``, ``StepSelection``,
``step_number_from_script_name``) and the additive manuscript-variable
round-trip API (``load_variables``, ``token_checksum``).
"""
