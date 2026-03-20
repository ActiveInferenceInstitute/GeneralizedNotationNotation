#!/usr/bin/env python3
"""
Tests for pipeline/context.py — PipelineContext and StepRecord.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestStepRecord:
    def test_to_dict_contains_required_fields(self):
        from pipeline.context import StepRecord
        rec = StepRecord(name="gnn_parse", step_num=3)
        d = rec.to_dict()
        assert d["name"] == "gnn_parse"
        assert d["step_num"] == 3
        assert d["status"] == "PENDING"
        assert isinstance(d["artifacts"], list)
        assert isinstance(d["errors"], list)

    def test_default_status_is_pending(self):
        from pipeline.context import StepRecord
        rec = StepRecord(name="step", step_num=0)
        assert rec.status == "PENDING"

    def test_custom_status_and_duration(self):
        from pipeline.context import StepRecord
        rec = StepRecord(name="step", step_num=1, status="SUCCESS", duration_seconds=1.23)
        d = rec.to_dict()
        assert d["status"] == "SUCCESS"
        assert d["duration_seconds"] == 1.23


class TestPipelineContext:
    def test_basic_construction(self):
        from pipeline.context import PipelineContext
        ctx = PipelineContext()
        assert ctx.output_dir == Path("output")
        assert ctx.target_dir == Path("input/gnn_files")

    def test_custom_output_dir(self):
        from pipeline.context import PipelineContext
        ctx = PipelineContext(output_dir=Path("/tmp/out"), target_dir=Path("/tmp/in"))
        assert ctx.output_dir == Path("/tmp/out")
        assert ctx.target_dir == Path("/tmp/in")

    def test_set_and_get(self):
        from pipeline.context import PipelineContext
        ctx = PipelineContext()
        ctx.set("key", "value")
        assert ctx.get("key") == "value"

    def test_get_missing_key_returns_default(self):
        from pipeline.context import PipelineContext
        ctx = PipelineContext()
        assert ctx.get("missing") is None
        assert ctx.get("missing", 42) == 42

    def test_record_step_stores_result(self):
        from pipeline.context import PipelineContext
        ctx = PipelineContext()
        ctx.record_step("gnn_parse", step_num=3, status="SUCCESS", duration=1.5)
        summary = ctx.summary()
        steps = {s["name"]: s for s in summary["steps"]}
        assert "gnn_parse" in steps
        assert steps["gnn_parse"]["status"] == "SUCCESS"
        assert steps["gnn_parse"]["duration_seconds"] == 1.5

    def test_summary_success_flag_all_success(self):
        from pipeline.context import PipelineContext
        ctx = PipelineContext()
        ctx.record_step("step_a", step_num=1, status="SUCCESS")
        ctx.record_step("step_b", step_num=2, status="SKIPPED")
        assert ctx.summary()["success"] is True

    def test_summary_success_flag_with_failure(self):
        from pipeline.context import PipelineContext
        ctx = PipelineContext()
        ctx.record_step("step_a", step_num=1, status="SUCCESS")
        ctx.record_step("step_b", step_num=2, status="FAILED")
        assert ctx.summary()["success"] is False

    def test_step_order_preserved(self):
        from pipeline.context import PipelineContext
        ctx = PipelineContext()
        ctx.record_step("first", step_num=1, status="SUCCESS")
        ctx.record_step("second", step_num=2, status="SUCCESS")
        ctx.record_step("third", step_num=3, status="SUCCESS")
        names = [s["name"] for s in ctx.summary()["steps"]]
        assert names == ["first", "second", "third"]

    def test_timings_property(self):
        from pipeline.context import PipelineContext
        ctx = PipelineContext()
        ctx.record_step("step", step_num=0, status="SUCCESS", duration=3.7)
        assert "step" in ctx.timings
        assert ctx.timings["step"] == 3.7

    def test_save_summary_writes_json(self, tmp_path):
        from pipeline.context import PipelineContext
        ctx = PipelineContext(output_dir=tmp_path)
        ctx.record_step("parse", step_num=3, status="SUCCESS")
        out_path = ctx.save_summary()
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert "steps" in data
        assert data["success"] is True

    def test_errors_aggregated_in_summary(self):
        from pipeline.context import PipelineContext
        ctx = PipelineContext()
        ctx.record_step("step", step_num=0, status="FAILED", errors=["something broke"])
        summary = ctx.summary()
        assert "something broke" in summary["errors"]

    def test_repr_is_informative(self):
        from pipeline.context import PipelineContext
        ctx = PipelineContext()
        ctx.record_step("step", step_num=0, status="SUCCESS")
        r = repr(ctx)
        assert "PipelineContext" in r
        assert "steps=1" in r

    def test_on_step_start_callback_fires(self):
        from pipeline.context import PipelineContext
        ctx = PipelineContext()
        calls = []
        ctx.on_step_start = lambda name, num: calls.append((name, num))
        ctx.trigger_step_start("gnn_parse", 3)
        assert ("gnn_parse", 3) in calls

    def test_on_step_complete_callback_fires(self):
        from pipeline.context import PipelineContext
        ctx = PipelineContext()
        calls = []
        ctx.on_step_complete = lambda name, num, status, dur: calls.append(status)
        ctx.record_step("step", step_num=0, status="SUCCESS", duration=1.0)
        assert "SUCCESS" in calls
