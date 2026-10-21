"""Fail-closed release scheduling controls (stdlib only)."""

import copy
import unittest

from release_gate import evaluate


class ReleaseGateControls(unittest.TestCase):
    def setUp(self):
        self.sha = "a" * 40
        self.run = dict(
            run_number=8,
            run_attempt=1,
            head_sha=self.sha,
            status="completed",
            conclusion="success",
        )
        self.jobs = [dict(status="completed", conclusion="success")]

    def state(self, runs=None, jobs=None):
        return evaluate(
            [self.run] if runs is None else runs,
            self.jobs if jobs is None else jobs,
            self.sha,
        )["state"]

    def test_complete(self):
        self.assertEqual(self.state(), "success")

    def test_missing(self):
        self.assertEqual(self.state(runs=[]), "missing")
        self.assertEqual(self.state(jobs=[]), "incomplete_jobs")

    def test_latest_failure_cannot_borrow_prior_success(self):
        failed = dict(self.run, run_number=9, conclusion="failure")
        self.assertEqual(self.state(runs=[failed, self.run]), "failed")

    def test_rerun_attempt_cannot_borrow_prior_success(self):
        pending = dict(self.run, run_attempt=2, status="in_progress", conclusion=None)
        self.assertEqual(self.state(runs=[self.run, pending]), "pending")

    def test_wrong_source(self):
        self.run["head_sha"] = "b" * 40
        self.assertEqual(self.state(), "wrong_source")

    def test_every_job_must_finish_successfully(self):
        for conclusion in ["skipped", "cancelled", "failure", None]:
            jobs = copy.deepcopy(self.jobs)
            jobs.append(dict(status="completed", conclusion=conclusion))
            self.assertEqual(self.state(jobs=jobs), "incomplete_jobs")


if __name__ == "__main__":
    unittest.main()
