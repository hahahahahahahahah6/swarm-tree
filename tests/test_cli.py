"""Tests for CLI helpers: summary-mode resolution, error counting, exit codes."""

import json
import os
import tempfile
import unittest
from unittest import mock

from swarm_tree import cli, config


def _node(summary_mode):
    d = {
        "name": "root",
        "provider": "anthropic",
        "model": "m",
        "role": "r",
    }
    if summary_mode is not None:
        d["summary_mode"] = summary_mode
    return config.validate(d)


class TestEffectiveSummaryMode(unittest.TestCase):
    def test_default_on(self):
        self.assertTrue(cli._effective_summary_mode(_node(None), False))

    def test_config_false_stays_off(self):
        self.assertFalse(cli._effective_summary_mode(_node(False), False))

    def test_config_true(self):
        self.assertTrue(cli._effective_summary_mode(_node(True), False))

    def test_flag_wins_over_config_true(self):
        # 2c: --full-context disables summaries even when config says true
        self.assertFalse(cli._effective_summary_mode(_node(True), True))

    def test_flag_with_config_false_stays_off(self):
        self.assertFalse(cli._effective_summary_mode(_node(False), True))

    def test_flag_with_default_off(self):
        self.assertFalse(cli._effective_summary_mode(_node(None), True))


class TestCountErrors(unittest.TestCase):
    def test_no_errors(self):
        rec = {"status": "ok", "children": [
            {"status": "ok", "children": []},
        ]}
        self.assertEqual(cli._count_errors(rec), 0)

    def test_nested_errors_counted(self):
        rec = {"status": "ok", "children": [
            {"status": "error", "children": []},
            {"status": "ok", "children": [
                {"status": "error", "children": []},
            ]},
        ]}
        self.assertEqual(cli._count_errors(rec), 2)

    def test_root_error_counted(self):
        self.assertEqual(
            cli._count_errors({"status": "error", "children": []}), 1)


def _record(status, error=None, children=()):
    return {
        "name": "root",
        "provider": "openai",
        "model": "m",
        "status": status,
        "duration_s": 0.1,
        "response": "answer" if status == "ok" else None,
        "error": error,
        "children": list(children),
        "tokens": {"input": 1, "output": 1, "cost_usd": 0.0},
    }


class TestExitCodes(unittest.TestCase):
    """0 clean, 3 degraded, 1 hard failure, 2 config/usage error."""

    def _run_main(self, record):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as fh:
            json.dump({"name": "root", "provider": "openai",
                       "model": "m", "role": "r"}, fh)
        self.addCleanup(os.unlink, path)
        out = path + ".transcript.json"
        self.addCleanup(lambda: os.path.exists(out) and os.unlink(out))
        with mock.patch.object(cli.engine, "execute_tree",
                               return_value=record):
            code = cli.main(["run", "--tree", path, "--task", "t",
                             "--dry-run", "--output", out])
        return code, out

    def test_exit_0_clean(self):
        code, _ = self._run_main(_record("ok"))
        self.assertEqual(code, 0)

    def test_exit_3_degraded(self):
        child = {"name": "c1", "provider": "openai", "model": "m",
                 "status": "error", "duration_s": 0.1, "response": None,
                 "error": "boom", "children": [],
                 "tokens": {"input": 1, "output": 1, "cost_usd": 0.0}}
        code, _ = self._run_main(_record("ok", children=[child]))
        self.assertEqual(code, 3)

    def test_exit_1_hard_failure(self):
        code, out = self._run_main(_record("error", error="boom"))
        self.assertEqual(code, 1)
        # the transcript is still dumped on hard failure
        self.assertTrue(os.path.exists(out))

    def test_cancelled_reraises_keyboard_interrupt(self):
        with self.assertRaises(KeyboardInterrupt):
            self._run_main(_record("cancelled", error="cancelled by user"))


if __name__ == "__main__":
    unittest.main()
