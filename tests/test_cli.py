"""Tests for CLI helpers: summary-mode resolution and error counting."""

import unittest

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


if __name__ == "__main__":
    unittest.main()
