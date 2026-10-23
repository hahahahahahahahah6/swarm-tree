"""Tests for serve mode: schedule parsing, due-node selection, state,
tick execution (dry-run), rollup briefings, and the new CLI commands."""

import datetime
import json
import os
import tempfile
import time
import unittest
from unittest import mock

from swarm_tree import cli, config, engine, serve


def _tree(extra_root=None, nodes=()):
    """Build a validated tree dict: root + child nodes (each a dict of fields)."""
    data = {
        "name": "ceo",
        "provider": "anthropic",
        "model": "m",
        "role": "ceo",
        "children": [
            {
                "name": n["name"],
                "provider": "deepseek",
                "model": "m",
                "role": "r",
                **{k: v for k, v in n.items() if k != "name"},
            }
            for n in nodes
        ],
    }
    if extra_root:
        data.update(extra_root)
    return config.validate(data)


class TestParseSchedule(unittest.TestCase):
    def test_intervals(self):
        self.assertEqual(config.parse_schedule("30s"), ("interval", 30))
        self.assertEqual(config.parse_schedule("5m"), ("interval", 300))
        self.assertEqual(config.parse_schedule("2h"), ("interval", 7200))
        self.assertEqual(config.parse_schedule("1d"), ("interval", 86400))
        self.assertEqual(config.parse_schedule("  10m "), ("interval", 600))

    def test_daily(self):
        self.assertEqual(config.parse_schedule("08:05"), ("daily", (8, 5)))
        self.assertEqual(config.parse_schedule("23:59"), ("daily", (23, 59)))
        self.assertEqual(config.parse_schedule("00:00"), ("daily", (0, 0)))

    def test_invalid(self):
        for bad in ("5x", "m", "", "  ", "25:00", "7:00", "08:60",
                    "-5m", "0m", "1.5h", "noon"):
            with self.assertRaises(config.TreeError, msg=bad):
                config.parse_schedule(bad)
        with self.assertRaises(config.TreeError):
            config.parse_schedule(5)


class TestScheduleConfig(unittest.TestCase):
    def test_node_knobs_accepted(self):
        node = _tree(nodes=[{"name": "a", "schedule": "30m",
                             "task": "check things", "enabled": False}])
        child = node["children"][0]
        self.assertEqual(child["schedule"], "30m")
        self.assertEqual(child["task"], "check things")
        self.assertFalse(child["enabled"])

    def test_node_knobs_default(self):
        node = _tree()
        self.assertIsNone(node["schedule"])
        self.assertIsNone(node["task"])
        self.assertTrue(node["enabled"])
        self.assertEqual(node["serve_dir"], "serve-data")
        self.assertTrue(node["rollup"])

    def test_bad_schedule_rejected(self):
        with self.assertRaises(config.TreeError):
            _tree(nodes=[{"name": "a", "schedule": "whenever"}])

    def test_empty_task_rejected(self):
        with self.assertRaises(config.TreeError):
            _tree(nodes=[{"name": "a", "schedule": "5m", "task": "   "}])

    def test_root_knobs(self):
        node = _tree(extra_root={"serve_dir": "./data", "rollup": False})
        self.assertEqual(node["serve_dir"], "./data")
        self.assertFalse(node["rollup"])


class TestNextRun(unittest.TestCase):
    def test_first_run_fires_now(self):
        now = time.time()
        self.assertEqual(serve.next_run_after("30m", None, now), now)

    def test_interval_from_last(self):
        now = 1_000_000.0
        self.assertEqual(serve.next_run_after("5m", now - 60, now), now + 240)

    def test_missed_interval_skips_backfill(self):
        # last run 10h ago on a 5m schedule: don't fire now, wait one interval
        now = 1_000_000.0
        self.assertEqual(serve.next_run_after("5m", now - 36000, now), now + 300)

    def test_daily_future_today(self):
        now = time.time()
        local = datetime.datetime.fromtimestamp(now).astimezone()
        future = (local + datetime.timedelta(hours=2)).strftime("%H:%M")
        nxt = serve.next_run_after(future, now - 86400, now)
        got = datetime.datetime.fromtimestamp(nxt).astimezone()
        self.assertEqual((got.hour, got.minute),
                         tuple(int(x) for x in future.split(":")))
        self.assertLessEqual(nxt - now, 2 * 3600 + 60)

    def test_daily_past_goes_tomorrow(self):
        now = time.time()
        local = datetime.datetime.fromtimestamp(now).astimezone()
        past = (local - datetime.timedelta(hours=2)).strftime("%H:%M")
        nxt = serve.next_run_after(past, now - 86400, now)
        self.assertGreater(nxt - now, 20 * 3600)  # roughly a day out


class TestDueNodes(unittest.TestCase):
    def test_selection(self):
        node = _tree(nodes=[
            {"name": "a", "schedule": "5m"},
            {"name": "b", "schedule": "5m", "enabled": False},
            {"name": "c"},  # no schedule
        ])
        now = time.time()
        due = serve.due_nodes(node, {}, now)
        self.assertEqual([n for n, _ in due], ["a"])  # first run: only enabled

    def test_not_due_yet(self):
        node = _tree(nodes=[{"name": "a", "schedule": "1h"}])
        now = time.time()
        state = {"a": {"last_run": now - 60}}
        self.assertEqual(serve.due_nodes(node, state, now), [])

    def test_due_after_interval(self):
        node = _tree(nodes=[{"name": "a", "schedule": "5m"}])
        now = time.time()
        state = {"a": {"last_run": now - 600}}
        due = serve.due_nodes(node, state, now)
        self.assertEqual([n for n, _ in due], ["a"])


class TestServeTick(unittest.TestCase):
    def _run_once(self, tmp, **kw):
        node = _tree(nodes=[
            {"name": "shop", "schedule": "5m", "task": "check shop metrics"},
            {"name": "quiet"},  # no schedule: never runs alone
        ])
        serve_dir = os.path.join(tmp, "data")
        results = serve.run_tick(
            node, engine.dry_run_provider, serve_dir=serve_dir,
            dry_run=True, version="0.3.0-test", now=time.time(), **kw)
        return node, serve_dir, results

    def test_tick_runs_due_and_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            node, serve_dir, results = self._run_once(tmp)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["name"], "shop")
            self.assertEqual(results[0]["status"], "ok")
            # transcript persisted
            runs = os.listdir(os.path.join(serve_dir, "runs", "shop"))
            self.assertEqual(len(runs), 1)
            self.assertTrue(runs[0].endswith(".json"))
            with open(os.path.join(serve_dir, "runs", "shop", runs[0])) as fh:
                data = json.load(fh)
            self.assertEqual(data["mode"], "serve")
            self.assertEqual(data["node"], "shop")
            # state persisted with a future next_run
            state = serve.load_state(serve_dir)
            self.assertIn("shop", state)
            self.assertGreater(state["shop"]["next_run"], time.time())
            self.assertNotIn("quiet", state)

    def test_second_tick_not_due(self):
        with tempfile.TemporaryDirectory() as tmp:
            node, serve_dir, first = self._run_once(tmp)
            second = serve.run_tick(
                node, engine.dry_run_provider, serve_dir=serve_dir,
                dry_run=True, version="0.3.0-test", now=time.time())
            self.assertEqual(second, [])

    def test_rollup_writes_briefing(self):
        with tempfile.TemporaryDirectory() as tmp:
            node, serve_dir, results = self._run_once(tmp)
            path = serve.write_rollup(
                node, results, engine.dry_run_provider, serve_dir=serve_dir,
                dry_run=True, version="0.3.0-test")
            self.assertIsNotNone(path)
            with open(path) as fh:
                body = fh.read()
            self.assertIn("shop", body)
            self.assertIn("Briefing", body)

    def test_rollup_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            node = _tree(extra_root={"rollup": False},
                          nodes=[{"name": "shop", "schedule": "5m"}])
            serve_dir = os.path.join(tmp, "data")
            results = serve.run_tick(
                node, engine.dry_run_provider, serve_dir=serve_dir,
                dry_run=True, version="0.3.0-test", now=time.time())
            self.assertIsNone(serve.write_rollup(
                node, results, engine.dry_run_provider, serve_dir=serve_dir,
                dry_run=True, version="0.3.0-test"))

    def test_state_survives_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            node, serve_dir, _ = self._run_once(tmp)
            reloaded = serve.load_state(serve_dir)
            # schedule resumes: next tick is in the future, not immediate
            due = serve.due_nodes(node, reloaded, time.time())
            self.assertEqual(due, [])

    def test_daily_due_after_slot(self):
        node = _tree(nodes=[{"name": "a", "schedule": "08:05"}])
        local = datetime.datetime.now().astimezone()
        # fixed "now": today 12:00 local
        now = local.replace(hour=12, minute=0, second=0, microsecond=0).timestamp()
        # last run yesterday 08:06 -> today's 08:05 slot passed -> due
        last = (local.replace(hour=8, minute=6, second=0, microsecond=0)
                - datetime.timedelta(days=1)).timestamp()
        due = serve.due_nodes(node, {"a": {"last_run": last}}, now)
        self.assertEqual([n for n, _ in due], ["a"])
        # last run today 08:06 -> next slot is tomorrow -> not due
        last2 = local.replace(hour=8, minute=6, second=0,
                              microsecond=0).timestamp()
        if last2 <= now:  # only valid when "now" really is after 08:06
            due2 = serve.due_nodes(node, {"a": {"last_run": last2}}, now)
            self.assertEqual(due2, [])

    def test_serve_loop_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            node = _tree(nodes=[{"name": "shop", "schedule": "5m"}])
            serve_dir = os.path.join(tmp, "data")
            seen = []
            rc = serve.serve_loop(
                node, engine.dry_run_provider, serve_dir=serve_dir,
                dry_run=True, version="0.3.0-test", once=True,
                on_tick=seen.append)
            self.assertEqual(rc, 0)
            self.assertEqual(len(seen), 1)
            self.assertEqual(len(seen[0]), 1)

    def test_serve_loop_no_schedules(self):
        node = _tree()
        with tempfile.TemporaryDirectory() as tmp:
            rc = serve.serve_loop(node, engine.dry_run_provider,
                                  serve_dir=tmp, dry_run=True,
                                  version="0.3.0-test", once=True)
            self.assertEqual(rc, 2)


class TestServeStatusReport(unittest.TestCase):
    def test_status_empty_state(self):
        node = _tree(nodes=[{"name": "a", "schedule": "5m"}])
        with tempfile.TemporaryDirectory() as tmp:
            out = serve.format_status(node, os.path.join(tmp, "d"))
            self.assertIn("a", out)
            self.assertIn("5m", out)

    def test_status_no_schedules(self):
        node = _tree()
        with tempfile.TemporaryDirectory() as tmp:
            out = serve.format_status(node, tmp)
            self.assertIn("no scheduled nodes", out)

    def test_read_report_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn("no briefings", serve.read_report(tmp))

    def test_read_report_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            node = _tree(nodes=[{"name": "shop", "schedule": "5m"}])
            serve_dir = os.path.join(tmp, "d")
            serve.run_tick(node, engine.dry_run_provider, serve_dir=serve_dir,
                           dry_run=True, version="0.3.0-test",
                           now=time.time())
            out = serve.read_report(serve_dir, "shop")
            self.assertIn("shop", out)
            self.assertIn("no runs recorded", serve.read_report(serve_dir, "nope"))


class TestServeCli(unittest.TestCase):
    def _write_tree(self, tmp, data):
        path = os.path.join(tmp, "tree.json")
        with open(path, "w") as fh:
            json.dump(data, fh)
        return path

    def _tree_data(self, **kw):
        d = {"name": "ceo", "provider": "anthropic", "model": "m", "role": "ceo",
             "children": []}
        d.update(kw)
        return d

    def test_serve_no_schedules_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_tree(tmp, self._tree_data())
            rc = cli.main(["serve", "--tree", path, "--dry-run", "--once",
                           "--serve-dir", os.path.join(tmp, "d")])
            self.assertEqual(rc, 2)

    def test_serve_once_dry_run(self):
        data = self._tree_data(children=[
            {"name": "a", "provider": "deepseek", "model": "m", "role": "r",
             "schedule": "5m", "task": "check"},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_tree(tmp, data)
            sdir = os.path.join(tmp, "d")
            with mock.patch("time.sleep", side_effect=AssertionError("no sleep")):
                rc = cli.main(["serve", "--tree", path, "--dry-run", "--once",
                               "--serve-dir", sdir])
            self.assertEqual(rc, 0)
            state = serve.load_state(sdir)
            self.assertIn("a", state)
            reports = [f for f in os.listdir(sdir) if f.startswith("report-")]
            self.assertEqual(len(reports), 1)

    def test_status_and_report_cli(self):
        data = self._tree_data(children=[
            {"name": "a", "provider": "deepseek", "model": "m", "role": "r",
             "schedule": "5m", "task": "check"},
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_tree(tmp, data)
            sdir = os.path.join(tmp, "d")
            self.assertEqual(
                cli.main(["status", "--tree", path, "--serve-dir", sdir]), 0)
            self.assertEqual(
                cli.main(["report", "--tree", path, "--serve-dir", sdir]), 0)

    def test_serve_bad_tree_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = cli.main(["serve", "--tree", os.path.join(tmp, "nope.json"),
                           "--dry-run", "--once"])
            self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
