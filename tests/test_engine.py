"""Tests for tree execution and ASCII rendering (all providers mocked)."""

import json
import threading
import time
import unittest

from swarm_tree import config, engine, render


def tree():
    return config.validate(
        {
            "name": "root",
            "provider": "anthropic",
            "model": "m1",
            "role": "coordinate",
            "children": [
                {
                    "name": "a",
                    "provider": "xai",
                    "model": "m2",
                    "role": "analyze",
                    "children": [
                        {
                            "name": "a1",
                            "provider": "deepseek",
                            "model": "m3",
                            "role": "review",
                        }
                    ],
                },
                {"name": "b", "provider": "gemini", "model": "m4", "role": "research"},
            ],
        }
    )


def two_leaf_tree():
    return config.validate(
        {
            "name": "root",
            "provider": "anthropic",
            "model": "m",
            "role": "r",
            "children": [
                {"name": "c1", "provider": "openai", "model": "m", "role": "r"},
                {"name": "c2", "provider": "openai", "model": "m", "role": "r"},
            ],
        }
    )


class ScriptedProvider:
    """Records every call; returns scripted (text, usage, finish_reason)."""

    USAGE = {
        "decompose": {"input_tokens": 100, "output_tokens": 20},
        "synthesize": {"input_tokens": 200, "output_tokens": 40},
        "leaf": {"input_tokens": 50, "output_tokens": 10},
        "summarize": {"input_tokens": 30, "output_tokens": 15},
    }
    # Long enough (>2400 chars) to take the summarize path, not passthrough.
    LONG_ANSWER = "work " * 600  # 3000 chars

    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, provider, model, system, user, kind, max_tokens):
        with self.lock:
            self.calls.append((provider, model, kind))
        usage = dict(self.USAGE[kind])
        if kind == "decompose":
            n = sum(1 for line in user.splitlines() if line.startswith("- "))
            return (
                "\n".join(f"{i + 1}) subtask {i + 1}" for i in range(n)),
                usage,
                "stop",
            )
        if kind == "synthesize":
            return "SYNTH", usage, "stop"
        if kind == "summarize":
            return "SUMMARY", usage, "stop"
        name = system.split(chr(39))[1]
        return f"LEAF:{name} {self.LONG_ANSWER}", usage, "stop"


class TestEngine(unittest.TestCase):
    def test_leaf(self):
        node = config.validate(
            {"name": "solo", "provider": "openai", "model": "m", "role": "work"}
        )
        fn = ScriptedProvider()
        rec = engine.execute_tree(node, "do it", fn)
        self.assertEqual(rec["status"], "ok")
        self.assertTrue(rec["response"].startswith("LEAF:solo"))
        self.assertEqual(len(rec["calls"]), 1)
        self.assertEqual(rec["calls"][0]["kind"], "leaf")

    def test_full_tree_call_counts(self):
        fn = ScriptedProvider()
        rec = engine.execute_tree(tree(), "big task", fn)
        self.assertEqual(rec["status"], "ok")
        kinds = sorted(c[2] for c in fn.calls)
        # root: decompose+synthesize; a: decompose+synthesize; a1,b: leaf.
        # summaries: a1 (by a) and b (by root) — one capped call each.
        # a itself passes through raw: its synthesized answer "SYNTH" is
        # short, so a summarize call would cost more than it saves.
        self.assertEqual(
            kinds,
            ["decompose", "decompose", "leaf", "leaf",
             "summarize", "summarize",
             "synthesize", "synthesize"],
        )
        self.assertEqual(rec["response"], "SYNTH")
        self.assertEqual(len(rec["children"]), 2)
        self.assertEqual(rec["children"][0]["children"][0]["name"], "a1")
        # a's summary is a passthrough of its short synthesized answer
        self.assertTrue(rec["children"][0]["summary"]["passthrough"])

    def test_children_run_in_parallel(self):
        entered, exited = [], []
        barrier = threading.Barrier(2)

        def slow(provider, model, system, user, kind, max_tokens):
            usage = {"input_tokens": 1, "output_tokens": 1}
            if kind == "decompose":
                return "1) x\n2) y", usage, "stop"
            if kind != "leaf":
                return "x", usage, "stop"
            entered.append(system)
            try:
                barrier.wait(timeout=5)
            except threading.BrokenBarrierError:
                pass
            exited.append(system)
            return "leafwork " + "x" * 3000, usage, "stop"

        node = config.validate(
            {
                "name": "root",
                "provider": "anthropic",
                "model": "m",
                "role": "r",
                "children": [
                    {"name": "c1", "provider": "openai", "model": "m", "role": "r"},
                    {"name": "c2", "provider": "openai", "model": "m", "role": "r"},
                ],
            }
        )
        rec = engine.execute_tree(node, "t", slow)
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(len(entered), 2)  # both children entered...
        self.assertEqual(len(exited), 2)  # ...and both finished

    def test_single_child_failure_fails_node(self):
        # A lone failed child means ALL children failed: the node must fail
        # instead of synthesizing an answer from nothing.
        def boom(provider, model, system, user, kind, max_tokens):
            if kind == "leaf":
                raise RuntimeError("provider exploded")
            return "1) x", {"input_tokens": 5, "output_tokens": 2}, "stop"

        node = config.validate(
            {
                "name": "root",
                "provider": "anthropic",
                "model": "m",
                "role": "r",
                "children": [
                    {"name": "c1", "provider": "openai", "model": "m", "role": "r"}
                ],
            }
        )
        rec = engine.execute_tree(node, "t", boom)
        # a lone failed child means ALL children failed -> node fails
        self.assertEqual(rec["status"], "error")
        self.assertIn("all 1 sub-agent(s)", rec["error"])
        child = rec["children"][0]
        self.assertEqual(child["status"], "error")
        self.assertIn("provider exploded", child["error"])

    def test_decompose_fallback_strips_preamble(self):
        def chatty(provider, model, system, user, kind, max_tokens):
            usage = {"input_tokens": 7, "output_tokens": 3}
            if kind == "decompose":
                return (
                    "Here are your subtasks:\n"
                    "research the topic thoroughly\n"
                    "write the final summary"
                ), usage, "stop"
            return "done " + "x" * 3000, usage, "stop"

        rec = engine.execute_tree(two_leaf_tree(), "t", chatty)
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(rec["subtask_parse"], "fallback")
        self.assertEqual(
            rec["children"][0]["subtask"], "research the topic thoroughly")
        self.assertEqual(rec["children"][1]["subtask"], "write the final summary")

    def test_decompose_too_few_items_raises(self):
        # 1 usable line for 2 children: no padding, a clear error instead.
        def stingy(provider, model, system, user, kind, max_tokens):
            usage = {"input_tokens": 7, "output_tokens": 3}
            if kind == "decompose":
                return "just do everything", usage, "stop"
            return "done", usage, "stop"

        rec = engine.execute_tree(two_leaf_tree(), "t", stingy)
        self.assertEqual(rec["status"], "error")
        self.assertIn("decompose", rec["error"])
        self.assertIn("2 children", rec["error"])

    def test_subtask_tagged_format(self):
        def tagged(provider, model, system, user, kind, max_tokens):
            usage = {"input_tokens": 1, "output_tokens": 1}
            if kind == "decompose":
                return (
                    '<subtask name="c1">Research phase</subtask>\n'
                    '<subtask name="c2">Writing phase</subtask>'
                ), usage, "stop"
            return "work " + "x" * 3000, usage, "stop"

        rec = engine.execute_tree(two_leaf_tree(), "t", tagged)
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(rec["subtask_parse"], "tagged")
        self.assertEqual(rec["children"][0]["subtask"], "Research phase")
        self.assertEqual(rec["children"][1]["subtask"], "Writing phase")

    def test_subtask_parse_recorded_exact(self):
        fn = ScriptedProvider()
        rec = engine.execute_tree(two_leaf_tree(), "t", fn)
        self.assertEqual(rec["subtask_parse"], "exact")

    def test_all_children_failed_skips_synthesize(self):
        calls = []

        def prov(provider, model, system, user, kind, max_tokens):
            calls.append(kind)
            usage = {"input_tokens": 1, "output_tokens": 1}
            if kind == "decompose":
                return "1) x\n2) y", usage, "stop"
            if kind == "leaf":
                raise RuntimeError("leaf down")
            return "ok", usage, "stop"

        rec = engine.execute_tree(two_leaf_tree(), "t", prov)
        self.assertEqual(rec["status"], "error")
        self.assertIn("all 2 sub-agent(s)", rec["error"])
        # no synthesize built from nothing, no summarize of errors either
        self.assertNotIn("synthesize", calls)
        for child in rec["children"]:
            self.assertEqual(child["status"], "error")
            self.assertNotIn("summarize", [c["kind"] for c in child["calls"]])

    def test_dry_run_end_to_end(self):
        rec = engine.execute_tree(tree(), "demo task", engine.dry_run_provider)
        self.assertEqual(rec["status"], "ok")
        self.assertIn("dry-run", rec["response"])

        def check(r):
            self.assertIsNotNone(r["response"])
            for c in r["children"]:
                check(c)

        check(rec)

    def test_usage_attached_to_every_call(self):
        rec = engine.execute_tree(tree(), "big task", ScriptedProvider())

        def check(r):
            for call in r["calls"]:
                self.assertIn("usage", call)
                self.assertIn("input_tokens", call["usage"])
                self.assertIn("output_tokens", call["usage"])
                self.assertIn("finish_reason", call)
                self.assertIn("provider", call)
                self.assertIn("model", call)
            for c in r["children"]:
                check(c)

        check(rec)

    def test_token_rollup(self):
        # tree(): root(decompose+synthesize), a(decompose+synthesize), a1 and b
        # (leaf+summarize each). a's own answer ("SYNTH") is short, so it
        # passes through with no summarize call. ScriptedProvider usage:
        # decompose 100/20, synthesize 200/40, leaf 50/10, summarize 30/15.
        rec = engine.execute_tree(tree(), "big task", ScriptedProvider())
        self.assertEqual(rec["status"], "ok")
        # grand total at root: in = 300 + 380 + 80 = 760, out = 60+85+25 = 170
        self.assertEqual(rec["tokens"]["input"], 760)
        self.assertEqual(rec["tokens"]["output"], 170)
        self.assertGreater(rec["tokens"]["cost_usd"], 0)
        # per-node: a1 leaf + its summarize call
        a1 = rec["children"][0]["children"][0]
        self.assertEqual(a1["tokens"]["input"], 80)
        self.assertEqual(a1["tokens"]["output"], 25)
        # a = own decompose+synthesize (no summarize: passthrough) + a1
        a = rec["children"][0]
        self.assertEqual(a["tokens"]["input"], 380)
        self.assertEqual(a["tokens"]["output"], 85)
        # b = leaf + summarize
        self.assertEqual(rec["children"][1]["tokens"]["input"], 80)
        self.assertEqual(rec["children"][1]["tokens"]["output"], 25)

    def test_cost_rollup_uses_price_table(self):
        # 1M in / 1M out on deepseek @ $0.27/$1.10 per 1M = $1.37
        def prov(provider, model, system, user, kind, max_tokens):
            return "x", {"input_tokens": 1_000_000,
                         "output_tokens": 1_000_000}, "stop"

        node = config.validate(
            {"name": "solo", "provider": "deepseek", "model": "m", "role": "r"})
        rec = engine.execute_tree(node, "t", prov)
        self.assertEqual(rec["tokens"]["cost_usd"], 1.37)

    def test_custom_prices_override(self):
        def prov(provider, model, system, user, kind, max_tokens):
            return "x", {"input_tokens": 1000, "output_tokens": 1000}, "stop"

        node = config.validate(
            {"name": "solo", "provider": "deepseek", "model": "m", "role": "r",
             "prices": {"deepseek": {"input": 1.0, "output": 1.0}}})
        rec = engine.execute_tree(node, "t", prov)
        # (1000*1.0 + 1000*1.0) / 1e6 = 0.002
        self.assertEqual(rec["tokens"]["cost_usd"], 0.002)

    def test_dry_run_usage_deterministic(self):
        rec1 = engine.execute_tree(tree(), "demo task", engine.dry_run_provider)
        rec2 = engine.execute_tree(tree(), "demo task", engine.dry_run_provider)
        self.assertEqual(rec1["tokens"], rec2["tokens"])
        self.assertGreater(rec1["tokens"]["input"], 0)
        self.assertGreater(rec1["tokens"]["output"], 0)
        # dry-run leaf input tokens ~= 1 per 4 prompt chars
        leaf_calls = [
            c for c in rec1["children"][1]["calls"] if c["kind"] == "leaf"
        ]
        self.assertEqual(len(leaf_calls), 1)
        call = leaf_calls[0]
        self.assertEqual(
            call["usage"]["input_tokens"],
            len(call["system"] + call["user"]) // 4,
        )
        self.assertEqual(call["usage"]["output_tokens"], 60)
        # root total = sum of every call's usage (no double counting)
        def all_calls(r):
            for c in r["calls"]:
                yield c
            for ch in r["children"]:
                yield from all_calls(ch)

        total_in = sum(c["usage"]["input_tokens"] for c in all_calls(rec1))
        total_out = sum(c["usage"]["output_tokens"] for c in all_calls(rec1))
        self.assertEqual(rec1["tokens"]["input"], total_in)
        self.assertEqual(rec1["tokens"]["output"], total_out)


class TestCancel(unittest.TestCase):
    def test_keyboard_interrupt_marks_cancelled_and_returns(self):
        def kb(provider, model, system, user, kind, max_tokens):
            if kind == "leaf":
                raise KeyboardInterrupt()
            if kind == "decompose":
                return "1) x\n2) y", {"input_tokens": 1, "output_tokens": 1}, "stop"
            return "ok", {"input_tokens": 1, "output_tokens": 1}, "stop"

        rec = engine.execute_tree(two_leaf_tree(), "t", kb)  # must not raise
        self.assertEqual(rec["status"], "cancelled")
        self.assertIn("KeyboardInterrupt", rec["error"])
        for child in rec["children"]:
            self.assertIn(child["status"], ("cancelled", "error"))
        # the transcript still serializes (nothing lost)
        json.dumps({"root": rec})


class TestFailFast(unittest.TestCase):
    def _prov(self, good_started=None, bad_done=None):
        def prov(provider, model, system, user, kind, max_tokens):
            usage = {"input_tokens": 1, "output_tokens": 1}
            if kind == "decompose":
                return "1) x\n2) y", usage, "stop"
            if kind == "leaf" and "'bad'" in system:
                # wait until good is provably in-flight, then fail
                self.assertTrue(good_started.wait(timeout=10))
                bad_done.set()
                raise RuntimeError("bad leaf exploded")
            if kind == "leaf":
                good_started.set()
                self.assertTrue(bad_done.wait(timeout=10))
                time.sleep(0.05)
                return "good " + "x" * 3000, usage, "stop"
            return "ok", usage, "stop"

        return prov

    def _node(self):
        return config.validate(
            {
                "name": "root",
                "provider": "anthropic",
                "model": "m",
                "role": "r",
                "children": [
                    {"name": "bad", "provider": "openai", "model": "m", "role": "r"},
                    {"name": "good", "provider": "openai", "model": "m", "role": "r"},
                ],
            }
        )

    def test_without_fail_fast_degraded_ok(self):
        rec = engine.execute_tree(
            self._node(), "t",
            self._prov(threading.Event(), threading.Event()), fail_fast=False)
        self.assertEqual(rec["status"], "ok")
        by_name = {c["name"]: c for c in rec["children"]}
        self.assertEqual(by_name["bad"]["status"], "error")
        self.assertEqual(by_name["good"]["status"], "ok")

    def test_fail_fast_aborts_on_first_failure(self):
        rec = engine.execute_tree(
            self._node(), "t",
            self._prov(threading.Event(), threading.Event()), fail_fast=True)
        self.assertEqual(rec["status"], "error")
        self.assertIn("bad leaf exploded", rec["error"])
        # the in-flight child was allowed to finish so the transcript stays whole
        by_name = {c["name"]: c for c in rec["children"]}
        self.assertEqual(by_name["good"]["status"], "ok")


class TestGlobalConcurrency(unittest.TestCase):
    def test_semaphore_caps_concurrent_calls(self):
        max_seen, current = 0, 0
        lock = threading.Lock()

        def prov(provider, model, system, user, kind, max_tokens):
            nonlocal max_seen, current
            usage = {"input_tokens": 1, "output_tokens": 1}
            if kind == "decompose":
                n = sum(1 for line in user.splitlines() if line.startswith("- "))
                return "\n".join(f"{i + 1}) x" for i in range(n)), usage, "stop"
            if kind in ("synthesize", "summarize"):
                return "ok", usage, "stop"
            with lock:
                current += 1
                max_seen = max(max_seen, current)
            try:
                time.sleep(0.05)
            finally:
                with lock:
                    current -= 1
            return "leaf " + "x" * 3000, usage, "stop"

        def internals(i):
            return {
                "name": f"g{i}",
                "provider": "openai",
                "model": "m",
                "role": "r",
                "children": [
                    {"name": f"g{i}l{j}", "provider": "openai",
                     "model": "m", "role": "r"}
                    for j in range(4)
                ],
            }

        node = config.validate(
            {
                "name": "root",
                "provider": "openai",
                "model": "m",
                "role": "r",
                "max_concurrency": 3,
                "children": [internals(i) for i in range(3)],
            }
        )
        rec = engine.execute_tree(node, "t", prov)
        self.assertEqual(rec["status"], "ok")
        # 12 leaf threads across nested pools, but never >3 concurrent calls
        self.assertLessEqual(max_seen, 3)
        self.assertGreater(max_seen, 1)  # real parallelism still happened


class TestSummaryMode(unittest.TestCase):
    def test_summary_records_attached_to_children(self):
        fn = ScriptedProvider()
        rec = engine.execute_tree(tree(), "big task", fn)
        self.assertEqual(rec["status"], "ok")
        for child in rec["children"]:
            s = child.get("summary")
            self.assertIsNotNone(s)
            self.assertEqual(s["name"], child["name"])
            self.assertEqual(s["provider"], child["provider"])
            self.assertEqual(s["model"], child["model"])
            self.assertEqual(s["status"], "ok")
            self.assertFalse(s["summary_fallback"])
            self.assertEqual(s["tokens"], child["tokens"])
            if child["name"] == "a":
                # a's synthesized answer is short: passthrough, no summarize call
                self.assertTrue(s["passthrough"])
                self.assertEqual(s["summary"], "SYNTH")
            else:
                self.assertFalse(s["passthrough"])
                self.assertEqual(s["summary"], "SUMMARY")
        # a1 was summarized by its parent a as well
        self.assertEqual(rec["children"][0]["children"][0]["summary"]["summary"], "SUMMARY")
        # root itself has no summary (no parent to summarize it for)
        self.assertNotIn("summary", rec)

    def test_summarize_call_uses_child_provider(self):
        fn = ScriptedProvider()
        engine.execute_tree(tree(), "big task", fn)
        summarize_calls = [(p, m) for p, m, k in fn.calls if k == "summarize"]
        # one per long-answer child: a1 on deepseek/m3 (by a), b on
        # gemini/m4 (by root). a's own answer is short -> passthrough.
        self.assertEqual(
            sorted(summarize_calls),
            [("deepseek", "m3"), ("gemini", "m4")],
        )

    def test_summarize_call_capped_at_300_tokens(self):
        seen = []

        def prov(provider, model, system, user, kind, max_tokens):
            usage = {"input_tokens": 1, "output_tokens": 1}
            if kind == "summarize":
                seen.append(max_tokens)
                return "SUM", usage, "stop"
            if kind == "decompose":
                return "1) x", usage, "stop"
            return "z" * 3000, usage, "stop"

        node = config.validate(
            {"name": "root", "provider": "anthropic", "model": "m", "role": "r",
             "children": [{"name": "c1", "provider": "openai", "model": "m",
                            "role": "r"}]})
        engine.execute_tree(node, "t", prov)
        self.assertEqual(seen, [engine.SUMMARIZE_MAX_TOKENS])
        self.assertLessEqual(engine.SUMMARIZE_MAX_TOKENS, 300)

    def test_short_answer_passthrough(self):
        # Answers under the threshold skip the summarize call entirely.
        def prov(provider, model, system, user, kind, max_tokens):
            usage = {"input_tokens": 1, "output_tokens": 1}
            if kind == "decompose":
                return "1) x", usage, "stop"
            if kind == "leaf":
                return "brief answer", usage, "stop"
            return "ok", usage, "stop"

        node = config.validate(
            {"name": "root", "provider": "anthropic", "model": "m", "role": "r",
             "children": [{"name": "c1", "provider": "openai", "model": "m",
                            "role": "r"}]})
        rec = engine.execute_tree(node, "t", prov)
        self.assertEqual(rec["status"], "ok")
        s = rec["children"][0]["summary"]
        self.assertTrue(s["passthrough"])
        self.assertEqual(s["summary"], "brief answer")
        self.assertFalse(s["summary_fallback"])
        kinds = [c["kind"] for c in rec["children"][0]["calls"]]
        self.assertNotIn("summarize", kinds)
        # the parent still sees the short raw answer
        synth = [c for c in rec["calls"] if c["kind"] == "synthesize"][0]["user"]
        self.assertIn("brief answer", synth)

    def test_failed_child_skips_summarize_call(self):
        calls = []

        def prov(provider, model, system, user, kind, max_tokens):
            calls.append((system, kind))
            usage = {"input_tokens": 1, "output_tokens": 1}
            if kind == "decompose":
                return "1) x\n2) y", usage, "stop"
            if kind == "leaf" and "'bad'" in system:
                raise RuntimeError("bad leaf down")
            return "good " + "z" * 3000, usage, "stop"

        node = config.validate(
            {
                "name": "root",
                "provider": "anthropic",
                "model": "m",
                "role": "r",
                "children": [
                    {"name": "good", "provider": "openai", "model": "m", "role": "r"},
                    {"name": "bad", "provider": "openai", "model": "m", "role": "r"},
                ],
            }
        )
        rec = engine.execute_tree(node, "t", prov)
        self.assertEqual(rec["status"], "ok")  # one survivor is enough
        by_name = {c["name"]: c for c in rec["children"]}
        # no provider call was spent summarizing the error string
        self.assertFalse(any(k == "summarize" for s, k in calls if "'bad'" in s))
        self.assertTrue(any(k == "summarize" for s, k in calls if "'good'" in s))
        bad_summary = by_name["bad"]["summary"]
        self.assertEqual(bad_summary["status"], "failed")
        self.assertIn("bad leaf down", bad_summary["summary"])

    def test_summarizer_override_root_and_node(self):
        seen = []

        def prov(provider, model, system, user, kind, max_tokens):
            usage = {"input_tokens": 1, "output_tokens": 1}
            if kind == "summarize":
                seen.append((provider, model))
                return "SUM", usage, "stop"
            if kind == "decompose":
                return "1) x", usage, "stop"
            return "z" * 3000, usage, "stop"

        def node(summarizer, child_summarizer=None):
            child = {"name": "c1", "provider": "openai", "model": "m", "role": "r"}
            if child_summarizer:
                child["summarizer"] = child_summarizer
            d = {"name": "root", "provider": "anthropic", "model": "m", "role": "r",
                 "children": [child]}
            if summarizer:
                d["summarizer"] = summarizer
            return config.validate(d)

        # root-level override
        engine.execute_tree(
            node({"provider": "deepseek", "model": "cheap"}), "t", prov)
        self.assertEqual(seen, [("deepseek", "cheap")])
        # per-node override wins over root
        seen.clear()
        engine.execute_tree(
            node({"provider": "deepseek", "model": "cheap"},
                 {"provider": "xai", "model": "grok"}), "t", prov)
        self.assertEqual(seen, [("xai", "grok")])
        # default: the child's own provider/model
        seen.clear()
        engine.execute_tree(node(None), "t", prov)
        self.assertEqual(seen, [("openai", "m")])

    def test_summary_mode_cost_math(self):
        # Honest economics, with realistic fake lengths: a cheap deepseek
        # child writes a long answer; an expensive anthropic parent reads it.
        # Summary mode must win on COST (not necessarily on total tokens —
        # the extra summarize call adds tokens, it just adds cheap ones).
        LONG = "x" * 5000
        seen = []

        def prov(provider, model, system, user, kind, max_tokens):
            seen.append((provider, kind))
            if kind == "decompose":
                n = sum(1 for ln in user.splitlines() if ln.startswith("- "))
                return ("\n".join(f"{i + 1}) part" for i in range(n)),
                        {"input_tokens": 100, "output_tokens": 20}, "stop")
            if kind == "synthesize":
                return ("SYNTH", {"input_tokens": len(user) // 4,
                                  "output_tokens": 40}, "stop")
            if kind == "summarize":
                return ("SUMMARY " + "y" * 280,
                        {"input_tokens": len(user) // 4,
                         "output_tokens": 75}, "stop")
            if "'long'" in system:
                return (LONG, {"input_tokens": 100,
                               "output_tokens": 1250}, "stop")
            return ("brief", {"input_tokens": 100, "output_tokens": 10}, "stop")

        def make_tree():
            return config.validate(
                {
                    "name": "root",
                    "provider": "anthropic",
                    "model": "m",
                    "role": "r",
                    "summarizer": {"provider": "deepseek", "model": "cheap"},
                    "children": [
                        {"name": "long", "provider": "deepseek",
                         "model": "m", "role": "r"},
                        {"name": "short", "provider": "deepseek",
                         "model": "m", "role": "r"},
                    ],
                }
            )

        sm = engine.execute_tree(make_tree(), "t", prov, summary_mode=True)
        fc = engine.execute_tree(make_tree(), "t", prov, summary_mode=False)
        by_name = {c["name"]: c for c in sm["children"]}
        # long answer was summarized; short answer passed through raw
        self.assertFalse(by_name["long"]["summary"]["passthrough"])
        self.assertTrue(by_name["short"]["summary"]["passthrough"])
        self.assertIn(("deepseek", "summarize"),
                      [(p, k) for p, k in seen])
        # the win is on cost: cheap tokens compress for an expensive parent
        self.assertLess(sm["tokens"]["cost_usd"], fc["tokens"]["cost_usd"])
        # ...even though total tokens went UP (the extra summarize call)
        self.assertGreater(sm["tokens"]["input"], fc["tokens"]["input"])

    def test_truncated_leaf_flagged_in_summary(self):
        def prov(provider, model, system, user, kind, max_tokens):
            if kind == "decompose":
                return "1) x", {"input_tokens": 1, "output_tokens": 1}, "stop"
            if kind == "leaf":
                return ("y" * 3000, {"input_tokens": 10, "output_tokens": 90},
                        "length")
            return "ok", {"input_tokens": 1, "output_tokens": 1}, "stop"

        node = config.validate(
            {"name": "root", "provider": "anthropic", "model": "m", "role": "r",
             "children": [{"name": "c1", "provider": "openai", "model": "m",
                            "role": "r"}]})
        rec = engine.execute_tree(node, "t", prov)
        self.assertEqual(rec["status"], "ok")
        child = rec["children"][0]
        leaf_call = [c for c in child["calls"] if c["kind"] == "leaf"][0]
        self.assertEqual(leaf_call["finish_reason"], "length")
        self.assertTrue(child["summary"]["truncated"])
        synth = [c for c in rec["calls"] if c["kind"] == "synthesize"][0]["user"]
        self.assertIn("(truncated)", synth)

    def test_failed_child_shows_in_synthesize_prompt(self):
        # 2a: failed child in summary mode — the parent's synthesize prompt
        # must show status: failed + the error, and the run continues.
        def boom(provider, model, system, user, kind, max_tokens):
            usage = {"input_tokens": 1, "output_tokens": 1}
            if kind == "leaf" and "'bad'" in system:
                raise RuntimeError("bad child blew up")
            if kind == "decompose":
                return "1) x\n2) y", usage, "stop"
            return "fine", usage, "stop"

        node = config.validate(
            {
                "name": "root",
                "provider": "anthropic",
                "model": "m",
                "role": "r",
                "children": [
                    {"name": "good", "provider": "openai", "model": "m", "role": "r"},
                    {"name": "bad", "provider": "openai", "model": "m", "role": "r"},
                ],
            }
        )
        rec = engine.execute_tree(node, "t", boom)
        self.assertEqual(rec["status"], "ok")  # run continues past the failure
        synth = [c for c in rec["calls"] if c["kind"] == "synthesize"][0]["user"]
        self.assertIn("status: failed", synth)  # bad child flagged
        self.assertIn("bad child blew up", synth)  # error in the summary
        self.assertIn("status: ok", synth)  # good child unaffected

    def test_depth4_chain_summary_mode(self):
        # 2b: max-depth-4 chain with summary mode — summaries nest correctly.
        # Only the leaf's long answer needs a real summarize call; the
        # intermediate "SYNTH" answers are short and pass through raw.
        def chain(depth):
            node = {"name": f"l{depth}", "provider": "openai", "model": "m",
                    "role": "r"}
            if depth < 4:
                node["children"] = [chain(depth + 1)]
            return node

        fn = ScriptedProvider()
        rec = engine.execute_tree(config.validate(chain(0)), "deep task", fn)
        self.assertEqual(rec["status"], "ok")
        kinds = [c[2] for c in fn.calls]
        self.assertEqual(kinds.count("summarize"), 1)  # only l4's long answer
        # nested summaries: each internal node's child record has one
        r = rec
        for expected in ("l1", "l2", "l3", "l4"):
            child = r["children"][0]
            self.assertEqual(child["name"], expected)
            self.assertIsNotNone(child.get("summary"))
            self.assertEqual(child["summary"]["status"], "ok")
            r = child
        # l4 was really summarized; l1-l3 passed their short answers through
        self.assertFalse(rec["children"][0]["children"][0]["children"][0]
                         ["children"][0]["summary"]["passthrough"])
        self.assertTrue(rec["children"][0]["summary"]["passthrough"])
        # the leaf has no children and no summary of its own to give upward
        self.assertEqual(r["children"], [])

    def test_summaries_run_in_parallel(self):
        # sibling summarize calls must overlap: if they ran sequentially,
        # the barrier would time out and every summary would fall back.
        barrier = threading.Barrier(3)

        def prov(provider, model, system, user, kind, max_tokens):
            usage = {"input_tokens": 1, "output_tokens": 1}
            if kind == "summarize":
                barrier.wait(timeout=10)  # BrokenBarrierError if sequential
                return "SUM", usage, "stop"
            if kind == "decompose":
                n = sum(1 for line in user.splitlines() if line.startswith("- "))
                return "\n".join(f"{i + 1}) x" for i in range(n)), usage, "stop"
            return "leafwork " + "x" * 3000, usage, "stop"

        node = config.validate(
            {
                "name": "root",
                "provider": "anthropic",
                "model": "m",
                "role": "r",
                "children": [
                    {"name": f"c{i}", "provider": "openai", "model": "m", "role": "r"}
                    for i in range(3)
                ],
            }
        )
        rec = engine.execute_tree(node, "t", prov)
        self.assertEqual(rec["status"], "ok")
        for child in rec["children"]:
            self.assertFalse(child["summary"]["summary_fallback"])
            self.assertEqual(child["summary"]["summary"], "SUM")

    def test_one_slow_summary_does_not_block_siblings(self):
        # one child's summarize hanging must not delay the others forever:
        # siblings finish; the slow one falls back only if IT fails.
        entered = threading.Event()
        release = threading.Event()

        def prov(provider, model, system, user, kind, max_tokens):
            usage = {"input_tokens": 1, "output_tokens": 1}
            if kind == "leaf":
                return "leafwork " + "x" * 3000, usage, "stop"
            if kind == "summarize" and "'slow'" in system:
                entered.set()
                release.wait(timeout=10)
                return "SLOW-SUM", usage, "stop"
            if kind == "decompose":
                return "1) x\n2) y", usage, "stop"
            return "ok", usage, "stop"

        node = config.validate(
            {
                "name": "root",
                "provider": "anthropic",
                "model": "m",
                "role": "r",
                "children": [
                    {"name": "fast", "provider": "openai", "model": "m", "role": "r"},
                    {"name": "slow", "provider": "openai", "model": "m", "role": "r"},
                ],
            }
        )
        rec = engine.execute_tree(node, "t", prov)
        self.assertEqual(rec["status"], "ok")
        by_name = {c["name"]: c for c in rec["children"]}
        self.assertEqual(by_name["fast"]["summary"]["summary"], "ok")
        self.assertEqual(by_name["slow"]["summary"]["summary"], "SLOW-SUM")
        self.assertTrue(entered.is_set())

    def test_safe_summary_isolation_on_unexpected_error(self):
        # defense in depth: if _make_summary itself blows up (not just the
        # provider call), the sibling summaries must still land.
        real_make = engine._make_summary

        def sabotage(ctx, node, child_record):
            if node["name"] == "victim":
                raise ValueError("unexpected bug")
            real_make(ctx, node, child_record)

        engine._make_summary = sabotage
        try:
            fn = ScriptedProvider()
            node = config.validate(
                {
                    "name": "root",
                    "provider": "anthropic",
                    "model": "m",
                    "role": "r",
                    "children": [
                        {"name": "victim", "provider": "openai", "model": "m", "role": "r"},
                        {"name": "okc", "provider": "openai", "model": "m", "role": "r"},
                    ],
                }
            )
            rec = engine.execute_tree(node, "t", fn)
            self.assertEqual(rec["status"], "ok")
            by_name = {c["name"]: c for c in rec["children"]}
            self.assertEqual(by_name["victim"]["summary"]["status"], "failed")
            self.assertTrue(by_name["victim"]["summary"]["summary_fallback"])
            self.assertEqual(by_name["okc"]["summary"]["summary"], "SUMMARY")
            self.assertFalse(by_name["okc"]["summary"]["summary_fallback"])
        finally:
            engine._make_summary = real_make

    def test_synthesize_sees_summaries_not_raw_outputs(self):
        fn = ScriptedProvider()
        rec = engine.execute_tree(tree(), "big task", fn)
        synth_calls = [c for c in rec["calls"] if c["kind"] == "synthesize"]
        self.assertEqual(len(synth_calls), 1)
        user = synth_calls[0]["user"]
        self.assertIn("SUMMARY", user)  # compact summaries present
        self.assertIn("transcript.json", user)  # pointers present
        self.assertNotIn("LEAF:", user)  # raw child outputs absent

    def test_synthesize_prompt_cites_and_flags(self):
        # item 3: the synthesize prompt must instruct citation and explicit
        # flagging of failed/incomplete/contradictory children.
        prompt = engine._synthesize_prompt("task", [], True)
        self.assertIn("Cite which sub-agent each key point comes from", prompt)
        self.assertIn("say so explicitly instead of guessing", prompt)

    def test_summarize_failure_falls_back_to_truncation(self):
        long_answer = "x" * 3000  # over the summarize threshold

        def flaky(provider, model, system, user, kind, max_tokens):
            usage = {"input_tokens": 1, "output_tokens": 1}
            if kind == "summarize":
                raise RuntimeError("summarizer down")
            if kind == "decompose":
                return "1) x", usage, "stop"
            return long_answer, usage, "stop"

        node = config.validate(
            {
                "name": "root",
                "provider": "anthropic",
                "model": "m",
                "role": "r",
                "children": [
                    {"name": "c1", "provider": "openai", "model": "m", "role": "r"}
                ],
            }
        )
        rec = engine.execute_tree(node, "t", flaky)
        self.assertEqual(rec["status"], "ok")  # run survives the summarize failure
        s = rec["children"][0]["summary"]
        self.assertTrue(s["summary_fallback"])
        self.assertEqual(s["summary"], "x" * 500 + "...")

    def test_full_context_disables_summaries(self):
        fn = ScriptedProvider()
        rec = engine.execute_tree(tree(), "big task", fn, summary_mode=False)
        self.assertEqual(rec["status"], "ok")
        kinds = [c[2] for c in fn.calls]
        self.assertNotIn("summarize", kinds)
        for child in rec["children"]:
            self.assertNotIn("summary", child)
        synth_calls = [c for c in rec["calls"] if c["kind"] == "synthesize"]
        user = synth_calls[0]["user"]
        self.assertIn("LEAF:", user)  # raw outputs restored
        self.assertNotIn("SUMMARY", user)

    def test_dry_run_summarize_deterministic(self):
        rec1 = engine.execute_tree(tree(), "demo task", engine.dry_run_provider)
        rec2 = engine.execute_tree(tree(), "demo task", engine.dry_run_provider)
        self.assertEqual(rec1["tokens"], rec2["tokens"])
        # a1 is a leaf with a long dry-run answer: it gets a real summarize call
        summarize_calls = [
            c for c in rec1["children"][0]["children"][0]["calls"]
            if c["kind"] == "summarize"
        ]
        self.assertEqual(len(summarize_calls), 1)
        self.assertIn("dry-run summary", summarize_calls[0]["response"])
        self.assertEqual(summarize_calls[0]["usage"]["output_tokens"], 50)

    def test_summary_artifact_extraction(self):
        def rich(provider, model, system, user, kind, max_tokens):
            usage = {"input_tokens": 1, "output_tokens": 1}
            if kind == "decompose":
                return "1) x", usage, "stop"
            if kind == "leaf":
                return (
                    "Done. Wrote the plan to /tmp/plan.md and "
                    "src/report.json; details at https://example.com/report.",
                    usage,
                    "stop",
                )
            return "ok", usage, "stop"

        node = config.validate(
            {
                "name": "root",
                "provider": "anthropic",
                "model": "m",
                "role": "r",
                "children": [
                    {"name": "c1", "provider": "openai", "model": "m", "role": "r"}
                ],
            }
        )
        rec = engine.execute_tree(node, "t", rich)
        arts = rec["children"][0]["summary"]["artifacts"]
        self.assertIn("/tmp/plan.md", arts)
        self.assertIn("src/report.json", arts)
        self.assertIn("https://example.com/report", arts)

    def test_extract_artifacts_unit(self):
        self.assertEqual(
            engine._extract_artifacts("see /a/b.py and https://x.io/y, also z.md"),
            ["https://x.io/y", "/a/b.py", "z.md"],  # URLs first, then paths
        )
        self.assertEqual(engine._extract_artifacts(""), [])
        self.assertEqual(engine._extract_artifacts(None), [])
        # no false positives on plain prose
        self.assertEqual(engine._extract_artifacts("the cat sat on the mat"), [])


class TestRender(unittest.TestCase):
    def test_ascii_structure(self):
        rec = engine.execute_tree(tree(), "demo", engine.dry_run_provider)
        out = render.render_tree(rec)
        self.assertIn("● root [anthropic/m1] ok", out)
        self.assertIn("├── ● a [xai/m2] ok", out)
        self.assertIn("└── ● b [gemini/m4] ok", out)
        self.assertIn("a1 [deepseek/m3]", out)
        self.assertIn("↳", out)

    def test_error_glyph(self):
        rec = {
            "name": "x",
            "provider": "openai",
            "model": "m",
            "status": "error",
            "duration_s": 0.1,
            "response": None,
            "error": "boom",
            "children": [],
        }
        self.assertIn("✖", render.render_tree(rec))

    def test_tokens_shown_per_node(self):
        rec = engine.execute_tree(tree(), "demo", engine.dry_run_provider)
        out = render.render_tree(rec)
        # compact in/out display, e.g. "1.2k/80"; root line carries the rollup
        toks = rec["tokens"]
        expect = (
            f"{render._fmt_tokens(toks['input'])}/{render._fmt_tokens(toks['output'])}"
        )
        self.assertIn(expect, out.splitlines()[0])

    def test_cost_shown_per_node(self):
        rec = engine.execute_tree(tree(), "demo", engine.dry_run_provider)
        out = render.render_tree(rec)
        # dry-run cost is $0.00 (no prices apply to mock usage... actually
        # dry-run records real provider names, so cost is estimated)
        self.assertIn("~$", out.splitlines()[0])

    def test_fmt_tokens(self):
        self.assertEqual(render._fmt_tokens(950), "950")
        self.assertEqual(render._fmt_tokens(3200), "3.2k")
        self.assertEqual(render._fmt_tokens(0), "0")

    def test_fmt_cost(self):
        self.assertEqual(render._fmt_cost(1.5), "$1.50")
        self.assertEqual(render._fmt_cost(0.0042), "$0.0042")
        self.assertEqual(render._fmt_cost(0), "$0.0000")


if __name__ == "__main__":
    unittest.main()
