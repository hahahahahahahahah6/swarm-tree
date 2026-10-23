"""Tests for tree execution and ASCII rendering (all providers mocked)."""

import threading
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


class ScriptedProvider:
    """Records every call; returns scripted (text, usage) responses."""

    USAGE = {
        "decompose": {"input_tokens": 100, "output_tokens": 20},
        "synthesize": {"input_tokens": 200, "output_tokens": 40},
        "leaf": {"input_tokens": 50, "output_tokens": 10},
    }

    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, provider, model, system, user, kind):
        with self.lock:
            self.calls.append((provider, model, kind))
        usage = dict(self.USAGE[kind])
        if kind == "decompose":
            n = len(user.split("Your sub-agents:")[-1].splitlines())
            # count child lines like "- a: ..."
            n = sum(1 for line in user.splitlines() if line.startswith("- "))
            return "\n".join(f"{i + 1}) subtask {i + 1}" for i in range(n)), usage
        if kind == "synthesize":
            return "SYNTH", usage
        return f"LEAF:{system.split(chr(39))[1]}", usage


class TestEngine(unittest.TestCase):
    def test_leaf(self):
        node = config.validate(
            {"name": "solo", "provider": "openai", "model": "m", "role": "work"}
        )
        fn = ScriptedProvider()
        rec = engine.execute_tree(node, "do it", fn)
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(rec["response"], "LEAF:solo")
        self.assertEqual(len(rec["calls"]), 1)
        self.assertEqual(rec["calls"][0]["kind"], "leaf")

    def test_full_tree_call_counts(self):
        fn = ScriptedProvider()
        rec = engine.execute_tree(tree(), "big task", fn)
        self.assertEqual(rec["status"], "ok")
        kinds = sorted(c[2] for c in fn.calls)
        # root: decompose+synthesize; a: decompose+synthesize; a1,b: leaf
        self.assertEqual(
            kinds, ["decompose", "decompose", "leaf", "leaf", "synthesize", "synthesize"]
        )
        self.assertEqual(rec["response"], "SYNTH")
        self.assertEqual(len(rec["children"]), 2)
        self.assertEqual(rec["children"][0]["children"][0]["name"], "a1")

    def test_children_run_in_parallel(self):
        entered, exited = [], []
        barrier = threading.Barrier(2)

        def slow(provider, model, system, user, kind):
            if kind != "leaf":
                return "x", {"input_tokens": 1, "output_tokens": 1}  # root's decompose/synthesize don't count
            entered.append(system)
            try:
                barrier.wait(timeout=5)
            except threading.BrokenBarrierError:
                pass
            exited.append(system)
            return "x", {"input_tokens": 1, "output_tokens": 1}

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

    def test_child_failure_recorded(self):
        def boom(provider, model, system, user, kind):
            if kind == "leaf":
                raise RuntimeError("provider exploded")
            return "1) x", {"input_tokens": 5, "output_tokens": 2}

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
        self.assertEqual(rec["status"], "error")
        self.assertIn("c1", rec["error"])
        self.assertEqual(rec["children"][0]["status"], "error")

    def test_decompose_fallback_when_not_numbered(self):
        def chatty(provider, model, system, user, kind):
            usage = {"input_tokens": 7, "output_tokens": 3}
            if kind == "decompose":
                return "just do your best, both of you", usage  # no numbered list
            return "done", usage

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
        rec = engine.execute_tree(node, "t", chatty)
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(len(rec["children"]), 2)

    def test_dry_run_end_to_end(self):
        rec = engine.execute_tree(tree(), "demo task", engine.dry_run_provider)
        self.assertEqual(rec["status"], "ok")
        self.assertIn("dry-run", rec["response"])
        # every node got a deterministic mock response
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
            for c in r["children"]:
                check(c)

        check(rec)

    def test_token_rollup(self):
        # tree(): root(decompose+synthesize), a(decompose+synthesize),
        # a1(leaf), b(leaf). ScriptedProvider usage:
        # decompose 100/20, synthesize 200/40, leaf 50/10.
        rec = engine.execute_tree(tree(), "big task", ScriptedProvider())
        self.assertEqual(rec["status"], "ok")
        # grand total at root: in = 100+200+100+200+50+50 = 700
        self.assertEqual(rec["tokens"], {"input": 700, "output": 140})
        # per-node: a1 leaf alone
        a1 = rec["children"][0]["children"][0]
        self.assertEqual(a1["tokens"], {"input": 50, "output": 10})
        # a = own decompose+synthesize + a1
        a = rec["children"][0]
        self.assertEqual(a["tokens"], {"input": 350, "output": 70})
        # b leaf alone
        self.assertEqual(rec["children"][1]["tokens"], {"input": 50, "output": 10})

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
        self.assertEqual(rec1["tokens"], {"input": total_in, "output": total_out})


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

    def test_fmt_tokens(self):
        self.assertEqual(render._fmt_tokens(950), "950")
        self.assertEqual(render._fmt_tokens(3200), "3.2k")
        self.assertEqual(render._fmt_tokens(0), "0")


if __name__ == "__main__":
    unittest.main()
