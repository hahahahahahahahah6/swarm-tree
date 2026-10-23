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
    """Records every call; returns scripted responses."""

    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, provider, model, system, user, kind):
        with self.lock:
            self.calls.append((provider, model, kind))
        if kind == "decompose":
            n = len(user.split("Your sub-agents:")[-1].splitlines())
            # count child lines like "- a: ..."
            n = sum(1 for line in user.splitlines() if line.startswith("- "))
            return "\n".join(f"{i + 1}) subtask {i + 1}" for i in range(n))
        if kind == "synthesize":
            return "SYNTH"
        return f"LEAF:{system.split(chr(39))[1]}"


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
                return "x"  # root's decompose/synthesize don't count
            entered.append(system)
            try:
                barrier.wait(timeout=5)
            except threading.BrokenBarrierError:
                pass
            exited.append(system)
            return "x"

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
            return "1) x"

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
            if kind == "decompose":
                return "just do your best, both of you"  # no numbered list
            return "done"

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


if __name__ == "__main__":
    unittest.main()
