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
        "summarize": {"input_tokens": 30, "output_tokens": 15},
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
        if kind == "summarize":
            return "SUMMARY", usage
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
        # root: decompose+synthesize; a: decompose+synthesize; a1,b: leaf;
        # summaries: a1 (by a), a+b (by root) — one cheap call each
        self.assertEqual(
            kinds,
            ["decompose", "decompose", "leaf", "leaf",
             "summarize", "summarize", "summarize",
             "synthesize", "synthesize"],
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

    def test_child_failure_degraded_not_fatal(self):
        def boom(provider, model, system, user, kind):
            if kind == "leaf":
                raise RuntimeError("provider exploded")
            if kind == "summarize":
                # echo the input: a real model would restate the failure
                return f"SUMMARIZED: {user[-60:]}", {"input_tokens": 5, "output_tokens": 2}
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
        # the run continues: parent still synthesizes an answer
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(rec["response"], "1) x")
        child = rec["children"][0]
        self.assertEqual(child["status"], "error")
        self.assertIn("provider exploded", child["error"])
        # ...but the failure is visible in the summary the parent sees
        self.assertEqual(child["summary"]["status"], "failed")
        self.assertIn("provider exploded", child["summary"]["summary"])

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
        # tree(): root(decompose+synthesize), a(decompose+synthesize+summarize),
        # a1(leaf+summarize), b(leaf+summarize). ScriptedProvider usage:
        # decompose 100/20, synthesize 200/40, leaf 50/10, summarize 30/15.
        rec = engine.execute_tree(tree(), "big task", ScriptedProvider())
        self.assertEqual(rec["status"], "ok")
        # grand total at root: in = 300 + 410 + 80 = 790, out = 60+100+25 = 185
        self.assertEqual(rec["tokens"], {"input": 790, "output": 185})
        # per-node: a1 leaf + its summarize call
        a1 = rec["children"][0]["children"][0]
        self.assertEqual(a1["tokens"], {"input": 80, "output": 25})
        # a = own decompose+synthesize+summarize + a1
        a = rec["children"][0]
        self.assertEqual(a["tokens"], {"input": 410, "output": 100})
        # b = leaf + summarize
        self.assertEqual(rec["children"][1]["tokens"], {"input": 80, "output": 25})

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
            self.assertEqual(s["summary"], "SUMMARY")
            self.assertFalse(s["summary_fallback"])
            self.assertEqual(s["tokens"], child["tokens"])
        # a1 was summarized by its parent a as well
        self.assertEqual(rec["children"][0]["children"][0]["summary"]["summary"], "SUMMARY")
        # root itself has no summary (no parent to summarize it for)
        self.assertNotIn("summary", rec)

    def test_summarize_call_uses_child_provider(self):
        fn = ScriptedProvider()
        engine.execute_tree(tree(), "big task", fn)
        summarize_calls = [(p, m) for p, m, k in fn.calls if k == "summarize"]
        # one per child: a1 on deepseek/m3 (by a), a on xai/m2, b on gemini/m4 (by root)
        self.assertEqual(
            sorted(summarize_calls),
            [("deepseek", "m3"), ("gemini", "m4"), ("xai", "m2")],
        )

    def test_failed_child_shows_in_synthesize_prompt(self):
        # 2a: failed child in summary mode — the parent's synthesize prompt
        # must show status: failed + a partial summary, and the run continues.
        def boom(provider, model, system, user, kind):
            usage = {"input_tokens": 1, "output_tokens": 1}
            if kind == "leaf" and "bad" in system:
                raise RuntimeError("bad child blew up")
            if kind == "decompose":
                return "1) x\n2) y", usage
            if kind == "summarize":
                # echo the input: a real model would restate the failure
                return f"SUMMARIZED: {user[-60:]}", usage
            return "fine", usage

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
        fn = boom
        rec = engine.execute_tree(node, "t", fn)
        self.assertEqual(rec["status"], "ok")  # run continues past the failure
        synth = [c for c in rec["calls"] if c["kind"] == "synthesize"][0]["user"]
        self.assertIn("status: failed", synth)  # bad child flagged
        self.assertIn("bad child blew up", synth)  # partial summary has the error
        self.assertIn("status: ok", synth)  # good child unaffected

    def test_depth4_chain_summary_mode(self):
        # 2b: max-depth-4 chain with summary mode — summaries nest correctly.
        def chain(depth):
            node = {"name": f"l{depth}", "provider": "openai", "model": "m",
                    "role": "r"}
            if depth < 4:
                node["children"] = [chain(depth + 1)]
            return node

        fn = ScriptedProvider()
        rec = engine.execute_tree(config.validate(chain(0)), "deep task", fn)
        self.assertEqual(rec["status"], "ok")
        # every internal node summarized its child on the child's provider
        kinds = [c[2] for c in fn.calls]
        self.assertEqual(kinds.count("summarize"), 4)  # l3,l2,l1,l0's children
        # nested summaries: each internal node's child record has one
        r = rec
        for expected in ("l1", "l2", "l3", "l4"):
            child = r["children"][0]
            self.assertEqual(child["name"], expected)
            self.assertIsNotNone(child.get("summary"))
            self.assertEqual(child["summary"]["status"], "ok")
            r = child
        # the leaf has no children and no summary of its own to give upward
        self.assertEqual(r["children"], [])

    def test_summaries_run_in_parallel(self):
        # sibling summarize calls must overlap: if they ran sequentially,
        # the barrier would time out and every summary would fall back.
        barrier = threading.Barrier(3)

        def prov(provider, model, system, user, kind):
            usage = {"input_tokens": 1, "output_tokens": 1}
            if kind == "summarize":
                barrier.wait(timeout=10)  # BrokenBarrierError if sequential
                return "SUM", usage
            if kind == "decompose":
                n = sum(1 for line in user.splitlines() if line.startswith("- "))
                return "\n".join(f"{i + 1}) x" for i in range(n)), usage
            return "ok", usage

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

        def prov(provider, model, system, user, kind):
            usage = {"input_tokens": 1, "output_tokens": 1}
            if kind == "summarize" and "'slow'" in system:
                entered.set()
                release.wait(timeout=10)
                return "SLOW-SUM", usage
            if kind == "decompose":
                return "1) x\n2) y", usage
            return "ok", usage

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

        def sabotage(fn, node, child_record, max_tokens):
            if node["name"] == "victim":
                raise ValueError("unexpected bug")
            real_make(fn, node, child_record, max_tokens)

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
        long_answer = "x" * 800

        def flaky(provider, model, system, user, kind):
            usage = {"input_tokens": 1, "output_tokens": 1}
            if kind == "summarize":
                raise RuntimeError("summarizer down")
            if kind == "decompose":
                return "1) x", usage
            return long_answer, usage

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
        summarize_calls = [
            c for c in rec1["children"][0]["calls"] if c["kind"] == "summarize"
        ]
        self.assertEqual(len(summarize_calls), 1)
        self.assertIn("dry-run summary", summarize_calls[0]["response"])
        self.assertEqual(summarize_calls[0]["usage"]["output_tokens"], 50)

    def test_summary_artifact_extraction(self):
        def rich(provider, model, system, user, kind):
            usage = {"input_tokens": 1, "output_tokens": 1}
            if kind == "decompose":
                return "1) x", usage
            if kind == "leaf":
                return (
                    "Done. Wrote the plan to /tmp/plan.md and "
                    "src/report.json; details at https://example.com/report.",
                    usage,
                )
            return "ok", usage

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

    def test_fmt_tokens(self):
        self.assertEqual(render._fmt_tokens(950), "950")
        self.assertEqual(render._fmt_tokens(3200), "3.2k")
        self.assertEqual(render._fmt_tokens(0), "0")


if __name__ == "__main__":
    unittest.main()
