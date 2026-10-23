"""swarm-tree command line interface."""

from __future__ import annotations

import argparse
import datetime
import json
import sys

from . import __version__, config, engine, providers, render

EXAMPLE_TREE = """\
name: orchestrator
provider: anthropic
model: claude-sonnet-4-5
role: You are the top-level coordinator. Break hard problems into subtasks for
  your sub-agents, then merge their results into one coherent answer.
# Optional root knobs (with their defaults):
# summary_mode: true      # compress long child outputs before the parent reads them
# max_concurrency: 8      # global cap on concurrent provider calls
# timeout: 120            # seconds per provider call
# summarizer: {provider: deepseek, model: deepseek-chat}  # cheap compressor model
children:
  - name: grok-agent
    provider: xai
    model: grok-4
    role: You are a contrarian analyst. Find flaws, edge cases, and risks others miss.
  - name: gemini-agent
    provider: gemini
    model: gemini-2.5-pro
    role: You are a researcher. Gather facts, evidence, and precedents.
  - name: gpt-agent
    provider: openai
    model: gpt-5
    role: You are a planner. Turn goals into concrete step-by-step plans.
    children:
      - name: code-reviewer
        provider: deepseek
        model: deepseek-chat
        role: You review code diffs and point out bugs.
      - name: writer
        provider: anthropic
        model: claude-sonnet-4-5
        role: You write clear, concise summaries.
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swarm-tree",
        description="Hierarchical multi-agent orchestration: a tree of LLM agents, "
                    "each node backed by a different provider. "
                    "The runtime core is standard-library only; YAML tree configs "
                    "require PyYAML (pip install 'swarm-tree[yaml]'), JSON configs "
                    "always work.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a task through the agent tree.")
    run.add_argument("--tree", required=True, help="Path to tree.yaml / tree.json")
    run.add_argument("--task", required=True, help="The task for the root agent")
    run.add_argument("--dry-run", action="store_true",
                     help="Use deterministic fake providers (no API keys, no cost)")
    run.add_argument("--output", default="transcript.json",
                     help="Where to write the JSON transcript (default: transcript.json)")
    run.add_argument("--max-tokens", type=int, default=1024,
                     help="Max tokens per provider call (default: 1024)")
    run.add_argument("--full-context", action="store_true",
                     help="Disable summary mode: pass full child outputs to the "
                          "synthesize step (the pre-v0.2 behavior)")
    run.add_argument("--fail-fast", action="store_true",
                     help="Abort the run on the first child failure instead of "
                          "continuing in degraded mode")
    run.add_argument("--max-concurrency", type=int, default=None,
                     help="Cap on concurrent provider calls across the whole tree "
                          "(default: from config, 8)")
    run.add_argument("--timeout", type=float, default=None,
                     help="Seconds per provider call (default: from config, 120)")

    sub.add_parser("validate", help="Validate a tree config without running it.").add_argument(
        "--tree", required=True, help="Path to tree.yaml / tree.json"
    )
    sub.add_parser("example", help="Print an example tree.yaml to stdout.")
    return parser


def _cmd_example() -> int:
    print(EXAMPLE_TREE, end="")
    return 0


def _cmd_validate(args) -> int:
    try:
        node = config.load_config(args.tree)
    except (config.TreeError, OSError, json.JSONDecodeError) as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 2
    leaf_count = _count_leaves(node)
    sm = "on" if node.get("summary_mode", True) else "off"
    print(f"valid: {leaf_count} leaves, "
          f"providers: {', '.join(config.providers_used(node))}, "
          f"summary_mode: {sm}, max_concurrency: {node.get('max_concurrency', 8)}")
    return 0


def _count_leaves(node: dict) -> int:
    if not node["children"]:
        return 1
    return sum(_count_leaves(c) for c in node["children"])


def _effective_summary_mode(node: dict, full_context: bool) -> bool:
    """Resolve whether summary mode is on for a run.

    The config's root-level `summary_mode` (default true) ANDed with the
    --full-context flag: the flag always wins, so --full-context turns
    summaries off regardless of the config.
    """
    return bool(node.get("summary_mode", True)) and not full_context


def _count_errors(record: dict) -> int:
    """Count nodes with status "error" anywhere in a finished record tree."""
    n = 1 if record.get("status") == "error" else 0
    return n + sum(_count_errors(c) for c in record.get("children", []))


def _write_transcript(args, record: dict) -> None:
    transcript = {
        "tool": "swarm-tree",
        "version": __version__,
        "task": args.task,
        "dry_run": args.dry_run,
        "generated_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "root": record,
    }
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(transcript, fh, indent=2, ensure_ascii=False)
    print()
    print(f"transcript written to {args.output}")


def _cmd_run(args) -> int:
    """Exit codes: 0 clean run, 3 degraded (answer produced but some
    sub-agents failed), 1 hard failure (root failed/cancelled), 2
    config/usage error."""
    try:
        node = config.load_config(args.tree)
    except (config.TreeError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        provider_fn = engine.dry_run_provider
    else:
        # Fail fast with a clear message before spending anything.
        missing = []
        for provider in config.providers_used(node):
            try:
                providers.api_key_for(provider)
            except providers.ProviderError:
                missing.append(providers.ENV_VARS[provider])
        if missing:
            print(
                f"error: missing API keys for this tree: {', '.join(missing)} "
                f"(set them in the environment or use --dry-run)",
                file=sys.stderr,
            )
            return 2
        provider_fn = None

    max_concurrency = args.max_concurrency or node.get("max_concurrency", 8)
    timeout = args.timeout or node.get("timeout", 120)

    print(f"swarm-tree v{__version__} — task: {args.task!r}")
    print(f"mode: {'dry-run (mock providers, $0.00)' if args.dry_run else 'live'}")
    summary_mode = _effective_summary_mode(node, args.full_context)
    print(f"summary mode: {'on' if summary_mode else 'off (full context)'}")
    print(f"max concurrency: {max_concurrency}, timeout: {timeout}s, "
          f"fail-fast: {'on' if args.fail_fast else 'off'}")
    print()

    record = engine.execute_tree(
        node, args.task, provider_fn, args.max_tokens,
        summary_mode=summary_mode, fail_fast=args.fail_fast,
        max_concurrency=max_concurrency, timeout=timeout)
    n_failed = _count_errors(record)
    try:
        print(render.render_tree(record))
        print()
        if record["status"] == "ok":
            print("=== FINAL ANSWER ===")
            print(record["response"])
            if n_failed:
                print()
                print(f"=== WARNING: {n_failed} sub-agent(s) failed "
                      f"(see {args.output} for details) ===")
        else:
            print(f"=== RUN FAILED: {record['error']} ===")
    finally:
        # The transcript is always dumped — even for cancelled runs — so a
        # Ctrl-C never loses the partial results.
        _write_transcript(args, record)
    if record["status"] == "cancelled":
        raise KeyboardInterrupt  # transcript dumped; let the shell see Ctrl-C
    if record["status"] != "ok":
        return 1  # hard failure
    if n_failed:
        return 3  # degraded: answer produced, but some sub-agents failed
    return 0


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "validate":
        return _cmd_validate(args)
    if args.command == "example":
        return _cmd_example()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
