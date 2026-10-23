"""Tree execution: decompose -> parallel children -> synthesize.

A leaf node calls its provider once with (role + task) and returns text.

An internal node:
  1. calls its provider with (role + task + the children's roles) to produce
     one subtask per child (decompose step);
  2. runs its children in parallel (threads); a failed child is recorded
     (status/error) and the parent keeps going in degraded mode instead of
     aborting — unless --fail-fast is set. If EVERY child failed, the node
     fails instead of synthesizing an answer from nothing;
  3. summarizes each long child answer with one capped call to a summarizer
     model (summary mode, default on) — short answers pass through raw, and
     failed children get error summaries without a provider call;
  4. calls its provider again with the child results to produce the final
     answer (synthesize step). In summary mode the synthesize prompt sees
     only the compact summaries, never the raw child outputs; full outputs
     stay in the transcript with pointers.

Concurrency: a single semaphore caps concurrent provider calls across the
whole tree (root `max_concurrency`, default 8); per-node pools are bounded
by the same number. Nested pools can no longer turn into a 429 storm.

Every provider call is recorded in the node transcript (provider, model,
prompts, response, finish reason, timings, usage) so the whole run can be
audited afterwards.
"""

from __future__ import annotations

import datetime
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import config as _config
from . import providers

# provider_fn(provider, model, system, user, kind, max_tokens)
#     -> (text, usage, finish_reason)
# kind is one of "leaf", "decompose", "synthesize", "summarize"; usage is
# {"input_tokens": int, "output_tokens": int}; finish_reason is the
# provider's stop/finish reason (or None).

SUMMARIZE_MAX_TOKENS = 300
# Summarize a child's answer only when it exceeds ~2x the summary budget
# (~300 tokens ~= 1200 chars). Shorter answers pass through raw: a summarize
# call would cost more than it saves.
SUMMARY_BUDGET_CHARS = 1200
SUMMARY_THRESHOLD_CHARS = 2 * SUMMARY_BUDGET_CHARS


class TreeRunError(RuntimeError):
    """Run-time failure: bad decompose output, all children failed, etc."""


class _Ctx:
    """Per-run settings threaded through execution."""

    def __init__(self, *, fn, max_tokens, summary_mode, fail_fast,
                 max_concurrency, timeout, prices, summarizer):
        self.fn = fn
        self.max_tokens = max_tokens
        self.summary_mode = summary_mode
        self.fail_fast = fail_fast
        self.max_concurrency = max_concurrency
        self.semaphore = threading.Semaphore(max_concurrency)
        self.timeout = timeout
        self.prices = prices
        self.summarizer = summarizer  # root-level {provider, model} or None
        self.summary_threshold = SUMMARY_THRESHOLD_CHARS


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _new_record(node: dict, subtask: str | None = None) -> dict:
    return {
        "name": node["name"],
        "provider": node["provider"],
        "model": node["model"],
        "depth": node["depth"],
        "role": node["role"],
        "subtask": subtask,
        "status": "running",
        "started_at": _now_iso(),
        "duration_s": 0.0,
        "calls": [],  # each: {kind, provider, model, system, user, response,
        #              duration_s, usage, finish_reason}
        "tokens": {"input": 0, "output": 0, "cost_usd": 0.0},
        "response": None,
        "error": None,
        "children": [],
    }


def _finish(record: dict, started: float) -> None:
    record["duration_s"] = round(time.monotonic() - started, 3)


def _call(ctx: _Ctx, node: dict, user_prompt: str, kind: str, record: dict,
          max_tokens: int | None = None, provider: str | None = None,
          model: str | None = None) -> str:
    """One provider call under the global concurrency semaphore, recorded."""
    prov = provider or node["provider"]
    mod = model or node["model"]
    system = f"You are the '{node['name']}' agent. {node['role']}"
    started = time.monotonic()
    with ctx.semaphore:
        text, usage, finish_reason = ctx.fn(
            prov, mod, system, user_prompt, kind, max_tokens or ctx.max_tokens
        )
    duration = round(time.monotonic() - started, 3)
    record["calls"].append(
        {"kind": kind, "provider": prov, "model": mod, "system": system,
         "user": user_prompt, "response": text, "duration_s": duration,
         "usage": usage, "finish_reason": finish_reason}
    )
    return text


_NUMBERED = re.compile(r"^\s*\d+[.)]\s*(.+?)\s*$")
_SUBTASK_TAG_RE = re.compile(
    r'<subtask\s+name="([^"]+)"\s*>(.*?)</subtask\s*>',
    re.IGNORECASE | re.DOTALL,
)
_PREAMBLE_RE = re.compile(
    r"^(here\s+(are|is)|sure|certainly|of\s+course|absolutely|below\s+(are|is)|"
    r"i('ve| have)\s+(broken|split|divided|decomposed).*)",
    re.IGNORECASE,
)
_CLOSER_RE = re.compile(
    r"^(let\s+me\s+know|feel\s+free|hope\s+this\s+helps).*",
    re.IGNORECASE,
)


def _strip_preamble(lines: list) -> list:
    """Drop chatty intro/outro lines around a non-conforming subtask list."""
    out = list(lines)
    while out and (out[0].rstrip().endswith(":") or _PREAMBLE_RE.match(out[0])):
        out.pop(0)
    while out and _CLOSER_RE.match(out[-1]):
        out.pop()
    return out


def _parse_subtasks(text: str, children: list) -> tuple:
    """Split a decompose response into one subtask per child.

    Returns (items, parse) where parse is "exact" (numbered list), "tagged"
    (<subtask name="..."> blocks), or "fallback" (line distribution after
    stripping preambles).

    Raises TreeRunError when the model produced fewer usable items than
    children — padding with duplicates would silently duplicate work, which
    is worse than failing loudly.
    """
    text = text or ""
    tagged = _SUBTASK_TAG_RE.findall(text)
    if tagged:
        items = [body.strip() for _name, body in tagged if body.strip()]
        parse = "tagged"
    else:
        items = [
            match.group(1)
            for line in text.splitlines()
            for match in [_NUMBERED.match(line)]
            if match
        ]
        parse = "exact"
    if len(items) == len(children):
        return items, parse
    lines = _strip_preamble(
        [line.strip() for line in text.splitlines() if line.strip()]
    )
    if len(lines) >= len(children):
        return lines[: len(children)], "fallback"
    raise TreeRunError(
        f"decompose step produced {len(items)} subtask(s) for {len(children)} "
        f"children; reply with a numbered list (or "
        f'<subtask name="...">...</subtask> blocks) containing exactly '
        f"{len(children)} items"
    )


def _decompose_prompt(node: dict, task: str) -> str:
    roster = "\n".join(f"- {c['name']}: {c['role']}" for c in node["children"])
    n = len(node["children"])
    return (
        f"Task:\n{task}\n\n"
        f"Your sub-agents:\n{roster}\n\n"
        f"Write one concrete subtask per sub-agent, in the same order as listed above. "
        f"Reply with a numbered list containing exactly {n} items, one line per item "
        f'(or {n} <subtask name="agent-name">...</subtask> blocks).'
    )


def _full_child_block(child: dict) -> str:
    """One child's raw result for the full-context synthesize prompt."""
    error = child.get("error")
    body = child["response"] if child["response"] else f"(sub-agent failed: {error})"
    return f"[{child['name']}]:\n{body}"


def _synthesize_prompt(task: str, child_records: list, summary_mode: bool) -> str:
    if summary_mode:
        # Parent sees compact summaries + pointers, never raw child outputs.
        parts = []
        for c in child_records:
            s = c.get("summary", {})
            toks = s.get("tokens", {})
            status = s.get("status", "?")
            if s.get("truncated"):
                status = f"{status} (truncated)"
            parts.append(
                f"[{c['name']} ({c['provider']}/{c['model']}, "
                f"status: {status})]:\n"
                f"{s.get('summary', '')}\n"
                f"Artifacts: {', '.join(s.get('artifacts', [])) or 'none'}\n"
                f"Tokens used by this sub-agent (in/out): "
                f"{toks.get('input', 0)}/{toks.get('output', 0)}\n"
                f"(full transcript for this agent is in transcript.json "
                f"under node '{c['name']}')"
            )
        results = "\n\n".join(parts)
    else:
        results = "\n\n".join(_full_child_block(c) for c in child_records)
    return (
        f"Original task:\n{task}\n\n"
        f"Sub-agent results:\n{results}\n\n"
        f"Combine these into one coherent final answer to the original task. "
        f"Cite which sub-agent each key point comes from (by name). If a "
        f"sub-agent failed, or its summary looks incomplete or contradicts "
        f"the others, say so explicitly instead of guessing."
    )


_SUMMARIZE_PROMPT = (
    "Summarize what you accomplished, the key decisions you made, and where "
    "the full output is stored. Keep it to ~150 words max."
)

_URL_RE = re.compile(r"https?://[^\s)>\"]+")
# file-ish tokens: src/foo.py, /tmp/bar.md, docs/a/b.json
_PATH_RE = re.compile(
    r"(?:^|[\s(\[{'\"])"
    r"(/?(?:[A-Za-z0-9_.\-]+/)*[A-Za-z0-9_.\-]+\.[A-Za-z0-9]{1,5})"
    r"(?=[\s)\]}\",'\";:.!?]|$)"
)


def _extract_artifacts(text: str | None) -> list:
    """Pull file paths and URLs out of a child's final answer (best effort)."""
    if not text:
        return []
    found = []
    for m in _URL_RE.finditer(text):
        found.append(m.group(0).rstrip(".,;)"))
    for m in _PATH_RE.finditer(text):
        found.append(m.group(1))
    seen, out = set(), []
    for item in found:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out[:10]


def _was_truncated(child_record: dict) -> bool:
    """True if any of the child's own calls was cut off by the token budget."""
    return any(
        call.get("finish_reason") in providers.TRUNCATED_REASONS
        for call in child_record.get("calls", [])
    )


def _resolve_summarizer(ctx: _Ctx, node: dict) -> tuple:
    """(provider, model) to use for summarizing this child's output.

    Per-node `summarizer:` wins over the root-level one; both default to the
    child's own provider/model.
    """
    override = node.get("summarizer") or ctx.summarizer
    if override:
        return override["provider"], override["model"]
    return node["provider"], node["model"]


def _error_summary(ctx: _Ctx, child_record: dict) -> None:
    """Build a summary for a failed child directly from its error.

    No provider call: paying to summarize an error string is waste, and a
    shared-fate summarize call could 429 alongside the failure.
    """
    _rollup(child_record, ctx.prices)
    child_record["summary"] = {
        "name": child_record["name"],
        "provider": child_record["provider"],
        "model": child_record["model"],
        "status": "failed",
        "summary": f"(sub-agent failed: {child_record.get('error')})",
        "artifacts": [],
        "tokens": dict(child_record["tokens"]),
        "truncated": False,
        "summary_fallback": False,
        "passthrough": False,
    }


def _make_summary(ctx: _Ctx, node: dict, child_record: dict) -> None:
    """Summarize a finished child into child_record["summary"].

    Honest economics: the summarize call only happens when the child's
    answer is long enough that compressing it saves the (usually pricier)
    parent from re-reading it. Short answers pass through raw — a summarize
    call would cost more than it saves. The summarize call itself is capped
    at ~300 output tokens and may use a cheaper `summarizer` model.
    """
    answer = child_record.get("response") or ""
    artifacts = _extract_artifacts(answer)
    _rollup(child_record, ctx.prices)
    status = child_record.get("status", "unknown")
    if len(answer) <= ctx.summary_threshold:
        child_record["summary"] = {
            "name": node["name"],
            "provider": node["provider"],
            "model": node["model"],
            "status": status,
            "summary": answer or "(no output)",
            "artifacts": artifacts,
            "tokens": dict(child_record["tokens"]),
            "truncated": _was_truncated(child_record),
            "summary_fallback": False,
            "passthrough": True,
        }
        return
    summarizer_provider, summarizer_model = _resolve_summarizer(ctx, node)
    user = f"{_SUMMARIZE_PROMPT}\n\nYour completed work:\n{answer}"
    fallback = False
    try:
        summary_text = _call(
            ctx, node, user, "summarize", child_record,
            max_tokens=SUMMARIZE_MAX_TOKENS,
            provider=summarizer_provider, model=summarizer_model,
        )
    except Exception:  # noqa: BLE001 - fallback below, never crash the run
        summary_text = (answer[:500] + ("..." if len(answer) > 500 else "")) \
            or "(no output)"
        fallback = True
    _rollup(child_record, ctx.prices)  # include the summarize call itself
    child_record["summary"] = {
        "name": node["name"],
        "provider": node["provider"],
        "model": node["model"],
        "status": status,
        "summary": summary_text,
        "artifacts": artifacts,
        "tokens": dict(child_record["tokens"]),
        "truncated": _was_truncated(child_record),
        "summary_fallback": fallback,
        "passthrough": False,
    }


def _safe_summary(ctx: _Ctx, node: dict, child_record: dict) -> None:
    """_make_summary that can never raise: sibling summaries are isolated.

    _make_summary already falls back internally on provider failure; this
    wrapper is defense-in-depth for anything unexpected (a malformed record,
    a broken provider fn) so one bad child can never take down the rest.
    """
    try:
        _make_summary(ctx, node, child_record)
    except Exception as exc:  # noqa: BLE001 - last-resort fallback below
        child_record["summary"] = {
            "name": node["name"],
            "provider": node["provider"],
            "model": node["model"],
            "status": "failed",
            "summary": f"(summary unavailable: {exc})",
            "artifacts": [],
            "tokens": dict(child_record.get("tokens",
                                            {"input": 0, "output": 0,
                                             "cost_usd": 0.0})),
            "truncated": False,
            "summary_fallback": True,
            "passthrough": False,
        }


def _run_children(ctx: _Ctx, node: dict, subtasks: list, child_records: list) -> None:
    """Run children in parallel; a child failure is recorded, not fatal.

    With fail_fast, the first child failure aborts the wait: pending futures
    are cancelled, in-flight children finish so the transcript stays
    complete, then the failure propagates. A BaseException (Ctrl-C) cancels
    everything pending immediately and propagates.
    """
    pool = ThreadPoolExecutor(
        max_workers=min(len(child_records), ctx.max_concurrency))
    first_failure = None
    try:
        futures = [
            pool.submit(_run_child, ctx, child, subtask, crec)
            for child, subtask, crec in zip(node["children"], subtasks,
                                           child_records)
        ]
        for future in futures:
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - recorded on the child
                if ctx.fail_fast and first_failure is None:
                    first_failure = exc
                    break
                # otherwise the parent keeps going in degraded mode
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True, cancel_futures=True)
    if first_failure is not None:
        raise first_failure


def _summarize_children(ctx: _Ctx, node: dict, child_records: list) -> None:
    """Summarize children concurrently; failures are isolated per child."""
    todo = []
    for child, crec in zip(node["children"], child_records):
        if crec["status"] == "error":
            _error_summary(ctx, crec)  # no provider call for failed children
        else:
            todo.append((child, crec))
    if not todo:
        return
    pool = ThreadPoolExecutor(max_workers=min(len(todo), ctx.max_concurrency))
    try:
        futures = [pool.submit(_safe_summary, ctx, child, crec)
                   for child, crec in todo]
        for future in futures:
            future.result()  # _safe_summary never raises
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)


def _run_node(ctx: _Ctx, node: dict, task: str, record: dict) -> None:
    """Fill in `record` by executing `node`; raises on failure (record keeps the error)."""
    if not node["children"]:
        record["response"] = _call(ctx, node, f"Task:\n{task}", "leaf", record)
        return
    decomposition = _call(ctx, node, _decompose_prompt(node, task),
                          "decompose", record)
    subtasks, parse = _parse_subtasks(decomposition, node["children"])
    record["subtask_parse"] = parse
    child_records = [_new_record(child, subtask=st)
                     for child, st in zip(node["children"], subtasks)]
    record["children"] = child_records  # attached before threads run: partial transcripts survive
    _run_children(ctx, node, subtasks, child_records)
    if all(c["status"] == "error" for c in child_records):
        raise TreeRunError(
            f"all {len(child_records)} sub-agent(s) of '{node['name']}' failed; "
            f"skipping synthesize (refusing to build an answer from nothing)"
        )
    if ctx.summary_mode:
        # Summarize long child answers so the synthesize step sees compact
        # summaries instead of full child transcripts. Siblings summarize
        # concurrently; _safe_summary isolates failures per child.
        _summarize_children(ctx, node, child_records)
    record["response"] = _call(
        ctx, node, _synthesize_prompt(task, child_records, ctx.summary_mode),
        "synthesize", record)


def _run_child(ctx: _Ctx, child: dict, subtask: str, record: dict) -> None:
    started = time.monotonic()
    try:
        _run_node(ctx, child, subtask, record)
        record["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - recorded verbatim in the transcript
        record["status"] = "error"
        record["error"] = str(exc)
        raise
    finally:
        _finish(record, started)


def _call_cost_usd(provider: str | None, usage: dict, prices: dict) -> float:
    """Estimated USD for one call from the (approximate) price table."""
    table = prices.get(provider) if provider else None
    if not table:
        return 0.0
    return (
        usage.get("input_tokens", 0) * table.get("input", 0)
        + usage.get("output_tokens", 0) * table.get("output", 0)
    ) / 1_000_000


def _rollup(record: dict, prices: dict) -> dict:
    """Sum token usage and estimated cost over a node's own calls plus all
    descendants. Mutates record["tokens"] in place (idempotent) and returns
    it, so the root's "tokens" is the grand total of the whole run."""
    total_in = sum(c.get("usage", {}).get("input_tokens", 0)
                   for c in record.get("calls", []))
    total_out = sum(c.get("usage", {}).get("output_tokens", 0)
                    for c in record.get("calls", []))
    total_cost = sum(
        _call_cost_usd(c.get("provider"), c.get("usage", {}), prices)
        for c in record.get("calls", [])
    )
    for child in record.get("children", []):
        child_totals = _rollup(child, prices)
        total_in += child_totals["input"]
        total_out += child_totals["output"]
        total_cost += child_totals["cost_usd"]
    record["tokens"] = {"input": total_in, "output": total_out,
                        "cost_usd": round(total_cost, 6)}
    return record["tokens"]


def _mark_cancelled(record: dict) -> None:
    """Mark a record and every descendant still 'running' as 'cancelled'."""
    if record.get("status") == "running":
        record["status"] = "cancelled"
    for child in record.get("children", []):
        _mark_cancelled(child)


def execute_tree(node: dict, task: str, provider_fn=None, max_tokens: int = 1024,
                 summary_mode: bool = True, fail_fast: bool = False,
                 max_concurrency: int | None = None,
                 timeout: int | float | None = None) -> dict:
    """Run the tree on a task. Always returns the root record (errors and
    cancellations are recorded, never raised — not even KeyboardInterrupt).

    summary_mode=True (default): long child answers are compressed by one
    capped summarize call each (cheap summarizer optional); short answers
    pass through raw. summary_mode=False passes full child outputs up (see
    --full-context). fail_fast=True aborts on the first child failure.
    max_concurrency/timeout default to the config's root keys (8 and 120).
    """
    mc = max_concurrency or node.get(
        "max_concurrency", _config.DEFAULT_MAX_CONCURRENCY)
    to = timeout or node.get("timeout", _config.DEFAULT_TIMEOUT)
    prices = node.get("prices")
    if not prices:
        prices = {p: dict(v) for p, v in providers.PRICES.items()}
    ctx = _Ctx(fn=None, max_tokens=max_tokens, summary_mode=summary_mode,
               fail_fast=fail_fast, max_concurrency=mc, timeout=to,
               prices=prices, summarizer=node.get("summarizer"))

    def default_fn(provider, model, system, user, kind, max_tokens):
        return providers.call(provider, model, system, user,
                              max_tokens=max_tokens, timeout=ctx.timeout)

    ctx.fn = provider_fn or default_fn
    started = time.monotonic()
    record = _new_record(node)
    try:
        _run_node(ctx, node, task, record)
        record["status"] = "ok"
    except KeyboardInterrupt:
        _mark_cancelled(record)
        record["status"] = "cancelled"
        record["error"] = "cancelled by user (KeyboardInterrupt)"
    except Exception as exc:  # noqa: BLE001 - recorded verbatim in the transcript
        record["status"] = "error"
        record["error"] = str(exc)
    finally:
        _finish(record, started)
        _rollup(record, ctx.prices)  # children are all settled by now
    return record


_CHILD_RE = re.compile(r"^-\s*(\S+?)\s*:", re.MULTILINE)
_RESULT_RE = re.compile(r"^\[.+\]:", re.MULTILINE)
_NAME_RE = re.compile(r"You are the '([^']+)' agent")


# Fixed fake output tokens per call kind in dry-run mode.
_DRY_RUN_OUTPUT = {"leaf": 60, "decompose": 40, "synthesize": 80, "summarize": 50}


def dry_run_provider(provider: str, model: str, system: str, user: str, kind: str,
                     max_tokens: int = 1024) -> tuple:
    """Deterministic fake provider: no API keys, no cost. For demos and tests.

    Returns (text, usage, finish_reason). Fake usage is deterministic:
    input_tokens is ~1 token per 4 prompt chars, output_tokens is fixed per
    call kind — so --dry-run exercises the whole token-accounting path for
    $0.00. Leaf answers are deliberately long (>2400 chars) so dry-run also
    exercises summary mode end to end. These are illustrative mock numbers,
    not measurements of any real model.
    """
    name = _NAME_RE.search(system)
    name = name.group(1) if name else "agent"
    usage = {
        "input_tokens": len(system + user) // 4,
        "output_tokens": _DRY_RUN_OUTPUT.get(kind, 50),
    }
    finish_reason = "stop"
    if kind == "decompose":
        children = _CHILD_RE.findall(user.split("Your sub-agents:")[-1])
        task_line = user.splitlines()[1] if len(user.splitlines()) > 1 else user[:60]
        lines = [
            f"{i + 1}) Do the {child} part of: {task_line[:60]}"
            for i, child in enumerate(children)
        ]
        return "\n".join(lines), usage, finish_reason
    if kind == "summarize":
        return (
            f"[{name}] dry-run summary: finished the assigned subtask; "
            f"key decisions and artifacts are in the full dry-run transcript."
        ), usage, finish_reason
    if kind == "synthesize":
        n = len(_RESULT_RE.findall(user))
        return (
            f"[{name}] dry-run synthesis: combined {n} sub-agent result(s) "
            f"into one final answer (providers: mixed, cost: $0.00)."
        ), usage, finish_reason
    text = (
        f"[{name}] dry-run result from {provider}/{model}: "
        f"{user[:80].replace(chr(10), ' ')} " + "pad " * 700
    ).strip()
    return text, usage, finish_reason
