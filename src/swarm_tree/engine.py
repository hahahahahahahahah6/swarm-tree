"""Tree execution: decompose -> parallel children -> synthesize.

A leaf node calls its provider once with (role + task) and returns text.

An internal node:
  1. calls its provider with (role + task + the children's roles) to produce
     one subtask per child (decompose step);
  2. runs its children in parallel (threads); a failed child is recorded
     (status/error) and the parent keeps going in degraded mode instead of
     aborting — the synthesize step is told explicitly which sub-agent
     failed;
  3. summarizes each child concurrently with one cheap call each to that
     child's own provider/model (summary mode, default on) — or skips this
     with summary_mode=False; one child's summarize failure never affects
     the others;
  4. calls its provider again with the child results to produce the final
     answer (synthesize step). In summary mode the synthesize prompt sees
     only the compact summaries, never the raw child outputs; full outputs
     stay in the transcript with pointers.

Every provider call is recorded in the node transcript (prompts, response,
timings) so the whole run can be audited afterwards.
"""

from __future__ import annotations

import datetime
import re
import time
from concurrent.futures import ThreadPoolExecutor

from . import providers

# provider_fn(provider, model, system, user, kind) -> (text, usage)
# kind is one of "leaf", "decompose", "synthesize"; usage is
# {"input_tokens": int, "output_tokens": int}.


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
        "calls": [],  # each: {kind, system, user, response, duration_s, usage}
        "tokens": {"input": 0, "output": 0},  # own calls + descendants, after rollup
        "response": None,
        "error": None,
        "children": [],
    }


def _finish(record: dict, started: float) -> None:
    record["duration_s"] = round(time.monotonic() - started, 3)


def _call(fn, node: dict, user_prompt: str, kind: str, record: dict, max_tokens: int) -> str:
    system = f"You are the '{node['name']}' agent. {node['role']}"
    started = time.monotonic()
    response, usage = fn(node["provider"], node["model"], system, user_prompt, kind)
    duration = round(time.monotonic() - started, 3)
    record["calls"].append(
        {"kind": kind, "system": system, "user": user_prompt,
         "response": response, "duration_s": duration,
         "usage": usage}
    )
    return response


_NUMBERED = re.compile(r"^\s*\d+[.)]\s*(.+?)\s*$")


def _parse_subtasks(text: str, children: list) -> list:
    """Split a decompose response into one subtask per child.

    Falls back to distributing non-empty lines (padding with the last line)
    when the model doesn't produce an exact numbered list.
    """
    items = [_NUMBERED.match(line).group(1) for line in text.splitlines()
             if _NUMBERED.match(line)]
    if len(items) != len(children):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) >= len(children):
            items = lines[: len(children)]
        elif lines:
            items = lines + [lines[-1]] * (len(children) - len(lines))
        else:
            items = [f"Subtask for {c['name']}" for c in children]
    return items


def _decompose_prompt(node: dict, task: str) -> str:
    roster = "\n".join(f"- {c['name']}: {c['role']}" for c in node["children"])
    return (
        f"Task:\n{task}\n\n"
        f"Your sub-agents:\n{roster}\n\n"
        f"Write one concrete subtask per sub-agent, in the same order as listed above. "
        f"Reply with a numbered list containing exactly {len(node['children'])} items, "
        f"one line per item."
    )


def _synthesize_prompt(task: str, child_records: list, summary_mode: bool) -> str:
    if summary_mode:
        # Parent sees compact summaries + pointers, never raw child outputs.
        parts = []
        for c in child_records:
            s = c.get("summary", {})
            toks = s.get("tokens", {})
            parts.append(
                f"[{c['name']} ({c['provider']}/{c['model']}, "
                f"status: {s.get('status', '?')})]:\n"
                f"{s.get('summary', '')}\n"
                f"Artifacts: {', '.join(s.get('artifacts', [])) or 'none'}\n"
                f"Tokens used by this sub-agent (in/out): "
                f"{toks.get('input', 0)}/{toks.get('output', 0)}\n"
                f"(full transcript for this agent is in transcript.json "
                f"under node '{c['name']}')"
            )
        results = "\n\n".join(parts)
    else:
        results = "\n\n".join(
            f"[{c['name']}]:\n"
            f"{c['response'] if c['response'] else f'(sub-agent failed: {c.get("error")})'}"
            for c in child_records
        )
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


def _make_summary(fn, node: dict, child_record: dict, max_tokens: int) -> None:
    """Summarize a finished child into child_record["summary"].

    One extra call to the child's own provider/model. On failure the summary
    falls back to a 500-char truncation of the child's answer — the run never
    crashes because of a summarize call. The summarize call's own usage is
    recorded on the child's calls list, so it shows up in token rollups.

    A child that failed still gets a summary: its status is reported as
    "failed" and the summary notes the error, so the parent's synthesize
    step can account for it explicitly instead of silently missing it.
    """
    answer = child_record.get("response") or ""
    if not answer and child_record.get("error"):
        answer = f"(sub-agent failed: {child_record['error']})"
    artifacts = _extract_artifacts(answer)
    system = f"You are the '{node['name']}' agent. {node['role']}"
    user = f"{_SUMMARIZE_PROMPT}\n\nYour completed work:\n{answer}"
    started = time.monotonic()
    fallback = False
    try:
        response, usage = fn(node["provider"], node["model"], system, user,
                             "summarize")
        child_record["calls"].append(
            {"kind": "summarize", "system": system, "user": user,
             "response": response, "duration_s": round(time.monotonic() - started, 3),
             "usage": usage}
        )
        summary_text = response
    except Exception:  # noqa: BLE001 - fallback below, never crash the run
        summary_text = (answer[:500] + ("..." if len(answer) > 500 else "")) \
            or "(no output)"
        fallback = True
    _rollup(child_record)  # subtree tokens, including this summarize call
    status = child_record.get("status", "unknown")
    if status == "error":
        status = "failed"  # the synthesize prompt speaks "failed", not "error"
    child_record["summary"] = {
        "name": node["name"],
        "provider": node["provider"],
        "model": node["model"],
        "status": status,
        "summary": summary_text,
        "artifacts": artifacts,
        "tokens": dict(child_record["tokens"]),
        "summary_fallback": fallback,
    }


def _safe_summary(fn, node: dict, child_record: dict, max_tokens: int) -> None:
    """_make_summary that can never raise: sibling summaries are isolated.

    _make_summary already falls back internally on provider failure; this
    wrapper is defense-in-depth for anything unexpected (a malformed record,
    a broken provider fn) so one bad child can never take down the rest.
    """
    try:
        _make_summary(fn, node, child_record, max_tokens)
    except Exception as exc:  # noqa: BLE001 - last-resort fallback below
        child_record["summary"] = {
            "name": node["name"],
            "provider": node["provider"],
            "model": node["model"],
            "status": "failed",
            "summary": f"(summary unavailable: {exc})",
            "artifacts": [],
            "tokens": dict(child_record.get("tokens", {"input": 0, "output": 0})),
            "summary_fallback": True,
        }


def _run_node(node: dict, task: str, fn, max_tokens: int, record: dict,
              summary_mode: bool) -> None:
    """Fill in `record` by executing `node`; raises on failure (record keeps the error)."""
    if not node["children"]:
        record["response"] = _call(fn, node, f"Task:\n{task}", "leaf", record, max_tokens)
        return
    decomposition = _call(fn, node, _decompose_prompt(node, task),
                          "decompose", record, max_tokens)
    subtasks = _parse_subtasks(decomposition, node["children"])
    child_records = [_new_record(child, subtask=st)
                     for child, st in zip(node["children"], subtasks)]
    record["children"] = child_records  # attached before threads run: partial transcripts survive
    with ThreadPoolExecutor(max_workers=len(node["children"])) as pool:
        futures = [
            (child["name"], pool.submit(_run_child, child, subtask, fn, max_tokens,
                                        crec, summary_mode))
            for child, subtask, crec in zip(node["children"], subtasks, child_records)
        ]
        for name, future in futures:
            try:
                future.result()
            except Exception:
                # Recorded on the child (status/error); the parent keeps
                # going in degraded mode and the synthesize step is told
                # which sub-agent failed instead of the whole run dying.
                pass
    if summary_mode:
        # One cheap summarize call per child (on the child's own
        # provider/model) so the synthesize step sees compact summaries
        # instead of full child transcripts. Siblings summarize
        # concurrently; _safe_summary isolates failures per child.
        with ThreadPoolExecutor(max_workers=len(child_records)) as pool:
            futures = [
                pool.submit(_safe_summary, fn, child, crec, max_tokens)
                for child, crec in zip(node["children"], child_records)
            ]
            for future in futures:
                future.result()  # _safe_summary never raises
    record["response"] = _call(
        fn, node, _synthesize_prompt(task, child_records, summary_mode),
        "synthesize", record, max_tokens)


def _run_child(child: dict, subtask: str, fn, max_tokens: int, record: dict,
               summary_mode: bool) -> None:
    started = time.monotonic()
    try:
        _run_node(child, subtask, fn, max_tokens, record, summary_mode)
        record["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - recorded verbatim in the transcript
        record["status"] = "error"
        record["error"] = str(exc)
        raise
    finally:
        _finish(record, started)


def _rollup(record: dict) -> dict:
    """Sum token usage over a node's own calls plus all descendants.

    Mutates record["tokens"] in place and returns it, so the root's
    "tokens" is the grand total of the whole run.
    """
    total_in = sum(c.get("usage", {}).get("input_tokens", 0)
                   for c in record.get("calls", []))
    total_out = sum(c.get("usage", {}).get("output_tokens", 0)
                    for c in record.get("calls", []))
    for child in record.get("children", []):
        child_totals = _rollup(child)
        total_in += child_totals["input"]
        total_out += child_totals["output"]
    record["tokens"] = {"input": total_in, "output": total_out}
    return record["tokens"]


def execute_tree(node: dict, task: str, provider_fn=None, max_tokens: int = 1024,
                 summary_mode: bool = True) -> dict:
    """Run the tree on a task. Always returns the root record (errors are recorded).

    summary_mode=True (default): each internal node summarizes its children
    with one cheap call each, and the synthesize step sees only those
    summaries. summary_mode=False passes full child outputs up (the old
    behavior; see --full-context).
    """
    fn = provider_fn or (
        lambda provider, model, system, user, kind: providers.call(
            provider, model, system, user, max_tokens=max_tokens
        )
    )
    started = time.monotonic()
    record = _new_record(node)
    try:
        _run_node(node, task, fn, max_tokens, record, summary_mode)
        record["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - recorded verbatim in the transcript
        record["status"] = "error"
        record["error"] = str(exc)
    finally:
        _finish(record, started)
        _rollup(record)  # children are all settled by now
    return record


_CHILD_RE = re.compile(r"^-\s*(\S+?)\s*:", re.MULTILINE)
_RESULT_RE = re.compile(r"^\[.+\]:", re.MULTILINE)
_NAME_RE = re.compile(r"You are the '([^']+)' agent")


# Fixed fake output tokens per call kind in dry-run mode.
_DRY_RUN_OUTPUT = {"leaf": 60, "decompose": 40, "synthesize": 80, "summarize": 50}


def dry_run_provider(provider: str, model: str, system: str, user: str, kind: str) -> tuple:
    """Deterministic fake provider: no API keys, no cost. For demos and tests.

    Returns (text, usage). Fake usage is deterministic: input_tokens is
    ~1 token per 4 prompt chars, output_tokens is fixed per call kind —
    so --dry-run exercises the whole token-accounting path for $0.00.
    """
    name = _NAME_RE.search(system)
    name = name.group(1) if name else "agent"
    usage = {
        "input_tokens": len(system + user) // 4,
        "output_tokens": _DRY_RUN_OUTPUT.get(kind, 50),
    }
    if kind == "decompose":
        children = _CHILD_RE.findall(user.split("Your sub-agents:")[-1])
        task_line = user.splitlines()[1] if len(user.splitlines()) > 1 else user[:60]
        lines = [
            f"{i + 1}) Do the {child} part of: {task_line[:60]}"
            for i, child in enumerate(children)
        ]
        return "\n".join(lines), usage
    if kind == "summarize":
        return (
            f"[{name}] dry-run summary: finished the assigned subtask; "
            f"key decisions and artifacts are in the full dry-run transcript."
        ), usage
    if kind == "synthesize":
        n = len(_RESULT_RE.findall(user))
        return (
            f"[{name}] dry-run synthesis: combined {n} sub-agent result(s) "
            f"into one final answer (providers: mixed, cost: $0.00)."
        ), usage
    return (
        f"[{name}] dry-run result from {provider}/{model}: "
        f"{user[:80].replace(chr(10), ' ')}"
    ), usage
