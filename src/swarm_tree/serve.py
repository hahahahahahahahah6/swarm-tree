"""swarm-tree serve mode: persistent, scheduled agent trees.

A node may carry a `schedule` ("30m", "2h", "08:05") and a `task`. Under
`swarm-tree serve` the tree stays resident: each scheduled node runs its own
task on its own cadence, transcripts are persisted per run, and (when
`rollup` is on) the root node aggregates the tick's fresh outputs into a
daily markdown briefing — the "CEO briefing" analog.

Everything is standard-library only. State lives in <serve_dir>/state.json,
so a restart resumes the schedule without replaying missed ticks.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
import time

from . import config, engine

STATE_FILE = "state.json"
RUNS_DIR = "runs"
REPORT_PREFIX = "report-"


def scheduled_nodes(node: dict) -> list:
    """Return [(name, node)] for every enabled node carrying a schedule."""
    found = []
    if node.get("enabled", True) and node.get("schedule"):
        found.append((node["name"], node))
    for child in node["children"]:
        found.extend(scheduled_nodes(child))
    return found


def _daily_next(hour: int, minute: int, now: float) -> float:
    """Next occurrence of HH:MM strictly after `now` (local time)."""
    local = datetime.datetime.fromtimestamp(now).astimezone()
    candidate = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate.timestamp() <= now:
        candidate += datetime.timedelta(days=1)
    return candidate.timestamp()


def next_run_after(spec: str, last_run: float | None, now: float) -> float:
    """Compute the next run time for a schedule spec.

    First-ever run (last_run is None) fires immediately. Missed ticks are
    skipped, never backfilled: an interval whose slot already passed is
    rescheduled one full interval from now, and a daily time goes to its
    next occurrence.
    """
    kind, value = config.parse_schedule(spec)
    if last_run is None:
        return now
    if kind == "interval":
        nxt = last_run + value
        if nxt <= now:
            nxt = now + value
        return nxt
    return _daily_next(value[0], value[1], now)


def due_nodes(node: dict, state: dict, now: float) -> list:
    """Return [(name, node)] whose schedule slot has arrived.

    First-ever runs fire immediately. A node whose last run is older than
    one interval (or whose daily slot passed since its last run) is due —
    it runs once, then resumes its normal cadence (missed ticks are not
    backfilled; see next_run_after).
    """
    due = []
    for name, sub in scheduled_nodes(node):
        kind, value = config.parse_schedule(sub["schedule"])
        last_run = state.get(name, {}).get("last_run")
        if last_run is None:
            due.append((name, sub))
        elif kind == "interval":
            if last_run + value <= now:
                due.append((name, sub))
        else:  # daily: due if a scheduled slot passed since the last run
            if _daily_next(value[0], value[1], last_run) <= now:
                due.append((name, sub))
    return due


def load_state(serve_dir: str) -> dict:
    path = os.path.join(serve_dir, STATE_FILE)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_state(serve_dir: str, state: dict) -> None:
    os.makedirs(serve_dir, exist_ok=True)
    path = os.path.join(serve_dir, STATE_FILE)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utcnow_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


TRIGGER_TIMEOUT = 30  # seconds for a trigger check command


def run_trigger_check(command: str) -> tuple:
    """Run a node's `trigger` shell command.

    Returns (ok, stdout_hash, note). ok=False means the check itself failed
    (nonzero exit, timeout, OSError) — callers must FAIL OPEN and run the
    node anyway. ok=True carries the sha256 of stdout for change detection.
    """
    try:
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=TRIGGER_TIMEOUT)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return (False, None, f"trigger check failed: {exc}")
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        detail = f"exit {proc.returncode}" + (f": {err[:200]}" if err else "")
        return (False, None, f"trigger check failed: {detail}")
    digest = hashlib.sha256(proc.stdout.encode("utf-8")).hexdigest()
    return (True, digest, "")


def harvest_decisions(response_text) -> list:
    """Collect `DECISION:` lines from a node's output text."""
    found = []
    for line in (response_text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("DECISION:"):
            text = stripped[len("DECISION:"):].strip()
            if text:
                found.append(text)
    return found


def get_queue(state: dict) -> list:
    """Pending owner-action items: [{"node", "text", "added_at"}]."""
    queue = state.get("queue", [])
    return queue if isinstance(queue, list) else []


def queue_add(state: dict, name: str, texts: list) -> int:
    """Append decision texts to the queue, deduped on (node, text).

    Returns the number of items actually added."""
    queue = get_queue(state)
    seen = {(item.get("node"), item.get("text")) for item in queue}
    added = 0
    for text in texts:
        if (name, text) not in seen:
            queue.append({"node": name, "text": text,
                          "added_at": _utcnow_iso()})
            seen.add((name, text))
            added += 1
    state["queue"] = queue
    return added


def queue_clear(state: dict) -> int:
    """Empty the owner-action queue. Returns the number of items removed."""
    removed = len(get_queue(state))
    state["queue"] = []
    return removed


def _node_spend(state: dict, name: str) -> float:
    spend = state.get("_spend", {})
    if not isinstance(spend, dict):
        return 0.0
    try:
        return float(spend.get(name, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _add_spend(state: dict, name: str, cost_usd: float) -> float:
    """Accumulate a node's lifetime spend. Returns the new total."""
    spend = state.get("_spend")
    if not isinstance(spend, dict):
        spend = {}
        state["_spend"] = spend
    try:
        cost = float(cost_usd or 0.0)
    except (TypeError, ValueError):
        cost = 0.0
    total = round(_node_spend(state, name) + cost, 6)
    spend[name] = total
    return total


def _write_run_transcript(serve_dir: str, name: str, task: str,
                          record: dict, dry_run: bool, version: str) -> str:
    run_dir = os.path.join(serve_dir, RUNS_DIR, name)
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, f"{_stamp()}.json")
    transcript = {
        "tool": "swarm-tree",
        "version": version,
        "mode": "serve",
        "node": name,
        "task": task,
        "dry_run": dry_run,
        "generated_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "root": record,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(transcript, fh, indent=2, ensure_ascii=False)
    return path


def _excerpt(text, limit: int = 800) -> str:
    text = (text or "").strip()
    if len(text) > limit:
        return text[:limit] + "…"
    return text


def run_tick(node: dict, provider_fn, *, serve_dir: str, dry_run: bool,
             version: str, max_tokens: int = 1024, summary_mode: bool = True,
             fail_fast: bool = False, max_concurrency: int | None = None,
             timeout=None, now: float | None = None) -> list:
    """Run every due scheduled node once. Returns a list of tick results:

    [{"name", "schedule", "task", "status", "transcript", "record"}].
    State is updated and persisted as each node finishes.
    """
    now = time.time() if now is None else now
    state = load_state(serve_dir)
    results = []
    for name, sub in due_nodes(node, state, now):
        task = sub.get("task") or f"Run your scheduled check for '{name}'."
        entry = state.get(name, {})
        if not isinstance(entry, dict):
            entry = {}

        # --- v0.4 budget gate: cumulative spend vs per-node budget_usd ---
        budget = sub.get("budget_usd")
        spent = _node_spend(state, name)
        if budget is not None and spent >= budget:
            ran_at = time.time()
            state[name] = {
                "schedule": sub["schedule"],
                "last_run": ran_at,
                "next_run": next_run_after(sub["schedule"], ran_at, ran_at),
                "last_status": "budget-exceeded",
                "last_transcript": entry.get("last_transcript"),
                "trigger": entry.get("trigger"),
                "trigger_hash": entry.get("trigger_hash"),
            }
            save_state(serve_dir, state)
            print(f"tick: {name} -> budget exceeded "
                  f"(spent ${spent:.6f} >= budget ${budget:.6f}); skipping")
            results.append({
                "name": name,
                "schedule": sub["schedule"],
                "task": task,
                "status": "budget-exceeded",
                "transcript": None,
                "record": {"status": "budget-exceeded", "response": "",
                           "tokens": {"input": 0, "output": 0,
                                      "cost_usd": 0.0}},
            })
            continue

        # --- v0.4 trigger gate: skip the LLM call when nothing changed ---
        trigger_cmd = sub.get("trigger")
        trigger_outcome = None
        trigger_note = ""
        if trigger_cmd:
            ok, digest, note = run_trigger_check(trigger_cmd)
            if ok and digest == entry.get("trigger_hash"):
                ran_at = time.time()
                record = {"status": "skipped",
                          "response": "trigger skipped: stdout unchanged "
                                      f"since last tick ({trigger_cmd})",
                          "tokens": {"input": 0, "output": 0,
                                     "cost_usd": 0.0}}
                path = _write_run_transcript(serve_dir, name, task, record,
                                             dry_run, version)
                state[name] = {
                    "schedule": sub["schedule"],
                    "last_run": ran_at,
                    "next_run": next_run_after(sub["schedule"], ran_at, ran_at),
                    "last_status": "skipped",
                    "last_transcript": path,
                    "trigger": "skipped",
                    "trigger_hash": digest,
                }
                save_state(serve_dir, state)
                print(f"tick: {name} -> skipped (trigger: stdout unchanged)")
                results.append({
                    "name": name,
                    "schedule": sub["schedule"],
                    "task": task,
                    "status": "skipped",
                    "transcript": path,
                    "record": record,
                    "trigger": "skipped",
                })
                continue
            if ok:
                entry["trigger_hash"] = digest
                trigger_outcome = "ran"
            else:
                # Fail open: a broken check must never silence the node.
                trigger_outcome = "check-failed"
                trigger_note = note
                print(f"tick: {name} -> trigger check failed, "
                      f"running anyway ({note})")

        record = engine.execute_tree(
            sub, task, provider_fn, max_tokens,
            summary_mode=summary_mode, fail_fast=fail_fast,
            max_concurrency=max_concurrency, timeout=timeout)
        if trigger_note:
            record = dict(record)
            record["trigger_note"] = trigger_note
        # v0.4 owner queue: harvest DECISION: lines from the node output.
        n_new = queue_add(state, name,
                          harvest_decisions(record.get("response")))
        if n_new:
            print(f"tick: {name} -> queued {n_new} decision(s) for owner")
        # v0.4 spend tracking (cumulative, survives restarts).
        total = _add_spend(state, name,
                           record.get("tokens", {}).get("cost_usd", 0.0))
        path = _write_run_transcript(serve_dir, name, task, record,
                                     dry_run, version)
        ran_at = time.time()
        state[name] = {
            "schedule": sub["schedule"],
            "last_run": ran_at,
            "next_run": next_run_after(sub["schedule"], ran_at, ran_at),
            "last_status": record.get("status"),
            "last_transcript": path,
            "trigger": trigger_outcome or entry.get("trigger"),
            "trigger_hash": entry.get("trigger_hash"),
            "spend_usd": total,
        }
        save_state(serve_dir, state)  # persist per node: a crash loses nothing
        results.append({
            "name": name,
            "schedule": sub["schedule"],
            "task": task,
            "status": record.get("status"),
            "transcript": path,
            "record": record,
            "trigger": trigger_outcome,
        })
    return results


def _rollup_node(root: dict) -> dict:
    """A synthetic single leaf node so the rollup reuses execute_tree."""
    return {
        "name": "rollup",
        "provider": root["provider"],
        "model": root["model"],
        "role": ("You are a chief-of-staff. Aggregate the department "
                 "briefings below into one concise markdown briefing: "
                 "headlines per department, notable changes, and items "
                 "needing the human's attention. Markdown only, no preamble."),
        "depth": 0,
        "max_children": 0,
        "summarizer": None,
        "schedule": None,
        "task": None,
        "enabled": True,
        "children": [],
    }


def write_rollup(root: dict, tick_results: list, provider_fn, *,
                 serve_dir: str, dry_run: bool, version: str,
                 max_tokens: int = 1024) -> str | None:
    """Aggregate this tick's fresh outputs into the daily markdown briefing.

    Returns the report path, or None when rollup is disabled / nothing new.
    A rollup failure is recorded but never breaks the tick.
    """
    if not root.get("rollup", True) or not tick_results:
        return None
    lines = []
    for res in tick_results:
        rec = res["record"]
        lines.append(
            f"## {res['name']} ({res['schedule']}) — {res['status']}\n\n"
            f"{_excerpt(rec.get('response'))}"
        )
        if rec.get("status") != "ok" and rec.get("error"):
            lines.append(f"\n_error: {_excerpt(rec.get('error'), 300)}_")
    prompt = ("Aggregate these department briefings into one markdown briefing:\n\n"
              + "\n\n".join(lines))
    try:
        record = engine.execute_tree(
            _rollup_node(root), prompt, provider_fn, max_tokens,
            summary_mode=False)
        body = record.get("response") or "(rollup produced no text)"
        if record.get("status") != "ok":
            body = f"_Rollup degraded ({record.get('error')}). Raw briefings:_\n\n" \
                   + "\n\n".join(lines)
    except Exception as exc:  # noqa: BLE001 - rollup must never break a tick
        body = ("_Rollup failed: %s. Raw briefings:_\n\n" % exc
                + "\n\n".join(lines))
    # v0.4 owner queue: pending decisions always close the briefing.
    queue = get_queue(load_state(serve_dir))
    if queue:
        items = "\n".join(
            f"- [ ] {it.get('node')}: {it.get('text')}" for it in queue)
        body += ("\n\n## 待你拍板 (Owner Action Queue)\n" + items)
    day = datetime.datetime.now().astimezone().strftime("%Y-%m-%d")
    path = os.path.join(serve_dir, f"{REPORT_PREFIX}{day}.md")
    os.makedirs(serve_dir, exist_ok=True)
    ts = datetime.datetime.now().astimezone().strftime("%H:%M")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"\n\n# Briefing — {day} {ts}\n\n{body}\n")
    return path


def format_status(node: dict, serve_dir: str, now: float | None = None) -> str:
    """Render the schedule table. Needs no API keys."""
    now = time.time() if now is None else now
    state = load_state(serve_dir)
    rows = []
    for name, sub in scheduled_nodes(node):
        entry = state.get(name, {})
        if not isinstance(entry, dict):
            entry = {}
        nxt = next_run_after(sub["schedule"], entry.get("last_run"), now)
        nxt_s = datetime.datetime.fromtimestamp(nxt).astimezone().strftime(
            "%m-%d %H:%M")
        last = entry.get("last_run")
        last_s = (datetime.datetime.fromtimestamp(last).astimezone().strftime(
            "%m-%d %H:%M") if last else "-")
        # v0.4: trigger outcome marker.
        trig = sub.get("trigger")
        trig_s = entry.get("trigger") or ("armed" if trig else "-")
        # v0.4: spend vs budget, flagged when the budget tripped.
        budget = sub.get("budget_usd")
        spent = _node_spend(state, name)
        if budget is not None:
            spend_s = f"${spent:.4f}/${budget:g}"
            if spent >= budget:
                spend_s += " BUDGET-EXCEEDED"
        else:
            spend_s = "-" if spent == 0 else f"${spent:.4f}"
        rows.append((name, sub["schedule"],
                     "on" if sub.get("enabled", True) else "off",
                     last_s, nxt_s, entry.get("last_status", "-"),
                     trig_s, spend_s))
    if not rows:
        return "no scheduled nodes (add a `schedule:` to a node to use serve mode)"
    header = (f"{'node':<20}{'schedule':<10}{'en':<4}{'last run':<12}"
              f"{'next run':<12}{'status':<16}{'trigger':<12}{'spend'}")
    out = [header]
    for r in rows:
        out.append(f"{r[0]:<20}{r[1]:<10}{r[2]:<4}{r[3]:<12}{r[4]:<12}"
                   f"{r[5]:<16}{r[6]:<12}{r[7]}")
    pending = get_queue(state)
    if pending:
        out.append(f"\nowner queue: {len(pending)} pending "
                   f"(swarm-tree queue to review)")
    return "\n".join(out)


def read_report(serve_dir: str, node_name: str | None = None) -> str:
    """Return the latest daily briefing, or one node's latest transcript."""
    if node_name:
        run_dir = os.path.join(serve_dir, RUNS_DIR, node_name)
        try:
            files = sorted(f for f in os.listdir(run_dir) if f.endswith(".json"))
        except OSError:
            return f"no runs recorded for node '{node_name}'"
        if not files:
            return f"no runs recorded for node '{node_name}'"
        path = os.path.join(run_dir, files[-1])
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        rec = data.get("root", {})
        return (f"# {node_name} — {data.get('generated_at', '?')} "
                f"({rec.get('status', '?')})\n\n"
                f"{_excerpt(rec.get('response'), 2000)}")
    try:
        reports = sorted(f for f in os.listdir(serve_dir)
                         if f.startswith(REPORT_PREFIX) and f.endswith(".md"))
    except OSError:
        return "no briefings yet (serve_dir is empty)"
    if not reports:
        return "no briefings yet"
    with open(os.path.join(serve_dir, reports[-1]), "r", encoding="utf-8") as fh:
        return fh.read()


def serve_loop(root: dict, provider_fn, *, serve_dir: str, dry_run: bool,
               version: str, max_tokens: int = 1024, summary_mode: bool = True,
               fail_fast: bool = False, max_concurrency: int | None = None,
               timeout=None, once: bool = False,
               on_tick=None) -> int:
    """The daemon loop. `on_tick` is a test seam: called with tick results."""
    scheduled = scheduled_nodes(root)
    if not scheduled:
        return 2
    print(f"serve: {len(scheduled)} scheduled node(s): "
          + ", ".join(f"{n} ({s['schedule']})" for n, s in scheduled))
    print(f"serve: state in {serve_dir} (Ctrl-C to stop, state is kept)")
    try:
        while True:
            results = run_tick(
                root, provider_fn, serve_dir=serve_dir, dry_run=dry_run,
                version=version, max_tokens=max_tokens,
                summary_mode=summary_mode, fail_fast=fail_fast,
                max_concurrency=max_concurrency, timeout=timeout)
            if results:
                for res in results:
                    print(f"tick: {res['name']} -> {res['status']} "
                          f"({res['transcript']})")
                path = write_rollup(
                    root, results, provider_fn, serve_dir=serve_dir,
                    dry_run=dry_run, version=version, max_tokens=max_tokens)
                if path:
                    print(f"briefing appended to {path}")
            if on_tick is not None:
                on_tick(results)
            if once:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nserve: stopped (state kept)")
    return 0
