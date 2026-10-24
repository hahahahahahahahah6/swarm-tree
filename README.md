# swarm-tree

Hierarchical multi-agent orchestration: **a tree of LLM agents, each node backed
by a different provider.**

Draw it on paper: a root orchestrator at the top, three child agents below it
(one running on Grok, one on Gemini, one on GPT), and one of those children
with two sub-agents of its own. That drawing is the whole idea.

- **Leaf nodes** do the work: one provider call with (role + task), returning text.
- **Internal nodes** coordinate: they ask their provider to split the task into
  one subtask per child (decompose), run all children **in parallel**, then ask
  their provider again to merge the child results into a final answer (synthesize).

Each node can use a different model from a different provider, so the tree is
a genuine multi-model swarm — the contrarian on Grok, the researcher on Gemini,
the planner on GPT, coordinated by Claude at the top.

The runtime core is standard-library only (no vendor SDKs, no frameworks).
Tree configs can be YAML — which requires PyYAML (`pip install "swarm-tree[yaml]"`)
— or plain JSON, which always works.

## Install

```bash
pip install swarm-tree
```

## Quick start

1. Describe your tree in `tree.yaml`:

```yaml
name: orchestrator
provider: anthropic
model: claude-sonnet-4-5
role: You are the top-level coordinator. Break hard problems into subtasks for
  your sub-agents, then merge their results into one coherent answer.
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
```

2. Set API keys (only for the providers your tree actually uses):

| Provider    | Env var            | API endpoint used                          |
|-------------|--------------------|--------------------------------------------|
| `anthropic` | `ANTHROPIC_API_KEY`| `api.anthropic.com/v1/messages`            |
| `openai`    | `OPENAI_API_KEY`   | `api.openai.com/v1/chat/completions`       |
| `gemini`    | `GEMINI_API_KEY`   | `generativelanguage.googleapis.com/v1beta` |
| `xai`       | `XAI_API_KEY`      | `api.x.ai/v1/chat/completions`             |
| `deepseek`  | `DEEPSEEK_API_KEY` | `api.deepseek.com/chat/completions`        |

3. Run it:

```bash
swarm-tree run --tree tree.yaml --task "Design a weekend hackathon project"
```

## Dry run (no keys, no cost)

```bash
swarm-tree run --tree tree.yaml --task "Design a weekend hackathon project" --dry-run
```

`--dry-run` swaps every provider call for a deterministic mock, so you can
demo and test the whole pipeline for $0.00 without any API keys.

## What you get

The finished tree rendered as ASCII — every node shows its name,
provider/model, status, timing, and a truncated result:

```
● orchestrator [anthropic/claude-sonnet-4-5] ok 0.0s 757/480 ~$0.01
  ↳ "[orchestrator] dry-run synthesis: combined 3 sub-agent result(s) into one…"
├── ● grok-agent [xai/grok-4] ok 0.0s 43/60 ~$0.0000
│   ↳ "[grok-agent] dry-run result from xai/grok-4: Do the grok-agent part of:…"
├── ● gemini-agent [gemini/gemini-2.5-pro] ok 0.0s 41/60 ~$0.0000
│   ↳ "[gemini-agent] dry-run result from gemini/gemini-2.5-pro: Do the gemini-…"
└── ● gpt-agent [openai/gpt-5] ok 0.0s 334/240 ~$0.0033
    ↳ "[gpt-agent] dry-run synthesis: combined 2 sub-agent result(s) into one…"
    ├── ● code-reviewer [deepseek/deepseek-chat] ok 0.0s 43/60 ~$0.0001
    │   ↳ "[code-reviewer] dry-run result from deepseek/deepseek-chat: Do the…"
    └── ● writer [anthropic/claude-sonnet-4-5] ok 0.0s 38/60 ~$0.0000
        ↳ "[writer] dry-run result from anthropic/claude-sonnet-4-5: Do the wri…"
```

The trailing `757/480 ~$0.01` on each node is input/output tokens plus the
estimated cost: each node's number is its own calls **plus all its
descendants**, so the root line is the grand total for the whole run. That
is the whole point — you can see at a glance which branch is eating your
quota (and your money).

Then the root's final answer, and a full JSON transcript
(`transcript.json`) with every node, every prompt, every raw response,
per-call timings, **and per-call token usage** (`usage.input_tokens` /
`usage.output_tokens` parsed from each provider's API response) — so the
whole run is auditable.

## Token accounting and cost estimates

Token usage is parsed from each provider's native response format and
normalized to `{"input_tokens": int, "output_tokens": int}`:

| Provider | Source fields |
|---|---|
| anthropic | `usage.input_tokens` / `usage.output_tokens` |
| openai, xai, deepseek | `usage.prompt_tokens` / `usage.completion_tokens` |
| gemini | `usageMetadata.promptTokenCount` / (`usageMetadata.candidatesTokenCount` + `thoughtsTokenCount`) |

Missing or malformed usage degrades to `0/0` instead of failing the run.

Each call's estimated USD cost is computed from a small built-in per-provider
price table (per 1M tokens, **approximate as of 2026-10** — real prices vary by
exact model and change often; override with the `prices:` root config key) and
rolled up next to the token counts, so the ASCII tree shows e.g.
`757/480 ~$0.01` per node and the root line is the grand total.

In `--dry-run` mode the mock provider returns deterministic fake usage
(`input_tokens` ≈ 1 per 4 prompt chars, `output_tokens` fixed per call
kind: 60 leaf / 40 decompose / 80 synthesize) so the entire accounting
path — parsing, rollup, rendering — is exercised for $0.00 with no keys.
These are illustrative mock numbers, not measurements of any real model.

## Summary mode

The classic multi-agent failure: every reviewer re-reads every worker's full
output, and the "coordination" costs more tokens than the work itself. By
default, swarm-tree avoids the worst of this.

After each child subtree finishes, **if its answer is long** (over ~2400
chars, about 2x the summary budget), the parent makes one extra call capped
at ~300 output tokens: *"summarize what you accomplished, key decisions, and
where the full output is stored (~150 words max)."* Short answers pass
through raw — a summarize call would cost more than it saves. The synthesize
step then sees only these compact summaries — plus extracted artifacts (file
paths / URLs mentioned in the child's answer), the subtree's token usage,
truncation flags, and a pointer to the full transcript — **never the raw
child outputs**. Nothing is lost: `transcript.json` still records every
prompt, every raw response, and every per-call token count.

The summarize call goes to a `summarizer` model you choose — a cheap model
compressing for an expensive parent is the whole point:

```yaml
# tree.yaml (root level; per-node override also accepted)
summarizer:
  provider: deepseek
  model: deepseek-chat
```

Default is each child's own provider/model. Failed children are never
summarized: their summary is built directly from the error string (no
provider call, no shared fate on 429s).

Control summary mode two ways:

```yaml
# tree.yaml (root level, default true)
summary_mode: false
```

```bash
swarm-tree run --tree tree.yaml --task "..." --full-context  # old behavior: full child outputs passed up
```

`swarm-tree validate` reports the effective setting for a config.

A failed sub-agent no longer kills the run: the child is recorded with its
error, its summary is marked `failed`, and the parent synthesizes an answer
from whatever the surviving children produced — flagging the failure
explicitly instead of silently working around it. If **every** child of a
node failed, the node fails instead of synthesizing an answer from nothing.
The CLI exits 3 and prints a warning when any sub-agent failed, so a
degraded run never looks clean (see exit codes below).

## How it saves tokens — honestly

An earlier version of this section claimed summary mode reduces total token
spend. That was backwards: each summarize call *adds* tokens (it re-reads
the child's full answer). The honest claim is narrower — summary mode is
**cost arbitrage**:

- A cheap `summarizer` model (e.g. DeepSeek) compresses a long child output
  into ~150 words, so an expensive parent (e.g. Claude) reads the summary
  instead of the full output. Total tokens go *up* slightly; total *cost*
  goes down when the summarizer is cheaper per token than the parent.
- Short answers skip summarization entirely (passthrough), so you never pay
  a summarize call that saves nothing.
- Truncation is surfaced, not hidden: if any call was cut off by the token
  budget, the child's summary is flagged `(truncated)` in the synthesize
  prompt.

Use the per-node `~$` cost estimates in the ASCII tree to check whether
summary mode is actually saving you money on your workload — if your parent
and children cost the same per token, `--full-context` may be cheaper.

## Other commands

```bash
swarm-tree validate --tree tree.yaml   # check a config without running it
swarm-tree example                      # print a starter tree.yaml
swarm-tree run --tree tree.yaml --task "..." --fail-fast         # abort on first child failure
swarm-tree run --tree tree.yaml --task "..." --max-concurrency 4 # tighter API budget
```

## Serve mode (v0.3): a tree that stays up

`run` executes one task and exits. `serve` keeps the tree resident: any node
with a `schedule` runs its own `task` on its own cadence, transcripts persist
per run, and the root rolls the tick's fresh outputs into a daily markdown
briefing — think "CEO briefing" for your agent team.

```yaml
name: ceo
provider: anthropic
model: claude-sonnet-4-5
role: You are the chief-of-staff. Aggregate department briefings.
rollup: true            # root aggregates each tick into a daily briefing
serve_dir: ./serve-data # transcripts, state, briefings land here
children:
  - name: shop-agent
    provider: deepseek
    model: deepseek-chat
    role: You watch shop metrics and flag anomalies.
    schedule: 30m       # interval: 30s | 5m | 2h | 1d
    task: Check today's orders, refunds and GMV; flag anything unusual.
  - name: research-agent
    provider: gemini
    model: gemini-2.5-pro
    role: You track the market each morning.
    schedule: "08:05"   # daily at 08:05 local time
    task: Summarize overnight market moves relevant to our watchlist.
  - name: helper        # no schedule: only aggregates, never fires alone
    provider: xai
    model: grok-4
    role: You answer ad-hoc questions from the CEO.
```

```bash
swarm-tree serve --tree tree.yaml --dry-run   # resident loop ($0.00)
swarm-tree serve --tree tree.yaml --once      # one tick, then exit (cron/systemd)
swarm-tree status --tree tree.yaml            # schedule table, no keys needed
swarm-tree report --tree tree.yaml            # latest daily briefing
swarm-tree report --tree tree.yaml --node shop-agent  # one node's latest output
```

Semantics: the first run fires immediately; afterwards each node follows its
schedule. Missed ticks are skipped, never backfilled (a restart resumes the
cadence instead of replaying the backlog). State lives in
`<serve_dir>/state.json`, transcripts in `<serve_dir>/runs/<node>/`, briefings
in `<serve_dir>/report-YYYY-MM-DD.md`. `enabled: false` takes a node out of
the rotation without deleting it. Ctrl-C stops the loop; state is always kept.

### Serve-mode governance trio (v0.4)

Three small mechanisms that keep a resident tree cheap and safe — all
configured per node, all stdlib, all visible in `swarm-tree status`:

- **Trigger condition** (`trigger:`): a shell command run before each tick.
  Its stdout is hashed; when the hash is unchanged since the last tick, the
  LLM call is skipped for $0.00. A failing check fails open — the node still
  runs, so a broken sensor can never silence an agent.
- **Owner action queue**: any node output line starting with `DECISION:` is
  harvested into a queue in `state.json`. Every briefing ends with a
  `## 待你拍板 (Owner Action Queue)` section, and `swarm-tree queue` lists
  what's pending (`--clear` empties it after you've handled the items).
  Agents prepare everything up to the last click; humans make the call.
- **Per-node budget** (`budget_usd:`): a cumulative USD cap per node,
  tracked in `state.json` across restarts. Once a node's lifetime spend
  reaches its budget, its ticks are skipped with a `budget-exceeded` status
  until you raise the cap. `status` shows `$spent/$budget` per node.

```bash
swarm-tree queue --tree tree.yaml            # pending owner decisions
swarm-tree queue --tree tree.yaml --clear    # empty the queue
```

### Operating loop (7 steps, from Simon Lin's e-commerce loop)

`examples/loop-7step.yaml` is a serve-mode template for a business that
runs itself in a loop, borrowed from Simon Lin's e-commerce AI loop —
not his code (which is not open source) and not his data, just the
operating shape that made it work:

| Loop step | swarm-tree mechanism |
|---|---|
| 1. Data first | `data-agent` pulls and cleans today's data; its `trigger` hashes the source, so a quiet day costs $0.00 |
| 2. Look at the numbers | `analyst-agent` flags anomalies with numbers, inventing none |
| 3. One recommendation only | `advisor-agent` proposes exactly one action, ending in a `DECISION:` line |
| 4. Owner decides | the owner queue is the gate; nothing proceeds without approval |
| 5. Worker prepares | `worker-agent` prepares execution up to the last click — it never publishes, charges, or sends on its own |
| 6. Independent verification | `verifier-agent` runs on a different provider from the worker and outputs `VERDICT: PASS/FAIL`; a FAIL is a veto via `DECISION:` |
| 7. Feedback is the next trigger | tomorrow's `data-agent` tick runs only when its trigger hash changed, so the loop re-runs on real change, not on a timer |

Two honest limits, kept on purpose. First, serve mode does not run a
sequential pipeline: the steps are ordered by staggered daily times
(08:00, 08:15, 08:30, 09:00, 09:30) and each node reads the previous
node's latest output — the engine does not enforce the order. Second,
the verifier is a soft veto, not a hard block: a FAIL lands in the
owner queue, and a human still has to honour it. What was deliberately
*not* copied: a heavy resident company platform, a shared data
platform, and any auto-publish step. The loop stays light on purpose.

```bash
swarm-tree serve --tree examples/loop-7step.yaml --dry-run --once
swarm-tree status --tree examples/loop-7step.yaml
swarm-tree queue --tree examples/loop-7step.yaml
```

## Web mode (v0.5): a website for your trees

`swarm-tree web` starts a local, stdlib-only web app (no frameworks, no new
dependencies):

```bash
swarm-tree web --port 8080     # http://127.0.0.1:8080
```

What you get in the browser:

- **Accounts** — sign up / log in (pbkdf2-hmac passwords, HMAC-signed
  session cookies, login rate limiting).
- **Tree editor** — create trees and edit them in the visual studio
  (the same tree builder from `studio/`, with a Save button wired to the
  server); one-click validation before saving.
- **Chat** — pick a tree, send a message, watch the agents work via a
  live progress stream; full transcripts are kept per run.
- **Image generation** — prompt → PNG, via OpenAI (`gpt-image-2.5-flare`)
  or Gemini (`gemini-3.1-flash-image`); your images are saved under
  `serve-data/images/`.
- **BYOK keys** — paste your own provider API keys in Settings; they are
  stored in a 0600 SQLite file, used server-side only, and never rendered
  back or logged. Chat and image calls use *your* keys, per request.

Security notes: passwords are salted + stretched (pbkdf2-hmac/sha256,
200k rounds); sessions are HMAC-signed, HttpOnly, SameSite=Lax, 7-day
expiry. For anything beyond localhost: put it behind TLS, set
`SWARM_TREE_SECRET` from a real secret manager, and consider moving keys
out of SQLite into a KMS.

## Rules enforced on the tree

- Each node needs `name`, `provider`, `model`, `role`; `children` is optional.
  Node names must be unique tree-wide (transcript pointers like "node 'X'"
  would otherwise be ambiguous).
- Providers: `anthropic`, `openai`, `gemini`, `xai`, `deepseek`.
- Max depth 4 (root is depth 0); cycles are rejected.
- Each node accepts `max_children` (default 8); more children is rejected.
- Unknown config keys produce a warning (never a crash) — a `chilren:`
  typo warns instead of silently dropping a subtree.
- Root knobs (all optional): `summary_mode: true|false` (default `true`,
  must be a boolean), `max_concurrency` (default 8, global cap on
  concurrent provider calls), `timeout` (default 120, seconds per provider
  call), `prices` (override the built-in approximate price table),
  `summarizer: {provider, model}` (cheap compressor model; per-node
  override also accepted), `serve_dir` (default `./serve-data`, where serve
  mode keeps state/transcripts/briefings), `rollup: true|false` (default
  `true`, daily briefing aggregation in serve mode).
- Per-node serve knobs (all optional): `schedule` (`"30m"` interval or
  `"08:05"` daily local time), `task` (the node's recurring task text),
  `enabled: true|false` (default `true`).
- Configs can be YAML (requires PyYAML: `pip install "swarm-tree[yaml]"`)
  or plain JSON (always works).

## Exit codes

- `0` — clean run, all agents succeeded.
- `3` — degraded: an answer was produced, but one or more sub-agents failed
  (details in the transcript; a warning is printed).
- `1` — hard failure: the root node failed or the run was cancelled (Ctrl-C).
- `2` — config or usage error (invalid tree, missing API keys, bad flags).

The transcript is always written, even for cancelled runs — Ctrl-C marks
in-flight nodes `cancelled` and dumps the partial transcript instead of
losing everything.

## Tests

```bash
python -m unittest discover -s tests
```

125 tests, all green. Network calls are stubbed; provider tests never touch
the real APIs. CI runs the suite on Python 3.9–3.13
(`.github/workflows/ci.yml`).

## License

MIT. See [LICENSE](LICENSE).
