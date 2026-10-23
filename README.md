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

## Studio (visual editor)

Don't want to hand-write YAML? Open
[`studio/swarm-tree-studio.html`](studio/swarm-tree-studio.html) in a browser —
a single-file visual tree builder: click nodes to edit them, pick a provider
per agent, live validation against the CLI's rules, and one-click YAML/JSON
export that `swarm-tree validate` accepts.

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
  override also accepted).
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
