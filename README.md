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

Standard library only. No vendor SDKs, no frameworks.

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
● orchestrator [anthropic/claude-sonnet-4-5] ok 0.0s 757/480
  ↳ "[orchestrator] dry-run synthesis: combined 3 sub-agent result(s) into one…"
├── ● grok-agent [xai/grok-4] ok 0.0s 43/60
│   ↳ "[grok-agent] dry-run result from xai/grok-4: Do the grok-agent part of:…"
├── ● gemini-agent [gemini/gemini-2.5-pro] ok 0.0s 41/60
│   ↳ "[gemini-agent] dry-run result from gemini/gemini-2.5-pro: Do the gemini-…"
└── ● gpt-agent [openai/gpt-5] ok 0.0s 334/240
    ↳ "[gpt-agent] dry-run synthesis: combined 2 sub-agent result(s) into one…"
    ├── ● code-reviewer [deepseek/deepseek-chat] ok 0.0s 43/60
    │   ↳ "[code-reviewer] dry-run result from deepseek/deepseek-chat: Do the…"
    └── ● writer [anthropic/claude-sonnet-4-5] ok 0.0s 38/60
        ↳ "[writer] dry-run result from anthropic/claude-sonnet-4-5: Do the wri…"
```

The trailing `757/480` on each node is input/output tokens: each node's
number is its own calls **plus all its descendants**, so the root line is
the grand total for the whole run. That is the whole point — you can see
at a glance which branch is eating your quota.

Then the root's final answer, and a full JSON transcript
(`transcript.json`) with every node, every prompt, every raw response,
per-call timings, **and per-call token usage** (`usage.input_tokens` /
`usage.output_tokens` parsed from each provider's API response) — so the
whole run is auditable.

## Token accounting

Token usage is parsed from each provider's native response format and
normalized to `{"input_tokens": int, "output_tokens": int}`:

| Provider | Source fields |
|---|---|
| anthropic | `usage.input_tokens` / `usage.output_tokens` |
| openai, xai, deepseek | `usage.prompt_tokens` / `usage.completion_tokens` |
| gemini | `usageMetadata.promptTokenCount` / `usageMetadata.candidatesTokenCount` |

Missing or malformed usage degrades to `0/0` instead of failing the run.
In `--dry-run` mode the mock provider returns deterministic fake usage
(`input_tokens` ≈ 1 per 4 prompt chars, `output_tokens` fixed per call
kind: 60 leaf / 40 decompose / 80 synthesize) so the entire accounting
path — parsing, rollup, rendering — is exercised for $0.00 with no keys.

## Summary mode

The classic multi-agent failure: every reviewer re-reads every worker's full
output, and the "coordination" costs more tokens than the work itself. By
default, swarm-tree avoids this.

After each child subtree finishes, the parent makes **one extra call to that
child's own provider/model**: *"summarize what you accomplished, key
decisions, and where the full output is stored (~150 words max)."* The
synthesize step then sees only these compact summaries — plus extracted
artifacts (file paths / URLs mentioned in the child's answer), the subtree's
token usage, and a pointer to the full transcript — **never the raw child
outputs**. Nothing is lost: `transcript.json` still records every prompt,
every raw response, and every per-call token count.

The economics: one cheap summarize call per child versus re-reading every
child's entire output in the synthesize prompt. If the summarize call itself
fails, the summary falls back to a 500-char truncation of the child's
answer — the run never crashes because of it.

Control it two ways:

```yaml
# tree.yaml (root level, default true)
summary_mode: false
```

```bash
swarm-tree run --tree tree.yaml --task "..." --full-context  # old behavior: full child outputs passed up
```

`swarm-tree validate` reports the effective setting for a config.

## Other commands

```bash
swarm-tree validate --tree tree.yaml   # check a config without running it
swarm-tree example                      # print a starter tree.yaml
```

## Rules enforced on the tree

- Each node needs `name`, `provider`, `model`, `role`; `children` is optional.
- Providers: `anthropic`, `openai`, `gemini`, `xai`, `deepseek`.
- Max depth 4 (root is depth 0); cycles are rejected.
- Root may set `summary_mode: true|false` (default `true`); must be a boolean.
- Configs can be YAML (parsed with PyYAML when installed) or plain JSON.

## Tests

```bash
python -m unittest discover -s tests
```

59 tests, all green. Network calls are stubbed; provider tests never touch
the real APIs.

## License

MIT. See [LICENSE](LICENSE).
