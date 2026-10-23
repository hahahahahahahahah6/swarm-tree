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
● orchestrator [anthropic/claude-sonnet-4-5] ok 0.0s
  ↳ "[orchestrator] dry-run synthesis: combined 3 sub-agent result(s) into one…"
├── ● grok-agent [xai/grok-4] ok 0.0s
│   ↳ "[grok-agent] dry-run result from xai/grok-4: Do the grok-agent part of:…"
├── ● gemini-agent [gemini/gemini-2.5-pro] ok 0.0s
│   ↳ "[gemini-agent] dry-run result from gemini/gemini-2.5-pro: Do the gemini-…"
└── ● gpt-agent [openai/gpt-5] ok 0.0s
    ↳ "[gpt-agent] dry-run synthesis: combined 2 sub-agent result(s) into one…"
    ├── ● code-reviewer [deepseek/deepseek-chat] ok 0.0s
    │   ↳ "[code-reviewer] dry-run result from deepseek/deepseek-chat: Do the…"
    └── ● writer [anthropic/claude-sonnet-4-5] ok 0.0s
        ↳ "[writer] dry-run result from anthropic/claude-sonnet-4-5: Do the wri…"
```

Then the root's final answer, and a full JSON transcript
(`transcript.json`) with every node, every prompt, every raw response,
and per-call timings — so the whole run is auditable.

## Other commands

```bash
swarm-tree validate --tree tree.yaml   # check a config without running it
swarm-tree example                      # print a starter tree.yaml
```

## Rules enforced on the tree

- Each node needs `name`, `provider`, `model`, `role`; `children` is optional.
- Providers: `anthropic`, `openai`, `gemini`, `xai`, `deepseek`.
- Max depth 4 (root is depth 0); cycles are rejected.
- Configs can be YAML (parsed with PyYAML when installed) or plain JSON.

## Tests

```bash
python -m unittest discover -s tests
```

36 tests, all green. Network calls are stubbed; provider tests never touch
the real APIs.

## License

MIT. See [LICENSE](LICENSE).
