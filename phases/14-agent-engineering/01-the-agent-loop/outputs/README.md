# Agent Loop — ReAct Reference Implementation

## Stop-Condition Choice

Primary stop: **explicit `finish` action** (the model decides the task is done).

Safety belts (in priority order):
1. `no_tool_calls` — if the assistant emits an empty action, treat as done
2. `max_turns` — hard cap scaled to task class (see Turn Budget below)
3. `max_tokens` — optional total-token budget for cost control
4. `guardrail` — external check can block any tool name at runtime

Why `finish` as primary: it lets the model signal completion with a final answer, giving the user a natural stopping point. The other conditions are defensive — they prevent runaway loops when the model fails to emit `finish`.

## Turn Budget Justification

| Task class | Budget | Rationale |
|------------|--------|-----------|
| `short` | 10 | Quick tool chain: compute → store → verify. Most simple Q&A tasks finish in 3–6 steps. |
| `computer_use` | 200 | GUI agents: observe screen → click → observe → type. Anthropic reports dozens-to-hundreds of steps per task. |
| `deep_research` | 400 | Long-horizon research: multi-source retrieval, cross-referencing, report synthesis. |

The budget is configurable per `AgentLoop` instance — pick the class that fits your task, not a one-size-fits-all.

## Worked Trace (Anthropic provider)

```
User: What is 120 plus 15% tax, stored in kv?

[step 0] | Thought: store the base price
          | Action: kv_set({'key': 'base', 'value': '120'})
          | Observation: stored base

[step 1] | Thought: compute 15% tax
          | Action: calculator({'expr': '120 * 0.15'})
          | Observation: 18.0

[step 2] | Thought: store the tax
          | Action: kv_set({'key': 'tax', 'value': '18.0'})
          | Observation: stored tax

[step 3] | Thought: compute total
          | Action: calculator({'expr': '120 + 18.0'})
          | Observation: 138.0

[step 4] | Thought: confirm stored values
          | Action: kv_get({'key': 'base'})
          | Observation: 120

[step 5] | Thought: the total including 15% tax is 138.0
          | (finish)

final answer: the total including 15% tax is 138.0
turns used:   5
stop reason:  finish
```

Each step follows Thought → Action → Observation. The model reasons about what to do next based on the previous observation. The loop stops when the model emits `finish`.

## Message Schemas

The implementation never silently swaps between provider schemas:

- **Anthropic**: `tool_use` blocks inside assistant content, `tool_result` blocks inside user messages
- **OpenAI**: `tool_calls` array on assistant message, `tool` role messages for results

Pick your provider at `AgentLoop` construction time. The same tool registry works with both — only the serialization changes.

## What to Read Next

- **Lesson 02 (ReWOO planning)** — if your task is long-horizon and benefits from planning before execution. ReWOO separates planning from action, reducing redundant tool calls.
- **Lesson 03 (Reflexion)** — if your task is repeat-of-previous and benefits from self-critique. Reflexion adds a critic that evaluates the trace and feeds back corrections.
- **Lesson 27 (prompt injection)** — if your tools touch untrusted content (web pages, PDFs, user files). Tool outputs are untrusted input — only the user message carries permission.
