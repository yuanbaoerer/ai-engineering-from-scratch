"""ReAct agent loop — stdlib only, async.

Same architecture as main.py (sync) but tool dispatch is async.
Reference: Yao et al. "ReAct" (ICLR 2023, arXiv:2210.03629).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable


# ---------------------------------------------------------------------------
# Provider schemas — same as sync, never mix Anthropic and OpenAI
# ---------------------------------------------------------------------------

class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


@dataclass
class AnthropicToolUse:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class AnthropicToolResult:
    tool_use_id: str
    content: str
    is_error: bool = False


@dataclass
class OpenAIToolCall:
    id: str
    type: str
    function: dict[str, Any]


@dataclass
class OpenAIToolResult:
    role: str
    tool_call_id: str
    content: str


# ---------------------------------------------------------------------------
# Message buffer
# ---------------------------------------------------------------------------

class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FINAL = "final"


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[AnthropicToolUse] | list[OpenAIToolCall] | None = None
    tool_results: list[AnthropicToolResult] | list[OpenAIToolResult] | None = None


@dataclass
class MessageBuffer:
    messages: list[Message] = field(default_factory=list)

    def append(self, msg: Message) -> None:
        self.messages.append(msg)

    def as_prompt(self, provider: Provider) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in self.messages:
            if provider == Provider.ANTHROPIC:
                out.extend(_to_anthropic(m))
            else:
                out.extend(_to_openai(m))
        return out


def _to_anthropic(msg: Message) -> list[dict[str, Any]]:
    if msg.role == Role.USER:
        return [{"role": "user", "content": msg.content}]
    if msg.role == Role.FINAL:
        return [{"role": "assistant", "content": msg.content}]
    if msg.role == Role.ASSISTANT:
        blocks: list[dict[str, Any]] = []
        if msg.content:
            blocks.append({"type": "text", "text": msg.content})
        if msg.tool_calls:
            for tc in msg.tool_calls:
                blocks.append({
                    "type": "tool_use", "id": tc.id,
                    "name": tc.name, "input": tc.input,
                })
        return [{"role": "assistant", "content": blocks}]
    results: list[dict[str, Any]] = []
    if msg.tool_results:
        for tr in msg.tool_results:
            results.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tr.tool_use_id,
                    "content": tr.content,
                    "is_error": tr.is_error,
                }],
            })
    return results


def _to_openai(msg: Message) -> list[dict[str, Any]]:
    if msg.role == Role.USER:
        return [{"role": "user", "content": msg.content}]
    if msg.role == Role.FINAL:
        return [{"role": "assistant", "content": msg.content}]
    if msg.role == Role.ASSISTANT:
        entry: dict[str, Any] = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            entry["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": tc.function}
                for tc in msg.tool_calls
            ]
        return [entry]
    results: list[dict[str, Any]] = []
    if msg.tool_results:
        for tr in msg.tool_results:
            results.append({
                "role": "tool", "tool_call_id": tr.tool_call_id,
                "content": tr.content,
            })
    return results


# ---------------------------------------------------------------------------
# Tool registry — async dispatch, input validation
# ---------------------------------------------------------------------------

@dataclass
class ToolDef:
    name: str
    fn: Callable[..., Any]  # sync or async callable
    param_types: dict[str, type] | None = None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}

    def register(self, name: str, fn: Callable[..., Any],
                 param_types: dict[str, type] | None = None) -> None:
        self._tools[name] = ToolDef(name=name, fn=fn, param_types=param_types)

    def names(self) -> list[str]:
        return sorted(self._tools)

    async def dispatch(self, name: str, args: dict[str, Any]) -> str:
        td = self._tools.get(name)
        if td is None:
            return f"error: unknown tool {name!r}"
        if td.param_types:
            for k, expected in td.param_types.items():
                if k in args and not isinstance(args[k], expected):
                    return (f"error: {k} should be {expected.__name__}, "
                            f"got {type(args[k]).__name__}")
        try:
            result = td.fn(**args)
            if asyncio.iscoroutine(result):
                return await result
            return result
        except TypeError as e:
            return f"error: bad args for {name}: {e}"
        except Exception as e:
            return f"error: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Trace record
# ---------------------------------------------------------------------------

class StopReason(str, Enum):
    FINISH = "finish"
    NO_TOOL_CALLS = "no_tool_calls"
    MAX_TURNS = "max_turns"
    MAX_TOKENS = "max_tokens"
    GUARDRAIL = "guardrail"


@dataclass
class TraceEntry:
    step: int
    thought: str
    action: str | None
    action_args: dict[str, Any] | None
    observation: str | None
    timestamp: float

    def __str__(self) -> str:
        parts = [f"[step {self.step}]"]
        if self.thought:
            parts.append(f"Thought: {self.thought}")
        if self.action:
            parts.append(f"Action: {self.action}({self.action_args or {}})")
        if self.observation:
            parts.append(f"Observation: {self.observation}")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# OpenTelemetry GenAI spans (optional)
# ---------------------------------------------------------------------------

def _try_otel_span(name: str, attributes: dict[str, Any] | None = None):
    try:
        from opentelemetry import trace
        tracer = trace.get_tracer("agent-loop")
        return tracer.start_as_current_span(name, attributes=attributes or {})
    except Exception:
        import contextlib
        return contextlib.nullcontext()


# ---------------------------------------------------------------------------
# Toy LLM — scripted async policy
# ---------------------------------------------------------------------------

class ToyLLM:
    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.script = script
        self.cursor = 0

    async def respond(self, history: MessageBuffer) -> dict[str, Any]:
        if self.cursor >= len(self.script):
            return {"kind": "finish", "content": "no more actions"}
        entry = self.script[self.cursor]
        self.cursor += 1
        return entry


# ---------------------------------------------------------------------------
# Agent loop — async
# ---------------------------------------------------------------------------

TURN_BUDGETS = {
    "short": 10,
    "computer_use": 200,
    "deep_research": 400,
}


@dataclass
class AgentLoop:
    llm: ToyLLM
    tools: ToolRegistry
    provider: Provider = Provider.ANTHROPIC
    max_turns: int = TURN_BUDGETS["short"]
    max_total_tokens: int | None = None
    guardrail_check: Callable[[str], bool] | None = None
    history: MessageBuffer = field(default_factory=MessageBuffer)
    trace: list[TraceEntry] = field(default_factory=list)
    _total_tokens: int = 0

    async def run(self, user_message: str) -> str:
        self.history.append(Message(role=Role.USER, content=user_message))
        stop_reason = StopReason.MAX_TURNS

        with _try_otel_span("invoke_agent", {"user_message": user_message}):
            for step in range(self.max_turns):
                reply = await self.llm.respond(self.history)

                if reply["kind"] == "finish":
                    stop_reason = StopReason.FINISH
                    self._record_trace(step, reply.get("thought", ""),
                                       None, None, None)
                    self.history.append(
                        Message(role=Role.FINAL, content=reply["content"]))
                    return reply["content"]

                thought = reply.get("thought", "")
                action_name = reply["action"]
                action_args = reply.get("args", {})

                if self.guardrail_check and self.guardrail_check(action_name):
                    stop_reason = StopReason.GUARDRAIL
                    obs = "guardrail blocked action"
                    self._record_trace(step, thought, action_name,
                                       action_args, obs)
                    self.history.append(
                        Message(role=Role.FINAL,
                                content=f"stopped by guardrail on {action_name}"))
                    return obs

                if (self.max_total_tokens is not None
                        and self._total_tokens >= self.max_total_tokens):
                    stop_reason = StopReason.MAX_TOKENS
                    self._record_trace(step, thought, action_name,
                                       action_args, None)
                    self.history.append(
                        Message(role=Role.FINAL, content="token budget exceeded"))
                    return "token budget exceeded"

                with _try_otel_span("tool_call", {"tool": action_name}):
                    observation = await self.tools.dispatch(
                        action_name, action_args)

                self._record_trace(step, thought, action_name,
                                   action_args, observation)

                self._append_assistant_turn(thought, action_name, action_args)
                self._append_tool_turn(action_name, observation)

                if not action_name:
                    stop_reason = StopReason.NO_TOOL_CALLS
                    self.history.append(
                        Message(role=Role.FINAL, content=thought or "done"))
                    return thought or "done"

            self.history.append(
                Message(role=Role.FINAL, content="budget exhausted"))
            return "budget exhausted"

    def _record_trace(self, step: int, thought: str, action: str | None,
                      args: dict[str, Any] | None,
                      observation: str | None) -> None:
        self.trace.append(TraceEntry(
            step=step, thought=thought, action=action,
            action_args=args, observation=observation,
            timestamp=time.time(),
        ))

    def _append_assistant_turn(self, thought: str, action_name: str,
                               action_args: dict[str, Any]) -> None:
        if self.provider == Provider.ANTHROPIC:
            tc = AnthropicToolUse(
                id=f"toolu_{uuid.uuid4().hex[:8]}",
                name=action_name, input=action_args,
            )
            self.history.append(Message(
                role=Role.ASSISTANT, content=thought, tool_calls=[tc]))
        else:
            tc = OpenAIToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function={"name": action_name,
                          "arguments": str(action_args)},
            )
            self.history.append(Message(
                role=Role.ASSISTANT, content=thought, tool_calls=[tc]))

    def _append_tool_turn(self, action_name: str, observation: str) -> None:
        if self.provider == Provider.ANTHROPIC:
            tool_use_id = ""
            for m in reversed(self.history.messages):
                if m.tool_calls and isinstance(m.tool_calls[0], AnthropicToolUse):
                    tool_use_id = m.tool_calls[-1].id
                    break
            tr = AnthropicToolResult(
                tool_use_id=tool_use_id, content=observation,
                is_error=observation.startswith("error:"))
            self.history.append(Message(role=Role.TOOL, tool_results=[tr]))
        else:
            tool_call_id = ""
            for m in reversed(self.history.messages):
                if m.tool_calls and isinstance(m.tool_calls[0], OpenAIToolCall):
                    tool_call_id = m.tool_calls[-1].id
                    break
            tr = OpenAIToolResult(
                role="tool", tool_call_id=tool_call_id, content=observation)
            self.history.append(Message(role=Role.TOOL, tool_results=[tr]))


# ---------------------------------------------------------------------------
# Sample tools
# ---------------------------------------------------------------------------

def calculator(expr: str) -> str:
    """Evaluate a simple arithmetic expression."""
    allowed = set("0123456789+-*/(). ")
    if not set(expr).issubset(allowed):
        return "error: illegal character in expr"
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


class KVStore:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def get(self, key: str) -> str:
        """Async get value by key."""
        await asyncio.sleep(0)  # yield to event loop
        return self._store.get(key, f"missing:{key}")

    async def set(self, key: str, value: str) -> str:
        """Async set key-value pair."""
        await asyncio.sleep(0)
        self._store[key] = value
        return f"stored {key}"


# ---------------------------------------------------------------------------
# Demo builder
# ---------------------------------------------------------------------------

def build_demo_agent(provider: Provider = Provider.ANTHROPIC) -> AgentLoop:
    tools = ToolRegistry()
    tools.register("calculator", calculator)
    kv = KVStore()
    tools.register("kv_get", kv.get)
    tools.register("kv_set", kv.set)

    script: list[dict[str, Any]] = [
        {"kind": "action", "thought": "store the base price",
         "action": "kv_set", "args": {"key": "base", "value": "120"}},
        {"kind": "action", "thought": "compute 15% tax",
         "action": "calculator", "args": {"expr": "120 * 0.15"}},
        {"kind": "action", "thought": "store the tax",
         "action": "kv_set", "args": {"key": "tax", "value": "18.0"}},
        {"kind": "action", "thought": "compute total",
         "action": "calculator", "args": {"expr": "120 + 18.0"}},
        {"kind": "action", "thought": "confirm stored values",
         "action": "kv_get", "args": {"key": "base"}},
        {"kind": "finish", "content": "the total including 15% tax is 138.0"},
    ]
    return AgentLoop(llm=ToyLLM(script), tools=tools, provider=provider,
                     max_turns=TURN_BUDGETS["short"])


def pretty_trace(trace: list[TraceEntry]) -> None:
    for entry in trace:
        print(entry)


async def main() -> None:
    print("=" * 70)
    print("REACT AGENT LOOP (ASYNC) — Phase 14, Lesson 01")
    print("=" * 70)

    for provider in (Provider.ANTHROPIC, Provider.OPENAI):
        print(f"\n--- Provider: {provider.value} ---")
        agent = build_demo_agent(provider=provider)
        final = await agent.run("What is 120 plus 15% tax, stored in kv?")
        print()
        pretty_trace(agent.trace)
        print()
        print(f"final answer:  {final}")
        print(f"turns used:    {sum(1 for t in agent.trace if t.action)}")
        print(f"tools:         {agent.tools.names()}")
        print(f"provider msgs: {len(agent.history.as_prompt(provider))}")


if __name__ == "__main__":
    asyncio.run(main())
