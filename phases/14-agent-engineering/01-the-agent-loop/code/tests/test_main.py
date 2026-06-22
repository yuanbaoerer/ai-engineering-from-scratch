"""Tests for the sync ReAct agent loop."""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import (
    ToolRegistry, MessageBuffer, Message, Role, Provider,
    AgentLoop, ToyLLM, StopReason, TraceEntry,
    AnthropicToolUse, AnthropicToolResult,
    OpenAIToolCall, OpenAIToolResult,
    calculator, KVStore, TURN_BUDGETS,
)


class TestToolRegistry(unittest.TestCase):
    def test_register_and_names(self):
        reg = ToolRegistry()
        reg.register("a", lambda: "ok")
        reg.register("b", lambda: "ok")
        self.assertEqual(reg.names(), ["a", "b"])

    def test_dispatch_unknown_tool(self):
        reg = ToolRegistry()
        result = reg.dispatch("nope", {})
        self.assertEqual(result, "error: unknown tool 'nope'")

    def test_dispatch_success(self):
        reg = ToolRegistry()
        reg.register("add", lambda a, b: str(a + b))
        result = reg.dispatch("add", {"a": 1, "b": 2})
        self.assertEqual(result, "3")

    def test_dispatch_type_error(self):
        reg = ToolRegistry()
        reg.register("add", lambda a, b: str(a + b))
        result = reg.dispatch("add", {"a": 1})
        self.assertIn("error: bad args", result)

    def test_dispatch_exception(self):
        def boom():
            raise ValueError("kaboom")
        reg = ToolRegistry()
        reg.register("boom", boom)
        result = reg.dispatch("boom", {})
        self.assertIn("error: ValueError: kaboom", result)

    def test_dispatch_with_param_types(self):
        reg = ToolRegistry()
        reg.register("echo", lambda x: str(x), param_types={"x": str})
        result = reg.dispatch("echo", {"x": 42})
        self.assertIn("error: x should be str", result)

    def test_dispatch_param_types_valid(self):
        reg = ToolRegistry()
        reg.register("echo", lambda x: str(x), param_types={"x": str})
        result = reg.dispatch("echo", {"x": "hello"})
        self.assertEqual(result, "hello")


class TestCalculator(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(calculator("2 + 3"), "5")

    def test_precedence(self):
        self.assertEqual(calculator("2 + 3 * 4"), "14")

    def test_parentheses(self):
        self.assertEqual(calculator("(2 + 3) * 4"), "20")

    def test_illegal_char(self):
        result = calculator("2 + x")
        self.assertIn("error: illegal character", result)

    def test_division(self):
        self.assertEqual(calculator("10 / 4"), "2.5")


class TestKVStore(unittest.TestCase):
    def test_set_get(self):
        kv = KVStore()
        self.assertEqual(kv.set("a", "1"), "stored a")
        self.assertEqual(kv.get("a"), "1")

    def test_missing_key(self):
        kv = KVStore()
        self.assertEqual(kv.get("nope"), "missing:nope")


class TestMessageBuffer(unittest.TestCase):
    def test_anthropic_prompt(self):
        buf = MessageBuffer()
        buf.append(Message(role=Role.USER, content="hello"))
        buf.append(Message(role=Role.ASSISTANT, content="hi",
                           tool_calls=[AnthropicToolUse(
                               id="t1", name="search", input={"q": "x"})]))
        buf.append(Message(role=Role.TOOL, tool_results=[
            AnthropicToolResult(tool_use_id="t1", content="result")]))
        prompt = buf.as_prompt(Provider.ANTHROPIC)
        self.assertEqual(prompt[0]["role"], "user")
        self.assertEqual(prompt[1]["role"], "assistant")
        self.assertEqual(prompt[1]["content"][1]["type"], "tool_use")
        self.assertEqual(prompt[2]["role"], "user")  # tool result wraps in user

    def test_openai_prompt(self):
        buf = MessageBuffer()
        buf.append(Message(role=Role.USER, content="hello"))
        buf.append(Message(role=Role.ASSISTANT, content="hi",
                           tool_calls=[OpenAIToolCall(
                               id="c1", type="function",
                               function={"name": "search", "arguments": "{}"})]))
        buf.append(Message(role=Role.TOOL, tool_results=[
            OpenAIToolResult(role="tool", tool_call_id="c1", content="result")]))
        prompt = buf.as_prompt(Provider.OPENAI)
        self.assertEqual(prompt[0]["role"], "user")
        self.assertEqual(prompt[1]["role"], "assistant")
        self.assertIn("tool_calls", prompt[1])
        self.assertEqual(prompt[2]["role"], "tool")


class TestAgentLoop(unittest.TestCase):
    def test_finish_stops_loop(self):
        script = [{"kind": "finish", "content": "done"}]
        agent = AgentLoop(llm=ToyLLM(script), tools=ToolRegistry())
        result = agent.run("hello")
        self.assertEqual(result, "done")
        self.assertEqual(len(agent.trace), 1)
        self.assertIsNone(agent.trace[0].action)

    def test_max_turns_exhausted(self):
        script = [{"kind": "action", "thought": "hmm",
                    "action": "noop", "args": {}}] * 20
        agent = AgentLoop(llm=ToyLLM(script), tools=ToolRegistry(),
                          max_turns=3)
        result = agent.run("go")
        self.assertEqual(result, "budget exhausted")
        self.assertEqual(len(agent.trace), 3)

    def test_tool_dispatch(self):
        reg = ToolRegistry()
        reg.register("calc", lambda expr: calculator(expr))
        script = [
            {"kind": "action", "thought": "compute", "action": "calc",
             "args": {"expr": "2+3"}},
            {"kind": "finish", "content": "5"},
        ]
        agent = AgentLoop(llm=ToyLLM(script), tools=reg)
        result = agent.run("compute 2+3")
        self.assertEqual(result, "5")
        self.assertEqual(agent.trace[0].observation, "5")

    def test_guardrail_blocks(self):
        def block(name: str) -> bool:
            return name == "danger"
        reg = ToolRegistry()
        reg.register("danger", lambda: "boom")
        script = [
            {"kind": "action", "thought": "hmm",
             "action": "danger", "args": {}},
        ]
        agent = AgentLoop(llm=ToyLLM(script), tools=reg,
                          guardrail_check=block)
        result = agent.run("do it")
        self.assertIn("guardrail", result)

    def test_anthropic_schema_output(self):
        script = [
            {"kind": "action", "thought": "search",
             "action": "q", "args": {"x": "1"}},
            {"kind": "finish", "content": "done"},
        ]
        reg = ToolRegistry()
        reg.register("q", lambda x: "r")
        agent = AgentLoop(llm=ToyLLM(script), tools=reg,
                          provider=Provider.ANTHROPIC)
        agent.run("query")
        prompt = agent.history.as_prompt(Provider.ANTHROPIC)
        # assistant turn should have tool_use block
        asst = [m for m in prompt if m["role"] == "assistant"]
        self.assertTrue(any(
            isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_use" for b in m["content"])
            for m in asst
        ))

    def test_openai_schema_output(self):
        script = [
            {"kind": "action", "thought": "search",
             "action": "q", "args": {"x": "1"}},
            {"kind": "finish", "content": "done"},
        ]
        reg = ToolRegistry()
        reg.register("q", lambda x: "r")
        agent = AgentLoop(llm=ToyLLM(script), tools=reg,
                          provider=Provider.OPENAI)
        agent.run("query")
        prompt = agent.history.as_prompt(Provider.OPENAI)
        asst = [m for m in prompt if m["role"] == "assistant"]
        self.assertTrue(any("tool_calls" in m for m in asst))

    def test_trace_records_thought(self):
        script = [
            {"kind": "action", "thought": "thinking...",
             "action": "calc", "args": {"expr": "1"}},
            {"kind": "finish", "content": "ok"},
        ]
        reg = ToolRegistry()
        reg.register("calc", lambda expr: "1")
        agent = AgentLoop(llm=ToyLLM(script), tools=reg)
        agent.run("go")
        self.assertEqual(agent.trace[0].thought, "thinking...")

    def test_demo_builds(self):
        from main import build_demo_agent
        for p in (Provider.ANTHROPIC, Provider.OPENAI):
            agent = build_demo_agent(p)
            result = agent.run("test")
            self.assertEqual(result, "the total including 15% tax is 138.0")


if __name__ == "__main__":
    unittest.main()
