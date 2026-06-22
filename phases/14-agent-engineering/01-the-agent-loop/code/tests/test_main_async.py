"""Tests for the async ReAct agent loop."""

import asyncio
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main_async import (
    ToolRegistry, MessageBuffer, Message, Role, Provider,
    AgentLoop, ToyLLM, StopReason,
    calculator, KVStore, TURN_BUDGETS,
)


def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestAsyncToolRegistry(unittest.TestCase):
    def test_dispatch_sync_fn(self):
        reg = ToolRegistry()
        reg.register("add", lambda a, b: str(a + b))
        result = run_async(reg.dispatch("add", {"a": 1, "b": 2}))
        self.assertEqual(result, "3")

    def test_dispatch_async_fn(self):
        async def aadd(a, b):
            return str(a + b)
        reg = ToolRegistry()
        reg.register("aadd", aadd)
        result = run_async(reg.dispatch("aadd", {"a": 1, "b": 2}))
        self.assertEqual(result, "3")

    def test_dispatch_unknown(self):
        reg = ToolRegistry()
        result = run_async(reg.dispatch("nope", {}))
        self.assertEqual(result, "error: unknown tool 'nope'")

    def test_dispatch_exception(self):
        def boom():
            raise ValueError("kaboom")
        reg = ToolRegistry()
        reg.register("boom", boom)
        result = run_async(reg.dispatch("boom", {}))
        self.assertIn("error: ValueError: kaboom", result)


class TestAsyncKVStore(unittest.TestCase):
    def test_async_set_get(self):
        kv = KVStore()
        self.assertEqual(run_async(kv.set("a", "1")), "stored a")
        self.assertEqual(run_async(kv.get("a")), "1")

    def test_async_missing(self):
        kv = KVStore()
        self.assertEqual(run_async(kv.get("x")), "missing:x")


class TestAsyncAgentLoop(unittest.TestCase):
    def test_finish(self):
        script = [{"kind": "finish", "content": "done"}]
        agent = AgentLoop(llm=ToyLLM(script), tools=ToolRegistry())
        result = run_async(agent.run("hello"))
        self.assertEqual(result, "done")

    def test_max_turns(self):
        script = [{"kind": "action", "thought": "h",
                    "action": "noop", "args": {}}] * 20
        agent = AgentLoop(llm=ToyLLM(script), tools=ToolRegistry(),
                          max_turns=3)
        result = run_async(agent.run("go"))
        self.assertEqual(result, "budget exhausted")

    def test_tool_dispatch_async(self):
        async def acalc(expr: str) -> str:
            return calculator(expr)
        reg = ToolRegistry()
        reg.register("calc", acalc)
        script = [
            {"kind": "action", "thought": "compute", "action": "calc",
             "args": {"expr": "2+3"}},
            {"kind": "finish", "content": "5"},
        ]
        agent = AgentLoop(llm=ToyLLM(script), tools=reg)
        result = run_async(agent.run("compute 2+3"))
        self.assertEqual(result, "5")
        self.assertEqual(agent.trace[0].observation, "5")

    def test_guardrail(self):
        def block(name):
            return name == "danger"
        reg = ToolRegistry()
        reg.register("danger", lambda: "boom")
        script = [{"kind": "action", "thought": "hmm",
                    "action": "danger", "args": {}}]
        agent = AgentLoop(llm=ToyLLM(script), tools=reg,
                          guardrail_check=block)
        result = run_async(agent.run("do it"))
        self.assertIn("guardrail", result)

    def test_demo_builds(self):
        from main_async import build_demo_agent
        for p in (Provider.ANTHROPIC, Provider.OPENAI):
            agent = build_demo_agent(p)
            result = run_async(agent.run("test"))
            self.assertEqual(result, "the total including 15% tax is 138.0")


if __name__ == "__main__":
    unittest.main()
