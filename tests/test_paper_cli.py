"""Exercise CLI configuration without importing API clients or making requests."""

import argparse
import ast
import asyncio
import contextlib
import datetime
import io
import logging
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


class PaperCliTests(unittest.TestCase):
    def run_cli(self, extra=(), environ=None):
        source = Path(__file__).resolve().parents[1] / "main.py"
        tree = ast.parse(source.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        captured = {}

        class Runner:
            def __init__(self, **kwargs):
                pass

            def run(self, **kwargs):
                captured.update(kwargs)
                return {}

        modules = {}
        for name in ("mas_arena.agents", "mas_arena.evaluators", "mas_arena.memory.memory_registry"):
            modules[name] = types.ModuleType(name)
        modules["mas_arena.agents"].AVAILABLE_AGENT_SYSTEMS = {"bench_agent": object()}
        modules["mas_arena.evaluators"].BENCHMARKS = {"math": {}}
        modules["mas_arena.memory.memory_registry"].memory_registry = types.SimpleNamespace(
            get_available_memory_names=lambda: []
        )
        namespace = dict(argparse=argparse, asyncio=asyncio, os=os, Path=Path, sys=sys, datetime=datetime,
                         List=list, Optional=__import__("typing").Optional,
                         load_dotenv=lambda: None, BenchmarkRunner=Runner,
                         logger=logging.getLogger(__name__))
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules), \
                patch.dict(os.environ, environ or {}, clear=True), \
                patch.object(sys, "argv", ["main.py", "--results-dir", str(Path(directory) / "run_1/math"), *extra]), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(namespace["main"](), 0)
        return captured["agent_config"] or {}

    def test_default_tools_do_not_contain_model_name(self):
        config = self.run_cli()
        self.assertNotIn("manager_tools", config)

    def test_model_environment_is_model_only(self):
        config = self.run_cli(environ={"MODEL_NAME": "test-backend"})
        self.assertEqual(config.get("model_name"), "test-backend")
        self.assertNotIn("manager_tools", config)

    def test_explicit_model_and_empty_tools(self):
        config = self.run_cli(("--model-name", "explicit", "--manager-tools", "none",
                               "--search-tools", "none"), {"MODEL_NAME": "environment"})
        self.assertEqual(config["model_name"], "explicit")
        self.assertEqual(config["manager_tools"], [])
        self.assertEqual(config["search_tools"], [])

    def test_explicit_completion_cap_reaches_workflow_config(self):
        config = self.run_cli(("--max-completion-tokens", "8192"), {"MAX_TOKEN_SIZE": "123"})
        self.assertEqual(config["max_completion_tokens"], 8192)

    def test_nonpositive_completion_cap_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            self.run_cli(("--max-completion-tokens", "0"))
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
