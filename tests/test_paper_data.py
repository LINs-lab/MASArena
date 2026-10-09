import importlib.util
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PaperDataTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("prepare_paper_data", ROOT / "scripts/prepare_paper_data.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_sample_matches_paper_seed_without_touching_global_rng(self):
        state = random.getstate()
        actual = self.module.select_indices(486, 400)
        self.assertEqual(actual, random.Random(42).sample(range(486), 400))
        self.assertEqual(random.getstate(), state)

    def test_small_split_preserves_all_records(self):
        self.assertEqual(self.module.select_indices(30, 30), list(range(30)))
        with self.assertRaises(ValueError):
            self.module.select_indices(29, 30)

    def test_rejects_wrong_source_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "math_test.jsonl").write_text('{"problem":"altered"}\n')
            manifest = {"benchmarks": {"math": {"filename": "math_test.jsonl", "source_count": 486,
                        "evaluated_count": 400, "sha256": "0" * 64}}}
            with self.assertRaises(ValueError):
                self.module.prepare(root, root / "out", manifest)
            self.assertFalse((root / "out/math_test.jsonl").exists())

    def test_paper_commands_pin_protocol_completion_cap(self):
        spec = importlib.util.spec_from_file_location("run_paper", ROOT / "scripts/run_paper.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        protocol = json.loads((ROOT / "configs/paper/protocol.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            selection = {"total_instances": 3302, "benchmarks": {}}
            for benchmark, count in protocol["benchmark_sizes"].items():
                payload = b'{"problem":"offline fixture"}\n' * count
                filename = benchmark + ".jsonl"
                (data / filename).write_bytes(payload)
                selection["benchmarks"][benchmark] = {"count": count, "filename": filename,
                    "sha256": hashlib.sha256(payload).hexdigest()}
            (data / "selection.json").write_text(json.dumps(selection))
            commands = module.build_commands(ROOT, data, data / "results", protocol)
        self.assertEqual(len(commands), 210)
        for command in commands:
            index = command.index("--max-completion-tokens")
            self.assertEqual(command[index + 1], str(protocol["max_completion_tokens"]))


if __name__ == "__main__":
    unittest.main()
