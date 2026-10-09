import importlib.util
import json
from pathlib import Path
import unittest


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "scripts/build_paper_release.py"
        spec = importlib.util.spec_from_file_location("release", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_credentials_in_nested_serialized_text_are_removed(self):
        fake = "sk-" + "x" * 36
        raw = json.dumps({"content": json.dumps({"Authorization": "Bearer " + fake}),
                          "api_key": "private-value", "total_tokens": 123})
        result = self.module.sanitize_text(raw, json_mode=True)
        self.assertNotIn(fake, result)
        self.assertNotIn("private-value", result)
        self.assertEqual(json.loads(result)["total_tokens"], 123)

    def test_archive_paths_cannot_escape(self):
        for path in ["../secret", "/absolute", "a/../../b"]:
            with self.assertRaises(ValueError):
                self.module.safe_name(path)

    def test_local_identity_removed_but_task_url_preserved(self):
        source = "file /" + "Users/researcher/project/input.txt and https://example.com/article"
        cleaned = self.module.sanitize_text(source)
        self.assertNotIn("researcher", cleaned)
        self.assertIn("https://example.com/article", cleaned)


if __name__ == "__main__":
    unittest.main()
