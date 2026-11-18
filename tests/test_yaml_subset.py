"""Tests for the built-in YAML subset reader used for config files."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from src.pipeline.config import load_pipeline_config
from src.utils.yaml_subset import YAMLSubsetError, load, load_file


class YAMLSubsetReaderTests(unittest.TestCase):
    """Good input parses as PyYAML's safe_load would; anything else is refused with file and line."""

    def test_mappings_lists_and_scalars(self) -> None:
        text = (
            "# comment\n"
            "---\n"
            "ollama:\n"
            "  base_url: \"http://localhost:11434\"  # trailing comment\n"
            "  timeout: 600\n"
            "  batch_delay: 5.0\n"
            "  disable_thinking: true\n"
            "  level: low\n"
            "  empty:\n"
            "list:\n"
            "  - \"gsm8k\"\n"
            "  - 'it''s'\n"
            "  - name: a\n"
            "    items: [1, two, \"three\"]\n"
            "same_indent:\n"
            "- x\n"
            "- y\n"
            "flow: {a: 1, b: [x, y], c: }\n"
            "empty_list: []\n"
            "deep:\n"
            "    inner:\n"
            "        value: ~\n"
        )
        self.assertEqual(
            load(text),
            {
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "timeout": 600,
                    "batch_delay": 5.0,
                    "disable_thinking": True,
                    "level": "low",
                    "empty": None,
                },
                "list": ["gsm8k", "it's", {"name": "a", "items": [1, "two", "three"]}],
                "same_indent": ["x", "y"],
                "flow": {"a": 1, "b": ["x", "y"], "c": None},
                "empty_list": [],
                "deep": {"inner": {"value": None}},
            },
        )

    def test_yaml_1_1_resolution_matches_pyyaml(self) -> None:
        data = load(
            "a: yes\nb: Off\nc: 017\nd: 0x1F\ne: 0b11\nf: 1_000\ng: .5\nh: 1e3\ni: -.inf\nj: Null\n"
            "k: '5'\nl: \"\\u00e9\\t\"\n1: int key\n"
        )
        self.assertIs(data["a"], True)
        self.assertIs(data["b"], False)
        self.assertEqual(data["c"], 15)
        self.assertEqual(data["d"], 31)
        self.assertEqual(data["e"], 3)
        self.assertEqual(data["f"], 1000)
        self.assertEqual(data["g"], 0.5)
        self.assertEqual(data["h"], "1e3")  # PyYAML (YAML 1.1) needs a dot for a float
        self.assertTrue(math.isinf(data["i"]) and data["i"] < 0)
        self.assertIsNone(data["j"])
        self.assertEqual(data["k"], "5")
        self.assertEqual(data["l"], "é\t")
        self.assertEqual(data[1], "int key")

    def test_empty_and_comment_only_documents_are_none(self) -> None:
        self.assertIsNone(load(""))
        self.assertIsNone(load("# nothing here\n\n"))

    def test_refused_input_names_file_and_line(self) -> None:
        refused = {
            "a: &anchor 1\n": 1,
            "a: *alias\n": 1,
            "a: !!str 1\n": 1,
            "a: |\n  block\n": 1,
            "a: b: c\n": 1,
            "a: 2026-11-18\n": 1,
            "a: 1\n---\nb: 2\n": 2,
            "a:\n\tb: 1\n": 2,
            "a: 'open\n": 1,
            "a: [1, 2\n": 1,
            "a: plain\n  continued\n": 2,
            "<<: {a: 1}\n": 1,
            "? key\n": 1,
            "a:\n  b: 1\n c: 2\n": 3,
        }
        for text, line in refused.items():
            with self.subTest(text=text):
                with self.assertRaises(YAMLSubsetError) as caught:
                    load(text, "fake.yaml")
                self.assertEqual(caught.exception.line, line)
                self.assertIn("fake.yaml, line", str(caught.exception))

    def test_repository_config_files_load(self) -> None:
        root = Path(__file__).resolve().parents[1] / "config"
        for path in sorted(root.glob("*.yaml")):
            if path.name == "secrets.yaml":
                continue  # never read a real secrets file in tests
            with self.subTest(path=path.name):
                load_file(path)
        config = load_file(root / "config.example.yaml")
        self.assertEqual(config["evaluation"]["benchmarks"]["mmlu"]["subjects"], ["all"])

    def test_pipeline_config_merges_a_secrets_file(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config_path = Path(tempdir) / "config.yaml"
            config_path.write_text("huggingface:\n  use_pro_features: true\n", encoding="utf-8")
            (Path(tempdir) / "secrets.yaml").write_text(
                "huggingface:\n  token: \"fake-test-token\"\n", encoding="utf-8"
            )
            config = load_pipeline_config(str(config_path), ["gsm8k"])
        self.assertEqual(config["huggingface"], {"use_pro_features": True, "token": "fake-test-token"})

    def test_pipeline_config_error_names_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config_path = Path(tempdir) / "config.yaml"
            config_path.write_text("ollama:\n  base_url: &x \"http://localhost\"\n", encoding="utf-8")
            with self.assertRaises(YAMLSubsetError) as caught:
                load_pipeline_config(str(config_path), ["gsm8k"])
        self.assertIn("config.yaml, line 2", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
