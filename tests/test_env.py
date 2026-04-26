"""Tests for repo-local environment loading."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.utils.env import huggingface_token, load_repository_env


class EnvTests(unittest.TestCase):
    """Validate .env aliases without exposing real secrets."""

    def test_hf_key_sets_huggingface_env_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            env_path = Path(tempdir) / ".env"
            env_path.write_text("HF_KEY=hf_test_token\n", encoding="utf-8")

            with mock.patch.dict(os.environ, {}, clear=True):
                load_repository_env(env_path)

                self.assertEqual(os.environ["HF_KEY"], "hf_test_token")
                self.assertEqual(os.environ["HF_TOKEN"], "hf_test_token")
                self.assertEqual(os.environ["HUGGINGFACE_HUB_TOKEN"], "hf_test_token")

    def test_huggingface_token_ignores_placeholder_config(self) -> None:
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_from_env"}, clear=True):
            self.assertEqual(huggingface_token("YOUR_HF_TOKEN_HERE"), "hf_from_env")


if __name__ == "__main__":
    unittest.main()
