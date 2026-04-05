"""Tests for local runtime readiness messaging."""

from __future__ import annotations

import unittest

from src.cli.setup_checks import (
    LMStudioStatus,
    OllamaStatus,
    SetupReport,
    build_setup_report_lines,
    parse_ollama_list_output,
)


class SetupCheckTests(unittest.TestCase):
    """Validate setup messaging for common local-runtime states."""

    def test_parse_ollama_list_output_returns_model_names(self) -> None:
        stdout = "NAME ID SIZE MODIFIED\nllama3.2 abc 2.0 GB 1 day ago\nqwen2.5:def ghi 1.0 GB 2 days ago\n"
        self.assertEqual(parse_ollama_list_output(stdout), ["llama3.2", "qwen2.5:def"])

    def test_setup_lines_explain_existing_ollama_models_are_reused(self) -> None:
        report = SetupReport(
            ollama=OllamaStatus(
                cli_path="C:/Program Files/Ollama/ollama.exe",
                running=True,
                models=("llama3.2", "qwen2.5:0.5b"),
                source="api",
            ),
            lm_studio=LMStudioStatus(running=False, detail="connection refused"),
        )

        lines = build_setup_report_lines(report)

        self.assertIn("Ollama: ready (2 model(s) detected)", lines)
        self.assertIn(
            "  Existing Ollama models are reused automatically. You do not need to pull them again.",
            lines,
        )
        self.assertIn("smaLLMs can run right now with the local models above.", lines)

    def test_setup_lines_offer_clear_next_step_when_no_models_exist(self) -> None:
        report = SetupReport(
            ollama=OllamaStatus(
                cli_path="C:/Program Files/Ollama/ollama.exe",
                running=False,
                models=(),
                detail="connection refused",
            ),
            lm_studio=LMStudioStatus(running=False, detail="connection refused"),
        )

        lines = build_setup_report_lines(report)

        self.assertIn("Ollama: installed, but not responding on http://localhost:11434", lines)
        self.assertIn(
            "  If you already pulled models before, just start Ollama again. You do not need to reinstall them.",
            lines,
        )
        self.assertIn("No local models were detected yet.", lines)
        self.assertIn(
            "Fastest fix: start Ollama, run `ollama pull llama3.2` once, then relaunch the menu.",
            lines,
        )


if __name__ == "__main__":
    unittest.main()
