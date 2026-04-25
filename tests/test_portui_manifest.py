"""Regression tests for the project-local PortUI manifest."""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[1]
PORTUI_DIR = ROOT / "portui"


def _read_env(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value
    return data


class PortUIManifestTests(unittest.TestCase):
    """Keep the PortUI app wired as a portable project-local command surface."""

    def test_manifest_uses_portui_project_variables(self) -> None:
        manifest = _read_env(PORTUI_DIR / "manifest.env")

        self.assertEqual(manifest["VARIABLE_repo"], "{{projectDir}}")
        self.assertEqual(manifest["VARIABLE_webRepo"], "{{workspaceDir}}{{pathSep}}websmaLLMs")
        self.assertEqual(manifest["VARIABLE_webPublicData"], "{{webRepo}}{{pathSep}}public{{pathSep}}data")

    def test_action_ids_are_unique_and_cross_platform(self) -> None:
        seen = set()
        for action_path in sorted((PORTUI_DIR / "actions").glob("*.env")):
            action = _read_env(action_path)
            with self.subTest(action=action_path.name):
                self.assertRegex(action.get("ID", ""), r"^[a-z0-9][a-z0-9-]*$")
                self.assertNotIn(action["ID"], seen)
                seen.add(action["ID"])
                self.assertIn("TITLE", action)
                self.assertIn("DESCRIPTION", action)
                self.assertIn("CWD", action)
                self.assertTrue(action.get("PROGRAM") or action.get("POSIX_PROGRAM"))
                self.assertTrue(action.get("PROGRAM") or action.get("WINDOWS_PROGRAM"))

    def test_launch_menu_hands_terminal_to_nested_tui(self) -> None:
        action = _read_env(PORTUI_DIR / "actions" / "01-launch-menu.env")

        self.assertEqual(action["INTERACTIVE"], "true")
        self.assertEqual(action["TIMEOUT_SECONDS"], "0")

    def test_project_local_portui_runtime_is_vendored(self) -> None:
        for relative_path in [
            "portui.ps1",
            "portui.sh",
            "portui.cmd",
            ".portui-runtime/portui.ps1",
            ".portui-runtime/portui.sh",
            ".portui-runtime/portui.cmd",
        ]:
            with self.subTest(path=relative_path):
                self.assertTrue((ROOT / relative_path).is_file())

        self.assertIn(".portui-runtime", (ROOT / "portui.ps1").read_text(encoding="utf-8"))
        self.assertIn(".portui-runtime", (ROOT / "portui.sh").read_text(encoding="utf-8"))
        self.assertIn(".portui-runtime", (ROOT / "portui.cmd").read_text(encoding="utf-8"))

    def test_json_actions_keep_machine_readable_cli_args(self) -> None:
        json_action_ids = {"doctor", "discover-models", "list-benchmarks"}
        for action_path in sorted((PORTUI_DIR / "actions").glob("*.env")):
            action = _read_env(action_path)
            if action.get("ID") not in json_action_ids:
                continue
            with self.subTest(action=action["ID"]):
                args = "|".join([action.get("ARGS", ""), action.get("POSIX_ARGS", ""), action.get("WINDOWS_ARGS", "")])
                self.assertIn("--json", args)
                self.assertNotEqual(action.get("INTERACTIVE", "false"), "true")

    def test_windows_batch_launcher_does_not_hide_child_failures(self) -> None:
        launcher = (ROOT / "start.bat").read_text(encoding="utf-8")

        self.assertIn(":use_venv", launcher)
        self.assertIn(":use_py", launcher)
        self.assertIn(":use_python", launcher)
        self.assertIsNone(re.search(r"\([^)]*exit /b %errorlevel%[^)]*\)", launcher, re.IGNORECASE | re.DOTALL))


if __name__ == "__main__":
    unittest.main()
