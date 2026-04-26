"""
Cross-platform arrow-key terminal menus for smaLLMs.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.pipeline.benchmarks import (
    DEFAULT_BENCHMARKS,
    list_benchmark_suites,
    list_supported_benchmarks,
)


if os.name == "nt":
    import ctypes
    import msvcrt
else:
    import select
    import termios
    import tty


@dataclass(frozen=True)
class MenuOption:
    """One selectable terminal option."""

    label: str
    value: Any
    hint: str = ""
    disabled: bool = False


class TerminalMenuError(RuntimeError):
    """Raised when the menu cannot run in the current terminal."""


class CrossPlatformKeyReader:
    """Read arrow-key input on both Windows and POSIX terminals."""

    def __init__(self) -> None:
        self._fd: Optional[int] = None
        self._old_settings: Any = None

    def __enter__(self) -> "CrossPlatformKeyReader":
        if not sys.stdin.isatty():
            raise TerminalMenuError("Interactive terminal input is required for arrow-key mode.")

        if os.name != "nt":
            self._fd = sys.stdin.fileno()
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if os.name != "nt" and self._fd is not None and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def read_key(self) -> str:
        """Read one normalized key token."""
        if os.name == "nt":
            return self._read_windows_key()
        return self._read_posix_key()

    def _read_windows_key(self) -> str:
        while True:
            char = msvcrt.getwch()
            if char in {"\x00", "\xe0"}:
                extended = msvcrt.getwch()
                mapping = {"H": "up", "P": "down", "K": "left", "M": "right"}
                return mapping.get(extended, "unknown")
            if char in {"\r", "\n"}:
                return "enter"
            if char == " ":
                return "space"
            if char == "\x1b":
                return "escape"
            if char in {"\x08", "\x7f"}:
                return "backspace"
            if char == "\x03":
                return "interrupt"
            return char.lower()

    def _read_posix_key(self) -> str:
        char = sys.stdin.read(1)
        if char in {"\r", "\n"}:
            return "enter"
        if char == " ":
            return "space"
        if char == "\x03":
            return "interrupt"
        if char in {"\x7f", "\b"}:
            return "backspace"
        if char != "\x1b":
            return char.lower()

        ready, _, _ = select.select([sys.stdin], [], [], 0.03)
        if not ready:
            return "escape"

        second = sys.stdin.read(1)
        if second != "[":
            return "escape"

        third = sys.stdin.read(1)
        mapping = {"A": "up", "B": "down", "C": "right", "D": "left"}
        return mapping.get(third, "escape")


class TerminalMenuApp:
    """Fullscreen-ish terminal UI built on plain ANSI control sequences."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    GRAY = "\033[90m"

    def __init__(self, app: Any) -> None:
        self.app = app
        self._enable_virtual_terminal_sequences()

    def run(self) -> None:
        """Run the main terminal menu loop."""
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            raise TerminalMenuError("Arrow-key mode requires an interactive TTY.")

        with CrossPlatformKeyReader() as reader:
            self._hide_cursor()
            try:
                while True:
                    action = self._select_main_action(reader)
                    if action == "run":
                        self._run_benchmark_flow(reader)
                    elif action == "setup":
                        self._show_setup_status(reader)
                    elif action == "discover":
                        self._show_discovered_models(reader)
                    elif action == "catalog":
                        self._show_benchmark_catalog(reader)
                    elif action == "status":
                        self._show_latest_run(reader)
                    elif action == "export":
                        self._export_latest_run(reader)
                    elif action == "quit":
                        break
            except KeyboardInterrupt:
                pass
            finally:
                self._show_cursor()
                self._clear_screen()

    def _select_main_action(self, reader: CrossPlatformKeyReader) -> str:
        options = [
            MenuOption("Run local benchmark", "run", "Select models, benchmark suite, and sample count"),
            MenuOption("Setup and model status", "setup", "Check whether Ollama or LM Studio is ready to use"),
            MenuOption("Discover local models", "discover", "Inspect Ollama and LM Studio endpoints"),
            MenuOption("Browse supported benchmarks", "catalog", "Runnable suites and benchmarks"),
            MenuOption("Latest run summary", "status", "Open the newest artifact summary"),
            MenuOption("Export latest run", "export", "Refresh website bundle from the latest artifacts"),
            MenuOption("Quit", "quit", "Exit smaLLMs"),
        ]
        return self._select_one(
            reader,
            title="smaLLMs",
            subtitle="Local-first benchmarking. If Ollama already has models installed, smaLLMs will reuse them automatically.",
            options=options,
        )

    def _show_setup_status(self, reader: CrossPlatformKeyReader) -> None:
        self._show_message(
            reader,
            title="Setup and Model Status",
            lines=self.app.build_setup_status_lines(),
        )

    def _run_benchmark_flow(self, reader: CrossPlatformKeyReader) -> None:
        orchestrator = self.app._get_orchestrator()
        discovered = asyncio.run(orchestrator.discover_local_models())
        all_models = self._flatten_models(discovered)
        if not all_models:
            self._show_message(
                reader,
                title="No Local Models Found",
                lines=[
                    "smaLLMs could not reach any local model endpoint.",
                    "",
                    "Checked providers:",
                    "  - Ollama on http://localhost:11434",
                    "  - LM Studio on http://localhost:1234",
                    "",
                    "If you already pulled Ollama models before, you do not need to pull them again.",
                    "Just make sure Ollama or LM Studio is running, then retry this screen.",
                ],
            )
            return

        model_options = [
            MenuOption(
                label=model["name"],
                value=model["name"],
                hint=self._format_model_hint(model),
            )
            for model in all_models
        ]
        selected_models = self._select_many(
            reader,
            title="Select Models",
            subtitle="Use arrow keys to move, space to toggle, enter to continue.",
            options=model_options,
            default_values=[model["name"] for model in all_models],
        )
        if not selected_models:
            return

        benchmark_mode = self._select_one(
            reader,
            title="Benchmark Scope",
            subtitle="Suites are faster to operate; custom selection gives full control.",
            options=self._benchmark_scope_options(),
        )
        if benchmark_mode is None:
            return

        benchmarks: List[str]
        if benchmark_mode[0] == "suite":
            benchmarks = [benchmark_mode[1]]
        else:
            benchmark_options = [
                MenuOption(
                    label=f"{entry['display_name']} ({entry['key']})",
                    value=entry["key"],
                    hint=f"{entry['category']} | {entry['description']}",
                )
                for entry in list_supported_benchmarks()
            ]
            benchmarks = self._select_many(
                reader,
                title="Custom Benchmark Selection",
                subtitle="Only locally runnable benchmarks appear here.",
                options=benchmark_options,
                default_values=list(DEFAULT_BENCHMARKS),
            )
            if not benchmarks:
                return

        sample_count = self._select_one(
            reader,
            title="Sample Count",
            subtitle="Higher counts are slower but more stable.",
            options=[
                MenuOption("3 samples", 3, "Quick smoke test"),
                MenuOption("10 samples", 10, "Useful default"),
                MenuOption("25 samples", 25, "Stable signal"),
                MenuOption("50 samples", 50, "Leaderboard-style comparison"),
                MenuOption("100 samples", 100, "Long-form serious sweep"),
                MenuOption("Back", None, "Return to the previous screen"),
            ],
        )
        if sample_count is None:
            return

        benchmark_label = ", ".join(benchmarks)
        confirmed = self._select_one(
            reader,
            title="Confirm Run",
            subtitle="Review the benchmark job before it starts.",
            options=[
                MenuOption(
                    "Start run",
                    True,
                    f"{len(selected_models)} model(s) | {benchmark_label} | {sample_count} samples",
                ),
                MenuOption("Back", False, "Return to the previous screen"),
            ],
        )
        if not confirmed:
            return

        self._clear_screen()
        try:
            asyncio.run(
                self.app.run_benchmarks(
                    models=selected_models,
                    benchmarks=benchmarks,
                    samples=sample_count,
                    export_after_run=True,
                    all_local=True,
                )
            )
            self._wait_for_key(reader, "Run finished. Press any key to return to the menu.")
        except (KeyboardInterrupt, asyncio.CancelledError):
            self.app.terminal.finish_live_screen()
            self._show_message(
                reader,
                title="Run Cancelled",
                lines=[
                    "Benchmark run cancelled.",
                    "",
                    "Any benchmark artifacts written before cancellation are still in the artifacts directory.",
                ],
            )
        except Exception as exc:
            self._show_message(
                reader,
                title="Run Failed",
                lines=[
                    f"{type(exc).__name__}: {exc}",
                    "",
                    "The live run renderer exited early. Review the terminal output or logs and retry.",
                ],
            )

    def _benchmark_scope_options(self) -> List[MenuOption]:
        """Build benchmark suite options from the shared catalog."""
        options = [
            MenuOption(
                label=suite["display_name"],
                value=("suite", suite["key"]),
                hint=(
                    f"{suite['description']} "
                    f"({len(suite['benchmarks'])} benchmark{'s' if len(suite['benchmarks']) != 1 else ''})"
                ),
            )
            for suite in list_benchmark_suites()
        ]
        options.extend(
            [
                MenuOption("Custom runnable benchmarks", ("custom", None), "Select individual runnable benchmarks"),
                MenuOption("Back", None, "Return to the main menu"),
            ]
        )
        return options

    def _show_discovered_models(self, reader: CrossPlatformKeyReader) -> None:
        discovered = asyncio.run(self.app._get_orchestrator().discover_local_models())
        lines: List[str] = []
        total = 0
        for provider in ["ollama", "lm_studio"]:
            models = discovered.get(provider, [])
            total += len(models)
            lines.append(f"{provider}: {len(models)} model(s)")
            if models:
                for model in models:
                    lines.append(f"  - {model['name']} | {self._format_model_hint(model)}")
            else:
                lines.append("  - none detected")
            lines.append("")

        if total == 0:
            lines.append("No reachable local models were discovered.")
        self._show_message(reader, title="Local Model Discovery", lines=lines)

    def _show_benchmark_catalog(self, reader: CrossPlatformKeyReader) -> None:
        suites = list_benchmark_suites()
        benchmarks = list_supported_benchmarks()
        lines: List[str] = []
        lines.append(f"Supported suites: {len(suites)}")
        for suite in suites:
            lines.append(
                f"  - {suite['key']} | {suite['display_name']} | {', '.join(suite['benchmarks'])}"
            )
        lines.append("")
        lines.append(f"Supported benchmarks: {len(benchmarks)}")
        for entry in benchmarks:
            lines.append(
                f"  - {entry['key']} [{entry['category']}] | {entry['display_name']} | {entry['description']}"
            )
        self._show_message(reader, title="Supported Benchmarks", lines=lines)

    def _show_latest_run(self, reader: CrossPlatformKeyReader) -> None:
        latest_run_id = self.app.artifact_store.latest_run_id()
        if not latest_run_id:
            self._show_message(
                reader,
                title="Latest Run",
                lines=["No run artifacts exist yet.", "", "Start a benchmark run first."],
            )
            return

        run_data = self.app.artifact_store.load_run(latest_run_id)
        summary = run_data.get("summary", {})
        totals = summary.get("totals", {})
        leaderboard = summary.get("leaderboard", [])

        lines = [
            f"Run ID: {latest_run_id}",
            f"Models: {totals.get('models', 0)}",
            f"Benchmarks: {totals.get('benchmarks', 0)}",
            f"Evaluations: {totals.get('evaluations', 0)}",
            f"Samples: {totals.get('samples', 0)}",
            f"Accuracy: {totals.get('accuracy', 0.0):.4f}",
            f"Errors: {totals.get('errors', 0)}",
            f"Total tokens: {totals.get('total_tokens', 0)}",
            "",
            "Top leaderboard rows:",
        ]
        if leaderboard:
            for row in leaderboard[:10]:
                lines.append(
                    f"  - #{row.get('rank', '?')} {row['model_name']} | "
                    f"acc {row.get('overall_accuracy', 0.0):.4f} | "
                    f"lat {row.get('avg_latency_sec', 0.0):.2f}s"
                )
        else:
            lines.append("  - no completed evaluations")
        self._show_message(reader, title="Latest Run Summary", lines=lines)

    def _export_latest_run(self, reader: CrossPlatformKeyReader) -> None:
        try:
            exported = self.app.exporter.export_run()
        except Exception as exc:
            self._show_message(
                reader,
                title="Export Failed",
                lines=[f"{type(exc).__name__}: {exc}"],
            )
            return

        lines = ["Website bundle refreshed from the latest run.", ""]
        for name, path in exported.items():
            lines.append(f"{name}: {path}")
        self._show_message(reader, title="Export Complete", lines=lines)

    def _select_one(
        self,
        reader: CrossPlatformKeyReader,
        title: str,
        subtitle: str,
        options: Sequence[MenuOption],
    ) -> Any:
        if not options:
            return None

        index = self._first_enabled_index(options)
        while True:
            self._render_options(
                title=title,
                subtitle=subtitle,
                options=options,
                cursor=index,
                selected=None,
                multi_select=False,
            )
            key = reader.read_key()
            if key == "interrupt":
                raise KeyboardInterrupt
            if key == "up":
                index = self._move_cursor(options, index, -1)
            elif key == "down":
                index = self._move_cursor(options, index, 1)
            elif key == "enter":
                option = options[index]
                if not option.disabled:
                    return option.value
            elif key in {"escape", "q"}:
                return None

    def _select_many(
        self,
        reader: CrossPlatformKeyReader,
        title: str,
        subtitle: str,
        options: Sequence[MenuOption],
        default_values: Optional[Sequence[Any]] = None,
    ) -> List[Any]:
        if not options:
            return []

        selected: Set[Any] = set(default_values or [])
        index = self._first_enabled_index(options)
        while True:
            self._render_options(
                title=title,
                subtitle=subtitle,
                options=options,
                cursor=index,
                selected=selected,
                multi_select=True,
            )
            key = reader.read_key()
            if key == "interrupt":
                raise KeyboardInterrupt
            if key == "up":
                index = self._move_cursor(options, index, -1)
            elif key == "down":
                index = self._move_cursor(options, index, 1)
            elif key == "space":
                option = options[index]
                if option.disabled:
                    continue
                if option.value in selected:
                    selected.remove(option.value)
                else:
                    selected.add(option.value)
            elif key == "a":
                enabled_values = {option.value for option in options if not option.disabled}
                if selected == enabled_values:
                    selected.clear()
                else:
                    selected = set(enabled_values)
            elif key == "enter":
                if selected:
                    ordered = [option.value for option in options if option.value in selected]
                    return ordered
            elif key in {"escape", "q"}:
                return []

    def _render_options(
        self,
        title: str,
        subtitle: str,
        options: Sequence[MenuOption],
        cursor: int,
        selected: Optional[Set[Any]],
        multi_select: bool,
    ) -> None:
        self._clear_screen()
        width = max(72, min(shutil.get_terminal_size((120, 40)).columns, 140))
        print(
            f"{self.CYAN}╔{'═' * (width - 2)}╗{self.RESET}\n"
            f"{self.CYAN}║ {self.BOLD}{title}{self.RESET}{self.CYAN}{' ' * max(0, width - len(title) - 3)}║{self.RESET}\n"
            f"{self.CYAN}╚{'═' * (width - 2)}╝{self.RESET}"
        )
        print(f"{self.GRAY}{subtitle}{self.RESET}\n")

        start, end = self._window_bounds(cursor, len(options), max_visible=12)
        for idx in range(start, end):
            option = options[idx]
            is_cursor = idx == cursor
            cursor_prefix = f"{self.CYAN}›{self.RESET}" if is_cursor else " "
            if multi_select:
                checkbox = "[x]" if selected and option.value in selected else "[ ]"
                marker = checkbox
            else:
                marker = "•"

            label_color = self.YELLOW if is_cursor else self.WHITE
            if option.disabled:
                label_color = self.GRAY

            print(f"{cursor_prefix} {label_color}{marker} {option.label}{self.RESET}")
            if option.hint:
                print(f"    {self.GRAY}{option.hint}{self.RESET}")

        if end < len(options):
            print(f"\n{self.GRAY}… more options below …{self.RESET}")

        footer = "Use ↑/↓ to move, Enter to confirm, Q or Esc to go back."
        if multi_select:
            footer = "Use ↑/↓ to move, Space to toggle, A to select all, Enter to confirm, Q or Esc to go back."
        print(f"\n{self.GRAY}{footer}{self.RESET}")
        if multi_select:
            print(f"{self.GRAY}Selected: {len(selected or set())}{self.RESET}")
        sys.stdout.flush()

    def _show_message(self, reader: CrossPlatformKeyReader, title: str, lines: Sequence[str]) -> None:
        self._clear_screen()
        print(f"{self.BOLD}{title}{self.RESET}\n")
        for line in lines:
            print(line)
        self._wait_for_key(reader, "\nPress any key to return.")

    def _wait_for_key(self, reader: CrossPlatformKeyReader, prompt: str) -> None:
        print(prompt)
        sys.stdout.flush()
        reader.read_key()

    def _window_bounds(self, cursor: int, total: int, max_visible: int) -> Tuple[int, int]:
        if total <= max_visible:
            return 0, total
        half = max_visible // 2
        start = max(0, cursor - half)
        end = min(total, start + max_visible)
        start = max(0, end - max_visible)
        return start, end

    def _move_cursor(self, options: Sequence[MenuOption], current: int, step: int) -> int:
        index = current
        for _ in range(len(options)):
            index = (index + step) % len(options)
            if not options[index].disabled:
                return index
        return current

    def _first_enabled_index(self, options: Sequence[MenuOption]) -> int:
        for index, option in enumerate(options):
            if not option.disabled:
                return index
        return 0

    def _flatten_models(self, discovered: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        models: List[Dict[str, Any]] = []
        for provider in ["ollama", "lm_studio"]:
            models.extend(discovered.get(provider, []))
        return models

    def _format_model_hint(self, model: Dict[str, Any]) -> str:
        details = [model.get("provider", "unknown")]
        size_gb = float(model.get("size_gb") or 0.0)
        if size_gb:
            details.append(f"{size_gb:.1f}GB")
        parameters = model.get("parameters")
        if parameters and parameters != "unknown":
            details.append(str(parameters))
        family = model.get("family")
        if family and family != "unknown":
            details.append(str(family))
        quantization = model.get("quantization")
        if quantization and quantization != "unknown":
            details.append(str(quantization))
        return " | ".join(details)

    def _clear_screen(self) -> None:
        print("\033[2J\033[H", end="")

    def _hide_cursor(self) -> None:
        print("\033[?25l", end="")
        sys.stdout.flush()

    def _show_cursor(self) -> None:
        print("\033[?25h", end="")
        sys.stdout.flush()

    def _enable_virtual_terminal_sequences(self) -> None:
        if os.name != "nt":
            return
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
