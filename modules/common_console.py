from __future__ import annotations

from collections import deque

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text


class MergeConsole:
    def __init__(self):
        self._console = Console()
        self._capture_console = Console(
            width=120,
            force_terminal=False,
            color_system=None,
            no_color=True,
        )
        self._live: Live | None = None
        self._logs: deque[str] = deque(maxlen=14)
        self._run_overview = Panel(
            Text("Waiting for configuration...", style="dim"),
            title="[bold]Run Overview[/bold]",
            border_style="blue",
        )
        self._request_overview = Panel(
            Text("No active merge request.", style="dim"),
            title="[bold]Current Request[/bold]",
            border_style="cyan",
        )
        self._merge_info = Panel(
            Text("No active layer.", style="dim"),
            title="[bold]Merge Info[/bold]",
            border_style="magenta",
        )
        self._step_progress = self._create_progress()
        self._layer_progress = self._create_progress()
        self._step_task_id = self._step_progress.add_task("Configs", total=1, completed=0)
        self._layer_task_id = self._layer_progress.add_task(
            "Layers",
            total=1,
            completed=0,
            visible=False,
        )

    @property
    def is_live(self) -> bool:
        return self._live is not None

    def _create_progress(self) -> Progress:
        return Progress(
            SpinnerColumn(),
            TextColumn("{task.description}", justify="left"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            expand=True,
        )

    def _render_dashboard(self):
        logs_text = (
            "\n".join(self._logs)
            if self._logs
            else "No logs yet. Warnings and status messages will appear here."
        )
        progress_group = Group(self._step_progress, self._layer_progress)

        layout = Layout()
        layout.split_column(
            Layout(name="top", size=11),
            Layout(name="middle", size=11),
            Layout(name="bottom"),
        )
        layout["top"].split_row(
            Layout(self._run_overview, name="overview"),
            Layout(self._request_overview, name="request"),
        )
        layout["middle"].split_row(
            Layout(self._merge_info, name="merge_info"),
            Layout(
                Panel(
                    progress_group,
                    title="[bold]Progress[/bold]",
                    border_style="green",
                ),
                name="progress",
            ),
        )
        layout["bottom"].update(
            Panel(logs_text, title="[bold]Recent Logs[/bold]", border_style="yellow")
        )
        return layout

    def _refresh(self):
        if self._live is not None:
            self._live.update(self._render_dashboard(), refresh=True)

    def _capture_text(self, *objects, sep=" ", end="\n", **kwargs) -> str:
        with self._capture_console.capture() as capture:
            self._capture_console.print(*objects, sep=sep, end=end, **kwargs)
        return capture.get().strip()

    def _append_log(self, message: str):
        if not message:
            return
        for line in message.splitlines():
            self._logs.append(line.rstrip())

    def start_live(self):
        if self._live is not None or not self._console.is_terminal:
            return

        self._live = Live(
            self._render_dashboard(),
            console=self._console,
            refresh_per_second=8,
            transient=False,
            screen=False,
        )
        self._live.start()
        self._refresh()

    def stop_live(self):
        if self._live is None:
            return
        self._refresh()
        self._live.stop()
        self._live = None

    def print(self, *objects, sep=" ", end="\n", **kwargs):
        if not self.is_live:
            self._console.print(*objects, sep=sep, end=end, **kwargs)
            return

        self._append_log(self._capture_text(*objects, sep=sep, end=end, **kwargs))
        self._refresh()

    def rule(self, title="", **kwargs):
        if not self.is_live:
            self._console.rule(title, **kwargs)
            return

        normalized_title = title if isinstance(title, str) else str(title)
        self._append_log(f"=== {normalized_title} ===")
        self._refresh()

    def set_run_overview(self, renderable):
        if not self.is_live:
            self._console.print(renderable)
            return
        self._run_overview = renderable
        self._refresh()

    def set_request_overview(self, renderable):
        if not self.is_live:
            self._console.print(renderable)
            return
        self._request_overview = renderable
        self._refresh()

    def set_merge_info(self, renderable):
        if not self.is_live:
            self._console.print(renderable)
            return
        self._merge_info = renderable
        self._refresh()

    def configure_run_progress(self, total_steps: int):
        self._step_progress.update(
            self._step_task_id,
            total=max(total_steps, 1),
            completed=0,
            description="Configs",
            visible=True,
        )
        self._refresh()

    def start_request(self, index: int, total: int, name: str, operation: str):
        self._step_progress.update(
            self._step_task_id,
            description=f"Configs {index + 1}/{total}: {name} ({operation})",
        )
        self._refresh()

    def advance_request(self, status: str | None = None):
        description = "Configs"
        if status:
            description = f"Configs ({status})"
        self._step_progress.update(
            self._step_task_id,
            advance=1,
            description=description,
        )
        self._refresh()

    def reset_layer_progress(self, total_layers: int, description: str):
        visible = total_layers > 0
        self._layer_progress.update(
            self._layer_task_id,
            total=max(total_layers, 1),
            completed=0,
            description=description,
            visible=visible,
        )
        self._refresh()

    def advance_layer(self, layer_key: str | None = None):
        description = "Layers"
        if layer_key:
            description = f"Layer: {layer_key}"
        self._layer_progress.update(
            self._layer_task_id,
            advance=1,
            description=description,
        )
        self._refresh()

    def complete_layer_progress(self):
        task = self._layer_progress.tasks[self._layer_task_id]
        self._layer_progress.update(
            self._layer_task_id,
            completed=task.total,
            description="Layers complete",
            visible=task.total > 0,
        )
        self._refresh()


console = MergeConsole()
