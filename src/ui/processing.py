"""Live progress display during transcript processing."""

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from src.agent import PROCESSING_STEPS, STEP_LABELS, process_transcript_stream

console = Console()


def _build_progress(completed: list[str], active: str | None, skipped: set[str]) -> Group:
    """Build a renderable showing step progress."""
    lines: list[Text] = []

    for step in PROCESSING_STEPS:
        label = STEP_LABELS.get(step, step)
        if step in completed:
            line = Text()
            line.append("  \u2714 ", style="green")
            line.append(label, style="green")
            lines.append(line)
        elif step in skipped:
            line = Text()
            line.append("  \u2500 ", style="dim")
            line.append(label, style="dim strikethrough")
            lines.append(line)
        elif step == active:
            line = Text()
            line.append("  \u25cf ", style="yellow")
            line.append(label, style="bold yellow")
            line.append(" ...", style="dim yellow")
            lines.append(line)
        else:
            line = Text()
            line.append("  \u25cb ", style="dim")
            line.append(label, style="dim")
            lines.append(line)

    return Group(*lines)


def run_with_progress(transcript: str, source: str = "sample") -> dict:
    """Process a transcript while showing live progress.

    Returns:
        Final accumulated state dict.
    """
    completed: list[str] = []
    skipped: set[str] = set()
    final_state: dict = {}

    # Determine which step comes next so we can show it as active
    remaining = list(PROCESSING_STEPS)

    with Live(console=console, refresh_per_second=8, transient=False) as live:
        # Show initial state with first step active
        active = remaining[0] if remaining else None
        live.update(
            Panel(
                _build_progress(completed, active, skipped),
                title="[bold blue]Processing Transcript[/bold blue]",
                border_style="blue",
                expand=False,
                padding=(1, 2),
            )
        )

        for node_name, state_update in process_transcript_stream(transcript, source):
            # Merge update into final state
            for k, v in state_update.items():
                if isinstance(v, list) and isinstance(final_state.get(k), list):
                    final_state[k] = v
                else:
                    final_state[k] = v

            # Skip the internal trace metadata event in progress display
            if node_name == "__trace__":
                continue

            # Mark this node as completed
            completed.append(node_name)
            if node_name in remaining:
                remaining.remove(node_name)

            # If extract_action_items yielded no items, mark later steps as skipped
            if node_name == "extract_action_items" and not final_state.get("action_items"):
                for s in ("assign_owners", "determine_deadlines", "send_emails"):
                    skipped.add(s)
                    if s in remaining:
                        remaining.remove(s)

            # Next active step
            active = remaining[0] if remaining else None

            live.update(
                Panel(
                    _build_progress(completed, active, skipped),
                    title="[bold blue]Processing Transcript[/bold blue]",
                    border_style="blue",
                    expand=False,
                    padding=(1, 2),
                )
            )

        # Final: show all done
        live.update(
            Panel(
                _build_progress(completed, None, skipped),
                title="[bold green]Processing Complete[/bold green]",
                border_style="green",
                expand=False,
                padding=(1, 2),
            )
        )

    return final_state
