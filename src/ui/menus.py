"""Main menu and navigation UI components."""

import re
import sys
from pathlib import Path

from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.config import settings

console = Console()

BANNER = r"""
 ╦ ╦╦ ╦╔╦╗╔╦╗╦  ╔═╗  ╔═╗╦
 ╠═╣║ ║ ║║ ║║║  ║╣   ╠═╣║
 ╩ ╩╚═╝═╩╝═╩╝╩═╝╚═╝  ╩ ╩╩
"""


def show_banner() -> None:
    """Display the application banner."""
    console.print(
        Panel(
            Text(BANNER, style="bold cyan", justify="center")
            + Text("\n  Meeting Intelligence Agent", style="dim", justify="center"),
            border_style="blue",
            expand=False,
            padding=(0, 4),
        )
    )


def show_main_menu() -> str | None:
    """Display main menu and return user choice.

    Returns:
        "process", "assistant", or None to exit.
    """
    console.print()
    console.print("[bold]What would you like to do?[/bold]\n")
    console.print("  [cyan]1.[/cyan] Process Meeting Transcript")
    console.print("  [cyan]2.[/cyan] Assistant (RAG-powered chat)")
    console.print("  [dim]q.[/dim] Quit\n")

    try:
        choice = pt_prompt(HTML("<b>> </b>")).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None

    if choice in ("1", "process"):
        return "process"
    if choice in ("2", "assistant"):
        return "assistant"
    if choice in ("q", "quit", "exit"):
        return None

    console.print("[red]Invalid choice. Please enter 1, 2, or q.[/red]")
    return show_main_menu()


def _parse_transcript_metadata(path: Path) -> dict:
    """Extract metadata from the first few lines of a transcript file."""
    text = path.read_text()
    lines = text.strip().splitlines()

    title = path.stem.replace("-", " ").title()
    date = ""
    duration = ""
    attendees = ""

    for line in lines[:5]:
        if line.startswith("Meeting:"):
            title = line.split(":", 1)[1].strip()
        elif line.startswith("Date:"):
            date = line.split(":", 1)[1].strip()
        elif line.startswith("Duration:"):
            duration = line.split(":", 1)[1].strip()
        elif line.startswith("Attendees:"):
            attendees = line.split(":", 1)[1].strip()

    return {
        "path": path,
        "title": title,
        "date": date,
        "duration": duration,
        "attendees": attendees,
        "size": len(text),
    }


def show_transcript_menu() -> tuple[str, str] | None:
    """Show transcript selection submenu.

    Returns:
        Tuple of (transcript_text, source_label), or None to go back.
    """
    console.print()
    console.print("[bold]Select transcript source:[/bold]\n")
    console.print("  [cyan]1.[/cyan] Use Sample Transcript")
    console.print("  [cyan]2.[/cyan] Paste Transcript")
    console.print("  [dim]b.[/dim] Back\n")

    try:
        choice = pt_prompt(HTML("<b>> </b>")).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None

    if choice in ("1", "sample"):
        return _select_sample_transcript()
    if choice in ("2", "paste"):
        return _paste_transcript()
    if choice in ("b", "back"):
        return None

    console.print("[red]Invalid choice.[/red]")
    return show_transcript_menu()


def _select_sample_transcript() -> tuple[str, str] | None:
    """Display navigable list of sample transcripts."""
    transcripts_dir = settings.SAMPLE_TRANSCRIPTS_DIR
    files = sorted(transcripts_dir.glob("*.txt"))

    if not files:
        console.print("[red]No sample transcripts found.[/red]")
        return None

    samples = [_parse_transcript_metadata(f) for f in files]

    console.print()
    table = Table(
        title="Sample Transcripts",
        show_lines=True,
        border_style="blue",
        title_style="bold blue",
    )
    table.add_column("#", style="cyan", width=3, justify="right")
    table.add_column("Meeting", style="bold", min_width=20)
    table.add_column("Date", style="dim")
    table.add_column("Duration", style="dim")
    table.add_column("Attendees", max_width=40, overflow="ellipsis")

    for i, s in enumerate(samples, 1):
        table.add_row(
            str(i),
            s["title"],
            s["date"],
            s["duration"],
            s["attendees"],
        )

    console.print(table)
    console.print()

    try:
        choice = pt_prompt(
            HTML("<b>Select transcript # (or <i>b</i> to go back): </b>")
        ).strip()
    except (EOFError, KeyboardInterrupt):
        return None

    if choice.lower() in ("b", "back"):
        return None

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(samples):
            selected = samples[idx]
            transcript = selected["path"].read_text()
            label = selected["title"]
            console.print(f"\nSelected: [bold cyan]{label}[/bold cyan]")
            console.print(f"[dim]{len(transcript):,} characters[/dim]\n")
            return transcript, label
        else:
            console.print("[red]Number out of range.[/red]")
            return _select_sample_transcript()
    except ValueError:
        console.print("[red]Please enter a valid number.[/red]")
        return _select_sample_transcript()


def _paste_transcript() -> tuple[str, str] | None:
    """Accept pasted transcript with preview."""
    console.print()
    console.print(
        "[bold]Paste your meeting transcript below.[/bold]"
    )
    console.print(
        "[dim]Enter an empty line followed by Ctrl+D (or type END on its own line) to finish.[/dim]\n"
    )

    lines: list[str] = []
    try:
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
    except EOFError:
        pass
    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        return None

    transcript = "\n".join(lines).strip()
    if not transcript or len(transcript) < 20:
        console.print("[red]Transcript is too short (minimum 20 characters).[/red]")
        return None

    # Show preview
    preview = transcript[:200]
    if len(transcript) > 200:
        preview += "..."

    console.print()
    console.print(
        Panel(
            preview,
            title="Preview",
            border_style="yellow",
            expand=False,
        )
    )
    console.print(f"[dim]Total: {len(transcript):,} characters[/dim]\n")

    try:
        confirm = pt_prompt(
            HTML("<b>Process this transcript? (y/n): </b>")
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None

    if confirm in ("y", "yes"):
        return transcript, "Pasted Transcript"

    console.print("[dim]Cancelled.[/dim]")
    return None
