"""Huddle AI - Main entry point."""

import sys

from rich.console import Console

from src.ui.menus import show_banner, show_main_menu, show_transcript_menu
from src.ui.processing import run_with_progress
from src.ui.results import display_results

console = Console()


def _handle_process() -> None:
    """Handle the Process Meeting Transcript flow."""
    result = show_transcript_menu()
    if result is None:
        return

    transcript, label = result
    console.print(f"Processing: [bold cyan]{label}[/bold cyan]\n")

    state = run_with_progress(transcript, source="sample" if label != "Pasted Transcript" else "pasted")
    display_results(state)


def _handle_assistant() -> None:
    """Placeholder for assistant mode (Phase 3)."""
    console.print(
        "\n[yellow]Assistant mode is not yet implemented (coming in Phase 3).[/yellow]\n"
    )


def main() -> None:
    """Run the Huddle AI main loop."""
    show_banner()

    while True:
        choice = show_main_menu()
        if choice is None:
            console.print("\n[dim]Goodbye![/dim]\n")
            break
        elif choice == "process":
            _handle_process()
        elif choice == "assistant":
            _handle_assistant()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/dim]\n")
        sys.exit(0)
