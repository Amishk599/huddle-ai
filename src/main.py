"""Huddle AI - Main entry point."""

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.agent import process_transcript
from src.config import settings

console = Console()


def main():
    """Run Huddle AI - process a sample transcript."""
    console.print(
        Panel("[bold blue]Huddle AI[/bold blue]\nMeeting Intelligence Agent", expand=False)
    )

    # Load sample transcript
    transcript_path = settings.SAMPLE_TRANSCRIPTS_DIR / "meeting-with-todd.txt"
    if not transcript_path.exists():
        console.print("[red]Sample transcript not found.[/red]")
        return

    transcript = transcript_path.read_text()
    console.print(f"\nProcessing: [cyan]{transcript_path.name}[/cyan]")
    console.print(f"Transcript length: {len(transcript)} chars\n")

    # Process
    console.print("[yellow]Processing transcript...[/yellow]\n")
    result = process_transcript(transcript, source="sample")

    # Display results
    if result.get("errors"):
        console.print("[red]Errors:[/red]")
        for error in result["errors"]:
            console.print(f"  - {error}")

    # Summary
    console.print(Panel(result.get("summary", "N/A"), title="Meeting Summary"))

    if result.get("key_topics"):
        console.print("[bold]Key Topics:[/bold]", ", ".join(result["key_topics"]))

    if result.get("participants"):
        console.print("[bold]Participants:[/bold]", ", ".join(result["participants"]))

    # Action items
    items = result.get("action_items", [])
    if items:
        console.print()
        table = Table(title=f"Action Items ({len(items)})")
        table.add_column("#", style="dim")
        table.add_column("Description")
        table.add_column("Assignee")
        table.add_column("Priority")
        table.add_column("Deadline")

        for item in items:
            priority = item.get("priority", "MEDIUM")
            priority_color = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "green"}.get(
                priority, "white"
            )
            table.add_row(
                item.get("id", ""),
                item.get("description", ""),
                f"{item.get('assignee', 'N/A')} ({item.get('assignee_email', '')})",
                f"[{priority_color}]{priority}[/{priority_color}]",
                item.get("deadline", "N/A"),
            )

        console.print(table)
    else:
        console.print("\n[dim]No action items found.[/dim]")

    # Emails sent
    emails = result.get("emails_sent", [])
    if emails:
        console.print(f"\n[bold]Emails Sent ({len(emails)}):[/bold]")
        for email in emails:
            console.print(f"  [green]✓[/green] {email}")

    console.print()


if __name__ == "__main__":
    main()
