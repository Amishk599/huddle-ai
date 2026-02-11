"""Results display formatting."""

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

PRIORITY_STYLES = {
    "HIGH": ("red", "\u25b2"),
    "MEDIUM": ("yellow", "\u25cf"),
    "LOW": ("green", "\u25bc"),
}


def display_results(state: dict) -> None:
    """Display full processing results with Rich formatting."""
    console.print()

    # Errors
    errors = state.get("errors", [])
    if errors:
        error_text = "\n".join(f"  - {e}" for e in errors)
        console.print(
            Panel(
                error_text,
                title="[bold red]Errors[/bold red]",
                border_style="red",
                expand=False,
            )
        )
        console.print()

    # Summary
    summary = state.get("summary", "N/A")
    console.print(
        Panel(
            summary,
            title="[bold blue]Meeting Summary[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # Topics + Participants side by side
    topics = state.get("key_topics", [])
    participants = state.get("participants", [])

    if topics or participants:
        parts = []
        if topics:
            topic_text = Text()
            for t in topics:
                topic_text.append("  \u2022 ", style="cyan")
                topic_text.append(t + "\n")
            parts.append(
                Panel(
                    topic_text,
                    title="[bold]Key Topics[/bold]",
                    border_style="dim",
                    expand=True,
                )
            )
        if participants:
            part_text = Text()
            for p in participants:
                part_text.append("  \u2022 ", style="cyan")
                part_text.append(p + "\n")
            parts.append(
                Panel(
                    part_text,
                    title="[bold]Participants[/bold]",
                    border_style="dim",
                    expand=True,
                )
            )
        console.print(Columns(parts, equal=True, expand=True))

    # Action items table
    items = state.get("action_items", [])
    if items:
        console.print()
        table = Table(
            title=f"Action Items ({len(items)})",
            show_lines=True,
            border_style="blue",
            title_style="bold blue",
            expand=True,
        )
        table.add_column("#", style="dim", width=7, justify="center")
        table.add_column("Description", ratio=3)
        table.add_column("Assignee", ratio=2)
        table.add_column("Priority", width=10, justify="center")
        table.add_column("Deadline", width=12, justify="center")

        for item in items:
            priority = item.get("priority", "MEDIUM")
            color, icon = PRIORITY_STYLES.get(priority, ("white", " "))

            assignee_text = item.get("assignee", "Unassigned")
            email = item.get("assignee_email", "")
            if email:
                assignee_text += f"\n[dim]{email}[/dim]"

            table.add_row(
                item.get("id", ""),
                item.get("description", ""),
                assignee_text,
                f"[{color}]{icon} {priority}[/{color}]",
                item.get("deadline") or "[dim]N/A[/dim]",
            )

        console.print(table)
    else:
        console.print()
        console.print("[dim]No action items found in this transcript.[/dim]")

    # Emails sent
    emails = state.get("emails_sent", [])
    if emails:
        console.print()
        email_lines = Text()
        for email in emails:
            email_lines.append("  \u2714 ", style="green")
            email_lines.append(f"Notification sent to {email}\n")
        console.print(
            Panel(
                email_lines,
                title=f"[bold]Email Notifications ({len(emails)})[/bold]",
                border_style="green",
                expand=False,
            )
        )

    # LangSmith trace URL
    trace_url = state.get("trace_url", "")
    console.print()
    if trace_url:
        console.print(
            Panel(
                f"[link={trace_url}]{trace_url}[/link]",
                title="[bold magenta]LangSmith Trace[/bold magenta]",
                border_style="magenta",
                expand=False,
            )
        )
    else:
        console.print(
            "[dim]View traces in LangSmith: https://smith.langchain.com/projects[/dim]"
        )
    console.print()
