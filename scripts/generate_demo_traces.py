"""Generate demo traces in LangSmith for presentation.

Processes all sample transcripts through the meeting pipeline and runs
a few assistant queries, generating tagged LangSmith traces that can be
explored in the LangSmith UI.

By default, all traces are made **publicly shareable** so that anyone
with the URL can view them — perfect for showcasing in interviews or
portfolio reviews.

Usage:
    python scripts/generate_demo_traces.py            # public traces (default)
    python scripts/generate_demo_traces.py --private   # workspace-only traces
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add project root so we can import src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

from src.agent import process_transcript  # noqa: E402
from src.assistant.chain import ask  # noqa: E402
from src.config import settings  # noqa: E402

console = Console()

SAMPLE_ASSISTANT_QUERIES = [
    "Who on the team has Python expertise?",
    "What was discussed in the meeting with Todd?",
    "Who is the QA lead?",
    "What design work is being done on the component library?",
    "What is LangGraph?",
]


def main() -> None:
    """Process all sample transcripts and assistant queries to generate traces."""
    share = "--private" not in sys.argv
    visibility = "public" if share else "private"

    console.print()
    console.print(
        Panel(
            "[bold magenta]Huddle AI — Demo Trace Generator[/bold magenta]\n"
            f"[dim]Project: {settings.LANGCHAIN_PROJECT}  |  Traces: {visibility}[/dim]",
            border_style="magenta",
            expand=False,
        )
    )

    if share:
        console.print(
            "\n[yellow]Traces will be shared publicly "
            "(pass --private to keep them workspace-only)[/yellow]"
        )

    # --- Process sample transcripts ---
    console.print("\n[bold blue]Processing Sample Transcripts[/bold blue]\n")

    transcripts_dir = settings.SAMPLE_TRANSCRIPTS_DIR
    files = sorted(transcripts_dir.glob("*.txt"))

    if not files:
        console.print(f"[red]No transcript files found in {transcripts_dir}[/red]")
        return

    trace_table = Table(
        title="Meeting Processing Traces",
        show_lines=True,
        border_style="blue",
    )
    trace_table.add_column("Transcript", ratio=2)
    trace_table.add_column("Actions", justify="center", width=10)
    trace_table.add_column("Emails", justify="center", width=10)
    trace_table.add_column(f"Trace URL ({visibility})", ratio=3)

    for filepath in files:
        name = filepath.stem.replace("-", " ").title()
        console.print(f"  Processing: [cyan]{name}[/cyan] ... ", end="")

        try:
            transcript = filepath.read_text()
            result = process_transcript(transcript, source="demo", share=share)

            num_actions = len(result.get("action_items", []))
            num_emails = len(result.get("emails_sent", []))
            trace_url = result.get("trace_url", "")

            trace_table.add_row(
                name,
                str(num_actions),
                str(num_emails),
                trace_url[:80] + "..." if len(trace_url) > 80 else trace_url,
            )
            console.print("[green]✓[/green]")

        except Exception as e:
            trace_table.add_row(name, "ERR", "ERR", str(e)[:60])
            console.print(f"[red]✗ {e}[/red]")

    console.print()
    console.print(trace_table)

    # --- Process assistant queries ---
    console.print("\n[bold cyan]Processing Assistant Queries[/bold cyan]\n")

    assistant_table = Table(
        title="Assistant Query Traces",
        show_lines=True,
        border_style="cyan",
    )
    assistant_table.add_column("Query", ratio=2)
    assistant_table.add_column("Source", justify="center", width=18)
    assistant_table.add_column("Answer (preview)", ratio=3)

    for query in SAMPLE_ASSISTANT_QUERIES:
        console.print(f"  Asking: [cyan]{query}[/cyan] ... ", end="")
        try:
            answer, source = ask(query)
            preview = answer[:80] + "..." if len(answer) > 80 else answer
            assistant_table.add_row(query, source, preview)
            console.print("[green]✓[/green]")
        except Exception as e:
            assistant_table.add_row(query, "ERR", str(e)[:60])
            console.print(f"[red]✗ {e}[/red]")

    console.print()
    console.print(assistant_table)

    # --- Summary ---
    console.print()
    share_note = (
        "Traces are [bold]publicly shared[/bold] — anyone with the URL can view them."
        if share
        else "Traces are [bold]private[/bold] to your LangSmith workspace."
    )
    console.print(
        Panel(
            f"[bold green]Demo traces generated successfully![/bold green]\n\n"
            f"{share_note}\n"
            f"View in LangSmith: https://smith.langchain.com\n"
            f"Project: [bold]{settings.LANGCHAIN_PROJECT}[/bold]",
            border_style="green",
            expand=False,
        )
    )
    console.print()


if __name__ == "__main__":
    main()
