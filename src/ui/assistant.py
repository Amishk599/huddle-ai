"""Chat interface for the RAG-powered assistant."""

from langchain_core.messages import AIMessage, HumanMessage
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

console = Console()

HELP_TEXT = """
[bold]Assistant Commands:[/bold]

  [cyan]/help[/cyan]   Show this help message
  [cyan]/clear[/cyan]  Clear conversation history
  [cyan]/exit[/cyan]   Return to main menu
  [cyan]q[/cyan]       Return to main menu
"""

SOURCE_STYLES = {
    "Team Directory": "magenta",
    "Meeting History": "cyan",
    "General Knowledge": "green",
}


def _show_assistant_banner() -> None:
    console.print()
    console.print(
        Panel(
            Text("RAG-Powered Assistant", style="bold", justify="center")
            + Text(
                "\n  Ask about your team, past meetings, or anything else.",
                style="dim",
                justify="center",
            ),
            border_style="magenta",
            expand=False,
            padding=(0, 4),
        )
    )
    console.print("[dim]  Type /help for commands, /exit to return to menu.[/dim]\n")


def _display_answer(answer: str, source: str) -> None:
    """Display the assistant's response with source label."""
    color = SOURCE_STYLES.get(source, "white")
    source_tag = f"[{color}]{source}[/{color}]"

    console.print()
    console.print(
        Panel(
            Markdown(answer),
            title=f"[bold]Assistant[/bold]  [{color}]({source})[/{color}]",
            border_style=color,
            padding=(1, 2),
        )
    )


def run_assistant() -> None:
    """Run the interactive assistant chat loop."""
    from src.assistant.chain import ask

    _show_assistant_banner()

    history: list[HumanMessage | AIMessage] = []

    while True:
        try:
            user_input = pt_prompt(HTML("<b><ansigreen>You</ansigreen> > </b>")).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Returning to main menu.[/dim]\n")
            break

        if not user_input:
            continue

        # Handle commands
        cmd = user_input.lower()
        if cmd in ("/exit", "/quit", "q"):
            console.print("[dim]Returning to main menu.[/dim]\n")
            break
        if cmd == "/help":
            console.print(HELP_TEXT)
            continue
        if cmd == "/clear":
            history.clear()
            console.print("[dim]Conversation history cleared.[/dim]\n")
            continue

        # Ask the assistant
        try:
            answer, source = ask(user_input, history)

            # Update conversation history
            history.append(HumanMessage(content=user_input))
            history.append(AIMessage(content=answer))

            _display_answer(answer, source)
            console.print()
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")
