"""Evaluation suite for Huddle AI meeting intelligence agent.

Runs evaluations for both meeting processing (action item extraction,
assignee matching, summary quality) and assistant query routing,
then reports results via Rich tables and LangSmith experiments.

Usage:
    python evals/run_evals.py              # Run all evaluations
    python evals/run_evals.py --meeting    # Meeting processing only
    python evals/run_evals.py --assistant  # Assistant routing only
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Ensure .env is loaded before any LangChain / OpenAI imports
load_dotenv()

# Add project root to path so we can import src modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import process_transcript  # noqa: E402
from src.assistant.chain import classify_query  # noqa: E402

console = Console()

# ---------------------------------------------------------------------------
# Load evaluation dataset
# ---------------------------------------------------------------------------

EVAL_DATASET_PATH = Path(__file__).parent / "eval_dataset.json"


def _load_dataset() -> dict:
    with open(EVAL_DATASET_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Target functions (what we are evaluating)
# ---------------------------------------------------------------------------


def meeting_target(inputs: dict) -> dict:
    """Run the full meeting processing pipeline on a transcript.

    This is the *target function* that LangSmith's ``evaluate()`` calls
    for each example in the dataset.
    """
    transcript = inputs["transcript"]
    result = process_transcript(transcript, source="eval")
    return {
        "action_items": result.get("action_items", []),
        "action_item_count": len(result.get("action_items", [])),
        "assignees": [
            item.get("assignee", "") for item in result.get("action_items", [])
        ],
        "priorities": [
            item.get("priority", "MEDIUM") for item in result.get("action_items", [])
        ],
        "summary": result.get("summary", ""),
        "key_topics": result.get("key_topics", []),
    }


def assistant_target(inputs: dict) -> dict:
    """Classify a user query to test assistant routing."""
    query = inputs["query"]
    category = classify_query(query)
    return {"category": category}


# ---------------------------------------------------------------------------
# Custom evaluators — meeting processing
# ---------------------------------------------------------------------------


def action_item_count_evaluator(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    """Score whether the correct number of action items were extracted.

    Returns a binary score (1.0 = exact match, 0.0 = mismatch) plus
    a precision-style partial credit score.
    """
    expected = reference_outputs.get("expected_action_items", 0)
    actual = outputs.get("action_item_count", 0)

    exact_match = 1.0 if actual == expected else 0.0

    # Partial credit: how close are we?
    if expected == 0:
        partial = 1.0 if actual == 0 else 0.0
    else:
        partial = max(0.0, 1.0 - abs(actual - expected) / max(expected, 1))

    return {
        "results": [
            {"key": "action_item_count_exact", "score": exact_match},
            {"key": "action_item_count_partial", "score": round(partial, 2)},
        ]
    }


def assignee_accuracy_evaluator(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    """Score whether action items were assigned to the correct people.

    Checks both single-assignee and multi-assignee test cases.
    Returns precision and recall scores.
    """
    actual_assignees = [a.lower().strip() for a in outputs.get("assignees", []) if a]

    # Handle both single and multi-assignee expected formats
    expected_raw = reference_outputs.get("expected_assignees", [])
    if not expected_raw:
        single = reference_outputs.get("expected_assignee", "")
        expected_raw = [single] if single else []
    expected_assignees = [a.lower().strip() for a in expected_raw]

    if not expected_assignees:
        # No expected assignees — score 1.0 if none assigned
        score = 1.0 if not actual_assignees else 0.5
        return {"results": [{"key": "assignee_accuracy", "score": score}]}

    # Count matches (fuzzy: expected name appears as substring of actual)
    matches = 0
    for expected in expected_assignees:
        for actual in actual_assignees:
            if expected in actual or actual in expected:
                matches += 1
                break

    precision = matches / len(actual_assignees) if actual_assignees else 0.0
    recall = matches / len(expected_assignees) if expected_assignees else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "results": [
            {"key": "assignee_precision", "score": round(precision, 2)},
            {"key": "assignee_recall", "score": round(recall, 2)},
            {"key": "assignee_f1", "score": round(f1, 2)},
        ]
    }


def priority_detection_evaluator(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    """Score whether high-priority items were correctly flagged."""
    expected_priority = reference_outputs.get("expected_priority", "").upper()
    if not expected_priority:
        return {"results": [{"key": "priority_detection", "score": 1.0}]}

    actual_priorities = [p.upper() for p in outputs.get("priorities", [])]
    if not actual_priorities:
        return {"results": [{"key": "priority_detection", "score": 0.0}]}

    # Check if at least one item matches expected priority
    match = any(p == expected_priority for p in actual_priorities)
    return {"results": [{"key": "priority_detection", "score": 1.0 if match else 0.0}]}


def summary_quality_evaluator(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    """LLM-as-judge scorer for summary quality.

    Uses gpt-4o-mini to rate summary quality on a 0-1 scale.
    """
    from langchain_openai import ChatOpenAI

    summary = outputs.get("summary", "")
    transcript = inputs.get("transcript", "")

    if not summary or not transcript:
        return {"results": [{"key": "summary_quality", "score": 0.0}]}

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = (
        "You are evaluating the quality of a meeting summary.\n\n"
        "TRANSCRIPT:\n{transcript}\n\n"
        "SUMMARY:\n{summary}\n\n"
        "Rate the summary on a scale from 0.0 to 1.0 based on:\n"
        "- Accuracy: Does it correctly capture what was discussed?\n"
        "- Completeness: Does it cover the key points?\n"
        "- Conciseness: Is it appropriately brief without losing important details?\n\n"
        "Respond with ONLY a single decimal number between 0.0 and 1.0."
    )

    try:
        response = llm.invoke(
            prompt.format(transcript=transcript[:2000], summary=summary[:1000])
        )
        score = float(response.content.strip())
        score = max(0.0, min(1.0, score))
    except (ValueError, Exception):
        score = 0.5  # Default if parsing fails

    return {"results": [{"key": "summary_quality", "score": round(score, 2)}]}


# ---------------------------------------------------------------------------
# Custom evaluator — assistant routing
# ---------------------------------------------------------------------------


def routing_accuracy_evaluator(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    """Score whether the assistant routed the query to the correct source."""
    expected = reference_outputs.get("expected_source", "").lower()
    actual = outputs.get("category", "").lower()

    score = 1.0 if actual == expected else 0.0
    return {"results": [{"key": "routing_accuracy", "score": score}]}


# ---------------------------------------------------------------------------
# Run evaluations using LangSmith SDK
# ---------------------------------------------------------------------------


def _create_or_get_dataset(
    client: Client,
    name: str,
    examples: list[dict],
    input_key_map: dict,
    output_key_map: dict,
) -> str:
    """Create a LangSmith dataset (or reuse existing) and populate with examples.

    Returns the dataset name.
    """
    # Delete existing dataset to ensure fresh examples
    try:
        existing = client.read_dataset(dataset_name=name)
        client.delete_dataset(dataset_id=existing.id)
    except Exception:
        pass

    dataset = client.create_dataset(
        dataset_name=name,
        description=f"Huddle AI evaluation dataset: {name}",
    )

    # Build examples
    ls_examples = []
    for ex in examples:
        inputs = {v: ex.get(k, "") for k, v in input_key_map.items()}
        outputs = {k: ex.get(k, "") for k in output_key_map}
        # Preserve all expected_* fields as reference outputs
        for key in ex:
            if key.startswith("expected_") and key not in outputs:
                outputs[key] = ex[key]
        ls_examples.append({"inputs": inputs, "outputs": outputs})

    client.create_examples(
        dataset_id=dataset.id,
        examples=ls_examples,
    )

    return name


def run_meeting_evals(client: Client, dataset_json: dict) -> dict:
    """Run meeting processing evaluations via LangSmith."""
    console.print("\n[bold blue]Running Meeting Processing Evaluations...[/bold blue]\n")

    examples = dataset_json["meeting_processing"]
    dataset_name = _create_or_get_dataset(
        client,
        name="huddle-ai-meeting-evals",
        examples=examples,
        input_key_map={"input": "transcript"},
        output_key_map={
            "expected_action_items": "expected_action_items",
        },
    )

    results = client.evaluate(
        meeting_target,
        data=dataset_name,
        evaluators=[
            action_item_count_evaluator,
            assignee_accuracy_evaluator,
            priority_detection_evaluator,
            summary_quality_evaluator,
        ],
        experiment_prefix="huddle-meeting-eval",
        metadata={
            "app": "huddle-ai",
            "eval_type": "meeting_processing",
        },
        max_concurrency=2,
    )

    return _collect_results(results, "Meeting Processing")


def run_assistant_evals(client: Client, dataset_json: dict) -> dict:
    """Run assistant routing evaluations via LangSmith."""
    console.print("\n[bold cyan]Running Assistant Routing Evaluations...[/bold cyan]\n")

    examples = dataset_json["assistant_queries"]
    dataset_name = _create_or_get_dataset(
        client,
        name="huddle-ai-assistant-evals",
        examples=examples,
        input_key_map={"query": "query"},
        output_key_map={
            "expected_source": "expected_source",
        },
    )

    results = client.evaluate(
        assistant_target,
        data=dataset_name,
        evaluators=[routing_accuracy_evaluator],
        experiment_prefix="huddle-assistant-eval",
        metadata={
            "app": "huddle-ai",
            "eval_type": "assistant_routing",
        },
        max_concurrency=2,
    )

    return _collect_results(results, "Assistant Routing")


def _collect_results(experiment_results, eval_name: str) -> dict:
    """Extract scores from experiment results and return summary dict."""
    scores: dict[str, list[float]] = {}

    for result in experiment_results:
        eval_results = result.get("evaluation_results", {})
        feedback_list = eval_results.get("results", [])
        for feedback in feedback_list:
            key = feedback.key if hasattr(feedback, "key") else str(feedback.get("key", "unknown"))
            score = feedback.score if hasattr(feedback, "score") else feedback.get("score", 0)
            if score is not None:
                scores.setdefault(key, []).append(float(score))

    # Compute averages
    averages = {k: round(sum(v) / len(v), 3) if v else 0.0 for k, v in scores.items()}

    return {
        "eval_name": eval_name,
        "scores": averages,
        "num_examples": sum(1 for _ in experiment_results) if not scores else max(len(v) for v in scores.values()),
    }


# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------


def _display_results(meeting_results: dict | None, assistant_results: dict | None) -> None:
    """Display evaluation results in a formatted Rich table."""
    console.print()
    console.print(
        Panel(
            "[bold]Huddle AI — Evaluation Results[/bold]",
            border_style="blue",
            expand=False,
        )
    )

    all_results = []
    if meeting_results:
        all_results.append(meeting_results)
    if assistant_results:
        all_results.append(assistant_results)

    for result in all_results:
        table = Table(
            title=f"{result['eval_name']}",
            show_lines=True,
            border_style="cyan",
            title_style="bold cyan",
        )
        table.add_column("Metric", style="bold", ratio=2)
        table.add_column("Score", justify="center", ratio=1)
        table.add_column("Status", justify="center", ratio=1)

        for metric, score in result["scores"].items():
            # Color coding
            if score >= 0.8:
                color = "green"
                status = "✓ PASS"
            elif score >= 0.5:
                color = "yellow"
                status = "~ WARN"
            else:
                color = "red"
                status = "✗ FAIL"

            score_text = Text(f"{score:.3f}", style=color)
            status_text = Text(status, style=color)
            table.add_row(metric, score_text, status_text)

        console.print(table)
        console.print()

    # Summary
    total_metrics = sum(len(r["scores"]) for r in all_results)
    passing = sum(
        1
        for r in all_results
        for s in r["scores"].values()
        if s >= 0.8
    )
    console.print(
        f"[bold]Overall: {passing}/{total_metrics} metrics passing (≥ 0.8)[/bold]"
    )
    console.print()
    console.print(
        "[dim]View detailed results in LangSmith: "
        "https://smith.langchain.com[/dim]"
    )
    console.print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    """Run the evaluation suite."""
    run_meeting = "--assistant" not in sys.argv
    run_assistant = "--meeting" not in sys.argv

    console.print()
    console.print(
        Panel(
            "[bold magenta]Huddle AI — Evaluation Suite[/bold magenta]\n"
            "[dim]Running evaluators with LangSmith tracing enabled[/dim]",
            border_style="magenta",
            expand=False,
        )
    )

    # Verify LangSmith connection
    try:
        client = Client()
        console.print("[green]✓[/green] Connected to LangSmith")
    except Exception as e:
        console.print(f"[red]✗ Failed to connect to LangSmith: {e}[/red]")
        console.print("[dim]Make sure LANGCHAIN_API_KEY is set in your .env file[/dim]")
        sys.exit(1)

    dataset_json = _load_dataset()

    meeting_results = None
    assistant_results = None

    start_time = time.time()

    if run_meeting:
        meeting_results = run_meeting_evals(client, dataset_json)

    if run_assistant:
        assistant_results = run_assistant_evals(client, dataset_json)

    elapsed = time.time() - start_time

    _display_results(meeting_results, assistant_results)
    console.print(f"[dim]Completed in {elapsed:.1f}s[/dim]\n")


if __name__ == "__main__":
    main()
