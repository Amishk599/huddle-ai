"""LangGraph state machine for meeting processing."""

import json
import re
from collections.abc import Generator
from datetime import date

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from src.config import settings
from src.models import (
    ActionItemList,
    AssigneeMatch,
    DeadlineResolution,
    MeetingState,
    MeetingSummary,
)
from src.prompts import (
    ASSIGN_OWNER_PROMPT,
    DETERMINE_DEADLINES_PROMPT,
    EXTRACT_ACTIONS_PROMPT,
    SUMMARIZE_PROMPT,
)
from src.rag import lookup_team_member
from src.tools import send_action_item_email


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=settings.DEFAULT_MODEL)


# --- Graph nodes ---


def intake(state: MeetingState) -> dict:
    """Validate the transcript input."""
    transcript = state.get("transcript", "")
    if not transcript or len(transcript.strip()) < 20:
        return {
            "errors": ["Transcript is empty or too short (minimum 20 characters)."],
            "processing_step": "intake",
        }
    return {"processing_step": "intake"}


def summarize(state: MeetingState) -> dict:
    """Generate a structured meeting summary."""
    try:
        llm = _get_llm()
        chain = SUMMARIZE_PROMPT | llm.with_structured_output(
            MeetingSummary, method="function_calling"
        )
        result = chain.invoke({"transcript": state["transcript"]})
        return {
            "summary": result.summary,
            "key_topics": result.key_topics,
            "participants": result.participants,
            "processing_step": "summarize",
        }
    except Exception as e:
        return {
            "summary": "Error generating summary.",
            "key_topics": [],
            "participants": [],
            "errors": [f"Summarize error: {e}"],
            "processing_step": "summarize",
        }


def extract_action_items(state: MeetingState) -> dict:
    """Extract action items from the transcript."""
    try:
        llm = _get_llm()
        chain = EXTRACT_ACTIONS_PROMPT | llm.with_structured_output(
            ActionItemList, method="function_calling"
        )
        result = chain.invoke(
            {
                "transcript": state["transcript"],
                "summary": state.get("summary", ""),
            }
        )

        items = []
        for i, item in enumerate(result.action_items):
            d = item.model_dump()
            d["id"] = f"ai-{i + 1:03d}"
            items.append(d)

        return {
            "action_items": items,
            "processing_step": "extract_action_items",
        }
    except Exception as e:
        return {
            "action_items": [],
            "errors": [f"Extract action items error: {e}"],
            "processing_step": "extract_action_items",
        }


def assign_owners(state: MeetingState) -> dict:
    """Assign owners to action items using RAG lookup."""
    llm = _get_llm()
    updated_items = []

    for item in state.get("action_items", []):
        try:
            # Build search query from the item
            query = item.get("description", "")
            context = item.get("context", "") or ""
            mentioned = item.get("assignee") or "not specified"

            # RAG lookup for candidates — include mentioned name for better matching
            rag_query = f"{mentioned} {query} {context}"
            docs = lookup_team_member(rag_query, k=3)
            candidates_text = "\n".join(
                f"- {doc.metadata['name']} ({doc.metadata['role']}), "
                f"Email: {doc.metadata['email']}\n  {doc.page_content}"
                for doc in docs
            )

            chain = ASSIGN_OWNER_PROMPT | llm.with_structured_output(
                AssigneeMatch, method="function_calling"
            )
            match = chain.invoke(
                {
                    "task_description": query,
                    "task_context": context,
                    "mentioned_assignee": mentioned,
                    "team_candidates": candidates_text,
                }
            )

            item = dict(item)
            item["assignee"] = match.name
            item["assignee_email"] = match.email
        except Exception as e:
            item = dict(item)
            item.setdefault("assignee", "Unassigned")

        updated_items.append(item)

    return {
        "action_items": updated_items,
        "processing_step": "assign_owners",
    }


def determine_deadlines(state: MeetingState) -> dict:
    """Resolve relative deadlines to absolute dates."""
    items = state.get("action_items", [])
    if not items:
        return {"processing_step": "determine_deadlines"}

    # Try to extract meeting date from transcript
    meeting_date = _extract_meeting_date(state.get("transcript", ""))

    try:
        llm = _get_llm()
        # Prepare items summary for the LLM
        items_for_llm = [
            {
                "index": i,
                "description": item.get("description", ""),
                "raw_deadline": item.get("deadline"),
            }
            for i, item in enumerate(items)
        ]

        chain = DETERMINE_DEADLINES_PROMPT | llm.with_structured_output(
            DeadlineResolution, method="function_calling"
        )
        result = chain.invoke(
            {
                "action_items_json": json.dumps(items_for_llm, indent=2),
                "transcript": state["transcript"],
                "meeting_date": meeting_date,
            }
        )

        # Apply resolved deadlines
        updated_items = [dict(item) for item in items]
        for entry in result.deadlines:
            idx = entry.index
            if 0 <= idx < len(updated_items):
                updated_items[idx]["deadline"] = entry.deadline

        return {
            "action_items": updated_items,
            "processing_step": "determine_deadlines",
        }
    except Exception as e:
        return {
            "errors": [f"Deadline resolution error: {e}"],
            "processing_step": "determine_deadlines",
        }


def send_emails(state: MeetingState) -> dict:
    """Send mock email notifications to assignees."""
    sent = []
    summary = state.get("summary", "")

    for item in state.get("action_items", []):
        email = item.get("assignee_email")
        name = item.get("assignee", "Team Member")
        if not email:
            continue

        result = send_action_item_email(
            recipient_email=email,
            recipient_name=name,
            action_item=item,
            meeting_summary=summary,
        )
        if result["status"] == "sent":
            sent.append(email)

    return {
        "emails_sent": sent,
        "processing_step": "send_emails",
    }


# --- Routing ---


def should_continue(state: MeetingState) -> str:
    """Route based on whether action items were found."""
    if state.get("action_items"):
        return "assign_owners"
    return END


# --- Graph construction ---


def build_graph():
    """Build and compile the meeting processing graph."""
    graph = StateGraph(MeetingState)

    # Add nodes
    graph.add_node("intake", intake)
    graph.add_node("summarize", summarize)
    graph.add_node("extract_action_items", extract_action_items)
    graph.add_node("assign_owners", assign_owners)
    graph.add_node("determine_deadlines", determine_deadlines)
    graph.add_node("send_emails", send_emails)

    # Add edges
    graph.add_edge(START, "intake")
    graph.add_edge("intake", "summarize")
    graph.add_edge("summarize", "extract_action_items")
    graph.add_conditional_edges(
        "extract_action_items",
        should_continue,
        {"assign_owners": "assign_owners", END: END},
    )
    graph.add_edge("assign_owners", "determine_deadlines")
    graph.add_edge("determine_deadlines", "send_emails")
    graph.add_edge("send_emails", END)

    return graph.compile()


PROCESSING_STEPS = [
    "intake",
    "summarize",
    "extract_action_items",
    "assign_owners",
    "determine_deadlines",
    "send_emails",
]

STEP_LABELS = {
    "intake": "Validating transcript",
    "summarize": "Generating summary",
    "extract_action_items": "Extracting action items",
    "assign_owners": "Assigning owners (RAG)",
    "determine_deadlines": "Resolving deadlines",
    "send_emails": "Sending notifications",
}


def _initial_state(transcript: str, source: str) -> dict:
    return {
        "transcript": transcript,
        "transcript_source": source,
        "summary": "",
        "key_topics": [],
        "participants": [],
        "action_items": [],
        "emails_sent": [],
        "errors": [],
        "processing_step": "",
    }


def process_transcript(transcript: str, source: str = "sample") -> dict:
    """Process a meeting transcript through the full pipeline.

    Args:
        transcript: The meeting transcript text.
        source: Source type ("sample" or "pasted").

    Returns:
        Final state dict with summary, action_items, emails_sent, etc.
    """
    graph = build_graph()
    return graph.invoke(_initial_state(transcript, source))


def process_transcript_stream(
    transcript: str, source: str = "sample"
) -> Generator[tuple[str, dict], None, None]:
    """Process a transcript, yielding (node_name, state_update) after each step.

    Use this for real-time progress display.
    """
    graph = build_graph()
    for event in graph.stream(
        _initial_state(transcript, source), stream_mode="updates"
    ):
        for node_name, state_update in event.items():
            yield node_name, state_update


# --- Helpers ---


def _extract_meeting_date(transcript: str) -> str:
    """Try to extract meeting date from transcript, fallback to today."""
    # Look for common date patterns
    patterns = [
        r"Date:\s*(\w+ \d{1,2},?\s*\d{4})",
        r"Date:\s*(\d{4}-\d{2}-\d{2})",
        r"(\w+ \d{1,2},?\s*\d{4})",
    ]
    for pattern in patterns:
        match = re.search(pattern, transcript)
        if match:
            return match.group(1)

    return date.today().isoformat()
