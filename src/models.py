"""Pydantic models and LangGraph state definitions."""

import operator
from typing import Annotated, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# --- Pydantic models for structured LLM output ---


class ActionItem(BaseModel):
    """A single action item extracted from a meeting transcript."""

    description: str = Field(description="Clear description of the task")
    assignee: Optional[str] = Field(
        default=None, description="Name of the person assigned (from transcript)"
    )
    assignee_email: Optional[str] = Field(
        default=None, description="Email of the assignee"
    )
    priority: str = Field(
        default="MEDIUM",
        description="Priority level: HIGH, MEDIUM, or LOW",
    )
    deadline: Optional[str] = Field(
        default=None,
        description="Deadline mentioned in the transcript (raw text, e.g. 'by Friday')",
    )
    context: Optional[str] = Field(
        default=None, description="Additional context from the meeting"
    )


class ActionItemList(BaseModel):
    """Wrapper for extracting a list of action items."""

    action_items: list[ActionItem] = Field(
        default_factory=list,
        description="List of action items extracted from the transcript. Empty if none found.",
    )


class MeetingSummary(BaseModel):
    """Structured meeting summary."""

    summary: str = Field(description="Concise summary of the meeting discussion")
    key_topics: list[str] = Field(description="Main topics discussed")
    participants: list[str] = Field(
        description="Names of meeting participants mentioned"
    )


class AssigneeMatch(BaseModel):
    """Result of matching an action item to a team member."""

    name: str = Field(description="Full name of the best matching team member")
    email: str = Field(description="Email of the matched team member")
    reasoning: str = Field(description="Brief reason for this match")


class DeadlineEntry(BaseModel):
    """A single resolved deadline."""

    index: int = Field(description="0-based index of the action item")
    deadline: str = Field(description="Resolved deadline in ISO format YYYY-MM-DD")


class DeadlineResolution(BaseModel):
    """Resolved deadlines for action items."""

    deadlines: list[DeadlineEntry] = Field(
        description="List of resolved deadlines for each action item"
    )


# --- LangGraph state ---


class MeetingState(TypedDict):
    """State object passed between LangGraph nodes."""

    # Input
    transcript: str
    transcript_source: str  # "sample" or "pasted"

    # Processing results
    summary: str
    key_topics: list[str]
    participants: list[str]
    action_items: list[dict]  # serialized ActionItem dicts

    # Output
    emails_sent: list[str]  # recipient emails
    errors: Annotated[list[str], operator.add]  # accumulated across nodes

    # Metadata
    processing_step: str
