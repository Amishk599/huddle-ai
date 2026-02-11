"""Prompt templates for LLM calls."""

from langchain_core.prompts import ChatPromptTemplate

SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert meeting analyst. Produce a concise, structured summary of the meeting transcript provided.",
        ),
        (
            "human",
            """Analyze the following meeting transcript and produce a structured summary.

TRANSCRIPT:
{transcript}

Provide:
1. A concise summary (2-4 sentences) capturing the key decisions and outcomes
2. The main topics discussed
3. The names of all participants mentioned""",
        ),
    ]
)

EXTRACT_ACTIONS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at identifying action items from meeting transcripts. "
            "Extract every task, assignment, or commitment made during the meeting. "
            "If no action items exist, return an empty list.",
        ),
        (
            "human",
            """Extract all action items from this meeting transcript.

TRANSCRIPT:
{transcript}

MEETING SUMMARY:
{summary}

For each action item, identify:
- description: What needs to be done
- assignee: Who was assigned (use their name as mentioned in the transcript, or null if unclear)
- priority: HIGH (urgent, blocking, ASAP), MEDIUM (normal), or LOW (nice-to-have, no rush)
- deadline: The deadline as mentioned in the transcript (e.g. "by Friday", "next week", "end of Q2"), or null if none mentioned
- context: Brief context from the discussion about why this task matters

If no action items are found, return an empty list.""",
        ),
    ]
)

ASSIGN_OWNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are matching action items to team members. Given a task description and a list of candidate team members, "
            "select the single best match based on their role, expertise, and current projects.",
        ),
        (
            "human",
            """Match this action item to the best team member.

TASK: {task_description}
CONTEXT: {task_context}
MENTIONED ASSIGNEE: {mentioned_assignee}

CANDIDATE TEAM MEMBERS:
{team_candidates}

Select the team member who best matches. If a specific person was mentioned by name in the transcript, prefer them.
Return their full name and email.""",
        ),
    ]
)

DETERMINE_DEADLINES_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are resolving relative date references into absolute ISO dates (YYYY-MM-DD). "
            "Use the meeting date as the reference point for relative dates.",
        ),
        (
            "human",
            """Resolve the deadlines for these action items into absolute dates.

MEETING DATE: {meeting_date}

ACTION ITEMS:
{action_items_json}

Rules:
- "by Friday" or "end of this week" → the next upcoming Friday from the meeting date
- "next week" → the following Monday from the meeting date
- "next Monday", "next Wednesday" etc. → the next occurrence of that day
- "ASAP" or "immediately" → 2 business days from the meeting date
- "end of month" → last day of the meeting's month
- "end of Q1/Q2/Q3/Q4" → last day of that quarter
- If no deadline was mentioned → 7 days from the meeting date
- Specific dates like "February 12th" → use that date directly

Return a list where each entry has the action item index (0-based) and the resolved ISO date.""",
        ),
    ]
)
