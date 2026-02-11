"""RAG-powered assistant chain with question routing."""

from collections.abc import Generator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import settings
from src.rag import search_meetings, search_team

# --- Question classification ---


class QueryClassification(BaseModel):
    """Classification of a user query for routing."""

    category: str = Field(
        description="One of: 'team', 'meeting', or 'general'"
    )
    reasoning: str = Field(description="Brief reason for classification")


CLASSIFY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Classify the user's question into one of three categories:\n"
            "- 'team': Questions about team members, roles, expertise, projects, "
            "or who someone is/does (e.g. 'Who is the PM?', 'Who knows Python?')\n"
            "- 'meeting': Questions about past meetings, discussions, decisions, "
            "or what was talked about (e.g. 'What was discussed with Todd?', "
            "'What are the action items from sprint planning?')\n"
            "- 'general': Everything else — general knowledge, definitions, "
            "opinions, or questions unrelated to the team or meetings\n\n"
            "Respond with the category and a brief reasoning.",
        ),
        MessagesPlaceholder("history", optional=True),
        ("human", "{question}"),
    ]
)


# --- Response generation ---


TEAM_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for a team. Answer the user's question "
            "using the team directory information provided below. Be concise and "
            "direct. If the information doesn't fully answer the question, say so.\n\n"
            "TEAM DIRECTORY CONTEXT:\n{context}",
        ),
        MessagesPlaceholder("history", optional=True),
        ("human", "{question}"),
    ]
)


MEETING_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions about past meetings. "
            "Use the meeting transcript excerpts provided below to answer the user's "
            "question. Be specific and reference details from the transcripts. If the "
            "information doesn't fully answer the question, say so.\n\n"
            "MEETING CONTEXT:\n{context}",
        ),
        MessagesPlaceholder("history", optional=True),
        ("human", "{question}"),
    ]
)


GENERAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the user's question clearly "
            "and concisely.",
        ),
        MessagesPlaceholder("history", optional=True),
        ("human", "{question}"),
    ]
)


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=settings.DEFAULT_MODEL)


def classify_query(
    question: str,
    history: list[HumanMessage | AIMessage] | None = None,
) -> str:
    """Classify a user question into 'team', 'meeting', or 'general'.

    Returns:
        The category string.
    """
    llm = _get_llm()
    chain = CLASSIFY_PROMPT | llm.with_structured_output(
        QueryClassification, method="function_calling"
    )
    result = chain.invoke({
        "question": question,
        "history": history or [],
    })
    return result.category


def _build_team_chain_inputs(
    question: str,
    history: list[HumanMessage | AIMessage] | None = None,
) -> tuple[dict, str]:
    """Prepare inputs for a team RAG query."""
    docs = search_team(question, k=3)
    context = "\n\n".join(
        f"[{doc.metadata.get('name', 'Unknown')} - {doc.metadata.get('role', '')}]\n"
        f"{doc.page_content}"
        for doc in docs
    )
    return {
        "question": question,
        "context": context,
        "history": history or [],
    }, "Team Directory"


def _build_meeting_chain_inputs(
    question: str,
    history: list[HumanMessage | AIMessage] | None = None,
) -> tuple[dict, str]:
    """Prepare inputs for a meeting RAG query."""
    docs = search_meetings(question, k=3)
    context = "\n\n".join(
        f"[{doc.metadata.get('meeting', 'Unknown Meeting')} - "
        f"{doc.metadata.get('date', '')}]\n{doc.page_content}"
        for doc in docs
    )
    return {
        "question": question,
        "context": context,
        "history": history or [],
    }, "Meeting History"


def _build_general_chain_inputs(
    question: str,
    history: list[HumanMessage | AIMessage] | None = None,
) -> tuple[dict, str]:
    """Prepare inputs for a general knowledge query."""
    return {
        "question": question,
        "history": history or [],
    }, "General Knowledge"


_CATEGORY_CONFIG: dict[str, tuple] = {
    "team": (TEAM_RAG_PROMPT, _build_team_chain_inputs),
    "meeting": (MEETING_RAG_PROMPT, _build_meeting_chain_inputs),
    "general": (GENERAL_PROMPT, _build_general_chain_inputs),
}


def _resolve_category(
    question: str,
    history: list[HumanMessage | AIMessage] | None,
) -> tuple[ChatPromptTemplate, dict, str]:
    """Classify the question and return (prompt, chain_inputs, source_label)."""
    category = classify_query(question, history)
    prompt, builder = _CATEGORY_CONFIG.get(
        category, (GENERAL_PROMPT, _build_general_chain_inputs)
    )
    inputs, source = builder(question, history)
    return prompt, inputs, source


def ask(
    question: str,
    history: list[HumanMessage | AIMessage] | None = None,
) -> tuple[str, str]:
    """Classify a question and route to the appropriate handler.

    Args:
        question: The user's question.
        history: Conversation history (list of HumanMessage/AIMessage).

    Returns:
        Tuple of (answer_text, source_label).
    """
    prompt, inputs, source = _resolve_category(question, history)
    chain = prompt | _get_llm()
    result = chain.invoke(inputs)
    return result.content, source


def ask_stream(
    question: str,
    history: list[HumanMessage | AIMessage] | None = None,
) -> tuple[str, Generator[str, None, None]]:
    """Classify a question and stream the response token-by-token.

    Classification is blocking (fast structured output call).
    The answer is streamed.

    Args:
        question: The user's question.
        history: Conversation history (list of HumanMessage/AIMessage).

    Returns:
        Tuple of (source_label, token_generator).
        The generator yields content strings for each chunk.
    """
    prompt, inputs, source = _resolve_category(question, history)
    chain = prompt | _get_llm()

    def _token_generator() -> Generator[str, None, None]:
        for chunk in chain.stream(inputs):
            if chunk.content:
                yield chunk.content

    return source, _token_generator()
