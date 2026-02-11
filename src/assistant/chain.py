"""RAG-powered assistant chain with question routing."""

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


def answer_team_query(
    question: str,
    history: list[HumanMessage | AIMessage] | None = None,
) -> tuple[str, str]:
    """Answer a team-related question using RAG.

    Returns:
        Tuple of (answer, source_label).
    """
    docs = search_team(question, k=3)
    context = "\n\n".join(
        f"[{doc.metadata.get('name', 'Unknown')} - {doc.metadata.get('role', '')}]\n"
        f"{doc.page_content}"
        for doc in docs
    )

    llm = _get_llm()
    chain = TEAM_RAG_PROMPT | llm
    result = chain.invoke({
        "question": question,
        "context": context,
        "history": history or [],
    })
    return result.content, "Team Directory"


def answer_meeting_query(
    question: str,
    history: list[HumanMessage | AIMessage] | None = None,
) -> tuple[str, str]:
    """Answer a meeting-related question using RAG.

    Returns:
        Tuple of (answer, source_label).
    """
    docs = search_meetings(question, k=3)
    context = "\n\n".join(
        f"[{doc.metadata.get('meeting', 'Unknown Meeting')} - "
        f"{doc.metadata.get('date', '')}]\n{doc.page_content}"
        for doc in docs
    )

    llm = _get_llm()
    chain = MEETING_RAG_PROMPT | llm
    result = chain.invoke({
        "question": question,
        "context": context,
        "history": history or [],
    })
    return result.content, "Meeting History"


def answer_general_query(
    question: str,
    history: list[HumanMessage | AIMessage] | None = None,
) -> tuple[str, str]:
    """Answer a general question directly with the LLM.

    Returns:
        Tuple of (answer, source_label).
    """
    llm = _get_llm()
    chain = GENERAL_PROMPT | llm
    result = chain.invoke({
        "question": question,
        "history": history or [],
    })
    return result.content, "General Knowledge"


# Route table
_ROUTE_HANDLERS = {
    "team": answer_team_query,
    "meeting": answer_meeting_query,
    "general": answer_general_query,
}


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
    category = classify_query(question, history)
    handler = _ROUTE_HANDLERS.get(category, answer_general_query)
    return handler(question, history)
