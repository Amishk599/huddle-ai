"""RAG setup and retrieval functions for team directory and meeting history."""

import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.config import settings

_team_store: FAISS | None = None
_meetings_store: FAISS | None = None


def _get_embeddings() -> OpenAIEmbeddings:
    """Get the embedding model."""
    return OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)


def _store_path() -> str:
    """Get the path for persisting the FAISS index."""
    return str(settings.VECTOR_STORES_DIR / "team_db")


def get_team_vector_store() -> FAISS:
    """Get or create the team directory vector store."""
    global _team_store
    if _team_store is not None:
        return _team_store

    store_dir = _store_path()
    if Path(store_dir).exists() and (Path(store_dir) / "index.faiss").exists():
        _team_store = FAISS.load_local(
            store_dir, _get_embeddings(), allow_dangerous_deserialization=True
        )
    else:
        # If no persisted store, initialize it
        initialize_team_store()
        _team_store = FAISS.load_local(
            store_dir, _get_embeddings(), allow_dangerous_deserialization=True
        )

    return _team_store


def initialize_team_store() -> int:
    """Load team directory JSON and populate the vector store.

    Returns:
        Number of documents indexed.
    """
    global _team_store

    team_file = settings.DATA_DIR / "team_directory.json"
    with open(team_file) as f:
        team_members = json.load(f)

    documents = []
    for member in team_members:
        text = (
            f"Name: {member['name']}\n"
            f"Role: {member['role']}\n"
            f"Expertise: {', '.join(member['expertise'])}\n"
            f"Current Projects: {', '.join(member['current_projects'])}\n"
            f"Reports To: {member['reports_to']}"
        )
        doc = Document(
            page_content=text,
            metadata={
                "id": member["id"],
                "name": member["name"],
                "email": member["email"],
                "role": member["role"],
            },
        )
        documents.append(doc)

    store = FAISS.from_documents(documents, _get_embeddings())

    # Persist to disk
    store_dir = _store_path()
    Path(store_dir).mkdir(parents=True, exist_ok=True)
    store.save_local(store_dir)

    _team_store = store
    return len(documents)


def lookup_team_member(query: str, k: int = 3) -> list[Document]:
    """Search for team members matching a task description.

    Args:
        query: Task description or context to match against.
        k: Number of results to return.

    Returns:
        List of matching team member documents.
    """
    store = get_team_vector_store()
    return store.similarity_search(query, k=k)


# --- Meeting history vector store ---


def _meetings_store_path() -> str:
    """Get the path for persisting the meetings FAISS index."""
    return str(settings.VECTOR_STORES_DIR / "meetings_db")


def get_meetings_vector_store() -> FAISS:
    """Get or create the meetings vector store."""
    global _meetings_store
    if _meetings_store is not None:
        return _meetings_store

    store_dir = _meetings_store_path()
    if Path(store_dir).exists() and (Path(store_dir) / "index.faiss").exists():
        _meetings_store = FAISS.load_local(
            store_dir, _get_embeddings(), allow_dangerous_deserialization=True
        )
    else:
        initialize_meetings_store()
        _meetings_store = FAISS.load_local(
            store_dir, _get_embeddings(), allow_dangerous_deserialization=True
        )

    return _meetings_store


def initialize_meetings_store() -> int:
    """Index all sample transcripts into the meetings vector store.

    Each transcript is split into chunks (by speaker turn) for better retrieval.

    Returns:
        Number of documents indexed.
    """
    global _meetings_store

    transcripts_dir = settings.SAMPLE_TRANSCRIPTS_DIR
    files = sorted(transcripts_dir.glob("*.txt"))

    if not files:
        raise FileNotFoundError(f"No transcript files found in {transcripts_dir}")

    documents = []
    for filepath in files:
        text = filepath.read_text()
        meeting_name = filepath.stem.replace("-", " ").title()

        # Extract metadata from header lines
        header_info = _parse_transcript_header(text)

        # Split transcript into chunks by speaker turns for granular retrieval
        chunks = _split_transcript_into_chunks(text, chunk_size=500)

        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": filepath.name,
                    "meeting": meeting_name,
                    "date": header_info.get("date", ""),
                    "attendees": header_info.get("attendees", ""),
                    "chunk_index": i,
                },
            )
            documents.append(doc)

    store = FAISS.from_documents(documents, _get_embeddings())

    store_dir = _meetings_store_path()
    Path(store_dir).mkdir(parents=True, exist_ok=True)
    store.save_local(store_dir)

    _meetings_store = store
    return len(documents)


def search_meetings(query: str, k: int = 3) -> list[Document]:
    """Search meeting history for relevant context.

    Args:
        query: The user's question about past meetings.
        k: Number of results to return.

    Returns:
        List of matching meeting transcript chunks.
    """
    store = get_meetings_vector_store()
    return store.similarity_search(query, k=k)


def search_team(query: str, k: int = 3) -> list[Document]:
    """Search team directory for relevant members.

    Convenience alias for lookup_team_member.
    """
    return lookup_team_member(query, k=k)


# --- Helpers ---


def _parse_transcript_header(text: str) -> dict:
    """Extract metadata from transcript header lines."""
    info: dict[str, str] = {}
    for line in text.strip().splitlines()[:5]:
        if line.startswith("Date:"):
            info["date"] = line.split(":", 1)[1].strip()
        elif line.startswith("Attendees:"):
            info["attendees"] = line.split(":", 1)[1].strip()
        elif line.startswith("Meeting:"):
            info["meeting"] = line.split(":", 1)[1].strip()
    return info


def _split_transcript_into_chunks(text: str, chunk_size: int = 500) -> list[str]:
    """Split a transcript into overlapping chunks for indexing.

    Tries to split on speaker turns first, then falls back to character-based chunking.
    """
    lines = text.strip().splitlines()

    # Separate header from body
    body_lines: list[str] = []
    header_lines: list[str] = []
    in_body = False
    for line in lines:
        if not in_body and ":" in line and not line.startswith(("Meeting:", "Date:", "Duration:", "Attendees:")):
            in_body = True
        if in_body:
            body_lines.append(line)
        else:
            header_lines.append(line)

    header = "\n".join(header_lines).strip()
    body = "\n".join(body_lines).strip()

    if not body:
        return [text.strip()] if text.strip() else []

    # Chunk the body text, prepending header context to each chunk
    chunks: list[str] = []
    current_chunk = header + "\n\n"

    for line in body_lines:
        if len(current_chunk) + len(line) > chunk_size and len(current_chunk) > len(header) + 10:
            chunks.append(current_chunk.strip())
            current_chunk = header + "\n\n" + line + "\n"
        else:
            current_chunk += line + "\n"

    if current_chunk.strip() and len(current_chunk) > len(header) + 10:
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text.strip()]
