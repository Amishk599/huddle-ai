"""RAG setup and retrieval functions for team directory."""

import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.config import settings

_team_store: FAISS | None = None


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
