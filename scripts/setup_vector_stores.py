"""Initialize vector stores with team directory and sample transcripts."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag import initialize_team_store


def main():
    """Set up vector stores."""
    print("Initializing team directory vector store...")
    count = initialize_team_store()
    print(f"Indexed {count} team members into vector store.")
    print("Vector store setup complete.")


if __name__ == "__main__":
    main()
