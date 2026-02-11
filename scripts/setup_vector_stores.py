"""Initialize vector stores with team directory and sample transcripts."""

import json
from pathlib import Path

# TODO: Implement in Phase 1
# - Load team directory JSON
# - Create Chroma vector store for team
# - Index sample transcripts for meetings store


def main():
    """Set up vector stores."""
    print("Vector store setup - implement in Phase 1")

    # Verify data files exist
    base_dir = Path(__file__).parent.parent
    team_file = base_dir / "data" / "team_directory.json"
    transcripts_dir = base_dir / "data" / "sample_transcripts"

    if team_file.exists():
        with open(team_file) as f:
            team = json.load(f)
        print(f"Found {len(team)} team members in directory")

    transcripts = list(transcripts_dir.glob("*.txt"))
    print(f"Found {len(transcripts)} sample transcripts")


if __name__ == "__main__":
    main()
