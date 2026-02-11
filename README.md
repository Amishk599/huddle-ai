# Huddle AI

A terminal-based agentic application that processes meeting transcripts and extracts actionable intelligence using LangGraph, LangChain, and LangSmith.

## Features

- **Meeting Processing**: Transform chaotic meeting discussions into structured action items
- **Automatic Assignee Detection**: RAG-powered team member matching based on expertise
- **Deadline Extraction**: Intelligent parsing and inference of due dates
- **Mock Email Notifications**: Simulated email sending to action item owners
- **Interactive Assistant**: RAG-powered chat for team and meeting queries
- **Full Observability**: LangSmith integration for tracing and evaluation

## Architecture

```
Main Menu
├─ 1. Process Meeting Transcript
│  ├─ 1. Use Sample Transcript (navigable list)
│  └─ 2. Paste Transcript (with preview)
│
└─ 2. Assistant (RAG-powered chat)
   ├─ General questions → Direct LLM
   ├─ Team queries → Team Directory RAG
   └─ Meeting queries → Meeting History RAG
```

### LangGraph State Machine

```
intake → summarize → extract_action_items
                           ↓
                    (if actions found)
                           ↓
       assign_owners → determine_deadlines → send_emails → display_results
              ↑
          (RAG lookup)
```

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key
- LangSmith API key (for tracing)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd huddle-ai
   ```

2. **Create and activate virtual environment**
   ```bash
   make setup
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   make install
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Initialize vector stores**
   ```bash
   make init-stores
   ```

6. **Run the application**
   ```bash
   make run
   ```

## Project Structure

```
huddle-ai/
├── src/
│   ├── main.py          # Application entry point
│   ├── agent.py         # LangGraph state machine
│   ├── tools.py         # LangChain tools (email sender)
│   ├── rag.py           # Vector store setup and retrieval
│   ├── prompts.py       # Prompt templates
│   ├── models.py        # Pydantic models
│   ├── config.py        # Application configuration
│   ├── ui/              # Terminal UI components
│   │   ├── menus.py     # Main menu and navigation
│   │   ├── processing.py # Progress display
│   │   ├── results.py   # Results formatting
│   │   └── assistant.py # Chat interface
│   └── assistant/       # RAG-powered assistant
│       └── chain.py     # Question routing and chains
│
├── data/
│   ├── team_directory.json      # Team member data for RAG
│   ├── sample_transcripts/      # Example meeting transcripts
│   └── vector_stores/           # Chroma databases
│
├── evals/
│   ├── run_evals.py     # Evaluation runner
│   └── eval_dataset.json # Test cases
│
├── scripts/
│   ├── setup_vector_stores.py   # Initialize Chroma stores
│   └── generate_demo_traces.py  # Create demo traces
│
└── docs/
    ├── architecture.md  # Design decisions
    └── langsmith-guide.md # LangSmith setup
```

## Usage

### Process a Meeting Transcript

1. Run `make run`
2. Select "1. Process Meeting Transcript"
3. Choose a sample transcript or paste your own
4. Watch the processing pipeline execute
5. View extracted action items with assignees and deadlines

### Use the Assistant

1. Run `make run`
2. Select "2. Assistant"
3. Ask questions about:
   - Team members: "Who has Python expertise?"
   - Past meetings: "What was discussed with Todd?"
   - General topics: "What is LangGraph?"

## Development

### Run Tests
```bash
make test
```

### Run Evaluations
```bash
make eval
```

### Clean Up
```bash
make clean
```

## Technologies

- **LangGraph** - State machine orchestration
- **LangChain** - LLM calls, RAG, tools
- **LangSmith** - Tracing and evaluations
- **OpenAI** - gpt-4o-mini
- **Chroma** - Vector store
- **Rich** - Terminal UI
- **Prompt Toolkit** - Interactive menus
- **Pydantic** - Structured outputs

## License

MIT
