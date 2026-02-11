# LangSmith Integration Guide

## Setup

1. Create a LangSmith account at https://smith.langchain.com

2. Get your API key from Settings > API Keys

3. Add to your `.env`:
   ```
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   LANGCHAIN_API_KEY=lsv2_pt_your_key_here
   LANGCHAIN_PROJECT=huddle-ai
   ```

## Automatic Tracing

All LangGraph and LangChain calls are automatically traced when:
- `LANGCHAIN_TRACING_V2=true`
- Valid `LANGCHAIN_API_KEY` is set

No additional code changes needed.

## Viewing Traces

1. Go to https://smith.langchain.com
2. Select your project ("huddle-ai")
3. View individual runs with:
   - Input/output for each node
   - Token usage
   - Latency
   - Error details

## Metadata Tagging

Runs are tagged with metadata for filtering:
- `source`: Where the transcript came from
- `transcript_length`: Character count
- `action_count`: Number of extracted actions

## Running Evaluations

```bash
make eval
```

This will:
1. Load test cases from `evals/eval_dataset.json`
2. Run the processing pipeline
3. Score outputs against expected results
4. Push results to LangSmith

## Debugging Tips

- Use "Playground" to test prompts interactively
- Compare runs to identify regressions
- Filter by metadata to find specific run types
- Export datasets for offline analysis
