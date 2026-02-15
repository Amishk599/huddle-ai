# Architecture Documentation

## Overview

Huddle AI uses a LangGraph state machine to process meeting transcripts and extract actionable intelligence.

## Design Decisions

### 1. State Machine Architecture

We chose LangGraph for orchestrating the processing pipeline because:
- Clear separation of concerns (each node has single responsibility)
- Easy to debug with LangSmith tracing
- Conditional routing for flexible flow control
- State management built-in

### 2. Dual RAG Strategy

Two separate vector stores:
- **Team Directory**: For assignee matching based on expertise
- **Meeting History**: For assistant queries about past meetings

This separation provides:
- Cleaner retrieval results
- Easier maintenance
- Better relevance for different query types

### 3. Mock Email System

Emails are written to files instead of sent because:
- No external dependencies
- Easy to verify outputs
- Safe for testing
- Clear audit trail

### 4. Pydantic for Structured Output

Using Pydantic models ensures:
- Type safety throughout the pipeline
- Validation of LLM outputs
- Clear data contracts
- Easy serialization

## Flow Diagram

```
┌─────────┐
│  Start  │
└────┬────┘
     │
     ▼
┌─────────┐
│ Intake  │ ← Validate transcript
└────┬────┘
     │
     ▼
┌──────────┐
│Summarize │ ← Generate meeting summary
└────┬─────┘
     │
     ▼
┌─────────────────────┐
│Extract Action Items │ ← Parse tasks from transcript
└────────┬────────────┘
         │
    ┌────┴────┐
    │ Actions │
    │ Found?  │
    └────┬────┘
    No   │   Yes
    │    │
    ▼    ▼
┌─────┐ ┌──────────────┐
│Done │ │Assign Owners │ ← RAG lookup
└─────┘ └──────┬───────┘
               │
               ▼
        ┌──────────────────┐
        │Determine Deadlines│ ← Parse/infer dates
        └────────┬─────────┘
                 │
                 ▼
          ┌────────────┐
          │Send Emails │ ← Write to files
          └──────┬─────┘
                 │
                 ▼
        ┌────────────────┐
        │Display Results │
        └────────────────┘
```

## Future Improvements

- Real email integration (SendGrid, SES)
- Slack/Teams webhook notifications
- PII redaction
- Multi-language support
