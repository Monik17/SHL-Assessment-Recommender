# SHL Conversational Assessment Recommender

A conversational AI agent that recommends SHL assessments based on hiring needs.

## Architecture

- **FastAPI** REST service with `GET /health` and `POST /chat`
- **Claude claude-sonnet-4-20250514** as the reasoning engine
- **Keyword-based RAG** retrieval over the SHL Individual Test Solutions catalog
- **Stateless design**: full conversation history sent on every request

## Quick Start (Local)

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY=your_key_here

# Start server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Test health
curl http://localhost:8000/health

# Test chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I need assessments for a Java developer"}]}'
```

## Run Evaluations

```bash
python evaluate.py --url http://localhost:8000
```

## Deploy to Render

1. Push this repo to GitHub
2. Create a new Web Service on [Render](https://render.com)
3. Connect your GitHub repo
4. Set env variable: `ANTHROPIC_API_KEY`
5. Build command: `pip install -r requirements.txt`
6. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## API Specification

### GET /health
Returns `{"status": "ok"}` with HTTP 200.

### POST /chat
**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure. What is seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

**Response:**
```json
{
  "reply": "Got it. Here are 5 assessments that fit a mid-level Java dev...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

## Conversation Behaviors

| Behavior | How it works |
|----------|-------------|
| **Clarify** | Agent asks focused questions when query is vague |
| **Recommend** | 1–10 catalog assessments once context is sufficient |
| **Refine** | Updates shortlist when user adds/changes constraints |
| **Compare** | Grounds comparison in catalog data only |
| **Refuse** | Declines off-topic, legal, and injection attempts |

## Catalog Coverage

60+ Individual Test Solutions including:
- **Cognitive**: Verify series (Numerical, Verbal, Inductive, Deductive), General Ability
- **Personality**: OPQ32r, MQ, Workplace Personality Inventory
- **Technical/Knowledge**: Java, Python, JavaScript, SQL, C#, React, Angular, Spring, ML, Data Science, AWS, DevOps, Cyber Security
- **Situational Judgment**: SJT Manager, Leadership Judgment Indicator
- **Simulations**: Automata Fix, Automata Pro, Contact Center Simulation
- **Behavioral**: Universal Competency Questionnaire, Customer Contact Competencies Inventory

## Evaluation Metrics

- **Schema compliance**: Every response has `reply`, `recommendations[]`, `end_of_conversation`
- **Recall@10**: Fraction of expected assessments appearing in top-10 recommendations
- **Behavior probes**: Refusal rate, clarification on vague queries, refinement honoring
