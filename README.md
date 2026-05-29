# Entertainment Recommendation & Curation System
**DSA 2020A – Lab 2 | Multi-Agent AI System (LangGraph + Groq)**

## Chosen Use Case & Rationale
**Entertainment Recommendation & Curation Team**

This use case was selected because it demonstrates all required multi-agent coordination patterns without requiring sensitive disclaimers or complex external APIs. The domain is naturally suited to a pipeline of specialized agents: preference extraction → content discovery → taste matching → curation → plan generation → quality review. Each step has a clear, distinct responsibility that benefits from agent specialization.

---

## Agent Team Diagram

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────┐
│            SUPERVISOR / MANAGER AGENT           │
│   (LangGraph StateGraph — LLM-based routing)    │
│  Decides task order, delegation, termination    │
└────────────────────┬────────────────────────────┘
                     │ delegates to
    ┌────────────────┼───────────────────────┐
    ▼                ▼                       ▼
┌──────────┐  ┌──────────────┐  ┌─────────────────┐
│Preference│  │   Content    │  │  Taste Matcher  │
│ Profiler │  │  Researcher  │  │                 │
│          │  │              │  │ Scores content  │
│Genre     │  │DuckDuckGo    │  │ vs. profile     │
│Matcher   │  │search tool   │  │ Genre Matcher   │
│tool      │  │              │  │ tool            │
└────┬─────┘  └──────┬───────┘  └────────┬────────┘
     │               │                   │
     └───────────────┴───────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Diversity & Serendip │◄── HUMAN-IN-THE-LOOP
         │      Curator          │    (user approves list)
         │ DuckDuckGo search     │
         └──────────┬────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  Entertainment Plan  │
         │      Builder         │
         │ Schedule Planner tool│
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │   Quality Reviewer   │◄── REFLECTION/CRITIQUE LOOP
         │  (critique & approve)│
         └──────────┬───────────┘
                    │
                    ▼
             Final Plan Output
```

**Communication Flow:** Sequential pipeline with shared state. Each agent receives the full `AgentState` message history (all prior agent outputs). The LangGraph supervisor reads the last speaker's name and routes to the next agent. Human-in-the-loop pauses occur at the curation step via `interrupt_before`.

---

## Framework & Architecture
- **Framework:** LangGraph (hierarchical supervisor pattern — strongly recommended by assignment)
- **LLM:** Groq `llama-3.3-70b-versatile` via `langchain-groq`
- **Pattern:** `StateGraph` with supervisor node + `MemorySaver` checkpointer for human-in-the-loop
- **State:** Shared `AgentState` TypedDict with message history passed across all nodes

## Agents & Roles

| Agent | Role | Tools |
|-------|------|-------|
| Supervisor | Routes tasks between workers, decides next agent, terminates on FINISH | — |
| Preference Profiler | Extracts user taste profile from request | Genre Matcher |
| Content Researcher | Searches for matching content | DuckDuckGo Search |
| Taste Matcher | Scores content vs. profile (1-10) | Genre Matcher |
| Diversity Curator | Ensures variety + serendipity pick *(HITL pause)* | DuckDuckGo Search |
| Plan Builder | Builds formatted schedule | Schedule Planner |
| Quality Reviewer | Critiques and approves final plan *(reflection loop)* | — |

## Tools

| Tool | Type | Purpose |
|------|------|---------|
| DuckDuckGoSearchRun | LangChain (no API key needed) | Real-time content discovery |
| genre_matcher | Custom `@tool` | Maps preference text to genre categories |
| schedule_planner | Custom `@tool` | Generates time-slot schedule from available hours |

---

## Setup & How to Run

### 0. Python version
Requires **Python 3.10–3.13**. If you only have Python 3.14:
```bash
# Windows — install Python 3.12 via winget
winget install Python.Python.3.12
# Then create a virtualenv:
py -3.12 -m venv .venv
.venv\Scripts\activate
```

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a Groq API Key (free)
1. Go to [console.groq.com](https://console.groq.com) and sign up
2. Create an API key under "API Keys"

### 3. Set environment variable
```bash
# Option A: Create a .env file
echo GROQ_API_KEY=your_key_here > .env

# Option B: Set in terminal
# Windows PowerShell:
$env:GROQ_API_KEY = "your_key_here"
# Linux/Mac:
export GROQ_API_KEY="your_key_here"
```

### 4. Run
```bash
# Activate venv first
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac/Linux

python main.py
```

### 5. Jupyter Demo
```bash
.venv\Scripts\activate
pip install jupyter ipykernel   # if not installed
jupyter notebook demo.ipynb
```

---

## Example Interaction Transcripts

### Transcript 1 — Movie Night Request
**User:** "I want something exciting for tonight, I love sci-fi and action movies, have about 3 hours"

**Preference Profiler Output:**
```json
{
  "genres": ["Science Fiction", "Action", "Thriller"],
  "formats": ["movies"],
  "mood": "exciting",
  "estimated_hours": 3,
  "liked_styles": ["fast-paced", "high-stakes"],
  "exclusions": []
}
```

**Content Researcher** → Found: Dune Part 2, The Creator, Mission Impossible Dead Reckoning, Arrival, Mad Max Fury Road, Interstellar

**Taste Matcher** → Ranked: Dune Part 2 (9/10), Interstellar (8/10), The Creator (8/10), Mission Impossible (7/10), Arrival (7/10)

**Diversity Curator** → Added serendipity pick: "Everything Everywhere All at Once" (slightly quirky but action-packed)

**Plan Builder** → Generated 3-hour schedule: Main Feature (Dune Part 2, 2h36m) + short wind-down music playlist

**Quality Reviewer** → Approved with note: "Plan balances spectacular sci-fi visuals with high-stakes action perfectly."

---

### Transcript 2 — Chill Weekend Request
**User:** "I want something relaxing for the weekend, mix of shows and music, maybe 5 hours total"

**Preference Profiler Output:**
```json
{
  "genres": ["Slice of Life", "Drama", "Ambient"],
  "formats": ["shows", "music"],
  "mood": "relaxing",
  "estimated_hours": 5,
  "liked_styles": ["slow-paced", "atmospheric"],
  "exclusions": []
}
```

**Agent Pipeline** → Recommended: Chef's Table (Netflix), The Bear S1 (Hulu), Nils Frahm album "All Melody", Studio Ghibli film Spirited Away, Serendipity pick: "Our Planet" documentary

**Final Plan:** 2-hour main show binge → 45-min episode → 30-min ambient music playlist → weekend extension picks

---

### Transcript 3 — Gaming + Music Combo
**User:** "Looking for games and music to match — I like indie stuff and hip hop, have about 4 hours"

**Agent Pipeline** → Profile: indie gaming + hip-hop music → Researcher found Hades, Hollow Knight, Kendrick Lamar "Mr. Morale", Tyler the Creator "Flower Boy" → Curator added serendipity: "Stardew Valley" as cozy counterbalance → Plan Builder created 4-hour gaming+listening session

---

## Key Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| **Groq daily token limits** on free tier | Switch to `llama-4-scout-17b` (separate quota); fall back to deterministic supervisor to save tokens |
| **Hallucinated platforms** (agent says "on Netflix" incorrectly) | Quality Reviewer agent with explicit fact-checking instructions |
| **Agent routing loops** (supervisor re-calling same agent) | LLM supervisor given explicit last-speaker context; deterministic fallback prevents infinite loops |
| **Context loss** between agents | Full `AgentState` message history passed to every node; agents read all prior outputs |
| **Serendipity pick too random** | Curator agent given explicit constraint: "close enough to taste to be enjoyable" |

---

## Reflection Report

**Multi-Agent Advantages vs. Single Agent**

A single generalist agent attempting this task would face a fundamental tension: it must simultaneously understand user psychology, know current content across all platforms, apply objective scoring logic, curate for diversity, format for readability, AND verify factual accuracy — all in one pass. This creates a context overload problem where the agent's attention is divided across conflicting priorities. In contrast, the multi-agent system divides these cognitive tasks cleanly: the Preference Profiler operates only on psychology, the Content Researcher only on discovery, and the Quality Reviewer only on verification. Each agent has a focused system prompt that makes it reliably better at its specific subtask than a generalist attempting everything at once.

The multi-agent approach also enables a genuine critique loop that a single agent cannot provide for itself. Having the Quality Reviewer operate as an independent agent — with no memory of producing the plan — creates authentic skepticism. It approaches the plan as an outside evaluator would, catching errors and misalignments that the original agents, having produced the content, are cognitively "invested" in. The human-in-the-loop checkpoint at the curation stage further demonstrates a key advantage of multi-agent architecture: natural breakpoints where human judgment can intercept the workflow before irreversible decisions are made, a pattern that would be awkward to implement in a single-agent pipeline.

Finally, the parallelization potential of multi-agent systems provides scalability benefits beyond this assignment. While this implementation runs sequentially, the architecture naturally supports running the Content Researcher and Genre Matcher in parallel, reducing latency. A single agent that internally "debates with itself" lacks this property. The division of labor also makes debugging and improvement straightforward: if recommendations are poor, only the Taste Matcher or Researcher prompts need tuning, rather than re-engineering a monolithic agent.

One honest limitation surfaced during development: the LLM-based supervisor occasionally entered routing loops when the underlying model (LLaMA 3.3-70B on Groq's free tier) misread the conversation history. The solution was to supply the supervisor with explicit last-speaker context and a deterministic fallback — a reminder that in production multi-agent systems, supervisors often need a rule-based safety net alongside their LLM judgment. This trade-off between flexibility and reliability is a real engineering challenge the assignment exposed.
