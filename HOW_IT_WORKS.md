# How It Works — Entertainment Recommendation & Curation System
**DSA 2020A – Lab 2 | Multi-Agent AI (LangGraph + Groq)**

---

## Table of Contents
1. [High-Level Overview](#1-high-level-overview)
2. [Project Files](#2-project-files)
3. [Shared State — the single source of truth](#3-shared-state)
4. [The Supervisor](#4-the-supervisor)
5. [The Six Worker Agents](#5-the-six-worker-agents)
6. [The Three Tools](#6-the-three-tools)
7. [Human-in-the-Loop](#7-human-in-the-loop)
8. [The Reflection / Critique Loop](#8-the-reflection--critique-loop)
9. [Graph Wiring & Execution Flow](#9-graph-wiring--execution-flow)
10. [Streaming Output](#10-streaming-output)
11. [Termination](#11-termination)
12. [How to Run](#12-how-to-run)
13. [Common Pitfalls & Solutions](#13-common-pitfalls--solutions)

---

## 1. High-Level Overview

The system is a **pipeline of 6 specialized AI agents** that collaborate to turn a free-text entertainment request into a polished, time-slotted plan. A **Supervisor** node orchestrates the pipeline by reading the message history and deciding which agent acts next. All agents share the same message history through a single `AgentState` object.

```
User request
     │
     ▼
 SUPERVISOR ──────────────────────────────────────────────────────┐
     │                                                             │
     │  routes to (in order)                                       │
     ▼                                                             │
 PROFILER → RESEARCHER → MATCHER → CURATOR* → PLANNER → REVIEWER │
     │           │           │         │           │         │    │
     └───────────┴───────────┴─────────┴───────────┴─────────┘    │
                             each node loops back ─────────────────┘
                             supervisor decides FINISH after reviewer
```

`*` CURATOR is preceded by a **Human-in-the-Loop** interrupt — execution pauses and waits for user approval or feedback before the curator runs.

---

## 2. Project Files

| File | Purpose |
|------|---------|
| [main.py](main.py) | **Production entry point.** Full interactive pipeline with Human-in-the-Loop via `input()`. Run this to use the system. |
| [test_run.py](test_run.py) | **Automated test.** Same pipeline but without HITL — auto-runs a hardcoded request end-to-end. Good for verifying the system works without user interaction. |
| [demo.ipynb](demo.ipynb) | **Jupyter notebook demo.** Step-by-step walkthrough of all components. Runs the same pipeline interactively in a notebook, with HITL handled via a `human_feedback` variable between cells. |
| [requirements.txt](requirements.txt) | Python dependency list. |
| [.env](.env) | Stores `GROQ_API_KEY`. Never commit a real key. |

---

## 3. Shared State

All agents communicate through a single `AgentState` TypedDict defined in both `main.py` and the notebook:

```python
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]   # full conversation history
    user_request: str                          # original user input (unchanged)
    next: str                                  # supervisor's routing decision
```

**Key design choice — `Annotated[list, operator.add]`:** When LangGraph merges state updates, it uses `operator.add` to *append* new messages rather than replace the list. This means every agent's output is accumulated and every subsequent agent has the full history of all prior outputs in its context window.

**MemorySaver checkpointer:** The compiled graph uses `MemorySaver()` as a checkpointer. This saves the full state after every node execution, enabling the graph to be *paused* (at the HITL interrupt) and *resumed* later with a new input — the state is restored exactly where it left off.

---

## 4. The Supervisor

**File:** `main.py:97–132` | **Notebook:** cell 6

The supervisor is a LangGraph node (not a separate process) that runs after every worker agent and decides what happens next.

```python
def supervisor_node(state: AgentState) -> dict:
    # 1. Find the last agent that wrote to state
    last_agent = None
    for msg in reversed(state['messages']):
        name = getattr(msg, 'name', None)
        if name and name in PIPELINE:
            last_agent = name
            break

    # 2. Build a routing prompt with explicit last-speaker context
    system = f"""Pipeline: profiler → researcher → matcher → curator → planner → reviewer → FINISH
Last agent: {last_agent or 'NONE'}
Return exactly the next agent name (or FINISH)."""

    # 3. LLM structured output → Route(next=...)
    try:
        result = llm.with_structured_output(Route).invoke([SystemMessage(content=system)])
        return {'next': result.next}
    except Exception:
        # 4. Deterministic fallback — prevents infinite loops
        if last_agent is None:
            return {'next': 'profiler'}
        idx = PIPELINE.index(last_agent)
        return {'next': PIPELINE[idx + 1] if idx + 1 < len(PIPELINE) else 'FINISH'}
```

**Why `llm.with_structured_output(Route)`?** It forces the LLM to return a Pydantic `Route` model whose `next` field is a `Literal` of valid agent names. This prevents the LLM from returning freeform text that can't be used as a graph edge key.

**Why a deterministic fallback?** On Groq's free tier, token limits can cause the LLM call to fail. The fallback reads `last_agent`, looks it up in the ordered `PIPELINE` list, and returns the next index — guaranteeing forward progress regardless of LLM availability.

**Important:** Do not wrap the LLM in `.with_retry()`. That returns a `RunnableRetry` object, which loses the `.with_structured_output()` and `.bind_tools()` methods that `ChatGroq` exposes. Always use plain `ChatGroq(...)`.

---

## 5. The Six Worker Agents

Each agent is a Python function that receives the full `AgentState`, calls the LLM (sometimes with tools), and returns a dict with one new `AIMessage` that carries the agent's `name` field. That name field is how the supervisor detects who last spoke.

### 5.1 Preference Profiler (`profiler`)

**Role:** Understand *what* the user wants.

**How it works (3-step):**
1. Calls the LLM to extract a comma-separated keyword list from the user request.
2. Passes those keywords to the `genre_matcher` tool to get a structured genre map.
3. Calls the LLM again with both the keywords and genre map to produce a full JSON taste profile.

**Output example:**
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

---

### 5.2 Content Researcher (`researcher`)

**Role:** Discover actual content that could match the profile.

**How it works (3-step):**
1. Asks the LLM to write a focused DuckDuckGo search query based on the full conversation.
2. Runs `DuckDuckGoSearchRun` with that query, truncating results to 2000 characters.
3. Asks the LLM to format the raw search results into a clean list of 6–8 options with title, type, rating, platform, and description.

**Why 3 steps?** If you just pass the user request directly to DuckDuckGo, the query is too broad. Having the LLM generate the query first produces much more targeted results.

---

### 5.3 Taste Matcher (`matcher`)

**Role:** Score and rank the researcher's list against the profiler's taste profile.

**How it works:** Single LLM call. The system prompt instructs it to score each item 1–10 with one-sentence reasoning, then return the top 5 ranked highest to lowest. Because the full `AgentState` message history is in context, the matcher sees both the profile (profiler's output) and the raw list (researcher's output) simultaneously.

---

### 5.4 Diversity & Serendipity Curator (`curator`)

**Role:** Ensure variety and add surprise; then wait for human approval.

**How it works:**
1. LLM call with instructions to enforce: ≥2 content formats, max 2 items per exact genre, and exactly one `[SERENDIPITY PICK]` slightly outside the user's usual taste.
2. **HUMAN-IN-THE-LOOP:** calls `interrupt({...})` which signals LangGraph to pause execution and surface the curated list for user review. Execution suspends here until the graph is resumed.
3. On resume, if the user typed feedback, it appends that feedback to the curator's message before continuing.

This node is also pre-empted by `interrupt_before=['curator']` in the graph compilation — the graph pauses *before* the node even runs, which is why the streaming loop in Phase 1 ends just before the curator.

---

### 5.5 Plan Builder (`planner`)

**Role:** Build a time-slotted, formatted entertainment plan.

**How it works:**
1. Calls `schedule_planner` tool with `"3"` hours to get a JSON schedule structure.
2. Calls the LLM with that schedule and the curated list, asking for a formatted table: `Title | Type | Platform | Duration | Time Slot | Why it's perfect for you`.
3. Marks the serendipity pick as `*** SURPRISE PICK ***`.

---

### 5.6 Quality Reviewer (`reviewer`)

**Role:** Independent critique and fact-checking. This is the **reflection loop**.

**How it works:** Single LLM call. The system prompt gives it the role of an outside quality auditor: check that titles are real, platforms are correct, the plan matches user preferences, and there is good variety. Apply corrections if needed, then append a 2-sentence "WHY THIS PLAN WORKS FOR YOU" summary.

**Why is this meaningful?** The reviewer has no memory of *producing* the plan — it approaches the output with genuine skepticism, just like a human reviewer would. This is categorically different from the same agent self-reviewing, which tends to confirm rather than critique.

---

## 6. The Three Tools

| Tool | Type | Where Used |
|------|------|-----------|
| `DuckDuckGoSearchRun` | LangChain community (no API key) | `researcher_node` |
| `genre_matcher` | Custom `@tool` | `profiler_node` |
| `schedule_planner` | Custom `@tool` | `planner_node` |

### `genre_matcher`
Takes a string of preference keywords, scans for matches against a 12-category mapping, and returns a JSON object of `{keyword: [genres]}`. Used to translate vague user language ("I like exciting stuff") into structured genre categories the matcher and curator can reason about.

### `schedule_planner`
Takes hours as a string, parses it to a float, and generates a time-slot JSON:
- ≥2 hours → Main Feature slot (~2 hrs)
- ≥0.75 remaining → Mid-session slot (~45 min)
- ≥0.5 remaining → Wind-down slot (~30 min)

The planner uses this structure to assign each content pick to the most appropriate time slot.

---

## 7. Human-in-the-Loop

**Implementation:** `langgraph.types.interrupt()` + `interrupt_before=['curator']`

The HITL is implemented in two places that work together:

**Graph compilation** (`main.py:290–293`):
```python
return g.compile(
    checkpointer=MemorySaver(),
    interrupt_before=['curator'],   # pause BEFORE curator node runs
)
```

**Inside curator_node** (`main.py:230–233`):
```python
feedback = interrupt({
    "prompt": "Approve or type changes:",
    "curated_list": content
})
```

**Execution flow for HITL:**

```
Phase 1: graph.stream(initial_state, config)
         → supervisor → profiler → supervisor → researcher → supervisor → matcher
         → supervisor decides "curator"
         → graph PAUSES (interrupt_before fires before curator starts)
         → streaming loop exits

User reviews the matcher's top-5, types feedback (or presses Enter)

graph.update_state(config, {"messages": [HumanMessage(human_input)]})
         → injects human message into the saved checkpoint state
         → graph remains paused (update_state never triggers execution)

Phase 2: graph.stream(None, config)
         → None tells LangGraph to RESUME from the existing checkpoint
         → curator_node runs, reads the HumanMessage from state["messages"][-1]
         → incorporates feedback if the message isn't a plain approval
         → supervisor → planner → supervisor → reviewer → supervisor → FINISH
```

**Critical detail — why you must use `update_state` + `stream(None)`:**
Passing a state dict directly to `stream()` (e.g. `stream({"messages": [...]}, config)`) updates the checkpoint but does NOT resume execution — the graph stays paused. You must call `stream(None, config)` to signal LangGraph to actually continue from where it stopped. `update_state` is the correct way to inject the human message before that resume.

The `MemorySaver` checkpointer is what makes the pause-and-resume possible: it saves complete state after each node, and the `thread_id` in the config identifies which checkpoint to restore.

---

## 8. The Reflection / Critique Loop

The Quality Reviewer is structurally independent from the agents that produced the plan. It receives the full message history (all prior agent outputs) but its system prompt positions it as an outside auditor, not a co-author. This asymmetry — producer vs. independent evaluator — is the key to the reflection loop:

- Producers optimize for completing their task.
- The reviewer optimizes for catching problems.

In practice, the reviewer catches platform errors (e.g., "Dune: Part Two is not on Netflix"), preference mismatches (e.g., a horror pick for a user who asked for relaxing content), and format diversity gaps.

---

## 9. Graph Wiring & Execution Flow

```python
g.add_edge(START, "supervisor")              # entry point

g.add_conditional_edges(
    "supervisor",
    lambda s: s["next"],                     # routing key from supervisor
    {
        "profiler":   "profiler",
        "researcher": "researcher",
        "matcher":    "matcher",
        "curator":    "curator",
        "planner":    "planner",
        "reviewer":   "reviewer",
        "FINISH":     END,                   # terminates the graph
    }
)

for member in PIPELINE:
    g.add_edge(member, "supervisor")         # every worker loops back to supervisor
```

Every worker agent has an unconditional edge back to the supervisor. The supervisor then evaluates the new state and routes forward. This supervisor-loop pattern is LangGraph's **hierarchical supervisor architecture**, which is the recommended approach for sequential multi-agent pipelines.

**State at each step:**

```
Initial:   messages = [HumanMessage("I love sci-fi...")]
After profiler:   messages = [..., AIMessage(name="profiler", content="...JSON...")]
After researcher: messages = [..., AIMessage(name="researcher", content="...list...")]
After matcher:    messages = [..., AIMessage(name="matcher", content="...top 5...")]
[HITL PAUSE]
After curator:    messages = [..., AIMessage(name="curator", content="...curated...")]
After planner:    messages = [..., AIMessage(name="planner", content="...plan...")]
After reviewer:   messages = [..., AIMessage(name="reviewer", content="...reviewed...")]
```

---

## 10. Streaming Output

```python
for event in graph.stream(state, config, stream_mode="values"):
    msgs = event.get("messages", [])
    if msgs:
        m = msgs[-1]
        label = getattr(m, "name", None) or type(m).__name__
        if label not in PIPELINE:
            continue
        key = id(m)
        if key in seen:          # deduplication — same message object can appear
            continue             # in multiple stream events
        seen.add(key)
        print(f"[{label.upper()}]:\n{m.content[:500]}")
```

`stream_mode="values"` yields the full state dict after each node execution. The deduplication via `id(m)` (object identity) prevents the same message from printing twice when it appears in consecutive events as part of the accumulated history.

---

## 11. Termination

The supervisor returns `{"next": "FINISH"}` after the reviewer completes. The conditional edge maps `"FINISH"` to `END`, which is LangGraph's built-in terminal node. When the graph reaches `END`, execution stops cleanly. There is no max-turn counter needed because the supervisor enforces a strictly sequential, non-repeating pipeline — each agent name can only appear once before `FINISH` is reached.

---

## 12. How to Run

### Prerequisites
- Python 3.10–3.13 (not 3.14 — LangGraph is not tested on 3.14 yet)
- A free Groq API key from [console.groq.com](https://console.groq.com)

### Setup
```bash
# Create and activate virtualenv
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Add your key to .env
echo GROQ_API_KEY=gsk_your_key_here > .env
```

### Run the full interactive system
```bash
python main.py
# Enter your entertainment request when prompted
# When the HITL pause appears, press Enter to approve or type feedback
```

### Run the automated test (no interaction required)
```bash
python test_run.py
# Runs a hardcoded sci-fi request end-to-end with no pauses
```

### Run the Jupyter notebook demo
```bash
pip install jupyter ipykernel
jupyter notebook demo.ipynb
# Run cells top-to-bottom
# Cell 12 streams Phase 1 (profiler → matcher)
# Edit human_feedback in cell 13, then run it to resume
# Cell 14 prints the final plan
```

---

## 13. Common Pitfalls & Solutions

| Problem | Root Cause | Fix |
|---------|-----------|-----|
| `AttributeError: 'RunnableRetry' object has no attribute 'with_structured_output'` | `llm = ChatGroq(...).with_retry(...)` — `.with_retry()` strips the ChatModel interface | Use plain `ChatGroq(...)` with no `.with_retry()` wrapper |
| `AttributeError: 'RunnableRetry' object has no attribute 'bind_tools'` | Same as above | Same fix |
| `LangGraphDeprecatedSinceV10: create_react_agent has been moved` | `from langgraph.prebuilt import create_react_agent` is deprecated | Use `from langchain.agents import create_agent` or, better, implement direct-LLM node functions as in `main.py` |
| Supervisor enters a routing loop | LLM misreads conversation history without explicit last-speaker context | Pass `last_agent` explicitly in the supervisor prompt + keep the deterministic fallback |
| `AttributeError: 'NoneType' object has no attribute 'upper'` | `getattr(m, 'name', type(m).__name__)` — when `name=None`, returns `None` not the type name | Use `getattr(m, 'name', None) or type(m).__name__` |
| Groq `RateLimitError` (429) | Free tier has low TPM on large models | Switch to `meta/llama-4-scout-17b-16e-instruct` which has a much higher free-tier quota |
| Hallucinated streaming platforms | LLM fabricates "available on Netflix" | Quality Reviewer agent explicitly instructed to fact-check platform claims |
| HITL doesn't resume | Wrong `thread_id` in resume call | Both `stream()` calls (Phase 1 and Phase 2) must use the identical `config` dict with the same `thread_id` |
