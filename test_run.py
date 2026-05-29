"""
Entertainment Recommendation System - Test Run (auto-approves HITL)
Direct LLM node approach: no ReAct loop, reliable across models.
"""
import os, json, operator, warnings
warnings.filterwarnings("ignore")
from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0.7,
    max_tokens=1500,
)

# ── Tools ─────────────────────────────────────────────────────────────────────
search_tool = DuckDuckGoSearchRun()

@tool
def genre_matcher(preferences: str) -> str:
    """Maps preferences to genres."""
    mapping = {
        "action": ["Action", "Thriller", "Adventure"],
        "sci-fi": ["Science Fiction", "Space Opera", "Cyberpunk"],
        "comedy": ["Comedy", "Sitcom"], "romance": ["Romance", "Drama"],
        "horror": ["Horror", "Mystery"], "fantasy": ["Fantasy"],
        "music":  ["Pop", "Jazz", "Hip-hop"], "gaming": ["RPG", "Indie"],
        "relaxing": ["Slice of Life", "Ambient"], "animation": ["Anime", "Animation"],
    }
    result = {k: v for k, v in mapping.items() if k in preferences.lower()}
    return json.dumps(result or {"general": ["Drama", "Comedy", "Trending"]}, indent=2)

@tool
def schedule_planner(hours_available: str) -> str:
    """Creates time-slot schedule from available hours."""
    try:
        h = float(hours_available.strip().split()[0])
    except Exception:
        h = 3.0
    slots = []
    if h >= 2.0:
        slots.append({"slot": "Main Feature", "type": "Movie / 2-ep binge", "duration": "~2 hrs"})
    if h >= 2.75:
        slots.append({"slot": "Short Pick", "type": "Episode or short film", "duration": "~45 min"})
    if h >= 0.5:
        slots.append({"slot": "Wind-down", "type": "Music / Podcast", "duration": "~30 min"})
    return json.dumps({"hours": h, "slots": slots}, indent=2)

# ── Shared State ──────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    user_request: str
    next: str

PIPELINE = ["profiler", "researcher", "matcher", "curator", "planner", "reviewer"]

# ── Supervisor (deterministic) ────────────────────────────────────────────────
def supervisor_node(state: AgentState) -> dict:
    last = None
    for msg in reversed(state["messages"]):
        n = getattr(msg, "name", None)
        if n and n in PIPELINE:
            last = n
            break
    if last is None:
        return {"next": "profiler"}
    idx = PIPELINE.index(last)
    return {"next": PIPELINE[idx + 1] if idx + 1 < len(PIPELINE) else "FINISH"}

# ── Direct-LLM Agent Nodes ────────────────────────────────────────────────────
def invoke_agent(system_prompt: str, state: AgentState, name: str) -> dict:
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=response.content, name=name)]}

def profiler_node(state):
    return invoke_agent(
        "You are an entertainment preference profiler. "
        "Extract from the user's request a JSON profile with: "
        "genres, formats (movies/shows/music/games), mood, estimated_hours, liked_styles. "
        "Return only the JSON block.",
        state, "profiler"
    )

def researcher_node(state):
    # Step 1: generate search query
    q_msgs = [SystemMessage(content="Based on the conversation, write ONE web search query (max 10 words) to find great entertainment recommendations. Reply with just the query.")] + state["messages"]
    query  = llm.invoke(q_msgs).content.strip().strip('"')

    # Step 2: search
    try:
        results = search_tool.run(query)[:2000]
    except Exception as e:
        results = f"Search unavailable: {e}"

    # Step 3: format results
    fmt_msgs = [
        SystemMessage(content="You are a content researcher. Format the following search results into a clean list of 6-8 entertainment options. Each entry: title, type, rating (if known), platform, 1-sentence description."),
        *state["messages"],
        HumanMessage(content=f"Search query: '{query}'\n\nResults:\n{results}")
    ]
    response = llm.invoke(fmt_msgs)
    return {"messages": [AIMessage(content=response.content, name="researcher")]}

def matcher_node(state):
    return invoke_agent(
        "You are a taste matcher. Score each item from the researcher's list (1-10) "
        "against the preference profile. One-sentence reasoning per item. "
        "Return top 5 ranked by score.",
        state, "matcher"
    )

def curator_node(state):
    return invoke_agent(
        "You are a diversity curator. Review the top-5 list: "
        "ensure 2+ content formats (e.g. movie + show or music), "
        "max 2 items per exact genre. "
        "Add one [SERENDIPITY PICK] slightly outside the user's usual taste. "
        "Return the final 5-item curated list.",
        state, "curator"
    )

def planner_node(state):
    # Get schedule from tool
    try:
        schedule = schedule_planner.invoke({"hours_available": "3"})
    except Exception:
        schedule = '{"hours": 3, "slots": [{"slot": "Main Feature", "duration": "~2 hrs"}]}'

    plan_msgs = [
        SystemMessage(content=(
            f"You are an entertainment concierge. "
            f"Use this schedule: {schedule}\n"
            "Format the curated 5 picks as:\n"
            "Title | Type | Platform | Duration | Time Slot | Why it's perfect for you\n"
            "Mark the serendipity pick as '*** SURPRISE PICK ***' at the end."
        )),
        *state["messages"]
    ]
    response = llm.invoke(plan_msgs)
    return {"messages": [AIMessage(content=response.content, name="planner")]}

def reviewer_node(state):
    return invoke_agent(
        "You are a quality reviewer. Check the plan: "
        "are titles real? platforms correct? does it match user preferences? "
        "good variety? Apply corrections if needed. "
        "End with a 2-sentence 'WHY THIS PLAN WORKS FOR YOU' summary.",
        state, "reviewer"
    )

# ── Build Graph ───────────────────────────────────────────────────────────────
g = StateGraph(AgentState)
g.add_node("supervisor", supervisor_node)
g.add_node("profiler",   profiler_node)
g.add_node("researcher", researcher_node)
g.add_node("matcher",    matcher_node)
g.add_node("curator",    curator_node)
g.add_node("planner",    planner_node)
g.add_node("reviewer",   reviewer_node)

g.add_edge(START, "supervisor")
g.add_conditional_edges("supervisor", lambda s: s["next"], {
    "profiler":   "profiler",
    "researcher": "researcher",
    "matcher":    "matcher",
    "curator":    "curator",
    "planner":    "planner",
    "reviewer":   "reviewer",
    "FINISH":     END,
})
for m in PIPELINE:
    g.add_edge(m, "supervisor")

graph = g.compile(checkpointer=MemorySaver())

# ── Run ───────────────────────────────────────────────────────────────────────
USER_REQUEST = "I love sci-fi and action movies, have about 3 hours tonight"
config = {"configurable": {"thread_id": "test-1"}}
state  = {"messages": [HumanMessage(content=USER_REQUEST)], "user_request": USER_REQUEST, "next": ""}

print("="*60)
print(f"User: {USER_REQUEST}")
print("="*60 + "\n")

seen = set()
for event in graph.stream(state, config, stream_mode="values"):
    msgs = event.get("messages", [])
    if msgs:
        m     = msgs[-1]
        label = getattr(m, "name", None) or type(m).__name__
        if label not in PIPELINE:
            continue
        key = id(m)
        if key in seen:
            continue
        seen.add(key)
        if hasattr(m, "content") and m.content:
            preview = m.content[:600] + ("..." if len(m.content) > 600 else "")
            print(f"[{label.upper()}]:\n{preview}\n{'-'*50}\n")

final = graph.get_state(config).values.get("messages", [])
if final:
    print("\n" + "="*60)
    print("FINAL ENTERTAINMENT PLAN")
    print("="*60)
    print(final[-1].content)
