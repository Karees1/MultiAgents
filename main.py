"""
Entertainment Recommendation & Curation System
DSA 2020A – Lab 2 | Multi-Agent AI (LangGraph + Groq)

Agents: Profiler → Researcher → Matcher → Curator (HITL) → Planner → Reviewer
Supervisor orchestrates the pipeline; Quality Reviewer is the reflection/critique loop.
"""

import os
import json
import operator
import warnings
warnings.filterwarnings("ignore")
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────
# llama-3.3-70b-versatile gives best quality; llama-4-scout also works.
LLM_MODEL = os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

llm = ChatGroq(
    model=LLM_MODEL,
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0.7,
    max_tokens=1500,
)

# ── Tools ─────────────────────────────────────────────────────────────────────
search_tool = DuckDuckGoSearchRun()

@tool
def genre_matcher(preferences: str) -> str:
    """Maps user preference keywords to entertainment genres and formats."""
    mapping = {
        "action":      ["Action", "Thriller", "Adventure", "Superhero"],
        "sci-fi":      ["Science Fiction", "Space Opera", "Cyberpunk", "Dystopian"],
        "comedy":      ["Comedy", "Sitcom", "Stand-up"],
        "romance":     ["Romance", "Drama", "Rom-com"],
        "horror":      ["Horror", "Psychological Thriller", "Mystery"],
        "documentary": ["Documentary", "True Crime", "Nature"],
        "fantasy":     ["Fantasy", "Epic Fantasy", "Mythology"],
        "animation":   ["Animation", "Anime"],
        "music":       ["Pop", "Jazz", "Hip-hop", "Rock", "R&B"],
        "gaming":      ["RPG", "Indie", "Action-Adventure", "Strategy"],
        "relaxing":    ["Slice of Life", "Nature Documentary", "Ambient"],
        "exciting":    ["Action", "Thriller", "Competition"],
    }
    result = {k: v for k, v in mapping.items() if k in preferences.lower()}
    return json.dumps(result or {"general": ["Drama", "Comedy", "Trending"]}, indent=2)


@tool
def schedule_planner(hours_available: str) -> str:
    """Creates a time-slot entertainment schedule from available hours."""
    try:
        hours = float(hours_available.strip().split()[0])
    except Exception:
        hours = 3.0
    slots = []
    remaining = hours
    if remaining >= 2.0:
        slots.append({"slot": "Main Feature", "type": "Movie / 2-episode binge", "duration": "~2 hours"})
        remaining -= 2.0
    if remaining >= 0.75:
        slots.append({"slot": "Mid-session Pick", "type": "Single episode or short film", "duration": "~45 min"})
        remaining -= 0.75
    if remaining >= 0.5:
        slots.append({"slot": "Wind-down", "type": "Music playlist or podcast", "duration": "~30 min"})
    return json.dumps({"available_hours": hours, "schedule": slots}, indent=2)


# ── Shared State ──────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    user_request: str
    next: str


PIPELINE = ["profiler", "researcher", "matcher", "curator", "planner", "reviewer"]

from typing import Literal
from pydantic import BaseModel

class Route(BaseModel):
    next: Literal["profiler", "researcher", "matcher", "curator", "planner", "reviewer", "FINISH"]

def supervisor_node(state: AgentState) -> dict:
    """LLM-based supervisor: reads conversation context and routes to the next agent."""
    # Detect who last spoke to give the LLM explicit context
    last_agent = None
    for msg in reversed(state["messages"]):
        name = getattr(msg, "name", None)
        if name and name in PIPELINE:
            last_agent = name
            break

    system = f"""You are the supervisor of an entertainment recommendation team.
Pipeline (always sequential, never skip or go backwards):
  profiler → researcher → matcher → curator → planner → reviewer → FINISH

Last agent to respond: {last_agent or 'NONE — pipeline has not started yet'}

Routing rules:
- NONE       → profiler
- profiler   → researcher
- researcher → matcher
- matcher    → curator
- curator    → planner
- planner    → reviewer
- reviewer   → FINISH

Return exactly the next agent name (or FINISH)."""

    try:
        result = llm.with_structured_output(Route).invoke([SystemMessage(content=system)])
        return {"next": result.next}
    except Exception:
        # Deterministic fallback if LLM routing fails
        if last_agent is None:
            return {"next": "profiler"}
        idx = PIPELINE.index(last_agent)
        return {"next": PIPELINE[idx + 1] if idx + 1 < len(PIPELINE) else "FINISH"}


# ── Agent Node Helpers ────────────────────────────────────────────────────────
def call_llm(system_prompt: str, state: AgentState) -> str:
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    return llm.invoke(messages).content


def make_simple_node(system_prompt: str, name: str):
    def node(state: AgentState) -> dict:
        content = call_llm(system_prompt, state)
        return {"messages": [AIMessage(content=content, name=name)]}
    return node


# ── Specialized Agent Nodes ───────────────────────────────────────────────────
def profiler_node(state: AgentState) -> dict:
    """Builds a taste profile, then calls genre_matcher to enrich genre mapping."""
    # Step 1: extract raw preferences from user request
    raw = call_llm(
        "You are an entertainment preference profiler. "
        "Extract from the user request a comma-separated list of genre/mood keywords "
        "(e.g. 'sci-fi, action, relaxing'). Reply with ONLY the keywords.",
        state
    )
    # Step 2: run genre_matcher tool to get structured genres
    genre_map = genre_matcher.invoke({"preferences": raw})
    # Step 3: build full JSON profile
    profile_msgs = [
        SystemMessage(content=(
            "You are an entertainment preference profiler. "
            "Using the keyword list and genre map below, return a complete JSON profile with: "
            "genres, formats (movies/shows/music/games), mood, estimated_hours, liked_styles, exclusions.\n\n"
            f"Keywords: {raw}\nGenre map: {genre_map}"
        )),
        *state["messages"]
    ]
    content = llm.invoke(profile_msgs).content
    return {"messages": [AIMessage(content=content, name="profiler")]}

matcher_node = make_simple_node(
    "You are a taste matcher. Score each item in the researcher's list (1-10) "
    "against the preference profile. Provide one-sentence reasoning per item. "
    "Return the top 5 ranked highest to lowest.",
    "matcher"
)

reviewer_node = make_simple_node(
    "You are a quality reviewer. Check the entertainment plan: "
    "are titles real? are platforms correct? does it match user preferences? "
    "is there variety? Apply corrections if needed. "
    "Finish with a 2-sentence 'WHY THIS PLAN WORKS FOR YOU' summary.",
    "reviewer"
)


def researcher_node(state: AgentState) -> dict:
    # Step 1: generate a focused search query
    q_msgs = [
        SystemMessage(content="Based on the conversation, write ONE web search query (max 10 words) to find great entertainment recommendations. Reply with just the query, nothing else."),
        *state["messages"]
    ]
    query = llm.invoke(q_msgs).content.strip().strip('"')

    # Step 2: perform search
    try:
        results = search_tool.run(query)[:2000]
    except Exception as e:
        results = f"Search unavailable: {e}"

    # Step 3: format results
    fmt_msgs = [
        SystemMessage(content="You are a content researcher. Format these search results into 6-8 entertainment options. For each: title, type, rating (if known), streaming platform, 1-sentence description."),
        *state["messages"],
        HumanMessage(content=f"Search query: '{query}'\n\nRaw results:\n{results}")
    ]
    content = llm.invoke(fmt_msgs).content
    return {"messages": [AIMessage(content=content, name="researcher")]}


def curator_node(state: AgentState) -> dict:
    # Check if the last message is human feedback injected via the HITL resume
    human_feedback = ""
    last_msg = state["messages"][-1] if state["messages"] else None
    if last_msg and isinstance(last_msg, HumanMessage):
        candidate = last_msg.content.strip().lower()
        if candidate not in ("", "approved", "approve", "yes", "ok", "looks good", "approved, looks great!"):
            human_feedback = last_msg.content.strip()

    feedback_context = (
        f"\n\nThe human reviewer said: '{human_feedback}' — incorporate this feedback."
        if human_feedback else ""
    )

    content = call_llm(
        "You are a diversity and serendipity curator. Review the top-5 ranked list: "
        "ensure 2+ content formats, max 2 items per exact genre. "
        f"Add one [SERENDIPITY PICK] slightly outside the user's usual taste.{feedback_context} "
        "Return the final 5-item curated list with the serendipity pick clearly labeled.",
        state
    )
    return {"messages": [AIMessage(content=content, name="curator")]}


def planner_node(state: AgentState) -> dict:
    # Call schedule tool
    try:
        schedule = schedule_planner.invoke({"hours_available": "3"})
    except Exception:
        schedule = '{"hours": 3, "slots": [{"slot": "Main Feature", "duration": "~2 hrs"}]}'

    plan_msgs = [
        SystemMessage(content=(
            f"You are an entertainment concierge. Use this time schedule: {schedule}\n\n"
            "Build a polished plan from the curated picks. For each item:\n"
            "Title | Type | Platform | Duration | Best Time Slot | Why it's perfect for you\n\n"
            "Highlight the serendipity pick at the end under '*** SURPRISE PICK ***'."
        )),
        *state["messages"]
    ]
    content = llm.invoke(plan_msgs).content
    return {"messages": [AIMessage(content=content, name="planner")]}


# ── Build the Graph ───────────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("supervisor",  supervisor_node)
    g.add_node("profiler",    profiler_node)
    g.add_node("researcher",  researcher_node)
    g.add_node("matcher",     matcher_node)
    g.add_node("curator",     curator_node)
    g.add_node("planner",     planner_node)
    g.add_node("reviewer",    reviewer_node)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        lambda s: s["next"],
        {
            "profiler":   "profiler",
            "researcher": "researcher",
            "matcher":    "matcher",
            "curator":    "curator",
            "planner":    "planner",
            "reviewer":   "reviewer",
            "FINISH":     END,
        },
    )
    for member in PIPELINE:
        g.add_edge(member, "supervisor")

    return g.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["curator"],  # HUMAN-IN-THE-LOOP pause before curator
    )


# ── CLI Entry ─────────────────────────────────────────────────────────────────
def run(user_request: str):
    graph  = build_graph()
    config = {"configurable": {"thread_id": "session-1"}}
    state  = {
        "messages":     [HumanMessage(content=user_request)],
        "user_request": user_request,
        "next":         "",
    }

    print("\n" + "="*60)
    print("Streaming agent pipeline...")
    print("="*60 + "\n")

    seen = set()

    def stream_events(input_state, cfg):
        for event in graph.stream(input_state, cfg, stream_mode="values"):
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
                    preview = m.content[:500] + ("..." if len(m.content) > 500 else "")
                    print(f"[{label.upper()}]:\n{preview}\n{'-'*50}\n")

    # Phase 1: run until HITL pause (before curator)
    stream_events(state, config)

    # Handle HITL — graph is paused at interrupt_before=['curator']
    snapshot = graph.get_state(config)
    if snapshot.next:
        # Show the matcher's top-5 so the user knows what they're approving
        for msg in reversed(snapshot.values.get("messages", [])):
            if getattr(msg, "name", None) == "matcher":
                print("\n" + "="*60)
                print("HUMAN-IN-THE-LOOP: Curator will work with this top-5 list.")
                print("Review it below, then approve or suggest changes.")
                print("="*60)
                print(msg.content[:900])
                break

        print("\n" + "="*60)
        print("Press Enter to approve, or type feedback for the curator:")
        print("="*60)
        human_input = input("> ").strip()

        # inject human input into the checkpoint, then resume with None
        graph.update_state(
            config,
            {"messages": [HumanMessage(content=human_input or "Approved")]}
        )
        stream_events(None, config)

    # Final output
    final_msgs = graph.get_state(config).values.get("messages", [])
    if final_msgs:
        final_plan = final_msgs[-1].content
        print("\n" + "="*60)
        print("FINAL ENTERTAINMENT PLAN")
        print("="*60)
        print(final_plan)
        return final_plan
    return "No output."


if __name__ == "__main__":
    print("=" * 60)
    print("  Entertainment Recommendation & Curation System")
    print(f"  Multi-Agent AI  |  LangGraph + Groq {LLM_MODEL}")
    print("=" * 60)
    print()
    request = input("What kind of entertainment are you in the mood for?\n> ")
    run(request)
