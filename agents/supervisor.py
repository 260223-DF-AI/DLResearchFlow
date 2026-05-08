"""
ResearchFlow — Supervisor Graph

Builds and returns the main LangGraph StateGraph that orchestrates
the Planner, Retriever, Analyst, Fact-Checker, and Critique nodes.
"""

from agents.state import ResearchState
from agents.retriever import retriever_node
from agents.analyst import analyst_node, analyst_node_stream
from agents.fact_checker import fact_checker_node
from memory.store import (
    get_user_preferences,
    get_query_history,
    append_query
)
from agents.prompts import PLANNER_NODE_PROMPT, ROUTER_PROMPT, ROUTER_PROMPT_TEMP

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import NodeInterrupt
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

import boto3
from dotenv import load_dotenv
import os
import re


class Plan(BaseModel):
    """Structured output schema for the planner."""
    subtasks: list[str] = Field(
        description=(
            "An ordered JSON array of sub-task strings (1–4 entries). "
            "Each element MUST be a string. Do NOT return a single concatenated string."
        ),
    )


PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You decompose research questions into 1–4 ordered, independently-"
     "answerable sub-tasks. Prefer fewer, larger sub-tasks over many tiny "
     "ones. Each sub-task should be answerable from a single retrieval."
     "Sub-tasks should use specific phrases to prevent generalizing the question.\n\n"
     "Output schema: return JSON with a single key 'subtasks' whose value is "
     "a JSON array of strings. Never return a single concatenated string."),
    ("human",
     "User preferences: {preferences}\n"
     "Recent past questions from this user: {history}\n\n"
     "New question: {question}\n\n"
     "Return the sub-tasks as a JSON list of strings."),
])


HITL_THRESHOLD = 0.8
MAX_ITERATIONS = 3

load_dotenv()

def remove_reasoning(text: str) -> str:
    return re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL).strip()

def planner_node(state: ResearchState) -> dict:
    """
    Decompose the user's question into actionable sub-tasks.

    TODO:
    - Use Bedrock LLM to analyze the question.
    - Return a list of sub-tasks (Plan-and-Execute pattern).
    - Write to the scratchpad for observability.
    """
    user_id = state.get("user_id", "default")
    prefs = get_user_preferences(user_id)
    history = get_query_history(user_id, limit=3)
    append_query(user_id, state["question"])

    llm = ChatBedrock(
        model_id=os.environ["BEDROCK_MODEL_ID"],
        region_name=os.environ["AWS_REGION"],
        model_kwargs={"max_tokens": 512, "temperature": 0.0},
    )
    chain = PLANNER_PROMPT | llm.with_structured_output(Plan)
    plan = chain.invoke({
        "question": state["question"],
        "preferences": prefs,
        "history": history or ["<none>"],
    })

    return {
        "plan": plan.subtasks,
        "current_subtask_index": 0,
        "iteration_count": 0,
        "needs_hitl": False,
        "scratchpad": [f"[planner] decomposed into {len(plan.subtasks)} sub-tasks"],
    }

def router(state: ResearchState) -> str:
    """
    Conditional edge: decide which agent to invoke next.

    TODO:
    - Inspect the current plan and state to choose the next node.
    - Return the node name as a string (used by add_conditional_edges).
    """
    if not state.get("retrieved_chunks"):
        return "retriever"
    if not state.get("analysis"):
        return "analyst"
    if not state.get("fact_check_report"):
        return "fact_checker"
    return "critique"


def critique_node(state: ResearchState) -> dict:
    """
    Evaluate the aggregated response and decide: accept, retry, or escalate.

    TODO:
    - Check confidence_score against the HITL threshold.
    - If below threshold and iterations < max, loop back for refinement.
    - If below threshold and iterations >= max, trigger HITL interrupt.
    - If above threshold, accept and route to END.
    - Increment iteration_count.
    """
    iteration = state.get("iteration_count", 0) + 1
    confidence = state.get("confidence_score", 0.0)
    threshold = float(os.environ.get("HITL_CONFIDENCE_THRESHOLD", 0.6))
    max_iter = int(os.environ.get("MAX_REFINEMENT_ITERATIONS", 3))

    log = [f"[critique] iter={iteration}, conf={confidence:.2f}, "
           f"threshold={threshold}, max_iter={max_iter}"]

    plan = state.get("plan", [])
    idx = state.get("current_subtask_index", 0)

    # Path 1: confident enough → accept.
    if confidence >= threshold and not state.get("needs_hitl"):
        # If there are more subtasks, advance the pointer and reset
        # downstream artifacts so the graph executes the next subtask.
        if idx + 1 < len(plan):
            next_idx = idx + 1
            log.append(f"[critique] accepted sub-task {idx + 1}/{len(plan)}")
            log.append(f"[critique] advancing to sub-task {next_idx + 1}/{len(plan)}")
            return {
                "current_subtask_index": next_idx,
                "iteration_count": 0,
                "confidence_score": 0.0,
                "needs_hitl": False,
                "retrieved_chunks": [],
                "analysis": {},
                "fact_check_report": {},
                "scratchpad": log,
            }

        log.append("[critique] accepted final sub-task; finishing")
        return {"iteration_count": iteration, "scratchpad": log}

    # Path 2: budget exhausted → escalate.
    if iteration >= max_iter:
        log.append("[critique] max iterations reached — escalating to HITL")
        # NodeInterrupt pauses the graph; resume by graph.update_state(...).
        raise NodeInterrupt(
            f"Confidence {confidence:.2f} below threshold {threshold} "
            f"after {iteration} iterations. Human review required."
        )

    # Path 3: retry — clear downstream state so the router re-runs them.
    log.append("[critique] retrying — clearing analysis & fact_check")
    return {
        "iteration_count": iteration,
        "retrieved_chunks": [],          # forces retriever to re-fetch
        "analysis": {},                  # forces analyst to re-synthesize
        "fact_check_report": {},
        "scratchpad": log,
    }


def _critique_router(state: ResearchState) -> str:
    """Edge after critique_node — END if accepted/final, else continue loop."""
    confidence = state.get("confidence_score", 0.0)
    threshold = float(os.environ.get("HITL_CONFIDENCE_THRESHOLD", 0.6))
    if confidence >= threshold and not state.get("needs_hitl"):
        return END
    return "retriever"

def build_supervisor_graph():
    """
    Construct and compile the Supervisor StateGraph.

    TODO:
    - Instantiate StateGraph with ResearchState.
    - Add nodes: planner, retriever, analyst, fact_checker, critique.
    - Add edges and conditional edges (router).
    - Set entry point to planner.
    - Compile and return the graph.

    Returns:
        A compiled LangGraph that can be invoked with an initial state.
    """
    # instantiate stategraph
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("fact_checker", fact_checker_node)
    graph.add_node("critique", critique_node)

    graph.add_edge(START, "planner")
    # After planner, the router picks whichever specialist is needed first
    # (almost always the retriever). Listed below so LangGraph knows the
    # full set of valid destinations.
    graph.add_conditional_edges(
        "planner", router,
        {"retriever": "retriever", "analyst": "analyst",
         "fact_checker": "fact_checker", "critique": "critique"},
    )
    # Linear within a sub-task: retriever → analyst → fact_checker → critique
    graph.add_edge("retriever", "analyst")
    graph.add_edge("analyst", "fact_checker")
    graph.add_edge("fact_checker", "critique")

    # Critique decides whether to loop or end.
    graph.add_conditional_edges(
        "critique", _critique_router,
        {"retriever": "retriever", END: END},
    )

    # MemorySaver = in-process checkpointer; switch to a persistent saver
    # (PostgresSaver / DynamoDBSaver) when deploying to Lambda.
    return graph.compile(checkpointer=MemorySaver())

async def run_streaming(question: str, user_id: str = "default", thread_id: str = "stream-1"):
    graph = build_supervisor_graph()
    config = {"configurable": {"thread_id": thread_id}}

    print("Streaming response:\n")
    
    final_analysis = {}
    final_confidence = 0.0
    final_iterations = 0
    final_fact_report = {}

    async for chunk in graph.astream(
        {"question": question, "user_id": user_id},
        config=config,
        stream_mode="updates",
    ):
        for node_name, state_update in chunk.items():
            if node_name == "planner":
                plan = state_update.get("plan", [])
                print(f"[planner] decomposed into {len(plan)} sub-tasks:")
                for i, t in enumerate(plan, 1):
                    print(f"  {i}. {t}")
                print()

            elif node_name == "retriever":
                n = len(state_update.get("retrieved_chunks", []))
                print(f"[retriever] fetched {n} chunks\n")

            elif node_name == "analyst":
                analysis = state_update.get("analysis", {})
                if analysis:
                    print(f"[analyst] confidence={state_update.get('confidence_score', 0):.2f}")
                    print(f"  answer: {analysis.get('answer', '')}\n")
                    final_analysis = analysis
                    final_confidence = state_update.get("confidence_score", 0.0)

            elif node_name == "fact_checker":
                report = state_update.get("fact_check_report", {})
                if report:
                    for v in report.get("verdicts", []):
                        print(f"[fact_checker] [{v['verdict']}] {v['claim'][:80]}")
                    print()
                    # ADDED: keep the most recent fact check report
                    final_fact_report = report

            elif node_name == "critique":
                conf = state_update.get("confidence_score")
                itr = state_update.get("iteration_count")
                if conf is not None:
                    print(f"[critique] confidence={conf:.2f}, iteration={itr}")
                # ADDED: track final iteration count
                if itr is not None:
                    final_iterations = itr
    print("\n[stream complete]")

    # ADDED: final formatted report, mirrors main.py's output block
    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(final_analysis.get("answer", "<no answer>"))
    print("\nCITATIONS")
    for c in final_analysis.get("citations", []):
        page = f", p.{c['page_number']}" if c.get("page_number") else ""
        print(f"  • {c['source']}{page}: {c.get('excerpt', '')[:120]}")
    print(f"\nCONFIDENCE: {final_confidence:.2f}")
    print(f"ITERATIONS: {final_iterations}")
    if final_fact_report:
        print("\nFACT-CHECK REPORT")
        for v in final_fact_report.get("verdicts", []):
            print(f"  [{v['verdict']}] {v['claim'][:80]}")


