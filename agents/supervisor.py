"""
ResearchFlow — Supervisor Graph

Builds and returns the main LangGraph StateGraph that orchestrates
the Planner, Retriever, Analyst, Fact-Checker, and Critique nodes.
"""

from agents.state import ResearchState
from agents.retriever import retriever_node
from agents.analyst import analyst_node
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

# -----
# TEMP-DELETE LATER
# -----
def verify_bedrock_access():
    """Test that Bedrock is accessible with current credentials."""
    # Create Bedrock client
    bedrock = boto3.client(
        service_name='bedrock',
        region_name='us-east-1'  # Adjust to your region
    )
    
    # List available foundation models
    response = bedrock.list_foundation_models()
    
    print("Available Bedrock Models:")
    print("-" * 50)
    for model in response['modelSummaries']:
        print(f"  {model['modelId']}")
        print(f"    Provider: {model['providerName']}")
        print(f"    Input: {model['inputModalities']}")
        print(f"    Output: {model['outputModalities']}")
        print()

# -----
# TEMP- DELETE LATER
# -----
def test_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("planner_node", planner_node)
    graph.add_node("critique_node", critique_node)
    graph.add_edge(START, "planner_node")
    graph.add_conditional_edges("planner_node", router)
    graph = graph.compile()

    state = ResearchState(
        question="What is the capital of France?",
        plan=[],
        retrieved_chunks=[],
        analysis={},
        fact_check_report={},
        confidence_score=.90,
        iteration_count=0,
        scratchpad=[],
        user_id="1",
    )

    return graph, state
if __name__ == "__main__":
    load_dotenv()

    from agents.supervisor import build_supervisor_graph

    graph = build_supervisor_graph()
    config = {"configurable": {"thread_id": "demo-1"}}

    result = graph.invoke(
        {"question": "How is loss backpropagated through a feed-forward neural network?", "user_id": "ben"},
        config=config,
    )
    print("FINAL ANSWER:")
    print(result["analysis"]["answer"])
    print("\nCONFIDENCE:", result["confidence_score"])
    print("\nSCRATCHPAD:")
    for line in result["scratchpad"]:
        print(" ", line)