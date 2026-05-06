"""
ResearchFlow — Supervisor Graph

Builds and returns the main LangGraph StateGraph that orchestrates
the Planner, Retriever, Analyst, Fact-Checker, and Critique nodes.
"""

import os
from dotenv import load_dotenv
import re
import ast

import boto3

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_aws import ChatBedrock
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from agents.state import ResearchState
from agents.retriever import retriever_node
from agents.fact_checker import fact_checker_node
from agents.analyst import analyst_node
from agents.prompts import PLANNER_NODE_PROMPT, ROUTER_PROMPT
from memory.store import (
    get_user_preferences,
    get_query_history,
    append_query,
)
from utils.utils import remove_reasoning

load_dotenv()

HITL_THRESHOLD = 0.8
MAX_ITERATIONS = 3


def planner_node(state: ResearchState) -> dict:
    """
    Decompose the user's question into actionable sub-tasks.

    TODO:
    - Use Bedrock LLM to analyze the question.
    - Return a list of sub-tasks (Plan-and-Execute pattern).
    - Write to the scratchpad for observability.
    """
    agent = ChatBedrock(
        model_id = os.getenv("BEDROCK_MODEL_ID"),
        region_name = os.getenv("AWS_REGION"),
        model_kwargs={
            "max_tokens": 512,
            "temperature" : 0.1
        }
    )

    user_id = state["user_id"]
    prefs = get_user_preferences(user_id)
    history = get_query_history(user_id)
    append_query(user_id, state["question"])


    message = []
    question = state["question"]
    message.append(SystemMessage(content=PLANNER_NODE_PROMPT))
    message.append(SystemMessage(content=f"User Preferences: {prefs}"))
    message.append(SystemMessage(content=f"Recent Query History: {history}"))
    message.append(HumanMessage(content=question))
    response = agent.invoke(message).content
    response = remove_reasoning(response)

    try:
        parsed_response = ast.literal_eval(response)
        if isinstance(parsed_response, list) and all(isinstance(item, str) for item in parsed_response):
            response = parsed_response
        else:
            response = [str(parsed_response)]
    except (ValueError, SyntaxError):
        response = [response]

    

    state["scratchpad"].append(f"Question: {question}")
    state["scratchpad"].append(f"Plan: {response}")

    state["plan"] = response
    
    return state

def router(state: ResearchState) -> str:
    """
    Conditional edge: decide which agent to invoke next.

    TODO:
    - Inspect the current plan and state to choose the next node.
    - Return the node name as a string (used by add_conditional_edges).
    """
    agent = ChatBedrock(
        model_id = os.getenv("BEDROCK_MODEL_ID"),
        region_name = os.getenv("AWS_REGION"),
        model_kwargs={
            "temperature" : 0.1
        }
    )
    message = []
    message.append(SystemMessage(content=ROUTER_PROMPT))
    message.append(HumanMessage(content=str(state)))
    response = agent.invoke(message).content
    response = remove_reasoning(response)

    return response


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
    confidence = state["confidence_score"]
    iterations = state["iteration_count"]
    critique_result = ""
    
    if confidence < HITL_THRESHOLD and iterations < MAX_ITERATIONS:
        critique_result = "retry"
    elif confidence < HITL_THRESHOLD and iterations >= MAX_ITERATIONS:
        critique_result = "escalate to human intervention"
    else:
        critique_result = "accept current response"

    state["iteration_count"] += 1
    print(f"Iteration {state['iteration_count']}: Critique Result - {critique_result} (Confidence: {confidence:.2f})")
    state["scratchpad"].append(f"Critique: {critique_result}")
    return state

def _critique_router(state: ResearchState) -> str:
    """Edge after critique_node — END if accepted, else loop."""
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

    # add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("fact_checker", fact_checker_node)
    graph.add_node("critique", critique_node)

    # static edges
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
    
    # compile and return graph
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
    '''
    graph = StateGraph(ResearchState)
    graph.add_node("planner_node", planner_node)
    graph.add_node("critique_node", critique_node)
    graph.add_edge(START, "planner_node")
    graph.add_conditional_edges("planner_node", router)
    graph = graph.compile(checkpointer=MemorySaver())
    '''
    from agents.supervisor import build_supervisor_graph
    graph = build_supervisor_graph()

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
    # invoked = graph.invoke(state)
    # print(invoked["scratchpad"])
    load_dotenv()

    graph, state = test_graph()

    config = {"configurable": {"thread_id": "demo-1"}}

    result = graph.invoke(state, config=config)
    print("FINAL ANSWER:")
    print(result["analysis"]["overall_answer"])
    print("\nCONFIDENCE:", result["confidence_score"])
    print("\nSCRATCHPAD:")
    for line in result["scratchpad"]:
        print(" ", line)