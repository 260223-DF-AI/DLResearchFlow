"""
ResearchFlow — Supervisor Graph

Builds and returns the main LangGraph StateGraph that orchestrates
the Planner, Retriever, Analyst, Fact-Checker, and Critique nodes.
"""

from agents.state import ResearchState
from agents.retriever import retriever_node
from agents.fact_checker import fact_checker_node
from agents.analyst import analyst_node
from agents.prompts import PLANNER_NODE_PROMPT, ROUTER_PROMPT, ROUTER_PROMPT_TEMP
from utils.utils import remove_reasoning

from langgraph.graph import StateGraph, START, END
from langchain_aws import ChatBedrock
from langchain_core.messages import SystemMessage, HumanMessage
import boto3
from dotenv import load_dotenv
import os
import re

HITL_THRESHOLD = 0.8
MAX_ITERATIONS = 3

load_dotenv()

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
            "temperature" : 0.1
        }
    )

    message = []
    question = state["question"]
    message.append(SystemMessage(content=PLANNER_NODE_PROMPT))
    message.append(HumanMessage(content=question))
    response = agent.invoke(message).content
    response = remove_reasoning(response)

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
    state["scratchpad"].append(f"Critique: {critique_result}")

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
    graph.add_node("planner_node", planner_node)
    graph.add_node("retriever_node", retriever_node)
    graph.add_node("analyst_node", analyst_node)
    graph.add_node("fact_checker_node", fact_checker_node)
    graph.add_node("critique_node", critique_node)

    # static edges
    graph.add_edge(START, "planner_node")

    # conditional edges
    graph.add_conditional_edges("planner_node", router)
    graph.add_conditional_edges("retriever_node", router)
    graph.add_conditional_edges("analyst_node", router)
    graph.add_conditional_edges("fact_checker_node", router)
    graph.add_conditional_edges("critique_node", router)
    
    # compile and return graph
    graph = graph.compile()
    return graph

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
    graph, state = test_graph()
    invoked = graph.invoke(state)
    print(invoked["scratchpad"])