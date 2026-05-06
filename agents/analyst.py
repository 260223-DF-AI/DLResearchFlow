"""
ResearchFlow — Analyst Agent

Synthesizes retrieved context into a structured, cited research
response using AWS Bedrock, with Pydantic-validated output.
"""

from pydantic import BaseModel
from dotenv import load_dotenv
import os
import json
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import ResearchState
from agents.prompts import ANALYST_NODE_PROMPT
from utils.utils import remove_reasoning

load_dotenv()

# ---------------------------------------------------------------------------
# Structured Output Schema
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    """A single supporting citation."""
    source: str
    page_number: int | None = None
    excerpt: str

class SingleClaim(BaseModel):
    """A single claim."""
    claim: str
    answer: str
    citations: list[Citation]
    confidence: float


class AnalysisResult(BaseModel):
    """Pydantic model enforcing structured analyst output."""
    overall_answer: str
    claims: list[SingleClaim]
    overall_confidence: float  # 0.0 – 1.0


# ---------------------------------------------------------------------------
# Agent Node
# ---------------------------------------------------------------------------

def analyst_node(state: ResearchState) -> dict:
    """
    Synthesize retrieved chunks into a structured research response.

    TODO:
    - Build a prompt from the question, sub-task, and retrieved_chunks.
    - Invoke AWS Bedrock (e.g., Claude) with structured output enforcement.
    - Parse the response into an AnalysisResult.
    - Support streaming for real-time feedback.
    - Log actions to the scratchpad.

    Returns:
        Updated state with "analysis" and "confidence_score" keys.
            - "analysis": Dict with "analysis" key containing the AnalysisResult as a dict,
            - "confidence_score": updated from the model's self-assessment, 0.0-1.0.
    """
    agent = ChatBedrock(
        model_id=os.getenv("BEDROCK_MODEL_ID"),
        region_name=os.getenv("AWS_REGION"),
        model_kwargs={
            "temperature": 0.1
        }
    )

    # give the model prompt & system state
    message = []
    message.append(SystemMessage(content=ANALYST_NODE_PROMPT))
    message.append(HumanMessage(content=str(state)))
    response = agent.invoke(message).content

    # remove reasoning
    response = remove_reasoning(response)
    
    # parse response
    total_confidence = 0.0
    claims = []
    data = json.loads(response)

    for claim in data["claims"]:
        total_confidence += claim["confidence"]
        claims.append(SingleClaim.model_validate(claim))
    
    report = AnalysisResult(
        overall_answer=data["overall_answer"],
        claims=claims,
        overall_confidence=total_confidence/len(claims)
    )


    state["analysis"] = report.model_dump()
    state["confidence_score"] = report.overall_confidence
    state["scratchpad"].append(f"Analysis: {state['analysis']}")

    return state

# -----
# TEMP- DELETE LATER
# -----
if __name__ == "__main__":
    state = ResearchState(
        question="Does the capital of France or the capital of Germany have more people?",
        plan=["Search for the capital of France", "Search for the population of the capital of France", "Search for the capital of Germany", "Search for the population of the capital of Germany", "Verify the information is up to date"],
        retrieved_chunks=["Paris is the capital of France.", "Paris has a population of 5 million people.", "Berlin is the capital of Germany.", "Berlin has a population of 3 million people."],
        analysis={},
        fact_check_report={},
        confidence_score=0,
        iteration_count=0,
        scratchpad=["Question: Does the capital of France or the capital of Germany have more people?", "Plan: 'Search for the capital of France', 'Search for the population of the capital of France', 'Search for the capital of Germany', 'Search for the population of the capital of Germany', 'Verify the information is up to date'", "Retrieved Chunks: 'Paris is the capital of France.', 'Paris has a population of 5 million people.', 'Berlin is the capital of Germany.', 'Berlin has a population of 3 million people.'"],
        user_id="1",
    )

    response = analyst_node(state)
    print(response["analysis"])