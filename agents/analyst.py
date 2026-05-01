"""
ResearchFlow — Analyst Agent

Synthesizes retrieved context into a structured, cited research
response using AWS Bedrock, with Pydantic-validated output.
"""

from pydantic import BaseModel
from dotenv import load_dotenv
import os
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage

from agents.state import ResearchState
from agents.prompts import ANALYST_NODE_PROMPT
from agents.supervisor import remove_reasoning

load_dotenv()

# ---------------------------------------------------------------------------
# Structured Output Schema
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    """A single supporting citation."""
    source: str
    page_number: int | None = None
    excerpt: str


class AnalysisResult(BaseModel):
    """Pydantic model enforcing structured analyst output."""
    answer: str
    citations: list[Citation]
    confidence: float  # 0.0 – 1.0


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

    message = []
    message.append(SystemMessage(content=ANALYST_NODE_PROMPT))
    message.append(HumanMessage(content=str(state)))
    response = agent.invoke(message).content
    response = remove_reasoning(response)
    response = AnalysisResult(**response)

    state["analysis"] = response
    state["confidence_score"] = response.confidence
    state["scratchpad"].append(f"Analysis: {response}") 

    return state

# -----
# TEMP- DELETE LATER
# -----