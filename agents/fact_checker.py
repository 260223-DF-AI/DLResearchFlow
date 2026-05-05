"""
ResearchFlow — Fact-Checker Agent

Cross-references the Analyst's claims against the fact-check
namespace in Pinecone and produces a verification report.
Triggers HITL interrupt when confidence is below threshold.
"""

from pydantic import BaseModel
from langchain_aws import ChatBedrock
import os
import json
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import ResearchState
from agents.prompts import FACT_CHECKER_PROMPT
from utils.utils import remove_reasoning

load_dotenv()

class ClaimVerdict(BaseModel):
    """Verification result for a single claim."""
    claim: str
    verdict: str  # "Supported" | "Unsupported" | "Inconclusive"
    evidence: str | None = None
    confidence: float


class FactCheckReport(BaseModel):
    """Full verification report across all claims."""
    verdicts: list[ClaimVerdict]
    overall_confidence: float


def fact_checker_node(state: ResearchState) -> dict:
    """
    Verify the Analyst's response against trusted reference sources.

    TODO:
    - Extract claims from state["analysis"].
    - Query the 'fact-check-sources' Pinecone namespace for each claim.
    - Produce per-claim verdicts.
    - If confidence < threshold, trigger HITL interrupt.
    - Support Time Travel via state checkpointing.
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
    message.append(SystemMessage(content=FACT_CHECKER_PROMPT))
    message.append(HumanMessage(content=str(state)))
    response = agent.invoke(message).content

    # remove reasoning
    response = remove_reasoning(response)

    # parse response for claims, build single FactCheckReport
    verdicts = []
    total_confidence = 0

    data = json.loads(response)
    for claim in data["claims"]:
        verdicts.append(ClaimVerdict.model_validate(claim))
        total_confidence += claim["confidence"]

    report = FactCheckReport(
        verdicts=verdicts,
        overall_confidence=total_confidence / len(verdicts)
    )

    state["fact_check_report"] = report.model_dump()
    state["confidence_score"] = report.overall_confidence
    state["scratchpad"].append(f"Fact Check Report: {report}")

    return state

# -----
# TEMP- DELETE LATER
# -----
if __name__ == "__main__":
    state = ResearchState(
        question="What is the capital of France?",
        plan=["Search for the capital of France", "Verify the information is up to date"],
        retrieved_chunks=["Paris is the capital of France."],
        analysis={'answer': 'Paris', 'citations': [{'source': 'chunk_0', 'page_number': None, \
                    'excerpt': 'Paris is the capital of France.'}], 'confidence': 0.95},
        fact_check_report={},
        confidence_score=0.95,
        iteration_count=0,
        scratchpad=["Question: What is the capital of France?", "Plan: Search for the capital of France, Verify the information is up to date", "Retrieved Chunks: Paris is the capital of France.", "Analysis: {'answer': 'Paris', 'citations': [{'source': 'chunk_0', 'page_number': None, 'excerpt': 'Paris is the capital of France.'}], 'confidence': 0.95}"],
        user_id="1",
    )

    response = fact_checker_node(state)
    print(response["fact_check_report"])