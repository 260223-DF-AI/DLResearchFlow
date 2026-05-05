"""
ResearchFlow — Fact-Checker Agent

Cross-references the Analyst's claims against the fact-check
namespace in Pinecone and produces a verification report.
Triggers HITL interrupt when confidence is below threshold.
"""

from pydantic import BaseModel
from langchain_aws import ChatBedrock, BedrockEmbeddings
import os
import json
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from pinecone import Pinecone

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

def _query_fact_check_namespace(claims: list[str], top_k: int = 3) -> list[str]:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    embedder = BedrockEmbeddings(
        model_id=os.getenv("BEDROCK_EMBEDDING_MODEL_ID"),
        region_name=os.getenv("AWS_REGION")
    )

    seen_ids: set[str] = set()
    chunks: list[str] = []

    for claim in claims:
        vector = embedder.embed_query(claim)
        results = index.query(
            vector=vector,
            top_k=top_k,
            namespace="fact-check-sources",
            include_metadata=True,
        )
        for match in results["matches"]:
            if match["id"] not in seen_ids:
                seen_ids.add(match["id"])
                chunks.append(match["metadata"].get("text", ""))

    return chunks

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

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    analysis = state.get("analysis", {})
    claim_texts: list[str] = [c["claim"] for c in analysis.get("claims", [])]

    fact_check_chunks = _query_fact_check_namespace(claim_texts)
    augmented_state = dict(state)
    augmented_state["fact_check_chunks"] = fact_check_chunks

    # give the model prompt & system state
    message = []
    message.append(SystemMessage(content=FACT_CHECKER_PROMPT))
    message.append(HumanMessage(content=str(augmented_state))) 

    response = agent.invoke(message).content

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
        overall_confidence=total_confidence / len(verdicts) if len(verdicts)> 0 else 1
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
        question="Does the capital of France or the capital of Germany have more people?",
        plan=["Search for the capital of France", "Search for the population of the capital of France", "Search for the capital of Germany", "Search for the population of the capital of Germany", "Verify the information is up to date"],
        retrieved_chunks=["Paris is the capital of France.", "Paris has a population of 5 million people.", "Berlin is the capital of Germany.", "Berlin has a population of 3 million people."],
        analysis={},
        fact_check_report={},
        confidence_score=0,
        iteration_count=0,
        scratchpad=['Question: Does the capital of France or the capital of Germany have more people?', "Plan: 'Search for the capital of France', 'Search for the population of the capital of France', 'Search for the capital of Germany', 'Search for the population of the capital of Germany', 'Verify the information is up to date'", "Retrieved Chunks: 'Paris is the capital of France.', 'Paris has a population of 5 million people.', 'Berlin is the capital of Germany.', 'Berlin has a population of 3 million people.'", "Analysis: {'overall_answer': 'Paris, the capital of France, has a larger population than Berlin, the capital of Germany.', 'claims': [{'claim': 'Paris has more people than Berlin.', 'answer': 'Paris has a population of 5 million people, while Berlin has a population of 3 million people, so Paris has more people.', 'citations': [{'source': 'Paris is the capital of France.', 'page_number': 1, 'excerpt': 'Paris is the capital of France.'}, {'source': 'Paris has a population of 5 million people.', 'page_number': 1, 'excerpt': 'Paris has a population of 5 million people.'}, {'source': 'Berlin is the capital of Germany.', 'page_number': 1, 'excerpt': 'Berlin is the capital of Germany.'}, {'source': 'Berlin has a population of 3 million people.', 'page_number': 1, 'excerpt': 'Berlin has a population of 3 million people.'}], 'confidence': 0.9}, {'claim': 'The population of Paris is 5 million and the population of Berlin is 3 million.', 'answer': 'Paris has 5 million residents and Berlin has 3 million residents.', 'citations': [{'source': 'Paris has a population of 5 million people.', 'page_number': 1, 'excerpt': 'Paris has a population of 5 million people.'}, {'source': 'Berlin has a population of 3 million people.', 'page_number': 1, 'excerpt': 'Berlin has a population of 3 million people.'}], 'confidence': 0.9}], 'overall_confidence': 0.9}"],
        user_id="1"
    )

    response = fact_checker_node(state)
    print(response["fact_check_report"])