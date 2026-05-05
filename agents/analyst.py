"""
ResearchFlow — Analyst Agent

Synthesizes retrieved context into a structured, cited research
response using AWS Bedrock, with Pydantic-validated output.
"""

import os
from dotenv import load_dotenv

from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.state import ResearchState

load_dotenv()


# ---------------------------------------------------------------------------
# Structured Output Schema
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    """A single supporting citation."""
    source: str = Field(description="Source filename, e.g. 'Artemis_II.pdf'")
    page_number: int | None = Field(
        default=None,
        description="Page number within the source, if known",
    )
    excerpt: str = Field(default="", description="Short supporting excerpt from the source")


class AnalysisResult(BaseModel):
    """Pydantic model enforcing structured analyst output."""
    answer: str = Field(description="The synthesized answer to the user's question")
    citations: list[Citation] = Field(
        default_factory=list,
        description=(
            "A list of citation objects. Each object MUST have a 'source' string "
            "and an optional 'page_number' integer. Do NOT return a single string."
        ),
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Self-assessed confidence on a 0.0–1.0 scale",
    )


_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a research analyst. Synthesize a precise answer to the user's "
     "question using ONLY the numbered context chunks below. Every factual "
     "claim must cite at least one chunk by its source filename and page. "
     "If the context does not support an answer, say so explicitly and set "
     "confidence below 0.4.\n\n"
     "Self-assess your confidence on a 0.0–1.0 scale where:\n"
     "  • 0.9+ = direct quote answers the question\n"
     "  • 0.6–0.9 = answer is supported by the context but requires inference\n"
     "  • <0.6 = context is partial, conflicting, or off-topic\n\n"
     "Output schema: return JSON with 'answer' (string), 'citations' "
     "(a JSON array of objects, each with 'source' and 'page_number'), "
     "and 'confidence' (number 0.0–1.0). Never return citations as a single string."),
    ("human",
     "Question: {question}\n\n"
     "Sub-task: {sub_task}\n\n"
     "Context chunks:\n{context_block}"),
])


def _format_chunks(chunks: list[dict]) -> str:
    """Render retrieved chunks into a numbered, citeable block."""
    lines = []
    for i, c in enumerate(chunks, start=1):
        page = f", p.{c['page_number']}" if c.get("page_number") else ""
        lines.append(f"[{i}] (source: {c['source']}{page})\n{c['content']}")
    return "\n\n".join(lines)

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
        Dict with "analysis" key containing the AnalysisResult as a dict,
        and "confidence_score" updated from the model's self-assessment.
    """
    chunks = state.get("retrieved_chunks", [])
    log = [f"[analyst] synthesizing from {len(chunks)} chunks"]

    if not chunks:
        empty = AnalysisResult(
            answer="No relevant context was retrieved; cannot answer reliably.",
            citations=[],
            confidence=0.0,
        )
        return {
            "analysis": empty.model_dump(),
            "confidence_score": 0.0,
            "scratchpad": log + ["[analyst] short-circuit: no chunks"],
        }

    plan = state.get("plan", [])
    idx = state.get("current_subtask_index", 0)
    sub_task = plan[idx] if plan else state["question"]

    # ChatBedrock + structured output — the LLM is forced into AnalysisResult.
    llm = ChatBedrock(
        model_id=os.environ["BEDROCK_MODEL_ID"],
        region_name=os.environ["AWS_REGION"],
        model_kwargs={"max_tokens": 1024, "temperature": 0.2},
    )
    chain = _PROMPT | llm.with_structured_output(AnalysisResult)

    result: AnalysisResult = chain.invoke({
        "question": state["question"],
        "sub_task": sub_task,
        "context_block": _format_chunks(chunks),
    })
    log.append(f"[analyst] confidence={result.confidence:.2f}, "
               f"citations={len(result.citations)}")

    return {
        "analysis": result.model_dump(),
        "confidence_score": float(result.confidence),
        "scratchpad": log,
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    from agents.retriever import retriever_node
    
    state = {
        "question": "Question your corpus can actually answer.",
        "plan": ["Question your corpus can actually answer."],
        "current_subtask_index": 0,
    }
    state.update(retriever_node(state))
    state.update(analyst_node(state))
    import json
    print(json.dumps(state["analysis"], indent=2))
    print("Confidence:", state["confidence_score"])