"""
ResearchFlow — Fact-Checker Agent

Cross-references the Analyst's claims against the fact-check
namespace in Pinecone and produces a verification report.
Triggers HITL interrupt when confidence is below threshold.
"""
import os
import re

from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone
from pydantic import BaseModel, Field

from agents.state import ResearchState


class ClaimVerdict(BaseModel):
    """Verification result for a single claim."""
    claim: str
    verdict: str = Field(pattern=r"^(Supported|Unsupported|Inconclusive)$")
    evidence: str | None = None


class FactCheckReport(BaseModel):
    """Full verification report across all claims."""
    verdicts: list[ClaimVerdict] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0)


_VERDICT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict fact-checker. Given a claim and supporting evidence, "
     "decide one of: Supported, Unsupported, Inconclusive.\n"
     "  • Supported = the evidence directly states or strongly implies the claim.\n"
     "  • Unsupported = the evidence contradicts the claim.\n"
     "  • Inconclusive = the evidence is silent on the claim.\n"
     "Quote a short snippet from the evidence as your justification.\n\n"
     "Output schema: return JSON with 'verdict' (one of the three labels above, "
     "exactly as spelled) and 'evidence' (a short string snippet from the input)."),
    ("human",
     "Claim: {claim}\n\nEvidence:\n{evidence}"),
])


class SingleVerdict(BaseModel):
    """Schema the verdict-LLM is forced into."""
    verdict: str = Field(
        pattern=r"^(Supported|Unsupported|Inconclusive)$",
        description="Exactly one of: Supported, Unsupported, Inconclusive",
    )
    evidence: str = Field(
        description="A short quoted snippet from the evidence justifying the verdict",
    )


_embedder = None
_pinecone_index = None
_verdict_llm = None


def _lazy_init():
    """Module-level singletons; lets unit tests monkeypatch."""
    global _embedder, _pinecone_index, _verdict_llm
    if _embedder is None:
        _embedder = BedrockEmbeddings(
            model_id=os.environ.get("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"),
            region_name=os.environ["AWS_REGION"],
        )
    if _pinecone_index is None:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        _pinecone_index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
    if _verdict_llm is None:
        _verdict_llm = ChatBedrock(
            model_id=os.environ.get(
                "FACT_CHECK_MODEL_ID",
                "anthropic.claude-3-haiku-20240307-v1:0",
            ),
            region_name=os.environ["AWS_REGION"],
            model_kwargs={"max_tokens": 256, "temperature": 0.0},
        )


def _split_into_claims(answer: str) -> list[str]:
    """Heuristic claim extraction — split on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    return [s for s in sentences if len(s) > 20]


def _verify_claim(claim: str) -> ClaimVerdict:
    query_vec = _embedder.embed_query(claim)
    raw = _pinecone_index.query(
        vector=query_vec, top_k=3,
        namespace="fact-check-sources",
        include_metadata=True,
    )
    matches = raw.get("matches", []) if isinstance(raw, dict) else raw["matches"]
    if not matches:
        return ClaimVerdict(claim=claim, verdict="Inconclusive",
                            evidence="No supporting documents found.")

    evidence_block = "\n\n---\n\n".join(
        m["metadata"].get("content", "") for m in matches
    )
    chain = _VERDICT_PROMPT | _verdict_llm.with_structured_output(SingleVerdict)
    out: SingleVerdict = chain.invoke({"claim": claim, "evidence": evidence_block})
    return ClaimVerdict(claim=claim, verdict=out.verdict, evidence=out.evidence)

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
    _lazy_init()
    log = ["[fact_checker] starting verification"]

    analysis = state.get("analysis") or {}
    answer = analysis.get("answer", "")
    claims = _split_into_claims(answer)
    log.append(f"[fact_checker] extracted {len(claims)} claims")

    if not claims:
        report = FactCheckReport(verdicts=[], overall_confidence=0.0)
        return {
            "fact_check_report": report.model_dump(),
            "confidence_score": 0.0,
            "needs_hitl": True,
            "scratchpad": log + ["[fact_checker] no claims, escalating to HITL"],
        }

    verdicts = [_verify_claim(c) for c in claims]
    counts = {"Supported": 0, "Unsupported": 0, "Inconclusive": 0}
    for v in verdicts:
        counts[v.verdict] = counts.get(v.verdict, 0) + 1

    # Confidence = (supported - unsupported) / total, clamped to [0, 1].
    total = max(len(verdicts), 1)
    raw = (counts["Supported"] - counts["Unsupported"]) / total
    overall = max(0.0, min(1.0, raw))

    threshold = float(os.environ.get("HITL_CONFIDENCE_THRESHOLD", 0.6))
    needs_hitl = overall < threshold or counts["Unsupported"] > 0

    report = FactCheckReport(verdicts=verdicts, overall_confidence=overall)
    log.append(
        f"[fact_checker] supported={counts['Supported']}, "
        f"unsupported={counts['Unsupported']}, inconclusive={counts['Inconclusive']}, "
        f"overall={overall:.2f}, hitl={needs_hitl}"
    )

    return {
        "fact_check_report": report.model_dump(),
        "confidence_score": overall,
        "needs_hitl": needs_hitl,
        "scratchpad": log,
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()                                    # must run BEFORE agent imports

    from agents.retriever import retriever_node
    from agents.analyst import analyst_node
    
    state = {
        "question": "How is loss backpropagated through a feed-forward neural network?",
        "plan": ["How is loss backpropagated through a feed-forward neural network?"],
        "current_subtask_index": 0,
    }
    state.update(retriever_node(state))
    state.update(analyst_node(state))
    state.update(fact_checker_node(state))

    print("Overall confidence:", state["confidence_score"])
    print("Needs HITL:", state["needs_hitl"])
    for v in state["fact_check_report"]["verdicts"]:
        print(f"  [{v['verdict']}] {v['claim'][:60]}...")