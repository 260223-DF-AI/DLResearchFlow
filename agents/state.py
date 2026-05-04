"""
ResearchFlow — Graph State Definition

Defines the TypedDict that flows through the Supervisor StateGraph.
All nodes read from and write to this shared state.
"""

from typing import Annotated, TypedDict
from operator import add


class ResearchState(TypedDict, total=False):
    """Shared state for the Supervisor graph.

    `total=False` lets nodes return partial dicts without LangGraph
    complaining about missing keys.
    """
    # --- inputs ---
    question: str
    user_id: str

    # --- planner output ---
    plan: list[str]
    current_subtask_index: int

    # --- retrieval / analysis / verification ---
    retrieved_chunks: list[dict]
    analysis: dict                  # serialized AnalysisResult
    fact_check_report: dict         # serialized FactCheckReport

    # --- loop control ---
    confidence_score: float
    iteration_count: int
    needs_hitl: bool                # True → critique routes to HITL interrupt

    # --- observability — uses the `add` reducer so each node APPENDS ---
    scratchpad: Annotated[list[str], add]