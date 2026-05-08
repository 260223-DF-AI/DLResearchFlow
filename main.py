"""
ResearchFlow — Main Entry Point

Parses CLI arguments and invokes the Supervisor graph to answer
a research question against the ingested document corpus.
"""

import argparse
import uuid

from dotenv import load_dotenv

from langgraph.errors import GraphInterrupt

from agents.supervisor import build_supervisor_graph
from middleware.guardrails import detect_injection, sanitize_input
from middleware.pii_masking import mask_pii


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ResearchFlow: Adaptive Multi-Agent Research Assistant"
    )
    parser.add_argument(
        "--user-id",
        "-U",
        type=str,
        default="anonymous",
        help="User ID for cross-thread memory (Store interface).",
    )
    parser.add_argument(
        "--verbose",
        "-V",
        action="store_true",
        help="Enable step-wise scratchpad logging.",
    )
    return parser.parse_args()


def main() -> None:
    """
    High-level flow:
    1. Load environment variables.
    2. Initialize the Supervisor graph (see agents/supervisor.py).
    3. Repeatedly prompt for a question and invoke the graph.
    4. Print the structured research report for each turn.
    """

    load_dotenv()
    args = parse_args()

    # --- graph + addressable thread -----------------------------------------
    graph = build_supervisor_graph()
    thread_id = f"cli-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}
    print("ResearchFlow interactive mode. Type 'exit' or press Ctrl+C to quit.")

    while True:
        try:
            raw_question = input("\nQuestion> ").strip()
        except EOFError:
            print("\nExiting.")
            break
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        if not raw_question:
            continue
        if raw_question.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        # --- input boundary --------------------------------------------------
        if detect_injection(raw_question):
            print("Input rejected: possible prompt injection.")
            continue
        question = mask_pii(sanitize_input(raw_question))

        initial_state = {
            "question": question,
            "user_id": args.user_id,
        }

        # --- invoke, handling HITL interrupts -------------------------------
        try:
            result = graph.invoke(initial_state, config=config)
        except GraphInterrupt as interrupt:
            # NodeInterrupt percolates up wrapped in GraphInterrupt.
            print("\n=== HUMAN-IN-THE-LOOP REVIEW REQUIRED ===")
            print(f"Reason: {interrupt}")
            # Show the reviewer the current state so they can decide.
            snapshot = graph.get_state(config)
            analysis = snapshot.values.get("analysis", {})
            print("\nDraft answer:\n", analysis.get("answer", "<empty>"))
            decision = input("\nApprove answer as-is? [y/n]: ").strip().lower()
            if decision != "y":
                print("Rejected by reviewer. Aborting.")
                continue
            # Override the state to mark approved, then resume.
            graph.update_state(config, {"needs_hitl": False, "confidence_score": 1.0})
            result = graph.invoke(None, config=config)

        # --- output boundary -------------------------------------------------
        analysis = result.get("analysis", {})
        safe_answer = mask_pii(analysis.get("answer", ""))

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(safe_answer)
        print("\nCITATIONS")
        for c in analysis.get("citations", []):
            page = f", p.{c['page_number']}" if c.get("page_number") else ""
            print(f"  • {c['source']}{page}: {c.get('excerpt','')[:120]}")
        print(f"\nCONFIDENCE: {result.get('confidence_score', 0.0):.2f}")
        print(f"ITERATIONS: {result.get('iteration_count', 0)}")

        if args.verbose:
            print("\nSCRATCHPAD")
            for line in result.get("scratchpad", []):
                print(" ", line)

        if result.get("fact_check_report"):
            print("\nFACT-CHECK REPORT")
            for v in result["fact_check_report"]["verdicts"]:
                print(f"  [{v['verdict']}] {v['claim'][:80]}")

if __name__ == "__main__":
    main()
