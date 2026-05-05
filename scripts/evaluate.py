"""
ResearchFlow — RAGAS Evaluation Pipeline

Loads a golden dataset and runs a formal RAGAS evaluation measuring
faithfulness, answer relevancy, and context precision.

Usage:
    python scripts/evaluate.py --golden-dataset ./data/golden_dataset.json
"""

import os
import argparse
import json

from dotenv import load_dotenv

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

from langchain_aws import ChatBedrock

from agents.supervisor import build_supervisor_graph

load_dotenv()


def parse_args() -> argparse.Namespace:
    """Parse evaluation CLI arguments."""
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation.")
    parser.add_argument(
        "--golden-dataset",
        type=str,
        required=True,
        help="Path to the golden dataset JSON file.",
    )
    return parser.parse_args()


def load_golden_dataset(filepath: str) -> list[dict]:
    """
    Load the golden dataset from a JSON file.

    Expected format: see data/golden_dataset.json for the schema.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_predictions(dataset: list[dict]) -> list[dict]:
    """
    Run each question through the ResearchFlow pipeline and collect predictions.

    TODO:
    - For each entry in the dataset, invoke the Supervisor graph.
    - Capture the generated answer and the retrieved contexts.
    - Return a list of dicts with keys: question, answer, contexts.
    """
    graph = build_supervisor_graph()
    out = []
    for i, entry in enumerate(dataset):
        config = {"configurable": {"thread_id": f"eval-{i}"}}
        try:
            result = graph.invoke(
                {"question": entry["question"], "user_id": "evaluator"},
                config=config,
            )
        except Exception as e:
            print(f"  [warn] entry {i} failed: {e}")
            out.append({"question": entry["question"], "answer": "", "contexts": []})
            continue
        analysis = result.get("analysis", {}) or {}
        contexts = [c["content"] for c in result.get("retrieved_chunks", [])]
        out.append({
            "question": entry["question"],
            "answer": analysis.get("answer", ""),
            "contexts": contexts,
            "ground_truth": entry["ground_truth_answer"],
        })
        print(f"  [{i+1}/{len(dataset)}] done")
    return out


def run_ragas_evaluation(predictions: list[dict], golden: list[dict]) -> dict:
    """
    Evaluate predictions against the golden dataset using RAGAS.

    TODO:
    - Construct a RAGAS Dataset from predictions and ground truth.
    - Evaluate with metrics: faithfulness, answer_relevancy, context_precision.
    - Return a dict of metric_name → score.
    """

    ds = Dataset.from_list(predictions)
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    # `result` is a RAGASResult; `.to_pandas()` gives per-row, but we want
    # the aggregate summary as a flat dict.
    return {k: float(v) for k, v in result._scores_dict.items()}


def main() -> None:
    """Orchestrate the evaluation pipeline."""
    load_dotenv()
    args = parse_args()

    golden = load_golden_dataset(args.golden_dataset)
    predictions = generate_predictions(golden)
    results = run_ragas_evaluation(predictions, golden)

    print("\n📊 RAGAS Evaluation Results:")
    print("-" * 40)
    for metric, score in results.items():
        print(f"  {metric:<25} {score:.4f}")
    print("-" * 40)


if __name__ == "__main__":
    main()
