"""
Step 5: Human vs LLM Evaluation
Three-way comparison: individual humans, human consensus, and LLM.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
from pathlib import Path
import json

from utils import compute_consensus, CATEGORIES, ANNOTATOR_COLS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


def main():
    all_path = DATA_DIR / "annotations_all.csv"
    if not all_path.exists():
        print("ERROR: annotations_all.csv not found. Run steps 1-3 first.")
        return

    df = pd.read_csv(all_path)
    print(f"Loaded {len(df)} items")

    has_llm = "llm_annotator" in df.columns and df["llm_annotator"].notna().all()
    if not has_llm:
        print("WARNING: LLM annotations missing. Running evaluation without LLM comparison.")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Compute consensus
    df["consensus"] = compute_consensus(df)

    # --- Accuracy comparison ---
    print("\n" + "=" * 60)
    print("ACCURACY vs GROUND TRUTH")
    print("=" * 60)
    accuracy_results = {}
    for col in ANNOTATOR_COLS:
        acc = (df[col] == df["true_label"]).mean()
        accuracy_results[col] = acc
        print(f"  {col:20s}: {acc:.4f}")

    consensus_acc = (df["consensus"] == df["true_label"]).mean()
    accuracy_results["consensus"] = consensus_acc
    print(f"  {'consensus':20s}: {consensus_acc:.4f}")

    if has_llm:
        llm_acc = (df["llm_annotator"] == df["true_label"]).mean()
        accuracy_results["llm_annotator"] = llm_acc
        print(f"  {'llm_annotator':20s}: {llm_acc:.4f}")

    # --- Per-category F1 ---
    print("\n" + "=" * 60)
    print("PER-CATEGORY F1 SCORES")
    print("=" * 60)

    consensus_f1 = f1_score(df["true_label"], df["consensus"], labels=CATEGORIES, average=None)
    print("\n  Human Consensus:")
    f1_results = {"consensus": {}}
    for cat, f in zip(CATEGORIES, consensus_f1):
        print(f"    {cat:10s}: {f:.4f}")
        f1_results["consensus"][cat] = float(f)

    if has_llm:
        llm_f1 = f1_score(df["true_label"], df["llm_annotator"], labels=CATEGORIES, average=None)
        print("\n  LLM (Claude):")
        f1_results["llm_annotator"] = {}
        for cat, f in zip(CATEGORIES, llm_f1):
            print(f"    {cat:10s}: {f:.4f}")
            f1_results["llm_annotator"][cat] = float(f)

    if has_llm:
        # --- Disagreement analysis ---
        print("\n" + "=" * 60)
        print("DISAGREEMENT ANALYSIS: LLM vs CONSENSUS")
        print("=" * 60)

        disagree = df[df["llm_annotator"] != df["consensus"]]
        print(f"\n  Total disagreements: {len(disagree)} / {len(df)} ({len(disagree)/len(df)*100:.1f}%)")

        # Category breakdown of disagreements
        print("\n  Disagreement by true category:")
        for cat in CATEGORIES:
            cat_disagree = disagree[disagree["true_label"] == cat]
            cat_total = (df["true_label"] == cat).sum()
            print(f"    {cat:10s}: {len(cat_disagree):4d} / {cat_total} ({len(cat_disagree)/cat_total*100:.1f}%)")

        # --- LLM correct, consensus wrong (and vice versa) ---
        print("\n" + "=" * 60)
        print("COMPLEMENTARITY ANALYSIS")
        print("=" * 60)

        llm_right_cons_wrong = df[(df["llm_annotator"] == df["true_label"]) & (df["consensus"] != df["true_label"])]
        llm_wrong_cons_right = df[(df["llm_annotator"] != df["true_label"]) & (df["consensus"] == df["true_label"])]
        both_wrong = df[(df["llm_annotator"] != df["true_label"]) & (df["consensus"] != df["true_label"])]
        both_right = df[(df["llm_annotator"] == df["true_label"]) & (df["consensus"] == df["true_label"])]

        print(f"  Both correct:              {len(both_right):4d} ({len(both_right)/len(df)*100:.1f}%)")
        print(f"  LLM correct, consensus wrong: {len(llm_right_cons_wrong):4d} ({len(llm_right_cons_wrong)/len(df)*100:.1f}%)")
        print(f"  Consensus correct, LLM wrong: {len(llm_wrong_cons_right):4d} ({len(llm_wrong_cons_right)/len(df)*100:.1f}%)")
        print(f"  Both wrong:                {len(both_wrong):4d} ({len(both_wrong)/len(df)*100:.1f}%)")

        # Complementarity score
        consensus_errors = (df["consensus"] != df["true_label"]).sum()
        complementarity = len(llm_right_cons_wrong) / consensus_errors if consensus_errors > 0 else 0
        print(f"\n  Complementarity score: {complementarity:.4f}")
        print(f"  (When consensus is wrong, LLM gets it right {complementarity*100:.1f}% of the time)")

        # Category breakdown of complementarity
        print("\n  LLM corrects consensus errors by category:")
        for cat in CATEGORIES:
            cat_corrections = llm_right_cons_wrong[llm_right_cons_wrong["true_label"] == cat]
            cat_cons_errors = ((df["consensus"] != df["true_label"]) & (df["true_label"] == cat)).sum()
            if cat_cons_errors > 0:
                print(f"    {cat:10s}: {len(cat_corrections):3d} / {cat_cons_errors} ({len(cat_corrections)/cat_cons_errors*100:.1f}%)")

        # Classification reports
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT: HUMAN CONSENSUS")
        print("=" * 60)
        print(classification_report(df["true_label"], df["consensus"], labels=CATEGORIES))

        print("=" * 60)
        print("CLASSIFICATION REPORT: LLM (CLAUDE)")
        print("=" * 60)
        print(classification_report(df["true_label"], df["llm_annotator"], labels=CATEGORIES))

        # Save evaluation data
        eval_results = {
            "accuracy": accuracy_results,
            "f1_scores": f1_results,
            "disagreements": len(disagree),
            "complementarity_score": complementarity,
            "llm_corrects_consensus": len(llm_right_cons_wrong),
            "consensus_corrects_llm": len(llm_wrong_cons_right),
            "both_correct": len(both_right),
            "both_wrong": len(both_wrong),
        }
    else:
        eval_results = {
            "accuracy": accuracy_results,
            "f1_scores": f1_results,
        }

    with open(OUTPUTS_DIR / "evaluation_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    # Save consensus column
    df.to_csv(DATA_DIR / "annotations_all.csv", index=False)
    print(f"\nEvaluation results saved to {OUTPUTS_DIR / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
