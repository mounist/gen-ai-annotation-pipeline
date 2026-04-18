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
    has_ft = "finetuned_model" in df.columns and df["finetuned_model"].notna().all()
    if not has_llm:
        print("WARNING: LLM annotations missing. Running evaluation without LLM comparison.")
    if not has_ft:
        print("WARNING: Fine-tuned model predictions missing. Run src/07_finetune_classifier.py for 3-way comparison.")

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

    if has_ft:
        ft_acc = (df["finetuned_model"] == df["true_label"]).mean()
        accuracy_results["finetuned_model"] = ft_acc
        print(f"  {'finetuned_model':20s}: {ft_acc:.4f}")

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

    if has_ft:
        ft_f1 = f1_score(df["true_label"], df["finetuned_model"], labels=CATEGORIES, average=None)
        print("\n  Fine-tuned DistilBERT:")
        f1_results["finetuned_model"] = {}
        for cat, f in zip(CATEGORIES, ft_f1):
            print(f"    {cat:10s}: {f:.4f}")
            f1_results["finetuned_model"][cat] = float(f)

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

        if has_ft:
            # --- 3-way comparison: consensus vs Claude vs fine-tuned DistilBERT ---
            print("\n" + "=" * 60)
            print("3-WAY AGREEMENT: CONSENSUS vs CLAUDE vs FINE-TUNED")
            print("=" * 60)
            cons_ok = df["consensus"] == df["true_label"]
            llm_ok = df["llm_annotator"] == df["true_label"]
            ft_ok = df["finetuned_model"] == df["true_label"]
            all_right = (cons_ok & llm_ok & ft_ok).sum()
            all_wrong = ((~cons_ok) & (~llm_ok) & (~ft_ok)).sum()
            only_ft = ((~cons_ok) & (~llm_ok) & ft_ok).sum()
            only_llm = ((~cons_ok) & llm_ok & (~ft_ok)).sum()
            only_cons = (cons_ok & (~llm_ok) & (~ft_ok)).sum()
            print(f"  All three correct:       {all_right:4d} ({all_right/len(df)*100:.1f}%)")
            print(f"  All three wrong:         {all_wrong:4d} ({all_wrong/len(df)*100:.1f}%)")
            print(f"  Only fine-tuned correct: {only_ft:4d}")
            print(f"  Only Claude correct:     {only_llm:4d}")
            print(f"  Only consensus correct:  {only_cons:4d}")

            print("\n" + "=" * 60)
            print("CLASSIFICATION REPORT: FINE-TUNED DISTILBERT")
            print("=" * 60)
            print(classification_report(df["true_label"], df["finetuned_model"], labels=CATEGORIES))

            eval_results["three_way"] = {
                "all_correct": int(all_right),
                "all_wrong": int(all_wrong),
                "only_finetuned_correct": int(only_ft),
                "only_llm_correct": int(only_llm),
                "only_consensus_correct": int(only_cons),
            }
    else:
        eval_results = {
            "accuracy": accuracy_results,
            "f1_scores": f1_results,
        }

    ft_path = OUTPUTS_DIR / "finetuned_results.json"
    if ft_path.exists():
        eval_results["finetuned_meta"] = json.loads(ft_path.read_text())

    with open(OUTPUTS_DIR / "evaluation_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    # Save consensus column
    df.to_csv(DATA_DIR / "annotations_all.csv", index=False)
    print(f"\nEvaluation results saved to {OUTPUTS_DIR / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
