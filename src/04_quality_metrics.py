"""
Step 4: Quality Metrics Pipeline
IAA, error rates, confusion matrices, bias detection, annotator ranking.
"""

import pandas as pd
import numpy as np
import krippendorff
from itertools import combinations
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from pathlib import Path

from utils import compute_consensus, CATEGORIES, ANNOTATOR_COLS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


def per_annotator_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy for each annotator vs ground truth."""
    rows = []
    for col in ANNOTATOR_COLS:
        acc = (df[col] == df["true_label"]).mean()
        rows.append({"annotator": col, "accuracy": acc})
    if "llm_annotator" in df.columns and df["llm_annotator"].notna().all():
        acc = (df["llm_annotator"] == df["true_label"]).mean()
        rows.append({"annotator": "llm_annotator", "accuracy": acc})
    return pd.DataFrame(rows)


def compute_krippendorff_alpha(df: pd.DataFrame) -> dict:
    """Compute Krippendorff's alpha overall and per category."""
    cat_to_int = {c: i for i, c in enumerate(CATEGORIES)}

    # Build reliability data matrix (annotators x items)
    cols = ANNOTATOR_COLS[:]
    if "llm_annotator" in df.columns and df["llm_annotator"].notna().all():
        cols.append("llm_annotator")

    data = np.array([[cat_to_int.get(v, np.nan) for v in df[col]] for col in cols], dtype=float)
    overall_alpha = krippendorff.alpha(reliability_data=data, level_of_measurement="nominal")

    results = {"overall": overall_alpha}

    # Per-category: binary agreement (category vs not-category)
    for cat in CATEGORIES:
        binary_data = np.array(
            [[(1 if v == cat else 0) for v in df[col]] for col in cols], dtype=float
        )
        try:
            alpha = krippendorff.alpha(reliability_data=binary_data, level_of_measurement="nominal")
        except Exception:
            alpha = float("nan")
        results[cat] = alpha

    return results


def compute_cohen_kappa_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Cohen's kappa for each annotator pair."""
    cols = ANNOTATOR_COLS[:]
    if "llm_annotator" in df.columns and df["llm_annotator"].notna().all():
        cols.append("llm_annotator")

    rows = []
    for a, b in combinations(cols, 2):
        kappa = cohen_kappa_score(df[a], df[b])
        rows.append({"annotator_a": a, "annotator_b": b, "kappa": kappa})
    return pd.DataFrame(rows)


def compute_fleiss_kappa(df: pd.DataFrame) -> float:
    """Compute Fleiss' kappa for all human annotators."""
    n_items = len(df)
    n_annotators = len(ANNOTATOR_COLS)
    n_categories = len(CATEGORIES)

    # Build category count matrix
    cat_to_idx = {c: i for i, c in enumerate(CATEGORIES)}
    counts = np.zeros((n_items, n_categories), dtype=int)
    for col in ANNOTATOR_COLS:
        for i, val in enumerate(df[col]):
            if val in cat_to_idx:
                counts[i, cat_to_idx[val]] += 1

    # Fleiss' kappa formula
    p_j = counts.sum(axis=0) / (n_items * n_annotators)
    P_i = (np.sum(counts ** 2, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    P_bar = P_i.mean()
    P_e = np.sum(p_j ** 2)

    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1.0 - P_e)


def compute_confusion_matrices(df: pd.DataFrame) -> dict:
    """Compute confusion matrix for each annotator."""
    results = {}
    cols = ANNOTATOR_COLS[:]
    if "llm_annotator" in df.columns and df["llm_annotator"].notna().all():
        cols.append("llm_annotator")

    for col in cols:
        cm = confusion_matrix(df["true_label"], df[col], labels=CATEGORIES)
        results[col] = cm
    return results


def detect_systematic_bias(df: pd.DataFrame) -> list[dict]:
    """Flag annotators with systematic bias (error rate on specific pairs >2x average)."""
    flagged = []

    # Compute average error rates for each (true, predicted) category pair across annotators
    pair_errors = {}
    for col in ANNOTATOR_COLS:
        errors = df[df[col] != df["true_label"]]
        for true_cat in CATEGORIES:
            for pred_cat in CATEGORIES:
                if true_cat == pred_cat:
                    continue
                count = ((errors["true_label"] == true_cat) & (errors[col] == pred_cat)).sum()
                total = (df["true_label"] == true_cat).sum()
                rate = count / total if total > 0 else 0
                pair_errors.setdefault((true_cat, pred_cat), []).append(rate)

    avg_pair_rates = {k: np.mean(v) for k, v in pair_errors.items()}

    # Check each annotator
    for col in ANNOTATOR_COLS:
        errors = df[df[col] != df["true_label"]]
        for true_cat in CATEGORIES:
            for pred_cat in CATEGORIES:
                if true_cat == pred_cat:
                    continue
                count = ((errors["true_label"] == true_cat) & (errors[col] == pred_cat)).sum()
                total = (df["true_label"] == true_cat).sum()
                rate = count / total if total > 0 else 0
                avg_rate = avg_pair_rates[(true_cat, pred_cat)]
                if avg_rate > 0 and rate > 2 * avg_rate and count >= 5:
                    flagged.append({
                        "annotator": col,
                        "true_category": true_cat,
                        "predicted_category": pred_cat,
                        "error_rate": rate,
                        "avg_error_rate": avg_rate,
                        "ratio": rate / avg_rate,
                        "count": count,
                    })

    return flagged


def rank_annotators(df: pd.DataFrame, kappa_df: pd.DataFrame, bias_flags: list[dict]) -> pd.DataFrame:
    """Rank annotators by composite score: accuracy + agreement + bias penalty."""
    consensus = compute_consensus(df)

    rows = []
    for col in ANNOTATOR_COLS:
        acc = (df[col] == df["true_label"]).mean()
        agreement = (df[col] == pd.Series(consensus)).mean()

        # Average kappa with other annotators
        kappas = kappa_df[(kappa_df["annotator_a"] == col) | (kappa_df["annotator_b"] == col)]["kappa"]
        avg_kappa = kappas.mean() if len(kappas) > 0 else 0

        # Bias penalty: number of flagged bias patterns
        n_biases = sum(1 for f in bias_flags if f["annotator"] == col)
        bias_penalty = n_biases * 0.05

        composite = 0.4 * acc + 0.3 * avg_kappa + 0.2 * agreement - 0.1 * bias_penalty
        rows.append({
            "annotator": col,
            "accuracy": acc,
            "avg_kappa": avg_kappa,
            "consensus_agreement": agreement,
            "n_bias_flags": n_biases,
            "composite_score": composite,
        })

    ranking = pd.DataFrame(rows).sort_values("composite_score", ascending=False).reset_index(drop=True)
    ranking.index += 1
    ranking.index.name = "rank"
    return ranking


def main():
    # Try annotations_all.csv first, fall back to annotations_human.csv
    all_path = DATA_DIR / "annotations_all.csv"
    human_path = DATA_DIR / "annotations_human.csv"
    if all_path.exists():
        df = pd.read_csv(all_path)
        print(f"Loaded {len(df)} items from {all_path}")
    elif human_path.exists():
        df = pd.read_csv(human_path)
        print(f"Loaded {len(df)} items from {human_path} (no LLM annotations)")
    else:
        print("ERROR: No annotation file found. Run steps 1-2 first.")
        return

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Per-annotator accuracy
    print("\n" + "=" * 60)
    print("PER-ANNOTATOR ACCURACY")
    print("=" * 60)
    acc_df = per_annotator_accuracy(df)
    for _, row in acc_df.iterrows():
        print(f"  {row['annotator']:20s}: {row['accuracy']:.4f}")

    # 2a. Krippendorff's alpha
    print("\n" + "=" * 60)
    print("KRIPPENDORFF'S ALPHA")
    print("=" * 60)
    alpha_results = compute_krippendorff_alpha(df)
    for key, val in alpha_results.items():
        print(f"  {key:15s}: {val:.4f}")

    # 2b. Cohen's kappa
    print("\n" + "=" * 60)
    print("COHEN'S KAPPA (PAIRWISE)")
    print("=" * 60)
    kappa_df = compute_cohen_kappa_pairs(df)
    for _, row in kappa_df.iterrows():
        print(f"  {row['annotator_a']:15s} vs {row['annotator_b']:15s}: {row['kappa']:.4f}")

    # 2c. Fleiss' kappa
    print("\n" + "=" * 60)
    print("FLEISS' KAPPA (ALL HUMAN ANNOTATORS)")
    print("=" * 60)
    fleiss = compute_fleiss_kappa(df)
    print(f"  Fleiss' kappa: {fleiss:.4f}")

    # 3. Confusion matrices
    print("\n" + "=" * 60)
    print("CONFUSION MATRICES")
    print("=" * 60)
    cm_dict = compute_confusion_matrices(df)
    for name, cm in cm_dict.items():
        print(f"\n  {name}:")
        print(f"  {'':15s} " + " ".join(f"{c:>10s}" for c in CATEGORIES))
        for i, true_cat in enumerate(CATEGORIES):
            row_str = " ".join(f"{cm[i, j]:10d}" for j in range(len(CATEGORIES)))
            print(f"  {true_cat:15s} {row_str}")

    # 4. Systematic bias detection
    print("\n" + "=" * 60)
    print("SYSTEMATIC BIAS DETECTION")
    print("=" * 60)
    bias_flags = detect_systematic_bias(df)
    if bias_flags:
        for flag in bias_flags:
            print(
                f"  [!] {flag['annotator']}: {flag['true_category']} -> {flag['predicted_category']} "
                f"(rate={flag['error_rate']:.3f}, avg={flag['avg_error_rate']:.3f}, "
                f"ratio={flag['ratio']:.1f}x, n={flag['count']})"
            )
    else:
        print("  No systematic biases detected.")

    # 5. Annotator ranking
    print("\n" + "=" * 60)
    print("ANNOTATOR RANKING")
    print("=" * 60)
    ranking = rank_annotators(df, kappa_df, bias_flags)
    print(ranking.to_string())

    # Save metrics for downstream use
    acc_df.to_csv(OUTPUTS_DIR / "annotator_accuracy.csv", index=False)
    kappa_df.to_csv(OUTPUTS_DIR / "cohen_kappa_pairs.csv", index=False)
    ranking.to_csv(OUTPUTS_DIR / "annotator_ranking.csv")
    pd.DataFrame(bias_flags).to_csv(OUTPUTS_DIR / "bias_flags.csv", index=False)

    # Save alpha and fleiss as JSON
    import json
    metrics = {"krippendorff_alpha": alpha_results, "fleiss_kappa": fleiss}
    with open(OUTPUTS_DIR / "iaa_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
