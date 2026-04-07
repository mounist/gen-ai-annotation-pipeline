"""
Step 2: Simulate 5 Human Annotators
Each annotator has a different accuracy and bias profile.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"]
SEED = 42


def simulate_annotation(true_label: str, accuracy: float, bias: dict, rng: np.random.Generator) -> str:
    """Simulate a single annotation with given accuracy and bias profile.

    Args:
        true_label: The ground-truth category.
        accuracy: Probability of returning the correct label.
        bias: Dict mapping (true_category, confused_category) -> weight multiplier.
              e.g. {("Business", "Sci/Tech"): 3} means when the true label is Business,
              Sci/Tech is 3x more likely to be chosen as the wrong answer.
        rng: NumPy random generator.
    """
    if rng.random() < accuracy:
        return true_label

    # Build weights for wrong categories
    wrong_cats = [c for c in CATEGORIES if c != true_label]
    weights = np.ones(len(wrong_cats), dtype=float)
    for i, cat in enumerate(wrong_cats):
        if (true_label, cat) in bias:
            weights[i] *= bias[(true_label, cat)]
    weights /= weights.sum()

    return rng.choice(wrong_cats, p=weights)


# Annotator profiles
ANNOTATORS = {
    "annotator_1": {"accuracy": 0.92, "bias": {}},  # Expert
    "annotator_2": {"accuracy": 0.85, "bias": {}},  # Good
    "annotator_3": {"accuracy": 0.78, "bias": {}},  # Average
    "annotator_4": {  # Confuses Business <-> Sci/Tech
        "accuracy": 0.80,
        "bias": {
            ("Business", "Sci/Tech"): 3.0,
            ("Sci/Tech", "Business"): 3.0,
        },
    },
    "annotator_5": {  # Noisy, specific mislabeling patterns
        "accuracy": 0.70,
        "bias": {
            ("World", "Business"): 3.0,
            ("Sports", "World"): 3.0,
        },
    },
}


def main():
    df = pd.read_csv(DATA_DIR / "ag_news_sample.csv")
    print(f"Loaded {len(df)} items")

    rng = np.random.default_rng(SEED)

    for name, profile in ANNOTATORS.items():
        annotations = [
            simulate_annotation(row["true_label"], profile["accuracy"], profile["bias"], rng)
            for _, row in df.iterrows()
        ]
        df[name] = annotations
        acc = (df[name] == df["true_label"]).mean()
        print(f"{name}: accuracy = {acc:.3f} (target ~{profile['accuracy']})")

    out_path = DATA_DIR / "annotations_human.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved annotations to {out_path}")

    # Quick sanity check on annotator_4 bias
    a4_errors = df[df["annotator_4"] != df["true_label"]]
    biz_to_sci = ((a4_errors["true_label"] == "Business") & (a4_errors["annotator_4"] == "Sci/Tech")).sum()
    sci_to_biz = ((a4_errors["true_label"] == "Sci/Tech") & (a4_errors["annotator_4"] == "Business")).sum()
    print(f"\nAnnotator 4 bias check: Business->Sci/Tech={biz_to_sci}, Sci/Tech->Business={sci_to_biz}")


if __name__ == "__main__":
    main()
