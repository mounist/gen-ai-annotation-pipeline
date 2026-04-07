"""
Step 1: Data Preparation
Download AG News dataset from HuggingFace and sample 1000 items (250 per category).
"""

import pandas as pd
from datasets import load_dataset
from pathlib import Path

LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
SAMPLE_PER_CATEGORY = 250
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    print("Downloading AG News dataset from HuggingFace...")
    ds = load_dataset("ag_news", split="test")
    df = pd.DataFrame(ds)

    df["label_name"] = df["label"].map(LABEL_MAP)
    print(f"Full dataset size: {len(df)}")
    print(f"Label distribution:\n{df['label_name'].value_counts()}\n")

    # Balanced sampling: 250 per category
    sampled = (
        df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(n=SAMPLE_PER_CATEGORY, random_state=42))
        .reset_index(drop=True)
    )
    sampled = sampled.sample(frac=1, random_state=42).reset_index(drop=True)
    sampled["item_id"] = range(len(sampled))

    result = sampled[["item_id", "text", "label_name"]].rename(
        columns={"label_name": "true_label"}
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "ag_news_sample.csv"
    result.to_csv(out_path, index=False)
    print(f"Saved {len(result)} items to {out_path}")
    print(f"Label distribution:\n{result['true_label'].value_counts()}")


if __name__ == "__main__":
    main()
